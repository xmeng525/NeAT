import mcubes
import numpy as np

import torch
import torch.nn.functional as F


def extract_fields(bound_min, bound_max, resolution, query_func):
    N = 64
    X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
    Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
    Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

    u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
    with torch.no_grad():
        for xi, xs in enumerate(X):
            for yi, ys in enumerate(Y):
                for zi, zs in enumerate(Z):
                    xx, yy, zz = torch.meshgrid(xs, ys, zs)
                    pts = torch.stack([xx, yy, zz], dim=-1).reshape(N, -1, 3)
                    val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                    u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
    return u


def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
    print('threshold: {}'.format(threshold))
    u = extract_fields(bound_min, bound_max, resolution, query_func)
    vertices, triangles = mcubes.marching_cubes(u, threshold)
    b_max_np = bound_max.detach().cpu().numpy()
    b_min_np = bound_min.detach().cpu().numpy()

    vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
    return vertices, triangles


class NeATRenderer:
    def __init__(self,
                 sdf_network,
                 deviation_network,
                 color_network,
                 n_samples,
                 perturb,
                 validity_enhance_param):
        """Initializes a new NeATRenderer.

        :param sdf_network: The SDF-Net and the Validity-Net.
        :param deviation_network: The network to predict the deviation.
        :param color_network: The Color-Net.
        :param n_samples: (scalar) The number of samples in each ray.
        :param perturb: (scalar) The scale of perterbation, ranging in [0, 1].
        :param validity_enhance_param: (scalar) We only apply validity for the sampls with SDF close to 0.
            validity_enhance_param determines the level of "close".
        """
        self.sdf_network = sdf_network
        self.deviation_network = deviation_network
        self.color_network = color_network
        self.n_samples = n_samples
        self.perturb = perturb
        self.validity_enhance_param = validity_enhance_param
        self.z_vals = None

    def render_core(self,
                    rays_o,
                    rays_d,
                    z_vals,
                    sample_dist,
                    cos_anneal_ratio,
                    sigmoid_factor):
        """The core function of NeAT neural rendering.

        :param rays_o: (B, 3) The origin of the rays.
        :param rays_d: (B, 3) The normalized directions of the rays.
        :param z_vals: (B, Nsample) The depth values of the rays.
        :param sample_dist: (scalar) The distance between two samples.
        :param cos_anneal_ratio: (scalar) The scalar grows from 0 to 1 in the beginning training iterations. The anneal
            strategy makes the cos value "not dead" at the beginning training iterations, for better convergence.
        :param sigmoid_factor: (scalar) sigmoid_factor determines the width of the valid SDF region.
        """
        batch_size, n_samples = z_vals.shape

        # Section length
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape)], -1)
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3
        dirs = rays_d[:, None, :].expand(pts.shape).reshape(-1, 3)

        # Gets the sdf, positional features, and the gradients.
        sdf_nn_output = self.sdf_network(pts)
        self.sdf = sdf_nn_output[:, 0:1]
        self.gradients = self.sdf_network.gradient(pts)
        feature_vector = sdf_nn_output[:, 2:]

        # Gets the "predicted" validity probability of the sampled positions.
        self.sdf_is_valid_pred = sdf_nn_output[:, 1:2]
        
        # Enhances the predicted validity probability of the sampled positions.
        sample_is_valid = self.sdf_is_valid_pred * torch.exp(-self.validity_enhance_param * self.sdf * self.sdf)
        self.validity_probability = torch.sigmoid((sample_is_valid - 0.5) * sigmoid_factor)
        
        # Gets the sampled color of the sampled positions.
        self.sampled_color = self.color_network(pts.reshape(-1, 3),
                                                self.gradients,
                                                dirs,
                                                feature_vector).reshape(batch_size, n_samples, 3)

        # Gets the deviation.
        self.inv_s = self.deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)  # Single parameter
        self.inv_s = self.inv_s.expand(batch_size * n_samples, 1)

        # Calculates the cos value of the angle between the view directions and the gradients.
        self.cos = (dirs * self.gradients).sum(-1, keepdim=True)
        self.true_cos = -torch.abs(self.cos)

        # Use the annual strategy to make the cos value "not dead" at the beginning training iterations.
        self.iter_cos = -(F.relu(-self.true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio)
                          + F.relu(-self.true_cos) * cos_anneal_ratio)  # always non-positive

        # Then we calculate the opacity density of the sampled points. According to Equation 5 in the paper, we have
        # rho(t) = [-d[Phi(-sign(v * n) * sdf)] / dt] / Phi(-sign(v * n) * sdf)
        
        # Firstly, estimates the numerator. We have:
        # pdf = -d[Phi(-sign(v * n) * sdf)] / dt
        #     = -(Phi(-sign(v * n) * sdf + d(-sign(v * n) * sdf)) - Phi(-sign(v * n) * sdf - d(-sign(v * n) * sdf)))
        #     = Phi(-sign(v * n) * sdf - d(-sign(v * n) * sdf)) - Phi(-sign(v * n) * sdf + d(-sign(v * n) * sdf))
        # If sign(v * n) = 1,
        # pdf = Phi(-sdf - d(-sdf)) - Phi(-sdf + d(-sdf))
        #     = Phi(sdf + d(-sdf)) - Phi(sdf - d(-sdf))
        #     = Phi(sdf - d(sdf)) - Phi(sdf + d(sdf))

        # If sign(v * n) = -1,
        # pdf = Phi(sdf - d(sdf)) - Phi(sdf + d(sdf))
        # Therefore, sign(v * n) doesn't affect the calculation of pdf.

        # Calculates: sdf + d(sdf)
        self.estimated_next_sdf = self.sdf + self.iter_cos * dists.reshape(-1, 1) * 0.5
        # Calculates: sdf - d(sdf)
        self.estimated_prev_sdf = self.sdf - self.iter_cos * dists.reshape(-1, 1) * 0.5

        # Calculates: Phi(sdf + d(sdf))
        self.next_cdf = torch.sigmoid(self.estimated_next_sdf * self.inv_s)
        # Calculates: Phi(sdf - d(sdf))
        self.prev_cdf = torch.sigmoid(self.estimated_prev_sdf * self.inv_s)
        
        # Calculates: pdf = Phi(sdf - d(sdf)) - Phi(sdf + d(sdf))
        self.p = self.prev_cdf - self.next_cdf
        
        # Secondly, estimates the denominator. i.e. the cdf of the sampled points:
        # cdf = Phi(-sign(v * n) * sdf)
        self.c = torch.sigmoid(self.sdf * self.inv_s * torch.sign(self.cos) * (-1))

        # Estimates the discrete opacity = p / c * validity_probability. (Equation 8 in the paper)
        self.alpha = ((self.p) / (self.c + 1e-6)).reshape(batch_size, n_samples).clip(0.0, 1.0) \
            * self.validity_probability.reshape(batch_size, n_samples)

        # Estimates the rendering weights.
        self.accum_trans = torch.cumprod(torch.cat([torch.ones([batch_size, 1]),
                                                    1. - self.alpha + 1e-7], -1), -1)[:, :-1]
        self.weights = self.alpha * self.accum_trans

        # Estimates the color of the sampled pixels.
        color = (self.sampled_color * self.weights[:, :, None]).sum(dim=1)
        
        # Eikonal loss
        gradient_error = ((torch.linalg.norm(self.gradients.reshape(
            batch_size, n_samples, 3), ord=2, dim=-1) - 1.0) ** 2).mean()
        gradients_dir_error = F.relu(self.gradients * self.validity_probability).mean()

        return {
            'color_fine': color,
            'sdf': self.sdf,
            'gradients': self.gradients.reshape(batch_size, n_samples, 3),
            's_val': 1.0 / self.inv_s,
            'weights': self.weights,
            'cdf_fine': self.c.reshape(batch_size, n_samples),
            'gradient_error': gradient_error,
            'gradients_dir_error': gradients_dir_error,
        }

    def render(self,
               rays_o,
               rays_d,
               near,
               far,
               perturb_overwrite=-1,
               cos_anneal_ratio=0.0,
               sigmoid_factor=10):
        batch_size = len(rays_o)
        sample_dist = 2.0 / self.n_samples   # Assuming the region of interest is a unit sphere
        z_vals = torch.linspace(0.0, 1.0, self.n_samples)
        z_vals = near + (far - near) * z_vals[None, :]

        perturb = self.perturb
        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1]) - 0.5)
            z_vals = z_vals + t_rand * 2.0 / self.n_samples

        # Render core
        ret_fine = self.render_core(rays_o, rays_d, z_vals, sample_dist, cos_anneal_ratio, sigmoid_factor)
        return ret_fine

    def extract_geometry(self, bound_min, bound_max, resolution, threshold=0.0, query_func=None):
        if query_func is None:
            return extract_geometry(bound_min,
                                    bound_max,
                                    resolution=resolution,
                                    threshold=threshold,
                                    query_func=lambda pts: -self.sdf_network.sdf_withnan(pts))
        else:
            return extract_geometry(bound_min,
                                    bound_max,
                                    resolution=resolution,
                                    threshold=threshold,
                                    query_func=query_func)
