import argparse
import logging
import os

import cv2 as cv
import numpy as np
from pyhocon import ConfigFactory
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import trimesh
from shutil import copyfile
from tqdm import tqdm

from models.dataset import Dataset
from models.fields import RenderingNetwork
from models.fields import SDFValidityNet
from models.fields import SingleVarianceNetwork
from models.mesh_utils import remove_nan_from_mesh
from models.renderer import NeATRenderer


class Runner:
    def __init__(self, 
                 conf_path,
                 mode='train',
                 case='CASE_NAME',
                 is_continue=False,
                 prev_epoch=""):
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        os.makedirs(self.base_exp_dir, exist_ok=True)
        self.dataset = Dataset(self.conf['dataset'])
        self.iter_step = 0

        # Training parameters
        self.end_iter = self.conf.get_int('train.end_iter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_image_freq = self.conf.get_int('train.val_image_freq')
        self.val_mesh_freq = self.conf.get_int('train.val_mesh_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.validate_resolution_level = self.conf.get_int('train.validate_resolution_level')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.learning_rate_sdf = self.conf.get_float('train.learning_rate_sdf')
        self.learning_rate_validity = self.conf.get_float('train.learning_rate_validity')
        self.learning_rate_dev = self.conf.get_float('train.learning_rate_dev')
        self.learning_rate_alpha = self.conf.get_float('train.learning_rate_alpha')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.thin_shell_reg_end = self.conf.get_float('train.thin_shell_reg_end', default=1e5)
        self.anneal_end = self.conf.get_float('train.anneal_end', default=0.0)
        self.change_sigmoid_factor = self.conf.get_int('train.change_sigmoid_factor', default=0)
        # use_weighted_mask: 
        # 0: don't use weighted mask; 
        # 1: use occ/free as weighted mask;
        # 2: add sensitive region
        self.use_weighted_mask = self.conf.get_int('train.use_weighted_mask', default=1)
        self.is_mutliview_sample = self.conf.get_bool('train.is_mutliview_sample', default=False)

        # Weights
        self.igr_weight = self.conf.get_float('train.igr_weight')
        self.mask_weight = self.conf.get_float('train.mask_weight')
        self.nan_reg_weight = self.conf.get_float('train.nan_reg_weight')
        self.bce_reg_weight = self.conf.get_float('train.bce_reg_weight')

        self.is_continue = is_continue
        self.mode = mode
        self.model_list = []
        self.writer = None

        # Networks
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)
        params_to_train = []
        params_to_train += list(self.color_network.parameters())

        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        params_to_train_dev = []
        params_to_train_dev += list(self.deviation_network.parameters())

        self.sdf_network = SDFValidityNet(**self.conf['model.sdf_network']).to(self.device)
        params_to_train_sdf = []
        for l in range(0, self.sdf_network.num_layers - 1):
            params_to_train_sdf += list(getattr(self.sdf_network, "lin" + str(l)).parameters())

        params_to_train_validity = []
        for l in range(len(self.sdf_network.is_valid_dims) - 1):
            params_to_train_validity += list(getattr(self.sdf_network, "is_valid_lin" + str(l)).parameters())

        self.optimizer = torch.optim.Adam(params_to_train, lr=self.learning_rate)
        self.sdf_optimizer = torch.optim.Adam(params_to_train_sdf, lr=self.learning_rate_sdf)
        self.validity_optimizer = torch.optim.Adam(params_to_train_validity, lr=self.learning_rate_validity)
        self.dev_optimizer = torch.optim.Adam(params_to_train_dev, lr=self.learning_rate_dev)

        self.renderer = NeATRenderer(self.sdf_network,
                                     self.deviation_network,
                                     self.color_network,
                                     **self.conf['model.neat_renderer'])
        self.bce = torch.nn.BCELoss()
        # Load checkpoint
        latest_model_name = None
        if is_continue:
            if len(prev_epoch) > 0:
                latest_model_name = '{epoch}.pth'.format(epoch=prev_epoch)
            else:
                if os.path.exists(os.path.join(self.base_exp_dir, 'checkpoints')):
                    model_list = os.listdir(os.path.join(self.base_exp_dir, 'checkpoints'))
                    model_list.sort()
                    latest_model_name = model_list[-1]
        if latest_model_name is not None:
            logging.info('Find checkpoint: {}'.format(latest_model_name))
            self.load_checkpoint(latest_model_name)

        # Backup codes and configs for debug
        if self.mode[:5] == 'train':
            self.file_backup()

        if self.iter_step >= self.thin_shell_reg_end:
            self.nan_reg_weight = 0
            self.bce_reg_weight = 0

    def train(self):
        self.writer = SummaryWriter(log_dir=os.path.join(self.base_exp_dir, 'logs'))
        self.update_learning_rate()
        res_step = self.end_iter - self.iter_step
        image_perm = self.get_image_perm()

        for iter_i in tqdm(range(res_step)):
            if self.is_mutliview_sample:
                data = self.dataset.get_multiview_random_rays_at(self.batch_size)
            else:
                data = self.dataset.gen_random_rays_at(
                    image_perm[self.iter_step % len(image_perm)], self.batch_size)
            rays_o, rays_d, true_rgb, mask = data[:, :3], data[:, 3: 6], data[:, 6: 9], data[:, 9: 10]
            wt_mask_val = data[:, 10: 11]
            near, far = self.dataset.near_far_from_sphere(rays_o, rays_d)

            if self.mask_weight > 0.0:
                mask = (mask > 0.5).float()
            else:
                mask = torch.ones_like(mask)

            sigmoid_factor = 10 + self.change_sigmoid_factor * self.iter_step / self.end_iter

            render_out = self.renderer.render(rays_o, rays_d, near, far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              sigmoid_factor=sigmoid_factor)
            color_fine = render_out['color_fine']
            s_val = render_out['s_val']
            cdf_fine = render_out['cdf_fine']
            gradient_error = render_out['gradient_error']
            gradients_dir_error = render_out['gradients_dir_error']
            weight_max = torch.max(render_out['weights'], dim=-1, keepdim=True)[0]
            weight_sum = render_out['weights'].sum(dim=-1, keepdim=True)
            gradients = render_out['gradients']
            sdf = render_out['sdf']

            # Loss
            mask_sum = mask.sum() + 1e-5

            color_error = (color_fine - true_rgb) * mask
            color_fine_loss = F.l1_loss(color_error, torch.zeros_like(color_error), reduction='sum') / mask_sum
            psnr = 20.0 * torch.log10(1.0 / (((color_fine - true_rgb)**2 * mask).sum() / (mask_sum * 3.0)).sqrt())

            eikonal_loss = gradient_error

            if self.use_weighted_mask == 2:
                mask_wt = -19 * wt_mask_val + 20
                mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask, weight=mask_wt)
            elif self.use_weighted_mask == 1:
                mask_wt = -19 * mask + 20
                mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask, weight=mask_wt)
            elif self.use_weighted_mask == 0:
                mask_loss = F.binary_cross_entropy(weight_sum.clip(1e-3, 1.0 - 1e-3), mask)
            else:
                print("wrong param: use_weighted_mask")
                exit()

            valid_reg = self.renderer.sdf_is_valid_pred.mean()

            nan_reg_weight = np.clip(self.iter_step / self.warm_up_end * self.nan_reg_weight, 0, 1)
            bce_reg_weight = np.clip(self.iter_step / self.warm_up_end * self.bce_reg_weight, 0, 1)

            sivp = torch.sqrt(self.renderer.sdf_is_valid_pred.clip(1e-6, 1-1e-6))
            bce_reg = torch.mean(-sivp * torch.log(sivp) - (1.0 - sivp) * torch.log(1.0 - sivp))
            
            loss = color_fine_loss +\
                   eikonal_loss * self.igr_weight +\
                   mask_loss * self.mask_weight +\
                   valid_reg * nan_reg_weight+\
                   bce_reg * bce_reg_weight

            self.optimizer.zero_grad()
            self.sdf_optimizer.zero_grad()
            self.validity_optimizer.zero_grad()
            self.dev_optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.sdf_optimizer.step()
            self.validity_optimizer.step()
            self.dev_optimizer.step()
            self.iter_step += 1

            if self.iter_step % self.report_freq == 0:
                self.writer.add_scalar('Loss/loss', loss, self.iter_step)
                self.writer.add_scalar('Loss/color_loss', color_fine_loss, self.iter_step)
                self.writer.add_scalar('Loss/eikonal_loss', eikonal_loss, self.iter_step)
                self.writer.add_scalar('Loss/mask_loss', mask_loss, self.iter_step)
                self.writer.add_scalar('Loss/valid_reg', valid_reg, self.iter_step)
                self.writer.add_scalar('Loss/bce_reg', bce_reg, self.iter_step)
                self.writer.add_scalar('Statistics/s_val', s_val.mean(), self.iter_step)
                self.writer.add_scalar('Statistics/cdf', (cdf_fine[:, :1] * mask).sum() / mask_sum, self.iter_step)
                self.writer.add_scalar('Statistics/weight_max', (weight_max * mask).sum() / mask_sum, self.iter_step)
                self.writer.add_scalar('Statistics/psnr', psnr, self.iter_step)
                self.writer.add_scalar('Statistics/sigmoid_factor', sigmoid_factor, self.iter_step)

            if self.iter_step % self.save_freq == 0:
                self.save_checkpoint()

            if self.iter_step % self.val_image_freq == 0:
                self.validate_image(resolution_level=1)

            if self.iter_step % self.val_mesh_freq == 0:
                self.validate_mesh()

            if self.iter_step % self.report_freq == 0:
                self.update_learning_rate(self.writer)
                self.update_learning_rate_sdf(self.writer)
                self.update_learning_rate_validity(self.writer)
            else:
                self.update_learning_rate()
                self.update_learning_rate_sdf()
                self.update_learning_rate_validity()

            if self.iter_step % len(image_perm) == 0:
                image_perm = self.get_image_perm()

            # if self.iter_step >= self.warm_up_end:
            #     self.nan_reg_weight = 0
            #     self.bce_reg_weight = 0

    def get_image_perm(self):
        return torch.randperm(self.dataset.n_images)

    def get_cos_anneal_ratio(self):
        if self.anneal_end == 0.0:
            return 1.0 - 1e-6
        else:
            return np.min([1.0 - 1e-6, self.iter_step / self.anneal_end])

    def update_learning_rate(self, writer=None):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
        else:
            alpha = self.learning_rate_alpha
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            learning_factor = (np.cos(np.pi * progress) + 1.0) * 0.5 * (1 - alpha) + alpha

        for g in self.optimizer.param_groups:
            g['lr'] = self.learning_rate * learning_factor
        for g in self.dev_optimizer.param_groups:
            g['lr'] = self.learning_rate_dev * learning_factor

        if writer is not None:
            writer.add_scalar('Lr/color', self.learning_rate * learning_factor, self.iter_step)
            writer.add_scalar('Lr/dev', self.learning_rate_dev * learning_factor, self.iter_step)

    def update_learning_rate_sdf(self, writer=None):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
            for g in self.sdf_optimizer.param_groups:
                g['lr'] = self.learning_rate_sdf * learning_factor
            if writer is not None:
                writer.add_scalar('Lr/sdf', self.learning_rate_sdf * learning_factor, self.iter_step)
        else:
            for g in self.sdf_optimizer.param_groups:
                g['lr'] = self.learning_rate_sdf
            if writer is not None:
                writer.add_scalar('Lr/sdf', self.learning_rate_sdf, self.iter_step)


    def update_learning_rate_validity(self, writer=None):
        if self.iter_step < self.warm_up_end:
            learning_factor = self.iter_step / self.warm_up_end
            for g in self.validity_optimizer.param_groups:
                g['lr'] = self.learning_rate_validity * learning_factor
            if writer is not None:
                writer.add_scalar('Lr/validity', self.learning_rate_validity * learning_factor, self.iter_step)
        else:
            progress = (self.iter_step - self.warm_up_end) / (self.end_iter - self.warm_up_end)
            for g in self.validity_optimizer.param_groups:
                g['lr'] = self.learning_rate_validity + (self.learning_rate_sdf - self.learning_rate_validity) * progress
            if writer is not None:
                writer.add_scalar('Lr/validity', self.learning_rate_validity + (self.learning_rate_sdf - self.learning_rate_validity) * progress, self.iter_step)
    
    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.sdf_optimizer.load_state_dict(checkpoint['sdf_optimizer'])
        self.validity_optimizer.load_state_dict(checkpoint['validity_optimizer'])
        self.dev_optimizer.load_state_dict(checkpoint['dev_optimizer'])
        self.iter_step = checkpoint['iter_step']

        logging.info('End')

    def save_checkpoint(self):
        checkpoint = {
            'sdf_network_fine': self.sdf_network.state_dict(),
            'variance_network_fine': self.deviation_network.state_dict(),
            'color_network_fine': self.color_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'sdf_optimizer': self.sdf_optimizer.state_dict(),
            'validity_optimizer': self.validity_optimizer.state_dict(),
            'dev_optimizer': self.dev_optimizer.state_dict(),
            'iter_step': self.iter_step,
        }

        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))

    def validate_image(self, idx=-1, resolution_level=-1):
        if idx < 0:
            idx = np.random.randint(self.dataset.n_images)

        print('Validate: iter: {}, camera: {}'.format(self.iter_step, idx))

        if resolution_level < 0:
            resolution_level = self.validate_resolution_level
        rays_o, rays_d = self.dataset.gen_rays_at(idx, resolution_level=resolution_level)
        H, W, _ = rays_o.shape
        rays_o = rays_o.reshape(-1, 3).split(self.batch_size//4)
        rays_d = rays_d.reshape(-1, 3).split(self.batch_size//4)

        out_rgb_fine = []
        out_mask_fine = []
        out_normal_fine = []

        for rays_o_batch, rays_d_batch in zip(rays_o, rays_d):
            near, far = self.dataset.near_far_from_sphere(rays_o_batch, rays_d_batch)

            sigmoid_factor = 10 + self.change_sigmoid_factor * self.iter_step / self.end_iter

            render_out = self.renderer.render(rays_o_batch,
                                              rays_d_batch,
                                              near,
                                              far,
                                              cos_anneal_ratio=self.get_cos_anneal_ratio(),
                                              sigmoid_factor=sigmoid_factor)
            weight_sum = render_out['weights'].sum(dim=-1, keepdim=True)

            def feasible(key): return (key in render_out) and (render_out[key] is not None)

            if feasible('color_fine'):
                out_rgb_fine.append(render_out['color_fine'].detach().cpu().numpy())
                out_mask_fine.append(weight_sum.detach().cpu().numpy())
            if feasible('gradients') and feasible('weights'):
                normals = render_out['gradients'] * render_out['weights'][:, :self.renderer.n_samples, None]
                if feasible('inside_sphere'):
                    normals = normals * render_out['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()
                out_normal_fine.append(normals)
            del render_out

        img_fine = None
        if len(out_rgb_fine) > 0:
            img_fine = (np.concatenate(out_rgb_fine, axis=0).reshape([H, W, 3, -1]) * 255).clip(0, 255)
            mask_fine = (np.concatenate(out_mask_fine, axis=0).reshape([H, W, 1, -1]) * 255).clip(0, 255)

        normal_img = None
        if len(out_normal_fine) > 0:
            normal_img = np.concatenate(out_normal_fine, axis=0)
            rot = np.linalg.inv(self.dataset.pose_all[idx, :3, :3].detach().cpu().numpy())
            normal_img = (np.matmul(rot[None, :, :], normal_img[:, :, None])
                          .reshape([H, W, 3, -1]) * 128 + 128).clip(0, 255)

        os.makedirs(os.path.join(self.base_exp_dir, 'validations_fine'), exist_ok=True)
        os.makedirs(os.path.join(self.base_exp_dir, 'normals'), exist_ok=True)

        for i in range(img_fine.shape[-1]):
            if len(out_rgb_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'validations_fine',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           np.concatenate([img_fine[..., i],
                                           self.dataset.image_at(idx, resolution_level=resolution_level)]))
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'validations_fine',
                                        '{:0>8d}_{}_{}_mask.png'.format(self.iter_step, i, idx)),
                           np.concatenate([mask_fine[..., 0, i],
                                           self.dataset.image_at(idx, resolution_level=resolution_level)[:,:,0]]))
            if len(out_normal_fine) > 0:
                cv.imwrite(os.path.join(self.base_exp_dir,
                                        'normals',
                                        '{:0>8d}_{}_{}.png'.format(self.iter_step, i, idx)),
                           normal_img[..., i])

    def validate_mesh(self, world_space=False, resolution=64, threshold=0.0):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32)
        os.makedirs(os.path.join(self.base_exp_dir, 'meshes'), exist_ok=True)

        sigmoid_factor = 10 + self.change_sigmoid_factor * self.iter_step / self.end_iter
        vertices, triangles =\
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        vertices, triangles = remove_nan_from_mesh(vertices, triangles)
        if len(vertices) > 0:
            if world_space:
                vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]
            mesh = trimesh.Trimesh(vertices, triangles)
            mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}.ply'.format(self.iter_step)))

        vertices, triangles =\
            self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold,
                query_func=lambda pts: -self.renderer.sdf_network.sdf(pts))
        vertices, triangles = remove_nan_from_mesh(vertices, triangles)
        if len(vertices) > 0:
            if world_space:
                vertices = vertices * self.dataset.scale_mats_np[0][0, 0] + self.dataset.scale_mats_np[0][:3, 3][None]
            mesh = trimesh.Trimesh(vertices, triangles)
            mesh.export(os.path.join(self.base_exp_dir, 'meshes', '{:0>8d}_sdf.ply'.format(self.iter_step)))
        logging.info('End')


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    logging.basicConfig(level=logging.DEBUG, format=FORMAT)

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/base.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcube_threshold', type=float, default=0.0)
    parser.add_argument('--is_continue', default=False, action="store_true")
    parser.add_argument('--is_show', default=False, action="store_true")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--case', type=str, default='')
    parser.add_argument('--img_idx', type=int, default=0)
    parser.add_argument('--prev_epoch', type=str, default="")
    parser.add_argument('--res', type=int, default=512)
    parser.add_argument('--is_world_space', default=False, action="store_true")
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = Runner(args.conf, args.mode, args.case, args.is_continue, args.prev_epoch)

    if args.mode == 'train':
        runner.train()
    elif args.mode == 'validate_mesh':
        runner.validate_mesh(world_space=args.is_world_space, resolution=args.res, threshold=args.mcube_threshold)
    elif args.mode == 'validate_image':
        runner.validate_image(idx=args.img_idx, resolution_level=1)
