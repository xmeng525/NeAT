import os
import json
import math
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from plyfile import PlyData
import numpy as np
from PIL import Image
import argparse

# Data structures and functions for rendering
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor
)

from cam_utils import getModelViewMatrix


def generate_scale_mat(res):
    scale_mat = np.eye(4)
    scale_mat[0,0] *= (res - 1) / 2
    scale_mat[1,1] *= (res - 1) / 2
    scale_mat[0,2] += (res - 1) / 2
    scale_mat[1,2] += (res - 1) / 2
    return scale_mat


def generate_cam_positions(elev_start,
                           elev_end,
                           elev_step,
                           elev_offset,
                           azim_start,
                           azim_end,
                           azim_step):
    elev_unit = (elev_end - elev_start) / elev_step
    elev_seq = np.linspace(elev_start + elev_offset, elev_end - elev_offset, elev_step)

    azim_unit = (azim_end - azim_start) / azim_step
    azim_seq = np.linspace(azim_start + azim_unit / 2, azim_end - azim_unit / 2, azim_step)

    elev_azim_pairs = np.stack([np.tile(elev_seq.reshape(-1, 1), (1, azim_step)),
                                np.tile(azim_seq.reshape(1, -1), (elev_step, 1))], axis=-1).reshape(-1, 2)
    elev_azim_pairs = np.insert(elev_azim_pairs, 0, np.asarray([elev_start + 1e-3, 1e-3]))
    elev_azim_pairs = np.insert(elev_azim_pairs, 0, np.asarray([elev_end - 1e-3, azim_end - 1e-3]))

    elev_azim_pairs = elev_azim_pairs.reshape(-1, 2)

    total_views = elev_step * azim_step + 2

    print("total_views = ", total_views)
    print("elev_seq = ", elev_seq)
    print("azim_seq = ", azim_seq)

    return elev_azim_pairs, total_views


def save_image(input, path):
    im = Image.fromarray((255*input).astype(np.uint8))
    im.save(path)


def read_ply_xyzrgb(filename):
    """ read XYZ RGB normals point cloud from filename PLY file """
    assert(os.path.isfile(filename))
    with open(filename, 'rb') as f:
        ply_data = PlyData.read(f)
        num_verts = ply_data['vertex'].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:,0] = ply_data['vertex'].data['x']
        vertices[:,1] = ply_data['vertex'].data['y']
        vertices[:,2] = ply_data['vertex'].data['z']
        vertices[:,3] = ply_data['vertex'].data['red']
        vertices[:,4] = ply_data['vertex'].data['green']
        vertices[:,5] = ply_data['vertex'].data['blue']
    return vertices 


def render_view(conf, device, eye, up=np.asarray([0,0,1])):
    raster_settings = PointsRasterizationSettings(
        image_size=conf['res'], 
        radius=conf['point_radius'],
        points_per_pixel=conf['points_per_pixel'],
    )
    model_view_mat = getModelViewMatrix(eye, up)
    R = torch.Tensor(model_view_mat[:3, :3].T).unsqueeze(0)
    T = torch.Tensor(model_view_mat[:3, 3]).unsqueeze(0)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T, znear=conf['znear'], zfar=conf['zfar'], fov=conf['fov'])

    world2camera_mat = np.eye(4)
    world2camera_mat[:3, :3] = R[0].numpy().T
    world2camera_mat[:3, 3] = T[0].numpy()

    camera2image_mat = np.eye(4)
    camera2image_mat[:3, :3] = cameras.get_projection_transform().get_matrix()[0].cpu().numpy().T[:3, :3]
    camera2image_mat[2, 2] = 1
    
    camera_mat = camera2image_mat@model_view_mat

    rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
    renderer = PointsRenderer(rasterizer=rasterizer, compositor=AlphaCompositor())
    return renderer, camera_mat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/conf_render_pc.json')
    args = parser.parse_args()
    with open(args.conf, 'r') as f:
        conf = json.load(f)

    with open(conf['category_id_file'], 'r') as f:
        category_id_dict = json.load(f)

    # Setup
    if torch.cuda.is_available():
        device = torch.device("cuda:%d"%conf['gpu_id'])
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    elev_azim_pairs, total_views = generate_cam_positions(conf['elev_start'],
                                                          conf['elev_end'],
                                                          conf['elev_step'],
                                                          conf['elev_offset'],
                                                          conf['azim_start'],
                                                          conf['azim_end'],
                                                          conf['azim_step'])
    output_dir = conf['output_dir'].format(n_views=np.round(total_views).astype(int))
    dist = conf['dist_scaling'] * (1. / math.tan(math.radians(conf['fov'] / 2)) + 1)

    if conf['category'] in category_id_dict:
        data_ids = category_id_dict[conf['category']]
    else:
        print('category %s does not exist!'%conf['category'])
        exit()

    for data_id in data_ids:
        print("Processing {data_id}".format(data_id=data_id))
        mesh_input_path = os.path.join(conf['input_dir'], '{data_id}_pc.ply'.format(data_id=data_id))
        output_path = os.path.join(output_dir, '{data_id}'.format(data_id=data_id))
        pointcloud = read_ply_xyzrgb(mesh_input_path)
        vertices = torch.Tensor(pointcloud[:, 0:3]).to(device)
        rgb = torch.Tensor(pointcloud[:, 3:6]).to(device) / 255

        camera_all = dict()
        camera_index = 0
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(os.path.join(output_path, "image"), exist_ok=True)
        os.makedirs(os.path.join(output_path, "mask"), exist_ok=True)

        for elev, azim in elev_azim_pairs:
            # print("elev = %d, azim = %d"%(elev, azim))
            eye = dist * np.asarray([np.cos(elev / 180 * math.pi) * np.sin(azim / 180 * math.pi), 
                                     np.cos(elev / 180 * math.pi) * np.cos(azim / 180 * math.pi),
                                     np.sin(elev / 180 * math.pi) + 1e-3])
            renderer, camera_mat = render_view(conf, device, eye)
            camera_all["world_mat_%d"%camera_index] = generate_scale_mat(conf['res'])
            camera_all["scale_mat_%d"%camera_index] = camera_mat

            point_cloud = Pointclouds(points=[vertices], features=[rgb])
            image = np.flipud(np.fliplr(renderer(point_cloud)[0, ..., :3].cpu().numpy()))
            save_image(image, os.path.join(output_path, "image", "%06d.png"%(camera_index)))

            point_cloud = Pointclouds(points=[vertices], features=[torch.ones_like(vertices).to(device)])
            image = np.flipud(np.fliplr(renderer(point_cloud)[0, ..., 0].cpu().numpy()))
            save_image(image, os.path.join(output_path, "mask", "%06d.png"%(camera_index)))
            camera_index += 1

        np.savez(os.path.join(output_path, "cameras_sphere.npz"), **camera_all)
        np.savez(os.path.join(output_path, "cam_info.npz"), fov=conf['fov'], elev_azim_pairs=elev_azim_pairs)

if __name__ == '__main__':
    main()
