import argparse
import json
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import math

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import soft_renderer as sr

from get_camera import getCubicCamera
from get_camera import getCubicCameraWithTop


def transform_cam_to_neus_format(load_camera_dict, data_path, n_view, h, w):
    save_camera_dict = dict()
    for view_id in range(n_view):
        camera_mat = load_camera_dict["camera_mat_%d"%view_id]
        world_mat = load_camera_dict["world_mat_%d"%view_id]
        scale_mat = np.eye(4)
        scale_mat[0,0] *= (w - 1) / 2
        scale_mat[1,1] *= (h - 1) / 2

        scale_mat[0,2] += (w - 1) / 2
        scale_mat[1,2] += (h - 1) / 2

        save_camera_dict["world_mat_%d"%view_id] = scale_mat
        save_camera_dict["scale_mat_%d"%view_id] = camera_mat@world_mat

    np.savez(os.path.join(data_path, "cameras_sphere.npz"), **save_camera_dict)


def save_image(input, path):
    im = Image.fromarray((255*input).astype(np.uint8))
    im.save(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-conf', '--config-path', type=str, default='./confs/conf_render_obj_perspective.json')
    parser.add_argument('--visualize_camera',action='store_true')
    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        conf = json.load(f)

    mesh_path = conf['input_mesh_path']
    texture_type = conf['input_mesh_texture_type']
    output_dir = conf['output_dir']
    image_size = conf['resolution']
    num_loop = conf['num_loop']
    num_sample_per_loop = conf['num_sample_per_loop']
    camera_mode = conf['camera_mode']
    viewing_angle = conf['viewing_angle']
    dist_scaling = conf['dist_scaling']
    near_far = conf['near_far']
    keep_alpha = conf['keep_alpha']

    eye_dist_val = dist_scaling * (1. / math.tan(math.radians(viewing_angle)) + 1)
    scalings = [1 / np.tan(viewing_angle / 180 * math.pi), 1 / np.tan(viewing_angle / 180 * math.pi), 1]
    camera_all = dict()
    total_camera_num = getCubicCamera(num_loop, num_sample_per_loop, eye_dist_val, scalings, camera_all)

    # Visualizes the camera positions.
    if args.visualize_camera:
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        cam_pos_arr = []
        for dir_idx in range(total_camera_num):
            cam_pos_arr.append(camera_all["camera_pos_%d"%dir_idx])
        cam_pos_arr = np.asarray(cam_pos_arr)
        ax.scatter3D(cam_pos_arr[:, 0], cam_pos_arr[:, 1], cam_pos_arr[:, 2])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()

    # Saves the camera and renders the images & masks.
    if os.path.exists(mesh_path):
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "image"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "mask"), exist_ok=True)

        # Saves the cameras.
        transform_cam_to_neus_format(camera_all, output_dir, total_camera_num, image_size, image_size)

        # Loads normalized_mesh -> *= 0.45 -> save gt mesh.
        mesh = sr.Mesh.from_obj(mesh_path,
                                normalization=True,
                                load_texture=True, 
                                texture_res=32,
                                texture_type=texture_type)
        mesh.vertices *= 0.45
        mesh.save_obj(os.path.join(output_dir, "GT.obj"), save_texture=True)

        # Loads the gt mesh -> render images
        mesh = sr.Mesh.from_obj(os.path.join(output_dir, "GT.obj"),
                                load_texture=True, 
                                texture_res=32,
                                texture_type=texture_type)
        for index in range(total_camera_num):
            # Creates a renderer with SoftRas.
            camera = camera_all["camera_mat_%d"%index]            
            renderer = sr.SoftRenderer(image_size=image_size, camera_mode='projection', \
                P=torch.tensor(camera).float().cuda().unsqueeze(0)[:,0:3,:],
                perspective=(camera_mode=="perspective"), orig_size=image_size/2, 
                near=-near_far, far=near_far, background_color=[0,0,0], 
                eps=1e-5, gamma_val=1e-6, sigma_val=1e-7, anti_aliasing=True)
            image = np.flipud(renderer.render_mesh(mesh).detach().cpu().numpy()[0].transpose((1, 2, 0)))
            if keep_alpha:
                save_image(image, os.path.join(output_dir, "image", "%06d.png"%(index)))
            else:
                save_image(image[:,:,0:3], os.path.join(output_dir, "image", "%06d.png"%(index)))
            save_image(image[:,:,3] > 0, os.path.join(output_dir, "mask", "%03d.png"%(index)))
            mesh.reset_()


if __name__ == '__main__':
    main()
