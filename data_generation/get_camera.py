import numpy as np

from cam_utils import getCameraMat


def getCubicCamera(num_loop,
                   num_sample_per_loop,
                   eye_dist_val, 
                   scalings,
                   camera_all=dict(),
                   eps=1e-6):
    '''Uniformly samples cameras on a sphere.

    Example: We sample 3 loops and 4 samples per loop.
             The angle of the three loops and the y-axis are 30 degree, 90 degree, and 150 degree.
             The angle of the sampled cameras on a loop and the x-axis are 0 degree, 90 degree, 180 degree, 270 degree.

    :param num_loop: The number of loops along the y-axis.
    :param num_sample_per_loop: The number of cameras sampled on each loop.
    :param eye_dist_val: The radius of the sphere.
    :param scalings: The scaling of the object.
    :param camera_all: The dict storing all the sampled camera matrices. The keys are the strings of camera names. The
        values are the camera matrices.
    :param eps: The small variable to prevent singularity.

    :return: The number of sampled cameras.
    '''
    for id_loop in range(num_loop):
        angle_u = (id_loop + 0.5) / num_loop * np.pi
        for id_sample in range(num_sample_per_loop):
            angle_v = id_sample / num_sample_per_loop * 2.0 * np.pi
            
            eye = eye_dist_val * np.asarray([
                np.sin(angle_u) * np.cos(angle_v),
                np.cos(angle_u),
                np.sin(angle_u) * np.sin(angle_v),
            ]) + eps
            
            up = np.asarray([eps, 1, eps])

            camera = getCameraMat(eye, up, scalings)

            camera_index = id_sample + id_loop * num_sample_per_loop

            camera_all["camera_mat_%d"%(camera_index)] = camera
            camera_all["world_mat_%d"%(camera_index)] = np.eye(4)
            camera_all["scale_mat_%d"%(camera_index)] = np.eye(4)
            camera_all["camera_pos_%d"%(camera_index)] = eye

    return num_loop * num_sample_per_loop


def getCubicCameraWithTop(num_loop,
                             num_sample_per_loop,
                          eye_dist_val,
                          scalings,
                          camera_all=dict(),
                          eps=1e-3):
    '''Uniformly samples cameras on a sphere.

    Example: We sample 4 loops and 4 samples per loop.
             The angle of the three loops and the y-axis are 0 degree, 60 degree, 120 degree and 180 degree.
             The angle of the sampled cameras on a loop and the x-axis are 0 degree, 90 degree, 180 degree, 270 degree.

    :param num_loop: The number of loops along the y-axis.
    :param num_sample_per_loop: The number of cameras sampled on each loop.
    :param eye_dist_val: The radius of the sphere.
    :param scalings: The scaling of the object.
    :param camera_all: The dict storing all the sampled camera matrices. The keys are the strings of camera names. The
        values are the camera matrices.
    :param eps: The small variable to prevent singularity.

    :return: The number of sampled cameras.
    '''
    for id_loop in range(num_loop):
        angle_u = id_loop / (num_loop - 1) * np.pi
        for id_sample in range(num_sample_per_loop):
            angle_v = id_sample / num_sample_per_loop * 2.0 * np.pi
            
            eye = eye_dist_val * np.asarray([
                np.sin(angle_u) * np.cos(angle_v),
                np.cos(angle_u),
                np.sin(angle_u) * np.sin(angle_v),
            ]) + eps

            up = np.asarray([eps, 1, eps])

            camera = getCameraMat(eye, up,scalings)

            camera_index = id_sample + id_loop * num_sample_per_loop

            camera_all["camera_mat_%d"%(camera_index)] = camera
            camera_all["world_mat_%d"%(camera_index)] = np.eye(4)
            camera_all["scale_mat_%d"%(camera_index)] = np.eye(4)
            camera_all["camera_pos_%d"%(camera_index)] = eye

    return num_loop * num_sample_per_loop
