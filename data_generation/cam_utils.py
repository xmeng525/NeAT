import numpy as np


def normalize(v):
    '''Normalizes the input vector.
    :param v (3, ): The input vector to be normalized.

    :return (3, ): The normalized vector. 
    '''
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm


def getModelViewMatrix(eye, up=np.asarray([0,1,0])):
    '''Uses the lookAt camera to generate the model view matrix
    
    :param eye_dist (3, ): The eye position of the camera.
    :param up (3, ): The up vector of the lookat matrix.

    :return (4, 4): The modelview matrix.
    '''
    forward = normalize(-eye)
    left = normalize(np.cross(up, forward))
    upDir = np.cross(forward, left)

    m = np.identity(4)
    # Sets the rotation part
    m[0, :3] = left
    m[1, :3] = upDir
    m[2, :3] = forward

    # Sets the translation part
    m[0, 3] = -left[0] * eye[0] - left[1] * eye[1] - left[2] * eye[2]
    m[1, 3] = -upDir[0] * eye[0] - upDir[1] * eye[1] - upDir[2] * eye[2]
    m[2, 3] = -forward[0] * eye[0] - forward[1] * eye[1] - forward[2] * eye[2]
    return m


def getPinHoleMatrix(scalings=[1, 1, 1]):
    '''Generate the pin-hole camera matrix.
    :param scalings (3, ): The scaling of the x-, y-, and z-axis.

    :return (4, 4): The pin-hole camera matrix.
    '''
    m = np.identity(4)
    m[0,0] = scalings[0]
    m[1,1] = scalings[1]
    m[2,2] = scalings[2]
    return m


def getCameraMat(eye, up=np.asarray([0,1,0]), scalings=[1, 1, 1]):
    '''Generates the camera matrix for the softras renderer.
    :param eye (3, ): The eye position of the camera.
    :param up (3, ): The up vector of the lookat matrix.
    :param scalings (3, ): The scaling of the x-, y-, and z-axis.

    :return (4, 4): The camera matrix for the softras renderer.
    '''
    model_view_mat = getModelViewMatrix(eye, up)
    pin_hole_mat = getPinHoleMatrix(scalings)
    camera_mat = pin_hole_mat@model_view_mat
    return camera_mat
