import numpy as np


def remove_nan_from_mesh(verts, faces):
    """Remove the vertices with nan coordinate in the input mesh. 

    :param verts: the vertices of the input mesh (containing nan coordinates)
    :param faces: the faces of the input mesh

    return: the vertices and faces of the mesh without nan coordinates.
    """
    if len(verts) == 0:
        return verts, faces

    index_map = dict()
    new_verts = []
    new_faces = []

    vert_id = 0
    for i, v in enumerate(verts):
        if np.isnan(v).any():
            index_map[i] = -1
        else:
            index_map[i] = vert_id
            vert_id += 1
            new_verts.append(v)

    for i, f in enumerate(faces):
        vid0, vid1, vid2 = f[0], f[1], f[2]
        vid0, vid1, vid2 = index_map[vid0], index_map[vid1], index_map[vid2]
        if vid0 == -1 or vid1 == -1 or vid2 == -1:
            continue
        new_faces.append([vid0, vid1, vid2])

    return np.asarray(new_verts), np.asarray(new_faces)
