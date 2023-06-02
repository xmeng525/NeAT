# NeAT Data Generation
Given a point cloud or a mesh file as the input. You may uniformly sample cameras on a sphere and render images with cameras info used for NeAT reconstruction.

# Render arbitrary obj
To use the code, please install [SoftRas](https://github.com/ShichenLiu/SoftRas).

To generate image data with camera info, in this directory, run 
```
python render_obj.py --conf ./confs/conf_render_obj_perspective.json
```
Feel free to change the input, output, and the camera settings in `./confs/conf_render_obj_perspective.json` or `./confs/conf_render_obj_orthogonal.json`


# Render pointcloud of [deep fashion 3d](https://kv2000.github.io/2020/03/25/deepFashion3DRevisited/)
To use the code, please install [PyTorch3D](https://pytorch3d.org/).

To generate image data with camera info, in this directory, run 
```
python render_pc.py --conf ./confs/conf_render_pc.json
```
Feel free to change the input, output, and the camera settings in `./confs/conf_render_pc.json`.