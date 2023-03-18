# NeAT
This is the official repo for the implementation of **NeAT: Learning Neural Implicit Surfaces with Arbitrary Topologies from Multi-view Images**.
![alt text](https://xmeng525.github.io/xiaoxumeng.github.io/projects/cvpr23_neat/teaser.png)

## [Project page](https://xmeng525.github.io/xiaoxumeng.github.io/projects/cvpr23_neat) |  [Paper](https://arxiv.org/abs/2106.10689) | [Data](https://www.dropbox.com/sh/utn5rnohmr0y2c8/AACdets4PQrP5CB1KwGkpOFUa?dl=0)


## Usage

### Setup the environment

You can create an anaconda environment called neat_env by running the following commands:

```
git clone https://github.com/xmeng525/NeAT.git
cd NeAT
conda create -n neat_env python=3.8
conda activate neat_env
pip install -r requirements.txt
```

## Demo

You can now test our code on the provided input images and checkpoints.

For example, run the following command to reconstruct a T-shirt from the [MGN dataset](https://virtualhumans.mpi-inf.mpg.de/mgn/):
```
python exp_runner.py --case MGN/TShirtNoCoat_125611500935128 --conf ./confs/wmask.conf --is_continue --mode validate_mesh --res 256
```

This script takes the images in `data/MGN/TShirtNoCoat_125611500935128` as the input, and loads the pretrained checkpoint in `exp/MGN/TShirtNoCoat_125611500935128/checkpoints`. 

The generated meshes are saved in `exp/MGN/TShirtNoCoat_125611500935128/meshes`:

* The generated result mesh with open surfaces is `00320000.ply`, where `00320000` is the number of training iterations.

* The mesh corresponding to the final signed distance field is `00320000_sdf.ply`.


Welcome to download more dataset and checkpoints from [HERE](https://www.dropbox.com/sh/utn5rnohmr0y2c8/AACdets4PQrP5CB1KwGkpOFUa?dl=0).

Similar to the example above, please throw the downloaded data into `data/DATASET_NAME` and throw the corresponding checkpoint into `exp/DATASET_NAME/CASE_NAME/wmask/checkpoints`.

Note:

For the data `data_DTU/*`, please use `--conf ./confs/wmask_dtu.conf`.

For the data `others/cat_mask`, please use `--conf ./confs/wmask_onlypos.conf`.

### Running
- **Prepare your own data**

Please refer to the [Data Conversion in NeuS](https://github.com/Totoro97/NeuS#data-convention) to generate your own data.

- **Training**

With the images and masks ready, you may reconstruct neural implicit surface with arbitrary topologies from multi-view images by running:
```
python exp_runner.py --case YOUR_CASE --conf ./confs/wmask.conf
```

- **Evaluation**

In the evaluation stage, you may export the mesh reconstructed with the multi-view images by running:
```
python exp_runner.py --case YOUR_CASE --conf ./confs/wmask.conf --is_continue --mode validate_mesh --res 512
```

## Citation

If you find our work useful in your research, please consider citing:

```
@article{meng_2023_neat,
	title={NeAT: Learning Neural Implicit Surfaces with Arbitrary Topologies from Multi-view Images},
	author={Meng, Xiaoxu and Chen, Weikai and Yang, Bo},
	journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
	month={June},
	year={2023}
}
```

## Acknowledgement

Some code snippets are borrowed from: [IDR](https://github.com/lioryariv/idr), [NeRF-pytorch](https://github.com/yenchenlin/nerf-pytorch), and [NeuS](https://github.com/Totoro97/NeuS). Thanks for these great projects!