# OmniGS
### [Homepage](https://liquorleaf.github.io/research/OmniGS/) | [arXiv](https://arxiv.org/abs/2404.03202)

**OmniGS: Fast Radiance Field Reconstruction using Omnidirectional Gaussian Splatting** <br>
[Longwei Li](https://github.com/liquorleaf)<sup>1</sup>, [Huajian Huang](https://huajianup.github.io)<sup>2</sup>, [Sai-Kit Yeung](https://saikit.org/)<sup>2</sup>, and [Hui Cheng](https://cse.sysu.edu.cn/teacher/ChengHui)<sup>1</sup> <br>
Sun Yat-Sen University<sup>1</sup>, The Hong Kong University of Science and Technology<sup>2</sup> <br>
2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)

## Usage
### Omnidirectional Rasterizer
The omnidirectional rasterizer is in `cuda_rasterizer/`. See the function `RasterizeGaussiansCUDA` in `src/rasterize_points.cu` for the interface. Set `camera_type = 3` to render omnidirectional images.

### OmniGS Examples
Otherwise you may want to run our C++ OmniGS examples:

### Prerequisites
```
sudo apt install libeigen3-dev libjsoncpp-dev libopengl-dev mesa-utils libglfw3-dev libglm-dev
```

| Dependencies | Tested with |
| ---- | ---- |
| OS | Ubuntu 20.04 LTS |
| gcc | 10.5.0 |
| cmake | 3.27.0, 3.27.5 |
| [CUDA](https://developer.nvidia.com/cuda-toolkit-archive) | 11.8 |
| [cuDNN](https://developer.nvidia.com/rdp/cudnn-archive) | v8.9.3, for CUDA 11.x |
| [OpenCV](https://opencv.org/releases/) | 4.7.0 (built with opencv_contrib-4.7.0 and CUDA 11.8)|
| [LibTorch](https://pytorch.org/get-started/locally/) | cxx11-abi-shared-with-deps-2.0.1+cu118 |
|||


#### Using LibTorch
If you do not have the LibTorch installed in the system search paths for CMake, you need to add additional options to `build.sh` help CMake find LibTorch. See `build.sh` for details. Otherwise, you can also add one line before `find_package(Torch REQUIRED)` of `CMakeLists.txt`:

[Option 1] Conda. If you are using Conda to manage your python packages and have installed compatible Pytorch, you could set the 
```
# export TORCH_INSTALL=https://developer.download.nvidia.cn/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
# pip install --no-cache $TORCH_INSTALL

set(Torch_DIR /the_path_to_conda/python3.x/site-packages/torch/share/cmake/Torch)
```

[Option 2] You cloud download the libtorch, e.g., [cu118](https://download.pytorch.org/libtorch/cu118) and then extract them to the folder `./the_path_to_where_you_extracted_LibTorch`. 
```
set(Torch_DIR /the_path_to_where_you_extracted_LibTorch/libtorch/share/cmake/Torch)
```

#### Using OpenCV with opencv_contrib and CUDA
Take version 4.7.0 for example, look into [OpenCV realeases](https://github.com/opencv/opencv/releases) and [opencv_contrib](https://github.com/opencv/opencv_contrib/tags), you will find [OpenCV 4.7.0](https://github.com/opencv/opencv/archive/refs/tags/4.7.0.tar.gz) and the corresponding [opencv_contrib 4.7.0](https://github.com/opencv/opencv_contrib/archive/refs/tags/4.7.0.tar.gz), download them to the same directory (for example, `~/opencv`) and extract them. Then open a terminal and run:
```
cd ~/opencv
cd opencv-4.7.0/
mkdir build
cd build

# The build options we used in our tests:
cmake -DCMAKE_BUILD_TYPE=RELEASE -DWITH_CUDA=ON -DWITH_CUDNN=ON -DOPENCV_DNN_CUDA=ON -DWITH_NVCUVID=ON -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.8 -DOPENCV_EXTRA_MODULES_PATH="../../opencv_contrib-4.7.0/modules" -DBUILD_TIFF=ON -DBUILD_ZLIB=ON -DBUILD_JASPER=ON -DBUILD_CCALIB=ON -DBUILD_JPEG=ON -DWITH_FFMPEG=ON ..

# Take a moment to check the cmake output, see if there are any packages needed by OpenCV but not installed on your device

make -j8
# NOTE: We found that compilation of OpenCV may stuck at 99%, this may be caused by the final linking process. We just waited it for a while until it completed and exited without errors.
```
To install OpenCV into the system path:
```
sudo make install
```
If you prefer installing OpenCV to a custom path by adding `-DCMAKE_INSTALL_PREFIX=/your_preferred_path` option to the `cmake` command, remember to help Photo-SLAM find OpenCV by adding additional cmake options. See `build.sh` for details. Otherwise, you can also add the following line to `CMakeLists.txt`, `ORB-SLAM3/CMakeLists.txt` and `ORB-SLAM3/Thirdparty/DBoW2/CMakeLists.txt`, just like what we did for LibTorch.
```
set(OpenCV_DIR /your_preferred_path/lib/cmake/opencv4)
```

### OmniGS Installation
```
git clone https://github.com/liquorleaf/OmniGS.git
cd OmniGS/
chmod +x ./build.sh
./build.sh
```

## Examples

### Dataset
The benchmark datasets mentioned in our paper: [360Roam](https://huajianup.github.io/research/360Roam/), [EgoNeRF](https://github.com/changwoonchoi/EgoNeRF).

### Using openMVG to format the dataset
Follow instructions at [openMVG](https://github.com/openMVG/openMVG) and place the built binaries under somewhere you like (noted as`${openMVG_bin_dir}`). Modify `openMVG_bin_dir` and `dataset_dir` in `scripts/360roam_to_openmvg_lonlat_pose_known.sh` and `scripts/egonerf_to_openmvg_lonlat.sh`, then perform these scripts to format the dataset.

### Training and Testing
Change all `dataset_dir` in `scripts/360roam_run.py` and `scripts/egonerf_run.py`, then perform them to start training and testing.

:exclamation: Though OmniGS has the option to train and test perspective images, there exists some unknown bug for this function (see [Photo-SLAM #24](https://github.com/HuajianUP/Photo-SLAM/issues/24)). So we used [the original 3DGS](https://github.com/graphdeco-inria/gaussian-splatting) as the perspective
 baseline in our paper.

### View the result
Need the absolute path of result .ply files. See `scripts/view_lonlat.sh` for example.
