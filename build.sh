cd ./third_party/Sophus

# Sophus
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4

# Photo-SLAM
cd ../../..
mkdir build
cd build
# cmake .. # add Torch_DIR and/or OpenCV_DIR definitions if needed, example:
cmake .. -DTorch_DIR=/mnt/SSD0/linux_libs/pytorch/libtorch/share/cmake/Torch #-DOpenCV_DIR=/home/rapidlab/libs/opencv/lib/cmake/opencv4
make -j4
