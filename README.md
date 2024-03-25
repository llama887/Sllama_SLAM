# Visual Navigation Game

This is the course project for NYU ROB-GY 6203 Robot Perception. 
For more information, please reach out to AI4CE lab (cfeng at nyu dot edu).

# Installing Dependencies (Ubuntu)
All command line operations start from the project root directory.
## Game Dependencies
```commandline
conda update conda
git clone --recursive https://github.com/llama887/vis_nav_player.git
cd vis_nav_player
conda env create -f environment.yml
conda activate game
```
## OpenVSLAM-Python-bindings

### stella_vslam
Installation instructions extracted from: https://stella-cv.readthedocs.io/en/latest/installation.html
```commandline
sudo apt update -y
sudo apt upgrade -y --no-install-recommends
# basic dependencies
sudo apt install -y build-essential pkg-config cmake git wget curl unzip
# g2o dependencies
sudo apt install -y libatlas-base-dev libsuitesparse-dev
# OpenCV dependencies
sudo apt install -y libgtk-3-dev ffmpeg libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libavresample-dev libtbb-dev
# eigen dependencies
sudo apt install -y gfortran
# backward-cpp dependencies (optional)
sudo apt install -y binutils-dev
# other dependencies
sudo apt install -y libyaml-cpp-dev libgflags-dev sqlite3 libsqlite3-dev

# (if you plan on using PangolinViewer)
# Pangolin dependencies
sudo apt install -y libglew-dev

# (if you plan on using SocketViewer)
# Protobuf dependencies
sudo apt install -y autogen autoconf libtool
# Node.js
curl -sL https://deb.nodesource.com/setup_12.x | sudo -E bash -
sudo apt install -y nodejs

cd /tmp
wget -q https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.tar.bz2
tar xf eigen-3.3.7.tar.bz2 && rm -rf eigen-3.3.7.tar.bz2
cd eigen-3.3.7
mkdir -p build && cd build
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    ..
make -j4 && sudo make install

cd /tmp
# Download OpenCV
wget -q https://github.com/opencv/opencv/archive/4.5.5.zip
unzip -q 4.5.5.zip && rm -rf 4.5.5.zip
# Download aruco module (optional)
wget -q https://github.com/opencv/opencv_contrib/archive/refs/tags/4.5.5.zip -O opencv_contrib-4.5.5.zip
unzip -q opencv_contrib-4.5.5.zip && rm -rf opencv_contrib-4.5.5.zip
mkdir -p extra && mv opencv_contrib-4.5.5/modules/aruco extra
rm -rf opencv_contrib-4.5.5
# Build and install OpenCV
cd opencv-4.5.5
mkdir -p build && cd build
cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DBUILD_DOCS=OFF \
    -DBUILD_EXAMPLES=OFF \
    -DBUILD_JASPER=OFF \
    -DBUILD_OPENEXR=OFF \
    -DBUILD_PERF_TESTS=OFF \
    -DBUILD_TESTS=OFF \
    -DBUILD_PROTOBUF=OFF \
    -DBUILD_opencv_apps=OFF \
    -DBUILD_opencv_dnn=OFF \
    -DBUILD_opencv_ml=OFF \
    -DBUILD_opencv_python_bindings_generator=OFF \
    -DENABLE_CXX11=ON \
    -DENABLE_FAST_MATH=ON \
    -DWITH_EIGEN=ON \
    -DWITH_FFMPEG=ON \
    -DWITH_TBB=ON \
    -DWITH_OPENMP=ON \
    -DOPENCV_EXTRA_MODULES_PATH=/tmp/extra \
    ..
make -j4 && sudo make install

# Install stella_vslam core library
mkdir -p ~/lib
cd ~/lib
git clone --recursive https://github.com/stella-cv/stella_vslam.git
cd stella_vslam
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
make -j4
sudo make install
```


# Playing
```commandline
python sllama_slam.py
```
