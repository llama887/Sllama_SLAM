# Visual Navigation Game

This is the course project for NYU ROB-GY 6203 Robot Perception. 
For more information, please reach out to AI4CE lab (cfeng at nyu dot edu).

# Installing Dependencies (Ubuntu)
All command line operations start from the project root directory.
1. Game Dependencies
```commandline
conda update conda
git clone --recursive https://github.com/llama887/vis_nav_player.git
cd vis_nav_player
conda env create -f environment.yml
conda activate game
```
2. Pangolin
```commandline
# Pangolin Installation/Building
cd Pangolin

# Install dependencies (as described above, or your preferred method)
./scripts/install_prerequisites.sh recommended

# Configure and build
cmake -B build
cmake --build build

# with Ninja for faster builds (sudo apt install ninja-build)
cmake -B build -GNinja
cmake --build build

# GIVEME THE PYTHON STUFF!!!! (Check the output to verify selected python version)
cmake --build build -t pypangolin_pip_install
```
3. ORB-SLAM3 Python Bindings
```commandline
sudo apt install libopencv-dev
sudo apt-get install libssl-dev
sudo apt install libopencv-dev
dpkg -l libopencv-dev
cd ORB-SLAM3-python
python setup.py install
```

# Playing
Play using the default keyboard player
```commandline
python sllama_slam.py
```
