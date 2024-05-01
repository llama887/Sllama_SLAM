# Visual Navigation Game

This is forked from the course project platform for NYU ROB-GY 6203 Robot Perception. 
For more information about the course project, please reach out to AI4CE lab (cfeng at nyu dot edu).

# Instructions for Players
Note: Instructions Tested on Unbuntu Linux and WSL: Unbuntu.
1. Install
```commandline
conda update conda
git clone https://github.com/llama887/grave_digger.git
cd vis_nav_player
conda env create -f environment.yaml
conda activate game
sudo apt install libopenblas-base libomp-dev
```

2. Play
```commandline
python grave_digger.py
```

Move using arrow keys. LSHIFT allows moving the robot without updating the map and is used for making minor positional adjustments without messing up the map.

When you are done with exploration, press ENTER to generate a vocabulary. Once vocabulary is generated, press ESC to move into navigation phase. Launching navigation will open a large image displaying the target view and the top 12 views that the robot thinks matches the target. Click on the window and press ENTER to proceed. Then enter the index-1 (0-11) of the image that correlates to the target view in the terminal. This will pop up a new window with only the target view to use as reference. Click on that window and hit ENTER to proceed. Reclicking on the game window will allow you to navigate to the target following the plotted trajectory. 