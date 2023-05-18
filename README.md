# Instructions to run the script

- clone the repository and cd into the project-sim-pavan/AssignReid (this will be your root directory)
- Create a root directory and place all the images within the data folder
- Run the following commands sequentially inside the command prompt of anaconda
    - `conda create env --name AssignReid python==3.10`
    - `conda activate AssignReid`
    - `conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia`
    - `pip install -r requirements.txt`
    - `python ./Infernece/src/main.py --t .75 --bs 32` *(this command will be unique for the type of OS, please check while using)*

- The number of unique person value will be generated within the console (I found 3,996 unique persons out of 25,259 images for a threshold of 0.75)
- A folder named results inside inference has the activation mapping upon the original images
