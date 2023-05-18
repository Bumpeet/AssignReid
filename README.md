# # Problem: Estimate the number of unique people


## Summary

We provide a sample of crops taken from an open-source person re-id dataset. The task of the problem is to use deep learning techniques to cluster and identify the no. of unique people in this dataset. 

**Task 1:** Estimate the no. of unique people in the dataset

**Task 2:** Convert the deep learning model to ONNX and TensorRT (optional)

**Task 3** Visualise the clusters of people and the activation maps of the person re-id features

**The data:** The directory contains 25000 crops of some unique people. You can download the dataset from this [link](https://drive.google.com/file/d/109SrSu-muQm1UwuiyYLZI6ps72PHlaZU/view?pli=1).

## Structure
1. Folder **research**: It will have the experimentation/scratch codes. 
2. Folder **inference**  
	a.  **src**: final code to estimate the no. of unique persons in the image.  
	b.  **results**: visualisation of the inferred results.  
3. Folder **models**: This folder should contain your model files that could be visualised using Netron.
4. **README**: listing the instructions that are required for running the notebook/python codes. 


## Instructions to run the script

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
