# TWO-VIEW CT
Two-View CT Reconstruction Based on A Noise Adaptive Regularization Guided Conditional Diffusion Model
![system](https://github.com/Xlab2024/TWCT/blob/main/system.jpg?raw=true)
## Install requirements
Create a conda environment with the file provided:
```
$ conda env create -f environment.yaml
```
## Dataset
Convert the CT slices in your own dataset into 256x256 png images and store them in “./data/CT/AAPM/train/full_dose”
## Training
```
$ python main.py
```
## Predicting
```
$ python inverse_problem_solver_AAPM_3d_total.py
```
## A more detailed description will be provided shortly

