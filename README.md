# Visual Odometry using SfM

This project implements a visual odometry pipeline on the Oxford Robotcar Dataset. Demo video can be viewed [here](https://drive.google.com/file/d/1PHvL80Kd2P5Srsl9RrBpmU_2RR0EHPnn/view?usp=sharing)

# Dependencies
- opencv
- matplotlib
- numpy
- scipy

# Instructions
- Download the dataset from this [link](https://drive.google.com/drive/folders/1f2xHP_l8croofUL_G5RZKmJo2YE9spx9).
- To run the code for the project, type the following commands in the terminal:
```
python proj5.py
```
- The result data will be generated in the result folder. A plot will also be generated as program is run.

# Future work/Improvements
- Implement kalman filter to improve results.
- Implement bundle adjustment to run at a slower rate and keep optimizing self-pose and 3dpoint cloud.
