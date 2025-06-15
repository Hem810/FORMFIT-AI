# FORMFIT-AI
An AI-powered fitness companion that provides real-time guidance and training entirely on-device using edge AI technology.

## Table of Contents
1. [About](#about)
2. [Setup](#setup)
3. [Test](#test)
4. [Run](#run)
5. [Features](#features)


## About
FORMFIT-AI is an intelligent fitness companion application that leverages edge AI capabilities for real-time workout guidance, form correction, and exercise tracking. The application is utilizing on-device AI processing for pose detection and analysis to ensure privacy and low-latency performance.
On the Snapdragon X Elite, the model is optimized to leverage the Neural Processing Unit (NPU) at inference runtime. Elsewhere, it will run using the CPU.

## Setup
 1. git clone repo
      ```
      >>git clone https://github.com/hem810/FORMFIT-AI.git
      ```
   2. Create virtual environment
      ```
      >> python -m virtual_env venv
      ```
   3. Activate virtual environment
      ```
      >> venv\Scripts\activate 
      ```
   4. Install dependencies
      ```
      >> pip install -r requirements.txt
      ```
   4. Download model from AI Hub 
      https://aihub.qualcomm.com/compute/models/hrnet_pose?domain=Computer+Vision&useCase=Pose+Estimation

   6. Transfer model to FORMFIT-AI/models/
      ```
      >> mv Downloads/hrnet_pose.onnx FORMFIT-AI/models/
      ```
## Test
To test the setup run
```
 >>python test_setup.py
```
## Run
(from FORMFIT-AI directory)
```
>> python scripts/main.py 
```

## Features
1.The code gives the option to train on prebuilt workouts as well a create personal workouts from the given 
  set of exerscies and then use that to train.
2.The code will analyse the angles between different points on the body and give suggestion if your pose is incorrect.
## Use
1.The first page is just the homepage
2. The generate workout page you can seect the exerscies from the left and the set the reps and number of sets on the right and click create workout.Then you will be prompted to enter the name.after you enter the name click on the bottom right button to continue.
3.Then select the workout on the select_workout page
4.Click on start on the last page to start the workout.You can use the next to skip the exescise or stop to quit.

