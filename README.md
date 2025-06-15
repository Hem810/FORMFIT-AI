# FORMFIT-AI
An AI-powered fitness companion that provides real-time guidance and training entirely on-device using edge AI technology.

## Table of Contents
1. [About](#about)
2. [Setup](#setup)
3. [Test Setup](#test setup)
4. [Run](#run)


## About
FORMFIT-AI is an intelligent fitness companion application that leverages edge AI capabilities for real-time workout guidance, form correction, and exercise tracking. The application is specifically designed for Qualcomm Snapdragon X-powered devices, utilizing on-device AI processing for pose detection and analysis to ensure privacy and low-latency performance[1][2].
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
## Test Setup
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

