# End-to-End self-driving
&emsp;&emsp;In this project, I used Pytorch to achieve end-to-end self-driving. I use gym-donkey to achieve self-driving based on OpenCV and generate dataset. Then, I used those data to successfully train the model and successfully acommplish self-driving in gym.

![Static Badge](https://img.shields.io/badge/Python%203.7%2B-make?&logo=python&logoColor=white&labelColor=blue&color=gray)
&emsp;![Static Badge](https://img.shields.io/badge/PyTorch,Torchvision-make?logo=pytorch&logoColor=white&labelColor=orange&color=white)
&emsp;![Static Badge](https://img.shields.io/badge/OpenCV-make?logo=opencv&logoColor=white&labelColor=green&color=white)
&emsp;![Static Badge](https://img.shields.io/badge/Gym-make?logo=OpenAI&logoColor=white&labelColor=black&color=white)

## 1. Introduction
&emsp;&emsp;In this project, based on the paper "End to End Learning for Self-Driving Cars", I successfully accomplish self-driving in the Gym. 

![Alt Text](https://example.com/path/to/your/gif.gif) 
&emsp;&emsp;Paper: https://arxiv.org/abs/1604.07316 

&emsp;&emsp;Simulate: [Gym-Donkey](https://github.com/tawnkramer/gym-donkeycar)

## 2. Building virtual environment and simulator 
### 2.1 Env and dependencies
```
### git clone this project
$ git clone https://github.com/unswimmingduck/End2End_Driving.git
$ cd End2End_Driving

### create virtual env and install depencies
$ conda create -n Driving_env python=3.7
$ conda activate Driving_env

### install depencies
$ pip install -r requirement.txt 
```
### 2.2 Simuulator
&emsp;&emsp; In this project, I choose Gym-Donkey as the simulator. If you want to get more detail, you can [click here](https://github.com/tawnkramer/gym-donkeycar) to get more information. 

&emsp;&emsp; Firstly, we should download simulator binaries in: https://github.com/tawnkramer/gym-donkeycar/releases. You could choose the right version depending on your computer and unzip to your file. 

&emsp;&emsp; Secondly, we should install the gym and gym_donkey lib. 
```
$ pip install git+https://github.com/tawnkramer/gym-donkeycar
```
## 3. Works
&emsp;&emsp; To achieveing self-driving, there are four parts work to do: **collecting data**, **building CNN model and dataset**, **training CNN model**, **deploying CNN model in simulator**. In the following, I will introduce those four parts work.
### 3.1 Collecting data
&emsp;&emsp; In order to collecting data and generate dataset, I use OpenCV to achieve self-driving based on lane detectation and the steel angle will be collected in the OpenCV self-driving. In the following parts, I will introduce how I use OpenCV to realize self-driving based on lane detectation.
&emsp;&emsp; Firstly, in Gym-Donkey, we can get the image from virtual camera by function ```img, _, _, _ = env.step(np.array([steel_angle, speed])```. And I realize the self-driving in ```env = gym.make("donkey-generated-roads-v0")```. The road img shows below.
