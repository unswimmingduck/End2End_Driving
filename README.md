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
#### 3.1.1 Generating dataset
&emsp;&emsp; If you want to generate the data, you can use below command. ```--speed``` sets the speed of the donkey car. ```--num``` sets the num of frame in one roads. ```--epoch``` sets the num of raod ( In Gym-Donkey, everytime you excute ```gym.make("donkey-generated-roads-v0")```, it will randomly generate the road in the same surronding environment. So, if you want to get more information, you can set more epoch ). ```--ratio``` sets the ratio of val data in all collection data. ```--path``` sets the path to save those collecting data. So the following command will collecting 1000*5 data and split the data 4500 for train and 500 for validate in ```End2End_Driving\dataset\train``` and ```End2End_Driving\dataset\val``` respectively.  
```
$ python datasets/get_data.py --speed 0.3 --num 1000 --epoch 5 --ratio 0.1 --path dataset
```
#### 3.1.2 Realization
&emsp;&emsp; In order to collecting data and generate dataset, I use OpenCV to achieve self-driving based on lane detectation and the steel angle will be collected in the OpenCV self-driving. In the following parts, I will introduce how I use OpenCV to realize self-driving based on lane detectation.  
&emsp;&emsp; Firstly, in Gym-Donkey, we can get the image from virtual camera by function ```img, _, _, _ = env.step(np.array([steel_angle, speed])```. And I realize the self-driving in ```env = gym.make("donkey-generated-roads-v0")```. The road img shows below (the img in the size of 120x160). I use three lane to help the donkey car to know where it is. So in order to detect the right lane, center lane and left lane, I apply OpenCV to process the images.   
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![img](https://github.com/unswimmingduck/End2End_Driving/assets/111033998/3a431b44-ba0f-4d38-afe9-674dc8dcd3cd)   
&emsp;&emsp; Secondly, I firstly use ```cv2.cvtColor``` convert the format of img to HSV format. What's more, in order to deminish the noise, I apply ```cv2.GaussianBlur``` to got better detectation result. Then, I use ```cv2.inRange``` to get the mask of white lane in the left side and right side and the yellow lane in the center of road.   
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![hsv](https://github.com/unswimmingduck/End2End_Driving/assets/111033998/b1aebef9-c391-4996-acde-6ee2c71d5d53)
&emsp;&emsp;![white](https://github.com/unswimmingduck/End2End_Driving/assets/111033998/a5fd8864-dfdb-48d5-ba29-5e87e6042a6c)
&emsp;&emsp;![yellow](https://github.com/unswimmingduck/End2End_Driving/assets/111033998/c8f0de9f-6353-443f-abc7-3a5e1205226f)   
&emsp;&emsp; Thirdly, I apply ```cv2.Canny``` to get the outline of white lanes and yellow lane. And I crop the left lane RoI, center lane RoI, and right lane RoI respectively. After that, I use ```cv2.HoughLinesP``` to got the line in those RoIs. What's, more, I used Mean-Filtering Algorithm to filter some useless line like some horizontal line segments that may make some error in getting the pose of lanes. Then, I apply Linear-Regression Algorithm to to fit the pose of the lane.  
&emsp;&emsp; Lastly, depending on those poses of lanes and different number of lanes has been detected, a function named ```offset_control``` in ```get_data.py``` will decide how much steel angle should be used. The following shows that based on lanes dectection, Donkey car can run in the road smoothly. 


