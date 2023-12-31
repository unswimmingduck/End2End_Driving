# End-to-End self-driving

![Static Badge](https://img.shields.io/badge/Python%203.7%2B-make?&logo=python&logoColor=white&labelColor=blue&color=gray)
&emsp;![Static Badge](https://img.shields.io/badge/PyTorch,Torchvision-make?logo=pytorch&logoColor=white&labelColor=orange&color=white)
&emsp;![Static Badge](https://img.shields.io/badge/OpenCV-make?logo=opencv&logoColor=white&labelColor=green&color=white)
&emsp;![Static Badge](https://img.shields.io/badge/Gym-make?logo=OpenAI&logoColor=white&labelColor=black&color=white)

## 1. Introduction
&emsp;&emsp;In this project, based on the paper "End to End Learning for Self-Driving Cars", I used Pytorch to achieve end-to-end self-driving. I use gym-donkey to achieve self-driving based on OpenCV and generate dataset. Then, I used those data to successfully train the model and successfully acommplish self-driving in gym. 

&emsp;&emsp;Paper: https://arxiv.org/abs/1604.07316 

&emsp;&emsp;Simulator: [Gym-Donkey](https://github.com/tawnkramer/gym-donkeycar)  

&emsp;&emsp;**The following shows the performance of end-to-end self-driving works in gym-donkey. (the video may be unfluent because of the net)**  
![image](https://github.com/unswimmingduck/End2End_Driving/blob/main/End2end.mp4)

<img src="https://img.shields.io/badge/YouTube-FF0000?style=for-the-badge&logo=youtube&logoColor=white">  

&emsp;&emsp; You could also see the performance of the end-to-end self-driving CNN model in gym-donkey in [my YouTube channels](https://youtu.be/EANfsycsA-0?si=7uxOH4wzZ-gtPLKH)

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
### 2.2 Simulator
&emsp;&emsp; In this project, I choose Gym-Donkey as the simulator. If you want to get more detail, you can [click here](https://github.com/tawnkramer/gym-donkeycar) to get more information. 

&emsp;&emsp; Firstly, we should download **simulator binaries** in https://github.com/tawnkramer/gym-donkeycar/releases. You could choose the right version depending on your computer and unzip to your file. 

&emsp;&emsp; Secondly, we should install the gym and gym_donkey lib. 
```
$ pip install git+https://github.com/tawnkramer/gym-donkeycar
```
## 3. Training and Driving
### 3.1 Training
#### 3.1.1 Collecting data
&emsp;&emsp; Before training, we should collecting the data and generate the dataset.  
&emsp;&emsp; Firstly, you should operate ```DonkeySimWin/donkey_sim.exe```(from simulator binaries) to lauch the simulator.  
&emsp;&emsp; Secondly, you can excute the following command to collect data and generate dataset. 
```
$ python datasets/get_data.py --speed 0.3 --num 1000 --epoch 5 --ratio 0.1 --path dataset
```
 &emsp;&emsp; In the command, ```--speed``` sets the speed of the donkey car. ```--num``` sets the num of frame in one roads. ```--epoch``` sets the num of raod ( In Gym-Donkey, everytime you excute ```gym.make("donkey-generated-roads-v0")```, it will randomly generate the road in the same surronding environment. So, if you want to get more information, you can set more epoch ). ```--ratio``` sets the ratio of val data in all collection data. ```--path``` sets the path to save those collecting data. So the following command will collecting 1000*5 data and split the data 4500 for train and 500 for validate in ```End2End_Driving\dataset\train``` and ```End2End_Driving\dataset\val``` respectively.
#### 3.1.2 Training
&emsp;&emsp; After data collection, you can excute the following command to start training. In the training, the config is used to configurate relavent parameters in training. 
```
$ python tools/train.py  config/Donkey_gym_config.yaml
```
&emsp;&emsp; If you want to train other environment in Donkey_gym, you could use my 128 epochs training checkpoint in ```checkpoint/epoch_128.pth```. What's more, you can see the training log in ```doc/```.
### 3.2 Driving
&emsp;&emsp; After you training successfully, we can test self-driving in donkey-gym.  
&emsp;&emsp; Firstly, you should operate ```DonkeySimWin/donkey_sim.exe```(from simulator binaries)  to lauch the simulator.   
&emsp;&emsp; Then, you can excute the following command to achieve self-driving.
```
$ python tools/driving.py --checkpoint checkpoint/epoch_128.pth --speed 0.3 --frames 1000
```
&emsp;&emsp; In the command, ```--checkpoint``` means the path of the well trained model, ```--speed``` sets the speed of donkey car in simulator(you better set the same value as you set in Section 3.1.1 Collecting data). ```--frames``` sets the num of frame in one test, so this value decide how long the self-driving will excute.(But the length of road is limited, if you set you speed=0.3, you better set the frames=1000).
## 4. Works
&emsp;&emsp; To achieveing self-driving, there are four parts work to do: **collecting data**, **building CNN model and dataset**, **training CNN model**, **deploying CNN model in simulator**. In the following, I will introduce those four parts work.
### 4.1 Collecting data
&emsp;&emsp; In order to collecting data and generate dataset, I use OpenCV to achieve self-driving based on lane detectation and the steel angle will be collected in the OpenCV self-driving. In the following parts, I will introduce how I use OpenCV to realize self-driving based on lane detectation.  
&emsp;&emsp; Firstly, in Gym-Donkey, we can get the image from virtual camera by function ```img, _, _, _ = env.step(np.array([steel_angle, speed])```. And I realize the self-driving in ```env = gym.make("donkey-generated-roads-v0")```. The road img shows below (the img in the size of 120x160). I use three lane to help the donkey car to know where it is. So in order to detect the right lane, center lane and left lane, I apply OpenCV to process the images.   
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![img](https://github.com/unswimmingduck/End2End_Driving/assets/111033998/3a431b44-ba0f-4d38-afe9-674dc8dcd3cd)   
&emsp;&emsp; Secondly, I firstly use ```cv2.cvtColor``` convert the format of img to HSV format. What's more, in order to deminish the noise, I apply ```cv2.GaussianBlur``` to got better detectation result. Then, I use ```cv2.inRange``` to get the mask of white lane in the left side and right side and the yellow lane in the center of road.   
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;![hsv](https://github.com/unswimmingduck/End2End_Driving/assets/111033998/b1aebef9-c391-4996-acde-6ee2c71d5d53)
&emsp;&emsp;![white](https://github.com/unswimmingduck/End2End_Driving/assets/111033998/a5fd8864-dfdb-48d5-ba29-5e87e6042a6c)
&emsp;&emsp;![yellow](https://github.com/unswimmingduck/End2End_Driving/assets/111033998/c8f0de9f-6353-443f-abc7-3a5e1205226f)   
&emsp;&emsp; Thirdly, I apply ```cv2.Canny``` to get the outline of white lanes and yellow lane. And I crop the left lane RoI, center lane RoI, and right lane RoI respectively. After that, I use ```cv2.HoughLinesP``` to got the line in those RoIs. What's, more, I used Mean-Filtering Algorithm to filter some useless line like some horizontal line segments that may make some error in getting the pose of lanes. Then, I apply Linear-Regression Algorithm to to fit the pose of the lane.  
&emsp;&emsp; Lastly, depending on those poses of lanes and different number of lanes has been detected, a function named ```offset_control``` in ```get_data.py``` will decide how much steel angle should be used. The following shows that based on lanes dectection, Donkey car can run in the road smoothly. And the ```doc/get_data.log``` shows the process of collecting data and generate the train and val dataset. 
![image]()

### 4.2 Building CNN Model and Dataset
#### 4.2.1 Model Architecture
&emsp;&emsp; Based on "[End to End Learning for Self-Driving Cars](https://arxiv.org/abs/1604.07316 )", the following picture shows the architecture of this end-to-end self driving net. If you want to get more details, you can see ```e2e_cnn/model/e2e_cnn.py```.
&emsp;&emsp; &emsp;![image](https://github.com/unswimmingduck/End2End_Driving/assets/111033998/a6e979c3-efba-46f7-8e15-e12d72071ad2)
#### 4.2.2 Dataset
&emsp;&emsp; I built a dataset for Gym_Donkey to load data to the e2e_cnn model in ```e2e_cmm/dataset```. In the function of ```__getitem__``` this dataset, I return two parameters: **images(in size of 120x160) and the gruond truth of steel angle**. **So if you want to use this model to train other data, you could build a custome dataset that need to return the same size images and steel angle.**  
#### 4.2.3 Data Augment
&emsp;&emsp; What's more, to make the model can predict better steel angle in random generated road, I also built data augment in the dataset.  
&emsp;&emsp; Firstly, I randomly filp the image horizontally and nagetivate the steel angle. In this way, the model can learn more information.  
&emsp;&emsp; Secondly, I apply random brightness adjustment to make the model could achieve successful prediction in bad light condition.

### 4.3 Training
&emsp;&emsp; In section 3.1.2, you can see how to train. What's more, you also can see train log in ```doc/trai_log.log```

### 4.4 Driving
&emsp;&emsp; I have deployed end-to-ednd self-driving in gym-donkey and you can use the test it in simulator by ```tools/driving.py```. And you can got more information in section 3.1.3 to konw how to start self-driving in gym-donkey.



