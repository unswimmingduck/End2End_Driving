# In this project, I use Opencv to detect left, center, right line and use this data to control the DonkeyCar
# When the car running on the road, the data will be collect and output to corresponding file folder

import gym
import gym_donkeycar

import cv2
import numpy as np
import math

import argparse
import os
import json
import shutil

import random

import logging

# color detection threshold
yellow_lower = np.array([80, 40, 40])
yellow_upper = np.array([120, 120, 130])

white_lower = np.array([0, 0, 200])
white_upper = np.array([45, 20, 230])

# logger init
Data_logger = logging.getLogger('Data-Generator')
Data_logger.setLevel(logging.INFO)

formator = logging.Formatter(fmt = " %(levelname)s  -  %(message)s",datefmt="%d %X")

loggerfile = logging.FileHandler(filename=os.path.join(os.getcwd(),'dataset', "get_data.log"))
loggerfile.setFormatter(formator)
loggerfile.setLevel(logging.INFO)

sh = logging.StreamHandler()
sh.setFormatter(formator)
sh.setLevel(logging.INFO)

Data_logger.addHandler(sh)
Data_logger.addHandler(loggerfile)



def get_args():
    parser = argparse.ArgumentParser('get Data')
    parser.add_argument('--speed', type=float, default=0.3, help='set the speed of the car')
    parser.add_argument('--num', type=float, default=1000, help='set the number of data you want to gather in one epcho')
    parser.add_argument('--path', type=str, default='dataset',help='path to stor the data')
    parser.add_argument('--ratio', type=float, default=0.01,help='set the ratio of val in dataset')
    parser.add_argument('--epoch', type=float, default=5,help='set the epoch in data collection')
    args = parser.parse_args()
    return args



def generate_dataset(data_num, data_path, car_speed, ratio, epoch):
    # make train, val folder
    train_path = os.path.join(os.getcwd(),data_path,"train")
    val_path = os.path.join(os.getcwd(),data_path,"val")
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(val_path):
        os.makedirs(val_path)
    Data_logger.info("                        train path:{}".format(train_path))
    Data_logger.info("                        val path:{}".format(val_path))
    Data_logger.info("------------------- Collection Start ---------------")
    collect_data(data_num, train_path, car_speed, epoch)
    spilt_data(ratio, train_path, val_path)


def collect_data(data_num, train_path, car_speed, epoch):
    # initilzing the env
    data_info = {}
    Data_logger.info("------------------- Collection Start ---------------")
    
    #  start collection epoch  
    for e in range(epoch):
        Data_logger.info(" --- {} epoch collection start".format(e+1)) 
        # gnerate a new road 
        env = gym.make("donkey-generated-roads-v0")
        obs = env.reset()

        height, width,_ = obs.shape
        action = np.array([0.0, car_speed])

        obs, reward, done, info = env.step(action)
         
        for idx in range(data_num):
            data = {}
            data_idx = idx + e*data_num
            hsv = cv2.cvtColor(obs, cv2.COLOR_BGR2HSV)
            hsv = cv2.GaussianBlur(hsv, (1,1), 3)

            yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
            white_mask = cv2.inRange(hsv, white_lower, white_upper)

            yellow_edge = cv2.Canny(yellow_mask, 50, 150)
            white_edge = cv2.Canny(white_mask, 50, 150)

            left_x, center_x, right_x = generate_center(yellow_edge, white_edge, height)
            dir = offset_control(left_x, center_x, right_x, height, width)
            action[0] = dir
            
            data_path = os.path.join(train_path, "{:06d}_{:0.4f}.jpg".format(data_idx ,dir))
            cv2.imwrite(data_path, cv2.cvtColor(obs,cv2.COLOR_BGR2RGB))

            data["img"] = data_path
            data['dir'] = dir
            data_info[str(data_idx)] = data

            obs, reward, done, info = env.step(action)

        Data_logger.info(" --- {} epoch collection over, totally collect {} data".format(e+1,data_idx+1))
        env.close()

    Data_logger.info("------------------- Collection over ---------------")
    # generate data_info.json
    with open(os.path.join(train_path, "data_info.json"), 'w') as json_file:
        json.dump(data_info, json_file)
    Data_logger.info("Dump data info to {}".format(os.path.join(train_path, "data_info.json")))
    Data_logger.info("This collection totally collect {} data".format(len(data_info)))




def generate_center(yellow_mask, white_mask, height):
    # crop ROI
    left_roi = crop_roi(white_mask, dir='left')
    center_roi = crop_roi(yellow_mask, dir='center')
    right_roi = crop_roi(white_mask, dir='right')

    left_line = get_HoghLine(left_roi)
    center_line = get_HoghLine(center_roi)
    right_line = get_HoghLine(right_roi)

    left_x = get_offset_point(left_line, height)
    center_x = get_offset_point(center_line, height)
    right_x= get_offset_point(right_line, height)
    
    return left_x, center_x, right_x
    

def crop_roi(mask, dir:str):
    height, width = mask.shape
    tmp = np.zeros_like(mask)

    if dir == 'left':
        polygon = np.array([[(0, height * 3/8),
                             (width * 1/2, height * 3/8),
                             (width * 1/2, height), 
                             (0, height)]], np.int32)
    elif dir == 'center':
        polygon = np.array([[(width * 1/4, height * 3/8),
                             (width * 3/4, height * 3/8), 
                             (width * 3/4, height),
                             (width * 1/4, height)]], np.int32)
    elif dir == 'right':
        polygon = np.array([[(width * 1/2, height * 3/8),
                             (width , height * 3/8), 
                             (width , height),
                             (width * 1/2, height)]], np.int32)    

    cv2.fillPoly(tmp, polygon, 255)

    roi_mask = cv2.bitwise_and(mask, tmp)
    return roi_mask    

def get_HoghLine(img):
    k_threshold = 0.8  
    # get HoughLine
    lines =  cv2.HoughLinesP(img,1, np.pi/180, threshold=10, minLineLength=8, maxLineGap=8)

    if lines is not None:    
        # calculate the k in the line( y = k*x + b )
        k = []
        for line in lines:
            k.append((line[0][1] - line[0][3])/(line[0][0] - line[0][2]))
        k_mean = np.mean(np.array(k))
        k = abs(k - k_mean)
        id = np.where(k < k_threshold)[0]

        if len(id) == 0:
            return None

        line_x = []
        line_y = []
        for i in id:
            line_x.append(lines[i][0][0])
            line_x.append(lines[i][0][2])
            line_y.append(lines[i][0][1])
            line_y.append(lines[i][0][3])
        return line_x, line_y
    else:
        return None

def linear_regression(line):
    if line is not None:
        k, b = np.polyfit(line[0], line[1], 1)
        return k, b, True
    else:
        return False, False, False

def get_offset_point(line, height):
    k, b, flag = linear_regression(line)
    if flag:
        x = (height/2 - b)/k
        return x
    else:
        return flag


def offset_control(left_x, center_x, right_x, height, width):
    if left_x and center_x and right_x:
        offset_x = (left_x + center_x + right_x)/3 - width/2
    elif not center_x : 
        offset_x = (left_x + right_x)/2 - width/2
    else:
        offset_x = center_x - width/2
    
    offset_y = height/2
    offset_angle = int(math.atan(offset_x / offset_y) * 180.0 / math.pi) 
    control_angle = offset_angle/40
    return control_angle



def spilt_data(ratio, train_path, val_path):
    # gerate the val dataset and val data_info.json
    data_list = os.listdir(train_path)
    random.seed(256)
    random.shuffle(data_list)
    val_idx = int(len(data_list)*ratio)
    val_data = data_list[:val_idx]
    val_data_info = {}
    i = 0
    for val in val_data:
        val_info = {}
        shutil.copy2(os.path.join(train_path,val), val_path)
        val_info["img"] = os.path.join(val_path, val)
        val_info['dir'] = float(val.split('.')[0].split('_')[-1])
        val_data_info[str(i)] = val_info
        i += 1

    with open(os.path.join(val_path, "data_info.json"), 'w') as json_file:
        json.dump(val_data_info, json_file)

    Data_logger.info("--- Data split over ---")
    Data_logger.info(" Train data: {}".format(len(data_list)))
    Data_logger.info(" Val data: {}".format(val_idx))

    Data_logger.info(" --------- Data collect and split successfully ---------")

if __name__ == "__main__":
    args = get_args()
    Data_logger.info('Data collection Config: ')
    Data_logger.info("                        data num in one epoch:{}".format(args.num))
    Data_logger.info("                        epoch:{}".format(args.epoch))
    Data_logger.info("                        val ratio:{}".format(args.ratio))
    generate_dataset(args.num, args.path, args.speed, args.ratio, args.epoch)

