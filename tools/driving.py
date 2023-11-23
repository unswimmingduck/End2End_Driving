import gym
import gym_donkeycar
import argparse

import numpy as np

import logging

from e2e_cnn.model.e2e_cnn import e2e_net
import torch
from torchvision import transforms

import cv2




logger = logging.getLogger("Self-Driving")
logger.setLevel(logging.DEBUG)



def get_args():
    parser = argparse.ArgumentParser('E2E_CNN')
    parser.add_argument('--checkpoint', type=str, default="checkpoint\epoch_70.path", help='path to load model you train')
    parser.add_argument('--speed', type=float, default=0.3, help='path to load model you train')
    parser.add_argument('--frames', type=int, default=1000, help='number of frames in one epoch self-driving')
    args = parser.parse_args()
    # filter the input info 
    try:
        f = open(args.checkpoint, 'r')
    except IOError:
        print("Please input the right path of checkpoint")

    try:
        if args.speed < 0:
            raise Exception()
    except IOError:
        print("Please input the speed > 0")

    try:
        if args.frames < 0:
            raise Exception()
    except IOError:
        print("Please input the frames > 0")
    return args


def main():
    args = get_args()
    env = gym.make("donkey-generated-roads-v0")
    obs = env.reset()
    speed = args.speed
    dir = 0
    action = np.array([dir, 0])
    model = model_init(args)
    logger.info('---------------- E2E_CNN model load successfully ----------------')
    img, reward, done, info = env.step(action)

    for _ in range(args.frames):
        # process the img
        tensor = img_process(img)
        angle = model.validate(tensor).cpu()
        print(angle.item())
        action = np.array([angle.item(), speed])
        
        img, reward, done, info = env.step(action)



def model_init(args):
    # load the weight to model
    model = e2e_net().cuda()
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model'])
    
    return model



def img_process(img):
    # convert BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # conver img to tensor
    to_tensor = transforms.ToTensor()
    tensor = to_tensor(img).unsqueeze(0).cuda()
    
    return tensor




if __name__ == "__main__":
    main()