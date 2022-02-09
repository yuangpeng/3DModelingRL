import numpy as np
from numpy.lib.function_base import delete
import torch
import scipy.ndimage as sn
import copy
from PIL import Image
from utils import utils
import config as p
import gym
from gym import Env
from gym.spaces import Box
import random
import math
from typing import Optional
from gym import spaces
from gym.utils import seeding
GT_VOX_RESO_L = 16
GT_VOX_RESO_H = 64
REF_IMG_RESO = 128

BOX_NUM = 27
ACTION_NUM = 756
LOOP_NUM = 10
MAX_STEP = 300

# hyperparamters for training
BATCH_SIZE = 64
LR = 0.00008                 # learning rate
EPSILON = 0.98               # greedy policy
GAMMA = 0.9                  # reward discount
TARGET_REPLACE_ITER = 4000   # target update frequency
MEMORY_LONG_CAPACITY = 2000
MEMORY_SELF_CAPACITY = 1000  # shared by D_short and D_self
DAGGER_EPOCH = 1
DAGGER_ITER = 4
DAGGER_LEARN = 4000
RL_EPOCH = 2000

COLORS = [[166, 206, 227], [31, 120, 180], [178, 223, 138], [51, 160, 44],
          [251, 154, 153], [227, 26, 28], [253, 191, 111], [255, 127, 0],
          [166, 216, 84], [106, 61, 154], [255, 255, 153], [177, 89, 40],
          [102, 194, 165], [252, 141, 98], [141, 160, 203], [231, 138, 195],
          [166, 216, 84], [255, 217, 47], [27, 158, 119], [217, 95, 2],
          [117, 112, 179], [231, 41, 138], [102, 166, 30], [230, 171, 2],
          [166, 118, 29], [102, 102, 102], [229, 196, 148]]

class Environment():
    def __init__(self):
        #模型初始参数
        self.vox_size_l=GT_VOX_RESO_L
        self.vox_size_h=GT_VOX_RESO_H
        self.ref_size=REF_IMG_RESO
        self.box_num=BOX_NUM
        self.action_num=ACTION_NUM
        self.max_step=MAX_STEP
        self.loop_num=LOOP_NUM
        
        self.ref= np.zeros((1, self.ref_size, self.ref_size))
        self.target = np.zeros((1, self.vox_size_l, self.vox_size_l, self.vox_size_l))
        self.target_h = np.zeros((1, self.vox_size_l, self.vox_size_l, self.vox_size_l))        
        self.target_points = np.zeros((1, 3))
        self.target_h_points = np.zeros((1, 3))

        self.init_boxes = self.initialize_box()
        self.all_boxes = copy.copy(self.init_boxes)
        
        self.name = 'Shit Mountain of chimera'
        
        self.last_IOU = 0
        self.last_delete = 0
        self.last_local_IOU = 0
        self.step_count = 0
        self.step_vec=np.zeros((self.max_step), dtype=np.int)  
        
        #action
        
        max_action = np.ones((162,), dtype = np.float32)
        self.max_action = max_action
        min_action =np.full((162,),-1,np.float32,)
        
        self.min_action = min_action
        
        self.action_space = spaces.Box(
            low=self.min_action, high=self.max_action, dtype=np.float32
        )
        #observation
        def initialize_box(self):
          padding = 2
          size = int((self.vox_size_l - padding) / 3.0)
          init_boxes = np.zeros((self.box_num, 6), dtype=np.float32)
          count = 0
          for i in range(3):
            for j in range(3):
                for k in range(3):
                    x, y, z = padding + i * size, padding + j * size, padding + k * size
                    x_, y_, z_ = x + size, y + size, z + size
                    init_boxes[count] = [x, y, z, x_, y_, z_]
                    count+=1
          return init_boxes
       
        self.low_state = np.full((162,),-16,np.float32,)
        self.high_state =np.full((162,),16,np.float32,)
        self.observation_space = spaces.Box(
            low=self.low_state, high=self.high_state, dtype=np.float32
        )
        #inite state
        booox=initialize_box(self)
        self.state =np.reshape(booox,(162,),)
        self.colors = np.array(p.COLORS)
    
    
      
    
    
        
        
    
    def step(self,action):
        self.state += action

        
        IOU, local_IOU= self.compute_increment(self.all_boxes)
        reward=self.compute_reward(IOU, local_IOU,delete_count)
        
        self.last_IOU = IOU
        self.last_delete = delete_count
        self.last_local_IOU = local_IOU        
        
        self.step_count += 1
        self.step_vec=np.zeros((self.max_step), dtype=np.int)
        
        if self.step_count==self.max_step:
            done=True
        else:
            done=False
            self.step_vec[self.step_count]=1
        info={self.ref, self.all_boxes/self.vox_size_l, self.step_vec}
        return self.state, reward, done,info
    
    
 
    
    def compute_increment(self, box):
        # voxelize the boxes
        # canvas -> 和 self.target 张一样的全0矩阵
        # target -> target_h
        canvas = np.zeros_like(self.target_h, dtype=np.int)

        # box 大小限定在[0, self.vox_size_l - 1] -> clip_box
        clip_box = np.clip(box, 0, self.vox_size_l-1)
        # box -> canvas
        left = lambda x: int(x * 4) + round(math.ceil(x * 4) - x * 4)
        right = lambda x: int(x * 4) + round(x * 4 - math.floor(x * 4))
        for i in range(self.box_num):
            [x, y, z, x_, y_, z_] = clip_box[i][0:6]
            canvas[left(x):right(x_), left(y):right(y_), left(z):right(z_)] = 1

        intersect = canvas & self.target_h
        i_count = np.sum(intersect == 1)

        union = canvas | self.target_h
        u_count = np.sum(union == 1)

        delete_count = 0
        sum_single_iou = 0
        for i in range(0, 162, 6):
          if self.state[i]>= self.state[i+3] or self.state[i+1]>=self.state[i+4] or self.state[i+2]>= self.state[i+5]:
            delete_count+=1
          else:
            single_canvas = np.zeros((self.vox_size_h,self.vox_size_h,self.vox_size_h), dtype=np.int)
            [x, y, z, x_, y_, z_] = clip_box[i][0:6]
            single_canvas[left(x):right(x_), left(y):right(y_), left(z):right(z_)] = 1
            single_intersect = single_canvas & self.target_h
            s_i_count = np.sum(single_intersect == 1)
            
            single_union = single_canvas | self.target_h
            s_u_count = np.sum(single_union == 1)

            local_iou = float(s_i_count) / float(s_u_count)
            sum_single_iou += local_iou

        iou=float(i_count)/float(u_count)
        
        if delete_count==self.box_num:
            local_iou = 0
        else:
            local_iou = sum_single_iou / (self.box_num - delete_count)
            
        return iou, local_iou, delete_count
    def compute_reward(self, iou, local_iou, delete_count):
        
        r_iou = iou - self.last_IOU
        r_local = local_iou - self.last_local_IOU
        r_parsimony = delete_count - self.last_delete
        
        a=0.1
        b=0.01
        
        reward = r_iou + a*r_local + b*r_parsimony 
        if reward==0:
          reward -=1
        return reward
    
    
    
    
    def render(self):
      pass
    def reset(self, shape_infopack):
        def initialize_box(self):
          padding = 2
          size = int((self.vox_size_l - padding) / 3.0)
          init_boxes = np.zeros((self.box_num, 6), dtype=np.float32)
          count = 0
          for i in range(3):
            for j in range(3):
                for k in range(3):
                    x, y, z = padding + i * size, padding + j * size, padding + k * size
                    x_, y_, z_ = x + size, y + size, z + size
                    init_boxes[count] = [x, y, z, x_, y_, z_]
                    count+=1
          return init_boxes
        boox=initialize_box(self)
        self.state =np.reshape(boox,(162,),)
        #the information data of the new shape
        self.name, vox_l_fn, vox_h_fn, ref_type, ref_fn = shape_infopack

        #reset all
        self.all_boxes = copy.copy(self.init_boxes)
        self.last_IOU = 0
        self.last_local_IOU = 0
        self.last_delete = 0
        self.step_count = 0
        self.step_vec=np.zeros((self.max_step), dtype=np.int)
        
        #load reference image
        img = Image.open(ref_fn)
        if ref_type=='rgb':
            image = Image.new('RGB', size=(600, 600), color=(255,255,255))
            image.paste(img, (0, 0), mask=img)
            img = image
        
        #process and reset reference image
        img = img.convert('L')        
        img = img.resize((self.ref_size, self.ref_size), Image.ANTIALIAS)
        self.raw_img=copy.copy(img)
        img = np.array(img)
        img = np.expand_dims(img, axis=0)
        self.ref=img/255.0
        
        #load groundtruth data
        shape=utils.load_voxel_data(vox_l_fn).astype(np.int)
        shape_h=utils.load_voxel_data(vox_h_fn).astype(np.int)
       
        #process and reset groundtruth
        shape = sn.binary_dilation(shape)
        shape_h= sn.binary_dilation(shape_h)
        self.target_points = np.argwhere(shape == 1)
        self.target_h_points = np.argwhere(shape_h == 1)
        self.target = shape
        self.target_h = shape_h
        
        return self.state,self.ref, self.all_boxes/self.vox_size_l, self.step_vec
        
        
    
        
    
    


