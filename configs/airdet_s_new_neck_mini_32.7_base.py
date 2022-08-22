#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
from airdet.config import Config as MyConfig

class Config(MyConfig):
    def __init__(self):
        super(Config, self).__init__()

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.miscs.eval_interval_epochs = 10

        # self.training.resume_path = './workdirs/airdet_s_new_neck_mini_I/epoch_235_ckpt.pth'

        self.dataset.train_ann = ('coco_2017_train_mini',)

        self.model.backbone.act = 'relu'
        self.model.backbone.wid_mul = 0.5
        self.model.backbone.reparam = True

        GiraffeNeckV2 = {"name": "GiraffeNeckV2",
            "depth": 0.5,
            "width": 0.5,
            "in_features": [2, 3, 4],
            "in_channels": [256, 512, 1024],
            "out_channels": [192, 384, 768],
            "depthwise": False,
            "act": "relu",
            "spp": False,
            "reparam_mode": True,
            "block_name": 'BasicBlock_3x3_Reverse',
        }

        self.model.neck = GiraffeNeckV2

        self.model.head.in_channels =[96, 192, 384]

        self.model.head.stacked_convs = 0
        self.model.head.act = 'silu'
        self.model.head.use_ese = False
        self.model.head.conv_groups = 1
        self.model.head.feat_channels = [96, 192, 384]
        self.model.head.reg_max = 16




