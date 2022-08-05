#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
from airdet.config import Config as MyConfig

class Config(MyConfig):
    def __init__(self):
        super(Config, self).__init__()

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.miscs.eval_interval_epochs = 10

        self.training.train_ann = ('coco_2017_train_mini',)

        self.model.backbone.act = 'relu'
        self.model.backbone.wid_mul = 0.5

        GiraffeNeckV2 = {"name": "GiraffeNeckV2",
        "depth": 0.33,
        "width": 0.5,
        "in_features": [2, 3, 4],
        "in_channels": [256,512,1024],
        "depthwise": False,
        "act": "relu",
        "spp": False,
        }

        self.model.neck = GiraffeNeckV2

        self.model.head.in_channels =[128, 256, 512]

        self.model.head.stacked_convs = 1
        self.model.head.act = 'relu'
        self.model.head.use_ese = False
        self.model.head.conv_groups = 1
        self.model.head.feat_channels = [128, 256, 512]




