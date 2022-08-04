#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
from airdet.config import Config as MyConfig

class Config(MyConfig):
    def __init__(self):
        super(Config, self).__init__()

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        GiraffeNeckV2 = {"name": "GiraffeNeckV2",
        "depth": 0.33,
        "width": 0.5,
        "in_features": [2, 3, 4],
        "in_channels": [256,512,1024],
        "depthwise": False,
        "act": "silu",
        "spp": False,
        }
       
        self.model.neck = GiraffeNeckV2

        self.model.head.in_channels =[128, 256, 512]

        self.model.head.stacked_convs = 1
        self.model.head.act = 'relu'
        self.model.head.use_ese = True
        self.model.head.conv_groups = 1
        self.model.head.feat_channels = [128, 256, 512]


      

