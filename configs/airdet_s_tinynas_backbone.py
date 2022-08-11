#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
from airdet.config import Config as MyConfig

class Config(MyConfig):
    def __init__(self):
        super(Config, self).__init__()

        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.miscs.eval_interval_epochs = 10

        self.dataset.train_ann = ('coco_2017_train_mini',)

        GiraffeNeckV2 = {"name": "GiraffeNeckV2",
        "depth": 0.33,
        "width": 0.5,
        "in_features": [2, 3, 4],
        "in_channels": [256,512,1024],
        "depthwise": False,
        "act": "relu",
        "spp": False,
        "reparam_mode": True,
        }
        self.model.neck = GiraffeNeckV2

        backbone_struct = '''
        [ {'class': 'ConvKXBNRELU', 'in': 3, 'k': 3, 'nbitsA': 8, 'nbitsW': 8, 'out': 32, 's': 2},
          { 'L': 1,
            'btn': 16,
            'class': 'SuperResConvK1KX',
            'in': 32,
            'inner_class': 'ResConvK1KX',
            'k': 3,
            'nbitsA': [8, 8, 8],
            'nbitsW': [8, 8, 8],
            'out': 80,
            's': 2},
          { 'L': 3,
            'btn': 40,
            'class': 'SuperResConvK1KX',
            'in': 80,
            'inner_class': 'ResConvK1KX',
            'k': 3,
            'nbitsA': [8, 8, 8, 8, 8, 8, 8, 8, 8],
            'nbitsW': [8, 8, 8, 8, 8, 8, 8, 8, 8],
            'out': 160,
            's': 2},
          { 'L': 3,
            'btn': 80,
            'class': 'SuperResConvK1KX',
            'in': 160,
            'inner_class': 'ResConvK1KX',
            'k': 3,
            'nbitsA': [8, 8, 8, 8, 8, 8, 8, 8, 8],
            'nbitsW': [8, 8, 8, 8, 8, 8, 8, 8, 8],
            'out': 320,
            's': 2},
          { 'L': 3,
            'btn': 80,
            'class': 'SuperResConvK1KX',
            'in': 320,
            'inner_class': 'ResConvK1KX',
            'k': 3,
            'nbitsA': [8, 8, 8, 8, 8, 8, 8, 8, 8],
            'nbitsW': [8, 8, 8, 8, 8, 8, 8, 8, 8],
            'out': 320,
            's': 1},
          { 'L': 3,
            'btn': 152,
            'class': 'SuperResConvK1KX',
            'in': 320,
            'inner_class': 'ResConvK1KX',
            'k': 3,
            'nbitsA': [8, 8, 8, 8, 8, 8, 8, 8, 8],
            'nbitsW': [8, 8, 8, 8, 8, 8, 8, 8, 8],
            'out': 640,
            's': 2}]
        '''
        TinyNAS = {
            "name": "TinyNAS",
            "net_structure_str": backbone_struct,
            "out_indices": (0, 1, 2, 4, 5),
            "out_channels": (None, None, 128, 256, 512),
            "with_spp": True,
            "use_focus": True,
            "act": "relu",
        }

        self.model.backbone = TinyNAS

        self.model.head.in_channels =[128, 256, 512]
        self.model.head.stacked_convs = 1
        self.model.head.act = 'relu'
        self.model.head.use_ese = False
        self.model.head.conv_groups = 1
        self.model.head.feat_channels = [128, 256, 512]

