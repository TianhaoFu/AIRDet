#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import os
from airdet.config import Config as MyConfig

class Config(MyConfig):
    def __init__(self):
        super(Config, self).__init__()
        self.exp_name = "distill_cwd_m2s_36ep_30_label_assign"
        # debug config
        self.training.total_epochs = 36
        self.training.no_aug_epochs = 0
        self.training.augmentation.use_autoaug = False
        self.training.augmentation.mosaic = False
        self.training.lr_scheduler = "multi_step"
        
        self.training.images_per_batch = 128