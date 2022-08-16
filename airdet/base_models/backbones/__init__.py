# Copyright (C) Alibaba Group Holding Limited. All rights reserved.


import copy

from .darknet import CSPDarknet
from .tinynas import load_tinynas_net

def build_backbone(cfg):
    backbone_cfg = copy.deepcopy(cfg)
    name = backbone_cfg.pop("name")
    if name == "CSPDarknet":
        return CSPDarknet(**backbone_cfg)
    elif name == "TinyNAS":
        return load_tinynas_net(backbone_cfg)
