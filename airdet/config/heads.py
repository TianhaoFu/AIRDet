# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

GFocalV2 = { "name" : "GFocalV2",
           "num_classes": 80,
           "nms": True,
           "use_ese": False,
           "in_channels": [96, 160, 384],
           "stacked_convs": 4,
           "reg_channels": 64,
           "feat_channels": 96,
           "reg_max": 14,
           "add_mean": True,
           "norm": "bn",
           "act": "silu",
           "start_kernel_size": 3,
           "conv_groups": 2,
           "conv_type": "BaseConv",
           "nms_conf_thre": 0.05,
           "nms_iou_thre": 0.7
         }


yolo_head = {"name": "YOLOX",
             "num_classes": 80,
             "decode_in_inference": True,
             "width": 0.5,
             "strides": [8, 16, 32],
             "in_channels": [256, 512, 1024],
             "act": "silu",
             "depthwise": False,
         }

