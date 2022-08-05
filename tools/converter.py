#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (C) Alibaba Group Holding Limited. All rights reserved.

import argparse
import os
from loguru import logger

import torch
from torch import nn

from airdet.base_models.core.base_ops import SiLU
from airdet.utils.model_utils import replace_module, get_model_info
from airdet.config.base import parse_config
from airdet.detectors.detector_base import Detector, build_local_model
from airdet.base_models.core.neck_ops import RepVGGBlock

def make_parser():
    parser = argparse.ArgumentParser("AIRDet converter deployment toolbox")
    # mode part
    parser.add_argument("--mode", default='onnx', type=str, help="onnx, trt_16 or trt_32")
    # model part
    parser.add_argument(
        "-f",
        "--config_file",
        default=None,
        type=str,
        help="expriment description file",
    )
    parser.add_argument("-c", "--ckpt", default=None, type=str, help="ckpt path")
    parser.add_argument("--trt", action='store_true', help="whether convert onnx into tensorrt")
    parser.add_argument("--half", action='store_true', help="whether use fp16")
    parser.add_argument(
        "--batch_size", type=int, default=None, help="inference image batch nums"
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default="640",
        help="inference image shape"
    )
    # onnx part
    parser.add_argument(
        "--input", default="images", type=str, help="input node name of onnx model"
    )
    parser.add_argument(
        "--output", default="output", type=str, help="output node name of onnx model"
    )
    parser.add_argument(
        "-o", "--opset", default=11, type=int, help="onnx opset version"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    return parser


@logger.catch
def trt_export(onnx_path, batch_size, inference_h, inference_w, half):
    import tensorrt as trt
    import sys
    if half:
        trt_mode = "fp16"
    else:
        trt_mode = "fp32"

    TRT_LOGGER = trt.Logger()
    engine_path = onnx_path.replace('.onnx', f'_{trt_mode}_bs{batch_size}.trt')

    EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

    logger.info(f"trt_{trt_mode} converting ...")
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) \
    as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = 1 << 30
        builder.max_batch_size = batch_size
        print('Loading ONNX file from path {}...'.format(onnx_path))

        if half:
            assert (builder.platform_has_fast_fp16 == True), "not support fp16"
            builder.fp16_mode = True

        with open(onnx_path, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                print ('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))

        network.get_input(0).shape = [batch_size, 3, inference_h, inference_w]
        print('Completed parsing of ONNX file')
        engine = builder.build_cuda_engine(network)
        with open(engine_path, "wb") as f:
            f.write(engine.serialize())
        logger.info("generated trt engine named {}".format(engine_path))


@logger.catch
def main():
    args = make_parser().parse_args()
    logger.info("args value: {}".format(args))

    output_name = os.path.join('deploy', args.ckpt.replace('.pth', '.onnx'))
    # init and load model
    config = parse_config(args.config_file)
    config.merge(args.opts)

    if args.batch_size is not None:
        config.testing.images_per_batch = args.batch_size

    # build model
    model = build_local_model(config, "cuda")
    print(model)
    info = get_model_info(model, (args.img_size, args.img_size))
    logger.info(info)
    # load model paramerters
    #ckpt = torch.load(args.ckpt, map_location="cpu")

    model.eval()
    #if "model" in ckpt:
    #    ckpt = ckpt["model"]
    #model.load_state_dict(ckpt, strict=True)
    logger.info("loading checkpoint done.")

    model = replace_module(model, nn.SiLU, SiLU)

    for layer in model.modules():
        if isinstance(layer, RepVGGBlock):
            layer.switch_to_deploy()

    # decouple postprocess
    model.head.nms = False

    dummy_input = torch.randn(args.batch_size, 3, args.img_size, args.img_size).to("cuda")
    predictions = model(dummy_input)
    torch.onnx._export(
        model,
        dummy_input,
        output_name,
        input_names=[args.input],
        output_names=[args.output],
        opset_version=args.opset,
    )
    logger.info("generated onnx model named {}".format(output_name))

    if args.trt:
        trt_export(output_name, args.batch_size, args.img_size, args.img_size, args.half)

if __name__ == "__main__":
    main()
