# python -m torch.distributed.launch --nproc_per_node=8 tools/distiller.py --tea_config configs/airdet_m.py --stu_config configs/airdet_s.py --tea_ckpt airdet_m.pth --d mimic --loss_weight 0.5
# python -m torch.distributed.launch --nproc_per_node=8 tools/train.py -f configs/airdet_s.py
# python -m torch.distributed.launch --nproc_per_node=8 tools/distiller.py --tea_config configs/airdet_s.py --stu_config configs/distill_align_mosaic.py --tea_ckpt airdet_s_438_ckpt.pth --d mimic --loss_weight 20
# python -m torch.distributed.launch --nproc_per_node=8 tools/distiller.py --tea_config configs/airdet_s.py --stu_config configs/distill_mgd_15_grad.py --tea_ckpt airdet_s_438_ckpt.pth --d mgd --loss_weight 15
# python -m torch.distributed.launch --nproc_per_node=8 tools/distiller.py --tea_config configs/airdet_m.py --stu_config configs/airdet_s.py --tea_ckpt airdet_m.pth --d cwd --loss_weight 20
# python -m torch.distributed.launch --nproc_per_node=8 tools/distiller.py --tea_config configs/airdet_m.py --stu_config configs/distill_cwd_m2s_300ep.py --tea_ckpt epoch_290_ckpt.pth --d cwd --loss_weight 1
# python -m torch.distributed.launch --nproc_per_node=8 tools/distiller.py --tea_config configs/airdet_m.py --stu_config configs/distill_cwd_m2s_300ep.py --tea_ckpt epoch_290_ckpt.pth --d cwd --loss_weight 2
# python -m torch.distributed.launch --nproc_per_node=8 tools/eval.py -f configs/airdet_s.py --ckpt airdet_s_280_ckpt.pth
# python -m torch.distributed.launch --nproc_per_node=8 tools/distiller.py --tea_config configs/airdet_m.py --stu_config configs/airdet_s_m.py --tea_ckpt airdet_m.pth --d cwd --loss_weight 0
# python -m torch.distributed.launch --nproc_per_node=8 tools/train.py -f configs/airdet_m.py

# from scratch
python -m torch.distributed.launch --nproc_per_node=8 tools/distiller.py --tea_config configs/airdet_m.py --stu_config configs/airdet_s.py --tea_ckpt epoch_290_ckpt.pth --d cwd --loss_weight 20

# 280
# python -m torch.distributed.launch --nproc_per_node=8 tools/distiller.py --tea_config configs/airdet_m.py --stu_config configs/distill_cwd_m2s_280ep.py --tea_ckpt epoch_290_ckpt.pth --d cwd --loss_weight 10