# DATASET_NAME="CUHK-PEDES" # ICFG-PEDES, RSTPReid, CUHK-PEDES

# CUDA_VISIBLE_DEVICES=0,1,2,3 \
# python -m torch.distributed.run --nproc_per_node=3 \

# CUDA_VISIBLE_DEVICES=3 \
# python train.py \
# --name dyn_low_pre2 \
# --img_aug \
# --MLM \
# --batch_size 64 \
# --dataset_name $DATASET_NAME \
# --loss_names 'itc+cdm+chm' \
# --num_epoch 60

# --resume \
# --resume_ckpt_file '/opt/data/private/crossreid/ICPG-main/logs/CUHK-PEDES/20250213_152326_icpg/best.pth'

## ICFG-PEDES, RSTPReid

DATASET_NAME="RSTPReid" # ICFG-PEDES, RSTPReid, CUHK-PEDES

CUDA_VISIBLE_DEVICES=1 \
python train.py \
--name dyn_pre \
--img_aug \
--MLM \
--batch_size 64 \
--dataset_name $DATASET_NAME \
--loss_names 'itc+cdm+chm' \
--num_epoch 60

# DATASET_NAME="ICFG-PEDES" # ICFG-PEDES, RSTPReid, CUHK-PEDES

# CUDA_VISIBLE_DEVICES=2 \
# python train.py \
# --name dyn_low_pre2 \
# --img_aug \
# --MLM \
# --batch_size 64 \
# --dataset_name $DATASET_NAME \
# --loss_names 'itc+cdm+chm' \
# --num_epoch 60



# --loss_names 'itc+cdm+chm' \


