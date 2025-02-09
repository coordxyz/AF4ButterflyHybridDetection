set -ex
#-----------------------------------------------
#--------Anomaly Factory
python train.py --dataroot ./Datasets/Images4TrainAFhybrid/List/train_AFhybrid/ \
--primitive seg_edges \
--no_instance --tps_aug 1 --tps_percent 0.99 --gpu_ids 0,1,2,3 \
--batchSize 40 --ngf 64 \
--name AF4hybrid \
--block_aug --reflect_aug \
--save_latest_freq 1000 --display_freq 100 \
--input_nc 6 --encoder_nc 6 --decoder_nc 3 --output_nc 3 --kRef 1 \
--netG AFsplitglobal \
--use_edgemask 0.0 \
--mask_loss_weight 10 \
--train_mode BootAF_I2DE \
# --load_pretrain ./checkpoints/AF4hybrid \
# --continue_train



