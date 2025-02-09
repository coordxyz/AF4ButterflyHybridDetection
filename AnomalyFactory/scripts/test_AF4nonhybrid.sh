set -ex

#---------------------AnomalyFactory-------------------------
CUDA_VISIBLE_DEVICES=0
python test.py --dataroot ./Datasets/Images4TrainAFnonhybrid/List/test_nonhybrid/ \
--primitive seg_edges --phase "test" --no_instance \
--results_dir ./results/butterflyContest/nh/ \
--input_nc 6 --encoder_nc 6 --decoder_nc 3 --output_nc 3 --kRef 1 \
--tps_aug 0 --tps_percent 0.99 \
--netG AFsplitglobal \
--how_many 5 --gpu_ids 0 \
--name AF4nonhybrid \
--train_mode BootAF_I2DE \
--test_mode BootAF_I2DE \
--use_edgemask 0.0 --refImgExt png \