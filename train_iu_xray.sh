device=0

python main_train.py\
    --device ${device} \
    --image_dir /root/R2G/data/iu_xray_data/images/ \
    --ann_path /root/R2G/data/iu_xray_data/iu_xray_with_retrival_reports_using_image_features.json \
    --dataset_name iu_xray \
    --max_seq_length 60 \
    --threshold 3 \
    --epochs 70 \
    --batch_size 16 \
    --lr_ve 5e-5 \
    --lr_ed 3e-4 \
    --step_size 10 \
    --gamma 0.8 \
    --num_layers 3 \
    --topk 32 \
    --cmm_size 2048 \
    --cmm_dim 512 \
    --seed 1229 \
    --beam_size 3 \
    --save_dir results/iu_xray0823_bs8_layer5_CMN/ \
    --log_period 50 \
    --early_stop 40 \
    >log/6.22/transformer_5e_5_3e_4_seed_1229_bs_16.log 2>&1 &
    #> log/6.22/lr_ve_5e_5_lr_ed_1e_4_seed_1229_bs32_device_1.log 2>&1 &
