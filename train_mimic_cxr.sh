python -u main_train.py \
    --device 1 \
    --image_dir /root/paddlejob/workspace/liujunyi_space/liujunyi_new_work_space/liujunyi05/mimic_cxr/images \
    --ann_path /root/paddlejob/workspace/liujunyi_space/liujunyi_new_work_space/liujunyi05/biyelunwen/MGCA/mimic_abn_with_retrival_reports.json \
    --dataset_name mimic_cxr \
    --max_seq_length 110 \
    --threshold 10 \
    --epochs 30 \
    --batch_size 16 \
    --lr_ve 1e-4 \
    --lr_ed 5e-4 \
    --step_size 3 \
    --gamma 0.8 \
    --num_layers 3 \
    --topk 32 \
    --cmm_size 2048 \
    --cmm_dim 512 \
    --seed 9153 \
    --beam_size 3 \
    --save_dir results/mimic_cxr/ \
    --log_period 1000 \
    > results/有反转_bs16_mimic_cxr_abn_纯正面图_device_1.log 2>&1 &
        
    #--use_rebuild_data \
     
