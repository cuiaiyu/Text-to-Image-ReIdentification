export CKPT_ROOT=checkpoints

python train.py \
--data_root "/path/to/data" \
--model_path "checkpoints/reID/baseline" \
--output_path "outputs/reID/baseline" \
--embed_size 1024 \
--batch_size 96 \
--dist_fn_opt "cosine" \
--cap_embed_type sent \
--image_melt_layer 6 \
--step_size 10 \
--load_ckpt_fn "0" \
--num_epochs_stage1 10 \
--num_epochs_stage2 20 \
--weight_decay 1e-5 \
--sent_token_length 60 \
--np_token_length 6 \
--num_np_per_sent 10 \
--np true \
--img_num_cut 6 \
--debug false