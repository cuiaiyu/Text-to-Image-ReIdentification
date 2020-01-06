export CUDA_VISIBLE_DEVICES=3
export CKPT_ROOT=/shared/rsaas/aiyucui2/wider_person/checkpoints/reID/baseline
export CKPT_EXP="dist_fn_cosine_imgbb_resnet50_capbb_bigru_embed_size_1024_batch_96_lr_0.0001_captype_sent_img_meltlayer_2_cos_margin_0.2_np_False_match_only"
# dist_fn_cosine_imgbb_resnet18_capbb_bigru_embed_size_512_batch_96_lr_0.0001_captype_sent_img_meltlayer_8_cos_margin_0.2_np_False
# dist_fn_cosine_imgbb_resnet18_capbb_bigru_embed_size_512_batch_96_lr_0.0001_captype_sent_img_meltlayer_8

python train.py \
--data_root "/data/aiyucui2/wider" \
--model_path "/shared/rsaas/aiyucui2/wider_person/checkpoints/reID/baseline" \
--output_path "/shared/rsaas/aiyucui2/wider_person/outputs/reID/baseline" \
--embed_size 1024 \
--batch_size 96 \
--num_gpus 1 \
--img_backbone_opt resnet50 \
--cap_backbone_opt bigru \
--dist_fn_opt "cosine" \
--cap_embed_type sent \
--image_melt_layer 8 \
--cos_margin 0.2 \
--load_ckpt_fn $CKPT_ROOT/$CKPT_EXP/stage_1_match_last.pt  \
--num_epochs_stage1 0 \
--num_epochs_stage2 15 \
--np false \
--img_num_cut 6 \
--debug false