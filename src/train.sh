export CUDA_VISIBLE_DEVICES=1
export CKPT_ROOT=/shared/rsaas/aiyucui2/wider_person/checkpoints/reID
export CKPT_EXP=dist_fn_cosine_imgbb_resnet18_capbb_bigru_embed_size_512_batch_96_lr_0.0001_captype_sent_img_meltlayer_8

python aiyustuff.py \
--data_root "/data/aiyucui2/wider" \
--embed_size 1024 \
--batch_size 96 \
--num_gpus 1 \
--img_backbone_opt resnet50 \
--cap_backbone_opt bigru \
--dist_fn_opt "cosine" \
--cap_embed_type word \
--image_melt_layer 1 \
--cos_margin 0.2 \
--load_ckpt_fn 0 \
--num_epochs_stage1 20 \
--debug false