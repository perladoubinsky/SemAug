export CUDA_VISIBLE_DEVICES=$1
python generate_augmentations.py \
--dataset fsc147 \
--data_path ../data/FSC147 \
--aug control-net-img2img \
--num_synthetic 5 \
--model_config ControlNet/models/cldm_v15.yaml \
--model_path ControlNet/pretrained/blip2_350.ckpt \
--guidance_scale 2.0 \
--steps 20 \
--t0 0.5 \
--captions captions/FSC147_captions_blip2_train.npy 

