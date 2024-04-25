export CUDA_VISIBLE_DEVICES=$1
python generate_augmentations.py \
--dataset fsc147 \
--data_path ../data/FSC147 \
--aug real-guidance \
--model_path runwayml/stable-diffusion-v1-5 \
--num_synthetic 5 \
--guidance_scale 7.5 \
--steps 20 \
--t0 0.5 \
--prompt_template 'a photo of a {name}'
