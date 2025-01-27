# sudo apt install nvidia-cuda-toolkit  

accelerate launch train_controlnet_flux.py \
 --pretrained_model_name_or_path="black-forest-labs/FLUX.1-schnell" \
 --output_dir="model_out" \
 --dataset_name="raulc0399/open_pose_controlnet" \
 --conditioning_image_column=conditioning_image \
 --image_column=image \
 --caption_column=text \
 --resolution=512 \
 --learning_rate=1e-5 \
 --train_batch_size=4 \
 --num_train_epochs=3 \
 --tracker_project_name="controlnet" \
 --enable_xformers_memory_efficient_attention \
 --checkpointing_steps=5000 \
 --validation_steps=5000 \
 --report_to wandb \
 --push_to_hub \
 --gradient_accumulation_steps=4 \
 --gradient_checkpointing \
 --use_8bit_adam \ 
 --set_grads_to_none \
 --enable_model_cpu_offload