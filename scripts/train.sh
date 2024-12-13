
# export CUDA_VISIBLE_DEVICES=0 &&
# python src/train.py \
#     --data_root data/mvtec_ad \
#     --class_name metal_nut \
#     --num_normal_samples -1 \
#     --ae_model conv_ae_1 \
#     --ae_hidden_dim 384 \
#     --bottleneck_dim 64 \
#     --backbone_model pdn_small \
#     --backbone_resume_path ./weights/teacher_small.pth \
#     --feature_dim 384 \
#     --feature_res 1 \
#     --patch_size 1 \
#     --num_enc_layers 3 \
#     --num_dec_layers 3 \
#     --num_heads 4 \
#     --mlp_ratio 4 \
#     --in_res 56 \
#     --img_size 256 \
#     --split train \
#     --batch_size 8 \
#     --num_epochs 100 \
#     --lr 0.0001 \
#     --momentum 0.9 \
#     --weight_decay 0.05 \
#     --optimizer adamw \
#     --scheduler cosine \
#     --grad_clip 1.0 \
#     --seed 46 \
#     --num_workers 1 \
#     --device cuda \
#     --log_interval 10 \
#     --log_eval \
#     --save_dir results & \

# export CUDA_VISIBLE_DEVICES=1 &&
# python src/train.py \
#     --data_root data/mvtec_ad \
#     --class_name tile \
#     --num_normal_samples -1 \
#     --ae_model conv_ae_1 \
#     --ae_hidden_dim 384 \
#     --bottleneck_dim 64 \
#     --backbone_model pdn_small \
#     --backbone_resume_path ./weights/teacher_small.pth \
#     --feature_dim 384 \
#     --feature_res 1 \
#     --patch_size 1 \
#     --num_enc_layers 3 \
#     --num_dec_layers 3 \
#     --num_heads 4 \
#     --mlp_ratio 4 \
#     --in_res 56 \
#     --img_size 256 \
#     --split train \
#     --batch_size 8 \
#     --num_epochs 100 \
#     --lr 0.0001 \
#     --momentum 0.9 \
#     --weight_decay 0.05 \
#     --optimizer adamw \
#     --scheduler cosine \
#     --grad_clip 1.0 \
#     --seed 46 \
#     --num_workers 1 \
#     --device cuda \
#     --log_interval 10 \
#     --log_eval \
#     --save_dir results & \

export CUDA_VISIBLE_DEVICES=2 &&
python src/train.py \
    --data_root data/mvtec_ad \
    --class_name carpet \
    --num_normal_samples -1 \
    --ae_model conv_ae_1 \
    --ae_hidden_dim 384 \
    --bottleneck_dim 64 \
    --backbone_model pdn_small \
    --backbone_resume_path ./weights/teacher_small.pth \
    --feature_dim 384 \
    --feature_res 1 \
    --patch_size 1 \
    --num_enc_layers 3 \
    --num_dec_layers 3 \
    --num_heads 4 \
    --mlp_ratio 4 \
    --in_res 56 \
    --img_size 256 \
    --split train \
    --batch_size 8 \
    --num_epochs 100 \
    --lr 0.0001 \
    --momentum 0.9 \
    --weight_decay 0.05 \
    --optimizer adamw \
    --scheduler cosine \
    --grad_clip 1.0 \
    --seed 46 \
    --num_workers 1 \
    --device cuda \
    --log_interval 10 \
    --log_eval \
    --save_dir results & \

export CUDA_VISIBLE_DEVICES=3 &&
python src/train.py \
    --data_root data/mvtec_ad \
    --class_name wood \
    --num_normal_samples -1 \
    --ae_model conv_ae_1 \
    --ae_hidden_dim 384 \
    --bottleneck_dim 64 \
    --backbone_model pdn_small \
    --backbone_resume_path ./weights/teacher_small.pth \
    --feature_dim 384 \
    --feature_res 1 \
    --patch_size 1 \
    --num_enc_layers 3 \
    --num_dec_layers 3 \
    --num_heads 4 \
    --mlp_ratio 4 \
    --in_res 56 \
    --img_size 256 \
    --split train \
    --batch_size 8 \
    --num_epochs 100 \
    --lr 0.0001 \
    --momentum 0.9 \
    --weight_decay 0.05 \
    --optimizer adamw \
    --scheduler cosine \
    --grad_clip 1.0 \
    --seed 46 \
    --num_workers 1 \
    --device cuda \
    --log_interval 10 \
    --log_eval \
    --save_dir results & \

export CUDA_VISIBLE_DEVICES=4 &&
python src/train.py \
    --data_root data/mvtec_ad \
    --class_name leather \
    --num_normal_samples -1 \
    --ae_model conv_ae_1 \
    --ae_hidden_dim 384 \
    --bottleneck_dim 64 \
    --backbone_model pdn_small \
    --backbone_resume_path ./weights/teacher_small.pth \
    --feature_dim 384 \
    --feature_res 1 \
    --patch_size 1 \
    --num_enc_layers 3 \
    --num_dec_layers 3 \
    --num_heads 4 \
    --mlp_ratio 4 \
    --in_res 56 \
    --img_size 256 \
    --split train \
    --batch_size 8 \
    --num_epochs 100 \
    --lr 0.0001 \
    --momentum 0.9 \
    --weight_decay 0.05 \
    --optimizer adamw \
    --scheduler cosine \
    --grad_clip 1.0 \
    --seed 46 \
    --num_workers 1 \
    --device cuda \
    --log_interval 10 \
    --log_eval \
    --save_dir results & \

export CUDA_VISIBLE_DEVICES=5 &&
python src/train.py \
    --data_root data/mvtec_ad \
    --class_name hazelnut \
    --num_normal_samples -1 \
    --ae_model conv_ae_1 \
    --ae_hidden_dim 384 \
    --bottleneck_dim 64 \
    --backbone_model pdn_small \
    --backbone_resume_path ./weights/teacher_small.pth \
    --feature_dim 384 \
    --feature_res 1 \
    --patch_size 1 \
    --num_enc_layers 3 \
    --num_dec_layers 3 \
    --num_heads 4 \
    --mlp_ratio 4 \
    --in_res 56 \
    --img_size 256 \
    --split train \
    --batch_size 8 \
    --num_epochs 100 \
    --lr 0.0001 \
    --momentum 0.9 \
    --weight_decay 0.05 \
    --optimizer adamw \
    --scheduler cosine \
    --grad_clip 1.0 \
    --seed 46 \
    --num_workers 1 \
    --device cuda \
    --log_interval 10 \
    --save_dir results \
    --log_eval & \

wait

# export CUDA_VISIBLE_DEVICES=0 &&
# python src/train.py \
#     --data_root data/mvtec_ad \
#     --class_name metal_nut \
#     --num_normal_samples -1 \
#     --ae_model conv_ae_2 \
#     --ae_hidden_dim 384 \
#     --bottleneck_dim 64 \
#     --backbone_model pdn_small \
#     --backbone_resume_path ./weights/teacher_small.pth \
#     --feature_dim 384 \
#     --feature_res 1 \
#     --patch_size 1 \
#     --num_enc_layers 3 \
#     --num_dec_layers 3 \
#     --num_heads 4 \
#     --mlp_ratio 4 \
#     --in_res 56 \
#     --img_size 256 \
#     --split train \
#     --batch_size 8 \
#     --num_epochs 100 \
#     --lr 0.0001 \
#     --momentum 0.9 \
#     --weight_decay 0.05 \
#     --optimizer adamw \
#     --scheduler cosine \
#     --grad_clip 1.0 \
#     --seed 45 \
#     --num_workers 1 \
#     --device cuda \
#     --log_interval 10 \
#     --log_eval \
#     --save_dir results & \

# export CUDA_VISIBLE_DEVICES=1 &&
# python src/train.py \
#     --data_root data/mvtec_ad \
#     --class_name tile \
#     --num_normal_samples -1 \
#     --ae_model conv_ae_2 \
#     --ae_hidden_dim 384 \
#     --bottleneck_dim 64 \
#     --backbone_model pdn_small \
#     --backbone_resume_path ./weights/teacher_small.pth \
#     --feature_dim 384 \
#     --feature_res 1 \
#     --patch_size 1 \
#     --num_enc_layers 3 \
#     --num_dec_layers 3 \
#     --num_heads 4 \
#     --mlp_ratio 4 \
#     --in_res 56 \
#     --img_size 256 \
#     --split train \
#     --batch_size 8 \
#     --num_epochs 100 \
#     --lr 0.0001 \
#     --momentum 0.9 \
#     --weight_decay 0.05 \
#     --optimizer adamw \
#     --scheduler cosine \
#     --grad_clip 1.0 \
#     --seed 45 \
#     --num_workers 1 \
#     --device cuda \
#     --log_interval 10 \
#     --log_eval \
#     --save_dir results & \

export CUDA_VISIBLE_DEVICES=2 &&
python src/train.py \
    --data_root data/mvtec_ad \
    --class_name carpet \
    --num_normal_samples -1 \
    --ae_model conv_ae_2 \
    --ae_hidden_dim 384 \
    --bottleneck_dim 64 \
    --backbone_model pdn_small \
    --backbone_resume_path ./weights/teacher_small.pth \
    --feature_dim 384 \
    --feature_res 1 \
    --patch_size 1 \
    --num_enc_layers 3 \
    --num_dec_layers 3 \
    --num_heads 4 \
    --mlp_ratio 4 \
    --in_res 56 \
    --img_size 256 \
    --split train \
    --batch_size 8 \
    --num_epochs 100 \
    --lr 0.0001 \
    --momentum 0.9 \
    --weight_decay 0.05 \
    --optimizer adamw \
    --scheduler cosine \
    --grad_clip 1.0 \
    --seed 45 \
    --num_workers 1 \
    --device cuda \
    --log_interval 10 \
    --log_eval \
    --save_dir results & \

export CUDA_VISIBLE_DEVICES=3 &&
python src/train.py \
    --data_root data/mvtec_ad \
    --class_name wood \
    --num_normal_samples -1 \
    --ae_model conv_ae_2 \
    --ae_hidden_dim 384 \
    --bottleneck_dim 64 \
    --backbone_model pdn_small \
    --backbone_resume_path ./weights/teacher_small.pth \
    --feature_dim 384 \
    --feature_res 1 \
    --patch_size 1 \
    --num_enc_layers 3 \
    --num_dec_layers 3 \
    --num_heads 4 \
    --mlp_ratio 4 \
    --in_res 56 \
    --img_size 256 \
    --split train \
    --batch_size 8 \
    --num_epochs 100 \
    --lr 0.0001 \
    --momentum 0.9 \
    --weight_decay 0.05 \
    --optimizer adamw \
    --scheduler cosine \
    --grad_clip 1.0 \
    --seed 45 \
    --num_workers 1 \
    --device cuda \
    --log_interval 10 \
    --log_eval \
    --save_dir results & \

export CUDA_VISIBLE_DEVICES=4 &&
python src/train.py \
    --data_root data/mvtec_ad \
    --class_name leather \
    --num_normal_samples -1 \
    --ae_model conv_ae_2 \
    --ae_hidden_dim 384 \
    --bottleneck_dim 64 \
    --backbone_model pdn_small \
    --backbone_resume_path ./weights/teacher_small.pth \
    --feature_dim 384 \
    --feature_res 1 \
    --patch_size 1 \
    --num_enc_layers 3 \
    --num_dec_layers 3 \
    --num_heads 4 \
    --mlp_ratio 4 \
    --in_res 56 \
    --img_size 256 \
    --split train \
    --batch_size 8 \
    --num_epochs 100 \
    --lr 0.0001 \
    --momentum 0.9 \
    --weight_decay 0.05 \
    --optimizer adamw \
    --scheduler cosine \
    --grad_clip 1.0 \
    --seed 45 \
    --num_workers 1 \
    --device cuda \
    --log_interval 10 \
    --log_eval \
    --save_dir results & \

export CUDA_VISIBLE_DEVICES=5 &&
python src/train.py \
    --data_root data/mvtec_ad \
    --class_name hazelnut \
    --num_normal_samples -1 \
    --ae_model conv_ae_2 \
    --ae_hidden_dim 384 \
    --bottleneck_dim 64 \
    --backbone_model pdn_small \
    --backbone_resume_path ./weights/teacher_small.pth \
    --feature_dim 384 \
    --feature_res 1 \
    --patch_size 1 \
    --num_enc_layers 3 \
    --num_dec_layers 3 \
    --num_heads 4 \
    --mlp_ratio 4 \
    --in_res 56 \
    --img_size 256 \
    --split train \
    --batch_size 8 \
    --num_epochs 100 \
    --lr 0.0001 \
    --momentum 0.9 \
    --weight_decay 0.05 \
    --optimizer adamw \
    --scheduler cosine \
    --grad_clip 1.0 \
    --seed 45 \
    --num_workers 1 \
    --device cuda \
    --log_interval 10 \
    --save_dir results \
    --log_eval & \

wait

# export CUDA_VISIBLE_DEVICES=0 &&
# python src/train.py \
#     --data_root data/mvtec_ad \
#     --class_name metal_nut \
#     --num_normal_samples -1 \
#     --ae_model conv_ae_3 \
#     --ae_hidden_dim 384 \
#     --bottleneck_dim 64 \
#     --backbone_model pdn_small \
#     --backbone_resume_path ./weights/teacher_small.pth \
#     --feature_dim 384 \
#     --feature_res 1 \
#     --patch_size 1 \
#     --num_enc_layers 3 \
#     --num_dec_layers 3 \
#     --num_heads 4 \
#     --mlp_ratio 4 \
#     --in_res 56 \
#     --img_size 256 \
#     --split train \
#     --batch_size 8 \
#     --num_epochs 100 \
#     --lr 0.0001 \
#     --momentum 0.9 \
#     --weight_decay 0.05 \
#     --optimizer adamw \
#     --scheduler cosine \
#     --grad_clip 1.0 \
#     --seed 44 \
#     --num_workers 1 \
#     --device cuda \
#     --log_interval 10 \
#     --log_eval \
#     --save_dir results & \

# export CUDA_VISIBLE_DEVICES=1 &&
# python src/train.py \
#     --data_root data/mvtec_ad \
#     --class_name tile \
#     --num_normal_samples -1 \
#     --ae_model conv_ae_3 \
#     --ae_hidden_dim 384 \
#     --bottleneck_dim 64 \
#     --backbone_model pdn_small \
#     --backbone_resume_path ./weights/teacher_small.pth \
#     --feature_dim 384 \
#     --feature_res 1 \
#     --patch_size 1 \
#     --num_enc_layers 3 \
#     --num_dec_layers 3 \
#     --num_heads 4 \
#     --mlp_ratio 4 \
#     --in_res 56 \
#     --img_size 256 \
#     --split train \
#     --batch_size 8 \
#     --num_epochs 100 \
#     --lr 0.0001 \
#     --momentum 0.9 \
#     --weight_decay 0.05 \
#     --optimizer adamw \
#     --scheduler cosine \
#     --grad_clip 1.0 \
#     --seed 44 \
#     --num_workers 1 \
#     --device cuda \
#     --log_interval 10 \
#     --log_eval \
#     --save_dir results & \

export CUDA_VISIBLE_DEVICES=2 &&
python src/train.py \
    --data_root data/mvtec_ad \
    --class_name carpet \
    --num_normal_samples -1 \
    --ae_model conv_ae_3 \
    --ae_hidden_dim 384 \
    --bottleneck_dim 64 \
    --backbone_model pdn_small \
    --backbone_resume_path ./weights/teacher_small.pth \
    --feature_dim 384 \
    --feature_res 1 \
    --patch_size 1 \
    --num_enc_layers 3 \
    --num_dec_layers 3 \
    --num_heads 4 \
    --mlp_ratio 4 \
    --in_res 56 \
    --img_size 256 \
    --split train \
    --batch_size 8 \
    --num_epochs 100 \
    --lr 0.0001 \
    --momentum 0.9 \
    --weight_decay 0.05 \
    --optimizer adamw \
    --scheduler cosine \
    --grad_clip 1.0 \
    --seed 44 \
    --num_workers 1 \
    --device cuda \
    --log_interval 10 \
    --log_eval \
    --save_dir results & \

export CUDA_VISIBLE_DEVICES=3 &&
python src/train.py \
    --data_root data/mvtec_ad \
    --class_name wood \
    --num_normal_samples -1 \
    --ae_model conv_ae_3 \
    --ae_hidden_dim 384 \
    --bottleneck_dim 64 \
    --backbone_model pdn_small \
    --backbone_resume_path ./weights/teacher_small.pth \
    --feature_dim 384 \
    --feature_res 1 \
    --patch_size 1 \
    --num_enc_layers 3 \
    --num_dec_layers 3 \
    --num_heads 4 \
    --mlp_ratio 4 \
    --in_res 56 \
    --img_size 256 \
    --split train \
    --batch_size 8 \
    --num_epochs 100 \
    --lr 0.0001 \
    --momentum 0.9 \
    --weight_decay 0.05 \
    --optimizer adamw \
    --scheduler cosine \
    --grad_clip 1.0 \
    --seed 44 \
    --num_workers 1 \
    --device cuda \
    --log_interval 10 \
    --log_eval \
    --save_dir results & \

export CUDA_VISIBLE_DEVICES=4 &&
python src/train.py \
    --data_root data/mvtec_ad \
    --class_name leather \
    --num_normal_samples -1 \
    --ae_model conv_ae_3 \
    --ae_hidden_dim 384 \
    --bottleneck_dim 64 \
    --backbone_model pdn_small \
    --backbone_resume_path ./weights/teacher_small.pth \
    --feature_dim 384 \
    --feature_res 1 \
    --patch_size 1 \
    --num_enc_layers 3 \
    --num_dec_layers 3 \
    --num_heads 4 \
    --mlp_ratio 4 \
    --in_res 56 \
    --img_size 256 \
    --split train \
    --batch_size 8 \
    --num_epochs 100 \
    --lr 0.0001 \
    --momentum 0.9 \
    --weight_decay 0.05 \
    --optimizer adamw \
    --scheduler cosine \
    --grad_clip 1.0 \
    --seed 44 \
    --num_workers 1 \
    --device cuda \
    --log_interval 10 \
    --log_eval \
    --save_dir results & \

export CUDA_VISIBLE_DEVICES=5 &&
python src/train.py \
    --data_root data/mvtec_ad \
    --class_name hazelnut \
    --num_normal_samples -1 \
    --ae_model conv_ae_3 \
    --ae_hidden_dim 384 \
    --bottleneck_dim 64 \
    --backbone_model pdn_small \
    --backbone_resume_path ./weights/teacher_small.pth \
    --feature_dim 384 \
    --feature_res 1 \
    --patch_size 1 \
    --num_enc_layers 3 \
    --num_dec_layers 3 \
    --num_heads 4 \
    --mlp_ratio 4 \
    --in_res 56 \
    --img_size 256 \
    --split train \
    --batch_size 8 \
    --num_epochs 100 \
    --lr 0.0001 \
    --momentum 0.9 \
    --weight_decay 0.05 \
    --optimizer adamw \
    --scheduler cosine \
    --grad_clip 1.0 \
    --seed 44 \
    --num_workers 1 \
    --device cuda \
    --log_interval 10 \
    --save_dir results \
    --log_eval & \

wait

# export CUDA_VISIBLE_DEVICES=0 &&
# python src/train.py \
#     --data_root data/mvtec_ad \
#     --class_name metal_nut \
#     --num_normal_samples -1 \
#     --ae_model conv_ae_4 \
#     --ae_hidden_dim 384 \
#     --bottleneck_dim 64 \
#     --backbone_model pdn_small \
#     --backbone_resume_path ./weights/teacher_small.pth \
#     --feature_dim 384 \
#     --feature_res 1 \
#     --patch_size 1 \
#     --num_enc_layers 3 \
#     --num_dec_layers 3 \
#     --num_heads 4 \
#     --mlp_ratio 4 \
#     --in_res 56 \
#     --img_size 256 \
#     --split train \
#     --batch_size 8 \
#     --num_epochs 100 \
#     --lr 0.0001 \
#     --momentum 0.9 \
#     --weight_decay 0.05 \
#     --optimizer adamw \
#     --scheduler cosine \
#     --grad_clip 1.0 \
#     --seed 43 \
#     --num_workers 1 \
#     --device cuda \
#     --log_interval 10 \
#     --log_eval \
#     --save_dir results & \

# export CUDA_VISIBLE_DEVICES=1 &&
# python src/train.py \
#     --data_root data/mvtec_ad \
#     --class_name tile \
#     --num_normal_samples -1 \
#     --ae_model conv_ae_4 \
#     --ae_hidden_dim 384 \
#     --bottleneck_dim 64 \
#     --backbone_model pdn_small \
#     --backbone_resume_path ./weights/teacher_small.pth \
#     --feature_dim 384 \
#     --feature_res 1 \
#     --patch_size 1 \
#     --num_enc_layers 3 \
#     --num_dec_layers 3 \
#     --num_heads 4 \
#     --mlp_ratio 4 \
#     --in_res 56 \
#     --img_size 256 \
#     --split train \
#     --batch_size 8 \
#     --num_epochs 100 \
#     --lr 0.0001 \
#     --momentum 0.9 \
#     --weight_decay 0.05 \
#     --optimizer adamw \
#     --scheduler cosine \
#     --grad_clip 1.0 \
#     --seed 43 \
#     --num_workers 1 \
#     --device cuda \
#     --log_interval 10 \
#     --log_eval \
#     --save_dir results & \

export CUDA_VISIBLE_DEVICES=2 &&
python src/train.py \
    --data_root data/mvtec_ad \
    --class_name carpet \
    --num_normal_samples -1 \
    --ae_model conv_ae_4 \
    --ae_hidden_dim 384 \
    --bottleneck_dim 64 \
    --backbone_model pdn_small \
    --backbone_resume_path ./weights/teacher_small.pth \
    --feature_dim 384 \
    --feature_res 1 \
    --patch_size 1 \
    --num_enc_layers 3 \
    --num_dec_layers 3 \
    --num_heads 4 \
    --mlp_ratio 4 \
    --in_res 56 \
    --img_size 256 \
    --split train \
    --batch_size 8 \
    --num_epochs 100 \
    --lr 0.0001 \
    --momentum 0.9 \
    --weight_decay 0.05 \
    --optimizer adamw \
    --scheduler cosine \
    --grad_clip 1.0 \
    --seed 43 \
    --num_workers 1 \
    --device cuda \
    --log_interval 10 \
    --log_eval \
    --save_dir results & \

export CUDA_VISIBLE_DEVICES=3 &&
python src/train.py \
    --data_root data/mvtec_ad \
    --class_name wood \
    --num_normal_samples -1 \
    --ae_model conv_ae_4 \
    --ae_hidden_dim 384 \
    --bottleneck_dim 64 \
    --backbone_model pdn_small \
    --backbone_resume_path ./weights/teacher_small.pth \
    --feature_dim 384 \
    --feature_res 1 \
    --patch_size 1 \
    --num_enc_layers 3 \
    --num_dec_layers 3 \
    --num_heads 4 \
    --mlp_ratio 4 \
    --in_res 56 \
    --img_size 256 \
    --split train \
    --batch_size 8 \
    --num_epochs 100 \
    --lr 0.0001 \
    --momentum 0.9 \
    --weight_decay 0.05 \
    --optimizer adamw \
    --scheduler cosine \
    --grad_clip 1.0 \
    --seed 43 \
    --num_workers 1 \
    --device cuda \
    --log_interval 10 \
    --log_eval \
    --save_dir results & \

export CUDA_VISIBLE_DEVICES=4 &&
python src/train.py \
    --data_root data/mvtec_ad \
    --class_name leather \
    --num_normal_samples -1 \
    --ae_model conv_ae_4 \
    --ae_hidden_dim 384 \
    --bottleneck_dim 64 \
    --backbone_model pdn_small \
    --backbone_resume_path ./weights/teacher_small.pth \
    --feature_dim 384 \
    --feature_res 1 \
    --patch_size 1 \
    --num_enc_layers 3 \
    --num_dec_layers 3 \
    --num_heads 4 \
    --mlp_ratio 4 \
    --in_res 56 \
    --img_size 256 \
    --split train \
    --batch_size 8 \
    --num_epochs 100 \
    --lr 0.0001 \
    --momentum 0.9 \
    --weight_decay 0.05 \
    --optimizer adamw \
    --scheduler cosine \
    --grad_clip 1.0 \
    --seed 43 \
    --num_workers 1 \
    --device cuda \
    --log_interval 10 \
    --log_eval \
    --save_dir results & \

export CUDA_VISIBLE_DEVICES=5 &&
python src/train.py \
    --data_root data/mvtec_ad \
    --class_name hazelnut \
    --num_normal_samples -1 \
    --ae_model conv_ae_4 \
    --ae_hidden_dim 384 \
    --bottleneck_dim 64 \
    --backbone_model pdn_small \
    --backbone_resume_path ./weights/teacher_small.pth \
    --feature_dim 384 \
    --feature_res 1 \
    --patch_size 1 \
    --num_enc_layers 3 \
    --num_dec_layers 3 \
    --num_heads 4 \
    --mlp_ratio 4 \
    --in_res 56 \
    --img_size 256 \
    --split train \
    --batch_size 8 \
    --num_epochs 100 \
    --lr 0.0001 \
    --momentum 0.9 \
    --weight_decay 0.05 \
    --optimizer adamw \
    --scheduler cosine \
    --grad_clip 1.0 \
    --seed 43 \
    --num_workers 1 \
    --device cuda \
    --log_interval 10 \
    --save_dir results \
    --log_eval & \

wait


# export CUDA_VISIBLE_DEVICES=0 &&
# python src/train.py \
#     --data_root data/mvtec_ad \
#     --class_name metal_nut \
#     --num_normal_samples -1 \
#     --ae_model conv_ae_5 \
#     --ae_hidden_dim 384 \
#     --bottleneck_dim 64 \
#     --backbone_model pdn_small \
#     --backbone_resume_path ./weights/teacher_small.pth \
#     --feature_dim 384 \
#     --feature_res 1 \
#     --patch_size 1 \
#     --num_enc_layers 3 \
#     --num_dec_layers 3 \
#     --num_heads 4 \
#     --mlp_ratio 4 \
#     --in_res 56 \
#     --img_size 256 \
#     --split train \
#     --batch_size 8 \
#     --num_epochs 100 \
#     --lr 0.0001 \
#     --momentum 0.9 \
#     --weight_decay 0.05 \
#     --optimizer adamw \
#     --scheduler cosine \
#     --grad_clip 1.0 \
#     --seed 42 \
#     --num_workers 1 \
#     --device cuda \
#     --log_interval 10 \
#     --log_eval \
#     --save_dir results & \

# export CUDA_VISIBLE_DEVICES=1 &&
# python src/train.py \
#     --data_root data/mvtec_ad \
#     --class_name tile \
#     --num_normal_samples -1 \
#     --ae_model conv_ae_5 \
#     --ae_hidden_dim 384 \
#     --bottleneck_dim 64 \
#     --backbone_model pdn_small \
#     --backbone_resume_path ./weights/teacher_small.pth \
#     --feature_dim 384 \
#     --feature_res 1 \
#     --patch_size 1 \
#     --num_enc_layers 3 \
#     --num_dec_layers 3 \
#     --num_heads 4 \
#     --mlp_ratio 4 \
#     --in_res 56 \
#     --img_size 256 \
#     --split train \
#     --batch_size 8 \
#     --num_epochs 100 \
#     --lr 0.0001 \
#     --momentum 0.9 \
#     --weight_decay 0.05 \
#     --optimizer adamw \
#     --scheduler cosine \
#     --grad_clip 1.0 \
#     --seed 42 \
#     --num_workers 1 \
#     --device cuda \
#     --log_interval 10 \
#     --log_eval \
#     --save_dir results & \

export CUDA_VISIBLE_DEVICES=2 &&
python src/train.py \
    --data_root data/mvtec_ad \
    --class_name carpet \
    --num_normal_samples -1 \
    --ae_model conv_ae_5 \
    --ae_hidden_dim 384 \
    --bottleneck_dim 64 \
    --backbone_model pdn_small \
    --backbone_resume_path ./weights/teacher_small.pth \
    --feature_dim 384 \
    --feature_res 1 \
    --patch_size 1 \
    --num_enc_layers 3 \
    --num_dec_layers 3 \
    --num_heads 4 \
    --mlp_ratio 4 \
    --in_res 56 \
    --img_size 256 \
    --split train \
    --batch_size 8 \
    --num_epochs 100 \
    --lr 0.0001 \
    --momentum 0.9 \
    --weight_decay 0.05 \
    --optimizer adamw \
    --scheduler cosine \
    --grad_clip 1.0 \
    --seed 42 \
    --num_workers 1 \
    --device cuda \
    --log_interval 10 \
    --log_eval \
    --save_dir results & \

export CUDA_VISIBLE_DEVICES=3 &&
python src/train.py \
    --data_root data/mvtec_ad \
    --class_name wood \
    --num_normal_samples -1 \
    --ae_model conv_ae_5 \
    --ae_hidden_dim 384 \
    --bottleneck_dim 64 \
    --backbone_model pdn_small \
    --backbone_resume_path ./weights/teacher_small.pth \
    --feature_dim 384 \
    --feature_res 1 \
    --patch_size 1 \
    --num_enc_layers 3 \
    --num_dec_layers 3 \
    --num_heads 4 \
    --mlp_ratio 4 \
    --in_res 56 \
    --img_size 256 \
    --split train \
    --batch_size 8 \
    --num_epochs 100 \
    --lr 0.0001 \
    --momentum 0.9 \
    --weight_decay 0.05 \
    --optimizer adamw \
    --scheduler cosine \
    --grad_clip 1.0 \
    --seed 42 \
    --num_workers 1 \
    --device cuda \
    --log_interval 10 \
    --log_eval \
    --save_dir results & \

export CUDA_VISIBLE_DEVICES=4 &&
python src/train.py \
    --data_root data/mvtec_ad \
    --class_name leather \
    --num_normal_samples -1 \
    --ae_model conv_ae_5 \
    --ae_hidden_dim 384 \
    --bottleneck_dim 64 \
    --backbone_model pdn_small \
    --backbone_resume_path ./weights/teacher_small.pth \
    --feature_dim 384 \
    --feature_res 1 \
    --patch_size 1 \
    --num_enc_layers 3 \
    --num_dec_layers 3 \
    --num_heads 4 \
    --mlp_ratio 4 \
    --in_res 56 \
    --img_size 256 \
    --split train \
    --batch_size 8 \
    --num_epochs 100 \
    --lr 0.0001 \
    --momentum 0.9 \
    --weight_decay 0.05 \
    --optimizer adamw \
    --scheduler cosine \
    --grad_clip 1.0 \
    --seed 42 \
    --num_workers 1 \
    --device cuda \
    --log_interval 10 \
    --log_eval \
    --save_dir results & \

export CUDA_VISIBLE_DEVICES=5 &&
python src/train.py \
    --data_root data/mvtec_ad \
    --class_name hazelnut \
    --num_normal_samples -1 \
    --ae_model conv_ae_5 \
    --ae_hidden_dim 384 \
    --bottleneck_dim 64 \
    --backbone_model pdn_small \
    --backbone_resume_path ./weights/teacher_small.pth \
    --feature_dim 384 \
    --feature_res 1 \
    --patch_size 1 \
    --num_enc_layers 3 \
    --num_dec_layers 3 \
    --num_heads 4 \
    --mlp_ratio 4 \
    --in_res 56 \
    --img_size 256 \
    --split train \
    --batch_size 8 \
    --num_epochs 100 \
    --lr 0.0001 \
    --momentum 0.9 \
    --weight_decay 0.05 \
    --optimizer adamw \
    --scheduler cosine \
    --grad_clip 1.0 \
    --seed 42 \
    --num_workers 1 \
    --device cuda \
    --log_interval 10 \
    --save_dir results \
    --log_eval & \