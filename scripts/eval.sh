python src/eval.py \
    --data_root data/mvtec_ad \
    --class_name bottle \
    --ae_model conv_ae_sp1 \
    --ae_hidden_dim 384 \
    --bottleneck_dim 64 \
    --backbone_model pdn_small \
    --ae_resume_path ./results/bottle_conv_ae_sp2_dim64_pdn_small/weights.pth \
    --backbone_resume_path ./weights/teacher_small.pth \
    --feature_dim 384 \
    --feature_res 2 \
    --img_size 256 \
    --split test \
    --batch_size 8 \
    --seed 42 \
    --device cuda 

python src/eval.py \
    --data_root data/mvtec_ad \
    --class_name bottle \
    --ae_model conv_ae_sp1 \
    --ae_hidden_dim 384 \
    --bottleneck_dim 64 \
    --backbone_model pdn_small \
    --ae_resume_path ./results/bottle_conv_ae_sp4_dim64_pdn_small/weights.pth \
    --backbone_resume_path ./weights/teacher_small.pth \
    --feature_dim 384 \
    --feature_res 4 \
    --img_size 256 \
    --split test \
    --batch_size 8 \
    --seed 42 \
    --device cuda 

python src/eval.py \
    --data_root data/mvtec_ad \
    --class_name bottle \
    --ae_model conv_ae_sp1 \
    --ae_hidden_dim 384 \
    --bottleneck_dim 64 \
    --backbone_model pdn_small \
    --ae_resume_path ./results/bottle_conv_ae_sp8_dim64_pdn_small/weights.pth \
    --backbone_resume_path ./weights/teacher_small.pth \
    --feature_dim 384 \
    --feature_res 8 \
    --img_size 256 \
    --split test \
    --batch_size 8 \
    --seed 42 \
    --device cuda 

python src/eval.py \
    --data_root data/mvtec_ad \
    --class_name bottle \
    --ae_model conv_ae_sp1 \
    --ae_hidden_dim 384 \
    --bottleneck_dim 64 \
    --backbone_model pdn_small \
    --ae_resume_path ./results/bottle_conv_ae_sp16_dim64_pdn_small/weights.pth \
    --backbone_resume_path ./weights/teacher_small.pth \
    --feature_dim 384 \
    --feature_res 16 \
    --img_size 256 \
    --split test \
    --batch_size 8 \
    --seed 42 \
    --device cuda 
