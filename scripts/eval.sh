python src/eval.py \
    --data_root data/mvtec_ad \
    --class_name pill \
    --ae_model conv_ae_sp1 \
    --ae_hidden_dim 384 \
    --bottleneck_dim 384 \
    --backbone_model pdn_small \
    --ae_resume_path ./results/pill_conv_ae_sp1_dim384_pdn_small/weights.pth \
    --backbone_resume_path ./weights/teacher_small.pth \
    --feature_dim 384 \
    --feature_res 1 \
    --img_size 256 \
    --split test \
    --batch_size 8 \
    --seed 42 \
    --device cuda 

python src/eval.py \
    --data_root data/mvtec_ad \
    --class_name pill \
    --ae_model conv_ae_sp1 \
    --ae_hidden_dim 384 \
    --bottleneck_dim 256 \
    --backbone_model pdn_small \
    --ae_resume_path ./results/pill_conv_ae_sp1_dim256_pdn_small/weights.pth \
    --backbone_resume_path ./weights/teacher_small.pth \
    --feature_dim 384 \
    --feature_res 1 \
    --img_size 256 \
    --split test \
    --batch_size 8 \
    --seed 42 \
    --device cuda 

python src/eval.py \
    --data_root data/mvtec_ad \
    --class_name pill \
    --ae_model conv_ae_sp1 \
    --ae_hidden_dim 384 \
    --bottleneck_dim 128 \
    --backbone_model pdn_small \
    --ae_resume_path ./results/pill_conv_ae_sp1_dim128_pdn_small/weights.pth \
    --backbone_resume_path ./weights/teacher_small.pth \
    --feature_dim 384 \
    --feature_res 1 \
    --img_size 256 \
    --split test \
    --batch_size 8 \
    --seed 42 \
    --device cuda 

python src/eval.py \
    --data_root data/mvtec_ad \
    --class_name pill \
    --ae_model conv_ae_sp1 \
    --ae_hidden_dim 384 \
    --bottleneck_dim 64 \
    --backbone_model pdn_small \
    --ae_resume_path ./results/pill_conv_ae_sp1_dim64_pdn_small/weights.pth \
    --backbone_resume_path ./weights/teacher_small.pth \
    --feature_dim 384 \
    --feature_res 1 \
    --img_size 256 \
    --split test \
    --batch_size 8 \
    --seed 42 \
    --device cuda 

python src/eval.py \
    --data_root data/mvtec_ad \
    --class_name pill \
    --ae_model conv_ae_sp1 \
    --ae_hidden_dim 384 \
    --bottleneck_dim 32 \
    --backbone_model pdn_small \
    --ae_resume_path ./results/pill_conv_ae_sp1_dim32_pdn_small/weights.pth \
    --backbone_resume_path ./weights/teacher_small.pth \
    --feature_dim 384 \
    --feature_res 1 \
    --img_size 256 \
    --split test \
    --batch_size 8 \
    --seed 42 \
    --device cuda 

python src/eval.py \
    --data_root data/mvtec_ad \
    --class_name pill \
    --ae_model conv_ae_sp1 \
    --ae_hidden_dim 384 \
    --bottleneck_dim 16 \
    --backbone_model pdn_small \
    --ae_resume_path ./results/pill_conv_ae_sp1_dim16_pdn_small/weights.pth \
    --backbone_resume_path ./weights/teacher_small.pth \
    --feature_dim 384 \
    --feature_res 1 \
    --img_size 256 \
    --split test \
    --batch_size 8 \
    --seed 42 \
    --device cuda 

python src/eval.py \
    --data_root data/mvtec_ad \
    --class_name pill \
    --ae_model conv_ae_sp1 \
    --ae_hidden_dim 384 \
    --bottleneck_dim 8 \
    --backbone_model pdn_small \
    --ae_resume_path ./results/pill_conv_ae_sp1_dim8_pdn_small/weights.pth \
    --backbone_resume_path ./weights/teacher_small.pth \
    --feature_dim 384 \
    --feature_res 1 \
    --img_size 256 \
    --split test \
    --batch_size 8 \
    --seed 42 \
    --device cuda 
