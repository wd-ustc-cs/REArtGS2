export CUDA_VISIBLE_DEVICES=0
# dataset=paris
# subset=sapien
# scenes=(foldchair_102255 washer_103776 fridge_10905 blade_103706 storage_45135 oven_101917 stapler_103111 USB_100109 laptop_10211 scissor_11100)
# subset=realscan
# scenes=(real_fridge real_storage)
# n_iter=3000

# dataset=dta
# subset=sapien
# scenes=(fridge_10489 storage_47254)
# n_iter=3000

dataset=artgs
subset=sapien
scenes=(oven_101908 table_25493 storage_45503 storage_47648 table_31249)
scenes=(table_25493 table_31249)
n_iter=5000

model_name=pred
res=8

for scene in ${scenes[@]};do
    model_path=outputs/${dataset}/${subset}/${scene}/${model_name}
    python train_predict.py \
        --dataset ${dataset} \
        --subset ${subset} \
        --scene_name ${scene} \
        --model_path ${model_path} \
        --eval \
        --resolution ${res} \
        --iterations ${n_iter} \
        --densify_grad_threshold 0.001 \
        --coarse_name coarse_gs \
        --random_bg_color 
done

--dataset paris --subset sapien --scene_name scissor_11100 --model_path outputs/paris/sapien/scissor_11100/pred --eval --resolution 8 --iterations 5000 --densify_grad_threshold 0.001 --coarse_name coarse_gs --random_bg_color