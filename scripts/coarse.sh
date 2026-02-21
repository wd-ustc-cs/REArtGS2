export CUDA_VISIBLE_DEVICES=0
 dataset=paris
 subset=sapien
 scenes = (scissor_11100)
# scenes=(foldchair_102255 washer_103776 fridge_10905 blade_103706 storage_45135 oven_101917 stapler_103111 USB_100109 laptop_10211 scissor_11100)
# subset=realscan
# scenes=(real_fridge real_storage)

# dataset=dta
# subset=sapien
# scenes=(fridge_10489 storage_47254)

#dataset=artgs
#subset=sapien
#scenes=(oven_101908 table_25493 storage_45503 storage_47648 table_31249)
#scenes=(table_25493 table_31249)


model_name=coarse_gs
res=2
for scene in ${scenes[@]};do
    model_path=outputs/${dataset}/${subset}/${scene}/${model_name}
    python train_coarse.py \
        --dataset ${dataset} \
        --subset ${subset} \
        --scene_name ${scene} \
        --model_path ${model_path} \
        --resolution ${res} \
        --iterations 10000 \
        --opacity_reg_weight 0.1 \

done

--dataset paris --subset sapien --scene_name scissor_11100 --model_path outputs/paris/sapien/scissor_11100/coarse_gs --resolution 2 --iterations 10000 --opacity_reg_weight 0.1 --random_bg_color