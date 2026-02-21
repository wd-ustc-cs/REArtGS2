export CUDA_VISIBLE_DEVICES=0
# dataset=paris
# subset=sapien
# scenes=(foldchair_102255 washer_103776 fridge_10905 blade_103706 storage_45135 oven_101917 stapler_103111 USB_100109 laptop_10211 scissor_11100)
# subset=realscan
# scenes=(real_fridge real_storage)

# dataset=dta
# subset=sapien
# scenes=(fridge_10489 storage_47254)

dataset=artgs
subset=sapien
scenes=(oven_101908 storage_47648 table_31249 storage_45503)
scenes=(table_25493)

model_name=artgs
res=1
iter=best

for scene in ${scenes[@]};do
    # model_path=outputs/ArtGS_ckpt/${dataset}/${scene}
    model_path=outputs/${dataset}/${subset}/${scene}/${model_name}
    python render.py \
        --dataset ${dataset} \
        --subset ${subset} \
        --scene_name ${scene} \
        --model_path ${model_path} \
        --resolution ${res} \
        --iteration ${iter} \
        --skip_test 
        # --visualize \
        # --eval_app \
done

--dataset paris --subset sapien --scene_name scissor_11100 --model_path outputs/paris/sapien/scissor_11100/artgs --resolution 1 --iteration best --skip_test