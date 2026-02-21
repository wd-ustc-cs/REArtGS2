export CUDA_VISIBLE_DEVICES=0
dataset=paris
subset=sapien
scenes=(fridge_10905 blade_103706 storage_45135 oven_101917 stapler_103111 USB_100109 laptop_10211 scissor_11100)
scenes=(foldchair_102255 scissor_11100 stapler_103111)
# scenes=(foldchair_102255)
# subset=realscan
# scenes=(real_fridge real_storage)
# scenes=(real_storage)

# dataset=dta
# subset=sapien
# scenes=(fridge_10489)
# scenes=(storage_47254)

# dataset=artgs
# subset=sapien
# scenes=(oven_101908 table_25493 storage_47648 table_31249 storage_45503)
# scenes=(table_31249)


model_name=base
res=1
iter=best

for scene in ${scenes[@]};do
    model_path=outputs/best/${dataset}/${scene}
    # model_path=outputs/${dataset}/${subset}/${scene}/${model_name}
    python render_video.py \
        --dataset ${dataset} \
        --subset ${subset} \
        --scene_name ${scene} \
        --model_path ${model_path} \
        --resolution ${res} \
        --iteration ${iter} \
        --white_background \
        --N_frames 30 
done

