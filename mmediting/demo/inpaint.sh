python demo/inpainting_demo.py \
    configs/inpainting/deepfillv2/deepfillv2_256x256_8x2_places.py \
    models/deepfillv2_256x256_8x2_places_20200619-10d15793.pth \
    ~/mmsegmentation/demo/inputs/cars1.png \
    ~/mmsegmentation/demo/selected_masks/cars1.png \
    ~/mmsegmentation/demo/inpainted/cars1.png