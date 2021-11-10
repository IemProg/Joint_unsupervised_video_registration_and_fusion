# Joint_unsupervised_video_registration_and_fusion
This project targets joint registration and fusion of two different imaging modalities namely, RGB and Infrared (IR)

### Abstract ###
We present a system to perform joint registration and fusion
for RGB and Infrared (IR) video pairs. While RGB is related to
human perception, IR is associated with heat. However, IR images often lack contour and texture information. The goal with
the fusion of the visible and IR images is to obtain more information from them. This requires two completely matched images.
However, classical methods assuming ideal imaging conditions
fail to achieve satisfactory performance in actual cases. From
the data-dependent modeling point of view, labeling the dataset
is costly and impractical.


In this context, we present a framework that tackles two
challenging tasks. First, a video registration procedure that aims
to align IR and RGB videos. Second, a fusion method brings all
the essential information from the two video modalities to a single video. We evaluate our approach on a challenging dataset of
RGB and IR video pairs collected for firefighters to handle their
tasks effectively in challenging visibility conditions such as heavy
smoke after a fire.

![](example.png)

# Requirements:
```
pip install -r requirements.txt
```

- In order to create masks for the images, pre-trained model of MaskRCNN is required. Download link(https://github.com/matterport/Mask_RCNN/releases).

- For the Fusion part, we use pre-trained model of VGG-19 (Fusion-Zero).

### Data preparation

Your directory tree should be look like this:
````bash
$Data_IR_RGB
├── Segmented
│   ├── SegmentedVIS
│       ├── SAVE_1_AFI
│       ├── SAVE_2_AFI
│       └── SAVE_3_HAK
│       └── SAVE_4_ARO
│ 
├── VISIBLE & IR
│   ├── SAVE_1_AFI
│   ├── SAVE_2_AFI
│   └── SAVE_3_HAK
│   └── SAVE_4_ARO

````

### Training

For example, train the provided data_set with a batch size of 100 on 1 GPUs:
````
python train.py --inputFrames1=PathToRGB_Frames --inputFrames2=PathToIR_Frames --output=PATH_TO_SAVE_MODEL ----name=MODEL_NAME  --epochs=NBR_OF_EPOCHS
````

- We designed the scripts to be as similar as possible to the tensorflow/keras version adapted from VoxelMorph implementation.

### Test

For example, evaluating our model using pretrained model "registration_videoAll.ph":
````bash
python train.py --inputVid1=PathToRGB_Frames --inputVid2=PathToIR_Frames --inputSegFrame1=PathToSegmentedRGB_Frames --inputSegFrame2=PathToSegmentedIR_Frames --output=PATH_TO_SAVE_MODEL --model="registration_videoAll.ph"  --save=True/False to save samples
````

### Citing
```
@InProceedings{Joint_Unsupersvised_IR_RGB_Video_Fusion_Registration,
author = {Imad Eddine Marouf and Lucas Barras and Hakki Can Karaimer and Sabine Süsstrunk},
title = {Joint Unsupervised Joint Infrared-RGB Video Registration and Fusion},
booktitle = {London Imaging Meeting (LIM)},
year = {2021}
}

```
# Contact:
For any problems or questions please open an issue in github.

## Acknowledgement
- We adopt VoxelMorph-Architecture implemented by [Dalca](https://github.com/voxelmorph/voxelmorph).
- Dataset provided by IVRL Laboroatory at EPFL, Switzerland.
