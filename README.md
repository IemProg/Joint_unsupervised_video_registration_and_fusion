# Joint_unsupervised_video_registration_and_fusion
This project targets joint registration and fusion of two different imaging modalities namely, RGB and Infrared (IR)


![](example.png)

# Requirements:
```
pip install -r requirements.txt
```

- In order to create masks for the images, pre-trained model of MaskRCNN is required. Download Link(https://github.com/matterport/Mask_RCNN/releases).

- For the Fusion part, we need pre-trained model of VGG-19 is also required.

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

# Papers: 

1 - https://github.com/voxelmorph/voxelmorph
```
@article{Balakrishnan_2018,
   title={An Unsupervised Learning Model for Deformable Medical Image Registration},
   ISBN={9781538664209},
   url={http://dx.doi.org/10.1109/CVPR.2018.00964},
   DOI={10.1109/cvpr.2018.00964},
   journal={2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition},
   publisher={IEEE},
   author={Balakrishnan, Guha and Zhao, Amy and Sabuncu, Mert R. and Dalca, Adrian V. and Guttag, John},
   year={2018},
   month={Jun}
}
```


2 - https://github.com/IVRL/Fast-Zero-Learning-Fusion
```
@inproceedings{fayez2019zero,
title={Fast and Efficient Zero-Learning Image Fusion},
author={Fayez, Lahoud and Sabine, Süsstrunk},
journal={arXiv preprint arXiv:1902.00730},
year={2019}
}
```
3 - https://github.com/matterport/Mask_RCNN
```
@misc{matterport_maskrcnn_2017,
  title={Mask R-CNN for object detection and instance segmentation on Keras and TensorFlow},
  author={Waleed Abdulla},
  year={2017},
  publisher={Github},
  journal={GitHub repository},
  howpublished={\url{https://github.com/matterport/Mask_RCNN}},
}
```

# Contact:
For any problems or questions please open an issue in github.

## Acknowledgement
- We adopt VoxelMorph-Architecture implemented by [Dalca](https://github.com/voxelmorph/voxelmorph).
- Dataset provided by IVRL Laboroatory at EPFL, Switzerland.
