# Joint_unsupervised_video_registration_and_fusion
This project targets joint registration and fusion of two different imaging modalities namely, RGB and Infrared (IR)


![](example.png)

# Requirements:
```
pip install -r requirements.txt
```

# Usage:
```
python  main.py --input /path/to/input/video.avi --output /path/to/your/result.avi
```


# Training
We designed the scripts to be as similar as possible to the tensorflow/keras versions.

```
train.py --data_dir /my/path/to/data --gpu 0 --model_dir /my/path/to/save/models 
```

### Data preparation
You need to download the [Cityscapes](https://www.cityscapes-dataset.com/), [LIP](http://sysu-hcp.net/lip/) and [PASCAL-Context](https://cs.stanford.edu/~roozbeh/pascal-context/) datasets.

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
├── VISIBLE
│   ├── SAVE_1_AFI
│   ├── SAVE_2_AFI
│   └── SAVE_3_HAK
│   └── SAVE_4_ARO

````

### Train and test
For example, train the HRNet-W48 on Cityscapes with a batch size of 12 on 4 GPUs:
````bash
python tools/train.py --cfg experiments/cityscapes/seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml
````

For example, evaluating our model on the Cityscapes validation set with multi-scale and flip testing:
````bash
python tools/test.py --cfg experiments/cityscapes/seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml \
                     TEST.MODEL_FILE hrnet_w48_cityscapes_cls19_1024x2048_trainset.pth \
                     TEST.SCALE_LIST 0.5,0.75,1.0,1.25,1.5,1.75 \
                     TEST.FLIP_TEST True
````
Evaluating our model on the Cityscapes test set with multi-scale and flip testing:
````bash
python tools/test.py --cfg experiments/cityscapes/seg_hrnet_w48_train_512x1024_sgd_lr1e-2_wd5e-4_bs_12_epoch484.yaml \
                     DATASET.TEST_SET list/cityscapes/test.lst \
                     TEST.MODEL_FILE hrnet_w48_cityscapes_cls19_1024x2048_trainset.pth \
                     TEST.SCALE_LIST 0.5,0.75,1.0,1.25,1.5,1.75 \
                     TEST.FLIP_TEST True
````
Evaluating our model on the PASCAL-Context validation set with multi-scale and flip testing:
````bash
python tools/test.py --cfg experiments/pascal_ctx/seg_hrnet_w48_cls59_480x480_sgd_lr4e-3_wd1e-4_bs_16_epoch200.yaml \
                     DATASET.TEST_SET testval \
                     TEST.MODEL_FILE hrnet_w48_pascal_context_cls59_480x480.pth \
                     TEST.SCALE_LIST 0.5,0.75,1.0,1.25,1.5,1.75,2.0 \
                     TEST.FLIP_TEST True
````
Evaluating our model on the LIP validation set with flip testing:
````bash
python tools/test.py --cfg experiments/lip/seg_hrnet_w48_473x473_sgd_lr7e-3_wd5e-4_bs_40_epoch150.yaml \
                     DATASET.TEST_SET list/lip/testvalList.txt \
                     TEST.MODEL_FILE hrnet_w48_lip_cls20_473x473.pth \
                     TEST.FLIP_TEST True \
                     TEST.NUM_SAMPLES 0
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
