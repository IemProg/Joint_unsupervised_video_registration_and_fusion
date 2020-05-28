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

# Contact:
For any problems or questions please open an issue in github.
