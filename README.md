# GenerativeSkinLesion  

Using [progressive gorwing of GANs](https://arxiv.org/abs/1710.10196) to augmentation skin lesion dataset.  

This project was originated as SFU CMPT-726 (18 Fall) course project. Collaborator: [Yongjun He](https://github.com/Nju141250047), [Ruochen Jiang](https://github.com/VHUCXAONG).   

![](https://github.com/SaoYan/GenerativeSkinLesion/blob/master/sample_images.png)   

***

Be careful to use the code!  

This may not be full implementation of progressive GANs, and we haven't run the code on that same dataset as the original paper.  

***

## Dependences  
* PyTorch (>=0.4.1) with torchvision  
* [tensorboardX](https://github.com/lanpa/tensorboardX)  
* [Pillow](https://github.com/python-pillow/Pillow)  

## Data  
*Melanoma* image in ISIC 2017 or ISIC 2018 classification task.  

## Training  

```
python main.py --mode train
```

The training taks about 4 days on 4 NVIDIA P100 Pascal GPUs.  

## (After training) Generating  

```
python main.py --mode test --num 1000
```

## Acknowledgement  

* [tkarras/progressive_growing_of_gans](https://github.com/tkarras/progressive_growing_of_gans) (official Tensorflow implementation)  
* [akanimax/pro_gan_pytorch](https://github.com/akanimax/pro_gan_pytorch)  
* [nashory/pggan-pytorch](https://github.com/nashory/pggan-pytorch)  
