# GenerativeSkinLesion  

Using [progressive gorwing of GANs](https://arxiv.org/abs/1710.10196) to augmentation skin lesion dataset.  

This project was originated as SFU CMPT-726 (18 Fall) course project. Collaborator: [Yongjun He](https://github.com/YongjunHe), [Ruochen Jiang](https://github.com/VHUCXAONG).   

I wrote an article (in Chinese) to discuss some tricks and implementation details. [Zhihu article](https://zhuanlan.zhihu.com/p/56244285)   

![](https://github.com/SaoYan/GenerativeSkinLesion/blob/master/sample_images.png)   

***

Be careful to use the code!  

* This may not be full implementation of progressive GANs, and we haven't run the code on that same dataset as the original paper.  
* Unsolved issue: the training fails when using PyTorch 1.0. I'm still not sure what difference between 1.0 and 0.4 causes this issue...

***

## Dependences  
* PyTorch 0.4 with torchvision  
* [tensorboardX](https://github.com/lanpa/tensorboardX)  
* [Pillow](https://github.com/python-pillow/Pillow)  

## Data  
ISIC 2018 classification task.  

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
