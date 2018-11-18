# GenerativeSkinLesion
Using GANs to augmentation skin lesion dataset

## Dependences  
* PyTorch (>=0.4.1) with torchvision  
* [tensorboardX](https://github.com/lanpa/tensorboardX)  
* [Pillow](https://github.com/python-pillow/Pillow)  

## Data  
[Google Drive Link](https://drive.google.com/drive/folders/1lndmIp75e1uo2cmsdV15yYDMhWRXeCUc?usp=sharing)  

## How to run  

### Step 1 : Adjust the default parameters  

* Use one GPU    
```
device_ids = [0]
```  

* Use smaller batch size (otherwise OOM for one GPU)  
```
parser.add_argument("--batch_size", type=int, default=8)
```  

* Path of the dataset   
```
if opt.preprocess:
      preprocess_data_gan('YourOwnPath')
```  

### Step 2 : Train   

```
python train_gan.py --preprocess
```  

## Which parameters can be further tuned?

* --unit_epoch: train for more epochs
* --num_aug: maybe try 5 for data_2017 and 10 for data_2016, that is around 1000 images per "actual" epoch

## 更新日志
* 11.17：参照Tensorflow源码，修正了两个细节    
    * [Refer to this commit](https://github.com/SaoYan/GenerativeSkinLesion/commit/9747160c1424b8c5a45aed2fef856c7bf46aadc1?diff=unified) 训练某个分辨率的阶段，需要将256x256的图片下采样到相应的分辨率作为训练样本，之前代码中这个下采样采用了nearest neighboring，现在修正成了PIL.Image.ANTIALIAS（该参数参见[Pillow文档](https://pillow.readthedocs.io/en/3.0.0/reference/Image.html#PIL.Image.Image.resize)）    
    * [Refer to this commit](https://github.com/SaoYan/GenerativeSkinLesion/commit/d9ad43bb83ec800a539849ccd894545d086f2a16) 在G的fade in阶段，调换了old_to_rgb和old_upsample的顺序
