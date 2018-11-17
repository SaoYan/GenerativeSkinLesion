# GenerativeSkinLesion
Using GANs to augmentation skin lesion dataset

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
