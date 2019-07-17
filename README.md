# SEE-Net
The Tensorflow implementation of "Order Matters: Shuffling Sequence
Generation for Video Prediction" (BMVC2019) by Junyan Wang, Bingzhang Hu, Yang Long, Yu Guan.

## Python packages
* Python: 3.6.8
* Tensorflow: 1.12.0

## Dataset & Pretrained Models
* The preprocessed KTH dataset can be downloaded from https://www.dropbox.com/sh/bad23wroxv7pmvl/AAB-8OIipqnHSdOHpB9z72iEa?dl=0
* The pretrained SEENet Model is in `./pretrained` folder.

## Train
Make a directory `./models` for saving models and a directory `./logs` for saving logs.

To train motion and content features:
> python3 ./src/main.py --train_feature True

To train predict part:
> python3 ./src/main.py --train_feature False

The predicted samples can be seen in `./samples` folder. The detailed arguments can be set up in `./src/args.py`
 

## Test
> python3 ./src/main.py --test True

The test predicted samples can be seen in `./samples/test` folder

## Cite
If you use this code or reference our paper in your work please cite this publication as:
```

```


