# SEE-Net
The Tensorflow implementation of "[Order Matters: Shuffling Sequence
Generation for Video Prediction](https://bmvc2019.org/wp-content/uploads/papers/1023-paper.pdf)" (BMVC2019) by Junyan Wang, Bingzhang Hu, Yang Long, Yu Guan.


## Python packages
* Python: 3.6.8
* Tensorflow: 1.12.0
* CUDA 9.0

## Dataset & Pretrained Models
Make a directory `./data` for saving models and a directory `./pretrained` for pretrained models.
* The pretrained SEENet Model can be downloaded from https://www.dropbox.com/sh/m4jrus3x7cjyh9t/AADxVp06scNlQDWLsVTHXHD1a?dl=0
* The preprocessed KTH dataset can be downloaded from https://www.dropbox.com/sh/5wv2m9ov0usnikj/AAC6Fo3HMPIlXLho6pZKmNtUa?dl=0

## Train
Make a directory `./models` for saving models and a directory `./logs` for saving logs.

To train motion and content features:
> python3 ./src/main.py --train_feature True --test False

To train predict part:
> python3 ./src/main.py --train_feature False --test False

The predicted samples can be seen in `./samples` folder. The detailed arguments can be set up in `./src/args.py`
 

## Test
> python3 ./src/main.py --test True

The test predicted samples can be seen in `./samples/test` folder

## Examples

![KTHexamples](https://i.postimg.cc/G90WmFtS/all.png)

## Cite
If you use this code or reference our paper in your work please cite this publication as:

ArXiv Version:
```
@article{wang2019order,
  title={Order Matters: Shuffling Sequence Generation for Video Prediction},
  author={Wang, Junyan and Hu, Bingzhang and Long, Yang and Guan, Yu},
  journal={arXiv preprint arXiv:1907.08845},
  year={2019}
}
```

```
BMVC Version:
@inproceedings{wang2019order,
    author    = {Junyan Wang and
               Bingzhang Hu and
               Yang Long and
               Yu Guan},
    title     = {Order Matters: Shuffling Sequence Generation for Video Prediction},
    booktitle = {Proc. BMVA British Mach. Vis. Conf.},
    pages     = {275.1--275.14},
    year      = {2019}}
```
## Poster


