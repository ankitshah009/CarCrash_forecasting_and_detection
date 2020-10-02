Accident Forecasting in Traffic Camera CCTV Videos
====

## Requirements

Please use Python 2.7 and install required packages by
```bash
$ pip install -r requirements.txt
```
Please edit Keras configuration to load Tensorflow backend (we don't use Theano backend).
A GPU is recommended.
## Train/Test with original Faster R-CNN

### Data preparation

Assume that you have CADP dataset at `CADP_IMAGE_HOME`. 
For example, for video with ID 0, the frames should be in `CADP_IMAGE_HOME/000000/`.

##### Generating images

There are unlabeled regions in each images, therefore for task like object detections or tracking, anomalous detection, we should generate images with those regions are masked out.
To generate mask images, please modify the output path `CADP_MASK_HOME` in [cover_crowd_far.py](./analysis/cover_crowd_far.py), and then please run:

```bash
$ python analysis/cover_crowd_far.py --anno_dir=./data/annotations
```
where `anno_dir` is the directory containing VATIC format annotations (each file contains annotations for one video).

##### Generating CSV annotations
Please run the following command to output csv data for training/testing.

```bash
$ python analysis/generate_csv.py --anno_dir=./data/annotations/trainval/ --csv_output=./cadp_train.csv --use_mask=True
```
where `csv_output` is the output csv files and `use_mask` is the flag to specify whether to use masked images.

##### Download pretrained models
Please download pretrained Resnet-50 model from [https://github.com/fchollet/keras/tree/master/keras/applications](https://github.com/fchollet/keras/tree/master/keras/applications).
Assuming that you have put the pretrained Resnet model at `MODEL_DIR/resnet50_weights_tf_dim_ordering_tf_kernels.h5`.
Please set that path as the value of `cfg.base_net_weights` at L25 of [train_cadp_frcnn.py](./analysis/train_cadp_frcnn.py#L25).


### Training

After all above steps are done, please train Faster R-CNN with

```bash
$ python analysis/train_cadp_frcnn.py
```

### Testing

After training, to measure the mAP@IoU=0.5, please run

```bash
$ python analysis/measure_map.py --path=./cadp_test.csv --parser=simple
```
Where `path` is the CSV files containing the annotations of test set and `parser` must be `simple` to parse the annotations.


### Website - 

[Accident Forecasting Traffic Camera](https://ankitshah009.github.io/accident_forecasting_traffic_camera)

### Citation -
------------------------

```
@INPROCEEDINGS{8639160,
  author={A. P. {Shah} and J. {Lamare} and T. {Nguyen-Anh} and A. {Hauptmann}},
  booktitle={2018 15th IEEE International Conference on Advanced Video and Signal Based Surveillance (AVSS)}, 
  title={CADP: A Novel Dataset for CCTV Traffic Camera based Accident Analysis}, 
  year={2018},
  volume={},
  number={},
  pages={1-9},}
```
