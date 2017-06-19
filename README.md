# Faster RCNN via Tensorflow

This is experimental fork of [smallcorgi TF implementation](https://github.com/smallcorgi/Faster-RCNN_TF).

### What was changed

1. Deleted a lot of unused and dead code
2. Deleted all Matlab and selective search references
3. Switched to python3
4. Added tensorboard logging of metrics and validation into `tensorboard_data` folder
5. Added simple experiments runner via 'experiments_runner.py' + configs
6. Added validation on validation set during training.
7. Save boxes, models, metrics and cache data into `output/{EXP_NAME}` folder

### How to install

1. Install Tensorflow
2. Install python dependencies `pip3 install -r requirements.txt`
3. Clone this Faster R-CNN repository
4. Build the Cython modules
    ```Shell
    cd $FRCN_ROOT/faster_rcnn
    make
    ```
5. Download pre-trained ImageNet models

   Download the pre-trained ImageNet models [[Google Drive]](https://drive.google.com/open?id=0ByuDEGFYmWsbNVF5eExySUtMZmM) [[Dropbox]](https://www.dropbox.com/s/po2kzdhdgl4ix55/VGG_imagenet.npy?dl=0)
   
    ```Shell
    mkdir data
    cd data & mkdir pretrain_model
    mv VGG_imagenet.npy $FRCN_ROOT/data/pretrain_model/VGG_imagenet.npy
    ```

### How to run

This repo was tested and tied for single class detection for custom dataset called `dental` that was organized in VOC Pascal format. Also whole training pipeline was tested on GPU.

1. Create dataset in VOC format
2. Create symlink to `data` folder

   ```Shell
   cd $FRCN_ROOT/data
   ln -s $PATH_TO_DENTAL dental
   ```
3. Set up experiments params in new yml file in `experiments/cfgs` folder. You can do it by changing `base_config.yml` or create separate file
4. From `$FRCN_ROOT` folder run

   ```Shell
   python3 ./tools/train_net.py --cfg experiments/cfgs/{CONFIG_NAME}
   ```
5. Check progress in tensorboard

   ```
   tensorboard --logdir=$PATH_TO_PROJECT/tensorboard_data/{EXP_NAME}
   ```
6. Run prediction on validation for certain model iteration
   ```Shell
   python3 ./tools/test_net.py --device gpu --device_id 0  --cfg ./experiments/cfgs/{CONFIG_NAME}  --num_iter {ITER_NUM}
   ```
7. Trained model, bboxes and images will be in `output/{EXP_NAME}` folder


### How to customize to new dataset

Current datasets are tightly connected and hardcoded into library code (that's no good).
All workflow were tested via one class detection and many class can work not properly in validation pipeline.

To add a custom dataset:

1. Convert it to VOC Pascal
2. Make symlink and call it `dental` (because now this name is hardcoded in `faster_rcnn/datasets/factory.py`)

   ```Shell
   cd $FRCN_ROOT/data
   ln -s $PATH_TO_YOUR_DATA dental
   ```
3. Fix class name in 'faster_rcnn/datasets/factory.py#L27' from `dental` to your class name
4. Other stuff is similar as described in `How to run`

### How to run experiments query

For parallel and sequential experiments execution was written simple script - `experiments_runner.py`.
It takes basic config from `experiments/cfgs/base_config.yml` and experiments variations from `expertiments/experiment_list.py` and execute them in multiprocessing way.

In experiments store one-line diff variations for base params, which are merged and run by script.
It's important to change `EXP_NAME` in every experiment to save data into different folders.


### How to run forward pass on custom images folder

To run forward pass on folder with images you need to run following script

```Shell
python3 ./tools/predict_folder.py --device gpu --device_id 0  --cfg ./experiments/cfgs/{CONFIG_NAME}  --num_iter {ITER_NUM} --data_path {PATH_TO_DATA}
```

Result will be saved to `output/{EXP_NAME}/folder_predict/image_detections.csv`

### What is pending to fix/implement

1. Configurable datasets without hardcode :(
2. Weights sharing between validation/train :(
3. Remove more unused code. (Big part of preprocessing looks like not required)
4. Test with more params variations
5. More flexible model saving to reduce HDD space consumption
6. GPU parallelization for experiments


### Training Model on Pascal dataset

1. Download the training, validation, test data and VOCdevkit

	```Shell
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
	wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
	```

2. Extract all of these tars into one directory named `VOCdevkit`

	```Shell
	tar xvf VOCtrainval_06-Nov-2007.tar
	tar xvf VOCtest_06-Nov-2007.tar
	tar xvf VOCdevkit_08-Jun-2007.tar
	```

3. It should have this basic structure

	```Shell
  	$VOCdevkit/                           # development kit
  	$VOCdevkit/VOCcode/                   # VOC utility code
  	$VOCdevkit/VOC2007                    # image sets, annotations, etc.
  	# ... and several other directories ...
  	```

4. Create symlinks for the PASCAL VOC dataset

	```Shell
    cd $FRCN_ROOT/data
    ln -s $VOCdevkit VOCdevkit2007
    ```
5. Change in config dataset name to `DATASET_NAME: voc_2007`
6. Run training and validation as described in `How to run`

### References
[Faster R-CNN caffe version](https://github.com/rbgirshick/py-faster-rcnn)

[smallcorgi TF implementation](https://github.com/smallcorgi/Faster-RCNN_TF)

