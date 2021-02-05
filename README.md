# Adaptive LiDAR Sampling and Depth Completion using Ensemble Variance
Code repository for our *NN* version of ["Adaptive LiDAR Sampling and Depth Completion using Ensemble Variance"](https://arxiv.org/abs/2007.13834) by [Eyal Gofer](https://www.vision-and-sensing.com/eyal-gofer), [Shachar Praisler](https://www.vision-and-sensing.com/shahar-praizler) and [Guy Gilboa](https://www.vision-and-sensing.com/prof-guy-gilboa) at Technion - Israel Institute of Technology.

<p align="center">
	<img src="https://i.imgur.com/kM2iPv7.png" | height=250>
</p>


For comparison and demonstation videos, please [see](https://www.vision-and-sensing.com/post/adaptive-lidar-sampling-and-depth-completion-using-ensemble-variance-new-publication).


## Requirements
This code was implemented with Python 3.6.3 and PyTorch 1.3.1 on Ubunto 18.04. \
All relevant packages can be installed using pip:
```bash
pip install -r requirements.txt
```


## Data & directory tree
Download the [KITTI Depth](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_completion) dataset as mentioned [here](https://github.com/fangchangma/self-supervised-depth-completion), in the 'Data' section. 
Complete directory tree:
```
├── DepthCompVar # code
├── data  # KITTI's directory tree
|   ├── data_depth_annotated
|   |   ├── train
|   |   ├── val # won't be used
|   |   ├── test
|   ├── data_depth_velodyne
|   |   ├── train
|   |   ├── val # won't be used
|   |   ├── test
|   ├── depth_selection
|   |   ├── test_depth_completion_anonymous # won't be used
|   |   ├── test_depth_prediction_anonymous # won't be used
|   |   ├── val_selection_cropped # actual validation
|   └── data_rgb
|   |   ├── train
|   |   ├── val # won't be used
|   |   ├── test
├── data_new # empty, at first. Will contain relevant data for a specific run
├── outputs # empty, at first. Will contain results
```

**data** - contains the original KITTI depth dataset. 
- The original dataset doesn't have 'test' folders, so move the relevant data there by yourself (inside 'test' folders, keep formatting KITTI's path convention such as: 2011_xxx_sync ..., while xxx can be *whatever* you want). 
- For validation, we used the 'val_selection_cropped' folder and not the 'val'. We split the KITTI 'val_selection_cropped' (1000 original validation images) into 203 validation images that were remained inside their original folders, and 797 test images that were moved out from the 'val_selection_cropped', into the 'test' folders (inside 'data_depth_annotated', 'data_depth_velodyne', 'data_rgb'). In addition, we ignored some syncs inside the original train set, because they were used for calibrations or didn't resemble urban driving scenarios.
- To sum up:
  - Training - our 'train' had 78,378 images (out of the original 85,898), excluded: 2011_09_26_drive_0104_sync, 2011_09_28_drive_0053-0222_sync, 2011_09_26_drive_0104_sync (perfect for dense urban driving, hence was a 'test' set for us during debugging. Can be included again).
  - Validation - our 'val_selection_cropped' had 203 images from: 2011_09_26_drive_0002_sync, 2011_09_26_drive_0005_sync, 2011_09_26_drive_0079_sync, 2011_09_26_drive_0095_sync, 2011_09_29_drive_0026_sync. 
  - Test - our 'test' had 797 images taken from 'val_selection_cropped', from: 2011_09_26_drive_0013_sync, 2011_09_26_drive_0020_sync, 2011_09_26_drive_0023_sync, 2011_09_26_drive_0036_sync, 2011_09_26_drive_0113_sync, 2011_09_28_drive_0037_sync, 2011_09_30_drive_0016_sync, 2011_10_03_drive_0047_sync.

**data_new** - contains the relevant data for a specific run of our algorithm (each time we run the algorithm, the folder's content will be deleted and the relevant data will be created automatically). Some will be taken from 'data' folders, and some, like the new 'velodyne_raw' (the depth input to the NNs) for the next phase, will be produced during the algorithm run. Train & val sets will be created based on the given sizes parameters, and the test set will be taken as a whole, based on what's inside 'data'.
```
data_new
├── phase_1
|   ├── mini_set_1
|   |   ├── data_depth_annotated
|   |   ├── data_depth_velodyne
|   |   ├── data_rgb
|   |   ├── depth_selection
|   ├── ...
|   ├── mini_set_M
├── phase_2
|   ├── mini_set_1
|   |   ├── data_depth_velodyne # train & test: the accumulated depth samples, generated based on last phase
|   |   ├── depth_selection # validation: the accumulated depth samples, generated based on last phase
|   |   ├── predictions_tmp # train & test
|   |   ├── predictions_tmp_val_select # validation
|   ├── mini_set_2
|   |   ├── data_depth_velodyne # train
|   |   ├── predictions_tmp # train
|   ├── ...
|   ├── mini_set_M
├── ...
├── phase_K+1
├── var_final_NN # will contain entire relevant data (train, val, test - rgb, vel, gt)
|   ├── data_depth_annotated
|   ├── data_depth_velodyne
|   ├── data_rgb
|   ├── depth_selection
```

Clarifications:
- During the phases: ground-truth & rgb images will be placed only in phase_1 folder, because they don't change during the phases (unlike the accumulated 'velodyne_raw'). 
- During the phases: test & validation images will be placed only in mini_set_1 because they are shared by all. 
- The predictions (the images) of the k-th phase are saved in the k+1-th phase folder, same as the chosen samples ("velodyne_raw") that were based on them.

**outputs** - contains outputs and weights of the M depth completion predictors during the K phases (NN*x*\phase*y*), and the final predictor's (var_final_NN).


## Run algorithm
A complete list of parameters is available with:
```bash
python main.py -h
```

#### Generating samples (a) and predicting depth (b):
Complete algorithm (include train & save of final dense depth predictions on 'test' as images). Default parameters:
```bash
# train with rgb in phase 1 and with rgbd in all other phases + final
python main.py
```
Some examples:
```bash
# Desired total budget of 2048 samples, accumulated during 2 phases, 3 predictors/mini-sets, train with d input
# in all phases + final
python main.py -B 2048 -K 2 -M 3 --just-d

# Train with rgbd in all phases + final, Ma's NN with Resnet34, train batch size of 2, prediction batch size of 6
python main.py --allrgbd --layers 34 --train-bs 2 --pred-bs 6

# Work with a small portion of the KITTI training dataset, trying reach a minimum size of 4000 per mini-set 
# (taking entire syncs), 4 predictors/mini-sets, whole training set will be ~(M(=4) * 4000).
# [if entire train dataset is ~80K, so 4*4,000 << 80,000]
python main.py --miniset-size 4000 --big-portion False -M 4

# Work with a large portion of the KITTI training dataset, trying reach a minimum size of 15000 per mini-set 
# (taking entire syncs). [if entire train dataset is ~80K, so 4*15,000 < 80,000]
python main.py --miniset-size 15000 --big-portion True -M 4

# Work with entire KITTI train dataset (taking entire syncs)
# [if entire train dataset is ~80K, so 3*27,000 > 80,000. The exact number 27 isn't important, as long > remains]
python main.py --miniset-size 27000 --big-portion True -M 3

# Work with 1000 val_select images as validation set, sample stragety is greedy (MAX)
python main.py --val-size 1000 --samp-method greedy
```

Note:
- We sample from the ground-truth and not from the velodyne-raw, so we could have more sampling options.
- We're trying to reach equally (as possible) mini-sets sizes. '--big-portion True' will "take" the bigger syncs first, '--big-portion False' will "take" the smaller.

#### Generating samples (a) and predicting depth (b) - without training, only on 'test' set:
Inference 'test' set only based on given weights, resulting error metrics & saving dense depth prediction images:
```bash
python main.py --inference /path/to/the/text/file/with/your/weights --other_relevant_arguments
```
For instance (with default parameters),
```bash
python main.py --inference /home/Adaptive-LiDAR-Sampling/trained_weights.txt
```
Note:
- The .txt file has to have the same format as the given 'trained_weights.txt' file, meaning: all weights inside a phase are together, phases are divided by an empty line, the final NN weights are in the end. Relative path can be replaced by a full path.
- The results may vary a little bit from time to time because the PM process has randomness in it.
- Must relevant_arguments: same -K, -M, --just-d, --allrgbd as it was for the trained weights. Optional (most likely, you want it to be the same too): same -B, --samp-method.


## Runtime
It may be different to your machine, due to different CPUs.
Our complete algorithm:
- A run with default parameters takes ~18.6h on 2 GeForce RTX 2080 Ti. A similar run with 4 GPUs and pred-bs=8 saves ~33% of the time.
- A run with default parameters, beside pred-bs=16, takes ~16.5h on 2 Tesla V100-DGx-S-32GB. A similar run with 4 GPUs and pred-bs=32 saves ~33% of the time.
- A run with default parameters, beside train-bs=28 and pred-bs=32, takes ~9.5h on 4 Tesla V100-DGx-S-32GB.

Generating samples & depth predictions on 'test':
- Takes ~22 minutes for 797 images with default parameters on 2 GeForce RTX 2080 Ti.
- Takes ~20 minutes for 797 images with default parameters and pred-bs=16 on 2 Tesla V100-DGx-S-32GB. A similar run with 4 GPUs and pred-bs=32 takes ~19 min.


## Citation
If you find our work useful, please cite our paper:
```bash
@article{gofer2020adaptive,
  title={Adaptive LiDAR Sampling and Depth Completion using Ensemble Variance},
  author={Gofer, Eyal and Praisler, Shachar and Gilboa, Guy},
  journal={arXiv preprint arXiv:2007.13834},
  year={2020}
}
```

Feel free to place issues here or contact me via the e-mail in my personal page.


## Acknowledgements:
We took inspirations for this repository from [Fangchang Ma](https://github.com/fangchangma/self-supervised-depth-completion) and 
[Damian Kaliroff](https://github.com/dkaliroff/phitnet), thus we thank them for that.
In addition, this NN-based method uses the supervised version (sparse-to-dense) of Ma et al. [code](https://github.com/fangchangma/self-supervised-depth-completion).
