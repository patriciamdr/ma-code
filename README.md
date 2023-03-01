# Exploiting Spatial-Temporal Relationships for Occlusion-Robust 3D Human Pose Estimation

## Dependencies

- Cuda 11.6
- Python 3.10.4
- Pytorch 1.12.1

## Dataset setup

Please download the dataset from [Human3.6M](http://vision.imar.ro/human3.6m/) website and refer to [VideoPose3D](https://github.com/facebookresearch/VideoPose3D) to set up the Human3.6M dataset ('./dataset' directory). 
Or you can download the processed data from [here](https://drive.google.com/drive/folders/112GPdRC9IEcwcJRyrLJeYw9_YV4wLdKC?usp=sharing). 

```bash
${POSE_ROOT}/
|-- dataset
|   |-- h36m
|   |   |-- data_3d_h36m.npz
|   |   |-- data_2d_h36m_gt.npz
|   |   |-- data_2d_h36m_cpn_ft_h36m_dbb.npz
```

## Test the models

To test on pretrained models on Human3.6M:

### Baseline:
```bash
python main.py --test --reload --refine_reload --refine --spatial_module --frames 81 --previous_dir checkpoint/baseline/1024_2239_22_81
```

### Data-driven Approach (Occlusion Augmentation):
```bash
python main.py --test --reload --refine_reload --refine --spatial_module --frames 81 --previous_dir checkpoint/occlusion-augmentation/consecutive/1208_1312_58_81 
```

### Model-driven Approach (Noise Prediction):
```bash
python main.py --test --reload --refine_reload --refine --spatial_module --frames 81 --previous_dir checkpoint/visibility/noextra/1125_1855_35_81 --use_visibility --error_thr 0.2  
```

## Train the models

To train on Human3.6M:

### Pre-Training modified GraFormer (single-frame 3D HPE):
```bash
python pretrain_graformer.py
```

### Baseline:

Transfer Learning (Phase 1):

```bash
python main.py --frames 81 --spatial_module --pretrained_spatial_module_init --pretrained_spatial_module_dir [your pre-trained modified GraFormer directory path] --pretrained_spatial_module [your pre-trained modified GraFormer file name inside directory] 
```

Fine-Tuning (Phase 2):

```bash
python main.py --frames 81 --spatial_module --reload --spatial_module_lr 1e-3 --previous_dir [your phase-1 model saved directory path]
```

Pose Refinement (Phase 3):

```bash
python main.py --frames 81 --spatial_module --reload --spatial_module_lr 1e-3 --refine --lr_refine 1e-3 --previous_dir [your phase-2 model saved directory path]
```

### Data-driven Approach (Occlusion Augmentation):

Transfer Learning (Phase 1):

```bash
python main.py --frames 81 --spatial_module --occlusion_augmentation_train --num_occluded_j 1 --consecutive_frames --subset_size 6 --pretrained_spatial_module_init --pretrained_spatial_module_dir [your pre-trained modified GraFormer directory path] --pretrained_spatial_module [your pre-trained modified GraFormer file name inside directory] 
```

Fine-Tuning (Phase 2):

```bash
python main.py --frames 81 --spatial_module --occlusion_augmentation_train --num_occluded_j 1 --consecutive_frames --subset_size 6 --reload --spatial_module_lr 1e-3 --previous_dir [your phase-1 model saved directory path]
```

Pose Refinement (Phase 3):

```bash
python main.py --frames 81 --spatial_module --reload --occlusion_augmentation_train --num_occluded_j 1 --consecutive_frames --subset_size 6 --spatial_module_lr 1e-3 --refine --lr_refine 1e-3 --previous_dir [your phase-2 model saved directory path]
```

### Pre-Training Auxiliary Model (single-frame 3D HPE + Noise Prediction):

```bash
python pretrain_auxiliary_vis.py --self_supervision --lin_layers --error_thr 0.2 --lr 1e-6 --lr_vis 1e-3 --pose_weight_factor 10 --pretrained_graformer_init --pretrained_graformer [your pre-trained modified GraFormer /path/to/file]
```

### Model-driven Approach (Noise Prediction):

Transfer Learning (Phase 1):

```bash
python main.py --frames 81 --spatial_module --use_visibility --pose_weight_factor 10 --error_thr 0.2 --pretrained_spatial_module_init --pretrained_spatial_module_dir [your pre-trained auxiliary model directory path] --pretrained_spatial_module [your pre-trained auxiliary model file name inside directory] 
```

Fine-Tuning (Phase 2):

```bash
python main.py --frames 81 --spatial_module --use_visibility --pose_weight_factor 10 --error_thr 0.2 --reload --spatial_module_lr 1e-3 --previous_dir [your phase-1 model saved directory path]
```

Pose Refinement (Phase 3):

```bash
python main.py --frames 81 --spatial_module --use_visibility --pose_weight_factor 10 --error_thr 0.2 --reload --spatial_module_lr 1e-3 --refine --lr_refine 1e-3 --previous_dir [your phase-2 model saved directory path]
```

## Occlusion Robustness Analysis - Missing Keypoints

To test a model's robustness against missing keypoints (occluding a specific joint [0-16] across 30 frames):

```bash
cd scripts/occlusion_robustness_analysis
```

```bash
python joint_importance_analysis.py --frames --spatial_module --previous_dir ../../checkpoint/PATH/TO/MODEL_DIR --root_path ../../dataset
```

## Acknowledgement

The code is built on top of [StridedTransformer](https://github.com/Vegetebird/StridedTransformer-Pose3D).

