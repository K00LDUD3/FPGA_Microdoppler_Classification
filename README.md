# Radar Microdoppler Classification 

<!---
## [Project Proposal](https://drive.google.com/file/d/1YoAMCLCedslQCgVstt3f-OPIzyf1Hi53/view?usp=sharing, "private for now...")

## Rounds
1. *Due 15-03-2026:* Design and train an AI/ML model for a chosen real-world Edge AI application and implement the inference stage using Verilog RTL as a hardware accelerator.
2. *10-04-2026:* On-site deployment and real-time data interfacing using AMD/Xilinx FPGA platforms, along with expert sessions, workshops, poster presentations, and final evaluation.

## Dataset Gatherings
[All Datasets Gathered](https://docs.google.com/document/d/1vkHs6TT6vkpA83QuUCUTUFYyGv8xxEI254Gi0cglktQ/edit?tab=t.0#heading=h.smryuagthlx9, "Private for now...")
--->

<!-- ## Datasets We Had Considered: -->
## Radar Dataset Overview

### [**Open Radar Datasets**](https://github.com/openradarinitiative/open_radar_datasets)
- Primary: Outdoor Moving Object Dataset — stationary FMCW radar, targets include persons (walking/bicycling), UAVs/drones, vehicles; .npy dict format with Doppler spectra, SNR, positions, radar params
- Rich metadata and viewer scripts/notebooks for exploration and benchmarking micro-Doppler classification/recognition
- Large Scale (>30,000 time frames per class) 
- **Moving** Objects in front of the radar

<!-- 
### Other Datasets
#### [DARPA GOTCHA (Gotcha Volumetric SAR Data Set, Version 1.0)](https://www.sdms.afrl.af.mil/index.php?collection=gotcha)
- Real-world airborne X-band full-polarimetric spotlight SAR phase history data (raw k-space, not processed images)
- Circular 360° azimuth coverage over 8 elevation passes for volumetric/3D sparse aperture imaging
- ~11,520 .mat files (360 segments × 8 passes × 4 polarizations), 640 MHz bandwidth
- Urban parking lot scene with civilian vehicles (~9 cars + others) and calibration targets (trihedrals, dihedrals, tophat)
- Multi-pass temporal/sequential nature enables coherent processing, autofocus, change detection, and 3D reconstruction
- **Why we avoided this dataset:** Images are Top-Down View.




#### [RadarScenes](https://radar-scenes.com/dataset/structure/)
- Real-world automotive radar point cloud dataset from vehicle with 4 FMCW radars + documentary camera
- 158 sequences, >4 hours driving (>100 km), ~4 million annotated detections, >7,500 unique objects
- Point-wise semantic annotations + track IDs in HDF5/JSON format; temporal/sequential with ego-motion compensation
- 12 classes: passenger cars, trucks, buses, bicycles, pedestrians, animals, static environment, etc.
- Designed for radar-based perception in diverse urban/rural scenarios
- **Why we avoided this dataset:** Since the RadarScenes dataset offers radar point-cloud detections instead of fixed-size radar tensors, we decided not to use it. To create CNN-compatible inputs, this would necessitate extensive preprocessing (binning, gridding, multi-sensor fusion, and clutter filtering), which would increase engineering overhead and timeline risk. A dataset with pre-structured radar tensors and balanced class distributions is more compatible with fixed-dimension convolutional architectures and effective FPGA implementation, as our goal is to design and implement a hardware-accelerated CNN on FPGA. Because of its scalability and tensor-ready format, we chose the Open Radar moving-object dataset.
---> 
<!-- ## Idea Description -->
## Repo Structure

```text
├── Assets/                         -- Inputs for notebooks
│   ├── Dataset/                        -- Processed datasets
│   └── Models/                         -- Neural network architecture definitions (importable modules)
│
├── Logs/                           -- Experiment tracking and training logs
│   ├── experiments_metadata.json       -- Model hyperparameter and architecture + version registry
│   ├── proprocess_metadata.json        -- Processed dataset hyperparameter + version registry
│   └── training.log                    -- Epoch-level training logs
│
├── Outputs/Models                        -- Trained model state dictionaries and metrics 
│
├── Utils/                          -- Utilities for logs, outputs, and notebooks
│   ├── json_inter.py                   -- Used to intereface (read as csv/json,append) with .json logs 
│   └── plotting.py                     -- plotting dataset and training metrics
│   └── paths.py                        -- Collection of paths to validate structure, ensure everything exists
│   └── arch_logging.py                 -- Used to fetch non-default hyperparameters to log
└── Notebooks/                      -- Experimentation notebooks
```

## Results (Ongoing)
### Models
<!-- 
    | **Model**                   | **Parameters** | **MACs (Multiply-Adds)** | **Accuracy** | **Latencies**                  |
    |-----------------------------|:----------------:|:--------------------------:|:--------------:|:---------------------------------|
    | **BasicRadarCNN**         | 810,596        | 15,999,844               | 97.00%       | Torch CPU: 1.035 ms            <br> Torch GPU: 1.086 ms            <br> ONNX CPU: 0.206 ms             <br> ONNX GPU: 0.220 ms             |
    | **cnn_lite_v3**           | 32,932         | 18,764,164               | 90.89%       | Torch CPU: 1.059 ms            <br> Torch GPU: 0.570 ms            <br> ONNX CPU: 0.183 ms             <br> ONNX GPU: 0.261 ms             |
    | **cnn_distill_d3_2block** | 1,028          | 4,915,428                | 95.20%       | Torch CPU: 1.161 ms            <br> Torch GPU: 0.803 ms            <br> ONNX CPU: 0.312 ms             <br> ONNX GPU: 0.526 ms             | -->


| **Model**                   | **Parameters** | **MACs (Multiply-Adds)** | **Accuracy <br> (%)** | **Torch CPU Latency <br> (ms)** | **Torch GPU Latency <br>  (ms)** | **ONNX CPU Latency<br>(ms)** | **ONNX GPU Latency <br> (ms)** |
|-----------------------------|:--------------:|:------------------------:|:------------:|:---------------------:|:---------------------:|:--------------------:|:--------------------:|
| **[BasicRadarCNN](#1-initial-model---basicradarcnn)**           |     810,596    |      15,999,844          |    97.00    | &nbsp;1.035           | &nbsp;1.086           | &nbsp;0.206          | &nbsp;0.220          |
| **[cnn_lite_v3](#2-an-attempt-at-parameter-reduction---cnn_lite_v3)**             |     32,932     |      18,764,164          |    90.89    | &nbsp;1.059           | &nbsp;0.570           | &nbsp;0.183          | &nbsp;0.261          |
| **[cnn_distill_d3_2block](#3-using-a-depthwise--pointwise-conv2d-layer---most-lightweight-model-so-far---cnn_distill_d3_2block)**   |     1,028      |      4,915,428           |    95.20    | &nbsp;1.161           | &nbsp;0.803           | &nbsp;0.312          | &nbsp;0.526          |
<!-- <table style="border-collapse: collapse; width: 100%;">
  <tr>
    <th><strong>Model</strong></th>
    <th><strong>Parameters</strong></th>
    <th><strong>MACs (Multiply-Adds)</strong></th>
    <th><strong>Accuracy (%)</strong></th>
    <th style="border-left: 2px solid #000; padding-left: 10px; padding-right: 10px;"></th>
    <th><strong>Torch CPU Latency (ms)</strong></th>
    <th><strong>Torch GPU Latency (ms)</strong></th>
    <th><strong>ONNX CPU Latency (ms)</strong></th>
    <th><strong>ONNX GPU Latency (ms)</strong></th>
  </tr>
  <tr>
    <td>BasicRadarCNN</td>
    <td>810,596</td>
    <td>15,999,844</td>
    <td>97.00</td>
    <td></td> 
    <td style="border-left: 2px solid #000;">1.035</td>
    <td style="border-left: 2px solid #000;">1.086</td>
    <td style="border-left: 2px solid #000;">0.206</td>
    <td style="border-left: 2px solid #000;">0.220</td>
  </tr>
  <tr>
    <td>cnn_lite_v3</td>
    <td>32,932</td>
    <td>18,764,164</td>
    <td>90.89</td>
    <td></td> 
    <td style="border-left: 2px solid #ffffff;">1.059</td>
    <td style="border-left: 2px solid #000;">0.570</td>
    <td style="border-left: 2px solid #000;">0.183</td>
    <td style="border-left: 2px solid #000;">0.261</td>
  </tr>
  <tr>
    <td>cnn_distill_d3_2block</td>
    <td>1,028</td>
    <td>4,915,428</td>
    <td>95.20</td>
    <td></td> 
    <td style="border-left: 2px solid #000;">1.161</td>
    <td style="border-left: 2px solid #000;">0.803</td>
    <td style="border-left: 2px solid #000;">0.312</td>
    <td style="border-left: 2px solid #000;">0.526</td>
  </tr>
</table> -->
In the sequence of experimenting and creation:

#### 1. Initial Model - BasicRadarCNN
- Accuracy  : 97.00 %
- Params    : 810,596 float32
- Total MACs: 15,999,844
- Latencies
    - torch CPU   : 1.035 ms
    - torch GPU   : 1.086 ms
    - ONNX CPU    : 0.206 ms
    - ONNX GPU    : 0.220 ms

MODEL ARCH
```text 
======================================================================================================================================================================
Layer (type (var_name))                  Input Shape        Output Shape       Param #            Param %            Kernel Shape       Mult-Adds          Trainable
======================================================================================================================================================================
BasicRadarCNN (BasicRadarCNN)            [1, 1, 48, 128]    [1, 4]             --                      --            --                 --                 True
├─Sequential (features)                  [1, 1, 48, 128]    [1, 64, 6, 16]     --                      --            --                 --                 True
│    └─Conv2d (0)                        [1, 1, 48, 128]    [1, 16, 48, 128]   160                  0.02%            [3, 3]             983,040            True
│    └─BatchNorm2d (1)                   [1, 16, 48, 128]   [1, 16, 48, 128]   32                   0.00%            --                 32                 True
│    └─ReLU (2)                          [1, 16, 48, 128]   [1, 16, 48, 128]   --                      --            --                 --                 --
│    └─MaxPool2d (3)                     [1, 16, 48, 128]   [1, 16, 24, 64]    --                      --            2                  --                 --
│    └─Conv2d (4)                        [1, 16, 24, 64]    [1, 32, 24, 64]    4,640                0.57%            [3, 3]             7,127,040          True
│    └─BatchNorm2d (5)                   [1, 32, 24, 64]    [1, 32, 24, 64]    64                   0.01%            --                 64                 True
│    └─ReLU (6)                          [1, 32, 24, 64]    [1, 32, 24, 64]    --                      --            --                 --                 --
│    └─MaxPool2d (7)                     [1, 32, 24, 64]    [1, 32, 12, 32]    --                      --            2                  --                 --
│    └─Conv2d (8)                        [1, 32, 12, 32]    [1, 64, 12, 32]    18,496               2.28%            [3, 3]             7,102,464          True
│    └─BatchNorm2d (9)                   [1, 64, 12, 32]    [1, 64, 12, 32]    128                  0.02%            --                 128                True
│    └─ReLU (10)                         [1, 64, 12, 32]    [1, 64, 12, 32]    --                      --            --                 --                 --
│    └─MaxPool2d (11)                    [1, 64, 12, 32]    [1, 64, 6, 16]     --                      --            2                  --                 --
├─Sequential (classifier)                [1, 6144]          [1, 4]             --                      --            --                 --                 True
│    └─Linear (0)                        [1, 6144]          [1, 128]           786,560             97.03%            --                 786,560            True
│    └─ReLU (1)                          [1, 128]           [1, 128]           --                      --            --                 --                 --
│    └─Dropout (2)                       [1, 128]           [1, 128]           --                      --            --                 --                 --
│    └─Linear (3)                        [1, 128]           [1, 4]             516                  0.06%            --                 516                True
======================================================================================================================================================================
```

#### 2. An attempt at parameter reduction - cnn_lite_v3
- Accuracy  : 90.89 %
- Parameters: 32,932 float32
- Total MACs: 18,764,164
- Latencies
    - torch_cpu_ms   : 1.059 ms
    - torch_gpu_ms   : 0.570 ms
    - onnx_cpu_ms    : 0.183 ms
    - onnx_gpu_ms    : 0.261 ms


MODEL ARCH
```text
======================================================================================================================================================================
Layer (type (var_name))                  Input Shape        Output Shape       Param #            Param %            Kernel Shape       Mult-Adds          Trainable
======================================================================================================================================================================
cnn_lite_v3 (cnn_lite_v3)                [1, 1, 48, 128]    [1, 4]             --                      --            --                 --                 True
├─Sequential (features)                  [1, 1, 48, 128]    [1, 96, 6, 16]     --                      --            --                 --                 True
│    └─Conv2d (0)                        [1, 1, 48, 128]    [1, 16, 48, 128]   160                  0.49%            [3, 3]             983,040            True
│    └─ReLU (1)                          [1, 16, 48, 128]   [1, 16, 48, 128]   --                      --            --                 --                 --
│    └─MaxPool2d (2)                     [1, 16, 48, 128]   [1, 16, 24, 64]    --                      --            2                  --                 --
│    └─Conv2d (3)                        [1, 16, 24, 64]    [1, 32, 24, 64]    4,640               14.09%            [3, 3]             7,127,040          True
│    └─ReLU (4)                          [1, 32, 24, 64]    [1, 32, 24, 64]    --                      --            --                 --                 --
│    └─MaxPool2d (5)                     [1, 32, 24, 64]    [1, 32, 12, 32]    --                      --            2                  --                 --
│    └─Conv2d (6)                        [1, 32, 12, 32]    [1, 96, 12, 32]    27,744              84.25%            [3, 3]             10,653,696         True
│    └─ReLU (7)                          [1, 96, 12, 32]    [1, 96, 12, 32]    --                      --            --                 --                 --
│    └─MaxPool2d (8)                     [1, 96, 12, 32]    [1, 96, 6, 16]     --                      --            2                  --                 --
├─AdaptiveAvgPool2d (gap)                [1, 96, 6, 16]     [1, 96, 1, 1]      --                      --            --                 --                 --
├─Linear (fc)                            [1, 96]            [1, 4]             388                  1.18%            --                 388                True
======================================================================================================================================================================
```
<!-- Total params: 32,932
Trainable params: 32,932
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 18.76
======================================================================================================================================================================
Input size (MB): 0.02
Forward/backward pass size (MB): 1.47
Params size (MB): 0.13
Estimated Total Size (MB): 1.63
======================================================================================================================================================================

================================================================================
FPGA DEPLOYMENT METRICS
================================================================================
+--------------------------------+------------------------------------+
| Metric                         | Value                              |
+--------------------------------+------------------------------------+
| Total Parameters               | 32,932                             |
| Trainable Parameters           | 32,932                             |
| Non-Trainable Parameters       | 0                                  |
| Total MACs (Mult-Adds)         | 18,764,164                         |
| Estimated MACs (GOps)          | 0.0188 GOps                        |
| Parameter Memory (float32)     | 0.13 MB  /  128.64 KB              |
| INT8 Parameter Memory          | 0.03 MB  (4x reduction)            |
| INT4 Parameter Memory          | 0.02 MB  (8x reduction)            |
+--------------------------------+------------------------------------+ -->
#### 3. Using a Depthwise & Pointwise Conv2D Layer - **Most Lightweight Model So Far** - cnn_distill_d3_2block
- Accuracy  : 95.20 %
- Parameters: 1028 float32
- Total MACs: 4,915,428
- Latencies : 
    - torch CPU   : 1.161 ms
    - torch GPU   : 0.803 ms
    - ONNX CPU    : 0.312 ms
    - ONNX GPU    : 0.526 ms

*Note*: A quantized version of this exact model yields a 82.73% test accuracy. Will have to fine tune later.

MODEL ARCH

```text
================================================================================================================================================================================
Layer (type (var_name))                            Input Shape        Output Shape       Param #            Param %            Kernel Shape       Mult-Adds          Trainable
================================================================================================================================================================================
cnn_distill_d3_2block (cnn_distill_d3_2block)      [1, 1, 48, 128]    [1, 4]             --                      --            --                 --                 True
├─Sequential (block1)                              [1, 1, 48, 128]    [1, 16, 48, 128]   --                      --            --                 --                 True
│    └─Conv2d (0)                                  [1, 1, 48, 128]    [1, 16, 48, 128]   144                 14.01%            [3, 3]             884,736            True
│    └─BatchNorm2d (1)                             [1, 16, 48, 128]   [1, 16, 48, 128]   32                   3.11%            --                 32                 True
│    └─GELU (2)                                    [1, 16, 48, 128]   [1, 16, 48, 128]   --                      --            --                 --                 --
├─Sequential (block2)                              [1, 16, 48, 128]   [1, 32, 24, 64]    --                      --            --                 --                 True
│    └─DepthwiseSeparableConv (0)                  [1, 16, 48, 128]   [1, 32, 48, 128]   --                      --            --                 --                 True
│    │    └─Conv2d (depthwise)                     [1, 16, 48, 128]   [1, 16, 48, 128]   144                 14.01%            [3, 3]             884,736            True
│    │    └─Conv2d (pointwise)                     [1, 16, 48, 128]   [1, 32, 48, 128]   512                 49.81%            [1, 1]             3,145,728          True
│    │    └─BatchNorm2d (bn)                       [1, 32, 48, 128]   [1, 32, 48, 128]   64                   6.23%            --                 64                 True
│    └─GELU (1)                                    [1, 32, 48, 128]   [1, 32, 48, 128]   --                      --            --                 --                 --
│    └─MaxPool2d (2)                               [1, 32, 48, 128]   [1, 32, 24, 64]    --                      --            2                  --                 --
├─AdaptiveAvgPool2d (gap)                          [1, 32, 24, 64]    [1, 32, 1, 1]      --                      --            --                 --                 --
├─Linear (fc)                                      [1, 32]            [1, 4]             132                 12.84%            --                 132                True
================================================================================================================================================================================
```
<!-- Total params: 1,028
Trainable params: 1,028
Non-trainable params: 0
Total mult-adds (Units.MEGABYTES): 4.92
================================================================================================================================================================================
Input size (MB): 0.02
Forward/backward pass size (MB): 5.51
Params size (MB): 0.00
Estimated Total Size (MB): 5.53
================================================================================================================================================================================ -->