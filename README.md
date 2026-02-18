# FPGA_Hackathon_RMTS

## [Project Proposal](https://drive.google.com/file/d/1YoAMCLCedslQCgVstt3f-OPIzyf1Hi53/view?usp=sharing)

## Rounds
1. *Due 15-03-2026:* Design and train an AI/ML model for a chosen real-world Edge AI application and implement the inference stage using Verilog RTL as a hardware accelerator.
2. *10-04-2026:* On-site deployment and real-time data interfacing using AMD/Xilinx FPGA platforms, along with expert sessions, workshops, poster presentations, and final evaluation.

## Dataset Gatherings
[All Datasets Gathered](https://docs.google.com/document/d/1vkHs6TT6vkpA83QuUCUTUFYyGv8xxEI254Gi0cglktQ/edit?tab=t.0#heading=h.smryuagthlx9, "Private for now...")

<!-- ## Datasets We Had Considered: -->
## Further Radar Datasets Overview

### [DARPA GOTCHA (Gotcha Volumetric SAR Data Set, Version 1.0)](https://www.sdms.afrl.af.mil/index.php?collection=gotcha)
- Real-world airborne X-band full-polarimetric spotlight SAR phase history data (raw k-space, not processed images)
- Circular 360° azimuth coverage over 8 elevation passes for volumetric/3D sparse aperture imaging
- ~11,520 .mat files (360 segments × 8 passes × 4 polarizations), 640 MHz bandwidth
- Urban parking lot scene with civilian vehicles (~9 cars + others) and calibration targets (trihedrals, dihedrals, tophat)
- Multi-pass temporal/sequential nature enables coherent processing, autofocus, change detection, and 3D reconstruction
- **Why we avoided this dataset: (TODO)** 

### [Open Radar Initiative Datasets](https://github.com/openradarinitiative/open_radar_datasets)
- Curated index/collection of open real-world radar datasets focused on micro-Doppler signature analysis
- Primary: Outdoor Moving Object Dataset — stationary FMCW radar, targets include persons (walking/bicycling), UAVs/drones, vehicles; .npy dict format with Doppler spectra, SNR, positions, radar params
- Secondary: Assisted Living Dataset — indoor radar for human activity recognition (daily movements)
- Rich metadata and viewer scripts/notebooks for exploration and benchmarking micro-Doppler classification/recognition
- **Why we avoided this dataset: (TODO)**

### [RadarScenes](https://radar-scenes.com/dataset/structure/)
- Real-world automotive radar point cloud dataset from vehicle with 4 FMCW radars + documentary camera
- 158 sequences, >4 hours driving (>100 km), ~4 million annotated detections, >7,500 unique objects
- Point-wise semantic annotations + track IDs in HDF5/JSON format; temporal/sequential with ego-motion compensation
- 12 classes: passenger cars, trucks, buses, bicycles, pedestrians, animals, static environment, etc.
- Designed for radar-based perception in diverse urban/rural scenarios
- **Why we avoided this dataset: (TODO)**

<!-- ## Idea Description -->
