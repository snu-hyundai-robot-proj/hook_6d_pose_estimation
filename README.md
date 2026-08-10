.Hook 6D Pose Estimation

Zivid 3D camera and RF-DETR segmentation are used to estimate the 6D pose of a hook.The current version generates global pose candidates with PCA and FPFH-RANSAC, refines them with the Hook CAD model and a Canonical Observed Reference, and outputs the final hook pose in the robot base frame.

Current status: Only the Left camera configuration and Left canonical reference have been prepared/validated.

Repository Structure

workspace/
└── src/
    ├── Vision_/
    │   ├── camera/
    │   │   └── camera_setting_left.yml
    │   ├── models/
    │   │   ├── hook_model.ply
    │   │   └── hook_canonical_reference/
    │   │       └── left/
    │   │           ├── canonical_observed_stable.ply
    │   │           └── canonical_observed_core.ply
    │   └── segmentation/
    │       └── weights/
    │           └── rf_detr_best.pth
    │
    └── hook_vision/
        ├── hook_vision/
        │   ├── __init__.py
        │   └── hook_pose_estimator.py
        ├── resource/
        │   └── hook_vision
        ├── package.xml
        ├── setup.cfg
        └── setup.py

Required Environment

The following environment/packages are required.

ROS 2 (rclpy, colcon)

Zivid SDK and Zivid Python API

Python 3

PyTorch

RF-DETR

NumPy

SciPy

Open3D

OpenCV

Pillow

Python packages except the Zivid SDK/API can be installed in the active Python environment as needed, for example:

pip install numpy scipy open3d opencv-python pillow rfdetr

Install PyTorch separately according to the CUDA/CPU environment of the PC being used.The Zivid SDK and Zivid Python API must also be installed separately before running the node.

RF-DETR Weight File

src/Vision_/segmentation/weights/rf_detr_best.pth in this repository is an empty placeholder because the trained weight file is too large to upload to the repository.

Before running the code, replace it with the actual trained weight file using the same filename:

src/Vision_/segmentation/weights/rf_detr_best.pth

The program checks for the required files during startup, so execution will fail if the weight file or reference files are missing.

Build

The current code expects the repository/workspace path to be:

~/workspace/src/...

Therefore, place the repository so that Vision_ and hook_vision are located directly under ~/workspace/src/.

Then build the ROS 2 package:

cd ~/workspace
colcon build --packages-select hook_vision --symlink-install
source install/setup.bash

Run

Connect the Zivid camera and run:

ros2 run hook_vision hook_pose_estimator --ros-args -p hand_side:=Left

The current repository supports the Left configuration only.

The program performs one capture, estimates one 6D pose, saves the result/diagnostic images, and exits.

Output

Results are saved under:

~/workspace/src/Vision_/data/hook_pose_estimator_node/left/run_YYYYMMDD_HHMMSS/

The output includes the estimated hook pose and diagnostic images such as segmentation, CAD alignment, Canonical Reference alignment, and residual heatmaps.

Required Model / Reference Files

The following files are required for the current Left-camera version:

src/Vision_/camera/camera_setting_left.yml
src/Vision_/models/hook_model.ply
src/Vision_/models/hook_canonical_reference/left/canonical_observed_stable.ply
src/Vision_/models/hook_canonical_reference/left/canonical_observed_core.ply
src/Vision_/segmentation/weights/rf_detr_best.pth

Notes

Right-camera canonical reference files are not included yet, so Right pose estimation is not currently supported.

Camera-to-base calibration values are currently defined directly in hook_pose_estimator.py.

The Canonical Observed Reference was generated from previously captured hook point clouds and is used for final pose refinement and quality checking.

This repository estimates and saves the hook pose only. Verify the output and coordinate conventions before using the result for robot motion.
