# Hook 6D Pose Estimation

Zivid 3D Camera와 RF-DETR Segmentation을 이용하여 Hook의 6D Pose를 추정하는 ROS 2 패키지입니다.

Segmentation된 Hook Point Cloud와 CAD Model을 이용해 초기 자세 후보를 생성한 뒤, CAD 및 Canonical Observed Reference를 이용한 다단계 ICP 정합을 통해 최종 6D Pose를 계산합니다.

현재 버전은 **Left Camera만 지원합니다.**

---

## 1. Repository Structure

Repository는 아래와 같은 구조로 되어 있습니다.

```text
workspace/
└── src/
    ├── Vision_/
    │   ├── camera/
    │   │   ├── camera_setting_left.yml
    │   │   └── camera_setting_right.yml
    │   │
    │   ├── models/
    │   │   ├── hook_model.ply
    │   │   └── hook_canonical_reference/
    │   │       ├── left/
    │   │       │   ├── canonical_observed_stable.ply
    │   │       │   └── canonical_observed_core.ply
    │   │       └── right/
    │   │
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
```

코드 내부에서 `~/workspace/src/Vision_` 경로를 사용하기 때문에 Repository의 `src` 디렉토리가 아래 위치에 오도록 설치해야 합니다.

```text
~/workspace/src/
```

---

## 2. Required Environment

다음 환경 및 Python Package가 필요합니다.

### ROS 2

- ROS 2
- rclpy
- colcon

### Zivid

- Zivid SDK
- Zivid Python API

Zivid Camera가 PC에 연결되어 있어야 하며, Zivid SDK에서 Camera가 정상적으로 인식되는 상태여야 합니다.

### Python Packages

다음 Python Package를 사용합니다.

```text
numpy
scipy
open3d
opencv-python
Pillow
torch
rfdetr
zivid
```

예를 들어 일반 Python Package는 다음과 같이 설치할 수 있습니다.

```bash
pip install numpy scipy open3d opencv-python Pillow rfdetr
```

PyTorch는 사용하는 CUDA 및 GPU 환경에 맞는 버전으로 별도 설치하는 것을 권장합니다.

Zivid Python API는 사용 중인 Zivid SDK 환경에 맞게 설치해야 합니다.

---

## 3. RF-DETR Weight

다음 경로에 RF-DETR Segmentation Weight가 필요합니다.

```text
~/workspace/src/Vision_/segmentation/weights/rf_detr_best.pth
```

GitHub 파일 용량 제한으로 인해 Repository에 포함된

```text
rf_detr_best.pth
```

파일은 실제 Weight가 없는 빈 Placeholder 파일입니다.

코드를 실행하기 전에 학습된 실제 `rf_detr_best.pth` 파일을 별도로 다운로드하여 해당 파일을 교체해야 합니다.

최종 경로는 반드시 아래와 같아야 합니다.

```text
~/workspace/src/Vision_/segmentation/weights/rf_detr_best.pth
```

실제 Weight가 없는 상태에서는 RF-DETR 모델을 불러오는 과정에서 Error가 발생합니다.

---

## 4. Supported Camera

현재 Canonical Observed Reference는 Left Camera 데이터만 생성 및 검증되어 있습니다.

따라서 현재 버전에서는 다음 설정만 사용합니다.

```text
hand_side = Left
```

Left Camera에서 사용하는 파일은 다음과 같습니다.

```text
Vision_/camera/camera_setting_left.yml

Vision_/models/hook_canonical_reference/left/
├── canonical_observed_stable.ply
└── canonical_observed_core.ply
```

Right Camera용 Camera Setting은 Repository에 포함되어 있지만, Right Camera용 Canonical Reference는 아직 생성 및 검증되지 않았습니다.

따라서 현재는 Right Camera의 6D Pose Estimation 결과를 보장하지 않습니다.

---

## 5. Required Model Files

Pose Estimation 실행을 위해 다음 파일이 필요합니다.

### Hook CAD Model

```text
~/workspace/src/Vision_/models/hook_model.ply
```

### Canonical Observed Reference

```text
~/workspace/src/Vision_/models/hook_canonical_reference/left/canonical_observed_stable.ply
```

```text
~/workspace/src/Vision_/models/hook_canonical_reference/left/canonical_observed_core.ply
```

### Zivid Camera Setting

```text
~/workspace/src/Vision_/camera/camera_setting_left.yml
```

### RF-DETR Weight

```text
~/workspace/src/Vision_/segmentation/weights/rf_detr_best.pth
```

---

## 6. Download

코드 내부에서 `~/workspace/src` 경로를 사용하므로 Repository의 `src`가 `~/workspace/src`에 위치하도록 설치합니다.

새로운 Workspace를 만드는 경우:

```bash
cd ~
mkdir workspace
cd workspace
```

그 후 Repository 내용을 현재 `workspace` 디렉토리에 다운로드합니다.

Git을 사용하는 경우:

```bash
git clone <repository-url> .
```

다운로드 후 다음 구조인지 확인합니다.

```bash
ls ~/workspace/src
```

아래 두 디렉토리가 보여야 합니다.

```text
Vision_
hook_vision
```

이미 `~/workspace`를 사용하고 있는 경우에는 Repository의 `src/Vision_` 및 `src/hook_vision` 디렉토리가 각각 다음 위치에 오도록 복사하면 됩니다.

```text
~/workspace/src/Vision_
~/workspace/src/hook_vision
```

---

## 7. Build

먼저 ROS 2 환경을 source 합니다.

사용 중인 ROS 2 Distribution에 맞는 setup 파일을 사용합니다.

예:

```bash
source /opt/ros/<ros-distro>/setup.bash
```

Workspace로 이동합니다.

```bash
cd ~/workspace
```

`hook_vision` Package를 Build 합니다.

```bash
colcon build --packages-select hook_vision --symlink-install
```

Build가 완료되면 Workspace를 source 합니다.

```bash
source ~/workspace/install/setup.bash
```

새 Terminal을 열었을 경우 다시 source 해야 합니다.

```bash
source /opt/ros/<ros-distro>/setup.bash
source ~/workspace/install/setup.bash
```

---

## 8. Run

Left Camera를 사용하여 Hook 6D Pose Estimation을 실행합니다.

```bash
ros2 run hook_vision hook_pose_estimator --ros-args -p hand_side:=Left
```

현재 기본 Parameter가 `Left`이므로 다음과 같이 실행할 수도 있습니다.

```bash
ros2 run hook_vision hook_pose_estimator
```

프로그램을 한 번 실행하면 다음 과정이 수행됩니다.

```text
Zivid Capture
    ↓
RF-DETR Hook Segmentation
    ↓
Mask 영역의 3D Point Cloud 추출
    ↓
PCA / FPFH-RANSAC 기반 초기 자세 후보 생성
    ↓
CAD Coarse ICP
    ↓
CAD Trimmed / Local ICP
    ↓
Canonical Reference Trimmed / Local ICP
    ↓
정합 품질 및 Ambiguity 검사
    ↓
Hook 6D Pose 출력
```

현재 프로그램은 One-shot 방식으로 동작하며 한 번의 Zivid Capture 및 Pose Estimation을 수행한 뒤 종료됩니다.

---

## 9. Output

Pose Estimation 결과 및 정합 확인용 이미지는 다음 위치에 저장됩니다.

```text
~/workspace/src/Vision_/data/hook_pose_estimator_node/left/run_YYYYMMDD_HHMMSS/
```

주요 Output은 다음과 같습니다.

```text
01_rgb.png
02_mask.png
03_segmentation_result.png
04_cad_alignment.png
05_canonical_alignment.png
06_canonical_residual_heatmap.png
07_cad_residual_heatmap.png
pose_result.txt
```

`pose_result.txt`에는 Base Frame 기준 Hook 6D Pose가 저장됩니다.

```text
x [mm]
y [mm]
z [mm]

Euler x [deg]
Euler y [deg]
Euler z [deg]
```

---

## 10. Pose Estimation Method

현재 알고리즘은 다음 순서로 Hook Pose를 추정합니다.

### 1. Global Pose Candidate Generation

Segmentation된 Hook Point Cloud와 Hook CAD를 이용하여 초기 자세 후보를 생성합니다.

사용 방법:

```text
PCA
FPFH-RANSAC
```

PCA에서는 Point Cloud와 CAD의 주축 방향을 이용하여 여러 초기 자세 후보를 생성합니다.

FPFH-RANSAC에서는 Point Cloud와 CAD의 Local 3D Feature를 이용하여 추가적인 전역 자세 후보를 생성합니다.

### 2. CAD Registration

생성된 각 후보에 대해 Hook CAD와 Coarse ICP를 수행합니다.

정합 결과가 좋은 후보를 선정한 뒤 다음 정밀 정합을 수행합니다.

```text
CAD Trimmed ICP
CAD Local ICP
```

### 3. Canonical Reference Registration

CAD 정합 결과를 초기값으로 하여 실제 Zivid 관측 데이터를 기반으로 생성한 Canonical Observed Reference에 다시 정합합니다.

```text
Canonical Trimmed ICP
Canonical Local ICP
```

### 4. Quality Check

최종 후보는 Canonical Reference 및 CAD와의 거리 분포를 이용하여 품질을 검사합니다.

주요 평가 지표:

```text
Median distance
P90 distance
P95 distance
IR@2
CAD P95
```

서로 다른 Pose 후보가 비슷한 정합 품질을 가지는 경우 Ambiguous result로 판단하여 Pose를 출력하지 않습니다.

---

## 11. Notes

- 현재 Canonical Reference는 Left Camera 기준으로만 생성 및 검증되었습니다.
- `rf_detr_best.pth`는 Repository에 실제 Weight가 포함되어 있지 않으므로 실행 전에 반드시 교체해야 합니다.
- Hook CAD 및 Canonical Reference 파일의 위치를 변경할 경우 코드의 경로도 함께 수정해야 합니다.
- Camera-to-Base Calibration 값은 현재 코드 내부에 정의되어 있습니다.
- 다른 Camera 또는 Robot 환경에서 사용할 경우 Camera Calibration 값을 다시 설정해야 합니다.
- Zivid Camera가 정상적으로 연결되지 않은 경우 Capture 단계에서 실행이 실패합니다.
