# SurgPose: a Dataset for Articulated Robotic Surgical Tool Pose Estimation and Tracking

Zijian Wu, Adam Schmidt, Randy Moore, Haoying Zhou, Alexandre Banks, Peter Kazanzides, and Septimiu E. Salcudean (2025)

## 🧩 Problem to Solve

본 연구가 해결하고자 하는 핵심 문제는 수술용 로봇 도구의 정밀한 Pose Estimation 및 Tracking을 위한 고품질의 공개 데이터셋이 부족하다는 점이다.

로봇 보조 수술(Robot-assisted surgery, RAS)에서 도구의 포즈 정보는 증강 현실(AR), 다중 모달 데이터 등록, 시각적 서보 제어(Visual servoing) 및 학습 기반의 자율 조작 등 다양한 하위 애플리케이션에 필수적이다. 기존의 da Vinci 시스템과 같은 로봇의 관절 상태(Joint states)와 Kinematics 모델을 통해 포즈를 측정할 수 있으나, 케이블의 유연성, 도구의 굴곡, 그리고 백래시(Backlash)로 인해 실제 위치와 상당한 오차가 발생한다. 또한, 카메라 뷰와 로봇 좌표계 사이의 Hand-eye calibration 과정이 매우 까다롭고 비용이 많이 든다.

따라서 본 논문의 목표는 Kinematics에 의존하지 않고 카메라 영상만을 통해 도구의 포즈를 직접 계산할 수 있도록, 정밀하게 라벨링된 대규모 데이터셋인 SurgPose를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 가장 중심적인 아이디어는 **자외선(UV) 반응성 페인트(UV-reactive paint)**를 사용하여 투명하면서도 정확한 Keypoint 마커를 생성하는 것이다.

이 페인트는 일반적인 가시광선(White light) 아래에서는 투명하여 도구의 외형을 거의 변화시키지 않지만, UV 라이트(Black light) 아래에서는 형광색으로 밝게 빛나는 특성을 가진다. 연구진은 동일한 궤적을 가시광선 환경과 UV 라이트 환경에서 각각 두 번 수행함으로써, 실제 수술 환경과 유사한 'Raw 영상'과 정밀한 ground truth를 얻을 수 있는 '마커 영상'을 동시에 수집하였다.

이를 통해 수동 라벨링의 고충과 오차를 줄이면서도, 모델 학습 시에는 실제 외형과 차이가 없는 깨끗한 이미지를 사용할 수 있게 되었다.

## 📎 Related Works

### 기존 연구 및 한계

1. **Pose Estimation 방법론**: 템플릿 매칭(Template matching) 방식은 저수준 시각 특징에 의존하여 시맨틱 정보가 부족하며, 영상의 손상이나 환경 변화에 취약하다. 최근의 딥러닝 기반 방법들은 더 나은 강건성을 보이지만, 학습을 위한 대규모 고품질 데이터셋이 필수적이다.
2. **기존 데이터셋**: RMIT, EndoVis15, SurgRIPE, PhaKIR 등의 데이터셋이 존재하지만, 대부분의 경우 라벨링된 프레임 수가 적거나, 도구의 종류가 한정적이며, Stereo 영상이나 Kinematics/Joint states 데이터가 함께 제공되지 않는 등 한계가 있다.

### SurgPose의 차별점

SurgPose는 다음과 같은 점에서 기존 데이터셋과 차별화된다:

- **투명 마커 사용**: 이미지 왜곡 없이 정밀한 Keypoint 위치를 제공한다.
- **다중 모달 어노테이션**: End-effector kinematics, PSM의 Joint states, 파트 수준의 Segmentation mask를 함께 제공한다.
- **Rectified Stereo**: 2D 포즈를 3D로 확장할 수 있도록 정제된 스테레오 쌍을 제공한다.
- **다양한 도구 구성**: 6종의 서로 다른 수술 도구를 포함하여 포괄적인 알고리즘 평가가 가능하다.

## 🛠️ Methodology

### 1. UV-반응성 페인트 라벨링 및 궤적 생성

- **라벨링**: 투명도가 높고 UV 하에서 형광 반응이 뚜렷한 적색 락커 기반 UV-반응성 페인트를 사용하였다.
- **궤적 생성**: dVRK(da Vinci Research Kit)를 사용하여 PSM(Patient Side Manipulator) 1과 3을 제어한다.
  - **위치 궤적**: 3D 작업 공간 내에서 랜덤한 지점들을 선택하고, 이를 주기적 보간 큐빅 스플라인(Periodic interpolating cubic spline)으로 연결하여 부드럽고 닫힌 경로를 생성한다.
  - **C-space 궤적**: 관절 4-7 및 그리퍼 각도에 대해 주기적인 사인파(Sinusoidal) 궤적을 생성한다.
- **수집 절차**: 각 궤적을 가시광선 환경에서 한 번, UV 라이트 환경에서 한 번 수행하여 쌍을 이룬 영상을 획득한다.

### 2. 데이터 처리 및 어노테이션

- **Keypoint 추출**: UV 라이트 하에서 촬영된 영상의 형광 마커를 SAMv2(Segment Anything 2) 모델을 이용해 세그멘테이션하고, 해당 중심점을 Keypoint로 정의한다. 이후 사람이 직접 검수하고 수정하는 과정을 거쳤다.
- **Stereo Matching**: OpenCV를 이용해 카메라를 캘리브레이션하고 이미지를 Rectification한다. RAFT 알고리즘을 사용하여 좌우 이미지 간의 변위(Disparity)를 계산하며, 이를 다음과 같은 수식을 통해 깊이(Depth)로 변환한다.

$$\text{Depth} = \frac{\text{focal length} \times \text{baseline}}{|\text{disparity} + (c_{x1} - c_{x0})|}$$

여기서 $c_{x1} - c_{x0}$는 $x$축을 따른 주점(Principal points)의 차이를 의미한다.

### 3. 마커 가시성 검증 (Quantifying Visibility)

마커가 가시광선 하에서 실제로 보이지 않는지 확인하기 위해 GLCM(Gray Level Co-occurrence Matrix) 특징을 사용하였다. 마커가 있는 패치와 없는 패치의 텍스처 패턴을 분석하여 분류기로 구분하려 시도했으나, 특징 공간에서 두 데이터가 서로 얽혀 있어 알고리즘적으로 구분할 수 없음을 확인하였다. 이는 마커가 실질적으로 투명함을 시사한다.

## 📊 Results

### 1. 데이터셋 통계

SurgPose는 총 6종의 도구(Large/Mega Needle Driver, Micro Forceps, Curved Scissor, DeBakey/Prograsp Forceps)로 구성되며, 약 120k개의 인스턴스(학습용 80k, 검증용 40k)를 포함한다. 각 도구는 7개의 시맨틱 Keypoint로 라벨링되어 있으며, 해상도는 $986 \times 1400$이다.

### 2. 베이스라인 성능 평가

YOLOv8-pose-x, ViTPose, DeepLabCut 세 가지 모델을 사용하여 성능을 측정하였다. 평가지표로는 COCO 데이터셋의 OKS(Object Keypoint Similarity) 기반 mAP를 사용하였다.

| Method | mAP | Inference Latency (ms) |
| :--- | :---: | :---: |
| YOLOv8 | $63.3 \pm 1.2$ | 16.4 |
| ViTPose | $53.6 \pm 1.0$ | 67.7 |
| DeepLabCut | $66.2 \pm 2.0$ | 194.8 |

- **분석**: DeepLabCut이 가장 높은 정확도를 보였으나 추론 속도가 매우 느리다. 반면 YOLOv8과 ViTPose는 실시간 추론이 가능한 수준의 속도를 보여주었다.

## 🧠 Insights & Discussion

### 강점

본 연구는 UV-반응성 페인트라는 창의적인 방법을 통해, 데이터 수집 과정에서의 라벨링 비용을 획기적으로 줄이면서도 실제 수술 영상과 동일한 특성을 가진 고품질의 데이터셋을 구축하였다. 특히 스테레오 영상과 Kinematics 데이터를 동시에 제공함으로써 3D Pose Estimation 연구에 매우 유용한 벤치마크를 제시하였다.

### 한계 및 비판적 해석

논문에서 명시했듯이, 데이터셋의 규모는 커졌으나 배경(ex vivo 조직)의 다양성이 부족하여 모델이 과적합(Overfitting)되는 경향이 있다. 실제로 다른 배경의 영상으로 테스트했을 때 성능이 크게 하락하는 문제가 관찰되었다. 이는 본 데이터셋만으로는 실제 임상 환경에 바로 적용 가능한 모델을 학습시키기에 부족함을 의미하며, 향후 in vivo(생체 내) 데이터나 더 다양한 환경에서의 데이터 확장이 필수적이다.

또한, 도구 간의 상호 가려짐(Occlusion) 현상이 포함되지 않은 궤적으로 수집되었다는 점이 실용적 관점에서의 한계로 지적될 수 있다.

## 📌 TL;DR

SurgPose는 UV-반응성 페인트를 이용해 가시광선에서는 보이지 않고 UV 라이트에서만 보이는 마커를 사용함으로써, 이미지 왜곡 없이 정밀한 Keypoint 라벨링을 구현한 수술용 로봇 도구 포즈 추정 데이터셋이다. 6종의 도구에 대해 120k개의 인스턴스와 스테레오 영상, Kinematics 정보를 제공하며, 향후 소량 학습(Few-shot learning)이나 파운데이션 모델의 미세 조정(Fine-tuning)을 위한 핵심 자산으로 활용될 가능성이 높다.
