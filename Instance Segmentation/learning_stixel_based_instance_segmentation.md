# Learning Stixel-based Instance Segmentation

Monty Santarossa, Lukas Schneider, Claudius Zelenka, Lars Schmarje, Reinhard Koch and Uwe Franke (2021)

## 🧩 Problem to Solve

본 논문은 자율 주행 시스템에서 주변 환경을 정확하게 인식하기 위해 필수적인 Instance Segmentation의 연산 비용 문제를 해결하고자 한다. 도시 환경에서의 Instance Segmentation은 가림 현상(occlusion)과 다양한 객체 크기로 인해 매우 도전적인 과제이다. 기존의 픽셀 기반 CNN 방식은 정확도는 높으나 연산량이 매우 많아, Cityscapes 데이터셋 기준으로 최신 기법들조차 5 FPS 미만의 낮은 처리 속도를 보인다.

이를 해결하기 위해 저자들은 이미지의 데이터를 획기적으로 줄인 medium-level 표현 방식인 Stixel World를 활용한다. Stixel은 이미지의 픽셀을 수백 개의 직사각형 스틱(sticks)으로 단순화하여 데이터 볼륨과 깊이 노이즈를 줄여준다. 하지만 Stixel은 2D 공간에서 희소(sparse)하고 비정형(unstructured)된 특성을 가지기 때문에, 일반적인 CNN 구조를 그대로 적용하기 어렵다는 한계가 있다. 따라서 본 연구의 목표는 Stixel의 효율성을 유지하면서도 딥러닝을 통해 실시간으로 동작하는 Instance Segmentation 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Stixel 표현형을 포인트 클라우드(Point Cloud)와 유사한 비정형 데이터로 간주**하는 것이다. Stixel이 3D 공간에서의 위치와 형태 정보를 가지고 있다는 점에 착안하여, 포인트 기반 딥러닝 아키텍처인 PointNet을 Stixel 도메인에 적용한 **StixelPointNet**을 제안한다.

주요 기여 사항은 다음과 같다:

- **Stixel-level Ground Truth 생성**: 현재 Stixel 기반의 데이터셋이 부족한 문제를 해결하기 위해, 픽셀 수준의 어노테이션으로부터 Stixel 수준의 ground truth를 자동으로 생성하는 방법을 제안한다.
- **StixelPointNet 파이프라인**: 3D 공간의 Stixel을 입력으로 하여 PointNet 모델을 통해 빠르게 인스턴스를 분할하는 새로운 프레임워크를 구축하였다.
- **성능 및 속도 검증**: 기존의 클러스터링 기반 베이스라인 및 픽셀 기반 최신 기법들과 비교하여, Stixel 수준에서 state-of-the-art 성능을 달성함과 동시에 실시간 처리가 가능한 수준의 속도를 확보하였음을 입증하였다.

## 📎 Related Works

### 1. Point Cloud Feature Learning

비정형 포인트 데이터 학습을 위해 과거에는 Voxel 기반의 양자화(quantization) 방식과 Volumetric CNN이 사용되었으나, 포인트 클라우드의 희소성으로 인해 효율성이 떨어지는 문제가 있었다. 이를 해결하기 위해 Qi et al.이 제안한 PointNet은 shared MLP와 symmetric function(max pooling)을 사용하여 데이터의 순서나 크기에 상관없이 직접 학습할 수 있게 하였다. 본 논문은 이러한 PointNet의 구조를 Stixel에 적용한다.

### 2. Instance Segmentation on Point Clouds

Mask R-CNN의 구조를 3D에 적용한 Region-based PointNet, RGBD 이미지의 2D 제안 영역을 3D 프러스텀으로 확장한 Frustum PointNet, 그리고 제안 영역 없이 직접 bounding box를 예측하는 3D-BoNet 등이 연구되었다. StixelPointNet은 제안 영역(proposal) 기반의 분할이라는 일반적인 개념을 따르지만, Stixel의 특수한 기하학적 구조를 처리하기 위한 최적화를 적용하였다.

### 3. Instance Segmented Stixels

최근 Hehn et al.은 CNN을 통해 픽셀 수준의 인스턴스 정보를 추출하여 Stixel에 부여하는 Instance Stixels를 제안하였다. 그러나 본 논문의 저자들은 Stixel 생성 알고리즘 자체가 이미 깊이 정보를 통해 객체 경계를 어느 정도 구분하므로, 픽셀 수준의 정보를 추가하는 것보다 Stixel 도메인에서 직접 추론하는 것이 더 효율적이라고 주장한다.

## 🛠️ Methodology

### 전체 파이프라인

StixelPointNet의 전체 프로세스는 **Filtering $\rightarrow$ PointNet Model $\rightarrow$ Best Prediction Selection (BPS)**의 세 단계로 구성된다.

### 1. Filtering (제안 영역 추출)

먼저 SSD(Single Shot Detection)를 사용하여 객체의 candidate bounding box를 생성한다. 하지만 SSD 박스는 픽셀 수준에서 매우 타이트하게 생성되므로, 대응하는 Stixel들을 모두 포함하지 못하는 경우가 많다. 이를 해결하기 위해 **RoI box**라는 개념을 도입한다:

- **Scale Factor ($sc_{RoI}$)**: SSD 박스를 일정 비율로 확장하여 더 넓은 영역을 커버한다.
- **Overlap Threshold ($t_{RoI}$)**: Stixel의 면적 중 일정 비율 이상이 확장된 박스 내에 존재하면 해당 Stixel을 캡처한다.

### 2. PointNet Model (이진 분할)

필터링된 Stixel 세트를 입력으로 받아 해당 RoI 박스 내의 Stixel이 실제 객체에 속하는지 여부를 판단하는 이진 분할(binary segmentation)을 수행한다.

- **입력 특징 벡터**: $Stx' = [x, y, z, w, h, u', v', h', l, l_{bb}]$
  - $(x, y, z, w, h)$: 3D 위치 및 크기
  - $(u', v', h')$: RoI 박스 내에서의 정규화된 2D 위치 및 높이
  - $l, l_{bb}$: Stixel의 semantic label 및 bounding box의 predicted class label
- **아키텍처**: STN(Spatial Transform Networks)을 제외한 PointNet 구조를 사용한다. Shared MLP를 통해 각 Stixel의 특징을 추출하고, Max Pooling을 통해 전역 특징(Global Feature)을 생성한 후, 이를 다시 각 Stixel 특징과 결합하여 최종 이진 분류를 수행한다.

### 3. Best Prediction Selection (BPS)

하나의 Stixel이 여러 RoI 박스에 중복되어 포함될 수 있으므로, 최종적으로 어떤 인스턴스 ID를 부여할지 결정해야 한다. 본 논문에서는 다음과 같은 가중합(weighted sum) 방식을 사용하여 가장 신뢰도가 높은 예측을 선택한다:
$$\text{Score} = 0.75 \cdot c_{bb} + 0.25 \cdot p_c$$
여기서 $c_{bb}$는 RoI 박스의 신뢰도(confidence)이며, $p_c$는 PointNet 모델이 예측한 Stixel의 소속 확률(prediction confidence)이다.

## 📊 Results

### 실험 설정

- **데이터셋**: Cityscapes 및 여기서 파생된 Stixel-level 데이터셋 $SD_{0.35}$를 사용하였다. $SD_{0.35}$는 픽셀 수준의 ground truth와 Stixel 간의 중첩 면적이 0.35 이상일 때 동일 인스턴스로 간주하여 생성되었다.
- **비교 대상 (Baselines)**:
  - **Statistical Approach**: 클래스별 객체 Stixel의 평균 비율 $p_c$를 계산하고, RoI 박스 중심에서 가장 가까운 $p_c$ 만큼의 Stixel을 객체로 지정하는 단순 방식이다.
  - **HAC (Hierarchical Agglomerative Clustering)**: 3D 좌표 기반의 계층적 군집화를 수행하는 방식이며, RoI 내에서 수행하는 $HAC_{RoI}$와 전체 이미지에서 수행하는 $HAC_{img}$ 두 가지를 구현하였다.

### 주요 결과

- **Stixel-level 성능**: $SD_{0.35}$ 데이터셋에서 StixelPointNet은 대부분의 클래스(자동차 등)에서 가장 높은 AP를 기록하였다. 특히 자동차 클래스에서는 62% 이상의 AP를 달성하였다.
- **Pixel-level 성능**: 픽셀 수준의 AP 측정 시, 제안 방법이 기존의 Instance Stixels [13]보다 약 3% 높은 $AP_{50\%}$를 기록하였다. 이는 Stixel 도메인에서 직접 학습하는 것이 더 효과적임을 시사한다.
- **추론 속도**: NVIDIA GTX 1070 환경에서 StixelPointNet은 약 **35.2 FPS**의 속도를 기록하였다. 이는 픽셀 기반의 최신 기법인 UPS-Net(4.4 FPS)이나 Deep Snake(4.7 FPS)보다 **약 7.5배 빠른 수치**이다.

## 🧠 Insights & Discussion

### 강점 및 의의

본 논문은 Stixel이라는 매우 압축된 표현형을 포인트 클라우드로 해석함으로써, 연산 효율성과 인식 정확도라는 두 마리 토끼를 잡았다. 특히, 픽셀 수준의 복잡한 연산 없이도 3D 공간에서의 추론만으로 충분히 높은 수준의 Instance Segmentation이 가능하다는 것을 입증하였다.

### 한계 및 비판적 해석

- **데이터셋 의존성**: Stixel-level 데이터셋을 픽셀 수준에서 생성하여 사용했으므로, 생성 과정에서의 오차가 존재할 수 있다. 저자들 또한 Stixel 표현의 거친(rough) 특성 때문에 픽셀 수준의 AP가 낮게 측정되는 경향이 있음을 언급하였다.
- **객체 탐지기 성능**: 파이프라인의 첫 단계인 SSD의 성능이 전체 결과에 큰 영향을 미친다. SSD가 객체를 찾지 못하면 이후의 PointNet 모델이 아무리 뛰어나도 인스턴스를 분할할 수 없다.
- **가림 현상 처리**: 결과 분석에서 가림 현상이 심한 경우 하나의 객체가 여러 개로 쪼개지는 현상이 관찰되었다. 이는 PointNet의 국소적 특징 추출 능력의 한계로 보이며, 향후 PointNet++와 같은 계층적 구조 도입이 필요할 것으로 판단된다.

## 📌 TL;DR

본 논문은 Stixel 데이터를 포인트 클라우드로 간주하여 실시간 Instance Segmentation을 수행하는 **StixelPointNet**을 제안한다. SSD로 제안 영역을 추출하고 PointNet으로 이진 분할을 수행하는 파이프라인을 통해, 기존 픽셀 기반 방식보다 **약 7.5배 빠른 속도(35.2 FPS)**와 우수한 분할 성능을 달성하였다. 이 연구는 Stixel 도메인이 3D 딥러닝 작업에 활용될 수 있음을 보여주었으며, 향후 자율 주행 시스템의 경량화된 환경 인지 모듈로 적용될 가능성이 높다.
