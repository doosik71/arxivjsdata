# VIDEO-BASED SURGICAL TOOL-TIP AND KEYPOINT TRACKING USING MULTI-FRAME CONTEXT-DRIVEN DEEP LEARNING MODELS

Bhargav Ghanekar, Lianne R. Johnson, Jacob L. Laughlin, Marcia K. O’Malley, Ashok Veeraraghavan (2025)

## 🧩 Problem to Solve

본 연구가 해결하고자 하는 문제는 로봇 수술 영상 내에서 수술 도구의 주요 지점(Keypoints), 특히 Tool-tip과 같은 특정 지점을 정밀하게 지역화(Localization)하고 추적(Tracking)하는 것이다.

수술 도구의 Keypoint 추적은 외과 의사의 숙련도 평가(Skill assessment), 전문성 분석, 그리고 안전 구역(Safety zones) 설정과 같은 후속 분석 작업에 필수적인 기초 단계이다. 그러나 수술 영상은 조명 조건의 변화, 일시적인 가려짐(Occlusion), 초점 문제, 모션 블러(Motion blur) 및 도구의 복잡한 배치와 방향성으로 인해 Keypoint를 정확하게 추적하는 것이 매우 어렵다. 기존 연구들은 주로 도구 전체의 영역을 분할하는 Segmentation에 집중해 왔으며, 특정 Keypoint 추적에 관한 연구는 상대적으로 부족한 실정이다.

따라서 본 논문의 목표는 다중 프레임 문맥(Multi-frame context)을 활용하여 수술 도구의 Keypoint를 강건하게 추적할 수 있는 딥러닝 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Keypoint 추적 문제를 직접적인 좌표 예측이 아닌, Keypoint 주변의 작은 영역을 분할하는 Segmentation 문제로 정의하고, 여기에 시간적·기하학적 문맥을 추가하여 정밀도를 높이는 것이다.

가장 중심적인 설계 아이디어는 단일 프레임 기반의 분석(Single-frame context)을 넘어, 연속된 $K$개의 프레임 정보와 함께 Optical Flow 및 Monocular Depth 예측 맵을 보조 입력으로 사용하는 Multi-frame Context (MFC) 모델을 도입한 점이다. 이를 통해 도구의 급격한 움직임이나 모호한 포즈 상황에서도 보다 정확한 Keypoint 지역화가 가능하도록 설계하였다.

## 📎 Related Works

최근 수술 도구 검출 및 분할 분야에서는 감독 학습 기반의 딥러닝 솔루션들이 많이 제안되었다. 하지만 대부분의 현대적인 Pose Estimation 연구들은 인간의 포즈 추정에 집중되어 있으며, 이는 방대한 양의 학습 데이터를 필요로 한다. 반면, 수술 도구의 Keypoint 추적은 공개된 주석(Annotation) 데이터셋이 매우 부족하여 딥러닝 모델을 적용하기에 어려움이 있다.

기존의 일부 연구들은 전통적인 머신러닝 방법이나 단순한 CNN 기반의 추적 방식을 사용하였으나, 앞서 언급한 가려짐, 모션 블러, 비디오 압축 아티팩트 등의 문제로 인해 성능 향상에 한계가 있었다. 본 연구는 이러한 한계를 극복하기 위해 소규모 데이터셋만으로도 효율적으로 작동하며, 다중 프레임 문맥을 활용해 강건성을 확보한 접근 방식을 취한다.

## 🛠️ Methodology

### 전체 파이프라인 및 시스템 구조

본 연구의 추적 프레임워크는 다음의 2단계 과정으로 구성된다.

1. **Keypoint Region Segmentation**: 각 Keypoint 주변의 작은 영역(Keypoint ROI)을 픽셀 단위의 다중 클래스 분할 문제로 모델링하여 분할 맵을 생성한다.
2. **Keypoint Localization**: 생성된 분할 맵에서 가장 큰 연속된 영역(Blob)을 찾고, 해당 영역의 중심점(Centroid)을 최종 Keypoint 좌표로 결정한다.

### 주요 구성 요소 및 모델 아키텍처

제안된 모델은 문맥 활용 범위에 따라 두 가지 카테고리로 나뉜다.

- **SFC (Single-frame Context) 모델**: 특정 시점의 단일 프레임만을 입력으로 하여 분할 결과물을 출력하는 기본 모델이다. DeepLab-v3, HRNet, SegFormer 등이 이에 해당한다.
- **MFC (Multi-frame Context) 모델**: $K$개의 연속된 프레임을 입력으로 하며, 다음과 같은 보조 데이터를 활용한다.
  - **SFC Outputs**: $K$개 프레임 각각에 대해 SFC 모델이 예측한 분할 맵.
  - **Optical Flow Maps**: RAFT 모델을 통해 계산된, 현재 프레임과 이전 $K-1$개 프레임 사이의 픽셀 변위 정보.
  - **Depth Maps**: Depth-Anything-V2 모델을 통해 예측된 각 프레임의 픽셀별 깊이 정보.

이 데이터들은 **MFCNet**이라는 4층 구조의 CNN으로 입력되며, 두 가지 아키텍처 변형이 존재한다.

- **MFCNet-Basic (MFCNet-B)**: 깊이 맵, Optical Flow 맵, SFC 출력을 단순히 연결(Concatenate)하여 처리한다.
- **MFCNet-Warp (MFCNet-W)**: Optical Flow 맵을 이용하여 깊이 맵과 SFC 출력물들을 현재 프레임 기준으로 워핑(Warping)한 후 처리함으로써 시간적 정렬을 최적화한다.

### 훈련 목표 및 손실 함수

모델은 픽셀 단위의 다중 클래스 분할을 수행하며, 손실 함수는 다음과 같이 정의된다.

$$\text{Loss} = 0.7H - 0.3 \log J$$

여기서 $H$는 픽셀별 Negative Log-Likelihood (NLL) 손실이며, $\log J$는 분할 클래스들에 대한 Jaccard Index(IoU)의 로그 값이다. 클래스 불균형 문제를 해결하기 위해 NLL 손실 계산 시 배경(Background) 클래스의 가중치를 $1/100$로 낮추어 적용하였다.

### 학습 절차

1. SFC 모델을 Adam 옵티마이저(학습률 $3 \times 10^{-5}$)로 20 epoch 동안 먼저 학습시킨다.
2. 사전 학습된 SFC 모델을 고정하거나 낮은 학습률($10^{-6}$)로 미세 조정(Fine-tuning)하면서, MFCNet의 가중치를 학습률 $10^{-4}$로 20 epoch 동안 학습시킨다.
3. Optical Flow 계산 시 연산 부하를 줄이기 위해 $2\times$ 다운샘플링된 스케일을 사용하며, 10 epoch 이후 학습률을 $\gamma=0.1$ 비율로 감쇠시킨다.

## 📊 Results

### 실험 설정

- **데이터셋**: EndoVis'15 데이터셋(1850 프레임)과 자체 주석을 추가한 JIGSAWS 데이터셋(학습 1350 프레임, 테스트 450 프레임)을 사용하였다.
- **비교 대상**: 기존 연구(Du et al. 2018), 단일 프레임 모델(TernausNet, FCN, DeepLab-v3, HRNet, SegFormer).
- **측정 지표**: Localization RMSE (Root Mean Square Error, 픽셀 단위), Precision, Recall, Detection Accuracy.

### 주요 결과

- **EndoVis'15 데이터셋**:
  - 제안된 MFC 모델이 기존 Baseline 및 SFC 모델들보다 낮은 RMSE를 기록하며 성능 우위를 보였다.
  - 특히 DeepLab-v3를 기반으로 한 MFCNet-W 모델이 가장 우수한 성능을 보였으며, Localization RMS error는 $5.27$ pixels (Abstract 기준) 수준을 달성하였다.
  - Ablation Study 결과, $K=3$일 때 최적의 성능을 보였으며, 깊이(Depth) 정보가 없을 경우 성능이 저하됨을 확인하여 보조 입력의 중요성을 입증하였다.

- **JIGSAWS 데이터셋**:
  - MFC 모델들이 91% 이상의 Keypoint 검출 정확도(Detection Accuracy)를 기록하였다.
  - Localization RMSE는 $4.2$ pixels 미만으로 측정되어, 저해상도 영상과 까다로운 도구 포즈 상황에서도 강건한 추적 능력을 보여주었다.
  - 추론 속도는 Nvidia A100 GPU 기준 $\ge 3.4$ FPS로, 오프라인 분석에 적합한 속도를 확보하였다.

## 🧠 Insights & Discussion

본 연구는 단순한 이미지 분할을 넘어 시간적 문맥(Optical Flow)과 기하학적 정보(Depth)를 융합함으로써 수술 도구 추적의 정밀도를 높일 수 있음을 증명하였다. 특히 MFCNet-Warp 구조가 시간적 정렬을 통해 단일 프레임 모델의 한계를 극복하고 모션 블러와 같은 노이즈에 강건하게 대응한 점이 고무적이다.

다만, 몇 가지 한계점이 존재한다. 첫째, 데이터셋의 규모가 상대적으로 작아 모델의 일반화 성능에 제약이 있을 수 있다. 둘째, 추론 속도가 3.4 FPS 수준으로, 실시간(Real-time) 피드백 시스템에 직접 적용하기에는 속도 개선이 필요하다. 셋째, Depth-Anything-V2와 같은 외부 사전 학습 모델에 의존하고 있어, 해당 모델의 오차가 최종 결과에 영향을 줄 수 있다는 가정이 깔려 있다.

그럼에도 불구하고, 본 프레임워크는 모듈형 구조로 설계되어 향후 더 발전된 Segmentation 모델이나 Depth 추정 모델로 교체 가능한 확장성을 가지고 있다는 점에서 학술적 가치가 높다.

## 📌 TL;DR

본 논문은 수술 도구의 Keypoint 추적을 위해 **SFC(단일 프레임) 분할 맵, Optical Flow, Depth Map을 통합하여 처리하는 MFC(다중 프레임 문맥) 딥러닝 프레임워크**를 제안하였다. 이 방식은 특히 도구의 급격한 움직임이나 가려짐 상황에서 강건하며, EndoVis'15 및 JIGSAWS 데이터셋에서 기존 방식보다 낮은 RMSE와 높은 검출 정확도를 달성하였다. 이 연구는 향후 자동화된 수술 숙련도 평가 및 실시간 수술 보조 시스템 구축을 위한 핵심 기술로 활용될 가능성이 높다.
