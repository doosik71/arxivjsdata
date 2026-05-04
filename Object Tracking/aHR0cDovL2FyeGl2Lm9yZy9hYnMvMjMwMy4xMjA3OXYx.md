# OmniTracker: Unifying Object Tracking by Tracking-with-Detection

Junke Wang, Dongdong Chen, Zuxuan Wu, Chong Luo, Xiyang Dai, Lu Yuan, Yu-Gang Jiang (2023)

## 🧩 Problem to Solve

객체 추적(Object Tracking, OT)은 비디오 시퀀스 내에서 대상 객체의 위치를 추정하는 것을 목표로 한다. 현재 이 분야는 초기 상태의 지정 방식에 따라 크게 두 가지 범주로 나뉜다. 첫째는 첫 번째 프레임에서 주석(annotation)으로 대상이 지정되는 Instance Tracking(예: SOT, VOS)이며, 둘째는 특정 카테고리의 모든 객체를 탐지하고 추적하는 Category Tracking(예: MOT, MOTS, VIS)이다.

이처럼 서로 다른 설정으로 인해 각 작업마다 맞춤형 아키텍처와 하이퍼파라미터가 요구되었으며, 이는 모델 학습의 복잡성을 증가시키고 파라미터의 중복을 초래하였다. 본 논문의 목표는 이러한 다양한 추적 작업들을 하나의 완전하게 공유된 네트워크 아키텍처, 모델 가중치, 그리고 추론 파이프라인으로 해결할 수 있는 통합 모델인 OmniTracker를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 'Tracking-with-Detection'이라는 새로운 패러다임을 제시한 것이다. 기존의 방식들은 추적기가 탐지기의 탐색 영역을 지정하는 'Tracking-as-Detection' 방식이나, 탐지기가 먼저 박스를 찾고 추적기가 이를 연결하는 'Tracking-by-Detection' 방식으로 양분되어 있었다.

OmniTracker는 이 두 방식의 장점을 결합하여, 추적(Tracking)이 탐지(Detection)에 외형 정보(Appearance Priors)를 보완하고, 탐지가 추적에 연관성을 위한 후보 바운딩 박스(Candidate Bounding Boxes)를 제공하도록 설계되었다. 이를 통해 인스턴스 추적과 카테고리 추적이라는 서로 다른 성격의 작업을 하나의 통합된 프레임워크 내에서 수행할 수 있게 되었다.

## 📎 Related Works

기존의 추적 모델들은 특정 작업에 최적화된 Task-specific 모델들이 주를 이루었다. SOT는 Siamese 네트워크 기반의 매칭을, VOS는 공간-시간 메모리 네트워크(Spatio-temporal memory networks)를 통한 정밀한 매칭을 사용하였다. 반면 MOT와 같은 카테고리 추적은 탐지 후 연관시키는 Tracking-by-detection 방식이 주류를 이루었다.

최근에는 여러 작업을 통합하려는 시도가 있었으나 한계가 명확하였다. UniTrack은 공유 외형 모델을 사용했지만 헤드 구조의 차이로 인해 대규모 데이터셋 학습에 제약이 있었고, Unicorn은 타겟 프라이어(Target Prior)를 통해 통합을 시도했으나 작업별로 추론 파이프라인이 상이하고 박스 작업과 마스크 작업을 분리하여 학습시켜야 하는 문제가 있었다. OmniTracker는 모든 작업에 대해 동일한 가중치와 추론 파이프라인을 적용하며 공동 학습(Joint Training)을 수행한다는 점에서 차별점을 갖는다.

## 🛠️ Methodology

### 전체 파이프라인 및 시스템 구조

OmniTracker는 기본적으로 Deformable DETR을 기반으로 하며, 여기에 Reference-guided Feature Enhancement (RFE) 모듈을 추가하여 이전 프레임의 정보를 현재 프레임의 특징 맵에 반영한다. 전체 구조는 특징 추출을 위한 Backbone, 외형 정보를 보완하는 RFE 모듈, 그리고 최종 박스와 마스크를 예측하는 Deformable DETR 탐지기로 구성된다.

### 주요 구성 요소 및 역할

1. **RFE Module**: 이전 프레임($X_{t-1}$)의 정보를 활용해 현재 프레임($X_t$)의 특징을 강화한다.
    * **Instance Tracking**: 이전 프레임에서 추적된 박스들의 RoIAlign 특징을 사용한다.
    * **Category Tracking**: 이전 프레임 특징 맵을 다운샘플링하여 시간적 문맥 정보를 제공한다.
    * **작동 원리**: 현재 프레임 특징 $f_t$를 쿼리(Query)로, 이전 프레임 정보 $h_{t-1}$을 키(Key)와 값(Value)으로 사용하는 Cross-Attention을 통해 상관관계를 모델링하고, MLP를 거쳐 강화된 특징 $\tilde{f}_t$를 생성한다.
    $$\text{Equation (1): } h^i_{t-1} = \begin{cases} \text{RoIAlign}(f^i_{t-1}, \hat{b}_{t-1}), & \text{instance tracking} \\ \text{DownSample}(f^i_{t-1}), & \text{category tracking} \end{cases}$$
    $$\text{Equation (2): } g^i_t = \text{CrossAttn}(f^i_t, h^i_{t-1})$$

2. **Deformable DETR Detector**: 강화된 특징 $\tilde{F}$를 입력받아 Transformer Encoder와 Decoder를 통해 객체 쿼리를 업데이트하고, 이를 통해 바운딩 박스와 클래스를 예측한다. 마스크 생성의 경우 FPN 구조로 고해상도 특징 맵을 생성하고, 쿼리를 통해 생성된 커널 가중치 $\omega$를 사용하여 조건부 컨볼루션(CondConv)으로 예측한다.

### 훈련 목표 및 손실 함수

모델은 탐지 손실($L_{det}$)과 대조 학습 기반의 ReID 손실($L_{reid}$)의 합으로 학습된다.

* **Detection Loss**: 분류 손실($L_{cls}$), 박스 손실($L_{box}$, $L1 + \text{GIoU}$), 마스크 손실($L_{mask}$, $\text{Dice} + \text{Focal}$)의 가중 합으로 정의된다.
    $$L_{det} = L_{cls} + \lambda_1 L_{box} + \lambda_2 L_{mask}$$
* **Contrastive ReID Loss**: 객체 쿼리와 RoIAlign 특징을 결합하여 아이덴티티 임베딩 $e^t_k$를 생성하고, 이를 통해 서로 다른 프레임 간의 동일 객체를 구별하도록 학습한다.
    $$L_{reid} = \log \left[ 1 + \frac{\sum e^{-ref}}{\sum e^{+ref}} \exp(e_t \cdot e^{-ref} - e_t \cdot e^{+ref}) \right]$$

### 추론 절차 및 알고리즘 흐름

추론 시에는 모든 작업에 대해 동일한 파이프라인을 사용하며, 각 궤적(Trajectory)별로 메모리 뱅크를 유지하여 과거의 아이덴티티 임베딩을 저장한다.

1. **유사도 계산**: 현재 탐지된 객체의 임베딩 $e^t_n$과 메모리 뱅크의 가중 합 임베딩 $\tilde{e}_m$ 간의 양방향 유사도 $s_{n,m}$을 계산한다.
2. **모션 필터링**: Kalman Filter를 사용하여 위치를 예측하고, 예측 위치와 탐지 박스 간의 IoU가 임계값($\tau=0.25$)보다 낮은 경우를 필터링한다.
3. **최적 할당**: 계산된 유사도와 IoU 정보를 바탕으로 헝가리안 알고리즘(Hungarian Algorithm)을 통해 박스를 할당하며, 매칭되지 않은 고득점 박스는 새로운 궤적으로 초기화한다.

## 📊 Results

### 실험 설정

* **데이터셋**: LaSOT, TrackingNet(SOT), DAVIS16-17(VOS), MOT17(MOT), MOTS20(MOTS), YTVIS19(VIS) 등 총 7개의 벤치마크에서 실험을 수행하였다.
* **지표**: SOT에서는 Success rate와 Precision, VOS에서는 J&F score, MOT에서는 MOTA와 IDF1, VIS에서는 mAP 등을 사용하였다.

### 주요 결과

* **통합 성능**: OmniTracker-L(Large) 모델은 통합 모델 중 가장 뛰어난 성능을 보였으며, 특히 SOT와 MOT 작업에서 Unicorn을 상회하는 결과를 냈다.
* **작업별 성과**:
  * **SOT**: LaSOT와 TrackingNet에서 통합 모델 중 최상위 성능을 기록하며, 특정 작업 전용 모델(Task-specific)과 경쟁 가능한 수준에 도달했다.
  * **VOS**: Unicorn 대비 J&F 스코어가 DAVIS16에서 1.1%, DAVIS17에서 1.8% 향상되었으나, 전용 모델(예: ISVOS)과는 여전히 격차가 존재한다.
  * **MOT/MOTS**: MOT17에서 MOTA 79.1%, IDF1 75.6%를 기록하며 Unicorn보다 향상된 성능을 보였다. MOTS20의 sMOTSA에서도 PointTrackV2와 Unicorn을 앞질렀다.
  * **VIS**: 기존 통합 모델들은 VIS를 지원하지 않았으나, OmniTracker는 mAP 기준 laIDOL-L과 경쟁 가능한 수준(63.9%)의 성능을 보였다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 연구는 서로 다른 성격의 추적 작업들을 하나의 아키텍처와 가중치로 통합함으로써 모델의 범용성을 극대화하였다. 특히 '공동 학습(Joint Training)'의 효과가 뚜렷하게 나타났는데, 개별 학습(Separate Training)보다 공동 학습 시 모든 작업에서 더 높은 성능과 일반화 능력을 보였다. 또한, RFE 모듈이 탐지기의 정확도를 높이는 데 실질적인 기여를 함이 정량적/정성적 분석(특징 맵 시각화)을 통해 확인되었다.

### 한계 및 비판적 해석

VOS 작업에서 전용 모델과의 성능 격차가 여전히 존재하는 점은 주목할 만하다. 저자들은 그 원인을 OmniTracker가 콤팩트한 쿼리 기반 메모리를 사용하는 반면, VOS 전용 모델들은 고해상도 공간-시간 메모리를 사용하여 정밀한 픽셀 매칭을 수행하기 때문이라고 분석하였다. 이는 쿼리 기반의 압축된 표현 방식이 세밀한 마스크 예측에는 한계가 있을 수 있음을 시사한다.

또한, 추론 단계에서 Kalman Filter와 헝가리안 알고리즘이라는 전통적인 연관 방식을 사용하고 있는데, 이를 Transformer 기반의 end-to-end 연관 방식으로 완전히 대체할 수 있을지에 대한 논의가 추가되었다면 더욱 완성도 높은 연구가 되었을 것이다.

## 📌 TL;DR

OmniTracker는 인스턴스 추적(SOT, VOS)과 카테고리 추적(MOT, MOTS, VIS)을 하나의 네트워크와 파이프라인으로 통합한 모델이다. 추적이 탐지에 외형 정보를 제공하고, 탐지가 추적에 후보 박스를 제공하는 'Tracking-with-Detection' 패러다임을 통해 통합을 구현하였다. 실험 결과, 기존 통합 모델인 Unicorn보다 우수한 성능을 보였으며, 특히 VIS 작업까지 지원하는 범용성을 입증하였다. 이 연구는 향후 다양한 비디오 분석 작업을 단일 모델로 통합하려는 시도에 중요한 기반을 제공할 것으로 보인다.
