# Point Transformer V3 Extreme: 1st Place Solution for 2024 Waymo Open Dataset Challenge in Semantic Segmentation

Xiaoyang Wu et al. (2024)

## 🧩 Problem to Solve

본 논문은 자율주행 시스템의 핵심 요소인 3D Semantic Segmentation 작업을 해결하고자 한다. 특히, 고해상도 LiDAR 스캔 데이터와 방대한 주석을 제공하는 Waymo Open Dataset 챌린지에서 최상위 성능을 달성하는 것을 목표로 한다.

해결하고자 하는 구체적인 문제는 다음과 같다.

- **데이터 희소성 문제**: LiDAR 센서의 특성상 중심에서 멀리 떨어진 영역은 샘플링 밀도가 낮아 인식 성능이 저하되는 문제가 발생한다.
- **효율성과 확장성의 트레이드오프**: 기존의 Point Transformer 계열은 $k$-Nearest Neighbor (kNN) 연산에 의존하여 계산 복잡도가 높고 병렬화가 어려워, 모델의 규모를 확장(scaling up)하는 데 한계가 있었다.
- **데이터 손실**: 기존의 많은 인식 시스템들은 연산 효율성을 위해 특정 범위 밖의 포인트들을 제거하는 Clipping 방식을 사용하며, 이 과정에서 유용한 정보가 손실된다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Point Transformer V3 (PTv3)를 기반으로 하여, 이를 극한으로 최적화한 **Point Transformer V3 Extreme (PTv3-EX)** 솔루션을 제안한 것이다. 주요 설계 아이디어는 다음과 같다.

- **Serialization 기반의 구조적 접근**: 비정형 포인트 클라우드를 공간 채우기 곡선(Space-filling curve)을 통해 정형화된 1D 데이터로 변환하여 계산 효율성을 극대화하고 수용 영역(Receptive Field)을 확장하였다.
- **Multi-frame Training**: 현재 프레임뿐만 아니라 이전 2개 프레임의 데이터를 좌표 정렬 후 결합하여, 원거리 영역의 희소한 샘플링 문제를 보완하였다.
- **No-clipping Policy**: 기존의 Submanifold Sparse Convolution 기반 모델들이 가졌던 한계를 극복하여, 데이터 포인트를 임의로 제거하지 않고 전체를 활용하는 전략을 채택하였다.
- **Model Ensemble**: 최종 제출물을 위해 독립적으로 학습된 3개의 모델을 앙상블하여 정확도를 추가적으로 향상시켰다.

## 📎 Related Works

논문은 기존의 3D 인식 접근 방식과 PTv3의 차별점을 다음과 같이 설명한다.

- **전통적인 Point Transformer (e.g., PTv2)**: $k\text{NN}$을 사용하여 이웃 포인트를 쿼리하는 방식은 정확한 공간적 근접성을 제공하지만, 병렬화가 어려워 모델 크기를 키우는 데 제약이 있다.
- **SparseUNet 및 관련 연구**: 대규모 사전 학습을 통해 성능을 높였으나, 여전히 효율성 문제로 인해 모델의 확장성(Scalability)에 한계가 있었다.
- **OctFormer 및 FlatFormer**: 비정형 데이터를 정형 데이터로 변환하려는 시도를 하였으며, PTv3는 이러한 아이디어를 계승하여 포인트 클라우드를 언어 토큰과 같은 1D 구조로 변환하여 처리한다.

## 🛠️ Methodology

### 1. Point Transformer V3 (Base Architecture)

PTv3는 단순함과 효율성을 우선시하여 확장성을 확보하는 설계 원칙을 따른다.

- **Serialization (직렬화)**: 공간 채우기 곡선(Space-filling curve)을 사용하여 3D 포인트들을 1D 시퀀스로 정렬한다. 이 과정에서 공간적으로 가까운 포인트들이 시퀀스 상에서도 가깝게 배치되도록 하여 공간적 근접성을 유지한다.
- **Patch Grouping**: 직렬화된 데이터를 특정 패치 크기로 나누고, 필요한 경우 주변 패치에서 포인트를 빌려와 패딩(Padding)을 수행하여 고정된 크기의 텐서 형태로 만든다.
- **Attention Mechanism**:
  - **Local Attention** 및 **Flash Attention**을 적용하여 연산 속도를 높였다.
  - **Patch Interaction**: 서로 다른 직렬화 패턴을 층마다 순환적으로 적용하는 Shift Order 또는 무작위로 적용하는 Shuffle Order를 사용하여 패치 간 상호작용을 증진시킨다.
- **xCPE (Conditional Positional Encoding)**: Skip connection이 포함된 Sparse Convolution 레이어를 사용하여 조건부 위치 인코딩을 수행하며, 이를 통해 포인트의 위치 정보를 보존한다.

### 2. PTv3 Extreme의 추가 기술

기본 PTv3 아키텍처 위에 다음의 세 가지 기술을 적용하여 성능을 극대화하였다.

**가. Multi-frame Training**
원거리 영역의 부족한 샘플링을 해결하기 위해 현재 프레임 $t$와 이전 프레임 $t-1, t-2$를 결합한다.

- 좌표 정렬(Coordinate alignment)을 통해 과거 프레임을 현재 좌표계로 변환한 후 단순 결합(Concatenation)한다.
- 학습 시에는 편의를 위해 모든 프레임에 대해 지도 학습(Supervision)을 수행한다.

**나. Non-clipping Proxy**
일반적으로 Waymo 데이터셋에서는 $[-75.2, -75.2, -4, 75.2, 75.2, 2]$와 같은 특정 범위로 포인트를 Clipping 한다.

- 기존의 Submanifold Sparse Convolution 기반 모델들은 원거리의 고립된 포인트(Isolated points)를 처리하는 데 어려움이 있어 Clipping이 필수적이었다.
- 반면, PTv3는 데이터를 1D 배열로 처리하므로 고립된 포인트에 영향을 받지 않는다. 따라서 Clipping을 제거함으로써 더 많은 정보를 활용할 수 있게 되었으며, 이는 Waymo validation split에서 $\text{mIoU}$를 $72.1\%$에서 $74.8\%$로 상승시키는 결정적 요인이 되었다.

**다. Model Ensemble**

- 독립적으로 학습된 3개의 PTv3 모델을 준비하고, 각 모델이 예측한 Logits의 합 또는 평균을 구하여 최종 결과를 도출한다. (이 기술은 Test split에만 적용하였다.)

## 📊 Results

### 실험 설정

- **데이터셋**: Waymo Open Dataset
- **평가 지표**: mean Intersection over Union ($\text{mIoU}$)
- **하드웨어**: 학습은 4개의 NVIDIA A100 GPU에서 수행되었으며, 추론 지연 시간은 단일 RTX 4090 GPU에서 측정되었다.

### 정량적 결과 (Table 3 기준)

| 설정 | Validation $\text{mIoU}$ | Test $\text{mIoU}$ | 추론 지연 시간 (Single) |
| :--- | :---: | :---: | :---: |
| PTv3 (Single Frame) | $72.13\%$ | $70.68\%$ | $132\text{ms}$ |
| PTv3-EX (Multi-frame, No-clip) | $\mathbf{74.80\%}$ | $\mathbf{72.76\%}$ | $253\text{ms}$ |

- **성능 향상**: Multi-frame training과 No-clipping 전략을 통해 Validation $\text{mIoU}$가 약 $2.67\%$ 포인트 상승하였다.
- **클래스별 분석**: 특히 $\text{Motorcyclist}$, $\text{Bicyclist}$, $\text{Pedestrian}$ 등 동적 객체와 $\text{Pole}$, $\text{Curb}$ 등의 정적 객체 전반에서 성능 향상이 관찰되었다.
- **트레이드오프**: Multi-frame 데이터를 처리함에 따라 추론 지연 시간이 $132\text{ms}$에서 $253\text{ms}$로 약 2배 가까이 증가하였다.

## 🧠 Insights & Discussion

본 논문은 포인트 클라우드 처리에서 **"데이터의 구조화(Serialization)"**가 가져오는 이점을 명확히 보여준다. 기존의 Sparse Convolution 기반 모델들이 겪었던 고립 포인트 처리 문제(Isolated points problem)를 1D 시퀀스 변환이라는 구조적 변화를 통해 해결함으로써, Clipping 없이도 안정적인 학습이 가능함을 증명하였다.

또한, 단순한 모델 설계보다는 **확장성(Scalability)**과 **데이터 활용 극대화(Multi-frame, No-clip)**가 실제 챌린지 환경에서 성능을 결정짓는 핵심 요소임을 시사한다. 다만, 성능 향상을 위해 추론 시간과 메모리 사용량이 증가한 점은 실시간 자율주행 시스템 적용 시 최적화가 필요한 부분이다. 앙상블 기법을 Validation set에는 적용하지 않고 Test set에만 적용한 점은 공정한 비교를 위한 저자들의 학술적 판단으로 보인다.

## 📌 TL;DR

본 연구는 PTv3의 효율적인 1D 직렬화 구조를 기반으로 **Multi-frame training**, **No-clipping policy**, **Model ensemble**을 적용하여 2024 Waymo Open Dataset Semantic Segmentation 챌린지에서 1위를 차지하였다. 특히, 데이터의 정형화를 통해 기존 모델들이 포기했던 원거리 고립 포인트들을 효과적으로 활용함으로써 $\text{mIoU}$를 유의미하게 끌어올렸으며, 이는 향후 고해상도 LiDAR 데이터 처리 연구에 중요한 방향성을 제시한다.
