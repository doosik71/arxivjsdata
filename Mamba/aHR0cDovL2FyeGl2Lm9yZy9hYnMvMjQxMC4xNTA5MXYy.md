# SPATIAL-MAMBA: EFFECTIVE VISUAL STATE SPACE MODELS VIA STRUCTURE-AWARE STATE FUSION

Chaodong Xiao, Minghan Li, Zhengqiang Zhang, Deyu Meng, Lei Zhang (2025)

## 🧩 Problem to Solve

본 논문은 1차원 시퀀스 데이터 처리에 매우 효율적인 Selective State Space Models (SSMs), 특히 Mamba와 같은 모델을 2차원 이미지 데이터에 적용할 때 발생하는 한계를 해결하고자 한다. 이미지 데이터는 본질적으로 2차원 공간 구조를 가지고 있으나, 기존의 Visual SSMs는 이미지를 1차원 시퀀스로 변환하여 처리하는 방식을 취한다.

이를 위해 기존 연구들은 다양한 Scanning Patterns(예: bidirectional, four-way, continuous scan 등)를 도입하여 지역적 공간 의존성을 포착하려 했다. 그러나 이러한 방식은 두 가지 주요 문제점을 가진다. 첫째, 방향성 스캔은 픽셀 간의 인접 관계를 왜곡하여 이미지 고유의 공간 문맥(Spatial Context)을 파괴한다. 예를 들어, 스위핑 스캔(Sweeping scan)에서 좌우 인접 픽셀은 거리가 1이지만, 상하 인접 픽셀은 이미지 너비만큼의 거리가 떨어져 있게 된다. 둘째, 복잡한 공간 관계를 포착하기 위해 스캔 방향을 늘릴 경우 연산 비용이 급격히 증가한다. 따라서 본 논문의 목표는 스캔 방향에 의존하지 않고도 이미지의 복잡한 공간 구조를 효율적으로 포착할 수 있는 새로운 Visual SSM 구조를 설계하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 latent state space 내에서 직접적으로 이웃 간의 연결성(Neighborhood Connectivity)을 구축하는 **Structure-Aware State Fusion (SASF)** 메커니즘을 도입하는 것이다.

기존 Mamba가 순차적인 상태 전이(Sequential State Transition)에만 의존했던 것과 달리, Spatial-Mamba는 Dilated Convolutions를 활용하여 상태 변수들을 재가중치 부여하고 병합함으로써 이미지의 공간적 구조 의존성을 포착한다. 이를 통해 단일 방향 스캔(Unidirectional Scan)만으로도 다방향 스캔을 사용하는 기존 모델들보다 더 효과적으로 시각적 문맥 정보를 흐르게 할 수 있으며, 연산 효율성을 동시에 확보하였다. 또한, 이론적 분석을 통해 Spatial-Mamba가 Linear Attention 및 오리지널 Mamba와 동일한 행렬 곱셈 프레임워크로 통합될 수 있음을 증명하여 모델의 작동 원리에 대한 깊은 이해를 제공한다.

## 📎 Related Works

기존의 SSM 연구들은 S4, S5를 거쳐 데이터 의존적 선택 메커니즘을 도입한 Mamba로 발전하며 선형 복잡도로 긴 의존성을 모델링하는 성과를 거두었다. 이를 시각 작업에 적용하려는 시도로서 Vim과 VMamba는 양방향 또는 4방향 스캔을 통해 공간 민감도를 줄이려 했으며, PlainMamba는 연속적 스캔 순서를 도입해 공간적 연속성을 유지하려 하였다. 또한 LocalMamba는 이미지를 윈도우로 나누어 지역적 의존성을 포착하는 방식을 제안했다.

그러나 이러한 기존 접근 방식들은 모두 2D 데이터를 1D로 평탄화(Flattening)하는 과정에서 발생하는 공간적 왜곡 문제를 완전히 해결하지 못했다. Spatial-Mamba는 스캔 패턴을 최적화하는 대신, 상태 공간(State Space) 자체에서 공간적 융합을 수행함으로써 기존의 '스캔 전략 기반' 접근 방식과 차별화된다.

## 🛠️ Methodology

### 전체 파이프라인 및 시스템 구조

Spatial-Mamba의 처리 과정은 크게 세 단계로 구성된다.

1. **초기 상태 계산**: 입력 이미지를 단방향 스위핑 스캔을 통해 1차원 시퀀스로 변환하고, 오리지널 SSM의 상태 전이 방정식을 통해 상태 변수를 계산한 후 다시 2D 형식으로 재구성한다.
2. **공간 문맥 획득 (SASF)**: 재구성된 상태 변수들에 대해 Structure-Aware State Fusion (SASF)을 적용하여 이웃 상태 변수들을 병합한다.
3. **최종 출력 계산**: 융합된 상태 변수를 관측 방정식(Observation Equation)에 입력하여 최종 출력을 생성한다.

### 주요 방정식 및 상세 설명

Spatial-Mamba의 수식 체계는 다음과 같이 정의된다.

먼저, 상태 전이 방정식(State Transition Equation)을 통해 기본 상태 변수 $x_t$를 계산한다.
$$x_t = A_t x_{t-1} + B_t u_t$$

다음으로, 제안된 **SASF 방정식**을 통해 구조 인식 상태 변수 $h_t$를 생성한다.
$$h_t = \sum_{k \in \Omega} \alpha_k x_{\rho_k(t)}$$
여기서 $\Omega$는 이웃 집합, $\alpha_k$는 학습 가능한 가중치, $\rho_k(t)$는 위치 $t$의 $k$번째 이웃의 인덱스를 의미한다.

마지막으로, 관측 방정식(Observation Equation)을 통해 최종 출력 $y_t$를 산출한다.
$$y_t = C_t h_t + D u_t$$

실제 구현에서는 multi-scale dilated convolutions를 사용하여 이웃 상태 변수들에 선형 가중치를 부여한다. dilation factor $d \in \{1, 3, 5\}$인 세 개의 $3 \times 3$ depth-wise 필터를 사용하여 이웃 집합 $\Omega$를 구성하며, 이를 통해 넓은 수용 영역(Receptive Field)을 확보하고 skip connection 효과를 얻는다. 구체적인 구현 식은 다음과 같다.
$$h_t = \sum_{d=1,3,5} \sum_{i,j \in \Omega^d} k^d_{ij} \cdot x_{t+iw+j}$$
여기서 $w$는 이미지의 너비를 나타낸다.

### 네트워크 아키텍처

전체 구조는 Swin-Transformer와 유사한 4단계 계층적 구조를 가진다. Overlapped Stem layer를 통해 초기 특징 맵을 생성하고, 이후 4개의 Stage를 거치며 해상도를 줄이고 채널을 늘린다. 각 Stage는 여러 개의 **Spatial-Mamba Block**으로 구성되며, 이 블록은 Structure-aware SSM과 Feed-Forward Network (FFN) 및 skip connection으로 이루어져 있다. 특히 Structure-aware SSM 내부에서는 기존 Mamba의 1D causal convolution을 $3 \times 3$ depth-wise convolution으로 대체하고, S6 모듈을 SASF 모듈로 대체하였다.

## 📊 Results

### 실험 설정

본 논문은 ImageNet-1K(분류), COCO(검출 및 분할), ADE20K(시맨틱 분할) 데이터셋을 사용하여 성능을 검증하였다. 비교 대상으로는 ConvNeXt, Swin-T, NAT, VMamba, LocalVMamba 등 CNN, Transformer, SSM 기반의 최신 모델들이 포함되었다.

### 주요 결과

1. **이미지 분류 (ImageNet-1K)**:
   - Spatial-Mamba-T는 Top-1 정확도 83.5%를 기록하며, 유사한 파라미터 및 FLOPs를 가진 ConvNeXt-T(82.1%)와 Swin-T(81.3%)를 상회하였다.
   - 특히 SSM 기반 모델 중에서는 VMamba-T(82.6%)보다 1.0% 높은 성능을 보였으며, Base 모델(Spatial-Mamba-B)은 85.3%의 높은 정확도를 달성하였다.

2. **객체 검출 및 인스턴스 분할 (COCO)**:
   - Mask R-CNN 헤드를 사용한 실험에서 Spatial-Mamba-T는 box mAP 47.6, mask mAP 42.9를 기록하여 VMamba-T보다 우수한 성능을 보였다.
   - 3x multi-scale 학습 시 Spatial-Mamba-S는 box mAP 50.5를 기록하여 경쟁 모델들 중 최상위권의 성능을 입증하였다.

3. **시맨틱 분할 (ADE20K)**:
   - UPerNet을 사용한 결과, Spatial-Mamba-T는 multi-scale mIoU 49.4를 기록하여 NAT-T(48.4)와 VMamba-T(48.8)보다 우수하였다.

4. **효율성**:
   - Spatial-Mamba-S 및 B 모델은 VMamba보다 처리량(Throughput)이 더 높으며, CNN 및 Transformer 기반 모델들보다 현저히 빠르다.

## 🧠 Insights & Discussion

### 이론적 통합 및 해석

논문은 Linear Attention, Mamba, Spatial-Mamba를 $y = Mu$라는 단일 행렬 곱셈 프레임워크로 통합하여 분석하였다.

- **Linear Attention과 Mamba**는 행렬 $M$이 하삼각 행렬(Lower Triangular Matrix) 형태를 띠며, 이는 인과적(Causal) 순서에 따른 정보 흐름을 의미한다.
- **Spatial-Mamba**는 행렬 $M$이 인접 행렬(Adjacency Matrix) 형태를 띠며, 이는 특정 순서가 아닌 공간적 이웃 간의 가중치 합으로 정보를 통합함을 의미한다.

### 시각적 분석 및 ERF

Effective Receptive Field (ERF) 분석 결과, 기존의 VMamba 등은 다방향 스캔으로 인해 수평/수직 방향으로 정보가 축적되는 방향성 편향(Directional Bias)이 관찰되었다. 반면, Spatial-Mamba는 단방향 스캔을 사용함에도 불구하고 SASF를 통해 이러한 편향을 효과적으로 제거하고 글로벌한 수용 영역을 확보함을 확인하였다. 또한, SASF 적용 전후의 상태 변수를 시각화했을 때, 융합 후의 상태 변수 $h_t$가 배경과 전경을 훨씬 더 명확하게 구분하는 능력을 보였다.

### 한계 및 확장 가능성

SASF 모듈을 Vim이나 VMamba 같은 다른 백본에 통합했을 때도 성능 향상이 관찰되었다는 점은 SASF가 일반적인 공간 모델링 도구로 활용될 수 있음을 시사한다. 다만, Depth-wise Convolution 대신 Dynamic Convolution을 사용하면 성능은 약간 상승하지만 연산 속도(Throughput)가 크게 저하되는 트레이드-오프 관계가 존재한다.

## 📌 TL;DR

본 논문은 Visual SSM의 고질적인 문제인 '스캔 패턴으로 인한 공간 구조 왜곡'을 해결하기 위해, 상태 공간 내에서 직접 이웃 정보를 융합하는 **Structure-Aware State Fusion (SASF)**을 제안하였다. 이를 통해 단 한 번의 스캔만으로도 기존 다방향 스캔 모델들을 능가하는 성능을 달성하였으며, 이론적으로는 SSM과 Attention을 아우르는 통합 행렬 프레임워크를 제시하였다. 이 연구는 향후 효율적인 시각적 상태 공간 모델 설계 및 공간적 인덕티브 바이어스(Inductive Bias)를 SSM에 주입하는 방법론에 중요한 이정표가 될 것으로 보인다.
