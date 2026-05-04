# A hybrid approach for improving U-Net variants in medical image segmentation

Aitik Gupta, Joydip Dhar (2023)

## 🧩 Problem to Solve

본 논문은 의료 영상 분할(Medical Image Segmentation) 분야에서 딥러닝 모델, 특히 U-Net 변형 모델들이 직면한 효율성 문제를 해결하고자 한다. 의료 영상 분할은 진단, 수술 계획 및 치료 평가를 위해 매우 중요하지만, 최신 모델들은 다음과 같은 문제점을 가지고 있다.

- **과도한 파라미터 수**: Attention U-Net이나 MultiResUNet과 같은 최신 변형 모델들은 정확도를 높이기 위해 복잡한 구조를 채택하고 있으며, 이는 학습 가능한 파라미터 수의 급격한 증가로 이어진다.
- **과적합(Overfitting) 위험**: 파라미터 수가 너무 많을 경우, 사용 가능한 데이터 양에 비해 모델이 너무 복잡해져 훈련 데이터에만 특화되고 새로운 데이터에 대한 일반화 성능이 떨어지는 과적합 문제가 발생한다.
- **추론 시간 및 계산 비용 증가**: 높은 파라미터 요구 사항은 결과적으로 추론 시간(Inference time)을 증가시켜 실시간 진단이나 자원이 제한된 환경에서의 적용을 어렵게 만든다.

따라서 본 연구의 목표는 Depthwise Separable Convolutions를 도입하여 네트워크 파라미터 수를 획기적으로 줄이면서도, Attention 시스템과 Residual Connection을 통해 의료 영상 분할 작업(특히 피부 병변 분할)에서의 성능을 유지하거나 향상시키는 최적화된 하이브리드 접근 방식을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 모델의 경량화와 성능 유지라는 두 마리 토끼를 잡기 위해 세 가지 핵심 요소를 결합한 하이브리드 아키텍처를 설계한 것이다.

1. **Depthwise Separable Convolutions의 도입**: 표준 합성곱(Standard Convolution)을 Depthwise와 Pointwise 단계로 분리하여 계산량과 파라미터 수를 대폭 감소시킨다.
2. **Soft Attention 메커니즘 적용**: Skip Connection 단계에 Attention Gate를 추가하여, 디코더가 인코더의 특징 맵 중에서 중요한 정보에만 집중하게 함으로써 분할 정확도를 높인다.
3. **Residual Connection 및 밀집 연결(Densely Connected Blocks) 활용**: 그라디언트 소실(Vanishing Gradient) 문제를 방지하고 깊은 층에서도 특징 학습이 효율적으로 이루어지도록 설계하였다.

## 📎 Related Works

논문은 의료 영상 분할의 발전 과정을 세 단계로 나누어 설명한다.

- **전통적 방법**: 
    - **임계값 처리(Thresholding)**: 픽셀 강도를 기준으로 이진화하는 방법으로, 간단하지만 노이즈에 취약하다. Global, Adaptive, Otsu 방법 등이 있다.
    - **군집화(Clustering)**: K-Means와 같이 픽셀 유사성을 바탕으로 영역을 나누는 방법이다. 초기 중심점 설정에 민감하며 최적의 분할을 보장하지 못한다.
- **CNN 및 U-Net**: 
    - **U-Net**: 인코더(수축 경로)와 디코더(확장 경로) 구조 및 Skip Connection을 도입하여 의료 영상 분할의 표준이 되었다.
    - **Attention U-Net**: Attention Gate를 통해 중요 특징에 가중치를 부여하여 정확도를 높였으나 파라미터 수가 증가하는 단점이 있다.
    - **MultiResUNet**: 다중 해상도 스케일을 사용하여 미세하고 거친 정보를 모두 포착하도록 설계되었으며 성능은 우수하지만 매우 무겁다.
- **Residual Learning**: ResNet에서 제안된 잔차 연결(Residual Connection)은 입력값을 출력값에 더해줌으로써 매우 깊은 네트워크에서도 학습이 가능하게 하여 그라디언트 소실 문제를 해결하였다.

## 🛠️ Methodology

### 전체 시스템 구조
제안된 아키텍처는 기본적으로 U-Net의 인코더-디코더 구조를 따르며, 각 구성 요소에 경량화 및 성능 향상 모듈을 결합하였다.

1. **Encoder**: 여러 개의 밀집 연결 블록(Densely Connected Blocks)으로 구성된다. 각 블록은 Depthwise Separable Convolution, Batch Normalization, ReLU 활성화 함수를 포함하며, Residual Connection을 통해 다음 블록으로 연결된다.
2. **Attention Gate**: 인코더의 특징 맵과 디코더의 특징 맵을 입력으로 받아 Attention 계수를 계산한다. 이를 통해 디코더가 필요한 정보만을 선택적으로 수용하도록 돕는다.
3. **Decoder**: 업샘플링 층과 Skip Connection을 통해 인코더의 특징을 결합한다. 이후 다시 밀집 연결 블록을 통과하며 해상도를 복원한다.
4. **Output**: 최종적으로 $1 \times 1$ Convolution 층을 통해 각 픽셀을 클래스 수에 맞게 매핑하여 분할 맵을 생성한다.

### 주요 구성 요소 상세 설명

#### 1. Depthwise Separable Convolutions
표준 합성곱을 다음 두 단계로 나누어 수행함으로써 파라미터를 줄인다.
- **Depthwise Convolution**: 각 입력 채널에 대해 개별적인 필터를 적용하여 공간적 특징을 추출한다.
- **Pointwise Convolution**: $1 \times 1$ 필터를 사용하여 Depthwise 단계에서 생성된 채널들을 결합하고 출력 채널 수를 조정한다.

#### 2. Soft Attention in Skip Connections
단순히 특징을 복사하는 Skip Connection 대신, 가중치 함수를 통해 중요도를 조절하는 Soft Attention을 적용한다. 이는 이진 마스크를 사용하는 Hard Attention보다 더 세밀한 제어가 가능하며, 네트워크가 자동으로 어떤 특징을 결합할지 학습하게 한다.

#### 3. 기본 CNN 블록 및 방정식
각 블록은 Depthwise Separable Convolution $\rightarrow$ Group Normalization (GN) $\rightarrow$ LeakyReLU 순으로 구성된다. 특징 맵의 융합은 다음과 같은 수식으로 설명된다.

$$x_0 = F_{conv 1\times 1}(x_{input})$$
$$x_i = F_{conv 3\times 3}(\text{SUM}(x_0; x_1; x_2; \dots; x_{i-1}))$$

여기서 $x_i \in \mathbb{R}^{C \times H \times W}$는 $i$번째 층의 특징 맵을 의미하며, 이전 층들의 합(SUM)을 통해 정보를 융합함으로써 추가 파라미터 없이 추출된 정보를 통합한다.

**참고**: 논문 내에서 학습에 사용된 구체적인 손실 함수(Loss Function)와 학습 최적화 알고리즘(Optimizer)에 대한 명시적인 언급은 없다.

## 📊 Results

### 실험 설정
- **데이터셋**: 
    - **HAM10000**: 피부 병변 이미지 데이터셋 (7개 클래스, 총 10,015장). 256x256 크기로 전처리 후 훈련(80%), 테스트(10%), 검증(10%)으로 분할하였다.
    - **Thyroid Gland Dataset**: 갑상선 분할 데이터셋.
- **평가 지표**: 
    - **Dice Coefficient**: 예측과 정답의 겹침 정도를 측정. $\text{Dice} = \frac{2 \times TP}{2TP + FN + FP}$
    - **IoU (Intersection over Union)**: 합집합 대비 교집합의 영역을 측정. $\text{IoU} = \frac{TP}{TP + FN + FP}$
    - **ASSD (Average Symmetric Surface Distance)**: 예측 경계와 실제 경계 사이의 평균 거리를 측정하며, 값이 작을수록 정확하다.

### 정량적 결과 분석

#### 1. 피부 병변 분할 (Skin Lesion Segmentation)
| 접근 방식 | Accuracy | Dice Coefficient | 파라미터 수 |
| :--- | :---: | :---: | :---: |
| U-Net | 0.8777 | 0.8739 | 7.76M |
| Attention-UNet | 0.8812 | 0.8854 | 34.88M |
| MultiResUNet | 0.9221 | 0.9179 | 64.8M |
| **Hybrid (제안 방식)** | **0.9082** | **0.8872** | **2.3M** |

- 제안 모델은 U-Net 및 Attention U-Net보다 높은 정확도와 Dice 계수를 보였다.
- MultiResUNet보다는 수치적으로 낮지만, 파라미터 수를 약 **97%** 수준으로 획기적으로 줄이면서도 경쟁력 있는 성능을 유지하였다.

#### 2. 갑상선 분할 (Thyroid Gland Segmentation)
| 접근 방식 | Accuracy | Dice Coefficient | 파라미터 수 |
| :--- | :---: | :---: | :---: |
| U-Net | 0.9347 | 0.9332 | 7.76M |
| Attention-UNet | 0.9526 | 0.9169 | 34.88M |
| MultiResUNet | 0.9727 | 0.9544 | 64.8M |
| **Hybrid (제안 방식)** | **0.9766** | **0.9493** | **2.3M** |

- 갑상선 분할에서는 MultiResUNet을 포함한 모든 비교 대상보다 높은 정확도($0.9766$)를 달성하였으며, 파라미터 효율성 또한 극도로 높다.

#### 3. 소거 연구 (Ablation Study - 피부 병변 기준)
- **U-Net + DC**: IoU 0.7164 / Dice 0.8232 / ASSD 1.6189
- **U-Net + DC + RC**: IoU 0.7672 / Dice 0.8612 / ASSD 1.0833
- **U-Net + DC + RC + Attention Pooling**: IoU 0.8157 / Dice 0.8872 / ASSD 0.8364
- 결과적으로 Depthwise Conv, Residual Connection, Attention Pooling이 단계적으로 추가될 때마다 성능이 지속적으로 향상됨을 확인하였다.

## 🧠 Insights & Discussion

본 논문의 강점은 **극단적인 파라미터 최적화**에 있다. 일반적인 U-Net 변형 모델들이 성능 향상을 위해 모델의 크기를 키우는 방향으로 발전한 반면, 본 연구는 Depthwise Separable Convolution을 통해 파라미터 수를 2.3M까지 낮추면서도 SOTA 모델들에 근접하거나 이를 능가하는 성능을 보였다.

특히 갑상선 분할 데이터셋에서 MultiResUNet보다 높은 정확도를 기록한 점은, 단순히 모델이 크다고 해서 항상 최적의 성능을 내는 것이 아니며, 적절한 Attention 메커니즘과 구조적 최적화가 더 중요하다는 것을 시사한다.

다만, 몇 가지 한계점과 논의 사항이 존재한다.
- **데이터셋 편향**: 피부 병변 데이터셋(HAM10000)의 경우 특정 클래스(NV)에 데이터가 매우 쏠려 있는 class imbalance 문제가 명시되어 있으나, 이를 해결하기 위한 손실 함수(예: Weighted Cross Entropy)나 샘플링 전략에 대한 설명이 부족하다.
- **계산 복잡도 분석**: 파라미터 수는 줄었으나, 실제 추론 속도(Inference latency)나 FLOPs에 대한 정량적 비교 데이터가 제시되지 않아 실질적인 속도 향상 폭을 정확히 알 수 없다.

## 📌 TL;DR

본 논문은 U-Net의 성능을 유지하면서 파라미터 수를 획기적으로 줄인 **하이브리드 경량 분할 네트워크**를 제안한다. **Depthwise Separable Convolutions, Soft Attention, Residual Connections**를 결합하여, MultiResUNet 대비 파라미터 수를 약 97% 줄이면서도 피부 병변 및 갑상선 분할 작업에서 매우 효율적이고 정확한 성능을 입증하였다. 이 연구는 자원이 제한된 의료 환경에서 고성능 분할 모델을 배포하는 데 중요한 기초 자료가 될 가능성이 높다.