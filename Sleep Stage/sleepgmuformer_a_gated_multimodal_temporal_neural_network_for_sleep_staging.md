# SleepGMUformer: A gated multimodal temporal neural network for sleep staging

Chenjun Zhao, Xuesen Niu, Xinglin Yu, Long Chen, Na Lv, Huiyu Zhou, Aite Zhao (2025)

## 🧩 Problem to Solve

수면 단계 분류(Sleep Staging)는 수면의 질을 평가하고 수면 장애를 진단하는 핵심적인 방법이다. 전통적으로 수면 다원 검사(Polysomnography, PSG)를 통해 뇌파(EEG), 안구전도(EOG) 등 다양한 생체 신호를 기록하고 전문가가 수동으로 분류해 왔으나, 이는 노동 집약적이며 주관적인 편향이 개입될 가능성이 크다.

최근 딥러닝 기반의 자동 수면 단계 분류 연구가 활발히 진행되고 있으나, 다음과 같은 한계점이 존재한다:

1. **모달리티 기여도 무시**: 기존의 다중 모달 융합(Multimodal Fusion) 방식은 주로 단순 연결(Concatenation) 방식을 사용하는데, 이는 각 수면 단계별로 서로 다른 생체 신호(모달리티)가 기여하는 정도가 다르다는 점을 간과한다.
2. **데이터 전처리 부족**: 처리되지 않은 수면 데이터는 주파수 도메인 정보에 간섭을 일으켜 분석의 정확도를 떨어뜨릴 수 있다.
3. **장비의 제약**: PSG는 높은 정확도를 제공하지만 비용이 많이 들고 환자에게 신체적 불편함을 주어 장기적인 모니터링이 어렵다. 이에 따라 웨어러블 기기 데이터를 활용한 낮은 자원 환경에서의 수면 단계 인식 필요성이 대두되었다.

본 논문의 목표는 PSG 데이터와 웨어러블 기기 데이터를 모두 처리할 수 있으며, 각 모달리티의 기여도를 동적으로 조정하는 Gated Multimodal Units (GMU) 기반의 네트워크를 통해 수면 단계 분류의 정확도와 해석 가능성을 높이는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **데이터 특성에 맞는 맞춤형 전처리**와 **인스턴스 수준의 동적 모달리티 가중치 부여**에 있다.

1. **이종 데이터 통합 처리**: 전통적인 PSG 데이터뿐만 아니라 심박수, 호흡, 움직임과 같은 웨어러블 센서 데이터를 동일한 프레임워크 내에서 처리하여 모델의 범용성을 확장하였다.
2. **Gated Multimodal Units (GMU) 도입**: 각 모달리티에서 추출된 특징들을 단순히 합치는 것이 아니라, 현재 입력된 데이터 샘플(인스턴스)에서 어떤 모달리티가 수면 단계 결정에 더 중요한지를 결정하는 게이팅 메커니즘을 통해 동적으로 융합한다.
3. **의학적 지식 기반 전처리**: EEG 신호의 저주파 추세 제거(De-trending)와 웨어러블 데이터의 시간적 정렬 및 결측치 처리를 통해 데이터의 품질을 높이고 임상적 분석 결과와 일치하도록 설계하였다.

## 📎 Related Works

기존의 수면 단계 분류 모델들은 다음과 같은 발전을 거쳐왔다:

- **DeepSleepNet**: CNN과 Bi-LSTM을 결합하여 시간 불변 특징과 시계열 정보를 동시에 추출하였다. 하지만 입력-출력이 일대일 대응 구조여서 수면 단계 간의 전이 규칙(Transition rules)을 무시하는 한계가 있었다.
- **SeqSleepNet**: Many-to-Many 구조의 LSTM을 사용하여 수면 단계 간의 문맥적 관계와 전이 규칙을 학습하였다. 그러나 EEG 전극의 비유클리드 공간 분포 및 공간적 상관관계를 고려하지 못했다.
- **Spatiotemporal GCN**: 동적/정적 시공간 맵 컨볼루션 네트워크와 어텐션 블록을 결합하여 EEG 신호 간의 장기 의존성과 공간 정보를 캡처하였다.

그럼에도 불구하고, 기존 연구들은 주로 **단일 모달리티(주로 EEG)**에 집중하거나, 다중 모달리티를 사용할 때 단순히 특징 벡터를 연결(Concatenation)하는 방식을 채택하여 각 모달리티의 상대적 중요도를 반영하지 못했다는 차별점이 있다.

## 🛠️ Methodology

### 1. 전체 파이프라인 구조

SleepGMUformer는 **[전처리 모듈] $\rightarrow$ [단일 채널 특징 추출 모듈] $\rightarrow$ [다중 채널 동적 특징 융합 모듈] $\rightarrow$ [분류기]** 순으로 구성된다.

### 2. 전처리 모듈 (Preprocessing)

데이터셋에 따라 서로 다른 전처리 과정을 거친다.

- **PSG 데이터 (SleepEDF-78)**:
  - **De-trending**: 최소자승법(Least Squares Method) 기반의 다항식 피팅을 통해 저주파 추세를 제거한다.
  - **STFT (Short-Time Fourier Transform)**: 원시 신호를 시간-주파수 이미지로 변환하여 $T=29, F=128$ 크기의 텐서로 만든다.
- **웨어러블 데이터 (WristHR-Motion-Sleep)**:
  - **정렬 및 보간**: 서로 다른 샘플링 주기를 가진 심박수, 호흡, 걸음 수 데이터를 30초 단위(Epoch)로 정렬하고, 결측치는 보간법(Interpolation)으로 처리한다.
- **공통 처리**: 데이터를 $[0, 1]$ 범위로 정규화(Normalization)하거나 표준화(Standardization)한 후, MLP 레이어를 통해 모든 모달리티를 동일한 고차원 공간 $P$로 매핑한다.

### 3. 단일 채널 특징 추출 (Single-channel Temporal Feature Extraction)

각 채널(모달리티)은 독립적인 Transformer Block으로 구성된 특징 추출기를 통과한다.

- **Positional Encoding**: Transformer의 순서 무시 특성을 보완하기 위해 $\sin, \cos$ 함수 기반의 위치 인코딩 $PE$를 더한다.
- **CLASS Token**: 최종 분류를 위해 학습 가능한 $\tilde{f}_0$ 토큰을 입력 시퀀스 앞에 추가한다.
- **Self-Attention**: Multi-head Attention을 통해 시간적 전역 의존성을 파악한다.
  - $\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$
  - 여기서 $Q, K, V$는 각각 Query, Key, Value 벡터이다.
- 최종적으로 CLASS 토큰의 출력값 $F_t \in \mathbb{R}^P$가 해당 채널의 대표 특징 벡터가 된다.

### 4. 다중 채널 동적 특징 융합 (GMU)

단순 연결 방식의 한계를 극복하기 위해 GMU 모듈을 사용하여 모달리티별 가중치를 동적으로 결정한다.

- **은닉 특징($h_i$) 계산**: 각 모달리티 $x_i$에 대해 $\tanh$ 활성화 함수를 사용하여 특징을 변환한다.
    $$h_i = \tanh(W_i x_i^T)$$
- **게이트($z_i$) 계산**: 모든 모달리티의 특징을 결합하여 각 모달리티가 현재 인스턴스에서 얼마나 중요한지를 결정하는 가중치 $z_i$를 $\sigma(\text{sigmoid})$ 함수로 계산한다.
    $$z_i = \sigma(W_{z_i} [x_1, \dots, x_N])$$
- **최종 융합**: 가중치 $z_i$와 은닉 특징 $h_i$를 요소별 곱(Element-wise product)하여 합산한다.
    $$h = \sum_{i=1}^n z_i \odot h_i$$

### 5. 분류기 (Classifier)

융합된 특징 벡터 $h$는 두 개의 Fully Connected Layer(ReLU 활성화 함수 및 Dropout 적용)를 거쳐 Softmax 함수를 통해 5개의 수면 단계(Wake, N1, N2, N3, REM) 중 하나로 분류된다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: SleepEDF-78 (PSG 데이터), WristHR-Motion-Sleep (웨어러블 데이터)
- **평가 지표**: Accuracy(정확도), Kappa coefficient(일치도), Macro F1 score(MF1), Sensitivity, Specificity
- **하이퍼파라미터**: Adam optimizer, 학습률 $5 \times 10^{-3}$, 배치 크기 256, Attention head $H=8$

### 2. 정량적 결과

| 데이터셋 | 모델 | Accuracy | Kappa | MF1 |
| :--- | :--- | :---: | :---: | :---: |
| **SleepEDF-78** | **SleepGMUformer** | **85.03%** | **0.83** | **79.2%** |
| | XSleepNet2 | 84.0% | 0.80 | 77.9% |
| | SeqSleepNet | 82.9% | 0.80 | 76.9% |
| **WristHR-Motion-Sleep**| **SleepGMUformer** | **94.54%** | **0.91** | **89.6%** |
| | Neural net | 93.5% | - | 52.3% |
| | MultiChannel-SleepNet| 83.5% | 0.80 | 73.1% |

- **SleepEDF-78**에서 기존 SOTA 모델인 XSleepNet2보다 정확도가 $1.00\%$ 높았으며, SeqSleepNet 대비 $2.10\%$ 향상된 결과를 보였다.
- **WristHR-Motion-Sleep**에서는 $94.54\%$라는 매우 높은 정확도를 기록하며 웨어러블 데이터 기반 분류에서도 탁월한 성능을 입증하였다.

### 3. 절제 실험 (Ablation Study)

- **모달리티 조합**: EOG 데이터가 포함되었을 때 N1과 REM 단계의 분류 성능이 크게 향상되었다. 특히 Fpz-Cz 채널이 N2, N3 단계 인식에 더 유리함이 확인되었다.
- **융합 방식**: 단순 연결(Concatenation) 방식보다 GMU 기반의 동적 융합 방식이 모든 데이터셋에서 더 높은 성능을 보였다. 이는 모달리티별 기여도가 수면 단계마다 다르다는 가설을 뒷받침한다.

## 🧠 Insights & Discussion

### 1. 강점 및 분석

- **동적 가중치의 효과**: t-SNE 시각화 결과, GMU 통과 후 특징들이 더 조밀하게 클러스터링되는 것을 확인하였다. 이는 모델이 각 모달리티에서 유의미한 정보를 선택적으로 추출하여 분류 성능을 높였음을 의미한다.
- **데이터 범용성**: PSG와 웨어러블 기기라는 서로 다른 도메인의 데이터를 하나의 구조로 처리함으로써, 저자원 환경에서도 적용 가능한 수면 분석 가능성을 제시하였다.

### 2. 한계점 및 향후 과제

- **N1 단계 분류의 어려움**: 두 데이터셋 모두에서 N1 단계의 F1-score가 상대적으로 낮게 나타났다. 이는 N1이 각성(Wake)과 수면 사이의 전이 단계로서 특징이 모호하기 때문이며, 이에 대한 세밀한 특징 추출 방법 연구가 필요하다.
- **시계열 문맥 부족**: 현재 모델은 개별 에포크(Epoch)를 독립적으로 입력받아 처리하므로, 에포크 간의 시간적 순서나 전이 관계를 고려하지 못한다. 이로 인해 수면 단계 전이 구간에서 예측 결과가 불연속적으로 나타나는 경향이 있다.
- **채널 확장성**: 현재 사용된 PSG 채널 외에 EMG, 체온 등 추가적인 생체 신호를 통합하여 성능을 더욱 개선할 여지가 있다.

## 📌 TL;DR

본 논문은 수면 단계 분류를 위해 **Gated Multimodal Units (GMU)**를 도입한 **SleepGMUformer** 모델을 제안한다. 이 모델은 PSG와 웨어러블 기기의 이종 데이터를 통합 처리하며, 각 샘플마다 모달리티별 중요도를 동적으로 조절하여 융합함으로써 정확도를 높였다. 실험 결과 SleepEDF-78에서 $85.0\%$, WristHR-Motion-Sleep에서 $94.5\%$의 정확도를 달성하여 SOTA 모델들을 능가하였다. 이는 향후 웨어러블 기반의 실시간 수면 모니터링 및 정밀 진단 시스템 구축에 중요한 기여를 할 것으로 기대된다.
