# miMamba: EEG-based Emotion Recognition with Multi-scale Inverted Mamba Models

Xin Zhou, Dawei Huang, Xiaojiang Peng, Lijun Yin (2021/2025)

## 🧩 Problem to Solve

본 논문은 뇌-컴퓨터 인터페이스(BCI) 분야에서 핵심적인 과제인 뇌전도(EEG) 기반의 감정 인식(Emotion Recognition) 문제를 다룬다. EEG 신호는 뇌의 복잡한 공간적 토폴로지와 동적인 시간적 의존성을 동시에 가지고 있어, 변별력 있는 시공간적 특성(spatiotemporal features)을 추출하는 것이 매우 어렵다.

기존의 연구들은 다음과 같은 한계를 가지고 있다. 첫째, 미분 엔트로피(Differential Entropy)나 전력 스펙트럼 밀도(Power Spectral Density)와 같은 도메인 특화된 시간-주파수 특징 추출에 의존하는데, 이는 많은 도메인 지식을 요구하며 긴 시계열 정보를 단일 고유값으로 압축하는 과정에서 귀중한 시간적 정보를 손실시킨다. 둘째, 딥러닝 기반 방법론들조차 시간적 의존성과 공간적 특성을 각각 별도의 브랜치에서 분석하는 경향이 있어, 국소-전역 관계와 시공간적 역학 사이의 상호작용을 충분히 활용하지 못한다.

따라서 본 논문의 목표는 도메인 특화된 수동 특징 추출 없이, EEG 신호의 국소적 세부 사항과 전역적 시간 의존성, 그리고 시공간적 상호작용을 동시에 학습할 수 있는 새로운 네트워크인 MS-iMamba를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **다중 스케일 시간 블록(Multi-Scale Temporal Block, MSTB)**과 **시공간 융합 블록(Temporal-Spatial Fusion Block, TSFB)**을 결합하여 시공간적 의존성을 극대화하는 것이다.

1.  **MSTB를 통한 다중 스케일 표현 학습**: FFT(Fast Fourier Transform)를 통해 주요 주파수 성분을 분석하고, 이를 기반으로 서로 다른 크기의 패치(patch)를 생성한다. 이를 통해 단일 타임스텝이 갖는 의미적 한계를 극복하고, 국소적 디테일과 전역적 관계를 동시에 포착한다.
2.  **Inverted Mamba 구조의 TSFB**: 기존의 임베딩 방식(동일 시간대의 서로 다른 채널을 하나의 토큰으로 묶는 방식) 대신, **Inverted Embedding**을 적용하여 동일 채널의 여러 시간 스텝을 하나의 토큰으로 묶는다. 여기에 Selective State-Space Model(SSM)인 Mamba를 적용함으로써 채널 간의 공간적 상관관계와 시간적 역동성을 효율적으로 모델링한다.

## 📎 Related Works

### 다중 스케일 표현 학습 (Multi-Scale Representation Learning)
시계열 데이터에서 단일 포인트는 의미적 정보가 부족하므로, 데이터를 패치 단위로 묶어 표현하는 방식이 효과적이다. TimesNet과 같은 모델은 1D 시계열을 2D 텐서로 변환하여 주기 내(intra-period) 및 주기 간(inter-period) 변동을 캡처한다. EEG 분야에서도 AMCNN-DGCN이나 MS-AMF와 같이 다중 스케일 특징을 융합하여 일반화 성능을 높이려는 시도가 있었다.

### 시공간 표현 학습 (Spatiotemporal Representation Learning)
CNN, GNN(Graph Neural Networks), Transformer 등을 이용하여 EEG의 공간적 구조와 시간적 흐름을 동시에 학습하려는 연구가 많았다. 특히 SGCN과 Bi-LSTM을 결합하거나 Dynamic Graph Convolution을 사용하는 방식이 제안되었다. 최근에는 iTransformer와 같이 시계열의 각 변수(채널)를 독립적인 토큰으로 취급하여 다변량 상관관계를 학습하는 Inverted 구조가 주목받고 있다.

본 논문은 이러한 다중 스케일 학습과 Inverted 구조의 아이디어를 Mamba(SSM) 모델에 접목하여, 기존의 분리된 시공간 분석 체계를 통합된 상호작용 모델로 발전시켰다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 전체 파이프라인
MS-iMamba 네트워크는 입력 EEG 신호를 받아 $\text{MSTB} \rightarrow \text{TSFB} \rightarrow \text{Linear Classifier}$ 순으로 처리하여 감정 상태를 분류한다.

### 1. Multi-Scale Temporal Block (MSTB)
MSTB는 수동 특징 추출 없이 신호 자체에서 최적의 패치 크기를 결정하고 다중 스케일 특징을 추출한다.

*   **패치 크기 결정**: 입력 신호 $X^{1D}$에 FFT를 적용하여 진폭이 가장 큰 상위 $k$개의 주파수 $\{f_1, \dots, f_k\}$를 선택한다. 각 주파수에 해당하는 주기 $p_i = \lceil L/f_i \rceil$를 패치 크기로 설정한다.
*   **2D 재구성**: 신호를 패치 크기 $p_i$에 따라 2D 형태로 Reshape 한다.
    $$X_{2D}^i = \text{Reshape}_{p_i, f_i}(\text{Padding}(X^{1D}))$$
    여기서 수직 방향은 패치 내부 변동(국소 디테일), 수평 방향은 패치 간 변동(전역 관계)을 나타낸다.
*   **특징 추출 및 융합**: MSP(Multi-Scale Perception) 모듈의 컨볼루션 연산을 통해 특징을 추출한 후, 다시 1D로 복원한다. 최종 표현은 FFT 진폭 기반의 가중치 $W_{f_i}$를 이용한 가중합으로 계산된다.
    $$X^{1D} = \sum_{i=1}^{k} W_{f_i} \times X_{1D}^i$$

### 2. Temporal-Spatial Fusion Block (TSFB)
TSFB는 MSTB에서 생성된 다중 스케일 표현을 입력으로 받아 시공간적 상호작용을 모델링한다.

*   **Inverted Embedding**: 일반적인 방식은 $\text{Time} \times \text{Channel}$ 구조에서 시간축을 기준으로 토큰화하지만, 본 모델은 $\text{Channel} \times \text{Time}$ 구조로 변환하여 **채널별 전체 시계열을 하나의 토큰**으로 취급한다.
    $$\hat{X}^{1D} = \text{Reshape}_{C, L}(X^{1D})$$
    이는 서로 다른 물리적 의미를 가진 채널들이 하나의 토큰에 섞여 정보가 희석되는 것을 방지하고, 채널 간의 상관관계를 더 명확히 포착하게 한다.

*   **iMamba (Selective SSM)**: Inverted Embedding된 데이터를 Mamba 모델에 입력한다. Mamba는 다음과 같은 상태 공간 방정식에 기반한다.
    $$\begin{aligned} H'(t) &= Ah(t) + Bx(t) \\ y(t) &= Ch(t) \end{aligned}$$
    이산화(Discretization) 과정을 통해 $\bar{A} = \exp(\Delta A)$, $\bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I) \cdot \Delta B$로 변환되어 시퀀스 데이터를 처리하며, 병렬 계산을 위해 컨볼루션 형태로 구현된다.

### 3. 학습 절차 및 손실 함수
최종 출력 $\hat{Y}_i$는 Linear 레이어와 Softmax를 통해 계산되며, 정답 라벨 $Y_{ij}$와의 차이를 줄이기 위해 Cross-Entropy 손실 함수를 사용한다.
$$L_{cls} = -\sum_{i=1}^{n} \mathbb{1}[i = \hat{Y}_i] \log(\hat{Y}_i)$$

## 📊 Results

### 실험 설정
*   **데이터셋**: DEAP, DREAMER, SEED (3종의 공개 데이터셋).
*   **입력 채널**: 전두극 영역(Frontal Polar Region)의 **단 4개 채널**만 사용 (예: FP1, FP2, AF3, AF4).
*   **평가 지표**: 분류 정확도(Accuracy).
*   **비교 모델**: iTransformer, DLinear, TimesNet, NTransformer, Informer, TCN.

### 주요 결과
1.  **정량적 성능**: MS-iMamba는 4개 채널만 사용했음에도 불구하고 DEAP(94.86%), DREAMER(94.94%), SEED(91.36%)에서 매우 높은 정확도를 기록하며 SOTA 성능을 달성하였다.
2.  **피험자 내/간 실험 (Intra/Inter-subject)**:
    *   **Intra-subject**: 모든 데이터셋에서 baseline 모델들을 압도하였다. 특히 SEED 데이터셋의 Inter-session 시나리오에서 2위 모델 대비 약 22.39%의 성능 향상을 보였다.
    *   **Inter-subject**: 데이터 변동성이 커짐에 따라 성능이 하락하지만, 그럼에도 불구하고 다른 모델들보다 훨씬 높은 강건함(Robustness)과 일반화 능력을 보였다. 특히 DLinear와 같은 선형 모델이 Inter-subject 환경에서 성능이 급락하는 것과 대조적이다.
3.  **SOTA 비교**: 수동 특징 추출(PSD, DE 등)과 모든 채널을 사용한 기존 SOTA 모델(EESCN, V-IAG 등)과 비교했을 때, Raw 데이터와 단 4개 채널만으로 유사하거나 더 높은 성능을 냈다는 점이 매우 고무적이다.

## 🧠 Insights & Discussion

### 강점 및 해석
*   **효율적인 채널 활용**: 전두엽 영역이 감정과 밀접한 관련이 있다는 뇌과학적 근거를 바탕으로 채널 수를 대폭 줄였음에도 높은 성능을 낸 것은, 모델의 특징 추출 능력이 매우 강력함을 시사한다.
*   **Inverted 구조의 유효성**: Ablation Study 결과, MSTB보다 Inverted Embedding(TSFB)이 성능 향상에 더 크게 기여했다. 이는 EEG 신호에서 시간축의 단순 나열보다 채널 간의 상호작용을 토큰화하여 분석하는 것이 훨씬 효율적임을 증명한다.
*   **Mamba의 가능성**: Transformer 기반 모델들이 제한된 채널 수에서 기대 이하의 성능을 보인 반면, SSM 기반의 Mamba 구조는 시공간적 의존성을 더 효과적으로 캡처하였다.

### 한계 및 향후 과제
*   **Cross-subject/session의 어려움**: 성능은 높지만, 피험자가 바뀌거나 세션이 바뀔 때 발생하는 데이터 드리프트(Data Drift) 문제는 여전히 해결해야 할 과제로 남아 있다.
*   **데이터 규모의 한계**: 대규모 EEG 데이터셋 확보가 어려운 상황에서, 소수 피험자 데이터만으로 미지의 피험자를 정확히 예측하는 일반화 능력의 추가 개선이 필요하다.

## 📌 TL;DR

본 논문은 EEG 기반 감정 인식을 위해 **다중 스케일 시간 블록(MSTB)**과 **Inverted Mamba 구조의 시공간 융합 블록(TSFB)**을 결합한 **MS-iMamba**를 제안한다. 이 모델은 수동 특징 추출 없이 Raw 데이터에서 직접 특징을 학습하며, 특히 채널별 시계열을 토큰화하는 Inverted Embedding을 통해 시공간적 상호작용을 극대화한다. 실험 결과, **단 4개의 EEG 채널만으로도** 기존의 전 채널 기반 SOTA 모델들을 능가하는 성능을 보였으며, 이는 향후 저비용/고효율의 실시간 감정 인식 시스템 구축에 중요한 기여를 할 것으로 기대된다.