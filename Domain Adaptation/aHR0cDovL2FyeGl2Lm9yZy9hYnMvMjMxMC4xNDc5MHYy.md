# Weighted Joint Maximum Mean Discrepancy Enabled Multi-Source-Multi-Target Unsupervised Domain Adaptation Fault Diagnosis

Zixuan Wang, Haoran Tang, Haibo Wang, Bo Qin, Mark D. Butala, Weiming Shen, and Hongwei Wang (2023)

## 🧩 Problem to Solve

본 논문은 산업 현장의 기계 고장 진단(Fault Diagnosis)에서 발생하는 **Domain Shift** 문제를 해결하고자 한다. 데이터 기반의 지능형 고장 진단 기술은 일반적으로 학습 데이터와 테스트 데이터가 동일한 분포를 가진다고 가정하며, 충분한 양의 레이블링된 데이터를 필요로 한다. 그러나 실제 산업 환경에서는 다양한 작동 상태(Operating states)가 존재하여 데이터 분포가 달라지며, 모든 작동 조건에 대해 데이터를 레이블링하는 것은 비용과 시간이 매우 많이 소요된다.

기존의 Unsupervised Domain Adaptation(UDA) 방법들은 레이블이 없는 타겟 도메인의 데이터를 처리할 수 있게 해주었으나, 대부분 **Single-Source-Single-Target(단일 소스-단일 타겟)** 시나리오에 국한되어 있었다. 단일 소스 도메인만으로는 충분한 특징 표현을 추출하기 어렵고, 여러 타겟 도메인이 존재할 때 각각의 모델을 학습시키거나 단순히 데이터를 합치는 방식은 비용 효율성이 떨어지거나 분포의 왜곡을 초래한다. 따라서 본 논문은 **Multi-Source-Multi-Target(다중 소스-다중 타겟)** 환경에서 동시에 효과적인 고장 진단을 수행하는 것을 목표로 한다.

## ✨ Key Contributions

본 연구의 핵심 기여는 고장 진단 분야에서 최초로 **Multi-Source-Multi-Target Unsupervised Domain Adaptation (WJMMD-MDA)** 방법론을 제안한 것이다. 

중심 아이디어는 여러 개의 레이블링된 소스 도메인으로부터 풍부한 정보를 추출함과 동시에, 개선된 **Weighted Joint Maximum Mean Discrepancy (WJMMD)** 거리 손실 함수를 통해 소스 도메인들과 타겟 도메인들 간의 특징 분포를 정렬하는 것이다. 이를 통해 도메인에 불변하면서도(Domain-invariant) 판별력이 높은(Discriminative) 특징을 학습하여, 복잡한 다중 도메인 환경에서도 높은 진단 정확도를 확보하고자 한다.

## 📎 Related Works

### 1. Unsupervised Domain Adaptation (UDA)
UDA는 소스 도메인의 레이블 정보를 활용해 타겟 도메인의 레이블 없는 데이터를 분류하는 기술이다.
- **Metric Discrepancy 기반 방법**: Maximum Mean Discrepancy (MMD)와 같이 두 분포의 기댓값 차이를 최소화하여 정렬한다.
- **Adversarial Learning 기반 방법**: DANN, ADDA와 같이 도메인 판별기(Discriminator)를 통해 도메인을 구분할 수 없도록 특징을 학습시킨다.
- **한계**: 대부분의 기존 연구는 단일 소스-단일 타겟 시나리오에 집중되어 있어, 실제 다중 도메인 환경에서의 활용도가 떨어진다.

### 2. Multiple Domains Adaptation
최근 다중 도메인 적응 연구가 진행되고 있으나, 주로 다음과 같은 방향으로 이루어졌다.
- **Multi-Source $\rightarrow$ Single-Target**: 여러 소스에서 정보를 모아 하나의 타겟에 적용하는 방식.
- **Single-Source $\rightarrow$ Multi-Target**: 하나의 소스 정보를 여러 타겟으로 전이하는 방식.
- **차별점**: 본 논문은 이 두 가지를 결합하여 **Multi-Source $\rightarrow$ Multi-Target** 시나리오를 고장 진단 분야에 처음으로 적용했다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조
WJMMD-MDA의 구조는 크게 **Feature Extractor(특징 추출기)**, **Domain Feature Alignment(도메인 특징 정렬)**, 그리고 **Classifier(분류기)**로 구성된다. 모든 도메인의 데이터는 가중치를 공유(Shared weights)하는 특징 추출기를 통해 공통의 특징 공간으로 매핑된다.

- **Feature Extractor**: Convolution 레이어들과 Bottleneck(Linear layer $\rightarrow$ ReLU $\rightarrow$ Dropout)으로 구성된다.
- **Classifier**: 특징 공간에서 최종 고장 클래스를 예측하는 Linear layer이다.

### 상세 방법 및 손실 함수

#### 1. 분류 손실 함수 (Classification Loss)
레이블이 존재하는 $G$개의 소스 도메인 $\{D^s_1, D^s_2, \dots, D^s_G\}$에 대해 Cross-Entropy 손실을 사용하여 학습한다. 각 소스 도메인 $D^s_i$에 대한 손실 $L^s_{ic}$는 다음과 같다.

$$L^s_{ic} = \mathbb{E}_{(x^s_{ij}, y^s_{ij}) \sim D^s_i} l_{ce}(C(x^s_{ij}), y^s_{ij})$$

전체 분류 손실 $L_c$는 모든 소스 도메인 손실의 합으로 정의된다.
$$L_c = \sum_{i=1}^{G} L^s_{ic}$$

#### 2. 도메인 정렬 손실 함수 (Distance Loss)
단순한 MMD는 레이블 정보를 무시하므로, 본 논문은 특징과 레이블의 결합 분포를 고려하는 **Joint Maximum Mean Discrepancy (JMMD)**를 사용한다. 

$$L_{JMMD} = \left\| \mathbb{E}_{x^s_i \sim D^s} \left[ \bigotimes_{l=1}^{|L|} \phi_l(z^s_l) \right] - \mathbb{E}_{x^t_j \sim D^t} \left[ \bigotimes_{l=1}^{|L|} \phi_l(z^t_l) \right] \right\|^2_{\bigotimes_{l=1}^{|L|} \otimes_l}$$

여기서 $\otimes$는 텐서 힐베르트 공간으로의 매핑을 의미하며, $z^s_l$과 $z^t_l$은 각각 소스와 타겟의 $l$번째 레이어 활성화 값이다.

다중 소스-다중 타겟 환경을 위해, 모든 소스-타겟 쌍 $(s_i, t_j)$에 대해 JMMD 거리를 계산하고 이를 **가중합(Weighted Sum)** 한다.

$$L_{dis} = \sum_{i=1}^{G} \sum_{j=1}^{H} W^s_{it}^j L_{JMMD}(s_i, t_j)$$

이때 가중치 $W^s_{it}^j$는 소프트맥스(Softmax) 형태의 함수로 정의되어, 거리가 먼 도메인 쌍에 더 큰 가중치를 부여함으로써 더 적극적으로 정렬하도록 유도한다.

$$W^s_{it}^j = \frac{e^{L_{JMMD}(s_i, t_j)}}{\sum_s \sum_t e^{L_{JMMD}(s, t)}}$$

#### 3. 최종 목적 함수 및 학습 절차
최종 손실 함수 $L$은 분류 손실과 거리 손실의 가중 합으로 결정된다.

$$L = \lambda L_c + (1-\lambda) L_{dis}$$

학습은 $\lambda$라는 트레이드오프 파라미터를 설정하고, 경사 하강법(Gradient Descent)을 통해 특징 추출기 $\theta_f$와 분류기 $\theta_y$를 동시에 최적화하는 방식으로 진행된다.

## 📊 Results

### 실험 설정
- **데이터셋**: CWRU(베어링), PU(베어링), PHM2009(기어박스)의 3가지 데이터셋을 사용하였다.
- **비교 대상(Baselines)**: JMMD, MK-MMD, CORAL, DANN, CDAN, SDAFDN.
- **평가 지표**: 정확도(Accuracy).
- **전처리**: Z-score 정규화 및 Fast Fourier Transform(FFT)을 적용하여 입력 데이터를 구성하였다.

### 주요 결과
- **정량적 결과**: 모든 데이터셋에서 제안 방법이 기존 baseline들보다 우수한 성능을 보였다. 특히 가장 난이도가 높은 **PHM2009 데이터셋에서 가장 우수한 baseline 대비 평균 정확도가 12.86% 높게** 나타났다.
- **절제 연구 (Ablation Study)**:
    - **MSST (Multi-Source $\rightarrow$ Single-Target)** 보다 본 제안 방법이 우수하며, 이는 여러 타겟 도메인의 정보를 활용하는 것이 이점이 있음을 시사한다.
    - **SSMT (Single-Source $\rightarrow$ Multi-Target)** 보다 본 제안 방법이 우수하며, 이는 여러 소스 도메인의 레이블 정보를 활용하는 것이 성능 향상에 기여함을 보여준다.
    - **c-MSMT (단순 타겟 통합)**: 여러 타겟 도메인을 단순히 하나로 합쳐 학습시키는 방식은 오히려 **Negative Transfer(부정적 전이)** 현상을 일으켜 성능이 저하되었다.
- **도메인 수의 영향**: 소스 도메인의 수가 증가할수록 레이블 정보가 많아져 성능이 향상되는 경향을 보였으나, 타겟 도메인의 수가 증가할수록 데이터 분포가 복잡해져 적응 난이도가 상승하는 결과를 보였다.

## 🧠 Insights & Discussion

### 강점
본 논문은 단순한 단일-단일 적응을 넘어 다중 소스-다중 타겟이라는 실제 산업 환경에 가까운 복잡한 시나리오를 성공적으로 다루었다. 특히 JMMD에 가중치 메커니즘을 도입하여, 거리 차이가 큰 도메인들을 더 효율적으로 정렬함으로써 성능을 극대화한 점이 돋보인다.

### 한계 및 논의사항
1. **타겟 도메인 증가에 따른 성능 저하**: 실험 결과에서 타겟 도메인이 많아질수록 성능이 떨어지는 현상이 확인되었다. 이는 다중 타겟 간의 분포 간섭이나 복잡성 증가 때문으로 추정되며, 향후 이를 해결하기 위한 타겟 도메인 간의 관계 모델링 연구가 필요할 것으로 보인다.
2. **계산 복잡도**: 모든 소스와 타겟의 쌍(Pair)에 대해 JMMD를 계산하고 가중치를 산출하므로, 도메인의 개수 $G$와 $H$가 매우 커질 경우 계산 비용이 급격히 증가할 가능성이 있다.
3. **가정**: 본 모델은 모든 소스 도메인이 동일한 클래스 레이블 체계를 가지고 있다고 가정한다. 만약 소스 도메인마다 클래스 정의가 다르다면 적용이 어려울 것이다.

## 📌 TL;DR

본 논문은 고장 진단 분야 최초로 **다중 소스-다중 타겟 비지도 도메인 적응(MSMT-UDA)** 방법인 **WJMMD-MDA**를 제안하였다. 가중치가 적용된 JMMD 거리 손실 함수를 통해 여러 소스 도메인의 지식을 효과적으로 추출하고 다수의 타겟 도메인에 동시에 전이시켰으며, 실험을 통해 기존 단일 소스/타겟 기반 방법들보다 월등한 성능을 입증하였다. 이 연구는 다양한 작동 조건이 혼재하는 실제 산업 현장에서 데이터 레이블링 비용을 줄이면서도 고정밀 진단을 가능하게 하는 기반 기술이 될 가능성이 높다.