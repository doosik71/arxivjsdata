# Reciprocal Normalization for Domain Adaptation

Zhiyong Huang, Kekai Sheng, Ke Li, Jian Liang, Taiping Yao, Weiming Dong, Dengwen Zhou, Xing Sun (2021)

## 🧩 Problem to Solve

본 논문은 비지도 도메인 적응(Unsupervised Domain Adaptation, UDA) 환경에서 딥러닝 모델의 성능을 저해하는 **채널 미정렬(Channel Misalignment)** 문제를 해결하고자 한다. 

일반적으로 딥러닝 모델에서 널리 사용되는 Batch Normalization(BN)은 데이터의 평균과 분산을 통해 도메인 관련 지식을 표현하는데, 이는 소스 도메인과 타겟 도메인 간의 통계적 특성이 크게 다른 UDA 작업에서는 오히려 전이 성능을 떨어뜨리는 요인이 된다. 기존의 BN 변형 방법들(AdaBN, AutoDIAL, TN 등)은 각 도메인의 통계량을 분리하거나 결합하여 사용하지만, 주로 **동일한 인덱스의 채널(corresponding channels)** 간의 관계만을 고려한다.

하지만 저자들은 실제 도메인 간의 특성 차이로 인해, 소스 도메인과 타겟 도메인에서 동일하거나 유사한 패턴이 서로 다른 채널에 의해 캡처될 수 있다는 점에 주목하였다. 즉, 동일한 인덱스의 채널을 강제로 정렬하려는 기존 방식은 도메인 특유의 정보를 손실시키고 최적의 전이 성능을 달성하는 데 한계가 있다. 따라서 본 논문의 목표는 도메인 간의 상호 관계를 유연하게 모델링하여 채널 미정렬 문제를 해결하는 새로운 정규화 방법인 **Reciprocal Normalization(RN)**을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 동일 채널 간의 정렬을 넘어, **도메인 간 모든 채널 쌍의 상관관계(cross-domain channel-wise correlation)를 분석하여 상호 보완적인 정보를 추출**하는 것이다.

핵심 기여 사항은 다음과 같다:
1. **Reciprocal Normalization(RN) 설계**: 도메인 간 채널 미정렬 문제를 해결하기 위해 상호 보완적인 정보를 획득하고 이를 적응적으로 결합하는 새로운 정규화 스킴을 제안하였다.
2. **구조적 도메인 정렬**: Reciprocal Compensation(RC) 모듈을 통해 전역적인 채널 상관관계를 캡처하고, Reciprocal Aggregation(RA) 모듈을 통해 이를 통합함으로써 도메인 간의 구조적 정렬을 수행한다.
3. **범용적인 플러그 앤 플레이 모듈**: RN은 특정 알고리즘에 종속되지 않고 기존의 다양한 도메인 적응 방법론(DANN, CDAN 등)에 쉽게 통합되어 성능을 향상시킬 수 있는 범용성을 가진다.

## 📎 Related Works

### 1. 도메인 적응 (Domain Adaptation)
기존 연구들은 주로 손실 함수 설계나 네트워크 구조 변경에 집중하였다.
- **통계량 일치**: MMD(Maximum Mean Discrepancy)나 Wasserstein Distance 등을 사용하여 소스와 타겟의 분포 차이를 최소화하는 방식(DDC, DAN, SWD 등)이 있다.
- **적대적 학습**: 도메인 판별기(Domain Discriminator)를 도입하여 도메인 간의 구분을 어렵게 만드는 방식(DANN, ADDA, CDAN 등)이 주를 이룬다.
- **한계**: 이러한 방법들은 특징 수준의 정렬에 집중하지만, 정규화 모듈 내에서 발생하는 채널 간의 미정렬 문제는 간과하는 경향이 있다.

### 2. 정규화 기술 (Normalization Techniques)
BN의 한계를 극복하기 위한 여러 변형들이 제안되었다.
- **AdaBN**: 추론 단계에서 타겟 도메인의 통계량을 사용한다.
- **AutoDIAL**: 소스와 타겟의 통계량을 채널별로 가중 합산하여 통합한다.
- **DSBN**: 소스와 타겟의 BN 파라미터를 완전히 분리하여 관리한다.
- **TN (Transferable Normalization)**: 채널 어텐션 메커니즘을 통해 전이 가능성이 높은 채널에 집중한다.
- **차별점**: 기존 방법들은 여전히 '동일 인덱스 채널' 간의 관계만을 고려하거나 단순히 분리하는 수준에 머물러 있다. 반면, RN은 **비-동일 채널(non-corresponding channels)** 간의 상관관계를 모델링함으로써 더 넓은 범위의 도메인 정보를 활용한다.

## 🛠️ Methodology

RN은 크게 **Reciprocal Compensation(RC)**과 **Reciprocal Aggregation(RA)** 두 단계로 구성된다. 이해를 돕기 위해 소스($s$)에서 타겟($t$)으로의 흐름을 중심으로 설명한다.

### 1. Reciprocal Compensation (RC)
RC 모듈은 타겟 도메인의 각 채널이 소스 도메인의 어떤 채널들과 유사한지를 분석하여 보완적인 통계량을 생성한다.

- **상관관계 계산**: 타겟 채널 $i$와 소스 채널 $j$ 사이의 평균($\mu$)과 분산($\sigma^2$)에 대해 음의 $l_2$ 거리를 사용하여 상관관계 $E$를 계산한다.
  $$E^\mu_{i,j} = -(\mu_{i,t} - \mu_{j,s})^2, \quad E^{\sigma^2}_{i,j} = -(\sigma^2_{i,t} - \sigma^2_{j,s})^2$$
- **확률 가중치 도출**: 행(row) 방향으로 Softmax를 적용하여 정규화된 상관관계 가중치 $\rho$를 구한다.
  $$\rho^\mu_{t \to s} = \text{softmax}(E^\mu_{t \to s}, \text{dim}=1)$$
- **보완적 통계량 생성**: 이 가중치를 소스 도메인의 통계량에 곱하여 타겟 채널을 위한 보완적 통계량(compensatory)을 계산한다.
  $$\mu_{t,cc} = \rho^\mu_{t \to s} \cdot \mu_s, \quad \sigma^2_{t,cc} = \rho^{\sigma^2}_{t \to s} \cdot \sigma^2_s$$

### 2. Reciprocal Aggregation (RA)
보완적 통계량을 그대로 사용하면 기존 도메인의 특성 정보가 소실될 수 있으므로, 학습 가능한 게이트 파라미터 $g \in [0.5, 1]$를 도입하여 적응적으로 결합한다.

- **적응적 결합**: 원본 통계량과 보완적 통계량을 Hadamard product($*$)를 통해 결합한다.
  $$\tilde{\mu}_t = g^\mu_t * \mu_t + (1 - g^\mu_t) * \mu_{t,cc}$$
  $$\tilde{\sigma}^2_t = g^{\sigma^2}_t * \sigma^2_t + (1 - g^{\sigma^2}_t) * \sigma^2_{t,cc}$$
- 여기서 $g$는 초기값으로 1로 설정되어 학습 초기에는 도메인 특유의 정규화를 수행하고, 학습이 진행됨에 따라 점진적으로 도메인 간 간극을 좁히도록 학습된다.

### 3. 정규화 및 추론 절차
결합된 통계량 $\tilde{\mu}, \tilde{\sigma}^2$를 사용하여 특징 맵을 정규화하며, 학습 가능한 아핀 파라미터 $\gamma, \beta$를 적용한다.
$$\hat{x}^{(i)}_t = \gamma^{(i)} \left( \frac{x^{(i)}_t - \tilde{\mu}^{(i)}_t}{\sqrt{\tilde{\sigma}^{2(i)}_t + \epsilon}} \right) + \beta^{(i)}$$

**추론(Inference)** 단계에서는 계산 효율성을 위해 BN과 유사하게 학습 과정에서 지수 이동 평균(EMA)으로 업데이트된 누적 통계량 $\bar{\mu}, \bar{\sigma}^2$를 사용하여 직접 정규화를 수행한다.

## 📊 Results

### 1. 실험 설정
- **데이터셋**: ImageCLEF-DA (소규모), Office-Home (중규모), VisDA-C (대규모)
- **작업 시나리오**: Closed-set UDA, Partial-set DA (PDA), Multi-source DA (MSDA)
- **베이스라인**: BN, AutoDIAL, DSBN, TN 및 최신 UDA 방법론(DANN, CDAN 등)
- **백본**: ResNet-50, ResNet-101

### 2. 주요 결과
- **UDA 성능 향상**: Office-Home 데이터셋에서 CDAN에 RN을 적용했을 때 평균 정확도가 65.8%에서 70.6%로 크게 상승하였다. VisDA-C(대규모)에서는 CDAN+RN이 ResNet-50 기준 79.6%의 정확도를 기록하며 기존 CDAN(70.0%) 대비 9.6%p 향상되는 압도적인 결과를 보였다.
- **범용성 검증**: DANN, CDAN, ETN, $\text{BA}^3\text{US}$ 등 서로 다른 성격의 방법론들에 RN을 적용했을 때 모두 성능 향상이 관찰되었다. 특히 PDA와 MSDA 시나리오에서도 일관된 성능 향상을 보여 RN의 범용성을 입증하였다.
- **정규화 모듈 비교**: Table IV에서 RN은 AutoDIAL, DSBN, TN 등 기존 정규화 변형들보다 일관되게 높은 성능을 보였으며, 특히 데이터셋의 규모가 커질수록 그 효과가 뚜렷하게 나타났다.
- **효율성**: 학습 시간은 BN보다 약간 증가하지만, 추론 시간은 BN과 거의 동일한 수준으로 매우 효율적이다.

## 🧠 Insights & Discussion

### 1. 이론적 분석
저자들은 도메인 적응의 학습 상한선(Learning Bound) 이론을 통해 RN의 효과를 분석하였다. 실험 결과, RN을 적용했을 때 **A-distance**($d_{\mathcal{H}\Delta\mathcal{H}}(S,T)$)와 **$\lambda$** 값이 모두 낮게 측정되었다. 이는 RN이 도메인 간의 불일치를 효과적으로 줄여 더 전이 가능한 표현(transferable representation)을 학습하게 함을 의미한다.

### 2. 채널 정렬의 실재
최근 특징 층의 채널 간 거리를 측정한 결과, 가장 가까운 채널 쌍 중에서 동일 인덱스 채널이 차지하는 비중이 매우 낮음을 확인하였다. 이는 저자들이 주장한 '채널 미정렬' 현상이 실제로 존재하며, 이를 해결하기 위해 비-동일 채널 간의 관계를 고려하는 RN의 접근 방식이 타당함을 뒷받침한다.

### 3. 한계 및 해석
- **데이터 규모 의존성**: 대규모 데이터셋에서 성능 향상 폭이 더 컸는데, 이는 도메인 통계량이 더 정확하게 추정될수록 RN의 보완적 통계량 계산이 더 정교해지기 때문으로 해석된다.
- **게이트 파라미터**: $g$ 값의 범위 $[0.5, 1]$ 제약이 중요하며, $0.5$ 미만으로 내려갈 경우 원본 도메인 정보의 손실로 인해 성능이 하락하는 경향을 보였다.

## 📌 TL;DR

본 논문은 UDA에서 발생하는 **채널 미정렬(Channel Misalignment)** 문제를 해결하기 위해, 도메인 간 모든 채널의 상관관계를 분석하여 보완적인 통계량을 생성하고 이를 적응적으로 결합하는 **Reciprocal Normalization(RN)**을 제안하였다. RN은 단순한 플러그 앤 플레이 모듈로서 기존의 다양한 DA 방법론에 통합 가능하며, 특히 대규모 데이터셋과 다양한 적응 시나리오(PDA, MSDA)에서 탁월한 성능 향상을 입증하였다. 이 연구는 향후 객체 검출이나 세그멘테이션과 같은 복잡한 CV 작업의 도메인 적응 연구에 중요한 기반을 제공할 가능성이 높다.