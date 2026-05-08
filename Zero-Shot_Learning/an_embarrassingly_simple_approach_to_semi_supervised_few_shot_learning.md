# An Embarrassingly Simple Approach to Semi-Supervised Few-Shot Learning

Xiu-Shen Wei, He-Yang Xu, Faen Zhang, Yuxin Peng, Wei Zhou (2022)

## 🧩 Problem to Solve

본 논문은 **Semi-Supervised Few-Shot Learning (SSFSL)** 문제를 해결하고자 한다. SSFSL은 극소량의 라벨링된 데이터(Support set)와 일정량의 라벨링되지 않은 데이터(Unlabeled data)를 활용하여 새로운 클래스들에 빠르게 적응하는 분류기를 학습시키는 작업이다.

이 문제의 핵심적인 어려움은 라벨링된 데이터가 극도로 부족한 상황(예: 1-shot)에서는 모델이 초기 단계에서 매우 부정확한 예측을 수행한다는 점이다. 기존의 많은 SSFSL 방법론들은 라벨이 없는 데이터에 대해 직접적으로 **Positive pseudo-label**(어떤 클래스에 속하는지에 대한 가짜 라벨)을 예측하여 학습 데이터를 확장하려 시도하지만, 모델의 초기 성능이 낮아 잘못된 pseudo-label이 생성되고 이것이 학습 과정에서 오염(noise)으로 작용하는 문제가 발생한다. 따라서 본 논문의 목표는 매우 제한된 라벨 환경에서도 신뢰할 수 있는 pseudo-label을 생성하여 분류 성능을 효과적으로 향상시키는 단순하고 강력한 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **"어떤 클래스인지(Positive) 맞히는 것보다, 어떤 클래스가 아닌지(Negative) 맞히는 것이 훨씬 쉽다"**는 직관에서 출발한다. 이를 위해 저자들은 **MUSIC (Method of sUccesSIve exClusions)**이라는 방법론을 제안한다.

MUSIC의 중심 설계 아이디어는 다음과 같다.

1. **간접 학습 관점의 접근**: Unlabeled 데이터에 대해 가장 낮은 확신도를 가진 클래스를 찾아 이를 **Negative pseudo-label**(해당 클래스가 아님을 의미)로 지정한다.
2. **연속적 배제(Successive Exclusion)**: 한 번의 Negative label을 찾고 모델을 업데이트한 뒤, 해당 클래스를 제외한 나머지 클래스들 중에서 다시 Negative label을 찾는 과정을 반복한다.
3. **Positive label의 자연스러운 도출**: 모든(또는 대부분의) 클래스가 Negative label로 배제되고 나면, 마지막으로 남은 클래스가 자연스럽게 해당 데이터의 Positive label이 된다.

## 📎 Related Works

### Few-Shot Learning (FSL)

FSL은 소수의 예제만으로 새로운 카테고리를 학습하는 것을 목표로 하며, 크게 두 가지 흐름으로 나뉜다.

- **Meta-learning 기반 방법**: Metric-based(예: Prototypical Networks)와 Optimization-based(예: MAML) 방법론이 있으며, 이는 '학습하는 법을 학습(learning-to-learn)'하여 빠른 적응을 꾀한다.
- **Transfer-learning 기반 방법**: 대량의 base-class 데이터로 모델을 사전 학습(pre-train)한 후, 이를 novel-class 분류에 활용한다.

### Semi-Supervised Few-Shot Learning (SSFSL)

라벨이 없는 데이터를 활용해 FSL의 성능을 높이려는 시도로, TPN, ICI, iLPC, PLCM 등의 방법론이 제안되었다. 대부분의 최신 SSFSL 방법론들은 pseudo-labeling을 통해 support set을 확장하는 방식을 취하지만, 본 논문은 기존의 Positive pseudo-labeling 방식 대신 **Complementary labels(Negative learning)**를 도입했다는 점에서 차별점을 갖는다.

### Negative Learning (NL)

Negative Learning은 "입력 이미지가 특정 라벨에 속하지 않는다"라는 정보를 학습하는 간접 학습 패러다임이다. 일반적인 Positive Learning(PL)보다 라벨 수집 비용이 적고 노이즈에 강한 특성이 있으며, 본 논문은 이를 SSFSL의 pseudo-label 생성 과정에 최초로 도입하였다.

## 🛠️ Methodology

### 전체 파이프라인 및 시스템 구조

MUSIC의 전체 과정은 사전 학습된 특성 추출기(Feature Extractor)를 사용하여 이미지의 임베딩을 얻은 후, 단순한 선형 분류기를 통해 연속적인 배제 과정을 거쳐 pseudo-label을 생성하고 학습하는 구조이다.

1. **특성 추출**: 사전 학습된 네트워크 $F(\cdot; \Theta)$를 통해 이미지 $I$를 벡터 $x$로 변환한다.
   $$x=F(I; \Theta) \in \mathbb{R}^d$$
2. **초기 학습**: 극소량의 support set $S$를 사용하여 분류기 $f(\cdot; \theta)$를 Cross-Entropy loss로 초기 학습시킨다.
   $$L(f,y) = -\sum_{k} y_k \log p_k$$
3. **연속적 배제(Successive Exclusion) 과정**:
   - Unlabeled 데이터 $I^u$에 대해 예측 확률 $p^u$를 구한다.
   - 가장 낮은 확률을 가진 클래스 $k$를 찾고, 그 값이 임계값 $\delta$보다 작을 때 이를 Negative pseudo-label $y^u_k = 1$로 할당한다.
     $$y^u_k = \begin{cases} 1 & \text{if } k = \arg \min(p^u) \text{ and } p^u_k \leq \delta \\ \text{rejection} & \text{otherwise} \end{cases}$$
   - 이 Negative label을 사용하여 모델을 업데이트한다. 이때 사용되는 손실 함수는 다음과 같다.
     $$L(f, y^u) = -\sum_{k} y^u_k \log(1-p^u_k)$$
   - 업데이트된 모델을 사용하여 $k$번째 클래스를 제외한 나머지 클래스들 중에서 다시 Negative label을 찾는 과정을 반복한다.
4. **Positive label 복구 및 최종 학습**: 모든 Negative label이 결정되면, 자동으로 남은 클래스가 Positive label이 된다. 이를 사용하여 최종적으로 분류기를 업데이트한다.

### 추가 구성 요소

- **Minimum-Entropy Loss (MinEnt)**: pseudo-label의 확신도를 높여 분포를 더욱 뾰족하게(sharp) 만들기 위해 다음의 손실 함수를 추가한다.
  $$L(f, p^u) = -\sum_{k} p^u_k \log p^u_k$$
- **Reject Option ($\delta$)**: 확신도가 충분하지 않은 경우 pseudo-label을 생성하지 않도록 하여 학습의 안정성을 확보한다.

## 📊 Results

### 실험 설정

- **데이터셋**: miniImageNet, tieredImageNet, CIFAR-FS, CUB (총 4개 벤치마크).
- **백본**: ResNet-12.
- **설정**: Inductive inference(쿼리 셋을 학습 시 보지 않음)와 Transductive inference(쿼리 셋을 학습 시 활용) 두 가지 환경에서 모두 실험하였다.
- **지표**: 평균 정확도(Average Accuracy) 및 95% 신뢰 구간.

### 주요 결과

- **정량적 성과**: Table 1과 2에서 확인된 바와 같이, MUSIC은 모든 데이터셋에서 기존의 SOTA FSL 및 SSFSL 방법론들을 큰 격차로 상회하는 성능을 보였다.
- **강건성 검증**: Unlabeled set에 support set에 없는 클래스가 섞여 있는 **Distractive setup**에서도 MUSIC은 가장 우수한 성능을 기록하였다. 이는 Positive label보다 Negative label이 오류 발생 가능성이 훨씬 낮다는 가정을 뒷받침한다.
- **데이터 양에 따른 성능**: Unlabeled 데이터의 양이 증가함에 따라 성능이 안정적으로 향상되는 경향을 보였으며, 이는 타 방법론 대비 더 가파른 상승 곡선을 그렸다.

## 🧠 Insights & Discussion

### 분석 및 고찰

1. **Negative $\rightarrow$ Positive 순서의 중요성**: 실험 결과, Negative label을 먼저 예측하고 나중에 Positive label을 도출하는 방식이 그 반대 경우보다 성능이 높았다. 이는 라벨 제약이 심한 환경에서 Negative 학습이 모델 학습의 더 좋은 기초(foundation)를 제공함을 의미한다.
2. **Pseudo-label의 정확도**: 분석 결과, MUSIC이 생성한 Negative pseudo-label의 에러율은 매우 낮았으며(최종 단계에서도 6.7% 이하), Positive pseudo-label의 에러율 또한 기존 SOTA 방법론(25% 이상)보다 훨씬 낮은 약 10% 수준이었다.
3. **분포의 균형**: MUSIC을 통해 생성된 pseudo-label들은 각 클래스별로 매우 균형 잡힌 분포를 보였으며, 이는 특정 클래스로 편향되지 않은 강건한 분류기 학습에 기여하였다.
4. **$\delta$의 역할**: 임계값 $\delta$를 제거했을 때 성능이 하락하는 것을 통해, 확신도가 낮은 샘플을 배제하는 전략이 필수적임을 확인하였다.

### 한계 및 비판적 해석

본 논문은 매우 단순한 구조로 뛰어난 성능을 냈으나, 이론적인 수렴성(convergence)이나 추정 오차의 한계(estimation error bound)에 대한 수학적 분석은 제공하지 않았다. 또한, 제안된 방법이 일반적인 Semi-Supervised Learning 문제에서도 동일하게 작동할지에 대해서는 추가 연구가 필요하다.

## 📌 TL;DR

본 논문은 SSFSL에서 "무엇이 아닌지"를 예측하는 것이 "무엇인지"를 예측하는 것보다 쉽다는 직관을 바탕으로, **연속적 배제(Successive Exclusion)** 방식의 **MUSIC** 방법론을 제안한다. Negative Learning을 통해 신뢰도 높은 pseudo-label을 단계적으로 생성하고 이를 통해 support set을 확장함으로써, 복잡한 설계 없이도 4개의 벤치마크 데이터셋에서 SOTA 성능을 달성하였다. 이 연구는 극소량의 데이터 환경에서 Negative learning이 매우 효율적인 pseudo-labeling 전략이 될 수 있음을 시사한다.
