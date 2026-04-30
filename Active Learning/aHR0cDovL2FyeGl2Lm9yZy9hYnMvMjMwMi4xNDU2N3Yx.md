# Active Learning with Combinatorial Coverage

Sai Prathyush Katragadda, Tyler Cody, Peter Beling, Laura Freeman (2023)

## 🧩 Problem to Solve

본 논문은 능동 학습(Active Learning, AL)에서 발생하는 **모델 의존성(Model Dependency)** 문제를 해결하고자 한다. 기존의 능동 학습 방법들은 특정 모델이 어떤 데이터를 필요로 하는지에 집중하는 모델 중심적(Model-centric) 접근 방식을 취한다. 이로 인해 다음과 같은 두 가지 핵심적인 문제가 발생한다.

1.  **데이터 전이성(Data Transferability) 부족**: 특정 모델을 위해 샘플링된 데이터가 다른 모델을 학습시킬 때는 효과적이지 않을 수 있다. 실제 배포 환경에서는 모델의 종류가 지속적으로 업데이트되므로, 한 번 레이블링된 데이터가 다양한 모델에 범용적으로 사용될 수 있어야 하지만 기존 방식으로는 이것이 어렵다.
2.  **샘플링 편향(Sampling Bias)**: 모델 의존적인 샘플링은 특성 공간(Feature Space)의 특정 영역에만 치우쳐 데이터를 수집하게 만들어, 전체 데이터 분포를 충분히 반영하지 못하는 편향 문제를 야기한다.

따라서 본 연구의 목표는 모델이 아닌 데이터 자체의 특성에 집중하는 **데이터 중심적(Data-centric)** 능동 학습 방법을 제안하여, 데이터 전이성을 높이고 샘플링 편향을 줄이는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 소프트웨어 테스트 분야에서 사용되는 **조합 커버리지(Combinatorial Coverage, CC)** 개념을 능동 학습에 도입하는 것이다. 

- **데이터 중심적 샘플링**: 특정 모델의 불확실성에만 의존하지 않고, 데이터셋 내의 다양한 특성 조합(Interaction)이 얼마나 포함되었는지를 측정하여 샘플을 선택한다.
- **범용적 데이터셋 구축**: 조합 커버리지를 통해 특성 공간을 골고루 커버하는 데이터를 샘플링함으로써, 이후 어떤 모델을 적용하더라도 효과적으로 작동할 수 있는 전이 가능한(Transferable) 레이블 데이터셋을 구축하고자 한다.
- **세 가지 새로운 샘플링 전략 제안**: Coverage Density Sampling (CDS), Informative Coverage Density Sampling (ICDS), 그리고 Uncertainty Sampling Weighted by Coverage Density (USWCD)를 제안한다.

## 📎 Related Works

### 기존 능동 학습 연구
능동 학습은 크게 Membership Query Synthesis, Stream-based Selective Sampling, Pool-based Sampling으로 나뉜다. 본 논문은 전체 풀에서 샘플을 뽑는 **Pool-based Sampling**에 집중한다. 대표적인 전략으로는 모델의 불확실성이 높은 데이터를 뽑는 Uncertainty Sampling, 여러 모델의 의견 차이를 이용하는 Query by Committee (QBC), 데이터의 대표성을 고려하는 Information Density 등이 있다. 그러나 이들은 모두 모델의 상태에 의존적이라는 한계가 있다.

### 조합 상호작용 테스트 (Combinatorial Interaction Testing, CIT)
CIT는 시스템의 모든 가능한 $t$-way 상호작용(특성 값들의 조합)을 보장하는 테스트 세트를 설계하는 기법이다. 최근 딥러닝 모델의 테스트나 설명 가능한 AI(XAI) 분야에 적용되기 시작했다. 특히 **Set Difference Combinatorial Coverage (SDCC)**는 두 데이터셋 간의 상호작용 차이를 측정하는 지표로, 데이터 커버리지가 높을수록 모델의 성능이 향상된다는 연구 결과가 보고된 바 있다.

## 🛠️ Methodology

### 1. 기본 개념 및 SDCC
본 논문은 레이블링된 데이터셋 $D^L$과 레이블링되지 않은 데이터셋 $D^U$ 사이의 **$t$-way Set Difference Combinatorial Coverage (SDCC)**를 활용한다. 

$$\text{SDCC}_t(D^U, D^L) = \frac{|D^U_t \setminus D^L_t|}{|D^U_t|}$$

여기서 $D^U_t \setminus D^L_t$는 $D^U$에는 존재하지만 $D^L$에는 존재하지 않는 $t$-way 상호작용 조합의 집합을 의미한다.

### 2. 커버리지 밀도 (Coverage Density)
특정 데이터 포인트 $i \in D^U$가 얼마나 많은 '누락된 상호작용'을 가지고 있는지를 측정하여 우선순위를 결정한다. 상호작용 수준 $t$가 낮을수록 더 많은 클래스와 연관될 가능성이 높으므로, $t=1$부터 $6$까지 $\frac{1}{t}$의 가중치를 부여하여 합산한다.

데이터 포인트 $i$의 레벨 $t$에서의 커버리지 밀도 $c_i^t$는 다음과 같다.
$$c_i^t = \sum_{j \in D^L} \frac{1}{t} \quad \text{if } i_t \notin j_t$$
최종 커버리지 밀도 $c_i$는 모든 $t \in \{1, \dots, T\}$에 대한 $\sum c_i^t$로 정의된다.

### 3. 제안하는 세 가지 샘플링 방법

**① Coverage Density Sampling (CDS)**
가장 단순하게 커버리지 밀도 $c_i$가 높은 포인트들을 예산 $b$ 내에서 선택한다.
$$\arg \max_i \sum_i c_i x_i \quad \text{s.t.} \sum_i x_i \leq b$$

**② Informative Coverage Density Sampling (ICDS)**
CDS가 이상치(Outlier)에 취약할 수 있다는 점을 보완하기 위해, 커버리지 밀도에 코사인 유사도(Cosine Similarity) 기반의 정보량을 곱하여 가중치를 둔다.
$$I(x_i) = c_i \frac{1}{U} \sum_{x \in X} \frac{x \cdot x_i}{\|x\| \|x_i\|}$$
여기서 $U$는 레이블되지 않은 집합의 크기이며, 유사도가 높은(대표성이 있는) 데이터 중 커버리지가 높은 것을 선택한다.

**③ Uncertainty Sampling Weighted by Coverage Density (USWCD)**
기존의 모델 기반 불확실성(Entropy)과 데이터 중심적 커버리지 밀도를 결합한 방식이다.
$$I(x_i) = H(x_i) c_i$$
$$H(x_i) = -\sum_{y \in Y} p(y_i) \log(p(y_i))$$
모델이 불확실해하면서 동시에 데이터 공간의 커버리지를 높일 수 있는 포인트를 우선적으로 선택한다.

## 📊 Results

### 실험 설계
- **데이터셋**: UCI Machine Learning Repository의 6개 데이터셋 (Tic-Tac-Toe, Balance Scale, Car Evaluation, Chess, Nursery, Monk).
- **지표**: F1-Score, 학습 곡선 아래 면적(AUC), 샘플링 편향(Sampling Bias).
- **비교 대상**: Random Sampling, Uncertainty Sampling, QBC, Information Density.
- **실험 구성**:
    - **Experiment 1**: Random Forest(RF)로 샘플링하고 동일한 RF로 성능 측정.
    - **Experiment 2**: RF로 샘플링한 데이터를 Decision Tree(DT)와 SVM에 적용하여 전이성 측정.

### 주요 결과
1.  **단일 모델 성능 (Exp 1)**: QBC가 가장 좋은 성능을 보이는 경우가 많았으나, 제안 방법 중 **USWCD**가 그에 근접한 경쟁력 있는 성능을 보여주었다.
2.  **모델 전이 성능 (Exp 2)**: 데이터가 새로운 모델(DT, SVM)로 전이되었을 때, **USWCD가 다른 모든 벤치마크 방법보다 월등한 성능**을 보였다. 특히 Random Sampling보다 성능이 떨어지는 경우가 없었으며, 전이 가능한 데이터셋 구축에 매우 효과적임이 입증되었다.
3.  **샘플링 편향**: USWCD의 중앙값(Median) 샘플링 편향이 다른 방법들보다 낮게 나타나, 데이터 공간을 더 균일하게 샘플링함을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 기여
본 논문은 능동 학습의 고질적인 문제인 모델 의존성을 해결하기 위해 소프트웨어 공학의 '조합 커버리지' 개념을 성공적으로 도입하였다. 특히 **USWCD**는 모델의 불확실성과 데이터의 다양성을 모두 고려함으로써, 현재 모델의 성능 유지와 미래 모델로의 전이 가능성이라는 두 마리 토끼를 모두 잡은 전략이라고 평가할 수 있다.

### 한계 및 비판적 해석
- **샘플링 편향 개선의 폭**: 결과적으로 편향이 낮아지긴 했으나, 바이올린 플롯의 분포를 보면 기존 방법들과의 차이가 극명하게 크지는 않다. 저자 또한 클래스 다양성이 더 높은 데이터셋에서의 추가 검증이 필요함을 언급하고 있다.
- **계산 복잡도**: $t$-way 상호작용을 모두 계산하는 과정이 특성 수가 많아질 경우 계산 비용이 급격히 증가할 수 있다. 본 논문에서는 $t=6$까지 제한하여 이를 완화하였으나, 고차원 데이터에 대한 확장성 논의가 부족하다.

### 향후 연구 방향
- 연속형 데이터(Continuous data)를 위한 이산화(Discretization) 방법 적용.
- 매 반복마다 재학습이 필요한 Uncertainty Sampling과 달리, 커버리지 밀도는 모델 재학습 없이도 샘플링이 가능하므로 딥러닝과 같은 무거운 모델에 적용 시 큰 효율성을 기대할 수 있다.

## 📌 TL;DR

본 논문은 모델에 지나치게 의존적인 기존 능동 학습의 한계를 극복하기 위해, **조합 커버리지(Combinatorial Coverage)** 기반의 데이터 중심적 샘플링 방법론을 제안한다. 특히 모델의 불확실성과 데이터 커버리지를 결합한 **USWCD** 방식은 단일 모델에서의 성능을 유지하면서도, **샘플링된 데이터를 다른 모델로 전이했을 때 압도적인 성능 향상**을 보여준다. 이는 실제 머신러닝 배포 환경에서 모델이 변경되더라도 레이블링된 데이터를 재사용할 수 있게 하여 데이터 구축 비용을 획기적으로 줄일 수 있는 가능성을 제시한다.