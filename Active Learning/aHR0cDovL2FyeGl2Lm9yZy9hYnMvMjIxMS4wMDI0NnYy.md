# Batch Active Learning from the Perspective of Sparse Approximation

Maohao Shen, Bowen Jiang, Jacky Y. Zhang, Oluwasanmi Koyejo (2022)

## 🧩 Problem to Solve

본 논문은 딥러닝 모델 학습 시 발생하는 고비용의 데이터 레이블링 문제를 해결하기 위한 Batch Active Learning(배치 능동 학습) 프레임워크를 제안한다. Active Learning의 핵심은 제한된 레이블링 예산 내에서 모델 성능을 최대화하기 위해 가장 정보량이 많은 데이터 샘플을 선택하는 Query Strategy를 설계하는 것이다.

기존의 쿼리 전략은 크게 두 가지 방향으로 나뉜다. 첫째, Uncertainty-based 접근법은 모델이 불확실하다고 판단하는 샘플을 선택하지만, 배치 단위로 선택할 때 서로 유사하고 중복된 샘플들이 함께 선택되는 경향이 있다. 둘째, Representation-based 접근법은 전체 데이터셋을 잘 대표하는 부분 집합을 찾으려 하지만, 연산 비용이 매우 높고 배치 크기에 민감하다는 단점이 있다.

따라서 본 연구의 목표는 Uncertainty(불확실성)와 Representation(대표성) 사이의 트레이드오프를 원칙적으로 균형 있게 조절하면서, 베이지안(Bayesian) 및 비베이지안(non-Bayesian) 신경망 모두에 적용 가능한 효율적인 배치 선택 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Batch Active Learning 문제를 Sparse Approximation(희소 근사) 관점에서 재정의하는 것이다. 연구진은 레이블이 없는 전체 데이터 풀의 손실 함수(Loss Function)를 가장 잘 근사하는 희소한 가중치 벡터 $w$를 찾는 문제로 정식화하였다.

주요 기여 사항은 다음과 같다.
1. **희소 근사 기반 프레임워크 제안**: 배치 선택 문제를 전체 데이터셋의 손실 함수와 선택된 부분 집합의 손실 함수 사이의 차이를 최소화하는 문제로 정의하여, 베이지안과 비베이지안 설정 모두에서 작동하는 유연한 구조를 설계하였다.
2. **Bias-Variance 분해를 통한 균형 설계**: 손실 함수의 근사 오차를 Bias(대표성)와 Variance(불확실성)로 분해하여, 두 요소 간의 트레이드오프를 수학적으로 정교하게 조절할 수 있는 상한선(Upper Bound)을 도출하였다.
3. **효율적인 최적화 알고리즘 제공**: 도출된 불연속 최적화 문제를 해결하기 위해 Greedy 알고리즘과 Proximal Iterative Hard Thresholding(IHT) 알고리즘을 제안하여, 기존 하이브리드 방식(예: BADGE)보다 훨씬 낮은 연산 복잡도로 유사하거나 더 우수한 성능을 달성하였다.

## 📎 Related Works

기존의 Active Learning 연구들은 주로 예측의 불확실성(Entropy, Mutual Information)이나 데이터의 분포(KCenter, Core-set)에 의존하였다. 특히 최근의 하이브리드 방식인 BADGE는 그라디언트 임베딩 공간에서 다양성을 확보하려 하지만, 배치 크기가 커질수록 연산 시간이 급격히 증가하는 문제가 있다.

또한 Bayesian Coreset 연구들이 존재하지만, 이들은 주로 베이지안 모델에 국한되어 설계되었다. 본 논문은 손실 함수 공간에서의 세미노름(semi-norm)을 도입함으로써, 모델의 종류와 상관없이 손실 함수 자체의 근사 성능을 측정하는 보다 일반적인 접근 방식을 취한다는 점에서 기존 연구와 차별화된다.

## 🛠️ Methodology

### 1. 문제 정의: Sparse Approximation 관점의 AL
연구진은 레이블이 없는 데이터셋 $D_u$에 대해, 각 데이터에 가중치 $w_j$를 부여한 가중 손실 함수 $\tilde{L}_w$가 전체 데이터셋의 손실 함수 $\tilde{L}$을 근사하도록 설계하였다. 이를 위해 시프트 불변 세미노름(shift-invariant seminorm) $q(\cdot)$를 정의하고 다음과 같은 최적화 문제를 구성하였다.
$$\arg \min_{w \in \mathbb{R}^{n_u}_+} \mathbb{E}_P[q(\tilde{L} - \tilde{L}_w)] \quad \text{s.t. } \|w\|_0 = b$$
여기서 $\|w\|_0 = b$는 정확히 $b$개의 샘플만 선택해야 함을 의미한다.

### 2. Bias와 Variance의 분해
위의 기대값 계산은 연산량이 너무 많으므로, 연구진은 삼각 부등식을 통해 이를 Bias(근사 편향)와 Variance(분산)의 합으로 상한선을 도출하였다.
- **Bias**: 선택된 부분 집합이 전체 데이터의 기대 손실을 얼마나 잘 대표하는가 하는 척도이다.
- **Variance**: 레이블이 부여되지 않은 샘플들이 가지는 개별적인 불확실성이다.

### 3. 구체적인 최적화 문제 (Problem 2)
베이지안 설정(후험 분포 샘플링 이용)과 비베이지안 설정(로컬 윈도우 내 그라디언트 근사 이용)을 통합하여, 최종적으로 다음과 같은 유한 차원 최적화 문제를 해결한다.
$$\arg \min_{w \in \mathbb{R}^{n_u}_+} \|v - \Phi w\|_2^2 - \alpha \sum_{x_j \in D_u} 1(w_j > 0) \cdot \sigma_j^2 + \beta \|w-1\|_2^2 \quad \text{s.t. } \|w\|_0 = b$$
- $\|v - \Phi w\|_2^2$: Representation을 측정하는 Bias 항이다.
- $\alpha \sum 1(w_j > 0) \cdot \sigma_j^2$: Uncertainty를 측정하는 Variance 항이며, $\alpha$를 통해 비중을 조절한다.
- $\beta \|w-1\|_2^2$: 수치적 안정성을 위한 정규화 항이다.
- $v, \Phi, \sigma_j$는 각각 전체 데이터의 평균 그라디언트(또는 손실 샘플), 개별 데이터의 그라디언트, 그리고 그라디언트의 분산을 나타낸다.

### 4. 최적화 알고리즘
본 문제는 $\|w\|_0$ 제약 조건과 불연속 함수 $f_2$ 때문에 비볼록(non-convex) 최적화 문제이다. 이를 해결하기 위해 두 가지 알고리즘을 제안한다.
- **Greedy Algorithm**: 매 단계마다 목적 함수를 가장 많이 감소시키는 인덱스 $j$를 순차적으로 선택한다.
- **Proximal-IHT Algorithm**: 그라디언트 하강법으로 업데이트한 후, Hard Thresholding 연산과 근사 연산자(Proximal Operator)를 통해 희소성 제약을 강제하며 반복적으로 최적화한다.

## 📊 Results

### 실험 설정
- **데이터셋**: Fashion MNIST, CIFAR-10, CIFAR-100, MNIST, SVHN.
- **모델**: ResNet-18 (Bayesian), LeNet-5, VGG-16 (General).
- **비교 대상**: Random, BALD, Batch BALD, Bayesian Coreset, Entropy, KCenter, BADGE.
- **측정 지표**: 테스트 정확도의 학습 곡선 및 AUC(Area Under Curve) 스코어, 쿼리 획득 시간(Runtime).

### 주요 결과
1. **성능**: 베이지안 및 일반 CNN 설정 모두에서 제안 방법(Ours-Greedy, Ours-IHT)이 대부분의 베이스라인보다 우수하거나 대등한 성능을 보였다. 특히 베이지안 설정에서는 Bayesian Coreset보다 높은 AUC를 기록하였다.
2. **효율성**: 가장 강력한 경쟁 모델인 BADGE와 비교했을 때, 정확도는 비슷하거나 높으면서 연산 시간은 획기적으로 단축되었다. 예를 들어 SVHN 데이터셋(VGG-16)에서 BADGE가 약 732초가 걸린 반면, 제안 방법(Greedy)은 약 201초 만에 쿼리를 완료하였다.
3. **시간 복잡도**: 제안 알고리즘의 시간 복잡도는 $O(nb)$(Greedy) 및 $O(n \log b)$(IHT)로, BADGE의 $O(nb^2)$보다 훨씬 효율적임을 입증하였다.

## 🧠 Insights & Discussion

본 논문은 Active Learning의 고전적인 딜레마인 '불확실성'과 '대표성'의 충돌을 '희소 근사'라는 수학적 틀 안에서 통합하여 해결하였다는 점에서 강점이 있다. 특히, 모델의 내부 구조에 의존하지 않고 손실 함수의 근사라는 관점에서 접근했기에 베이지안과 비베이지안 모델 모두에 적용 가능하다는 범용성을 확보하였다.

다만, 성능이 하이퍼파라미터 $\alpha$(불확실성 가중치)와 $\beta$(정규화 가중치)에 의존한다는 점이 한계로 지적될 수 있다. 논문의 Ablation Study에 따르면 데이터셋마다(예: CIFAR-10) Variance 항이 더 중요하게 작용하는 경우가 있어, 최적의 $\alpha$ 값을 찾는 과정이 필요하다. 또한, 실제 구현에서 사용된 근사 기법들이 이론적 상한선과 얼마나 밀접하게 맞닿아 있는지에 대한 추가적인 분석이 필요할 것으로 보인다.

## 📌 TL;DR

본 연구는 Batch Active Learning을 **Sparse Approximation** 문제로 정식화하여, 전체 데이터셋의 손실 함수를 가장 잘 근사하는 부분 집합을 효율적으로 찾는 프레임워크를 제안하였다. 이 방법은 **Bias-Variance 분해**를 통해 대표성과 불확실성을 동시에 고려하며, **Greedy 및 Proximal-IHT 알고리즘**을 통해 기존 하이브리드 방식(BADGE 등)보다 훨씬 빠른 속도로 고성능의 데이터 샘플을 선택할 수 있다. 향후 대규모 데이터셋에서 레이블링 비용을 획기적으로 줄여야 하는 실제 딥러닝 학습 파이프라인에 적용될 가능성이 매우 높다.