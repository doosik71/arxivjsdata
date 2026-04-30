# Adaptive Expert Models for Personalization in Federated Learning

Martin Isaksson, Edvin Listo Zec, Rickard Cöster, Daniel Gillblad and Šarnas Girdzijauskas (2022)

## 🧩 Problem to Solve

본 논문은 데이터가 여러 조직이나 기기에 분산되어 있어 중앙 집중화가 불가능한 환경에서의 Federated Learning (FL) 성능 저하 문제를 다룬다. 특히, 클라이언트 간의 데이터 분포가 서로 다른 Heterogeneous 및 Non-IID (non-Independent and Identically Distributed) 상황에서 기존의 전역 모델(Global Model) 방식은 최적의 성능을 내지 못하는 한계가 있다.

이러한 문제의 중요성은 현실 세계의 데이터(예: 의료 데이터, 센서 데이터)가 법적 규제(GDPR 등)나 통신 제한으로 인해 이동이 불가능하며, 지역적/개인적 특성에 따라 데이터 분포의 왜곡(Skew)이 심하게 나타난다는 점에 있다. 따라서 본 논문의 목표는 데이터의 이질성에 적응하여 각 클라이언트에게 최적화된 추론 결과를 제공하는 실용적이고 강건한 개인화(Personalization) 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 여러 개의 전역 클러스터 모델과 개별 로컬 모델을 **Mixture of Experts (MoE)** 구조로 결합하고, 클러스터 모델의 학습 과정에서 **$\epsilon$-greedy 탐색 전략**을 도입하여 개인화를 달성하는 것이다.

1.  **$\epsilon$-greedy 기반의 클러스터 모델 학습**: 기존의 Iterative Federated Clustering Algorithm (IFCA)이 일부 모델만 수렴하는 문제를 해결하기 위해, 탐색(Exploration)과 활용(Exploitation)의 균형을 맞추는 $\epsilon$-greedy 알고리즘을 도입하여 더 많은 클러스터 모델이 유의미하게 수렴하도록 유도하였다.
2.  **MoE 기반의 개인화 추론**: 수렴된 여러 개의 클러스터 모델들을 Expert로 활용하고, 여기에 로컬 모델을 추가하여 gating function이 각 모델의 가중치를 동적으로 조절하게 함으로써 개인화된 추론 성능을 극대화하였다.
3.  **Non-IID 환경에 대한 광범위한 분석**: Label distribution skew, Feature distribution skew 등 다양한 Non-IID 시나리오에서 제안 방법론의 강건성을 검증하였다.

## 📎 Related Works

논문에서는 다음과 같은 기존 연구들을 언급하며 차별점을 제시한다.

-   **FedAvg**: 가장 일반적인 FL 알고리즘이나, Non-IID 데이터 환경에서는 수렴 속도가 느리거나 성능이 매우 떨어진다.
-   **Iterative Federated Clustering Algorithm (IFCA)**: 데이터를 클러스터링하여 각 클러스터별 전역 모델을 학습하는 방식이다. 하지만 IFCA는 추론 시 단 하나의 최적 클러스터 모델만을 선택하고 나머지는 폐기하므로, 가용한 정보(다른 클러스터 모델들)를 충분히 활용하지 못한다는 한계가 있다.
-   **MoE for FL**: 전역 모델과 로컬 모델을 결합하여 가중치를 학습하는 방식이 제안된 바 있으나, 본 논문은 여기서 더 나아가 **여러 개의 전역 클러스터 모델**을 전문가(Expert)로 포함시킨다는 점에서 확장성을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조 및 파이프라인
제안된 프레임워크는 클러스터 모델을 학습하는 단계와 이를 MoE 구조에 통합하여 개인화된 추론을 수행하는 단계로 구성된다.

### 1. $\epsilon$-greedy 기반 클러스터 모델 학습
기존 IFCA는 클라이언트가 단순히 손실(Loss)이 가장 낮은 모델을 선택하게 하여 특정 모델만 빠르게 업데이트되는 '승자 독식' 현상이 발생한다. 이를 방지하기 위해 다음과 같은 선택 로직을 사용한다.

-   **탐색 (Exploration)**: $\epsilon$의 확률로 무작위 클러스터 모델 $\hat{j} \in \{1, \dots, J\}$를 선택한다.
-   **활용 (Exploitation)**: $1-\epsilon$의 확률로 로컬 학습 데이터에서 손실이 가장 낮은 모델을 선택한다.
    $$\hat{j} = \text{argmin}_{j \in J} \sum_{i \in P_k} l(x_i, y_i, w^j_g)$$

이를 통해 더 많은 수의 클러스터 모델이 서로 다른 데이터 특성에 맞게 수렴할 수 있도록 유도한다.

### 2. MoE 기반의 개인화 추론
학습된 $J$개의 전역 클러스터 모델 $f^j_g(x)$와 클라이언트 개별 로컬 모델 $f^k_l(x)$를 Expert로 구성한다. Gating model $f^k_h(x)$는 입력 $x$에 대해 각 Expert의 가중치를 출력하며, 최종 추론 결과 $\hat{y}_h$는 다음과 같이 계산된다.

$$\hat{y}_h = g^k_l f^k_l(x) + \sum_{j=0}^{J-1} g^k_j f^j_g(x)$$

여기서 $g^k_l$은 로컬 모델의 가중치, $g^k_j$는 $j$번째 클러스터 모델의 가중치이며, 이들의 합은 1이 된다.

### 3. 학습 절차 및 손실 함수
-   **손실 함수**: 모든 모델은 Negative Log-Likelihood loss를 사용하여 학습된다.
-   **최적화 알고리즘**: 
    -   로컬 모델 및 Gating 모델: **AdamW** 옵티마이저를 사용한다.
    -   클러스터 모델: 모델 평균화(Averaging) 과정에서 모멘텀 파라미터로 인한 문제를 방지하기 위해 **SGD**를 사용한다.
-   **통신 효율화**: 특정 클러스터 모델이 수렴했다고 판단되면 Early stopping을 적용하여 더 이상 해당 모델을 전송하지 않음으로써 통신 오버헤드를 줄인다.

## 📊 Results

### 실험 설정
-   **데이터셋**: CIFAR-10 (Label skew 제어), Rotated CIFAR-10 (Feature skew), FEMNIST (복합 Non-IID).
-   **비교 대상 (Baselines)**: IFCA, Local model, Ensemble model (동일 가중치 합산), Fine-tuned model.
-   **평가 지표**: Validation Accuracy 및 클라이언트 간 정확도 분산(Variance).

### 주요 결과
1.  **정확도 향상**: 특히 Pathological Non-IID 설정(데이터 분포가 극단적으로 치우친 경우)에서 IFCA 대비 최대 $29.78\%$, 단순 로컬 모델 대비 최대 $4.38\%$ 높은 정확도를 달성하였다.
2.  **강건성 (Robustness)**: IID 설정($p=0.2$)에서 하이퍼파라미터를 튜닝했음에도 불구하고, 극단적인 Non-IID 설정($p=1$)에서도 성능이 유지되거나 오히려 향상되는 강건함을 보였다.
3.  **공정성 (Fairness)**: 클라이언트 간의 성능 편차(Inter-client variance)를 크게 줄였으며, 이는 CDF(Cumulative Distribution Function) 그래프를 통해 확인되었다. IFCA보다 훨씬 많은 수의 클라이언트가 높은 정확도 구간에 분포함을 보였다.
4.  **$\epsilon$-greedy의 효과**: 탐색 전략이 없을 때보다 있을 때 더 많은 클러스터 모델이 균등하게 활용되었으며, 이것이 MoE의 성능 향상으로 이어졌다.

## 🧠 Insights & Discussion

### 강점
-   **적응형 구조**: 단일 모델이 아닌 MoE 구조를 채택함으로써, 데이터 분포가 극단적으로 다른 환경에서도 로컬 모델과 전역 클러스터 모델 사이의 최적의 균형점을 동적으로 찾을 수 있다.
-   **탐색 전략의 도입**: MAB(Multi-Armed Bandit)의 개념을 FL의 클러스터 할당에 적용하여, 전역 모델들의 수렴 가능성을 높인 점이 매우 실용적이다.

### 한계 및 비판적 해석
-   **통신 오버헤드**: IFCA는 단일 모델 혹은 마지막 레이어만 공유하는 반면, 본 제안 방법은 모든 클러스터 모델의 전체 가중치를 공유한다. 이는 통신 비용을 증가시키는 요인이 된다. 저자들은 Early stopping으로 이를 완화하려 했으나, 근본적인 해결책은 아니며 향후 연구 과제로 남겨두었다.
-   **하이퍼파라미터 의존성**: $\epsilon$ 값과 클러스터의 개수 $J$에 따라 성능 변화가 있으며, 특히 Rotated CIFAR-10과 같은 특정 케이스에서는 더 큰 $J$와 높은 $\epsilon$ 값이 필요하다는 점이 관찰되었다.

## 📌 TL;DR

본 논문은 Non-IID 데이터 환경에서 Federated Learning의 개인화를 위해 **$\epsilon$-greedy 탐색 기반의 클러스터 학습**과 **MoE(Mixture of Experts) 추론 구조**를 제안하였다. 이 방법은 여러 개의 전역 클러스터 모델과 로컬 모델을 유연하게 결합하여, 극단적인 데이터 불균형 상황에서도 기존 방식(IFCA 등)보다 훨씬 높은 정확도와 클라이언트 간 공정성을 달성하였다. 이는 향후 초개인화된 분산 학습 시스템 구축에 중요한 기여를 할 것으로 평가된다.