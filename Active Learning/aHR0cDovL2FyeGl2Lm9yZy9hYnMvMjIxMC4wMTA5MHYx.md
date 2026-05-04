# Nonstationary Data Stream Classification with Online Active Learning and Siamese Neural Networks

Kleanthis Malialis, Christos G. Panayiotou, Marios M. Polycarpou (2022)

## 🧩 Problem to Solve

본 논문은 실시간으로 데이터가 유입되는 Data Stream 환경에서 예측 모델을 학습시킬 때 발생하는 네 가지 주요 난제를 해결하고자 한다. 첫째, 모든 유입 데이터에 대해 정답 라벨을 얻는 것은 비용이 많이 들거나 불가능한 경우가 많아 **Limited Labelled Data** 문제가 발생한다. 둘째, 시간이 흐름에 따라 데이터의 분포가 변화하는 **Nonstationary Data (Concept Drift)** 현상이 나타나 모델의 성능을 저하시킨다. 셋째, 특정 클래스의 데이터가 극도로 적은 **Class Imbalance** 상황에서는 소수 클래스에 대한 식별 능력이 현저히 떨어진다. 마지막으로, 실시간 시스템의 특성상 데이터를 저장하기 위한 **Memory** 자원이 제한적이라는 점이다.

결과적으로 본 연구의 목표는 실시간 응답이 가능하면서도, 제한된 라벨 데이터만으로 학습하고, Concept Drift와 심각한 Class Imbalance 상황에서도 강건하게 동작하는 온라인 학습 알고리즘을 설계하는 것이다.

## ✨ Key Contributions

본 논문은 위의 문제들을 해결하기 위해 **Online Active Learning**, **Siamese Neural Networks**, 그리고 **Multi-queue Memory**를 유기적으로 결합한 **ActiSiamese** 알고리즘을 제안한다.

핵심적인 설계 아이디어는 기존의 Active Learning이 주로 불확실성(Uncertainty)에 기반하여 라벨을 요청했던 것과 달리, **Latent Space(잠재 공간)에서의 유사도(Similarity) 기반 밀도 샘플링 전략**을 도입한 것이다. Siamese Network를 통해 입력 데이터를 저차원의 인코딩 공간으로 매핑하고, 이 공간에서의 유사도를 측정함으로써 더 정보 가치가 높은 샘플을 효율적으로 선택하여 라벨을 요청한다. 또한, 클래스별로 독립된 큐(Queue)를 유지하는 메모리 구조를 통해 데이터 불균형과 개념 변화에 동시에 대응한다.

## 📎 Related Works

기존의 온라인 학습 연구는 크게 세 가지 방향으로 진행되었다. Concept Drift 대응을 위해 Sliding Window 기반의 Memory-based 방법, 통계적 검정을 통한 Change Detection 방법, 그리고 여러 분류기를 결합하는 Ensembling 방법이 사용되었다. Class Imbalance 문제의 경우, 알고리즘 수준에서 손실 함수를 수정하거나 데이터 수준에서 Resampling을 수행하는 방식이 제안되었으며, 특히 클래스별 큐를 사용하는 Queue-Based Resampling 방식이 제시된 바 있다.

Active Learning 분야에서는 결정 경계 근처의 샘플을 선택하는 Uncertainty Sampling이 주를 이루었으나, 이는 임계값(Threshold) 설정에 민감하며 데이터 분포의 전체적인 특성을 반영하지 못한다는 한계가 있다. Density Sampling은 입력 공간의 밀도를 고려하여 샘플을 선택하지만, 고차원 데이터에서는 '차원의 저주'로 인해 거리 측정의 신뢰도가 떨어진다는 문제가 있다. 본 논문은 이러한 한계를 극복하기 위해 입력 공간이 아닌 학습된 Latent Space에서 유사도를 측정하는 방식을 제안하여 차별점을 둔다.

## 🛠️ Methodology

### 전체 시스템 구조

ActiSiamese의 전체 파이프라인은 **Multi-queue Memory $\rightarrow$ Classifier (or Ensemble) $\rightarrow$ Online Active Learning Strategy** 순으로 구성된다. 새로운 인스턴스 $x_t$가 유입되면 메모리의 데이터를 참조하여 클래스를 예측하고, Active Learning 전략에 따라 라벨 요청 여부를 결정하며, 라벨이 제공될 경우에만 모델을 점진적으로 학습시킨다.

### 주요 구성 요소 및 상세 설명

**1. Multi-queue Memory**
클래스 개수 $K$만큼의 FIFO(First-In-First-Out) 큐를 유지한다. 각 큐의 용량은 $L$로 고정되며, $\text{q}_c$는 클래스 $c$에 속하는 최신 샘플들을 저장한다. 이는 데이터 불균형 상황에서도 각 클래스의 대표 샘플을 일정 수 유지함으로써 Oversampling 효과를 내며, 오래된 샘플이 자연스럽게 제거되어 Concept Drift에 대응하게 한다.

**2. Siamese Neural Networks**
두 개의 동일한 네트워크(Twins)가 입력 $x$를 Latent Space의 인코딩 $e(x)$로 매핑한다. 두 샘플 $(x_i, x_j)$ 사이의 거리는 다음과 같이 정의된다.
$$d(x_i, x_j) = |e(x_i) - e(x_j)|$$
이 거리 값은 Sigmoid 함수를 통해 두 샘플이 동일 클래스일 확률 $\hat{p}(x_i, x_j) \in [0, 1]$로 변환된다.

**3. 클래스 예측 및 Active Learning 전략**

- **예측(Prediction):** 유입된 $x_t$와 각 클래스 큐에 저장된 샘플들 간의 평균 유사도를 계산하여 가장 높은 값을 가진 클래스를 선택한다.
$$\hat{y}_t = \text{argmax}_{c \in \{1, \dots, K\}} \frac{1}{L} \sum_{i=1}^L \hat{p}(x_t, x_{c,i})$$
- **RVSS (Randomised Variable Similarity Sampling):** 예측된 클래스 $c$ 내에서 가장 유사한 샘플과의 유사도 $v = \max_{i} \hat{p}(x_t, x_{c,i})$를 계산한다. 이 $v$ 값을 가변 임계값 $\theta$와 비교하여 라벨을 요청하며, $\theta$는 정규분포 $\eta \sim \mathcal{N}(1, \delta)$를 이용하여 동적으로 조정됨으로써 탐색 효율을 높인다.

**4. 학습 절차 및 손실 함수**
라벨이 제공되면 큐에 저장된 샘플들로 양성 쌍($Q_{pos}$: 동일 클래스)과 음성 쌍($Q_{neg}$: 서로 다른 클래스)을 생성한다. 이때 두 집합의 크기를 동일하게 맞춰 불균형을 방지한다. 학습 목표는 Binary Cross-Entropy 손실 함수 $J_t$를 최소화하는 것이다.
$$J_t = \frac{1}{|Q_{train}|} \sum_{(x_i, x_j) \in Q_{train}} l(y_{i,j}, \hat{p}(x_i, x_j))$$
학습은 Incremental Stochastic Gradient Descent를 통해 가중치를 업데이트하는 방식으로 이루어진다.

**5. ActiSiamese-WM (Ensemble)**
복수의 ActiSiamese 분류기를 사용하며, Weighted Majority (WM) 알고리즘을 통해 각 분류기의 가중치를 관리한다. 예측 시에는 가중 평균 확률을 사용하며, 오답을 낸 분류기의 가중치는 지수적으로 감소시킨다.

## 📊 Results

### 실험 설정

- **데이터셋:** 합성 데이터(Sea, Circles, Blobs 등)와 실제 데이터(Gestures, MNIST, Forest, Insects 등)를 모두 사용하였다. 특히 Extreme Imbalance (0.1%)와 Abrupt/Recurrent Drift 상황을 포함하여 가혹한 환경을 구축하였다.
- **비교 대상:** RVUS (One-pass, Uncertainty-based), ActiQ (Memory-based, Uncertainty-based), RVSS (Memory-based, Input-space Similarity-based) 및 각각의 Ensemble 버전.
- **지표:** 클래스 불균형에 강건한 **G-mean**을 주요 성능 지표로 사용하였으며, Prequential evaluation 방식을 적용하였다.

### 주요 결과

1. **학습 속도 및 효율성:** ActiSiamese는 다른 메모리 기반 방법(ActiQ, RVSS)보다 초기 학습 속도가 현저히 빨랐다. 이는 Siamese Network의 Few-shot learning 능력이 효과적으로 작용했음을 시사한다.
2. **클래스 불균형 대응:** 극심한 불균형(0.1%) 상황에서 ActiSiamese는 다른 모든 베이스라인을 압도하는 성능을 보였다. 이는 클래스별 큐 유지와 Balanced Pair 생성 전략의 효과이다.
3. **Concept Drift 강건성:** Multi-queue Memory를 사용한 방법들은 One-pass 학습자인 RVUS보다 Drift 상황에서 훨씬 빠르게 회복하였다.
4. **Latent Space의 중요성:** 입력 공간에서 유사도를 측정하는 RVSS보다 Latent Space에서 측정하는 ActiSiamese의 성능이 우수하였으며, 이는 고차원 데이터에서 인코딩된 특징 공간이 더 유의미한 유사도 정보를 제공함을 증명한다.
5. **실제 데이터 성능:** Gestures, Keystroke 등의 데이터셋에서 유의미한 성능 향상을 보였으며, 특히 제한된 라벨 예산($B=1\%$) 하에서도 강건한 성능을 유지하였다.

## 🧠 Insights & Discussion

### 강점 및 해석

ActiSiamese의 성공 요인은 세 가지 메커니즘의 시너지에 있다. 첫째, Siamese Network를 통해 고차원 입력을 변별력 있는 저차원 공간으로 매핑하여 데이터 효율성을 극대화했다. 둘째, 클래스별 독립 큐를 통해 소수 클래스 샘플을 강제로 유지함으로써 데이터 불균형 문제를 구조적으로 해결했다. 셋째, Latent Space 기반의 Active Learning 전략을 통해 라벨링 예산을 매우 효율적으로 사용하였다.

### 한계 및 비판적 논의

본 모델의 가장 큰 한계는 **One-pass learner가 아니라는 점**이다. 즉, 데이터를 한 번 보고 바로 버리는 것이 아니라 고정된 크기의 메모리에 저장하여 재사용해야 한다. 비록 메모리 크기가 작고 고정되어 있어 실용적이지만, 진정한 의미의 Single-pass 학습을 원하는 환경에서는 제약이 될 수 있다. 또한, 현재의 Drift 대응 방식은 점진적 학습을 통한 '암시적' 대응에 의존하고 있어, 급격한 변화가 일어날 때 명시적인 Drift Detection 메커니즘이 결합된다면 더 빠른 반응성을 가질 수 있을 것으로 판단된다.

## 📌 TL;DR

본 논문은 데이터 스트림의 세 가지 난제인 **제한된 라벨, Concept Drift, Class Imbalance**를 동시에 해결하기 위해 **Siamese Neural Networks와 Multi-queue Memory를 결합한 ActiSiamese**를 제안한다. 특히 잠재 공간에서의 유사도를 이용한 새로운 Active Learning 전략을 통해 매우 적은 라벨만으로도 빠른 학습과 높은 정확도를 달성하였다. 이 연구는 실시간 모니터링, 보안 시스템 등 라벨 획득 비용이 높고 데이터 분포가 계속 변하는 실제 산업 현장의 온라인 분류 문제에 적용될 가능성이 매우 높다.
