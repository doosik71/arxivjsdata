# Meta-learning with an Adaptive Task Scheduler

Huaxiu Yao, Yu Wang, Ying Wei, Peilin Zhao, Mehrdad Mahdavi, Defu Lian, Chelsea Finn (2021)

## 🧩 Problem to Solve

본 논문은 메타학습(Meta-learning) 과정에서 메타-트레이닝 태스크(meta-training tasks)를 샘플링하는 기존의 방식이 가진 한계점을 해결하고자 한다.

대부분의 기존 메타학습 알고리즘은 모든 태스크가 동일하게 중요하다고 가정하고, 균등한 확률(uniform probability)로 태스크를 무작위 샘플링하여 학습한다. 그러나 실제 환경에서는 다음과 같은 두 가지 주요 문제가 발생한다.

1. **노이즈 섞인 태스크(Noisy Tasks):** 일부 태스크는 잘못된 레이블링이나 측정 오류로 인해 노이즈를 포함하고 있으며, 이러한 태스크가 학습에 포함될 경우 메타-모델(meta-model)의 성능을 저하시키고 오염시킬 수 있다.
2. **불균형한 태스크 분포(Imbalanced Task Distribution):** 메타-트레이닝 태스크의 수가 제한적인 경우, 특정 클래스의 태스크가 다수를 차지하여 모델이 특정 방향으로 편향될 수 있다.

따라서 본 논문의 목표는 메타-모델의 학습 진행 상태에 따라 어떤 태스크를 학습에 사용할지 동적으로 결정하는 **Adaptive Task Scheduler (ATS)**를 설계하여, 모델의 일반화 능력을 극대화하고 메타-오버피팅(meta-overfitting)을 방지하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 메타-모델과 신경망 기반의 스케줄러를 함께 학습시키는 **Bi-level Optimization** 구조를 도입하는 것이다. 단순히 사람이 정의한 규칙(heuristic)에 따라 태스크를 선택하는 것이 아니라, 신경망 스케줄러가 메타-모델의 상태를 입력받아 각 태스크의 샘플링 확률을 예측하도록 한다.

특히, 태스크의 난이도와 중요도를 판단하기 위해 다음의 두 가지 핵심 요소(factors)를 스케줄러의 입력으로 사용한다.

1. **학습 결과 측면 (Learning Outcome):** 쿼리 세트(query set)에서의 손실 값(loss)을 통해 태스크의 현재 난이도를 측정한다.
2. **학습 과정 측면 (Learning Process):** 서포트 세트(support set)와 쿼리 세트의 그래디언트 유사도(gradient similarity)를 측정하여, 서포트 세트에서의 학습이 쿼리 세트로 얼마나 잘 일반화되는지를 파악한다.

## 📎 Related Works

메타학습 분야에서 태스크 샘플링 전략에 대한 초기 연구들은 주로 고정된 규칙을 사용하였다.

- **기존 접근 방식:** 클래스 선택 전략을 조정하거나, self-paced regularizer를 사용하고, 혹은 정보량(information amount)에 기반해 태스크를 랭킹화하는 방식들이 제안되었다.
- **한계점:** 이러한 방식들은 수동으로 정의된(manually defined) 고정된 스케줄러이므로, 메타-모델의 복잡한 학습 역학(learning dynamics)에 유연하게 대응하지 못하며 일반화 능력을 명시적으로 최적화하지 않는다.
- **차별점:** ATS는 신경망 기반의 스케줄러를 통해 학습 과정 중에 동적으로 샘플링 확률을 조정하며, 검증 세트(validation set)의 성능을 보상(reward)으로 사용하여 일반화 능력을 직접적으로 최적화한다.

## 🛠️ Methodology

### 1. 전체 파이프라인 및 시스템 구조

ATS는 다음과 같은 순환 구조로 동작한다.

1. **특성 추출:** 모든 후보 태스크에 대해 메타-모델 기반의 두 가지 요소(Loss, Gradient Similarity)를 계산한다.
2. **태스크 샘플링:** 신경망 스케줄러 $g$가 위 요소들을 입력받아 샘플링 확률 $w_i^{(k)}$를 예측하고, 이에 따라 $B$개의 태스크를 샘플링한다.
3. **메타-모델 업데이트:** 샘플링된 태스크를 사용하여 메타-모델 $\theta_0$를 업데이트한다.
4. **스케줄러 최적화:** 업데이트된 모델을 검증 세트에 적용하여 얻은 성능(보상)을 바탕으로 신경망 스케줄러 $\phi$를 업데이트한다.

### 2. 주요 구성 요소 및 방정식

#### A. 샘플링 확률 결정

태스크 $T_i$의 샘플링 확률 $w_i^{(k)}$는 다음과 같이 정의된다.
$$w_i^{(k)} = g(L(D^q_i; \theta_i^{(k)}), \langle \nabla_{\theta_0^{(k)}} L(D^s_i; \theta_0^{(k)}), \nabla_{\theta_0^{(k)}} L(D^q_i; \theta_0^{(k)}) \rangle ; \phi^{(k)})$$
여기서 $L(D^q_i; \theta_i^{(k)})$는 쿼리 세트의 손실이며, $\langle \cdot, \cdot \rangle$는 서포트 세트와 쿼리 세트 그래디언트 간의 내적(inner product)으로 일반화 격차(generalization gap)를 의미한다.

#### B. Bi-level Optimization

ATS는 메타-모델 $\theta_0$와 스케줄러 $\phi$를 동시에 최적화하는 계층적 구조를 가진다.
$$\min_{\phi} \frac{1}{N_v} \sum_{v=1}^{N_v} L_{val}(T_v; \theta_0^*(\phi))$$
$$\text{s.t. } \theta_0^*(\phi) = \arg \min_{\theta_0} \frac{1}{B} \sum_{i=1}^{B} L_{tr}(T_i; \theta_0, \phi)$$
즉, 하위 루프에서는 샘플링된 태스크로 메타-모델을 최적화하고, 상위 루프에서는 검증 세트의 손실을 최소화하도록 스케줄러를 최적화한다.

#### C. 신경망 스케줄러 학습 (REINFORCE)

샘플링 과정은 미분 불가능하므로, 강화학습의 **REINFORCE** 알고리즘을 사용하여 $\phi$를 업데이트한다.
$$\phi^{(k+1)} \leftarrow \phi^{(k)} - \gamma \nabla_{\phi^{(k)}} \log P(W^{(k)}; \phi^{(k)}) \left( \frac{1}{N_v} \sum_{i=1}^{N_v} R_i^{(k)} - b \right)$$
여기서 $R_i^{(k)}$는 검증 태스크의 정확도(보상)이며, $b$는 그래디언트 분산을 줄이기 위한 baseline(이동 평균 정확도)이다.

### 3. 이론적 분석

논문은 Proposition 1과 2를 통해 스케줄러의 효과를 수학적으로 증명한다.

- **Proposition 1:** 메타-학습 손실은 샘플링 확률 $w$가 손실 $L$과는 음의 상관관계를, 그래디언트 유사도 $\nabla$와는 양의 상관관계를 가질 때 감소함을 보인다.
- **Proposition 2:** 이상적인 스케줄러를 사용할 경우, 최적점 $\theta_0^*$에서 멀리 떨어진 경우 그래디언트가 더 가팔라져 학습 속도가 빨라지고, 최적점 근처에서는 landscape가 평탄해져 일반화 능력이 향상됨을 보인다.

## 📊 Results

### 1. 실험 설정

- **데이터셋:** miniImageNet (분류), Drug activity prediction (회귀)
- **평가 지표:** 분류는 Accuracy, 회귀는 Pearson coefficient의 제곱($R^2$) 및 $R^2 > 0.3$인 태스크 수 사용.
- **비교 대상:** Uniform sampling, SPL, FocalLoss, DAML, GCP, PAML.
- **백본 알고리즘:** ANIL (Almost No Inner Loop).

### 2. 주요 결과

- **노이즈 환경 (Meta-learning with Noise):**
  - miniImageNet에서 ATS는 기존 baseline 대비 최대 13% 성능 향상을 보였다.
  - Drug 데이터셋에서도 최대 18%의 성능 향상을 기록하였다.
  - 분석 결과, ATS는 노이즈가 섞인 태스크의 가중치를 낮게 유지함으로써 모델의 강건성(robustness)을 높였다.
- **제한된 예산 환경 (Limited Budgets):**
  - 메타-트레이닝 태스크 수가 적은 상황에서도 ATS가 일관되게 우수한 성능을 보였다.
  - 특히 miniImageNet에서 학습 데이터(클래스 수)가 적을수록 Uniform sampling 대비 ATS의 성능 향상 폭이 더 컸다.

## 🧠 Insights & Discussion

### 강점 및 분석

ATS의 가장 큰 강점은 **상황에 따른 적응성(adaptability)**에 있다. 실험 결과, 스케줄러는 환경에 따라 서로 다른 전략을 취하는 것으로 나타났다.

1. **노이즈가 많은 경우:** 손실 값이 큰 태스크를 '노이즈'로 판단하여 샘플링 확률을 낮춘다.
2. **데이터가 부족한 경우:** 손실 값이 큰 태스크를 '학습 가치가 높은 어려운 태스크'로 판단하여 샘플링 확률을 높인다.
이는 고정된 규칙의 스케줄러로는 불가능하며, 메타-모델의 상태와 일반화 능력을 함께 고려하는 신경망 스케줄러만이 가능한 동작이다.

### 한계 및 비판적 해석

- **계산 비용:** 신경망 스케줄러와 메타-모델을 교차 학습시키고, 매 반복마다 후보 태스크들의 특성을 계산해야 하므로 단순 무작위 샘플링보다 계산 비용이 높다.
- **범위의 제한:** 본 논문은 태스크 수준(task-level)의 스케줄링만 다루고 있다. 각 태스크 내부의 개별 샘플 수준(sample-level) 스케줄링과 어떻게 결합할지에 대한 논의가 부족하다.
- **가정:** 검증 세트(validation set)가 깨끗하고 대표성이 있다는 가정을 전제로 한다. 만약 검증 세트마저 오염되었다면 스케줄러의 학습 방향이 잘못될 위험이 있다.

## 📌 TL;DR

본 논문은 메타-트레이닝 태스크의 노이즈와 불균형 문제를 해결하기 위해, 쿼리 손실과 그래디언트 유사도를 기반으로 태스크 샘플링 확률을 예측하는 **Adaptive Task Scheduler (ATS)**를 제안한다. Bi-level Optimization과 REINFORCE 알고리즘을 통해 일반화 능력을 직접 최적화하며, 이를 통해 노이즈가 많은 환경과 데이터가 부족한 환경 모두에서 기존 메타학습 방식보다 뛰어난 성능과 강건성을 입증하였다. 이 연구는 향후 데이터 효율적인 메타학습 및 실제 세계의 노이즈 섞인 데이터셋 적용에 중요한 기여를 할 것으로 보인다.
