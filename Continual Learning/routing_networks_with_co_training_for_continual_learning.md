# Routing Networks with Co-training for Continual Learning

Mark Collier, Efi Kokiopoulou, Andrea Gesmundo, Jesse Berent (2020)

## 🧩 Problem to Solve

본 논문은 Continual Learning(지속 학습)의 핵심 난제인 Catastrophic Forgetting(파괴적 망각) 문제를 해결하고자 한다. Catastrophic Forgetting은 신경망이 일련의 태스크를 순차적으로 학습할 때, 새로운 태스크를 학습하면서 이전에 학습했던 태스크에 대한 성능이 급격히 저하되는 현상을 의미한다. 특히, 학습하는 태스크들 간의 유사성이 낮을수록 이러한 망각 현상이 더욱 심하게 나타나는 경향이 있다.

따라서 본 연구의 목표는 서로 다른 태스크 간의 간섭(Interference)은 최소화하면서, 유사한 태스크 간의 긍정적 전이(Positive Transfer)는 가능하게 하는 네트워크 구조와 학습 방법을 제안하는 것이다. 특히, 기존의 많은 아키텍처 기반 방법들이 새로운 태스크가 추가될 때마다 모델의 용량(Capacity)을 선형적으로 확장해야 했던 한계를 극복하고, 고정된 용량 내에서 효율적으로 학습하는 것을 지향한다.

## ✨ Key Contributions

본 논문의 중심적인 아이디어는 Sparse Routing Networks(희소 라우팅 네트워크)를 Continual Learning에 도입하고, 이를 최적화하기 위한 Co-training 기법을 제안하는 것이다.

1. **Sparse Routing Networks의 도입**: 입력 데이터에 따라 네트워크 내의 서로 다른 경로(전문가 집합, Experts)를 활성화함으로써, 서로 다른 태스크는 서로 다른 전문가 집합을 사용하게 하여 간섭을 줄이고, 유사한 태스크는 전문가를 공유하게 하여 지식 전이를 유도한다.
2. **Co-training 방법론 제안**: 라우팅 네트워크를 순차적으로 학습시킬 때, 일부 전문가만 선택되고 나머지는 사용되지 않는 '탐욕적 선택(Greedy choice)' 문제를 해결하기 위해, 사용되지 않은 전문가들도 현재 데이터를 통해 함께 학습시키는 Co-training 기법을 제안한다.
3. **고정 용량 아키텍처 구현**: 태스크 수에 따라 모델 크기가 커지지 않는 고정 용량의 Mixture-of-Experts(MoE) 구조를 사용하여 실제 적용 가능성을 높였다.

## 📎 Related Works

논문에서는 Continual Learning의 접근 방식을 네 가지 범주로 나누어 설명하고 있다.

- **Dynamic Model Architectures**: Progressive Networks나 DEN(Dynamically Expandable Networks)과 같이 새로운 태스크가 들어올 때 모델의 용량을 확장하는 방식이다. 하지만 태스크 수에 따라 용량이 선형적으로 증가한다는 치명적인 한계가 있다.
- **Loss Regularization**: EWC나 MER과 같이 손실 함수를 수정하여 중요한 가중치가 변하지 않도록 규제하는 방식이다.
- **Memory-based Methods**: GEM이나 A-GEM과 같이 과거 데이터의 일부를 저장하는 episodic memory를 사용하여 과거 태스크의 그래디언트와 충돌하지 않도록 제어하는 방식이다.
- **Model Weights Importance**: 가중치의 불확실성이나 Fisher 정보 행렬을 사용하여 가중치의 중요도를 추정하고 이를 보존하는 방식이다.

본 논문의 제안 방식은 위 방법들과 달리 **고정된 용량의 희소 아키텍처**를 사용한다는 점에서 차별성을 가지며, 동시에 Episodic Memory와 같은 메모리 기반 방법과 결합하여 시너지 효과를 낼 수 있는 범용적인 구조를 가진다.

## 🛠️ Methodology

### 전체 시스템 구조

본 논문에서 제안하는 구조는 Mixture-of-Experts(MoE) 기반의 라우팅 네트워크이다. 전체 파이프라인은 크게 Router, Experts, 그리고 Aggregator로 구성된다.

- **Router**: 입력 및 태스크 ID $i$에 기반하여 어떤 전문가를 활성화할지 결정하는 확률 분포를 출력한다. 본 논문에서는 라우팅 행렬 $R$을 사용하여 태스크 $i$가 전문가 $j$를 선택할 확률을 관리한다.
- **Experts**: 각각의 전문가는 독립적인 신경망 층이다. 라우터에 의해 선택된 소수의 전문가만이 연산에 참여한다.
- **Aggregator**: 선택된 전문가들의 출력을 가중 합산(Weighted sum)하여 다음 층으로 전달한다.

### Co-training 학습 절차

단순히 라우팅 네트워크를 순차 학습시키면, 초기 태스크에서 이미 학습된 전문가들만 계속 선택되고, 나머지 전문가들은 무작위 초기화 상태로 남게 되어 효율적인 용량 활용이 불가능해진다. 이를 해결하기 위해 다음과 같은 Co-training 절차를 수행한다.

1. **표준 학습 단계**: 현재 태스크의 데이터와 Episodic Memory에서 추출한 데이터를 사용하여, 라우터가 선택한 전문가들만을 통해 표준 SGD 업데이트를 수행한다.
2. **전문가 사용 여부 추적**: 각 층의 전문가 중 라우팅 확률 상위 $k$위 안에 들어 실제 연산에 참여한 전문가들을 '사용됨(Used)' 상태로 기록한다.
3. **Co-training 단계**: 아직 어떤 태스크에서도 사용되지 않은 전문가들($1 - E$)을 강제로 활성화하여, 동일한 데이터 배치에 대해 추가적인 그래디언트 업데이트를 수행한다. 이를 통해 새로운 태스크가 등장했을 때 라우터가 선택할 수 있는 '잘 초기화된' 전문가 풀을 확보한다.

### 주요 방정식 및 알고리즘

학습 과정은 다음과 같은 흐름을 따른다.

- 표준 업데이트: $(\theta, R) \leftarrow \text{SGD}(x, y, \alpha, \theta_{\hat{R}}, R)$
- Co-training 업데이트: $(\theta, \cdot) \leftarrow \text{SGD}(x \cup x^M, y \cup y^M, \alpha^c, \theta_{(1-E)}, R)$
여기서 $\theta_{\hat{R}}$은 라우터가 선택한 전문가의 가중치이며, $\theta_{(1-E)}$는 사용되지 않은 전문가들의 가중치를 의미한다.

## 📊 Results

### 실험 설정

- **데이터셋**: MNIST-Permutations 및 MNIST-Rotations (각 20개 태스크)
- **비교 대상 (Baseline)**: Shared Bottom 신경망 + Reservoir Sampling 기반 Episodic Memory (1,000개 샘플)
- **평가 지표**:
  - **Average Accuracy (ACC)**: 모든 태스크 학습 후의 평균 정확도.
  - **Backward Transfer (BWT)**: 새로운 태스크 학습이 이전 태스크의 성능에 미치는 영향 (음수 값이 클수록 망각이 심함).
- **파라미터**: MoE 네트워크는 2개의 은닉층을 가지며, 각 층당 20명의 전문가 중 $k=4$명을 활성화한다. 공정한 비교를 위해 전체 파라미터 수는 Shared Bottom 네트워크와 동일하게 맞추었다.

### 정량적 결과

실험 결과, **MoE + Replay Buffer + Co-training** 조합이 가장 우수한 성능을 보였다.

| Method | MNIST-Perm (BWT / ACC) | MNIST-Rot (BWT / ACC) |
| :--- | :---: | :---: |
| Shared Bottom | $-0.219 / 0.726$ | $-0.269 / 0.693$ |
| Shared Bottom + Replay | $-0.057 / 0.912$ | $-0.057 / 0.920$ |
| MoE + Replay | $-0.040 / 0.918$ | $-0.038 / 0.923$ |
| **MoE + Replay + Co-training** | $\mathbf{-0.038 / 0.920}$ | $\mathbf{-0.034 / 0.929}$ |

결과적으로 MoE 기반 방식이 Shared Bottom 방식보다 BWT가 낮고(망각이 적고) ACC가 높음을 확인할 수 있다.

### 정성적 분석 및 해석 가능성

MNIST-Rotations 실험에서 라우팅 행렬 $R$을 분석한 결과, 회전 각도가 유사한 태스크들은 전문가를 공유하고, 각도가 크게 차이 나는 태스크들은 서로 다른 전문가를 사용하는 블록 대각 구조(Block diagonal structure)의 태스크 유사도 행렬이 형성됨을 확인하였다. 이는 네트워크가 스스로 태스크 간의 유사성을 학습하여 적절히 경로를 분리하고 있음을 입증한다.

## 🧠 Insights & Discussion

### 강점

본 연구는 고정된 모델 용량 내에서 Sparse Routing을 통해 Catastrophic Forgetting을 효과적으로 억제하였다. 특히 Co-training을 통해 전문가들의 초기화 문제를 해결함으로써 MoE 구조의 잠재력을 극대화하였다. 또한, 학습된 라우팅 행렬을 통해 태스크 간 유사도를 시각화할 수 있어 모델의 의사결정 과정에 대한 해석 가능성(Interpretability)을 제공한다.

### 한계 및 비판적 해석

논문에서 언급되었듯, MoE 네트워크는 학습 초기 단계에서 Shared Bottom 방식보다 정확도가 낮게 나타나는 경향이 있다. 이는 라우팅 결정의 확률적 특성(Stochasticity)으로 인해 샘플 효율성(Sample efficiency)이 떨어지기 때문으로 분석된다. 또한, 본 실험이 MNIST와 같은 비교적 단순한 데이터셋에서 진행되었으므로, 매우 복잡한 고차원 데이터셋에서도 동일한 수준의 전문가 분리 및 전이 효과가 나타날지는 추가적인 검증이 필요하다.

## 📌 TL;DR

본 논문은 Continual Learning에서 발생하는 파괴적 망각을 줄이기 위해 **고정 용량의 Sparse Routing Network(MoE)**와 **Co-training** 기법을 제안한다. 유사한 태스크는 전문가를 공유하고 서로 다른 태스크는 분리된 경로를 사용하게 하여 간섭을 최소화하며, Co-training을 통해 모든 전문가가 균형 있게 학습되도록 보장한다. 실험적으로 MNIST 벤치마크에서 기존 방식보다 적은 망각과 높은 평균 정확도를 달성하였으며, 이는 향후 고정 용량 모델 기반의 지속 학습 연구에 중요한 기초를 제공할 수 있다.
