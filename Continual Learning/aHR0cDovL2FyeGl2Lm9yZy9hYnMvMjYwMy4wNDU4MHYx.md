# Why Do Neural Networks Forget: A Study of Collapse in Continual Learning

Yunqin Zhu, Jun Jin (2026)

## 🧩 Problem to Solve

본 논문은 지속 학습(Continual Learning, CL)의 핵심 난제인 치명적 망각(Catastrophic Forgetting) 현상을 분석한다. 치명적 망각이란 모델이 새로운 태스크를 학습하는 과정에서 이전 태스크에 대해 학습했던 지식을 급격히 소실하는 현상을 의미한다.

기존의 많은 연구는 망각의 정도를 측정하기 위해 단순히 태스크 정확도(Task Accuracy)와 같은 외부 성능 지표에 의존해 왔다. 그러나 이러한 접근 방식은 모델 내부의 구조적 변화를 간과하며, 왜 특정 전략이 망각을 억제하는지에 대한 근본적인 메커니즘을 설명하지 못한다. 본 연구의 목표는 모델 내부의 구조적 붕괴(Structural Collapse)와 표현 붕괴(Representational Collapse)가 치명적 망각과 어떤 상관관계를 갖는지 분석하고, 이를 정량적으로 측정하여 망각의 기하학적 원인을 규명하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 치명적 망각을 단순한 성능 저하가 아닌, 모델 내부 표현 공간의 차원이 급격히 축소되는 '기하학적 실패(Geometric Failure)'로 보는 것이다. 연구진은 이를 측정하기 위해 유효 랭크(Effective Rank, eRank)라는 지표를 도입하였다.

핵심 직관은 모델이 새로운 정보를 학습하기 위해 특징 공간(Feature Space)을 확장할 수 있는 능력, 즉 가소성(Plasticity)을 잃게 되면, 결국 기존의 표현을 덮어쓰게 되어 망각이 발생한다는 것이다. 따라서 가중치(Weight)의 eRank를 통해 구조적 붕괴를, 활성화 값(Activation)의 eRank를 통해 표현 붕괴를 추적함으로써 망각의 진행 과정을 정량적으로 모니터링할 수 있음을 제안한다.

## 📎 Related Works

지속 학습은 크게 Task-Incremental Learning(Task-IL), Class-Incremental Learning(Class-IL), Domain-Incremental Learning의 세 가지 설정으로 구분된다. 본 논문은 이 중 Task-IL과 Class-IL을 중점적으로 다룬다.

기존 연구들은 치명적 망각의 원인을 주로 상충하는 그래디언트(Conflicting Gradients)나 분류기 드리프트(Classifier Drift)에서 찾았다. 이를 해결하기 위해 다음과 같은 전략들이 제안되었다:

- **파라미터 기반 정규화(Parameter-based Regularization):** EWC나 MAS와 같이 중요 가중치의 변화를 억제하는 방식이다.
- **기능적 정규화(Functional Regularization):** LwF와 같이 지식 증류(Knowledge Distillation)를 통해 이전 태스크의 출력 행동을 보존하는 방식이다.
- **경험 재생(Experience Replay, ER):** 과거 데이터의 일부를 버퍼에 저장하고 새 데이터와 함께 학습시켜 표현 드리프트를 방지하는 방식이다.

본 논문은 이러한 기존 방법론들이 성능은 개선시키지만, 내부 표현 공간의 차원 유지(Plasticity)에 어떤 영향을 주는지는 충분히 분석되지 않았다는 점을 차별점으로 삼는다.

## 🛠️ Methodology

### 전체 시스템 구조 및 분석 파이프라인

본 연구는 네 가지 서로 다른 아키텍처(MLP, ConvGRU, ResNet-18, Bi-ConvGRU)를 대상으로 세 가지 학습 전략(SGD, LwF, ER)을 적용하여 그 효과를 비교 분석한다. 데이터셋으로는 단순한 환경인 Split MNIST(Task-IL)와 복잡한 환경인 Split CIFAR-100(Class-IL)을 사용한다.

### 주요 구성 요소 및 학습 절차

1. **학습 전략:**
   - **SGD:** 아무런 방어 기제 없이 순차적으로 학습하는 베이스라인이다.
   - **Experience Replay (ER):** Reservoir Sampling을 통해 최대 2,000개의 샘플을 저장하는 버퍼를 운영하며, 매 배치마다 새 데이터와 버퍼 데이터를 함께 학습한다.
   - **Learning without Forgetting (LwF):** 학습 전 현재 모델을 Teacher 모델로 복사하여 고정하고, Student 모델이 Teacher의 출력을 따르도록 하는 증류 손실(Distillation Loss)을 추가한다.
     $$\mathcal{L} = \mathcal{L}_{task} + \lambda \cdot \mathcal{L}_{distill}$$
     여기서 $\mathcal{L}_{distill}$은 두 모델의 소프트맥스 출력 간의 KL Divergence로 계산된다.

2. **분석 지표: Effective Rank (eRank)**
   행렬의 특이값(Singular Values) 분포의 엔트로피를 이용하여 유효 차원을 측정한다. 특이값 $\sigma_i$를 정규화한 분포를 $p_i$라고 할 때, eRank는 다음과 같이 정의된다:
   $$\text{eRank} = \exp\left( -\sum_i p_i \log p_i \right)$$
   - **Weight eRank:** 각 레이어의 가중치 행렬 $W$에 대해 계산하며, 구조적 붕괴를 측정한다.
   - **Activation eRank:** 마지막 은닉층의 활성화 행렬 $A$에 대해 계산하며, 표현 붕괴를 측정한다.
   - **Peak-Normalized eRank ($\text{eRank}_{pct}$):** 아키텍처마다 다른 차원 크기를 보정하기 위해, 과거 최대 eRank 대비 현재 유지 비율을 계산한다.
     $$\text{eRank}_{pct} = \frac{\text{eRank}(t)}{\max_{u \le t} \text{eRank}(u)}$$

### 아키텍처 상세

- **MLP:** 단순 완전 연결 층으로 구성되어 붕괴에 가장 취약한 하한선 역할을 한다.
- **ConvGRU:** Convolution과 Gated Recurrent Unit을 결합하여 시간적 메모리와 게이팅 메커니즘을 통해 정보 흐름을 조절한다.
- **ResNet-18:** Residual Skip Connection을 통해 그래디언트 흐름을 안정화하고 특징 재사용을 가능하게 한다.
- **Bi-ConvGRU:** ConvGRU를 양방향으로 확장하여 더 풍부한 공간 메모리를 확보한다.

## 📊 Results

### Split MNIST 실험 (MLP, ConvGRU)

- **정확도 및 망각:** SGD는 심각한 성능 저하를 보였으며, 특히 MLP에서 가장 두드러졌다. LwF는 어느 정도 완화시키지만, ER이 모든 태스크에서 가장 높은 정확도를 유지하고 망각을 거의 완벽하게 억제했다.
- **Activation eRank:** MLP-SGD의 경우 학습 초기에는 eRank가 상승하다가 태스크 2~5 사이에서 급격히 붕괴하며, 이는 정확도 하락 시점과 일치한다. 반면 MLP-ER은 학습이 진행됨에 따라 eRank가 지속적으로 상승한다.
- **Weight eRank:** 모든 모델에서 가중치 랭크가 감소하는 구조적 붕괴가 관찰되었으며, ER이 이 붕괴 속도를 가장 효과적으로 늦추었다. LwF는 출력 행동은 보존하지만 가중치 랭크의 하락을 막는 데는 한계가 있었다.

### Split CIFAR-100 실험 (ResNet-18, Bi-ConvGRU)

- **정확도 및 망각:** SGD 모델들은 태스크 1 이후 정확도가 급락했다. ER은 약 80%의 높은 정확도를 유지한 반면, LwF는 ResNet-18에서 약 60%, Bi-ConvGRU에서 40% 수준의 성능을 보였다.
- **Activation eRank:** ResNet-18-SGD는 초기에는 높은 eRank를 가지나 결국 0에 가깝게 붕괴한다. ER과 LwF는 이를 안정화시키지만, ER의 성능이 더 우수하다.
- **Weight eRank:** 초기 5개 태스크에서 공통적으로 급격한 하락이 발생한다. 특히 ResNet-18은 Skip Connection이 있음에도 불구하고 장기적으로는 가중치 붕괴가 일어나 가소성을 상실하는 모습이 관찰되었다.

## 🧠 Insights & Discussion

### 표현 붕괴와 망각의 상관관계

본 연구는 $\text{eRank}$의 하락(붕괴)이 망각의 직접적인 원인임을 입증하였다. 새로운 태스크 학습 시 정규화가 부족하면 가중치가 새로운 특징에 맞춰지면서 기존 특징들이 저차원 부분 공간으로 압축(Collapse)된다. 이로 인해 클래스 간 구분 능력이 상실되어 결정 경계가 붕괴되고, 결과적으로 치명적 망각이 발생한다.

### 아키텍처별 붕괴 특성

- **Feed-forward 모델 (MLP, ResNet-18):** 초기 표현 용량은 매우 크지만, 보호 기제가 없으면 붕괴 속도가 매우 빠르다. ResNet-18의 Skip Connection은 초기 붕괴를 지연시킬 뿐, 장기적인 가소성 상실을 완전히 막지는 못한다.
- **Recurrent 모델 (ConvGRU, Bi-ConvGRU):** 게이팅 메커니즘이 그래디언트 간섭을 줄여 구조적 실패를 늦춘다. 하지만 이는 초기부터 표현을 공격적으로 압축하여 안정성을 얻는 방식(Trade-off)이므로, 절대적인 표현 풍부함(Representational Richness)과 장기적 용량은 피드포워드 모델보다 낮다.

### CL 전략의 효과성 분석

- **ER의 우수성:** ER은 과거 데이터를 직접 재생함으로써 모델이 과거와 현재의 특징을 모두 수용할 수 있는 풍부한 특징 부분 공간을 강제로 유지하게 한다. 이는 가중치와 활성화 값 모두에서 eRank를 보존하여 가소성을 유지하는 결과를 낳는다.
- **LwF의 한계:** LwF는 '출력 행동'을 고정하여 망각을 늦추지만, 내부 가중치 행렬의 구조적 붕괴(Weight Collapse)를 막지는 못한다. 즉, 겉으로는 성능을 유지하는 듯 보이나 내부 용량은 계속 잠식되므로, 학습 태스크가 늘어날수록 결국 한계에 부딪히게 된다.

## 📌 TL;DR

본 논문은 치명적 망각의 원인을 모델 내부 표현 공간의 차원이 축소되는 **'구조적 및 표현적 붕괴'**에서 찾고, 이를 **Effective Rank (eRank)**라는 지표로 정량화하여 분석했다. 실험 결과, 망각은 eRank의 하락과 강하게 결합되어 있으며, **Experience Replay (ER)** 전략이 가중치의 유효 차원을 가장 잘 보존하여 가소성을 유지함으로써 망각을 효과적으로 억제함을 보였다. 이 연구는 지속 학습 모델을 설계할 때 단순히 출력 성능을 높이는 것뿐만 아니라, 내부 표현 공간의 차원(Capacity)을 어떻게 유지할 것인가가 핵심임을 시사한다.
