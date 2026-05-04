# Improved Knowledge Distillation for Pre-trained Language Models via Knowledge Selection

Chenglong Wang, Yi Lu, Yongyu Mu, Yimin Hu, Tong Xiao, and Jingbo Zhu (2023)

## 🧩 Problem to Solve

본 논문은 사전 학습된 언어 모델(Pre-trained Language Models, PLMs)의 거대한 파라미터 수로 인해 발생하는 배포 비용 문제를 해결하기 위해 Knowledge Distillation(KD)을 활용한다. 일반적으로 KD 과정에서는 교사 모델(Teacher model)로부터 추출된 여러 유형의 지식(Knowledge)을 학생 모델(Student model)에게 전달한다.

그러나 저자들은 사전 연구를 통해 다음과 같은 두 가지 핵심 문제를 발견하였다.
1. 모든 유형의 지식이 좋은 학생 모델을 학습시키는 데 필수적인 것은 아니며, 지식의 종류에 따라 성능 차이가 발생한다.
2. 학습 단계(Training step)에 따라 학생 모델에게 도움이 되는 지식의 유형이 동적으로 변화한다.

따라서 본 논문의 목표는 KD 과정 중 각 학습 단계에서 가장 적절한 지식을 선택하여 전이함으로써, 학생 모델의 성능을 극대화하는 Knowledge Selection 전략을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Actor-Critic 강화학습 프레임워크를 도입하여 지식 선택 모듈(Knowledge Selection Module, KSM)을 설계**하는 것이다. 

단순히 고정된 지식을 사용하거나 무작위로 선택하는 것이 아니라, 현재 모델의 상태를 반영하여 미래의 보상을 최대화할 수 있는 지식을 동적으로 선택한다. 또한, 보상 계산에 따른 연산 부담을 줄이기 위해 학습 과정을 여러 단계로 나누는 **Multi-phase training** 기법과 로컬 옵티마 탈출을 위한 **Exploration reward**를 제안하였다.

## 📎 Related Works

기존의 KD 연구들은 크게 세 가지 방향으로 분류된다.
1. **다양한 지식 활용:** 여러 교사 모델을 사용하거나, 교사 모델의 중간 레이어 지식, 혹은 파라미터 자체를 전이하는 방식이다.
2. **효율적인 학습 전략:** task-specific 데이터를 증강하거나 2단계 학습 프레임워크를 설계하여 학습 효율을 높이는 방식이다.
3. **데이터 및 교사 모델 선택:** 학생 모델의 역량에 따라 중요한 데이터만 선택하거나 교사 모델의 가중치를 조정하는 방식이다.

기존 연구들이 "어떻게 하면 전이된 지식을 더 잘 배울 것인가"에 집중했다면, 본 논문은 **"어떤 지식을 언제 선택하여 전이할 것인가"**라는 지식 선택(Knowledge Selection) 문제에 집중한다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 전체 파이프라인
본 방법론은 두 단계로 구성된다.
1. **KSM 학습 단계:** Actor-Critic 알고리즘을 통해 상태($s$)에 따라 최적의 지식($a$)을 선택하는 KSM을 학습시킨다.
2. **학생 모델 증류 단계:** 학습된 KSM을 사용하여 각 학습 단계에서 최적의 지식을 선택하고 이를 학생 모델에게 전이한다.

### 주요 구성 요소 및 상세 설명

#### 1. 지식의 유형 (Knowledge Types)
본 논문은 다음과 같은 네 가지 지식 유형을 정의하고 각각에 맞는 손실 함수를 설계하였다.
- **Response Knowledge (ResK):** 교사 모델의 마지막 레이어 출력값(Soft labels)을 모방한다. $\text{KL Divergence}$를 사용한다.
- **Feature Knowledge (FeaK):** 교사 모델의 중간 레이어 특징 표현을 학습한다. $\text{L2 distance}$를 사용한다.
- **Relation Knowledge (RelK):** 레이어 간 또는 샘플 간의 관계를 학습한다. FSP(Flow of Solution Procedure) 행렬의 $\text{MSE}$를 사용한다.
- **Finetune Knowledge (FinK):** 정답 레이블(Ground-truth)을 통해 직접 학습하는 일반적인 파인튜닝 지식이다. $\text{Cross-Entropy}$를 사용한다.

#### 2. Knowledge Selection Module (KSM)
KSM은 상태를 입력받아 어떤 지식을 사용할지 결정하는 에이전트 역할을 한다.

- **State ($s_t$):** 학습 단계 $t$에서 학생 모델과 교사 모델의 마지막 레이어 $\text{[CLS]}$ 임베딩을 추출하고, 각각 2층 MLP인 $N_{fea}^S, N_{fea}^T$를 통과시켜 특징 벡터 $v$를 생성한 후 이를 결합하여 상태를 정의한다.
- **Action ($a_t$):**
    - **Soft Action:** 4가지 지식 유형에 대해 $[0, 1]$ 사이의 가중치를 부여한다.
      $$L^{soft}_t = a^t_1 L_{FinK} + a^t_2 L_{ResK} + a^t_3 L_{FeaK} + a^t_4 L_{RelK}$$
    - **Hard Action:** 임계값 $\lambda$를 기준으로 지식 사용 여부를 0 또는 1로 결정한다.
- **Reward ($r_t$):** 개발 세트(Development set)에서의 $\text{Cross-Entropy loss}$ 차이를 즉각적인 보상으로 정의한다.

#### 3. 최적화 알고리즘
- **Actor ($\mu_\theta$):** 3층 MLP로 구성되며, 정책 경사(Policy Gradient)를 통해 장기 보상을 최대화하도록 업데이트된다.
  $$\nabla_\theta J \approx \frac{1}{N} \sum_{t=1}^N \nabla_\theta Q_\phi(s_t, a_t)$$
- **Critic ($Q_\phi$):** 3층 MLP로 구성되며, $\text{Temporal-Difference (TD)}$ 학습을 통해 상태-행동 가치 함수를 근사한다.
  $$Q^*_\phi(s_t, a_t) = \gamma r_t + Q_\phi(s_{t+1}, a_{t+1})$$
  손실 함수는 예측 가치와 목표 가치 사이의 $\text{MSE}$를 최소화하는 방향으로 학습된다.

#### 4. 효율성 및 안정성 향상 기법
- **Multi-phase Training:** 매 스텝마다 보상을 계산하는 비용을 줄이기 위해 전체 과정을 여러 페이즈(Phase)로 나누고, 각 페이즈가 끝날 때 한 번만 보상을 계산하여 Critic을 업데이트한다.
- **Exploration Reward:** Actor가 다양한 행동을 시도하도록 유도한다. Soft action의 경우 이전 페이즈 행동과의 코사인 유사도를 이용하고, Hard action의 경우 행동의 반복 횟수를 기반으로 보상을 부여한다.

## 📊 Results

### 실험 설정
- **데이터셋:** GLUE 벤치마크의 6개 데이터셋(MNLI, QQP, QNLI, SST-2, MRPC, RTE).
- **모델:** 교사 모델은 $\text{BERT}_{BASE}$, 학생 모델은 $\text{BERT}_6$(6층) 및 $\text{BERT}_3$(3층)를 사용하였다.
- **비교 대상:** Vanilla KD, PKD, DistilBERT, Dynamic KD, Finetune 및 무작위 선택(Random-Hard/Soft) 베이스라인.

### 주요 결과
1. **정량적 성능:** 제안 방법은 $\text{BERT}_6$와 $\text{BERT}_3$ 모든 설정에서 기존 베이스라인들을 유의미하게 능가하였다. 특히 $\text{BERT}_6$ 학생 모델은 파라미터 수를 약 61%로 줄이면서도 교사 모델 성능의 98.5% 이상을 달성하였다.
2. **Action 전략 비교:** Soft action이 Hard action보다 성능이 더 좋게 나타났다. 이는 Soft action의 탐색 공간이 더 넓어 최적의 지식 조합을 찾을 가능성이 높기 때문으로 분석된다.
3. **데이터 증강과의 결합:** 데이터 증강(Data Augmentation)을 적용했을 때 $\text{TinyBERT}$보다 우수한 성능을 보였으며, 이는 본 방법론이 다른 KD 기법들과 상호보완적(Orthogonal)임을 시사한다.
4. **Ablation Study:** Multi-phase training은 보상의 안정성을 높여 KSM 학습을 돕고, Exploration reward는 더 빠르게 최적의 지식 선택 전략에 도달하게 함을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 해석
본 연구는 지식 전이가 고정된 프로세스가 아니라 학습 상태에 따라 변화해야 하는 **동적인 과정**임을 입증하였다. 특히, 지식 선택을 일종의 **정규화(Regularization)** 관점에서 해석한 점이 인상적이다. 특정 교사의 행동에 과하게 의존하는 'co-adaptation' 현상을 방지함으로써, 학생 모델이 다양한 지식을 적절히 학습하게 하여 과적합을 줄이고 일반화 성능을 높였다고 평가할 수 있다.

### 한계 및 미해결 질문
1. **자원 소모:** KSM을 학습시키기 위해 반복적인 에피소드 학습이 필요하므로, 시간과 계산 자원 소모가 크다.
2. **확장성 문제:** 데이터셋의 규모가 매우 커질 경우, 학습 스텝 수가 증가하여 지식 선택 문제가 더 복잡해지고 상태 공간($s$)의 복잡도가 증가하여 KSM이 혼란을 겪을 가능성이 있다.

## 📌 TL;DR

본 논문은 PLM 압축을 위한 KD 과정에서 **어떤 지식을 언제 전이할지 결정하는 Actor-Critic 기반의 지식 선택 모듈(KSM)**을 제안하였다. 이를 통해 고정된 지식 전이 방식의 한계를 극복하고 학생 모델의 성능을 유의미하게 향상시켰으며, 특히 소형 모델($\text{BERT}_3, \text{BERT}_6$)에서 교사 모델에 근접하는 성능을 보였다. 이 연구는 향후 동적 지식 전이 및 모델 압축 분야에서 효율적인 학습 스케줄링 전략을 세우는 데 중요한 기초가 될 것으로 보인다.