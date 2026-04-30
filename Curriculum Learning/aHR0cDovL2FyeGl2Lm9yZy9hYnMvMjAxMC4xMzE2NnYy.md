# A Survey on Curriculum Learning

Xin Wang, Yudong Chen, and Wenwu Zhu (2021)

## 🧩 Problem to Solve

전통적인 머신러닝 알고리즘은 훈련 데이터를 모델에 무작위로 제공하는 방식을 취한다. 이는 데이터 샘플마다 가지고 있는 복잡성의 차이와 현재 모델의 학습 상태를 완전히 무시하는 접근 방식이다. 반면, 인간의 교육 과정(Curriculum)은 기초적인 단계에서 시작하여 점차 복잡한 개념으로 나아가는 '의미 있는 순서'를 가지고 있다.

본 논문은 이러한 인간의 학습 방식을 모방하여, 쉬운 데이터부터 어려운 데이터 순으로 학습시키는 Curriculum Learning (CL) 전략을 체계적으로 분석하고자 한다. CL의 핵심 목표는 훈련 순서를 최적화함으로써 모델의 일반화 성능(Generalization capacity)을 향상시키고 수렴 속도(Convergence rate)를 가속화하는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 기여는 파편화되어 있던 다양한 CL 방법론을 하나의 통합된 프레임워크로 정의하고 분류했다는 점이다. 구체적인 기여 사항은 다음과 같다.

1.  **통합 프레임워크 제시**: 대부분의 CL 방법론이 $\text{Difficulty Measurer} + \text{Training Scheduler}$라는 일반적인 구조를 따르고 있음을 밝히고 이를 체계화하였다.
2.  **포괄적인 분류 체계 구축**: CL을 크게 Predefined CL과 Automatic CL로 나누고, 특히 Automatic CL을 다시 Self-paced Learning, Transfer Teacher, RL Teacher, 그리고 기타 자동화 방법으로 세분화하여 분석하였다.
3.  **이론적 배경 분석**: CL이 왜 효과적인지를 최적화 문제(Optimization problem)와 데이터 분포(Data distribution) 관점에서 분석하여, '가이드(Guide)'와 '잡음 제거(Denoise)'라는 두 가지 핵심 동기를 제시하였다.
4.  **상호 관계 분석**: CL과 전이 학습(Transfer Learning), 메타 학습(Meta-learning), 지속 학습(Continual Learning), 능동 학습(Active Learning) 간의 관계를 데이터 분포 관점에서 해석하였다.

## 📎 Related Works

논문은 CL의 기원을 인간 및 동물의 행동 과학, 그리고 초기 로보틱스 및 NLP 연구에서 찾는다. 특히 Bengio 등이 제안한 초기 CL 정의를 기반으로 하며, 이후 확장된 다양한 연구들을 다룬다.

기존의 접근 방식들은 주로 특정 태스크에 맞춘 수동적인 커리큘럼 설계에 의존했다. 하지만 이러한 방식은 다음과 같은 한계가 있다.
- **전문가 지식 의존성**: 데이터의 '쉬움'과 '어려움'을 정의하기 위해 도메인 전문가의 지식이 필수적이다.
- **유연성 부족**: 학습 과정 중 모델의 상태 변화를 반영하지 못하고 고정된 순서로만 데이터를 제공한다.
- **최적 조합 탐색의 어려움**: 어떤 난이도 측정법과 스케줄러의 조합이 최적인지 알 수 없어 단순한 시행착오(Trial and error)에 의존해야 한다.

## 🛠️ Methodology

### 1. CL의 일반적 정의 (General Definitions)
논문은 CL을 세 가지 단계의 정의로 확장하여 설명한다.

- **Original CL (Definition 1)**: 훈련 단계 $t$에 따른 일련의 재가중치 부여 기준 $C = \langle Q_1, \dots, Q_T \rangle$로 정의된다.
  $$Q_t(z) \propto W_t(z)P(z) \quad \forall \text{example } z \in \text{training set } D$$
  여기서 분포의 엔트로피 $H(Q_t)$는 점차 증가해야 하며, 최종적으로는 전체 데이터 분포 $P(z)$에 도달해야 한다.
- **Data-level Generalized CL (Definition 2)**: 위 조건들을 완화하여, 단순히 훈련 분포에 재가중치를 부여하는 전략으로 본다. 여기에는 '어려운 데이터부터 학습'하는 Anti-curriculum이나 Hard Example Mining (HEM)도 포함된다.
- **Generalized CL (Definition 3)**: 데이터 수준을 넘어 모델 용량, 손실 함수, 학습 목표 등 모든 훈련 기준(Training criteria)의 시퀀스로 정의를 확장한다.

### 2. 통합 프레임워크: $\text{Difficulty Measurer} + \text{Training Scheduler}$
모든 CL 설계는 다음 두 구성 요소로 요약된다.
- **Difficulty Measurer**: 각 데이터 샘플의 상대적인 '쉬움' 정도를 결정한다.
- **Training Scheduler**: 측정된 난이도를 바탕으로 훈련 과정에서 어떤 데이터 서브셋을 언제 투입할지 결정한다.

### 3. 세부 방법론

#### (1) Predefined CL
- **Difficulty Measurer**: 구조적 복잡성(문장 길이, 객체 수), 다양성(단어 희소성, 엔트로피), 잡음 추정(데이터 소스, SNR) 등을 기반으로 수동 설계한다.
- **Training Scheduler**: 
    - **Discrete**: 데이터를 버킷(Bucket)으로 나누어 순차적으로 추가하는 Baby Step 방식이나, 이전 버킷을 버리고 다음으로 넘어가는 One-Pass 방식이 있다.
    - **Continuous**: epoch $t$에 따라 사용 가능한 데이터 비율 $\lambda(t)$를 결정하는 함수를 사용한다. (예: Linear, Root, Geometric function)

#### (2) Automatic CL
- **Self-Paced Learning (SPL)**: 모델 스스로가 교사가 되어, 현재 모델의 Loss가 낮은 데이터를 '쉬운 데이터'로 간주한다.
  $$\min_{w, v \in [0,1]^N} \sum_{i=1}^N v_i l_i + g(v; \lambda)$$
  여기서 $v_i$는 샘플 가중치, $g(v; \lambda)$는 SP-regularizer이며, $\lambda$는 학습 속도를 조절하는 Age parameter이다.
- **Transfer Teacher**: 미리 학습된 강력한 교사 모델(Teacher model)이 데이터의 난이도를 측정하여 학생 모델에게 전달한다.
- **RL Teacher**: 강화학습(RL) 에이전트가 교사가 되어, 학생 모델의 피드백(State, Reward)을 바탕으로 최적의 데이터를 동적으로 선택(Action)한다.
- **Other Automatic CL**: Bayesian Optimization, Meta-learning, Hypernetworks 등을 통해 최적의 커리큘럼(가중치 또는 손실 함수)을 직접 학습한다.

## 📊 Results

### 1. 정량적 결과 및 성능 경향
논문은 다양한 실험 결과들을 종합하여 다음과 같은 결론을 제시한다.
- **스케줄러 성능**: Continuous Root-p function $\gt$ Discrete Baby Step $\gt$ Discrete One-Pass 순으로 효과적인 경향을 보인다.
- **SPL 정규화**: Implicit regularizers $\gt$ Soft regularizers $\gt$ Hard regularizers 순으로 성능이 높다.
- **효과성**: CL은 특히 타겟 태스크가 매우 어렵거나, 데이터에 잡음이 많은 경우, 또는 모델 크기가 작고 데이터셋이 제한적인 상황에서 더 큰 성능 향상을 보인다.

### 2. 계산 복잡도 분석
- **Predefined CL**: $O(n \log n + M)$으로 가장 저렴하다. (정렬 비용 + 훈련 반복 횟수)
- **SPL**: $O(Mn)$ 또는 $O(Mnx)$로 매 반복마다 가중치를 업데이트해야 한다.
- **Transfer Teacher**: $O(T + n \log n + M)$으로 교사 모델의 사전 학습 비용 $T$가 추가된다.
- **RL Teacher**: $O(RM + xMn)$으로 교사 모델의 업데이트 비용 $R$이 발생하며, Deep RL을 사용할 경우 비용이 매우 높아질 수 있다.

## 🧠 Insights & Discussion

### 1. "Easier First" vs "Harder First"
CL(쉬운 것부터)과 HEM(어려운 것부터) 중 무엇이 더 좋은가는 데이터의 특성에 따라 다르다.
- **CL**은 라벨 잡음(Label noise)이나 이상치(Outlier)가 많은 데이터셋에서 모델의 강건성(Robustness)과 수렴 속도를 높이는 데 유리하다.
- **HEM**은 깨끗한 데이터셋에서 더 효율적이며, SGD의 안정성을 높이는 경향이 있다.

### 2. 다른 학습 패러다임과의 연결성
- **Transfer Learning**: CL은 초기 태스크를 통해 최종 태스크를 가이드하는 특수한 형태의 전이 학습으로 볼 수 있다.
- **Meta-Learning**: Automatic CL은 '가르치는 법을 배우는 것(Learning to teach)'이며, 이는 메타 학습의 관점과 일치한다.
- **Active Learning**: 두 방식 모두 동적 데이터 선택을 수행하지만, CL은 성능 향상과 수렴 가속이 목적이고 AL은 라벨링 비용 절감이 목적이라는 차이가 있다.

### 3. 한계 및 비판적 해석
본 논문은 광범위한 서베이를 제공하지만, CL의 효과가 '언제' 그리고 '왜' 발생하는지에 대한 통일된 수학적 증명보다는 경험적 증거에 의존하는 경향이 있다. 특히, 자동화된 CL 방법론들이 주는 성능 이득이 추가적인 계산 비용(Computational cost)을 정당화할 만큼 충분한지에 대한 심층적인 분석이 더 필요하다.

## 📌 TL;DR

본 논문은 인간의 교육 과정을 모방하여 쉬운 데이터부터 학습시키는 Curriculum Learning (CL)을 체계적으로 분석한 서베이 논문이다. 모든 CL을 $\text{Difficulty Measurer} + \text{Training Scheduler}$ 프레임워크로 통합하였으며, 수동 설계(Predefined)부터 자기 주도 학습(SPL), 강화학습 기반 교사(RL Teacher)까지의 방법론을 분류하였다. CL은 특히 잡음이 많은 데이터나 난이도가 높은 태스크에서 일반화 성능을 높이고 수렴을 가속화하며, 이는 최적화 경로를 가이드하거나 데이터 잡음을 제거하는 효과를 통해 달성된다. 향후 연구 방향으로 통일된 벤치마크 구축과 더 정교한 이론적 보장(Theoretical guarantee)의 필요성을 제시한다.