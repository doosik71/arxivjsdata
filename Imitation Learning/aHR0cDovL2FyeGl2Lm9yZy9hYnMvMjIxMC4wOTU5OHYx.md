# Planning for Sample Efficient Imitation Learning

Zhao-Heng Yin, Weirui Ye, Qifeng Chen, Yang Gao (2022)

## 🧩 Problem to Solve

본 논문은 모방 학습(Imitation Learning, IL)에서 발생하는 **성능(Performance)**과 **환경 내 샘플 효율성(In-environment Sample Efficiency)** 사이의 트레이드오프 문제를 해결하고자 한다.

기존의 대표적인 모방 학습 방식인 Behavioral Cloning (BC)은 환경과의 추가적인 상호작용이 필요 없어 샘플 효율성이 매우 높지만, 학습 데이터와 테스트 데이터의 분포가 달라지는 Covariate Shift 문제로 인해 성능이 저하되는 한계가 있다. 반면, Adversarial Imitation Learning (AIL)은 분포 매칭(Distribution Matching) 방식을 통해 더 높은 성능을 달성할 수 있지만, 학습을 위해 방대한 양의 온라인 상호작용이 필요하며 특히 이미지 입력(Image-based input) 환경에서 효율성이 크게 떨어진다는 단점이 있다.

따라서 본 연구의 목표는 BC의 샘플 효율성과 AIL의 높은 성능을 동시에 달성할 수 있는 새로운 프레임워크를 제안하는 것이며, 이를 위해 Planning 기반의 모방 학습 방법론인 **EfficientImitate (EI)**를 제시한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **MCTS(Monte Carlo Tree Search) 기반의 Planning**을 통해 서로 양립하기 어려웠던 BC와 AIL을 자연스럽게 통합하는 것이다.

1.  **AIL의 MCTS 확장**: 기존의 Model-free 방식인 AIL을 Model-based 설정으로 확장하여, MCTS를 통해 미래의 보상을 예측하고 최적의 행동을 탐색하도록 설계하였다.
2.  **BC와 AIL의 통합**: MCTS의 노드 확장(Expansion) 단계에서 BC 정책에서 샘플링된 행동을 후보군에 포함시킨다. 이를 통해 BC는 탐색을 위한 '거친 가이드(Coarse solution)' 역할을 하고, AIL 보상은 '장기적인 목표(Long-term goal)'를 제시함으로써 두 방법론의 장점을 모두 취한다.
3.  **샘플 효율성 극대화**: EfficientZero의 자기지도 표현 학습(Self-supervised Representation Learning)을 도입하여 이미지 기반의 복잡한 환경에서도 매우 적은 샘플만으로 전문가 수준의 성능에 도달하게 하였다.

## 📎 Related Works

- **Imitation Learning (IL)**: BC는 지도 학습 기반으로 단순하지만 Covariate Shift에 취약하며, IRL(Inverse RL) 및 AIL은 보상 함수를 추론하여 분포를 맞추려 하지만 샘플 효율성이 낮다. 기존 연구들은 off-policy 학습이나 모델 기반 접근(VMAIL 등)을 통해 효율성을 높이려 했으나, BC와 AIL을 효과적으로 통합하는 데는 어려움이 있었다.
- **AIL과 BC의 결합**: 일부 연구에서는 BC로 정책을 초기화하거나 BC 손실 함수를 정규화(Regularization) 용도로 사용하였으나, 이는 AIL 학습 과정에서 BC의 지식이 소실되거나 잘못된 BC 가이드가 탐색을 방해하는 문제가 있었다.
- **Sample Efficient RL**: EfficientZero와 같은 모델 기반 RL은 MCTS와 표현 학습을 통해 샘플 효율성을 획기적으로 높였다. 본 논문은 이러한 RL의 성과를 모방 학습 영역으로 가져와 적용하였다.

## 🛠️ Methodology

### 1. 전체 시스템 구조
EfficientImitate는 다음과 같은 구성 요소로 이루어진 파이프라인을 가진다.
- **World Model**: 상태 $s$를 추상 상태 $h$로 변환하는 Encoder $f$, 다음 상태를 예측하는 Dynamics $g$, 보상을 예측하는 Reward network $R$.
- **Planning Module**: MCTS를 사용하여 현재 상태에서 최적의 행동과 가치를 탐색한다.
- **Policy & Value Networks**: MCTS의 탐색 결과(Target)를 학습하여 실제 추론 시 빠른 의사결정을 가능하게 한다.

### 2. 상세 방법론 및 방정식

#### (1) Multi-step Discriminator 학습 (AIL의 확장)
판별자(Discriminator) $D$가 모델의 추상 상태($h$)에서도 잘 작동하도록, 실제 데이터뿐만 아니라 모델 기반의 롤아웃(Rollout) 데이터를 사용하여 학습한다.
$$L^D = -\mathbb{E}_{(s_t, a_{t:t+n}) \sim B, (s^E_{t'}, a^E_{t':t'+n}) \sim D} \left[ \sum_{i=0}^n \log(D(h_{t+i}, a_{t+i})) + \log(1-D(h^E_{t'+i}, a^E_{t'+i})) \right]$$
여기서 $h$는 Dynamics 네트워크를 통해 생성된 추상 상태이다. 보상 함수는 GAIL 방식을 따라 $R(h, a) = -\log(1-D(h, a))$로 정의된다.

#### (2) BC와 AIL의 통합 (MCTS Expansion)
MCTS의 각 노드에서 행동을 샘플링할 때, 현재의 정책 $\pi$와 BC 정책 $\pi^{BC}$를 혼합한 $\tilde{\pi}$를 사용한다.
$$\tilde{\pi} = \alpha \pi^{BC} + (1-\alpha)\pi$$
여기서 $\alpha$는 혼합 계수(본 논문에서는 0.25 사용)이다. 이렇게 샘플링된 BC 행동들은 MCTS의 Planning 과정을 통해 평가되며, 장기적으로 AIL 보상을 극대화하는 행동일 경우에만 선택된다. 이는 BC가 틀렸을 때는 Planning이 이를 걸러낼 수 있게 하여 BC의 부작용을 방지한다.

#### (3) Multi-step BC Loss
MCTS 내에서의 분포 변화를 방지하기 위해 BC 정책 $\pi^{BC}$를 다음과 같이 다단계 손실 함수로 학습한다.
$$L^{BC} = \mathbb{E}_{(s^E_{t'}, a^E_{t':t'+n}) \sim D} \left[ \sum_{i=0}^n -\log(\pi^{BC}(a^E_{t'+i} | h^E_{t'+i})) \right]$$

#### (4) 최종 최적화 목표
전체 네트워크는 EfficientZero의 손실 함수 $L^{EZ}$와 판별자 손실 $L^D$, BC 손실 $L^{BC}$를 가중치 합산하여 동시에 최적화한다.
$$L = L^{EZ} + \lambda_d L^D + \lambda_{bc} L^{BC}$$

## 📊 Results

### 1. 실험 설정
- **데이터셋**: DeepMind Control Suite (State-based 및 Image-based).
- **평가 지표**: 전문가 성능 대비 정규화된 점수 (0.0 ~ 1.0).
- **비교 대상 (Baselines)**: BC, DAC, SQIL, ValueDICE, VMAIL.
- **샘플 예산**: 태스크 난이도에 따라 10k ~ 500k step으로 제한하여 샘플 효율성을 측정하였다.

### 2. 주요 결과
- **성능 및 효율성**: 모든 태스크에서 baseline 대비 압도적인 성능을 보였다. 특히 state-based 환경에서는 대부분의 태스크에서 전문가 수준의 성능에 빠르게 도달하였으며, 이미지 기반 환경에서도 SOTA(State-of-the-art) 결과를 달성하였다.
- **난이도 높은 태스크 해결**: 기존 방법론들이 적은 샘플로 해결하지 못했던 Humanoid Walk와 같은 복잡한 태스크에서도 성공적인 성능을 보였다.
- **샘플 효율성**: 일부 태스크(Walker, Cheetah)에서는 단 20k step(약 80개 궤적)만으로 전문가 수준에 도달하여, 실제 로봇 적용 가능성을 시사하였다.

## 🧠 Insights & Discussion

### 1. 강점 및 분석
- **BC 행동의 효과**: Ablation study를 통해 BC 행동을 MCTS에 넣지 않았을 때($\alpha=0$), 특히 Humanoid와 같은 고차원 태스크에서 Local Minima에 빠지는 현상이 확인되었다. 이는 BC가 탐색 공간을 효과적으로 좁혀주는 가이드 역할을 수행함을 증명한다.
- **Planning의 중요성**: 단순히 BC 손실을 정규화(Regularization)로 사용하는 방식(BC-Ann, BC-Reg)보다 MCTS 내에서 행동 후보로 사용하는 것이 훨씬 효과적이었다. 이는 Planning이 특정 행동의 '장기적 결과'를 계산할 수 있기 때문에 가능한 구조이다.
- **MCTS 파라미터**: 시뮬레이션 횟수 $N$이 성능에 큰 영향을 미치며, 샘플링 행동 수 $K$보다는 $N$을 늘리는 것이 더 효과적임이 밝혀졌다.

### 2. 한계점 및 비판적 해석
- **계산 비용**: MCTS 기반 방식은 Model-free 방식에 비해 추론 및 학습 시 계산 비용이 훨씬 높다. 이는 실시간 제어 시스템에 적용할 때 병목 현상이 될 수 있다.
- **객체 상호작용**: 본 연구는 단일 에이전트의 제어에 집중하였으며, 여러 객체와 상호작용하는 복잡한 환경(Robotic Manipulation 등)에 대한 검증은 부족하다.

## 📌 TL;DR

본 논문은 MCTS 기반의 Planning을 도입하여 모방 학습의 고질적인 문제인 **'샘플 효율성'**과 **'최종 성능'**을 동시에 잡은 **EfficientImitate (EI)**를 제안한다. 핵심은 AIL을 Model-based 설정으로 확장하고, MCTS 탐색 과정에 BC 행동을 후보로 넣어 BC의 가이드와 AIL의 목표 지향성을 통합한 것이다. 실험 결과, 이미지 기반 환경을 포함한 다양한 제어 태스크에서 기존 방법론 대비 4배 이상의 성능 향상과 획기적인 샘플 효율성을 입증하였다. 이 연구는 향후 실제 로봇의 데이터 효율적 학습에 중요한 기여를 할 것으로 보인다.