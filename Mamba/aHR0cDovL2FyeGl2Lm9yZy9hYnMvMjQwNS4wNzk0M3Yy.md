# Decision Mamba Architectures

André Correia, Luís A. Alexandre (2024)

## 🧩 Problem to Solve

본 논문은 Offline Reinforcement Learning(RL) 및 Imitation Learning(IL)에서 시퀀스 모델링을 통해 정책을 학습하는 기존 방식의 한계점을 해결하고자 한다. 특히, Decision Transformer(DT)와 그 확장형인 Hierarchical Decision Transformer(HDT)가 가진 다음과 같은 문제들에 집중한다.

첫째, Decision Transformer(DT)는 미래에 획득할 보상의 합계인 Returns-to-Go(RTG) 시퀀스에 크게 의존한다. 그러나 RTG를 설정하기 위해서는 각 작업에 특화된 정교한 보상 지식이 필요하며, 이는 사용자 개입을 야기하고 결정론적이지 않은 확률적 환경(stochastic environments)에서 성능 저하를 일으키는 원인이 된다.

둘째, Transformer 아키텍처는 컨텍스트 윈도우의 크기에 따라 계산 복잡도가 이차적으로 증가하는 Quadratic Scaling 문제를 가지고 있어, 긴 시퀀스를 처리할 때 효율성이 떨어진다.

따라서 본 연구의 목표는 Transformer를 대체하여 선형적 확장성을 가지면서도, 보상 시퀀스 없이도 효과적인 가이드 신호를 제공할 수 있는 Mamba 아키텍처 기반의 Decision Mamba(DM)와 Hierarchical Decision Mamba(HDM)를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Transformer 기반의 의사결정 모델을 State Space Model(SSM)의 최신 진화 형태인 Mamba 아키텍처로 대체한 것이다. 중심적인 직관은 Mamba의 진화 파라미터(evolutionary parameter)가 DT에서 RTG가 제공하던 가이드 신호의 역할을 대신할 수 있다는 점이다. 이를 통해 다음과 같은 성과를 거두었다.

1. **Decision Mamba(DM) 및 Hierarchical Decision Mamba(HDM) 제안**: 기존 DT와 HDT를 Mamba 아키텍처로 재설계하여 모델의 크기를 줄이고 속도를 높이면서도 정확도를 향상시켰다.
2. **보상 시퀀스 의존성 제거**: DM은 DT와 달리 RTG 시퀀스 없이도 작업 수행이 가능함을 보였다. 이는 보상 함수나 작업에 대한 사전 지식, 사용자의 개입 없이도 모방 학습이 가능함을 의미한다.
3. **성능 및 효율성 검증**: D4RL 벤치마크의 7가지 환경에서 실험을 진행하여, 대부분의 설정에서 Mamba 기반 모델이 Transformer 기반 모델보다 우수한 성능을 보이며 특히 추론 속도에서 이점이 있음을 입증하였다.

## 📎 Related Works

기존의 Offline RL은 데이터셋으로부터 직접 정책을 복제하는 Behavioural Cloning(BC) 방식에서 시작되었으나, 이는 데이터 품질에 의존적이며 compounding error 문제가 발생한다. 이를 해결하기 위해 DT는 RL 문제를 시퀀스 모델링 문제로 재정의하여 과거의 궤적과 RTG를 조건으로 액션을 예측하게 하였다. 

하지만 DT의 RTG 의존성 문제는 여러 후속 연구에서 다뤄졌다. 일부 연구는 가치 함수(Value function)를 미리 학습하여 RTG를 대체하거나, HDT와 같이 상위 모델이 하위 컨트롤러에게 서브골(sub-goal)을 제공하는 계층적 구조를 채택하였다. 그러나 이러한 방식들은 대개 추가적인 모델을 필요로 하거나 데이터 전처리 과정이 복잡하다는 한계가 있다.

본 논문은 이러한 맥락에서 최근 주목받는 Mamba 아키텍처를 도입한다. Mamba는 Transformer의 문맥 의존적 추론 능력과 SSM의 선형적 확장성을 동시에 갖추고 있어, 기존 Transformer 기반 RL 모델들의 효율성과 의존성 문제를 동시에 해결할 수 있는 대안으로 제시된다.

## 🛠️ Methodology

### 1. Structured State Space Sequence Models (SSM)
Mamba의 기반이 되는 SSM은 1차원 함수 $x(t)$를 은닉 상태 $h(t)$를 통해 $y(t)$로 매핑하는 연속 시스템이다. 이는 다음과 같은 선형 상미분 방정식(ODE)으로 표현된다.

$$h'(t) = Ah(t) + Bx(t), \quad y(t) = Ch(t)$$

여기서 $A$는 진화 파라미터(evolutionary parameter), $B$와 $C$는 투영 파라미터(projection parameters)이다. 딥러닝 적용을 위해 Zero-Order Hold(ZOH) 방식을 통한 이산화(discretization) 과정을 거치며, 타임스케일 파라미터 $\Delta$를 사용하여 이산 파라미터 $\bar{A}, \bar{B}$를 생성한다.

$$\bar{A} = \exp(\Delta A), \quad \bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I)\Delta B$$

최종 출력 $y(t)$는 구조화된 컨볼루션 커널 $\bar{K}$를 이용한 전역 컨볼루션 $\bar{K} = (C \bar{B}, C \bar{A} \bar{B}, \dots, C \bar{A}^{K-1} \bar{B})$를 통해 계산된다.

### 2. Decision Mamba (DM)
DM은 DT의 구조를 계승하되 Transformer를 Mamba 레이어로 교체한 모델이다.

- **입력 구성**: 상태(states), 액션(actions), 그리고 선택적으로 RTG의 시퀀스를 입력으로 받는다. 
- **구조적 단순화**: Mamba는 자체적으로 시퀀스 정보를 처리하므로, Transformer에서 필수적이었던 Causal Mask와 Positional Encoding이 필요 없다.
- **파이프라인**: 각 입력 시퀀스는 개별 선형 레이어를 통해 임베딩 차원으로 변환된 후 결합되어 Mamba 레이어들을 통과한다. 최종 출력은 Feed-forward 레이어와 $\tanh$ 활성화 함수를 거쳐 액션 공간으로 투영된다.
- **학습 목표**: 예측된 액션 시퀀스와 실제 정답 액션 시퀀스 간의 $L2$ 손실 함수를 최소화하는 것을 목표로 한다.

### 3. Hierarchical Decision Mamba (HDM)
HDM은 의사결정 과정을 상위 메커니즘(High-level mechanism)과 하위 컨트롤러(Low-level controller)로 분리한다.

- **상위 메커니즘**: 과거의 상태와 서브골 시퀀스를 입력받아, 하위 컨트롤러가 도달해야 할 다음 서브골 상태($s_g$)를 예측한다.
- **하위 컨트롤러**: 과거의 상태, 액션, 그리고 상위 모델이 제공한 서브골 시퀀스를 조건으로 하여 최적의 액션을 예측한다.
- **학습 과정**: 상위 모델은 예측 서브골과 실제 서브골 간의 $L2$ 손실로, 하위 모델은 예측 액션과 실제 액션 간의 $L2$ 손실로 각각 학습된다.

## 📊 Results

### 실험 설정
- **데이터셋**: D4RL 벤치마크의 7가지 작업(Ant, AntMaze, HalfCheetah, Hopper, Kitchen, Maze2D, Walker2D)을 사용하였으며, Expert, Medium, Mixed 등 다양한 품질의 데이터셋을 적용하였다.
- **비교 대상**: DT, HDT, DM (with/without Reward), HDM.
- **평가 지표**: 누적 보상(Accumulated Returns), 학습 및 추론 시간.
- **하이퍼파라미터**: 6개 레이어, 임베딩 크기 128, 컨텍스트 길이 20, 학습 횟수 1M 에포크.

### 주요 결과
1. **정량적 성능 (Table 1)**:
   - DM(보상 없음)은 27개 설정 중 15개에서 DT보다 우수한 성능을 보였다.
   - HDM은 27개 설정 중 22개에서 HDT보다 뛰어난 성과를 거두었다.
   - 특히 DM은 RTG 시퀀스가 없을 때 오히려 더 좋은 성능을 내는 경향이 확인되었으며, 이는 Mamba의 구조적 특성이 보상 정보 없이도 충분한 가이드를 제공함을 시사한다.

2. **시간 효율성 (Table 2)**:
   - **학습 시간**: Mamba 기반 모델과 Transformer 기반 모델 간의 큰 차이는 없었으나, 계층적 구조(HDT, HDM)는 모델이 두 개이므로 학습 시간이 약 2배 더 소요되었다.
   - **추론 시간**: Mamba 기반 모델(DM, HDM)이 Transformer 기반 모델(DT, HDT)보다 일관되게 빠른 추론 속도를 보였다.

## 🧠 Insights & Discussion

본 연구의 가장 중요한 발견은 **Mamba 아키텍처가 DT의 치명적인 약점인 RTG 의존성을 해결할 수 있다**는 점이다. DT에서는 RTG를 제거하면 학습이 불가능해지지만, DM은 RTG 없이도, 혹은 RTG가 있을 때보다 더 나은 성능을 보였다. 이는 SSM의 진화 파라미터 $A$가 시퀀스의 흐름을 제어하며 보상 시퀀스가 수행하던 '목표 지향적 가이드' 역할을 내부적으로 수행하고 있음을 암시한다.

또한, 계산 복잡도 측면에서 Mamba의 선형 확장성은 실시간 제어가 중요한 로보틱스 환경에서 큰 강점이 된다. 추론 속도가 향상되었다는 점은 실제 배포 시의 효율성을 보장한다.

다만, 실험 결과에서 모든 작업에 대해 일관되게 최적인 단일 하이퍼파라미터 구성(레이어 수, 임베딩 크기 등)을 찾지 못했다는 점은 모델의 설정이 환경 및 데이터셋의 특성에 민감할 수 있음을 나타낸다. 또한, HDM이 HDT보다 성능은 좋지만 여전히 두 개의 모델을 학습시켜야 하고 데이터 전처리가 필요하다는 점은 단순한 DM 구조에 비해 운영 비용이 높다는 한계가 있다.

## 📌 TL;DR

본 논문은 Decision Transformer의 보상 의존성 및 계산 효율성 문제를 해결하기 위해 Mamba 아키텍처를 도입한 **Decision Mamba(DM)**와 **Hierarchical Decision Mamba(HDM)**를 제안하였다. 실험 결과, DM은 보상 함수나 RTG 시퀀스 없이도 DT보다 우수한 성능을 보였으며, HDM 또한 HDT를 상회하는 성과를 거두었다. 특히 Mamba 모델들은 Transformer 대비 빠른 추론 속도를 제공한다. 이 연구는 Mamba가 모방 학습의 시퀀스 모델링에서 매우 유망한 아키텍처임을 입증하였으며, 향후 보상 함수 설계가 어려운 복잡한 제어 문제 해결에 기여할 가능성이 높다.