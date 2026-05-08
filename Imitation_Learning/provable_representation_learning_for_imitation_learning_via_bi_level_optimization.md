# Provable Representation Learning for Imitation Learning via Bi-level Optimization

Sanjeev Arora, Simon S. Du, Sham Kakade, Yuping Luo, and Nikunj Saunshi (2020)

## 🧩 Problem to Solve

본 논문은 모방 학습(Imitation Learning, IL) 분야에서 새로운 작업을 학습할 때 요구되는 방대한 양의 데이터(Sample Complexity) 문제를 해결하고자 한다. 일반적으로 현대의 학습 시스템은 여러 작업에 유용하게 사용될 수 있는 공통의 표현(Representation)을 학습하는 Representation Learning 전략을 사용하지만, 모방 학습 설정에서 이러한 접근 방식이 이론적으로 얼마나 유효하며 구체적으로 어느 정도의 샘플 효율성 이득을 제공하는지에 대한 증명은 부족한 상태였다.

특히, 서로 다른 보상 함수(Reward Function)나 전이 함수(Transition Function)를 가진 여러 전문가의 궤적(Trajectories)이 존재할 때, 이를 통해 공통의 표현을 학습함으로써 새로운 목표 작업(Target Task)에 필요한 전문가 시연 횟수를 획기적으로 줄이는 것을 목표로 한다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 표현 학습 문제를 **Bi-level Optimization(이단계 최적화)** 프레임워크로 공식화한 것이다.

1. **Bi-level Optimization 프레임워크 제안**:
    * **Outer Optimization**: 모든 작업에 걸쳐 공유되는 최적의 joint representation $\phi$를 학습한다.
    * **Inner Optimization**: 고정된 $\phi$를 바탕으로 각 작업에 특화된 파라미터(Task-specific parameters)를 학습하여 전문가의 정책을 모방한다.
2. **이론적 보장(Provable Guarantees) 제공**: Behavior Cloning(BC)과 Observation-Alone(OA)이라는 두 가지 서로 다른 모방 학습 설정에서, 충분한 수의 전문가 데이터가 있다면 표현 학습이 새로운 작업의 샘플 복잡도를 이론적으로 감소시킴을 증명하였다.
3. **실험적 검증**: 제안한 프레임워크를 통해 학습된 표현이 실제 환경(NoisyCombinationLock, SwimmerVelocity)에서 데이터 효율성을 높이며, 나아가 강화 학습(RL)의 정책 최적화 단계에서도 유용하게 사용될 수 있음을 확인하였다.

## 📎 Related Works

기존의 표현 학습 연구는 주로 지도 학습(Supervised Learning) 기반의 multi-task representation learning에 집중되어 왔으며, Maurer et al. [2016] 등이 샘플 복잡도 감소에 대한 이론적 분석을 수행한 바 있다.

모방 학습 분야에서는 Behavior Cloning(BC)이나 DAgger와 같은 방법론들이 제시되었으며, 최근에는 MAML과 같은 Meta-learning 기법을 통해 새로운 작업에 빠르게 적응하려는 시도가 있었다. 그러나 본 논문은 다음과 같은 차별점을 가진다.

* **명시적 표현 학습**: MAML과 같은 메타 알고리즘(Meta-algorithm) 학습이 아니라, 모든 작업이 공유하는 명시적인 Representation $\phi$를 학습하는 것에 집중한다.
* **이론적 증명**: 기존의 Meta-IL 연구들은 경험적인 성능 향상은 보여주었으나, 본 논문처럼 표현 학습이 샘플 복잡도를 구체적으로 어떻게 줄이는지에 대한 이론적 보장(Provable guarantees)을 제공하지 않았다.
* **일반적 손실 함수**: 볼록(Convex) 손실 함수에 국한되지 않고 일반적인 함수 클래스에 대해 Gaussian average를 이용한 분석을 수행하였다.

## 🛠️ Methodology

### 전체 시스템 구조 및 표현 정의

학습하고자 하는 정책 $\pi$는 표현 함수 $\phi \in \Phi$와 정책 함수 $f \in F$의 합성 함수 형태인 $\Pi = F \circ \Phi$로 정의된다. 여기서 $\phi: S \to \mathbb{R}^d$는 상태를 벡터로 매핑하며, $f: \mathbb{R}^d \to \Delta(A)$는 이 벡터를 기반으로 행동 분포를 생성한다. 본 논문에서는 $f$를 선형 함수 $\text{softmax}(Wx)$ 형태로 가정한다.

### Bi-level Optimization 공식화

표현 학습 문제는 다음과 같은 이단계 최적화 식으로 정의된다.

$$\min_{\phi \in \Phi} L(\phi) := \mathbb{E}_{\mu \sim \eta} \left[ \min_{\pi \in \Pi_\phi} \ell_\mu(\pi) \right]$$

여기서 $\ell_\mu$는 전문가 정책 $\pi^*_\mu$와 학습 정책 $\pi$ 사이의 차이를 측정하는 내적 손실 함수(Inner loss)이며, 외적 손실 함수 $L(\phi)$는 모든 작업 $\mu$에 대한 내적 손실의 기댓값을 최소화하는 $\phi$를 찾는 것이다. 실제 학습 시에는 $T$개의 작업 샘플을 이용한 경험적 손실 $\hat{L}(\phi)$를 최소화하여 $\hat{\phi}$를 구한다.

### 상세 설정 및 손실 함수

#### 1. Behavior Cloning (BC) 설정

에이전트가 전문가의 행동($a$)을 관찰할 수 있는 설정이다.

* **손실 함수**: 전문가의 상태-행동 분포 $\mu$에 대해 logistic loss를 사용한다.
    $$\ell_\mu(\pi) = \mathbb{E}_{(s,a) \sim \mu} [-\log(\pi(s)_a)]$$
* **학습 절차**: $T$개의 작업에서 수집된 $\{s, a\}$ 쌍을 이용하여 $\hat{\phi}$를 학습하고, 새로운 작업에서는 이 $\hat{\phi}$를 고정시킨 채 선형 레이어 $W$만을 학습하여 정책을 도출한다.

#### 2. Observation-Alone (OA) 설정

전문가의 상태($s$)만 관찰할 수 있고 행동은 알 수 없는 더 어려운 설정이다.

* **손실 함수**: 전문가가 유도하는 상태 분포 $\nu^*$와 학습 정책이 유도하는 상태 분포 $\nu^\pi$를 일치시키기 위해 판별자 클래스 $G$를 이용한 min-max 문제로 정의한다.
    $$\ell_{\mu, h}(\pi) = \max_{g \in G} \left[ \mathbb{E}_{s \sim \nu^*_{h, \mu}, a \sim \pi, \tilde{s} \sim P} g(\tilde{s}) - \mathbb{E}_{\bar{s} \sim \nu^*_{h+1, \mu}} g(\bar{s}) \right]$$
* **특이사항**: 이 설정에서는 샘플을 수집하기 위해 에이전트가 환경과 상호작용하며 데이터를 수집하는 과정이 필요하다.

## 📊 Results

### 실험 환경 및 지표

* **환경**: `NoisyCombinationLock`(노이즈가 섞인 금고 잠금 해제) 및 `SwimmerVelocity`(수영 로봇의 목표 속도 도달).
* **기준선(Baseline)**: 표현 학습 없이 각 작업마다 처음부터 정책 $\pi \in \Pi$를 학습하는 방식.
* **지표**: 평균 리턴(Average Return) 및 목표 작업에 필요한 궤적(Trajectories)의 수.

### 주요 결과

1. **이론 검증 (Figure 1)**: BC와 OA 설정 모두에서 학습에 참여한 전문가의 수($T$)가 많아질수록, 새로운 작업에서 필요한 데이터의 양이 줄어들며 성능(리턴)이 향상됨을 확인하였다.
2. **샘플 효율성**: 전문가 수가 충분할 때, 표현 학습을 사용한 방식은 Baseline보다 훨씬 적은 수의 궤적만으로도 유사하거나 더 높은 성능에 도달하였다.
3. **강화 학습으로의 확장 (Figure 2)**: 모방 학습을 통해 학습된 표현 $\hat{\phi}$를 PPO(Proximal Policy Optimization) 알고리즘의 입력으로 사용했을 때, 처음부터 학습한 Baseline보다 훨씬 적은 샘플로 더 높은 리턴을 얻었다. 이는 학습된 표현이 단순히 모방을 넘어 일반적인 제어 성능 향상에 기여함을 시사한다.

## 🧠 Insights & Discussion

### 강점 및 이론적 함의

본 논문은 Representation Learning이 모방 학습의 샘플 복잡도를 줄일 수 있음을 수학적으로 증명하였다. 특히 BC 설정에서 새로운 작업에 필요한 샘플 수 $n$이 표현 함수 클래스 $\Phi$의 복잡도가 아닌, 액션 공간의 크기 $K$와 관련된 상수에 의존하게 됨으로써, $\Phi$가 매우 복잡하더라도 다수의 작업을 통해 이를 미리 학습해 두면 개별 작업의 학습 비용을 획기적으로 낮출 수 있음을 보였다.

### 한계 및 해석

실험 결과에서 전문가의 수가 매우 적을 때는 Baseline이 오히려 더 높은 성능을 보이는 구간이 존재한다. 이는 불충분한 데이터로 학습된 표현 $\hat{\phi}$가 오히려 sub-optimal한 제약 조건으로 작용하기 때문으로 해석된다. 즉, 표현 학습의 이득을 얻기 위해서는 최소한의 임계치 이상의 작업 수($T$)가 확보되어야 한다. 또한, OA 설정의 경우 전문가의 행동 정보를 알 수 없으므로 BC보다 낮은 리턴을 기록했으며, 이는 손실 함수의 비관적(pessimistic) 특성과 정보 부족에 기인한 것으로 분석된다.

## 📌 TL;DR

본 논문은 모방 학습에서 여러 전문가의 데이터를 활용해 공통의 표현을 학습하는 문제를 **Bi-level Optimization**으로 정의하고, 이를 통해 새로운 작업의 샘플 복잡도를 낮출 수 있음을 이론적으로 증명하고 실험적으로 검증하였다. 특히 학습된 표현은 모방 학습뿐만 아니라 강화 학습의 정책 최적화 효율을 높이는 데에도 유효함이 입증되어, 향후 데이터 효율적인 에이전트 설계에 중요한 이론적 토대를 제공한다.
