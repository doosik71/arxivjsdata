# RIZE: Adaptive Regularization for Imitation Learning

Adib Karimi, Mohammad Mehdi Ebadzadeh (2025)

## 🧩 Problem to Solve

본 논문은 모방 학습(Imitation Learning, IL) 및 역강화학습(Inverse Reinforcement Learning, IRL)에서 보상 함수(reward function)를 설계할 때 발생하는 고정된 보상 구조의 경직성과 암시적 보상 정규화(implicit reward regularization)의 제한된 유연성 문제를 해결하고자 한다.

일반적으로 강화학습에서는 사람이 직접 보상 함수를 설계해야 하는데, 이는 도메인 전문가의 많은 시간과 노력이 필요하며 확장성이 떨어진다는 단점이 있다. 이를 해결하기 위해 전문가의 시연에서 보상을 추론하는 IRL 방식이 제안되었으며, 최근에는 보상 함수를 명시적으로 학습하지 않고 Q-값(Q-values)을 통해 보상을 암시적으로 표현하는 방식(예: IQ-Learn, LSIQ)이 주목받고 있다.

그러나 기존의 암시적 보상 정규화 방식(특히 LSIQ 등)은 전문가 샘플에는 $+1$, 에이전트 샘플에는 $-1$과 같이 **고정된 타겟(fixed targets)**을 할당한다. 이러한 접근 방식은 모든 태스크와 상태-행동 쌍을 동일하게 취급하므로 유연성이 부족하며, 이로 인해 복잡한 환경에서 성능이 제한되고 수렴 속도가 느려지는 문제가 발생한다. 따라서 본 논문의 목표는 학습 과정에서 동적으로 변화하는 **적응형 타겟(adaptive targets)**을 도입하여 보상의 범위를 유연하게 조정하고, 학습의 안정성과 성능을 높이는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 크게 두 가지 설계 아이디어로 요약된다.

1. **적응형 타겟(Adaptive Targets) 기반의 정규화**: 고정된 상숫값 대신 학습 가능한 타겟 $\lambda_{\pi_E}$(전문가용)와 $\lambda_{\pi}$(정책용)를 도입하였다. 이 타겟들은 학습 과정에서 동적으로 업데이트되어 보상이 과도하게 증가하거나 감소하는 것을 방지하는 적응형 경계(adaptive bounds) 역할을 수행하며, 이를 통해 정책 학습의 안정성을 확보한다.
2. **분포 강화학습(Distributional RL)의 통합**: 단일 값의 기대치인 $Q(s, a)$ 대신 리턴의 분포 $Z^\pi(s, a)$를 학습하는 Implicit Quantile Network (IQN)를 도입하였다. 이를 통해 환경의 불확실성을 더 풍부하게 캡처할 수 있으며, 결과적으로 더 강건한(robust) 의사결정이 가능하게 한다.

## 📎 Related Works

본 논문은 다음과 같은 기존 연구들의 흐름 속에서 차별점을 갖는다.

- **행동 복제(Behavior Cloning, BC)**: 상태에서 행동으로의 매핑을 직접 학습하는 단순한 방식이나, 분포 변화(covariate shift)로 인한 오류 누적 문제가 심각하다.
- **Maximum Entropy (MaxEnt) IRL 및 적대적 학습(Adversarial Training)**: GAIL과 같은 방식은 전문가와 에이전트의 분포 차이를 줄이는 적대적 게임 형태로 학습하지만, 학습 과정이 매우 불안정하다는 단점이 있다.
- **암시적 보상 학습(Implicit Reward Learning)**: IQ-Learn과 LSIQ는 적대적 학습을 피하기 위해 Bellman 방정식을 역전시켜 보상을 암시적으로 표현한다. 특히 LSIQ는 $\chi^2$ 발산을 최소화하여 squared TD-error 목적 함수를 사용하지만, 앞서 언급한 대로 **고정된 타겟**을 사용한다는 한계가 있다.

**RIZE의 차별점**은 이러한 암시적 보상 체계에 **학습 가능한 적응형 타겟**을 결합하여 보상의 범위를 이론적으로 제어하고, 여기에 **분포 강화학습(Distributional RL)**을 접목하여 비적대적 설정에서도 높은 성능과 안정성을 달성했다는 점이다.

## 🛠️ Methodology

### 1. 전체 파이프라인 및 시스템 구조

RIZE는 MaxEnt IRL 프레임워크를 기반으로 하며, **분포 기반 크리틱(Distributional Critic)**과 **적응형 정규화 기반의 보상 학습**, 그리고 **SAC(Soft Actor-Critic) 기반의 정책 최적화**로 구성된다.

### 2. 주요 구성 요소 및 상세 설명

#### (1) Distributional Value Integration

단순한 $Q$값 대신 리턴의 분위수 함수(quantile function) $Z^\tau(s, a)$를 학습한다. 정책 최적화에 필요한 $Q$값은 $Z$의 기대값으로 계산한다.
$$Q(s, a) = \sum_{i=0}^{N-1} (\tau_{i+1} - \tau_i) Z^{\tau_i}(s, a)$$
여기서 $\tau_i$는 분위수 수준을 나타내며, 이를 통해 리턴의 전체 분포 정보를 활용하여 더 안정적인 학습 신호를 얻는다.

#### (2) Implicit Reward 및 적응형 정규화

암시적 보상 $R^Q(s, a)$는 다음과 같이 정의된다.
$$R^Q(s, a) = Q(s, a) - \gamma \mathbb{E}_{P, \pi} [Q(s', a') - \alpha \log \pi(a'|s')]$$

RIZE는 이 $R^Q$에 대해 다음과 같은 적응형 정규화 항 $\Gamma$를 도입한다.
$$\Gamma(R^Q, \lambda) = \mathbb{E}_{\rho_E} [(R^Q(s, a) - \lambda_{\pi_E})^2] + \mathbb{E}_{\rho_\pi} [(R^Q(s, a) - \lambda_\pi)^2]$$
여기서 $\lambda_{\pi_E}$와 $\lambda_{\pi}$는 고정된 값이 아니라, 각각 전문가와 에이전트의 보상 추정치에 맞춰 동적으로 업데이트되는 학습 가능한 파라미터이다.

#### (3) 학습 목적 함수 및 절차

전체 손실 함수 $L(\pi, Q)$는 전문가와 에이전트의 보상 차이를 최대화하면서 정규화 항을 통해 보상 범위를 제한하는 형태를 띤다.
$$L(\pi, Q) = \mathbb{E}_{\rho_E} [R^Q(s, a)] - \mathbb{E}_{\rho_\pi} [R^Q(s, a)] - \alpha H(\pi) - c \Gamma(R^Q, \lambda)$$
여기서 $c$는 정규화 계수이며, $\lambda$ 값들은 다음의 목적 함수를 통해 업데이트된다.
$$\min_{\lambda_{\pi_E}} \mathbb{E}_{\rho_E} [(R^Q(s, a) - \lambda_{\pi_E})^2], \quad \min_{\lambda_\pi} \mathbb{E}_{\rho_\pi} [(R^Q(s, a) - \lambda_\pi)^2]$$

### 3. 이론적 분석 (Reward Bounding)

논문은 적응형 타겟을 사용할 때 최적의 암시적 보상 $R^*_Q$가 다음 범위 내에 존재함을 증명하였다 (Corollary 4.2).
$$R^*_Q(s, a) \in \left[ -\frac{1}{2c} + \lambda_{min}, \frac{1}{2c} + \lambda_{max} \right]$$
여기서 $\lambda_{min} = \min\{\lambda_{\pi_E}, \lambda_\pi\}$, $\lambda_{max} = \max\{\lambda_{\pi_E}, \lambda_\pi\}$이다. 이는 적응형 타겟이 보상 값을 특정 범위 내로 묶어둠으로써 크리틱의 업데이트를 안정화하고, 결과적으로 정책 최적화의 안정성을 보장함을 의미한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋 및 작업**: MuJoCo(HalfCheetah, Walker2d, Ant, Humanoid, Hopper) 및 Adroit(Hammer) 벤치마크.
- **비교 대상**: BC, SQIL, IQ-Learn, LSIQ, CSIL.
- **평가 지표**: 전문가 성능 대비 정규화된 리턴(Normalized Return), RLiable 지표(Median, IQM, Mean, Optimality Gap).
- **시나리오**: 전문가 시연 데이터 수를 3개와 10개로 설정하여 테스트.

### 2. 주요 결과

- **종합 성능**: RIZE는 대부분의 작업에서 baseline들보다 높은 Median, IQM, Mean 리턴을 기록했으며, Optimality Gap(최적성 간극)을 유의미하게 줄였다.
- **복잡한 작업 수행 능력**: 특히 **Humanoid-v2** 작업에서 모든 baseline이 실패한 반면, RIZE만이 성공적으로 전문가 수준의 성능을 달성하였다. Hammer-v1 작업에서도 RIZE, CSIL, BC가 상대적으로 우수한 성능을 보였다.
- **데이터 효율성**: 전문가 시연이 3개뿐인 극소량의 데이터 설정에서도 RIZE는 다른 방법들보다 안정적인 학습 곡선을 보여주었으며, 데이터가 10개로 늘어남에 따라 성능이 일관되게 향상되었다.

### 3. 분석 및 절제 연구(Ablation Study)

- **크리틱 구조 (IQN vs Q-Net)**: 분포 기반 크리틱($Z$)을 사용했을 때, 단일 값 크리틱($Q$)을 사용했을 때보다 분산이 낮고 샘플 효율성이 높았다. 특히 Humanoid-v2와 Hammer-v1 같은 복잡한 작업에서는 IQN 크리틱만이 전문가 수준의 성능을 낼 수 있었다.
- **보상 궤적 분석**: 실제 학습 과정에서 회복된 보상 값들이 이론적으로 예측된 범위 $[-\frac{1}{2c} + \lambda_\pi, \frac{1}{2c} + \lambda_{\pi_E}]$ 내에서 안정적으로 유지됨을 확인하였다. 반면, IQ-Learn이나 SQIL은 보상 값이 발산하거나 표류(drift)하는 경향을 보였다.

## 🧠 Insights & Discussion

### 강점 및 유효성

본 연구는 암시적 보상 학습에서 **'보상의 범위 제어'**가 학습 안정성에 얼마나 중요한지를 입증하였다. 단순히 값을 클리핑(clipping)하는 LSIQ 방식과 달리, 적응형 타겟을 통해 태스크의 특성에 맞춰 보상 경계를 유연하게 조정함으로써 복잡한 고차원 제어 작업(Humanoid 등)에서도 성공적인 학습을 이끌어냈다. 또한, Distributional RL을 통해 리턴의 불확실성을 모델링한 것이 단순한 점 추정 방식보다 훨씬 강력한 학습 신호를 제공함을 보여주었다.

### 한계 및 논의사항

- **수렴성 증명**: IQ-Learn의 수렴성 보장이 $\chi^2$-정규화(squared TD-error)로 확장되지 않는다는 점이 언급되었으며, RIZE의 교대 업데이트(alternating updates)에 대한 공식적인 수렴성 증명은 향후 과제로 남겨두었다.
- **하이퍼파라미터 민감도**: 적응형 타겟 $\lambda_\pi$의 초기값과 학습률(learning rate)에 대해 다소 민감한 모습을 보인다. 특히 $\lambda_\pi$는 매우 천천히 업데이트해야 학습이 안정된다는 점이 실험적으로 밝혀졌다.

## 📌 TL;DR

RIZE는 고정된 보상 타겟의 경직성을 해결하기 위해 **학습 가능한 적응형 타겟(Adaptive Targets)**과 **분포 강화학습(Distributional RL)**을 결합한 새로운 IRL 프레임워크이다. 이론적으로 보상 범위를 제한하여 학습 안정성을 확보하였으며, 특히 기존 방법들이 해결하지 못한 **Humanoid-v2**와 같은 복잡한 환경에서 전문가 수준의 성능을 달성하였다. 이 연구는 암시적 보상 정규화와 불확실성 인지 리턴 표현의 결합이 고차원 모방 학습의 효율성과 강건성을 크게 향상시킬 수 있음을 시사한다.
