# ProgAgent: A Continual RL Agent with Progress-Aware Rewards

Jinzhou Tan, Gabriel Adineera, Jinoh Kim (2026)

## 🧩 Problem to Solve

본 논문은 로봇의 평생 학습(Lifelong Learning)을 위한 지속적 강화학습(Continual Reinforcement Learning, CRL)에서 발생하는 두 가지 핵심 문제를 해결하고자 한다.

첫째는 **치명적 망각(Catastrophic Forgetting)** 문제이다. 로봇이 새로운 작업을 학습할 때 이전에 습득한 지식이 덮어씌워져 과거의 능력을 상실하는 현상으로, 이는 로봇의 장기적인 자율성을 저해하는 주요 원인이 된다.

둘째는 **보상 설계의 어려움(Reward Specification Problem)**이다. 다양한 조작 작업에 대해 밀집된(dense) 보상 함수를 수동으로 설계하는 것은 막대한 노동력이 필요하며, 실제 환경에서는 사실상 불가능에 가깝다. 이는 벤치마크 수준을 넘어선 실제 확장성을 제한하는 병목 현상이 된다.

결과적으로 본 연구의 목표는 전문가 비디오로부터 자동으로 밀집된 보상을 유도하고, 시스템 수준의 최적화를 통해 망각을 최소화하면서 새로운 기술을 빠르게 습득할 수 있는 통합 CRL 에이전트를 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 알고리즘적 혁신과 시스템적 최적화를 결합하여 **Stability-Plasticity(안정성-가소성)**의 균형을 맞춘 점에 있다.

1. **Progress-Aware Reward Model**: 액션 라벨이 없는 전문가 비디오에서 작업 진행도를 예측하는 모델을 학습시킨다. 이를 이론적으로 상태-포텐셜 함수(State-Potential Function)로 해석하여, 전문가의 행동과 정렬된 밀집 보상을 생성함으로써 정책 최적화 속도를 높인다.
2. **Adversarial Push-back Refinement**: 온라인 탐색 과정에서 발생하는 분포 변화(Distribution Shift)에 대응하기 위해, 전문가 궤적에서 벗어난 상태에 대해 보상 모델이 과신(overconfident)하지 않도록 규제하는 적대적 정제 메커니즘을 도입하였다.
3. **Unified JAX-native Architecture**: 보상 학습, 데이터 수집, 정책 최적화 전체 루프를 JIT(Just-In-Time) 컴파일하고 벡터화($vmap$)하여 초고속 병렬 처리를 가능케 하였다. 이를 통해 계산 비용이 높은 복잡한 CRL 알고리즘(Coreset Replay, Synaptic Intelligence 등)을 효율적으로 구현하고 통합하였다.

## 📎 Related Works

본 연구는 다음 세 가지 분야의 접점에 위치한다.

- **Continual RL**: 기존 연구들은 매개변수 규제(EWC, SI)나 리허설 버퍼(Experience Replay)를 통해 망각을 방지하려 했다. 그러나 이러한 방법들은 종종 안정성과 가소성 사이의 상충 관계(trade-off)를 해결하지 못하거나 계산 비용이 매우 높다는 한계가 있었다.
- **Learning Rewards from Perception**: 전문가 비디오에서 보상을 유도하는 Rank2Reward, TCN 등의 연구가 있었으나, 이들은 온라인 탐색 중 발생하는 Out-of-distribution(OOD) 상태에서 잘못된 보상을 주는 '보상 해킹' 문제에 취약했다.
- **High-Throughput RL Systems**: Isaac Gym이나 Brax와 같이 GPU 상에서 병렬 시뮬레이션을 수행하는 시스템들이 등장했다. 하지만 이러한 시스템들은 주로 단일 작업 가속에 집중했으며, 지속적 학습 환경에서의 동적 버퍼 관리나 통합 보상 정제와 같은 복잡한 요구사항은 충분히 다루지 않았다.

## 🛠️ Methodology

### 1. Progress-Aware Reward as a Learned Potential Function

ProgAgent는 전문가 비디오 $D_{expert}$를 사용하여 현재 상태가 목표 상태에 얼마나 근접했는지를 나타내는 진행도 비율 $\delta = |s_{curr} - s_{init}| / |s_{goal} - s_{init}|$를 예측하는 모델 $V_\phi$를 학습한다. 학습 손실 함수는 다음과 같이 KL divergence를 사용하여 정의된다.

$$L_{expert}(\phi) = \mathbb{E}_{(s_i, s_j, s_k) \sim D_{expert}} [ D_{KL} ( \mathcal{N}(\delta, \sigma^2) || V_\phi(s_i, s_j, s_k) ) ]$$

여기서 예측된 평균값은 상태-포텐셜 함수 $\Phi_\phi(s_t)$로 해석되며, 이를 통해 다음과 같이 형태가 지정된 보상(Shaped Reward)을 생성한다.

$$r_t(s_t, s_{t-1}; \phi) = \gamma \Phi_\phi(s_t) - \Phi_\phi(s_{t-1})$$

이 방식은 Ng 등이 제시한 보상 형성 이론에 따라 최적 정책의 불변성(Policy Invariance)을 보장하면서도, 희소한 보상 문제를 해결하고 전문가의 궤적을 따라가도록 가이드한다.

### 2. Adversarial Push-back Refinement

온라인 탐색 중 에이전트가 전문가 데이터에 없는 생소한 상태에 도달했을 때, 보상 모델이 잘못된 높은 보상을 주는 것을 방지하기 위해 적대적 정제(Adversarial Push-back)를 수행한다. 비전문가 궤적 $D_{\pi_\theta}$에 대해서는 예측값을 낮은 신뢰도의 사전 분포 $\mathcal{N}(0, \sigma^2_{prior})$로 밀어낸다.

$$L_{push}(\phi) = \mathbb{E}_{(s_i, s_j, s_k) \sim D_{\pi_\theta}} [ D_{KL} ( V_\phi(s_i, s_j, s_k) || \mathcal{N}(0, \sigma^2_{prior}) ) ]$$

최종 보상 모델의 손실 함수는 전문가 정렬 손실과 push-back 손실의 가중합으로 정의된다.
$$L_{reward}(\phi) = L_{expert} + \beta L_{push}$$

### 3. JAX-Native Architecture & Unified Objective

시스템적으로는 모든 시뮬레이션과 최적화 루프를 JAX의 JIT 컴파일러와 $vmap$을 통해 구현하여 수천 개의 환경에서 병렬 롤아웃을 수행한다.

정책 $\pi_\theta$는 PPO를 기반으로 하며, 망각을 방지하기 위해 Coreset Replay와 Synaptic Intelligence(SI)를 통합한 단일 목적 함수를 최적화한다.

$$L_{total}(\theta) = L_{PPO}(\theta; \phi) + \lambda_1 L_{replay}(\theta) + \lambda_2 L_{SI}(\theta)$$

- **Functional Replay ($L_{replay}$)**: Coreset $M$에서 샘플링한 과거 경험을 advantage-weighted 방식으로 재생하여 과거 지식을 유지한다.
- **Synaptic Importance ($L_{SI}$)**: 각 파라미터의 중요도 $\Omega$를 계산하여, 중요한 가중치가 크게 변하지 않도록 2차 페널티를 부여한다.
$$L_{SI} = \sum_k \Omega_k (\theta - \theta^*_k)^2$$

## 📊 Results

### 실험 설정

- **데이터셋**: ContinualBench (button-press, door-open, window-close) 및 Meta-World.
- **비교 대상**: OA (Online Agent), Perfect Memory (이상적 상한선), Coreset, SI, Fine-tuning, Rank2Reward, GAIL, TCN.
- **지표**: 성공률(Success Rate), 평가 보상(Eval Reward), 평균 성능(Average Performance, AP), 후회도(Regret).

### 주요 결과

- **정량적 성능**: ProgAgent는 ContinualBench의 모든 작업에서 모든 베이스라인을 압도하였다. 특히 놀라운 점은 모든 과거 데이터를 재학습하는 **Perfect Memory 에이전트보다 높은 성능**을 보였다는 것이다. 이는 단순한 메모리 용량보다 시스템적 효율성과 정교한 보상 설계가 더 중요함을 시사한다.
- **학습 속도**: 학습 곡선 분석 결과, ProgAgent는 다른 방법론보다 훨씬 빠르게 높은 보상 값에 도달하는 높은 샘플 효율성(Sample Efficiency)을 보였다.
- **정성적 분석**: 학습된 포텐셜 함수 $\Phi_\phi(s_t)$를 시각화했을 때, 전문가 및 성공 궤적에서는 단조 증가하는 매끄러운 곡선을 보인 반면, 실패 궤적에서는 값이 낮게 유지됨을 확인하였다. 이는 보상 모델이 단순한 시각적 특징 암기가 아닌 실제 작업 진행도를 정확히 파악하고 있음을 증명한다.
- **Ablation Study**:
  - Push-back 제거 시: 분포 변화로 인해 보상 해킹이 발생하며 성능이 급격히 하락한다.
  - CL Regularizers($L_{replay}, L_{SI}$) 제거 시: 새로운 작업은 빠르게 배우지만(가소성 $\uparrow$), 과거 작업 성능이 급락하는 치명적 망각(안정성 $\downarrow$)이 관찰되었다.

## 🧠 Insights & Discussion

본 논문의 가장 큰 통찰은 **"알고리즘적 정렬(Algorithmic Alignment)과 시스템적 효율성(Architectural Efficiency)의 결합이 CRL의 Stability-Plasticity 경계를 확장시킨다"**는 점이다.

단순히 망각 방지 알고리즘만 도입하거나 보상 모델만 개선해서는 한계가 있으며, JAX 기반의 고처리량 아키텍처를 통해 방대한 데이터를 병렬로 처리하고 복잡한 통합 손실 함수를 안정적으로 최적화함으로써 비로소 최적의 해를 찾을 수 있었다.

**한계점 및 향후 과제**:

- **전문가 데이터 의존성**: 제공된 전문가 비디오의 품질과 다양성이 낮을 경우, 포텐셜 함수가 잘못된 국소 최적점(local optima)으로 유도할 위험이 있다.
- **Sim-to-Real Gap**: 시뮬레이션에서의 성공이 실제 로봇의 시각적 도메인 갭을 극복하고 그대로 전이될지는 여전히 과제로 남아 있다.
- **하이퍼파라미터 튜닝**: $\beta, \lambda_1, \lambda_2$ 등 여러 계수를 수동으로 튜닝해야 하는 번거로움이 있으며, 향후 메타 학습을 통한 자동 튜닝이 필요하다.

## 📌 TL;DR

ProgAgent는 **전문가 비디오 기반의 진행도 예측 보상 모델**과 **초고속 JAX-native 시스템**을 결합한 지속적 강화학습 에이전트이다. 적대적 정제(Adversarial Push-back)를 통해 보상의 안정성을 높이고, PPO-Coreset-SI 통합 목적 함수를 통해 치명적 망각을 방지한다. 실험 결과, 모든 지표에서 SOTA를 달성했으며 심지어 이상적인 Perfect Memory 모델보다 우수한 성능을 보여, 효율적인 시스템 구조와 정교한 보상 설계의 시너지를 입증하였다.
