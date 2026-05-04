# DiffAIL: Diffusion Adversarial Imitation Learning

Bingzheng Wang, Guoqiang Wu, Teng Pang, Yan Zhang, Yilong Yin (2024)

## 🧩 Problem to Solve

본 논문은 실제 의사결정 작업에서 보상 함수(reward function)를 정의하기 어렵다는 문제점을 해결하기 위해 모방 학습(Imitation Learning)을 다룬다. 특히, 최근 널리 사용되는 적대적 모방 학습(Adversarial Imitation Learning, AIL) 프레임워크의 한계점에 집중한다.

전통적인 AIL에서는 전문가의 상태-행동 점유 측정치(state-action occupancy measures)와 에이전트의 것을 일치시켜 대리 보상(surrogate reward)을 얻는다. 그러나 기존의 판별자(discriminator)는 단순한 이진 분류기(binary classifier) 형태이기 때문에, 데이터의 복잡한 분포를 정확하게 학습하지 못하는 경향이 있다. 이로 인해 에이전트가 환경과 상호작용하며 생성한 '전문가 수준의 상태-행동 쌍'을 판별자가 제대로 식별하지 못하게 되며, 결과적으로 에이전트의 학습 효율과 성능이 저하되는 문제가 발생한다.

따라서 본 논문의 목표는 전문가 데이터의 분포를 보다 정교하게 캡처할 수 있는 강력한 판별자를 도입하여, AIL의 일반화 성능을 높이고 에이전트가 전문가 수준의 행동을 더 잘 학습하도록 하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **AIL 프레임워크의 판별자 구조에 확산 모델(Diffusion Model)을 통합**하는 것이다.

기존의 단순 분류기 대신, 상태-행동 쌍 $(s, a)$의 결합 분포를 조건 없는 확산 모델(unconditional diffusion model)로 모델링한다. 판별자가 단순히 "전문가 데이터인가 아닌가"를 분류하는 것이 아니라, 확산 모델의 학습 목표인 **확산 손실(diffusion loss)**을 판별 지표로 활용함으로써 전문가 데이터의 정밀한 분포를 학습하게 한다. 이를 통해 판별자는 전문가의 행동 양식을 더 정확하게 파악할 수 있으며, 에이전트에게 더욱 정교한 가이드(보상)를 제공할 수 있게 된다.

## 📎 Related Works

### 적대적 모방 학습 (Adversarial Imitation Learning)

GAIL, AIRL, f-GAIL 등의 연구들은 전문가와 에이전트 간의 점유 측정치 차이(f-divergence)를 최소화하는 방향으로 학습한다. 최근에는 샘플 효율성을 높이기 위해 오프-폴리시(off-policy) 방식을 도입한 DAC나, 분포 보정 추정(distribution correction estimation)을 사용하는 ValueDice, CFIL 등이 제안되었다. 하지만 여전히 판별자가 분포를 정확히 모델링하지 못하면 학습이 불안정해지는 한계가 있다.

### 강화학습과 확산 모델 (Diffusion Model with RL)

Diffusion-BC나 Diffuser와 같은 연구들은 확산 모델의 강력한 생성 능력을 활용하여 정책(policy) 자체를 확산 모델로 구현하거나 궤적(trajectory)을 생성하는 데 사용했다. 반면, 본 논문의 DiffAIL은 확산 모델을 정책 생성기가 아닌 **판별자의 분포 매칭 도구**로 사용한다는 점에서 기존 연구들과 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조

DiffAIL은 전통적인 AIL 프레임워크를 유지하면서, 판별자의 내부 메커니즘을 확산 모델 기반으로 교체한 구조이다. 전체적인 흐름은 전문가 데이터와 에이전트가 생성한 데이터를 확산 모델이 학습하고, 그 결과(손실 값)를 바탕으로 보상을 생성하여 강화학습 알고리즘(SAC 등)이 정책을 업데이트하는 루프를 가진다.

### 확산 모델 기반 판별자 설계

먼저, 상태-행동 쌍 $x = (s, a)$에 대한 확산 프로세스를 정의한다.
전방향 확산 과정(forward process)은 데이터에 점진적으로 가우시안 노이즈를 추가하는 과정이며, 역방향 과정(reverse process)은 노이즈로부터 원래의 데이터를 복구하는 과정이다.

본 논문에서는 확산 모델의 핵심인 노이즈 예측 손실 함수 $\text{Diff}_\phi$를 다음과 같이 정의한다:
$$\text{Diff}_\phi(x, \epsilon, t) = \|\epsilon - \epsilon_\phi(\sqrt{\alpha_t}x + \sqrt{1-\alpha_t}\epsilon, t)\|^2$$
여기서 $\epsilon$은 표준 가우시안 노이즈, $\epsilon_\phi$는 파라미터 $\phi$로 학습되는 노이즈 예측 네트워크이다.

### 적대적 학습 목표 및 보상 함수

판별자 $D_\phi$는 확산 손실의 음수 값에 지수 함수를 적용하여 $(0, 1)$ 범위의 확률 값으로 변환한다:
$$D_\phi(x, \epsilon, t) = \exp(-\text{Diff}_\phi(x, \epsilon, t))$$

최종적인 최적화 목적 함수는 GAIL의 minimax 구조를 따른다:
$$\min_{\pi_\theta} \max_{D_\phi} \mathbb{E}_{\epsilon, t} \left[ \mathbb{E}_{x \sim \pi^e}[\log(D_\phi(x, \epsilon, t))] + \mathbb{E}_{x \sim \pi^\theta}[\log(1 - D_\phi(x, \epsilon, t))] \right]$$

에이전트 $\pi_\theta$를 학습시키기 위한 대리 보상 함수 $R_\phi$는 다음과 같이 정의된다:
$$R_\phi(x, \epsilon) = -\frac{1}{T} \sum_{t=1}^T \log(1 - \exp(-\text{Diff}_\phi(x, \epsilon, t)))$$
즉, 확산 오류($\text{Diff}_\phi$)가 낮을수록(전문가 데이터와 유사할수록) 더 높은 보상을 부여하는 방식이다.

### 학습 절차

1. 에이전트 $\pi_\theta$를 통해 데이터를 수집하고 리플레이 버퍼에 저장한다.
2. 전문가 데이터와 에이전트 데이터를 샘플링하여 확산 판별자 $\phi$를 업데이트한다 (Gradient Ascent).
3. 업데이트된 판별자를 통해 대리 보상을 계산한다.
4. 계산된 보상을 바탕으로 SAC(Soft Actor-Critic) 알고리즘을 사용하여 정책 $\pi_\theta$와 Q-함수 $\omega$를 업데이트한다.

## 📊 Results

### 실험 설정

- **환경:** MuJoCo (Hopper, HalfCheetah, Walker2d, Ant)
- **데이터셋:** 전문가 궤적을 1, 4, 16개로 나누어 데이터 양에 따른 성능 측정
- **비교 대상:** BC, GAIL, ValueDice, CFIL, OPOLO
- **지표:** 에피소드당 평균 보상(Average Episodic Return)

### 주요 결과

- **SOTA 달성:** 단 1개의 전문가 궤적만 사용했을 때도 모든 태스크에서 기존 베이스라인보다 우수한 성능을 보였으며, 특히 HalfCheetah와 Ant에서는 전문가의 성능을 상회하는 결과를 얻었다.
- **상태 전용(State-only) 설정:** 행동 데이터 없이 상태만 주어진 환경에서도 동일한 하이퍼파라미터로 전문가 수준의 성능을 달성하였다.
- **판별 능력 및 일반화:** 학습에 사용되지 않은 전문가 궤적에 대해 테스트했을 때, GAIL보다 훨씬 높은 정확도로 전문가 데이터를 식별해내어 확산 모델 기반 판별자의 일반화 능력을 입증하였다.
- **보상 상관관계:** 대리 보상과 실제 보상 간의 피어슨 상관계수(Pearson Coefficient)를 분석한 결과, DiffAIL이 GAIL보다 훨씬 높은 선형 상관관계를 보였다. 이는 확산 기반 보상이 정책 학습을 더 정확하게 가이드함을 의미한다.

## 🧠 Insights & Discussion

### 강점

DiffAIL의 가장 큰 강점은 확산 모델의 강력한 분포 표현 능력을 판별자에 이식했다는 점이다. 단순 분류기는 데이터의 경계만을 학습하지만, 확산 모델은 데이터의 밀도 분포 자체를 학습하므로 전문가의 행동 패턴을 훨씬 정교하게 캡처할 수 있다. 이는 결과적으로 더 정확한 대리 보상으로 이어져, 적은 양의 데이터로도 전문가 이상의 성능을 낼 수 있게 한다.

### 한계 및 논의사항

- **계산 비용:** 확산 모델의 특성상 노이즈 샘플링 과정이 필요하며, 이는 기존의 단순 MLP 기반 판별자보다 계산 시간이 더 오래 걸린다. 실험 결과 확산 스텝($T$)이 많을수록 성능은 안정적이지만 학습 시간은 증가하는 트레이드-오프가 존재한다.
- **가정:** 본 연구에서는 MuJoCo와 같은 연속 제어 환경을 다루었으며, 매우 복잡한 고차원 데이터셋에서도 동일한 효율성이 유지될지는 추가적인 검증이 필요하다.

## 📌 TL;DR

본 논문은 AIL의 판별자가 데이터 분포를 정확히 학습하지 못해 성능이 저하되는 문제를 해결하기 위해, **판별자 내부에 확산 모델(Diffusion Model)을 도입한 DiffAIL**을 제안한다. 확산 손실(diffusion loss)을 기반으로 보상을 설계함으로써 전문가의 행동 분포를 정밀하게 캡처하였으며, 그 결과 MuJoCo 벤치마크에서 **적은 데이터만으로도 SOTA 및 전문가 수준 이상의 성능을 달성**하였다. 이 연구는 생성 모델의 분포 학습 능력을 판별기에 적용하는 새로운 방향성을 제시하며, 향후 더 효율적인 샘플러를 도입한다면 계산 비용 문제까지 해결하여 광범위하게 적용될 가능성이 크다.
