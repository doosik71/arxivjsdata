# Relational Mimic for Visual Adversarial Imitation Learning

Lionel Blondé, Yichuan Charlie Tang, Jian Zhang, Russ Webb (2019)

## 🧩 Problem to Solve

본 논문은 비디오 데모네스트레이션(Video Demonstrations)으로부터 행동을 학습하는 Visual Imitation Learning(VIL) 문제를 다룬다. 특히, 에이전트의 관절 각도나 속도와 같은 proprioceptive state(고유 수용성 상태) 정보 없이 오직 고차원 픽셀 입력(pixel inputs)만을 사용하여 복잡한 locomotion task(이동 작업)를 수행하는 것을 목표로 한다.

기존의 Behavioral Cloning(BC)은 데이터 부족 시 covariate shift 문제로 인해 매우 취약하며, Inverse Reinforcement Learning(IRL) 기반의 Apprenticeship Learning(AL)은 보상 함수를 복구하는 과정이 계산적으로 매우 비싸고 ill-posed 문제라는 한계가 있다. Generative Adversarial Imitation Learning(GAIL)은 이러한 문제를 완화하지만, 픽셀 입력만으로 복잡한 신체 구조의 상호작용과 조정(coordination)을 학습하기에는 표현력의 한계가 존재한다. 따라서 본 연구의 목표는 GAIL 프레임워크에 관계형 학습(Relational Learning)을 결합하여, 시각적 정보로부터 신체 부위 간의 공간적·시간적 관계를 추론하고 이를 통해 더 강력하고 효율적인 모방 학습을 달성하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 **Relational Mimic (RM)**이라는 새로운 방법론의 제안이다. RM의 중심 아이디어는 GAIL의 적대적 학습 구조에 **Self-attention** 메커니즘을 도입하여, 에이전트가 픽셀 데이터로부터 객체 간의 장거리 의존성(long-range dependencies)을 파악하는 '시각적 관계 추론' 능력을 갖추게 하는 것이다.

구체적으로, CNN 기반의 인지 스택에 Non-local block을 통합함으로써, 에이전트는 이미지 내의 서로 다른 공간적 위치뿐만 아니라 연속된 프레임 간의 시간적 관계를 동시에 고려할 수 있다. 이를 통해 고유 수용성 상태 정보 없이도 신체 부위 간의 조정을 최적화하여 locomotion 성능을 비약적으로 향상시켰다.

## 📎 Related Works

본 논문은 다음과 같은 관련 연구들을 배경으로 한다.

1. **Self-Attention & Non-local Networks**: Transformer의 Self-attention과 이미지 처리의 Non-local mean 연산은 데이터의 전역적인 관계를 캡처하는 데 탁월하다. 기존 연구들은 이를 주로 영상 예측(video prediction)이나 이미지 생성에 사용했으나, 본 논문은 이를 RL의 제어 정책(control policy)에 적용했다.
2. **Adversarial Imitation**: GAIL은 전문가의 상태-행동 분포와 에이전트의 분포를 구분하는 판별자(Discriminator)를 통해 보상을 생성한다. 최근 연구에서는 행동 정보가 없는 state-only 데모네스트레이션에서 학습하는 시도가 있었으나, 픽셀 입력을 사용하는 설정에서의 성능 최적화는 미흡했다.
3. **Relational Learning**: Graph Neural Networks(GNN)는 신체 구조를 그래프로 모델링하여 locomotion 성능을 높였으나, 이는 명시적인 신체 구조(proprioceptive state)가 주어졌을 때만 가능하다. 본 논문은 GNN과 달리 픽셀 입력에서 직접 관계를 발견하는 방식을 취함으로써 차별점을 가진다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

RM은 기본적으로 GAIL 프레임워크를 따르며, 정책 네트워크($\pi_\theta$), 가치 네트워크($V_\phi$), 그리고 보상을 생성하는 판별자 네트워크($D_\omega$)로 구성된다. 픽셀 입력의 부분 관측성(partial observability) 문제를 해결하기 위해 $k$개의 프레임을 쌓는 **Frame-stacking** 방식을 사용하며, 입력 데이터는 $84 \times 84 \times (colors \times k)$ 크기의 텐서가 된다.

### 2. Relational Block (Non-local Agent)

에이전트의 인지 능력 향상을 위해 도입된 Relational Block은 다음과 같은 Self-attention 메커니즘을 기반으로 한다. 입력 특징 맵 $u$에 대해, 쿼리(query) $q(u_i)$와 키(key) $k(u_j)$의 내적을 통해 유사도를 계산하고, 이를 softmax로 정규화하여 밸류(value) $v(u_j)$의 가중합을 구한다.

$$u_i \to \sum_{j=0}^{m} \text{softmax}(q(u_i)^T k(u_j)) v(u_j)$$

여기서 $q, k, v$는 각각 $1 \times 1$ convolution 레이어를 통해 생성된다. 최종 출력은 학습의 안정성을 위해 잔차 연결(residual connection)을 추가하여 다음과 같이 계산된다.

$$u_i \to e\left( \sum_{j=0}^{m} \text{softmax}(q(u_i)^T k(u_j)) v(u_j) \right) + u_i$$

### 3. 보상 학습 (Reward Learning)

판별자 $D_\omega$는 전문가의 궤적과 에이전트의 궤적을 구분하는 이진 분류기로 동작한다. GAIL의 목적 함수는 다음과 같은 minimax 문제로 정의된다.

$$\min_{\theta} \max_{\omega} [V(\theta, \omega)], \quad V(\theta, \omega) = \mathbb{E}_{\pi_\theta}[\log(1 - D_\omega(s, a))] + \mathbb{E}_{\pi_e}[\log D_\omega(s, a)]$$

전문가의 행동(action) 정보가 없는 경우, 환경의 결정론적(deterministic) 특성을 이용하여 상태 전이 $(s_t, s_{t+1})$를 $(s_t, a_t)$의 대리자(proxy)로 사용한다. 보상 신호 $r_\omega$는 판별자의 혼란 정도에 따라 다음과 같이 정의된다.

$$r_\omega(s^k_{t+1}) = -\log(1 - D_\omega(s^k_{t+1}))$$

학습의 안정성을 위해 판별자 네트워크에는 **Spectral Normalization**과 **Gradient Penalty**를 동시에 적용하였다.

### 4. 학습 절차

1. 정책 $\pi_\theta$를 통해 환경과 상호작용하며 데이터를 수집하고 저장한다.
2. 수집된 데이터와 전문가 데모네스트레이션을 사용하여 판별자 $D_\omega$를 업데이트한다.
3. 업데이트된 $D_\omega$가 제공하는 보상 $r_\omega$를 사용하여 PPO(Proximal Policy Optimization) 알고리즘으로 정책 $\pi_\theta$와 가치 네트워크 $V_\phi$를 업데이트한다.

## 📊 Results

### 1. 실험 설정

- **환경**: MuJoCo 기반의 `Hopper-v3`, `Walker2d-v3` (픽셀 입력 설정).
- **데이터**: 각 환경당 8개의 전문가 비디오 데모네스트레이션 사용.
- **지표**: Mean Episodic Return 및 CCDF(Complementary Cumulative Distribution Function)를 통해 성능과 안정성을 측정하였다.

### 2. 주요 결과

- **관계형 모듈의 효과**: RM-NL-NL(정책, 가치, 보상 네트워크 모두에 non-local block 적용) 모델이 baseline(관계형 모듈 없음)보다 월등한 성능을 보였다. 특히 복잡한 `Walker2d-v3` 환경에서 baseline은 거의 학습되지 않은 반면, RM은 성공적으로 수렴하였다.
- **모듈별 영향력**: ablation study 결과, non-local block을 **보상 네트워크(Reward module)**에 적용했을 때 성능 향상 폭이 가장 컸다. 이는 정확한 보상 신호를 생성하는 것이 모방 학습의 성패를 결정짓는 핵심임을 시사한다.
- **프레임 스택($k$)의 영향**: $k=4$에서 $k=8$로 늘렸을 때, RM 모델들은 전반적으로 성능이 향상되었다(Hopper의 경우 약 21% 증가). 반면, baseline은 오히려 성능이 하락하는 모습을 보였으며, 이는 관계형 추론 능력이 더 긴 시퀀스의 데이터를 효율적으로 처리하는 데 도움을 준다는 것을 의미한다.
- **RL 성능**: 보상 함수가 주어진 일반 RL 설정에서도 `NonLocalAgent`가 `Nature`나 `LargeImpala`와 같은 기존 SOTA 아키텍처보다 약 10% 높은 성능을 기록하였다.

## 🧠 Insights & Discussion

본 논문은 픽셀 기반의 locomotion 제어에서 **관계적 귀납 편향(relational inductive bias)**의 중요성을 입증하였다. 신체 부위 간의 유기적인 협응이 필수적인 locomotion 작업에서, 단순한 CNN은 국소적인 특징만을 추출하지만, Self-attention 기반의 Relational Block은 신체 각 부위의 상대적 위치와 움직임을 전역적으로 파악할 수 있게 한다.

특히 흥미로운 점은 보상 네트워크에 관계형 모듈을 적용했을 때 가장 큰 이득을 보았다는 것이다. 이는 판별자가 전문가의 행동을 구분할 때, 단순히 픽셀의 패턴을 보는 것이 아니라 '신체 부위들이 어떻게 상호작용하며 움직이는가'라는 고차원적인 관계를 파악해야 더 정확한 surrogate reward를 생성할 수 있기 때문으로 해석된다.

다만, GAIL의 고유한 문제인 샘플 효율성(sample-inefficiency) 문제는 여전히 남아 있다. 하지만 본 연구에서 제안한 아키텍처 개선은 알고리즘 변경과 독립적이므로, 향후 샘플 효율성을 높인 다른 GAIL 변형 알고리즘들과 결합하여 더 큰 시너지 효과를 낼 가능성이 높다.

## 📌 TL;DR

본 논문은 픽셀 입력만으로 로봇의 이동 동작을 배우는 Visual Adversarial Imitation Learning을 위해, GAIL에 **Self-attention 기반의 Relational Block**을 결합한 **Relational Mimic (RM)**을 제안한다. 이 방법은 신체 부위 간의 공간적·시간적 관계를 추론함으로써 복잡한 locomotion task에서 기존 baseline 및 RL 아키텍처 대비 월등한 성능을 보였으며, 특히 보상 함수 학습 단계에서 관계형 추론이 결정적인 역할을 함을 밝혔다. 이 연구는 시각적 관계 학습이 고차원 제어 문제의 효율성을 높이는 핵심 도구가 될 수 있음을 시사한다.
