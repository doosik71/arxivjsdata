# BOOTSTRAPPED META-LEARNING

Sebastian Flennerhag, Yannick Schroecker, Tom Zahavy, Hado van Hasselt, David Silver, Satinder Singh (2022)

## 🧩 Problem to Solve

본 논문은 Meta-learning(메타 학습) 과정에서 발생하는 두 가지 핵심적인 병목 현상을 해결하고자 한다. 일반적인 메타 학습은 학습 규칙(update rule)을 적용한 후 $K$ 단계 이후의 성능을 평가하여 메타 파라미터를 최적화한다. 그러나 이러한 방식은 다음과 같은 한계를 가진다.

1. **Myopia (근시안적 편향):** 메타 목적 함수가 $K$ 단계라는 짧은 horizon 내의 성능만을 평가하므로, 그 이후에 일어날 미래의 학습 역학(learning dynamics)을 무시하게 된다. 이는 Short-horizon bias로 이어져 장기적인 성능 저하를 초래한다.
2. **Curvature (곡률 문제):** 메타 목적 함수가 학습자의 목적 함수와 동일한 기하학적 구조(geometry)에 제약을 받는다. 따라서 학습자의 목적 함수가 ill-conditioned(악조건) 상태라면 메타 목적 함수 역시 동일한 문제를 겪게 되어 최적화가 매우 어려워진다.

따라서 본 연구의 목표는 미래의 학습 정보를 목적 함수에 주입하여 근시안적 편향을 줄이고, 매칭 공간(matching space)을 통해 곡률을 제어함으로써 메타 최적화의 효율성을 높이는 알고리즘을 설계하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **"메타 학습자가 자신의 미래 모습으로부터 학습하는 것(the meta-learner learns from its future self)"**이다. 이를 위해 다음 두 가지 핵심 메커니즘을 제안한다.

- **Meta-Bootstrap:** 메타 학습자의 출력값인 $x^{(K)}$로부터 타겟 $\tilde{x}$를 생성하는 bootstrapping 방식을 도입한다. 이는 미래의 학습 궤적에 대한 정보를 목적 함수에 주입하여 Myopia 문제를 완화한다.
- **Target Matching:** 메타 목적 함수를 단순히 성능을 최대화하는 것이 아니라, bootstrapped target $\tilde{x}$와 현재의 $x^{(K)}$ 사이의 거리(distance) 또는 발산(divergence)을 최소화하는 문제로 재정의한다. 이를 통해 메타 손실 함수의 landscape(곡률)를 직접 제어할 수 있다.

## 📎 Related Works

본 논문은 다음과 같은 기존 연구들과 연관된다.

- **Reinforcement Learning (RL)의 Bootstrapping:** 미래의 예측값을 타겟으로 사용하는 TD(Temporal Difference) 학습 방식에서 영감을 얻었다.
- **Multi-task Meta-learning:** SGD의 초기값을 학습할 때 최적 파라미터와의 유클리드 거리를 최소화하는 방식과 유사하나, BMG는 임의의 메타 파라미터, 매칭 함수, 타겟 생성 방식을 허용함으로써 이를 일반화한다.
- **Knowledge Distillation:** KL-divergence를 이용한 타겟 매칭은 스승(teacher) 네트워크의 출력을 학생(student) 네트워크가 따라가게 하는 증류 방식과 구조적으로 유사하다.
- **Trust-region Methods:** 거리 함수를 도입해 그래디언트 업데이트를 정규화하는 방식에서 아이디어를 얻어 메타 최적화의 안정성을 꾀했다.

## 🛠️ Methodology

### 1. 전체 파이프라인

학습자의 목적 함수를 $f(x)$라 하고, 메타 파라미터 $w$를 이용한 업데이트 규칙을 $\phi$라고 할 때, $K$번의 업데이트 후의 파라미터를 $x^{(K)}(w)$라고 정의한다.

표준적인 Meta-Gradient(MG)는 다음과 같이 업데이트된다:
$$w' = w - \beta \nabla_w f(x^{(K)}(w))$$

반면, 제안된 **Bootstrapped Meta-Gradient(BMG)**는 다음의 두 단계를 거쳐 업데이트를 수행한다.

**단계 1: Target Bootstrap (TB) 생성**
현재의 $x^{(K)}$로부터 타겟 $\tilde{x}$를 생성한다. 본 논문에서는 주로 $\phi$를 추가로 $L-1$단계 더 실행하고, 마지막에 목적 함수 $f$에 대한 그래디언트 단계를 적용하는 방식을 사용한다:
$$\tilde{x} = x^{(K+L-1)} - \alpha \nabla f(x^{(K+L-1)})$$
이때, 중요하게는 **타겟 $\tilde{x}$를 생성하는 과정에 대해서는 역전파(backpropagation)를 수행하지 않는다.** 이는 TD 학습처럼 타겟을 고정된 목표로 취급하는 것이다.

**단계 2: Matching Function을 통한 최적화**
메타 파라미터 $w$를 업데이트하기 위해, $x^{(K)}(w)$와 $\tilde{x}$ 사이의 유사도를 측정하는 매칭 함수 $\mu$를 사용하여 다음과 같이 업데이트한다:
$$\tilde{w} = w - \beta \nabla_w \mu(\tilde{x}, x^{(K)}(w))$$

### 2. 주요 구성 요소 설명

- **Matching Function ($\mu$):** 파라미터 공간에서의 유클리드 거리뿐만 아니라, 확률 분포를 출력하는 모델의 경우 KL-divergence를 사용할 수 있다. 이는 메타 최적화의 곡률을 제어하는 역할을 한다.
- **Target Bootstrap ($\xi$):** 메타 학습자가 도달해야 할 미래의 상태를 정의한다. $L$ 값을 조절함으로써 backpropagation의 계산 비용을 크게 늘리지 않고도 실질적인 메타 학습 horizon을 확장할 수 있다.

## 📊 Results

### 1. 실험 설정

- **Non-stationary Grid-World:** 보상이 주기적으로 바뀌는 환경에서 에이전트가 얼마나 효율적으로 재탐색(re-exploration)하는지 측정한다.
- **Atari ALE:** 57개 게임을 대상으로 Human Normalized Score (HNS)를 측정하며, STACX를 베이스라인으로 사용한다.
- **Few-shot Learning:** MiniImagenet 데이터셋에서 MAML과 비교하여 데이터 및 계산 효율성을 측정한다.

### 2. 주요 결과

- **Atari ALE:** BMG는 Median HNS 약 **610%**를 달성하여 기존 SOTA 및 STACX를 크게 상회하는 성능을 보였다. 특히, 타겟 생성 시 RMSProp을 사용한 곡률 교정(45% 기여), 매칭 함수 변경(25% 기여), $L$ 증가를 통한 Myopia 완화(30% 기여)가 성능 향상의 주요 원인임을 확인하였다.
- **Grid-World:** BMG는 $\epsilon$-greedy 탐색 전략을 성공적으로 메타 학습하였다. 특히 업데이트 규칙을 통한 역전파 없이도 행동 정책(behavior policy)의 일부를 학습할 수 있음을 입증하였다.
- **Few-shot Learning:** MAML 대비 데이터 효율성이 높았으며, 동일한 성능에 도달하는 데 걸리는 시간이 약 절반 수준으로 단축되었다.

## 🧠 Insights & Discussion

### 1. 강점 및 이론적 근거

- **이론적 보장:** Theorem 1을 통해 BMG가 적절한 타겟과 매칭 함수를 선택할 경우 성능 개선을 보장함을 수학적으로 증명하였다. 또한, Corollary 1에서는 BMG가 표준 MG보다 더 큰 지역적 개선을 이뤄낼 수 있음을 보였다.
- **계산 효율성:** $K$를 늘려 역전파 경로를 길게 가져가는 대신, 타겟을 bootstrapping 하는 $L$을 늘림으로써 메모리와 계산 비용을 획기적으로 줄이면서도 긴 horizon의 효과를 얻었다.

### 2. 한계 및 비판적 해석

- **타겟 선택의 트레이드-오프:** 타겟 $\tilde{x}$가 너무 멀리 떨어져 있으면 학습 신호는 강해지지만 곡률로 인한 왜곡(distortion)이 심해진다. 따라서 $L$의 적절한 설정이 중요하며, 이는 실험적인 튜닝에 의존하는 경향이 있다.
- **가정 사항:** 본 분석은 노이즈가 없는(noiseless) 설정에서의 이론적 증명을 포함하고 있어, 실제 매우 노이즈가 심한 환경에서의 수렴성 보장은 추가적인 연구가 필요해 보인다.

## 📌 TL;DR

본 논문은 메타 학습의 고질적인 문제인 **근시안적 편향(Myopia)**과 **곡률 문제(Curvature)**를 해결하기 위해, 메타 학습자가 자신의 미래 상태를 타겟으로 삼아 이를 추적하게 만드는 **Bootstrapped Meta-Gradients (BMG)** 알고리즘을 제안한다. BMG는 미래의 타겟을 생성하는 과정에서 역전파를 생략함으로써 계산 효율성을 극대화하면서도 실질적인 학습 horizon을 확장한다. 실험 결과, Atari ALE에서 새로운 SOTA를 달성하였으며, Few-shot 학습에서도 높은 데이터 및 계산 효율성을 입증하였다. 이 연구는 메타 학습의 목적 함수를 단순 성능 최적화에서 **타겟 매칭**으로 전환함으로써 최적화 안정성과 효율성을 동시에 잡을 수 있음을 보여준다.
