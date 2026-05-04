# Extrinsically Rewarded Soft Q Imitation Learning with Discriminator

Ryoma FURUYAMA, Daiki KUYOSHI, and Satoshi YAMANE (2024)

## 🧩 Problem to Solve

본 논문은 모방 학습(Imitation Learning)에서 발생하는 **상태 분포 변화(Distribution Shift)** 문제와 **보상 설계(Reward Design)**의 어려움을 해결하고자 한다.

일반적으로 보상 함수를 설계하기 어렵거나 보상이 희소(Sparse)한 환경에서 모방 학습이 사용된다. 기존의 Behavioral Cloning(BC)은 전문가의 데이터를 직접 학습하는 지도 학습 방식이지만, 학습 데이터에 없는 상태에 진입했을 때 오차가 누적되어 성능이 급격히 저하되는 distribution shift 문제에 취약하다.

이를 해결하기 위해 제안된 Soft Q Imitation Learning(SQIL)은 BC와 Soft Q-learning을 결합하여 분포 변화 문제를 완화했다. 그러나 SQIL은 전문가 데이터에는 상숫값 1을, 샘플링 데이터에는 0의 보상을 부여하는 고정 보상 방식을 사용한다. 이러한 방식은 학습이 진행됨에 따라 에이전트가 획득한 샘플 데이터가 전문가 데이터와 매우 유사하더라도 여전히 0의 보상을 받게 하여, 학습 과정에서 노이즈로 작용하거나 학습 효율을 떨어뜨리는 한계가 있다. 따라서 본 논문의 목표는 고정 보상 대신 **판별자(Discriminator)를 이용한 가변 보상 함수**를 도입하여 더 효율적이고 강건한 모방 학습 알고리즘인 **DSQIL(Discriminator Soft Q Imitation Learning)**을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 GAN(Generative Adversarial Networks)의 판별자 구조를 SQIL의 보상 체계에 통합하는 것이다.

핵심 직관은 에이전트가 수집한 샘플 데이터가 전문가의 행동과 얼마나 유사한지를 판별자가 수치적으로 계산하게 하고, 이를 보상으로 환원하여 제공하는 것이다. 이를 통해 에이전트는 단순히 전문가 데이터를 복제하는 것을 넘어, 전문가와 유사한 행동을 수행했을 때 긍정적인 보상을 받음으로써 더 효율적으로 정책을 최적화할 수 있다.

## 📎 Related Works

### Behavioral Cloning (BC)

전문가의 상태-행동 쌍 $(s^E, a^E)$를 정답으로 사용하는 지도 학습 방식이다. 구현이 간단하지만, 전문가 데이터의 분포에서 벗어난 상태(unseen states)에 직면했을 때 적절한 행동을 결정하지 못하는 분포 변화 문제에 취약하다.

### Soft Q Imitation Learning (SQIL)

BC의 한계를 극복하기 위해 제안된 방법으로, Soft Q-learning을 기반으로 하며 전문가 데이터와 샘플링 데이터에 상수를 부여하는 방식으로 학습한다. 기존의 Adversarial Imitation Learning 방식들보다 적은 학습 단계로 효율적인 모방이 가능하다고 보고되었다.

### Generative Adversarial Networks (GAN)

생성자(Generator)와 판별자(Discriminator)가 서로 경쟁하며 데이터 분포를 학습하는 구조이다. GAIL(Generative Adversarial Imitation Learning)이나 AIRL(Adversarial Inverse Reinforcement Learning)과 같은 연구들이 이 구조를 모방 학습에 도입하여 전문가의 정책을 효율적으로 학습하는 성과를 거두었다.

## 🛠️ Methodology

### 전체 시스템 구조

DSQIL은 전문가 데이터 버퍼($\beta^{demo}$)와 에이전트가 수집한 샘플 데이터 버퍼($\beta^{sample}$)를 동시에 사용한다. 판별자 $D$는 이 두 버퍼의 데이터를 입력받아 해당 데이터가 전문가의 것인지 판별하며, 이 판별 결과가 곧 에이전트의 보상 함수 $R$이 된다.

### 주요 구성 요소 및 학습 절차

1. **판별자(Discriminator) 학습**: 전문가 데이터와 샘플 데이터를 구분하도록 학습한다. 판별자 $D$의 가중치를 $\phi$라고 할 때, 다음과 같은 Binary Cross Entropy 손실 함수를 최소화한다.
   $$\mathcal{L}(\phi) \triangleq -\mathbb{E}_{(s,a) \sim \beta^{demo}} [\log(D_\phi(s,a))] - \mathbb{E}_{(s,a) \sim \beta^{sample}} [\log(1-D_\phi(s,a))]$$
2. **보상 함수 정의**: 학습된 판별자의 출력값을 보상으로 사용한다. 본 논문에서는 보상 범위를 SQIL과 유사하게 제한하기 위해 판별자 출력의 절반 값을 보상으로 설정하였다.
   $$R = \frac{D(s, a)}{2}$$
3. **정책(Policy) 업데이트**: Soft Q-learning 기반의 업데이트 식을 사용하며, 전문가 데이터와 샘플 데이터에 대해 각각 계산된 보상을 적용한다.

### 주요 방정식 및 업데이트 식

에이전트의 파라미터 $\theta$는 다음과 같은 업데이트 식에 의해 갱신된다.
$$\theta \leftarrow \theta - \eta \nabla_\theta \left( \omega_{demo} \delta^2(\beta^{demo}, R(\beta^{demo}) + \frac{1}{2\omega_{demo}}) + \omega_{sample} \delta^2(\beta^{sample}, R(\beta^{sample})) \right)$$

여기서 $\delta^2$는 **Squared Soft Bellman Error**를 의미하며, 다음과 같이 정의된다.
$$\delta^2(\beta, r) \triangleq \frac{1}{|\beta|} \sum_{(s,a,s') \in \beta} \left( Q_\theta(s,a) - (r + \gamma \log \sum_{a' \in \mathcal{A}} \exp(Q_\theta(s',a'))) \right)^2$$

이 식은 현재의 Soft Q-값과 보상 및 다음 상태의 Soft Value(Log-Sum-Exp) 합 사이의 오차를 최소화함으로써 정책을 전문가의 행동에 가깝게 유도한다.

## 📊 Results

### 실험 설정

- **환경**: MuJoCo의 세 가지 환경(Hopper-v3, Walker2d-v3, HalfCheetah-v3)
- **비교 대상**: BC, SQIL (SAC 기반)
- **측정 지표**: 전문가 데이터의 양 $\{2, 4, 8, 16, 32\}$ 에피소드 변화에 따른 평균 점수
- **에이전트**: Soft Actor-Critic(SAC) 사용

### 정량적 결과 분석

- **BC 대비 성능**: 모든 환경에서 DSQIL이 BC보다 월등한 성능을 보였다.
- **SQIL 대비 성능**:
  - **단순 환경(Hopper, Walker2d)**: 전문가 데이터가 충분할 경우 SQIL과 DSQIL의 성능이 비슷하다. 다만, 데이터가 매우 적을 때는 오히려 SQIL이 약간 더 나은 모습을 보이기도 했다.
  - **복잡한 환경(HalfCheetah)**: 전문가 데이터의 양과 상관없이 DSQIL이 SQIL을 압도하였다. 특히 전문가 데이터가 적을 때 그 차이가 매우 뚜렷하게 나타났다.

### 학습 속도 및 보상 변화

- **학습 속도**: 단순 환경에서는 SQIL이 더 빠르게 수렴한다. 이는 판별자가 충분히 학습될 때까지 시간이 걸리며, 초기 단계의 부정확한 판별자가 잘못된 보상을 제공하여 학습을 방해할 수 있기 때문이다. 반면, 복잡한 환경(HalfCheetah)에서는 DSQIL의 수렴 속도가 더 빨랐는데, 이는 가변 보상을 통한 학습 이득이 판별자 학습 비용보다 크기 때문이다.
- **보상 추이**: 학습이 진행됨에 따라 샘플 데이터에 부여되는 보상이 점차 증가하는 경향을 보였다. 이는 에이전트가 전문가와 유사한 행동을 생성하기 시작했음을 의미한다.

## 🧠 Insights & Discussion

### 강점

DSQIL은 고정 보상의 한계를 극복하고 판별자를 통해 '전문가와 얼마나 유사한가'에 대한 정밀한 피드백을 제공한다. 특히 HalfCheetah와 같은 복잡한 제어 작업에서 적은 양의 전문가 데이터만으로도 효율적인 학습이 가능하다는 점이 입증되었다.

### 한계 및 비판적 해석

1. **환경 복잡도에 따른 효율성 역전**: 단순한 환경에서는 오히려 판별자를 학습시키는 오버헤드가 성능 저하나 학습 속도 저하로 이어진다. 이는 판별자의 정확도가 낮을 때 제공되는 보상이 일종의 노이즈로 작용할 수 있음을 시사한다.
2. **보상 수렴 문제**: 학습 후반부에 전문가 데이터와 샘플 데이터의 보상 값이 비슷하게 수렴하는 현상이 관찰되었다. 이는 전문가의 성능을 정확히 추종하는 데는 도움이 될 수 있으나, 전문가를 뛰어넘는 성능을 목표로 할 때는 제약이 될 가능성이 있다.

## 📌 TL;DR

본 논문은 SQIL의 고정 보상 체계를 GAN의 판별자 기반 가변 보상 체계로 교체한 **DSQIL**을 제안한다. 이를 통해 에이전트는 전문가 데이터와 유사한 행동을 했을 때 더 높은 보상을 받게 되어, 특히 복잡한 환경에서 데이터 효율성과 학습 성능이 크게 향상된다. 본 연구는 모방 학습에서 단순한 데이터 복제보다 정교한 보상 셰이핑(Reward Shaping)이 복잡한 태스크 해결에 핵심적임을 보여준다.
