# Robust Meta-Reinforcement Learning with Curriculum-Based Task Sampling

Morio Matsumoto, Hiroya Matsuba, and Toshihiro Kujirai (2022)

## 🧩 Problem to Solve

본 논문은 Meta-Reinforcement Learning (Meta-RL)에서 발생하는 **Meta-overfitting** 문제를 해결하고자 한다. Meta-RL의 목표는 다양한 태스크 분포에서 빠르게 적응할 수 있는 meta-policy를 학습하는 것이지만, 기존의 MAML과 같은 방식은 태스크를 무작위로 샘플링하여 학습한다. 이 과정에서 에이전트가 쉽게 높은 점수를 얻을 수 있는 '쉬운 태스크(easy tasks)'에 과하게 최적화되는 meta-overfitting이 발생하며, 결과적으로 태스크 분포가 넓어질수록 '어려운 태스크(difficult tasks)'에 대한 성능이 급격히 저하되는 문제가 발생한다.

이 문제는 실제 환경에 RL을 적용할 때 매우 중요하다. 시뮬레이션과 실제 환경 사이의 모델 오차(model errors)가 존재하는 상황에서, 에이전트가 학습 과정에서 접하지 못한 더 어려운 상황에서도 강건하게(robust) 작동해야 하기 때문이다. 따라서 본 논문의 목표는 커리큘럼 기반의 태스크 샘플링 기법을 도입하여 meta-overfitting을 줄이고, 넓은 태스크 분포에서도 강건한 성능을 내는 **RMRL-GTS (Robust Meta-Reinforcement Learning with Guided Task Sampling)** 방법을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 태스크 샘플링 과정에 커리큘럼(Curriculum)을 도입하여, 학습 단계에 따라 샘플링되는 태스크의 영역을 제한하고(Restriction), 성능이 낮은 어려운 태스크에 더 많은 가중치를 두어 샘플링하는(Prioritization) 것이다.

단순히 어려운 태스크를 집중적으로 학습하는 것뿐만 아니라, 학습 초기에 샘플링 영역을 제한했다가 점진적으로 확장하는 방식이 결합되어야만 진정한 의미의 강건한 Meta-RL을 달성할 수 있다는 통찰을 제시한다.

## 📎 Related Works

- **MAML (Model-Agnostic Meta-Learning):** 가장 대표적인 gradient-based meta-RL 방법론으로, 적은 수의 gradient step만으로 새로운 태스크에 빠르게 적응하는 초기 파라미터를 학습한다. 그러나 태스크 분포가 넓을 때 특정 태스크에 오버피팅되는 경향이 있다.
- **PEARL:** Probabilistic embeddings를 사용하여 성능을 개선했으나, 여전히 넓은 분포의 어려운 태스크에서는 낮은 점수를 보이는 편향(bias) 문제가 존재한다.
- **Meta-ADR:** Active Domain Randomization (ADR)을 MAML에 결합하여 커리큘럼 학습을 시도했다. 하지만 MuJoCo의 Ant-Velocity와 같이 태스크 난이도가 급격히 증가하는 환경에서는 MAML보다 낮은 성능을 보이기도 했다.
- **AdMRL (Model-based Adversarial Meta-RL):** 보상 함수(reward function)의 gradient 정보를 활용하여 넓은 분포에서 높은 성능을 보이지만, 실제 환경에서는 보상 함수의 수식이나 정보가 명확하지 않은 경우가 많아 적용에 한계가 있다.

본 논문의 RMRL-GTS는 보상 함수의 내부 정보 없이 오직 얻어진 **점수(score)**와 **에포크(epoch)** 정보만을 이용하여 샘플링 확률을 조정하므로, AdMRL보다 범용적이며 meta-ADR보다 어려운 태스크에 대해 더 강건한 성능을 보인다.

## 🛠️ Methodology

RMRL-GTS는 MAML 프레임워크를 기반으로 하며, 태스크 샘플링 단계를 다음과 같은 두 가지 접근 방식으로 개선한다.

### 1. MAML 기본 구조 (Baseline)
기본적으로 MAML은 다음과 같은 절차를 따른다.
1. 태스크 $\tau_i$에 대해 현재 파라미터 $\theta$에서 gradient step을 수행하여 적응된 파라미터 $\theta'_i$를 계산한다.
   $$\theta'_i = \theta - \alpha \nabla_\theta L_{\tau_i}(\pi_\theta)$$
2. 모든 샘플링된 태스크에 대한 손실 함수의 합을 최소화하도록 메타 파라미터 $\theta$를 업데이트한다.
   $$\theta = \theta - \beta \nabla_\theta \sum_{\tau_i \sim p(\tau)} L_{\tau_i}(\pi_{\theta'_i})$$

### 2. RMRL-GTS의 핵심 구성 요소

#### Approach I: 태스크 샘플링 영역의 제한 (Restriction of Task Sampling Region)
태스크의 난이도 $\tau$가 클수록 어렵다고 가정할 때, 학습 초기에 모든 영역을 샘플링하지 않고 영역을 세 가지($T_{easy}, T_{middle}, T_{difficult}$)로 나눈다.

- **초기 설정 ($\text{epoch}=0$):** 태스크를 일정 간격으로 샘플링하여 평균 보상을 계산하고, 평균 보상에 가장 가까운 값을 $\tau_{mean}$으로 설정하여 영역의 기준점으로 삼는다.
- **영역 정의:**
  - $T_{easy}: \tau_{min} \le \tau < \tau_{middle1}$
  - $T_{middle}: \tau_{middle1} \le \tau < \tau_{middle2}$
  - $T_{difficult}: \tau_{middle2} \le \tau \le \tau_{max}$
- **동적 확장:** 학습의 절반($0.5 N_{epoch}$)까지는 $T_{easy}$와 $T_{middle}$에서만 샘플링한다. 그 이후부터는 일정 주기($n_{batch}$)마다 $\tau_{middle1}$과 $\tau_{middle2}$를 오른쪽으로 이동시켜 $T_{easy}$ 영역을 확장하고 $T_{difficult}$ 영역을 축소한다. 결국 학습 종료 시점에는 전체 영역이 샘플링 대상이 된다.

#### Approach II: 점수 기반 우선순위 샘플링 (Prioritized Sampling with Score)
단순 무작위 샘플링 대신, 현재 성능이 낮은 태스크가 더 많이 뽑히도록 확률 분포 $p(\tau)$를 조정한다.

- **가중 점수 계산:** 최근 에포크일수록 더 높은 가중치를 부여하여 평균 보상 $f(\tau)$를 계산한다.
  $$f(\tau) = \frac{1}{n_{bin}} \sum_{\tau < \tau' < \tau + d\tau_{bin}} \frac{\sum_{i=0}^{n_{ce}} c_i(\tau') R_{mean,i}(\tau')}{\sum c_i(\tau')}$$
  여기서 $c_i$는 에포크 $i$에 따른 가중치이며, $R_{mean,i}(\tau)$는 해당 에포크에서의 평균 보상이다.
- **확률 분포 생성:** 계산된 점수를 min-max 정규화한 $\bar{f}(\tau)$를 이용하여, 점수가 낮을수록(어려울수록) 샘플링 확률이 높아지도록 설계한다.
  $$p(\tau) = \frac{1 - \bar{f}(\tau)}{\sum_{\tau} (1 - \bar{f}(\tau))}$$
- **보완책:** $T_{easy}$ 영역의 태스크들이 너무 오랫동안 샘플링되지 않아 점수가 왜곡되는 것을 막기 위해, 일정한 비율 $\delta$로 $T_{easy}$ 영역에서 무작위 샘플링을 병행한다.

## 📊 Results

### 실험 설정
- **데이터셋 및 환경:** MuJoCo의 Ant-Velocity, HalfCheetah-Velocity, 그리고 2D Navigation.
- **비교 대상:** MAML, 이전의 커리큘럼 기반 방법론 [5].
- **평가 지표:** 평균 총 보상(Mean Total Reward), Variance, 그리고 **Bias Score**.
  $$\text{Bias Score} = (\max_{\tau'} R(\tau') - R(\tau))$$
  Bias Score가 낮을수록 태스크 분포 전반에 걸쳐 성능이 균일하며 강건하다는 것을 의미한다.

### 주요 결과
1. **Ant-Velocity:** 
   - RMRL-GTS는 MAML보다 어려운 태스크 영역($v \ge 1.5$)에서 월등히 높은 점수를 기록했다.
   - 특히 $v \in [0, 3]$ 범위에서의 Variance가 MAML보다 현저히 낮았으며, 이는 Meta-overfitting이 억제되었음을 보여준다.
   - 학습 분포 밖의 영역($v \in [3, 5]$)에서도 MAML보다 더 높은 적응 성능을 보였다.
2. **HalfCheetah-Velocity:**
   - Ant-Velocity와 마찬가지로 어려운 태스크($v \ge 2$)에서 높은 성능을 보였으며, 학습 에포크가 증가함에 따라 어려운 태스크의 성능이 지속적으로 향상되었다.
3. **2D Navigation:**
   - 2D 좌표계의 넓은 목표 지점 분포에서도 MAML보다 높은 평균 보상을 기록하며 강건함을 입증했다.

### 절제 실험 (Ablation Study)
- **Approach I만 사용 시:** MAML과 유사한 결과가 나타나, 단순한 영역 제한만으로는 부족함을 알 수 있다.
- **Approach II만 사용 시:** 어려운 태스크에 어느 정도 적응하지만, 전체적인 성능은 RMRL-GTS(I+II 결합)보다 낮았다.
- **결론:** 영역 제한(I)과 우선순위 샘플링(II)이 모두 결합되어야 샘플링 효율이 극대화되고 강건한 Meta-policy를 얻을 수 있다.

## 🧠 Insights & Discussion

본 논문은 Meta-RL에서 단순히 "어려운 것을 많이 학습시킨다"는 아이디어만으로는 부족하며, **"어느 시점에 어떤 난이도의 범위를 학습시킬 것인가"**에 대한 전략적 접근(Curriculum)이 필수적임을 보여주었다.

**강점:**
- 보상 함수의 수학적 구조를 몰라도 점수만으로 작동하므로 실제 환경 적용 가능성이 높다.
- 단순한 샘플링 전략 수정만으로 MAML의 고질적인 문제인 Meta-overfitting을 효과적으로 해결했다.

**한계 및 논의사항:**
- 태스크 난이도가 $\tau$ 값에 비례한다는 가정이 필요하다. 비록 좌표 변환을 통해 해결 가능하다고 언급했으나, 난이도를 정의하기 어려운 복잡한 다차원 태스크에서는 이 가정을 설정하는 것이 어려울 수 있다.
- 제안된 방법은 샘플링 효율을 높이지만, MAML 자체의 gradient 기반 최적화 한계(예: Local minima)까지 해결하는 것은 아니다.

## 📌 TL;DR

이 논문은 Meta-RL의 Meta-overfitting 문제를 해결하기 위해 **태스크 샘플링 영역을 단계적으로 확장(Approach I)**하고, **성능이 낮은 태스크를 우선적으로 샘플링(Approach II)**하는 **RMRL-GTS** 방법을 제안한다. 실험 결과, Ant-Velocity 및 HalfCheetah-Velocity 등 다양한 벤치마크에서 MAML보다 어려운 태스크에 대해 더 강건한 성능을 보였으며, 점수 편향(Bias)을 크게 줄였다. 이 연구는 보상 함수의 내부 정보 없이도 샘플링 전략만으로 Meta-RL의 강건성을 높일 수 있음을 증명하여, 향후 RL의 실제 세계 적용에 기여할 가능성이 크다.