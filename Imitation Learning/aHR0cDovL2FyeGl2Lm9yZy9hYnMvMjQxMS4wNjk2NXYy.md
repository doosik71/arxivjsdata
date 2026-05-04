# Imitation from Diverse Behaviors: Wasserstein Quality Diversity Imitation Learning with Single-Step Archive Exploration

Xingrui Yu, Zhenglin Wan, David Mark Bossens, Yueming Lyu, Qing Guo, Ivor W. Tsang (2025)

## 🧩 Problem to Solve

본 논문은 제한된 수의 전문가 시연(demonstrations)으로부터 다양하고 고성능인 행동(behaviors)을 학습하는 문제를 해결하고자 한다. 기존의 Imitation Learning (IL) 방법들은 여러 시연이 주어지더라도 보통 하나의 특정한 행동을 학습하도록 설계되어 있어, 다양성과 품질을 동시에 확보해야 하는 Quality Diversity (QD) 목표를 달성하는 데 한계가 있다.

특히, 기존의 Adversarial IL 기반 QDIL 접근 방식은 다음과 같은 두 가지 핵심적인 문제를 가지고 있다:
1. **학습 불안정성(Training Instability):** GAIL과 같은 적대적 학습 기반 방법은 GAN의 고질적인 문제인 학습 불안정성을 그대로 계승하며, 이는 QD 설정에서 여러 정책을 동시에 학습시켜야 하므로 더욱 심화된다.
2. **행동 과적합 보상(Behavior-Overfitted Reward):** 전문가의 시연이 제한적일 때, 보상 모델이 전문가가 보여준 특정 행동 주변에만 보상을 집중적으로 부여하게 된다. 이로 인해 에이전트가 시연 데이터 너머의 더 다양한 행동을 탐색하도록 유도하지 못하고 특정 행동에만 과적합되는 현상이 발생한다.

따라서 본 연구의 목표는 적대적 학습의 불안정성을 해결하고, 보상 모델의 과적합을 방지하여 제한된 데이터로도 전문가 수준 혹은 그 이상의 다양성과 성능을 가진 정책 아카이브를 구축하는 것이다.

## ✨ Key Contributions

본 논문은 위에서 언급한 두 가지 문제를 해결하기 위해 **Wasserstein Quality Diversity Imitation Learning (WQDIL)**과 **Single-Step Archive Exploration (SSAE)**이라는 두 가지 핵심 전략을 제안한다.

1. **WQDIL을 통한 학습 안정화:** Wasserstein Auto-Encoder (WAE)의 잠재 공간(latent space) 내에서 Wasserstein 적대적 학습을 수행함으로써 보상 모델 학습의 안정성을 획기적으로 개선한다.
2. **SSAE를 통한 행동 탐색 촉진:** 보상 함수에 Measure-conditioning을 적용하고, 단일 단계 아카이브 탐색 보너스(single-step archive exploration bonus)를 도입하여 에이전트가 전문가의 행동에만 매몰되지 않고 행동 공간 전체를 탐색하도록 유도한다.

## 📎 Related Works

### 관련 연구 및 한계
- **Imitation Learning (IL):** Behavior Cloning (BC)은 에러 누적 문제가 있으며, Inverse Reinforcement Learning (IRL) 및 GAIL과 같은 적대적 IL은 보상 함수를 추정하여 정책을 학습시키지만 학습이 매우 불안정하다는 단점이 있다.
- **Quality Diversity Reinforcement Learning (QDRL):** PPGA와 같은 최신 기법들은 PPO와 Differentiable QD를 결합하여 고성능의 다양한 행동을 찾지만, 이는 매우 정교하게 설계된 보상 함수가 필요하다는 전제가 있다. 실제 환경에서 이러한 보상 함수를 직접 설계하는 것은 매우 어렵다.
- **Wasserstein GAN & WAE:** WGAN은 JS-divergence 대신 Wasserstein distance를 사용하여 GAN의 안정성을 높였으며, WAE는 VAE의 안정적인 학습 특성과 GAN의 고품질 생성 능력을 결합한 모델이다.

### 기존 방식과의 차별점
기존의 Adversarial QDIL이 단순히 GAIL의 판별자(discriminator)를 보상 모델로 사용했다면, 본 연구는 이를 **WAE의 잠재 공간으로 확장**하여 안정성을 확보하고, 단순히 전체 궤적의 Measure를 보는 것이 아니라 **단일 단계(single-step)의 Measure 기반 보너스**를 통해 탐색 효율을 극대화했다는 점에서 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조
본 프레임워크는 **Proximal Policy Gradient Arborescence (PPGA)** 알고리즘을 기반으로 하며, 여기에 WQDIL 보상 모델과 SSAE 탐색 메커니즘을 통합한 구조이다. 전체 흐름은 다음과 같다:
1. 전문가 시연 $D$로부터 WQDIL 보상 모델 $R$을 학습시킨다.
2. 학습된 보상 모델 $R$과 SSAE 보너스를 결합하여 각 정책의 성능을 평가한다.
3. VPPO를 통해 목적 함수와 Measure 함수의 그래디언트를 근사하고, 이를 통해 정책 아카이브를 업데이트한다.

### 핵심 구성 요소 및 방법론

#### 1. Single-Step Archive Exploration (SSAE)
행동 과적합 문제를 해결하기 위해, 상태-행동 쌍뿐만 아니라 단일 단계의 Measure Proxy $\delta(s)$를 기반으로 탐색 보너스를 부여한다.
- **단일 단계 아카이브 ($A_{single}$):** $\delta(s)$에 따라 구획된 셀(cell)들의 방문 횟수 $n(C_i)$를 기록한다.
- **탐색 보너스 ($\mathcal{r}_{exp}$):** 특정 행동 영역의 방문 비율 $p(\delta(s))$가 낮을수록 더 높은 보상을 부여한다.
  $$\mathcal{r}_{exp}(s, a, \delta(s)) = \frac{1}{1 + p(\delta(s))}$$
  여기서 $p(\delta(s)) = \frac{n(C(\delta(s)))}{\sum_i n(C_i)}$ 이다.

#### 2. Measure-Conditioned Reward
보상 모델이 단순히 "전문가와 닮았는가"를 판단하는 것이 아니라, 현재의 행동 특성(Measure)에 따라 서로 다른 보상을 제공하도록 $\delta(s)$를 조건으로 입력한다.
$$\tilde{R}(s, a, \delta(s)) = \max_\pi \min_D \mathbb{E}_{(s,a)\sim D}[-\log D(s, a, \delta(s))] + \mathbb{E}_{(s,a)\sim \pi}[-\log(1 - D(s, a, \delta(s)))]$$
최종 보상은 적대적 보상과 탐색 보너스의 합으로 정의된다: $R = \tilde{R} + \mathcal{r}_{exp}$.

#### 3. WQDIL: WAE-WGAIL
학습 안정성을 위해 $\text{state-action-measure}$ 트리플렛을 잠재 공간 $Z$로 매핑하는 WAE 구조를 사용한다.
- **구조:** 인코더 $E_\phi$와 디코더 $D_\psi$를 통해 데이터를 재구성한다.
- **손실 함수:** 재구성 손실(Reconstruction Loss)과 잠재 공간에서의 1-Wasserstein 거리 $W_1$을 최소화하는 적대적 손실을 결합한다.
  $$\mathcal{L}(\phi, \psi) = \mathcal{L}_{recon}^E + \mathcal{L}_{recon}^\pi + \lambda \mathbb{E}_{z_e, z_\pi} [f_W(z_e) - f_W(z_\pi)]$$
  여기서 $f_W$는 1-Lipschitz 연속 함수인 Wasserstein 판별자이다.

#### 4. mCWAE-WGAIL (최종 제안 모델)
단순히 $(s, a)$만 사용하는 것이 아니라 $\delta(s)$까지 포함하여 $x = (s, a, \delta(s))$로 인코딩함으로써, 제한된 시연 데이터 상황에서도 에이전트가 마주칠 수 있는 다양한 행동에 대해 잠재 공간이 적응적으로 학습되도록 한다.

## 📊 Results

### 실험 설정
- **환경:** MuJoCo의 HalfCheetah, Humanoid, Walker2d.
- **시연 데이터:** PPGA 아카이브에서 가장 성능이 좋은 엘리트 중 다양성이 높은 4개의 정책에서 추출한 각 1개의 에피소드 (매우 제한적인 데이터 설정).
- **비교 대상:** GAIL, PWIL, AIRL, MaxEntIRL, GIRIL 및 제안 방법의 변형들(WAE-WGAIL 등).
- **평가 지표:**
    - **QD-Score:** 아카이브 내 모든 비어있지 않은 셀의 점수 합 (가장 중요).
    - **Coverage:** 비어있지 않은 셀의 비율 (다양성 측정).
    - **Best/Average Reward:** 최고 및 평균 보상 (품질 측정).

### 주요 결과
- **정량적 성과:** $\text{mCWAE-WGAIL-Bonus}$는 모든 환경에서 기존 IL 방법들보다 월등히 높은 **QD-Score**를 기록하였다. 특히 Humanoid 환경에서는 전문가(PPGA-trueReward)보다 12% 더 높은 QD-Score를 달성하며 전문가를 뛰어넘는 성능을 보였다.
- **다양성 및 품질:** 시각화 결과(Figure 4), GAIL은 넓은 영역을 탐색하지만 대부분 저성능인 반면, 제안 방법은 넓은 영역을 탐색하면서도 고성능 정책을 훨씬 더 많이 확보하였다.
- **샘플 효율성:** HalfCheetah 환경에서 다른 방법들보다 훨씬 적은 반복 횟수로 유사하거나 더 높은 QD-Score에 도달하여 뛰어난 샘플 효율성을 입증하였다.

## 🧠 Insights & Discussion

### 강점 및 분석
- **안정성의 핵심:** Ablation 연구를 통해 잠재 공간에서의 Wasserstein 적대적 학습이 QD-Score 향상에 가장 크게 기여함을 확인하였다. 이는 기존 GAIL의 불안정성을 WAE 구조가 효과적으로 억제했음을 의미한다.
- **탐색의 시너지:** $\text{mCWAE-WGAIL}$에 탐색 보너스(Bonus)와 Measure-conditioning을 추가했을 때, 특히 Humanoid 환경에서 QD-Score가 대폭 상승하였다. 이는 적대적 보상이 주는 '품질' 가이드와 SSAE가 주는 '다양성' 가이드가 상호 보완적으로 작용했기 때문이다.

### 한계 및 논의사항
- **보너스 영향의 차이:** 탐색 보너스의 효과가 HalfCheetah에서는 미미했으나 Walker2d와 Humanoid에서는 매우 컸다. 이는 환경의 복잡도나 Measure 정의 방식에 따라 탐색 보너스의 영향력이 달라질 수 있음을 시사한다.
- **가정:** 본 연구는 $\delta(s)$라는 Markovian Measure Proxy가 존재한다는 가정하에 작동한다. 만약 적절한 단일 단계 Measure를 정의하기 어려운 복잡한 태스크의 경우, 본 방법론의 효율성이 떨어질 가능성이 있다.

## 📌 TL;DR

본 논문은 제한된 전문가 시연으로부터 다양하고 고품질인 행동을 배우는 **Wasserstein Quality Diversity Imitation Learning (WQDIL)**을 제안한다. **WAE 기반의 잠재 공간 적대적 학습**으로 학습 불안정성을 해결하고, **단일 단계 아카이브 탐색 보너스 및 Measure-conditioning**을 통해 행동 과적합 문제를 극복하였다. 실험 결과, MuJoCo 환경에서 기존 SOTA IL 방법들을 압도하며 일부 태스크에서는 전문가 이상의 성과를 거두었다. 이 연구는 보상 함수 설계가 어려운 복잡한 로봇 제어 태스크에서 다양하고 효율적인 행동 라이브러리를 구축하는 데 중요한 기여를 할 것으로 기대된다.