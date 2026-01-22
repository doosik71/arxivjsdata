# Imitation Learning from Imperfect Demonstration

Yueh-Hua Wu, Nontawat Charoenphakdee, Han Bao, Voot Tangkaratt, Masashi Sugiyama

## 🧩 Problem to Solve

모방 학습(Imitation Learning, IL)은 전문가의 시연(demonstration)을 통해 최적의 정책(policy)을 학습하는 것을 목표로 합니다. 그러나 실제 시연 데이터는 최적의 시연을 수집하는 데 드는 높은 비용 때문에 불완전한 경우가 많습니다. 기존 IL 방법론은 종종 최적의 시연이 완벽하게 주어진다고 가정하지만, 이는 실제 환경에서는 충족되기 어렵습니다.

따라서 본 연구의 주요 문제는 불완전한 시연 데이터, 특히 일부 시연에만 품질을 나타내는 '신뢰도 점수(confidence score)'가 부여되어 있고 나머지는 '레이블이 없는(unlabeled)' 상황에서 어떻게 효과적으로 최적의 정책을 학습할 것인가입니다.

## ✨ Key Contributions

* **두 가지 새로운 신뢰도 기반 모방 학습(IL) 방법론 제안:**
  * **Two-step Importance Weighting IL (2IWIL):** 레이블이 없는 시연 데이터에 대한 신뢰도 점수를 예측한 후, 이 가중치를 기반으로 GAIL 목표를 최적화합니다.
  * **Generative Adversarial IL with Imperfect Demonstration and Confidence (IC-GAIL):** 시연 데이터의 최적 및 비최적 혼합 분포를 활용하여 최적 정책의 점유 측정(occupancy measure)과 직접 일치시키는 방법을 제안합니다.
* **신뢰도 점수의 효과 입증:** 비최적 시연의 작은 부분에만 신뢰도 점수를 부여하는 것만으로도 모방 학습의 성능을 이론적 및 실증적으로 크게 향상시킬 수 있음을 보여줍니다.
* **이론적 분석 제공:** 두 방법론 모두에 대해 추정 오차 경계(estimation error bound)를 포함한 이론적 분석을 제시하여 방법론의 건전성을 뒷받침합니다.
* **성능과 수렴 속도 간의 트레이드오프 분석:** IC-GAIL이 2IWIL보다 더 나은 최종 성능을 달성하지만 수렴 속도가 더 느리다는 점을 분석하고 실험적으로 확인합니다.

## 📎 Related Works

* **비최적 시연을 통한 학습:**
  * **Distance Minimization Inverse RL (DM-IRL) [Burchfiel et al., 2016]:** 상태의 특징 함수를 사용하고 참 보상 함수가 선형이라고 가정했습니다. 인간 피드백은 누적 보상 추정치로 제공되었으나, 신뢰도 부여보다 어려웠고 고차원 문제에는 부적합했습니다.
  * **Semi-supervised IRL (SSIRL) [Valko et al., 2012]:** Abbeel과 Ng [2004]의 IRL 방법을 확장하여 최적 및 준최적 궤적을 사용하고 전이 SVM(Transductive SVM)을 적용했습니다. 본 연구는 최적 시연 대신 신뢰도 점수를 사용하며, DM-IRL 및 SSIRL과 달리 고차원 문제에 더 적합합니다.
* **신뢰도 데이터를 활용한 준지도 분류(Semi-supervised Classification):**
  * **Zhou et al. [2014], Wang et al. [2013]:** 소수의 하드 레이블 데이터를 사용하여 가우시안 혼합 모델(Gaussian mixture models)이나 커널 밀도 추정기(kernel density estimator)로 레이블 없는 샘플의 신뢰도를 추정했습니다.
  * **El-Zahhar and El-Gayar [2010]:** 소프트 레이블($z \in [0,1]$)을 퍼지 입력으로 간주하고 k-최근접 이웃(k-nearest neighbors) 기반 분류 방식을 제안했습니다. 고차원 작업 확장과 이론적 보장이 부족했습니다.
  * **Ishida et al. [2018]:** 신뢰도가 부여된 긍정 데이터만으로 분류기를 학습하는 방식을 제안했습니다.
  * 본 연구의 2IWIL은 적은 신뢰도 데이터와 많은 레이블 없는 데이터를 함께 사용하여 분류기를 학습한다는 점에서 차별점을 가집니다.

## 🛠️ Methodology

본 연구에서는 불완전한 시연 데이터로부터 학습하기 위한 두 가지 접근 방식인 2IWIL과 IC-GAIL을 제안합니다.

* **문제 설정 (Section 4.1):**
  * 주어진 불완전한 시연은 최적 정책($\pi_{opt}$)과 비최적 정책($\Pi = \{\pi_i\}$)의 혼합으로 가정합니다.
  * 상태-행동 쌍 $x=(s,a)$에 대한 정규화된 점유 측정(normalized occupancy measure)은 $p(x) = \alpha p_{opt}(x) + (1-\alpha)p_{non}(x)$로 정의되며, 여기서 $\alpha = P(y=+1)$는 최적 정책의 사전 확률(class-prior probability)입니다.
  * 신뢰도 점수 $r(x) = P(y=+1|x)$는 최적 정책으로부터 $x$가 샘플링될 확률을 나타냅니다.
  * 데이터셋은 신뢰도 점수가 부여된 소량의 데이터 $D_c = \{(x_{c,i}, r_i)\}_{i=1}^{n_c}$와 레이블 없는 다량의 데이터 $D_u = \{x_{u,i}\}_{i=1}^{n_u}$로 구성됩니다.

* **Two-step Importance Weighting Imitation Learning (2IWIL) (Section 4.2):**
    1. **클래스 사전 확률 추정:** 신뢰도 데이터 $D_c$를 사용하여 $\hat{\alpha} = \frac{1}{n_c} \sum_{i=1}^{n_c} r_i$를 추정합니다.
    2. **확률 분류기 학습:** 레이블 없는 데이터의 신뢰도 점수를 예측하기 위해 준지도 분류(Semi-Conf, SC) 문제를 해결하는 확률 분류기를 학습합니다. 이 분류기는 다음의 위험 함수(risk function)를 최소화합니다:
        $$ R_{SC,L}(g) = E_{x,r \sim q}[r(L(g(x)) - L(-g(x))) + (1-\beta)L(-g(x))] + E_{x \sim p}[\beta L(-g(x))] $$
        여기서 $g$는 예측 함수, $L$은 엄격하게 적절한 합성 손실 함수(strictly proper composite loss function, 예: 로지스틱 손실)입니다. $\beta = \frac{n_u}{n_u + n_c}$를 사용하여 추정치의 분산을 최소화합니다 (Proposition 4.2). 또한 과적합 방지를 위해 비음수 위험 추정량(non-negative risk estimator)을 적용합니다.
    3. **신뢰도 점수 예측:** 학습된 분류기를 사용하여 $D_u$에 대한 신뢰도 점수 $\hat{r}_{u,i}$를 예측합니다.
    4. **가중치 기반 GAIL 수행:** 예측된 신뢰도 점수를 가중치로 사용하여 표준 GAIL 목적 함수를 최적화합니다. 에이전트 정책 $\pi_\theta$는 TRPO(Trust Region Policy Optimization)를 통해 업데이트됩니다.

* **Generative Adversarial Imitation Learning with Imperfect Demonstration and Confidence (IC-GAIL) (Section 4.3):**
    1. **점유 측정 직접 일치:** 2IWIL과 달리, IC-GAIL은 레이블 없는 데이터에 대한 신뢰도 예측 과정 없이 종단간(end-to-end) 방식으로 학습합니다. 이 방법은 주어진 시연의 점유 측정 $p$와 에이전트 정책 $\pi_\theta$ 및 비최적 정책의 혼합 점유 측정 $p' = \alpha p_\theta + (1-\alpha)p_{non}$ 사이의 젠슨-섀넌 발산(Jensen-Shannon Divergence, JSD)을 최소화하는 것을 목표로 합니다. Theorem 4.4에 따르면, 이 발산은 $p_\theta = p_{opt}$일 때만 최소화됩니다.
    2. **변환된 목적 함수:** 다음 min-max 목적 함수를 사용합니다:
        $$ \min_{\theta} \max_{w} E_{x \sim p}[\log(1-D_w(x))] + E_{x \sim p'}[\log D_w(x)] $$
        이는 실제 데이터($D_u$, $\pi_\theta$에서 샘플링된 $D_a$, $D_c$)를 사용하여 추정 가능하도록 다음과 같이 변환됩니다:
        $$ \tilde{V}(\pi_\theta, D_w) = E_{x \sim p}[\log(1-D_w(x))] + \alpha E_{x \sim p_\theta}[\log D_w(x)] + E_{x,r \sim q}[(1-r) \log D_w(x)] $$
    3. **실용적인 구현:** 클래스 사전 확률 $\hat{\alpha}$를 추정한 후, 에이전트 항의 영향이 미미해지는 것을 방지하기 위해 $\lambda = \max\{\tau, \hat{\alpha}\}$ (여기서 $\tau$는 임계값)를 적용하여 스케일링을 조절합니다. $\theta$와 $w$에 대해 교대로 경사 하강법(gradient descent)을 수행합니다.

## 📊 Results

* **성능 비교 (그림 1):** 제안된 2IWIL과 IC-GAIL 방법은 HalfCheetah-v2, Ant-v2, Hopper-v2, Swimmer-v2, Walker2d-v2와 같은 다양한 Mujoco 환경에서 모든 기준선 GAIL 방법보다 훨씬 우수한 성능을 보였습니다.
  * IC-GAIL은 일반적으로 2IWIL보다 최종 성능은 더 좋았지만 수렴 속도는 느렸습니다.
  * 신뢰도 정보를 활용한 GAIL (Reweight)은 신뢰도 정보를 사용하지 않은 GAIL (C)보다 우수하여 가중치 적용의 이점을 확인했습니다.
  * 표준 GAIL인 GAIL (U+C)는 불량한 성능을 보여 신뢰도 정보의 중요성을 강조했습니다.
* **노이즈에 대한 견고성 (그림 2):** Ant-v2 환경에서의 실험 결과, 신뢰도 점수에 가우시안 노이즈가 추가되더라도 2IWIL과 IC-GAIL 모두 상당히 견고한 성능을 유지했습니다. 이는 사람 라벨러의 잠재적인 부정확성에도 불구하고 실제 적용 가능성이 높음을 시사합니다.
* **레이블 없는 데이터의 영향 (그림 3):** 레이블 없는 데이터의 양이 증가함에 따라 두 방법 모두의 성능이 향상되는 것을 확인했습니다. 이는 신뢰도 데이터가 부족할 때 레이블 없는 데이터를 활용하는 것이 모방 학습 성능을 개선하는 데 유용하다는 동기를 뒷받침합니다.

## 🧠 Insights & Discussion

* **레이블 없는 데이터의 역할:**
  * **2IWIL:** 경험적 위험 추정량(empirical risk estimator)의 분산(variance)을 줄여주는 역할을 합니다.
  * **IC-GAIL:** '유도된 탐색(guided exploration)'과 유사하게 작동합니다. 신뢰도 정보가 희소한 보상 함수와 같다고 비유될 때, 비최적 시연으로부터 모방하여 학습된 정책을 신뢰도 정보를 사용하여 개선하는 데 도움을 줍니다.
* **신뢰도 데이터의 역할:**
  * **2IWIL:** 분류기를 학습하고 최적 분포 $p_{opt}$를 가중치화하는 데 사용되며, 이 두 단계 과정에서 오류가 누적될 수 있습니다.
  * **IC-GAIL:** 주어진 불완전 시연 내의 비최적 분포 $p_{non}$ 부분을 직접 보상하여 분포 $p$의 구성을 모방합니다. 이는 예측 오류를 피하고 종단간(end-to-end) 학습을 가능하게 합니다.
* **클래스 사전 확률 $\alpha$의 영향:**
  * **2IWIL:** $p_{opt}$를 재가중치화하는 정규화 상수로 작용하여 에이전트 정책의 수렴에 직접적인 영향을 미치지 않습니다.
  * **IC-GAIL:** 에이전트 항 $p_\theta$가 $\alpha$에 의해 직접 스케일링됩니다. $\alpha$가 작을 경우, 보상 함수가 거의 상수에 가까워져 에이전트의 학습 속도가 느려질 수 있습니다. 실용적인 구현에서 임계값 $\lambda = \max\{\tau, \alpha\}$를 사용하는 것은 이러한 문제를 완화하기 위함입니다.
* **트레이드오프:** 2IWIL은 더 빠른 수렴 속도를 보여 환경과의 상호작용 비용이 높은 경우 유리할 수 있습니다. 반면 IC-GAIL은 종단간 특성 덕분에 더 나은 최종 성능을 달성하지만 학습 속도는 느릴 수 있습니다.
* **일반적인 적용 가능성:** 제안된 접근 방식들은 다른 IL 및 IRL 방법론으로도 쉽게 확장될 수 있습니다.
* **향후 연구:** 본 연구의 신뢰도 개념을 이산 시퀀스 생성(discrete sequence generation)과 같은 다양한 응용 분야(예: 화합물의 용해도 예측)로 확장할 수 있습니다.

## 📌 TL;DR

본 논문은 불완전한 시연 데이터(일부는 신뢰도 점수 포함, 대부분은 레이블 없음)로부터 최적 정책을 학습하는 모방 학습(IL) 문제를 다룹니다. 이를 위해 **2IWIL(Two-step Importance Weighting IL)**과 **IC-GAIL(Generative Adversarial IL with Imperfect Demonstration and Confidence)**이라는 두 가지 새로운 신뢰도 기반 IL 방법론을 제안합니다.

**2IWIL**은 제한된 신뢰도 데이터와 풍부한 레이블 없는 데이터를 사용하여 확률 분류기를 학습하고, 이를 통해 모든 시연에 대한 신뢰도 점수를 예측한 후 GAIL 목표에 가중치로 적용하는 2단계 접근 방식입니다. 반면 **IC-GAIL**은 비최적 분포에 맞춰 직접 점유 측정(occupancy measure)을 일치시키는 종단간(end-to-end) 방식으로, 2IWIL의 예측 오류 누적을 피합니다. 또한 에이전트 정책의 영향을 조절하기 위해 클래스 사전 확률을 임계값과 함께 사용합니다.

실험 결과, 두 방법론 모두 기존 GAIL 기준선보다 훨씬 우수한 성능을 보였으며, 신뢰도 점수의 활용과 레이블 없는 데이터의 가치를 입증했습니다. 또한 신뢰도 점수의 노이즈에 대해 견고하며, 레이블 없는 데이터가 많을수록 성능이 향상됨을 확인했습니다. IC-GAIL은 최종 성능이 더 우수하지만 2IWIL보다 수렴 속도가 느리다는 트레이드오프를 가집니다.
