# Restoring Noisy Demonstration for Imitation Learning with Diffusion Models

Shang-Fu Chen, Co Yong, Shao-Hua Sun (2025)

## 🧩 Problem to Solve

모방 학습(Imitation Learning, IL)은 전문가의 시연(demonstrations)으로부터 정책(policy)을 학습하는 것을 목표로 하며, 환경과의 상호작용이나 보상 신호 없이도 학습이 가능하다는 장점이 있다. 그러나 대부분의 기존 IL 알고리즘은 전문가의 시연 데이터가 완벽(perfect)하다는 가정을 전제로 한다.

실제 환경에서 수집되는 데이터는 인간 전문가의 실수, 센서의 오차, 또는 제어 시스템의 부정확성으로 인해 필연적으로 노이즈를 포함하게 된다. 이러한 노이즈가 포함된 시연 데이터를 그대로 사용할 경우, 학습된 정책의 성능이 심각하게 저하되는 문제가 발생한다. 따라서 본 논문의 목표는 내재된 노이즈가 존재하는 전문가 시연 데이터를 효과적으로 활용하기 위해, 이를 필터링하고 복구(restore)하여 신뢰할 수 있는 데이터를 생성하는 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 **Diffusion-Model-Based Demonstration Restoration (DMDR)**라고 불리는 '필터링 후 복구(filter-and-restore)' 프레임워크를 제안한 것이다.

중심적인 직관은 단순히 노이즈가 섞인 데이터를 제거하거나 낮은 가중치를 부여하는 기존 방식(예: BCND)과 달리, **데이터의 일부를 정제하여 깨끗한 샘플을 추출하고, 이를 통해 학습된 조건부 확산 모델(Conditional Diffusion Model)을 사용하여 나머지 노이즈 데이터를 복구**함으로써 데이터 효율성을 극대화하는 것이다. 특히 상태(state)와 행동(action) 사이의 상관관계를 활용하여 서로를 복구하는 상호 보완적 구조를 설계하였다.

## 📎 Related Works

기존의 모방 학습 연구는 크게 온라인 IL과 오프라인 IL로 나뉜다. BC(Behavioral Cloning), IBC(Implicit BC), Diffusion Policy와 같은 오프라인 방식은 환경 상호작용이 불가능한 상황에서 유용하지만, 데이터 품질에 극도로 의존한다.

노이즈가 섞인 데이터로부터 학습하려는 기존 시도들은 다음과 같은 한계를 가진다:

- **가중치 기반 방식(예: BCND):** 노이즈가 심한 샘플에 낮은 가중치를 부여하여 무시하는 전략을 취하지만, 이는 노이즈 데이터 내에 존재할 수 있는 유용한 정보를 완전히 버린다는 단점이 있다.
- **데이터 복구 방식(예: DDRM, GibbsDDRM):** 확산 모델을 이용한 복구를 시도하지만, 이는 선형 퇴화 모델(linear degradation model)과 같은 명확한 물리적/수학적 손상 모델이 미리 정의되어 있어야 한다. 하지만 실제 IL 환경에서는 어떤 노이즈가 섞였는지 알 수 없는 'Blind' 상황이 대부분이다.

DMDR은 이러한 한계를 극복하기 위해 사전 정의된 손상 모델 없이, 데이터 자체에서 정제된 부분집합을 찾아내고 이를 기반으로 조건부 확산 모델을 학습시키는 방식을 취한다.

## 🛠️ Methodology

DMDR 프레임워크는 크게 두 단계인 '시연 필터링(Demonstration Filtering)'과 '시연 복구(Demonstration Restoration)' 단계로 구성된다.

### 1. Demonstration Filtering

상태 $s$와 행동 $a$가 서로 다른 센서에 의해 측정되어 독립적으로 노이즈가 발생한다고 가정하고, 각각을 독립적으로 필터링한다.

- **Autoencoder(AE)를 통한 특징 추출:** 상태와 행동 각각에 대해 오토인코더를 학습시켜 데이터의 전역적 특징을 캡처한다. 손실 함수는 다음과 같은 재구성 손실(reconstruction loss)을 사용한다.
  $$L_{s}^{rec} = \mathbb{E}_{(s,a) \sim D} [\|\phi(s) - s\|^2], \quad L_{a}^{rec} = \mathbb{E}_{(s,a) \sim D} [\|\phi(a) - a\|^2]$$
- **LOF(Local Outlier Factor) 적용:** 단순히 재구성 오차만 사용하면 드물지만 중요한 샘플(예: 물체를 잡는 순간)을 노이즈로 오인해 삭제할 위험이 있다. 이를 방지하기 위해 AE의 보틀넥 특징(bottleneck features) 공간에서 LOF 알고리즘을 적용하여 국소 밀도를 기반으로 이상치를 탐지한다.
- **데이터 분할:** LOF 점수를 기준으로 하위 50%를 깨끗한 샘플($\hat{s}, \hat{a}$)로, 나머지 50%를 노이즈 샘플($s', a'$)로 라벨링하여 네 가지 부분집합($D_{(\hat{s},\hat{a})}, D_{(s',\hat{a})}, D_{(\hat{s},a')}, D_{(s',a')}$)으로 나눈다.

### 2. Demonstration Restoration

필터링된 깨끗한 부분집합 $D_{(\hat{s},\hat{a})}$를 사용하여 조건부 DDPM(Denoising Diffusion Probabilistic Models)을 학습한다.

- **조건부 확산 모델:** 상태를 복구하는 모델 $\theta_s$는 깨끗한 행동 $\hat{a}$를 조건으로 하며, 행동을 복구하는 모델 $\theta_a$는 깨끗한 상태 $\hat{s}$를 조건으로 한다.
  $$L_{s}^{diff} = \mathbb{E}_{t \sim U(0,T), (\hat{s},\hat{a}) \sim D_{(\hat{s},\hat{a})}} [\|\epsilon - \epsilon_{\theta_s}(s_t, \hat{a}, t)\|^2]$$
  $$L_{a}^{diff} = \mathbb{E}_{t \sim U(0,T), (\hat{s},\hat{a}) \sim D_{(\hat{s},\hat{a})}} [\|\epsilon - \epsilon_{\theta_a}(a_t, \hat{s}, t)\|^2]$$
- **노이즈 타임스텝 예측기(Noise Timestep Predictor):** 모든 노이즈 데이터가 동일한 수준으로 오염되지 않았으므로, 각 샘플에 맞는 적절한 역확산 단계(denoising step)를 결정하기 위해 예측기 $\psi_s, \psi_a$를 학습한다.
  $$L_{s}^{pred} = \mathbb{E}_{t \sim U(0,T), (\hat{s},\hat{a}) \sim D_{(\hat{s},\hat{a})}} [\|t - \psi_s(s_t, \hat{a})\|^2]$$

### 3. 추론 및 복구 절차

- 노이즈 상태 $s'$가 포함된 $(s', \hat{a})$ 쌍에 대해, 예측기 $\psi_s$를 통해 노이즈 타임스텝 $t^*$를 예측한다.
- $t^*$가 임계값 $t_{thres}$보다 작으면 그대로 깨끗한 데이터로 간주하고, 크면 $\theta_s$를 이용해 복구된 상태 $s^*$를 생성한다.
- 최종적으로 정제된 데이터셋을 구축하고, 이를 BC, IBC, Diffusion Policy 등 기존의 IL 알고리즘의 학습 데이터로 사용한다.

## 📊 Results

### 실험 설정

- **작업 영역:** Robot Arm Manipulation (FETCHPICK, FETCHPUSH), Dexterous Manipulation (HANDROTATE), Locomotion (WALKER).
- **노이즈 설정:** 전문가 데이터에 확률 $p$로 가우시안 노이즈를 주입하여 시뮬레이션하였다.
- **비교 대상:** BC, Ensemble BC, BCND.

### 정량적 결과

모든 작업에서 DMDR이 기존 베이스라인을 압도하는 성능을 보였다. 특히 Ensemble DMDR의 경우 모든 태스크에서 가장 높은 성공률(Success Rate)과 리턴(Return) 값을 기록하였다.

- **FETCHPICK:** BC(44.38%) $\rightarrow$ DMDR(90.52%) $\rightarrow$ Ensemble DMDR(91.80%)
- **WALKER:** BC(4456.8) $\rightarrow$ DMDR(5066.4) $\rightarrow$ Ensemble DMDR(6168.1)

### 정성적 결과 및 분석

- **t-SNE 시각화:** FETCHPICK 환경에서 복구된 샘플(blue)이 노이즈 샘플(red)보다 깨끗한 샘플(green)의 분포에 훨씬 가깝게 위치함을 확인하여 복구 성능을 시각적으로 증명하였다.
- **필터링 절차 분석:** AE 단독 또는 LOF 단독 사용보다 AE+LOF 결합 방식이 훨씬 높은 성공률을 보였으며, 이는 전역적 특징과 국소적 밀도를 모두 고려하는 것이 중요함을 시사한다.
- **노이즈 강건성:** 가우시안뿐만 아니라 라플라스(Laplacian), 편향된(Biased), 혼합(Mixed) 노이즈 상황에서도 일관된 성능 향상을 보였다. 다만, 균등 분포(Uniform) 노이즈의 경우 LOF의 밀도 기반 탐지 한계로 인해 성능 향상이 미미했다.

## 🧠 Insights & Discussion

### 강점

DMDR은 특정 노이즈 모델에 의존하지 않고 데이터 자체에서 정제된 부분을 찾아 이를 이용해 나머지 데이터를 복구한다는 점에서 실용성이 매우 높다. 또한 특정 IL 알고리즘에 종속되지 않고, 데이터 전처리 단계로서 작동하므로 BC, IBC, Diffusion Policy 등 다양한 정책 학습 모델에 즉시 적용 가능하다는 확장성을 가진다.

### 한계 및 비판적 해석

- **Uniform Noise 취약성:** LOF는 주변 샘플과의 밀도 차이를 이용하는데, 균등 분포 노이즈는 공간을 고르게 채우기 때문에 밀도 차이가 발생하지 않아 필터링 성능이 떨어진다. 이는 향후 시간적(temporal) 구조나 모델 기반의 이상치 탐지 기법 도입이 필요함을 보여준다.
- **데이터 버림 문제:** 상태와 행동이 모두 오염된 부분집합 $D_{(s', a')}$는 복구를 위한 조건(condition)이 부족하여 현재 모두 폐기된다. 이 데이터를 어떻게 재활용할 것인가가 데이터 효율성 측면에서 해결해야 할 과제이다.
- **최적성(Optimality) 문제:** 본 논문은 '손상(corruption)'된 데이터를 복구하는 데 집중하고 있으며, 데이터가 깨끗하더라도 전문가의 실력 자체가 낮은 '차선(sub-optimal)' 데이터에 대한 해결책은 제시하지 않았다.

## 📌 TL;DR

본 논문은 노이즈가 섞인 전문가 시연 데이터로부터 효율적으로 학습하기 위해 **Autoencoder+LOF 필터링**과 **조건부 확산 모델 기반 복구**를 결합한 **DMDR 프레임워크**를 제안한다. 이 방법은 노이즈 데이터를 단순히 버리는 것이 아니라, 깨끗한 샘플을 통해 학습한 모델로 노이즈 데이터를 복구하여 학습에 활용함으로써, 다양한 로봇 제어 작업에서 기존 BCND 등의 방식보다 월등한 성능 향상을 이끌어냈다. 이 연구는 실제 세계의 불완전한 데이터셋을 정제하여 고성능 정책을 학습시키는 데 중요한 기여를 할 것으로 보인다.
