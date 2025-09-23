# FLOW MATCHING FOR GENERATIVE MODELING

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le

## 🧩 Problem to Solve

최근 이미지 생성 분야에서 놀라운 발전을 이끈 딥 생성 모델, 특히 Diffusion 모델은 주로 단순한 확산(diffusion) 프로세스에 의존합니다. 이는 샘플링 확률 경로의 공간을 제한하고, 결과적으로 매우 긴 학습 시간과 효율적인 샘플링을 위한 특수 방법의 필요성을 야기합니다. 한편, Continuous Normalizing Flows (CNFs)는 임의의 확률 경로를 모델링할 수 있는 일반적인 프레임워크임에도 불구하고, 확장 가능한 학습 알고리즘이 부족합니다. 기존 최대 우도(Maximum Likelihood) 학습은 비용이 많이 드는 수치 ODE 시뮬레이션을 필요로 하며, 시뮬레이션-프리(simulation-free) 방법들은 다루기 힘든 적분이나 편향된 기울기 문제에 직면했습니다. 따라서 Diffusion 모델의 한계를 극복하고 CNF를 대규모로 효율적으로 학습할 수 있는 새로운 생성 모델링 패러다임이 필요합니다.

## ✨ Key Contributions

- **Flow Matching (FM) 프레임워크 도입:** 고정된 조건부 확률 경로의 벡터 필드를 회귀(regressing)하는 시뮬레이션-프리 접근 방식을 통해 CNF를 대규모로 학습하기 위한 새로운 패러다임을 제안합니다.
- **Conditional Flow Matching (CFM) 목표 함수:** 다루기 힘든 주변(marginal) 벡터 필드에 대한 명시적 지식 없이도 주변 확률 경로를 생성하는 CNF를 학습할 수 있게 하는 목표 함수를 개발했습니다. 이는 기존 Score Matching의 아이디어를 벡터 필드 매칭으로 일반화한 것입니다.
- **일반적인 가우시안 확률 경로군:** 기존 Diffusion 경로를 특수한 경우로 포함하는 일반적인 가우시안 조건부 확률 경로군과 호환되는 FM 프레임워크를 제시합니다.
- **Diffusion 경로와 FM의 결합:** 기존 Diffusion 경로에서도 FM을 사용하면 Score Matching보다 더 강건하고 안정적인 학습 및 우수한 성능을 달성함을 발견했습니다.
- **Optimal Transport (OT) 확률 경로:** Diffusion 경로보다 효율적인 Optimal Transport (OT) 변위 보간을 기반으로 한 새로운 확률 경로를 도입했습니다. 이 경로는 더 빠른 학습, 샘플링, 더 나은 일반화 성능을 제공합니다.
- **이미지넷(ImageNet) 규모 CNF 학습:** FM을 사용하여 ImageNet 데이터셋에서 CNF를 성공적으로 학습하여, 기존 Diffusion 기반 방법들보다 우도(likelihood) 및 샘플 품질 면에서 일관되게 우수한 성능을 달성했으며, 상용 ODE 솔버를 이용한 빠르고 신뢰할 수 있는 샘플 생성을 가능하게 했습니다.
- **이론적 증명:** 주변 벡터 필드가 주변 확률 경로를 생성한다는 것(정리 1)과 FM 및 CFM 목표 함수의 기울기가 기대값에서 동일하다는 것(정리 2)을 증명하여 CFM의 유효성을 뒷받침했습니다.

## 📎 Related Works

- **Continuous Normalizing Flows (CNFs):** (Chen et al., 2018)에 의해 처음 소개되었으며, (Grathwohl et al., 2018) 등에서 최대 우도 학습의 높은 계산 비용이 지적되었습니다. (Dupont et al., 2019; Finlay et al., 2020) 등의 연구는 ODE의 해결을 용이하게 하기 위한 정규화 기법을 시도했습니다.
- **시뮬레이션-프리 CNF 학습:** (Rozen et al., 2021)은 선형 보간을 사용했지만 고차원에서의 적분 문제에 직면했고, (Ben-Hamu et al., 2022)는 일반적인 확률 경로를 다루었으나 편향된 기울기 문제가 있었습니다.
- **Diffusion Models:** (Sohl-Dickstein et al., 2015; Ho et al., 2020; Song & Ermon, 2019) 등에서 확산 프로세스를 통해 확률 경로를 간접적으로 정의했습니다. (Song et al., 2020b)는 Diffusion 모델이 denoising score matching으로 학습된다는 점을 밝혔으며, (Dhariwal & Nichol, 2021; Nichol & Dhariwal, 2021) 등에서 손실 재조정, 분류기 가이던스 등을 통해 성능을 개선했습니다.
- **Optimal Transport (OT):** (McCann, 1997) 등의 이론적 배경을 바탕으로, 본 연구는 OT 변위 맵을 조건부 확률 경로 정의에 활용합니다.
- **유사 동시 연구:** (Liu et al., 2022; Albergo & Vanden-Eijnden, 2022) 등에서도 시뮬레이션-프리 CNF 학습을 위한 유사한 조건부 목표 함수가 독립적으로 제안되었습니다.

## 🛠️ Methodology

1. **Flow Matching (FM) 목표 함수:**
   - 주변 확률 경로 $p_t(x)$와 이를 생성하는 목표 벡터 필드 $u_t(x)$가 주어졌을 때, 학습 가능한 CNF 벡터 필드 $v_t(x)$를 $u_t(x)$에 매칭시키는 목표 함수를 정의합니다.
   - $$ L*{\text{FM}}(\theta) = E*{t, p_t(x)} ||v_t(x) - u_t(x)||^2 $$
   - 하지만 $p_t(x)$와 $u_t(x)$를 직접 알기 어렵기 때문에 이 목표를 그대로 사용하기는 어렵습니다.
2. **조건부 확률 경로 및 벡터 필드 구성:**
   - 각 데이터 샘플 $x_1$에 대해 조건부 확률 경로 $p_t(x|x_1)$를 정의합니다. 이 경로는 $t=0$에서 단순한 분포($N(x|0,I)$)에서 시작하여 $t=1$에서 $x_1$ 주변에 집중된 분포($N(x|x_1, \sigma^2 I)$)로 이동합니다.
   - 주변 확률 경로 $p_t(x)$와 주변 벡터 필드 $u_t(x)$는 이러한 조건부 경로 및 벡터 필드를 데이터 분포 $q(x_1)$에 대해 주변화(marginalize)하여 구성됩니다.
   - **정리 1**은 주변 벡터 필드가 주변 확률 경로를 생성함을 증명하여, 조건부 경로로부터 전체 경로를 모델링하는 이론적 기반을 마련합니다.
3. **Conditional Flow Matching (CFM) 목표 함수:**
   - 주변 경로와 필드의 다루기 힘든 적분 문제를 피하기 위해, 조건부 벡터 필드 $u_t(x|x_1)$에 직접 회귀하는 CFM 목표 함수를 제안합니다.
   - $$ L*{\text{CFM}}(\theta) = E*{t, q(x_1), p_t(x|x_1)} ||v_t(x) - u_t(x|x_1)||^2 $$
   - **정리 2**는 $L_{\text{FM}}(\theta)$와 $L_{\text{CFM}}(\theta)$의 기울기가 기대값에서 동일함을 증명하여, CFM을 최적화하는 것이 FM을 최적화하는 것과 같음을 보입니다. 이는 효율적인 학습을 가능하게 합니다.
4. **가우시안 조건부 확률 경로 및 벡터 필드:**
   - 조건부 확률 경로로 $p_t(x|x_1) = N(x|\mu_t(x_1), \sigma_t(x_1)^2 I)$ 형태의 가우시안 분포를 사용합니다.
   - $t=0$에서는 $\mu_0=0, \sigma_0=1$ (표준 가우시안 노이즈), $t=1$에서는 $\mu_1=x_1, \sigma_1=\sigma_{\text{min}}$ (데이터 샘플 $x_1$에 집중된 분포)으로 경계 조건을 설정합니다.
   - **정리 3**은 이러한 가우시안 경로를 생성하는 고유한 벡터 필드 $u_t(x|x_1)$가 다음 형태를 가짐을 증명합니다:
     $$ u_t(x|x_1) = \frac{\sigma'\_t(x_1)}{\sigma_t(x_1)}(x - \mu_t(x_1)) + \mu'\_t(x_1) $$
5. **특수 인스턴스:**
   - **Diffusion 조건부 VFs:** 기존의 Variance Exploding (VE) 및 Variance Preserving (VP) Diffusion 경로들이 $\mu_t, \sigma_t$의 특정 함수를 사용하여 FM 프레임워크에 통합될 수 있음을 보여줍니다.
   - **Optimal Transport (OT) 조건부 VFs:** $\mu_t(x_1) = tx_1$ 및 $\sigma_t(x_1) = 1-(1-\sigma_{\text{min}})t$와 같이 평균과 표준편차가 시간 $t$에 따라 선형적으로 변하도록 정의합니다. 이는 두 가우시안 분포 사이의 Optimal Transport 변위 맵에 해당하며, 입자가 항상 직선 궤적과 일정한 속도로 이동하는 특성을 가집니다. 이 OT 벡터 필드는 Diffusion VF보다 학습하기 더 간단합니다.
6. **샘플링:** 학습된 CNF 벡터 필드 $v_t$를 사용하여 표준 ODE 솔버로 $\frac{d}{dt}\phi_t(x) = v_t(\phi_t(x))$를 $t \in [0,1]$ 구간에서 해결하여 노이즈 $x_0 \sim N(0,I)$로부터 데이터 샘플 $\phi_1(x_0)$를 생성합니다.

## 📊 Results

- **이미지넷(ImageNet) 생성 성능:**
  - CIFAR-10 및 ImageNet 32x32, 64x64, 128x128 해상도에서 평가되었습니다.
  - **Flow Matching w/ OT (FM-OT)**는 Negative Log-Likelihood (NLL, bits per dimension) 및 Frechet Inception Distance (FID) 면에서 경쟁 방법 대비 일관되게 **가장 우수한 성능**을 보였습니다. 또한, 함수 평가 횟수(NFE)도 가장 낮아 효율성을 입증했습니다 (Table 1).
  - ImageNet-128에서 FID 20.9를 달성하며, 조건부 모델인 IC-GAN을 제외하고 최첨단 성능을 기록했습니다.
- **빠른 학습:**
  - FM은 기존 Score Flow 및 VDM 등 다른 Diffusion 모델에 비해 **훨씬 빠르게 수렴**하며 더 낮은 FID를 달성했습니다 (Figure 5).
  - ImageNet-128 모델 학습 시, 기존 Diffusion 모델 대비 33% 적은 이미지 처리량으로 더 적은 반복 횟수(500k vs 1.3M/10M)를 사용했습니다.
- **효율적인 샘플링:**
  - **FM-OT** 모델은 사용된 모든 ODE 솔버에서 **가장 효율적인 샘플러**를 제공했습니다.
  - OT 경로 모델은 Diffusion 경로 모델보다 **더 이른 시점에 이미지 생성을 시작**하며, Diffusion 경로는 경로의 마지막에 가서야 노이즈가 제거되는 경향을 보였습니다 (Figure 6).
  - 낮은 NFE(≤100) 샘플링에서, **FM w/ OT**는 유사한 수치 오차를 달성하는 데 Diffusion 모델의 **약 60%의 NFE만 필요**했으며, 낮은 NFE에서도 우수한 FID를 달성하여 샘플 품질과 비용 간의 더 나은 균형을 제공했습니다 (Figure 7).
- **조건부 생성 (Super-resolution):**
  - 64x64 이미지를 256x256으로 업샘플링하는 조건부 이미지 생성 태스크에서 **FM w/ OT**는 SR3과 유사한 PSNR, SSIM 값을 보이면서도 FID 및 IS (Inception Score)를 크게 개선했습니다 (Table 2).

## 🧠 Insights & Discussion

- **CNF 학습의 새로운 시대:** Flow Matching은 CNF 학습을 위한 시뮬레이션-프리 프레임워크를 제공하여, 이전에는 불가능했던 대규모 CNF 학습의 길을 열었습니다.
- **Diffusion 모델의 재해석 및 확장:** FM은 Diffusion 모델을 새로운 관점에서 바라보게 하며, 확률 경로를 직접 명세함으로써 확률적/확산 구성에 얽매이지 않고 더 유연하게 모델을 설계할 수 있음을 시사합니다.
- **OT 경로의 탁월한 이점:** Optimal Transport (OT) 확률 경로의 사용은 입자의 직선 궤적 및 일정한 속도 특성 덕분에 더 빠른 학습과 샘플링, 더 나은 일반화 및 샘플 품질로 이어집니다. 이는 Diffusion 경로의 "오버슈트" 문제를 효과적으로 방지합니다.
- **강건성과 안정성:** Diffusion 경로와 함께 FM을 사용하더라도 기존 Score Matching보다 더 강건하고 안정적인 학습을 제공하여, Diffusion 모델 학습의 대안으로 활용될 수 있습니다.
- **확장성 및 유연성:** 조건부 구성을 통해 높은 차원의 이미지 데이터셋에 대한 학습이 용이하며, 향후 비등방성 가우시안 또는 더 일반적인 커널과 같은 다양한 확률 경로를 탐색할 수 있는 문을 열었습니다.
- **에너지 효율성:** 더 적은 기울기 업데이트와 이미지 처리량으로 모델을 학습할 수 있는 능력은 상당한 시간 및 에너지 절약으로 이어져, 지속 가능한 딥러닝 연구에 기여할 수 있습니다.

## 📌 TL;DR

이 논문은 대규모 Continuous Normalizing Flows (CNF) 학습의 비효율성을 해결하기 위해 **Flow Matching (FM)**이라는 새로운 시뮬레이션-프리 생성 모델링 패러다임을 제안합니다. FM은 데이터 샘플에 대한 조건부 확률 경로의 벡터 필드를 직접 회귀하는 방식으로 CNF를 학습시키며, 특히 기존 Diffusion 경로를 포괄하면서도 더 효율적인 **Optimal Transport (OT) 기반 확률 경로**를 도입합니다. 실험 결과, FM은 ImageNet과 같은 대규모 데이터셋에서 기존 Diffusion 기반 방법보다 **우도 및 샘플 품질 모두에서 우수한 성능**을 달성했으며, 더 빠르고 안정적인 학습과 효율적인 샘플 생성을 가능하게 하여 CNF 기반 생성 모델링의 새로운 가능성을 제시합니다.
