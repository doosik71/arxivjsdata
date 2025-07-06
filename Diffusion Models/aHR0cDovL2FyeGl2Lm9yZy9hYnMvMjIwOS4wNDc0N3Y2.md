# Diffusion Models in Vision: A Survey
Florinel-Alin Croitoru, Vlad Hondru, Radu Tudor Ionescu, Mubarak Shah

## 🧩 Problem to Solve
확산 모델(Diffusion Models)이 컴퓨터 비전 분야에서 인상적인 생성 능력을 보이며 빠르게 부상하고 있지만, 이 모델들에 대한 포괄적이고 시의적절한 이론 및 실용적 관점의 조망이 부족합니다. 다양한 응용 분야와 기본 프레임워크를 체계적으로 분류하고, 현재의 한계를 분석하며, 향후 연구 방향을 제시하는 것이 이 논문의 목표입니다.

## ✨ Key Contributions
- 컴퓨터 비전 분야에서 노이즈 제거 확산 모델(Denoising Diffusion Models)에 대한 시의적절하고 포괄적인 문헌을 검토하여 확산 모델링 프레임워크에 대한 빠른 이해를 돕습니다.
- 확산 모델의 세 가지 주요 프레임워크(DDPMs, NCSNs, SDEs)를 정의하고 제시합니다.
- VAEs, GANs, EBMs, Autoregressive Models, Normalizing Flows 등 다른 딥 생성 모델과의 관계를 분석합니다.
- 컴퓨터 비전 분야에 적용된 확산 모델을 여러 관점(기반 프레임워크, 대상 작업, 노이즈 제거 조건, 데이터셋)에서 다각도로 분류하는 체계를 제안합니다.
- 확산 모델의 현재 한계점(예: 샘플링 시 낮은 속도)을 제시하고, 이를 극복할 수 있는 흥미로운 미래 연구 방향을 제시합니다.

## 📎 Related Works
이 논문은 확산 모델 분야의 포괄적인 개요를 제공하며, 다음을 포함한 다양한 선행 연구를 참조합니다.
- **주요 확산 모델:**
  - Denoising Diffusion Probabilistic Models (DDPMs) [1, 2]
  - Noise Conditioned Score Networks (NCSNs) [3]
  - 확률 미분 방정식(Stochastic Differential Equations, SDEs) 기반 모델 [4]
- **다른 딥 생성 모델:**
  - Variational Auto-Encoders (VAEs) [50]
  - Generative Adversarial Networks (GANs) [52]
  - Energy-Based Models (EBMs) [60, 61]
  - Autoregressive Models [62, 70]
  - Normalizing Flows [63, 64]
- **획기적인 모델:**
  - Imagen [12]
  - Latent Diffusion Models (LDMs) [10] (Stable Diffusion [10] 포함)

## 🛠️ Methodology
확산 모델은 데이터에 점진적으로 노이즈를 추가하여 순수한 가우시안 노이즈로 변환하는 **순방향 확산(Forward Diffusion)** 단계와, 이 노이즈를 점진적으로 제거하여 원본 데이터를 복원하는 **역방향 확산(Reverse Diffusion)** 단계를 기반으로 합니다.
- **Denoising Diffusion Probabilistic Models (DDPMs) [1, 2]:**
  - 비평형 열역학에서 영감을 받아 가우시안 노이즈를 점진적으로 추가하여 데이터를 손상시킵니다.
  - 순방향 프로세스는 마르코프 체인(Markovian process)으로 정의되며, 특정 시간 $t$에서의 노이즈가 추가된 이미지 $x_t$는 원본 이미지 $x_0$와 관계식: $p(x_t|x_0) = \mathcal{N}(x_t; \sqrt{\hat{\beta}_t} \cdot x_0, (1 - \hat{\beta}_t) \cdot I)$ 로 직접 샘플링될 수 있습니다. 여기서 $\hat{\beta}_t = \prod_{i=1}^t \alpha_i$ 이고 $\alpha_t = 1 - \beta_t$ 입니다.
  - 역방향 프로세스(생성)는 신경망(주로 U-Net [65] 기반)을 훈련시켜 각 단계에서 추가된 노이즈를 예측하고, 이를 통해 원본 이미지를 복구하도록 학습합니다.
  - 목적 함수는 실제 노이즈 $z_t$와 모델의 노이즈 추정치 $z_{\theta}(x_t, t)$ 사이의 제곱 오차를 최소화하는 단순화된 형태로 사용됩니다: $L_{\text{simple}} = \mathbb{E}_{t \sim [1,T]} \mathbb{E}_{x_0 \sim p(x_0)} \mathbb{E}_{z_t \sim \mathcal{N}(0,I)} ||z_t - z_{\theta}(x_t, t)||^2$.
- **Noise Conditioned Score Networks (NCSNs) [3]:**
  - 스코어 함수($\nabla_{x}\text{log }p(x)$)를 추정하도록 신경망을 훈련합니다.
  - 훈련은 노이즈 제거 스코어 매칭(denoising score matching) [67]을 통해 다양한 노이즈 수준에서 교란된 데이터 분포의 스코어 함수를 학습합니다. 목적 함수는 $\mathbb{E} ||s_{\theta}(x_t, \sigma_t) + \frac{x_t - x}{\sigma_t^2}||^2$와 같은 형태입니다.
  - 샘플링은 담금질 랑주뱅 동역학(Annealed Langevin dynamics) [3]을 사용하며, 여러 노이즈 스케일($\sigma_t$)에 걸쳐 점진적으로 노이즈를 제거하여 데이터를 생성합니다.
- **확률 미분 방정식 (Stochastic Differential Equations, SDEs) [4]:**
  - DDPM과 NCSN을 일반화하여 확산 과정을 연속적인 확률 미분 방정식으로 모델링합니다.
  - 순방향 SDE는 데이터를 노이즈로 변환하고, 역방향 SDE는 노이즈에서 데이터를 복구하는 과정을 나타냅니다.
  - 신경망은 스코어 함수 $s_{\theta}(x, t) \approx \nabla_{x}\text{log }p_t(x)$를 추정하며, 훈련 목적 함수는 연속적인 경우에 맞춰진 $L^*_{\text{dsm}}$ 형태입니다.
  - 샘플링은 Euler-Maruyama [4] 또는 Predictor-Corrector [4]와 같은 수치 SDE 솔버를 통해 수행됩니다.

## 📊 Results
- 확산 모델은 이미지 생성에서 GAN(Generative Adversarial Networks)을 능가하는 샘플 품질과 다양성을 보여주며(예: Imagen, Stable Diffusion), 매우 인상적인 결과를 달성했습니다.
- 훈련 시 보지 못했던 비현실적인 시나리오를 생성하는 등 높은 일반화 능력을 입증했습니다. (논문 Figure 2 참조)
- 이미지 생성(비조건부/조건부), 초해상도, 인페인팅, 이미지 편집, 이미지 간 변환, 분할, 의료 영상, 이상 탐지, 비디오 생성 등 광범위한 컴퓨터 비전 작업에 성공적으로 적용되었습니다. (논문 Table 1은 적용된 작업과 데이터셋에 대한 포괄적인 목록을 제공합니다.)
- 학습된 잠재 표현을 활용하여 이미지 분할, 분류, 이상 탐지 등 판별 작업에서도 유망한 결과를 보입니다.

## 🧠 Insights & Discussion
- **한계:**
  - **계산 부담:** 가장 큰 단점은 샘플당 수천 단계의 계산이 필요하여 추론 시간이 매우 길다는 점입니다. 이는 GANs에 비해 여전히 비효율적입니다.
  - **텍스트-이미지 생성의 한계:** CLIP 임베딩을 사용하는 텍스트-이미지 생성 모델은 CLIP 임베딩이 철자 정보를 포함하지 않아 이미지 내에서 읽을 수 있는 텍스트를 생성하는 데 어려움을 겪을 수 있습니다(예: Imagen).
- **의미:** 확산 모델은 생성 모델링의 기준을 높였으며, GANs가 겪는 훈련 난이도나 모드 붕괴 문제 없이 안정적인 훈련 과정과 다양한 샘플 생성을 제공합니다.
- **미래 방향:**
  - **효율성 향상:** 샘플링 속도 개선을 위한 효율적인 옵티마이저의 업데이트 규칙 도입 등.
  - **새로운 컴퓨터 비전 작업:** 이미지 디헤이징(dehazing), 비디오 이상 탐지, 시각적 질의응답(VQA) 등 확산 모델이 적용될 수 있는 새로운 분야 탐색.
  - **잠재 공간 활용:** 확산 모델이 학습한 잠재 표현의 품질과 판별 작업(분류, 회귀, 객체 탐지)에서의 유용성 평가.
  - **비디오 모델링 확장:** 비디오에서 가능한 미래를 시뮬레이션하거나, 장기적인 시간적 관계 및 객체 간 상호작용을 모델링하는 연구.
  - **다목적 모델 개발:** 텍스트, 클래스 레이블, 이미지 등 다양한 유형의 데이터를 조건으로 여러 유형의 출력을 생성하는 다목적 모델을 개발하여 AGI(인공 일반 지능)에 더 가까이 다가가는 것.

## 📌 TL;DR
- **문제:** 컴퓨터 비전 분야에서 빠르게 발전하는 확산 모델에 대한 포괄적인 조사를 제공합니다.
- **방법:** DDPM, NCSN, SDE 세 가지 핵심 확산 모델링 프레임워크를 소개하고, 다른 생성 모델과의 관계를 분석하며, 확산 모델의 다양한 응용 분야를 분류합니다.
- **주요 발견:** 확산 모델이 인상적인 생성 품질과 다양성, 광범위한 적용 가능성을 보여주지만, 샘플링 시 높은 계산 비용이라는 한계를 가지고 있음을 지적합니다. 이 한계를 극복하고 새로운 응용 분야를 탐색하기 위한 미래 연구 방향을 제시합니다.