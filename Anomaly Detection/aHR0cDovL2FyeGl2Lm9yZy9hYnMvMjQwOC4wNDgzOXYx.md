# Adversarially Robust Industrial Anomaly Detection Through Diffusion Model

Yuanpu Cao, Lu Lin, Jinghui Chen

## 🧩 Problem to Solve

산업 현장에서 딥러닝 기반 이상 탐지 모델은 높은 정확도를 보이지만, 적대적 예제(adversarial examples)에 취약하여 실제 배포에 큰 위협이 됩니다. 최근 확산 모델(Diffusion Model)이 적대적 노이즈를 정화하여 분류기의 견고성을 높일 수 있음이 밝혀졌지만, 이를 이상 탐지에 단순히 적용(정화기를 이상 탐지기 앞에 배치)하면 이상 신호와 적대적 교란이 동시에 제거되어 '이상 미탐지율(anomaly miss rate)'이 높아지는 문제가 발생합니다. 이 논문은 이러한 한계를 극복하고 이상 탐지 모델의 적대적 견고성을 확보하는 것을 목표로 합니다.

## ✨ Key Contributions

- 산업 이상 탐지 영역에서 적대적 견고성 연구를 촉진하기 위해 다양한 이상 탐지기를 위한 통합 적대적 공격 프레임워크를 구축했습니다.
- 확산 모델이 이상 탐지기 및 적대적 정화기 역할을 동시에 수행하는 새로운 적대적으로 견고한 산업 이상 탐지 방법인 AdvRAD를 제안했습니다. 또한, $l_2$-norm 제한 교란에 대한 인증된 견고성(certified robustness)을 제공하도록 확장했습니다.
- 광범위한 실험을 통해 제안된 방법이 뛰어난 (인증된) 적대적 견고성을 보여주면서도 MVTec AD, ViSA, BTAD와 같은 산업 이상 탐지 벤치마크 데이터셋에서 최신 방법과 동등한 강력한 이상 탐지 성능을 유지함을 입증했습니다.

## 📎 Related Works

- **이상 탐지 방법 (Anomaly Detection Methods):**
  - **재구성 기반(Reconstruction-based):** Autoencoder를 훈련하고 입력과 재구성된 이미지 사이의 $l_p$ norm 거리를 이상 점수로 사용하거나, GAN의 생성자를 Autoencoder로 구현하여 재구성 품질을 높이는 방법 (예: OCR-GAN, Skip-GANomaly).
  - **특징 기반(Feature-based):** 사전 훈련된 ResNet, Vision Transformer 등을 사용하여 정상 이미지의 특징을 추출하고, Flow-based 모델, KNN, 가우시안 분포 모델링 등으로 특징 분포를 추정하여 이상 점수를 계산하는 방법 (예: SPADE, CFlow, FastFlow, CFA, PANDA).
- **이상 탐지기에 대한 적대적 공격 및 방어 (Adversarial Attacks and Defenses for Anomaly Detectors):**
  - 기존에는 Autoencoder 기반 모델에 초점을 맞춤. Goodge et al. [14]는 이상 데이터를 정상으로 분류하도록 재구성 오차를 줄이는 공격을 고려하고 APAE 방어를 제안. Lo et al. [23]는 Principal Latent Space를 방어 전략으로 제안.
- **확산 모델 (Diffusion Models):**
  - 높은 샘플 품질과 강력한 모드 커버리지로 최근 주목받는 생성 모델. Nie et al. [27]는 확산 모델을 사용하여 적대적 교란을 정화하는 DiffPure를 제안하여 분류 작업에서 강력한 견고성을 입증.
  - 의료 이미지 진단 분야에서 확산 모델을 활용한 연구 (Wolleb et al. [39], Wyatt et al. [40])가 있으나, 대부분 픽셀 수준의 이상 탐지/localization에 초점을 맞추고 있으며, 적대적 견고성 개선에 대한 연구는 없었음.

## 🛠️ Methodology

AdvRAD는 확산 모델이 이상 탐지기와 적대적 정화기 역할을 동시에 수행하도록 설계되었습니다.

1. **통합 적대적 공격 프레임워크 (Unified Adversarial Attacks):**

   - PGD(Projected Gradient Descent) [24] 공격을 사용하여 이상 탐지 모델의 취약점을 평가합니다. 목표는 이상 샘플의 이상 점수를 낮추고 정상 샘플의 이상 점수를 높여 오탐지를 유발하는 것입니다.
   - $l_\infty$ 또는 $l_2$ norm 제약 하에 다음 최적화 목표를 따릅니다: $\arg \max_x L_\theta(x,y) = yA_\theta(x)$ (여기서 $y$는 레이블, $A_\theta(x)$는 이상 점수).

2. **순진한 접근 방식의 실패 분석 (Failure of Naive DiffPure):**

   - DiffPure [27]를 이상 탐지 모델(예: CFA [20]) 앞에 배치할 경우, 낮은 정화 수준에서는 적대적 교란을 완전히 제거하지 못하고, 높은 정화 수준에서는 이상 신호까지 함께 제거하여 표준 AUC가 급격히 감소하는 것을 확인했습니다.

3. **확산 모델을 통한 이상 탐지 및 적대적 정화 통합 (Merging Anomaly Detection and Adversarial Purification):**
   - **강력한 재구성 (Robust Reconstruction):** 확산 모델이 훈련된 정상 데이터에 대해 재구성을 수행할 때, 정상 데이터는 원본과 거의 동일하게 재구성되지만, 이상 데이터는 확산 모델이 "이상 영역"을 복구하려 시도하여 원본과의 재구성 오류가 커집니다. 또한, 적대적 노이즈는 이상 신호와 함께 제거되므로, 적대적으로 교란된 이상 샘플도 재구성 후 높은 오류를 보여 이상으로 탐지될 수 있습니다.
   - **One-shot Denoising:** 반복적인 디노이징 과정은 시간이 많이 소요되므로, AdvRAD는 O(1) 추론 시간 효율성을 위해 one-shot 디노이징 (Algorithm 1)을 채택합니다. 이는 임의의 시간 단계 $k$에서 $x_k = \sqrt{\alpha_k}x_0 + \sqrt{1-\alpha_k}\epsilon$로 확산된 이미지 $x_k$에서 바로 $\tilde{x}_0 = \frac{1}{\sqrt{\alpha_k}}(x_k - \sqrt{1-\alpha_k}\epsilon_\theta(x_k,k))$를 통해 원본 $\tilde{x}$를 재구성하는 방식입니다.
   - **이상 점수 계산 (Anomaly Score Calculation):** 최종 이상 점수를 견고하고 안정적으로 계산하기 위해, 픽셀 단위 및 패치 단위 재구성 오류를 모두 고려하는 다중 스케일 재구성 오류 맵(Multiscale Reconstruction Error Map, Err${}_{ms}$)을 사용합니다. 각 스케일 $l \in L=\{1, 1/2, 1/4, 1/8\}$에 대해 다운샘플링된 입력 $x_l$과 재구성 $\tilde{x}_l$ 간의 오류 맵 $\text{Err}(x, \tilde{x})_l = \frac{1}{C}\sum_{c=1}^C (x_l - \tilde{x}_l)^2[c,:,:]$를 계산한 후 원본 해상도로 업샘플링합니다. 최종 Err${}_{ms}$는 각 스케일의 오류 맵을 평균하고 평균 필터를 적용하여 얻습니다. 최종 이상 점수는 Err${}_{ms}(x, \tilde{x})$와 정상 훈련 데이터의 Err${}_{ms}$ 평균 간의 픽셀별 절대 편차의 최댓값을 사용합니다.

## 📊 Results

- **SOTA 이상 탐지 모델과의 비교:** AdvRAD는 MVTec AD, ViSA, BTAD 데이터셋에서 $l_\infty$-PGD ($\epsilon=2/255$) 및 $l_2$-PGD ($\epsilon=0.2$) 공격에 대해 기존 SOTA 모델들을 크게 능가하는 견고한 AUC를 달성했습니다. 예를 들어, MVTec AD에서 평균 견고한 AUC를 81.1%로 개선하여 최소 78.8%p 향상시켰습니다. 클린 데이터에 대한 표준 AUC는 SOTA 모델들과 동등하거나 (ViSA에서는) 모든 기준선을 능가했습니다.
- **다른 확산 모델 기반 이상 탐지기와의 비교:** 의료 영상에 초점을 맞춘 AnoDDPM [40]과 비교했을 때, AdvRAD는 표준 AUC 및 견고한 AUC 모두에서 우수한 성능을 보였습니다. 특히 AnoDDPM의 one-shot 재구성 버전과 비교해도 성능이 뛰어났습니다.
- **모델 불가지론적 방어 전략과의 비교:** DiffPure [27] + SOTA 이상 탐지기 및 Adversarial Training [24] + SOTA 이상 탐지기와 비교했을 때, AdvRAD는 평균 표준 AUC와 견고한 AUC 모두에서 훨씬 우수했습니다. 이는 DiffPure가 이상 신호를 보존하면서 적대적 노이즈만 제거하는 것이 어렵고, Adversarial Training은 정상 데이터에만 적용 가능하여 이상 데이터의 견고성을 보호하지 못한다는 한계를 보여줍니다.
- **방어 기능이 있는 이상 탐지기와의 비교:** APAE [14], PLS [23], Robust Autoencoder (RAE) [43] 등 기존 방어 기능이 있는 모델들과 비교했을 때, AdvRAD는 모든 공격에서 월등히 뛰어난 성능을 보였습니다.
- **더 강력한 적대적 공격에 대한 방어:** EOT-PGD (확률적 방어 회피) 및 AutoAttack [12]과 같은 적응형 공격에 대해서도 AdvRAD는 경험적으로 강력한 견고성을 유지했습니다.
- **인증된 적대적 견고성 (Certified Adversarial Robustness):** 랜덤화된 스무딩 (Randomized Smoothing) [10]을 AdvRAD에 적용하여 $l_2$ norm 교란에 대한 인증된 견고성을 달성했습니다. 예를 들어, MVTec AD의 Grid 데이터셋에서 $l_2$ 반경 0.2에서 98.2%의 인증된 AUC를 달성했습니다.

## 🧠 Insights & Discussion

- **확산 모델의 잠재력:** 확산 모델이 데이터 정화 능력뿐만 아니라, 이상 탐지에서도 강력한 재구성 특성을 통해 적대적 견고성을 자연스럽게 확보할 수 있음을 보여줍니다. 이는 확산 모델의 다재다능한 능력을 강조합니다.
- **동시 수행의 중요성:** 이상 탐지와 적대적 정화를 분리하여 순차적으로 수행하는 대신, AdvRAD처럼 확산 모델 내부에서 두 가지 작업을 동시에 수행하는 것이 핵심입니다. 이를 통해 정화기가 이상 신호와 적대적 교란을 구분해야 하는 어려운 문제를 회피합니다.
- **효율성:** One-shot 디노이징의 도입은 확산 모델의 주요 단점 중 하나인 느린 추론 속도를 크게 개선하여 실시간 산업 애플리케이션에 대한 적용 가능성을 높였습니다.
- **인증된 견고성:** 확산 모델 기반 이상 탐지기에 랜덤화된 스무딩을 적용하여 이론적으로 보장된 견고성을 제공하는 것은 이 분야의 중요한 진전입니다. 그러나 높은 노이즈 수준에서는 이상 특징이 가우시안 노이즈에 의해 가려질 수 있다는 한계도 존재합니다.
- **산업적 의의:** AdvRAD는 실제 산업 환경에서 적대적 공격에 강인하면서도 정확한 이상 탐지 솔루션을 제공하여, 딥러닝 기반 시스템의 신뢰성과 배포 가능성을 높이는 데 기여합니다.

## 📌 TL;DR

기존 이상 탐지 모델은 적대적 공격에 취약하며, 확산 모델을 통한 단순 정화는 이상 신호까지 제거하는 문제가 있습니다. 이 논문은 확산 모델이 이상 탐지와 적대적 정화를 동시에 수행하는 `AdvRAD`를 제안합니다. `AdvRAD`는 `one-shot denoising`을 통한 `강력한 재구성`을 기반으로, `PGD`, `AutoAttack` 등 다양한 적대적 공격에 대해 `SOTA` 모델 대비 월등히 `뛰어난 견고성`을 보이며, `높은 이상 탐지 성능`과 `인증된 견고성`까지 달성하여 산업 현장의 `적대적 견고성` 문제를 효과적으로 해결했습니다.
