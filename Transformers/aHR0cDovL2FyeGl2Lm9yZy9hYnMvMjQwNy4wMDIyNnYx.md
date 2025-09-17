# Transformer-based Image and Video Inpainting: Current Challenges and Future Directions

Omar Elharrouss, Rafat Damseh, Abdelkader Nasreddine Belkacem, Elarbi Badidi, Abderrahmane Lakas

## 🧩 Problem to Solve

이미지 인페인팅(Inpainting)은 손상되거나 누락된 이미지/비디오 영역을 복원하거나 채우는 컴퓨터 비전의 핵심 과제입니다. 딥러닝(특히 CNN과 GAN)의 발전으로 상당한 개선이 있었으나, 최근 NLP에서 강력한 성능을 보인 트랜스포머(Transformer) 아키텍처가 컴퓨터 비전 분야, 특히 이미지/비디오 인페인팅에 도입되면서 장거리 종속성(long-range dependencies) 캡처 능력으로 전역적인 맥락 이해에 탁월한 성능을 보이고 있습니다. 이 논문은 트랜스포머 기반 이미지/비디오 인페인팅 기법의 현재 상태를 포괄적으로 검토하고, 주요 개선 사항을 조명하며, 이 분야의 새로운 연구자들에게 지침을 제공하는 것을 목표로 합니다.

## ✨ Key Contributions

- **트랜스포머 기반 인페인팅 기법의 종합적인 검토:** 이미지 및 비디오 인페인팅을 위한 트랜스포머 기반 기술에 초점을 맞춰 현재 접근 방식을 포괄적으로 분석합니다.
- **유형별 분류:** 아키텍처 구성(블라인드, 마스크 필요, GAN 기반), 손상 유형(마스크), 성능 지표에 따라 트랜스포머 기반 기법을 체계적으로 분류합니다.
- **주요 개선 사항 강조:** 트랜스포머가 인페인팅 작업에 가져온 중요한 개선점들을 강조합니다.
- **현존하는 도전 과제 분석:** 이미지/비디오 인페인팅 분야의 현재 도전 과제들을 체계적으로 종합하여 제시합니다.
- **미래 연구 방향 제시:** 트랜스포머 기반 인페인팅 분야의 미래 연구 방향을 제안하여 연구자들에게 실질적인 지침을 제공합니다.

## 📎 Related Works

이 논문은 기존 이미지/비디오 인페인팅 관련 리뷰 및 설문 조사들을 언급합니다. 이러한 기존 연구들은 주로 다음과 같은 기준에 따라 분류됩니다:

- **데이터 유형 기반:** 긁힌 사진, 깊이(Depth) 이미지, RGB 이미지, 법의학 이미징 등.
- **기술 기반:** 패치 기반, 예시 기반과 같은 전통적인 방법, 또는 CNN 및 GAN 기반의 딥러닝 접근 방식.

하지만 이 논문은 기존 연구들이 특정 데이터 또는 기술에 집중하는 반면, **트랜스포머 기반 인페인팅 기술에만 초점을 맞춰** 기존 설문 조사들과 차별성을 둡니다.

## 🛠️ Methodology

이 검토 논문은 트랜스포머 기반 이미지/비디오 인페인팅 알고리즘을 다음 분류에 따라 분석합니다.

1. **마스크 유형 분류:**

   - **블록(Blocks):** 사각형 형태의 손상 영역.
   - **객체(Object):** 이미지 내 특정 객체 영역.
   - **노이즈(Noise):** 밝기나 색상의 무작위 변동.
   - **스크리블(Scribble):** 손으로 그린 불규칙한 선이나 낙서 형태의 마스크.
   - **텍스트(Text):** 워터마크, 캡션 등 이미지에 오버레이된 텍스트.
   - **스크래치(Scratches):** 물리적 손상으로 인한 얇고 긴 선.

2. **트랜스포머 네트워크 아키텍처 분류:**

   - **블라인드 이미지 인페인팅 네트워크 (Blind Image Inpainting Networks):** 손상된 이미지 자체만을 입력으로 사용하여 누락된 영역을 채웁니다. (예: CTN, ICT, MAT, BAT-Fill, T-former, PUT, TransCNN-HAE, InstaFormer, Campana et al., U2AFN, CBNet, CoordFill, CMT, TransInpaint, NDMA, Blind-Omni-Wav-Net)
   - **마스크 필수 이미지 인페인팅 네트워크 (Mask-Required Image Inpainting Networks):** 손상된 이미지와 함께 마스크 정보를 추가 입력으로 사용합니다. (예: ZITS, APT, SPN, SWMH, ZITS++, TransRef, UFFC)
   - **GAN과 트랜스포머 결합 인페인팅 (GAN With Transformer Image Inpainting):** GAN의 생성자와 판별자 아키텍처에 트랜스포머를 통합하여 사실적인 이미지 생성을 목표로 합니다. (예: ACCP-GAN, AOT-GAN, HiMFR, Wang et al., Li et al., Swin-GAN, SFI-Swin, IIN-GCMAM, WAT-GAN, UFFC, PATMAT, GCAM)

3. **비디오 인페인팅 방법론:**

   - 트랜스포머 모델을 활용하여 비디오 시퀀스의 누락되거나 손상된 부분을 채웁니다. (예: FuseFormer, FAST, DSTT, E2FGVI, FGT, DeViT, DLFormer, DMT, FGT++, Liao et al., FITer, ProPainter, SViT, FSTT)

4. **손실 함수(Loss Functions) 분석:**

   - **맥락 기반(Contextual-based):** L1 손실, 재구성(Reconstruction) 손실.
   - **스타일 기반(Style-based):** 지각(Perceptual) 손실, 스타일(Style) 손실, 적대적(Adversarial) 손실, 특징 맵(Feature Map) 손실.
   - 기타(Hinge loss, Cross-entropy loss 등)도 사용될 수 있습니다.

5. **평가 데이터셋 및 지표:**
   - **데이터셋:** Paris Street View, CelebA-HQ, Places2, FFHQ (이미지); YouTube-VOS, DAVIS (비디오).
   - **평가 지표:**
     - **픽셀 기반(Pixel-based):**
       - PSNR (Peak Signal-to-Noise Ratio): $\text{PSNR} = 10 \log_{10} \left( \frac{\text{MAX}_{\text{I}}^2}{\text{MSE}} \right)$ (높을수록 좋음)
       - SSIM (Structural Similarity Index): (두 이미지의 구조적 유사성, 높을수록 좋음)
     - **패치 기반/지각 기반(Patch-based/Perceptual):**
       - LPIPS (Learned Perceptual Image Patch Similarity): $\text{LPIPS}(\text{I}_1, \text{I}_2) = \frac{1}{\text{N}} \sum_{i=1}^{\text{N}} \|\phi(\text{I}_1)_i - \phi(\text{I}_2)_i\|_2$ (낮을수록 좋음)
       - FID (Fréchet Inception Distance): (생성된 이미지의 품질 평가, 낮을수록 좋음)

## 📊 Results

- **이미지 인페인팅 성능:**
  - **PSV 및 Places2 데이터셋 (마스크 비율 40-50%):** Blind-Omni-Wav-Net이 PSNR과 SSIM에서 가장 높은 성능을 보였으며, 이는 고품질 이미지 재구성 능력을 시사합니다. TransInpaint는 가장 낮은 LPIPS와 FID 값을 달성하여 우수한 텍스처와 세부 정확도를 보여주었습니다. CoordFill은 고해상도 이미지 처리에서 뛰어난 성능을 보였습니다.
  - **CelebA-HQ 및 FFHQ 데이터셋 (마스크 비율 40-50%, 얼굴 이미지):** CelebA-HQ에서 대부분의 방법이 24 이상의 PSNR과 80% 이상의 SSIM을 달성했습니다. Blind-Omni-Wav-Net이 PSNR과 SSIM에서 가장 우수했으며, LPIPS에서는 Campana et al.과 CoordFill이 가장 낮은 값을 보였습니다. FHHQ에서는 ZITS++가 LPIPS와 FID에서 가장 우수한 성능을 나타냈습니다.
  - **마스크 비율에 따른 PSNR 변화:** 마스크 비율이 증가할수록 PSNR 값이 감소하는 경향을 보였습니다. Li et al.의 방법은 낮은 마스크 비율에서, APT와 SwMH는 높은 마스크 비율에서 강점을 보였습니다.
- **비디오 인페인팅 성능:**
  - **YouTube-VOS 및 DAVIS 데이터셋:** 모든 방법이 높은 PSNR (YouTube-VOS에서 33 이상)과 SSIM (YouTube-VOS에서 97% 이상)을 달성했습니다. FGT++는 YouTube-VOS에서, ProPainter와 DLFormer는 DAVIS에서 PSNR 기준 가장 우수한 성능을 보였습니다. 전반적으로 비디오 인페인팅 결과는 이미지 인페인팅보다 더 나은 품질을 보였습니다.
- **결론적으로,** 트랜스포머 기반 인페인팅 방법들은 전역적인 맥락 이해 능력 덕분에 이미지 품질, 구조적 유사성 등 여러 지표에서 상당한 개선을 이루었음을 보여주었습니다.

## 🧠 Insights & Discussion

- **도전 과제:**
  - **의미 보존(Preservation of Semantics):** 복원된 영역이 주변 맥락과 자연스럽게 어우러지며 이미지의 전체 의미를 유지해야 하는 어려움. 복잡한 텍스처나 미묘한 구조에서 공간적 일관성 유지 문제.
  - **맥락 이해(Context Understanding):** 이미지의 전역적 맥락(장면 의미, 객체 관계)을 정확히 이해하고, 누락된 가장자리를 재구성하며, 다양한 크기의 손상(작은 긁힘부터 큰 객체까지)을 처리하는 데 어려움.
  - **아키텍처 복잡성(Complexity of Architecture):** 트랜스포머 기반 모델은 복잡하여 훈련, 해석, 최적화가 어렵고, 계산 자원 및 모델 파라미터 수에 따라 성능 균형 맞추기가 어렵습니다.
  - **과적합(Overfitting):** 깊은 특징 추출 아키텍처는 훈련 데이터를 암기하는 과적합에 취약하며, 이를 완화하기 위한 최적의 파라미터 조합을 찾는 것이 어렵습니다.
  - **데이터 품질 요구사항(Data Quality Requirements):** 대규모 주석 데이터셋이 필요하며, 고해상도 이미지 훈련에는 막대한 계산 자원이 요구됩니다. 이미지를 작은 패치로 나누는 방식은 품질에 영향을 미칠 수 있습니다.
  - **계산 자원(Computational Resources):** 강력한 GPU/TPU가 필요하며, 실시간/대규모 애플리케이션의 구현이 어렵습니다.
  - **도메인 적응(Domain Adaptation):** 특정 데이터셋이나 작업에 훈련된 CNN은 도메인 변화나 편향으로 인해 다른 데이터셋이나 실제 환경에 적합하지 않을 수 있습니다.
- **미래 연구 방향:**
  - **장거리 종속성 처리 강화:** 인페인팅 정확도를 높이기 위해 장거리 종속성 처리 능력을 향상시키는 연구.
  - **다양한 데이터셋 탐구:** 표준 이미지 및 비디오 형식 외의 다양한 데이터셋에서 트랜스포머 기반 접근 방식의 성능을 연구하여 새로운 도전 과제와 기회 발굴.
  - **사실감 및 일관성 개선:** 복잡한 텍스처나 구조를 포함하는 시나리오에서 복원된 영역의 사실감과 일관성 정교화.
  - **시간적 일관성 및 손상 유형 강건성:** 비디오 인페인팅에서 프레임 간의 시간적 일관성 확보와 다양한 손상 유형(가려짐, 손상, 누락된 데이터)에 대한 강건성 보장.
  - **효율성 및 확장성 최적화:** 대규모 데이터셋 또는 실시간 애플리케이션을 위한 트랜스포머 기반 아키텍처의 효율성과 확장성 최적화.

## 📌 TL;DR

**문제:** 이미지/비디오 인페인팅은 손상된 영역을 채우는 작업으로, 특히 복잡한 텍스처와 장거리 맥락 이해가 중요합니다. 트랜스포머는 이러한 문제를 해결할 잠재력을 가지고 있습니다.
**방법:** 이 논문은 블라인드, 마스크 필수, GAN 기반 등 트랜스포머 기반 인페인팅 모델을 아키텍처별로 분류하고, 사용되는 마스크 유형, 손실 함수, 데이터셋 및 평가 지표를 체계적으로 검토합니다.
**발견:** 트랜스포머 기반 방법들은 전역적인 맥락 이해를 통해 이미지/비디오 인페인팅 품질을 크게 향상시켰으며, Blind-Omni-Wav-Net, FGT++와 같은 모델들이 우수한 성능을 보였습니다. 하지만 의미 보존, 계산 비용, 일반화 능력 등 여전히 많은 도전 과제가 남아있으며, 미래 연구는 이러한 문제들을 해결하고 더 강건하고 효율적인 실시간 인페인팅 솔루션 개발에 집중해야 합니다.
