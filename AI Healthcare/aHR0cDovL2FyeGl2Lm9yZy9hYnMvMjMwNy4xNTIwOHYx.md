# Generative AI for Medical Imaging: extending the MONAI Framework
Walter H. L. Pinaya, Mark S. Graham, Eric Kerfoot, Petru-Daniel Tudosiu, Jessica Dafflon, Virginia Fernandez, Pedro Sanchez, Julia Wolleb, Pedro F. da Costa, Ashay Patel, Hyungjin Chung, Can Zhao, Wei Peng, Zelong Liu, Xueyan Mei, Oeslle Lucena, Jong Chul Ye, Sotirios A. Tsaftaris, Prerna Dogra, Andrew Feng, Marc Modat, Parashkev Nachev, Sebastien Ourselin, and M. Jorge Cardoso

## 🧩 Problem to Solve
- 의학 영상 연구는 방대한 데이터셋을 필요로 하지만, 개인 정보 보호 문제로 인해 데이터 공유가 어렵습니다.
- 생성 모델은 합성 데이터 생성을 통한 프라이버시 보호, 이상 탐지, 이미지-투-이미지 변환, 노이즈 제거, 재구성 등 다양한 의학 영상 응용 분야에서 큰 잠재력을 가지고 있습니다.
- 하지만 이러한 생성 모델의 복잡성으로 인해 구현 및 재현이 어렵고, 이는 연구 진행을 저해하며 새로운 방법론의 비교를 어렵게 만듭니다.
- 생성 모델 개발 및 평가를 위한 표준화된 환경의 부족이 문제입니다.

## ✨ Key Contributions
- 자유롭게 사용 가능한 오픈소스 플랫폼인 MONAI Generative Models를 제안했습니다.
- 이를 통해 연구자 및 개발자가 의학 영상 분야에서 생성 모델 및 관련 애플리케이션을 쉽게 훈련, 평가, 배포할 수 있도록 지원합니다.
- 확산 모델, 자기회귀 트랜스포머, GAN 등 다양한 최신 아키텍처를 표준화된 방식으로 재현하고 사전 훈련된 모델을 제공합니다.
- 2D 및 3D 시나리오를 포함하여 CT, MRI, X-Ray 등 다양한 의학 영상 모달리티와 해부학적 부위에 걸쳐 일반화된 방식으로 구현 가능함을 입증했습니다.
- 장기적인 유지보수와 미래 기능 확장을 위해 모듈화되고 확장 가능한 접근 방식을 채택했습니다.

## 📎 Related Works
- **생성 모델 아키텍처:** 확산 모델(Diffusion Models) [19,40], 자기회귀 트랜스포머(Autoregressive Transformers) [36,11], 적대적 생성 신경망(GANs) [13], 변이형 오토인코더(VAEs) [28].
- **MONAI 내 특정 구현/개념:**
    - 확산 모델 구성 요소: `DDIMScheduler` [44], `PNDMScheduler` [31], 잠재 확산 모델(Latent Diffusion Model, LDM) [40,38], `AutoencoderKL`, `VQVAE`, `DiffusionModelEncoder` [54], `ControlNets` [57].
    - 트랜스포머 구성 요소: 어텐션 메커니즘(Attention mechanisms) [51], `Ordering` 클래스.
    - GAN 구성 요소: SPADE [35], `PatchDiscriminator` [22], `MultiScalePatchDiscriminator` [52].
- **평가 지표:** Fréchet Inception Distance (FID) [18], Maximum Mean Discrepancy (MMD) [16], Multi-Scale Structural Similarity Index Measure (MS-SSIM) [53].
- **손실 함수:** 스펙트럼 손실(Spectral losses) [9], 패치 기반 적대적 손실(patch-based adversarial loss) [22], 지각 손실(perceptual losses) [58,24]. 사전 훈련된 네트워크 (RadImageNet [33], MedicalNet [4]).
- **응용 분야:** 이상 탐지 [42,14,55,15,39], 이미지-투-이미지 변환 [55,12,57], 이미지 개선/노이즈 제거 [56], 이미지 재구성 [6,7], 초해상도(super-resolution) [20].
- **데이터셋:** ImageNet [8], LAION [43], Wikipedia Text Corpus, CommonCrawl, MIMIC-CXR [23], CSAW-M [45], UK Biobank [46], Retinal OCT [27], Medical Decathlon 데이터셋 [1].
- **기타:** Classifier Free Guidance [21], BiomedCLIP 모델 [59].

## 🛠️ Methodology
- **프레임워크 개발:** 기존 MONAI 프레임워크를 확장하여 MONAI Generative Models를 구축했습니다.
- **모델 구현:**
    - **확산 모델:** 타임스텝(`t`) 및 공간 트랜스포머 컨디셔닝을 위한 `DiffusionModelUNet`과 `DDIMScheduler` [44], `PNDMScheduler` [31] 같은 스케줄러를 포함하는 `Scheduler` 클래스를 구현했습니다. `linear`, `scaled linear`, `cosine` 노이즈 프로파일을 지원합니다.
    - **잠재 확산 모델 (LDM):** 잠재 표현 학습을 위한 `AutoencoderKL`과 `VQVAE` 클래스를 구현했습니다. 이상 탐지를 위한 `DiffusionModelEncoder`도 포함했습니다.
    - **ControlNets:** 사전 훈련된 확산 모델의 어댑터 역할을 하는 ControlNets를 이미지 변환을 위해 구현했습니다.
    - **자기회귀 트랜스포머:** 2D/3D 이미지를 1D 시퀀스로 변환하기 위한 `Ordering` 클래스(래스터 스캔, S-커브, 무작위 순서)를 제공했습니다.
    - **GAN:** SPADE [35] 구성 요소를 통합하여 공간 적응형 정규화를 구현했습니다. 적대적 훈련을 위해 `PatchDiscriminator` [22] 및 `MultiScalePatchDiscriminator` [52]를 포함했습니다.
- **지표 및 손실:**
    - **지표:** FID [18], MMD [16], MS-SSIM [53]을 MONAI 컨벤션에 따라 통합했습니다.
    - **손실:** 스펙트럼 손실 [9], 패치 기반 적대적 손실 [22], 지각 손실 [58,24]을 추가했습니다. 지각 손실은 ImageNet, 2D용 RadImageNet [33], 3D용 MedicalNet [4]과 같은 사전 훈련된 네트워크를 사용합니다. 메모리 관리를 위해 3D 지각 손실에 2.5D 접근 방식을 구현했습니다.
- **실험적 검증:** 플랫폼의 기능을 시연하기 위해 다양한 의학 영상 작업, 모달리티, 차원(2D/3D)에 걸쳐 다섯 가지 실험을 수행했습니다. 모든 코드는 공개되었습니다.

## 📊 Results
- **실험 I (다양한 의료 영상 유형에 대한 적응성):**
    - 잠재 확산 모델(LDM)은 MIMIC-CXR, CSAW-M, 망막 OCT, UK Biobank 2D/3D 뇌 MRI 등 다양한 데이터셋에서 고품질 이미지를 성공적으로 생성했습니다.
    - 예를 들어, UK Biobank 3D에서 FID는 $0.0051$, MS-SSIM은 다양성 $0.9217$, 재구성 $0.9820$으로 낮은 FID와 높은 MS-SSIM을 달성하여 높은 충실도와 다양성을 입증했습니다.
    - MIMIC-CXR 데이터셋에 대한 텍스트 조건부 생성은 CLIP 점수가 안내 값에 따라 예상대로 증가하며 좋은 이미지-텍스트 정렬을 보여주었습니다.
- **실험 II (잠재 모델의 모듈성):**
    - MIMIC-CXR 데이터셋에서 동일한 VQ-VAE 압축 모델을 사용하여 VQ-VAE + 트랜스포머 (FID = $9.1995$)와 VQ-VAE + 확산 모델 (FID = $8.0457$) 간의 상호 교환 가능성을 입증하여 모듈식 설계를 확인했습니다. VQ-VAE 재구성 MS-SSIM은 $0.9689$였습니다.
- **실험 III (이상 탐지 적용):**
    - `VQ-VAE + Transformer` 기술은 BRATs 데이터로 훈련했을 때 3D Medical Decathlon 데이터셋의 모든 클래스에서 OOD(Out-of-Distribution) 탐지에 대해 우수한 AUC 점수 ($1.0$)를 달성했습니다.
- **실험 IV (이미지 변환 적용):**
    - ControlNets는 2D FLAIR MRI 슬라이스를 T1-가중 이미지로 높은 충실도(PSNR = $26.2458 \pm 1.0092$, MAE = $0.02632 \pm 0.0036$, MS-SSIM = $0.9526 \pm 0.0111$)로 성공적으로 변환했습니다.
- **실험 V (이미지 초해상도 적용):**
    - UK Biobank 데이터에 3D 초해상도를 위한 캐스케이드(cascaded) LDM 접근 방식을 적용했습니다.
    - 저해상도 모델은 FID = $0.0009$를 달성했습니다. 업스케일러 LDM은 고해상도 이미지에 대해 FID = $0.0024$, MS-SSIM = $0.9141$을 보여주며, 단일 단계 생성보다 향상된 충실도를 나타냈습니다.
    - 저해상도 테스트 이미지를 초해상도화하는 데 PSNR = $29.8042 \pm 0.4173$, MAE = $0.0181 \pm 0.0009$, MS-SSIM = $0.9806 \pm 0.0017$을 달성하여 고품질 업스케일링 능력을 확인했습니다.
    - 캐스케이드 방식은 더 나은 품질을 제공하지만 이미지 생성 시간이 훨씬 더 길어지는 단점이 있습니다(예: 3D 이미지 생성에 13분 대 22초).

## 🧠 Insights & Discussion
- **의의:** MONAI Generative Models는 의학 영상 분야에서 생성 모델을 개발하고 배포하는 데 필요한 시간과 노력을 크게 줄여줍니다. 이 플랫폼은 표준화를 촉진하고 이미지 합성을 넘어 다양한 애플리케이션을 가능하게 합니다.
- **모듈성 및 적응성:** 프레임워크의 모듈성은 구성 요소(예: 압축 모델, 생성 모델)의 손쉬운 교환을 허용하며, 그 적응성은 모델의 원래 범위를 다양한 시나리오(2D/3D, 모달리티)로 확장합니다.
- **트레이드오프:** 캐스케이드 방식(예: 초해상도)은 더 높은 품질을 달성할 수 있지만, 추론 시간이 상당히 길어지는 단점이 있어 실제 배포 시 고려해야 할 사항입니다.
- **향후 연구:** 더 최신 모델을 통합하여 모델 간 비교를 용이하게 하고, MRI 재구성과 같은 다른 애플리케이션에 대한 지원을 강화하는 데 중점을 둘 것입니다.

## 📌 TL;DR
- **문제:** 의학 영상 분야의 기존 생성 모델은 구현, 재현이 복잡하고 개인 정보 보호 문제로 인해 데이터 공유가 어려워 연구 발전을 저해합니다.
- **방법:** MONAI를 확장한 오픈소스 프레임워크인 MONAI Generative Models를 제시하며, 2D/3D 의료 모달리티에 걸쳐 다양한 생성 아키텍처(확산 모델, 트랜스포머, GAN)의 훈련, 평가, 배포를 위한 표준화된 도구를 제공합니다.
- **주요 발견:** 이 플랫폼이 고품질 합성 의료 이미지를 생성하고, 모듈형 모델 조합을 가능하게 하며, 이상 탐지, 이미지-투-이미지 변환, 초해상도와 같은 다운스트림 작업을 효과적으로 지원하는 데 있어 다재다능함과 성능을 입증했습니다.