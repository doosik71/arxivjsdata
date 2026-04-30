# Generative Adversarial Networks (GAN) Powered Fast Magnetic Resonance Imaging — Mini Review, Comparison and Perspectives

Guang Yang, Jun Lv, Yutong Chen, Jiahao Huang, and Jin Zhu (2021)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 자기공명영상(Magnetic Resonance Imaging, MRI)의 치명적인 단점인 **느린 데이터 획득 속도(Slow Image Acquisition Rate)**이다. MRI는 방사선 노출이 없고 연부 조직 대조도가 뛰어나다는 장점이 있지만, k-space에서 데이터를 수집하는 과정(특히 phase-encoding 단계)으로 인해 촬영 시간이 매우 길다. 이는 응급 상황에서의 사용을 제한하며, 환자에게 폐쇄공포증이나 불안감을 유발하고, 호흡 조절이 어려운 환자(소아, 비만 환자 등)에게 큰 제약이 된다.

기존의 압축 센싱(Compressed Sensing, CS-MRI) 기반 재구성 방식은 언더샘플링(Undersampling)을 통해 속도를 높이려 했으나, 반복적인 최적화 과정으로 인해 계산 시간이 오래 걸리고, 결과물에서 인위적인 매끄러움(Smoothing)이나 블록 형태의 아티팩트(Blocky artefacts)가 발생하는 문제가 있었다. 최근 등장한 딥러닝(DNN) 기반 방식 역시 $L_1$ 또는 $L_2$ 거리 기반의 손실 함수를 사용할 경우, 국소적인 해부학적 선명도(Anatomical sharpness)를 반영하지 못해 결과 영상이 흐릿해지는(Blurry) 경향이 있다. 따라서 본 논문의 목표는 GAN(Generative Adversarial Networks)을 활용하여 MRI의 획득 속도를 높이면서도, 시각적으로 매우 정교하고 고품질인 영상을 재구성하는 방법론들을 검토하고 비교 분석하는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 기여는 GAN을 이용한 고속 MRI 재구성 방법론들에 대한 포괄적인 리뷰와 벤치마킹을 수행한 것이다. 핵심 아이디어는 **생성자(Generator)**가 언더샘플링된 데이터로부터 원본 영상을 복원하도록 학습하고, **판별자(Discriminator)**가 복원된 영상과 실제 완전 샘플링된 영상(Ground Truth)을 구분하도록 경쟁시키는 적대적 학습 구조를 통해, 단순한 픽셀 값의 유사성을 넘어 실제 사진과 같은(Photo-realistic) 고해상도 세부 정보를 복원하는 것이다. 특히 DAGAN, KIGAN, ReconGAN/RefineGAN 등 주요 모델들의 구조와 손실 함수를 분석하고, 다양한 해부학적 데이터셋과 샘플링 마스크 환경에서 이들의 성능을 정량적·정성적으로 비교하여 최적의 모델을 제시하였다.

## 📎 Related Works

논문에서는 기존의 MRI 가속화 접근 방식을 다음과 같이 분류하고 한계를 지적한다.

1.  **전통적 압축 센싱(CS-MRI):** 이미지의 희소성(Sparsity)을 가정하고 Total Variation(TV)이나 Dictionary Learning(DL) 같은 정규화 함수를 사용하여 재구성한다. 하지만 반복적 최적화로 인한 시간 소요와 영상의 과도한 평활화(Smoothing) 문제가 존재한다.
2.  **CNN 기반 재구성:** 
    *   **End-to-End 방식:** 언더샘플링된 입력을 재구성된 출력으로 직접 매핑한다(예: U-Net).
    *   **Unrolled Optimization 방식:** 전통적인 CS-MRI의 반복 최적화 과정을 신경망 구조로 풀어내어, CNN이 최적의 정규화 방법을 학습하게 한다(예: DC-CNN, Variational Network). 이 방식은 데이터 충실도(Data fidelity)는 높으나 재구성 시간이 상대적으로 길다.
3.  **GAN 기반 재구성:** 기존 CNN 방식들이 겪는 흐릿한 결과물 문제를 해결하기 위해 적대적 손실을 도입하여 지각적 품질(Perceptual quality)을 극대화한다.

## 🛠️ Methodology

### 1. MRI 재구성의 기본 원리
언더샘플링된 MRI 재구성은 다음의 수식으로 모델링된다.
$$y_u = Ax_t$$
여기서 $y_u$는 언더샘플링된 k-space 신호, $x_t$는 원래의 완전 샘플링된 영상, $A$는 연산자로 $A = \Psi F$로 정의된다. ($\Psi$는 언더샘플링 마스크, $F$는 푸리에 변환 연산자)
전통적인 CS 모델은 다음의 최적화 문제를 푼다.
$$\arg \min_{\hat{x}_u} \lambda \frac{1}{2} \|y_u - A\hat{x}_u\|_2^2 + R(\hat{x}_u)$$
첫 번째 항은 k-space의 데이터 충실도를 보장하며, $R(\hat{x}_u)$는 이미지의 매끄러움이나 희소성을 보장하는 정규화 함수이다.

### 2. 분석 대상 GAN 모델 아키텍처

#### (1) DAGAN (Deep De-Aliasing GAN)
- **구조:** 생성자로 수정된 U-Net을 사용하며, 판별자는 11층의 CNN으로 구성된다. 
- **특징:** 생성자의 출력을 $\hat{x}_u = G(x_u) + x_u$로 설정하여, 모델이 전체 영상을 새로 만드는 것이 아니라 누락된 정보만 보완하는 Refinement 함수로 동작하게 하여 학습 안정성을 높였다.
- **손실 함수:** 
  $$L_{TOTAL} = \alpha L_{iMSE} + \beta L_{fMSE} + \gamma L_{VGG} + L_{adv}$$
  이미지 도메인 MSE($L_{iMSE}$), 주파수 도메인 MSE($L_{fMSE}$), VGG Perceptual Loss($L_{VGG}$), 그리고 적대적 손실($L_{adv}$)을 결합하여 데이터 충실도와 지각적 품질을 동시에 잡았다.

#### (2) KIGAN
- **구조:** k-space 생성자($G_K$)와 이미지 공간 생성자($G_{IM}$)가 직렬로 연결된 구조이다.
- **특징:** 인접한 k-space 슬라이스들을 입력으로 받아 k-space에서 1차 복원 후, 다시 이미지 공간에서 정밀하게 복원하는 단계적 접근 방식을 취한다.
- **손실 함수:** 이미지 MSE, 주파수 MSE, 적대적 손실의 합으로 구성된다.

#### (3) ReconGAN / RefineGAN
- **구조:** 두 개의 U-shaped 생성자($G_1, G_2$)가 체인 형태로 연결된 구조이다. $G_1$의 결과물은 ReconGAN, $G_2$까지 거친 최종 결과물은 RefineGAN으로 정의한다.
- **특징:** $\bar{x}_u = G_1(x_u) + x_u$ 및 $\hat{x}_u = G_2(\bar{x}_u) + \bar{x}_u$ 형태의 잔차 학습(Residual Learning)을 적용하여 수렴 속도를 높이고 세부 묘사를 강화했다.
- **손실 함수:** 이미지 MSE, 주파수 MSE, 적대적 손실을 사용한다.

## 📊 Results

### 1. 실험 설정
- **데이터셋 및 작업:** 뇌(Brain) 및 무릎(Knee) MRI 데이터셋을 사용하여 재구성 성능을 측정하였다.
- **샘플링 마스크:** Cartesian, Radial, Spiral 세 가지 마스크를 사용하였다.
- **가속도/샘플링률:** Cartesian의 경우 $2\times, 4\times, 6\times$ 가속을, Radial/Spiral의 경우 $50\%, 30\%, 20\%$ 샘플링률을 적용하였다.
- **측정 지표:** 
    - **Fidelity (충실도):** PSNR(Peak Signal-to-Noise Ratio), SSIM(Structural Similarity Index)
    - **Perceptual (지각적 품질):** MOS(Mean Opinion Score), FID(Frechet Inception Distance)

### 2. 주요 결과
- **정성적 결과:** Zero-Filled(ZF) 영상은 강한 아티팩트가 존재하며, DAGAN과 KIGAN은 아티팩트를 제거하지만 여전히 일부 잔여 아티팩트나 흐릿함(Blurring)이 관찰되었다. 반면, **RefineGAN**은 Ground Truth에 가장 근접한 세밀한 혈관 구조와 뇌 조직의 경계를 복원하였다.
- **정량적 결과:** Table 3와 4에서 확인된 바와 같이, 거의 모든 마스크와 가속도 조건에서 **RefineGAN이 가장 높은 PSNR과 SSIM**을 기록하였다. 특히 가속도가 높아질수록(또는 샘플링률이 낮아질수록) 타 모델과의 성능 격차가 뚜렷해졌으며, RefineGAN은 높은 신호 대 잡음비(SNR)를 유지하며 우수한 복원력을 보였다.

## 🧠 Insights & Discussion

### 1. GAN 기반 방식의 강점과 한계
GAN 기반 방법론은 단순한 픽셀 일치보다 인간의 시각적 인지와 유사한 고품질 영상을 생성하는 데 탁월하다. 하지만 다음과 같은 한계점이 논의되었다.
- **학습 불안정성:** GAN 특유의 학습 불안정성과 느린 수렴 속도가 문제이며, 이를 해결하기 위해 Wasserstein GAN(WGAN)이나 Spectral Normalization 등의 기법이 제안되고 있다.
- **고주파 노이즈:** GAN 손실 함수가 때때로 고주파 노이즈를 생성하거나 반대로 영상을 너무 매끄럽게 만드는 경향이 있다. 이는 Least Square GAN이나 $L_1$ 보조 손실 함수로 완화 가능하다.

### 2. 재구성 불안정성 (Reconstruction Instability)
Antun et al. (2020)의 연구를 인용하여, GAN 기반 모델이 학습 데이터와 다른 분포의 입력(예: 이미지에 임의의 글자를 추가한 경우)에 대해 매우 취약하며, 언더샘플링 비율이 조금만 달라져도 성능이 급격히 저하되는 불안정성을 보인다는 점이 지적되었다. 이는 GAN이 실제 물리적 복원이 아닌 데이터의 통계적 패턴을 "학습"하여 "그려내는" 성향이 강하기 때문일 수 있다.

### 3. 일반화 성능 (Generalizability)
불안정성 논란에도 불구하고, GAN 모델은 **Zero-shot Inference**(학습하지 않은 병변 복원)에서 강점을 보인다. 예를 들어, 건강한 뇌 영상으로 학습된 DAGAN이 뇌종양 환자의 영상을 복원할 때 종양 구조를 충실히 보존했다는 점은, GAN이 실제 해부학적 변이를 어느 정도 수용할 수 있음을 시사한다.

## 📌 TL;DR

본 논문은 MRI 촬영 시간을 획기적으로 단축하기 위해 GAN을 이용한 고속 재구성 방법론들을 리뷰하고 벤치마킹하였다. 분석 결과, **RefineGAN**이 다양한 샘플링 마스크와 높은 가속도 환경에서도 가장 우수한 이미지 충실도(PSNR, SSIM)와 지각적 품질을 보였다. GAN 기반 방식은 학습 및 재구성의 불안정성이라는 잠재적 위험이 존재하지만, 병리적 구조를 보존하는 일반화 능력 또한 확인되었다. 향후 연구는 MRI 물리 법칙의 결합과 설명 가능한 AI(XAI) 모듈의 도입을 통해 임상 적용 가능성을 높이는 방향으로 나아가야 한다.