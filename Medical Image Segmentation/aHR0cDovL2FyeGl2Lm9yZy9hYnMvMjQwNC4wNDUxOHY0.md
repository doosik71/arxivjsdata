# MedIAnomaly: A comparative study of anomaly detection in medical images

Yu Cai, Weiwen Zhang, Hao Chen, Kwang-Ting Cheng (2025)

## 🧩 Problem to Solve

본 논문은 의료 영상 분야의 이상치 탐지(Anomaly Detection, AD) 연구에서 발생하는 불공정하고 불완전한 평가 체계 문제를 해결하고자 한다. 의료 AD는 정상 데이터만으로 학습하여 희귀 질병을 인식하거나 건강 스크리닝을 수행하는 중요한 역할을 하지만, 기존 연구들은 다음과 같은 한계를 지니고 있다.

첫째, 서로 다른 데이터셋이나 데이터 분할 방식을 사용함으로써 재현성과 비교 가능성이 떨어진다. 특히 Hyper-Kvasir나 OCT2017과 같은 일부 데이터셋은 정상과 비정상 샘플 간의 단순한 편향(bias)이나 명확한 저수준 차이가 존재하여, 단순한 Auto-Encoder(AE)만으로도 거의 완벽한 성능이 나오는 등 AD 모델의 변별력을 평가하기에 너무 쉬운 특성을 보인다.

둘째, 동일한 패러다임의 방법론이라 하더라도 통일된 구현체 없이 서로 다른 네트워크 아키텍처와 학습 기법을 사용함으로써, 방법론 자체의 효율성보다는 구현상의 디테일이 결과에 영향을 주는 불공정한 비교가 이루어지고 있다.

따라서 본 연구의 목표는 통일된 비교 벤치마크를 구축하여 다양한 의료 AD 방법론을 공정하게 평가하고, 각 구성 요소가 성능에 미치는 영향을 체계적으로 분석하여 향후 연구 방향을 제시하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 의료 영상 AD를 위한 포괄적이고 공정한 평가 프레임워크를 제안한 것이며, 세부 내용은 다음과 같다.

- **AD 방법론의 체계적 분류(Taxonomy)**: 기존 방법론을 Reconstruction-based(Image/Feature), Self-supervised learning-based(One-stage/Two-stage), Feature reference-based(Knowledge distillation/Feature modeling)의 세 가지 범주로 분류하고 문헌 조사를 수행하였다.
- **광범위한 벤치마크 데이터셋 구축**: 흉부 X-ray, 뇌 MRI, 망막 저저 사진, 피부경 영상, 조직 병리 영상 등 5가지 모달리티를 포함하는 7개의 의료 데이터셋을 큐레이션하여 평가에 활용하였다.
- **대규모 비교 실험 수행**: 총 30개의 대표적인 AD 방법론을 대상으로 이미지 수준의 이상치 분류(AnoCls)와 픽셀 수준의 이상치 분할(AnoSeg) 성능을 비교 분석하였다.
- **핵심 구성 요소의 영향 분석**: AE의 잠재 공간(Latent space) 크기, 거리 함수(Distance function), ImageNet 사전 학습 가중치(Pre-trained weights)의 영향 등을 심층 분석하여 의료 AD의 내재적 특성을 밝혀냈다.

## 📎 Related Works

논문은 기존의 의료 AD 서베이 및 벤치마크 연구들을 언급하며 본 연구와의 차별점을 명시한다. Fernando et al. (2021)은 기법 중심의 리뷰를 제공했으나 실험적 검증이 없었으며, Baur et al. (2021)은 재구성 기반 방법론에 집중했으나 단일 모달리티(뇌 MRI)와 비공개 데이터셋을 사용하여 재현성이 낮다는 한계가 있다. 또한 Lagogiannis et al. (2023)은 최신 SOTA 방법론들을 심층 분석했으나, 모든 방법론에 대해 통일된 네트워크 아키텍처를 적용하지 않아 구성 요소 자체의 고유한 특성을 파악하는 데 한계가 있었다.

본 연구는 이러한 한계를 극복하기 위해 최대한 공정한 네트워크 설정을 적용하고, 이미지-수준 분류와 픽셀-수준 분할 모두를 아우르는 다각적인 분석을 수행한다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 1. AD 방법론 분류 (Taxonomy)
본 논문은 AD를 정상 학습 집합 $D_{train} = \{x_i\}_{i=1}^N$을 통해 이상치 점수 함수 $A(\cdot; \theta)$를 학습하는 One-class classification(OCC) 문제로 정의하며, 방법론을 다음과 같이 분류한다.

- **Reconstruction-based**: 정상 이미지의 매니폴드를 학습하여 재구성하고, 입력값과 재구성값의 차이(Reconstruction error)를 점수로 사용한다. 이미지 공간에서 직접 수행하는 Image-reconstruction과 고수준 특징 맵을 재구성하는 Feature-reconstruction으로 나뉜다.
- **Self-supervised learning (SSL)-based**: 가짜 이상치(Pseudo anomalies)를 생성하여 학습한다. 가짜 이상치를 탐지하도록 직접 학습시키는 One-stage 방식과, pretext task를 통해 표현(Representation)을 학습한 후 One-class classifier를 구축하는 Two-stage 방식으로 구분된다.
- **Feature reference-based**: 사전 학습된 교사 네트워크의 특징이나 정상 데이터의 프로토타입을 참조점으로 삼아 현재 입력과의 거리 차이를 측정한다.

### 2. 벤치마크 설정
#### 데이터셋 및 태스크
- **Image-level AnoCls**: RSNA, VinDr-CXR, Brain Tumor, LAG, ISIC2018, Camelyon16 총 6개 데이터셋을 사용한다.
- **Pixel-level AnoSeg**: BraTS2021 데이터셋을 재구성하여 사용하며, 뇌종양 영역의 픽셀 단위 분할 성능을 평가한다.

#### 평가 지표
- **AnoCls**: Threshold-independent 지표인 AUC(Area Under the ROC Curve)와 AP(Average Precision)를 사용한다.
- **AnoSeg**: 픽셀 수준의 $AP_{pix}$와 최적의 동작 지점에서 계산된 $\lceil \text{Dice} \rceil$ score를 사용한다.

### 3. 통일된 구현 (Unified Implementation)
공정한 비교를 위해 다음과 같은 기본 설정을 적용한다.
- **Reconstruction methods**: Encoder가 입력을 잠재 표현 $z = f_e(x) \in \mathbb{R}^d$로 압축하고, Decoder가 $\hat{x} = f_d(z) \in \mathbb{R}^{C \times H \times W}$로 복원하는 구조를 가진다. 입력 크기는 $64 \times 64$, 잠재 공간 크기 $d=16$을 기본으로 한다.
- **SSL methods**: ResNet18을 특징 추출기($f$)로 사용하며, 입력 해상도는 $224 \times 224$로 설정한다. Two-stage 방식의 경우 Gaussian Density Estimator(GDE)를 One-class classifier($\psi$)로 사용한다.

## 📊 Results

### 1. SOTA 방법론 비교
- **이미지 수준 분류(AnoCls)**: 재구성 기반에서는 AE-PL, AE-U, FAE-SSIM이 우수한 성능을 보였다. SSL 기반에서는 ImageNet 사전 학습 가중치를 활용하거나 미세 조정(Fine-tuning)한 방법들이 가장 경쟁력이 높았다.
- **픽셀 수준 분할(AnoSeg)**: 재구성 기반 방법론이 SSL 기반보다 압도적으로 우수한 성능을 보였다. 특히 DAE(Denoising AE)가 가장 높은 성능을 기록하였는데, 이는 UNet 구조의 skip-connection과 self-supervised denoising task의 시너지가 정상 영역의 정밀한 복원과 이상 영역의 효과적인 제거를 가능케 했기 때문이다.

### 2. AE 구성 요소 분석
- **입력 크기 및 네트워크 복잡도**: 입력 크기를 $64 \times 64$에서 $128 \times 128$로 키우거나 네트워크의 깊이와 너비를 늘려도 성능 향상이 미미했다. 이는 현재의 AE 구조가 고해상도 이미지의 세부 정보를 충분히 활용하지 못하고 있음을 시사한다.
- **잠재 공간(Latent Space) 제한**:
    - 국소적 이상치(Local anomalies)가 포함된 데이터셋에서는 잠재 공간 크기 $d$를 매우 작게(4~32) 설정했을 때 성능이 크게 향상되었다. 이는 좁은 병목(Bottleneck)이 모델의 일반화 능력을 제한하여 이상치까지 복원해버리는 현상을 방지하기 때문이다.
    - 반면, 전역적 의미 이상치(Global semantic anomalies)가 있는 ISIC2018 데이터셋에서는 오히려 큰 잠재 공간이 유리한 경향을 보였다.
- **거리 함수(Distance Function)**: 단순 $\ell_2$ 또는 $\ell_1$ 손실보다 SSIM 및 Perceptual Loss(PL)가 훨씬 뛰어난 성능을 보였다. 특히 AE-PL은 대부분의 데이터셋에서 SOTA 수준의 성능을 기록했다.

### 3. 기타 분석 결과
- **DDPM 기반 방법론**: Gaussian noise보다 Simplex noise를 사용하는 AnoDDPMSimplex가 뇌 MRI 등에서 훨씬 우수한 성능을 보였으며, 이는 Simplex noise가 이상 영역을 더 효과적으로 오염(corrupt)시켜 pseudo-healthy reconstruction을 유도하기 때문이다.
- **ImageNet 가중치의 위력**: 복잡한 SSL 학습 과정 없이 단순히 ImageNet으로 사전 학습된 ResNet의 특징을 추출하여 One-class classifier에 입력하는 것만으로도 많은 최신 SSL 방법론보다 높은 성능을 기록했다.

## 🧠 Insights & Discussion

### 1. 재구성 품질과 AD 성능의 괴리
본 논문은 재구성된 이미지의 시각적 품질(Fidelity)과 실제 AD 성능 사이에 상관관계가 매우 낮다는 점을 강조한다. 예를 들어, AnoDDPM(Gaussian)은 시각적으로는 가장 완벽한 재구성을 보여주지만, 이상치까지 너무 잘 복원하기 때문에 AD 성능은 최하위였다. 반면 AE-PL은 시각적으로는 왜곡이 심하지만, 특징 공간(Feature space)에서 의미적 차이를 포착하므로 성능은 매우 높았다. 이는 의료 AD에서 중요한 것이 '얼마나 똑같이 복원하느냐'가 아니라 '정상과 비정상의 잔차(Residual) 분포를 얼마나 잘 분리하느냐'임을 보여준다.

### 2. 잠재 공간 엔트로피의 중요성
실험을 통해 데이터셋의 복잡도(정보량)에 따라 최적의 잠재 공간 크기가 달라짐을 확인하였다. 뇌 MRI와 같이 정보량이 많은 데이터는 더 큰 $d$를 필요로 한다. 저자들은 정상 데이터의 정보 엔트로피와 잠재 공간의 엔트로피를 일치시키는 self-adaptive 방식이 이론적 최적해를 찾는 방향이 될 것이라고 제안한다.

### 3. ImageNet 가중치 활용의 역설
ImageNet 가중치가 매우 강력함에도 불구하고, 이를 의료 도메인에 맞게 미세 조정(Fine-tuning)하는 기존 방법론(PANDA, MSC 등)들이 단순 가중치 사용보다 성능이 낮게 나오는 현상이 발견되었다. 이는 의료 영상의 특성에 맞는 효과적인 fine-tuning 전략이 아직 부족함을 의미하며, 향후 연구의 중요한 과제로 제시된다.

### 4. 임상적 유용성과 지표의 한계
현재 널리 쓰이는 Dice score는 과분할(Over-segmentation) 경향이 있는 모델에 유리하게 작용하는 특성이 있다. 특성 공간에서 계산되는 점수(AE-PL 등)는 성능 지표는 높지만, 픽셀 값의 절대적 차이가 모호하여 실제 임상 전문가가 해석하기 어려울 수 있다. 따라서 단순 지표를 넘어 임상적 유용성을 반영하는 보정(Calibration) 전략과 새로운 지표 개발이 필요하다.

## 📌 TL;DR

본 연구는 의료 영상 이상치 탐지(AD)의 불공정한 평가 문제를 해결하기 위해 7개 데이터셋과 30개 방법론을 포함한 **통일된 벤치마크(MedIAnomaly)**를 제안하였다. 분석 결과, 픽셀 수준 분할에서는 **DAE**가, 이미지 수준 분류에서는 **ImageNet 사전 학습 가중치**를 활용한 방법들이 가장 우수했다. 특히 **잠재 공간의 크기를 극도로 제한**하거나 **Perceptual Loss**를 사용하는 것이 성능 향상의 핵심임을 밝혀냈으며, 재구성 이미지의 시각적 품질보다는 정상/비정상 간의 특징 분리 능력이 더 중요하다는 통찰을 제공한다. 이 연구는 향후 의료 AD 모델 설계 시 불필요한 모델 복잡도 증가보다는 잠재 공간 최적화와 적절한 거리 함수 선택에 집중해야 함을 시사한다.