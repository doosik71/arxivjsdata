# FASL-Seg: Anatomy and Tool Segmentation of Surgical Scenes

Muraam Abdel-Ghani, Mahmoud Ali, Mohamed Ali, Fatmaelzahraa Ahmed, Mohamed Arsalan, Abdulaziz Al-Ali, Shidin Balakrishnan (2025)

## 🧩 Problem to Solve

본 논문은 로봇 최소 침습 수술(robotic minimally invasive surgeries)의 훈련 및 기술 평가를 위한 수술 장면의 시맨틱 세그멘테이션(semantic segmentation) 문제를 다룬다. 수술 장면을 정확하게 이해하기 위해서는 수술 도구뿐만 아니라 해부학적 구조물(anatomical objects)에 대한 정밀한 위치 파악이 필수적이다.

그러나 기존 연구들은 주로 수술 도구 세그멘테이션에 집중하고 해부학적 구조물을 간과하는 경향이 있다. 또한, 최신(SOTA) 모델들조차 고수준의 문맥적 특징(high-level contextual features)과 저수준의 엣지 특징(low-level edge features)을 동시에 포착하여 균형을 맞추는 데 어려움을 겪고 있다. 특히, 도구의 엣지나 작은 객체는 인코더의 초기 단계에서 추출되는 반면, 해부학적 구조나 큰 도구의 문맥 정보는 후기 단계에서 추출되므로, 이들 다중 스케일 특징(multiscale features)을 효율적으로 처리할 방법이 필요하다.

따라서 본 논문의 목표는 저수준 및 고수준 특징을 각각 독립적으로 처리하는 스트림을 통해, 해부학적 구조와 수술 도구 모두를 정밀하게 세그멘테이션할 수 있는 Feature-Adaptive Spatial Localization (FASL-Seg) 모델을 제안하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 인코더에서 추출되는 특징 맵의 해상도와 성격에 따라 서로 다른 처리 경로를 적용하는 **이원화된 특징 투영 스트림(Dual Feature Projection Streams)** 설계에 있다.

1. **LLFP (Low-Level Feature Projection) 스트림**: 인코더의 초기 레이어에서 추출된 저수준 특징을 처리한다. 여기에는 엣지 정보와 작은 객체의 디테일이 포함되어 있으며, Multi-Head Self Attention (MHSA)를 적용하여 노이즈를 제거하고 국소적/전역적 의존성을 포착함으로써 정밀한 위치 파악을 가능하게 한다.
2. **HLFP (High-Level Feature Projection) 스트림**: 인코더의 후기 레이어에서 추출된 고수준 특징을 처리한다. 이미 전역적인 문맥 정보가 충분히 인코딩되어 있으므로, 불필요한 어텐션 메커니즘을 배제하고 ConvChain을 통해 채널 정보를 압축하고 문맥적 특성을 보존하는 데 집중한다.
3. **단일 백본 및 얕은 디코더 구조**: SegFormer 백본 하나와 얕은 디코더(shallow decoder)를 사용함으로써 모델의 복잡도를 낮추면서도, LLFP와 HLFP를 통해 추출된 다중 스케일 특징을 통합하여 최적의 세그멘테이션 결과를 도출한다.

## 📎 Related Works

기존의 수술 장면 세그멘테이션 연구들은 주로 다음과 같은 접근 방식을 취했다.

- **MedT 및 TransUNet**: 트랜스포머 블록을 통합하여 글로벌 및 로컬 특징을 추출하려 했으나, TransUNet의 경우 바늘이나 실과 같은 매우 가는(thin) 객체를 세그멘테이션하는 데 어려움을 겪었다.
- **ViT-CNN 하이브리드 모델**: 두 경로 간의 상호 통신을 위한 어댑터 모듈을 사용하였으나, 클립 어플라이어(clip applier)나 초음파 프로브와 같은 특정 도구들에 대해 여전히 성능 저하가 관찰되었다.
- **MA-TIS**: Mask2Former 백본과 시간적 일관성 모듈을 사용하여 성능을 높였으나, 여러 백본과 디코더를 사용하여 모델 복잡도가 증가하는 단점이 있다.

본 연구는 이러한 기존 방식과 달리, 하나의 SegFormer 백본을 사용하되 인코더 출력물을 점진적으로 통합하는 대신 **별도의 처리 스트림(LLFP, HLFP)을 통해 독립적으로 처리 후 병합**함으로써 다양한 해상도의 특징 맵에서 정보를 최대한 유지하고자 한다. 또한, 많은 연구가 도구 세그멘테이션에만 치중한 반면, 본 논문은 해부학적 구조와 도구 부품 전체에 대한 클래스별 성능 지표를 상세히 보고하여 분석의 공백을 메우고자 한다.

## 🛠️ Methodology

### 전체 시스템 구조

FASL-Seg는 **SegFormer**를 인코더 백본으로 사용하며, 인코더의 4개 블록에서 나오는 출력물을 두 가지 서로 다른 스트림(LLFP, HLFP)으로 나누어 처리한 후, 이를 병합하여 얕은 디코더로 전달하는 구조이다.

### 주요 구성 요소 및 절차

#### 1. LLFP (Low-Level Feature Projection) Stream

인코더의 1, 2번 블록 출력($F_1, F_2$)을 처리한다.

- **ConvBlock**: Point-wise Convolution ($1 \times 1$ kernel) $\rightarrow$ Batch Normalization $\rightarrow$ Leaky ReLU 순으로 처리하여 공간 해상도를 유지하며 특징을 정제한다.
- **MHSA (Multi-Head Self Attention)**: 정제된 특징을 쿼리(Q), 키(K), 값(V)으로 입력하여 국소적/전역적 의존성을 학습하고 노이즈를 제거한다.
- **UpChain**: bilinear interpolation을 통해 특징 맵의 크기를 확대하여 최종 병합 크기에 맞춘다.
- 최종 출력 식: $\hat{F}_i = \text{UpChain}_N(\text{MHSA}(\text{ConvBlock}(F_i)))$

#### 2. HLFP (High-Level Feature Projection) Stream

인코더의 3, 4번 블록 출력($F_3, F_4$)을 처리한다.

- **ConvChain**: 여러 개의 ConvBlock을 연결하여 고수준의 문맥 특징을 보존하면서 채널 수를 압축한다.
- **특징**: 고수준 특징은 이미 전역 문맥을 포함하고 있어 어텐션을 적용할 경우 오히려 핵심 의미 정보가 손실될 수 있으므로 MHSA를 사용하지 않는다.
- **UpChain**: interpolation을 통해 크기를 확대한다.
- 최종 출력 식: $\hat{F}_i = \text{UpChain}_N(\text{ConvChain}_N(F_i))$

#### 3. 특징 융합 및 디코더

네 개의 스트림에서 나온 출력 $\hat{F}_1, \hat{F}_2, \hat{F}_3, \hat{F}_4$를 채널 방향으로 결합(Concat)하여 통합 특징 맵 $\hat{F}_{EM}$을 생성한다.
$$\hat{F}_{EM} = \text{Concat}(\hat{F}_1, \hat{F}_2, \hat{F}_3, \hat{F}_4)$$
이후 4개의 ConvBlock으로 구성된 얕은 디코더를 통해 채널을 압축하고, Laplacian convolution 레이어를 거쳐 최종 클래스별 세그멘테이션 맵을 생성한다.

#### 4. 학습 목표 및 손실 함수

모델은 Tversky Loss($L_{\text{tversky}}$)와 Cross Entropy Loss($L_{\text{CE}}$)의 조합을 사용하여 학습한다.
$$L_{\text{tversky}} = \frac{TP}{TP + \alpha FP + \beta FN}$$
$$L_{\text{total}} = \alpha L_{\text{tversky}} + (1-\alpha) L_{\text{CE}}$$
여기서 $\alpha=0.5$를 적용하여 두 손실 함수의 기여도를 동일하게 설정하였으며, Tversky Loss 내부의 $\alpha=0.7, \beta=0.3$ 설정을 통해 과세그멘테이션(oversegmenting)을 방지하고 오탐(False Positive)에 더 큰 패널티를 부여하였다.

## 📊 Results

### 실험 설정

- **데이터셋**: EndoVis17 및 EndoVis18 (신장 절제술 영상).
- **평가 지표**: mean Intersection over Union (mIoU) 및 Dice Similarity Coefficient (Dice).
- **비교 대상**: UNet, Mask R-CNN, DeepLabV3, SegFormer-b5, TransUNet, MedT, MATIS, ViTxCNN 등.

### 주요 결과

1. **EndoVis18 (부품 및 해부학 세그멘테이션)**:
    - FASL-Seg는 mIoU 72.71%를 달성하여 기존 SOTA 대비 약 5% 향상된 성능을 보였다.
    - 특히 MedT 대비 mIoU는 8%, Dice는 9% 더 높게 나타났다.
2. **수술 도구 세그멘테이션**:
    - **EndoVis18**: mIoU 85.61% 달성, 전체 성능 및 특정 클래스(Large Needle Driver, Monopolar Curved Scissors)에서 SOTA를 능가하였다.
    - **EndoVis17**: mIoU 72.78% 달성, 전반적인 성능에서 SOTA 모델들을 상회하였다.
3. **오탐률(FPR)**:
    - SegFormer-b5와 비교했을 때, 해부학 전용 클래스에서 1.04%, 도구 세그멘테이션에서 1% 정도 FPR이 개선되어 전반적인 마스크 생성 능력이 향상되었음을 입증하였다.

## 🧠 Insights & Discussion

### 강점 및 분석

- **특징 처리의 적응성**: ablation study를 통해 LLFP에 어텐션을 적용하는 것은 성능을 높이지만, HLFP에 적용하는 것은 오히려 성능을 떨어뜨림을 확인하였다. 이는 저수준 특징은 노이즈 제거와 정밀화가 필요하지만, 고수준 특징은 이미 추상화된 문맥 정보를 담고 있어 추가적인 어텐션이 유효한 정보를 손실시킬 수 있음을 시사한다.
- **업샘플링 방식**: Convolution Transpose보다 단순 Bilinear interpolation이 더 나은 성능을 보였는데, 이는 수술 영상의 특징을 확대할 때 단순 보간법이 중요한 세부 정보를 더 잘 유지했기 때문으로 해석된다.
- **효율성**: SegFormer-b5보다 파라미터 수가 적으면서도(81.99M) 더 높은 성능을 냈으며, GPU 메모리 점유율 또한 0.92GB로 낮아 효율적인 구조임을 보여준다.

### 한계 및 비판적 해석

- **실시간성 부족**: 추론 속도가 2.14 FPS로 매우 느리다. 이는 실시간 수술 보조 도구로 사용하기에는 부적합하며, 현재로서는 수술 후 분석(post-operative analysis) 용도로만 적합하다.
- **일반화 검증 미비**: 신장 절제술이라는 특정 수술 데이터셋에서만 검증되었으므로, 다른 종류의 수술 장면에서도 동일한 성능을 낼 수 있는지에 대한 교차 도메인(cross-domain) 테스트가 필요하다.
- **특정 클래스 저성능**: Error analysis 결과, Instrument Clasper나 Covered Kidney와 같은 일부 클래스에서는 여전히 성능이 낮게 나타났으며, 이는 색상 및 대비 증강(hue and contrast augmentation)과 같은 추가적인 데이터 증강 기법으로 해결해야 할 과제로 남아 있다.

## 📌 TL;DR

본 논문은 수술 장면에서 해부학적 구조와 도구를 정밀하게 분할하기 위해, 저수준 특징(엣지/디테일)과 고수준 특징(문맥)을 서로 다른 경로로 처리하는 **FASL-Seg** 아키텍처를 제안한다. SegFormer 백본을 기반으로 LLFP(Attention 적용)와 HLFP(ConvChain 적용) 스트림을 구축하여 다중 스케일 특징을 효과적으로 융합하였으며, 그 결과 EndoVis17/18 데이터셋에서 기존 SOTA 모델들을 뛰어넘는 mIoU 및 Dice 성능을 달성하였다. 다만, 낮은 추론 속도로 인해 실시간 적용보다는 수술 후 분석에 더 적합하며, 향후 경량화 백본 도입을 통한 실시간성 확보가 필요하다.
