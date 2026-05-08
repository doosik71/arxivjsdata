# Test-time generative augmentation for medical image segmentation

Xiao Ma, Yuhui Tao, Zetian Zhang, Yuhan Zhang, Xi Wang, Sheng Zhang, Zexuan Ji, Yizhe Zhang, Qiang Chen, and Guang Yang (2025/2026)

## 🧩 Problem to Solve

의료 영상 분할(Medical Image Segmentation)은 임상 진단과 치료 계획 수립에 필수적이지만, 실제 환경에서는 조직의 폐쇄(occlusions), 모호한 경계(ambiguous boundaries), 영상 장비 및 프로토콜의 다양성으로 인해 모델의 예측 불확실성이 높다는 문제가 있다.

기존의 테스트 단계 보강(Test-Time Augmentation, TTA) 기법들은 주로 정해진 기하학적 변환(회전, 뒤집기 등)이나 광도 변환에 의존한다. 이러한 방식은 의료 영상의 복잡하고 이질적인 특성에 유연하게 대응하지 못하며, 단순한 무작위성으로 인해 실제 임상적으로 유의미한 변형을 생성하는 데 한계가 있다. 또한, 모델 파라미터를 업데이트하는 Test-Time Model Adaptation(TTMA)이나 Dropout 기반의 TTD 방식은 각각 불확실성 측정의 어려움이나 결과의 불안정성이라는 문제를 안고 있다.

본 논문의 목표는 도메인 특화 생성 모델을 활용하여 테스트 이미지의 콘텐츠와 시맨틱(semantics)에 최적화된 다양하고 현실적인 증강 이미지를 생성함으로써, 분할 정확도를 높이고 픽셀 단위의 오류 추정(pixel-wise error estimation) 성능을 향상시키는 **TTGA(Test-Time Generative Augmentation)** 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **도메인 적응형 생성 모델을 이용해 테스트 시점에 각 이미지의 특성에 맞는 맞춤형 증강 데이터를 생성**하는 것이다.

1. **TTGA 프레임워크 제안**: 의료 데이터로 파인튜닝된 확산 모델(Diffusion Model)을 사용하여, 테스트 샘플의 전역적 레이아웃은 유지하면서 국소적 구조의 분포를 풍부하게 만드는 증강 전략을 도입하였다.
2. **Masked Null-text Inversion 기술**: 이미지의 특정 영역은 원본 정체성(identity)을 유지하고, 특정 영역은 제어된 변동성을 부여하기 위해 마스크 기반의 널 텍스트 인버전(null-text inversion) 기법을 개발하였다.
3. **Dual Denoising Pathway 설계**: 정체성 보존 경로(Identity-preserving path)와 증강 강화 경로(Augmentation-enhancing path)라는 두 가지 경로를 설계하여, 임상적 유효성을 해치지 않으면서도 모델의 강건성을 높이는 최적의 변형을 생성한다.

## 📎 Related Works

### 기존 접근 방식 및 한계

- **전통적 TTA**: 기하학적/광도 변환을 통해 여러 뷰를 생성하고 앙상블한다. 하지만 수작업으로 설계된 변환 세트에 의존하므로 증강 공간이 제한적이며, 의료 영상의 복잡한 콘텐츠를 의미 있게 변경하지 못한다.
- **TTMA & TTD**: 전자는 단일 최적 파라미터로 수렴하여 불확실성 측정에 불리하며, 후자는 제어되지 않은 무작위성으로 인해 정확도가 저하되거나 출력이 불안정할 수 있다.
- **확산 모델 기반 생성**: 고품질 이미지 생성이 가능하지만, 의료 분야에서는 텍스트-이미지 쌍 데이터가 부족하여 텍스트 기반 제어가 어렵다는 한계가 있다.

### 차별점

TTGA는 단순한 변환이나 무작위 섭동 대신, **LoRA(Low-Rank Adaptation)**를 통해 도메인에 적응된 생성 모델을 사용한다. 특히 텍스트 프롬프트에 전적으로 의존하는 대신, 이미지 자체에서 최적화된 널 텍스트 임베딩을 추출하여 정체성을 보존함으로써 의료 영상의 특수성을 반영한 정교한 증강을 수행한다.

## 🛠️ Methodology

### 전체 파이프라인

TTGA는 테스트 이미지 $\mathbf{x}'$가 입력되면 다음과 같은 절차를 거친다:

1. **One-step Null-text Optimization**: 원본 이미지의 특성을 캡처하는 최적의 널 텍스트 임베딩 $\emptyset^*_\tau$를 찾는다.
2. **Masked Null-text Inversion Generation**: 마스크를 통해 정체성 보존 영역과 증강 영역을 나누고, 두 가지 디노이징 경로를 통해 증강 이미지 세트를 생성한다.
3. **Ensemble Segmentation**: 생성된 $N$개의 증강 이미지와 원본 이미지를 기존 분할 모델에 입력하여 결과들을 앙상블하고, 최종 분할 맵과 불확실성 맵(Entropy 기반)을 도출한다.

### 상세 방법론 및 주요 방정식

#### 1. One-step Null-text Optimization

기존의 Null-text inversion은 모든 타임스텝에서 최적화를 수행하여 계산 비용이 매우 높다. 본 논문은 효율성을 위해 단일 중간 타임스텝 $\tau$에서만 최적화를 수행하는 방식을 제안한다.
$$\mathcal{L}_\tau = \|\mathbf{x}^*_0 - \mathbf{x}_0(\mathbf{x}_\tau, \tau, \mathbf{c}, \emptyset^*_\tau)\|^2_2$$
여기서 $\emptyset^*_\tau$만을 업데이트하여, 생성된 결과가 원본 이미지 $\mathbf{x}^*_0$와 최대한 유사해지도록 하여 정체성을 보존한다.

#### 2. Multi-condition Guidance

시맨틱 조건 $\mathbf{c}$(예: "폴립 사진")와 정체성 조건 $\mathbf{i}$를 동시에 적용하기 위해 확장된 Classifier-free guidance를 사용한다.
$$\ddot{\epsilon}_\theta(\mathbf{x}_t, t, \emptyset, \mathbf{i}, \mathbf{c}) = \epsilon_\theta(\mathbf{x}_t, t, \emptyset) + \lambda_c[\epsilon_\theta(\mathbf{x}_t, t, \mathbf{c}) - \epsilon_\theta(\mathbf{x}_t, t, \emptyset)] + \lambda_i[\tilde{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{i}, \mathbf{c}) - \epsilon_\theta(\mathbf{x}_t, t, \mathbf{c})]$$

- $\lambda_c$: 시맨틱 정렬 강도 (도메인 앵커 역할)
- $\lambda_i$: 원본 이미지의 정체성 유지 강도

#### 3. Dual Denoising Paths with Masks

이미지를 두 영역으로 나누어 서로 다른 디노이징 경로를 적용한다.
$$\bar{\mathbf{x}}_{t-\Delta t} = M^\mathcal{S}_{t-\Delta t} [\text{Identity Path}] + M^\mathcal{C}_{t-\Delta t} [\text{Augmentation Path}]$$

- **Identity-preserving path ($\mathcal{S}$)**: 최적화된 $\emptyset^*_\tau$를 사용하여 원본의 세부 사항을 엄격하게 유지한다.
- **Augmentation-enhancing path ($\mathcal{C}$)**: $\mathbf{c}$와 $\emptyset^*_\tau$를 모두 사용하여, 정체성을 유지하면서도 도메인 내에서 가능한 현실적인 변동성(텍스처, 조명 등)을 추가한다.
- **마스크 생성 전략**: 무작위(Bernoulli), 확산 모델의 어텐션 맵 기반(Attention), 그리고 이 둘을 섞은 Hybrid 방식을 사용한다.

#### 4. 의료 도메인 적용 (LoRA 및 조건 정의)

- **도메인 적응**: Stable Diffusion v1.5를 기반으로, 각 분할 작업의 학습 데이터셋을 사용하여 LoRA 파인튜닝을 수행한다.
- **시맨틱 조건 $\mathbf{c}$**: 단순한 클래스 레벨 텍스트 프롬프트(예: "a photo of a polyp")의 CLIP 임베딩을 사용하여 생성 방향을 의료 도메인으로 고정하는 'Domain Anchor' 역할을 하게 한다.
- **정체성 조건 $\mathbf{i}$**: 최적화된 $\emptyset^*_\tau$ 자체가 해당 이미지의 환자 고유한 해부학적 특성을 담은 컨테이너가 된다.

## 📊 Results

### 실험 설정

- **데이터셋 및 작업**:
  - Fundus: Optic Disc 및 Cup 분할 (REFUGE 데이터셋)
  - Polyp: 폴립 분할 (Kvasir, CVC-ClinicDB 등 5개 데이터셋)
  - Skin: 피부 병변 분할 (ISIC 2017, 2018, PH2 데이터셋)
- **기준 모델**: nnU-Net-v2 (범용) 및 각 작업별 SOTA 모델 (SegTran, HSNet, H-vmunet).
- **비교 대상**: Baseline(unaugmented), TTD, TTA, Deep Ensemble, DDIM Inversion, Null-text Inversion.
- **지표**:
  - 분할 정확도: DSC, AUC, HD95.
  - 오류 추정: DSC, AUC, NSD (Normalized Surface Distance).

### 주요 결과

1. **분할 정확도 향상**: TTGA는 모든 작업에서 baseline 대비 DSC를 $0.1\%$에서 $2.3\%$까지 향상시켰으며, 특히 HD95(경계 거리)에서 유의미한 감소(최대 $5.2$)를 보였다.
2. **픽셀 단위 오류 추정 성능**: 가장 극적인 향상을 보인 부분으로, baseline 대비 DSC가 $1.1\%$에서 $29.0\%$까지 크게 상승하였다. 이는 TTGA가 생성하는 증강 이미지가 모델이 어디에서 실수하는지를 정확하게 드러내는 유의미한 섭동(perturbation)을 제공함을 의미한다.
3. **형태학적 강건성**: 타겟의 크기(Normalized Size)와 원형도(Circularity)에 관계없이 일관된 성능 향상을 보였으며, 특히 복잡한 형태(low circularity)를 가진 피부 병변 데이터셋에서도 효과적임이 입증되었다.

## 🧠 Insights & Discussion

### 강점 및 해석

TTGA의 가장 큰 강점은 **'제어된 무작위성'**에 있다. 단순한 TTA나 TTD는 모델에 너무 과한 섭동을 주어 정확도를 떨어뜨리거나, 반대로 너무 약한 섭동을 주어 불확실성을 제대로 측정하지 못한다. 반면 TTGA는 생성 모델의 시맨틱 가이던스를 통해 의료 영상 분포 내에서 '있음직한' 변형만을 생성함으로써, 정확도 향상과 불확실성 추정이라는 두 마리 토끼를 동시에 잡았다.

### 한계 및 비판적 논의

- **모델 의존성**: baseline 모델이 데이터 섭동에 매우 둔감하거나 과적합(overfitting)된 경우, TTGA 단독으로는 성능 향상이 제한적일 수 있다. 논문에서도 폴립 분할 작업에서 TTD가 일부 지표에서 더 높게 나타난 점을 언급하며, 모델 기반 섭동과 데이터 기반 섭동의 결합(mixture solution) 필요성을 시사한다.
- **계산 비용**: 확산 모델의 인버전과 샘플링 과정이 포함되므로, 전통적인 TTA에 비해 추론 시간이 증가한다. 비록 One-step 최적화로 줄였으나, 실시간 진단 시스템에 적용하기에는 여전히 부담이 될 수 있다.

## 📌 TL;DR

본 논문은 의료 영상 분할의 정확도와 불확실성 추정을 개선하기 위해 **도메인 특화 생성 모델 기반의 테스트 단계 증강 기법인 TTGA**를 제안한다. Masked Null-text Inversion과 Dual Denoising Path를 통해 원본 이미지의 정체성은 보존하면서 의료적으로 타당한 변형을 생성하며, 이를 통해 다양한 의료 영상 작업에서 SOTA 성능을 달성하였다. 이 연구는 단순한 기하학적 변환을 넘어 **생성 AI를 이용한 맞춤형 데이터 증강이 의료 영상 분석의 강건성을 높이는 핵심 도구가 될 수 있음**을 보여준다.
