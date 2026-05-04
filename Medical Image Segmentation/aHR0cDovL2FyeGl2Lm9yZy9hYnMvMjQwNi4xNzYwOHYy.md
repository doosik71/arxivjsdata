# Test-time generative augmentation for medical image segmentation

Xiao Ma et al. (2025)

## 🧩 Problem to Solve

의료 영상 분할(Medical Image Segmentation)은 임상 진단과 치료 계획 수립에 필수적이지만, 폐쇄(occlusion), 모호한 경계, 촬영 장비 및 프로토콜의 다양성으로 인해 발생하는 불확실성으로 인해 성능이 저하되는 문제가 있다. 이를 해결하기 위해 추론 단계에서 데이터나 모델에 변형을 주는 다양한 Test-Time 기법들이 사용되어 왔다.

기존의 접근 방식들은 다음과 같은 한계를 가진다.

1. **Test-Time Augmentation (TTA):** 미리 정의된 기하학적 또는 광도 변환(회전, 반전 등)에 의존하므로, 의료 영상의 복잡하고 이질적인 특성에 유연하게 대응하지 못하며 증강 공간이 제한적이다.
2. **Test-Time Dropout (TTD):** 뉴런의 무작위 비활성화를 통해 불확실성을 측정하지만, 제어되지 않은 무작위성으로 인해 분할 정확도가 떨어지거나 결과가 불안정해질 수 있다.
3. **Test-Time Model Adaptation (TTMA):** 테스트 샘플에 맞춰 모델 파라미터를 업데이트하지만, 단일 최적 파라미터 설정으로 수렴하는 경향이 있어 불확실성 정량화에 필요한 다양한 예측 앙상블을 생성하기 어렵다.

본 논문의 목표는 의료 영상의 도메인 특성을 반영하면서도, 각 테스트 이미지의 콘텐츠에 최적화된 다양하고 맥락적인 증강 이미지를 생성하여 분할 정확도를 높이고 정밀한 픽셀 단위 오차 추정(Pixel-wise error estimation)을 가능하게 하는 **Test-Time Generative Augmentation (TTGA)** 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 도메인에 맞게 미세 조정된(domain-fine-tuned) 생성 모델을 사용하여 추론 시점에 각 이미지에 특화된 증강 샘플을 생성하는 것이다. 주요 기여 사항은 다음과 같다.

1. **TTGA 프레임워크 제안:** 의료 데이터로 학습된 확산 모델(Diffusion Model)을 활용하여, 테스트 이미지의 글로벌 레이아웃은 유지하면서 로컬 구조의 다양성을 확보한 현실적인 증강 이미지를 생성한다.
2. **Masked Null-text Inversion 기법:** 확산 모델의 인버전 과정을 개선하여 특정 영역에만 선택적으로 증강을 적용하는 기법을 제안한다. 이를 통해 정체성 보존(Identity preservation)과 제어된 변이(Controlled variability) 사이의 균형을 맞춘다.
3. **Dual Denoising Pathways 설계:** 이미지의 각 영역을 '정체성 보존 경로'와 '증강 강화 경로'로 나누어 처리함으로써, 임상적 유용성을 해치지 않으면서도 모델의 강건성을 높이는 섭동(perturbation)을 생성한다.
4. **광범위한 검증:** 세 가지 서로 다른 의료 영상 분할 작업과 9개의 데이터셋을 통해 TTGA가 분할 정확도를 높일 뿐만 아니라, 매우 정밀한 불확실성 맵(Error estimation map)을 제공함을 입증하였다.

## 📎 Related Works

### 기존 Test-Time 기법의 한계

기존 TTA는 단순한 기하학적 변환에 의존하여 이미지 콘텐츠의 유의미한 변경을 이끌어내지 못하며, TTD와 같은 모델 섭동 방식은 제어 불가능한 무작위성으로 인해 정확도를 희생시키는 경향이 있다.

### 확산 모델 및 이미지 편집

최근 Latent Diffusion Models (LDMs)와 같은 생성 모델이 고품질 이미지 합성에 널리 사용되고 있으며, 특히 Null-text Inversion과 같은 기법을 통해 실제 이미지의 세부 사항을 유지하며 편집하는 연구가 진행되었다. 그러나 의료 영상 분야에서는 텍스트-이미지 쌍 데이터의 부족으로 인해 텍스트 기반 제어가 어렵다는 한계가 있다.

### 차별점

TTGA는 복잡한 텍스트 프롬프트에 의존하는 대신, **LoRA(Low-Rank Adaptation)**를 통한 도메인 적응과 **마스크 기반의 지역적 제어**를 결합하여 의료 영상에 최적화된 증강 전략을 구축하였다는 점에서 기존 생성 기반 편집 방식과 차별화된다.

## 🛠️ Methodology

### 1. 전체 파이프라인

TTGA의 전체 과정은 **DDIM Inversion $\rightarrow$ One-step Null-text Optimization $\rightarrow$ Masked-guided Generation** 순으로 진행된다. 테스트 이미지가 입력되면, 이를 잠재 공간(Latent space)으로 인코딩한 후 최적화된 null-text 임베딩을 찾아내고, 이를 이용해 다양한 증강 이미지를 생성하여 최종적으로 분할 결과들을 앙상블한다.

### 2. One-step Null-text Optimization

기존의 null-text inversion은 모든 타임스텝에서 최적화를 수행하여 계산 비용이 매우 높다. 본 논문은 증강의 목적이 세밀한 시맨틱 편집이 아니라 외형 유지에 있다는 점에 착안하여, 특정 중간 타임스텝 $\tau$에서 단 한 번의 최적화를 수행하는 방식을 제안한다.

손실 함수는 다음과 같이 정의된다.
$$\mathcal{L}_\tau = \| \mathbf{x}^*_0 - \mathbf{x}_0(\mathbf{x}_\tau, \tau, \mathbf{c}, \emptyset^*_\tau) \|^2_2$$
여기서 $\mathbf{x}^*_0$는 인버전된 템플릿이며, $\emptyset^*_\tau$는 최적화 대상인 null-text 임베딩이다. 이를 통해 계산 효율성을 극대화하면서 원래 이미지의 정체성을 보존한다.

### 3. Dual Denoising Paths with Masks

이미지를 두 개의 경로로 나누어 처리하여 정체성 보존과 증강의 균형을 맞춘다.

- **Identity-Preserving Path ($\mathcal{S}$):** 최적화된 $\emptyset^*_\tau$를 사용하여 원래 이미지의 세부 사항을 엄격하게 유지한다.
- **Augmentation-Enhancing Path ($\mathcal{C}$):** 시맨틱 조건 $\mathbf{c}$와 정체성 조건 $\mathbf{I}$를 동시에 적용하여, 도메인 내에서 허용 가능한 현실적인 변이를 생성한다.

두 경로의 노이즈 예측은 다음과 같이 결합된다.
$$\bar{\mathbf{x}}_{t-\Delta t} = \mathbf{M}^\mathcal{S}_{t-\Delta t} [ \text{Identity Path Result} ] + \mathbf{M}^\mathcal{C}_{t-\Delta t} [ \text{Augmentation Path Result} ]$$
여기서 $\mathbf{M}^\mathcal{S}$와 $\mathbf{M}^\mathcal{C}$는 이미지를 두 영역으로 나누는 바이너리 마스크이다.

#### 마스크 생성 전략

1. **Bernoulli Scheme:** 무작위로 영역을 할당한다.
2. **Attention Scheme:** 확산 모델의 Attention Map을 활용하여 중요 해부학적 구조는 보존 경로에, 배경은 증강 경로에 할당한다.
3. **Hybrid Scheme:** 위 두 방식을 혼합하여 유연성과 정밀함을 동시에 확보한다. (본 논문에서 최적의 성능을 보임)

### 4. Multi-condition Guidance

시맨틱 조건 $\mathbf{c}$와 정체성 조건 $\mathbf{I}$를 동시에 반영하기 위해 Classifier-free guidance를 확장한 다중 조건 가이던스를 사용한다.
$$\ddot{\epsilon}_\theta(\mathbf{x}_t, t, \emptyset, \mathbf{I}, \mathbf{c}) = \epsilon_\theta(\mathbf{x}_t, t, \emptyset) + \lambda_c [ \epsilon_\theta(\mathbf{x}_t, t, \mathbf{c}) - \epsilon_\theta(\mathbf{x}_t, t, \emptyset) ] + \lambda_I [ \tilde{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{I}, \mathbf{c}) - \epsilon_\theta(\mathbf{x}_t, t, \mathbf{c}) ]$$
여기서 $\lambda_c$는 시맨틱 일치도를, $\lambda_I$는 정체성 보존 정도를 조절한다.

## 📊 Results

### 실험 설정

- **데이터셋 및 작업:**
  - Optic Disc/Cup 분할 (REFUGE 데이터셋)
  - Polyp 분할 (Kvasir, CVC-ClinicDB 등 5개 데이터셋)
  - Skin Lesion 분할 (ISIC 2017, 2018, PH2 데이터셋)
- **비교 대상:** Baseline(단일 예측), TTD(Test-time Dropout), TTA(기하학적 변환), Deep Ensemble, DDIM/Null-text Inversion.
- **평가 지표:**
  - 분할 정확도: Dice Similarity Coefficient (DSC), AUC, HD95.
  - 오차 추정 능력: DSC, AUC, Normalized Surface Distance (NSD).

### 주요 결과

1. **분할 정확도 향상:** 모든 작업에서 TTGA가 baseline 및 다른 TTA 기법보다 우수한 성능을 보였다. 특히 DSC 기준 $0.1\%$에서 $2.3\%$까지의 이득을 얻었으며, HD95(경계 정확도)에서 뚜렷한 개선이 나타났다.
2. **픽셀 단위 오차 추정(Uncertainty Estimation):** TTGA의 가장 강력한 성능이 나타난 부분이다. Baseline 대비 DSC가 최소 $1.1\%$에서 최대 $29.0\%$까지 향상되었다. 이는 TTGA가 생성한 증강 이미지들이 단순한 노이즈가 아니라, 모델이 헷갈려 하는 '의미 있는' 변이를 제공하기 때문이다.
3. **강건성:** 과적합(overfitting)된 모델(예: HSNet)의 경우, 무작위성이 강한 DDIM Inversion이나 TTD는 오히려 성능을 급격히 떨어뜨렸으나, TTGA는 정체성 보존 제어를 통해 안정적으로 성능을 향상시켰다.

## 🧠 Insights & Discussion

### 강점 및 분석

- **정체성 보존의 중요성:** 본 연구는 단순한 데이터 증강보다 '원래 이미지의 정체성을 유지하면서 변이를 주는 것'이 의료 영상 분할에서 얼마나 중요한지 보여주었다. 과적합된 모델일수록 무분별한 섭동에 취약하며, TTGA의 제어된 증강이 이를 해결하는 열쇠가 된다.
- **형태학적 일반화:** 분석 결과(Fig 6), TTGA는 타겟의 크기나 원형도(circularity)에 관계없이 일관된 성능 향상을 보였다. 특히 매우 불규칙한 형태의 피부 병변(Skin lesion)에서도 강건한 성능을 입증하여, 특정 형태에 국한되지 않은 범용적인 방법론임을 확인하였다.

### 한계 및 향후 연구

- **모델 민감도 의존성:** 분할 모델이 데이터 섭동에 매우 둔감하거나 극도로 과적합된 경우, TTGA 단독으로는 효과가 제한적일 수 있다. 저자들은 이를 해결하기 위해 모델 기반 섭동(TTD 등)과 TTGA를 결합한 혼합 솔루션(mixture solution)의 가능성을 언급하였다.
- **계산 비용:** 확산 모델의 인버전과 생성 과정이 포함되므로, 단순 TTA보다 추론 시간이 증가한다. 향후 샘플링 단계의 최적화나 적응형 샘플링 전략이 필요하다.
- **앙상블 전략:** 현재는 단순 평균(unweighted averaging)을 사용하고 있으나, 생성된 불확실성 맵을 가중치로 사용하는 지능형 융합 전략을 도입한다면 더 정밀한 경계 획득이 가능할 것이다.

## 📌 TL;DR

본 논문은 의료 영상 분할의 불확실성을 해결하기 위해, 도메인 적응된 확산 모델을 이용한 **Test-Time Generative Augmentation (TTGA)**를 제안한다. 특히 **Masked Null-text Inversion**과 **Dual Denoising Paths**를 통해 이미지의 핵심 정체성은 유지하면서도 분할 모델의 취약점을 자극하는 현실적인 변이를 생성한다. 실험 결과, TTGA는 분할 정확도를 높일 뿐만 아니라 픽셀 단위의 오차 추정 능력을 획기적으로 개선하여, 의료 AI 모델의 신뢰성과 강건성을 높이는 유용한 도구가 될 가능성을 보여주었다.
