# RemInD: Remembering Anatomical Variations for Interpretable Domain Adaptive Medical Image Segmentation

Xin Wang et al. (2025)

## 🧩 Problem to Solve

본 논문은 의료 영상 분할(Medical Image Segmentation) 분야에서 비지도 도메인 적응(Unsupervised Domain Adaptation, UDA) 시 발생하는 문제들을 해결하고자 한다. 의료 영상에서는 데이터 획득 장비나 프로토콜의 차이로 인해 소스 도메인(Source Domain)과 타겟 도메인(Target Domain) 간의 이미지 패턴이 달라지는 도메인 시프트(Domain Shift) 문제가 빈번하게 발생하며, 타겟 도메인의 레이블 데이터를 일일이 생성하는 것은 막대한 비용과 시간이 소모된다.

기존의 UDA 방법론들은 주로 고차원 특징 공간(High-dimensional feature space)에서 도메인 정렬(Domain alignment)을 수행하는 전략을 취해왔다. 그러나 이러한 접근 방식은 크게 두 가지 한계를 가진다. 첫째, 고차원 공간에서의 정렬은 계산 비용이 매우 높으며, 근사치에 의존하는 경우가 많아 정밀도가 떨어진다. 둘째, 명시적이고 설명 가능한 메커니즘이 부족하여, 정렬 과정에서 중요한 해부학적 구조 정보(Anatomical information)가 손실되거나 구조적으로 부정확한 예측 결과가 생성될 위험이 크다. 따라서 본 연구의 목표는 인간의 인지 과정과 유사하게 해부학적 변이를 기억하고 이를 활용하여 효율적이고 해석 가능한 도메인 적응 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 인간이 새로운 영상 도메인을 접할 때, 기억 속에 저장된 대표적인 구성 요소(Components)들의 조합을 회상하고 이를 약간의 변형(Warping)을 통해 적용하는 인지 과정에서 영감을 얻은 것이다. 이를 위해 다음과 같은 핵심 설계를 제안한다.

1. **Anchor-based Latent Manifold**: 소수의 앵커(Anchors)들로 정의되는 도메인 불가지론적(Domain-agnostic) 잠재 매니폴드를 학습한다. 각 이미지는 이 앵커들의 가중 합(Weighted average)으로 표현되어, 해부학적 구조의 변이를 효율적으로 메모리화하고 복원할 수 있다.
2. **Low-dimensional Alignment on Probability Simplex**: 고차원 특징 공간 대신, 저차원의 확률 심플렉스(Probability simplex) 상에서 앵커 가중치 벡터를 정렬함으로써 계산 효율성을 극대화하고 정렬 과정의 해석 가능성을 높였다.
3. **Geometrically Faithful Adaptation**: Fisher-Rao Metric을 도입하여 심플렉스의 비유클리드 기하학적 특성을 반영한 최적 운송(Optimal Transport) 정렬을 수행하며, 공간 변형(Spatial deformation)을 통해 개별 이미지의 세밀한 해부학적 차이를 보정한다.

## 📎 Related Works

기존의 UDA 연구들은 주로 다음과 같은 접근 방식을 사용하였다.

- **Adversarial 및 Semi-supervised 방식**: 판별자(Discriminator)나 유사 레이블(Pseudo-labels)을 통해 도메인 일관성을 강제한다. 하지만 이는 구조적 정확성보다 도메인 간 분포 일치에 우선순위를 두어, 핵심적인 해부학적 세부 사항을 손실할 위험이 있다.
- **Variational 및 Optimal Transport 방식**: 전역적인 특징 분포를 정렬하는 데 효과적이지만, 고차원 데이터 처리로 인해 계산 비용이 매우 크며 개별 특징의 품질을 세밀하게 제어하기 어렵다.

RemInD는 이러한 방식들과 달리, 특징 공간 전체를 정렬하는 대신 해부학적 대표성을 띠는 앵커들의 가중치만을 정렬함으로써 구조적 일관성을 유지하면서도 계산 효율성을 확보했다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조

RemInD는 베이즈 추론(Bayesian inference) 프레임워크를 기반으로 하며, 이미지를 세 가지 요소인 **콘텐츠 표현(Content representation $Z$)**, **공간 변형(Spatial transformation $\phi$)**, 그리고 **스타일 표현(Style representation $S$)**으로 분리하여 처리한다.

### 주요 구성 요소 및 학습 절차

**1. 베이즈 추론 및 ELBO**
모델은 변분 추론(Variational Inference)을 통해 잠재 변수 $w, z, v, s$를 추론한다. 여기서 $w$는 앵커 가중치, $z$는 아틀라스 표현, $v$는 공간 변형을 결정하는 속도장(Velocity field), $s$는 스타일 코드이다. 목적 함수인 ELBO(Evidence Lower Bound)는 다음과 같이 정의된다.

$$ \text{ELBO}(x,y) = \mathcal{L}_{\text{recon}}(x) + \mathcal{L}_{\text{seg}}(x,y) - D_{\text{KL}}[q(s|x)\|p(s)] - D_{\text{KL}}[q(w|x)\|p(w)] - \mathcal{L}_{\text{atlas}}(x) - \mathcal{L}_{\text{vel}}(x) $$

- $\mathcal{L}_{\text{recon}}$: 이미지 재구성 손실 (L1 loss 사용).
- $\mathcal{L}_{\text{seg}}$: 소스 도메인에서의 분할 손실 (Cross-entropy 및 Dice loss 사용).
- $\mathcal{L}_{\text{atlas}}, \mathcal{L}_{\text{vel}}$: 잠재 변수들에 대한 정규화 항.

**2. 앵커 기반 매니폴드 임베딩 (Anchor-Based Manifold)**
해석 가능성을 위해 아틀라스 표현 $z$를 $M$개의 학습 가능한 앵커 분포 $\{q^m(z)\}$의 가중 기하 평균으로 정의한다.

$$ q(z|w) \propto \prod_{m=1}^{M} q^{w_m}_m(z), \quad w \in \Delta $$

여기서 $\Delta$는 확률 심플렉스이다. 가중치 $w$는 서로 다른 해부학적 모양을 섞는 '모양 블렌딩(Shape blending)' 역할을 수행하며, 이는 인간이 기억 속의 구성 요소를 조합하는 과정과 유사하다. 계산 효율을 위해 $\mathcal{L}_{\text{atlas}}$ 대신 앵커 분포 자체를 정규화하는 $\mathcal{L}_{\text{anchor}}$를 사용한다.

**3. 효율적인 도메인 정렬 (Domain Alignment)**
고차원 특징 대신 저차원의 가중치 $w$를 정렬한다. 이를 위해 Sinkhorn Divergence 기반의 최적 운송(Optimal Transport) 손실 $\mathcal{L}_{\text{align}}$을 사용한다. 이때 심플렉스의 기하학적 특성을 반영하기 위해 Fisher-Rao Metric $D_{\text{FR}}$을 비용 함수로 채택한다.

$$ C(\hat{w}, \hat{w}') = D_{\text{FR}}(\hat{w}, \hat{w}') = 2 \arccos \left( \sum_{i=1}^{M} \sqrt{\hat{w}_i \hat{w}'_i} \right) $$

**4. 기하학적 정규화 (Geometry Regularization)**
가중치 공간의 거리와 실제 해부학적 구조의 차이 사이의 연관성을 부여하기 위해 $\mathcal{L}_{\text{geo}}$를 도입한다.

$$ \mathcal{L}_{\text{geo}} = \sum_{i,j} \left[ \left( 1 - \frac{D_{\text{FR}}(\hat{w}_{s_i}, \hat{w}_{s_j})}{\pi} \right) - \text{Sim}(\text{segmentation}_i, \text{segmentation}_j) \right]^2 $$

이는 잠재 매니폴드의 기하학적 구조를 정교하게 다듬어 학습 수렴 속도를 높이고 모드 붕괴(Mode collapse)를 방지한다.

### 최종 손실 함수

$$ L = -\frac{1}{B_s}\sum \mathcal{L}_{\text{seg}} - \lambda_1 (\text{Recon Loss}) + \lambda_2 (\text{Vel Loss}) + \lambda_3 \mathcal{L}_{\text{anchor}} + \lambda_4 \mathcal{L}_{\text{align}} + \lambda_5 \mathcal{L}_{\text{geo}} $$

## 📊 Results

### 실험 설정

- **데이터셋**:
  - **MS-CMR**: 심장 MRI (소스: bSSFP, 타겟: LGE).
  - **AMOS**: 복부 CT/MRI (소스: MRI, 타겟: CT).
- **평가 지표**: Dice Similarity Coefficient (DSC $\uparrow$), Average Symmetric Surface Distance (ASSD $\downarrow$).
- **비교 대상**: ADVENT, CyCMIS, VarDA, DARUNet, MAPSeg, VAMCEI 등 최신 UDA 기법들.

### 정량적 결과

RemInD는 단 하나의 정렬 항($\mathcal{L}_{\text{align}}$)만을 사용했음에도 불구하고, 여러 복잡한 정렬 전략을 사용하는 기존 방법론들보다 우수한 성능을 보였다.

- **MS-CMR**: 평균 DSC와 ASSD 모두에서 SOTA를 달성했으며, 특히 심근(Myocardium)의 ASSD를 기존 최고 성능 대비 18% 개선하였다.
- **AMOS**: 간, 비장, 신장 분할에서 우수한 성능을 보였으며, 특히 우측 신장의 DSC를 7%, 좌/우 신장의 ASSD를 각각 39%, 42% 개선하였다.

### 정성적 및 분석적 결과

- **시각적 분석**: 기존 방법들은 영상 품질이 낮거나 아티팩트가 있을 때 구조적으로 불가능한(예: 끊어진 심실) 예측을 내놓는 경향이 있으나, RemInD는 앵커를 통한 해부학적 지식 학습 덕분에 강건하고 일관된 구조를 생성한다.
- **가중치 시각화**: $w$를 3D 공간에 투영했을 때 소스-타겟 도메인이 잘 정렬되어 있으며, 환자 내 슬라이스 위치에 따라 가중치가 부드럽게 전이되는 것을 확인하였다. 이는 모델이 해부학적 연속성을 성공적으로 학습했음을 의미한다.
- **Ablation Study**: 공간 변형($\phi$)이 정확도에 결정적인 영향을 미치며, $\mathcal{L}_{\text{align}}$과 $\mathcal{L}_{\text{anchor}}$가 성능 향상에 크게 기여함을 확인하였다. $\mathcal{L}_{\text{geo}}$는 성능 자체보다는 학습 수렴 속도를 크게 앞당기는 역할을 한다.

## 🧠 Insights & Discussion

**강점 및 기여**
본 논문은 UDA의 고질적인 문제인 '계산 복잡도'와 '해부학적 불확실성'을 저차원 매니폴드와 앵커 시스템이라는 직관적인 설계를 통해 동시에 해결하였다. 특히 Fisher-Rao Metric을 통한 기하학적 정렬은 단순한 분포 일치를 넘어 통계적 매니폴드의 특성을 반영했다는 점에서 학술적 가치가 높다.

**한계 및 비판적 해석**
논문에서 언급되었듯이, 모든 이미지에 대해 앵커를 공유하는 구조는 네트워크의 유연성을 다소 제한하여 소스 도메인 자체의 성능은 약간 저하될 수 있다. 하지만 이는 타겟 도메인으로의 일반화 성능을 얻기 위한 합리적인 트레이드-오프(Trade-off)로 판단된다. 또한, 앵커의 개수 $M$에 대한 최적의 설정 방법이 구체적으로 제시되지 않았으며, $M$이 커질수록 계산 효율성이 어떻게 변하는지에 대한 심층 분석이 부족하다.

**향후 가능성**
소스 데이터에 접근할 수 없는 Source-free Domain Adaptation으로의 확장 가능성이 매우 높다. 소스 학습 단계에서 저장된 앵커 파라미터와 가중치만으로도 타겟 도메인에 적응할 수 있는 구조이기 때문이다.

## 📌 TL;DR

RemInD는 인간의 기억 및 회상 프로세스를 모방하여, 소수의 **앵커(Anchors)**로 구성된 저차원 매니폴드 상에서 의료 영상의 해부학적 변이를 학습하고 정렬하는 Bayesian UDA 프레임워크이다. 고차원 특징 정렬 대신 **확률 심플렉스 상의 가중치 정렬**과 **Fisher-Rao Metric**을 도입하여 계산 효율성과 해석 가능성을 동시에 확보했으며, 심장 및 복부 장기 분할 실험에서 기존 SOTA 모델들을 뛰어넘는 성능을 입증하였다. 이 연구는 향후 소스 데이터 없이 타겟 데이터만으로 적응하는 Source-free UDA 연구에 중요한 기초가 될 것으로 기대된다.
