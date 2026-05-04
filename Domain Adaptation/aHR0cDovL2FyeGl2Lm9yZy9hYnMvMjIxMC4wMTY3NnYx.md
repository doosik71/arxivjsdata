# Robust Target Training for Multi-Source Domain Adaptation

Zhongying Deng, Da Li, Yi-Zhe Song, Tao Xiang (2022)

## 🧩 Problem to Solve

본 논문은 Multi-Source Domain Adaptation (MSDA)에서 발생하는 **Source-domain-bias** 문제를 해결하고자 한다. 일반적으로 대부분의 MSDA 모델은 모든 소스 도메인과 타겟 도메인의 데이터를 한 번에 학습시키는 'one-step' 접근 방식을 취한다. 그러나 이러한 방식은 학습 데이터셋이 레이블이 있고 양이 훨씬 많은 소스 도메인 데이터에 의해 지배되기 때문에, 모델이 타겟 도메인보다는 소스 도메인에 편향되는 결과를 초래한다.

이러한 편향을 줄이기 위해 레이블이 없는 타겟 데이터만을 사용하여 모델을 미세 조정(fine-tuning)하는 두 번째 학습 단계를 도입하는 방법이 고려될 수 있다. 하지만 이때 사용되는 Pseudo-labels(가짜 레이블)는 필연적으로 노이즈를 포함하고 있으며, 이를 검증 없이 사용할 경우 오히려 모델 성능을 저하시키는 새로운 편향(bias toward erroneous pseudo labels)을 유발한다. 따라서 본 논문의 목표는 **소스 도메인 편향을 제거하면서도, 노이즈가 섞인 Pseudo-labels에 강건하게(robust) 대응할 수 있는 타겟 학습 방법론을 제안하는 것**이다.

## ✨ Key Contributions

본 논문은 **BORT$^2$ (Bi-level Optimization based Robust Target Training)**라는 새로운 방법론을 제안하며, 핵심 아이디어는 다음과 같다.

1. **Two-step Training Strategy**: 기존의 one-step MSDA 모델을 첫 번째 단계에서 학습시키고, 이를 Pseudo-labels를 생성하는 Labeling function으로 활용하여 타겟 모델을 학습시키는 두 단계 구조를 제안한다.
2. **Stochastic CNN for Noise Robustness**: 타겟 모델에 Stochastic CNN 레이어를 도입하여 각 인스턴스의 특징(feature)을 가우시안 분포로 모델링한다. 이를 통해 레이블 노이즈로 인한 불확실성을 측정하고, 노이즈가 심한 데이터의 영향을 자동으로 줄인다.
3. **Bi-level Optimization**: Labeling function과 타겟 모델을 교대로 최적화하는 Bi-level Optimization 구조를 설계한다. 특히 Implicit Differentiation(암시적 미분)과 Neumann approximation을 사용하여 수백만 개의 파라미터를 효율적으로 최적화함으로써 Pseudo-labels의 품질을 지속적으로 개선한다.

## 📎 Related Works

기존의 도메인 적응(Domain Adaptation) 연구는 크게 두 가지 방향으로 진행되었다.

- **Single-Source Domain Adaptation**: 소스 도메인과 타겟 도메인 간의 특징 분포를 정렬(alignment)하기 위해 MMD, KL-divergence 또는 DANN과 같은 적대적 학습(Adversarial training)을 사용한다.
- **Multi-Source Domain Adaptation (MSDA)**: 여러 소스 도메인이 존재할 때 공유 백본(shared backbone)을 통해 도메인 불변 특징을 추출하려 한다. MDAN, $M^3SDA-\beta$, LtC-MSDA 등이 대표적이며, 이들은 주로 특징 공간에서의 정렬에 집중한다.

**기존 방식과의 차별점**: 기존 MSDA 방법론들은 대부분 'one-step' 정렬 방식에 의존하므로 소스 도메인 편향에서 자유롭지 못하다. 반면 BORT$^2$는 정렬 자체보다는 **타겟 도메인 데이터를 어떻게 효과적으로 활용하여 소스 편향을 제거할 것인가**에 집중하며, 이를 위해 두 단계의 학습 파이프라인과 노이즈 강건 학습 체계를 도입했다는 점에서 차별화된다.

## 🛠️ Methodology

BORT$^2$의 전체 파이프라인은 크게 두 단계로 구성된다.

### 1. First-Step MSDA Training

먼저 기존의 MSDA 방법론(예: FixMatch-CM)을 사용하여 레이블이 있는 소스 데이터와 레이블이 없는 타겟 데이터를 함께 학습시킨다. 이때의 목적 함수는 다음과 같다.

$$\arg \min_{\theta} \sum_{x_s, y_s \sim S, x_t \sim T} L_{ce}(F_\theta(x_s), y_s) + L_{da}(F_\theta(x_s), F_\theta(x_t))$$

여기서 $L_{ce}$는 교차 엔트로피 손실이며, $L_{da}$는 도메인 적응 손실(Domain Adaptation loss)이다. 이 단계를 통해 학습된 모델 $F_\theta$는 이후 단계에서 Pseudo-labels를 생성하는 **Labeling function** 역할을 수행한다.

### 2. Noise-Robust Target Training (BORT$^2$)

두 번째 단계에서는 타겟 데이터만을 사용하여 모델 $M_\Psi$를 학습시킨다.

#### (1) Stochastic Feature Uncertainty Modeling

노이즈 섞인 Pseudo-labels에 대응하기 위해, 특징 출력층에 stochastic 레이어를 추가한다. 특징 $z_i$를 다음과 같은 가우시안 분포로 모델링한다.

$$z_i \sim \mathcal{N}(\mu_i, \sigma_i^2)$$

여기서 $\mu$와 $\sigma$는 각각 학습 가능한 레이어 $f_\mu^\Psi, f_\sigma^\Psi$에 의해 생성된다. 학습 시에는 Reparameterization trick($z_i = \mu_i + \sigma_i \cdot \epsilon, \epsilon \sim \mathcal{N}(0, I)$)을 사용하여 역전파가 가능하게 한다.

최종 학습 목표는 Cross-entropy 손실과 Entropy maximization 손실의 합을 최소화하는 것이다.

$$L_{trn} = \frac{1}{|T|} \sum_{x_i \sim T} \mathbb{1}(\max(p_i) \geq \tau) L_{ce}(g_{\Psi_1}(z_i), \hat{y}_i) + \lambda L_{ment}(f_\sigma^\Psi(f_0^\Psi(x_i)))$$

이때 $L_{ment}(\sigma_i) = (m - \sum \log(\sigma_i))^+$는 특징의 불확실성을 조절하며, 모델은 노이즈가 심한 레이블에 대해 더 큰 표준편차($\sigma$)를 할당하여 해당 데이터의 학습 신호를 억제하게 된다.

#### (2) Bi-level Optimization of Labeling Function

Labeling function $F_\theta$가 생성하는 Pseudo-labels의 품질을 높이기 위해 다음과 같은 Bi-level Optimization 문제를 정의한다.

$$\arg \min_{\theta} L_{val}(\arg \min_{\Psi} L_{trn}(M_\Psi, T_{trn}; F_\theta), T_{val})$$

- **Inner Loop**: 주어진 $\theta$에 대해 타겟 모델 $M_\Psi$를 최적화한다. 이때 Pseudo-label 생성 과정에 Gumbel-Softmax를 도입하여 미분 가능하게 만든다.
- **Outer Loop**: 최적화된 $M_\Psi$의 특징 불확실성(Entropy)을 최소화하도록 $F_\theta$를 업데이트한다. 즉, 타겟 모델이 예측한 특징의 불확실성이 낮을수록 레이블이 정확할 가능성이 높다는 직관을 이용한다.

이 과정에서 발생하는 거대한 Hessian 행렬의 역행렬 계산 문제를 해결하기 위해, **Neumann series 기반의 Implicit Function Theorem**을 사용하여 하이퍼그레디언트(hypergradient)를 효율적으로 계산한다.

## 📊 Results

### 실험 설정

- **데이터셋**: PACS, Digit-Five, DomainNet (대규모 데이터셋)
- **비교 대상**: DANN, MCD, $M^3SDA-\beta$, LtC-MSDA, DAC-Net, STEM 등 11종의 MSDA 모델
- **평가 지표**: Accuracy (정확도)

### 주요 결과

1. **정량적 성능**: BORT$^2$는 세 가지 벤치마크 모두에서 SOTA 성능을 달성했다.
    - **PACS**: 평균 정확도를 기존 베이스라인 대비 3.96% 향상시켰으며, 특히 Sketch 도메인과 같이 도메인 차이가 큰 경우 최대 8.26%의 성능 향상을 보였다.
    - **Digit-Five**: DRT 대비 평균 4%, DAC-Net 대비 1.4% 향상된 성능을 기록했으며, Oracle(정답 레이블 사용) 결과에 근접한 성능을 보였다.
    - **DomainNet**: 가장 도전적인 Quickdraw 도메인에서 타 방법론 대비 2% 이상의 성능 이득을 얻었다.
2. **Ablation Study**:
    - 단순히 두 단계 학습(Naive second-step)만 적용해도 성능이 향상되지만, BORT$^2$의 노이즈 강건 학습과 Bi-level 최적화를 모두 적용했을 때 최대 성능이 나타났다.
    - Bi-level 최적화를 제거할 경우 성능이 0.58% 하락하며, Stochastic 모델링을 제거할 경우 더 큰 폭으로 하락함을 확인하여 각 구성 요소의 필요성을 입증했다.

## 🧠 Insights & Discussion

본 논문은 MSDA에서 발생하는 고질적인 문제인 소스-타겟 간의 불균형(bias)을 '학습 단계의 분리'와 '노이즈 강건성 확보'라는 전략으로 해결했다.

**강점**:

- **Model-Agnostic**: 제안된 BORT$^2$ 프레임워크는 특정 MSDA 모델에 종속되지 않고, 어떤 one-step MSDA 모델 이후에도 적용 가능하다는 범용성을 가진다.
- **Theoretical Soundness**: 단순한 Pseudo-labeling을 넘어 Bi-level Optimization과 Implicit Differentiation이라는 수학적 최적화 도구를 사용하여 레이블 품질을 체계적으로 개선했다.

**한계 및 논의사항**:

- **학습 비용**: 두 번째 학습 단계와 Bi-level 최적화 과정으로 인해 추가적인 연산 시간이 소요된다. 논문에서는 이를 조절하여 학습 에포크를 제한하는 실험을 진행했으나, 여전히 one-step 방식보다는 비용이 높다.
- **가정**: 특징의 엔트로피(불확실성)가 레이블의 정확성과 직접적인 상관관계가 있다는 가정을 기반으로 한다. 이는 많은 경우 유효하지만, 모든 데이터 분포에서 항상 성립하는지에 대한 추가 논의가 필요할 수 있다.

## 📌 TL;DR

이 논문은 Multi-Source Domain Adaptation에서 소스 도메인 편향을 제거하기 위해 **두 단계 학습 전략(BORT$^2$)**을 제안한다. 첫 단계에서 일반적인 MSDA 모델을 학습시켜 Labeling function으로 쓰고, 두 번째 단계에서는 **Stochastic CNN**을 통해 노이즈에 강건한 타겟 모델을 학습시키며, **Bi-level Optimization**을 통해 레이블 품질을 최적화한다. 실험 결과 PACS, Digit-Five, DomainNet 등 주요 벤치마크에서 SOTA 성능을 달성하였으며, 특히 도메인 간 격차가 큰 환경에서 매우 효과적임을 입증했다. 이 연구는 향후 극심한 도메인 시프트가 존재하는 실제 환경의 AI 모델 적용에 중요한 기여를 할 것으로 보인다.
