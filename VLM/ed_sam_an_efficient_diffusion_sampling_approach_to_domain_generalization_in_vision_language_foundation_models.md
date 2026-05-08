# ED-SAM: An Efficient Diffusion Sampling Approach to Domain Generalization in Vision-Language Foundation Models

Thanh-Dat Truong, Xin Li, Bhiksha Raj, Jackson Cothren, Khoa Luu (2024)

## 🧩 Problem to Solve

최근 CLIP과 같은 시각-언어 파운데이션 모델(Vision-Language Foundation Models)은 대규모 데이터셋과 데이터 증강 기법을 통해 뛰어난 성능을 보이고 있으나, 학습 데이터와 다른 분포를 가진 미지의 데이터 분포에 대한 도메인 일반화(Domain Generalization) 문제는 여전히 해결해야 할 과제로 남아 있다.

기존의 데이터 증강 방식들은 주로 마스킹(masking), 적대적 섭동(adversarial perturbations), 색상 지터링(color jittering)과 같은 픽셀 수준의 수정에 집중하였다. 그러나 이러한 방식들은 시각적 개념의 세부적인 의미 정보(semantic information)를 풍부하게 만드는 데 한계가 있으며, 결과적으로 모델이 완전히 새로운 도메인의 데이터 분포에 직면했을 때 일반화 성능이 저하되는 문제가 발생한다. 본 논문의 목표는 Diffusion 모델을 활용하여 의미론적으로 다양하고 도전적인 적대적 샘플을 생성함으로써, 시각-언어 모델의 도메인 일반화 능력을 향상시키는 효율적인 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Diffusion 모델의 잠재 공간(latent space)에서 데이터 분포를 제어하며 이동시키는 **Transport Transformation**을 도입하는 것이다.

단순히 픽셀 수준에서 노이즈를 추가하는 대신, Diffusion 모델이 학습한 데이터의 분포 특성을 이용해 잠재 공간에서 소스 데이터로부터 일정 거리($\rho$)만큼 떨어진 지점으로 샘플을 이동시킨 후 이를 다시 이미지로 복원한다. 이를 통해 기존의 증강 기법보다 훨씬 더 넓은 분포의 적대적 샘플을 생성할 수 있으며, 이는 모델이 미지의 데이터 분포에 대해 더욱 견고한 표현(representation)을 학습하도록 유도한다. 특히, 본 연구는 Diffusion 모델과 도메인 일반화 사이의 관계를 이론적으로 분석하여 제안 방법의 타당성을 입증하였다.

## 📎 Related Works

**1. 시각-언어 파운데이션 모델:** CLIP, ALIGN과 같은 모델들은 대비 학습(Contrastive Learning)을 통해 이미지와 텍스트의 정렬을 학습한다. 이후 LaCLIP, SLIP 등은 텍스트 증강이나 자기지도 학습 기법을 통해 성능을 개선하려 노력하였으나, 주로 데이터 규모에 의존하는 경향이 있다.

**2. Denoising Diffusion Probabilistic Model (DDPM) 및 LDM:** Diffusion 모델은 데이터 분포를 정교하게 모델링하여 고품질의 이미지를 생성할 수 있다. 특히 Latent Diffusion Model(LDM)은 잠재 공간에서 확산 과정을 수행하여 효율성을 높였다.

**3. 도메인 일반화(Domain Generalization):** 기존 연구들은 데이터 증강(Masking), 불변 특징 학습(Invariant feature learning), 또는 적대적 데이터 증강(ADA) 등을 통해 일반화 성능을 높이려 했다. 그러나 ADA나 AdvStyle과 같은 방법들은 이미지의 콘텐츠 정보(객체의 형태, 배경 등)를 크게 변화시키지 못하고 픽셀 수준의 변형에 그친다는 한계가 있다. ED-SAM은 Diffusion 모델을 통해 의미론적 배경과 콘텐츠를 실질적으로 변화시킨 샘플을 생성함으로써 이들과 차별화된다.

## 🛠️ Methodology

### 전체 파이프라인 및 시스템 구조

ED-SAM의 전체 프로세스는 크게 **적대적 샘플 생성 단계**와 **파운데이션 모델 학습 단계**로 나뉜다. 학습 시간을 단축하기 위해 Diffusion 모델을 이용한 샘플 생성은 CLIP 학습 전에 미리 수행하는 오프라인 방식으로 진행된다.

1. **Forward Process:** 원본 이미지 $x_s$를 LDM의 인코더를 통해 잠재 변수 $z_s$로 변환한다.
2. **Transport Transformation:** 잠재 공간에서 $z_s$를 제어된 거리 $\rho$만큼 이동시켜 $z^*_t$를 생성한다.
3. **Backward Process:** 변환된 $z^*_t$와 대응하는 텍스트 프롬프트 $p_s$를 조건으로 LDM의 디코더를 통해 적대적 이미지 $x^*_t$를 생성한다.
4. **Model Training:** 생성된 적대적 샘플 $x^*_t$와 원본 샘플 $x_s$를 함께 사용하여 CLIP 모델을 학습시킨다.

### 주요 방정식 및 이론적 근거

**1. 도메인 일반화의 정식화 (Worst-case Formulation)**
본 논문은 도메인 일반화 문제를 소스 분포 $p_s$ 주변의 $\rho$-거리 내에 있는 최악의 분포 $p_t$에 대해 손실을 최소화하는 문제로 정의한다.
$$\theta^*_{F_x, \theta^*_{F_p}} = \arg \min_{\theta_{F_x}, \theta_{F_p}} \sup_{p_t : D(p_t, p_s) \le \rho} \mathbb{E}_{x_t, p_t \sim p_t} [L_{CLIP}(x_t, p_t)]$$
여기서 $D(\cdot, \cdot)$는 두 분포 사이의 거리를 측정하는 Wasserstein metric이다.

**2. Transport Transformation**
잠재 공간에서의 이동을 통해 위 조건을 만족시키기 위해 다음과 같은 변환 함수 $T$를 제안한다.
$$z^*_t = T(z_s, \rho) = \frac{z_s + N(\alpha\sqrt{2}, I)}{\sqrt{2}}, \quad \alpha \sim U(-\rho, \rho)$$
이 수식은 잠재 변수 $z_s$를 $\rho$ 범위 내에서 무작위로 이동시키면서도, Gaussian 분포의 특성을 유지하여 Diffusion 모델이 유효한 이미지를 생성할 수 있도록 보장한다.

**3. 학습 목표 (Training Objective)**
최종적으로 CLIP 모델은 원본 데이터와 생성된 적대적 샘플에 대해 다음과 같은 통합 손실 함수를 통해 학습된다.
$$\theta^*_{F_x, \theta^*_{F_p}} = \arg \min_{\theta_{F_x}, \theta_{F_p}} \mathbb{E}_{x_s, p_s, x^*_t} [L_{CLIP}(x_s, p_s) + L_{CLIP}(x^*_t, p_s)]$$

## 📊 Results

### 실험 설정

- **데이터셋:** CC3M, CC12M, LAION400M (다양한 규모의 데이터셋 사용)
- **평가 지표:** Zero-shot Classification Accuracy, Linear Probing Accuracy, Fine-tuning Accuracy
- **비교 대상:** Baseline CLIP, Masking(FLIP), ADA, AdvStyle
- **구현 세부사항:** ViT-B/16 백본 사용, $M=10$개의 적대적 샘플 생성, $\rho=0.5$ 설정

### 주요 결과

1. **정량적 성능 향상:** LAION400M 데이터셋에서 CLIP 모델에 ED-SAM을 적용했을 때, Zero-shot 성능이 $67.00\% \rightarrow 70.11\%$로 향상되었으며, SLIP 모델과 결합 시 $72.53\%$라는 SOTA 성능을 달성하였다.
2. **하이퍼파라미터 분석:**
    - $\rho$ 값의 영향: $\rho=0.5$일 때 최적의 성능을 보였다. $\rho$가 너무 작으면 분포 변화가 적어 효과가 없고, 너무 크면 생성된 이미지의 품질과 의미적 일관성이 떨어지기 때문이다.
    - 샘플 수 $M$의 영향: $M=10$까지는 성능이 꾸준히 상승하다가 이후 포화 상태에 이르는 경향을 보였다.
3. **비교 분석:** Masking, ADA, AdvStyle보다 ED-SAM이 일관되게 높은 성능을 보였다. 이는 픽셀 수준의 변형보다 Diffusion 기반의 의미론적 변형이 도메인 일반화에 훨씬 효과적임을 시사한다.
4. **범용성 확인:** STL-10, Caltech-101 등 6개의 외부 벤치마크 데이터셋에서도 Zero-shot 성능이 향상되어, 미지의 도메인에 대한 일반화 능력이 입증되었다.

## 🧠 Insights & Discussion

**강점 및 기여:**
본 논문은 단순한 데이터 증강을 넘어, Diffusion 모델의 잠재 공간을 수학적으로 제어하여 도메인 일반화를 달성하는 체계적인 방법론을 제시하였다. 특히 이미지의 픽셀이 아닌 '의미적 분포'를 확장함으로써, 기존의 적대적 학습이 가지지 못했던 강력한 일반화 능력을 확보하였다. 또한, 미리 샘플을 생성해두는 방식을 통해 학습 효율성을 높인 점이 실용적이다.

**한계 및 비판적 해석:**

- **계산 비용:** Diffusion 모델을 통해 수백만 장의 적대적 샘플을 미리 생성하는 과정은 여전히 상당한 계산 자원과 시간을 요구한다.
- **잠재 공간 가설:** 잠재 공간에서의 거리 이동이 실제 데이터 도메인의 전이(domain shift)를 완벽하게 대변하는지에 대한 추가적인 분석이 필요하다.
- **모델 의존성:** 제안 방법이 사전 학습된 LDM의 성능에 크게 의존하므로, LDM 자체가 가진 편향(bias)이나 환각(hallucination) 현상이 CLIP 모델에 전이될 위험이 있다.

## 📌 TL;DR

본 논문은 시각-언어 모델의 도메인 일반화 성능을 높이기 위해 Diffusion 모델의 잠재 공간을 제어하여 적대적 샘플을 생성하는 **ED-SAM**을 제안한다. 제안된 **Transport Transformation**을 통해 의미론적으로 다양한 샘플을 생성하여 학습시킨 결과, CC3M부터 LAION400M까지 다양한 규모의 데이터셋에서 기존의 픽셀 기반 증강 기법들을 압도하는 SOTA 성능을 달성하였다. 이 연구는 생성 AI 모델을 단순한 콘텐츠 제작이 아닌, 파운데이션 모델의 강건성과 일반화 성능을 높이는 데이터 엔진으로 활용할 수 있음을 보여주었다.
