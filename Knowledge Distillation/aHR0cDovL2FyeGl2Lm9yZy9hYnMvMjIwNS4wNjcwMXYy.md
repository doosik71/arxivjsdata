# Knowledge Distillation Meets Open-Set Semi-Supervised Learning

Jing Yang, Xiatian Zhu, Adrian Bulat, Brais Martinez, Georgios Tzimiropoulos (2024)

## 🧩 Problem to Solve

본 논문은 기존 지식 증류(Knowledge Distillation, KD) 방법론과 오픈셋 준지도 학습(Open-Set Semi-Supervised Learning, SSL)이 가진 한계점을 동시에 해결하고자 한다.

첫째, 기존의 KD 방법들은 주로 교사의 예측값(Prediction)이나 중간 활성화 값(Intermediate activation)을 증류하는 데 집중한다. 그러나 딥러닝 모델의 핵심 성능 요소인 구조적 표현(Structured representation), 즉 특징 차원 간의 복잡한 상호 의존성과 상관관계는 대부분 간과되어 왔다.

둘째, 기존의 오픈셋 SSL 연구들은 레이블이 없는 데이터(Unlabeled data)에 알려지지 않은 클래스(Unseen classes)가 포함되어 있을 때, 이를 Out-Of-Distribution(OOD) 샘플로 간주하여 탐지하고 제거하는 전략을 주로 사용한다. 하지만 저자들은 이러한 OOD 탐지 기반의 접근 방식이 실제의 제약 없는(Unconstrained) 대규모 데이터셋 환경에서는 매우 취약하며, 오히려 유용한 지식을 손실시킬 수 있다는 문제점을 제기한다.

따라서 본 논문의 목표는 특징 표현의 시맨틱한 구조를 효과적으로 증류하는 **Semantic Representational Distillation (SRD)** 방법을 제안하고, 이를 오픈셋 SSL 환경으로 확장하여 제약 없는 레이블 없는 데이터를 효과적으로 활용하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 교사 모델의 분류기(Classifier)를 **시맨틱 비평가(Semantic Critic)**로 활용하여 학생 모델의 표현 능력을 가이드하는 것이다.

1. **Semantic Representational Distillation (SRD):** 학생 모델의 표현(Representation)을 교사 모델의 분류기에 통과시켜 얻은 '교차 네트워크 로짓(Cross-network logit)'을 생성하고, 이를 교사 모델의 로짓과 정렬함으로써 고차원적인 시맨틱 상관관계를 증류한다.
2. **시맨틱 공간의 기저(Basis) 관점:** 학습 데이터의 기지 클래스(Seen classes)들을 시맨틱 공간의 기저(Basis)로 간주한다면, 미지의 클래스(Unseen classes) 또한 기지 클래스들의 특정 조합으로 근사할 수 있다는 가설을 제시한다. 이를 통해 OOD 샘플을 제거하는 대신, 증류 과정을 통해 미지 클래스의 데이터에서도 유용한 지식을 추출할 수 있도록 한다.
3. **오픈셋 SSL과의 연결:** KD와 오픈셋 SSL이라는 두 독립적인 분야를 통합하여, 제약 없는 대규모 데이터셋 환경에서도 강건하게 작동하는 프레임워크를 구축하고 기존 OOD 탐지 기반 SSL보다 우수함을 입증한다.

## 📎 Related Works

### Knowledge Distillation (KD)

기존 KD는 크게 두 가지 범주로 나뉜다.

- **Isolated knowledge based:** 교사의 소프트 타겟(Soft target)이나 특징 텐서, Attention map 등을 직접 복제하는 방식이다. 하지만 교사와 학생 간의 용량 차이(Capacity gap)로 인해 단순 복제가 성능 저하를 일으키는 경우가 많다.
- **Relational knowledge based:** 샘플 간의 거리나 각도 등 관계적 지식을 증류한다. 최근의 Contrastive Representation Distillation (CRD) 등은 상호 정보량(Mutual Information)을 최대화하려 하지만, 교사의 분류기가 가진 시맨틱 정보를 완전히 활용하지 못하며 대규모 배치 사이즈가 필요하다는 단점이 있다.

### Open-Set Semi-Supervised Learning (SSL)

대부분의 SSL은 레이블 없는 데이터가 레이블 있는 데이터와 동일한 클래스 분포를 가진다는 'Closed-set' 가정을 한다. 하지만 실제 데이터는 OOD 샘플을 포함하는 'Open-set' 환경이다. 기존의 오픈셋 SSL(예: UASD, T2T, OpenMatch)은 OOD 샘플을 식별하여 제거하거나 가중치를 낮추는 전략을 사용한다. 그러나 본 논문은 이러한 방식이 실제의 매우 제약 없는(Unconstrained) 데이터 환경에서는 제대로 작동하지 않음을 지적한다.

## 🛠️ Methodology

### 전체 시스템 구조

시스템은 먼저 레이블 데이터 $D_l$로 교사 모델 $T=\{f_t, h_t\}$를 사전 학습시킨다. 이후 고정된 교사 모델을 사용하여 학생 모델 $S=\{f_s, h_s\}$를 학습시키며, 이때 레이블 데이터 $D_l$과 레이블 없는 데이터 $D_u$를 모두 활용한다.

### 핵심 구성 요소 및 절차

1. **Cross-network Logit ($\hat{z}$):** 학생 모델의 특징 추출기 $f_s$에서 나온 표현 $x_s$를 특징 어댑터 $\phi$를 통해 변환한 후, 교사 모델의 분류기 $h_t$에 입력하여 얻은 로짓이다.
    $$\hat{z} = h_t(\phi(x_s))$$
    여기서 $\phi$는 학생의 표현을 교사의 분류기 입력 차원에 맞추기 위한 $1 \times 1$ Convolution 레이어이다.

2. **SRD 손실 함수 ($L_{srd}$):** 교사의 로짓 $z_t$와 교차 네트워크 로짓 $\hat{z}$ 사이의 거리를 최소화한다.
    $$L_{srd} = \text{dist}(z_t, \hat{z})$$
    논문에서는 세 가지 거리 함수를 제안하며, 실험적으로 MSE(Mean Square Error)가 가장 우수함을 확인하였다.
    $$L_{mse}^{srd} = \|z_t - \hat{z}\|^2 = \|(W_t)^\top (x_t - \phi(x_s))\|^2$$

3. **특징 정규화 ($R$):** 교사의 표현 $x_t$와 어댑터가 적용된 학생의 표현 $\phi(x_s)$ 간의 직접적인 정렬을 위해 Feature matching 손실을 추가한다.
    $$R = \|x_t - \phi(x_s)\|$$

### 최종 학습 목적 함수

레이블 있는 데이터 $D_l$과 레이블 없는 데이터 $D_u$에 대해 다음과 같이 손실 함수를 구성한다.
$$L_{l+u} = L_{ce}(D_l) + \alpha L_{srd}(D_l \cup D_u) + \beta R(D_l \cup D_u)$$
여기서 $L_{ce}$는 학생 모델의 예측값과 정답 레이블 간의 Cross-entropy 손실이다. 주목할 점은 $L_{srd}$와 $R$이 레이블 없는 데이터 $D_u$에도 동일하게 적용된다는 것이다. 이는 미지 클래스의 데이터라 할지라도 교사의 시맨틱 가이드를 통해 학생 모델의 표현력을 향상시킬 수 있음을 의미한다.

## 📊 Results

### 실험 설정

- **데이터셋:** CIFAR-10, CIFAR-100, ImageNet-1K, MegaFace(얼굴 인식), Tiny-ImageNet, Places365, CC3M.
- **모델:** Wide ResNet (WRN), ResNet, MobileNet, Vision Transformer (ViT).
- **비교 대상:** KD, AT, OFD, RKD, CRD (증류 방법론) / MixMatch, OpenMatch, T2T (SSL 방법론).

### 주요 결과

1. **일반적인 KD 성능:** CIFAR-10/100 및 ImageNet-1K의 다양한 네트워크 조합에서 SRD가 기존 SOTA 방법론들을 일관되게 능가하였다. 특히 MobileNet과 같은 경량 모델에서 성능 향상 폭이 컸다.
2. **특수 태스크 확장성:**
    - **얼굴 인식(Face Recognition):** 세밀한 표현력이 필요한 태스크에서도 SRD가 가장 우수하였다. 이는 교사 분류기의 클래스 프로토타입 정보를 직접 활용했기 때문이다.
    - **이진 네트워크 증류(Binary Network Distillation):** Real-valued 교사에서 Binary-valued 학생으로의 지식 전이에서도 SRD가 압도적인 성능을 보였다.
3. **오픈셋 SSL 환경:**
    - **OOD 탐지 vs SRD:** Tiny-ImageNet, Places365, CC3M과 같이 매우 제약 없는 데이터를 사용했을 때, 기존의 OOD 탐지 기반 SSL 방법론들은 성능이 오히려 저하되거나 Baseline 수준에 머물렀다. 반면 SRD는 레이블 없는 데이터를 활용할수록 성능이 꾸준히 향상되었다.
    - **결론:** 제약 없는 환경에서는 OOD 샘플을 제거하는 것보다, KD를 통해 그들로부터 잠재적인 지식(공통 속성 및 부분)을 추출하는 것이 훨씬 효과적이다.

## 🧠 Insights & Discussion

### 강점 및 통찰

- **시맨틱 가이드의 중요성:** 단순한 특징 값의 복제가 아니라, 교사의 분류기를 통해 '시맨틱하게 해석된' 표현을 정렬함으로써 학생 모델이 더 판별력 있는(Discriminative) 특징 공간을 학습하게 한다.
- **오픈셋 SSL에 대한 새로운 시각:** 기존 연구들이 OOD 데이터를 '독'으로 간주하여 제거하려 했다면, 본 논문은 이를 '잠재적 지식의 원천'으로 보고 KD의 관점에서 접근함으로써 더 높은 일반화 성능을 달성하였다.
- **아키텍처 불가지론(Architecture Agnostic):** CNN뿐만 아니라 ViT와 같은 서로 다른 구조 간의 증류에서도 효과적임을 입증하였다.

### 한계 및 논의사항

- **교사 모델 의존성:** 본 방법론은 강력하게 사전 학습된 교사 모델이 필수적이다. 만약 교사 모델의 성능이 낮거나 편향되어 있다면, 학생 모델 역시 잘못된 시맨틱 가이드를 받게 될 위험이 있다.
- **데이터 증강과의 충돌:** 실험 결과, 데이터 증강 기반의 일관성 규제(Consistency Regularization)가 SRD와 상충하여 성능을 저하시키는 현상이 발견되었다. 이는 교사 모델이 동일 이미지의 서로 다른 뷰(View)에 대해 서로 다른 예측을 내놓아 모순된 신호를 주기 때문으로 분석된다.

## 📌 TL;DR

본 논문은 교사의 분류기를 '시맨틱 비평가'로 활용하여 학생 모델의 표현을 정렬하는 **Semantic Representational Distillation (SRD)** 방법을 제안한다. 이 방법은 단순한 특징 복제를 넘어 고차원적인 시맨틱 구조를 전이시키며, 특히 미지 클래스가 섞여 있는 제약 없는 대규모 데이터셋 환경에서 기존의 OOD 탐지 기반 SSL보다 훨씬 강력한 성능을 보인다. 이는 지식 증류가 오픈셋 SSL 문제를 해결하는 매우 효과적인 전략이 될 수 있음을 시사하며, 향후 실무적인 경량 모델 최적화 및 준지도 학습 연구에 중요한 이정표를 제시한다.
