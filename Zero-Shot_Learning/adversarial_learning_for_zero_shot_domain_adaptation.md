# Adversarial Learning for Zero-shot Domain Adaptation

Jinghua Wang and Jianmin Jiang (2020)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 문제는 Zero-shot Domain Adaptation (ZSDA)이다. 일반적인 Domain Adaptation은 학습 단계에서 타겟 도메인의 데이터(레이블은 없을 수 있음)가 가용한 상태에서 소스 도메인의 지식을 전이하지만, ZSDA는 학습 단계에서 타겟 도메인의 데이터 샘플과 레이블 모두를 전혀 사용할 수 없는 극단적인 상황을 가정한다.

이러한 문제는 실제 환경에서 매우 중요하다. 예를 들어, 기존 카메라로 학습된 AI 시스템을 새로 설치된 다른 카메라에 적용해야 할 때, 새로운 카메라에서 캡처된 데이터에 즉각적으로 접근할 수 없는 상황이 발생할 수 있다. 따라서 본 논문의 목표는 타겟 도메인의 데이터에 접근하지 않고도 해당 도메인에서 잘 작동하는 머신러닝 모델을 도출하고, 나아가 타겟 도메인의 데이터를 합성해내는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 서로 다른 태스크 간에도 동일한 도메인 쌍(Domain Pair) 사이의 '도메인 시프트(Domain Shift)'가 공유될 것이라는 가설에서 출발한다.

가장 중점적인 기여는 다음과 같다. 첫째, 도메인 시프트를 두 도메인 간의 쌍을 이루는 샘플(Paired Samples)들 사이의 표현 차이(Representation Difference)의 분포로 정의하고, 이를 서로 다른 태스크 간에 전이하는 새로운 전략을 제안하였다. 둘째, 관련 없는 태스크(Irrelevant Task, IrT)에서 CoGAN을 통해 도메인 시프트를 학습한 후, 이를 관심 태스크(Task of Interest, ToI)로 전이하여 타겟 도메인의 데이터를 합성하고 모델을 학습시키는 프레임워크를 구축하였다. 셋째, 합성된 데이터의 품질을 높이기 위해 두 개의 Co-training Classifier를 도입하여 일관성(Consistency)을 통한 정규화 기법을 적용하였다.

## 📎 Related Works

기존의 Domain Adaptation 연구는 주로 소스와 타겟 도메인 간의 공통된 도메인 불변 특징(Domain-invariant features)을 학습하여 두 도메인의 괴리를 최소화하는 방향으로 진행되었다. 하지만 이러한 방법들은 학습 시 타겟 도메인 데이터가 필요하므로 ZSDA 문제를 해결할 수 없다.

기존의 ZSDA 접근 방식은 크게 세 가지로 나뉜다. 첫째는 여러 소스 도메인에서 일반화 성능이 좋은 불변 특징을 학습하는 방법(DICA, MTAE 등)이고, 둘째는 도메인을 공통 요인과 도메인 특유 요인으로 분해하여 공통 요인을 전이하는 방법이다. 셋째는 보조 태스크(Assistant Task)를 통해 도메인 간의 상관관계를 먼저 학습하는 방법(ZDDA, CoCoGAN 등)이다.

본 논문이 제안하는 방법은 세 번째 전략에 해당하지만, 기존 ZDDA나 CoCoGAN과 달리 도메인 시프트를 명시적으로 정의하고 이를 태스크 간에 전이한다는 점과, 타겟 도메인 데이터가 전혀 없는 상황에서도 CoGAN의 구조적 제약을 통해 더 유연하게 적용 가능하다는 점에서 차별점을 갖는다.

## 🛠️ Methodology

### 전체 파이프라인

본 방법론은 크게 두 단계의 CoGAN 학습으로 구성된다. 먼저 데이터가 모두 가용한 IrT에서 $\text{CoGAN-IrT}$를 학습하여 도메인 시프트를 캡처한다. 이후 이 시프트 정보를 $\text{CoGAN-ToI}$로 전이하여 타겟 도메인의 데이터를 생성하고 분류기를 학습시킨다.

### 1. 도메인 시프트의 정의 및 학습 ($\text{CoGAN-IrT}$)

CoGAN은 두 도메인의 결합 분포를 학습하는 GAN의 확장판으로, 생성기와 판별기의 일부 레이어를 공유하여 도메인 간의 상관관계를 학습한다. 본 논문에서는 도메인 시프트를 공유 레이어에서의 표현 차이로 정의한다. IrT의 소스 샘플 $x^\beta_s$와 타겟 샘플 $x^\beta_t$에 대해, 공유 레이어의 출력인 $R^h(\cdot)$를 이용하여 다음과 같이 도메인 시프트 $\delta^\beta_h$를 정의한다.

$$\delta^\beta_h = R^h(x^\beta_t) \ominus R^h(x^\beta_s)$$

여기서 $\ominus$는 요소별 차이(element-wise difference)를 의미한다. $\text{CoGAN-IrT}$는 다양한 노이즈 $z^\beta$를 입력받아 $\delta^\beta_h$의 분포 $p(\delta^\beta_h)$를 생성하게 된다.

### 2. Co-training Classifiers

$\text{CoGAN-ToI}$의 학습을 가이드하기 위해 두 개의 분류기 $\text{clf}_1, \text{clf}_2$를 도입한다. 이들은 서로 다른 관점에서 데이터를 분석하도록 유도되며, 다음과 같은 손실 함수로 학습된다.

$$L(\text{clf}_1, \text{clf}_2) = \lambda_w L^w(w_1, w_2) + \lambda_{\text{acc}} L_{\text{cls}}(X^\alpha_s) - \lambda_{\text{con}} L_{\text{con}}(X^\beta) + \lambda_{\text{diff}} L_{\text{diff}}(\tilde{X})$$

- $L^w$: 두 분류기의 파라미터 $w_1, w_2$ 사이의 코사인 유사도를 측정하여 두 모델이 서로 다르게 학습되도록 강제한다.
- $L_{\text{cls}}$: ToI의 소스 도메인 데이터 $X^\alpha_s$에 대한 표준 분류 손실이다.
- $L_{\text{con}}$: IrT의 샘플들에 대해 두 분류기가 일관된 예측을 내놓도록 유도한다.
- $L_{\text{diff}}$: 무작위 노이즈 이미지 등 관련 없는 샘플 $\tilde{X}$에 대해서는 서로 다른 예측을 하도록 유도한다.

### 3. 관심 태스크로의 전이 ($\text{CoGAN-ToI}$)

$\text{CoGAN-ToI}$는 타겟 도메인 데이터 $X^\alpha_t$가 없으므로, 다음 세 가지 제약 조건을 통해 학습된다.

1. 소스 도메인 브랜치는 가용한 $X^\alpha_s$를 사용하여 독립적인 GAN으로 학습한다.
2. **도메인 시프트 보존**: ToI의 시프트 $\delta^\alpha_h = R^h(x^\alpha_t) \ominus R^h(x^\alpha_s)$가 IrT의 시프트 $\delta^\beta_h$와 구별되지 않도록 Task Classifier를 통해 정렬한다.
3. **분류기 일관성**: 합성된 타겟 데이터 $x^\alpha_t$에 대해 $\text{clf}_1$과 $\text{clf}_2$가 일관된 예측을 하도록 유도한다.

최종적인 $\text{CoGAN-ToI}$의 목적 함수는 다음과 같다.

$$V(P^\alpha_g, T^\alpha_g, T^\alpha_d, P^\alpha_d) \equiv \lambda^\alpha_{\text{con}} \sum_{x^\alpha_t = g^\alpha_t(z^\alpha)} v_1(x^\alpha_t) \cdot v_2(x^\alpha_t) - L_{\text{clf}}(\delta^\alpha_h, \delta^\beta_h)$$

여기서 $v_1, v_2$는 각 분류기의 예측값이며, 두 번째 항은 ToI와 IrT의 도메인 시프트를 일치시키려는 시도이다.

## 📊 Results

### 실험 설정

- **데이터셋**: MNIST, Fashion-MNIST, NIST, EMNIST 등 4개의 그레이스케일 데이터셋을 사용하였다. 이를 기반으로 컬러(C-dom), 엣지(E-dom), 네거티브(N-dom) 도메인을 인위적으로 생성하여 실험하였다.
- **비교 대상**: ZDDA, CoCoGAN 및 Ablation Study를 위한 CTCC(분류기 일관성만 사용한 모델)와 비교하였다.
- **공공 데이터셋**: Office-Home 데이터셋을 사용하여, 카테고리 일부를 ToI로, 나머지를 IrT로 설정하여 실험을 진행하였다.

### 주요 결과

1. **합성 도메인 실험**: 다양한 도메인 쌍(G$\to$C, G$\to$E 등)에서 제안 방법이 ZDDA와 CoCoGAN보다 평균적으로 높은 정확도를 보였다. 특히 ToI가 EMNIST이고 IrT가 Fashion-MNIST일 때, ZDDA 대비 8.9%, CoCoGAN 대비 4.1% 향상된 성능을 기록하였다.
2. **Office-Home 실험**: 12가지의 소스-타겟 조합 모두에서 가장 우수한 성능을 보였으며, 특히 Rw $\to$ Cl 전이에서는 기존 방법들보다 10% 이상 높은 정확도를 달성하였다.
3. **분석**: CTCC(Baseline) 대비 성능이 크게 향상된 점을 통해, 도메인 시프트를 전이하는 메커니즘이 $\text{CoGAN-ToI}$ 학습에 결정적인 역할을 함을 입증하였다.

## 🧠 Insights & Discussion

본 논문은 타겟 도메인의 데이터가 전무한 상황에서도 도메인 시프트라는 개념을 추상화하여 다른 태스크에서 빌려옴으로써 ZSDA 문제를 해결할 수 있음을 보여주었다. 특히 단순히 분류 모델만 만드는 것이 아니라 타겟 도메인의 데이터를 실제로 합성해낼 수 있다는 점이 큰 강점이다.

다만, 본 방법론은 '도메인 시프트가 태스크 간에 공유된다'는 강한 가정에 의존한다. 저자들 역시 결론 부분에서 RGB-그레이와 같이 도메인 간 시프트가 명확하고 큰 경우에는 잘 작동하지만, 시프트가 작은 데이터셋(예: Office-31)에서는 성능이 저하될 수 있음을 명시하였다. 이는 도메인 시프트를 단순한 표현의 차이로 정의한 한계에서 기인하며, 향후에는 단순한 차이뿐만 아니라 더 복잡한 대응 관계를 학습할 수 있는 분류기를 도입해야 할 필요가 있다.

## 📌 TL;DR

이 논문은 타겟 도메인 데이터가 전혀 없는 Zero-shot Domain Adaptation 상황을 해결하기 위해, **관련 없는 태스크(IrT)에서 학습한 도메인 시프트를 관심 태스크(ToI)로 전이**하는 방법을 제안한다. CoGAN을 통해 도메인 간의 표현 차이를 캡처하고 Co-training Classifier의 일관성을 통해 합성 데이터의 품질을 높였으며, 이를 통해 타겟 도메인 모델 학습 및 데이터 합성을 동시에 달성하였다. 이 연구는 데이터 획득이 불가능한 환경에서 도메인 적응을 수행해야 하는 실무적인 AI 시스템 구축에 중요한 가능성을 제시한다.
