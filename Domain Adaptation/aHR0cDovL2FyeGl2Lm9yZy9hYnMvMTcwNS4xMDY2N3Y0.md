# Conditional Adversarial Domain Adaptation

Mingsheng Long, Zhangjie Cao, Jianmin Wang, and Michael I. Jordan

## 🧩 Problem to Solve

기존의 적대적 도메인 적응(Adversarial Domain Adaptation) 방법들은 분류 문제에서 발생하는 다중 모달(multimodal) 분포의 복잡한 구조를 효과적으로 정렬하지 못할 수 있습니다. 이는 도메인 분류기가 완전히 혼란스러워지더라도 두 분포가 충분히 유사하다고 보장할 수 없는 적대적 학습의 균형 문제에서 비롯됩니다. 또한, 판별적 정보(discriminative information)가 불확실할 경우 도메인 분류기(domain discriminator)를 이 정보에 기반하여 조건화(conditioning)하는 것이 위험할 수 있습니다.

## ✨ Key Contributions

- **조건부 적대적 도메인 적응(CDAN) 프레임워크 제안**: 분류기 예측(classifier predictions)에 담긴 판별적 정보를 조건으로 활용하여 적대적 적응 모델을 구성하는 원칙적인 프레임워크를 제시합니다.
- **두 가지 새로운 조건화 전략**:
  - **다중 선형 조건화(Multilinear Conditioning)**: 특징 표현($f$)과 분류기 예측($g$) 간의 교차 공분산(cross-covariance)을 포착하여 판별력(discriminability)을 향상시킵니다.
  - **엔트로피 조건화(Entropy Conditioning)**: 분류기 예측의 불확실성($H(g)$)을 제어하여 전이 가능성(transferability)을 보장합니다. 이를 통해 '쉬운(easy-to-transfer)' 예제에 더 높은 중요도를 부여합니다.
- **이론적 보장**: 일반화 오류 경계(generalization error bound)에 대한 이론적 분석을 제공합니다.
- **최고 성능 달성**: 최소한의 코드 변경으로 5가지 벤치마크 데이터셋에서 최신 기술(state-of-the-art)을 능가하는 성능을 달성합니다.

## 📎 Related Works

- **도메인 적응(Domain Adaptation)**: 주변 분포(marginal distributions) 또는 조건부 분포(conditional distributions)를 맞춰 학습자를 다른 분포의 도메인으로 일반화하는 방법입니다. 얕은(shallow) 아키텍처와 딥러닝 기반 방법들이 있습니다.
- **딥 도메인 적응(Deep Domain Adaptation)**: 딥 네트워크에 적응 모듈을 삽입하여 전이 가능한(transferable) 표현을 학습합니다. 주로 모멘트 매칭(moment matching)과 적대적 학습(adversarial training) 두 가지 기술을 활용합니다.
- **생성적 적대 신경망(GANs)**: 생성자(generator)와 판별자(discriminator) 간의 미니맥스 게임을 통해 데이터 분포를 학습합니다. 훈련 개선, 모드 붕괴(mode collapse) 해결 등의 발전이 있었으나, 두 분포를 정확히 일치시키는 데 어려움이 있습니다.
- **조건부 GANs (CGANs)**: 생성자와 판별자를 레이블이나 다른 정보에 조건화하여 이미지 생성 등의 성능을 향상시킵니다.
- **별도 도메인 판별자(Separate Domain Discriminators)**: 일부 연구는 특징과 클래스 분포를 별도의 도메인 판별자를 사용하여 정렬하지만, 특징과 클래스 간의 의존성을 통합된 조건부 도메인 판별자에서 탐색하지는 않습니다.

## 🛠️ Methodology

CDAN은 소스 분류기 $G$의 손실을 최소화하고, 조건부 도메인 판별자 $D$와 특징 표현 $F$ 및 분류기 $G$ 간의 미니맥스 게임을 수행하는 방식으로 학습됩니다.

1. **조건부 도메인 판별자 ($D(T(h))$)**:

   - 기존 적대적 도메인 적응의 두 가지 문제(다중 모달 분포 불일치, 판별 정보 불확실성)를 해결하기 위해 제안됩니다.
   - $h = (f, g)$는 특징 표현 $f=F(x)$와 분류기 예측 $g=G(x)$의 결합 변수입니다. 판별자 $D$는 이 $h$에 조건화됩니다.
   - **최적화 목표**:
     $$ \min*{G} E*{(x*s,y_s) \sim \mathcal{D}\_s} L(G(x_s),y_s) + \lambda \left( E*{x*s \sim \mathcal{D}\_s} \log [D(T(h_s))] + E*{x*t \sim \mathcal{D}\_t} \log [1-D(T(h_t))] \right) $$
        $$ \max_D E*{x*s \sim \mathcal{D}\_s} \log [D(T(h_s))] + E*{x_t \sim \mathcal{D}\_t} \log [1-D(T(h_t))] $$
        여기서 $L(\cdot, \cdot)$은 교차 엔트로피 손실이며, $\lambda$는 두 목표 사이의 균형을 조절하는 하이퍼파라미터입니다.

2. **다중 선형 조건화 ($T_{\otimes}(f,g) = f \otimes g$)**:

   - 특징 $f$와 예측 $g$를 단순히 연결(concatenation)하는 대신, 외적(outer product) $f \otimes g$를 사용하여 교차 공분산 의존성을 모델링합니다.
   - 이는 다중 모달 분포의 근간을 이루는 클래스 조건부 평균 $E_x[x|y=c]$와 같은 정보를 효과적으로 포착할 수 있습니다.

3. **무작위 다중 선형 조건화 ($T_{\circ}(f,g) = \frac{1}{\sqrt{d}}(R_f f) \circ (R_g g)$)**:

   - 다중 선형 맵 $f \otimes g$의 차원 폭발(dimension explosion) 문제를 해결하기 위해 무작위 투영(random projection)을 사용합니다.
   - $R_f$, $R_g$는 무작위 행렬이며, $\circ$는 요소별 곱(element-wise product)입니다.
   - **정리 1**은 $T_{\circ}$가 내적(inner product) 측면에서 $T_{\otimes}$의 불편향 추정량(unbiased estimator)임을 보장합니다.
   - $T(h)$는 $d_f \times d_g > 4096$일 때 $T_{\circ}$를, 그렇지 않을 때 $T_{\otimes}$를 사용합니다.

4. **엔트로피 조건화 (CDAN+E)**:

   - 분류기 예측의 불확실성($H(g) = -\sum_{c=1}^C g_c \log g_c$)을 사용하여 도메인 판별자를 훈련하는 예제에 가중치($w(H(g)) = 1 + e^{-H(g)}$)를 부여합니다.
   - 쉬운(확실한 예측의) 예제에 더 높은 중요도를 부여하여 전이 가능성(transferability)을 향상시키고, 목표 도메인에서 확실한 예측을 하도록 장려합니다.
   - **최적화 목표 (CDAN+E)**:
     $$ \min*{G} E*{(x*s,y_s) \sim \mathcal{D}\_s} L(G(x_s),y_s) + \lambda \left( E*{x*s \sim \mathcal{D}\_s} w(H(g_s)) \log [D(T(h_s))] + E*{x*t \sim \mathcal{D}\_t} w(H(g_t)) \log [1-D(T(h_t))] \right) $$
        $$ \max_D E*{x*s \sim \mathcal{D}\_s} w(H(g_s)) \log [D(T(h_s))] + E*{x_t \sim \mathcal{D}\_t} w(H(g_t)) \log [1-D(T(h_t))] $$

5. **일반화 오류 분석**: 도메인 적응 이론에 기반하여, CDAN의 목적 함수가 목표 위험(target risk)을 원천 위험(source risk)과 조건부 도메인 판별자가 정량화하는 분포 불일치(distribution discrepancy)로 바운딩(bound)하는 것과 관련 있음을 보여줍니다.

## 📊 Results

- **Office-31, ImageCLEF-DA, Office-Home**: AlexNet 및 ResNet-50 기반 실험에서 CDAN 모델들은 대부분의 전이 태스크에서 기존의 DAN, RTN, DANN, ADDA, JAN, GTA와 같은 최신 방법들을 능가합니다. 특히 CDAN+E는 CDAN보다 약간 더 나은 성능을 보입니다.
  - 특히, 소스 및 타겟 도메인이 상당히 다른 A→W 및 A→D와 같은 어려운 태스크에서 상당한 정확도 향상을 보여줍니다.
  - Office-Home과 같이 시각적으로 이질적이고 범주 수가 많은 어려운 데이터셋에서 CDAN 모델이 큰 성능 향상을 달성하여, 분류기 예측의 복잡한 다중 모달 구조를 활용한 적대적 도메인 적응의 강점을 입증합니다.
- **Digits (MNIST, USPS, SVHN) 및 VisDA-2017**: 생성적 픽셀 수준 적응(generative pixel-level adaptation) 방법인 UNIT, CyCADA, GTA가 강세를 보이는 이 데이터셋들에서도 CDAN+E는 경쟁력 있는 성능을 달성합니다. CDAN+E는 5개 데이터셋 모두에서 잘 작동하는 유일한 접근 방식입니다.
- **분석**:
  - **절단 연구(Ablation Study)**: 무작위 행렬 샘플링(Gaussian/Uniform)의 영향을 분석한 결과, 무작위 샘플링 없는 CDAN+E가 가장 좋은 성능을 보이며, 무작위 변형 중에서는 Uniform 샘플링이 가장 좋습니다. 엔트로피 조건화(CDAN+E)가 전이 가능성을 향상시킴을 입증합니다.
  - **조건화 전략**: 특징($f$)과 예측($g$)을 단순히 연결한 DANN-[f,g]보다 다중 선형 맵을 사용하는 CDAN이 훨씬 우수함을 보여줍니다. 엔트로피 가중치 $e^{-H(g)}$가 예측 정확도와 잘 일치하여, 올바른 예측 시 약 1에 가깝고, 불확실한 예측 시 1보다 훨씬 작음을 확인합니다.
  - **분포 불일치 (A-거리)**: CDAN 특징이 ResNet 및 DANN 특징보다 더 작은 A-거리를 보여 도메인 간 격차를 더 효과적으로 줄임을 시사합니다.
  - **수렴**: CDAN은 DANN보다 빠른 수렴 속도를 보입니다.
  - **시각화 (t-SNE)**: t-SNE 시각화 결과, CDAN-fg가 ResNet, DANN, CDAN-f보다 소스-타겟 정렬이 더 잘 되고 범주 식별이 더 명확하게 이루어짐을 보여줍니다.

## 🧠 Insights & Discussion

이 연구는 적대적 도메인 적응에서 분류기 예측이 제공하는 판별적 정보를 조건으로 활용하는 것의 중요성을 강조합니다. 기존 방법들이 특징 표현만을 일치시키는 데 집중하여 다중 모달 분포를 충분히 다루지 못했던 한계를, CDAN은 다중 선형 조건화와 엔트로피 조건화를 통해 극복합니다. 특히 다중 선형 조건화는 특징과 클래스 간의 교차 공분산을 포착하여 다중 모달 구조를 효과적으로 모델링하고 판별력을 높입니다. 엔트로피 조건화는 불확실한 예측을 가진 예제에 대한 판별자의 영향을 줄여 안전한 전이를 보장합니다. 이러한 접근 방식은 다양한 데이터셋에서 기존의 복잡한 생성 모델을 포함한 최신 방법들보다 뛰어난 성능을 달성하며, 효율성과 범용성을 겸비하고 있습니다.

## 📌 TL;DR

CDAN은 적대적 도메인 적응의 한계인 다중 모달 분포 불일치와 판별 정보 불확실성을 해결합니다. 특징과 분류기 예측의 교차 공분산을 포착하는 **다중 선형 조건화**와 분류기 예측의 불확실성을 제어하는 **엔트로피 조건화**를 통해 도메인 판별자를 조건화하여 판별력과 전이 가능성을 동시에 향상시킵니다. 이 간단하고 효과적인 방법은 다양한 벤치마크 데이터셋에서 최고 성능을 달성합니다.
