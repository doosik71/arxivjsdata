# Dual Adversarial Domain Adaptation

Yuntao Du, Zhiwen Tan, Qian Chen, Xiaowen Zhang, Yirong Yao, and Chongjun Wang (2020)

## 🧩 Problem to Solve

본 논문은 **Unsupervised Domain Adaptation (UDA)** 문제를 해결하고자 한다. UDA의 핵심은 라벨이 있는 소스 도메인(Source Domain)에서 학습된 지식을 라벨이 없는 타겟 도메인(Target Domain)으로 전이시켜 타겟 도메인에서의 분류 성능을 높이는 것이다.

이 과정에서 발생하는 주요 문제는 소스 도메인과 타겟 도메인 사이의 **분포 불일치(Distribution Discrepancy)**이다. 기존의 적대적 도메인 적응(Adversarial Domain Adaptation) 방식들은 주로 이진 분류기(Binary Discriminator)를 사용하여 도메인 수준의 정렬(Marginal Alignment)을 수행하거나, $K$-차원 출력의 분류기를 통해 클래스 수준의 정렬(Conditional Alignment)을 독립적으로 수행했다. 그러나 이러한 방식들은 다음과 같은 한계가 있다.

1. **독립적 정렬의 한계**: 도메인 수준과 클래스 수준의 정렬을 각각 따로 수행하면 두 정보 사이의 시맨틱(Semantic) 정보를 공유할 수 없다.
2. **단일 판별자의 정보 부족**: 하나의 판별자만으로는 도메인 간의 복잡한 구조와 유용한 정보를 모두 포착하기 어렵다.
3. **결정 경계 고려 부족**: 타겟 샘플이 소스 도메인의 서포트(Support) 밖에 존재할 경우, 판별적인 특징을 학습하기 어려워 성능이 저하되는 문제가 발생한다.

따라서 본 논문의 목표는 도메인 수준과 클래스 수준의 정렬을 동시에 수행하면서, 두 개의 판별자를 적대적으로 경쟁시켜 더 풍부한 정보를 학습하고 타겟 특징을 소스 도메인의 서포트 내부로 유도하는 **Dual Adversarial Domain Adaptation (DADA)** 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 설계 아이디어는 다음과 같다.

1. **$2K$-차원 출력 판별자 도입**: 단일 판별자가 도메인 정보와 클래스 정보를 동시에 학습할 수 있도록 $2K$-차원 출력을 설계하였다. 이를 통해 도메인 수준(Marginal)과 클래스 수준(Conditional)의 정렬을 동시에 수행한다.
2. **Dual Adversarial 전략**: 두 개의 판별자가 서로 경쟁하게 만드는 메커니즘을 설계하였다. 두 판별자 간의 불일치(Discrepancy)를 최대화하여 다양한 보완적 정보를 포착하게 하고, 특징 추출기(Feature Extractor)는 이 불일치를 최소화하도록 학습하여 타겟 특징이 소스 도메인의 서포트를 벗어나지 않도록 강제한다.
3. **SSL 정규화 적용**: 타겟 데이터의 라벨이 없으므로, 엔트로피 최소화(Entropy Minimization)와 Virtual Adversarial Training (VAT)을 통한 Semi-Supervised Learning (SSL) 정규화를 도입하여 특징의 판별력을 높였다.

## 📎 Related Works

논문에서는 도메인 적응 연구를 세 가지 단계로 구분하여 설명한다.

- **Shallow Domain Adaptation**: TCA, JDA, BDA, MEDA 등이 있으며, 주로 특징 매핑 과정에서 marginal 및 conditional 분포의 불일치를 줄이는 데 집중했다. 하지만 딥러닝의 강력한 특징 추출 능력을 활용하지 못한다는 한계가 있다.
- **Deep Domain Adaptation**: DDC, DAN 등이 대표적이며, MMD(Maximum Mean Discrepancy), KL-divergence, CORAL 등의 지표를 사용하여 네트워크의 특정 레이어에서 특징 분포의 차이를 직접 최소화하는 방식을 취한다.
- **Adversarial Domain Adaptation**: GAN의 아이디어를 차용하여 DANN, ADDA, MCD 등이 제안되었다. DANN은 도메인 판별기를 속이는 방식으로 도메인 불변 특징을 학습하며, MCD는 두 판별자의 불일치를 이용해 타겟 데이터를 소스 서포트 안으로 밀어 넣는 방식을 제안했다.

DADA는 이러한 기존 방식과 달리, **$2K$-차원 판별자를 통한 동시 정렬**과 **두 판별자 간의 적대적 경쟁(Dual Adversarial)**을 결합하여 조건부 분포 정렬의 정밀도를 높였다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. 시스템 구조 및 구성 요소
DADA는 다음과 같은 네 가지 주요 구성 요소로 이루어져 있다.
- **특징 추출기 $G$**: 입력 데이터로부터 도메인 불변 특징을 추출한다.
- **클래스 예측기 $F$**: 소스 데이터를 분류하고 타겟 데이터의 Pseudo-label을 생성한다.
- **두 개의 공동 판별자 $D_1, D_2$**: $2K$-차원 출력을 가지며, 도메인과 클래스를 동시에 판별한다.

### 2. 상세 방법론 및 손실 함수

#### (1) 클래스 예측기 손실 ($\mathcal{L}_{sc}$)
소스 데이터 $(x_s, y_s)$에 대해 표준 교차 엔트로피(Cross-Entropy, CE) 손실을 사용하여 학습한다.
$$\ell_{sc}(F) = \mathbb{E}_{(x_s, y_s) \sim P} \ell_{CE}(f(x), y)$$

#### (2) 단일 판별자 손실 (Joint Discriminator Loss)
판별자의 출력 $2K$차원 중 앞의 $K$개는 소스 클래스, 뒤의 $K$개는 타겟 클래스를 나타낸다.
- **소스 분류 손실**: 소스 샘플이 앞의 $K$개 클래스 중 하나에 속하도록 학습한다.
$$\ell_{dsc}(D_1) = \mathbb{E}_{(x_s, y_s) \sim P} \ell_{CE}(D_1(G(x_s)), [y_s, 0])$$
- **타겟 분류 손실**: 타겟 샘플에 대해 클래스 예측기 $F$가 생성한 Pseudo-label $\hat{y}_t$를 사용하여 뒤의 $K$개 클래스 중 하나에 속하도록 학습한다.
$$\ell_{dtc}(D_1) = \mathbb{E}_{x_t \sim q_t} \ell_{CE}(D_1(x_t), [0, \hat{y}_t])$$
- **적대적 정렬 손실**: 특징 추출기 $G$는 판별자를 속여 소스 샘플을 타겟으로, 타겟 샘플을 소스로 인식하게 하여 도메인 간 간극을 줄인다.
$$\ell_{dsa1}(G) = \mathbb{E}_{(x_s, y_s) \sim P} \ell_{CE}(D_1(G(x_s)), [0, y_s])$$
$$\ell_{dta1}(G) = \mathbb{E}_{x_t \sim q_t} \ell_{CE}(D_1(G(x_t)), [\hat{y}_t, 0])$$

#### (3) 판별자 간 적대적 손실 (Adversarial Loss Between Discriminators)
두 판별자 $D_1, D_2$의 예측 값 차이를 이용한 불일치(Discrepancy)를 정의한다.
$$d(f^{D_1}(x), f^{D_2}(x)) = \frac{1}{K} \sum_{k=1}^{K} |f^{D_1}(x)[k] - f^{D_2}(x)[k]|$$
- **판별자 학습**: $D_1$과 $D_2$는 이 불일치 $\ell_d$를 **최대화**하여 서로 다른 보완적 정보를 학습하고 타겟 샘플의 모호성을 포착한다.
- **특징 추출기 학습**: $G$는 이 불일치를 **최소화**하여 타겟 특징이 소스 도메인의 서포트 영역 안으로 들어오도록 유도한다.

#### (4) SSL 정규화 손실
타겟 데이터의 결정 경계를 명확히 하기 위해 엔트로피 최소화($\ell_{te}$)와 Virtual Adversarial Training (VAT) ($\ell_{svat}, \ell_{tvat}$)을 적용하여 모델의 국소 립시츠(locally-Lipschitz) 조건을 강화하고 판별력을 높인다.

### 3. 학습 절차 (Training Steps)
학습은 총 3단계로 나누어 반복 수행된다.
- **Step 1**: 소스 데이터만 사용하여 $G, F, D_1, D_2$가 소스 샘플을 정확히 분류하도록 초기화 학습한다.
- **Step 2**: $G$를 고정하고 $F, D_1, D_2$를 업데이트한다. 판별자들의 분류 정확도를 높이는 동시에, 두 판별자 간의 불일치($\ell_d$)를 최대화한다. 또한 SSL 정규화를 적용한다.
- **Step 3**: $F, D_1, D_2$를 고정하고 $G$를 업데이트한다. 판별자들을 속이는 적대적 정렬 손실과 두 판별자 간의 불일치($\ell_d$)를 최소화한다.

## 📊 Results

### 1. 실험 설정
- **데이터셋**: Office-31 (3개 도메인 간 6개 전이 작업), ImageCLEF-DA (3개 도메인 간 6개 전이 작업).
- **베이스라인**: DAN, DANN, ADDA, MADA, VADA, GTA, MCD, CDAN, TAT, RCA 등.
- **구현**: ResNet-50을 특징 추출기로 사용하였으며, PyTorch로 구현되었다.

### 2. 정량적 결과
- **Office-31**: DADA는 평균 정확도 **88.0%**를 기록하여 비교 대상 중 가장 높은 성능을 보였다. 특히 도메인 차이가 크고 불균형한 $\text{D} \to \text{A}$, $\text{W} \to \text{A}$ 작업에서도 강점을 보였다.
- **ImageCLEF-DA**: 평균 정확도 **89.3%**를 달성하여 기존 SOTA 모델들보다 우수한 성능을 확인하였다.

### 3. 분석 및 검증
- **Ablation Study**: SSL 정규화를 제거했을 때 평균 정확도가 1.0% 감소함을 확인하여, 엔트로피 최소화와 VAT의 효과를 입증하였다.
- **Feature Visualization**: t-SNE 시각화 결과, Source-only 모델에 비해 DADA가 소스와 타겟 샘플을 훨씬 더 잘 정렬시키고 타겟 샘플의 군집화(Discriminative prediction)가 뚜렷함을 보여주었다.
- **A-distance**: 도메인 간 분포 차이를 측정하는 A-distance 값이 DADA에서 가장 낮게 나타나, 특징 전이 가능성(Transferability)이 가장 높음을 입증하였다.

## 🧠 Insights & Discussion

본 논문은 Ben-David 등의 도메인 적응 이론을 근거로 하여, 타겟 도메인의 일반화 오차 $\epsilon_t(h)$가 소스 오차 $\epsilon_s(h)$, $H\Delta H$-거리, 그리고 이상적인 가설의 오차 $\lambda$의 합으로 상한선이 결정된다는 점에 주목하였다.

기존 DANN과 같은 방식은 $H\Delta H$-거리를 줄이는 데 집중하지만, 조건부 분포가 일치하지 않으면 $\lambda$ 값이 커져 성능이 제한된다. DADA의 $2K$-차원 판별자와 Dual Adversarial 구조는 **조건부 분포 정렬(Conditional Alignment)**을 정밀하게 수행함으로써 $\lambda$를 효과적으로 낮춘다. 

또한, 단순히 정렬만 하는 것이 아니라 두 판별자를 경쟁시켜 타겟 샘플이 소스의 서포트 영역 밖에 존재하는지 감지하고 이를 다시 내부로 밀어 넣는 메커니즘을 도입함으로써, 기존 적대적 학습이 가졌던 '불완전한 정렬' 문제를 효과적으로 해결하였다.

## 📌 TL;DR

- **핵심 기여**: $2K$-차원 출력 판별자를 통해 도메인/클래스 정렬을 동시에 수행하고, 두 개의 판별자를 적대적으로 경쟁시키는 **Dual Adversarial** 전략과 **SSL 정규화**를 제안함.
- **성과**: Office-31 및 ImageCLEF-DA 데이터셋에서 기존 SOTA 방법론들을 능가하는 분류 정확도를 달성함.
- **의의**: 조건부 분포 정렬을 강화하고 타겟 특징을 소스 서포트 영역 내로 유도함으로써, 더 강건하고 판별력 있는 도메인 전이 학습이 가능함을 보여줌. 향후 복잡한 도메인 시프트가 존재하는 실제 환경의 UDA 연구에 중요한 기반이 될 것으로 보임.