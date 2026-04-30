# Model Adaptation: Unsupervised Domain Adaptation without Source Data

Rui Li, Qianfen Jiao, Wenming Cao, Hau-San Wong, Si Wu (2025)

## 🧩 Problem to Solve

본 논문은 기존의 Unsupervised Domain Adaptation (UDA) 설정에서 한 단계 더 나아가, **소스 데이터(Source Data)가 전혀 없는 상태에서 타겟 도메인에 모델을 적응시키는 Unsupervised Model Adaptation** 문제를 해결하고자 한다. 

일반적인 UDA 방식은 레이블이 있는 소스 데이터와 레이블이 없는 타겟 데이터를 모두 사용하여 도메인 간의 차이(Domain Shift)를 줄인다. 그러나 실제 산업 현장에서는 데이터 프라이버시 및 보안 문제로 인해 학습된 모델만 제공될 뿐, 모델 학습에 사용된 원본 소스 데이터에 접근할 수 없는 경우가 많다. 또한, 소스 데이터셋의 규모가 너무 커서 플랫폼 간 이동이나 보관이 불가능한 물리적 한계가 존재할 수 있다. 

따라서 본 연구의 목표는 **오직 사전 학습된 소스 예측 모델($C$)과 레이블이 없는 타겟 데이터($X_t$)만을 이용하여, 타겟 도메인에서의 예측 성능을 극대화하는 방법론을 제안하는 것**이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Collaborative Class Conditional Generative Adversarial Networks (3C-GAN)** 프레임워크를 통해 소스 데이터에 대한 의존성을 제거하는 것이다.

중심적인 직관은 생성기(Generator)가 타겟 스타일의 가상 데이터를 생성하여 예측 모델을 학습시키고, 다시 개선된 예측 모델이 생성기에게 더 정확한 세만틱 가이드를 제공하는 **상호 협력적(Collaborative) 구조**를 구축하는 것이다. 여기에 모델의 급격한 변화를 막기 위한 가중치 제약(Weight Constraint)과 타겟 도메인의 특성을 반영하기 위한 클러스터링 기반 정규화(Clustering-based Regularization)를 추가하여 학습의 안정성과 판별력을 높였다.

## 📎 Related Works

기존의 UDA 연구들은 크게 두 가지 방향으로 진행되어 왔다.
1. **특징 정렬(Feature Alignment):** Maximum Mean Discrepancy (MMD)나 Adversarial Training을 통해 소스-타겟 간의 도메인 불변 특징(Domain-invariant features)을 학습하는 방식이다. (예: DAN, DANN, JAN 등)
2. **데이터 변환(Data Translation):** GAN을 이용하여 소스 데이터를 타겟 스타일로 직접 변환하거나, 그 반대의 과정을 거치는 방식이다. (예: CycleGAN, CyCADA, PixelDA 등)

**기존 방식의 한계 및 차별점:**
위의 모든 방법론은 적응(Adaptation) 과정 중에 반드시 **레이블이 있는 소스 데이터셋이 존재한다**고 가정한다. 소스 데이터가 없다면 소스-타겟 간의 분포 거리를 측정하거나 데이터를 변환하는 것이 불가능하기 때문에, 기존의 UDA 방식들은 본 논문이 제안하는 '소스 데이터 없는 설정'에서는 작동하지 않는다. 본 논문은 데이터가 아닌 '모델' 자체를 출발점으로 삼아 이 한계를 극복한다.

## 🛠️ Methodology

### 1. 전체 시스템 구조
제안된 프레임워크는 크게 세 가지 구성 요소로 이루어진다.
- **Predictor ($C$):** 소스 도메인에서 사전 학습된 모델이며, 타겟 도메인에 맞게 업데이트된다.
- **Generator ($G$):** 랜덤 노이즈 $z$와 클래스 레이블 $y$를 입력받아 타겟 스타일의 이미지 $x_g$를 생성한다.
- **Discriminator ($D$):** 생성된 이미지 $x_g$와 실제 타겟 이미지 $x_t$를 구분하여 $G$가 타겟 분포를 잘 따르도록 유도한다.

### 2. 3C-GAN (Collaborative Class Conditional GAN)
소스 데이터 없이 타겟 스타일의 학습 데이터를 확보하기 위해 다음과 같은 손실 함수를 사용한다.

**Discriminator ($D$)의 목표:**
실제 타겟 데이터 $x_t$는 1로, 생성 데이터 $G(y, z)$는 0으로 판별하도록 학습한다.
$$\max_{\theta_D} E_{x_t \sim D_t}[\log D(x_t)] + E_{y,z}[\log(1-D(G(y,z)))]$$

**Generator ($G$)의 목표:**
$D$를 속여 실제 타겟 데이터처럼 보이게 하는 Adversarial Loss $\ell_{adv}$와, 생성된 이미지가 입력 레이블 $y$의 의미를 유지하도록 하는 Semantic Similarity Loss $\ell_{sem}$를 동시에 최소화한다.
$$\ell_{adv}(G) = E_{y,z}[\log D(1-G(y,z))]$$
$$\ell_{sem}(G) = E_{y,z}[-y \log p_{\theta_C}(G(y,z))]$$
최종 목적 함수: $\min_{\theta_G} \ell_{adv} + \lambda_s \ell_{sem}$ (여기서 $p_{\theta_C}(\cdot)$는 예측 모델 $C$의 출력값이다.)

### 3. Prediction Model ($C$)의 최적화 및 정규화
예측 모델 $C$는 생성된 데이터 $\{x_g, y\}$를 통해 학습되며, 다음의 세 가지 손실 함수의 합으로 최적화된다.
$$\min_{\theta_C} \lambda_g \ell_{gen} + \lambda_w \ell_{wReg} + \lambda_{clu} \ell_{cluReg}$$

- **$\ell_{gen}$ (Generation Loss):** 생성된 데이터를 이용한 표준 교차 엔트로피 손실이다.
- **$\ell_{wReg}$ (Weight Regularization):** 적응된 모델의 파라미터 $\theta_C$가 사전 학습된 소스 모델의 파라미터 $\theta_{C_s}$에서 너무 멀어지지 않도록 제약하여 학습을 안정화하고 소스 지식을 보존한다.
$$\ell_{wReg} = \|\theta_C - \theta_{C_s}\|^2$$
- **$\ell_{cluReg}$ (Clustering-based Regularization):** 타겟 데이터의 밀도가 높은 지역에 결정 경계가 위치하지 않도록 조건부 엔트로피(Conditional Entropy)를 최소화한다. 이때, 모델의 지역적 매끄러움(Local Smoothness)을 보장하기 위해 가상 적대적 섭동(Adversarial Perturbation) $\tilde{r}$을 도입한 KL-Divergence 항을 추가한다.
$$\ell_{cluReg} = E_{x_t \sim D_t}[-p_{\theta_C}(x_t) \log p_{\theta_C}(x_t)] + [KL(p_{\theta_C}(x_t)||p_{\theta_C}(x_t + \tilde{r}))]$$

### 4. 학습 절차
1. $D$와 $G$를 교대로 업데이트하여 타겟 스타일의 유효한 샘플 $x_g$를 생성할 수 있게 한다.
2. 생성된 $x_g$와 타겟 데이터 $x_t$를 사용하여 $C$를 업데이트한다.
3. 개선된 $C$는 다시 $G$의 $\ell_{sem}$ 계산에 반영되어 더 정확한 가이드를 제공한다. 이 과정이 반복되며 $C$와 $G$가 서로를 강화한다.

## 📊 Results

### 1. 실험 설정
- **데이터셋:** Digit(MNIST, USPS, MNIST-M, SVHN, Syn.Digits), Sign(Syn.Signs, GTSRB), Office-31, VisDA17.
- **백본 네트워크:** Digit/Sign의 경우 소형 CNN, Office-31과 VisDA17의 경우 ResNet50/101을 사용하였다.
- **평가 지표:** Classification Accuracy (%)를 측정하였으며, 모든 실험에서 적응 과정 중 소스 데이터는 사용하지 않았다.

### 2. 주요 결과
- **Digit & Sign 데이터셋:** Source-Only(Baseline) 대비 성능이 비약적으로 향상되었다. 특히 MNIST $\rightarrow$ MNIST-M 작업에서 정확도가 약 40% 향상된 98.5%를 기록하였다. 소스 데이터를 사용하는 기존 UDA 방법론들과 비교해서도 대등하거나 더 높은 성능을 보였다.
- **Office-31:** 6가지 적응 작업의 평균 정확도에서 기존 SOTA 방법론(MADA, GenToAdapt 등)보다 약 3~4% 높은 성능을 보였으며, 특히 어려운 작업(A $\leftrightarrow$ D, A $\leftrightarrow$ W)에서 강세를 보였다.
- **VisDA17:** 매우 큰 규모의 소스-타겟 격차가 있는 데이터셋임에도 불구하고 81.6% $\sim$ 83.3%의 정확도를 달성하여, 소스 데이터를 사용하는 SimDA나 Self-Ensembling보다 우수한 성능을 기록하였다.

### 3. 절제 연구 (Ablation Study)
- $\ell_{gen}$이 없으면 모델이 수렴하지 않으며, 이는 생성 데이터가 적응 과정의 핵심임을 시사한다.
- $\ell_{wReg}$는 학습 곡선을 안정화하며, $\ell_{cluReg}$는 결정 경계를 최적화하여 최종 정확도를 1~3% 추가로 향상시킨다.
- Smoothness 제약(KL-Divergence 항)을 제거했을 때 성능이 하락하는 것을 통해, 조건부 엔트로피 추정의 정확도가 중요함을 확인하였다.

## 🧠 Insights & Discussion

본 연구의 가장 큰 강점은 **데이터 프라이버시 문제를 정면으로 해결하면서도, 모델 수준의 적응만으로 데이터 기반 UDA에 근접하거나 이를 능가하는 성능을 냈다는 점**이다.

특히 **협력적 루프(Collaborative Loop)** 구조가 인상적이다. 초기에는 $C$가 불완전하여 $G$가 생성하는 이미지의 세만틱 퀄리티가 낮지만, $G$가 생성한 타겟 스타일 데이터로 $C$를 학습시키면 $C$가 개선되고, 다시 개선된 $C$가 $G$에게 더 정확한 레이블 가이드를 제공함으로써 상호 진화하는 구조를 갖는다.

다만, 본 논문은 소스 모델 $C$가 이미 어느 정도 수준의 성능을 갖추고 있다는 전제하에 시작한다. 만약 사전 학습된 모델의 성능이 매우 낮거나 타겟 도메인과의 괴리가 극심하여 초기 생성 단계에서 유의미한 샘플을 만들지 못할 경우, 협력적 루프가 제대로 작동하지 않을 가능성이 있다. 또한, 생성 모델(GAN) 특유의 학습 불안정성 문제가 존재할 수 있으나, 본 논문에서는 Spectral Normalization과 Weight Regularization을 통해 이를 완화하였다.

## 📌 TL;DR

본 논문은 소스 데이터 없이 **사전 학습된 모델과 레이블 없는 타겟 데이터만으로 도메인 적응을 수행하는 Unsupervised Model Adaptation** 방법을 제안한다. 핵심은 **3C-GAN**을 통해 타겟 스타일의 가상 데이터를 생성하고, 이를 이용해 예측 모델을 학습시키는 동시에 모델과 생성기가 서로를 강화하는 협력적 구조를 구축한 것이다. 실험 결과, 소스 데이터를 사용하는 기존 UDA 방식보다 우수하거나 대등한 성능을 보였으며, 특히 데이터 보안이 중요한 실제 환경에서 매우 높은 활용 가능성을 가진다.