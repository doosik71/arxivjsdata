# Weakly Augmented Variational Autoencoder in Time Series Anomaly Detection

Zhangkai Wu, Longbing Cao, Qi Zhang, Junxian Zhou, Hui Chen (2024)

## 🧩 Problem to Solve

본 논문은 시계열 이상치 탐지(Time Series Anomaly Detection, TSAD)에서 변분 오토인코더(Variational Autoencoder, VAE) 기반 모델들이 겪는 **Latent Holes** 문제를 해결하고자 한다.

**1. 해결하고자 하는 문제**
VAE 기반의 TSAD 모델은 데이터의 우도(Likelihood)를 추정하여 정상 데이터를 학습하고, 이를 통해 이상치를 탐지한다. 그러나 실제 환경에서는 데이터 부족(Data Scarcity) 문제가 빈번하며, 특히 이상치 데이터가 잠재 공간(Latent Space)에 매핑될 때 정상 데이터의 시공간적 특성이 부족하여 잠재 공간 내에 불연속적인 영역인 Latent Holes가 발생한다.

**2. 문제의 중요성**
Latent Holes가 존재하면 잠재 공간의 연속성과 매끄러움(Smoothness)이 깨지게 된다. 이로 인해 모델이 잠재 공간에서 샘플링을 수행할 때, 입력 데이터와 재구성된 데이터 간의 불일치가 발생하며, 결과적으로 정상 데이터를 이상치로 오판하거나 그 반대의 경우가 발생하는 등 재구성의 강건성(Robustness)이 현저히 떨어진다.

**3. 논문의 목표**
자기지도학습(Self-Supervised Learning, SSL)과 데이터 증강(Data Augmentation)을 VAE 프레임워크에 통합하여 잠재 공간의 불연속성을 완화하고, 데이터 우도 추정 능력을 향상시켜 더욱 강건한 이상치 탐지 성능을 달성하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **입력 데이터의 약한 증강(Weak Augmentation)을 통해 잠재 표현을 풍부하게 만들고, 원본 모델과 증강 모델 간의 상호 정보량(Mutual Information, MI)을 최대화하여 잠재 공간의 밀도를 높이는 것**이다.

- **WAVAE(Weakly Augmented Variational Autoencoder) 제안**: 원본 데이터와 증강된 데이터가 동일한 파라미터를 공유하는 VAE 구조를 설계하여, 두 관점의 데이터가 동일한 분포를 학습하도록 유도한다.
- **상호 정보량 최대화 전략**: 원본 잠재 변수 $z^r$과 증강된 잠재 변수 $z^a$ 사이의 상호 정보량 $I(z^r, z^a)$를 최대화함으로써 잠재 공간의 Latent Holes를 메우고 표현의 강건성을 높인다.
- **두 가지 학습 접근법**: 상호 정보량 근사를 위해 얕은 학습(Shallow Learning) 기반의 **Contrastive Learning(InfoNCE)** 방식과 깊은 학습(Deep Learning) 기반의 **Adversarial Learning** 방식을 모두 구현하여 비교 분석하였다.
- **Weak Augmentation의 효용성 증명**: 강한 증강보다는 단순한 정규화(Normalization) 수준의 약한 증강이 시계열의 핵심 특성을 보존하면서 우도 적합도를 높이는 데 더 효과적임을 밝혔다.

## 📎 Related Works

**1. 관련 연구 설명 및 한계**

- **비확률적 생성 모델(AE, GAN)**: 오토인코더(AE)나 GAN 기반 모델들은 데이터를 강건하게 재구성하는 데 집중하지만, 데이터의 확률 분포를 명시적으로 모델링하지 않아 해석력이 부족하다.
- **확률적 생성 모델(VAE)**: VAE는 잠재 공간의 연속적인 표현을 학습하여 우도를 명시적으로 모델링할 수 있다는 장점이 있다. 최근에는 VRNN이나 메타-프라이어(Meta-priors)를 도입하여 시공간적 의존성을 캡처하려는 시도가 있었다.
- **한계점**: 기존 VAE 기반 방법들은 모델 구조 개선에 집중했을 뿐, 데이터 부족으로 인한 잠재 공간의 불연속성(Latent Holes) 문제나 데이터 활용 효율성 문제를 충분히 다루지 않았다.

**2. 기존 접근 방식과의 차별점**
기존 연구들이 주로 네트워크 아키텍처(예: RNN, Transformer)나 Prior 분포의 수정에 집중한 반면, 본 논문은 **SSL 기반의 데이터 증강과 두 표현 간의 정렬(Alignment)**이라는 학습 전략을 통해 잠재 공간 자체의 기하학적 구조를 개선했다는 점에서 차별성을 갖는다.

## 🛠️ Methodology

### 전체 파이프라인 및 시스템 구조

WAVAE는 원본 입력 $x^r$과 증강된 입력 $x^a$를 동시에 처리하는 듀얼 뷰(Dual-view) 구조를 가진다. 두 경로의 인코더($q_\phi$)와 디코더($p_\theta$)는 **파라미터를 공유**한다.

1. **입력 단계**: 원본 데이터 $x^r$에 약한 증강 연산 $O$를 적용하여 $x^a = O(x^r)$를 생성한다.
2. **인코딩 단계**: 공유된 인코더를 통해 각각 잠재 변수 $z^r \sim q_\phi(z^r|x^r)$와 $z^a \sim q_\phi(z^a|x^a)$를 추출한다.
3. **디코딩 단계**: 공유된 디코더를 통해 각각 $\hat{x}^r$과 $\hat{x}^a$로 재구성한다.
4. **정렬 단계**: $z^r$과 $z^a$ 사이의 상호 정보량을 최대화하는 손실 함수를 추가하여 두 표현을 일치시킨다.

### 훈련 목표 및 손실 함수

모델의 전체 최적화 목표는 다음과 같은 통합 손실 함수 $L_{AVAE}$를 최소화하는 것이다.

$$L_{AVAE} = L^r_{ELBO} + L^a_{ELBO} + I(z^r, z^a)$$

여기서 $L_{ELBO}$는 VAE의 표준 목적 함수로, 재구성 손실($L_R$)과 KL 발산 손실($L_I$)의 합으로 정의된다.

$$L_{ELBO} := \mathbb{E}_{q_\phi(z|x)} [\log p_\theta(x|z)] - \beta D_{KL}(q_\phi(z|x) || p(z))$$

### 상호 정보량(MI) 근사 방법

$I(z^r, z^a)$를 직접 계산하는 것은 불가능하므로, 논문은 두 가지 근사 방법을 제시한다.

**1. Shallow Learning (Contrastive Learning)**
$\text{InfoNCE}$ 손실 함수를 사용하여 $z^r$과 $z^a$가 서로 가깝게, 그리고 다른 샘플의 잠재 변수와는 멀게 학습시킨다.
$$L_{InfoNCE} = -\log \frac{\exp(z^r \cdot z^a / \tau)}{\sum \exp(z^r \cdot z^k / \tau)}$$

**2. Deep Learning (Adversarial Learning)**
판별자(Discriminator) $\Psi$를 도입하여 $(z^r, z^a)$ 쌍이 실제 쌍인지 가짜 쌍인지 구분하게 하며, 생성기(VAE)는 판별자를 속이도록 학습하여 두 분포를 일치시킨다.
$$L_{adversarial} \approx \log \frac{\Psi(z^r)}{1-\Psi(z^r)} + \log \frac{\Psi_a(z^a)}{1-\Psi_a(z^a)}$$

### 추론 및 이상치 탐지 절차

학습이 완료되면 증강 모델은 제거하고 원본 모델만을 사용한다.

- **이상치 점수(Anomaly Score)**: 원본 데이터 $x^r$과 재구성된 데이터 $\hat{x}^r$ 사이의 제곱 오차 합(SSE)을 계산한다.
- **판정**: 계산된 점수가 미리 설정된 임계값 $\eta$(예: 오차 분포의 99번째 백분위수)보다 크면 이상치로 판정한다.

## 📊 Results

### 실험 설정

- **데이터셋**: GD(로봇 신호), HSS(에너지 최적화 센서), ECG(심전도), TD(손 뼈대 궤적), Yahoo S5(서비스 메트릭) 총 5종의 데이터셋 사용.
- **비교 대상**: TAE, BGAN, RAE, CAE 등 6종의 일반 생성 모델 및 GMMVAE, VRAE, VQRAE 등 10종의 확률적 생성 모델(총 16종).
- **측정 지표**: AUROC (Area Under the ROC Curve) 및 PRAUC (Area Under the Precision-Recall Curve).

### 주요 결과

- **정량적 결과**: 모든 데이터셋에서 WAVAE가 기존 SOTA 모델들을 능가하는 성능을 보였다. 특히 $\text{WAVAE-Contrast}$(Contrastive Learning 적용) 모델이 $\text{WAVAE-Adversarial}$보다 전반적으로 더 높은 성능을 기록했다.
- **지표 수치**: 예를 들어 Yahoo S5 데이터셋에서 $\text{WAVAE-Contrast}$는 매우 높은 AUROC와 PRAUC 점수를 기록하며 압도적인 성능 향상을 보였다.
- **정성적 분석**: Ablation Study를 통해 잠재 변수의 차원 $z$가 14~20 사이일 때 최적의 성능이 나오며, MSE 손실 함수가 가장 효과적임을 확인하였다.

## 🧠 Insights & Discussion

**1. 강점**
본 연구는 VAE의 고질적인 문제인 Latent Holes를 데이터 증강과 MI 최대화라는 관점에서 접근하여 해결했다. 특히 복잡한 증강 기법이 아니라 단순한 정규화(Min-Max, Standardization)만으로도 잠재 공간의 밀도를 높여 강건성을 확보할 수 있음을 보인 점이 실용적이다.

**2. 한계 및 논의사항**

- **하이퍼파라미터 민감도**: 잠재 공간의 차원 $z$나 판별자의 레이어 수에 따라 성능 차이가 발생하므로, 새로운 데이터셋에 적용할 때 최적의 하이퍼파라미터를 찾는 과정이 필수적이다.
- **증강 기법의 선택**: 논문에서는 강한 증강(Strong Augmentation)이 오히려 우도 함수 생성에 방해가 된다고 언급했다. 이는 TSAD 작업에서 데이터의 원형 보존이 매우 중요하다는 것을 시사하며, 어떤 수준의 증강이 '적절한' 것인지에 대한 이론적 기준은 명확히 제시되지 않았다.

**3. 비판적 해석**
Contrastive Learning 방식이 Adversarial 방식보다 우수한 성능을 보인 이유는, 시계열 데이터의 특성상 판별자를 통한 분포 일치보다 샘플 간의 상대적 거리(Similarity)를 학습하는 것이 잠재 공간의 연속성을 확보하는 데 더 안정적이기 때문으로 분석된다.

## 📌 TL;DR

본 논문은 VAE 기반 시계열 이상치 탐지에서 데이터 부족으로 발생하는 **Latent Holes(잠재 공간의 불연속성)** 문제를 해결하기 위해, 약한 증강(Weak Augmentation)과 자기지도학습(SSL)을 결합한 **WAVAE**를 제안한다. 원본과 증강된 데이터의 잠재 표현 간 상호 정보량을 최대화함으로써 잠재 공간을 더욱 촘촘하고 강건하게 만들었으며, 실험을 통해 기존 SOTA 모델들보다 뛰어난 AUROC 및 PRAUC 성능을 입증하였다. 이 연구는 향후 데이터가 부족한 환경에서의 비지도 학습 기반 이상치 탐지 연구에 중요한 방법론적 기여를 할 것으로 기대된다.
