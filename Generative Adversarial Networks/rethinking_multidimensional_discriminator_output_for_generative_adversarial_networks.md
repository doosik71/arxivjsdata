# Rethinking Multidimensional Discriminator Output for Generative Adversarial Networks

Mengyu Dai, Haibin Hang, Anuj Srivastava (2022)

## 🧩 Problem to Solve

본 논문은 Generative Adversarial Networks(GAN)에서 판별자(Discriminator 또는 Critic)의 출력 차원이 일반적으로 1차원(스칼라 값)으로 제한되어 있다는 점에 주목한다. 기존의 Wasserstein GAN(WGAN) 프레임워크는 판별자가 실데이터와 생성 데이터의 분포를 1차원 공간으로 투영(Push-forward)한 뒤 그 차이를 측정한다.

그러나 이러한 1차원 투영 방식은 두 분포 사이의 중요한 차이점을 소실시킬 위험이 있으며, 판별자가 최적의 1차원 투영 함수를 찾아내야 한다는 점에서 학습의 부담이 크다. 따라서 본 연구의 목표는 판별자의 출력 차원을 다차원으로 확장하여 분포 간의 구별 능력을 높이고, 이를 통해 학습 속도 향상과 생성 결과물의 다양성을 확보하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 판별자의 다차원 출력을 이론적으로 정당화하고, 이를 효과적으로 활용하기 위한 구조적 장치를 제안한 것에 있다.

첫째, **maximal $p$-centrality discrepancy**라는 새로운 거리 척도를 제안하였다. 이는 $p$-Wasserstein 거리와 밀접한 관련이 있으며, 판별자의 출력 차원 $n$이 증가할수록 두 분포 사이의 괴리(discrepancy)가 커진다는 것을 이론적으로 증명하였다. 이를 통해 다차원 출력의 판별자가 더 풍부한 정보를 제공할 수 있음을 보였다.

둘째, **Square-Root Velocity Transformation (SRVT)** 블록을 도입하였다. 다차원 출력층이 단순히 Fully Connected(FC) 층으로 구성될 경우 발생하는 순열 대칭성(Permutation Symmetry) 문제를 해결하여, 각 출력 뉴런이 서로 다른 고유한 특징을 학습하도록 강제함으로써 다차원 출력의 이점을 극대화하였다.

## 📎 Related Works

본 논문은 다음과 같은 기존 연구들의 한계와 차별점을 논의한다.

- **Wasserstein 거리 기반 GAN**: WGAN은 1-Wasserstein 거리를 사용하여 학습 안정성을 높였으나, 여전히 Lipschitz 조건을 강제하기 위한 Weight Clipping이나 Gradient Penalty 등에 의존한다. Sliced Wasserstein Distance(SWD) 등의 연구가 무작위 방향으로의 투영을 통해 거리를 추정하려 했으나, 본 논문은 투영 공간 자체를 고차원으로 확장하여 접근한다.
- **Moment Matching 기반 GAN**: MMD GAN 등은 모멘트 매칭을 통해 분포를 구분한다. 본 논문이 제안하는 $p$-centrality 함수는 분포의 $p$차 모멘트의 $p$제곱근에 해당하므로, 제안 방법론은 특정 $p$차 모멘트를 매칭하려는 시도로 해석될 수 있다.
- **네트워크의 대칭성(Symmetry)**: 딥러닝 네트워크의 가중치 공간에 존재하는 대칭성은 중복성을 야기하고 일반화 능력을 저해한다. 기존에는 Random Initialization이나 Skip Connection으로 이를 해결하려 했으나, 본 논문은 출력층의 구조적 비대칭성을 부여하는 SRVT를 통해 이 문제를 해결한다.

## 🛠️ Methodology

### 1. 목적 함수 (Objective Function)

제안하는 GAN의 목적 함수는 다음과 같이 정의된다.

$$\min_{G} \max_{D} \left( \mathbb{E}_{x \sim P_r} [\|D(x)\|^p] \right)^{1/p} - \left( \mathbb{E}_{z \sim p(z)} [\|D(G(z))\|^p] \right)^{1/p}$$

여기서 $D$는 판별자로, 출력값은 $\mathbb{R}^n$ 공간의 $n$차원 벡터이다. $\|\cdot\|$는 $L_2$ 노름을 의미하며, $p$는 모멘트의 차수를 결정하는 하이퍼파라미터이다.

### 2. $p$-centrality 함수 및 거리 척도

$p$-centrality 함수 $\sigma_{P,p}(x)$는 거리 공간 $(M, d)$ 위의 확률 분포 $P$에 대해 다음과 같이 정의된다.

$$\sigma_{P,p}(x) := \left( \int_M d^p(x, y) dP(y) \right)^{1/p} = \left( \mathbb{E}_{y \sim P} [d^p(x, y)] \right)^{1/p}$$

본 논문은 이를 확장하여 **maximal $p$-centrality discrepancy** $\mathcal{L}_{p,n,K}$를 정의한다. 이는 $K$-Lipschitz 함수 $f: M \to \mathbb{R}^n$에 대해 다음의 값을 최대화하는 문제이다.

$$\mathcal{L}_{p,n,K}(P, Q) = \sup_{f \in \text{Lip}(K)} \left( \int \|f\|^p dP \right)^{1/p} - \left( \int \|f\|^p dQ \right)^{1/p}$$

이론적 분석 결과, 출력 차원 $n$이 커질수록 $\mathcal{L}_{p,n,K}$ 값은 증가하며($n < n' \implies \mathcal{L}_{p,n,K} \leq \mathcal{L}_{p,n',K}$), 이는 판별자가 실데이터와 가짜 데이터의 차이를 더 쉽게 포착할 수 있음을 의미한다.

### 3. Square Root Velocity Transformation (SRVT)

판별자의 출력층이 FC 층일 경우, 뉴런의 순서가 바뀌어도 결과가 동일한 순열 대칭성이 발생한다. 이를 깨기 위해 본 논문은 다음과 같은 비대칭 변환인 SRVT를 제안한다.

출력 벡터 $(x_1, x_2, \dots, x_n)$이 주어졌을 때, 변환된 벡터 $(y_1, y_2, \dots, y_n)$의 각 요소는 다음과 같다.

$$y_i = \text{sgn}(x_i - x_{i-1}) \sqrt{|x_i - x_{i-1}|}, \quad i=1, 2, \dots, n$$
(단, $x_0 = 0$으로 가정)

이 변환은 각 뉴런의 위치에 따른 구조적 고유성을 부여한다. 결과적으로 각 출력 뉴런은 서로 다른 기능을 수행하게 되어, 입력 분포의 서로 다른 특징들을 반영하도록 강제된다. 또한 제곱근 연산은 미분값의 크기를 집중시켜 특정 차원에만 의존하지 않고 균형 있게 특징을 학습하도록 돕는 평활화(smoothing) 효과를 제공한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: CIFAR-10, CIFAR-100, ImageNet-1K, STL-10, LSUN bedroom.
- **평가 지표**: FID(Frechet Inception Distance), KID(Kernel Inception Distance), Precision, Recall.
- **백본 네트워크**: StyleGAN2, ResNet, BigGAN.
- **구현 세부사항**: $n$의 값은 1, 16, 128, 1024 등으로 설정하여 비교하였으며, $p=1$을 기본값으로 사용하였다.

### 2. 주요 결과

- **출력 차원 $n$의 영향**: CIFAR-10 데이터셋에서 StyleGAN2를 사용한 실험 결과, $n$이 클수록(특히 $n=1024$) FID 수렴 속도가 빠르고 최종 성능이 우수함을 확인하였다.
- **SRVT의 효과**: ResNet 백본 실험에서 SRVT를 적용했을 때, 모든 $n$ 설정에서 성능이 향상되었으며, 특히 고차원 출력($n=1024$)에서 그 효과가 극대화되었다.
- **정량적 비교**:
  - CIFAR-10, STL-10, LSUN 데이터셋에서 제안 방법은 WGAN-GP, SNGAN, Sphere GAN 등 기존 방법론보다 낮은 FID를 기록하며 우수한 성능을 보였다. (예: CIFAR-10 ResNet 기반 FID 8.5 달성)
  - ImageNet-1K 대규모 실험에서도 $n=1024$ 설정이 $n=1$보다 더 낮은 FID와 더 높은 Recall(다양성)을 보였다.
  - 조건부 생성(Conditional Generation) 태스크인 BigGAN 설정에서도 기존 Hinge loss 대비 더 나은 성능을 기록하였다.

## 🧠 Insights & Discussion

본 논문은 판별자의 출력을 단순히 '진위 판별'을 위한 스칼라 값으로 보는 시각에서 벗어나, 이를 '분포의 특징을 추출하는 고차원 지표'로 재정의하였다.

**강점**:

- 이론적으로 고차원 출력이 왜 유리한지를 $p$-centrality discrepancy를 통해 증명함으로써, 단순한 휴리스틱이 아닌 수학적 근거를 제시하였다.
- SRVT라는 간단한 구조적 변경만으로 딥러닝의 고질적인 문제인 대칭성 문제를 해결하고 다차원 출력의 효율성을 높였다.

**한계 및 논의**:

- 고차원 출력을 사용할 때 StyleGAN2의 $\mathcal{R}_1$ 정규화 계수 $\gamma$를 $n$에 따라 조정해야 한다는 점이 발견되었다. 이는 고차원 출력으로 인해 목적 함수의 스케일이 변하기 때문으로 추측되나, 최적의 $\gamma$를 찾는 명확한 가이드는 제시되지 않았다.
- $p$ 값에 대한 실험에서 $p=1$ 혹은 $p=1, 2$ 조합이 효과적이었으나, $p > 2$인 경우 성능 향상이 없었다는 점은 $p$-centrality가 특정 차수 이상의 모멘트 정보까지는 필요하지 않음을 시사한다.

## 📌 TL;DR

본 연구는 GAN 판별자의 출력 차원을 1차원에서 다차원으로 확장하여 분포 간의 구별 능력을 높이는 방법을 제안한다. 이론적으로는 maximal $p$-centrality discrepancy를 통해 고차원 출력이 더 정밀한 거리 척도가 됨을 증명하였으며, 구조적으로는 SRVT 블록을 통해 출력 뉴런 간의 대칭성을 깨뜨려 각 뉴런이 고유한 특징을 학습하도록 하였다. 실험적으로는 CIFAR, ImageNet 등 다양한 데이터셋에서 기존 WGAN 계열보다 빠른 수렴 속도와 더 높은 생성 품질 및 다양성을 입증하였다. 이 연구는 향후 고해상도 이미지 생성 및 복잡한 분포 학습을 위한 판별자 설계에 중요한 방향성을 제시한다.
