# Are We Using Autoencoders in a Wrong Way?

Gabriele Martino, Davide Moroni, Massimo Martinelli (2023)

## 🧩 Problem to Solve

본 논문은 Autoencoder(AE)의 표준 학습 방식이 가진 한계를 지적하고 이를 개선하는 것을 목표로 한다. 일반적으로 Autoencoder는 입력 데이터를 그대로 복원(reconstruct)하도록 학습되며, 이 과정에서 병목(bottleneck) 구조를 통해 정보를 압축하여 Latent Space(LS)를 생성한다. 그러나 이러한 표준 방식은 Latent Space를 불규칙하게 만들어, 단순히 저차원 표현을 얻는 것 이상의 작업(예: 새로운 데이터 생성 또는 정교한 특징 추출)을 수행할 때 해석 가능성이 떨어지고 성능이 낮아지는 문제가 발생한다.

기존의 Variational Autoencoder(VAE) 등은 Latent Space에 명시적인 정규화 항(regularization term)을 추가하여 이 문제를 해결하려 했으나, 저자들은 정규화 항을 추가하는 대신 학습의 목적 함수(objective) 자체를 변경함으로써 Latent Space의 구조를 최적화하고 분류 성능을 높이는 방법을 탐구하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **"입력값과 동일한 값을 복원하는 대신, 동일한 분포에서 샘플링된 다른 값을 복원하게 함으로써 Latent Space를 변형하는 것"**이다.

1. **In-Class distribution Random Sampling Training (ICRST)**: 입력 데이터 $x$가 주어졌을 때, 동일한 클래스 분포에서 무작위로 샘플링된 다른 데이터 $y$를 복원하도록 학습한다. 이를 통해 각 클래스의 데이터들이 Latent Space 상에서 서로 응집되도록 유도한다.
2. **Total Random Sampling Training (TRST)**: 클래스 정보가 없는 완전 비지도 학습 환경에서, 전체 데이터셋에서 무작위로 선택된 샘플을 복원하도록 학습하여 데이터의 자연스러운 재배치를 유도한다.
3. **Manifold 조작 능력 입증**: Autoencoder가 단순히 데이터를 압축하는 것을 넘어, 매니폴드(manifold)를 수축(shrink)시키거나 재배치하여 특징 추출 성능을 극대화할 수 있음을 이론적 및 실험적으로 증명하였다.

## 📎 Related Works

본 논문은 Autoencoder의 다양한 변형 모델들과의 차별점을 설명한다.

- **Denoising AE (DAE) 및 Contractive AE (CAE)**: DAE는 노이즈가 섞인 입력에서 원본을 복원하며, CAE는 입력의 미세한 변화가 Latent Space의 미세한 변화로 이어지도록 정규화 항을 사용한다. 본 연구는 이러한 정규화 항 없이 샘플링 방식만으로 유사한 효과를 내고자 한다.
- **Variational AE (VAE)**: Latent Space가 특정 분포(주로 가우시안)를 따르도록 강제하여 정규성을 확보한다. 반면 본 연구는 분포를 강제하는 것이 아니라 복원 대상을 변경하여 자연스럽게 구조를 최적화한다.
- **Manifold Learning (Isomap, LLE 등)**: 기존의 비선형 차원 축소(NLDR) 기법들은 매니폴드의 위상(topology)이나 쌍별 거리(pairwise distance)를 보존하는 데 집중한다. 그러나 저자들은 클래스 간의 서브 매니폴드가 얽혀 있는 경우, 위상을 보존하는 것보다 오히려 이를 "수축"시켜 분리도를 높이는 것이 분류 작업에 더 유리하다고 주장한다.

## 🛠️ Methodology

### 1. 기본 Undercomplete Autoencoder

표준 AE는 인코더 $g(x, \theta)$와 디코더 $f(z, \phi)$를 통해 입력 $x$를 복원하며, 평균 제곱 오차(MSE)를 최소화한다.
$$\text{MSE}_{AE} = E[(x - f(g(x, \theta), \phi))^2]$$
이 최적화의 결과는 모델이 항등 함수(identity function)를 학습하여 $f(g(x)) = x$가 되는 것이다.

### 2. In-Class distribution Random Sampling Training (ICRST)

저자들은 입력 $x$와 복원 대상 $y$가 모두 동일한 클래스 $j$의 확률 분포 $p_j(x)$에서 독립적으로 샘플링되었다고 가정한다. 손실 함수는 다음과 같이 정의된다.
$$L(x, \theta, \phi) = \arg \min_{\theta, \phi} E_{j \in [1, \dots, M]} \left[ E_{y \sim p_j(x), x \sim p_j(x)} [(y - f(g(x, \theta), \phi))^2] \right]$$

이 방식에서 모델은 특정 샘플 $x$를 통해 동일 클래스의 임의의 샘플 $y$를 예측해야 하므로, 최적의 전략은 해당 클래스의 기대값(평균) $\mu_j$를 출력하는 것이 된다.
$$E_{x \sim p_j(x)} [f(g(x, \theta), \phi)] = \mu_j \quad \text{as } L \to 0$$
결과적으로 이는 Latent Space 상에서 동일 클래스 샘플들을 하나의 중심점으로 수축시키는 효과를 가져온다.

### 3. Total Random Sampling Training (TRST)

클래스 정보가 없을 때 사용하는 방법으로, 전체 데이터 분포 $p(x)$에서 무작위로 $x, y$를 추출하여 복원한다.
$$L(x, \theta, \phi) = \arg \min_{\theta, \phi} E_{y \sim p(x), x \sim p(x)} [(y - f(g(x, \theta), \phi))^2]$$
이는 모델이 데이터 간의 유사성을 스스로 파악하여 Latent Space를 자연스럽게 재배치하도록 유도하는 방식이다.

### 4. Manifold 관점의 해석

저자들은 ICRST를 DAE의 극단적인 케이스로 해석한다. DAE가 가우시안 노이즈를 제거하여 매니폴드로 되돌리는 것이라면, ICRST는 '다른 샘플'이라는 거대한 노이즈를 제거하여 매니폴드를 클래스 평균값으로 수축시킨다. 이를 통해 데이터들 사이의 Diffeomorphism(미분동형사상)이 생성되어 더 분리 가능한 특징 벡터를 추출할 수 있게 된다.

## 📊 Results

### 실험 설정

- **데이터셋**: MNIST, Fashion-MNIST, CIFAR-10, Caltech101, BreastCancer.
- **비교 방법**: Standard AE $\leftrightarrow$ ICRST (하이퍼파라미터 $p$를 통해 표준 방식과 ICRST 방식을 확률적으로 혼합하여 실험).
- **평가 지표**: Latent Space에서 추출한 특징을 이용하여 SVM, Random Forest, MLP, Gaussian Naive Bayes 분류기로 Accuracy 및 F1-Score 측정.

### 주요 결과

1. **분류 성능 향상**: ICRST 방식은 모든 데이터셋에서 표준 AE보다 월등한 성능을 보였다. 특히 MLP 분류기 기준 MNIST의 경우 $0.85 \to 0.97$로, CIFAR-10은 $0.19 \to 0.41$로 정확도가 크게 상승하였다.
2. **Latent Space 시각화**: t-SNE 투영 결과, $p=1.0$(완전한 ICRST)일 때 명시적인 정규화 항 없이도 클래스별 군집화가 매우 뚜렷하게 나타났다.
3. **TRST 분석**:
   - MNIST, Fashion-MNIST와 같이 클래스 내 유사성이 높은 데이터에서는 TRST가 Standard AE보다 높은 Mutual Information(MI)을 보이며 성능이 향상되었다.
   - 하지만 CIFAR-10, Caltech101과 같이 복잡한 데이터셋에서는 TRST의 효과가 미미하거나 오히려 성능이 하락하였다. 이는 복잡한 데이터의 경우 단순한 랜덤 샘플링만으로는 유의미한 매니폴드 재배치가 어렵기 때문으로 분석된다.

## 🧠 Insights & Discussion

본 논문은 Autoencoder를 학습시킬 때 '무엇을 복원하느냐'가 Latent Space의 기하학적 구조를 결정짓는 핵심 요소임을 보여준다.

**강점**:

- 매우 단순한 샘플링 방식의 변경만으로 복잡한 정규화 항(VAE의 KL-divergence 등) 없이도 Latent Space의 정규화 및 분리 가능성을 높였다.
- 특징 추출기로서의 AE 성능을 비약적으로 향상시켜 후속 분류 작업의 효율성을 증명하였다.

**한계 및 논의**:

- TRST의 경우 데이터의 복잡도에 따라 결과가 극명하게 갈린다. 이는 데이터의 내재적 특성(inductive bias)이 부족할 때 단순 랜덤 샘플링이 오히려 혼란을 줄 수 있음을 시사한다.
- 저자들은 복잡한 데이터셋에서 TRST의 성능을 높이기 위해 더 강력한 Bottleneck 구조가 필요할 것이라고 언급하였다.

**비판적 해석**:
ICRST는 사실상 준지도 학습(Semi-supervised learning)의 성격을 띠고 있다. 클래스 정보를 샘플링 단계에서 활용하기 때문에, 완전 비지도 학습이라기보다 클래스 레이블을 이용한 정규화 기법에 가깝다. 하지만 Loss function 자체를 수정하지 않고 샘플링 전략만으로 이를 구현했다는 점에서 실용적인 가치가 높다.

## 📌 TL;DR

본 연구는 입력 데이터를 그대로 복원하는 기존 Autoencoder 방식에서 벗어나, **동일 클래스의 다른 샘플을 복원(ICRST)**하거나 **전체 데이터에서 무작위 샘플을 복원(TRST)**하는 새로운 학습 프레임워크를 제안한다. 이를 통해 Latent Space의 매니폴드를 의도적으로 수축시켜 클래스 간 분리도를 높였으며, 결과적으로 다양한 데이터셋에서 특징 추출 및 분류 성능을 크게 향상시켰다. 이 연구는 향후 비지도 도메인 적응(Unsupervised Domain Adaptation) 및 해석 가능한 표현 학습 연구에 중요한 기초 아이디어를 제공할 가능성이 크다.
