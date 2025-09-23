# ANOMALY DETECTION USING ONE-CLASS NEURAL NETWORKS

Raghavendra Chalapathy, Aditya Krishna Menon, Sanjay Chawla

## 🧩 Problem to Solve

복잡한 데이터셋에서 이상 탐지(Anomaly Detection)를 수행하는 것은 중요한 과제입니다. 기존의 지도 학습 기반 이상 탐지 기법들은 레이블링된 이상 데이터를 필요로 하지만, 실제 환경에서는 이상 데이터가 희귀하고 불균형하게 분포하며, 정의하기 어려운 경우가 많아 비지도 학습 방식이 선호됩니다. One-Class SVM (OC-SVM)과 같은 전통적인 비지도 학습 기법은 고차원 및 복잡한 데이터셋에서 성능이 저하되는 한계가 있습니다.

최근 딥러닝 기반 이상 탐지 모델들은 주로 두 가지 접근 방식을 사용합니다:

1. **오토인코더(Autoencoder) 기반:** 재구성 오류(reconstruction error)를 사용하여 이상 여부를 판단합니다.
2. **하이브리드 모델(Hybrid Models):** 오토인코더 등으로 심층 특징(deep features)을 추출한 후, 이를 OC-SVM과 같은 별도의 전통적인 이상 탐지 알고리즘에 입력으로 사용합니다.

그러나 하이브리드 접근 방식은 특징 학습(feature learning)과 이상 탐지 목적이 분리되어 있어, 은닉층(hidden layers)에서의 표현 학습(representational learning)이 이상 탐지 작업에 최적화되지 못하는 문제가 있습니다. 이로 인해 최적화되지 않은(sub-optimal) 성능을 보일 수 있습니다.

## ✨ Key Contributions

- **새로운 One-Class 신경망(OC-NN) 모델 제안:** 신경망 훈련을 위해 One-Class SVM과 유사한 손실 함수를 사용하여 이상 탐지 목적에 맞게 은닉층 특징 표현을 학습하는 OC-NN 모델을 개발했습니다.
- **교대 최소화(Alternating Minimization) 학습 알고리즘 제안:** OC-NN 모델의 파라미터를 학습하기 위한 교대 최소화 알고리즘을 제안했으며, OC-NN 목적 함수의 하위 문제가 퀀타일(quantile) 선택 문제와 동일함을 증명했습니다.
- **광범위한 실험을 통한 성능 입증:** 복잡한 이미지 및 시퀀스 데이터셋(예: CIFAR, GTSRB)에서 OC-NN이 기존의 최첨단 딥러닝 이상 탐지 방법과 동등하거나 우수한 성능을 보였고, 일부 시나리오에서는 기존의 얕은(shallow) 방법들보다 뛰어났음을 설득력 있게 보여주었습니다.

## 📎 Related Works

- **전통적인 이상 탐지:**
  - **One-Class SVM (OC-SVM):** Schölkopf et al. [2001], Tax and Duin [2004] 등이 제안한 널리 사용되는 비지도 이상 탐지 기법으로, 데이터의 대부분을 둘러싸는 경계를 구축합니다.
  - **Isolation Forest (IF):** Liu et al. [2008]
  - **Kernel Density Estimation (KDE):** Parzen [1962]
- **딥러닝 기반 이상 탐지:**
  - **오토인코더 기반:** 재구성 오류를 활용하여 이상을 탐지합니다. Robust Deep Autoencoder (RDA) 또는 Robust Deep Convolutional Autoencoder (RCAE) (Zhou and Paffenroth [2017], Chalapathy et al. [2017]), Deep Convolutional Autoencoder (DCAE) (Masci et al. [2011]).
  - **하이브리드 모델:** 딥러닝(주로 오토인코더)으로 특징을 추출하고, 이를 OC-SVM과 같은 전통적인 이상 탐지 알고리즘에 입력합니다 (Erfani et al. [2016], Sohaib et al. [2017]). 전이 학습(transfer learning)을 통해 미리 학습된 네트워크(예: ImageNet-MatConvNet-VGG-F)의 표현을 활용하기도 합니다. 이 논문은 이러한 접근 방식의 "작업에 독립적인(task-agnostic)" 특징 학습 문제를 지적합니다.
  - **최근 딥 원-클래스 모델:** Deep One-Class Classification (예: Deep SVDD, Soft-Bound Deep SVDD) (Ruff et al. [2018])은 데이터의 신경망 표현을 감싸는 하이퍼스피어(hypersphere)의 볼륨을 최소화하는 방식으로 학습합니다.
  - **GAN 기반:** AnoGAN (Radford et al. [2015]).

## 🛠️ Methodology

OC-NN은 신경망 아키텍처에 OC-SVM과 동등한 손실 함수를 통합하여 설계되었습니다.

- **OC-NN 목적 함수:**
  OC-SVM의 핵심인 내적 $\langle \mathbf{w}, \Phi(\mathbf{X}_{n:})\rangle$을 신경망의 은닉층 출력인 $\langle \mathbf{w}, g(\mathbf{V}\mathbf{X}_{n:})\rangle$로 대체합니다. 여기서 $\mathbf{w}$는 은닉층에서 출력층으로의 가중치, $\mathbf{V}$는 입력에서 은닉층으로의 가중치 행렬이며, $g(\cdot)$는 활성화 함수입니다.
  OC-NN의 목적 함수는 다음과 같습니다:
  $$ \min*{\mathbf{w}, \mathbf{V}, r} \frac{1}{2}\|\mathbf{w}\|^2_2 + \frac{1}{2}\|\mathbf{V}\|^2_F + \frac{1}{\nu} \cdot \frac{1}{N} \sum*{n=1}^{N} \max(0, r - \langle\mathbf{w}, g(\mathbf{V}\mathbf{X}\_{n:})\rangle) - r $$
    여기서 $\nu \in (0,1)$는 이상치 허용 비율을 제어하는 파라미터입니다. 이 방식의 핵심은 데이터 표현 학습이 OC-NN 목적 함수에 의해 직접적으로 유도되므로, 이상 탐지에 최적화된 맞춤형 특징을 생성한다는 점입니다.

- **모델 학습 (교대 최소화 알고리즘):**
  목적 함수는 비볼록(non-convex)이므로 교대 최소화(alternating minimization) 방식을 사용합니다.

  1. **$r$ 고정 후 $\mathbf{w}, \mathbf{V}$ 최적화:** $r$ 값을 고정한 상태에서 표준 역전파(Backpropagation, BP) 알고리즘을 사용하여 $\mathbf{w}$와 $\mathbf{V}$를 업데이트합니다 (식 4).
  2. **$\mathbf{w}, \mathbf{V}$ 고정 후 $r$ 최적화:** $\mathbf{w}$와 $\mathbf{V}$를 고정한 상태에서 $r$을 최적화합니다. 이 때, $r$의 최적값은 모든 데이터 포인트의 스코어 $\hat{y}_n = \langle\mathbf{w}, g(\mathbf{V}\mathbf{x}_n)\rangle$의 $\nu$-퀀타일(quantile)이 됩니다 (식 5). 이는 논문의 Theorem 1에서 증명되었습니다.
  3. 이 과정을 수렴할 때까지 반복합니다.
  4. 수렴 후, 각 데이터 포인트 $\mathbf{X}_{n:}$에 대한 의사결정 점수(decision score)는 $S_n = \hat{y}_n - r$로 계산됩니다. $S_n \geq 0$이면 정상, $S_n < 0$이면 이상으로 분류됩니다.

- **사전 학습된 특징 사용:** 초기에는 딥 오토인코더를 학습시켜 대표 특징을 얻습니다. 이후 이 사전 학습된 오토인코더의 인코더 층을 복사하여 OC-NN의 입력으로 사용하며, 이 인코더의 가중치들은 고정되지 않고 OC-NN 목적 함수에 따라 추가적으로 학습됩니다. 이는 특징을 이상 탐지 작업에 맞게 미세 조정하는 효과를 가져옵니다.

## 📊 Results

- **합성(Synthetic) 데이터셋:** OC-NN은 10개의 이상 데이터를 정확하게 식별했으며, 고전적인 OC-SVM과 동등한 성능을 보였습니다. 이상치들은 음의 의사결정 점수를 가졌습니다.
- **MNIST 및 CIFAR-10 데이터셋:** (AUCs %로 측정)
  - **MNIST:** RCAE가 가장 뛰어난 성능을 보였으며, OC-NN도 경쟁력 있는 성능을 달성했습니다.
  - **CIFAR-10:**
    - RCAE가 전반적으로 우수했지만, 일부 클래스에서는 OC-NN, Soft-Bound/One-Class Deep SVDD가 RCAE에 필적하는 강건한 성능을 보였습니다.
    - 특히 전역 대비(global contrast)가 낮은 클래스(예: AUTOMOBILE, BIRD, DEER)에서는 OC-NN이 얕은 방법들과 Soft/One-Class Deep SVDD를 능가하는 성능을 보였습니다.
    - 반면, FROG, TRUCK과 같이 강한 전역 구조(strong global structures)를 가진 클래스에서는 OCSVM/SVDD와 KDE 같은 얕은 방법들이 딥러닝 기반 모델보다 더 나은 성능을 보이기도 했습니다. 이는 신경망 아키텍처 선택의 중요성을 시사합니다.
- **GTSRB (독일 교통 표지판 인식 벤치마크) 데이터셋 (적대적 공격 탐지):**
  - RCAE가 모든 딥 모델 중 가장 좋은 성능을 보였습니다. DCAE가 그 다음이었으며, OC-NN도 경쟁력 있는 성능을 나타냈습니다. 이 실험에서는 잘못 잘리거나 특이한 원근법으로 찍힌 이미지가 이상치로 간주되었습니다.

## 🧠 Insights & Discussion

- **OC-NN의 핵심 강점:** OC-NN은 은닉층의 데이터 표현 학습을 이상 탐지 목적 함수에 의해 직접적으로 제어하여, 이 작업에 특화된 특징을 생성합니다. 이는 특징 추출이 '작업에 독립적인' 하이브리드 모델들과의 중요한 차별점입니다.
- **경쟁력 있는 성능:** OC-NN은 복잡한 이미지 데이터셋에서 최첨단 딥러닝 및 기존 얕은 이상 탐지 방법과 비교하여 동등하거나 우수한 성능을 보여주었습니다. 특히, 이미지의 미묘한 이상 징후를 감지하는 데 효과적인 것으로 나타났습니다.
- **한계 및 관찰:**
  - OC-NN의 목적 함수는 비볼록(non-convex)하므로, 학습 알고리즘이 전역 최적해(global optimum)를 보장하지는 않습니다.
  - 일부 데이터셋 및 클래스에서는 전통적인 얕은 방법들이 여전히 좋은 성능을 보이기도 했으며, 이는 데이터 특성에 따른 모델 선택의 중요성을 강조합니다.
  - 강건한 오토인코더(RCAE)는 여러 데이터셋에서 일관되게 강력한 성능을 보였습니다.

이 연구는 딥러닝 아키텍처 내에 이상 탐지 목적을 직접적으로 통합하는 원칙적인 방법을 제시하며, 단순한 특징 추출과 분리된 탐지 단계를 넘어선 발전을 보여줍니다.

## 📌 TL;DR

**문제:** 기존 딥러닝 기반 이상 탐지 모델은 특징 학습과 이상 탐지 목적이 분리되어 있어, 복잡한 데이터셋에서 최적화되지 않은 성능을 보일 수 있습니다.
**방법:** 본 논문은 One-Class 신경망(OC-NN)을 제안합니다. OC-NN은 One-Class SVM과 유사한 손실 함수를 딥러닝 모델의 학습에 직접 통합하여, 은닉층이 이상 탐지 작업에 특화된 특징 표현을 학습하도록 유도합니다. 이를 위해 교대 최소화 알고리즘을 사용하여 모델 파라미터를 효율적으로 업데이트합니다.
**발견:** OC-NN은 MNIST, CIFAR-10, GTSRB 등 복잡한 이미지 데이터셋에서 최첨단 딥러닝 및 기존 얕은 이상 탐지 방법들과 비교하여 동등하거나 우수한 성능을 달성했습니다. 이는 이상 탐지 작업에 맞춤화된 특징 학습의 중요성을 입증합니다.
