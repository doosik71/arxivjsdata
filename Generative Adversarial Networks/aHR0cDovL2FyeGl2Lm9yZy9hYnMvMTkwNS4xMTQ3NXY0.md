# GAT: Generative Adversarial Training for Adversarial Example Detection and Robust Classification

Xuwang Yin, Soheil Kolouri, Gustavo K. Rohde (2020)

## 🧩 Problem to Solve

딥 뉴럴 네트워크(DNN)는 다양한 도메인에서 뛰어난 성능을 보이지만, Adversarial Examples(적대적 예제)에 매우 취약하다는 치명적인 약점이 있다. 이를 해결하기 위해 적대적 예제를 탐지(Detection)하려는 시도가 많았으나, 기존의 탐지 메커니즘들은 대부분 공격자가 탐지기의 존재를 모르는 Non-adaptive threat 상황을 가정한다.

하지만 공격자가 탐지 메커니즘을 알고 이를 우회하도록 공격을 설계하는 Adaptive attack 상황에서는 기존의 탐지 방법들이 매우 취약함이 밝혀졌다. 따라서 본 논문의 목표는 Norm-constrained white-box adaptive attack 상황에서도 견고하게 작동하는 원칙적인 적대적 예제 탐지 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 입력 공간을 기존 분류기의 결정 경계(Decision Boundary)를 기준으로 하위 공간(Subspace)으로 분할하고, 각 공간 내에서 깨끗한 샘플과 적대적 샘플을 구분하는 이진 분류기를 학습시키는 것이다.

구체적으로, $K$개의 클래스 분류 문제에서 각 클래스 $i$에 대해, 클래스 $i$의 깨끗한 데이터와 다른 클래스들로부터 생성되어 클래스 $i$로 오분류된 적대적 샘플을 구분하는 $K$개의 이진 분류기를 학습시킨다. 이를 통해 탐지 성능을 높이는 동시에, 각 이진 분류기를 클래스 조건부 데이터의 unnormalized density model로 해석함으로써 생성적 관점에서의 분류 및 탐지 접근 방식을 제시한다.

## 📎 Related Works

기존의 적대적 예제 탐지 연구들은 주로 깨끗한 샘플 집합 $D$와 공격받은 샘플 집합 $D'$를 구분하는 이진 분류기를 학습시키는 방식에 의존했다. 예를 들어, 이미지 콘텐츠에서 직접 탐지하거나, 분류 네트워크의 중간 레이어 특징(Intermediate layer features)을 이용하는 방식 등이 제안되었다.

그러나 이러한 방법들은 Heuristic한 접근 방식이 많아 성능 보장이 어렵다. 특히 Carlini & Wagner(2017a)와 Athalye et al.(2018)의 연구에 따르면, 대부분의 기존 탐지기들은 Adaptive attack에 노출되었을 때 그 효과가 급격히 떨어진다는 한계가 있다. 본 논문은 이러한 한계를 극복하기 위해 Robust Optimization 프레임워크를 도입하여 적대적 학습을 통해 탐지기를 강화한다.

## 🛠️ Methodology

### 전체 파이프라인 및 구조

제안된 방법론은 크게 두 가지 접근 방식으로 나뉜다.

1. **Integrated Detection**: 기존의 $K$-class 분류기 $f$를 사용하여 먼저 예측 레이블 $\hat{k}$를 얻고, 이후 $\hat{k}$번째 이진 분류기 $d_{\hat{k}}$를 사용하여 해당 샘플이 깨끗한 샘플인지 적대적 샘플인지 판별한다.
2. **Generative Detection**: $K$개의 이진 분류기 $\{d_k\}_{k=1}^K$를 One-versus-the-rest(OVR) 분류기로 사용하여 예측 레이블 $\hat{k}$를 결정하고, 동일하게 $d_{\hat{k}}$를 통해 탐지 여부를 결정한다.

### 학습 절차 및 손실 함수

각 이진 분류기 $d_k$는 클래스 $k$의 샘플($D_k$)과 다른 클래스의 샘플들($D_{\setminus k}$)로부터 생성된 적대적 샘플을 구분하도록 학습된다. Adaptive attack에 견고하게 만들기 위해 Madry et al.(2017)의 PGD(Projected Gradient Descent) 공격을 학습 과정에 포함시킨다.

최종적인 학습 목적 함수 $\rho(\theta_k)$는 다음과 같이 정의된다.

$$\rho(\theta_k) = \mathbb{E}_{x \sim D_{\setminus k}} \left[ \max_{\delta \in S} L(d_k(x+\delta; \theta_k), 0) \right] + \mathbb{E}_{x \sim D_k} [L(d_k(x; \theta_k), 1)]$$

여기서 $S$는 $\epsilon$-norm ball 제약 조건이며, $L$은 Negative Log Likelihood(NLL) 손실 함수이다. 이 식의 의미는 클래스 $k$가 아닌 샘플들을 적대적으로 변형시켜 $d_k$가 이를 클래스 $k$로 오인하게 만드는 최악의 경우(inner maximization)에도, $d_k$가 이를 0(적대적 샘플)으로 분류하도록 학습시킨다는 것이다.

### 생성적 접근 방식 (Generative Approach)

논문은 이진 분류기 $d_k$의 로짓(logit) 출력값 $z_{d_k}(x)$를 에너지 기반 모델(Energy-Based Model)의 에너지 함수로 해석한다. 클래스 조건부 확률은 다음과 같이 정의된다.

$$p(x|k) = \frac{\exp(-E_k(x))}{Z_k}, \quad E_k(x) = -z_{d_k}(x)$$

여기서 $Z_k$는 정규화 상수(Partition function)이다. 모든 $Z_k$와 클래스 사전 확률 $p(k)$가 동일하다고 가정하면, 베이즈 분류 규칙에 의해 다음과 같은 분류기가 도출된다.

$$H(x) = \arg \max_k p(k|x) = \arg \max_k z_{d_k}(x)$$

이 방식은 $d_k$가 단순히 구분선을 찾는 것이 아니라 클래스 $k$의 데이터 분포(Density)를 학습하게 함으로써, 의미 없는 노이즈(Rubbish examples)에 의한 오분류를 방지하고 해석 가능성을 높인다.

## 📊 Results

### 실험 설정

- **데이터셋**: MNIST, CIFAR10, Restricted ImageNet
- **비교 대상**: State-of-the-art 탐지기(Carlini & Wagner, 2017a), Softmax Robust Classifier(Madry et al., 2017)
- **지표**: AUC (Area Under the ROC Curve), Mean $L_2$ distortion, TPR(True Positive Rate), FPR(False Positive Rate)

### 주요 결과

1. **탐지 성능**: MNIST 데이터셋에서 Generative Detection은 0.95 TPR 및 1.0 FPR 기준, Mean $L_2$ distortion을 기존 SOTA인 $3.68$에서 최대 $5.65$까지 향상시켰다. CIFAR10에서도 $1.1$에서 $1.5$로 향상시키며 우수한 성능을 보였다.
2. **Adaptive Attack에 대한 견고성**: Combined attack(분류기와 탐지기를 동시에 공격) 상황에서도 Generative Detection이 Integrated Detection보다 더 높은 견고성을 보였다.
3. **분류 성능**: Softmax Robust Classifier는 $\epsilon$ 값이 커질 때(예: MNIST $\epsilon=0.4$) 성능이 급격히 하락하는 반면, 제안된 Generative Classifier는 Reject 옵션을 통해 표준 정확도와 강건한 에러 사이의 균형을 맞추며 더 높은 견고성을 유지했다.
4. **해석 가능성**: Targeted attack 결과, Softmax Robust Classifier를 공격하여 생성된 이미지는 시각적으로 무의미한 경우가 많았으나, Generative Classifier를 공격하여 생성된 이미지는 타겟 클래스의 시각적 특징이 명확하게 나타났다. 이는 모델이 실제 클래스 분포를 학습했음을 시사한다.

## 🧠 Insights & Discussion

### 강점 및 기여

본 연구는 적대적 예제 탐지를 단순한 이진 분류 문제에서 Robust Optimization 문제로 격상시켰다. 특히 $K$개의 이진 분류기를 통해 입력 공간을 분할 관리함으로써 Adaptive attack에 대한 방어력을 높였으며, 이를 확률 밀도 추정 문제로 연결하여 모델의 해석 가능성을 확보한 점이 매우 뛰어나다.

### 한계 및 비판적 해석

1. **계산 비용의 증가**: 가장 큰 한계는 비용 문제이다. $K$개의 클래스가 있을 때 $K$개의 이진 분류기를 유지하고 추론해야 하므로, 메모리 요구량과 추론 시간이 Softmax 분류기에 비해 약 $K$배(CIFAR10의 경우 10배) 증가한다.
2. **학습 시간**: Generative Adversarial Training(GAT)은 일반적인 적대적 학습보다 약 2.7배 느린 것으로 측정되었다. 실시간성이나 자원이 제한된 환경에서의 적용에는 어려움이 있을 수 있다.
3. **가정의 단순함**: 모든 클래스의 정규화 상수 $Z_k$와 사전 확률 $p(k)$가 동일하다고 가정한 점은 이론적 단순화를 위한 것이나, 실제 불균형 데이터셋에서는 성능 저하의 원인이 될 수 있다.

## 📌 TL;DR

본 논문은 Adaptive attack에 견고한 적대적 예제 탐지를 위해 $K$개의 이진 분류기를 이용한 하위 공간 분할 탐지 기법 및 Generative Adversarial Training(GAT)을 제안한다. 이 방법은 기존 SOTA 탐지기보다 높은 $L_2$ 왜곡 거리(Distortion)를 유도하여 공격을 어렵게 만들며, 특히 생성적 모델로 해석함으로써 "무의미한 입력"에 속지 않는 해석 가능한 강건한 분류를 가능케 한다. 다만, 클래스 수 $K$에 비례하여 증가하는 계산 비용이 실제 적용의 주요 병목이 될 것으로 보인다.
