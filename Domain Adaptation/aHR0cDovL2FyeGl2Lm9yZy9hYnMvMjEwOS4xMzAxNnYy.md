# Semi-Supervised Adversarial Discriminative Domain Adaptation

Thai-Vu Nguyen, Anh Nguyen, Nghia Le, Bac Le (2022)

## 🧩 Problem to Solve

본 논문은 딥러닝 모델이 학습 데이터셋에서는 우수한 성능을 보이지만, 실제 환경이나 완전히 다른 데이터셋(Testing dataset)에서는 성능이 급격히 저하되는 **Dataset Bias** 또는 **Domain Shift** 문제를 해결하고자 한다. 이러한 현상은 조명, 이미지 품질, 배경 등 다양한 요인으로 인해 소스 도메인(Source Domain)과 타겟 도메인(Target Domain)의 데이터 분포가 다르기 때문에 발생한다.

연구의 구체적인 목표는 라벨이 있는 소스 데이터셋과 라벨이 없는 타겟 데이터셋을 활용하는 **Unsupervised Domain Adaptation (UDA)** 상황에서, 두 도메인을 공통된 특징 공간(Common feature space)으로 매핑하여 타겟 도메인에 대한 분류 정확도를 높이는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Semi-supervised learning** 기법을 Adversarial 기반의 도메인 적응 방법론에 결합하는 것이다. 기존의 ADDA(Adversarial Discriminative Domain Adaptation)와 같은 방식은 판별기(Discriminator)가 단순히 소스 도메인과 타겟 도메인을 구분(Binary Classification)하는 역할만 수행했다.

반면, 제안된 **SADDA (Semi-Supervised Adversarial Discriminative Domain Adaptation)**는 판별기가 $N+1$개의 클래스를 예측하도록 설계되었다. 여기서 $N$은 분류 작업의 클래스 수이며, 마지막 $+1$은 해당 샘플이 소스 도메인인지 타겟 도메인인지를 구분하는 도메인 라벨이다. 이를 통해 판별기가 도메인 구분뿐만 아니라 소스 데이터의 클래스 정보까지 동시에 학습하게 함으로써, 인코더와 판별기의 일반화 능력을 동시에 향상시킨다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구들을 언급하며 SADDA의 차별점을 제시한다.

1. **Unsupervised Domain Adaptation (UDA):** MMD(Maximum Mean Discrepancy), DRCN(Deep Reconstruction Classification Network), Autoencoder 기반 방식 등이 존재하며, 공통적으로 소스와 타겟의 분포 차이를 최소화하는 데 집중한다.
2. **Adversarial-based DA:** GAN의 개념을 도입하여 도메인 판별기를 통해 도메인 간 혼동(Domain confusion)을 증가시키는 방식이다. 특히 **ADDA**는 소스 인코더와 타겟 인코더의 분포 거리를 좁히는 데 성공했으나, 판별기가 오직 도메인 구분만을 수행한다는 한계가 있다.
3. **Auxiliary Tasks 결합 방식:** DRCN처럼 분류와 재구성(Reconstruction)을 동시에 수행하는 연구가 있었으나, SADDA는 보조 작업이 아닌 판별기 자체의 구조를 변경하여 분류와 도메인 구분을 동일한 목표(특징 추출 능력 향상)로 통합했다는 점에서 차별화된다.

## 🛠️ Methodology

SADDA의 전체 파이프라인은 **Pre-training $\rightarrow$ Training target encoder $\rightarrow$ Testing**의 3단계로 구성된다.

### 1. Pre-training (사전 학습)

소스 도메인의 라벨링된 데이터 $(X_s, Y_s)$를 사용하여 소스 인코더($M_s$)와 소스 분류기($C_s$)를 학습시킨다. 이는 표준적인 지도 학습 분류 문제로, 다음과 같은 Categorical Cross-entropy 손실 함수를 최소화한다.
$$\arg \min_{M_s, C_s} L_{cls}(X_s, Y_s) = -\mathbb{E}_{(x,y) \sim (X_s, Y_s)} \sum_{n=1}^{N} y_n \log C_s(M_s(x_n)) \quad (1)$$

### 2. Training Target Encoder (타겟 인코더 학습)

이 단계에서는 판별기($D$)와 타겟 인코더($M_t$)를 동시에 학습시킨다.

**A. 판별기(Discriminator) 학습**
판별기는 두 가지 모드로 동작하며, 내부적으로 동일한 특징 추출 레이어를 공유한다.

- **Supervised mode ($D_{sup}$):** 소스 데이터의 $N$개 클래스를 예측한다.
$$\arg \min_{D_{sup}} L_{cls}(X_s, Y_s) = -\mathbb{E}_{(x,y) \sim (X_s, Y_s)} \sum_{n=1}^{N} y_n \log D_{sup}(M_s(x_n)) \quad (2)$$
- **Unsupervised mode ($D_{unsup}$):** 샘플이 소스인지 타겟인지 구분한다.
$$\arg \min_{D_{unsup}} L_{adv}^D(X_s, X_t, M_s, M_t) = -\mathbb{E}_{x_s \sim X_s} \log D_{unsup}(M_s(x_s)) - \mathbb{E}_{x_t \sim X_t} \log(1 - D_{unsup}(M_t(x_t))) \quad (3)$$

**B. 타겟 인코더($M_t$) 학습**
판별기 $D_{unsup}$을 속이도록 학습하여, 타겟 데이터를 소스 데이터와 유사한 특징 공간으로 매핑한다.
$$\arg \min_{M_t} L_{adv}^M(X_s, X_t, D) = -\mathbb{E}_{x_t \sim X_t} \log D_{unsup}(M_t(x_t)) \quad (4)$$

### 3. 판별기 모델의 상세 구조 및 Custom Activation

SADDA의 핵심은 $D_{sup}$과 $D_{unsup}$이 동일한 특징 추출 레이어를 사용한다는 점이다. $D_{unsup}$은 $D_{sup}$의 Softmax 직전 출력값들을 사용하여 다음과 같은 **Custom Activation** 함수를 통해 도메인을 판별한다.
$$D(x) = \frac{Z(x)}{Z(x) + 1} \quad (5)$$
여기서 $Z(x)$는 다음과 같이 정의된다.
$$Z(x) = \sum_{n=1}^{N} \exp[l_n(x)] \quad (6)$$

- **직관적 의미:** $Z(x)$는 출력 값들의 지수 합이다. 소스 샘플의 경우 특정 클래스에 대한 확신(Low entropy)이 높으므로 $Z(x)$가 커져 $D(x)$가 1.0에 가까워진다. 반면 타겟 샘플은 확신이 낮아(High entropy) $D(x)$가 0.0에 가깝게 출력된다.

### 4. 추론 절차 (Testing)

학습이 완료되면 타겟 인코더($M_t$)와 사전 학습된 소스 분류기($C_s$)를 결합하여 타겟 데이터 $X_t$의 라벨을 예측한다.

### 5. 아키텍처 설계 가이드라인

- **Image Classification:** DCGAN 구조에서 영감을 받아 Encoder(Conv layers)와 Discriminator(Transpose Conv layers)를 대칭적으로 구성한다. Pooling 레이어 대신 Stride $\ge 2$인 Convolution을 사용하고, ReLU(Encoder)와 LeakyReLU(Discriminator)를 적용한다.
- **Sentiment Classification:** Encoder-Decoder LSTM 구조를 사용한다. 텍스트 시퀀스를 처리하기 위해 LSTM 레이어를 배치하고, 과적합 방지를 위해 Dropout(0.2)을 적용한다.

## 📊 Results

### 1. Digit Recognition (숫자 인식)

- **데이터셋:** MNIST $\leftrightarrow$ USPS, MNIST $\rightarrow$ MNIST-M, SVHN $\rightarrow$ MNIST.
- **결과:** SVHN $\rightarrow$ MNIST 실험에서 Source only 대비 약 26% 향상된 86.5%의 정확도를 기록했다. 일부 최신 기법(SHOT, DFA-MCD)보다는 소폭 낮으나, ADDA보다는 높은 경쟁력을 보였다. t-SNE 시각화 결과, SADDA는 클래스 내 거리를 최소화하고 클래스 간 거리를 최대화하여 더 명확한 군집을 형성함을 확인했다.

### 2. Object Recognition (객체 인식)

- **데이터셋:** VLCS (PASCAL, LABELME, CALTECH, SUN).
- **결과:** 모든 케이스에서 Source only 모델보다 높은 정확도를 보였다. 예를 들어 PASCAL $\rightarrow$ CALTECH의 경우 45.10%에서 55.30%로 약 10% 향상되었다. 하지만 타겟 데이터로 직접 학습한 모델(Train on target)과의 간극은 여전히 컸다.

### 3. Sentiment Classification (감성 분류)

- **데이터셋:** Women's E-Commerce, Coronavirus tweets, Trip Advisor reviews.
- **결과:** 전반적으로 Source only 대비 소폭 향상되었으며, 특히 T $\rightarrow$ W 케이스에서는 41.07%에서 49.28%로 상승했다. 다만 일부 케이스(T $\rightarrow$ C)에서는 성능이 소폭 하락하는 모습이 관찰되었다.

## 🧠 Insights & Discussion

**강점:**

- 판별기의 구조를 변경하여 지도 학습과 비지도 학습의 이점을 동시에 취한 설계가 효율적이다.
- 컴퓨터 비전뿐만 아니라 NLP(감성 분류) 작업에도 적용 가능한 범용적인 프레임워크임을 입증했다.
- t-SNE 분석을 통해 특징 공간에서의 도메인 불변성(Domain invariance)이 실제로 개선되었음을 정성적으로 보여주었다.

**한계 및 비판적 해석:**

- **학습 안정성:** 판별기와 인코더가 동시에 학습되는 Adversarial process 특성상 Loss landscape가 매우 불안정하다. 논문에서는 이를 해결하기 위해 Nash equilibrium 상태에서 **Early Stopping**을 적용해야 한다고 명시하고 있다.
- **성능 격차:** 객체 인식과 감성 분류에서 Source only보다는 낫지만, 여전히 이상적인 성능(Target labels 존재 시)과는 큰 차이가 있다. 이는 단순한 도메인 정렬만으로는 해결되지 않는 도메인 간의 근본적인 차이가 존재함을 시사한다.
- **복잡도:** $N+1$ 클래스 예측을 위한 Custom Activation 설계가 정교하지만, 실제 구현 시 하이퍼파라미터(Learning rate, $\beta_1$ 등)에 매우 민감하게 반응할 가능성이 높다.

## 📌 TL;DR

본 논문은 판별기가 클래스 분류($N$)와 도메인 구분($+1$)을 동시에 수행하도록 설계된 **SADDA** 방법론을 제안하여 Unsupervised Domain Adaptation 문제를 해결하고자 하였다. 제안 방법은 숫자 인식, 객체 인식, 감성 분류 등 다양한 태스크에서 기존 Source-only 모델 및 일부 Adversarial 기반 모델보다 우수한 성능을 보였다. 특히 $N+1$ 분류 구조를 통한 특징 추출 능력의 상호 보완적 향상이 핵심이며, 향후 다양한 도메인 적응 작업의 기반 모델로 활용될 가능성이 있다.
