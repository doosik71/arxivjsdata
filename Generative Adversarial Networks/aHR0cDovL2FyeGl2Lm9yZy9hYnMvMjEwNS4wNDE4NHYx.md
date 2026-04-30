# Generative Adversarial Networks (GANs) in Networking: A Comprehensive Survey & Evaluation

Hojjat Navidan, Parisa Fard Moshiri, Mohammad Nabati, Reza Shahbazian, Seyed Ali Ghorashi, Vahid Shah-Mansouri, David Windridge (2021)

## 🧩 Problem to Solve

현대 컴퓨터 및 통신 네트워크, 특히 5G와 같은 차세대 네트워크는 매우 복잡하고 비선형적인 특성을 가지고 있어 기존의 고전적인 알고리즘만으로는 효율적인 자원 관리와 실시간 분석을 수행하는 데 한계가 있다. 이를 해결하기 위해 딥러닝 기반의 접근 방식이 널리 도입되었으나, 이러한 학습 모델들이 효과적으로 작동하기 위해서는 방대한 양의 학습 데이터가 필수적이다.

그러나 실제 네트워크 환경에서는 다음과 같은 심각한 문제들이 발생한다. 첫째, 데이터 접근성이 제한적이거나 데이터를 수집하는 비용이 지나치게 높다. 둘째, 실제 데이터의 클래스 분포가 매우 불균형(Class Imbalance)하여, 소수 클래스에 대한 학습이 제대로 이루어지지 않아 모델의 성능이 저하되는 문제가 발생한다.

본 논문의 목표는 이러한 데이터 부족 및 불균형 문제를 해결할 수 있는 Generative Adversarial Networks (GANs)의 네트워크 분야 적용 사례를 종합적으로 조사(Survey)하고, 특히 이미지 데이터가 아닌 일반 네트워크 데이터에 적용 가능한 성능 평가 프레임워크를 제시하여 다양한 GAN 모델의 성능을 정량적으로 비교 분석하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 GAN의 이론적 배경부터 실제 네트워크 응용 사례, 그리고 정량적 평가까지 통합적인 관점에서 분석한 것에 있다.

1.  **네트워크 도메인 중심의 GAN 서베이**: 기존의 GAN 연구가 주로 Computer Vision(CV)에 치중되어 있었으나, 본 논문은 이를 모바일 네트워크, 네트워크 분석, IoT, 물리 계층(Physical Layer), 사이버 보안이라는 다섯 가지 주요 카테고리로 분류하여 상세히 분석하였다.
2.  **비이미지 데이터 평가 프레임워크 제안**: GAN의 성능 평가가 주로 인간의 시각적 판단이나 이미지 전용 지표(Inception Score 등)에 의존했다는 점을 지적하며, 네트워크 데이터와 같은 수치형 데이터에 적용 가능한 Maximum Mean Discrepancy (MMD)와 Earth Mover Distance (EMD) 기반의 평가 체계를 제안하였다.
3.  **실증적 모델 비교 분석**: 제안한 프레임워크를 바탕으로 Vanilla GAN, CGAN, BiGAN, LSGAN, WGAN 등 5가지 주요 모델을 4가지 서로 다른 네트워크 데이터셋에 적용하여, 데이터의 특성(차원, 형태 등)에 따라 최적의 GAN 모델이 달라짐을 입증하였다.

## 📎 Related Works

기존의 GAN 관련 서베이 논문들은 대부분 이미지 생성, 얼굴 합성, 이미지 변환 등 Computer Vision 분야의 응용 사례에 집중되어 있었다. 일부 연구에서 음성 인식이나 언어 처리 분야를 다루기도 하였으나, 컴퓨터 및 통신 네트워크라는 특수한 도메인을 전문적으로 다룬 종합 서베이는 부재한 상태였다.

본 논문은 기존 연구들이 이미지 데이터의 시각적 품질 평가에 치중한 것과 달리, 네트워크 데이터의 통계적 유사성을 측정하는 정량적 지표의 필요성을 강조하며 기존의 CV 중심 서베이들과 차별점을 둔다.

## 🛠️ Methodology

### 1. GAN의 기본 구조 및 학습 원리
GAN은 생성자(Generator, $G$)와 판별자(Discriminator, $D$)라는 두 개의 신경망이 서로 대립하며 학습하는 게임 이론적 구조를 가진다. 생성자는 실제 데이터와 유사한 가짜 데이터를 만들어 판별자를 속이려 하고, 판별자는 입력된 데이터가 실제인지 가짜인지 정확히 구분하려 한다.

이들의 학습 목표는 다음과 같은 Minimax 목적 함수로 정의된다.
$$\min_{G} \max_{D} L(D,G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1-D(G(z)))]$$
여기서 $p_{data}(x)$는 실제 데이터의 분포이며, $p_z(z)$는 생성자의 입력으로 들어가는 잠재 공간(Latent Space)의 노이즈 분포이다.

### 2. 주요 GAN 변형 모델
논문에서는 네트워크 도메인에서 활용 가능한 주요 모델들을 다음과 같이 설명한다.
- **CGAN (Conditional GAN)**: 데이터에 클래스 레이블 $y$를 조건으로 추가하여 특정 클래스의 데이터를 생성할 수 있게 한다.
- **ACGAN (Auxiliary Classifier GAN)**: 판별자가 진위 여부뿐만 아니라 클래스 레이블까지 예측하도록 하여 생성 데이터의 품질을 높인다.
- **WGAN (Wasserstein GAN)**: Jensen-Shannon Divergence 대신 Wasserstein 거리(Earth Mover Distance)를 사용하여 학습의 안정성을 높이고 기울기 소실(Vanishing Gradient) 문제를 완화한다.
- **WGAN-GP**: WGAN의 가중치 클리핑(Weight Clipping) 문제를 해결하기 위해 Gradient Penalty 항을 추가하여 립시츠(Lipschitz) 연속성 조건을 강제한다.
- **LSGAN (Least Squares GAN)**: 교차 엔트로피 손실 함수를 최소제곱법(Least Squares) 손실 함수로 대체하여 기울기 소실을 줄이고 학습 안정성을 개선한다.
- **BiGAN (Bidirectional GAN)**: 인코더(Encoder)를 추가하여 실제 데이터에서 잠재 공간으로의 역매핑이 가능하게 한다.

### 3. 성능 평가 지표
비이미지 데이터의 유사도를 측정하기 위해 다음과 같은 수식을 사용한다.
- **EMD (Earth Mover Distance)**: 두 확률 분포 사이의 거리를 측정하며, WGAN의 판별자(Critic)를 통해 근사적으로 계산한다.
$$\hat{W}(x_r, x_g) = \frac{1}{N} \sum_{i=1}^{N} \hat{F}(x_r^{[i]}) - \frac{1}{N} \sum_{i=1}^{N} \hat{F}(x_g^{[i]})$$
- **MMD (Maximum Mean Discrepancy)**: 두 분포에서 추출한 샘플 간의 커널 평균 차이를 측정하며, 값이 작을수록 두 분포가 유사함을 의미한다.
$$MMD^2(p_r, p_g) = \mathbb{E}_{x,x'}[k(x,x')] - 2\mathbb{E}_{x,y}[k(x,y)] + \mathbb{E}_{y,y'}[k(y,y')]$$

## 📊 Results

### 1. 실험 설정
- **데이터셋**: WiFi RSSI, WiFi CSI, KDD99(네트워크 트래픽), RadioML(디지털 변조 신호) 총 4종의 데이터셋을 사용하였다.
- **비교 모델**: Vanilla GAN, CGAN, BiGAN, LSGAN, WGAN.
- **환경**: Keras-GAN 프레임워크, NVIDIA GeForce RTX 2060 GPU.

### 2. 주요 결과
실험 결과, 모든 데이터셋에서 압도적인 성능을 보이는 단일 모델은 존재하지 않았으며, 데이터의 차원과 특성에 따라 성능이 크게 달라짐을 확인하였다.

- **고차원 데이터의 한계**: WiFi CSI와 같이 데이터 차원이 매우 높은 경우, Vanilla GAN과 LSGAN은 MMD와 EMD 값이 매우 높게 나타나 성능이 현저히 떨어지는 모습을 보였다.
- **상대적 우위**: 전반적으로 WGAN과 CGAN이 다양한 데이터셋에서 안정적인 성능을 보였으나, 데이터셋마다 최적의 모델이 상이했다.
- **분포 시각화**: KDE(Kernel Density Estimation) 및 Q-Q Plot 분석을 통해 생성된 데이터가 실제 데이터의 통계적 분포를 어느 정도 모사하고 있음을 확인하였다. 그러나 급격한 변동성(Fast Fluctuations)을 가진 데이터의 세밀한 특징을 포착하는 데는 여전히 한계가 있었다.

## 🧠 Insights & Discussion

본 논문은 GAN이 네트워크 분야에서 데이터 부족 및 클래스 불균형 문제를 해결하는 데 매우 유용한 도구임을 입증하였다. 특히, 레이블이 없는 대량의 데이터를 통해 생성자를 먼저 학습시킨 후, 소량의 레이블된 데이터로 판별자를 미세 조정하는 Semi-supervised Learning 방식이 네트워크 데이터 분석에 큰 이점을 제공할 수 있음을 시사한다.

**한계 및 비판적 해석**:
1.  **이산 데이터 생성의 어려움**: 현재의 GAN 구조는 역전파(Back-propagation)에 의존하므로, 프로토콜 타입이나 플래그와 같은 이산형(Discrete) 데이터를 직접 생성하는 데 어려움이 있다.
2.  **모델 선택의 불확실성**: "No Free Lunch" 정리와 마찬가지로 모든 네트워크 상황에 적용 가능한 만능 GAN 모델은 없으며, 현재로서는 시행착오(Trial-and-Error)를 통해 모델을 선택해야 하는 번거로움이 있다.
3.  **평가 지표의 부족**: 이미지 데이터와 달리 수치형 데이터의 '품질'을 정의할 이론적/통계적 지표가 여전히 부족하며, 이는 모델 선택 과정을 더 어렵게 만든다.

## 📌 TL;DR

본 논문은 네트워크 도메인에서 데이터 부족 및 불균형 문제를 해결하기 위해 GAN을 적용한 최신 연구들을 종합적으로 분석한 서베이 논문이다. 특히 이미지 데이터가 아닌 수치형 네트워크 데이터의 유사도를 측정하기 위해 MMD와 EMD 기반의 평가 프레임워크를 제안하였고, 5가지 GAN 모델을 4가지 데이터셋으로 실험하여 데이터 특성에 따라 최적 모델이 다름을 입증하였다. 이 연구는 향후 네트워크 보안, 물리 계층 최적화 및 IoT 환경에서의 데이터 증강 연구에 중요한 가이드라인을 제공할 것으로 기대된다.