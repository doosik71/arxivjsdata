# Quaternion Generative Adversarial Networks

Eleonora Grassucci, Edoardo Cicero and Danilo Comminiello (2021)

## 🧩 Problem to Solve

최신 Generative Adversarial Networks (GANs)는 대규모 학습을 통해 뛰어난 성능을 보여주고 있으나, 수백만 개의 파라미터로 구성된 거대 모델을 사용함에 따라 막대한 계산 능력이 요구된다. 이러한 모델의 거대함은 재현 가능성을 떨어뜨리고 학습의 불안정성을 증가시키는 원인이 된다.

또한, 이미지나 오디오와 같은 다채널(multi-channel) 데이터를 처리할 때, 기존의 실수 값 기반 합성곱 신경망(real-valued convolutional networks)은 입력 채널을 평탄화(flatten)하거나 단순히 연결(concatenate)하여 처리한다. 이 과정에서 채널 간의 내부 공간 관계(intra-channel spatial relations)가 손실되는 문제가 발생한다. 본 논문은 이러한 모델의 복잡도 문제와 정보 손실 문제를 동시에 해결하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 쿼터니언 대수(quaternion algebra)의 특성, 특히 Hamilton product를 활용하여 다채널 데이터를 하나의 엔티티(single entity)로 처리하는 Quaternion Generative Adversarial Networks (QGANs) 제품군을 제안하는 것이다.

주요 기여 사항은 다음과 같다.

1. **QGAN 프레임워크 제안**: 쿼터니언 도메인에서 완전히 정의된 GAN을 통해 생성 능력을 향상시키면서 전체 파라미터 수를 최대 75%까지 줄였다.
2. **Quaternion Batch Normalization (QBN) 정의**: 이론적으로 올바른 QBN 접근 방식을 정의하고, 기존의 방식들이 이에 대한 근사치임을 규명하였다.
3. **Quaternion Spectral Normalization (QSN) 제안**: 쿼터니언 도메인에서의 Spectral Normalization을 정의하고, 이를 통해 학습 안정성을 확보하였다.

## 📎 Related Works

기존의 GAN 연구들은 주로 학습 안정화(Gradient Penalty, Spectral Normalization)나 아키텍처 혁신(Self-attention, Style-based generator), 혹은 모델의 규모를 확장(BigGAN)하는 방향으로 발전해 왔다. 그러나 이러한 접근법은 모델의 파라미터 수를 급격히 증가시켜 계산 자원 소모를 심화시켰다.

또한, 기존의 실수 기반 신경망은 다차원 입력을 처리할 때 채널을 독립적인 엔티티로 취급하여 채널 간의 상관관계를 활용하지 못한다는 한계가 있다. 이에 대한 대안으로 하이퍼컴플렉스(hypercomplex) 도메인의 신경망, 특히 Quaternion Neural Networks (QNNs)가 제안되었으며, QNNs는 Hamilton product를 통해 채널 간 관계를 유지하면서 파라미터 수를 획기적으로 줄일 수 있음을 보여주었다. 본 논문은 이러한 QNN의 이점을 생성적 적대 신경망(GAN) 구조에 완전히 통합하였다.

## 🛠️ Methodology

### 1. 쿼터니언 대수 기초

쿼터니언 $q$는 하나의 스칼라 성분과 세 개의 허수 성분으로 정의된다.
$$q = q_0 + q_1 \hat{i} + q_2 \hat{j} + q_3 \hat{k}$$
여기서 $\hat{i}^2 = \hat{j}^2 = \hat{k}^2 = -1$이며, Hamilton product는 비가환(non-commutative) 특성을 가진다.

### 2. Quaternion Core Blocks

- **Quaternion Fully Connected & Convolutional Layers**:
  실수 기반 레이어의 가중치 행렬 $W_r$ 대신 쿼터니언 가중치 $W = W_0 + W_1 \hat{i} + W_2 \hat{j} + W_3 \hat{k}$를 사용한다. 연산은 Hamilton product를 통해 수행된다.
  $$W \ast x = (W_0 \ast x_0 - W_1 \ast x_1 - W_2 \ast x_2 - W_3 \ast x_3) + \dots (\text{imaginary parts})$$
  이 구조는 4개의 하위 행렬이 가중치를 공유하므로, 동일한 출력 차원을 유지하면서 파라미터 수를 $\frac{1}{4}$로 줄인다.

- **Split Activation Functions**:
  쿼터니언의 각 성분에 대해 개별적으로 활성화 함수를 적용하는 방식을 사용한다. 예를 들어 ReLU의 경우 다음과 같이 적용된다.
  $$y = \text{ReLU}(z_0) + \text{ReLU}(z_1)\hat{i} + \text{ReLU}(z_2)\hat{j} + \text{ReLU}(z_3)\hat{k}$$

### 3. Quaternion Batch Normalization (QBN)

본 논문은 쿼터니언 신호가 자신의 involution과 상관관계가 없다는 $\text{Q-properness}$ 가정을 도입하여 계산 복잡도를 줄인 QBN을 제안한다.
$$\text{QBN}(x) = \gamma \frac{x - \mu_q}{\sqrt{\text{var}(x)}} + \beta$$
여기서 $\mu_q$는 쿼터니언 평균이며, $\gamma$는 스칼라, $\beta$는 쿼터니언 파라미터이다.

### 4. Quaternion Spectral Normalization (QSN)

Discriminator의 Lipschitz 연속성을 보장하기 위해 가중치의 Spectral norm을 제한한다. 본 논문은 두 가지 방식을 비교 분석하였다.

- **QSN Split**: 각 하위 행렬 $W_0 \dots W_3$를 독립적으로 정규화한다.
- **QSN Full**: 전체 쿼터니언 가중치 행렬 $W$의 Spectral norm을 계산하여 모든 성분을 동시에 정규화한다. 실험 결과, QSN Full이 더 안정적이고 높은 성능을 보였다.

### 5. QGAN 아키텍처

- **Vanilla QGAN**: DCGAN 구조를 쿼터니언 도메인으로 재정의한 모델이다.
- **QSNGAN (Advanced QGAN)**: SNGAN을 기반으로 하며, Quaternion Residual Blocks (QResBlock)를 사용한다. Generator에는 QBN을, Discriminator에는 QSN을 적용하며, Hinge loss를 통해 학습한다.

## 📊 Results

### 실험 설정

- **데이터셋**: CelebA-HQ ($128 \times 128$), 102 Oxford Flowers ($128 \times 128$), CIFAR10, STL10.
- **평가 지표**: Fréchet Inception Distance (FID, 낮을수록 좋음), Inception Score (IS, 높을수록 좋음).
- **비교 대상**: 실수 값 기반의 SNGAN.

### 주요 결과

1. **정량적 성능**:
   - CelebA-HQ 데이터셋에서 QSNGAN은 SNGAN보다 낮은 FID와 높은 IS를 기록하였다.
   - 특히 Discriminator의 반복 횟수(Critic iterations) 설정에 대해 QSNGAN이 SNGAN보다 훨씬 더 강건(robust)한 모습을 보였다.
2. **파라미터 효율성**:
   - Table 1에 따르면, QSNGAN의 전체 파라미터 수는 약 16.9M으로, SNGAN의 61.2M 대비 약 72% 이상 감소하였다. 디스크 메모리 사용량 또한 $115\text{GB} \rightarrow 35\text{GB}$로 대폭 절감되었다.
3. **정성적 성능**:
   - 시각적 분석 결과, QSNGAN은 배경과 피사체의 구분이 더 명확하며, 눈썹, 수염, 피부 톤 등 세부 속성에 대한 묘사 능력이 더 뛰어나고 색상이 더 선명한 이미지를 생성하였다.
4. **QSN 방식 비교**:
   - CIFAR10 및 STL10 실험을 통해 No QSN $\rightarrow$ QSN Split $\rightarrow$ QSN Full 순으로 FID 성능이 향상됨을 확인하여, QSN Full 방식의 우수성을 입증하였다.

## 🧠 Insights & Discussion

본 논문은 쿼터니언 대수를 딥러닝의 생성 모델에 적용함으로써 모델의 경량화와 성능 향상이라는 두 마리 토끼를 잡았다. 특히, RGB 채널을 독립적인 텐서로 처리하지 않고 쿼터니언의 구성 요소로 묶어 처리함으로써 채널 간의 내부 관계(internal relations)를 학습할 수 있었다는 점이 가장 큰 강점이다.

다만, 본 논문의 QBN 구현에서 $\text{Q-properness}$ 가정을 사용하여 계산량을 줄였는데, 이는 실제 데이터가 항상 이 가정을 만족하는지에 대한 추가적인 검토가 필요할 수 있다. 또한, 제안된 방법론이 매우 높은 해상도의 이미지 생성에서도 동일한 파라미터 절감 효율과 성능 우위를 유지할 수 있을지는 명시적으로 다뤄지지 않았다.

그럼에도 불구하고, 하이퍼컴플렉스 도메인에서 GAN 프레임워크를 완전히 정의하고, Spectral Normalization과 Batch Normalization을 이론적으로 정립하여 실질적인 성능 향상을 증명했다는 점에서 학술적 가치가 높다.

## 📌 TL;DR

본 연구는 쿼터니언 대수를 활용한 **Quaternion Generative Adversarial Networks (QGANs)**를 제안하여, 기존 실수 기반 GAN 대비 **파라미터 수를 약 75% 줄이면서도 이미지 생성 품질(FID, IS)을 향상**시켰다. 특히 쿼터니언 합성곱을 통해 이미지 채널 간의 상관관계를 보존하며, 쿼터니언 전용 Spectral Normalization(QSN)과 Batch Normalization(QBN)을 통해 학습 안정성을 확보하였다. 이 연구는 향후 계산 자원이 제한된 환경에서 고성능 생성 모델을 구축하는 데 중요한 기초가 될 것으로 보인다.
