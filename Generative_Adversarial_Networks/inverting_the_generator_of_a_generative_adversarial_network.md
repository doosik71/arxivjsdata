# Inverting The Generator Of A Generative Adversarial Network

Antonia Creswell, Anil Anthony Bharath (2016)

## 🧩 Problem to Solve

본 논문은 Generative Adversarial Networks (GANs)의 Generator $G$를 역전(Inversion)시켜, 이미지 공간(Image Space, $X$)의 데이터를 잠재 공간(Latent Space, $Z$)으로 투영하는 문제를 다룬다.

GAN은 기본적으로 잠재 공간의 벡터 $z$를 입력받아 고차원 이미지 $x$를 생성하는 $G: Z \to X$ 매핑을 학습한다. 만약 이 과정을 역으로 수행하여 이미지 $x$로부터 그에 대응하는 $z$를 찾아낼 수 있다면, 이미지 검색(Image Retrieval), 이미지 분류(Image Classification)와 같은 판별 작업(Discriminative Tasks)에서 잠재 공간의 풍부한 선형 구조를 활용할 수 있으며, 원본 이미지를 정밀하게 조작(Manipulation)하는 것도 가능해진다.

그러나 Generator는 다수의 비선형 계층으로 구성되어 있어 수학적으로 역함수를 구하는 것이 매우 어렵다. 기존의 접근 방식은 Generator와 함께 이미지 $x$를 $z$로 매핑하는 별도의 Decoder 네트워크를 학습시키는 방식이었으나, 이는 추가적인 파라미터 학습으로 인한 과적합(Overfitting) 위험이 있고, 이미 학습이 완료된 Pre-trained 네트워크에는 적용할 수 없다는 한계가 있다. 따라서 본 논문의 목표는 추가적인 네트워크 학습 없이, 기존의 Pre-trained GAN을 그대로 사용하여 이미지를 잠재 공간으로 투영하는 효율적인 기법을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 별도의 역전 네트워크를 학습시키는 대신, 주어진 이미지 $x$와 생성된 이미지 $G(z)$ 사이의 오차를 최소화하는 $z$를 **경사하강법(Gradient Descent)**을 통해 직접 최적화하여 찾는 것이다.

주요 기여 사항은 다음과 같다:

1. **범용적 역전 방법론 제안**: Generator의 계산 그래프(Computational Graph)만 존재한다면 어떤 Pre-trained GAN에도 적용 가능한 역전 기법을 제시하였다.
2. **Batch Normalization 문제 해결**: 단일 샘플 역전 시 발생할 수 있는 Batch Normalization의 불안정성 문제를 해결하기 위해, 여러 이미지를 동시에 역전시키는 **Batch Inversion** 방식을 제안하였다.
3. **잠재 공간 제약 조건 도입**: 잠재 공간의 사전 분포 $P(Z)$(Uniform 또는 Gaussian)를 고려하여, 최적화 과정에서 Clipping이나 Regularization 항을 추가함으로써 $z$가 유효한 영역 내에 존재하도록 유도하는 방법을 제시하였다.
4. **일반화 성능 입증**: Omniglot 데이터셋을 통해 학습 과정에서 보지 못한 새로운 알파벳의 문자라도 잠재 공간으로 성공적으로 투영될 수 있음을 보여줌으로써, One-shot Learning으로의 확장 가능성을 제시하였다.

## 📎 Related Works

본 논문은 크게 두 가지 기존 접근 방식과 차별점을 가진다.

첫째, Donahue et al. [3, 4] 및 Dumoulin et al. [5]은 Generator와 함께 이미지를 $z$로 매핑하는 별도의 Decoder 네트워크를 학습시키는 방식을 제안하였다. 하지만 이 방식은 재구성된 이미지의 품질이 낮고 특히 MNIST의 경우 스타일과 클래스 보존 능력이 떨어진다는 단점이 있다. 또한, 전술한 바와 같이 Pre-trained 모델에 적용 불가능하다.

둘째, Zhu et al. [12]은 이미지 역전을 수행하였으나, $\text{AlexNet}$과 같은 사전 학습된 CNN의 특징 맵(Feature Map)을 사용하여 재구성 손실을 계산하였다. 이는 자연 이미지(Natural Scenes)에는 효과적일 수 있으나, MNIST나 Omniglot과 같은 특수 데이터셋에는 적합하지 않다. 본 논문은 픽셀 단위의 손실(Pixel-wise loss)을 사용하여 데이터셋에 관계없이 범용적으로 적용 가능하며, 다른 외부 네트워크의 정보 없이 GAN 모델 자체의 특성을 분석할 수 있다는 점에서 차별화된다.

## 🛠️ Methodology

### 전체 파이프라인

본 연구의 역전 과정은 특정 이미지 $x$에 대해, $G(z)$가 $x$와 가장 유사해지도록 하는 $z^*$를 찾는 최적화 문제로 정의된다.

$$z^* = \min_{z} -\mathbb{E}_x \log[G(z)]$$

### 상세 학습 절차 (Algorithm 1)

1. **초기화**: 잠재 공간의 사전 분포 $P(Z)$에서 무작위로 $z$를 샘플링하여 초기값을 설정한다.
2. **손실 계산**: 입력 이미지 $x$와 Generator를 통해 생성된 이미지 $G(z^*)$ 사이의 Binary Cross Entropy (BCE) 손실 $L$을 계산한다.
   $$L = -(x \log[G(z^*)] + (1-x) \log[1-G(z^*)])$$
3. **업데이트**: $G$의 가중치는 고정하고, $z$에 대한 기울기 $\nabla_z L$을 계산하여 $z$ 값을 업데이트한다.
   $$z^* \leftarrow z^* - \alpha \nabla_z L$$
4. **반복**: 수렴할 때까지 위 과정을 반복하여 최종적인 $z^*$를 도출한다.

### Batch Normalization 및 Batch Inversion

GAN의 Generator에 Batch Normalization이 적용된 경우, 단일 $z$ 값을 입력하면 통계값(평균, 분산)이 의미 없게 되어 역전 결과가 부정확해질 수 있다. 이를 해결하기 위해 본 논문은 이미지들의 배치를 동시에 역전시키는 방식을 제안한다.

배치 $\mathbf{z}_b = \{z_1, z_2, \dots, z_B\}$에 대한 전체 손실은 개별 손실의 합으로 정의된다:
$$\nabla_{\mathbf{z}_b} L = \frac{\partial \sum_{i=1}^B L_i}{\partial (\mathbf{z}_b)} = \left( \frac{\partial L_1}{\partial z_1}, \frac{\partial L_2}{\partial z_2}, \dots, \frac{\partial L_B}{\partial z_B} \right)$$

배치 크기가 충분히 크다면 배치 통계값이 데이터셋의 상수 파라미터처럼 작동하므로, $z_i$의 업데이트는 오직 자신의 재구성 손실 $L_i$에 의해서만 결정된다. 이는 계산 효율성을 높이는 동시에 Batch Normalization 문제를 완화한다.

### 사전 분포 $P(Z)$의 활용

추론된 $z^*$가 GAN 학습 시 사용된 사전 분포 $P(Z)$의 유효 영역 내에 있도록 하기 위해 다음의 제약 조건을 사용한다.

- **Uniform Distribution $U[a, b]$**: 업데이트 후 $z^*$ 값을 $[a, b]$ 범위로 Clipping한다.
- **Gaussian Distribution $N[\mu, \sigma]$**: 손실 함수에 다음과 같은 Regularization 항을 추가한다.
  $$L(z, x) = \mathbb{E}_x \log[G(z)] + \gamma_1 ||\mu - \hat{\mu}||_2^2 + \gamma_2 ||\sigma - \hat{\sigma}||_2^2$$
  여기서 $\hat{\mu}$와 $\hat{\sigma}$는 배치 내 $z$ 값들의 평균과 표준편차이다.

## 📊 Results

### 실험 설정

- **데이터셋**: MNIST (손글씨 숫자), Omniglot (다양한 알파벳 문자)
- **모델**: 각 데이터셋에 대해 Uniform prior와 Normal prior를 사용한 총 4개의 Generator를 학습시켜 사용하였다.
- **평가 지표**:
  - 정량적 평가: 재구성된 이미지와 원본 이미지 사이의 평균 절대 픽셀 오차(Mean Absolute Pixel Error)를 측정하였다.
  - 정성적 평가: 원본 $x$와 재구성된 $G(z^*)$를 시각적으로 비교하여 정체성(Identity)과 스타일(Style) 보존 여부를 확인하였다.

### 주요 결과

1. **MNIST 결과**:
    - 시각적 분석 결과, 제안된 방법은 숫자의 정체성뿐만 아니라 개별 숫자만이 가진 고유한 스타일까지 매우 잘 보존하며 역전시켰다.
    - 정량적 분석(Table 2) 결과, Clipping이나 Regularization을 적용했을 때 오차가 오히려 약간 증가하거나 큰 차이가 없었다. 이는 사전 분포의 제약 없이도 유효한 역전이 가능함을 시사한다.

2. **Omniglot 결과**:
    - 특히 학습 단계에서 본 적 없는(unseen) 알파벳의 문자를 역전시켰을 때, 매우 날카롭고 세밀한 디테일(작은 원, 에지 등)을 가진 이미지로 재구성됨을 확인하였다.
    - 정량적 분석(Table 3)에서도 Regularization의 효과가 미미하여, $P(Z)$에 의존하지 않는 일반적인 역전이 가능함을 입증하였다.

## 🧠 Insights & Discussion

본 논문의 결과는 GAN의 잠재 공간 $Z$가 단순한 무작위 벡터의 집합이 아니라, 데이터의 핵심적인 특징(정체성 및 스타일)을 매우 효율적으로 인코딩하고 있음을 보여준다.

특히 주목할 점은 **Omniglot 데이터셋에서의 결과**이다. 학습되지 않은 알파벳의 문자가 잠재 공간으로 성공적으로 투영되었다는 것은, GAN이 특정 문자를 외우는 것이 아니라 '문자'라는 개념의 일반적인 시각적 특징을 학습했음을 의미한다. 이는 단 하나의 샘플만으로도 새로운 클래스를 정의할 수 있는 **One-shot Learning** 분야에 GAN의 잠재 공간 투영 기법이 활용될 수 있는 가능성을 강력하게 시사한다.

또한, 실험을 통해 $\text{BCE}$ 손실만으로도 충분히 좋은 결과가 나왔으며, 사전 분포에 기반한 강한 제약(Clipping, Regularization)이 반드시 필요하지 않다는 점을 발견하였다. 이는 사전 분포 $P(Z)$의 경계 바로 바깥 영역에서도 유의미한 데이터 분포가 생성될 수 있음을 암시한다.

## 📌 TL;DR

본 논문은 추가적인 네트워크 학습 없이 경사하강법을 통해 Pre-trained GAN의 Generator를 역전시켜 이미지를 잠재 공간 $Z$로 투영하는 방법을 제안하였다. Batch Inversion을 통해 Batch Normalization 문제를 해결하였으며, MNIST와 Omniglot 실험을 통해 숫자의 스타일 보존 및 미학습 문자의 성공적인 투영을 확인하였다. 이 연구는 GAN의 잠재 공간을 판별 작업이나 One-shot Learning에 활용할 수 있는 실질적인 경로를 제시하였다.
