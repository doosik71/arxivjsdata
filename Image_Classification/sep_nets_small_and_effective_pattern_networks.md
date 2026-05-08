# SEP-Nets: Small and Effective Pattern Networks

Zhe Li, Xiaoyu Wang, Xutao Lv, Tianbao Yang (2017)

## 🧩 Problem to Solve

최근 합성곱 신경망(Convolutional Neural Networks, CNN)은 층이 깊어짐에 따라 성능이 향상되는 경향을 보였으나, 이는 모델의 크기를 비대하게 만들어 모바일 및 임베디드 기기(예: FPGA)와 같은 자원 제한적인 환경에서의 배포를 어렵게 만든다. 특히 FPGA의 경우 온칩 메모리가 매우 제한적이어서 수십 MB에 달하는 기존의 대형 네트워크(ResNet, GoogLeNet 등)를 그대로 탑재하는 것이 불가능하다.

기존의 모델 압축 기술(양자화, 이진화, 가지치기 등)이 연구되어 왔으나, 모델 크기를 극단적으로 줄였을 때 발생하는 심각한 성능 저하(performance drop)가 주요 해결 과제로 남아 있다. 따라서 본 논문의 목표는 모델의 크기를 획기적으로 줄이면서도 대형 및 심층 CNN의 성능을 최대한 유지할 수 있는 효율적인 소형 네트워크 구조와 압축 방법을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 CNN 내에서 $1 \times 1$ 합성곱과 $k \times k$ ($k > 1$) 합성곱이 수행하는 역할이 서로 다르다는 직관에서 출발한다. $1 \times 1$ 합성곱은 데이터의 투영(projection) 및 변환(transformation)을 담당하는 반면, $k \times k$ 합성곱은 공간적인 패턴 추출(pattern extraction)을 담당한다. 이러한 기능적 차이를 바탕으로 다음과 같은 두 가지 핵심 기여를 제시한다.

1. **Pattern Binarization**: 모든 가중치를 이진화하는 기존 방식과 달리, 패턴 추출을 담당하는 $k \times k$ 합성곱 필터만을 이진화하고 $1 \times 1$ 필터는 부동 소수점(floating point) 형태를 유지함으로써 성능 저하를 최소화하며 모델 크기를 압축한다.
2. **Pattern Residual Block (PRB)**: 이진화된 $k \times k$ 합성곱으로 인한 정보 손실을 보완하기 위해, $1 \times 1$ 합성곱을 통한 변환된 특징 맵을 $k \times k$ 합성곱의 결과에 더해주는 새로운 잔차 블록 구조를 제안한다. 이를 통해 소형 네트워크에서도 모델 용량을 높이고 이진화의 부정적 영향을 상쇄한다.

## 📎 Related Works

기존의 연구들은 $1 \times 1$ 합성곱을 사용하여 파라미터 수를 줄이거나(Inception, ResNet), 가중치 가지치기(pruning), 양자화(quantization), 이진화(binarization) 등의 기법을 통해 모델을 압축하려 했다. SqueezeNet과 MobileNet과 같은 소형 네트워크 설계 연구도 진행되었으나, 여전히 모델 크기 대비 성능 효율성 문제가 존재한다.

본 연구는 특히 기존의 가중치 이진화 연구(Binarized Neural Networks 등)와 차별점을 가진다. 기존 방식은 모든 가중치를 이진화하여 학습 단계부터의 계산 비용을 줄이는 데 집중한 반면, 본 논문은 **학습이 완료된 모델을 대상으로 특정 필터($k \times k$)만을 이진화하고 이후 파인튜닝(fine-tuning)하는 2단계 접근법**을 사용하여 학습의 어려움을 피하고 성능 저하를 줄였다.

## 🛠️ Methodology

### 1. Pattern Binarization

본 논문은 임의의 성공적인 네트워크 구조를 압축하기 위해 다음과 같은 절차를 수행한다.

- **Step 1**: 전체 네트워크를 처음부터 학습시킨다.
- **Step 2**: 학습이 완료된 모델에서 $k \times k$ ($k > 1$) 합성곱 필터만을 이진화한다.
- **Step 3**: 모든 이진화된 $k \times k$ 필터의 스케일링 인자(scaling factor)와 모든 $1 \times 1$ 필터의 부동 소수점 표현을 역전파(back-propagation)를 통해 파인튜닝한다.

이진화 과정은 가중치 행렬 $W$를 $\alpha B$로 근사하는 문제로 정의되며, 이때 $B$는 $\{1, -1\}$의 원소를 갖는 이진 필터이고 $\alpha$는 스케일링 인자이다. 최적의 $\alpha, B$를 찾기 위한 목적 함수는 다음과 같다.

$$\min_{\alpha \in \mathbb{R}, B \in \{1, -1\}^{c \times k \times k}} \|W - \alpha B\|_F^2$$

여기서 최적의 $B^*$는 $W$의 부호를 기준으로 결정된다.
$$B^*_{i,j,l} = \begin{cases} 1 & \text{if } W_{i,j,l} \geq 0 \\ -1 & \text{if } W_{i,j,l} < 0 \end{cases}$$
최적의 스케일링 인자 $\alpha^*$는 다음과 같이 계산된다.
$$\alpha^* = \frac{\sum_{i,j,l} |W_{i,j,l}|}{c \times k \times k}$$

### 2. Pattern Residual Block (PRB) 및 SEP-Net 구조

**Pattern Residual Block (PRB)**는 입력 $x$에 대해 $k \times k$ 합성곱과 $1 \times 1$ 합성곱을 병렬로 수행하고 그 결과를 합산하는 구조이다.
$$O(x) = C_{k \times k}(x) + C_{1 \times 1}(x)$$
이 구조에서 $C_{1 \times 1}(x)$는 일종의 선형 매핑으로 작동하며, 특히 $C_{k \times k}$가 이진화되었을 때 발생하는 오차를 보완하는 잔차(residual) 역할을 수행하여 모델의 표현력을 유지한다.

**SEP-Net Module**은 다음과 같은 순서로 구성된다.

- 차원 축소 층 ($1 \times 1$ 합성곱) $\rightarrow$ 두 개의 PRB 블록 (서로 다른 출력 채널) $\rightarrow$ 차원 복원 층 ($1 \times 1$ 합성곱).
- 마지막 복원 층은 ResNet과 유사한 스킵 연결(skip connection)을 가능하게 하여 더 깊은 층을 쌓을 수 있게 한다.

**Group-wise Convolution**: 파라미터 수를 추가로 줄이기 위해 모든 합성곱 층에 그룹 합성곱을 적용한다. 입력 특징 맵을 $N$개의 그룹으로 나누어 처리함으로써, 단일 합성곱 대비 파라미터 수를 $1/N$로 줄인다. 본 논문에서는 $N=4$를 적용하였다.

## 📊 Results

### 1. CIFAR-10 실험

ResNet-20, 32, 44, 56 모델에 Pattern Binarization을 적용한 결과, 파인튜닝된 모델(Refined model)의 정확도가 원본 모델(Full model)과 매우 유사하게 유지됨을 확인하였다. 특히 ResNet-56의 경우, 파라미터 수를 약 86% 감소시켰음에도 성능 저하가 매우 적었다.

### 2. ImageNet 실험

GoogLeNet을 대상으로 필터 크기별 이진화 영향을 분석한 결과, $1 \times 1$ 필터를 이진화했을 때의 성능 저하가 $3 \times 3$ 또는 $5 \times 5$ 필터를 이진화했을 때보다 훨씬 컸다. 이는 $1 \times 1$ 필터가 데이터 투영이라는 중요한 역할을 수행하므로 부동 소수점 형태를 유지해야 한다는 본 논문의 가설을 뒷받침한다.

### 3. SOTA 모델과의 비교 (SEP-Net vs MobileNet vs SqueezeNet)

제안된 SEP-Net을 다양한 설정(SEP-Net-R: 원본 가중치, SEP-Net-B: 패턴 이진화, SEP-Net-BQ: 이진화 + 8비트 양자화)으로 실험하여 비교하였다.

- **정확도**: 1.3M 파라미터를 가진 SEP-Net-R은 ImageNet에서 65.8%의 Top-1 정확도를 기록하여, 동일한 크기의 MobileNet(63.7%)과 SqueezeNet(60.4%)보다 높은 성능을 보였다.
- **모델 크기**:
  - **SEP-Net-B (Small)**: 패턴 이진화를 통해 모델 크기를 4.2MB로 줄였을 때 63.7%의 정확도를 유지하였다.
  - **SEP-Net-BQ (Small)**: 추가로 $1 \times 1$ 필터를 8비트로 양자화하여 모델 크기를 **1.3MB**까지 줄였으며, 이때의 정확도는 63.5%로 MobileNet(5.2MB, 63.7%)과 대등한 성능을 달성하였다.

## 🧠 Insights & Discussion

본 논문은 CNN의 연산을 '데이터 변환($1 \times 1$)'과 '패턴 추출($k \times k$)'이라는 두 가지 관점에서 재정의함으로써 효율적인 압축 전략을 제시하였다. 특히, 모든 가중치를 동일하게 처리하지 않고 필터의 역할에 따라 차별적인 양자화 전략을 적용한 점이 매우 효과적이었다.

또한, PRB 구조는 단순히 층을 깊게 쌓는 것이 아니라, 이진화로 인해 발생하는 정보 손실을 $1 \times 1$ 합성곱 층이 보완하도록 설계되었다는 점에서 이론적 타당성을 갖는다. 다만, 본 논문은 이미 학습된 모델을 이진화하는 2단계 방식을 사용하였는데, 이는 학습 안정성은 높으나 처음부터 이진화된 네트워크를 최적화하여 학습시키는 방식에 비해 잠재적인 최적 성능에 도달하지 못했을 가능성이 있다.

## 📌 TL;DR

이 논문은 CNN의 $k \times k$ 필터만을 이진화하는 **Pattern Binarization**과 이를 보완하는 **Pattern Residual Block**을 제안하여, 성능 저하를 최소화하면서 모델 크기를 획기적으로 줄인 **SEP-Net**을 구현하였다. 특히 1.3MB 크기의 모델로 MobileNet 수준의 정확도를 달성함으로써, 극도로 제한된 메모리 환경(FPGA 등)에서의 딥러닝 모델 배포 가능성을 입증하였다.
