# Trainable Highly-expressive Activation Functions

Irit Chelly, Shahaf E. Finder, Shira Ifergane, and Oren Freifeld (2024)

## 🧩 Problem to Solve

딥러닝 모델의 성공에는 비선형성을 제공하는 활성화 함수(Activation Function, AF)의 선택이 결정적인 역할을 한다. 그러나 현재 대부분의 네트워크는 ReLU, GELU와 같은 고정된(fixed) 활성화 함수를 사용하며, 이는 모델의 표현력(expressiveness)을 제한하고 특정 학습 편향(learning bias)을 강제하는 문제가 있다.

기존의 학습 가능한 활성화 함수(Trainable Activation Functions, TAFs)인 PReLU, Swish 등은 일부 파라미터를 통해 형태를 조정할 수 있으나, 기본적으로 고정된 함수 형태를 계승하므로 표현력의 이득이 미미하다. 반면, Maxout Unit과 같은 방식은 표현력을 크게 높일 수 있지만, 네트워크의 뉴런 수에 비례하여 파라미터 수가 급격히 증가한다는 치명적인 단점이 있다. 따라서 본 논문은 **파라미터 증가를 최소화하면서도 매우 높은 표현력을 갖는 학습 가능한 활성화 함수를 설계**하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 미분 가능한 가역 함수인 **미분동형사상(Diffeomorphism)**, 특히 효율적인 **CPAB(CPA-Based) 변환**을 활성화 함수에 도입하는 것이다.

중심적인 설계 직관은 다음과 같다. 기존의 TAF들이 특정 형태(예: 선형 결합, 볼록 함수)로 제한된 것과 달리, CPAB 기반의 변환을 사용하면 매우 다양한 형태의 비선형 함수를 학습할 수 있다. 또한, CPAB는 소수의 파라미터만으로도 높은 표현력을 제공하며, 닫힌 형태(closed-form)의 수식과 기울기(gradient) 계산이 가능하여 기존 모델 아키텍처에 쉽게 통합될 수 있다.

## 📎 Related Works

### 기존 활성화 함수 및 한계

- **고정 AF (Sigmoid, Tanh, ReLU, GELU 등):** 계산 효율성은 높으나 표현력이 제한적이며, Sigmoid와 Tanh는 기울기 소실(vanishing gradient) 문제가 있다. ReLU 계열은 이를 일부 해결했으나 여전히 고정된 형태라는 한계가 있다.
- **기존 TAF (PReLU, Swish, PDELU 등):** 기본 함수의 형태를 유지한 채 일부 파라미터만 조정하므로 표현력 향상이 제한적이다.
- **고발현 TAF (Maxout, ACON 등):** 표현력은 높으나 파라미터 수가 채널 수나 뉴런 수에 비례하여 증가하여 모델의 효율성을 저하시킨다.

### CPAB 변환과의 차별점

CPAB 변환은 기존 연구들에서 주로 이미지의 공간적 도메인(Spatial Domain)이나 시계열의 시간적 도메인(Temporal Domain)을 변형하는 Spatial Transformer Net(STN) 등의 구조에서 사용되었다. 본 논문은 이를 최초로 **특징 맵의 값 범위(Range of feature maps)에 요소별(element-wise)로 적용하여 활성화 함수로 사용**함으로써 차별성을 갖는다.

## 🛠️ Methodology

### 1. CPAB 변환의 기초

CPAB 변환 $T_\theta$는 유한 구간 $\Omega = [a, b]$에서 정의되며, 연속 조각-선형(Continuous Piecewise-Affine, CPA) 속도장(velocity field) $v_\theta$의 적분을 통해 정의된다.

$$ \phi_\theta(x; t) = x + \int_0^t v_\theta(\phi_\theta(x; \tau)) d\tau $$

여기서 $T_\theta$는 $t=1$일 때의 결과값 $\phi_\theta(x; 1)$이며, 이는 단조 증가하는 미분동형사상(diffeomorphism)이 된다. $\Omega$의 분할(partition)이 세밀할수록 $T_\theta$의 표현력은 높아진다.

### 2. DiTAC 활성화 함수 구조

제안된 **DiTAC**은 CPAB 변환 $T_\theta$를 최신 AF인 GELU와 결합하여 구축한다. GELU는 $\text{GELU}(x) = x \cdot \Phi(x)$ (여기서 $\Phi$는 표준 정규 분포의 누적 분포 함수)로 정의된다.

DiTAC의 메인 버전은 다음과 같이 정의된다:
$$ \text{DiTAC}(x) = \tilde{x} \cdot \Phi(x), \quad \tilde{x} = \begin{cases} T_\theta(x) & \text{if } a \le x \le b \\ x & \text{otherwise} \end{cases} $$

입력값 $x$가 설정된 구간 $[a, b]$ 내에 있을 때만 학습 가능한 CPAB 변환을 적용하고, 구간 밖에서는 기존의 값을 유지하여 GELU의 특성을 따른다. 이 외에도 Leaky-ReLU와 결합한 Leaky-DiTAC, GELU를 음수 영역에만 적용한 GE-DiTAC, 양 끝단에서 아핀 변환을 확장한 inf-DiTAC 등의 변형 버전이 존재한다.

### 3. 정규화 및 학습 안정화

학습 과정에서 너무 극단적인 변환이 학습되는 것을 방지하기 위해 속도장에 대한 정규화 항을 추가한다.
$$ \mathcal{L}_{\text{reg}} = \sum_{l=1}^L \theta_l^T \Sigma_{\text{CPA}}^{-1} \theta_l $$
여기서 $\Sigma_{\text{CPA}}^{-1}$는 가우시안 평활도 사전 확률(Gaussian smoothness prior)과 관련된 공분산 행렬이며, 이를 통해 변환의 부드러움(smoothness)을 제어한다.

### 4. 계산 비용 절감 전략 (Lookup Table)

CPAB 변환을 모든 텐서 요소에 직접 적용하는 것은 계산 비용이 매우 크다. 이를 해결하기 위해 본 논문은 **양자화(Quantization) 기반의 룩업 테이블(Lookup Table, LUT)** 방식을 제안한다.

1. 구간 $[a, b]$를 균일하게 양자화하여 $n$개(보통 $2^8$)의 이산 값으로 나눈다.
2. 이 값들에 대해서만 CPAB 변환을 계산하여 LUT를 생성한다.
3. 실제 입력값 $x_i$에 대해서는 $\text{LUT}$를 참조하여 값을 출력한다.
4. 역전파 시에는 **Straight Through Estimator (STE)**를 사용하여, 양자화된 지점의 미분값을 입력값의 미분값으로 근사하여 전달한다.

이 방식을 통해 학습 시 계산 비용을 획기적으로 줄이며, 추론(inference) 시에는 단순히 테이블을 참조하므로 고정 AF와 동일한 수준의 효율성을 갖는다.

## 📊 Results

### 1. 토이 데이터 실험 (Toy Data)

- **분류 작업:** 2D-GMM 및 MNIST 데이터셋을 사용한 MLP 실험에서 DiTAC은 다른 AF/TAF들보다 높은 정확도를 보였다. 특히 2D-GMM에서 DiTAC은 클래스 간 경계를 훨씬 더 정교하게 학습함을 확인하였다 (표 1).
- **회귀 작업:** 1차원 및 2차원 함수 재구성 실험에서 DiTAC은 MSE를 낮추고 $R^2$ 점수를 높이며 압도적인 성능을 보였다. 시각화 결과, PReLU 등은 형태의 경직성으로 인해 부드러운 함수를 맞추는 데 어려움이 있었으나 DiTAC은 유연하게 적합되었다 (표 2).

### 2. 실제 데이터 실험 (Real-World Data)

- **이미지 분류:** ImageNet-50 (MobileNet-V3, ResNet) 및 ImageNet-100/1K (ConvNeXt-T, Swin-T) 실험에서 DiTAC은 GELU 및 다른 TAF들보다 일관되게 높은 Top-1 정확도를 기록하였다. ConvNeXt-T와 Swin-T에서는 각각 0.3%, 0.2%의 성능 향상을 보였다 (표 3, 4).
- **시맨틱 세그멘테이션:** ADE20K 및 Cityscapes 데이터셋에서 UperNet, PSPNet 프레임워크에 적용한 결과, baseline인 ReLU 및 GELU보다 높은 mIoU를 달성하였다 (표 5).
- **이미지 생성:** DCGAN(CelebA) 및 BigGAN(CIFAR-10) 실험에서 FID $\downarrow$ 및 IS $\uparrow$ 지표 모두에서 DiTAC이 가장 우수한 성능을 보였으며, 특히 DCGAN에서 GELU 대비 비약적인 성능 향상을 보였다 (표 6).

### 3. 효율성 분석

추론 시의 지연 시간(Latency)은 약간 증가하지만, 파라미터 수와 FLOPs의 증가는 무시할 수 있는 수준이다. 이는 DiTAC이 실용적인 관점에서도 도입 가능하다는 것을 시사한다 (표 10).

## 🧠 Insights & Discussion

### 강점 및 해석

DiTAC의 가장 큰 강점은 **최소한의 파라미터 추가만으로 극도로 유연한 비선형성**을 확보했다는 점이다. 실험 결과에서 알 수 있듯이, 단순한 분류/회귀부터 복잡한 생성 모델에 이르기까지 범용적으로 성능 향상을 이끌어냈다. 이는 각 레이어와 태스크마다 최적의 활성화 함수 형태가 다르다는 가설을 뒷받침하며, DiTAC이 데이터에 맞게 그 형태를 스스로 최적화할 수 있음을 보여준다.

### 한계 및 논의사항

- **계산 비용의 이론적 문제:** LUT 방식이 없었다면 CPAB 변환의 계산 비용이 매우 높았을 것이나, 본 논문은 이를 양자화와 STE로 해결하였다. 다만, 양자화 과정에서 발생하는 정보 손실이 극히 일부의 정밀한 작업에서 영향을 줄 가능성이 있다.
- **하이퍼파라미터 의존성:** 구간 $[a, b]$의 설정과 정규화 계수 $\lambda_{\text{var}}, \lambda_{\text{smooth}}$ 등의 하이퍼파라미터가 학습 결과에 영향을 줄 수 있다.

## 📌 TL;DR

본 논문은 미분동형사상(Diffeomorphism)의 일종인 CPAB 변환을 도입한 학습 가능한 활성화 함수 **DiTAC**을 제안한다. DiTAC은 매우 적은 수의 파라미터만 추가하면서도 기존의 고정 AF나 제한적인 TAF보다 훨씬 높은 표현력을 가지며, 이를 통해 이미지 분류, 세그멘테이션, 이미지 생성 등 다양한 컴퓨터 비전 작업에서 성능을 향상시킨다. 특히 룩업 테이블(LUT) 방식을 통해 학습 및 추론 효율성을 확보함으로써 실제 딥러닝 아키텍처에 즉시 적용 가능한 실용적인 대안을 제시하였다.
