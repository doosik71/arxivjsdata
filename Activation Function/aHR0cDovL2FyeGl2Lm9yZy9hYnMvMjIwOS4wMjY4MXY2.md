# How important are activation functions in regression and classification? A survey, performance comparison, and future directions

Ameya D. Jagtap, George Em Karniadakis

## 🧩 Problem to Solve

인공 신경망(ANN)의 핵심 구성 요소인 활성화 함수는 생물학적 뉴런에서 영감을 받아 학습 과정에서 필수적인 역할을 수행합니다. 분류 및 회귀 작업을 위한 다양한 활성화 함수들이 제안되었지만, 특정 문제에 최적의 활성화 함수를 찾는 것은 여전히 어려운 과제입니다. 특히 과학 계산 분야에서 부상하고 있는 물리학 기반 기계 학습(Physics-Informed Machine Learning, PIML) 프레임워크에서는 미분 가능성에 대한 엄격한 요구 사항으로 인해 활성화 함수의 선택이 더욱 중요합니다. 이 논문은 기존의 활성화 함수들을 종합적으로 조사하고, 분류 및 PIML 회귀 문제에서 이들의 성능을 체계적으로 비교하여, 각 함수의 장단점 및 적합성을 평가하는 것을 목표로 합니다.

## ✨ Key Contributions

- 고전적(고정) 활성화 함수(정류 장치, 진동 함수, 비표준 함수 포함)에 대한 종합적인 조사를 수행하고, 특정 작업에 가장 적합한 속성을 논의했습니다.
- 과학 계산 관점에서 응용 기반 활성화 함수 분류(실수값, 복소수값, 양자화 활성화)를 제안하고, 각각의 응용 분야와 이점을 설명했습니다.
- 최신 적응형 활성화 함수(고전적 함수보다 성능 우수, 훈련 가속화 및 예측 정확도 향상)에 대한 심층 논의를 제공하고, MNIST, CIFAR-10, CIFAR-100 데이터셋에 대한 고정 및 적응형 활성화 함수의 체계적인 성능 비교를 수행했습니다.
- 물리학 기반 기계 학습(PIML) 프레임워크에서 활성화 함수의 요구 사항을 논의하고, 다양한 고정 및 적응형 활성화 함수를 사용하여 여러 편미분 방정식(PDE)을 해결했습니다. TensorFlow, PyTorch, JAX와 같은 기계 학습 라이브러리를 사용한 예측 정확도 및 런타임 비교를 통해 PIML에 최적화된 활성화 함수를 식별했습니다.

## 📎 Related Works

이 논문은 초기 McCulloch and Pitts의 임계 논리 단위(Threshold Logic Unit)와 Rosenblatt의 퍼셉트론(Perceptron)부터 시작하여 Ivakhnenko and Lapa의 다층 신경망, Rumelhart et al.의 역전파(Backpropagation) 알고리즘에 이르는 신경망 발전 역사를 조명합니다. 활성화 함수의 중요성을 강조하며, Cybenko와 Hornik이 제시한 보편 근사(Universal Approximation)를 위한 비선형성, 유계성, 연속성 등의 바람직한 특성을 언급합니다.

초창기 인기 있었던 Sigmoid (Han and Moraga, 1995) 및 Hyperbolic Tangent (LeCun et al., 2012) 함수들의 기울기 소실(Vanishing Gradient) 문제를 지적하며, 이를 극복하기 위해 등장한 ReLU (Glorot et al., 2011; Maas et al., 2013)와 그 다양한 변형들(Leaky ReLU, Parametric ReLU, ELU, GELU, Swish, Mish 등)을 상세히 다룹니다.

또한, Agostinelli et al. (2014), Jagtap et al. (2020)의 전역 및 지역 적응형 활성화 함수, Gulcehre et al. (2016)의 잡음이 있는 활성화 함수, Ivanov (2018)의 분수 미분 기반 활성화 함수, 그리고 Xu and Zhang (2000)의 앙상블 활성화 함수와 같이 최신 연구 동향인 적응형(Adaptive) 및 학습 가능한(Trainable) 활성화 함수들을 광범위하게 참조합니다. 특히, Jagtap et al. (2022)의 Kronecker Neural Networks (KNN)는 적응형 활성화 함수의 일반적인 프레임워크로 소개됩니다.

물리학 기반 기계 학습(PIML) 분야에서는 Karniadakis et al. (2021)의 종합적인 리뷰와 Raissi et al. (2019)의 Physics-Informed Neural Networks (PINNs)와 같은 선구적인 연구들이 인용되어 활성화 함수가 과학적 문제 해결에 미치는 영향력을 강조합니다.

## 🛠️ Methodology

이 논문은 활성화 함수에 대한 심층적인 분석을 위해 다음과 같은 다단계 방법론을 채택했습니다.

1. **활성화 함수 분류**:
   - **특징 기반 분류**: 활성화 함수를 고정(Fixed), 적응형(Adaptive), 비표준(Non-standard) 세 가지 주요 범주로 나눕니다. 적응형 함수는 다시 매개변수 방식(Parametric)과 앙상블(Ensemble) 방식으로 세분화됩니다.
   - **응용 기반 분류**: 활성화 함수를 실수값(Real-valued), 복소수값(Complex-valued), 양자화(Quantized) 세 가지 범주로 분류하고, 각 유형의 과학 및 공학 응용 분야를 설명합니다.
2. **활성화 함수 조사 및 분석**:
   - **고정 활성화 함수**: 선형, 계단, Sigmoid (및 Bipolar-Sigmoid, Elliott, Scaled Sigmoid 등 변형), Hyperbolic Tangent (및 Scaled tanh, Hexpo, LiSHT 등 변형), ReLU (및 Leaky ReLU, Parametric ReLU, CReLU, ELU, GELU, Swish, Mish 등 변형), Softplus, Radial, Wavelet, Oscillatory, Maxout, Softmax 등 다양한 고정 활성화 함수들의 수학적 정의, 유도함수, 범위, 연속성 차수를 제시하고, 각 함수의 장단점과 특정 작업에 적합한 속성을 분석합니다.
   - **적응형 활성화 함수**: 매개변수 방식(예: PReLU, SReLU, P-TELU, Flexible ReLU, Elastic ELU 등), 확률론적/확률적 방식(예: Noisy activation, ProbAct), 분수 미분 기반 방식, 앙상블 방식(예: 다중 활성화 함수의 조합) 등 다양한 적응형 활성화 함수들의 원리와 구현 방식을 설명합니다.
3. **분류 작업 성능 비교**:
   - **데이터셋**: MNIST, CIFAR-10, CIFAR-100.
   - **모델**: MobileNet과 VGG16 (Convolutional Neural Network 기반).
   - **설정**: 학습률 $10^{-4}$, 배치 크기 (MNIST 64, CIFAR-10 128, CIFAR-100 64), Adam 옵티마이저, 교차 엔트로피 손실 함수.
   - **실험**: PyTorch 환경에서 10회 반복 실험 후 평균 정확도와 표준 편차를 보고합니다.
4. **물리학 기반 기계 학습(PIML) 성능 비교**:
   - **문제**: 선형 대류 방정식, 점성 버거스 방정식, 부시네스크 방정식 (미분 차수가 1차에서 4차까지 증가하는 PDE).
   - **방법**: Physics-Informed Neural Networks (PINNs)를 사용하며, 네트워크 출력이 데이터와 미분 방정식 형태의 물리 법칙을 모두 만족하도록 손실 함수를 구성합니다. 활성화 함수의 고차 미분 가능성 요구 사항을 중점적으로 분석합니다.
   - **훈련**: Adam 옵티마이저로 10,000회 반복 후 L-BFGS 옵티마이저를 사용하여 수렴까지 훈련합니다. 10회 반복 실험 후 상대 $L_2$ 오차의 평균과 표준 편차를 보고합니다.
5. **ML 라이브러리 비교 (PIML)**:
   - **라이브러리**: TensorFlow (TF2), PyTorch, JAX (JIT 컴파일러 포함).
   - **문제**: 4차 미분 항을 포함하는 부시네스크 방정식.
   - **데이터**: 깨끗한 데이터와 5% 가우시안 잡음이 추가된 데이터셋에 대해 성능을 비교합니다.
   - **지표**: 예측 정확도($L_2$ 오차)와 런타임을 측정합니다.
6. **함수 근사 및 헬름홀츠 방정식 해결**: 고주파수 성분과 불연속성을 포함하는 1D 함수 근사 및 2D 헬름홀츠 방정식 해결을 통해 활성화 함수의 성능을 추가로 비교합니다.

## 📊 Results

- **분류 작업 (MNIST, CIFAR-10, CIFAR-100)**:
  - MNIST 데이터셋의 경우 대부분의 활성화 함수가 높은 정확도를 보였습니다.
  - CIFAR-10 및 CIFAR-100에서는 Swish, ReLU, Leaky ReLU, ELU, Sine과 같은 활성화 함수가 좋은 성능을 나타냈습니다.
  - 특히, **적응형(Adaptive) 및 Rowdy 활성화 함수는 모든 분류 데이터셋에서 고정 활성화 함수보다 전반적으로 우수한 성능**을 달성했습니다.
- **물리학 기반 기계 학습(PIML) 작업**:
  - PDE 해결에 있어 Sine, Tanh, Swish는 모든 테스트 사례에서 일관되게 좋은 성능을 보였습니다.
  - Sigmoid는 고차 미분 방정식에서 미분값이 급격히 작아져 (Figure 14 참조) 성능이 크게 저하되었습니다.
  - ReLU 및 Leaky ReLU는 특정 지점에서 미분 불가능하여 PIML 문제에 부적합했으며, ELU는 1차 미분까지는 사용 가능했으나 고차 미분이 필요한 문제에는 한계가 있었습니다.
  - **적응형 및 Rowdy 활성화 함수는 고정형 활성화 함수에 비해 더 나은 예측 정확도**를 보여주었습니다.
- **ML 라이브러리 성능 비교 (PIML)**:
  - 부시네스크 방정식 해결 시, JAX (JIT 포함)는 TensorFlow (TF2) 및 PyTorch에 비해 깨끗한 데이터와 노이즈가 있는 데이터 모두에서 **모든 활성화 함수에 대해 예측 정확도를 약 10배 정도 향상**시켰습니다 (Table VI, VII 참조).
  - JAX는 TensorFlow 및 PyTorch보다 **계산적으로 훨씬 효율적**인 런타임을 기록했습니다 (Figure 15 참조).
  - 적응형 활성화 함수는 고정형에 비해 8-10%의 추가 비용이 들었으며, Rowdy 활성화 함수는 75-80%의 비용 증가가 있었습니다.
- **함수 근사 및 헬름홀츠 방정식**:
  - 고주파수 성분과 불연속성을 가진 함수 근사 및 헬름홀츠 방정식 해결에서 **Rowdy 활성화 함수가 다른 활성화 함수들보다 월등히 우수한 성능**을 보이며, '스펙트럼 편향' 문제 극복에 효과적임을 입증했습니다 (Table VIII 참조).

## 🧠 Insights & Discussion

- **활성화 함수의 문제 의존성**: 활성화 함수는 신경망 학습의 핵심이지만, 분류, 회귀, 특히 PIML과 같은 다양한 문제 유형에 따라 최적의 선택이 달라집니다. '이상적인' 단일 활성화 함수는 존재하지 않으며, 문제의 특성을 고려한 신중한 선택이 필요합니다.
- **PIML의 미분 가능성 요구사항**: PIML 프레임워크는 물리 법칙(미분 방정식)을 엄격하게 적용해야 하므로, 활성화 함수가 고차 연속 미분 가능해야 합니다. ReLU 및 그 변형은 이미지 분류에서 뛰어난 성능을 보였지만, 미분 불가능 지점 때문에 PIML에는 부적합합니다. 반면 Sine, Tanh, Swish와 같은 연속 미분 가능한 함수들이 PIML 회귀 문제에서 좋은 성능을 보였습니다.
- **적응형 활성화 함수의 강점**: 적응형 활성화 함수는 고정형보다 계산 비용이 높지만, 네트워크 훈련을 가속화하고 예측 정확도를 향상시키는 이점이 있습니다. 특히 다중 스케일 문제나 고주파수 성분을 포함하는 문제에서 스펙트럼 편향(Spectral Bias) 문제를 극복하는 데 Rowdy 활성화 함수와 같은 적응형 함수가 효과적임을 확인했습니다.
- **JAX의 PIML 효율성**: JAX 라이브러리는 자동 미분(AD)과 JIT 컴파일 기능을 통해 PIML 문제 해결에서 TensorFlow 및 PyTorch 대비 월등히 높은 예측 정확도와 계산 효율성을 제공합니다. 이는 고차 미분 연산이 빈번한 과학적 문제 해결에 JAX가 강력한 도구임을 시사합니다.
- **사전 지식 활용**: PIML에서 필드 변수의 알려진 범위(예: 유체 밀도는 항상 양수)와 같은 사전 지식을 출력층 활성화 함수 선택에 반영하면 네트워크의 수렴 속도를 크게 향상시킬 수 있습니다.

## 📌 TL;DR

신경망의 활성화 함수는 학습과 예측에 핵심적인 역할을 하지만, 문제에 따라 최적의 함수가 달라집니다. 이 논문은 고정형 및 적응형 활성화 함수의 광범위한 조사와 함께, 분류(MNIST, CIFAR) 및 물리학 기반 기계 학습(PIML) 작업에서의 성능을 비교합니다. ReLU 계열은 분류에 강력하지만, 미분 가능성 요구 사항으로 인해 PIML에는 Sine, Tanh, Swish가 더 적합함을 발견했습니다. 특히 **적응형 및 Rowdy 활성화 함수가 전반적으로 우수한 성능**을 보였으며, PIML에서는 JAX 라이브러리가 뛰어난 정확도와 효율성을 제공합니다. 대규모 모델과 데이터셋을 위해 양자화 적응형 활성화 함수 개발과 새로운 적응 전략 모색이 미래 연구 방향으로 제시됩니다.
