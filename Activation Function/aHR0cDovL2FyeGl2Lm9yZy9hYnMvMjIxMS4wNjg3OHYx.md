# EVALUATING CNN WITH OSCILLATORY ACTIVATION FUNCTION

Jeevanshi Sharma (2022)

## 🧩 Problem to Solve

본 논문은 Convolutional Neural Network(CNN)의 성능을 결정짓는 핵심 요소 중 하나인 활성화 함수(Activation Function)의 효율성을 탐구한다. CNN이 이미지로부터 고차원의 복잡한 특징을 학습할 수 있는 이유는 활성화 함수가 제공하는 비선형성(Non-linearity)에 있다. 특히, 기존의 표준적인 활성화 함수들이 가진 한계를 극복하기 위해 인간의 뇌 피질에서 영감을 받은 진동 활성화 함수(Oscillatory Activation Function)의 효용성을 검증하고자 한다.

본 연구의 구체적인 목표는 oscillatory activation function의 일종인 GCU(Growing Cosine Unit)를 AlexNet 아키텍처에 적용하여, MNIST와 CIFAR-10 데이터셋에서 기존의 널리 사용되는 활성화 함수들(ReLU, PReLU, Mish)과 비교해 어느 정도의 성능을 보이는지 분석하는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 아이디어는 생물학적 신경망의 특성을 모방한 진동 활성화 함수를 인공신경망에 도입하는 것이다. 특히 GCU(Growing Cosine Unit)의 핵심 직관은 단일 평면의 결정 경계(Decision Boundary)를 갖는 일반적인 활성화 함수와 달리, 무한히 일정한 간격으로 배치된 하이퍼플레인(Hyperplanes)을 생성한다는 점이다. 이러한 설계는 신경망이 더 복잡한 결정 경계를 형성할 수 있게 하며, 특히 기존 신경망이 해결하기 어려웠던 XOR 문제를 효과적으로 해결할 수 있는 잠재력을 가진다.

## 📎 Related Works

논문에서는 CNN의 비선형성을 부여하기 위해 사용되어 온 다양한 활성화 함수들을 소개한다.

1.  **ReLU (Rectified Linear Unit):** 기울기 소실(Vanishing Gradient) 문제를 완화하여 가장 널리 사용되지만, 음수 입력에 대해 출력이 0이 되어 뉴런이 죽는 'Dying ReLU' 문제가 존재한다.
2.  **PReLU (Parametric ReLU):** ReLU의 단점을 보완하기 위해 음수 영역에 학습 가능한 파라미터를 도입하여 적응형 상수를 통해 정확도를 높이려 한다.
3.  **Mish:** 비단조성(Non-monotonicity)과 매끄러운 프로파일을 가지며, 양수 영역에서는 무제한이고 음수 영역에서는 유한한 범위를 갖는 특성을 보인다.

기존의 접근 방식들이 주로 단일 평면 기반의 결정 경계를 생성하는 데 집중했다면, 본 논문에서 다루는 GCU는 생물학적 뉴런의 진동 특성을 반영하여 여러 개의 하이퍼플레인을 형성함으로써 인공 신경망과 생물학적 신경망 사이의 성능 격차를 줄이려는 차별점을 가진다.

## 🛠️ Methodology

### 전체 파이프라인 및 구조
본 연구는 CNN 아키텍처인 AlexNet을 기반으로 실험을 수행한다. 전체적인 시스템 구조는 Convolution layer(Conv), Down-sampling layer(Pooling), 그리고 Full-connection layer(FC)로 구성된다.

### 활성화 함수의 배치 및 적용
실험에서는 활성화 함수의 계산 비용을 고려하여 다음과 같이 배치하였다.
- **Convolutional Layers:** GCU, ReLU, PReLU, Mish를 각각 적용하여 성능을 비교한다.
- **Dense (Fully Connected) Layers:** 모든 실험에서 ReLU를 고정적으로 사용한다. 이는 특히 Mish와 같은 함수가 ReLU보다 계산 비용이 높기 때문에 효율성을 위해 Dense layer에는 ReLU만을 배치한 설정이다.

### 학습 절차 및 손실 함수
- **데이터셋:** MNIST 및 CIFAR-10.
- **최적화 알고리즘 (Optimizer):** CIFAR-10 데이터셋에는 SGD를 사용하였고, MNIST 데이터셋에는 Adam을 사용하였다.
- **손실 함수 (Loss Function):** Softmax 분류 헤드와 함께 Sparse Categorical Crossentropy 손실 함수를 사용하였다.
- **학습 횟수 (Epochs):** CIFAR-10은 50 epoch, MNIST는 40 epoch 동안 학습을 진행하였다.

## 📊 Results

본 논문은 AlexNet 구조에서 각 활성화 함수를 적용했을 때의 검증 정확도(Validation Accuracy), 테스트 정확도(Test Accuracy), 그리고 손실 값(Loss)을 측정하였다.

### MNIST 데이터셋 결과
MNIST 실험 결과, ReLU, PReLU, Mish 모두 매우 높은 테스트 정확도(약 $0.99$ 이상)를 기록하였다. GCU의 경우 테스트 정확도가 $0.9791$로 측정되어, 기존 함수들보다는 약간 낮지만 전반적으로 경쟁력 있는 성능을 보여주었다.

### CIFAR-10 데이터셋 결과
CIFAR-10 실험에서는 활성화 함수 간의 성능 차이가 더 명확하게 나타났다.
- **ReLU, PReLU, Mish:** 테스트 정확도가 각각 $0.9777, 0.9765, 0.9802$로 매우 높게 측정되었다.
- **GCU:** 테스트 정확도가 $0.8256$으로 기록되어, 다른 활성화 함수들에 비해 현저히 낮은 성능을 보였다.

결과적으로 MNIST와 같은 단순한 데이터셋에서는 GCU가 유사한 성능을 내지만, CIFAR-10과 같이 복잡한 데이터셋에서는 기존의 ReLU 기반 함수들이 GCU보다 훨씬 우수한 성능을 보임을 확인하였다.

## 🧠 Insights & Discussion

본 논문은 생물학적 영감을 받은 GCU가 CNN의 성능에 미치는 영향을 분석하였다. GCU의 이론적 강점은 무한한 하이퍼플레인을 통해 XOR 문제와 같은 복잡한 결정 경계를 쉽게 생성할 수 있다는 점에 있다.

그러나 실제 실험 결과, 표준적인 이미지 분류 작업(특히 CIFAR-10)에서는 GCU가 ReLU, PReLU, Mish보다 낮은 성능을 기록하였다. 이는 진동 활성화 함수가 이론적으로는 복잡한 매핑이 가능할 수 있으나, 실제 CNN의 합성곱 층에서 특징을 추출하는 과정에서는 기존의 단조 증가 또는 매끄러운 비단조 함수들이 더 안정적인 학습을 제공함을 시사한다.

또한, 저자는 결론에서 GCU가 "비교 가능한(comparable)" 성능을 보였다고 언급하였으나, 수치적으로는 CIFAR-10에서 상당한 성능 차이가 발생하였으므로 이에 대한 추가적인 분석(예: 과적합 여부, 학습률 민감도 등)이 필요해 보인다.

## 📌 TL;DR

본 연구는 진동 활성화 함수인 GCU(Growing Cosine Unit)를 AlexNet에 적용하여 MNIST와 CIFAR-10 데이터셋에서 성능을 평가하였다. 실험 결과, MNIST에서는 기존 함수들과 유사한 성능을 보였으나, CIFAR-10에서는 ReLU, PReLU, Mish에 비해 낮은 정확도를 기록하였다. 이는 GCU가 이론적인 복잡성 표현 능력에도 불구하고, 일반적인 이미지 분류 작업에서는 기존 활성화 함수들보다 효율성이 떨어질 수 있음을 보여준다. 향후 연구에서는 GCU의 하이퍼파라미터 최적화나 다른 아키텍처에서의 적용 가능성을 탐구하는 것이 중요할 것으로 보인다.