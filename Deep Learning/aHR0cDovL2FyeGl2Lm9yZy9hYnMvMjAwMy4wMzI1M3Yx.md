# Introduction to deep learning

Lihi Shiloh-Perl and Raja Giryes (2020)

본 문서는 특정 연구 문제를 해결하기 위한 개별 논문이라기보다, 딥러닝(Deep Learning, DL) 분야의 기초부터 최신 응용 기술까지를 체계적으로 정리한 교육적 성격의 튜토리얼 또는 서술형 리뷰 챕터이다. 따라서 새로운 알고리즘의 제안보다는 기존 기술의 집대성 및 분석에 초점이 맞춰져 있다.

## 🧩 Problem to Solve

본 텍스트가 해결하고자 하는 문제는 딥러닝이라는 방대하고 빠르게 발전하는 분야에 대해 입문자가 필요로 하는 기초 개념, 구조, 학습 절차 및 주요 응용 사례를 하나의 일관된 흐름으로 제공하는 것이다. 딥러닝의 역사적 배경부터 시작하여 하드웨어(GPU)와 데이터의 중요성, 그리고 수학적 기초와 고급 아키텍처까지를 포괄함으로써, 독자가 딥러닝 시스템의 전체 파이프라인을 이해하도록 돕는 것을 목표로 한다.

## ✨ Key Contributions

본 문서의 핵심 기여는 딥러닝의 복잡한 생태계를 다음과 같은 구조적 관점에서 체계적으로 정리했다는 점이다.

1. **역사적 맥락 제공**: Perceptron에서 시작하여 MLP, CNN, LSTM을 거쳐 AlexNet과 GAN에 이르기까지, 'NN Ice Age'와 'NN Winter'를 포함한 신경망의 발전 과정을 설명한다.
2. **기초 구성 요소의 수식화**: 신경망의 기본 빌딩 블록, 활성화 함수, 손실 함수, 그리고 역전파(Backpropagation) 과정을 수학적으로 명확히 정의한다.
3. **학습 최적화 및 정규화 전략 분석**: SGD부터 ADAM에 이르는 최적화 알고리즘과 Overfitting을 방지하기 위한 다양한 Regularization 기법을 상세히 다룬다.
4. **데이터 도메인별 아키텍처 확장**: 이미지(Detection, Segmentation), 시퀀스 데이터(RNN, Transformer), 불규칙 그리드 데이터(3D Point Cloud, Mesh, Graph)에 따른 맞춤형 구조를 분석한다.

## 📎 Related Works

본 문서는 특정 연구와 대립하는 것이 아니라, 수많은 선행 연구들을 인용하여 딥러닝의 흐름을 설명한다.

- **초기 신경망**: Rosenblatt의 Perceptron[105]과 그 한계(XOR 문제)를 지적한 Minsky와 Papert[82]의 연구를 언급한다.
- **부활의 계기**: Multi-Layer Perceptron(MLP)과 Backpropagation[107], 그리고 Convolutional Layer[68]의 도입이 NN의 정체기를 끝냈음을 설명한다.
- **현대적 돌파구**: 2012년 ImageNet 챌린지에서 hand-crafted feature 기반 방식보다 10% 이상의 성능 향상을 보인 AlexNet[62]이 현대 딥러닝의 촉매제가 되었음을 강조한다.
- **생성 모델**: 데이터 분포를 학습하는 GAN[39] 및 고해상도 이미지 생성을 위한 BigGAN[7] 등을 소개한다.

## 🛠️ Methodology

본 문서는 딥러닝의 작동 원리를 다음과 같은 단계적 구조로 설명한다.

### 1. 기본 구조 (Basic Structure)
신경망의 최소 단위는 선형 연산 후 비선형 함수를 적용하는 구조이다. 입력 데이터 $x \in \mathbb{R}^{d_0}$에 대해 출력은 다음과 같이 정의된다.
$$\psi(Wx + b)$$
여기서 $W \in \mathbb{R}^{d_1 \times d_0}$는 가중치(Weight), $b \in \mathbb{R}^{d_1}$는 편향(Bias), $\psi(\cdot)$는 비선형 활성화 함수이다. 이를 $K$개 층으로 쌓은 전체 네트워크의 임베딩 $\Phi$는 다음과 같다.
$$\Phi(x, W^{(1)}, \dots, W^{(K)}, b^{(1)}, \dots, b^{(K)}) = \psi(W^{(K)} \dots \psi(W^{(1)}x + b^{(1)}) \dots + b^{(K)})$$

### 2. 주요 레이어 및 함수
- **Linear Layers**: 모든 뉴런이 연결된 Fully Connected(FC) 레이어와 공간 정보를 유지하는 Convolutional 레이어로 구분한다.
- **Activation Functions**: $\text{ReLU}, \text{Leaky ReLU}, \text{ELU}, \text{Sigmoid}, \text{tanh}$ 등의 수식과 특성을 정의하며, 특히 $\text{ReLU}$ 계열이 CV 분야에서 선호됨을 명시한다.
- **Pooling**: $\text{Max}$, $\text{Mean}$, $L^p$ pooling을 통해 차원을 축소하고 지배적인 특징을 유지한다.
- **Softmax**: 벡터를 확률 분포로 변환하며, 다음과 같이 정의된다.
$$\text{softmax}(v_i) = \frac{e^{v_i}}{\sum_{j=1}^{N} e^{v_j}}$$

### 3. 손실 함수 (Loss Functions)
- **Regression**: MSE(Mean Squared Error), $\ell_1$ loss, SSIM 등을 사용한다.
- **Classification**: Cross-Entropy loss를 사용하며 수식은 다음과 같다.
$$L_{CE} = -\sum_{i=1}^{N} y_i \log(p_i)$$
- **Metric Learning**: 클래스 내 거리는 좁히고 클래스 간 거리는 넓히는 Triplet Loss를 제안한다.
$$L = \sum_{i} \max(0, \|\Phi(x_a^i) - \Phi(x_p^i)\|_2^2 - \|\Phi(x_a^i) - \Phi(x_n^i)\|_2^2 + \alpha)$$

### 4. 학습 절차 및 최적화
- **Backpropagation**: 연쇄 법칙(Chain Rule)을 이용하여 손실 함수의 기울기를 계산하고 가중치를 업데이트한다.
- **Optimizers**: 
    - $\text{SGD}$: 단일 샘플의 기울기만 사용하여 업데이트한다.
    - $\text{Momentum}$: 과거 기울기의 지수 이동 평균을 사용하여 진동을 줄인다.
    - $\text{ADAM}$: 1차 모멘트(평균)와 2차 모멘트(분산)를 모두 사용하여 학습률을 적응적으로 조절한다.
- **Regularization**: $\text{Weight Decay}, \text{Dropout}, \text{Batch Normalization}$ 및 $\text{Data Augmentation}$을 통해 Overfitting을 방지한다.

## 📊 Results

본 문서는 개별 실험을 수행한 논문이 아니므로, 직접적인 실험 결과 표는 제시되지 않는다. 대신, 기존 문헌의 성과를 다음과 같이 인용하여 설명한다.

- **AlexNet의 성과**: ImageNet 데이터셋(훈련 120만 장, 테스트 15만 장)에서 기존 hand-crafted feature 방식 대비 10% 이상의 정확도 향상을 달성하며 딥러닝의 가능성을 입증하였다.
- **3D 데이터 처리**: PointNet이 포인트 클라우드 데이터를 직접 처리하여 만족스러운 결과를 냈으며, 이후 PointNet++ 등으로 발전하였다.
- **성능 지표**: 객체 탐지 및 세그멘테이션 작업에서는 $\text{IoU}(\text{Intersection over Union})$, $\text{mAP}(\text{mean Average Precision})$, $\text{F1-score}$ 등을 통해 정량적 성능을 측정함을 설명한다.

## 🧠 Insights & Discussion

본 문서는 딥러닝의 성공 요인과 여전한 한계점을 동시에 논의한다.

- **성공 요인**: 단순한 알고리즘의 개선보다 **대규모 데이터의 가용성**과 **GPU 연산 능력의 향상**($100\times$ 가속)이 현대 딥러닝의 실질적인 핵심 동력(Key-enabler)이었음을 분석한다.
- **이론적 공백**: 신경망이 실무적으로는 매우 강력한 성능을 보이지만, 최적화, 일반화(Generalization), 표현력(Expressive power)에 대한 수학적/이론적 이해는 여전히 부족한 상태임을 지적한다.
- **데이터 의존성**: 지도 학습(Supervised Learning)의 성능이 높지만 라벨링 비용이 매우 크다는 점을 언급하며, 이를 해결하기 위한 Domain Adaptation, Transfer Learning, Few-shot Learning의 중요성을 강조한다.

## 📌 TL;DR

본 문서는 딥러닝의 역사, 기초 수학, 모델 아키텍처, 학습 최적화, 그리고 최신 응용 분야(3D, NLP, CV)를 총망라한 **딥러닝 종합 가이드라인**이다. 단순한 기술 소개를 넘어, 데이터와 하드웨어의 상호작용 및 이론적 한계까지 다루고 있어, 향후 딥러닝 모델을 설계하거나 새로운 아키텍처를 연구하려는 연구자에게 필수적인 기초 지식 체계를 제공한다.