# Neural Networks and Deep Learning

Deepak Alapatt, Pietro Mascagni, Vinkle Srivastav, Nicolas Padoy (2020)

## 🧩 Problem to Solve

본 논문(책의 챕터)은 현대 인공지능의 핵심인 Deep Neural Networks(DNN)의 이론적 배경과 실제 구현 방법을 외과 의사 및 의료 전문가들에게 전달하고자 한다. 수술은 데이터 집약적이며 매우 높은 위험(high-stake)을 수반하는 분야이기에, 이러한 계산적 방법론의 도입은 환자의 예후와 의료 시스템에 큰 이점을 줄 수 있다. 하지만 컴퓨터 과학자와 외과 의사 간의 지식 격차로 인해 실제 임상 적용에 어려움이 존재한다. 따라서 본 연구의 목표는 외과 의사들이 신경망의 직관적인 원리를 이해하고, Deep Learning(DL)의 핵심 개념과 과제, 구현 과정 및 수술 분야에서의 특수한 한계점을 파악하도록 돕는 교육적 가이드를 제공하는 것이다.

## ✨ Key Contributions

본 연구의 중심적인 기여는 복잡한 딥러닝 이론을 외과적 관점에서 재구성하여 체계적으로 설명했다는 점이다. 특히 단순한 이론 나열에 그치지 않고, 인공 뉴런의 수학적 기초부터 합성곱 신경망(Convolutional Neural Networks, CNN), 순환 신경망(Recurrent Neural Networks, RNN)과 같은 고급 아키텍처, 그리고 이를 수술 영상 분석(Tool detection, Phase recognition 등)에 어떻게 적용하는지에 대한 전체 파이프라인을 제시한다. 또한, 의료 데이터의 특수성(데이터 부족, 개인정보 보호, 설명 가능성)을 고려한 실질적인 해결 방안과 최신 최적화 기법을 함께 다룸으로써 임상 적용을 위한 가교 역할을 수행한다.

## 📎 Related Works

논문은 생물학적 뉴런 연구에서 시작된 인공 신경망의 역사적 흐름을 설명한다. 1943년 McCulloch와 Pitts의 수학적 모델과 1949년 Donald Hebb의 시냅스 가소성 개념이 기초가 되었으며, 1962년 Hubel과 Wiesel의 시각 피질 계층 구조 연구가 현대 CNN의 핵심인 계층적 특징 추출(Hierarchical processing)의 영감이 되었음을 명시한다. 

이후 1958년의 Perceptron, 1998년의 LeNet, 2012년의 AlexNet으로 이어지는 발전 과정을 다룬다. 특히 AlexNet이 GPU를 사용하여 대규모 데이터셋(ImageNet)을 학습함으로써 딥러닝의 실용성을 입증한 전환점이 되었음을 강조한다. 수술 분야에서는 Unet을 이용한 해부학적 구조의 픽셀 단위 세그멘테이션이나, AlexNet의 확장판인 EndoNet을 이용한 수술 워크플로우 분석 등이 기존 연구로 소개되고 있다.

## 🛠️ Methodology

### 1. 인공 뉴런과 신경망의 구조
인공 뉴런은 다차원 입력 $\mathbf{x} = [x_1, x_2, \dots, x_n]$을 하나의 출력값으로 매핑하는 기본 단위이다. 뉴런의 연산은 가중치(Weights) $w$, 편향(Bias) $b$, 그리고 활성화 함수(Activation function) $f$로 구성되며, 그 수식은 다음과 같다.

$$z = w_1x_1 + w_2x_2 + \dots + w_nx_n + b$$
$$\text{Output} = f(z)$$

활성화 함수는 비선형성을 추가하여 복잡한 실세계 데이터를 학습하게 하며, 대표적으로 ReLU, Sigmoid, Tanh, Step function이 사용된다. 다수의 뉴런이 연결된 Fully Connected Layer의 경우, 전체 연산은 행렬 곱셈으로 효율적으로 계산될 수 있다.

### 2. Convolutional Neural Networks (CNN)
고차원 이미지 데이터의 공간적 정보 유지와 파라미터 수 감소를 위해 Convolutional layer가 도입된다. 필터(Filter)가 입력 이미지 위를 슬라이딩하며 특징을 추출하는 방식이며, 이후 Pooling layer를 통해 차원을 축소하고 중요한 정보만을 남긴다. 최종 출력단에서는 다중 클래스 분류를 위해 Softmax 함수를 사용하여 확률 분포를 생성한다.

$$\text{Softmax}(z_i) = p_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

### 3. 학습 절차 및 최적화
신경망 학습은 Forward Propagation을 통해 예측값을 계산하고, 손실 함수(Loss function) $\mathcal{L}$을 통해 정답(Ground Truth)과의 오차를 측정하는 것으로 시작한다. 이후 Backpropagation을 통해 각 파라미터에 대한 그래디언트(Gradient)를 계산하며, Gradient Descent 알고리즘을 통해 손실을 최소화하는 방향으로 가중치를 업데이트한다.

$$w_{t+1} = w_t - \eta \nabla \mathcal{L}(w)$$

여기서 $\eta$는 학습률(Learning rate)이라는 하이퍼파라미터이다. 또한, 과적합(Overfitting)을 방지하기 위해 다음과 같은 기법들이 사용된다.
- **L2 Regularization**: 손실 함수에 가중치의 제곱합을 추가하여 가중치 값이 지나치게 커지는 것을 방지한다.
  $$\mathcal{L}_{L2} = \mathcal{L}_{orig} + \lambda \sum_{i=1}^{n} \|w_i\|^2$$
- **Dropout**: 학습 시 무작위로 일부 뉴런을 제외하여 특정 경로에 대한 의존도를 낮춘다.
- **Data Augmentation**: 회전, 이동 등의 변형을 통해 학습 데이터의 양을 인위적으로 늘린다.

### 4. 특수 목적 아키텍처
- **Semantic Segmentation**: Encoder-Decoder 구조를 통해 저차원에서 의미론적 정보를 추출하고 다시 원래 해상도로 복원하여 픽셀 단위 분류를 수행한다.
- **GAN (Generative Adversarial Networks)**: Generator와 Discriminator 두 네트워크가 서로 경쟁하며 실제와 유사한 가공의 데이터를 생성한다.
- **Temporal Models (RNN, LSTM, GRU)**: 수술 영상과 같은 시계열 데이터를 처리하기 위해 '기억' 능력을 가진 루프 구조를 추가한다. 특히 LSTM과 GRU는 Vanishing/Exploding Gradient 문제를 해결하기 위해 게이트(Gate) 메커니즘을 사용한다.

## 📊 Results

본 문서는 특정 실험의 결과물을 제시하는 연구 논문이 아니라 딥러닝의 원리와 적용법을 다루는 교육적 챕터이므로, 새로운 실험 데이터나 정량적 수치는 제시되지 않는다. 다만, 딥러닝이 수술 분야에서 다음과 같은 작업들에 성공적으로 적용될 수 있음을 설명한다.

- **Classification**: 병변의 양성/악성 구분, 수술 도구의 존재 여부 판단.
- **Detection**: 수술 도구의 위치를 Bounding box로 지정.
- **Semantic Segmentation**: 혈관 네트워크나 장기의 정밀한 영역 추출.
- **Temporal Recognition**: 수술 단계(Surgical phase)의 식별 및 수술 숙련도 평가.

또한, 전이 학습(Transfer Learning)을 통해 ImageNet으로 사전 학습된 모델을 Cholec80(담낭 절제술 데이터셋)과 같은 소규모 수술 데이터셋에 미세 조정(Fine-tuning)하는 것이 실무적으로 매우 효과적임을 언급한다.

## 🧠 Insights & Discussion

본 보고서는 딥러닝을 수술에 적용할 때 직면하는 현실적인 문제점들에 대해 깊이 있는 논의를 제공한다.

첫째, **데이터의 희소성과 어노테이션 비용**이다. 의료 데이터는 확보가 어렵고, 특히 수술 단계와 같은 고수준의 라벨링은 숙련된 외과 의사의 시간이 필요하므로 비용이 매우 높다. 이를 위해 적은 데이터로 학습 가능한 Weakly-supervised learning이나 합성 데이터 생성 기법이 필요하다.

둘째, **블랙박스(Black-box) 문제**이다. 의료 분야에서는 결정의 근거가 명확해야 하므로, 모델의 예측 근거를 시각화하는 Saliency maps나 Class Activation Maps (CAM)와 같은 설명 가능한 AI(XAI) 기술의 도입이 필수적이다.

셋째, **실시간성 및 하드웨어 제약**이다. 수술실(OR) 내에서 실시간 지원을 위해서는 MobileNet이나 EfficientNet과 같은 경량화 아키텍처, 그리고 연산 효율을 높이는 Mixed-precision computation의 활용이 중요함을 시사한다.

## 📌 TL;DR

본 연구는 외과 의사들이 딥러닝의 수학적 기초부터 최신 아키텍처(CNN, RNN, GAN 등)까지 이해하고 이를 수술 영상 분석에 적용할 수 있도록 구성된 종합 가이드이다. 데이터 부족, 설명 가능성, 실시간 연산이라는 의료 AI의 3대 난제를 해결하기 위한 방법론을 제시하며, 향후 컴퓨터 과학자와 의료진의 협업이 실제 환자의 예후 개선으로 이어질 수 있는 기술적 토대를 마련하였다.