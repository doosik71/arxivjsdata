# Neural ODE with Temporal Convolution and Time Delay Neural Networks for Small-Footprint Keyword Spotting

Hiroshi Fuketa and Yukinori Morita (2020)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 문제는 모바일 기기나 스마트 스피커와 같이 배터리로 작동하는 임베디드 환경에서의 **Small-Footprint Keyword Spotting (KWS)** 구현이다. KWS는 입력 오디오 데이터에서 미리 정의된 특정 키워드를 탐지하는 기술로, 최근 인공 신경망(NN)을 통해 높은 정확도를 달성하였다.

그러나 이러한 장치들은 메모리 용량과 계산 자원이 매우 제한적이다. 기존의 Convolutional Neural Network (CNN)나 Residual Network (ResNet) 기반 모델들은 높은 정확도를 제공하지만, 모델 파라미터 수가 많아 메모리 점유율이 높다는 한계가 있다. 따라서 본 연구의 목표는 모델의 파라미터 수와 연산량을 획기적으로 줄이면서도 KWS 작업에서 경쟁력 있는 정확도를 유지하는 신경망 모델을 설계하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Neural Ordinary Differential Equation (NODE)**를 KWS 모델에 최초로 적용하여 모델의 깊이를 연속적인 형태로 해석함으로써 파라미터 수를 극단적으로 줄이는 것이다. 주요 기여 사항은 다음과 같다.

1. **NODE 기반 KWS 모델 제안**: TCNN(Temporal Convolutional Neural Network)과 TDNN(Time Delay Neural Network) 구조를 NODE 프레임워크에 통합하여, 수십 개의 레이어를 쌓는 대신 단일 함수를 통한 ODE 풀이 과정으로 대체하였다.
2. **Layer-Dependent Batch Normalization (L-BN) 제안**: NODE의 연속적인 시간 축($t$) 특성상 기존의 Batch Normalization을 적용하기 어려운 문제를 해결하기 위해, 학습 시 층별 통계량을 저장하고 추론 시 선형 보간법을 사용하는 L-BN 기법을 제안하였다.
3. **추론 단계의 오차 허용치(Error Tolerance) 완화**: ODE Solver의 오차 허용치를 조정하여 정확도 손실을 최소화하면서 추론 시 발생하는 연산량을 크게 줄이는 최적화 방법을 제시하였다.
4. **파라미터 효율성 입증**: 제안된 모델이 기존 KWS 모델 대비 파라미터 수를 최대 68%까지 줄일 수 있음을 보였다.

## 📎 Related Works

기존의 KWS 연구들은 주로 다음과 같은 구조를 사용하였다.

- **CNN 및 ResNet**: Sainath와 Parada [2], Tang와 Lin [3] 등이 제안한 방식으로, 높은 정확도를 보이지만 파라미터 수가 많다.
- **TCNN 및 TDNN**: Choi 등 [4]과 Bai 등 [5]이 제안한 방식으로, 시계열 데이터의 특성을 반영하여 파라미터 수를 줄이려 시도하였다.

이러한 기존 방식들은 보통 5~15개의 레이어를 쌓아 올리는 구조를 가진다. 반면, 본 논문에서 채택한 NODE [6]는 Residual Network를 이산화된 ODE로 해석하여, 여러 층의 가중치를 학습하는 대신 하나의 미분 방정식 함수를 학습함으로써 레이어 수를 획기적으로 줄일 수 있다. 다만, 기존 NODE 연구는 MNIST와 같은 단순한 작업에만 적용되었으며, KWS와 같은 복잡한 시계열 작업으로의 확장 가능성과 ODE 풀이로 인한 연산량 증가 문제는 다뤄지지 않았었다.

## 🛠️ Methodology

### 1. 전체 시스템 구조 및 NODE 원리

본 논문은 NODE를 기반으로 한 두 가지 모델(TCNN 기반 및 TDNN 기반)을 제안한다. 기본적으로 입력 데이터인 오디오는 MFCC(Mel-Frequency Cepstrum Coefficients)로 변환되어 입력된다.

Residual Network에서 레이어 $t$에서 $t+1$로의 변환은 다음과 같이 정의된다.
$$p_{t+1} = p_t + f(p_t, \theta_t)$$
여기서 $\Delta t \to 0$으로 수렴한다고 가정하면, 이는 다음과 같은 상미분 방정식(ODE)으로 표현될 수 있다.
$$\frac{dp(t)}{dt} = f(p(t), t, \theta)$$
초기 상태 $p(t=0)$이 주어졌을 때, ODE Solver를 통해 $t=T$에서의 상태 $p(t=T)$를 구함으로써 네트워크의 출력을 얻는다. 기존 ResNet은 $N$개 레이어의 가중치 $\theta_1, \dots, \theta_N$가 필요하지만, NODE는 단일 함수 $f$에 대한 가중치 $\theta$만 필요하므로 파라미터 수가 매우 적다.

- **NODE-TCNN**: MFCC $\to$ CNN $\to$ Average Pooling $\to$ ODE Solver (TCNN 기반 함수 $f$) $\to$ Average Pooling & FC $\to$ Softmax 순으로 진행된다.
- **NODE-TDNN**: MFCC $\to$ TDNN-SUB (Subsampling) $\to$ ODE Solver (TDNN 기반 함수 $f$) $\to$ Average Pooling & FC $\to$ Softmax 순으로 진행된다.

### 2. Layer-Dependent Batch Normalization (L-BN)

기존 Batch Normalization (BN)은 이산적인 레이어 인덱스를 기반으로 평균과 분산을 계산한다. 하지만 NODE에서 $t$는 실수(real number)이므로 기존 방식을 그대로 적용할 수 없다. 또한, 추론 시 미니 배치 사이즈가 1인 경우가 많은 KWS 특성상, 추론 시점에 실시간으로 통계량을 계산하면 정확도가 급격히 떨어진다.

이를 해결하기 위해 **L-BN**을 제안한다.

- **학습 단계**: 각 레이어 $t$에서의 평균 $\mu[p]$와 분산 $\sigma^2[p]$를 계산하여 데이터베이스에 저장한다.
- **추론 단계**: 인덱스 $t$를 사용하여 데이터베이스에서 값을 가져온다. 만약 해당 $t$값이 데이터베이스에 없다면, 인접한 두 레이어의 데이터 포인트 사이에서 **선형 보간법(Linear Interpolation)**을 통해 값을 추정한다.

### 3. 추론 연산량 최적화 (Tolerance Relaxation)

ODE Solver의 오차 허용치(Error Tolerance)는 계산 정밀도를 결정하는 하이퍼파라미터이다. 본 논문은 학습 시에는 정밀한 값($10^{-3}$)을 사용하되, 추론 시에는 이 값을 완화(relax)하여도 정확도 하락이 미비하다는 점을 발견하였다.

- **TCNN 기반 모델**: 허용치를 $0.5$까지 완화하여 연산량(Multiplies)을 57% 감소시켰다.
- **TDNN 기반 모델**: 허용치를 $10^{-2}$까지 완화하여 연산량을 34% 감소시켰다.

## 📊 Results

### 실험 설정

- **데이터셋**: Google Speech Commands Dataset (1초 길이의 30개 단어 중 12개 클래스 분류).
- **특징 추출**: 40-dimensional MFCC (Window 30ms, Stride 10ms).
- **ODE Solver**: Dormand-Prince (DOPRI) 방법 사용.
- **비교 대상**: CNN 기반(trad-fpool3, tpool2), ResNet 기반(res8-narrow, res15), TCNN 기반(tc-resnet8, tc-resnet14-1.5), TDNN 기반(tdnn, swsa) 모델.

### 주요 결과

- **파라미터 수 및 정확도**:
  - `ode-tcnn30` 모델은 약 21k의 파라미터로 93.6%의 정확도를 기록하였으며, 이는 유사한 파라미터 규모의 `res8-narrow` (약 20k, 90.1%)보다 3.5% 높은 성능이다.
  - `ode-tdnn29` 모델은 파라미터 수가 약 6.4k로 매우 작으며, 동일 정확도 수준에서 기존 `res8-narrow` 대비 파라미터 수를 **68% 감소**시켰다.
- **연산량 (Multiplies)**:
  - 제안 모델은 CNN이나 ResNet 기반 모델보다 연산량이 적었으나, TCNN 기반 베이스라인 모델과는 유사한 수준이었다. 이는 ODE Solver가 정답을 찾기 위해 함수 $f$를 여러 번 호출하는 **NFE (Number of Function Evaluations)** 과정이 필요하기 때문이다.

## 🧠 Insights & Discussion

본 논문은 NODE를 KWS에 적용함으로써 **극도의 파라미터 효율성**을 달성할 수 있음을 증명하였다. 특히 L-BN을 통해 NODE의 연속성 문제를 해결하고, 추론 시 오차 허용치를 조절하여 실용적인 연산량 감소를 이끌어낸 점이 돋보인다.

**한계점 및 논의사항**:

- **연산 비용의 병목**: 파라미터 수는 획기적으로 줄었으나, 실제 추론 시의 연산 횟수(Multiplies)는 TCNN 모델들과 비슷하거나 오히려 NFE로 인해 부담이 될 수 있다. 이는 NODE가 가중치 저장 공간은 적게 차지하지만, 계산 과정은 더 복잡할 수 있음을 의미한다.
- **하드웨어 가속의 필요성**: 저자는 이러한 ODE Solver의 연산 오버헤드를 해결하기 위해 전용 하드웨어 가속기가 필요함을 언급하였다. 소프트웨어적인 최적화만으로는 기존의 단순 적층 구조(Stacked layers)보다 연산 속도 면에서 압도적인 우위를 점하기 어렵다는 점이 본 연구의 핵심적인 한계이자 향후 과제이다.

## 📌 TL;DR

본 연구는 **Neural ODE**를 Keyword Spotting에 적용하여, 기존 모델 대비 **파라미터 수를 최대 68%까지 줄인** 초소형 KWS 모델을 제안하였다. 특히 연속적인 시간 축에서 작동하는 NODE를 위해 **L-BN(Layer-Dependent Batch Normalization)**과 **추론 시 오차 허용치 완화** 기법을 도입하여 정확도와 효율성을 동시에 확보하였다. 이 연구는 메모리 제약이 극심한 엣지 디바이스 환경에서 모델 크기를 줄이는 새로운 방향성을 제시하며, 향후 ODE Solver 전용 하드웨어 가속기가 결합될 경우 실시간 KWS 시스템에 매우 중요한 역할을 할 것으로 기대된다.
