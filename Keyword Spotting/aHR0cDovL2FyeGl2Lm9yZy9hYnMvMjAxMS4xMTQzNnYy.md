# Speech Command Recognition in Computationally Constrained Environments with a Quadratic Self-organized Operational Layer

Mohammad Soltanian, Junaid Malik, Jenni Raitoharju, Alexandros Iosifidis, Serkan Kiranyaz, Moncef Gabbouj (2021)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 연산 자원이 제한된 임베디드 환경에서 효율적으로 동작하는 음성 명령 인식(Speech Command Recognition, SCR) 시스템을 구축하는 것이다. 일반적으로 최신 음성 인식 모델은 딥러닝 기반의 복잡한 네트워크 구조를 가지며, 이는 막대한 메모리와 에너지 소비를 야기한다. 따라서 로봇 공학이나 스마트 홈 어시스턴트와 같이 실시간성이 중요하고 하드웨어 자원이 한정된 기기에서는 이러한 무거운 모델을 그대로 구현하기 어렵다.

논문의 목표는 모델의 크기를 억지로 줄이는 'Squeezing' 방식이 아니라, 처음부터 연산 효율성이 높은 경량 네트워크를 설계하되, 제안하는 새로운 네트워크 레이어를 통해 인식 정확도를 높임으로써 계산 복잡도와 성능 사이의 절충안을 제시하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 **Self-organized Operational Neural Networks (SelfONNs)**와 **Quadratic-form Kernels**를 결합하여 입력 및 은닉층에서 더욱 풍부한 특징 표현(feature representation)을 생성하는 새로운 네트워크 레이어를 제안하는 것이다.

전통적인 Convolutional Neural Network (CNN)의 선형 연산은 데이터의 복잡한 비선형 관계를 포착하는 데 한계가 있다. 이를 해결하기 위해 Taylor expansion(테일러 전개)을 통해 함수를 근사하는 SelfONN의 아이디어와, 입력 요소 간의 상호 상관관계(cross-correlation)를 모델링하는 이차 형식(Quadratic form)의 커널을 결합함으로써, 경량 네트워크에서도 깊은 네트워크에 필적하는 표현력을 갖게 하여 인식 정확도를 향상시키고자 하였다.

## 📎 Related Works

논문에서는 기존의 SCR 접근 방식을 다음과 같이 구분하여 설명한다.

1.  **전통적 방식**: FFT(Fast Fourier Transform)와 같은 주파수 영역 특징을 추출하고, HMM(Hidden Markov Model)을 통해 단어 또는 음소의 시퀀스를 표현하였다.
2.  **딥러닝 기반 방식**: CNN, LSTM, GRU 등의 구조가 도입되었으며, 특히 raw audio에 CNN을 직접 적용하는 방식 등이 연구되었다. 하지만 이러한 모델들은 임베디드 장치에서 구동하기에는 연산 및 메모리 비용이 너무 높다는 한계가 있다.
3.  **경량화 전략**: 복잡한 모델을 압축하는 방식과, 처음부터 효율적인 모델을 설계하는 방식(예: POPs, GOPs, Depth-wise separable convolutions)으로 나뉜다.

본 논문은 특히 SelfONN의 생성적 뉴런(generative neurons) 개념을 차용하여, 고정된 연산자 세트에서 탐색하는 대신 학습 과정에서 최적의 노달 함수(nodal function)를 스스로 적응시키는 방식을 채택하여 기존 ONN의 느린 학습 속도와 초기 설정 의존성 문제를 해결하였다.

## 🛠️ Methodology

### 1. 기본 개념 및 구성 요소

제안하는 방법론을 이해하기 위해 먼저 기반이 되는 두 가지 개념을 정의한다.

**A. Self-organized Operational Neural Networks (SelfONNs)**
일반적인 CNN의 컨볼루션 연산 $Y = w^T x + b$를 테일러 급수 전개로 일반화한다. 각 뉴런은 다음과 같은 형태의 함수를 근사한다.
$$\Psi(x, w) = \sum_{q=1}^{Q} w_q^T x^q$$
여기서 $x^q$는 요소별 거듭제곱(element-wise power)을 의미하며, $w_q$는 학습 가능한 가중치이다. 즉, 단순 선형 결합이 아닌 고차항의 합으로 입력 특징을 처리한다.

**B. Quadratic-form Kernels**
입력 특징 간의 상호 상관관계를 포착하기 위해 다음과 같은 이차 형식의 연산을 수행한다.
$$Y = x^T \Omega x + w^T x + b$$
여기서 $\Omega$는 블록 대각 행렬(block-diagonal matrix) 형태의 커널이며, 이를 통해 수용 영역(receptive field) 내의 입력 요소들 사이의 관계를 모델링한다.

### 2. Proposed Quadratic SelfONN Layer

본 논문은 위 두 개념을 병합하여 다음과 같은 최종 연산 식을 제안한다.
$$Y_{c_{out}}(i, j) = \sum_{q=1}^{Q} (x^q(i, j))^T \Omega_{c_{out}, q} x^q(i, j) + \sum_{q=1}^{Q} w_{c_{out}, q}^T x^q(i, j) + b_{c_{out}}$$

이 식의 의미는 다음과 같다.
-   **첫 번째 항 (Quadratic term)**: 테일러 전개로 생성된 각 차수($q$)의 특징 벡터에 대해 이차 형식 연산을 수행하여, 고차원 특징 공간에서의 상호 상관관계를 학습한다.
-   **두 번째 항 (Linear term)**: 각 차수($q$)의 특징 벡터에 대해 선형 가중치를 곱하여 기본 특징을 유지한다.
-   **학습 파라미터**: $\Omega_{c_{out}, q}$와 $w_{c_{out}, q}$는 모두 경사 하강법을 통해 학습되는 파라미터이다.

### 3. 시스템 파이프라인 및 학습 절차
-   **전처리**: raw audio $\rightarrow$ 30ms 윈도우(10ms 오버랩) $\rightarrow$ MFCC 추출 $\rightarrow$ 20개의 계수 유지 $\rightarrow$ $[-1, 1]$ 범위로 정규화. 최종 입력 크기는 $20 \times 51$의 2D 신호이다.
-   **네트워크 구조**: LeNet-1과 유사한 매우 가벼운 구조를 사용한다. [Proposed Layer $\rightarrow$ Max Pooling $\rightarrow$ Tanh] 구조의 컨볼루션 층 2개와 Fully Connected 층 1개로 구성된다.
-   **학습 설정**: SGD (momentum=0.9, lr=0.01), 배치 사이즈 50, 커널 크기 $3 \times 3$, 최대 100 에포크 학습을 수행한다.

## 📊 Results

### 1. 실험 환경 및 데이터셋
-   **GSC (Google Speech Commands)**: 10개 클래스(on, off, yes, no 등)의 서브셋을 사용하였다.
-   **SSC (Synthetic Speech Commands)**: GSC의 TTS 버전으로, 노이즈가 심한 버전(very noisy version)을 사용하여 강건성을 평가하였다.
-   **지표**: 테스트 세트의 예측 정확도(Accuracy)와 추론 시간(Deployment time)을 측정하였다.

### 2. 정량적 결과
실험 결과, 제안된 Quadratic SelfONN 레이어가 일반 컨볼루션 및 SelfONN 레이어보다 높은 성능을 보였다.

| 데이터셋 | Ordinary Convolution | SelfONN | Proposed (Quadratic SelfONN) |
| :--- | :---: | :---: | :---: |
| **GSC** | 85.2% | 87.6% | **89.8%** |
| **SSC** | 95.5% | 97.1% | **97.9%** |

-   **정확도 향상**: 일반 CNN 대비 GSC에서는 약 4.6%, SSC에서는 약 2.2%의 성능 향상이 있었다. SelfONN 대비로도 GSC 2.4%, SSC 0.8%의 이득을 얻었다.
-   **테일러 차수($Q$)의 영향**: $Q$가 증가함에 따라 정확도가 상승하다가 특정 지점 이후 포화(saturation)되는 경향을 보였다.
-   **연산 비용**: 추론 시간은 일반 CNN보다는 높지만, SelfONN과는 매우 유사한 수준으로 나타나 경량 네트워크로서의 실용성을 입증하였다.

## 🧠 Insights & Discussion

### 1. 강점 및 의의
본 논문은 매우 얕고 가벼운 네트워크 구조(LeNet-1 기반)에서도 레이어 수준의 표현력을 높임으로써 성능을 유의미하게 끌어올릴 수 있음을 보여주었다. 특히 데이터가 오염된 노이즈 환경(SSC)에서도 안정적인 성능 향상을 보였다는 점은 임베디드 환경의 실제 적용 가능성을 시사한다.

### 2. 한계 및 비판적 해석
-   **모델 복잡도**: 비록 경량 네트워크를 사용했으나, 레이어 내부의 연산(이차 형식 및 테일러 전개)이 복잡하므로, 단순 CNN 대비 파라미터 수나 연산량의 정확한 증가분이 명시되지 않은 점이 아쉽다.
-   **비교 대상의 제한**: Kaggle 리더보드의 최상위 모델(약 91%)과 비교했을 때 성능 차이가 존재한다. 저자들은 이를 모델의 깊이와 입력 해상도의 차이로 설명하고 있으나, 제안한 레이어가 더 깊은 네트워크에서도 동일한 효율성을 유지할지에 대한 검증은 추가적으로 필요하다.
-   **가정**: 본 연구는 MFCC라는 고전적인 특징 추출 방식을 전제로 한다. End-to-end raw waveform 학습 방식과 비교했을 때 어느 정도의 효율성을 갖는지에 대한 논의가 부족하다.

## 📌 TL;DR

본 논문은 연산 자원이 제한된 환경에서 음성 명령 인식 성능을 높이기 위해, **테일러 전개(SelfONN)**와 **이차 형식 커널(Quadratic Kernels)**을 결합한 새로운 네트워크 레이어를 제안한다. 이 레이어는 입력 데이터의 고차원적 특징과 요소 간 상호 상관관계를 동시에 포착하여, 매우 가벼운 네트워크 구조에서도 일반 CNN 대비 GSC 데이터셋에서 약 4.6%의 정확도 향상을 달성하였다. 이는 향후 로봇 및 임베디드 기기의 저전력·고성능 음성 인식 시스템 구현에 중요한 기초 연구가 될 것으로 평가된다.