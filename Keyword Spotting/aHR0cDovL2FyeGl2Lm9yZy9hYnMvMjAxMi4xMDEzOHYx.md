# Resource-efficient DNNs for Keyword Spotting using Neural Architecture Search and Quantization

David Peter, Wolfgang Roth, Franz Pernkopf (2020)

## 🧩 Problem to Solve

본 논문은 자원이 제한된 환경(limited resource environments)에서 동작하는 키워드 검출(Keyword Spotting, KWS) 시스템을 위한 효율적인 심층 신경망(DNN) 모델을 설계하는 문제를 다룬다. 일반적으로 자동 음성 인식(ASR) 시스템은 계산 복잡도가 높아 항상 켜져 있는(always-on) 모드로 작동할 경우 모바일 기기의 배터리를 빠르게 소모시킨다. 따라서 낮은 전력으로 특정 키워드만 상시 감시하다가, 키워드가 검출되었을 때만 전체 ASR 시스템을 깨우는 저비용 KWS 시스템의 필요성이 매우 크다.

이 연구의 목표는 KWS 시스템이 갖추어야 할 세 가지 핵심 요구사항인 **자원 효율성(resource-efficiency)**, **실시간 동작(real-time execution)**, 그리고 **높은 정확도(accuracy)**를 동시에 만족하는 최적의 모델 구조를 자동으로 탐색하고, 가중치 양자화(weight quantization)를 통해 메모리 사용량을 극단적으로 줄이는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **미분 가능한 신경 구조 탐색(Differentiable Neural Architecture Search, NAS)**과 **Straight-Through Estimator (STE) 기반의 양자화**를 결합하여, 사람이 직접 설계하지 않고도 하드웨어 제약 조건에 최적화된 소형 모델을 자동으로 찾는 것이다. 특히, 정확도와 연산량(number of operations) 사이의 트레이드-오프(trade-off)를 최적화 목표로 설정하여, 다양한 연산 규모에 맞는 최적의 아키텍처를 도출하였다. 또한, 가중치를 1비트까지 낮추는 극단적인 양자화 상황에서도 성능 저하를 최소화할 수 있는 학습 방법을 제시하였다.

## 📎 Related Works

기존의 NAS 접근 방식은 강화 학습(RL), 경사 하강법(Gradient-based), 진화 알고리즘(Evolutionary methods) 등을 통해 탐색 공간을 탐색한다. 특히 많은 NAS 알고리즘이 연산 비용을 줄이기 위해 작은 데이터셋이나 짧은 학습 epoch를 사용하는 **Proxy task**에 의존하는데, 이는 타겟 태스크에서 최적이 아닐 수 있다는 한계가 있다. 본 논문은 이러한 한계를 극복하기 위해 타겟 태스크에서 직접 구조를 탐색하는 **ProxylessNAS** 방식을 채택하여 프록시 태스크 없이 직접 최적화를 수행한다.

KWS 분야에서는 DS-CNN(Depthwise Separable CNN)과 같은 모델들이 효율적인 베이스라인으로 사용되어 왔으며, 가중치를 8비트 고정 소수점으로 양자화하여 마이크로컨트롤러에 배포하는 방식이 일반적이다. 또한, 가중치와 활성화 함수를 $\{-1, 1\}$로 제한하는 이진 신경망(Binarized Neural Networks, BNNs) 연구가 진행되어 왔으며, 본 논문은 여기서 사용되는 STE 기법을 양자화 과정에 도입하여 성능을 유지한다.

## 🛠️ Methodology

### 1. 전체 시스템 구조 및 NAS 파이프라인
본 연구는 3단계(stage)로 구성된 CNN 아키텍처를 사용한다.
- **Stage (i):** 입력 단계로 $5 \times 11$ Convolution을 사용하여 고정된다.
- **Stage (ii):** 중간 단계로, **Mobile Inverted Bottleneck Convolutions (MBCs)** 블록들로 구성되며 이 단계의 구조가 NAS를 통해 최적화된다.
- **Stage (iii):** 출력 단계로 $1 \times 1$ Convolution 및 Global Average Pooling, Fully Connected 레이어로 구성되며 고정된다.

MBC 블록은 확장률(expansion rate) $e \in \{1, 2, 3, 4, 5, 6\}$와 커널 크기 $k \in \{3, 5, 7\}$라는 두 가지 학습 가능한 파라미터를 가지며, NAS는 이 중 최적의 조합 또는 zero operation(레이어 건너뛰기)을 선택한다.

### 2. NAS 학습 절차 및 방정식
NAS는 과매개변수화된 네트워크(overparameterized network)에서 시작하며, 각 연산 $o_i$에 대해 아키텍처 파라미터 $\alpha_i$를 할당한다. 이 파라미터는 Softmax 함수를 통해 선택 확률 $p_i$로 변환된다.

$$p_i = \frac{\exp(\alpha_i)}{\sum_{j} \exp(\alpha_j)}$$

이 확률에 따라 이진 게이트 $g_i$가 샘플링되어 최종 출력 $m_{Binary}^O$가 결정된다.

$$m_{Binary}^O = \sum_{i=1}^{N} g_i o_i(x)$$

가중치 파라미터와 아키텍처 파라미터 $\alpha_i$는 교대로 학습된다. 가중치 학습 시에는 $\alpha_i$를 고정하고, 아키텍처 학습 시에는 가중치를 고정하고 검증 세트에서 $\alpha_i$를 업데이트한다. 이때 샘플링 과정의 미분 불가능성을 해결하기 위해 STE를 사용하여 경사도를 근사한다.

### 3. 정확도와 연산량의 트레이드-오프 목적 함수
단순히 정확도만 높이는 것이 아니라 연산량을 제어하기 위해 다음과 같은 정규화된 손실 함수를 사용한다.

$$\text{loss}_{arch} = \text{CE}_{loss} \cdot \left( \frac{\log(\text{ops}_{exp})}{\log(\text{ops}_{target})} \right)^\beta$$

여기서 $\text{CE}_{loss}$는 교차 엔트로피 손실, $\text{ops}_{exp}$는 예상 연산량, $\text{ops}_{target}$은 목표 연산량, $\beta$는 정규화 파라미터이다. $\beta$ 값을 조정함으로써 다양한 연산 규모의 모델들을 생성할 수 있다.

### 4. 가중치 양자화 (Weight Quantization)
실수 값 가중치 $w$를 $k$-bit 정수 $\mathbf{w}^q$로 변환하기 위해 다음과 같은 균등 양자화(Uniform Quantization) 식을 사용한다.

$$\mathbf{w}^q = 2 \cdot \left[ \frac{1}{2^k-1} \text{round}\left((2^k-1)w + \frac{1}{2}\right) \right]_{0}^{1} - 1$$

양자화 방법은 두 가지로 비교되었다.
- **Post-processing bit-rounding:** 학습 완료 후 가중치를 반올림하여 양자화하는 방식.
- **Quantization Aware Training (QAT) with STE:** 학습 과정의 Forward pass에서는 양자화된 가중치를 사용하고, Backward pass에서는 양자화 함수의 미분값을 identity function의 미분값(1)으로 대체하여 가중치를 업데이트하는 방식.

## 📊 Results

### 1. 실험 설정
- **데이터셋:** Google Speech Commands (10개 핵심 단어 + unknown + silence 클래스).
- **입력 특징:** MFCC(Mel-frequency cepstral coefficients)를 사용하며, 기본적으로 10개의 특징을 추출한다.
- **데이터 증강:** 최대 100ms의 랜덤 시간 시프트 및 배경 소음 추가.

### 2. NAS 기반 아키텍처 성능
NAS를 통해 발견한 모델들은 기존의 Hello Edge DS-CNN 베이스라인과 비교하여 대등하거나 더 우수한 성능을 보이면서도 연산량을 획기적으로 줄였다. 예를 들어, 10 MFCC를 사용한 모델 중 하나는 95.4%의 정확도를 기록하면서 메모리 494.8 kB, 연산량 19.6M ops를 달성하였다.

### 3. 양자화 비트 수에 따른 영향
STE 기반의 QAT는 매우 낮은 비트 수에서도 성능 유지가 탁월했다. 8비트에서 1비트로 양자화했을 때 정확도 하락이 약 0.9%에 불과했다. 반면, 사후 처리 방식(post-processing)은 4비트 미만으로 내려갈 때 정확도가 급격히 하락하는 모습을 보였다.

### 4. MFCC 특징 수의 영향
입력 특징인 MFCC의 수를 10개에서 20개로 늘렸을 때 성능 향상이 뚜렷했다. 20 MFCC를 사용한 최적 모델은 **96.3%의 정확도, 340.1 kB 메모리, 27.1M ops**를 기록하여 베이스라인을 큰 차이로 앞질렀다. 30, 40 MFCC로 늘릴수록 정확도는 소폭 상승하지만 연산량과 메모리 사용량이 크게 증가하는 경향을 보였다.

## 🧠 Insights & Discussion

본 연구는 NAS를 통해 사람이 설계한 모델보다 더 효율적인 구조를 찾을 수 있음을 입증하였다. 특히 주목할 점은 **STE 기반의 양자화 학습이 극한의 저비트(1-bit) 환경에서도 모델의 표현력을 유지**할 수 있게 한다는 것이다. 이는 KWS 모델을 초소형 임베디드 장치나 마이크로컨트롤러에 탑재할 때 메모리 제약을 극복할 수 있는 강력한 방법이 된다.

또한, 모델 아키텍처 최적화만큼이나 **입력 특징(MFCC)의 차원을 결정하는 것이 성능에 결정적인 영향**을 미친다는 점을 확인하였다. 특징 수를 10에서 20으로 늘리는 것만으로도 아키텍처 변경보다 더 큰 정확도 이득을 얻을 수 있었다. 다만, 본 논문에서는 활성화 함수(activations)의 양자화는 다루지 않고 가중치 양자화에만 집중하였는데, 실제 하드웨어 가속을 위해서는 활성화 함수의 양자화 역시 필수적이므로 향후 연구 과제로 남는다.

## 📌 TL;DR

이 논문은 자원 제한 환경의 키워드 검출(KWS)을 위해 **미분 가능한 NAS(ProxylessNAS)**와 **STE 기반 양자화**를 적용하여 초소형·고효율 CNN 모델을 자동으로 탐색하는 방법을 제안한다. 실험 결과, 1비트 가중치 양자화에서도 성능 저하를 최소화하였으며, 특히 MFCC 특징 수를 20개로 최적화하여 정확도 96.3%의 고성능 소형 모델을 구현하였다. 이 연구는 하드웨어 제약이 심한 엣지 디바이스용 음성 인식 모델 설계에 있어 자동화된 구조 탐색과 양자화 학습의 중요성을 시사한다.