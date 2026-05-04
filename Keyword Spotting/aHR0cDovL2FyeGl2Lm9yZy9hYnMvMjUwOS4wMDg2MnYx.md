# Speech Command Recognition Using LogNNet Reservoir Computing for Embedded Systems

Yuriy Izotov and Andrei Velichko (2025)

## 🧩 Problem to Solve

본 논문은 마이크로컨트롤러(MCU) 기반의 임베디드 시스템에서 음성 명령 인식(Speech Command Recognition)을 구현할 때 발생하는 자원 제약 문제를 해결하고자 한다. 스마트 홈 자동화, 로보틱스, 산업용 보조 장치 등에서 핸즈프리 제어 인터페이스의 수요가 증가하고 있으나, 정교한 신경망 모델을 전력과 메모리가 제한된 임베디드 플랫폼에 탑재하는 것은 매우 어렵다.

특히, 기존의 딥러닝 모델(CNN, RNN 등)은 높은 인식 정확도를 제공하지만, 연산량과 메모리 요구량이 많아 ARM Cortex-M0+와 같은 저사양 마이크로컨트롤러에서 실시간으로 동작시키기에 한계가 있다. 따라서 본 연구의 목표는 최소한의 계산 자원을 사용하면서도 높은 인식 정확도를 유지하는 저전력·저사양 음성 명령 인식 파이프라인을 설계하고, 이를 실제 하드웨어에서 검증하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Reservoir Computing 기반의 LogNNet 분류기와 최적화된 MFCC 특징 추출 파이프라인을 결합하여, 극도로 제한된 자원 환경에서도 효율적인 음성 인식 시스템을 구축한 것이다.

가장 중심적인 설계 아이디어는 다음과 같다. 첫째, 연산 비용이 높은 심층 신경망 대신 카오스 동역학 시스템(Chaotic Dynamical Systems)을 활용한 LogNNet 구조를 채택하여 파라미터 수를 획기적으로 줄였다. 둘째, MFCC 특징 행렬을 1차원 벡터로 변환하는 네 가지 집계(Aggregation) 방식을 비교 분석하여, 정확도와 압축률의 균형이 가장 뛰어난 Adaptive Binning 방식을 도출하였다. 셋째, 이를 통해 ARM Cortex-M0+ 프로세서를 탑재한 Arduino Nano 33 IoT 보드에서 단 18 KB의 RAM만으로 약 90%의 실시간 인식 정확도를 달성함으로써 하드웨어적 실현 가능성을 입증하였다.

## 📎 Related Works

음성 명령 인식 분야에서는 Convolutional Neural Networks (CNNs)와 Depthwise Separable CNNs (DSCNNs)가 널리 사용되며, 특히 DSCNN은 낮은 전력 소모와 빠른 추론 시간으로 90% 이상의 정확도를 달성한 사례가 보고되었다. 또한, temporal dependencies를 모델링하기 위해 RNN이나 CRNN이 사용되지만, 이들은 일반적으로 더 많은 계산 자원을 요구하며 정교한 최적화 과정이 필수적이다.

특징 추출 단계에서는 Mel-Frequency Cepstral Coefficients (MFCCs)가 표준적으로 사용된다. 기존 연구들은 MFCC의 시간적 역동성을 보존하기 위해 정적 집계(Static aggregation)나 윈도우 통계량(Window statistics) 등을 활용하여 성능을 높이려 했다. 그러나 이러한 방식들은 모델의 복잡도를 높이거나 메모리 사용량을 증가시키는 경향이 있다. 본 논문은 이러한 기존 딥러닝 접근 방식들이 고성능 MCU(예: STM32F7/F4 시리즈)에서는 작동하지만, 더 낮은 사양의 Cortex-M0+ 환경에서는 여전히 부담스럽다는 점을 지적하며 LogNNet을 통한 대안을 제시한다.

## 🛠️ Methodology

### 1. 전체 파이프라인 구성
시스템은 **VAD(Voice Activity Detection) $\rightarrow$ MFCC 특징 추출 $\rightarrow$ LogNNet 분류**의 3단계 파이프라인으로 구성된다.

### 2. Voice Activity Detection (VAD)
에너지 임계값 기반의 VAD를 통해 음성 구간을 추출한다. 입력 신호를 1000 샘플 윈도우(8 kHz 기준 125 ms)로 나누고 300 샘플씩 이동하며 평균 제곱 에너지(Mean Squared Energy, MSE)를 계산한다. MSE의 수식은 다음과 같다.

$$\text{MSE}_i = \frac{1}{N} \sum_{j=1}^{N} x_j^2$$

여기서 $x_j$는 프레임 내 오디오 샘플의 진폭이며, $N$은 프레임당 샘플 수이다. 에너지가 사전 정의된 임계값(0.001)을 초과하면 활성 음성 구간으로 판단하며, 여러 구간이 검출될 경우 평균 에너지가 가장 높은 구간을 최종 선택한다.

### 3. MFCC 특징 추출 및 집계 (Aggregation)
8 kHz로 리샘플링된 신호에 대해 128-point FFT, 12개 Mel 필터뱅크, DCT를 적용하여 프레임당 8개의 MFCC 계수를 추출한다. 추출된 2차원 MFCC 행렬을 1차원 벡터로 변환하기 위해 네 가지 집계 방법을 평가하였다.

- **Basic statistical features**: 각 계수의 평균, 표준편차, 최솟값, 최댓값을 계산 (32차원).
- **Temporal dynamics**: 1차 및 2차 미분(Delta, Delta-Delta) 계수를 포함하여 속도와 가속도 변화를 캡처 (48차원).
- **Windowed statistical**: MFCC 행렬을 4개의 시간 윈도우로 나누고 각 윈도우 내 통계량 계산 (128차원).
- **Adaptive binning**: 시간축을 8개의 동일한 간격(Bin)으로 나누고 각 구간의 평균값을 계산 (64차원). 입력 신호 길이에 상관없이 고정된 차원을 유지한다는 점에서 "Adaptive"라고 명명하였다.

### 4. LogNNet Classifier
LogNNet은 Reservoir Computing 기반 구조로, 다음과 같은 절차로 동작한다.
1. **입력 및 정규화**: 입력 벡터 $F$에 bias ($Y_0=1$)를 추가하여 $N+1$ 차원의 벡터 $Y$를 생성하고, 학습 데이터의 최댓값으로 정규화한다.
2. **Reservoir 변환**: 카오스 매핑(Chaotic mapping)으로 생성된 특수 행렬 $W$를 $Y$에 곱하여 중간 벡터 $S$를 생성한다. 이후 다시 정규화 및 bias를 추가하여 $P+1$ 차원의 벡터 $S_h$를 만든다.
3. **선형 분류**: $S_h$를 입력으로 하여 은닉층($M$개 뉴런)과 출력층(4개 뉴런, Softmax)으로 구성된 MLP(Multi-Layer Perceptron)를 통해 최종 클래스를 분류한다.
- 네트워크 구조 표기법: $\text{LogNNet } N:P:M:4$ (입력:Reservoir:은닉층:출력).

## 📊 Results

### 1. 분류 성능 평가
'go', 'stop', 'left', 'right' 4개 명령어를 대상으로 실험을 진행하였다. 특히 모델의 일반화 성능을 측정하기 위해 훈련 세트와 테스트 세트의 화자가 완전히 분리된 **Speaker-Independent Split** 방식을 사용하였다.

- **Adaptive Binning** 방식이 Speaker-Independent 평가에서 **92.04%**의 정확도를 기록하며 가장 우수한 성능을 보였다.
- Windowed Statistical 방식(91.72%)과 성능은 비슷하지만, 특징 벡터의 차원이 절반(64 vs 128) 수준으로 메모리 효율이 훨씬 높다.
- Basic statistical method는 82.49%로 가장 낮았으며, 이는 단순 통계량만으로는 시간적 정보를 충분히 보존할 수 없음을 시사한다.

### 2. 특징 중요도 분석 (PFI)
Permutation Feature Importance (PFI) 분석 결과, 두 번째 MFCC 계수의 네 번째 bin(명령어의 중앙 부분)이 가장 큰 중요도를 가졌으며, 전체 중요도의 70%가 3~7번 bin에 집중되어 있었다. 이는 짧은 음성 명령어의 핵심 음향 정보가 중앙과 후반부에 집중되어 있다는 특성과 일치한다.

### 3. 하드웨어 구현 및 자원 사용량
Arduino Nano 33 IoT (ARM Cortex-M0+, 48 MHz, 32 KB RAM)에 $\text{LogNNet } 64:33:9:4$ 구조를 구현하였다.

- **인식 정확도**: 하드웨어 실측 결과 약 90%의 정확도를 달성하였다.
- **메모리 사용량**: 총 RAM 사용량은 **18,016 bytes**로, 전체 가용 RAM의 **54.9%**만 점유하였다. 
- **Flash 메모리**: 약 50 KB를 사용하였으며, 이는 부동 소수점 연산 라이브러리 및 FFT 구현체 등이 포함된 결과이다.

## 🧠 Insights & Discussion

본 연구는 저사양 임베디드 환경에서 Reservoir Computing이 전통적인 심층 신경망(DNN)의 강력한 대안이 될 수 있음을 입증하였다. 특히, 기존 DS-CNN 기반 시스템들이 80~128 KB의 RAM을 요구하는 것과 달리, LogNNet은 단 18 KB만으로 유사한 성능을 낼 수 있어 RAM 사용량을 3~5배 가량 절감하였다.

분석 과정에서 흥미로운 점은 특징 수 감소에 따른 정확도 변화가 비선형적으로 나타난다는 것이다. 40개 미만의 특징을 사용할 때는 성능이 급격히 저하되거나 불안정해지는데, 이는 특징 간의 시너지 효과가 존재하며 Adaptive Binning이 이미 매우 압축된 표현임을 의미한다.

한계점으로는 Cortex-M0+에 하드웨어 부동 소수점 연산 장치(FPU)가 없어 소프트웨어적으로 연산을 처리해야 하므로, 추론 속도가 FPU 탑재 모델보다 느릴 수 있다는 점이 있다. 또한, 배경 소음이 극심한 환경에서의 강건성에 대한 추가적인 검증이 필요하다.

## 📌 TL;DR

본 논문은 초저전력/저메모리 임베디드 시스템을 위한 음성 명령 인식 시스템을 제안한다. **Adaptive Binning 기반 MFCC 특징 추출**과 **LogNNet Reservoir Computing 분류기**를 결합하여, ARM Cortex-M0+ 기반의 Arduino 보드에서 **RAM 18 KB 사용 및 약 90%의 인식 정확도**를 달성하였다. 이는 기존 CNN 기반 방식보다 메모리 요구량을 획기적으로 줄이면서도 실용적인 성능을 유지한 결과로, 향후 배터리 기반 IoT 노드나 웨어러블 기기의 온디바이스 AI 구현에 핵심적인 참고 자료가 될 것으로 기대된다.