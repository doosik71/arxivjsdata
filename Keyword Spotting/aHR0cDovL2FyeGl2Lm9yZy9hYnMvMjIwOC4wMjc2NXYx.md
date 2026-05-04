# KEYWORD SPOTTING SYSTEM AND EVALUATION OF PRUNING AND QUANTIZATION METHODS ON LOW-POWER EDGE MICROCONTROLLERS

Jingyi Wang, Shengchen Li (2022)

## 🧩 Problem to Solve

본 연구는 저전력 엣지 디바이스, 특히 Cortex-M 기반의 마이크로컨트롤러(MCU)에서 실시간으로 동작하는 Keyword Spotting(KWS) 시스템을 구현하고 최적화하는 문제를 다룬다. KWS는 음성 기반의 사용자 인터랙션을 가능하게 하며, 스마트폰이나 IoT 기기에서 'Always-on' 상태로 동작해야 하므로 전력 소비, 대역폭 절약, 개인정보 보호를 위해 엣지 컴퓨팅 환경에서의 구현이 필수적이다.

그러나 마이크로컨트롤러는 다음과 같은 하드웨어적 제약 사항을 가지고 있다.

- **제한된 메모리 공간**: SRAM은 보통 $20\text{KB}$에서 $512\text{KB}$ 사이이며, Flash 메모리 역시 $64\text{KB}$에서 $1\text{MB}$ 정도로 매우 제한적이다.
- **낮은 연산 속도**: CPU 클럭 주파수가 일반적으로 $72\text{MHz}$에서 $216\text{MHz}$ 사이로 낮아, 딥러닝 모델의 높은 연산량과 저지연 요구사항을 충족하기 어렵다.

따라서 본 논문의 목표는 이러한 제약이 큰 STM32F7 마이크로컨트롤러 상에서 동작하는 소형 KWS 시스템을 구축하고, 다양한 Pruning(가지치기) 및 Quantization(양자화) 기법이 실제 하드웨어 성능(추론 시간 및 전력 소모)에 미치는 영향을 정밀하게 평가하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 이론적인 모델 압축 수치에 그치지 않고, 실제 MCU 하드웨어 상에서 다양한 최적화 기법의 실효성을 검증했다는 점이다. 구체적인 설계 아이디어는 다음과 같다.

1. **소형 KWS 시스템 구현**: 연산량을 최소화한 CNN-DNN 구조를 설계하여 STM32F7 하드웨어에서 실시간 오디오 특징 추출부터 추론까지 가능한 파이프라인을 구축하였다.
2. **Pruning 효율성 분석**: Unstructured pruning과 Structured pruning의 실제 가속 성능을 비교하여, MCU 환경에서는 구조적 가지치기가 훨씬 유리함을 입증하였다.
3. **Weight Prioritized Loop 제안**: Unstructured pruning의 가속화를 위해 루프 순서를 변경하여, 가중치가 0인 경우 연산을 한 번에 건너뛸 수 있도록 설계하여 추론 속도를 개선하였다.
4. **양자화 및 SIMD 적용**: float32 데이터를 int16으로 양자화하고, ARM Cortex-M의 SIMD(Single Instruction Multiple Data) 명령어인 `SMLAD`를 활용하여 연산 속도를 극대화하였다.

## 📎 Related Works

논문에서는 소형 KWS 구현을 위해 제안되었던 DNN, CNN, CRNN, LSTM 등의 네트워크 구조들을 언급한다. 특히, 모델의 파라미터 중 약 90%를 제거해도 정확도 손실이 적다는 Han et al.의 Pruning 연구를 소개하며, 이를 통해 모델 크기를 줄일 수 있음을 명시한다.

기존 접근 방식과의 차별점은 다음과 같다. 많은 연구가 모델의 파라미터 수나 이론적인 연산량(FLOPs) 감소에 집중하는 반면, 본 연구는 실제 MCU의 명령어 파이프라인, 메모리 접근 패턴, SIMD 명령어 활용 등 하드웨어 레벨에서의 실제 실행 시간(Latency)과 전력 소모량에 집중하여 분석하였다는 점이다.

## 🛠️ Methodology

### 1. 시스템 아키텍처 및 파이프라인

전체 시스템은 오디오 획득 $\rightarrow$ 특징 추출 $\rightarrow$ 모델 추론 단계로 구성되며, FreeRTOS를 통해 태스크가 스케줄링된다.

- **특징 추출(Feature Extraction)**:
  - $16\text{kHz}$로 샘플링된 오디오 신호에 Hanning window를 적용하고 $1024\text{-point FFT}$를 수행한다.
  - 이후 $40\text{-band Mel filter-bank}$를 곱하고 로그 진폭으로 변환하여 Mel spectrogram을 생성한다.
  - 최신 30개 컬럼(약 1초 분량)을 큐 형태로 저장하여 모델의 입력으로 사용한다.
- **모델 구조**:
  - **Convolutional Layer**: 하나의 합성곱 층을 사용한다. 커널 너비는 30(시간축 stride 없음), 높이는 8이며, 수직 stride를 4로 설정하여 풀링 층을 대체하였다.
  - **DNN Layer**: Conv 층의 출력을 Flatten 하여 4개의 Fully Connected(FC) 층에 연결한다.
  - **Output**: 최종적으로 Softmax를 통해 6개의 클래스("yes", "no", "left", "right", "background", "unknown")를 분류한다.

### 2. Pruning 및 가속 기법

- **Pruning Granularity**: Fine-grained(Unstructured), Vector-level, Kernel-level, Filter-level의 네 가지 수준의 희소성(Sparsity)을 평가하였다.
- **Weight Prioritized Loop**: 일반적인 루프는 출력 맵의 각 값에 대해 반복하지만, 제안된 방식은 가중치 값을 기준으로 반복한다.
  - 가중치 $w=0$인 경우, 해당 가중치와 관련된 모든 연산을 하나의 `if` 문으로 건너뜀으로써 불필요한 연산을 제거한다.
- **Quantization & SIMD**:
  - **Quantization**: 32-bit floating-point 데이터를 16-bit integer(int16)로 변환하여 메모리 사용량을 절반으로 줄이고 연산 비용을 낮춘다.
  - **SIMD (`SMLAD`)**: 두 쌍의 int16 값들을 한 번의 명령어로 곱하고 누적하여 int32 값으로 만드는 `SMLAD` 명령어를 사용하여 연산 효율을 높인다.

## 📊 Results

### 1. 실험 환경

- **하드웨어**: STM32F767IGT6 (Cortex-M7 @ 216MHz, 512KB SRAM).
- **데이터셋**: Speech Command Data Set v0.01 (4개 키워드 + 배경음 + 알 수 없는 음성).
- **평가 지표**: 추론 시간(Inference Time), 전력 소모(Power Consumption), 메모리 점유율.

### 2. 주요 결과

- **기본 시스템**: 실시간 특징 추출을 포함하여 약 $37\text{ms}$마다 분류 결과를 생성하며, 단일 추론 시간은 약 $31\text{ms}$이다. 모델 파라미터는 총 $119,936$개이며 약 $468.5\text{KB}$의 메모리를 사용한다.
- **Pruning 분석**:
  - Unstructured pruning의 경우 단순한 `if` 문 추가는 오히려 명령어 파이프라인 중단으로 인해 성능이 저하될 수 있다.
  - 반면, Vector-level 및 Filter-level의 Structured pruning은 적은 양의 가중치 제거만으로도 실제 성능 향상이 나타났다.
  - Weight Prioritized Loop를 적용한 Unstructured pruning은 희소성이 $80\%$ 이상일 때 효과적이며, $90\%$ 제거 시 속도가 약 2배 향상되었다.
- **양자화 및 SIMD 효과**:
  - $\text{float32} \rightarrow \text{int16}$ 양자화 시: 추론 시간이 $30.8\text{ms} \rightarrow 21.4\text{ms}$로 단축되고 메모리 사용량이 절반으로 감소하였다.
  - $\text{int16} + \text{SIMD}(\text{SMLAD})$ 적용 시: 추론 시간이 $15.6\text{ms}$까지 더욱 단축되었다.

## 🧠 Insights & Discussion

본 연구는 딥러닝 모델의 이론적 압축률이 실제 임베디드 하드웨어의 가속으로 직접 연결되지 않는다는 점을 명확히 보여준다. 특히 Unstructured pruning은 파라미터 수를 획기적으로 줄일 수 있음에도 불구하고, MCU의 메모리 접근 방식과 명령어 실행 구조로 인해 실제 연산 속도 향상을 이끌어내기 매우 어렵다는 점이 확인되었다.

강점으로는 실제 하드웨어 상에서 루프 순서 변경(`Weight Prioritized Loop`)과 하드웨어 특화 명령어(`SMLAD`)의 조합이 어떤 시너지를 내는지 정량적으로 분석했다는 점이다. 하지만, Weight Prioritized Loop가 다른 가속 기법들과 호환되지 않을 수 있다는 점은 한계로 지적된다. 또한, 정확도(Accuracy)에 대한 상세한 수치 변화보다는 실행 속도와 전력 효율에 집중하고 있어, 압축률 증가에 따른 성능 저하(Trade-off)에 대한 분석이 다소 부족하다.

## 📌 TL;DR

이 논문은 저전력 STM32 마이크로컨트롤러에서 동작하는 실시간 KWS 시스템을 구현하고, Pruning과 Quantization의 실제 하드웨어 가속 효율을 분석하였다. 연구 결과, MCU 환경에서는 Unstructured pruning보다 Structured pruning이 효율적이며, int16 양자화와 SIMD 명령어(`SMLAD`)를 결합했을 때 가장 비약적인 성능 향상(추론 시간 약 50% 감소)을 얻을 수 있음을 입증하였다. 이는 향후 엣지 디바이스용 AI 모델 최적화 시, 단순한 파라미터 감소보다는 하드웨어 명령어 세트와 메모리 구조를 고려한 최적화가 필수적임을 시사한다.
