# Implementing Keyword Spotting on the MCUX947 Microcontroller with Integrated NPU

Petar Jakuš, Hrvoje Džapo (2025)

## 🧩 Problem to Solve

본 연구는 자원이 극도로 제한된 임베디드 환경, 특히 마이크로컨트롤러(MCU)에서 실시간 Keyword Spotting(KWS) 시스템을 구현하는 것을 목표로 한다. IoT 기기의 확산으로 음성 제어 인터페이스의 중요성이 커지고 있으나, MCU는 연산 능력, 메모리 용량 및 전력 소모 측면에서 매우 엄격한 제약을 가지고 있다. 따라서 이러한 하드웨어 제약 내에서 정확도를 최대한 유지하면서도 추론 속도를 높이고 모델 크기를 줄이는 최적화된 KWS 솔루션을 구축하는 것이 핵심 문제이다.

## ✨ Key Contributions

본 논문의 중심적인 아이디어는 MFCC(Mel-Frequency Cepstral Coefficients) 기반의 특징 추출과 Quantization Aware Training(QAT)을 적용한 CNN 분류기를 결합하고, 이를 NXP MCXN947 MCU에 내장된 NPU(Neural Processing Unit)를 통해 가속화하는 것이다. 특히, 32비트 부동 소수점 모델을 8비트 정수형으로 변환할 때 발생하는 정확도 손실을 최소화하기 위해 학습 단계에서부터 양자화를 시뮬레이션하는 QAT 기법을 도입하여, 모델의 경량화와 실시간 성능이라는 두 가지 목표를 동시에 달성하였다.

## 📎 Related Works

논문에서는 KWS 시스템의 일반적인 구성 요소인 신호 획득, 특징 추출, 신경망 분류 단계를 설명한다. 기존 연구들에서 특징 추출을 위해 MFCC, RASTA-PLP 등이 널리 사용되었으며, 연산 복잡도를 낮추기 위해 FFT를 직접 적용하거나 아날로그 프론트엔드 방식을 사용하는 연구들이 제안되었다. 분류 단계에서는 CNN, LSTM, GRU 등의 아키텍처가 사용되었으며, 특히 모델 압축을 위해 Pruning 및 Quantization 기법이 적용되었다. 또한, 전용 코프로세서를 활용한 하드웨어 가속이 전력 소모를 유의미하게 줄인다는 점이 기존 문헌을 통해 제시되었다. 본 연구는 이러한 기존 접근 방식을 기반으로 최신 NPU 내장 MCU에 특화된 최적화 파이프라인을 제안한다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 1. 전처리 (Feature Extraction)
원시 오디오 신호를 신경망이 처리하기 적합한 형태로 변환하기 위해 다음과 같은 MFCC 파이프라인을 사용한다.

- **Framing 및 Windowing**: 오디오 신호를 25ms 길이의 프레임으로 나누며, 프레임 간 겹침(Hop size)은 10ms로 설정한다. 스펙트럼 누설을 줄이기 위해 각 프레임에 Hamming window를 적용하며, 수식은 다음과 같다.
$$w(n) = 0.54 - 0.46 \cdot \cos\left(\frac{2\pi n}{N}\right)$$
- **Spectral Analysis**: Fast Fourier Transform(FFT)를 통해 신호를 주파수 영역으로 변환한 후, 전력 스펙트럼 $P[k]$를 계산한다.
$$P[k] = \frac{1}{N}|X[k]|^2$$
여기서 $N$은 FFT 길이를 의미한다.
- **Mel-scale Filtering**: 인간의 청각 특성을 반영하여 40Hz에서 7.6kHz 범위의 40개 멜 필터 뱅크를 적용한다.
- **MFCC Computation**: 로그 멜 스펙트럼에 이산 코사인 변환(DCT)을 적용하여 최종적으로 프레임당 20개의 MFCC 특징을 생성한다.

### 2. 모델 아키텍처 및 양자화
본 연구에서는 엣지 디바이스에 최적화된 소형 CNN 구조를 설계하였다.

- **네트워크 구조**: 2개의 Convolutional Layer(각 층은 2D Convolution, Batch Normalization, Max Pooling으로 구성)와 2개의 Dense Layer(Fully Connected Layer)로 이루어져 있다. 최종 출력층은 "Marvin"이라는 특정 키워드의 존재 여부를 판단하는 이진 분류기 형태이다.
- **Quantization Aware Training (QAT)**: 32비트 부동 소수점 가중치를 8비트 고정 소수점으로 변환하기 위해 QAT를 적용하였다. 이는 순전파(Forward pass) 시에는 8비트 연산을 시뮬레이션하고, 역전파(Backward propagation) 시에는 전체 정밀도를 유지하여 학습하는 방식이다. 이를 통해 단순 사후 양자화(Post-Training Quantization)보다 정확도 저하를 효과적으로 막을 수 있다.

### 3. 모델 배포
학습된 TensorFlow Lite 모델은 NXP의 eIQ Toolkit을 통해 NPU 호환 포맷으로 변환되었다. 변환 과정에서 NPU의 전용 하드웨어 가속 기능을 최대한 활용할 수 있도록 일부 레이어 구조가 재구성되었으며, 최종 모델은 MCU의 플래시 메모리에 정적 배열 형태로 저장된다.

## 📊 Results

### 1. 실험 설정
- **데이터셋**: Google Speech Commands 데이터셋을 사용하였으며, 16kHz 샘플링 레이트의 1초 길이 음성 데이터를 활용하였다.
- **비교 대상**: 정밀도 모델(Regular), 양자화 모델(Quantized), NPU 가속 모델(NPU) 세 가지 버전을 비교하였다.
- **측정 지표**: 정확도(Accuracy), 모델 크기(Size), 추론 시간(Inference Time)을 측정하였다.

### 2. 주요 결과
실험 결과, NPU를 활용한 구현이 CPU 단독 실행 대비 압도적인 성능 향상을 보였다.

- **정확도**: Regular 모델은 $99.14\%$의 정확도를 보였으며, 양자화 및 NPU 모델은 $97.06\%$로 약간의 하락이 있었으나 매우 높은 수준을 유지하였다.
- **모델 크기**: Regular 모델($383,674$ bytes) 대비 NPU 모델($30,576$ bytes)은 약 $98.3\%$의 크기 감소를 달성하였다.
- **추론 속도**: MCXN947의 ARM Cortex-M33 CPU에서 양자화 모델의 추론 시간은 $228.2\text{ms}$였으나, NPU 가속을 적용했을 때는 $3.847\text{ms}$로 단축되었다. 이는 약 $59\times$의 속도 향상을 의미한다.
- **전체 파이프라인**: MFCC 특징 추출 시간(평균 $431\mu\text{s}$)과 NPU 추론 시간을 합산하면 전체 처리 시간이 $5\text{ms}$ 미만으로, 실시간 상호작용이 충분히 가능함을 입증하였다.

## 🧠 Insights & Discussion

본 연구는 QAT와 NPU 가속의 조합이 리소스 제한적인 MCU 환경에서 딥러닝 모델을 구동하는 가장 효율적인 방법 중 하나임을 보여준다. 특히 단순한 양자화만으로는 MCU CPU에서의 추론 속도가 실시간성(Real-time)을 확보하기 어렵지만, NPU를 통한 하드웨어 가속이 결합될 때 비로소 실용적인 수준의 응답 속도가 확보된다는 점이 인상적이다.

다만, 몇 가지 한계점이 존재한다. 첫째, 특정 키워드("Marvin") 하나에 대해서만 최적화된 모델을 평가하였으므로, 다중 키워드 인식 환경에서의 확장성 및 성능 변화에 대한 검증이 필요하다. 둘째, 하드웨어 가속을 통한 속도 향상은 명확히 제시되었으나, 실제 전력 소모량($\text{mW}$)에 대한 정량적 분석이 누락되어 있어 저전력 특성을 완전히 입증했다고 보기 어렵다. 마지막으로, 다양한 소음 환경에서의 강건성(Robustness) 테스트가 부족하여 실제 환경에서의 성능 예측에는 주의가 필요하다.

## 📌 TL;DR

본 논문은 NXP MCXN947 MCU의 내장 NPU를 활용하여 실시간 키워드 스포팅(KWS) 시스템을 구현한 연구이다. MFCC 특징 추출과 QAT 기반의 경량 CNN 모델을 적용하여, 정확도를 $97.06\%$로 유지하면서 모델 크기를 $98.3\%$ 줄였으며, CPU 대비 추론 속도를 $59$배 향상시켰다. 이 결과는 매우 제한된 자원을 가진 엣지 디바이스에서도 효율적인 음성 인터페이스 구현이 가능함을 시사하며, 향후 저전력 상시 대기(Always-on) 음성 인식 시스템 연구에 중요한 기초 자료가 될 것으로 보인다.