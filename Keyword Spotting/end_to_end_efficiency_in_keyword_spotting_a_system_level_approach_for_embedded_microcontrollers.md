# End-to-End Efficiency in Keyword Spotting: A System-Level Approach for Embedded Microcontrollers

Pietro Bartoli, Tommaso la Bondini, Christian Veronesi, Andrea Giudici, Niccolò Antonello, Franco Zappa (2025)

## 🧩 Problem to Solve

본 논문은 임베디드 및 IoT 기기에서 필수적인 기술인 Keyword Spotting (KWS)을 자원이 매우 제한적인 마이크로컨트롤러(MCU) 환경에 효율적으로 배포하는 문제를 다룬다. KWS는 연속적인 오디오 스트림에서 미리 정의된 특정 단어를 감지하는 작업으로, 배터리로 작동하는 MCU의 특성상 엄격한 메모리 제약과 에너지 소비 효율성이 요구된다.

기존의 많은 연구는 주로 신경망 모델의 추론(Inference) 단계 최적화에만 집중하는 경향이 있었다. 그러나 실제 시스템에서는 오디오 신호를 신경망이 처리할 수 있는 형태로 변환하는 전처리 과정, 즉 Mel-Frequency Cepstral Coefficient (MFCC) 특징 추출 과정에서도 상당한 연산 비용과 메모리가 소모된다. 따라서 본 논문은 모델의 정확도뿐만 아니라, 전처리부터 추론까지의 전체 파이프라인을 포함한 시스템 수준의 효율성을 분석하고 최적화하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 하드웨어 제약 사항을 고려하여 설계된 초경량 신경망 아키텍처인 Typman-KWS (TKWS)를 제안하고, 이를 다양한 MCU 플랫폼에서 시스템 전체 관점으로 검증한 것이다.

중심적인 설계 아이디어는 MobileNetV2의 Inverted Bottleneck 구조에서 영감을 얻은 것이다. TKWS는 Pointwise expansion, double 1D depth-wise convolutions, 그리고 Projection layer로 구성된 잔차 블록(Residual block)을 쌓아 올린 구조를 가진다. 이를 통해 파라미터 수를 획기적으로 줄이면서도 KWS 작업에 필요한 특징 추출 능력을 유지하도록 설계하였다. 또한, 단순한 모델 비교를 넘어 전처리 파라미터(MFCC window 수 및 필터 뱅크 수)와 하드웨어 가속기(NPU) 유무에 따른 에너지-지연 시간의 상관관계를 체계적으로 분석하였다.

## 📎 Related Works

논문에서는 기존의 경량 KWS 모델들로 DS-CNN, LiCO-Net, 그리고 TENet을 언급하며 비교 대상으로 삼았다.

- **DS-CNN**: 표준 컨볼루션을 Depthwise와 Pointwise 연산으로 분해하여 연산 효율성을 높인 모델이다.
- **LiCO-Net**: 스트리밍 추론을 위해 설계된 선형화된 컨볼루션 네트워크로, 본 연구에서는 one-shot 추론이 가능하도록 패딩 전략을 수정하여 사용하였다.
- **TENet**: 다중 분기(Multi-branch) 구조를 통해 다양한 시간적 스케일의 특징을 집계하는 모델이다.

기존 연구들의 한계점은 주로 모델 자체의 파라미터 수나 추론 시간만을 측정했다는 점이다. 본 논문은 이러한 한계를 극복하기 위해 특징 추출 단계부터 추론 단계까지의 전체 프로세스를 벤치마킹하며, 특히 하드웨어별 특성에 따른 시스템 효율성을 정량적으로 분석했다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 전체 파이프라인 구조

KWS 시스템은 [Raw Audio $\rightarrow$ MFCC 특징 추출 $\rightarrow$ Neural Network 추론 $\rightarrow$ Post-processing]의 순서로 동작한다. MFCC 과정에서는 오디오 신호를 Mel-scaled spectrogram으로 변환한 뒤, Discrete Cosine Transform (DCT)를 통해 차원을 축소하여 특징 벡터를 생성한다.

### proposed 모델: TKWS

TKWS는 MCU의 메모리 제약을 해결하기 위해 설계된 모델로, 두 가지 설정(TKWS-2: 잔차 블록 2개, TKWS-3: 잔차 블록 3개)으로 구현되었다.

- **구조**: Pointwise expansion $\rightarrow$ Double 1D Depth-wise Convolutions $\rightarrow$ Projection layer.
- **특징**: 2D 컨볼루션 대신 1D 컨볼루션을 사용하여 연산량을 줄였으며, MFCC의 시간적 시퀀스를 직접 처리한다.
- **출력층**: Global Average Pooling layer와 Softmax 활성화 함수를 가진 Fully Connected layer를 통해 클래스 확률을 출력한다.

### 학습 및 벤치마킹 절차

- **데이터셋**: Google Speech Commands Dataset (GSCD) v0.02의 10개 키워드 클래스를 사용하였다. 실제 환경의 강건성을 위해 $\mathcal{N}(10\text{ dB}, 5\text{ dB})$의 SNR을 가진 배경 소음을 합성하여 데이터 증강을 수행하였다.
- **MFCC 설정**: 시간 해상도 영향을 분석하기 위해 윈도우 수(32, 63)와 Mel 필터 뱅크 수(15, 30)를 조합하여 테스트하였다.
- **하드웨어**: STM32 N6 (Cortex-M55 + NPU), H7 (Cortex-M7), U5 (Cortex-M33) 세 가지 플랫폼을 사용하였다. 모든 모델은 8-bit 정수형으로 양자화(Quantization)되었다.

### 평가 지표

모델의 성능은 Weighted F1-score로 측정하였으며, 시스템 효율성은 다음과 같은 Energy-Delay Product (EDP) 지표를 사용하였다.
$$EDP = \text{Latency} \times \text{Energy Consumption}$$
이 지표는 지연 시간과 에너지 소비 사이의 균형을 측정하며, 값이 낮을수록 시스템 효율성이 높음을 의미한다.

## 📊 Results

### 모델 성능 분석

실험 결과, TKWS-3 모델은 MFCC 크기가 $15 \times 63$일 때 14.4k라는 매우 적은 파라미터 수로 92.4%의 F1-score를 달성하였다. 이는 LicoNet-S (17.4k params, 93.6% F1)와 유사한 성능을 보이면서도 파라미터 수는 약 17% 더 적으며, DS-CNN이나 TENet6에 비해 메모리 사용량을 3배 이상 줄이면서 동등하거나 더 높은 정확도를 보였다. 특히 시간 윈도우 수를 32에서 63으로 늘렸을 때 정확도가 일관되게 상승함을 확인하여, 높은 시간 해상도가 성능 향상에 중요함을 입증하였다.

### 시스템 효율성 분석 (EDP)

- **하드웨어 비교**: NPU가 통합된 STM32 N6 플랫폼이 가장 낮은 EDP를 기록하며 최적의 효율성을 보였다. N6는 DSP를 통한 빠른 MFCC 추출과 NPU를 통한 저전력 고속 추론이 가능하여, 높은 해상도의 특징($63$ timesteps)을 사용하더라도 에너지 비용 증가가 적었다.
- **플랫폼 특성**: H7은 지연 시간은 짧으나 에너지 소비가 컸고, U5는 속도는 느리지만 에너지 효율이 높았다. 따라서 실시간 응답이 중요한 경우에는 H7이, 초저전력 시나리오에서는 U5가 적합하다는 결론을 내렸다.
- **아키텍처 영향**: NPU가 없는 MCU(H7, U5)에서는 1D 컨볼루션 기반 모델이 2D 기반 모델(DS-CNN 등)보다 에너지 효율이 훨씬 좋았다. 반면 N6에서는 TENet 모델의 커널 크기가 NPU의 최적 실행 범위를 벗어나 CPU로 작업이 오프로드됨에 따라 오히려 EDP가 증가하는 현상이 관찰되었다.

## 🧠 Insights & Discussion

본 논문은 모델의 정확도라는 단일 지표가 실제 임베디드 환경에서의 유효성을 보장하지 않는다는 점을 강조한다. 시스템의 실제 효율성은 전처리 파라미터, 모델 아키텍처, 그리고 하드웨어 가속기의 특성이 모두 맞물려 결정된다.

특히 주목할 점은 하드웨어-소프트웨어 공동 설계(Co-design)의 중요성이다. NPU가 탑재된 최신 MCU에서는 단순히 모델을 가볍게 만드는 것보다, 하드웨어 가속기가 최적으로 처리할 수 있는 연산 단위(커널 크기 등)에 맞춰 모델을 설계하는 것이 더 중요하다는 것을 실험적으로 보여주었다. 반면 일반 MCU에서는 1D 컨볼루션과 같은 연산 단순화가 에너지 효율에 직접적인 영향을 미친다.

다만, 본 연구는 10개의 제한된 키워드 클래스만을 대상으로 하였으며, 더 많은 단어를 인식해야 하는 확장된 KWS 시나리오에서 TKWS의 파라미터 효율성이 어떻게 유지될지는 명시되지 않았다.

## 📌 TL;DR

본 연구는 MCU 환경에서 KWS의 전처리부터 추론까지의 전체 파이프라인을 분석하고, MobileNetV2 기반의 초경량 모델인 TKWS를 제안하였다. TKWS-3는 14.4k의 극소량 파라미터로 92.4%의 높은 정확도를 달성하였으며, STM32 N6와 같은 NPU 탑재 하드웨어에서 최적의 시스템 효율성(EDP)을 보였다. 이 연구는 임베디드 AI 배포 시 모델의 정확도뿐만 아니라 하드웨어 특성에 맞춘 전처리 및 아키텍처 설계가 필수적임을 시사하며, 향후 하드웨어-인식형 신경망 설계(Hardware-aware design)의 중요성을 강조한다.
