# End-to-end Keyword Spotting using Neural Architecture Search and Quantization

David Peter, Wolfgang Roth, Franz Pernkopf (2021)

## 🧩 Problem to Solve

본 논문은 자원이 제한된 환경(limited resource environments)에서 작동하는 종단간(end-to-end) 키워드 검출(Keyword Spotting, KWS) 모델을 자동으로 탐색하고 최적화하는 문제를 해결하고자 한다.

일반적인 자동 음성 인식(ASR) 시스템은 계산 복잡도가 높아 항상 켜져 있는(always-on) 모드로 작동할 경우 모바일 기기의 배터리를 빠르게 소모시키는 에너지 효율성 문제가 발생한다. 이를 해결하기 위해 특정 키워드만을 상시 감시하다가 키워드가 검출되었을 때만 전체 ASR 시스템을 트리거하는 저전력 KWS 시스템이 필요하다. 따라서 KWS 시스템은 다음과 같은 세 가지 핵심 요구사항을 충족해야 한다.
1. 에너지 소모를 줄이기 위한 자원 효율성(resource-efficiency).
2. 실시간 작동 가능성(real-time execution).
3. 사용자 경험 유지를 위한 높은 정확도(accuracy).

본 연구의 목표는 Neural Architecture Search(NAS)와 양자화(Quantization) 기술을 결합하여, 원시 오디오 파형(raw audio waveforms)을 직접 입력으로 사용하는 작고 효율적인 KWS 모델을 설계하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 다음과 같다.

- **원시 오디오 기반의 end-to-end KWS를 위한 NAS 적용**: 기존의 KWS NAS 연구들이 MFCC와 같은 수작업 특징량(hand-crafted features)에 의존했던 것과 달리, 원시 오디오 파형을 직접 처리하는 효율적인 모델 구조를 자동으로 탐색하였다.
- **특징 추출 방식의 비교 분석**: 원시 오디오 기반 모델과 전통적인 MFCC 기반 모델 간의 정확도, 연산량(operations), 파라미터 수를 정밀하게 비교하여 end-to-end 접근 방식의 효용성을 입증하였다.
- **최적화된 양자화 전략 제안**: 고정 비트 너비 양자화(fixed bit-width quantization)와 학습 가능 비트 너비 양자화(trained bit-width quantization)를 비교하여, 메모리 풋프린트를 극도로 줄이면서도 성능 하락을 최소화하는 방법을 제시하였다.

## 📎 Related Works

기존의 KWS 시스템은 주로 MFCC(Mel-frequency cepstral coefficient)와 같은 수작업 특징량을 추출하여 DNN에 입력하는 방식을 사용하였다. 그러나 MFCC 추출 과정에 포함되는 푸리에 변환(Fourier transform)은 계산 비용이 많이 들어 자원 제한적 환경에서는 부담이 될 수 있다. 이를 보완하기 위해 시간 영역의 특징량인 MFSTS(Multi-Frame Shifted Time Similarity) 등이 제안되었으나, 여전히 수작업 특징량이 KWS에 최적이라는 보장은 없다.

최근에는 SincNet과 같이 원시 오디오를 직접 입력으로 받는 CNN 구조가 제안되었으며, 이는 학습 가능한 파라미터 기반의 sinc 함수를 통해 커스텀 필터 뱅크를 구축함으로써 효율성을 높인다. 또한, DNN 구조 설계를 자동화하는 NAS(Neural Architecture Search) 기술이 발전하였으며, 특히 ProxylessNAS와 같이 타겟 하드웨어의 제약 조건을 고려하여 정확도와 연산량 사이의 트레이드오프를 최적화하는 방식이 주목받고 있다.

## 🛠️ Methodology

### 1. Neural Architecture Search (NAS)
본 연구는 $\text{ProxylessNAS}$를 사용하여 정확도와 연산량(number of operations)을 동시에 최적화하는 다중 목적 최적화를 수행한다.

- **전체 구조**: 모델은 총 5단계(stage)로 구성된다.
    - 스테이지 (i), (ii), (v)는 고정된 구조를 가진다.
    - 스테이지 (iii), (iv)는 NAS를 통해 최적화된다.
- **탐색 공간**: 스테이지 (iii)와 (iv)에서는 $\text{Mobile Inverted Bottleneck Convolutions (MBC)}$ 블록을 기본 단위로 사용한다.
    - **최적화 대상**: 확장 비율(expansion rate $e \in \{1, 2, 3, 4, 5, 6\}$), 커널 크기($k \in \{3, 5, 7\}$), 또는 $\text{Identity}$ 레이어 선택 여부.
    - MBC는 $1\times1$ convolution $\rightarrow$ depthwise-separable $k\times k$ convolution $\rightarrow$ $1\times1$ convolution 순으로 구성되며, 배치 정규화(Batch Normalization)와 ReLU 활성화 함수가 적용된다.

### 2. SincConv를 이용한 특징 추출
원시 오디오 입력을 처리하기 위해 첫 번째 레이어에 $\text{SincConv}$를 사용한다. SincConv는 학습 가능한 파라미터 $\theta = (f_1, f_2)$를 가진 밴드패스 필터로 작동한다.

필터 함수 $g$는 다음과 같이 정의된다.
$$g[n, f_1, f_2] = 2f_2 \text{sinc}(2\pi f_2 n) - 2f_1 \text{sinc}(2\pi f_1 n)$$
여기서 $\text{sinc}(x) = \sin(x)/x$이며, $f_1$과 $f_2$는 각각 밴드패스 필터의 하한 및 상한 컷오프 주파수를 의미한다. 

이 방식은 일반적인 1D-Conv와 달리 필터 전체를 저장할 필요 없이 두 개의 주파수 파라미터만 학습하면 되므로 메모리 효율적이다.

### 3. 가중치 및 활성화 함수 양자화
NAS로 탐색된 모델을 대상으로 $\text{Brevitas}$ 프레임워크를 이용한 양자화 인식 학습(Quantization-Aware Training, QAT)을 수행한다. 

- **Straight-Through Estimator (STE)**: 불연속적인 양자화 함수의 기울기를 근사하여 역전파(backpropagation)가 가능하게 한다.
- **동적 범위 $\alpha$**: 정수 값을 실제 값으로 매핑하는 스케일링 인자로, 가중치의 경우 최대 절대값을 사용하고 활성화 함수의 경우 2비트 이상일 때 학습 가능한 파라미터로 설정한다.
- **학습 가능 비트 너비 양자화 (Trained bit-width quantization)**: 비트 너비 자체를 학습 파라미터로 취급하며, 다음과 같은 손실 함수를 최적화한다.
$$L = L_{CE} + \lambda_w \cdot B_w + \lambda_a \cdot B_a$$
여기서 $L_{CE}$는 교차 엔트로피 손실이며, $B_w$와 $B_a$는 각각 가중치와 활성화 함수의 평균 비트 너비이다. $\lambda_w$와 $\lambda_a$는 비트 수를 줄이기 위한 규제 계수($0.04$)이다.

## 📊 Results

### 1. 실험 설정
- **데이터셋**: Google Speech Commands 데이터셋. 10개의 키워드 클래스, 'unknown' 클래스, 'silence' 클래스를 포함하여 총 12개 클래스로 구성된다.
- **데이터 증강**: 랜덤 타임 시프트 및 배경 소음 추가를 적용하였다.
- **비교 대상**: MFCC 기반 모델, 1D-Conv 기반 end-to-end 모델.

### 2. 주요 결과
- **SincConv vs MFCC vs 1D-Conv**:
    - 1D-Conv 모델은 정확도, 연산량, 파라미터 수 모든 측면에서 SincConv 모델보다 성능이 낮았다.
    - SincConv 모델은 MFCC 모델과 동일한 정확도를 달성하는 데 더 적은 파라미터 수를 필요로 하지만, 연산량은 약간 더 많았다.
    - NAS를 통해 도출된 최적 모델은 파라미터 75.7k개, 연산량 13.6M ops로 **95.55%의 정확도**를 달성하였다.

- **양자화 결과**:
    - **고정 비트 너비**: 활성화 함수의 비트 너비가 2비트 또는 1비트로 감소할 때 성능 하락이 가장 뚜렷하게 나타났다. 8비트 활성화 함수를 사용할 때는 가중치 비트 수에 관계없이 풀-프리시전(full-precision) 모델과 유사한 성능을 보였다.
    - **학습 가능 비트 너비**: 가중치 평균 2.51비트, 활성화 함수 평균 2.91비트를 사용하여 **93.76%의 정확도**를 기록하였다. 이는 동일 수준의 비트를 사용한 고정 비트 너비 모델(2비트-2비트, 정확도 90.35%)보다 3.41% 높은 성능이다.

## 🧠 Insights & Discussion

본 연구는 원시 오디오를 직접 처리하는 end-to-end KWS 모델이 적절한 NAS 구조와 SincConv 필터를 통해 MFCC 기반 시스템보다 효율적일 수 있음을 보여주었다. 특히 SincConv의 파라미터화된 필터 구조가 일반적인 1D-Conv보다 우수한 성능을 내는 점은, 음성 신호의 특성을 반영한 제약 조건이 모델의 학습 효율을 높인다는 것을 시사한다.

양자화 분석에서는 학습 가능 비트 너비 방식이 고정 방식보다 훨씬 높은 정확도를 유지함을 확인하였다. 흥미로운 점은 모델의 입력 레이어와 출력 레이어가 중간 레이어보다 더 많은 비트 수를 필요로 한다는 결과이다. 이는 기존 문헌에서 입력/출력 레이어를 풀-프리시전으로 유지하는 관행과 일치하는 결과이다.

다만, 학습 가능 비트 너비 양자화는 하드웨어 구현 난이도가 높다는 한계가 있다. 실제 칩 상에서 가변 비트 연산을 구현하는 것은 고정 비트 연산보다 훨씬 복잡하기 때문에, 실제 배포 단계에서는 정확도와 구현 복잡도 사이의 절충안이 필요할 것으로 판단된다.

## 📌 TL;DR

본 논문은 NAS와 양자화 기술을 결합하여 원시 오디오를 직접 입력으로 받는 초경량 KWS 모델을 제안하였다. $\text{SincConv}$ 레이어와 $\text{ProxylessNAS}$를 통해 최적의 구조를 찾았으며, 학습 가능 비트 너비 양자화를 적용해 평균 2.5~2.9비트의 매우 낮은 정밀도에서도 93.76%의 높은 정확도를 달성하였다. 이 연구는 극도의 자원 제한 환경에서 작동하는 온디바이스(on-device) 음성 인식 시스템 설계에 중요한 가이드라인을 제공한다.