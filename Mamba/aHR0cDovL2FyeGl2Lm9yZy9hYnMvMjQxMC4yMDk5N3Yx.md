# SepMamba: State-space models for speaker separation using Mamba

Thor Højhus Avenstrup, Boldizsár Elek, István László Mádi, András Bence Schin, Morten Mørup, Bjørn Sand Jensen and Kenny Olsen (2024)

## 🧩 Problem to Solve

본 논문은 단일 채널 화자 분리(single-channel speaker separation), 즉 소위 '칵테일 파티 문제(cocktail party problem)'를 해결하고자 한다. 화자 분리는 하나의 혼합 신호에서 개별 소스 신호를 추출하는 작업으로, 보청기나 통신 기기와 같은 실시간 오디오 처리 시스템에서 매우 중요하다.

최근 Transformer 기반의 Attention 메커니즘을 도입한 딥러닝 모델들이 뛰어난 성능 향상을 보였으나, 입력 시퀀스 길이에 대해 이차 복잡도(quadratic complexity)를 가지는 계산적 특성 때문에 막대한 연산 자원이 소모된다. 이는 계산 자원이 제한적인 저전력 환경(low-resource environments)에서의 실용적인 적용을 어렵게 만든다. 따라서 본 연구의 목표는 Transformer 수준의 모델링 능력을 유지하면서도, 연산 효율성을 획기적으로 높인 화자 분리 아키텍처를 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Transformer의 Attention 메커니즘 없이 오직 Mamba 레이어만을 사용하여 시간 영역(time-domain)에서 화자 분리를 수행하는 **SepMamba** 아키텍처를 제안한 것이다. 

중심 아이디어는 Mamba의 선형 시간 복잡도(linear-time complexity)를 활용하여 긴 의존성(long-range dependencies)을 효율적으로 학습하는 것이다. 이를 위해 Mamba 레이어를 U-Net 구조에 통합하여 다양한 스케일의 오디오 구조를 학습할 수 있도록 설계하였다. 특히, 양방향 Mamba 블록(Bidirectional Mamba, Bamba)을 도입하여 비인과적(non-causal) 설정에서의 성능을 극대화하였으며, 동시에 효율적인 인과적(causal) 변형 모델도 함께 제시하였다.

## 📎 Related Works

기존의 화자 분리 접근 방식은 크게 세 단계로 진화하였다. 

1. **STFT 기반 방식**: 입력 신호를 주파수 영역으로 변환하여 분리한 후 역STFT를 통해 복원한다. 하지만 위상(phase) 정보 모델링의 어려움과 긴 프레임 길이로 인한 높은 지연 시간(latency)이 한계로 지적된다.
2. **시간 영역(Time-domain) 방식**: TasNet, Conv-TasNet, SudoRM-RF 등은 학습 가능한 선형 기저를 사용하여 직접 파형을 처리함으로써 STFT의 한계를 극복하였다.
3. **Transformer 기반 방식**: SepFormer, MossFormer 등은 Attention 메커니즘을 통해 단기 및 장기 의존성을 효과적으로 포착하며 SOTA 성능을 달성하였다. 그러나 앞서 언급한 이차 복잡도로 인해 입력 데이터를 짧은 청크(chunk) 단위로 나누어 처리해야 하며, 이 과정에서 청크 간의 긴 의존성을 놓칠 수 있는 문제가 있다.

최근 State-Space Model(SSM)의 일종인 Mamba가 언어 모델링 및 이미지 세그멘테이션에서 Transformer에 필적하는 성능과 높은 효율성을 보여주었다. 기존의 SP-Mamba가 Mamba를 도입했으나, 여전히 Transformer 레이어에 의존하여 높은 연산 비용이 발생하는 한계가 있었다. SepMamba는 이러한 Transformer 의존성을 완전히 제거한 최초의 Mamba 기반 시간 영역 화자 분리 모델이라는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조
SepMamba는 U-Net 아키텍처를 기반으로 하며, 총 5단계의 다운샘플링(down-sampling)과 업샘플링(up-sampling) 과정으로 구성된다. 각 단계에는 **Bamba(Bidirectional Mamba)** 스택이 배치되어 시간적 의존성을 학습한다.

- **다운샘플링**: 표준 컨볼루션(Convolution) 레이어를 사용하여 해상도를 낮추고 채널 차원을 2배로 늘린다.
- **업샘플링**: 전치 컨볼루션(Transposed Convolution) 레이어를 사용하여 해상도를 복원하고 채널 차원을 절반으로 줄인다.
- **Skip Connection**: 다운샘플링 단계의 특징 맵을 업샘플링 단계로 전달하며, 이때 $1 \times 1$ 컨볼루션을 통해 차원을 맞춘다.
- **활성화 함수**: 네트워크 전체에 $\text{ReLU}$를 사용한다.

### Bamba 블록 및 Mamba 레이어
Bamba 블록은 입력 신호를 정방향과 역방향으로 각각 처리하여 합산하는 구조이다.

$$ \text{Bamba}(x) = \text{Mamba}_1(x) + \text{flip}(\text{Mamba}_2(\text{flip}(x))) $$

여기서 $\text{flip}(\cdot)$은 시퀀스를 반전시키는 연산이다. 인과적(causal) 변형 모델의 경우, 역방향 연산을 제거하고 정방향 Mamba 블록만을 사용한다.

Mamba 레이어의 핵심은 다음과 같은 이산화된 상태 공간 방정식으로 정의된다.

$$ h_t = Ah_{t-1} + Bx_t $$
$$ y_t = Ch_t $$

여기서 $A, B, C, \Delta$는 모델 파라미터이며, $A$와 $B$는 다음과 같은 이산화 규칙을 따른다.

$$ A = \exp(\Delta A) $$
$$ B = (\Delta A)^{-1} \exp(\Delta A - I) \cdot \Delta B $$

### 학습 절차 및 손실 함수
- **데이터셋**: WSJ0-2mix 데이터셋을 사용하며, Dynamic Mixing(DM) 증강 기법과 속도 섭동(speed perturbation)을 적용하였다.
- **손실 함수**: $\text{Scale-Invariant Signal-to-Distortion Ratio (SI-SDR)}$의 음수 값을 최소화하는 것을 목표로 하며, 화자 순서 문제를 해결하기 위해 $\text{utterance-level Permutation Invariant Training (uPIT)}$을 사용한다.
- **최적화**: $\text{AdamW}$ 옵티마이저를 사용하며, 초기 학습률은 $15 \times 10^{-5}$이다. 학습 중 수렴 상태에 따라 지수적 학습률 감소(exponential decay)를 적용한다.

## 📊 Results

### 실험 설정 및 지표
- **평가 지표**: $\text{SI-SDRi}$, $\text{SDRi}$, $\text{SI-SNRi}$를 통해 분리 성능을 측정한다.
- **효율성 지표**: 하드웨어 독립적인 연산량 지표인 $\text{GMAC/s}$, $\text{A100 GPU}$에서의 실제 실행 시간(wall-clock time), 피크 메모리 사용량(peak memory usage)을 측정한다.

### 주요 결과
1. **정량적 성능**: 
   - SepMamba (M) 모델은 Transformer 기반의 $\text{SepFormer}$ 및 $\text{MossFormer (M)}$보다 우수한 성능을 보이면서도 연산 및 메모리 사용량은 훨씬 적다.
   - Mamba를 부분적으로 사용한 $\text{SP-Mamba}$보다도 성능과 효율성 면에서 모두 우위에 있음을 입증하였다.
   - 소형 모델인 $\text{SepMamba (S)}$ 역시 유사한 파라미터 수를 가진 기존 모델들보다 뛰어난 성능을 보였다.
2. **인과적 설정(Causal Setting)**:
   - $\text{SepMamba}$의 인과적 변형 모델은 $\text{SI-SNRi}$ 기준 $21.4\text{dB}$를 달성하여, 기존 SOTA 인과적 모델인 $\text{UX-NET}(13.6\text{dB})$과 $\text{Causal Deep Casa}(15.2\text{dB})$를 크게 상회하였다.
3. **계산 효율성**:
   - **실행 시간**: $\text{SepMamba}$는 다른 모델들에 비해 전방 패스(forward pass) 시간이 압도적으로 짧다.
   - **메모리 사용량**: 피크 GPU 메모리 사용량이 $\text{SepFormer}$나 $\text{MossFormer 2}$보다 현저히 낮으며, $\text{Conv-TasNet}$ 수준의 효율성을 보인다.
   - **연산량**: $\text{GMAC/s}$ 관점에서 $\text{SepMamba (M)}$은 $\text{SepFormer}$의 극히 일부 연산량만으로 더 높은 성능을 낸다.

## 🧠 Insights & Discussion

### 강점 및 해석
SepMamba는 Transformer의 성능적 이점을 유지하면서도 SSM의 선형 복잡도를 활용해 계산 비용을 획기적으로 낮추었다. 특히, Transformer의 Attention 메커니즘은 입력 길이에 따라 메모리 사용량이 기하급수적으로 증가하는 반면, Mamba는 재귀적(recurrent) 특성 덕분에 추론 시 현재의 은닉 상태 $h_t$만 유지하면 되므로 실시간 처리 및 저전력 시스템에 매우 적합하다.

### 한계 및 논의 사항
논문에서는 파라미터 수 대비 성능 효율성 측면에서 일부 모델이 더 나은 결과를 보일 수 있음을 인정한다. 하지만 이는 파라미터 공유(parameter-sharing)나 컨볼루션 커널 크기 조정 등을 통해 최적화할 수 있는 여지가 많다고 분석한다.

또한, Mamba 레이어는 연산 집약도(arithmetic intensity)가 낮아 메모리 대역폭에 제한을 받는(memory-bound) 특성이 있다. 이는 GPU와 같이 병렬 연산 능력이 매우 높은 환경보다, 오히려 병렬성이 낮은 저사양 시스템에서 Transformer 대비 더 큰 성능 이점을 가질 수 있음을 시사한다.

## 📌 TL;DR

본 논문은 Transformer의 무거운 Attention 메커니즘을 제거하고, 효율적인 State-Space Model인 Mamba를 U-Net 구조에 결합한 **SepMamba**를 제안한다. 이 모델은 $\text{WSJ0-2mix}$ 데이터셋에서 기존 Transformer 기반 모델들을 능가하거나 대등한 성능을 보이면서도, 연산량(GMACs), 메모리 사용량, 실제 추론 시간을 획기적으로 줄였다. 특히 강력한 인과적 성능과 낮은 자원 소모 덕분에, 실시간성이 중요한 보청기나 저전력 임베디드 기기 등의 실제 환경에 적용될 가능성이 매우 높은 연구이다.