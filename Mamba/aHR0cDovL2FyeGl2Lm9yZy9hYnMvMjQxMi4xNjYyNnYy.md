# Mamba-SEUNet: Mamba UNet for Monaural Speech Enhancement

Junyu Wang, Zizhen Lin, Tianrui Wang, Meng Ge, Longbiao Wang, Jianwu Dang (2025)

## 🧩 Problem to Solve

본 논문은 단일 채널 음성 강화(Monaural Speech Enhancement, SE) 분야에서 기존 딥러닝 모델들이 가진 계산 복잡도와 성능 사이의 트레이드오프 문제를 해결하고자 한다. 최근 음성 강화 연구에서는 Transformer 및 그 변형 모델들이 주류를 이루고 있으나, Self-attention 메커니즘의 연산 복잡도가 시퀀스 길이에 대해 제곱(Quadratic complexity)으로 증가한다는 치명적인 단점이 있다. 이는 실제 서비스 환경이나 자원이 제한된 하드웨어에 모델을 배포하는 데 큰 제약이 된다.

따라서 본 연구의 목표는 Transformer 수준의 강력한 성능을 유지하면서도, 연산 효율성을 획기적으로 높여 실용적인 배포가 가능한 새로운 음성 강화 네트워크 아키텍처를 설계하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 State Space Model(SSM)의 최신 발전 형태인 **Mamba**를 **U-Net** 구조와 결합한 **Mamba-SEUNet**을 제안한 점이다.

중심적인 설계 아이디어는 다음과 같다.

1. **Bidirectional Mamba 적용**: 음성 신호의 전방향 및 후방향 의존성을 모두 캡처하기 위해 양방향 Mamba 블록을 사용하여 시계열 데이터의 컨텍스트 모델링 능력을 극대화하였다.
2. **다해상도 특징 추출**: U-Net의 인코더-디코더 구조와 Skip connection을 활용하여, 다양한 해상도에서 거친 특징(Coarse-grained)부터 세밀한 특징(Fine-grained)까지 다각도로 학습하고 융합하도록 설계하였다.
3. **효율적인 선형 복잡도**: Transformer의 제곱 복잡도 문제를 Mamba의 선형 복잡도로 대체하여, 모델 파라미터 수와 FLOPs를 크게 줄이면서도 최신 성능(SOTA)을 달성하였다.

## 📎 Related Works

음성 강화 연구는 크게 시간 도메인(Time-domain) 방식과 시간-주파수 도메인(T-F domain) 방식으로 나뉜다. 시간 도메인 방식은 신호의 고유 특성을 보존하지만 결과가 다소 거칠다는 단점이 있으며, T-F 도메인 방식은 음성과 잡음 사이의 세밀한 구조적 차이를 더 잘 구별하는 경향이 있다.

최근에는 Transformer나 Conformer 기반의 2단계(Two-stage) 아키텍처가 우수한 성능을 보였으나, 주파수 차원만 축소할 뿐 다양한 해상도에서의 특징을 충분히 포착하지 못한다는 한계가 있었다. 또한, 앞서 언급한 Self-attention의 높은 연산 비용이 문제였다. 이를 해결하기 위해 Taylor multi-head self-attention(T-MSA)을 사용한 MUSE와 같은 효율적 모델이 제안되었으나, 여전히 원래의 MHSA(Multi-head self-attention) 성능에는 미치지 못하는 한계가 있었다. 이에 본 논문은 선형 복잡도를 가지면서도 긴 시퀀스 모델링에 강점이 있는 Mamba를 대안으로 제시한다.

## 🛠️ Methodology

### 1. 전체 파이프라인

Mamba-SEUNet의 전체 구조는 다음과 같은 흐름으로 진행된다.

- **입력 및 전처리**: 잡음이 섞인 음성 신호 $y$를 STFT(Short-Time Fourier Transform)를 통해 크기 스펙트럼(Magnitude spectrum) $Y^m$과 위상 스펙트럼(Phase spectrum) $Y^p$으로 분리한다.
- **특징 인코더**: 분리된 스펙트럼을 특징 인코더에 통과시켜 중간 특징 공간 $Y \in \mathbb{R}^{T \times F \times C_1}$으로 변환한다.
- **계층적 처리 (U-Net)**: TS-Mamba 블록, Patch embedding 레이어, 다운샘플링 및 업샘플링 연산을 통해 다해상도 특징을 처리한다.
- **복원**: 강화된 크기와 위상 스펙트럼을 디코더를 통해 복원한 후, ISTFT(Inverse STFT)를 통해 최종 강화된 음성 신호를 생성한다.

### 2. Mamba 및 Selective SSM

Mamba는 입력에 따라 모델 파라미터가 동적으로 조정되는 선택 메커니즘을 가진 SSM이다. 기본적으로 입력 $x(t)$를 고차원 은닉 상태 $h(t)$를 통해 출력 $y(t)$로 매핑하며, 이는 다음과 같은 연속 시간 방정식으로 표현된다.

$$h'(t) = Ah(t) + Bx(t)$$
$$y(t) = Ch(t)$$

이를 이산화(Discretize)하기 위해 시간 척도 파라미터 $\Delta$를 사용하여 다음과 같이 근사한다.

$$A = e^{(\Delta A)}, \quad B = (\Delta A)^{-1}(e^{(\Delta A)} - I) \cdot (\Delta B)$$

이산화된 상태 방정식은 다음과 같다.

$$h(t) = Ah(t-1) + Bx(t)$$
$$y(t) = Ch(t)$$

최종적으로 출력 $y$는 입력 $x$와 전역 컨볼루션 커널 $K$의 합성곱으로 계산되어 효율적인 연산이 가능하다.

### 3. TS-Mamba Block

시간(Time)과 주파수(Frequency) 차원을 순차적으로 처리하는 TS-Mamba 블록을 사용한다.

- **양방향 모델링**: 과거와 미래의 정보를 모두 통합하기 위해 Forward Mamba와 Backward Mamba를 병렬로 배치한다.
- **연산 절차**:
  1. 입력을 Forward Mamba와 Flip(반전) 후 Backward Mamba에 각각 통과시킨다.
  2. 두 출력에 $\text{RMSNorm}$을 적용하고 더한 후, 이를 $\text{Concat}$ 하여 $\text{Linear}$ 레이어를 통과시킨다.
  
  이를 수식으로 표현하면 다음과 같다.
  $$x_f = \text{RMS}(\text{FMamba}(x)) + x$$
  $$x_b = \text{RMS}(\text{BMamba}(\text{Flip}(x))) + x$$
  $$x' = \text{Linear}(\text{Concat}(x_f, x_b))$$

- **내부 구조**: 각 Mamba 블록 내에서는 $\text{Linear} \to \text{Conv} \to \text{SiLU} \to \text{SSM}$ 과정으로 진행되며, SSM의 순차적 제약으로 인한 정보 손실을 방지하기 위해 $\text{Linear} \to \text{SiLU}$로 구성된 대칭적인 Gated branch를 병렬로 추가하여 최종 결합한다.

### 4. Encoder 및 Decoder

- **Encoder**: 두 개의 컨볼루션 레이어와 Dilated DenseNet(깊이 4, dilation factor 2)으로 구성되어 기본 스펙트럼 특징을 캡처한다.
- **Decoder**: 인코더와 유사한 Dilated DenseNet 구조를 가지며, 마지막에 2D Transposed Convolution과 $1 \times 1$ Convolution을 사용한다.
- **활성화 함수**: 크기(Magnitude) 디코더는 학습 가능한 시그모이드 함수($\text{L-Sigmoid}$)를, 위상(Phase) 디코더는 $\text{Arctan2}$ 함수를 사용하여 각각의 특성에 맞게 최적화하였다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: VCTK+DEMAND 데이터셋 (30명의 화자, 10가지 잡음 유형, SNR 0~15dB 등).
- **평가 지표**: WB-PESQ(음성 품질), STOI(명료도), MOS(CSIG, CBAK, COVL - 신호 왜곡, 배경 잡음, 전체 품질), FLOPs(연산량).

### 2. 주요 결과

- **성능 비교**: Mamba-SEUNet의 대형 모델(L)은 $\text{WB-PESQ } 3.59$를 달성하며 SOTA 성능을 기록하였다. 특히 가장 작은 모델인 XS 버전(0.99M 파라미터)만으로도 기존 MP-SENet과 유사한 성능을 내면서 연산량(FLOPs)은 획기적으로 낮췄다.
- **PCS 기법 적용**: Perceptual Contrast Stretching(PCS) 전처리를 결합했을 때, Mamba-SEUNet (L)의 $\text{PESQ}$ 점수는 $3.73$까지 상승하였다.
- **블록 수($N$)에 따른 영향**: $N$이 1에서 3까지 증가할 때는 성능이 꾸준히 향상되지만, 4 이상부터는 성능 향상이 정체되는 플래토(Plateau) 현상이 관찰되었다.
- **아키텍처 비교**: 동일한 U-Net 구조에서 블록만 교체하여 비교한 결과, Mamba가 Conformer나 Transformer보다 더 적은 연산량(FLOPs)으로 더 높은 $\text{PESQ}$ 점수를 기록하였다. (Mamba: 3.57 vs Conformer: 3.45 vs Transformer: 3.52)

## 🧠 Insights & Discussion

### 강점 및 기여

본 연구는 Mamba라는 새로운 SSM 구조를 음성 강화의 U-Net 프레임워크에 성공적으로 통합하였다. 특히 Transformer의 고질적인 문제인 제곱 복잡도를 선형 복잡도로 해결함으로써, 모델의 크기를 키우지 않고도 효율적으로 긴 시퀀스의 의존성을 모델링할 수 있음을 입증하였다. 양방향 Mamba를 통해 음성 신호의 전후 맥락을 모두 파악한 점이 성능 향상의 주요 요인으로 분석된다.

### 한계 및 논의사항

실험 결과에서 블록 수 $N$의 증가에 따른 성능 향상이 특정 지점에서 정체되는 현상이 나타났다. 이는 단순히 모델의 깊이를 더하는 것보다, Mamba 블록 내의 파라미터 최적화나 다른 형태의 특징 융합 방식이 더 필요할 수 있음을 시사한다. 또한, 본 논문은 특정 데이터셋(VCTK+DEMAND)에서의 결과에 집중하고 있어, 더 다양하고 극한의 잡음 환경에서도 동일한 효율성이 유지되는지에 대한 추가 검증이 필요하다.

## 📌 TL;DR

Mamba-SEUNet은 **Mamba(Selective SSM)**와 **U-Net**을 결합하여, 기존 Transformer 기반 음성 강화 모델의 높은 연산 복잡도 문제를 해결한 모델이다. 양방향 Mamba 블록을 통해 음성 신호의 다해상도 특징과 장거리 의존성을 효율적으로 학습하며, 결과적으로 **낮은 연산 비용(FLOPs)으로 최신 수준(SOTA)의 음성 강화 성능(PESQ 3.59, PCS 적용 시 3.73)**을 달성하였다. 이 연구는 실시간 음성 처리 시스템이나 자원 제한적 환경에서의 고성능 음성 강화 구현에 중요한 가능성을 제시한다.
