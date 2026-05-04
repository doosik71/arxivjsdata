# U-Mamba-Net: A highly efficient Mamba-based U-net style network for noisy and reverberant speech separation

Shaoxiang Dang, Tetsuya Matsumoto, Yoshinori Takeuchi, and Hiroaki Kudo (2024)

## 🧩 Problem to Solve

본 논문은 소음과 잔향이 존재하는 복잡한 환경에서 혼합된 음성 신호를 개별 화자의 음성 스트림으로 분리하는 Speech Separation 문제를 다룬다. 특히 소음과 잔향은 원하는 신호를 심하게 방해하며, 음성 분리에 필수적인 장기 의존성(long-term dependencies)을 포착하는 것을 어렵게 만든다.

기존의 고성능 모델들은 RNN, Transformer 기반의 Dual-path 구조 등을 사용하여 이러한 문제를 해결하려 했으나, Transformer의 Self-attention 메커니즘이 가진 이차 복잡도(quadratic scaling)로 인해 계산 비용과 학습 시간이 기하급수적으로 증가하는 문제가 있다. 또한, 복잡한 작업을 단순한 하위 작업으로 나누어 해결하는 Cascaded Multi-task Learning (CMTL) 방식은 성능 향상을 가져오지만, 모듈의 중첩으로 인해 모델 크기가 커지고 모듈 간의 gradient conflict가 발생하여 모델의 역량을 제한하는 한계가 있다. 따라서 본 논문의 목표는 연산 효율성을 유지하면서도 복잡한 환경에서 높은 성능을 내는 경량화된 음성 분리 모델을 개발하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 국소적 특징과 다해상도 특징 추출에 강점이 있는 U-Net 구조와, 선형 복잡도를 유지하면서 글로벌 의존성을 효율적으로 모델링할 수 있는 Mamba(Selective State Space Model)를 결합하는 것이다. Mamba를 일종의 특징 필터(feature filter)로 활용하여 U-Net 블록과 교대로 배치함으로써, 모델의 크기와 계산 비용을 낮게 유지하면서도 전역적인 문맥 정보를 효과적으로 학습하도록 설계하였다.

## 📎 Related Works

음성 분리를 위해 초기에는 시간-주파수 영역 및 시간 영역 처리에서 RNN이 널리 사용되었으며, 글로벌 정보를 모델링하기 위해 Dual-path(DP) 구조가 도입되었다. 이후 병렬 계산이 가능한 Transformer의 Self-attention 메커니즘이 DP 구조에 적용되었으나, 앞서 언급한 계산 오버헤드 문제가 제기되었다.

또한, 생물 의학 이미지 분할을 위해 개발된 U-Net 아키텍처가 Demucs나 SuDoRM-RF와 같은 연구를 통해 음악 및 오디오 소스 분리에서도 성공적으로 적용된 바 있다. U-Net은 완전 합성곱 신경망(fully convolutional neural network)으로 구성되어 크기가 작고 계산 복잡도가 낮다는 장점이 있지만, 전역적 관계를 학습하는 능력이 부족하다는 한계가 있다. 본 논문은 이러한 U-Net의 한계를 Mamba 모듈을 통해 보완하였다.

## 🛠️ Methodology

### 전체 시스템 구조

U-Mamba-Net은 크게 **Encoder $\rightarrow$ U-Mamba blocks $\rightarrow$ Decoder**의 구조로 이루어져 있다.

1. **Encoder**: 1차원 합성곱 층을 사용하여 파형(waveform) 데이터를 시간-주파수 유사 표현으로 매핑한다.
2. **U-Mamba blocks**: 본 모델의 핵심으로, 고도의 표현 능력을 갖춘 특징을 학습한다.
3. **Mask Estimation**: U-Mamba 블록 이후의 합성곱 층을 통해 각 음원(source)에 대한 마스크를 추정한다.
4. **Decoder**: 추정된 마스크를 혼합 음성 표현에 적용한 후, 1차원 전치 합성곱 층(transposed convolutional layer)을 통해 최종 분리된 음성을 복원한다.

### U-Mamba Block

하나의 U-Mamba 블록은 **U-net 모듈**과 **Mamba 모듈**로 구성된다.

- **U-net 모듈**: $L$개의 연속적인 다운샘플링 및 업샘플링 층으로 구성되며, 동일 깊이의 층 사이에는 residual connection이 연결되어 있다.
- **Mamba 모듈**: U-net 모듈의 출력을 입력으로 받아 전역적인 특징을 필터링한다.
- 최종적으로 U-Mamba 블록의 출력은 U-net 모듈의 출력과 다시 한번 residual connection으로 연결된다.

### Mamba (Selective SSM)

Mamba는 상태 공간 모델(State Space Models, SSMs)의 확장으로, 입력 $x(t) \in \mathbb{R}^F$를 은닉 상태 $h(t) \in \mathbb{R}^N$를 거쳐 출력 $y(t) \in \mathbb{R}^F$로 매핑한다.

연속형 수식은 다음과 같다:
$$h'(t) = Ah(t) + Bx(t)$$
$$y(t) = Ch(t) + Dx(t)$$
여기서 $A, B, C, D$는 상태 행렬이다. 이를 이산화(discretize)하면 다음과 같이 표현된다:
$$h_t = \bar{A}h_{t-1} + \bar{B}x_t$$
$$y_t = \bar{C}h_t + \bar{D}x_t$$
이때 $\bar{A}$와 $\bar{B}$는 bilinear method와 스텝 파라미터 $\Delta$를 사용하여 변환된다:
$$\bar{A} = (I - \frac{\Delta}{2}A)^{-1}(I + \frac{\Delta}{2}A)$$
$$\bar{B} = (I - \frac{\Delta}{2}A)^{-1}\Delta B$$

Mamba는 S4(Structured SSMs)의 HiPPO 초기화를 계승하여 신호를 직교 다항식으로 분해하는 능력을 가지며, 여기에 입력 의존적(input-dependent) 선택 메커니즘을 추가하여 계산 효율성과 표현력을 동시에 확보하였다.

### 학습 절차 및 손실 함수

모델은 Permutation-Invariant Scale-Invariant Signal-to-Noise Ratio (SI-SNR)를 손실 함수로 사용하여 학습한다:
$$\mathcal{L} = -\max_{\pi \in \mathcal{P}} \frac{1}{I} \sum_{i} \text{SI-SNR}(\hat{s}_{\pi(i)}, s_i)$$
여기서 $\pi$는 전체 SI-SNR을 최대화하는 최적의 순열 매핑 집합이다.

## 📊 Results

### 실험 설정

- **데이터셋**: Libri2mix를 사용하였으며, WHAM!의 소음과 Pyroomacoustics를 통한 잔향 시뮬레이션을 추가하여 복잡한 환경을 구축하였다. (샘플링 레이트 8 kHz)
- **비교 대상**: TasNet, SuDoRM-RF, Conv-TasNet, DPRNN (E2E 및 CMTL 방식)
- **평가 지표**: SI-SNRi, SDRi, SIRi (분리 성능), STOI, PESQ (지각적 품질), CSIG, CBAK, COVL (잡음 제거 성능), 모델 파라미터 수 및 GMACs (연산 효율성)

### 주요 결과

1. **정량적 성능**: U-Mamba-Net은 SI-SNRi 기준 8.50 dB를 기록하여 DPRNN (E2E)보다 0.92 dB 높았으며, CMTL 기반의 DPRNN보다도 0.42 dB 높은 성능을 보였다.
2. **연산 효율성**: 계산 비용(GMACs) 측면에서 압도적인 효율성을 보였다. U-Mamba-Net의 GMACs는 2.5인 반면, DPRNN (CMTL)은 40.2, DPRNN (E2E)은 23.9로, 각각 $1/16$, $1/9$ 수준의 연산량만으로 더 높은 성능을 달성하였다.
3. **지각적 품질 및 잡음 제거**: STOI 지표에서는 경쟁 모델들보다 우위에 있었으나, CBAK 등 잡음 제거 지표에서는 CMTL 기반의 DPRNN이 더 우수한 결과를 보였다. 이는 다단계 감독 학습을 수행하는 CMTL 구조의 특성으로 분석된다.

### Ablation Study

- **특징 차원 ($F$)**: $F$가 64 $\rightarrow$ 128 $\rightarrow$ 192로 증가함에 따라 SI-SNRi가 7.12 $\rightarrow$ 8.50 $\rightarrow$ 8.85 dB로 크게 향상되었으나 모델 크기도 함께 증가하였다.
- **블록 수 ($R$)**: 블록 수를 늘리면 성능이 향상되지만, 특징 차원만큼의 영향력은 낮았다.
- **모델 깊이 ($L$)**: 모델의 깊이를 과도하게 늘리는 것은 테스트 셋 성능에 부정적인 영향을 주었으며, 이는 분리 작업에서 너무 낮은 해상도가 유익하지 않기 때문으로 추정된다.
- **업샘플링 방법**: T-Conv1D, NN, Linear 방식 간의 차이는 미미하였다.

## 🧠 Insights & Discussion

본 논문은 Mamba의 선형 복잡도 특성과 U-Net의 다해상도 특징 추출 능력을 결합하여, 매우 적은 연산 비용으로도 기존의 무거운 모델(Transformer, DPRNN 등)보다 우수한 음성 분리 성능을 낼 수 있음을 입증하였다. 특히 GMACs 수치에서 보여준 효율성은 실시간 시스템이나 자원이 제한된 환경에서의 적용 가능성을 강력하게 시사한다.

다만, 단일 작업 모델로서의 한계도 명확히 드러났다. 잡음 제거 성능(CBAK 등)에서 CMTL 구조에 밀리는 모습을 보였는데, 이는 중간 단계의 레이블(noise-free mixture)을 활용하는 다단계 학습 방식이 특정 하위 작업(denoising)에서는 더 효과적임을 의미한다. 또한, 시각화 결과에서 DPRNN에 비해 스펙트로그램의 선명도가 다소 떨어진다는 점이 확인되었다.

결론적으로 U-Mamba-Net은 '효율성'과 '전반적인 분리 성능' 사이의 최적점을 잘 찾아낸 모델이지만, 매우 정교한 잡음 제거가 필요한 경우에는 CMTL과 같은 구조적 보완이 필요할 것으로 보인다.

## 📌 TL;DR

U-Mamba-Net은 U-Net의 구조적 장점과 Mamba의 효율적인 전역 모델링 능력을 결합한 경량 음성 분리 모델이다. 기존 DPRNN 대비 연산량(GMACs)을 1/10 수준으로 낮추면서도 SI-SNRi 등 주요 분리 지표에서 더 높은 성능을 달성하였다. 이 연구는 고비용의 Transformer나 RNN 기반 모델을 대체하여 저전력/실시간 음성 분리 시스템을 구축하는 데 중요한 기반이 될 가능성이 높다.
