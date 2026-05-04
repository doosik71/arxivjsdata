# HI-MAMBA: HIERARCHICAL MAMBA FOR EFFICIENT IMAGE SUPER-RESOLUTION

Junbo Qiao, Jincheng Liao, Wei Li, Yulun Zhang, Yong Guo, Yi Wen, Zhangxizi Qiu, Jiao Xie, Jie Hu, Shaohui Lin (2024)

## 🧩 Problem to Solve

본 논문은 단일 이미지 초해상도(Single Image Super-Resolution, SISR) 작업에서 State Space Models (SSM), 특히 Mamba 아키텍처를 효율적으로 활용하는 문제를 다룬다.

Mamba와 같은 SSM은 선형 복잡도로 장거리 의존성(long-range dependency)을 모델링할 수 있어 매우 유망하지만, 본질적으로 1차원 시퀀스 데이터를 처리하도록 설계되었다. 이를 2차원 이미지 데이터에 적용하기 위해 기존의 Vision Mamba 계열 연구(예: MambaIR)들은 이미지를 여러 방향으로 스캔하는 multi-direction scanning 전략을 사용한다. 하지만 이러한 방식은 다음과 같은 심각한 문제를 야기한다.

1. **계산 오버헤드 증가**: 여러 방향으로 반복적인 스캔을 수행함에 따라 SSM의 핵심 장점인 선형 복잡도의 이점이 상쇄되며, 연산량(FLOPs)과 파라미터 수가 크게 증가한다.
2. **고해상도 처리의 한계**: 계산 비용의 급격한 증가로 인해 실제 고해상도 이미지 처리 시 연산 부담이 매우 커져 실용성이 떨어진다.

따라서 본 논문의 목표는 multi-direction scanning으로 인한 계산 비용을 획기적으로 줄이면서도, 2차원 공간 의존성 모델링 능력을 유지하거나 향상시킨 효율적인 SR 네트워크인 **Hi-Mamba**를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **단방향 스캔(single-direction scanning)**을 유지하면서도, **계층적 구조(Hierarchical structure)**와 **방향 교차 배치(Direction Alternation)**를 통해 공간 정보 손실을 보완하는 것이다.

1. **Hierarchical Mamba Block (HMB)**: 단일 방향 스캔만 사용하는 Local SSM(L-SSM)과 Region SSM(R-SSM)을 병렬로 구성하여, 서로 다른 스케일의 표현력을 결합함으로써 컨텍스트 모델링 능력을 강화한다.
2. **Direction Alternation Hierarchical Mamba Group (DA-HMG)**: 동일한 단방향 스캔을 사용하는 HMB들을 수평(H), 수직(V), 역수평(RH), 역수직(RV) 순서로 교차 배치하여, 추가적인 연산 비용 없이 2차원 공간 관계 모델링을 풍부하게 한다.
3. **Gate Feed-Forward Network (G-FFN)**: FFN에 단순한 게이트 메커니즘을 도입하여 비선형 정보를 추가하고 채널 내 중복 정보를 줄여 모델의 표현력을 높인다.

## 📎 Related Works

### 1. Efficient CNNs and Transformers for SR

CNN 기반 SR 모델들은 효율성을 높이기 위해 cascading mechanism(CARN)이나 feature splitting(IMDN) 등을 도입했으나, 커널 크기의 제한으로 인해 장거리 의존성 모델링에 한계가 있다. Transformer 기반 모델들은 Self-Attention을 통해 이를 해결했지만, 시퀀스 길이에 따른 이차 복잡도(quadratic complexity) 문제로 인해 Window-based attention이나 channel-wise attention 등을 사용한다. 그러나 이러한 방식은 여전히 추론 시 계산 비용이 높고 수용 영역(receptive field)이 제한되는 문제가 있다.

### 2. Mamba and Applications for SR

최근 Mamba 아키텍처는 선형 복잡도로 긴 시퀀스를 처리할 수 있어 주목받고 있다. MambaIR과 같은 초기 SR 적용 사례들은 2D 특성을 잡기 위해 multi-sequence scanning 전략을 사용했다. 하지만 이는 계산 비용을 대폭 증가시킨다. Hi-Mamba는 이러한 기존 Mamba 기반 SR 모델들과 달리 **단일 시퀀스 스캔**만을 사용하여 효율성을 극대화하고, 계층적 구조와 방향 교차 전략으로 공간 모델링 능력을 보완한다는 점에서 차별화된다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

Hi-Mamba는 크게 세 단계의 파이프라인으로 구성된다.

- **Shallow Feature Extraction**: 단순한 convolution 레이어를 통해 입력 LR 이미지 $I^{LR}$로부터 로컬 특징 $F^l$을 추출한다.
- **Deep Feature Extraction**: $N_2$개의 DA-HMG 그룹을 통해 심층 특징 $F^d$를 추출하며, 각 그룹 내에는 $N_1$개의 HMB가 포함된다.
- **Image Reconstruction**: 추출된 $F^l$과 $F^d$를 합산한 후, $3 \times 3$ convolution과 Pixel Shuffle 연산을 통해 최종 고해상도 이미지 $I^r$을 생성한다.

학습은 정답 이미지 $I^{gt}$와 예측 이미지 $I^r$ 사이의 픽셀 단위 $L1$ 손실 함수를 사용하여 최적화한다.

### 2. Hierarchical Mamba Block (HMB)

HMB는 단일 방향 스캔을 수행하며, 두 개의 브랜치로 구성된다.

- **Local SSM (L-SSM)**: 입력 특징의 원본 해상도에서 로컬 의존성을 학습한다.
- **Region SSM (R-SSM)**: 입력 특징을 $n \times n$ 크기로 투영(projection)하여 저해상도 영역에서 전역적인 문맥을 학습한다.

두 브랜치의 출력은 **Fusion Module**을 통해 결합된다. Fusion Module은 R-SSM의 출력을 공간적으로 반복(repeat)하여 L-SSM의 크기와 맞춘 뒤, 학습 가능한 스케일링 인자 $S_f$를 사용하여 다음과 같이 융합한다:
$$F^{out} = S_f \cdot X^l_{out} + (1 - S_f) \cdot f_{re}(X^r_{out})$$
여기서 $f_{re}$는 2D repeat 연산이다.

이후 **Gate Feed-Forward Network (G-FFN)**를 거치는데, 이는 특징 맵을 채널 방향으로 두 부분으로 나누어 요소별 곱셈(element-wise multiplication)을 수행하는 게이트 구조를 가진다:
$$\text{G-FFN}(F^i) = w_2 * (\hat{F}_1 \odot \hat{F}_2)$$

### 3. Direction Alternation Hierarchical Mamba Group (DA-HMG)

DA-HMG는 계산 비용을 늘리지 않고 공간 모델링 능력을 높이기 위한 전략이다. 단일 방향 스캔만 사용하는 HMB를 다음과 같은 순서로 배치한다:
$$\text{HMB-H} \rightarrow \text{HMB-V} \rightarrow \text{HMB-RH} \rightarrow \text{HMB-RV}$$
각 블록은 오직 한 가지 방향으로만 스캔하지만, 레이어를 거치며 방향이 바뀌므로 결과적으로 전체 네트워크는 4방향의 공간 정보를 모두 통합하게 된다.

### 4. SSM 기초 이론 (Preliminaries)

본 논문에서 사용된 SSM은 연속 상태 공간 방정식 $\mathbf{h}'(t) = \mathbf{A}\mathbf{h}(t) + \mathbf{B}x(t), y(t) = \mathbf{C}\mathbf{h}(t) + \mathbf{D}x(t)$에서 시작하여, Zero-Order Hold (ZOH) 규칙을 통해 이산화(discretization)된 형태를 사용한다.
$$\mathbf{A} = \exp(\Delta \mathbf{A}), \mathbf{B} = (\Delta \mathbf{A})^{-1}(\exp(\mathbf{A}) - \mathbf{I}) \cdot \Delta \mathbf{B}$$
이를 통해 입력 시퀀스 $x$를 출력 $y$로 매핑하는 선형 연산으로 변환하여 계산 효율성을 확보한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: DIV2K, Flicker2K로 학습하고 Set5, Set14, BSD100, Urban100, Manga109 벤치마크에서 평가하였다.
- **지표**: PSNR, SSIM, 파라미터 수, FLOPs, 추론 시간(GPU Latency)을 측정하였다.
- **비교 대상**: CARN, IMDN, SwinIR-Light, SRFormer-Light, MambaIR 등 경량화 SR 모델들과 비교하였다.

### 2. 정량적 결과

- **경량 모델 비교 (Table 2)**: Hi-Mamba-S는 MambaIR 대비 FLOPs를 294G 줄이면서도 Urban100 ($\times 2$ SR)에서 PSNR을 0.21dB 향상시켰다. 또한 SRFormer-Light보다도 우수한 성능을 보였다.
- **고성능 모델 비교 (Table 3)**: 대규모 모델인 Hi-Mamba-L은 기존 SOTA 모델들을 능가하였으며, 특히 self-ensemble 전략을 적용한 Hi-Mamba-L+는 Manga109 ($\times 2$ SR)에서 SRFormer 대비 0.42dB, MambaIR 대비 0.21dB 높은 PSNR을 기록했다.
- **복잡도 및 효율성 (Table 4 & Fig 5)**: Hi-Mamba-L은 GRL-B 및 MambaIR보다 GFLOPs를 크게 낮추면서도 더 높은 PSNR을 달성하여, Latency-PSNR trade-off 관점에서 최적의 효율성을 보였다.

### 3. 정성적 결과

- 시각적 비교 결과, CNN이나 Transformer 기반 모델에서 나타나는 블러(blurry) 현상이나 왜곡이 줄어들었으며, 특히 건물 텍스처와 같은 세밀한 구조 복원 능력이 뛰어나다.
- Local Attribution Map (LAM) 시각화를 통해 Hi-Mamba가 다른 모델보다 더 넓은 범위의 정보를 활용하여 복원을 수행함을 확인하였다.

## 🧠 Insights & Discussion

### 1. 강점 및 분석

- **효율적인 공간 모델링**: Multi-direction scanning을 단순 반복하는 대신, 방향을 교차 배치하는 DA-HMG 전략이 연산량 증가 없이 성능을 높일 수 있음을 입증하였다.
- **계층적 구조의 이점**: L-SSM과 R-SSM을 결합한 구조가 단일 스케일 모델보다 훨씬 높은 PSNR 향상을 가져왔으며, 이는 다양한 수용 영역의 정보를 통합하는 것이 SR 작업에 필수적임을 시사한다.

### 2. 한계 및 논의사항

- **하이퍼파라미터 민감도**: R-SSM의 채널 수나 영역 크기($n \times n$)에 따라 성능과 속도가 달라지는데, 본 논문에서는 $4 \times 4$ 크기가 최적의 trade-off를 보인다고 명시하였다. 하지만 이는 데이터셋이나 배율에 따라 달라질 가능성이 있다.
- **가정**: 본 연구는 주로 PSNR 기반의 복원 성능에 집중하고 있으며, 지각적 품질(perceptual quality)에 대한 상세한 분석은 상대적으로 부족하다.

## 📌 TL;DR

본 논문은 Mamba의 선형 복잡도 장점을 유지하면서 2D 이미지의 공간 의존성을 효율적으로 모델링하는 **Hi-Mamba** 네트워크를 제안한다. 핵심은 **단방향 스캔을 사용하는 계층적 블록(HMB)**과 이를 **수평/수직/역방향으로 교차 배치하는 그룹 구조(DA-HMG)**에 있다. 이를 통해 기존 MambaIR 대비 연산량(FLOPs)을 획기적으로 줄이면서도 PSNR 성능을 향상시켰으며, 경량 모델과 고성능 모델 모두에서 최적의 효율성(Latency-PSNR trade-off)을 달성하였다. 이 연구는 추후 고해상도 이미지 복원 작업에서 SSM을 실용적으로 적용하는 데 중요한 기준이 될 것으로 보인다.
