# CU-Mamba: Selective State Space Models with Channel Learning for Image Restoration

Rui Deng, Tianpei Gu (2024)

## 🧩 Problem to Solve

본 논문은 노이즈 제거(Denoising), 블러 제거(Deblurring), 비 제거(Deraining)와 같이 손상된 이미지를 복원하는 이미지 복원(Image Restoration) 문제를 해결하고자 한다. 이미지 복원 분야에서는 전통적으로 CNN과 Transformer 기반의 모델들이 널리 사용되어 왔으나, 각각 치명적인 한계를 가지고 있다.

CNN은 국소적인 특징 추출에는 뛰어나지만 수용 영역(Receptive Field)이 제한적이라 이미지 내의 장거리 의존성(Long-range Dependency)을 캡처하는 데 어려움이 있다. 반면, Transformer는 Self-attention 메커니즘을 통해 전역적인 문맥을 파악할 수 있지만, 특성 맵의 크기에 따라 연산 비용이 이차적으로 증가하는 quadratic computational cost 문제가 발생하며, 복원에 필수적인 미세한 지역적 세부 사항을 놓칠 가능성이 있다.

최근 이를 해결하기 위해 선형 복잡도로 전역 문맥을 모델링할 수 있는 Mamba(Selective State Space Model)가 제안되었으나, 기존의 시각적 Mamba 모델들은 대부분 SSM 블록을 각 채널에 독립적으로 적용한다. 이는 채널 간의 정보 흐름(Channel Correlation)을 간과하게 만들며, 결과적으로 이미지의 세부 구조를 압축하고 재구성해야 하는 이미지 복원 작업에서 성능 저하를 야기한다. 따라서 본 논문의 목표는 전역 공간 문맥과 채널 간 상관관계를 동시에 학습할 수 있는 효율적인 모델을 설계하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 U-Net 구조에 전역 공간 정보를 학습하는 **Spatial SSM**과 채널 간의 상관관계를 학습하는 **Channel SSM**을 결합한 **Dual SSM 프레임워크**를 도입하는 것이다.

단순히 공간적인 스캔뿐만 아니라 채널 차원에서도 Selective SSM을 적용함으로써, 특성 맵의 압축 및 업샘플링 과정에서 발생하는 정보 손실을 최소화하고 풍부한 채널 문맥을 보존하고자 한다. 이를 통해 Transformer의 전역 수용 영역이라는 장점과 CNN의 효율성을 동시에 확보하면서, 기존 Mamba 모델들이 놓쳤던 채널 간 상호작용을 효과적으로 통합한 CU-Mamba 모델을 제안한다.

## 📎 Related Works

이미지 복원을 위한 기존 연구들은 크게 세 가지 흐름으로 나뉜다. 첫째, CNN 기반 접근 방식은 계층적 구조를 통해 공간적 위계를 잘 포착하며, 특히 Skip-connection을 가진 U-shaped 네트워크가 다중 스케일 특징 표현에 강점을 보였다. 둘째, Transformer 기반 방식은 Global self-attention을 통해 장거리 상호작용을 모델링하지만, 높은 연산 비용으로 인해 최근에는 로컬 윈도우나 채널 차원에서 어텐션을 수행하는 방식(예: Restormer)으로 최적화를 시도하고 있다.

셋째, 최근 주목받는 Structured State Space Models(SSMs), 특히 Mamba는 입력 데이터에 의존적인 선택적 매커니즘을 통해 선형 복잡도로 전역 문맥을 압축할 수 있음을 보여주었다. Vision Mamba나 U-Mamba와 같은 모델들이 시각 작업 및 의료 영상 분할에 적용되었으며, MambaMixer는 채널 믹싱을 위한 Mamba 블록을 제안하였다. 하지만 기존의 U-shaped Mamba 구조들은 이미지 복원에 필수적인 채널 SSM 모듈을 통합하지 않았다는 한계가 있으며, CU-Mamba는 바로 이 지점에서 차별성을 갖는다.

## 🛠️ Methodology

### 전체 파이프라인 및 구조

CU-Mamba는 4단계의 대칭적 인코더-디코더 구조를 가진 U-Net 아키텍처를 기반으로 한다. 입력 이미지 $I \in \mathbb{R}^{H \times W \times 3}$는 먼저 $3 \times 3$ 컨볼루션을 통해 저수준 특징 $X_0 \in \mathbb{R}^{H \times W \times C}$로 변환된다. 인코더의 각 레벨 $l$은 $N_l$개의 **CU-Mamba Block**과 다운샘플링 레이어로 구성되며, 디코더는 대칭적으로 CU-Mamba Block과 업샘플링 레이어로 구성된다. 특히 인코더의 특징을 디코더에 결합하기 위해 Skip-connection(Concatenation)을 사용하며, 최종 출력은 잔차 학습(Residual Learning) 형태인 $I' = I + R$로 계산된다.

### CU-Mamba Block

각 CU-Mamba Block은 두 가지 SSM 모듈이 순차적으로 배치된 구조이다:

1. **Spatial SSM Block**: 전역 공간 문맥 인코딩 수행.
2. **Channel SSM Block**: 채널 간 상관관계 특징 보존 및 학습 수행.

### 상세 모듈 설명

#### 1. Selective SSM Framework

Mamba의 핵심인 Selective SSM은 1D 시퀀스 입력 $x(t)$를 잠재 상태 $h(t) \in \mathbb{R}^N$를 통해 출력 $y(t)$로 매핑한다.
$$h_t = \bar{A}h_{t-1} + \bar{B}x_t$$
$$y_t = Ch_t$$
여기서 $\bar{A}, \bar{B}$는 고정된 변환을 통한 이산화된 버전이다. 특히 Mamba는 파라미터 $(B, C, \Delta)$를 입력 $x_t$에 따라 동적으로 결정하는 Linear 레이어를 통해 데이터 의존적인 선택적 매커니즘을 구현하여 필요한 정보만을 효율적으로 압축한다.

#### 2. Spatial SSM

입력 텐서 $X \in \mathbb{R}^{H \times W \times C}$에 대해 다음과 같은 절차를 거친다.

- Layer Normalization $\rightarrow 1 \times 1$ Conv $\rightarrow 3 \times 3$ Depth-wise Conv를 통해 지역적 문맥을 먼저 포착한다.
- 특징 맵을 $\hat{X} \in \mathbb{R}^{L \times C}$ (여기서 $L = H \times W$) 형태로 평탄화(Flatten)하여 시퀀스로 변환한다.
- $\hat{X}' = \text{SelectiveSSM}(\hat{X})$를 통해 이미지의 좌상단에서 우하단으로 선형 스캔하며 전역 의존성을 학습한다.
- 결과물을 다시 $\mathbb{R}^{H \times W \times C}$로 복원한다.

#### 3. Channel SSM

채널 차원의 의존성을 학습하기 위해 다음과 같은 절차를 수행한다.

- Spatial SSM과 유사하게 전처리를 수행한 후, 텐서를 전치(Transpose)하여 $\hat{X}^T \in \mathbb{R}^{C \times L}$ 형태로 변환한다.
- $\hat{X}^{T'} = \text{SelectiveSSM}(\hat{X}^T)$를 통해 채널 맵을 상단에서 하단으로 스캔하며 채널 간 특징을 믹싱하고 기억한다.
- 다시 원래 차원으로 복원한 뒤, LeakyReLU 활성화 함수가 포함된 두 개의 $3 \times 3$ Depth-wise Conv 레이어를 통과시켜 지역적 표현을 매끄럽게 다듬는다.

### 손실 함수 (Loss Function)

모델은 공간 도메인의 $\ell_2$ 손실과 주파수 도메인의 $\ell_1$ 손실을 결합하여 학습한다.
$$L(I', \hat{I}) = \sqrt{\|I' - \hat{I}\|^2 + \epsilon} + \lambda \|F(I') - F(\hat{I})\|_1$$
여기서 $\hat{I}$는 정답 이미지, $F$는 푸리에 변환(Fourier Transform)을 의미하며, $\epsilon=10^{-3}$, $\lambda=0.1$로 설정되었다.

### 연산 복잡도

전체 연산 복잡도는 $O(BE(L+C))$로, 시퀀스 길이($L=H \times W$)와 채널 수($C$)에 대해 선형적으로 증가한다. 이는 Transformer의 $O(L^2)$ 복잡도에 비해 매우 효율적이다.

## 📊 Results

### 실험 설정

- **데이터셋**: 실세계 노이즈 제거를 위해 SIDD, DND를 사용하였고, 모션 블러 제거를 위해 GoPro, HIDE, RealBlur-R/J를 사용하였다.
- **지표**: PSNR(Peak Signal-to-Noise Ratio)과 SSIM(Structural Similarity Index)을 사용하였다.

### 주요 결과

1. **Image Denoising**: SIDD 및 DND 데이터셋에서 CU-Mamba는 RIDNet, MPRNet과 같은 CNN 기반 모델은 물론, Uformer, Restormer와 같은 Transformer 기반 모델보다 우수한 성능을 보였다. 특히 SIDD에서 Restormer(40.02 dB)보다 높은 40.22 dB의 PSNR을 달성하였다.
2. **Image Deblurring**: GoPro 및 RealBlur-R 등 4개 데이터셋에서 평가한 결과, 최신 모델인 MRLPFNet을 능가하는 성능을 기록하였다. 특히 RealBlur-R에서 41.01 dB를 기록하며 SOTA 성능을 보였다.
3. **효율성**: GoPro 테스트 데이터셋 기준, Restormer 대비 추론 속도가 약 4배 빠르며(0.305s vs 1.218s), PSNR 또한 0.87 dB 향상됨을 확인하였다.
4. **Ablation Study**:
   - Spatial SSM만 적용했을 때 PSNR이 33.31 dB로 크게 향상되어 장거리 의존성 학습의 중요성이 입증되었다.
   - Channel SSM만 적용했을 때(33.07 dB)보다 두 모듈을 모두 적용했을 때(33.53 dB) 최적의 성능이 나타났다.

## 🧠 Insights & Discussion

본 논문은 이미지 복원 작업에서 전역 수용 영역(Global Receptive Field)의 필요성과 연산 효율성 사이의 트레이드오프를 Mamba 아키텍처를 통해 성공적으로 해결하였다. 특히 단순히 공간적인 전역 문맥을 보는 것을 넘어, 채널 차원에서의 SSM 스캔을 도입한 점이 매우 통찰력 있다. U-Net의 인코더-디코더 경로에서 특징이 압축되고 확장될 때, 채널 간의 상관관계가 명확히 유지되어야 고품질의 복원이 가능하다는 점을 실험적으로 증명하였다.

다만, 본 논문에서 제시된 스캔 방식은 기본적으로 선형 스캔에 의존하고 있으며, 이미지의 2차원적 특성을 완벽하게 반영하기 위한 다방향 스캔(Multi-directional scanning)에 대한 심층적인 논의는 부족한 편이다. 또한, 특정 데이터셋(GoPro 등)에서의 성능 향상은 뚜렷하나, 매우 다양한 종류의 손상(Degradation)이 혼합된 환경에서의 일반화 성능에 대한 분석은 추가적으로 필요할 것으로 보인다.

## 📌 TL;DR

CU-Mamba는 U-Net 구조에 **Spatial SSM**과 **Channel SSM**이라는 듀얼 Mamba 블록을 도입하여, 선형 복잡도로 전역 공간 문맥과 채널 간 상관관계를 동시에 학습하는 이미지 복원 모델이다. 실험 결과, 기존 CNN 및 Transformer 기반 SOTA 모델들보다 더 빠른 추론 속도와 더 높은 복원 성능(PSNR/SSIM)을 달성하였다. 이는 향후 고해상도 이미지 복원 작업에서 연산 효율성과 전역 문맥 유지라는 두 마리 토끼를 잡는 중요한 방향성을 제시한다.
