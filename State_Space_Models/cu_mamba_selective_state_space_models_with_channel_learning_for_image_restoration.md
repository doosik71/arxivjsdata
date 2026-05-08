# CU-Mamba: Selective State Space Models with Channel Learning for Image Restoration

Rui Deng, Tianpei Gu (2024)

## 🧩 Problem to Solve

본 논문은 노이즈, 블러, 비 오는 거리의 줄무늬 등 다양한 열화(degradation)가 발생한 이미지를 고품질 이미지로 복원하는 이미지 복원(Image Restoration) 문제를 다룬다.

기존의 이미지 복원 모델들은 주로 CNN(Convolutional Neural Networks)과 Transformer 기반의 아키텍처를 사용해 왔다. 그러나 CNN은 수용 영역(receptive field)이 제한적이어서 이미지 내의 장거리 의존성(long-range dependency)을 포착하는 데 한계가 있다. 반면, Transformer는 self-attention 메커니즘을 통해 글로벌 컨텍스트를 모델링할 수 있지만, 피처 맵의 크기에 따라 계산 복잡도가 제곱(quadratic)으로 증가하는 심각한 비용 문제가 발생하며, 때로는 복원에 필수적인 세밀한 지역적 세부 사항을 놓치는 경향이 있다.

최근 등장한 Mamba와 같은 구조적 상태 공간 모델(Structured State Space Models, SSMs)은 선형 계산 복잡도로 글로벌 수용 영역을 확보할 수 있는 대안으로 주목받고 있다. 하지만 대부분의 시각적 Mamba 모델들은 SSM 블록을 각 채널에 독립적으로 적용하기 때문에, 채널 간의 정보 흐름(information flow across channels)을 무시하게 된다. 이는 이미지 세부 사항을 압축하고 재구성해야 하는 이미지 복원 작업에서 치명적인 정보 손실을 야기할 수 있다. 따라서 본 논문의 목표는 글로벌 공간 컨텍스트와 채널 간 상관관계를 동시에 효율적으로 학습할 수 있는 모델을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 U-Net 구조 내에 **Dual State Space Model(SSM)** 프레임워크를 통합하여 공간적 정보와 채널 정보를 동시에 학습하는 **CU-Mamba(Channel-Aware U-Shaped Mamba)** 모델을 제안하는 것이다.

중심적인 설계 직관은 다음과 같다.

1. **Spatial SSM**: 이미지의 공간적 차원을 스캔하여 선형 복잡도로 장거리 의존성을 포착함으로써 글로벌 컨텍스트를 인코딩한다.
2. **Channel SSM**: 채널 차원을 스캔하여 채널 간의 특성 혼합(feature mixing)을 수행함으로써, U-Net의 다운샘플링 및 업샘플링 과정에서 발생하는 채널 간 상관관계 정보를 보존한다.

이러한 이중 SSM 접근 방식을 통해 모델은 연산 효율성을 유지하면서도, 광범위한 공간적 세부 사항과 복잡한 채널 간 상관관계 사이의 균형을 맞추어 복원 품질을 극대화한다.

## 📎 Related Works

- **CNN 기반 접근 방식**: U-Net과 같은 인코더-디코더 구조와 스킵 연결(skip connection)을 통해 계층적 다중 스케일 특성을 학습하며 우수한 성능을 보였으나, 지역적인 수용 영역으로 인해 글로벌 컨텍스트 학습에 한계가 있다.
- **Transformer 기반 접근 방식**: 글로벌 self-attention을 통해 장거리 상호작용을 캡처하며 초해상도, 디노이징, 디블러링 등에서 SOTA 성능을 달성했다. 하지만 계산 복잡도가 매우 높으며, 이를 줄이기 위해 로컬 윈도우나 채널 차원에서 attention을 수행하는 방식이 제안되었음에도 불구하고 여전히 오버헤드가 크다.
- **시각적 구조적 상태 공간 모델(Visual SSMs)**: Mamba 모델은 입력 의존적인 선택적 SSM(selective SSM)을 통해 선형 복잡도로 글로벌 컨텍스트를 압축한다. Vision Mamba와 같은 연구가 진행되었으며, MambaMixer는 채널 혼합 Mamba 블록을 도입했다. 그러나 기존의 U-shaped Mamba 아키텍처들은 이미지 복원에 필수적인 채널 SSM 모듈을 통합하지 않았다는 한계가 있다.

## 🛠️ Methodology

### 전체 파이프라인

CU-Mamba는 4단계의 대칭적 인코더-디코더 U-Net 구조를 따른다. 입력된 열화 이미지 $I \in \mathbb{R}^{H \times W \times 3}$는 먼저 $3 \times 3$ 컨볼루션을 통해 저수준 특징 $X_0 \in \mathbb{R}^{H \times W \times C}$로 변환된다.

- **Encoder**: 각 레벨 $l$에서 $N_l$개의 **CU-Mamba Block**과 다운샘플링 레이어로 구성된다. 다운샘플링은 공간 크기를 줄이고 채널 수를 확장한다.
- **Decoder**: 인코더와 대칭적인 CU-Mamba Block과 업샘플링 레이어로 구성된다. 업샘플링은 공간 크기를 두 배로 늘리고 채널 수를 절반으로 줄인다.
- **Skip Connection**: 인코더의 특징 맵을 디코더의 대응하는 특징 맵과 결합(concatenation)하여 세부 정보를 복원한다.
- **최종 출력**: 출력 투영 블록을 거쳐 잔차 $R$을 생성하며, 최종 복원 이미지는 $I' = I + R$로 계산된다.

### 손실 함수

모델은 다음과 같은 손실 함수를 사용하여 학습된다.
$$L(I', \hat{I}) = \sqrt{\|I' - \hat{I}\|^2 + \epsilon} + \lambda \|F(I') - F(\hat{I})\|_1$$
여기서 $\hat{I}$는 Ground-truth 이미지, $F$는 주파수 영역으로의 푸리에 변환(Fourier transform)을 의미한다. $\epsilon=10^{-3}$, $\lambda=0.1$로 설정되어 픽셀 값의 차이와 주파수 도메인에서의 일관성을 동시에 최적화한다.

### Selective SSM (Mamba) 프레임워크

기존 SSM은 데이터에 독립적인 파라미터를 가지나, Mamba는 입력 $x_t$에 따라 결정되는 데이터 의존적 파라미터 $(B, C, \Delta)$를 도입한다.

- $B_t = \text{Linear}_B(x_t), C_t = \text{Linear}_C(x_t), \Delta_t = \text{SoftPlus}(\text{Linear}_\Delta(x_t))$
이 메커니즘은 하드웨어 최적화를 통해 시퀀스 길이에 대해 선형적인 계산 및 메모리 복잡도를 유지하면서도 필요한 정보를 선택적으로 압축한다.

### Spatial SSM 블록

입력 텐서 $X \in \mathbb{R}^{H \times W \times C}$에 대해 다음 과정을 거친다.

1. $1 \times 1$ 컨볼루션으로 픽셀 수준의 채널 컨텍스트를 수집하고, $3 \times 3$ depth-wise 컨볼루션으로 공간 컨텍스트를 캡처한다.
2. 피처 맵을 $L = H \times W$ 길이의 시퀀스로 평탄화(flatten)하여 $\hat{X} \in \mathbb{R}^{L \times C}$를 생성한다.
3. Selective SSM을 적용하여 $\hat{X}' = \text{SelectiveSSM}(\hat{X})$를 얻는다. 이는 이미지를 좌상단에서 우하단으로 선형 스캔하며 각 픽셀이 이전의 모든 컨텍스트로부터 표현을 학습하는 것과 같다.
4. 결과를 다시 $H \times W \times C$ 형태로 복원한다.

### Channel SSM 블록

채널 간의 의존성을 학습하기 위해 다음 과정을 거친다.

1. Spatial SSM과 마찬가지로 $1 \times 1$ 및 $3 \times 3$ depth-wise 컨볼루션으로 지역 컨텍스트를 전처리한다.
2. 텐서를 전치(transpose)하고 평탄화하여 $\hat{X}^T \in \mathbb{R}^{C \times L}$를 생성한다. 이는 평탄화된 픽셀들을 채널의 표현으로 사용하는 것이다.
3. $\hat{X}^{T'} = \text{SelectiveSSM}(\hat{X}^T)$를 통해 채널 맵을 위에서 아래로 스캔하며 채널 간 특징을 혼합하고 기억한다.
4. 결과를 다시 원래 형태로 복원한 후, $3 \times 3$ depth-wise 컨볼루션 2개와 LeakyReLU 활성화 함수를 적용하여 지역적 표현을 매끄럽게 다듬는다.

### 계산 복잡도

배치 크기를 $B$, 시퀀스 길이를 $L(H \times W)$, 채널 차원을 $C$, 확장 계수를 $E$라고 할 때:

- Spatial SSM의 복잡도는 $O(BLE + EC)$이다.
- Channel SSM의 복잡도는 $O(BCE + EL)$이다.
- 전체 복잡도는 $O(BE(L+C))$로, 시퀀스 길이와 채널 차원에 대해 모두 **선형(linear)**이다.

## 📊 Results

### 실험 설정

- **데이터셋**: 실세계 노이즈 제거를 위해 SIDD, DND를 사용하였고, 모션 블러 제거를 위해 GoPro, HIDE, RealBlur-R, RealBlur-J를 사용하였다.
- **지표**: PSNR(Peak Signal-to-Noise Ratio)과 SSIM(Structural Similarity Index)을 사용하였다.

### 정량적 결과

1. **이미지 디노이징 (Denoising)**:
   - SIDD 데이터셋에서 CU-Mamba는 40.22 dB의 PSNR을 기록하며, RIDNet, MPRNet, Uformer, Restormer 등의 기존 SOTA 모델들보다 우수하거나 대등한 성능을 보였다.
   - DND 데이터셋에서도 40.34 dB를 달성하여 경쟁력을 입증하였다.

2. **이미지 디블러링 (Deblurring)**:
   - GoPro, HIDE, RealBlur-R, RealBlur-J의 4개 데이터셋 모두에서 매우 높은 성능을 보였다. 특히 RealBlur-R에서 MRLPFNet 대비 0.09 dB 향상된 성능을 보였다.
   - **추론 속도**: GoPro 테스트셋 기준, Restormer보다 약 **4배 빠른 추론 속도**($0.305\text{s}$ vs $1.218\text{s}$)를 기록하면서도 PSNR은 $0.87\text{dB}$ 더 높았다.

### 절제 연구 (Ablation Study)

GoPro 데이터셋을 이용한 분석 결과:

- **Baseline (UNet)**: PSNR 32.45 dB
- **Spatial SSM 단독**: 33.31 dB (가장 큰 성능 향상)
- **Channel SSM 단독**: 33.07 dB
- **Spatial + Channel SSM (CU-Mamba)**: 33.53 dB (최고 성능)
이를 통해 장거리 공간 의존성 학습이 매우 중요하며, 채널 SSM이 이를 보완하여 최적의 성능을 낸다는 것을 확인하였다.

## 🧠 Insights & Discussion

CU-Mamba는 Transformer의 글로벌 모델링 능력과 CNN의 효율성 사이의 트레이드-오프 문제를 SSM을 통해 성공적으로 해결하였다.

**강점 및 해석**:

- **효율적인 글로벌 수용 영역**: $O(L^2)$의 복잡도를 가진 Transformer와 달리 $O(L)$의 선형 복잡도로 글로벌 컨텍스트를 학습함으로써, 고해상도 이미지 복원 시 발생하는 연산 비용 문제를 획기적으로 줄였다.
- **채널-공간 이중 학습**: 기존 Mamba 기반 모델들이 간과했던 채널 간 상관관계를 Channel SSM으로 해결함으로써, 이미지 복원에 필수적인 풍부한 채널 특징을 보존하고 재구성할 수 있게 되었다.
- **실용성**: 추론 속도가 4배나 빠르면서 성능이 향상되었다는 점은 실제 실시간 이미지 처리 시스템에 적용될 가능성이 매우 높음을 시사한다.

**한계 및 논의**:

- 논문에서 제시된 실험 결과는 매우 긍정적이나, 다양한 종류의 열화(예: 복합 열화)에 대한 일반화 성능에 대한 언급은 부족하다.
- SSM의 스캔 방식이 기본적으로 선형적(Linear scan)이므로, 2차원 이미지의 특성을 더 잘 반영하기 위한 다방향 스캔(Bi-directional or multi-directional)의 최적화 가능성이 남아 있다.

## 📌 TL;DR

CU-Mamba는 U-Net 구조에 **Spatial SSM**과 **Channel SSM**을 결합하여, 연산 복잡도를 선형($O(L+C)$)으로 유지하면서도 이미지의 글로벌 공간 정보와 채널 간 상관관계를 모두 학습하는 이미지 복원 모델이다. 실세계 디노이징 및 디블러링 작업에서 기존 Transformer 기반 모델(Restormer 등)보다 훨씬 빠른 속도와 더 높은 복원 품질을 달성하였으며, 이는 향후 고해상도 실시간 이미지 복원 연구에 중요한 이정표가 될 것으로 보인다.
