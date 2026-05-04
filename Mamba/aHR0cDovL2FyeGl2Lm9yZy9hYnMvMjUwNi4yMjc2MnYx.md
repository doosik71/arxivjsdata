# VSRM: A Robust Mamba-Based Framework for Video Super-Resolution

Dinh Phu Tran, Dao Duy Hung, Daeyoung Kim (2025)

## 🧩 Problem to Solve

비디오 초해상도(Video Super-Resolution, VSR)의 핵심은 저해상도(LR) 비디오 프레임들로부터 고해상도(HR) 프레임을 복원하는 것이다. 이를 위해 인접한 여러 프레임에서 상보적인 정보를 효율적으로 추출하고 정렬하는 능력이 필수적이며, 특히 긴 시퀀스를 처리하기 위해 넓은 수용 영역(Receptive Field)을 확보하는 것이 중요하다.

기존의 접근 방식들은 다음과 같은 한계가 있다:
- **CNN 기반 방법**: 수용 영역이 국소적(local)으로 제한되어 있어, 프레임 간의 장거리 상관관계를 포착하는 데 한계가 있다.
- **Transformer 기반 방법**: Attention 메커니즘을 통해 강력한 성능을 보이지만, 연산 복잡도가 시퀀스 길이의 제곱에 비례하는 Quadratic Complexity 문제를 가진다. 이를 해결하기 위해 Window-based attention을 사용하기도 하지만, 이는 다시 수용 영역을 제한하는 결과를 초래한다.

본 논문의 목표는 연산 효율성(Linear Complexity)을 유지하면서도 넓은 수용 영역을 확보하여, 긴 비디오 시퀀스에서도 효율적이고 정확하게 고해상도 영상을 복원할 수 있는 Mamba 기반의 VSR 프레임워크인 VSRM을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 최근 NLP와 컴퓨터 비전 분야에서 주목받는 State Space Model(SSM)인 Mamba를 VSR 작업에 최초로 도입하여 효율성과 성능의 균형을 맞춘 것이다. 주요 기여 사항은 다음과 같다:

1. **Dual Aggregation Mamba Block (DAMB)**: 공간적 정보에서 시간적 정보로($S2T$), 그리고 시간적 정보에서 공간적 정보로($T2S$) 스캔하는 두 종류의 Mamba 블록을 구성하여, 장거리 시공간 특징(Spatio-temporal features)을 효과적으로 추출한다.
2. **Deformable Cross-Mamba Alignment (DCA)**: 객체의 움직임으로 인한 프레임 간 미정렬 문제를 해결하기 위해, Deformable window와 Cross-mamba 메커니즘을 결합하여 동적이고 유연한 정렬을 수행한다.
3. **Frequency Charbonnier-like Loss (FCL)**: 공간 도메인의 픽셀 손실뿐만 아니라, 푸리에 변환을 통한 주파수 도메인에서의 손실을 추가하여 고주파 성분(에지, 텍스처 등)을 더 잘 보존하고 복원한다.

## 📎 Related Works

### 기존 연구 및 한계
- **VSR 방법론**: early CNN 방식(VSR-net, TDAN)은 수용 영역의 한계가 있었고, 이후 BasicVSR 등이 효율적인 구성 요소를 제안했다. Transformer 기반의 VRT, PSRT, IART 등은 높은 성능을 보였으나, 연산 비용 문제로 인해 Window-based attention을 채택함으로써 다시 수용 영역이 제한되는 트레이드-오프 관계에 있었다.
- **State Space Model (SSM)**: S4, Mamba 등은 Linear Complexity로 긴 시퀀스를 모델링할 수 있음을 보여주었다. Vim이나 VideoMamba가 이미지 및 비디오 인식 작업에 적용되었고, MambaIR이 이미지 복원 작업에 쓰였으나 VSR에 적용된 사례는 없었다.
- **Spectral Bias**: 신경망이 고주파보다 저주파 성분에 편향되는 경향이 있다. 이를 해결하기 위해 Focal Frequency Loss(FFL) 등이 제안되었으나, 저주파와 고주파 성분의 그래디언트 균형을 맞추는 데 한계가 있었다.

### VSRM의 차별점
VSRM은 Mamba의 선형 복잡도와 전역 수용 영역이라는 장점을 VSR에 통합하였다. 특히 단순한 적용에 그치지 않고, VSR의 특성에 맞는 정렬 모듈(DCA)과 고주파 복원을 위한 전용 손실 함수(FCL)를 설계하여 기존 Transformer 기반 모델보다 효율적이면서도 더 넓은 수용 영역을 가진다.

## 🛠️ Methodology

### 1. 전체 시스템 구조
VSRM은 크게 **특징 추출(Feature Extraction)** 단계와 **업샘플러(Upsampler)** 단계로 구성된다. 특징 추출 단계에서는 Convolution 레이어와 Feature Propagation Block(FPB)을 통해 인접 프레임 간의 특징을 정렬하고 융합하며, 이후 업샘플러가 이를 고해상도로 확장한다.

### 2. Dual Aggregation Mamba Block (DAMB)
DAMB는 시공간 특징을 완전히 추출하기 위해 $N$개의 $\text{S2TMB}$와 1개의 $\text{T2SMB}$로 구성된다.

- **Spatial-to-Temporal Mamba (S2T-Mamba)**: 3D 시퀀스를 1D로 변환하여 먼저 공간적으로 스캔한 뒤 시간적으로 스캔한다. 전방향(Forward)과 후방향(Backward) SSM을 모두 사용하여 공간적 인지 능력을 보존한다.
  $$ \text{S2T-Mamba}(x,z) = \text{Linear}(x_1 \odot z + x_2 \odot z) $$
- **Temporal-to-Spatial Mamba (T2S-Mamba)**: 시간적 스캔을 우선시하여 S2TMB가 놓친 시간적 정보를 보완한다. 연산 효율성과 실험적 성능을 근거로 전방향 스캔만 수행한다.
  $$ \text{T2S-Mamba}(x,z) = \text{Linear}(x_1 \odot z) $$
- **Temporal Gated Feed-forward Network (TGFN)**: 기존 FFN의 단순함을 극복하기 위해 3D Depthwise Convolution(DW-3D Conv)과 Gating 메커니즘을 도입하여 인접 픽셀 간의 시공간 관계를 모델링한다.

### 3. Deformable Cross-Mamba Alignment (DCA)
프레임 간의 큰 움직임이나 미세한 움직임에 유연하게 대응하기 위한 정렬 모듈이다.
- **동적 참조 영역 생성**: SpyNet으로 추정된 Optical Flow를 기반으로 워핑(Warping)을 수행하며, 학습 가능한 오프셋 네트워크 $S(\cdot)$를 통해 고정된 윈도우 내에서 동적인 참조 영역 $\bar{r}$을 생성한다.
  $$ \bar{r} = \phi(w; r + \epsilon_r), \quad \epsilon_r = S(w) $$
- **Cross-Mamba 정렬**: 타겟 텐서 $Q$와 참조 텐서 $R$ 사이의 상호작용을 Cross-Mamba 모듈을 통해 계산하여 최종 정렬된 픽셀 $\bar{X}(x,y)$를 얻는다. 이는 고정된 가중치를 사용하는 선형 보간법보다 훨씬 유연한 정렬을 가능하게 한다.

### 4. 훈련 목표 및 손실 함수
모델은 공간 도메인의 $\text{Charbonnier Loss (CL)}$와 주파수 도메인의 $\text{Frequency Charbonnier-like Loss (FCL)}$를 동시에 최적화한다.
- **Charbonnier Loss (CL)**: 픽셀 간의 절대적 차이를 최소화한다.
  $$ L_{CL} = \sqrt{\|I^{SR} - I^{HR}\|^2 + \epsilon^2} $$
- **Frequency Charbonnier-like Loss (FCL)**: FFT를 통해 영상을 주파수 도메인으로 변환한 후, 실수부(Real)와 허수부(Imaginary)에 대해 각각 손실을 계산한다. 이는 크기(Magnitude)나 위상(Phase)을 사용할 때 발생하는 불연속성 문제를 피하고 고주파 세부 사항을 직접적으로 복원하기 위함이다.
  $$ L_{FCL} = \sum_{i \in \{Re, Im\}} \lambda_i \sqrt{\|iF(I^{SR}) - iF(I^{HR})\|^2 + \epsilon^2} $$
- **최종 손실 함수**: $L_{total} = \lambda L_{CL} + L_{FCL}$

## 📊 Results

### 실험 설정
- **데이터셋**: REDS4, Vimeo-90K-T, Vid4 (4배 확대 작업).
- **평가 지표**: PSNR, SSIM.
- **비교 대상**: EDVR, VSR-T, VRT, BasicVSR++, RVRT, PSRT-recurrent, IART.

### 주요 결과
- **정량적 성능**: 모든 벤치마크에서 SOTA(State-of-the-art)를 달성하였다. 특히 REDS4 데이터셋(16프레임 입력)에서 PSNR 33.11dB를 기록하며 IART(32.90dB)나 PSRT-recurrent(32.72dB)를 앞질렀다.
- **정성적 성능**: 시각적 비교 결과, 다른 방법론들이 뭉개뜨리는 번호판의 숫자나 스파이더맨 의상의 세밀한 패턴 등을 더 정확하게 복원하였다.
- **수용 영역(ERF)**: ERF 시각화 결과, CNN 기반의 EDVR이나 Window-based Transformer인 IART보다 훨씬 넓은 전역 수용 영역(Global Receptive Field)을 가짐이 확인되었다.

### Ablation Study 결과
- **Mamba의 효과**: Mamba 모듈을 Convolution이나 Window Attention으로 대체했을 때보다 PSNR이 각각 0.25dB, 0.12dB 상승하였다. Full Attention과 유사한 성능을 내면서 연산 비용은 훨씬 낮다.
- **DCA의 효과**: 정렬 모듈을 제거했을 때 성능이 0.22dB 하락하였으며, 기존의 FGDA나 IA 방식보다 DCA가 더 높은 PSNR을 기록했다.
- **T2SMB의 필요성**: T2SMB를 제거하면 성능이 0.14dB 하락하며, 이는 S2TMB만으로는 시간적 정보를 충분히 추출할 수 없음을 시사한다.
- **FCL의 효과**: FCL을 적용하지 않았을 때보다 PSNR이 0.12dB 상승하였으며, 다른 주파수 손실 함수(FFL, FSL, WHFL)보다 우수한 성능을 보였다.

## 🧠 Insights & Discussion

본 논문은 Mamba를 VSR에 도입함으로써 **'연산 복잡도 $\rightarrow$ 수용 영역 $\rightarrow$ 복원 성능'**으로 이어지는 기존의 트레이드-오프 관계를 효과적으로 해결하였다.

- **강점**: Mamba의 Linear Complexity 덕분에 긴 시퀀스를 처리하면서도 전역적인 컨텍스트를 유지할 수 있다. 또한, 주파수 도메인에서 실수부와 허수부를 직접 다루는 FCL은 기존의 복잡한 재가중치 행렬 방식보다 단순하면서도 고주파 성분을 효과적으로 복원하는 전략임을 입증하였다.
- **한계 및 논의**: 모델의 파라미터 수와 FLOPs 면에서는 효율적이지만, 실제 추론 시간(Runtime)은 Window-based Transformer보다 약간 느린 경향을 보인다. 이는 Mamba의 하드웨어 가속 및 최적화가 아직 진행 중인 단계이기 때문이며, 향후 커널 최적화를 통해 개선될 가능성이 크다.
- **비판적 해석**: DCA 모듈에서 SpyNet이라는 사전 학습된 모델에 의존하고 있는데, 이는 End-to-End 학습 관점에서 잠재적인 제약이 될 수 있다. 하지만 VSR 작업의 특성상 정확한 정렬이 우선시되므로 합리적인 선택으로 판단된다.

## 📌 TL;DR

VSRM은 Mamba 구조를 VSR에 최초로 적용하여, **선형 복잡도**와 **전역 수용 영역**이라는 두 마리 토끼를 잡은 프레임워크이다. 시공간 특징을 효율적으로 추출하는 **DAMB**, 유연한 정렬을 수행하는 **DCA**, 그리고 고주파 세부 사항을 복원하는 **FCL** 손실 함수를 통해 기존 SOTA 모델들을 능가하는 성능을 보였다. 이 연구는 향후 Transformer를 대체하여 비디오 복원 및 다양한 저수준 비전 작업(Low-level vision)의 새로운 베이스라인이 될 가능성이 매우 높다.