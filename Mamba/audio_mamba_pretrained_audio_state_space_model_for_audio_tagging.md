# AUDIO MAMBA: PRETRAINED AUDIO STATE SPACE MODEL FOR AUDIO TAGGING

Jiaju Lin, Haoxuan Hu (2024)

## 🧩 Problem to Solve

본 논문은 오디오 샘플을 해당 카테고리로 매핑하는 오디오 태깅(Audio Tagging) 작업에서 발생하는 계산 효율성 문제를 해결하고자 한다. 최근 오디오 표현 학습 분야에서는 Transformer 기반 모델들이 뛰어난 성능을 보이며 CNN을 대체하는 추세이다. 그러나 Transformer의 핵심인 Self-Attention 메커니즘은 입력 시퀀스 길이에 대해 이차 복잡도($O(n^2)$)의 계산 비용이 발생한다. 이러한 특성은 긴 오디오 스펙트로그램(Audio Spectrogram)을 처리할 때 모델의 확장성을 제한하며, 결과적으로 더 범용적인 오디오 모델의 개발을 저해하는 병목 현상으로 작용한다. 따라서 본 연구의 목표는 Self-Attention 없이도 긴 오디오 시퀀스의 의존성을 효과적으로 캡처할 수 있는 선형 복잡도($O(n)$)의 새로운 아키텍처를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 State Space Models (SSMs)를 오디오 태깅 작업에 도입한 **AudioMamba** 아키텍처를 제안한 것이다. 중심적인 설계 아이디어는 다음과 같다.

1. **선형 복잡도 구현**: Self-Attention을 제거하고 Mamba 아키텍처를 채택함으로써 입력 시퀀스 길이에 비례하는 선형 시간 복잡도를 달성하였다.
2. **계층적 구조의 통합**: HT-SAT의 계층적 구조 및 패치 임베딩 방식과 VMamba의 SSM 기반 백본을 결합하여, 다양한 스케일의 오디오 특징을 효율적으로 추출할 수 있도록 설계하였다.
3. **파라미터 효율성 입증**: SOTA(State-of-the-art) 성능을 보이는 오디오 스펙트로그램 Transformer 모델들과 비교하여, 약 3분의 1 수준의 파라미터만으로도 대등한 성능을 낼 수 있음을 실험적으로 증명하였다.

## 📎 Related Works

논문에서는 오디오 분류 및 탐지 작업을 위한 기존의 주요 연구들을 다음과 같이 설명한다.

- **HTS-AT**: 계층적 구조를 사용하여 모델 크기와 학습 시간을 줄이고, 토큰-시맨틱 모듈을 통해 이벤트 지역화 성능을 향상시킨 Transformer 모델이다.
- **Patchout**: Transformer의 계산 복잡도 문제를 해결하기 위해 학습 과정에서 입력 시퀀스의 일부를 무작위로 드롭하는 정규화 및 효율적 학습 방법을 제안하였다.
- **AST (Audio Spectrogram Transformer)**: 합성곱 층 없이 순수하게 Attention 메커니즘만을 사용하여 스펙트로그램을 직접 처리하는 모델이다.
- **PANNs (Pretrained Audio Neural Networks)**: 대규모 AudioSet 데이터셋을 통해 사전 학습된 CNN 기반 모델로, 전이 학습(Transfer Learning)의 효과를 입증하였다.

기존의 Transformer 기반 접근 방식들은 뛰어난 성능을 보이지만, 앞서 언급한 $O(n^2)$의 계산 비용 문제가 상존한다. AudioMamba는 이러한 한계를 SSM을 통해 극복함으로써 효율성과 성능의 균형을 맞추고자 한다.

## 🛠️ Methodology

AudioMamba는 오디오 스펙트로그램의 계층적 표현을 처리하기 위해 설계된 신경망이다. 전체 시스템은 크게 오디오 스펙트로그램 인코딩과 Mamba 백본 네트워크로 구성된다.

### 1. 오디오 스펙트로그램 인코딩 (Encode the Audio Spectrogram)

일반적인 Vision Transformer의 격자형(Grid-like) 패치 추출 방식은 이미지의 가로-세로가 모두 공간적 차원이지만, 오디오 멜-스펙트로그램(Mel-spectrogram)은 가로가 시간(Time), 세로가 주파수(Frequency)를 의미한다. 또한 시간 차원이 주파수 차원보다 훨씬 길다. 이를 해결하기 위해 본 모델은 다음과 같은 2단계 패치 추출 과정을 거친다.

1. 멜-스펙트로그램을 패치 윈도우 $w_1, w_2, \dots, w_n$으로 나눈다.
2. 각 윈도우를 다시 개별 패치로 분할한다.
3. 결과적인 패치 토큰의 순서를 $\text{time} \rightarrow \text{frequency} \rightarrow \text{window}$ 순으로 구성하여, 동일 시간 프레임 내의 서로 다른 주파수 빈(bin)들이 입력 시퀀스에서 인접하게 배치되도록 한다.

### 2. Audio Mamba 백본 (Audio Mamba Backbone)

모델은 다단계(Multi-stage) 구조를 통해 특징을 점진적으로 정제한다.

- **구조적 흐름**: Patch Embeddings $\rightarrow$ Stage 1 $\rightarrow$ Downsampling $\rightarrow$ Stage 2 $\rightarrow$ Downsampling $\rightarrow$ Stage 3 $\rightarrow$ Downsampling $\rightarrow$ Stage 4 순으로 진행된다.
- **해상도 변화**: 다운샘플링(Patch Merging)을 통해 각 단계에서 해상도를 $\frac{F}{8} \times \frac{T}{8}$, $\frac{F}{16} \times \frac{T}{16}$, $\frac{F}{32} \times \frac{T}{32}$로 점차 줄이며 채널 깊이를 증가시킨다.
- **SS Block (State Space Block)**: 모델의 핵심 구성 요소이며, 다음과 같은 모듈로 이루어져 있다.
  - **Depth-wise Convolution (DWConv)**: 지역적 특징을 추출한다.
  - **SiLU Activation**: 비선형 활성화 함수를 적용한다.
  - **SS2D**: Selective Scanning 기법을 2D 데이터에 적응시킨 연산으로, 2차원 데이터의 의존성을 캡처한다.
  - **FFN (Feed Forward Network)**: 최종적으로 특징 추출 능력을 강화한다.

## 📊 Results

### 실험 설정

- **데이터셋**: AudioSet (약 200만 개의 10초 길이 클립, 527개 클래스) 및 ESC-50.
- **전처리**: 32kHz 샘플링 레이트의 모노 오디오를 64 멜 빈(Mel bins)의 멜-스펙트로그램으로 변환. 최종 입력 크기는 $T=1024, F=64$이다.
- **평가 지표**:
  - **mAP (mean Average Precision)**: 모든 클래스에 대한 평균 정밀도.
  - **mAUC (mean Area Under Curve)**: ROC 곡선 아래 면적의 평균.
  - **d-prime ($d'$)**: 신호와 소음 분포의 평균 차이를 표준편차의 합으로 나눈 값.
    $$d' = \frac{\mu_1 - \mu_2}{\sqrt{\frac{\sigma_1^2 + \sigma_2^2}{2}}}$$

### 주요 결과

- **AudioSet 성능**: AudioMamba-Tiny 모델은 기존 SOTA 모델인 HT-SAT와 대등하거나 더 높은 mAP(0.440)를 기록했으며, 특히 mAUC(0.963)와 dPrime(2.51)에서 더 우수한 성능을 보였다.
- **모델 효율성**: AudioMamba-Micro(12.3M)와 Nano(5.2M) 모델은 파라미터 수를 획기적으로 줄이면서도 경쟁력 있는 성능을 유지하여, SSM의 파라미터 효율성을 입증하였다.
- **Ablation Study (ESC-50)**:
  - Transformer 블록 추가나 CutMix 데이터 증강은 성능 향상에 미미한 영향을 주었다.
  - 반면, ImageNet으로 사전 학습된 **VMamba 모델을 통합했을 때 성능이 비약적으로 향상**됨을 확인하였다.

## 🧠 Insights & Discussion

본 연구를 통해 SSM이 오디오 태깅 작업에서 Transformer를 대체할 수 있는 강력한 후보임을 확인하였다. 특히 대규모 데이터셋인 AudioSet에서 적은 파라미터로 높은 성능을 낸 점은 고무적이다.

그러나 분석 결과, AudioMamba는 소규모 데이터셋인 ESC-50에서 scratch 학습 시 HT-SAT보다 낮은 성능을 보였다. 이는 Mamba 아키텍처가 Swin Transformer와 같이 지역적 특징을 명시적으로 모델링하는 구조에 비해, 수렴을 위해 더 많은 양의 데이터를 필요로 함을 시사한다. 즉, 데이터 규모가 작을 때는 기존의 Transformer 계열이 유리할 수 있으나, 데이터 규모가 커질수록 Mamba의 확장성과 효율성이 극대화된다고 해석할 수 있다.

## 📌 TL;DR

본 논문은 Transformer의 $O(n^2)$ 복잡도 문제를 해결하기 위해 SSM 기반의 **AudioMamba**를 제안하였다. 이 모델은 특유의 2D Selective Scanning(SS2D)과 계층적 구조를 통해 오디오 스펙트로그램의 장기 의존성을 효율적으로 학습하며, **기존 SOTA 모델 대비 훨씬 적은 파라미터로 대등하거나 우월한 성능을 달성**하였다. 향후 오디오 자가 지도 학습(Self-supervised learning) 및 범용 오디오 아키텍처 설계에 있어 SSM의 가능성을 열어준 연구이다.
