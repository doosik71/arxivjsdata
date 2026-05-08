# From Vision to Audio and Beyond: A Unified Model for Audio-Visual Representation and Generation

Kun Su, Xiulong Liu, Eli Shlizerman (2024)

## 🧩 Problem to Solve

본 논문은 오디오-비주얼(Audio-Visual) 도메인에서 서로 분리되어 발전해 온 **오디오-비주얼 표현 학습(Representation Learning)**과 **비전-투-오디오 생성(Vision-to-Audio Generation)**을 하나의 통합된 프레임워크로 결합하고자 한다.

기존의 연구들은 오디오-비주얼 정렬을 통해 특징을 추출하는 결정론적(Deterministic) 모델에 집중하거나, 반대로 비주얼 조건 하에서 오디오를 생성하는 생성 모델에 치중하는 경향이 있었다. 특히 비디오와 오디오 데이터는 모두 고차원이며 시간적 의존성이 강해, 이를 동시에 처리하는 통합 모델을 구축하는 것은 계산 비용과 모델 복잡도 측면에서 매우 도전적인 과제이다. 따라서 본 연구의 목표는 잠재 공간(Latent Space) 내에서 표현 학습과 생성 모델링을 동시에 수행함으로써, 고품질의 오디오 생성 능력과 강력한 오디오-비주얼 이해 능력을 모두 갖춘 통합 모델인 **VAB(Vision to Audio and Beyond)**를 제안하는 것이다.

## ✨ Key Contributions

VAB의 핵심 아이디어는 원본 비디오 프레임과 오디오 신호를 직접 다루지 않고, **사전 학습된 토크나이저와 인코더를 통해 압축된 잠재 공간(Latent Space)에서 모든 학습과 생성을 수행**하는 것이다.

1. **통합 프레임워크 제안**: 오디오-비주얼 표현 학습과 조건부 오디오 생성을 하나의 모델에서 지원하는 최초의 파운데이션 모델 구조를 제시하였다.
2. **병렬 디코딩(Parallel Decoding) 도입**: 기존의 자기회귀(Autoregressive) 방식과 달리, 마스크 기반의 반복적 디코딩을 통해 생성 속도를 획기적으로 높였다. 이는 비전-투-오디오 생성 작업에 병렬 디코딩을 적용한 첫 번째 사례이다.
3. **범용적 적응성**: 사전 학습된 VAB 백본을 기반으로 대조 학습(Contrastive Learning) 및 선형 분류기(Linear Classifier)를 추가하여, 리트리벌(Retrieval) 및 분류(Classification) 등 다양한 다운스트림 작업에서 최신 성능(SOTA)을 달성하였다.

## 📎 Related Works

### 1. Video to Audio Generation

기존의 비전-투-오디오 생성 연구는 객체나 장면의 시맨틱 정렬을 이용하거나, 최근에는 Diffusion 모델 및 discrete token 기반의 생성 모델을 사용하였다. 그러나 이러한 모델들은 오직 생성 목적에만 특화되어 있어, 오디오-비주얼 간의 상호 이해나 표현 학습과 같은 분석적 작업에는 활용될 수 없다는 한계가 있다.

### 2. Audio-visual Representation Learning

자기지도 학습(Self-supervised Learning)을 통한 표현 학습 연구들은 주로 Contrastive Learning이나 Masked Autoencoder(MAE) 방식을 사용하였다. 하지만 대부분의 기존 모델들이 raw video frame이나 spectrogram patches 레벨에서 재구성 손실(Reconstruction loss)을 계산하므로, 데이터의 고차원성으로 인해 모델링의 복잡도가 높고 생성 작업으로의 확장이 어렵다는 문제가 있다.

### 3. Masked Generative Modeling

MaskGIT나 Muse와 같은 연구들은 bidirectional transformer를 이용해 마스크된 토큰을 예측함으로써 이미지 생성을 가속화하였다. VAB는 이러한 마스크 기반 생성 모델링의 개념을 오디오-비주얼 도메인으로 확장하여 적용하였다.

## 🛠️ Methodology

### 1. 잠재 공간으로의 변환 (Latent Space Transformation)

VAB는 계산 효율성과 학습 수렴 속도를 높이기 위해 입력을 다음과 같이 처리한다.

- **Visual Latents**: Frozen 상태의 사전 학습된 **CLIP** 이미지 인코더를 사용하여 1fps 속도로 프레임당 특징 벡터를 추출한다. 결과적으로 10초 비디오는 $V \in \mathbb{R}^{10 \times d}$의 형태를 가진다.
- **Audio Latents**: 사전 학습된 뉴럴 오디오 코덱인 **DAC** 또는 **Encodec**을 사용하여 오디오 파형을 이산 토큰(Discrete Tokens)으로 변환한다. 10초 오디오는 $K$개의 코드북을 가진 $A \in \mathbb{N}^{K \times 500}$ 형태의 토큰으로 압축된다.

### 2. 조건부 마스크 오디오 토큰 예측 (Conditional Masked Audio Token Prediction)

VAB의 사전 학습 핵심 작업은 비주얼 특징을 조건으로 마스킹된 오디오 토큰을 예측하는 것이다.

- **Masking 전략**: 평균 $0.55$ ($\text{std}=0.25$)의 절단 가우시안 분포를 사용하여 오디오 토큰을 무작위로 마스킹한다.
- **모델 구조 (Multiway Transformer Encoder)**:
  - **Modal-specific Experts**: 초기 $N_1$개 층은 오디오와 비전 각각을 위한 전용 Feed-forward 네트워크를 사용하여 각 모달리티의 특성을 학습한다.
  - **Shared Layers**: 이후 $N_2$개 층은 공유 네트워크를 통해 두 모달리티 간의 상호작용을 학습한다.
- **학습 목표**: 마스킹된 오디오 토큰 $a \in A_m$에 대해 다음의 음의 로그 가능도(Negative Log-Likelihood)를 최소화한다.
$$ \ell_{vab} = -\sum_{\forall a \in A_m} \log p(a|A_u, V, \theta) $$
여기서 $A_u$는 마스킹되지 않은 오디오 토큰, $V$는 비주얼 특징, $\theta$는 모델 파라미터이다.

### 3. 제로샷 비디오-투-오디오 생성 (Zero-Shot Generation)

사전 학습 후, 모든 오디오 토큰을 `[MASK]`로 초기화한 뒤 반복적 디코딩을 수행한다.

- **Confidence Score 계산**: 예측된 토큰 $\hat{a}_t$에 대해 온도 조절 Gumbel 노이즈를 추가하여 신뢰도 점수를 계산한다.
$$ z(\hat{a}_t) = \log(p(\hat{a}_t)) + \alpha \cdot g_t $$
- **반복적 재마스킹**: 신뢰도 점수가 가장 낮은 $k$개의 토큰을 선택하여 다음 반복 단계에서 다시 마스킹하고 예측한다. 이때 $\gamma$ (cosine schedule)를 통해 단계별로 교체되는 토큰 수를 조절한다.
- **Coarse-to-Fine 모델**: DAC 사용 시, 초기 4개 층의 거친(Coarse) 토큰을 먼저 생성하고, 이를 조건으로 나머지 세부(Fine) 토큰을 생성하는 추가 모델을 사용하여 품질을 높인다.

### 4. 다운스트림 작업으로의 적응 (Adaptation)

- **리트리벌 (Retrieval)**: 첫 $N_1$개 층을 대조 손실(Contrastive Loss) $\ell_c$를 사용하여 미세 조정(Fine-tuning)한다.
$$ \ell_c = -\frac{1}{N} \sum_{i=1}^N \log \frac{\exp(s_{i,i}/\tau)}{\sum_{k \neq i} \exp(s_{i,k}/\tau) + \exp(s_{i,i}/\tau)} $$
- **분류 (Classification)**: $N_1$개 층의 출력 특징을 평균 풀링(Average-pooling)한 뒤 선형 분류기(Linear Classifier)를 추가하여 학습한다.

## 📊 Results

### 1. 비디오-투-오디오 생성 성능

VGGSound 테스트 세트에서 FAD(Fréchet Audio Distance)와 KLD(Kullback–Leibler Distance) 지표를 사용하여 평가하였다.

- **정량적 결과**: VAB-Encodec과 VAB-DAC 모두 기존 베이스라인(SpecVQGAN, IM2WAV 등) 대비 경쟁력 있는 품질을 보였으며, 특히 **추론 속도**에서 압도적인 우위를 점했다. VAB-Encodec의 경우 기존 자기회귀 방식보다 약 **17배 빠른 생성 속도**를 기록하였다.
- **정성적 결과**: MOS(Mean Opinion Score) 평가에서 VAB-Encodec이 전체적인 품질과 비주얼 관련성 측면에서 가장 높은 선호도를 보였다.

### 2. 오디오-비주얼 리트리벌 성능

AudioSet, VGGSound, MSR-VTT 데이터셋에서 Recall@1, 5, 10 지표를 측정하였다.

- VAB 모델은 CAV-MAE, ImageBind, LanguageBind 등 기존 모델보다 훨씬 높은 성능을 보였으며, 특히 AudioSet과 VGGSound에서는 **약 2배 이상의 성능 향상**을 달성하였다.

### 3. 오디오-비주얼 이벤트 분류 성능

AudioSet-20K, AudioSet-2M, VGGSound에서 분류 정확도를 평가하였다.

- 비주얼 전용(V) 분류 작업에서 매우 높은 성능을 보였는데, 이는 강력한 사전 학습 모델인 CLIP 임베딩을 사용한 효과로 분석된다.
- 오디오 전용(A) 분류에서는 일부 손실이 발생하였는데, 이는 오디오 토큰화 과정에서의 손실 압축(Lossy Compression) 때문으로 보인다.

## 🧠 Insights & Discussion

### 1. 강점 및 분석

- **효율적인 통합**: 잠재 공간에서의 마스크 예측 작업을 통해 생성과 이해라는 두 마리 토끼를 잡았으며, 특히 병렬 디코딩을 통한 생성 속도 개선이 매우 고무적이다.
- **전문가 층(Expert Layers)의 중요성**: 분석 결과, 초기 층에 모달리티별 전용 전문가 네트워크($N_1$ layers)를 배치하는 것이 분류 작업 성능 유지에 필수적임을 확인하였다.

### 2. 한계점 및 가정

- **시간적 해상도 부족**: 1fps의 낮은 해상도로 비주얼 특징을 추출했기에, 오디오-비주얼 간의 완벽한 시간적 동기화(Perfect Synchronization)를 달성하는 데 한계가 있다.
- **공간 정보 소실**: 글로벌 CLIP 임베딩을 사용함으로써 픽셀 레벨의 공간 정보가 사라졌으며, 이로 인해 오디오-비주얼 로컬라이제이션(Localization)과 같은 작업에는 적용하기 어렵다.
- **오디오 압축 손실**: Neural Codec의 손실 압축 특성상, 비주얼 가이드가 없는 순수 오디오 분류 작업에서는 성능 저하가 발생한다.

### 3. 비판적 해석

논문은 사전 학습 작업의 순서(Masked Prediction $\rightarrow$ Contrastive)가 더 효율적임을 입증하며 학습 전략의 정당성을 부여하였다. 다만, 생성된 오디오의 '맥락적 적절성'은 기술적 지표(FAD, KLD)만으로는 완전히 평가될 수 없으므로, 향후 문화적/음향적 전문가의 정성적 평가가 더 보완되어야 할 것으로 보인다.

## 📌 TL;DR

본 논문은 오디오-비주얼 표현 학습과 생성을 통합한 **VAB(Vision to Audio and Beyond)** 프레임워크를 제안한다. CLIP과 Neural Audio Codec의 잠재 공간에서 **비주얼 조건부 마스크 오디오 토큰 예측**을 수행함으로써, 단일 모델로 고품질 오디오 생성과 강력한 AV 이해 능력을 동시에 확보하였다. 특히 **병렬 디코딩**을 통해 생성 속도를 기존 대비 17배 높였으며, 리트리벌 및 분류 작업에서도 SOTA 수준의 성능을 달성하였다. 이 연구는 향후 고효율 오디오-비주얼 파운데이션 모델 설계 및 콘텐츠 생성 도구 개발에 중요한 기여를 할 것으로 기대된다.
