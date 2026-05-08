# Conditional Latent Diffusion Models for Zero-Shot Instance Segmentation

Maximilian Ulmer, Wout Boerdijk, Rudolph Triebel, Maximilian Durner (2025)

## 🧩 Problem to Solve

본 논문은 **Zero-Shot Instance Segmentation (ZSI)** 문제를 해결하고자 한다. Instance Segmentation은 픽셀 수준에서 객체 인스턴스를 탐지하고 분리하는 핵심적인 컴퓨터 비전 작업이다. 기존의 접근 방식은 크게 두 가지 방향으로 나뉜다. 하나는 특정 타겟 객체에 대해 학습된 2D Segmentation 모델로, 성능은 높으나 새로운 객체를 추가할 때마다 재학습이 필요하여 비용이 매우 많이 든다. 다른 하나는 Class-agnostic 방법으로, 사전 지식 없이 임의의 인스턴스를 분리할 수 있지만, 시맨틱 정보가 부족하여 각 예측 결과에 특정 레이블을 할당하는 능력이 없다.

ZSI의 목표는 추론 시에만 조건부 정보(Conditioning information)를 제공하여, 학습 단계에서 보지 못한 임의의 객체에 대해서도 클래스 특이적인 인스턴스 예측을 수행하는 것이다. 기존의 ZSI 방법들은 주로 템플릿 이미지의 특징과 실제 이미지의 특징을 비교하는 2단계 방식(Region proposal 추출 $\rightarrow$ 템플릿 매칭)을 사용하며, 이는 Sim-to-real gap 문제와 다양한 스케일 변화에 취약하다는 한계가 있다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 **Object-Conditioned Diffusion Transformer (OC-DiT)**라는 새로운 확산 모델(Diffusion Model)을 제안하여, 객체 가설 생성(Object hypothesis generation)과 특징 매칭(Feature matching)을 하나의 엔드-투-엔드 generative process로 통합하는 것이다.

중심적인 설계 직관은 확산 모델의 Latent space 내에서 시각적 객체 기술자(Visual object descriptors)와 국소적 이미지 힌트(Localized image cues)를 조건으로 주어, 생성 과정 자체가 객체 인스턴스를 효과적으로 분리(Disentangle)하도록 만드는 것이다. 이를 위해 고해상도 이미지에서 초기 제안을 생성하는 **Coarse model**과, 이를 병렬적으로 정밀하게 다듬는 **Refinement model**의 2단계 구조를 제안한다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구들을 언급한다.

- **Zero-Shot Instance Segmentation**: 단어 벡터를 통한 시각-시맨틱 매핑이나 레이블이 없는 비디오를 활용한 연구들이 있었으며, 최근에는 CAD 모델 기반의 템플릿 매칭 방식이 주를 이룬다. 특히 SAM(Segment Anything Model)이나 FastSAM과 같은 Foundation Model을 사용하여 영역 제안을 생성하고 이를 템플릿과 매칭하는 방식(예: CNOS, SAM6D)이 사용되고 있다.
- **Diffusion Models for Segmentation**: 기존 확산 모델은 주로 이미지 합성(Synthesis)에 사용되었으나, 최근에는 확률적 샘플링 특성을 이용해 픽셀 단위의 불확실성 맵(Uncertainty maps)을 생성하거나 Few-shot 및 의료 이미지 세그멘테이션에 적용되는 추세이다.

OC-DiT는 기존의 '제안 후 매칭' 방식과 달리, 확산 과정 내에서 직접적으로 특징 매칭과 마스크 생성을 동시에 수행한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조

OC-DiT는 크게 세 가지 구성 요소로 이루어진다.

1. **Pre-trained ViT**: 입력 이미지와 템플릿 이미지에서 특징을 추출하는 고정된 백본이다.
2. **Pre-trained VAE**: 이진 세그멘테이션 마스크를 저차원의 Latent space로 압축하고 복원하는 역할을 한다.
3. **OC-DiT Transformer**: VAE의 Latent space에서 노이즈를 제거하며 조건부 마스크를 생성하는 핵심 모델이다.

### 상세 방법론

#### 1. Latent Diffusion of Bernoulli Distributions

이진 마스크(Bernoulli distribution)는 가우시안 분포를 가정하는 일반적인 확산 모델의 요구사항과 맞지 않는다. 이를 해결하기 위해 $\beta$-VAE를 사용하여 마스크 $b \in \{0,1\}^{H \times W}$를 저차원 잠재 코드 $x \in \mathbb{R}^{h \times w \times d}$로 압축한다. 모든 확산 과정은 이 Latent space에서 수행되며, 추론 마지막 단계에서만 VAE Decoder를 통해 다시 마스크로 복원한다.

#### 2. OC-DiT Transformer 아키텍처

모델은 Transformer 기반으로 설계되어 입력 해상도, 객체 수($N_O$), 템플릿 수($N_T$)에 대해 유연하게 대응한다.

- **Query Embeddings**: 각 타겟 객체에 대해 하나의 쿼리를 생성한다. 쿼리는 `Latent code`와 `Image features`의 결합으로 구성되며, 학습 시에는 노이즈가 섞인 Latent를, 추론 시에는 이전 단계의 예측 Latent를 사용한다.
- **Conditioning Object Templates**: 각 객체당 $N_T$개의 렌더링된 템플릿 뷰를 사용한다. ViT를 통해 특징을 추출하고, 2D 패치 위치, 객체 위치, 템플릿 위치를 인코딩한 Positional encoding을 추가한다.
- **Noise Embeddings**: 현재 노이즈 레벨 $\sigma$를 Fourier features를 통해 임베딩하여 모델에 전달한다.
- **OC-DiT Block**: 다음과 같은 순서로 처리한다.
  - **Cross-Attention**: 각 쿼리가 자신이 할당된 객체의 템플릿 토큰에만 주의를 기울이도록 제한한다.
  - **Self-Attention**: 쿼리들을 재배치하여 모든 쿼리 토큰이 서로를 참조하게 함으로써, 각 쿼리가 하나의 객체에 전념하도록 학습시킨다.
  - **adaLN (Adaptive Layer Norm)**: 노이즈 레벨에 따라 Scale($\gamma$), Shift($\beta$), Gate($\alpha$)를 조절하여 토큰을 변조한다.

#### 3. 학습 및 추론 절차

- **손실 함수**: 모델은 노이즈가 섞인 잠재 코드 $y = x + n$에서 원본 $x$를 복구하도록 학습된다. 손실 함수는 $L_2$ 에러를 최소화하는 방향으로 설정되며, 노이즈 레벨에 따른 불확실성 $u(\sigma)$를 이용해 가중치를 조절하는 Loss weighting 방식을 사용한다.
$$L(D_\theta, u) = \mathbb{E}_\sigma \left[ \frac{\lambda(\sigma)}{e^{u(\sigma)}} L(D_\theta; \sigma) + u(\sigma) \right]$$
- **추론 (Inference)**: 순수 노이즈 $x_N \sim \mathcal{N}(0, \sigma_{max}^2 I)$에서 시작하여 Probability flow ODE를 따라 $\sigma=0$까지 반복적으로 디노이징을 수행한다.
- **Ensembling**: 확산 과정의 확률적 특성을 활용하여 여러 번 추론한 뒤 결과 마스크 $\tilde{b}$를 평균 내어 최종 결과를 얻는다.

#### 4. Coarse-to-Refine 파이프라인

- **Coarse Model**: 전체 이미지에서 대략적인 객체 제안(Proposals)을 생성한다.
- **Refine Model**: Coarse 모델이 제안한 ROI(Region of Interest) 영역만을 잘라내어, 단일 객체 조건 하에 고해상도로 정밀하게 마스크를 생성한다.

## 📊 Results

### 실험 설정

- **데이터셋**: BOP Challenge의 YCB-V, TUDL, HB, LMO 벤치마크를 사용하였다. 학습은 Google Scanned Objects(GSO)와 Objaverse를 이용해 생성한 대규모 합성 데이터셋(약 100만 샘플)으로만 수행되었다.
- **지표**: Intersection over Union (IoU) 50%~95% 범위의 Average Precision (AP)을 측정하였다.

### 주요 결과

- **정량적 성과**: Table 1에 따르면, OC-DiT는 YCB-V, TUDL, HB 데이터셋에서 기존의 SOTA 방법들(CNOS, SAM6D, NIDS 등)보다 우수한 성능을 보였다. 특히 YCB-V에서는 Coarse 모델만으로도 SOTA에 도달하였으며, Refine 모델을 적용했을 때 성능이 더욱 향상되었다.
- **Refinement 효과**: Table 3에서 Refine 모델은 TUDL 데이터셋에서 AP를 크게 높였으며, 특히 정답 레이블 없이도(w/o labels) 높은 판별력을 보여주어 False Positive를 효과적으로 제거함을 입증하였다.
- **학습 데이터 영향**: Objaverse 데이터셋을 추가했을 때 성능 향상 폭이 가장 컸으며, 모든 합성 데이터를 함께 사용했을 때 최적의 성능을 보였다 (Table 4).

## 🧠 Insights & Discussion

### 강점 및 분석

- **Zero-Shot 일반화**: 실제 데이터로의 추가 학습 없이 합성 데이터만으로 학습했음에도 불구하고 실제 환경의 unseen objects에 대해 매우 강력한 성능을 보인다. 이는 VAE를 통한 Latent space 설계와 Transformer의 유연한 조건부 처리 덕분으로 해석된다.
- **확장성 (Scalability)**: 추론 시 타겟 객체 수를 늘리기 위해 'Random Interval Training' 방식의 Positional encoding을 사용했을 때, 단순히 더 많은 객체로 학습시킨 경우보다 오히려 더 높은 AP를 기록하였다 (Table 2). 이는 모델이 객체 순서에 구애받지 않고 일반적인 객체 관계를 학습했음을 시사한다.

### 한계점 및 비판적 해석

- **추론 속도**: 확산 모델의 고유한 특성인 반복적 샘플링으로 인해 실시간 적용이 불가능할 정도로 느리다. 특히 Ensemble size와 $\sigma$ 단계가 증가할수록 연산 시간이 선형적으로 증가한다 (Table 5).
- **객체 크기 편향**: LMO 데이터셋에서 상대적으로 낮은 성능을 보였는데, 이는 학습 데이터의 객체 크기가 LMO의 작은 객체들에 비해 전반적으로 컸기 때문이라는 분석이다.
- **템플릿 의존성**: 템플릿 렌더링 시 객체 주변을 타이트하게 크롭하기 때문에, 크기만 다르고 형태가 같은 객체(예: YCB-V의 Clamp)를 구분하는 데 어려움을 겪는 failure mode가 발견되었다.

## 📌 TL;DR

본 논문은 **Object-Conditioned Diffusion Transformer (OC-DiT)**를 통해, 추가 학습 없이 새로운 객체를 세그멘테이션하는 **Zero-Shot Instance Segmentation**의 새로운 패러다임을 제시하였다. 템플릿 정보와 이미지 특징을 조건으로 하는 Latent Diffusion 과정을 통해 객체 분리를 수행하며, Coarse-to-Refine 구조와 대규모 합성 데이터셋을 통해 SOTA 성능을 달성하였다. 추론 속도가 느리다는 한계가 있으나, 확산 모델을 인스턴스 세그멘테이션에 성공적으로 적용함으로써 향후 생성 모델 기반의 정밀 인식 연구에 중요한 기반을 마련하였다.
