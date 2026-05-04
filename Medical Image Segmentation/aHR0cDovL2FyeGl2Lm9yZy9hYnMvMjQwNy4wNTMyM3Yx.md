# Enhancing Label-efficient Medical Image Segmentation with Text-guided Diffusion Models

Chun-Mei Feng (2024)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 의료 영상 분할(Medical Image Segmentation) 분야에서 고품질의 픽셀 단위 주석(pixel-level annotations)을 확보하는 데 드는 막대한 비용과 노동력이다. 최근 Denoising Diffusion Probabilistic Models (DPM)가 의료 영상 생성뿐만 아니라 세그멘테이션과 같은 다운스트림 작업을 위한 강력한 표현 학습기(representation learner)로 활용될 수 있음이 밝혀졌으나, 이러한 잠재적 능력을 끌어내기 위해서는 여전히 많은 양의 픽셀 수준 레이블이 필요하다는 한계가 있다.

반면, 의료 영상과 함께 생성되는 진단 텍스트 정보는 상대적으로 획득 비용이 저렴하며, 영상 데이터에 상호 보완적인 정보를 제공한다. 따라서 본 논문의 목표는 비용이 많이 드는 픽셀 수준 주석에 대한 의존도를 낮추기 위해, 저비용의 의료 텍스트 주석을 활용하여 Diffusion 모델의 시각적-의미적 표현(visual-semantic representation)을 강화하는 **TextDiff** 모델을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 설계 아이디어는 사전 학습된 Diffusion 모델의 역과정(reverse process)에서 발생하는 중간 활성화 값(intermediate activations)과 의료 진단 텍스트 간의 정렬(alignment)을 통해 전문가 지식을 학습시키는 것이다.

핵심 기여 사항은 다음과 같다:

1. **TextDiff 프레임워크 제안**: 의료 텍스트 주석을 통해 Diffusion 모델의 세그멘테이션 성능을 향상시켜, 적은 수의 픽셀 수준 레이블만으로도 효과적인 학습이 가능한 Label-efficient한 구조를 설계하였다.
2. **시각-언어 연결성 확립**: DPM의 마르코프 단계(Markov step)에서 추출한 중간 활성화 값과 텍스트 진단 정보 사이의 강한 연결 고리를 구축하여 시각적-의미적 표현력을 높였다.
3. **효율적인 학습 전략**: 텍스트 인코더와 이미지 인코더의 가중치를 고정(freeze)하고, Cross-attention 메커니즘과 픽셀 분류기(pixel classifier)만 학습시킴으로써 파라미터 효율성을 극대화하였다.

## 📎 Related Works

논문에서는 기존의 의료 영상 분할 접근 방식과 그 한계를 다음과 같이 설명한다:

- **전통적/딥러닝 기반 방법**: UNet, TransUNet, SwinUNet 등은 우수한 성능을 보이지만, 대량의 정밀한 픽셀 레이블이 필수적이다.
- **약지도/준지도 학습**: Pseudo-label의 신뢰도에 크게 의존하며, 신뢰도가 낮은 레이블이 많을 경우 세그멘테이션 정확도가 크게 떨어진다는 한계가 있다.
- **멀티모달 접근 방식**: GLoRIA는 이미지 하위 영역과 텍스트를 대조하여 표현을 학습하고, LViT는 Vision Transformer에 텍스트 주석을 결합하여 시각적 표현을 보완한다. 하지만 LViT는 모델을 처음부터 학습시켜야 하며, GLoRIA는 효과적인 시각적 특징 추출에 한계가 있다.

본 논문의 **TextDiff**는 대규모 자연어 이미지로 사전 학습된 Diffusion 모델의 강력한 표현력을 활용하며, 전체 모델을 학습시키는 대신 특정 모듈만 튜닝함으로써 기존 멀티모달 방법론보다 효율적이고 강력한 성능을 제공한다는 점에서 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조

TextDiff는 시각 정보와 텍스트 정보를 각각 처리하는 **Dual-branch 구조**를 가진다. 이미지 브랜치는 사전 학습된 Diffusion 모델(UNet 아키텍처)을 사용하고, 텍스트 브랜치는 Clinical BioBERT를 사용한다. 두 브랜치에서 추출된 특징은 Cross-modal Attention 모듈을 통해 융합되며, 최종적으로 픽셀 분류기를 통해 세그멘테이션 맵을 생성한다.

### 주요 구성 요소 및 절차

#### 1. Image Encoding (Diffusion-based)

이미지 $x$에 대해 Diffusion 모델의 역과정을 통해 특징을 추출한다.

- **순방향 과정(Forward Process)**: 이미지 $x_0$에 가우시안 노이즈를 점진적으로 추가하여 $x_T$를 만드는 과정이다.
$$q(x_t | x_{t-1}) := \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)$$
- **역방향 과정(Reverse Process)**: 노이즈가 섞인 $x_t$로부터 원래 이미지 $x_0$를 복원하는 과정으로, 노이즈 예측기 $\epsilon_\theta(x_t, t)$ (UNet)가 핵심 역할을 한다.
$$p_\theta(x_{t-1} | x_t) := \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

TextDiff는 $\epsilon_\theta(x_t, t)$의 UNet 디코더 블록에서 발생하는 중간 활성화 값을 추출한다. 특히 디코더의 특정 블록 $B$와 특정 타임스텝 $t$에서의 특징을 추출한 뒤, Bilinear interpolation을 통해 원본 이미지 크기($H \times W$)로 업샘플링하여 픽셀 수준의 표현으로 사용한다.

#### 2. Text Encoding

의료 진단 텍스트 $t$는 MIMIC III 데이터셋으로 사전 학습된 **Clinical BioBERT**를 통해 임베딩 벡터 $\hat{t} = E_{Te}(t)$로 변환된다. 이 텍스트 특징은 이미지의 시각적 특징을 보완하는 전문가 지식의 역할을 한다.

#### 3. Cross-modal Attention for Knowledge Alignment

이미지에서 추출된 픽셀 수준 시각 특징 $h_{z,t}$와 텍스트 특징 $\hat{t}$를 정렬하기 위해 Scaled Dot-Product Attention을 사용한다.
$$H_{z,t} = \text{Softmax}\left(\frac{h_{z,t}W_q(\hat{t}W_k)^T}{\sqrt{d}}\right)\hat{t}W_v$$
여기서 $W_q, W_k, W_v$는 학습 가능한 파라미터 행렬이다. 다양한 스케일의 $H_{z,t}$를 추출하여 결합(concatenate)함으로써 텍스트 정보가 강화된 시각적 표현 $H$를 생성한다.

#### 4. 학습 절차 및 손실 함수

- **학습 대상**: 두 인코더(Diffusion, BioBERT)는 고정(freeze)하며, **Cross-modal Attention 모듈**과 **픽셀 분류기**만 학습시킨다.
- **손실 함수**: 세그멘테이션 성능 평가를 위해 Dice Loss ($\mathcal{L}_{Dice}$)와 Cross-Entropy Loss ($\mathcal{L}_{CE}$)의 합을 최소화한다.

## 📊 Results

### 실험 설정

- **데이터셋**:
  - **MoNuSeg**: 병리 이미지 데이터셋. label-efficient 성능을 증명하기 위해 단 5장의 이미지로만 학습을 진행하였다.
  - **QaTa-COVID19**: COVID-19 흉부 엑스레이 데이터셋. 150장의 이미지를 무작위로 선택하여 학습하였다.
- **비교 대상 (Baselines)**:
  - 고전적 방법: UNet, TransUNet, SwinUNet.
  - 텍스트 기반 방법: GLoRIA, LViT.
- **지표**: Dice coefficient (%) 및 IoU (%).

### 주요 결과

정량적 결과(Table 1)에 따르면, TextDiff는 모든 데이터셋에서 가장 높은 Dice와 IoU를 기록하였다.

- **MoNuSeg**: GLoRIA 대비 Dice가 $66.38\% \to 78.67\%$, IoU가 $49.83\% \to 64.98\%$로 대폭 향상되었다.
- **QaTa-COVID19**: 타 모델 대비 우수한 성능을 보였으며, 특히 적은 수의 샘플만으로도 높은 성능을 달성하였다.
- **효율성**: TextDiff의 학습 가능 파라미터 수는 $9.68\text{M}$으로, LViT($29.72\text{M}$)나 TransUNet($93.19\text{M}$)보다 훨씬 적어 모델 효율성이 매우 높다.

정성적 결과(Fig. 2)에서도 TextDiff의 결과물이 Ground-truth와 가장 유사하며, 다른 모델들에 비해 오차가 적음을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 분석

1. **텍스트 정보의 유효성**: Ablation study를 통해 텍스트 주석이 없을 때($\zeta_1$)보다 있을 때 성능이 크게 향상됨을 확인하였다. 이는 저비용의 텍스트 정보가 시각적 특징의 부족함을 효과적으로 보완할 수 있음을 시사한다.
2. **Diffusion 단계별 특성**: 역과정(reverse process)의 후반 단계(later steps) 특징이 초반 단계보다 세그멘테이션에 필요한 의미적 정보(semantic information)를 더 잘 포착한다는 점을 발견하였다.
3. **정렬 메커니즘의 중요성**: 단순 결합보다 Cross-modal Attention을 통한 정렬을 수행했을 때($\zeta_2$ 대비 Ours) 성능이 더 높았으며, 이는 두 모달리티 간의 깊은 정렬이 필수적임을 의미한다.

### 한계 및 논의사항

- **데이터셋의 다양성**: 두 개의 공개 데이터셋에서 우수한 성능을 보였으나, 더 다양한 의료 도메인(예: MRI, 초음파 등)에서의 일반화 성능에 대한 추가 검증이 필요하다.
- **추론 비용**: Diffusion 모델의 역과정에서 여러 타임스텝의 활성화를 추출해야 하므로, 실시간 추론 시 계산 복잡도가 증가할 가능성이 있다.

## 📌 TL;DR

본 논문은 사전 학습된 Diffusion 모델의 중간 활성화 값과 의료 진단 텍스트를 결합한 **TextDiff**를 제안하여, 픽셀 수준의 레이블이 부족한 환경에서도 고성능의 의료 영상 분할을 가능하게 하였다. 특히 모델의 백본을 고정하고 Cross-attention과 분류기만 학습시키는 전략을 통해 파라미터 효율성을 극대화하였으며, 적은 양의 데이터만으로 기존 SOTA 멀티모달 모델들을 상회하는 성능을 달성하였다. 이 연구는 향후 레이블 확보가 어려운 희귀 질환이나 특수 의료 영상 분할 연구에 중요한 전기를 마련할 것으로 보인다.
