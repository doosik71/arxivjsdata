# MambaMIM: Pre-training Mamba with State Space Token Interpolation and its Application to Medical Image Segmentation

Fenghe Tang, Bingkun Nian, Yingtai Li, Zihang Jiang, Jie Yang, Wei Liu, S Kevin Zhou (2025)

## 🧩 Problem to Solve

본 논문은 3D 의료 영상 분석에서 긴 시퀀스 모델링(long-sequence modeling)의 효율성을 제공하는 Mamba 아키텍처를 위한 효과적인 자기지도학습(Self-Supervised Learning, SSL) 방법론의 부재 문제를 해결하고자 한다.

구체적으로, 기존의 Masked Image Modeling(MIM) 방식들은 Masked 영역을 단순한 학습 가능 토큰(learnable tokens)으로 대체하여 복원하는 방식을 취한다. 그러나 Mamba의 핵심인 상태 공간 모델(State Space Model, SSM)은 인과적 특성(causal properties)과 입력 의존적 선택적 스캔(selective scan) 속성을 가지므로, 무작위로 초기화된 학습 가능 토큰을 삽입할 경우 상태 공간 내의 구조적 시퀀스 관계가 파괴되어 Mamba의 잠재력을 충분히 활용하지 못하게 된다.

또한, CNN과 Mamba가 결합된 하이브리드 아키텍처의 경우, 서로 다른 구성 요소 간의 마스킹 전략이 상이하여 마스킹 위치의 일관성(masking consistency)을 유지하기 어렵고, 이는 픽셀 강도 분포의 변화(distributional shifts)를 유발하여 표현 학습 능력을 저하시키는 문제가 있다. 따라서 본 연구의 목표는 Mamba의 인과적 특성을 보존하는 새로운 토큰 생성 전략과 하이브리드 구조에서도 일관성을 유지하는 마스킹 전략을 통해 고성능의 의료 영상 세그멘테이션 모델을 사전 학습시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Mamba 전용의 생성적 사전 학습 프레임워크인 **MambaMIM**을 제안한 것이다. 주요 설계 아이디어는 다음과 같다.

1. **TOKI (TOKen-Interpolation) 전략**: 마스킹된 영역을 단순한 고정 토큰으로 채우는 대신, 마스킹되지 않은 주변 상태 공간 시퀀스 간의 관계를 수학적으로 보간(interpolation)하여 토큰을 생성한다. 이를 통해 Mamba의 인과적 흐름을 유지하며 장거리 의존성(long-range dependency) 표현 능력을 극대화한다.
2. **Bottom-up 3D 하이브리드 마스킹 전략**: 하이브리드 아키텍처에서 마스킹 일관성을 유지하기 위해 가장 하위 단계(bottom stage)에서 마스크를 생성하고 이를 상위 단계로 매핑하는 방식을 도입하여, CNN과 Mamba 단계 전반에서 동일한 마스킹 위치를 보장한다.
3. **범용적 적용 가능성**: 제안된 방법론은 단일 Mamba 아키텍처뿐만 아니라 CNN-Mamba 하이브리드 구조 모두에 적용 가능하며, 대규모 3D CT 데이터셋을 통한 사전 학습의 유효성을 입증하였다.

## 📎 Related Works

**1. Masked Image Modeling (MIM)**
MAE(Masked Autoencoders)나 SimMIM과 같은 MIM 방식은 입력의 일부를 제거하고 이를 복원하는 과정을 통해 국소적 패턴(local attention patterns)을 학습하며, 세그멘테이션과 같은 다운스트림 작업으로의 전이 능력이 뛰어남이 증명되었다. 하지만 이러한 방법들은 Mamba의 상태 공간 내 구조적 시퀀스 관계를 무시한다는 한계가 있다.

**2. Medical Image SSL**
의료 영상 분야의 SSL은 주로 대조 학습(Contrastive Learning)에 의존해 왔으나, 이는 주로 고수준의 의미론적 표현(high-level semantic representations)을 학습하므로 데이터 분포가 다른 다운스트림 작업에서 편향이 발생할 수 있다. 반면, MIM 기반의 생성적 방법은 세밀한 국소 구조를 학습하므로 세그멘테이션 작업에 더 유리하다.

**3. Mamba in Medical Imaging**
Mamba는 Transformer의 이차 복잡도(quadratic complexity) 문제를 해결하여 고해상도 3D 의료 영상 처리에 적합한 대안으로 부상했다. 그러나 Mamba 기반의 모델을 위한 효과적인 사전 학습 방법론은 아직 충분히 탐구되지 않은 상태였다.

## 🛠️ Methodology

### 전체 시스템 구조

MambaMIM은 크게 **계층적 하이브리드 인코딩(Hierarchical Hybrid Encoding)**과 **계층적 하이브리드 디코딩(Hierarchical Hybrid Decoding)**, 그리고 **TOKI**라는 세 가지 핵심 요소로 구성된다.

### 1. 계층적 하이브리드 인코딩 (Hierarchical Hybrid Encoding)

하이브리드 모델(예: HyMamba)에서 CNN 단계와 Mamba 단계 간의 마스킹 일관성을 유지하기 위해 **Bottom-up masking** 전략을 사용한다.

- 최종 CNN 단계($n$-th stage)에서 랜덤하게 마스크 $M_n$을 초기화한다.
- 이 마스크를 상위 단계($1 \sim n-1$)로 업샘플링하여 $\{M_1, M_2, \dots, M_{n-1}\}$ 세트를 생성한다.
- CNN 단계에서는 3D Sparse Operator를 사용하여 마스킹된 위치의 계산을 건너뛰며 특징 맵 $S_i$를 추출한다.
- Mamba 단계에서는 $S_n$을 시퀀스로 직렬화하여 마스킹되지 않은 토큰 $x$만을 입력으로 받아 상태 공간 시퀀스 $y$를 생성한다.

### 2. 계층적 하이브리드 디코딩 (Hierarchical Hybrid Decoding)

인코더에서 추출된 계층적 특징 $\{y, S_{n-1}, \dots, S_1\}$을 사용하여 원래 영상을 복원한다.

- **CNN stage**: 학습 가능한 토큰 임베딩 $t_i$를 빈 공간에 채워 넣고, 업샘플링 블록과 스킵 연결(skip-connection)을 통해 조밀한 특징을 복원한다.
- **Mamba stage**: 단순 토큰 삽입 대신 **TOKI**를 적용하여 $y$를 $\hat{y}$로 복원함으로써 1D 선택적 구조 상태 공간 시퀀스의 연속성을 보존한다.

### 3. Selective Structured State Space Sequence Token-Interpolation (TOKI)

TOKI는 Mamba의 SSM 수식에 근거하여 마스킹된 토큰을 생성한다.
기존의 SSM은 다음과 같은 선형 상미분 방정식(ODE)으로 표현된다:
$$\begin{aligned} h'(t) &= Ah(t) + Bx(t) \\ y(t) &= Ch(t) \end{aligned}$$

이산화(Discretization) 과정을 거치면 출력 $y_j$는 다음과 같이 표현된다:
$$y_j = \sum_{i=1}^{j} C \cdot \bar{A}^{j-i} \cdot \bar{B}x_i$$

TOKI는 마스킹된 영역 앞뒤의 가시적 토큰 $\hat{y}_i$와 $\hat{y}_{i+1}$ 사이를 보간한다. 구체적으로, 마스킹된 개수가 $Q$개일 때, 가상의 시퀀스 $s'_n$을 다음과 같이 정의한다:
$$s'_n = \frac{Q+2-n}{Q+2} \cdot \hat{y}_i + \frac{n}{Q+2} \cdot \hat{y}_{i+1}$$

최종적으로 보간된 토큰 $z_j$는 다음과 같이 계산된다:
$$z_j = \sum_{n=0}^{j} \text{Pow}(\bar{A}', j-n) \Delta \bar{A}'^{-1} \cdot V$$
여기서 $V$는 $\hat{y}_i$와 $\hat{y}_{i+1}$의 선형 결합이며, $\bar{A}'$는 학습 가능한 파라미터이다. 이 과정을 통해 마스킹된 토큰들이 Mamba의 인과적 흐름(causal flow)을 따르도록 강제한다.

## 📊 Results

### 실험 설정

- **데이터셋**: 사전 학습을 위해 13개 공공 데이터셋에서 수집한 6,814개의 3D CT 스캔을 사용하였다.
- **다운스트림 작업**: BTCV, MSD, AMOS, CT-ORG, KiTS 23, BraTS 21 등 8개의 벤치마크에서 세그멘테이션 성능을 평가하였다.
- **평가 지표**: Dice Similarity Coefficient (DSC)를 사용하였다.
- **비교 대상**: MAE, SimMIM, SparK (MIM 계열), SimSiam, MoCov2, SwAV (대조 학습 계열), 그리고 UNETR, Swin UNETR, MedNeXt 등 최신 아키텍처.

### 주요 결과

1. **아키텍처 효율성**: HyMamba(MedNeXt + Vision Mamba)는 사전 학습 없이도 Vanilla Mamba보다 훨씬 높은 성능(BTCV 기준 77.56% vs 70.94%)을 보였다.
2. **MambaMIM의 효과**: HyMamba에 MambaMIM 사전 학습을 적용했을 때 DSC가 80.16%로 상승하여, 사전 학습이 없는 경우보다 2.6%p 향상되었다. 특히 췌장(Pancreas)과 같이 작은 장기의 세그멘테이션 성능이 크게 개선되었다.
3. **SSL 방법론 비교**: 동일한 HyMamba 백본에서 MambaMIM은 MoCov2, SimSiam, SwAV 등 대조 학습 기반 방법들보다 일관되게 높은 성능을 기록하였다.
4. **일반화 능력**: 사전 학습에 사용되지 않은 unseen 데이터셋(CT-ORG, KiTS 23)과 다른 모달리티인 MRI(BraTS 21)에서도 SOTA 성능을 달성하여, 모달리티에 독립적인 국소 표현(local representation)을 효과적으로 학습했음을 입증하였다.

## 🧠 Insights & Discussion

**강점 및 분석**

- **인과적 보간의 중요성**: Ablation Study 결과, 학습 가능한 토큰을 사용했을 때보다 TOKI를 사용했을 때 성능이 크게 향상(79.38% $\rightarrow$ 80.16%)되었다. 이는 Mamba의 상태 공간 특성을 유지하는 것이 단순한 토큰 삽입보다 훨씬 중요함을 시사한다.
- **MIM vs Contrastive**: 본 논문은 세그멘테이션과 같은 조밀한 예측(dense prediction) 작업에서는 고수준 의미론을 학습하는 대조 학습보다, 국소적 구조를 복원하는 MIM 방식이 더 효과적이라는 점을 확인하였다.
- **마스킹 비율**: 마스킹 비율이 75%일 때 최적의 성능을 보였으며, 이는 높은 마스킹 비율이 모델로 하여금 더 강력한 잠재 표현(latent representation)을 학습하도록 유도하기 때문으로 분석된다.

**한계 및 논의**

- 본 논문은 주로 3D CT와 MRI 영상에 집중하고 있으며, 다른 의료 영상 모달리티(예: 초음파, X-ray)에서의 성능은 명시적으로 다루지 않았다.
- TOKI의 수학적 유도 과정에서 일부 파라미터를 단순화($B'=C'=I$)하였는데, 이 가정이 모든 데이터 분포에서 최적인지에 대한 추가 연구가 필요할 수 있다.

## 📌 TL;DR

본 논문은 Mamba 아키텍처의 인과적 특성을 보존하는 새로운 사전 학습 프레임워크인 **MambaMIM**을 제안하였다. 핵심 기여는 마스킹된 토큰을 상태 공간 맥락에서 보간하는 **TOKI** 전략과 하이브리드 구조의 일관성을 보장하는 **Bottom-up 마스킹** 전략이다. 실험 결과, MambaMIM은 3D 의료 영상 세그멘테이션에서 기존의 대조 학습 및 MIM 기반 방법론들을 뛰어넘는 SOTA 성능을 달성하였으며, 특히 다양한 모달리티와 unseen 데이터셋에 대해 강력한 일반화 능력을 보여주어 향후 Mamba 기반 의료 AI 연구에 중요한 기반을 제공할 것으로 기대된다.
