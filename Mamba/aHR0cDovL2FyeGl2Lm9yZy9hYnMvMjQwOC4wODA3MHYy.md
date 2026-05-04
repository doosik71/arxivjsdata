# MambaMIM: Pre-training Mamba with State Space Token Interpolation and its Application to Medical Image Segmentation

Fenghe Tang, Bingkun Nian, Yingtai Li, Zihang Jiang, Jie Yang, Wei Liu, S Kevin Zhou (2025)

## 🧩 Problem to Solve

본 논문은 3D 의료 영상 분석에서 긴 시퀀스 모델링 능력이 뛰어난 State Space Model(SSM)인 Mamba를 효율적으로 사전 학습(Pre-training)시키기 위한 방법론을 제안한다. 기존의 Masked Image Modeling(MIM) 방식들을 Mamba에 그대로 적용할 때 발생하는 두 가지 핵심적인 문제를 해결하고자 한다.

첫째, **상태 공간 모델의 인과적 특성(Causal properties) 무시** 문제이다. 기존 MAE(Masked Autoencoders)와 같은 방식은 마스킹된 영역을 단순히 학습 가능한(learnable) 토큰으로 채운다. 그러나 Mamba의 Selective Scan 속성은 입력 데이터에 의존적인 인과 관계를 가지므로, 무작위로 초기화된 학습 가능 토큰을 삽입할 경우 상태 공간 내의 구조적 시퀀스 관계가 파괴되어 모델이 효과적인 표현력을 학습하지 못하게 된다.

둘째, **서로 다른 아키텍처 간의 마스킹 일관성(Masking consistency)** 문제이다. 특히 CNN과 Mamba가 결합된 하이브리드 구조에서, CNN의 Sparse Convolution 마스킹 방식과 Mamba의 토큰 드롭 방식은 서로 다르다. 이러한 불일치는 인코딩 과정에서 픽셀 강도 분포의 시프트(distributional shift)를 유발하여 모델의 표현 학습 능력을 저하시킨다.

결과적으로 본 논문의 목표는 Mamba의 인과적 특성을 보존하면서 하이브리드 구조에서도 일관된 마스킹을 유지하는 Mamba 전용 MIM 사전 학습 프레임워크인 MambaMIM을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Mamba의 특성에 최적화된 새로운 마스킹 및 토큰 생성 전략을 제안한 것에 있다.

1.  **TOKI (TOKen-Interpolation) 전략**: 마스킹된 영역을 단순히 독립적인 토큰으로 채우는 대신, 마스킹되지 않은 인접 토큰들 사이의 상태 공간 관계를 수학적으로 보간(interpolation)하여 채우는 방식을 제안한다. 이를 통해 Mamba의 인과적 흐름(causal flow)을 유지하며 장거리 의존성 학습 능력을 극대화한다.
2.  **Bottom-up 3D Hybrid Masking**: 하이브리드 아키텍처의 최하위 단계(가장 깊은 단계)에서 마스크를 먼저 생성하고 이를 상위 단계로 맵핑하는 방식을 통해, CNN과 Mamba 단계 전반에 걸쳐 마스킹 위치의 일관성을 보장한다.
3.  **Mamba 기반 의료 영상 사전 학습의 가능성 증명**: 대규모 CT 데이터셋(6.8K 스캔)으로 사전 학습한 MambaMIM을 8개의 공공 의료 영상 분할 벤치마크에 적용하여, 기존 SOTA(State-of-the-art) 모델들을 뛰어넘는 성능을 입증하였다. 특히 MedNeXt와 Vision Mamba를 결합한 하이브리드 모델(HyMamba)에서 가장 뛰어난 성능을 보였다.

## 📎 Related Works

본 논문은 다음과 같은 관련 연구들을 분석하고 차별점을 제시한다.

1.  **Masked Image Modeling (MIM)**: BERT에서 영감을 받은 MAE, SimMIM, SparK 등이 대표적이다. 이들은 국소적 패턴 학습 능력이 뛰어나 분할(Segmentation)과 같은 다운스트림 태스크에서 전이 능력이 높다. 하지만 이들은 주로 ViT나 CNN에 최적화되어 있으며, Mamba의 상태 공간 시퀀스 관계를 고려하지 않는다.
2.  **의료 영상 자가 지도 학습 (Medical SSL)**: 데이터 부족 문제를 해결하기 위해 대조 학습(Contrastive Learning) 기반 방법(MoCoV2, SimSiam 등)과 생성적 방법(MIM)이 연구되어 왔다. 대조 학습은 주로 고수준의 시맨틱 표현을 학습하므로 픽셀 수준의 정밀함이 필요한 분할 작업에서는 MIM보다 효율이 떨어지는 경향이 있다.
3.  **Medical Mamba**: 최근 Mamba를 의료 영상에 적용한 연구(SegMamba 등)들이 등장하였으나, 대부분 지도 학습(Supervised Learning)에 의존하거나 Mamba 전용의 효과적인 사전 학습 전략이 부재한 상태였다.

## 🛠️ Methodology

### 1. 전체 시스템 구조
MambaMIM은 크게 **계층적 하이브리드 인코딩(Hierarchical Hybrid Encoding)**과 **계층적 하이브리드 디코딩(Hierarchical Hybrid Decoding)**으로 구성된다. 특히 CNN(MedNeXt)과 Mamba(Vision Mamba)를 직렬로 연결한 **HyMamba** 구조를 기본 백본으로 사용한다.

### 2. Hierarchical Hybrid Encoding (Bottom-up Masking)
CNN과 Mamba 간의 마스킹 일관성을 위해 다음과 같은 절차를 따른다.
- 최하위 CNN 단계($n$-th stage)에서 랜덤 마스크 $M_n$을 생성한다.
- 이를 업샘플링하여 상위 단계 $\{M_1, \dots, M_{n-1}\}$에 맵핑한다.
- CNN 단계에서는 3D Sparse Operator를 사용하여 마스크 위치의 계산을 생략한다:
  $$\text{SparseOp}(S_{i-1}, M_i) = \text{Operator}(S_{i-1}) \cdot M_i$$
- Mamba 단계에서는 마스킹되지 않은 토큰 $\mathbf{x}$만을 직렬화하여 시퀀스 $\mathbf{y}$를 생성한다.

### 3. Hierarchical Hybrid Decoding
디코딩 단계에서는 아키텍처별 특성에 맞는 마스크 채우기 전략을 사용한다.
- **CNN 단계**: 학습 가능한 토큰 임베딩 $t_i$를 빈 위치에 채우고, 업샘플링 블록($B^{up}$)과 스킵 연결 블록($B^{skip}$)을 통해 조밀한 특징 맵을 재구성한다.
- **Mamba 단계**: 단순 토큰 삽입 대신 본 논문에서 제안하는 **TOKI**를 적용하여 시퀀스의 연속성을 보존한다.

### 4. Selective Structure State Space Sequence Token-Interpolation (TOKI)
TOKI는 Mamba의 SSM(State Space Model) 수식을 기반으로 마스킹된 토큰을 생성한다.

Mamba의 기본 이산화된 상태 방정식은 다음과 같다:
$$y_j = \sum_{i=1}^j C \bar{A}^{j-i} \bar{B} x_i$$
여기서 $\bar{A}$는 상태 행렬, $C$와 $\bar{B}$는 계산 파라미터이다.

기존 MIM은 마스킹된 위치에 상수 $\delta$를 넣지만, TOKI는 마스킹된 영역 전후의 가시적 토큰 $\hat{y}_i$와 $\hat{y}_{i+1}$ 사이를 보간한다.
1.  가시적 토큰 사이의 선형 보간 값 $V$를 계산한다: $V = (1-\alpha)\hat{y}_i + \alpha\hat{y}_{i+1}$ (여기서 $\alpha$는 위치에 따른 가중치).
2.  학습 가능한 파라미터 $\bar{A}'$를 사용하여 다음과 같이 보간된 토큰 $z_j$를 생성한다:
    $$z_j = \sum_{n=0}^j \text{Pow}(\bar{A}', j-n) \Delta \bar{A}'^{-1} \cdot V$$
이 과정은 마스킹된 토큰이 Mamba의 인과적 흐름을 따르도록 강제하여, 단순 학습 가능 토큰보다 훨씬 더 풍부한 문맥 정보를 제공한다.

### 5. 학습 절차 및 손실 함수
- **사전 학습 (Pre-training)**: 6,814개의 CT 스캔 데이터셋을 사용하며, 마스킹된 위치의 픽셀 값을 복원하는 $L^2$ (Mean Square Error) 손실 함수를 사용한다.
- **미세 조정 (Fine-tuning)**: Binary Cross Entropy (BCE) 손실과 Dice Similarity Coefficient (DSC) 손실의 합을 사용하여 최적화한다: $L_{seg} = L_{BCE} + L_{DSC}$.

## 📊 Results

### 1. 실험 설정
- **데이터셋**: BTCV, MSD, AMOS, CT-ORG, KiTS 23, BraTS 21 등 8개의 벤치마크 사용.
- **비교 대상**: MAE, SimMIM (Transformer 기반), SparK (CNN 기반), MoCoV2, SimSiam (대조 학습 기반) 및 다양한 최신 분할 네트워크(Swin UNETR, MedNeXt, SegMamba 등).
- **평가 지표**: Dice Similarity Coefficient (DSC).

### 2. 주요 결과
- **HyMamba의 우수성**: 사전 학습 없이 HyMamba만 사용했을 때 이미 Vanilla Mamba(70.94%)나 SegMamba(76.54%)보다 높은 77.56% (BTCV 기준)를 달성하여, 하이브리드 구조의 효율성을 입증했다.
- **MambaMIM의 효과**: HyMamba에 MambaMIM 사전 학습을 적용했을 때 DSC가 **80.16%**로 크게 상승했다. 특히 췌장(Pancreas), 위(Stomach), 담낭(Gallbladder)과 같이 크기가 작고 분할이 어려운 장기에서 성능 향상이 두드러졌다.
- **SOTA 달성**: BTCV 및 AMOS 데이터셋에서 기존의 어떤 사전 학습 방법보다 높은 성능을 보였으며, 특히 SparK나 MAE보다 뛰어난 결과를 냈다.
- **일반화 성능**: 사전 학습에 사용되지 않은 CT-ORG, KiTS 23 데이터셋에서도 SOTA 성능을 기록했으며, CT 도메인에서 학습한 모델을 MRI 도메인(BraTS 21)으로 전이했을 때도 89.91%라는 높은 DSC를 기록하며 모달리티 간 일반화 능력을 보여주었다.

## 🧠 Insights & Discussion

### 1. 강점 및 분석
본 연구는 Mamba라는 새로운 아키텍처를 의료 영상의 대규모 데이터셋에 어떻게 효과적으로 사전 학습시킬 수 있는지를 이론적, 실험적으로 증명했다. 특히 **TOKI**의 도입은 Mamba의 수학적 특성(SSM)을 직접적으로 고려한 설계라는 점에서 학술적 가치가 높다. 또한, 대조 학습보다 MIM 기반의 사전 학습이 분할 작업에 더 유리하다는 점을 재확인했는데, 이는 대조 학습은 전역적인 시맨틱 특징을 학습하는 반면, MIM은 국소적인 구조와 세부 패턴을 학습하기 때문으로 분석된다.

### 2. 한계 및 논의사항
- **계산 복잡도**: 하이브리드 구조와 TOKI의 보간 과정이 추가됨에 따라 단순 Mamba 모델보다 연산 오버헤드가 발생할 수 있다.
- **마스크 비율**: 실험 결과 마스크 비율 75%에서 최적의 성능이 나왔는데, 이는 너무 적은 마스킹은 학습 난이도를 낮춰 표현력 학습을 방해하고, 너무 많은 마스킹(90% 이상)은 복원할 정보가 부족해지기 때문으로 보인다.
- **스캔 방식의 강건성**: Raster, Zigzag, Hilbert, Shuffle 등 다양한 스캔 방식에 대해 성능 차이가 크지 않았다는 점은, 사전 학습된 모델이 입력 순서의 변화에 상당히 강건함을 시사한다.

## 📌 TL;DR

**MambaMIM**은 Mamba의 인과적 특성을 보존하는 **TOKI(토큰 보간)** 전략과 하이브리드 구조를 위한 **Bottom-up 마스킹**을 도입한 새로운 생성적 사전 학습 프레임워크이다. 대규모 3D CT 데이터로 사전 학습된 이 모델은 다양한 의료 영상 분할 작업에서 기존 SOTA를 뛰어넘는 성능을 보였으며, 특히 CT에서 MRI로의 모달리티 전이 학습에서도 강력한 일반화 능력을 입증하였다. 이는 향후 Mamba 기반의 의료 영상 분석 모델 설계에 있어 핵심적인 사전 학습 가이드라인을 제공할 것으로 기대된다.