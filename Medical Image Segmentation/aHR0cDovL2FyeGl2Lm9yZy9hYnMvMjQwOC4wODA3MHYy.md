# MambaMIM: Pre-training Mamba with State Space Token Interpolation and its Application to Medical Image Segmentation

Fenghe Tang et al. (2025)

## 🧩 Problem to Solve

본 논문은 3D 의료 영상 분석에서 효율적인 장거리 시퀀스 모델링 능력을 가진 State Space Model(SSM)인 Mamba를 활용하고자 한다. 하지만 기존의 generative self-supervised learning(SSL), 특히 Masked Image Modeling(MIM) 방법론을 Mamba에 그대로 적용하는 데에는 다음과 같은 두 가지 핵심적인 문제가 존재한다.

첫째, **상태 공간 모델의 인과적 특성(Causal Properties) 무시**이다. MAE와 같은 기존 MIM 방식은 마스킹된 영역을 랜덤하게 초기화된 learnable token으로 채운다. 그러나 Mamba의 핵심인 selective scan 구조는 입력 데이터에 의존적인 인과적 흐름을 가지므로, 무작위 토큰의 삽입은 상태 공간 내의 구조적 시퀀스 관계를 파괴하여 모델이 효과적인 표현을 학습하는 것을 방해한다.

둘째, **하이브리드 아키텍처에서의 마스킹 일관성(Masking Consistency) 유지 문제**이다. 최근 Mamba를 CNN과 결합한 하이브리드 구조가 많이 제안되고 있으나, CNN의 sparse convolution 방식과 Mamba의 시퀀스 처리 방식은 마스킹 전략이 서로 다르다. 서로 다른 구성 요소 간에 마스크 위치가 일치하지 않으면 픽셀 분포의 변화(distributional shift)가 발생하여 표현 학습 능력이 저하된다.

따라서 본 논문의 목표는 Mamba의 인과적 특성을 보존하면서 하이브리드 구조에서도 일관된 마스킹을 유지할 수 있는 전용 pre-training 프레임워크인 **MambaMIM**을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 설계 아이디어는 Mamba의 수학적 특성을 반영한 토큰 생성 전략과 하이브리드 구조를 위한 정교한 마스킹 파이프라인을 구축하는 것이다.

1. **TOKI (TOKen-Interpolation) 전략**: 단순히 learnable token을 삽입하는 대신, 마스킹되지 않은 주변 상태 공간 시퀀스 간의 관계를 수학적으로 보간(interpolation)하여 마스크 토큰을 생성한다. 이를 통해 Mamba의 인과적 흐름을 유지하며 장거리 의존성을 극대화한다.
2. **Bottom-up 3D Hybrid Masking**: 가장 하위 계층(bottom stage)에서 마스크를 먼저 생성하고 이를 상위 계층으로 전파(up-sampling)하는 방식을 통해, CNN과 Mamba라는 서로 다른 아키텍처 간의 마스킹 위치 일관성을 보장한다.
3. **범용적 Mamba Pre-training 프레임워크**: 제안된 방법론은 Vanilla Mamba뿐만 아니라 다양한 하이브리드 Mamba 구조에 적용 가능하며, 대규모 3D CT 데이터셋을 통해 그 효과를 입증하였다.

## 📎 Related Works

본 논문에서는 다음과 같은 관련 연구들의 한계를 지적하며 차별점을 제시한다.

- **Masked Image Modeling (MIM)**: MAE나 SimMIM 등은 ViT 기반으로 설계되었으며, SparK는 CNN 기반으로 설계되었다. 이러한 방법들은 Mamba의 핵심인 상태 공간(state space) 내부의 구조적 시퀀스 관계를 고려하지 않아 Mamba 모델의 잠재력을 충분히 끌어내지 못한다.
- **Medical SSL (Contrastive Learning)**: SimSiam, MoCoV2 등 대조 학습 기반 방법들은 주로 고수준의 시맨틱 표현(high-level semantic representations)을 학습한다. 이는 분류 작업에는 유리하나, 세밀한 국소적 특징 학습이 필요한 세그멘테이션(segmentation) 작업에서는 성능 한계가 있다.
- **Mamba in Medical Imaging**: 최근 Mamba 기반의 의료 영상 모델들이 제안되었으나, 대부분 지도 학습(supervised learning)에 의존한다. 의료 영상의 레이블 부족 문제를 해결하기 위해 Mamba 전용의 효율적인 SSL 방법론이 절실한 상황이다.

## 🛠️ Methodology

### 1. 전체 시스템 구조 및 베이스라인
MambaMIM은 크게 **Hierarchical Hybrid Encoding**과 **Decoding** 단계로 구성된다. 베이스라인으로는 두 가지 모델을 사용한다.
- **Vanilla Mamba**: UNETR 구조에서 ViT를 Vision Mamba로 대체한 모델이다.
- **HyMamba**: 상위 단계에는 CNN(MedNeXt)을 배치하고, 하위 단계에는 Vision Mamba를 배치한 직렬 하이브리드 구조이다.

### 2. Hierarchical Hybrid Encoding
하이브리드 구조에서 마스킹 일관성을 유지하기 위해 **Bottom-up masking** 전략을 사용한다.
- 가장 마지막 단계인 $n$-번째 CNN stage에서 랜덤하게 마스크 $M^n$을 생성한다.
- 이를 상위 단계($1$ ~ $n-1$ stage)로 up-sampling 하여 $\{M^1, \dots, M^{n-1}\}$ 세트를 생성함으로써 모든 계층에서 동일한 위치가 마스킹되도록 한다.
- CNN 단계에서는 3D Sparse Operator ($\text{SparseOp}$)를 사용하여 마스킹된 위치의 계산을 건너뛰며, Mamba 단계에서는 마스킹되지 않은 토큰들만 직렬화하여 처리한다.

### 3. Hierarchical Hybrid Decoding
인코더에서 생성된 계층적 sparse feature들을 다시 복원하는 과정이다.
- **CNN 부분**: learnable token embeddings를 빈 공간에 채우고, upsampling block과 skip-connection을 통해 조밀한(dense) 특징을 복원한다.
- **Mamba 부분**: 본 논문의 핵심인 **TOKI**를 적용하여 마스킹된 시퀀스를 복원한다.

### 4. TOKI (Selective Structure State Space Sequence Token-Interpolation)
TOKI는 Mamba의 SSM 공식을 기반으로 마스킹된 토큰을 생성한다. SSM의 기본 원리는 입력 $x(t)$를 중간 상태 $h(t)$를 통해 출력 $y(t)$로 매핑하는 것이며, 이는 다음과 같은 선형 상미분 방정식(ODE)으로 표현된다.
$$h'(t) = Ah(t) + Bx(t), \quad y(t) = Ch(t)$$

이산화(discretization)된 형태에서 출력 $y_j$는 다음과 같이 합산 형태로 표현될 수 있다.
$$y_j = \sum_{i=1}^j C \bar{A}^{j-i} \bar{B} x_i$$

기존 MIM은 마스킹된 인덱스 $\Omega$에 대해 단순히 학습 가능한 파라미터 $\delta$를 채워 넣었으나, TOKI는 마스킹 영역 앞뒤의 가시적 토큰 $\hat{y}_i$와 $\hat{y}_{i+1}$ 사이를 수학적으로 보간한다. 
구체적으로, 마스킹된 구간의 길이 $Q$에 대해, 보간된 값 $V$를 정의하고 이를 학습 가능한 파라미터 $\bar{A}'$를 이용해 다음과 같이 계산한다.
$$z_j = \sum_{n=0}^j \text{Pow}(\bar{A}', j-n) \Delta \bar{A}'^{-1} \cdot V$$
이 과정을 통해 생성된 $z_j$는 Mamba의 인과적 흐름을 깨지 않으면서도 주변 맥락을 반영한 토큰이 되어, 모델이 더 정교한 표현을 학습하게 한다.

### 5. 학습 절차 및 손실 함수
- **Pre-training**: 6,814개의 대규모 CT 스캔 데이터셋을 사용한다. 복원된 이미지 $\hat{X}$와 원본 이미지 $X$ 사이의 **Mean Square Error ($L_2$) loss**를 사용하여 학습한다.
- **Fine-tuning**: 다운스트림 세그멘테이션 작업에서는 **Binary Cross Entropy (BCE) loss**와 **Dice Similarity Coefficient (DSC) loss**의 결합 손실 함수를 사용한다.

## 📊 Results

### 1. 실험 설정
- **데이터셋**: BTCV, MSD, AMOS, CT-ORG, KiTS 23, BraTS 21 등 8개의 벤치마크를 사용하였다.
- **비교 대상**: MAE, SimMIM, SparK (MIM 계열), SimSiam, SwAV, MoCoV2 (Contrastive 계열) 및 UNETR, Swin UNETR, MedNeXt 등 최신 세그멘테이션 네트워크와 비교하였다.
- **평가 지표**: Dice Similarity Coefficient (DSC)를 사용하였다.

### 2. 주요 결과
- **HyMamba의 성능**: HyMamba를 scratch부터 학습시켰을 때 BTCV에서 77.56%의 DSC를 기록하여, Vanilla Mamba(70.94%)와 SegMamba(76.54%)보다 우수한 성능을 보였다.
- **MambaMIM의 효과**: HyMamba에 MambaMIM pre-training을 적용했을 때 DSC가 **80.16%**로 크게 향상되었다 (2.6%p 증가). 특히 췌장(Pancreas, 7.58% 증가)과 같이 작은 장기에서 성능 향상이 두드러졌다.
- **전이 학습 능력**:
    - **Unseen Datasets**: 학습에 사용되지 않은 CT-ORG와 KiTS 23에서도 각각 89.42%, 67.19%의 DSC를 기록하며 높은 일반화 능력을 보였다.
    - **Cross-modality**: CT 데이터로 학습한 모델을 MRI 데이터셋인 BraTS 21에 적용했을 때 89.91%의 DSC를 기록하며, 모달리티가 달라도 국소적 특징 학습 효과가 유효함을 입증하였다.

### 3. Ablation Study
- **Mask Ratio**: 마스킹 비율이 높을수록 성능이 향상되는 경향을 보였으며, **75%**에서 최적의 성능(80.16%)을 달성하였다. 이는 더 많은 영역을 마스킹할수록 모델이 잠재 표현(latent representation)을 더 강하게 학습해야 하기 때문으로 해석된다.
- **구성 요소 분석**: Bottom-up masking, Skip-connection, TOKI 모두 성능 향상에 기여하였으며, 특히 TOKI를 적용했을 때 learnable token 방식보다 성능이 크게 개선되었다.
- **Scanning Method**: Raster, Zigzag, Hilbert, Shuffle 등 다양한 스캔 방식에 대해 강건한 성능을 보였으며, Shuffle scan이 가장 높은 성능(80.89%)을 나타냈다.

## 🧠 Insights & Discussion

본 논문은 Mamba 아키텍처를 의료 영상 분석에 적용할 때, 단순한 모델 구조의 변경보다 **데이터의 특성과 모델의 수학적 메커니즘(SSM)을 반영한 학습 전략**이 훨씬 더 중요하다는 점을 시사한다.

**강점**:
- Mamba의 인과적 특성을 고려한 TOKI 전략을 통해 기존 MIM의 한계를 극복하였다.
- 하이브리드 구조에서 발생할 수 있는 마스킹 불일치 문제를 Bottom-up 방식으로 깔끔하게 해결하였다.
- CT에서 MRI로의 전이 학습 성공을 통해, 제안한 방법이 고수준 시맨틱이 아닌 범용적인 국소 특징을 효과적으로 학습함을 보였다.

**한계 및 논의**:
- TOKI의 수학적 유도 과정에서 $\bar{B}'$와 $\bar{C}'$를 identity matrix로 단순화하였는데, 이 부분이 실제 성능에 미치는 영향에 대한 추가 분석이 필요할 수 있다.
- 대규모 데이터셋을 사용한 pre-training이 필수적이므로, 연산 자원이 부족한 환경에서의 효율적인 학습 방안에 대한 연구가 향 uma la.

## 📌 TL;DR

MambaMIM은 Mamba 기반 의료 영상 모델을 위한 최초의 Masked Image Modeling(MIM) 프레임워크이다. Mamba의 인과적 시퀀스 특성을 보존하는 **TOKI(토큰 보간)** 전략과 하이브리드 구조를 위한 **Bottom-up 마스킹**을 도입하여, 3D 의료 영상 세그멘테이션에서 SOTA 성능을 달성하였다. 특히 MRI와 같은 타 모달리티로의 높은 일반화 성능을 보여, 향후 레이블이 부족한 다양한 의료 영상 분석 작업에 핵심적인 pre-training 방법론이 될 가능성이 높다.