# SliceMamba with Neural Architecture Search for Medical Image Segmentation

Chao Fan, Hongyuan Yu, Yan Huang, Liang Wang, Zhenghan Yang, and Xibin Jia (2024)

## 🧩 Problem to Solve

본 논문은 의료 영상 분할(Medical Image Segmentation) 분야에서 최근 주목받는 Mamba 기반 모델들이 가진 한계점을 해결하고자 한다. 기존의 Vision Mamba 모델들은 2D 이미지를 1D 시퀀스로 변환하기 위해 단방향 또는 다방향 스캔 메커니즘(예: SS2D)을 사용한다. 그러나 이러한 방식은 공간적으로 인접한 픽셀들이 스캔 시퀀스 내에서는 멀리 떨어지게 배치되는 문제를 야기하며, 결과적으로 의료 영상에서 병변이나 장기의 구조적 정보를 파악하는 데 필수적인 지역적 특징(Local Features)의 학습 능력을 저하시킨다.

따라서 본 연구의 목표는 공간적 인접성을 보존하는 새로운 스캔 메커니즘을 도입하여 지역적 특징과 전역적 문맥(Global Context)을 동시에 효과적으로 캡처할 수 있는 Mamba 기반의 의료 영상 분할 모델인 SliceMamba를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같다.

1. **Bidirectional Slice Scan (BSS) 모듈 제안**: 특징 맵을 수평 및 수직 방향으로 슬라이싱하고, 각 슬라이스의 형태에 최적화된 양방향 스캔 방식을 적용함으로써 공간적으로 인접한 특징들이 스캔 시퀀스 내에서도 가깝게 유지되도록 설계하였다.
2. **Adaptive Slice Search 방법론 도입**: 병변이나 장기의 크기와 모양이 데이터셋마다 다르다는 점에 착안하여, 최적의 슬라이스 크기 조합을 자동으로 탐색하는 NAS(Neural Architecture Search) 기반의 방법론을 제안하였다.
3. **Pure Mamba 기반 아키텍처**: 추가적인 Convolution 레이어에 의존하지 않고 Mamba 구조만으로 지역적/전역적 특징 추출을 수행하는 효율적인 모델을 구축하였다.
4. **다양한 의료 데이터셋 검증**: 피부 병변, 폴립, 다기관 분할 등 5개의 서로 다른 모달리티 데이터셋에서 기존 SOTA 모델 대비 우수한 성능을 입증하였다.

## 📎 Related Works

### 1. 의료 영상 분할 연구

- **CNN 기반**: UNet 등이 대표적이며 지역적 특징 추출에 강점이 있으나, 수용역(Receptive Field)의 한계로 인해 장거리 의존성(Long-range dependencies) 모델링이 어렵다.
- **Transformer 기반**: TransUNet, Swin-UNet 등이 전역적 문맥을 잘 파악하지만, 입력 크기에 따라 연산 복잡도가 제곱으로 증가하는 $\mathcal{O}(N^2)$ 문제가 있어 고해상도 의료 영상 적용에 제약이 있다.
- **Mamba 기반**: SSM(State Space Model)을 활용해 선형 복잡도 $\mathcal{O}(N)$로 장거리 의존성을 모델링한다. U-Mamba, VM-UNet 등이 제안되었으나, 본문에서 지적하듯 이미지의 공간적 상관관계를 무시하고 시퀀스로 펼치는 과정에서 지역적 특징이 소실되는 문제가 존재한다.

### 2. 신경망 구조 탐색 (NAS)

- 기존 NAS는 막대한 계산 비용이 소요되었으나, 최근에는 가중치 공유(Weight-sharing) 전략을 통해 비용을 줄이는 추세이다. 본 논문은 특히 Supernet 학습과 구조 탐색을 분리하여 가중치 커플링 문제를 해결한 **Single Path One-Shot (SPOS)** 방식을 채택하여 효율적으로 최적의 슬라이스 조합을 찾는다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

SliceMamba는 기본적인 UNet의 대칭적 인코더-디코더 구조를 따른다.

- **Encoder**: Patch Embedding layer $\rightarrow$ 4단계의 $S^3$ 블록 및 Patch Merging (PM) 모듈로 구성된다.
- **Decoder**: Patch Expanding (PE) 모듈 $\rightarrow$ $S^3$ 블록으로 구성되며, 인코더의 특징 맵을 Skip Connection을 통해 더해준다.
- **Final Mapping**: 최종적으로 마스크 형상에 맞게 특징 맵의 크기를 조정한다.

### 2. $S^3$ (Slice Selective Scan) 블록

$S^3$ 블록은 모델의 핵심 단위로, 다음과 같은 흐름으로 연산이 수행된다.

1. 입력 특징은 선형 임베딩 층을 거쳐 두 경로로 나뉜다.
2. **경로 1**: $3 \times 3$ Depth-wise Convolution $\rightarrow$ SiLU 활성화 함수 $\rightarrow$ **BSS 모듈** $\rightarrow$ Layer Normalization 순으로 처리된다.
3. **경로 2**: 입력 특징이 그대로 유지된다.
4. 두 경로의 출력물을 원소별 곱셈(Element-wise multiplication)한 후, 다시 선형 임베딩을 거쳐 입력값과 잔차 연결(Residual connection)을 수행한다.

### 3. BSS (Bidirectional Slice Scan) 모듈

BSS 모듈은 지역적 특징을 보존하기 위해 특징 맵 $F \in \mathbb{R}^{H \times W \times C}$를 다음과 같이 처리한다.

- **특징 슬라이싱**:
  - 수평 방향으로 $m$ 크기의 슬라이스로 분할하여 $F_h$ 집합을 생성한다.
  - 수직 방향으로 $n$ 크기의 슬라이스로 분할하여 $F_v$ 집합을 생성한다.
- **방향별 스캔**:
  - 수평 슬라이스($m \times W \times C$): 위$\rightarrow$아래 및 아래$\rightarrow$위 방향으로 스캔한다.
  - 수직 슬라이스($H \times n \times C$): 왼쪽$\rightarrow$오른쪽 및 오른쪽$\rightarrow$왼쪽 방향으로 스캔한다.
- **모델링 및 복원**: 생성된 4개의 시퀀스를 $S^6$ (Selective SSM) 블록에 입력하여 처리한 후, 다시 원래의 $H \times W \times C$ 형상으로 복원하고 원소별 덧셈으로 결합한다.

### 4. Adaptive Slice Search (ASS)

최적의 슬라이스 크기 $(m, n)$을 찾기 위해 다음 절차를 수행한다.

- **탐색 공간**: $\mathcal{S} = \{(2,2), (2,4), (4,2), (4,4)\}$의 4가지 후보 조합을 정의한다.
- **SPOS NAS 적용**:
    1. 모든 후보 조합을 포함하는 **Supernet**을 한 번 학습시킨다. 이때 가중치 $W^S$를 최적화한다:
       $$\min_{W} \mathcal{L}_{\text{train}}(N(\mathcal{S}, W))$$
    2. 학습된 가중치를 고정한 채, 검증 데이터셋에서 가장 높은 DSC(Dice Similarity Coefficient)를 보이는 조합 $c^*$를 진화 알고리즘(Evolutionary Algorithm)으로 탐색한다:
       $$c^* = \arg \max_{c \in \mathcal{S}} \text{DSC}_{\text{val}}(N(c, W^S(c)))$$
    3. 최종 선택된 조합으로 모델을 처음부터 다시 학습(Train from scratch)시킨다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: ISIC2017, ISIC2018 (피부 병변), Kvasir, ClinicDB (폴립), Synapse (다기관)
- **지표**: mIoU, DSC, Accuracy, Specificity, Sensitivity (피부/폴립), HD95 (Synapse)
- **구현**: RTX 4090 GPU, AdamW 옵티마이저, CosineAnnealingLR 스케줄러 사용.

### 2. 주요 결과

- **피부 병변 분할**: ISIC2017에서 mIoU 81.70%, DSC 89.93%를 기록하며 VM-UNet, TM-UNet 등 기존 Mamba 모델을 상회하였다.
- **폴립 분할**: Kvasir 및 ClinicDB에서 매우 큰 성능 향상을 보였다. 특히 TM-UNet 대비 mIoU가 최대 9.54% 향상되었는데, 이는 폴립 데이터셋이 병변과 배경의 유사성이 높아 지역적 특징 추출 능력이 매우 중요하기 때문으로 분석된다.
- **다기관 분할 (Synapse)**: DSC에서는 VM-UNet과 비슷했으나, 경계선 추출 능력을 나타내는 HD95 지표에서 $16.04$를 기록하여 VM-UNet($19.21$)보다 훨씬 우수한 성능을 보였다.

### 3. Ablation Study

- **BSS 및 ASS 효과**: Baseline(VM-UNet)에 BSS만 추가했을 때보다 ASS를 통해 최적의 슬라이스 조합을 찾았을 때 mIoU가 더 크게 향상됨을 확인하였다.
- **ImageNet 사전 학습**: 자연 이미지로 사전 학습한 가중치를 사용했을 때 성능이 향상되었다. 특히 데이터셋 규모가 작은 Synapse 데이터셋에서 DSC가 4.71% 상승하는 등 효과가 뚜렷하였다.

## 🧠 Insights & Discussion

### 강점

본 논문은 Mamba 모델이 가진 근본적인 문제인 '공간적 인접성 파괴'를 **슬라이싱 기반의 스캔 방식**이라는 단순하면서도 강력한 아이디어로 해결하였다. 특히, 모든 데이터에 동일한 구조를 강요하지 않고 NAS를 통해 데이터 특성에 맞는 최적의 슬라이스 크기를 결정한 점이 성능 향상의 주요 요인으로 판단된다.

### 한계 및 논의사항

- **탐색 공간의 제한**: 슬라이스 후보군을 $\mathcal{S} = \{(2,2), (2,4), (4,2), (4,4)\}$로 제한하여 설정하였는데, 더 넓은 범위의 슬라이스 크기가 존재할 때의 영향에 대해서는 명시되지 않았다.
- **계산 비용**: Supernet 학습 및 진화 알고리즘을 통한 탐색 과정이 추가되므로, 초기 모델 구축 시의 시간적 비용이 발생한다. 다만, 한 번 결정된 구조는 추론 시 추가 연산이 없다는 점이 긍정적이다.
- **사전 학습의 영향**: 자연 이미지 사전 학습이 의료 영상에서도 유효함을 보였으나, 의료 영상 전용 대규모 데이터셋으로 사전 학습했을 때 어느 정도의 추가 이득이 있을지에 대한 연구가 필요해 보인다.

## 📌 TL;DR

SliceMamba는 기존 Mamba 기반 의료 영상 분할 모델이 지역적 특징을 놓치는 문제를 해결하기 위해 **양방향 슬라이스 스캔(BSS)** 메커니즘을 도입하였다. 또한 **NAS(Adaptive Slice Search)**를 통해 대상 데이터에 최적화된 슬라이스 크기를 자동으로 탐색한다. 실험 결과, 특히 지역적 세부 정보가 중요한 폴립 분할 및 장기 경계 추출(HD95)에서 기존 모델들을 압도하는 성능을 보였으며, 이는 향후 Mamba 기반 비전 모델들이 공간적 구조를 어떻게 다뤄야 하는지에 대한 중요한 방향성을 제시한다.
