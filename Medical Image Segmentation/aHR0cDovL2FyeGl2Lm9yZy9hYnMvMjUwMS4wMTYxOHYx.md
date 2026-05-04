# Merging Context Clustering with Visual State Space Models for Medical Image Segmentation

Yun Zhu, Dong Zhang, Yi Lin, Yifei Feng, Jinhui Tang (2025)

## 🧩 Problem to Solve

본 논문은 의료 영상 분할(Medical Image Segmentation, MedISeg) 작업에서 전역적(global) 특징과 지역적(local) 특징 표현을 동시에 효율적으로 통합해야 하는 문제를 해결하고자 한다. 의료 영상 분할은 병변의 정밀한 경계를 구분해야 하므로 단거리 및 장거리 특징 상호작용을 모두 처리하는 능력이 필수적이다.

최근 Vision Mamba (ViM) 계열의 모델들이 선형 복잡도로 장거리 특징 상호작용을 처리하며 유망한 대안으로 떠올랐으나, 두 가지 핵심적인 한계가 존재한다. 첫째, 공간 토큰을 단순히 평탄화(flattening)하여 처리함으로써 지역적인 의존성(short-range local dependencies)을 보존하는 능력이 부족하다. 둘째, 고정된 스캐닝 패턴(fixed scanning patterns)을 사용하기 때문에 이미지의 동적인 공간 문맥 정보(dynamic spatial context information)를 유연하게 포착하지 못한다. 따라서 본 연구의 목표는 이러한 ViM의 한계를 극복하여 지역적-전역적 특징 상호작용을 모두 강화한 새로운 모델을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 고정된 스캐닝 방식 대신, 이미지를 데이터 포인트의 집합으로 간주하고 이를 동적으로 그룹화하는 **Context Clustering (CC)** 메커니즘을 도입하는 것이다.

주요 기여 사항은 다음과 같다:

1. **CCViM 모델 제안**: 의료 영상 분할을 위해 설계된 효율적이고 효과적인 Mamba 기반의 U-자형 아키텍처를 제안한다.
2. **CCS6 (Context Clustering Selective State Space) 레이어 설계**: 전역적 스캐닝 방향과 지역적 CC 레이어를 결합하여 모델의 특징 표현 능력을 향상시킨다.
3. **CC 레이어 도입**: 고정된 스캐닝 전략의 한계를 넘어, 지역 윈도우 내에서 데이터 포인트들을 동적으로 클러스터링함으로써 공간 문맥 정보를 적응적으로 추출한다.
4. **광범위한 검증**: 5개의 공공 의료 영상 데이터셋(Kumar, CPM17, ISIC17, ISIC18, Synapse)을 통해 기존 최신 모델(SOTA)보다 우수한 성능을 입증하였다.

## 📎 Related Works

기존의 의료 영상 분할 접근 방식은 크게 세 가지 흐름으로 구분된다.

- **CNN 기반 모델**: 지역적 특징 포착에 강점이 있고 Translation Equivalence와 같은 유효한 귀납적 편향(inductive bias)을 가지지만, 수용 영역(receptive field)이 제한적이어서 전역적 문맥을 파악하는 능력이 떨어진다.
- **Vision Transformer (ViT) 기반 모델**: 모든 토큰 간의 상호작용을 통해 전역적 특징을 효과적으로 추출하여 우수한 성능을 보이지만, 토큰 수에 비례하여 계산 복잡도가 제곱($O(N^2)$)으로 증가하는 치명적인 단점이 있다.
- **Mamba 기반 모델 (Visual State Space Models)**: 선형 복잡도로 장거리 의존성을 모델링할 수 있어 효율적이다. VMamba는 Cross-scan 모듈을 통해 2D 의존성을 해결하려 했고, LocalMamba는 지역 스캐닝 전략을 도입했다. 그러나 이들은 여전히 고정된 스캐닝 경로에 의존하며, 데이터의 특성에 따라 동적으로 공간 관계를 포착하는 능력이 부족하다.

본 논문은 이러한 고정적 접근법에서 벗어나, 이미지를 포인트 집합으로 처리하는 클러스터링 기반의 새로운 패러다임을 Mamba 모델에 통합함으로써 차별성을 갖는다.

## 🛠️ Methodology

### 1. 전체 시스템 구조 (Overall Architecture)

CCViM은 비대칭적 U-자형 구조를 가진다. 전체 파이프라인은 다음과 같다:

- **Patch Embedding**: 입력 이미지 $I \in \mathbb{R}^{3 \times W \times H}$를 $4 \times 4$ 크기의 패치로 나누어 $C \times \frac{W}{4} \times \frac{H}{4}$ 차원의 특징 맵을 생성한다.
- **Encoder**: 4단계로 구성되며, 각 단계는 2개의 CCViM 블록과 1개의 Patch Merging 레이어로 이루어져 있다. 채널 수는 $[C, 2C, 4C, 8C]$ 순으로 증가한다.
- **Decoder**: 4단계로 구성되며, 각 단계는 1개의 Patch Expanding 레이어와 2개의 CCViM 블록으로 구성된다. 채널 수는 $[8C, 4C, 2C, C]$ 순으로 감소한다.
- **Skip Connections**: 인코더와 디코더 사이에 단순 덧셈(addition) 기반의 연결을 사용하여 저수준 및 고수준 특징을 통합한다.

### 2. CCViM Block 및 CCS6 레이어

CCViM 블록의 핵심은 **CCS6 레이어**이다. 이 레이어는 입력 데이터를 처리하기 위해 총 6가지의 옵션 중 4가지를 선택하여 병렬로 처리한 후 S6(Selective State Space) 모듈에 입력한다.

- **전역 경로 (Global Paths)**: VMamba의 Cross-scan 모듈을 사용하여 가로, 가로-반전, 세로, 세로-반전의 4가지 방향으로 스캐닝한다.
- **지역 경로 (Local Paths)**: 서로 다른 클러스터 중심 개수(4개 및 25개)를 가진 2개의 CC 레이어를 적용하여 지역적 공간 문맥을 추출한다.

### 3. Context Clustering (CC) 레이어의 작동 원리

CC 레이어는 이미지를 픽셀 그리드가 아닌 데이터 포인트의 집합 $P \in \mathbb{R}^{d \times n}$으로 취급한다.

1. **윈도우 분할**: 계산 비용을 줄이기 위해 전체 이미지가 아닌 지역 윈도우 단위로 클러스터링을 수행한다.
2. **중심점 제안**: 각 윈도우에서 $t$개의 중심점을 제안하고, $k$-최근접 이웃(k-nearest points)의 평균을 통해 중심 특징을 계산한다.
3. **유사도 계산 및 할당**: 제안된 중심점과 포인트 간의 코사인 유사도를 계산하여 각 포인트를 가장 유사한 중심점에 할당한다.
4. **특징 응집 (Aggregation)**: 각 클러스터 내의 포인트들을 다음과 같은 식을 통해 하나의 응집된 특징 $g$로 통합한다:
   $$g = \frac{1}{T} \left( v_c + \sum_{i=1}^{m} \sigma(\alpha s_i + \beta) * p_i \right)$$
   여기서 $v_c$는 가치 중심(value center), $s_i$는 유사도, $\sigma$는 시그모이드 함수, $T$는 정규화 계수이다.
5. **특징 업데이트**: 응집된 특징 $g$를 다시 각 포인트 $p_i$에 전달하여 업데이트한다:
   $$p'_i = p_i + FC(\sigma(\alpha s_i + \beta) * g)$$

### 4. 학습 절차 및 손실 함수

모델은 ImageNet으로 사전 학습된 가중치를 초기값으로 사용하며, AdamW 옵티마이저와 CosineAnnealingLR 스케줄러를 적용한다. 손실 함수는 픽셀 수준의 정확도를 위한 Cross-Entropy ($L_{Ce}$)와 클래스 불균형 문제를 해결하기 위한 Dice Loss ($L_{Dice}$)를 결합하여 사용한다:
$$L = L_{Ce} + L_{Dice}$$

## 📊 Results

### 1. 실험 설정

- **데이터셋**: nuclei segmentation (Kumar, CPM17), skin lesion segmentation (ISIC17, ISIC18), multi-organ segmentation (Synapse).
- **지표**: PQ, Dice, AJI, DQ, SQ (핵 분할) / mIoU, DSC, Acc, Sen, Spe (피부 병변) / DSC, HD95 (다기관).
- **비교 대상**: UNet, TransUNet, Swin-Unet, VM-UNet, LocalMamba 등.

### 2. 정량적 결과

- **핵 분할 (Kumar, CPM17)**: CCViM은 PQ 지표에서 가장 우수한 성능을 보였다. 특히 Kumar 데이터셋에서 VM-UNet 대비 PQ가 0.61% 향상되었으며, 이는 작은 핵들의 정밀한 분리 능력이 강화되었음을 의미한다.
- **피부 병변 분할 (ISIC17, ISIC18)**: ISIC17에서 mIoU와 DSC를 각각 1.17%, 0.71% 향상(VM-UNet 대비)시키며 SOTA 성능을 달성했다.
- **다기관 분할 (Synapse)**: DSC를 1.57% 향상시키고 HD95(Hausdorff Distance)를 1.38% 감소시켜 경계 묘사 능력이 개선되었음을 입증했다.

### 3. 효율성 분석

- **계산 복잡도**: 파라미터 수와 FLOPs 측면에서 VM-UNet 및 LocalVMamba와 유사한 수준을 유지하면서 성능은 더 높다.
- **추론 속도**: 41.16 fps의 처리량(throughput)을 기록하여, 매우 효율적인 실시간 추론 가능성을 보여주었다. 이는 CC 레이어가 전역 연산이 아닌 지역 윈도우 단위로 작동하기 때문이다.

## 🧠 Insights & Discussion

### 강점 및 유효성

본 논문의 가장 큰 성과는 **적응적 지역 특징 추출(Adaptive Local Feature Extraction)**의 구현이다. 기존 Mamba 기반 모델들이 고정된 경로로 스캔하며 지역 정보를 놓쳤던 반면, CCViM은 데이터 포인트 간의 유사도를 기반으로 동적으로 클러스터를 형성함으로써 의료 영상 특유의 불규칙한 형태와 미세한 경계 세부 사항을 더 잘 포착할 수 있었다.

### 한계 및 비판적 해석

논문에서도 명시되었듯이, 현재의 스캐닝 방향과 클러스터 중심 개수($t=4, 25$) 설정은 **고정된 하이퍼파라미터**이다. 이는 다양한 해상도와 크기의 병변이 존재하는 실제 의료 환경에서 모든 케이스에 최적일 수 없다. 실제로 ISIC18 데이터셋의 일부 복잡한 구조에서 분할 실패 사례(failure examples)가 관찰되었으며, 이는 고정된 설정이 복잡한 경계 묘사에 한계가 있음을 시사한다.

### 향후 발전 방향

저자들은 입력 데이터의 특성에 따라 스캐닝 방향과 클러스터링 설정을 실시간으로 조정하는 **적응형 설정(Adaptive Configuration)** 알고리즘의 필요성을 언급하였다. 또한, Mamba 모델을 분할뿐만 아니라 등록(registration)이나 재구성(reconstruction) 작업으로 확장할 가능성을 제시하였다.

## 📌 TL;DR

CCViM은 Mamba의 선형 복잡도 효율성을 유지하면서, **Context Clustering(CC)**이라는 동적 그룹화 메커니즘을 통해 Mamba 모델의 고질적인 약점인 '지역적 특징 손실' 문제를 해결한 의료 영상 분할 모델이다. 전역 스캔과 지역 클러스터링을 결합한 **CCS6 레이어**를 통해 핵, 피부 병변, 복부 장기 등 다양한 의료 영상 작업에서 기존 SOTA 모델들을 능가하는 정밀도와 효율성을 입증하였다. 향후 적응형 구성 전략이 도입된다면 의료 AI 분야에서 더욱 강력한 범용 모델이 될 가능성이 높다.
