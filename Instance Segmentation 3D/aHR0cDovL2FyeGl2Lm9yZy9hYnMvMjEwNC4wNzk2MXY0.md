# ADVANCED DEEP NETWORKS FOR 3D MITOCHONDRIA INSTANCE SEGMENTATION

Mingxing Li, Chang Chen, Xiaoyu Liu, Wei Huang, Yueyi Zhang, Zhiwei Xiong (2021)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 문제는 전자 현미경(Electron Microscopy, EM) 이미지에서 3차원 미토콘드리아(Mitochondria)의 인스턴스 분할(Instance Segmentation)을 자동화하는 것이다.

미토콘드리아는 세포에 에너지를 공급하는 중요한 세포 소기관으로, 생명 과학 연구에 매우 높은 가치를 지닌다. 그러나 EM 이미지는 테라바이트(Terabyte) 단위의 거대한 용량을 차지하므로 사람이 직접 인스턴스 분할을 수행하는 것은 불가능에 가깝다. 따라서 고성능의 자동 분할 알고리즘이 필수적이다.

기존의 전통적인 방법들은 일반화 능력이 부족하여 대규모 데이터셋에 적용하기 어려웠으며, 최신 딥러닝 기반의 Top-down 방식(예: Mask-RCNN)은 미토콘드리아 특유의 길쭉하고 왜곡된 형태 때문에 적절한 Anchor 크기를 설정하기 어렵다는 한계가 있었다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 EM 이미지의 물리적 특성(비등방성 해상도 및 노이즈)을 고려한 네트워크 설계와 학습 전략을 도입하는 것이다. 주요 기여 사항은 다음과 같다.

1. **Anisotropic Convolution Block (ACB) 설계**: EM 데이터셋의 비등방성(Anisotropic) 해상도 특성을 반영하여, 수용 영역(Receptive Field)을 효과적으로 확장하는 전용 블록을 제안하였다.
2. **Res-UNet-R 및 Res-UNet-H 제안**: 샘플의 특성에 따라 네트워크 구조를 차별화하였다. 특히 노이즈가 더 많은 인간(Human) 샘플을 위해 시맨틱 마스크와 인스턴스 경계선을 각각 예측하는 듀얼 디코더 경로를 가진 Res-UNet-H를 설계하였다.
3. **Multi-scale Training (MT) 전략**: 입력 이미지의 크기를 단계적으로 변경하며 학습하는 전략을 통해 모델의 과적합을 방지하고 성능을 높였다.
4. **Denoising 전처리 도입**: 이미지 복원(Restoration)을 위한 Interpolation Network를 전처리에 도입하여, 테스트 셋의 노이즈 영향을 줄이고 모델의 일반화 성능을 향상시켰다.

## 📎 Related Works

논문에서는 기존 연구를 다음과 같이 구분하여 설명한다.

- **전통적 방법**: Supervoxel 기반 방법이나 대수 곡선(Algebraic curves) 및 Random Forest 분류기를 사용한 방법들이 있었으나, 대규모 데이터셋(MitoEM 등)에 대한 일반화 능력이 떨어진다.
- **CNN 기반 방법**:
  - **Top-down 방식**: Mask-RCNN 등이 대표적이지만, 미토콘드리아의 불규칙하고 길쭉한 형태 때문에 Anchor 기반의 검출이 어렵다.
  - **Bottom-up 방식**: 이진 마스크, Affinity map, 또는 경계선(Boundary)을 먼저 예측한 후 후처리를 통해 인스턴스를 구분하는 방식이다. 본 논문은 이 Bottom-up 방식을 채택하여 개선하였다.

## 🛠️ Methodology

### 1. 네트워크 구조 및 ACB

본 논문은 Bottom-up 접근 방식을 사용하며, 최종적으로 시맨틱 마스크(Semantic mask)와 인스턴스 경계선(Instance boundary)을 출력한다.

- **Anisotropic Convolution Block (ACB)**: $1 \times 3 \times 3$ 컨볼루션 층 하나와, skip connection이 포함된 두 개의 $3 \times 3 \times 3$ 컨볼루션 층을 직렬로 연결하여 비등방성 정보를 효과적으로 추출한다.
- **Res-UNet-R vs Res-UNet-H**:
  - **Res-UNet-R (Rat)**: 디코더가 하나의 경로로 구성되어 마스크와 경계선을 동시에 생성한다.
  - **Res-UNet-H (Human)**: 이미지 품질이 더 낮기 때문에, 시맨틱 마스크와 인스턴스 경계선을 각각 예측하는 두 개의 독립적인 디코더 경로를 가진다.

### 2. 손실 함수 (Loss Function)

클래스 불균형 문제를 해결하기 위해 가중치 기반의 이진 교차 엔트로피(Weighted Binary Cross Entropy, WBCE) 손실 함수를 사용한다.

개별 블록의 손실 함수는 다음과 같다.
$$L_{WBCE}(X_i, Y_i) = \frac{1}{DHW} \sum W_i L_{BCE}(X_i, Y_i)$$

여기서 $D, H, W$는 블록의 깊이, 높이, 너비이며, 가중치 $W_i$는 전경 픽셀 비율 $W_f$에 따라 다음과 같이 정의된다.
$$W_i = \begin{cases} Y_i + \frac{W_f}{1-W_f}(1-Y_i) & \text{if } W_f > 0.5 \\ (1-W_f)Y_i + W_f(1-Y_i) & \text{else} \end{cases}$$
($W_f = \text{sum}(Y_i) / DHW$)

전체 손실 함수 $L$은 시맨틱 마스크($X_M$)와 인스턴스 경계선($X_B$)의 손실 합으로 계산된다.
$$L = L_{WBCE}(X_M, Y_M) + L_{WBCE}(X_B, Y_B)$$

### 3. 후처리 및 전처리

- **Seed Map 생성**: 예측된 마스크 $X_M$과 경계선 $X_B$를 이용하여 시드 맵 $S_j$를 생성한다.
    $$S_j = \begin{cases} 1 & \text{if } X_j^M > T_1 \text{ and } X_j^B < T_2 \\ 0 & \text{else} \end{cases}$$
    이때 임계값은 $T_1 = 0.9, T_2 = 0.8$로 설정하며, 이후 연결 성분 레이블링(Connected Component Labeling, CCL)을 통해 최종 인스턴스를 분리한다.
- **Denoising 전처리**: 인접한 두 프레임을 입력으로 하여 노이즈 프레임을 복원하는 Interpolation Network를 사용하여 테스트 데이터의 품질을 높였다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: MitoEM (Rat, Human 샘플), Lucchi (시맨틱 분할용).
- **평가 지표**: MitoEM의 경우 3D AP-75(Intersection over Union $\ge 0.75$ 기준)를 사용하였고, Lucchi의 경우 Jaccard index와 Dice Similarity Coefficient (DSC)를 사용하였다.

### 2. 정량적 결과

- **인스턴스 분할 (MitoEM)**: 제안 방법이 AP-75 지표에서 Rat 샘플 0.917, Human 샘플 0.828을 기록하며 기존의 Wei [4], Nightingale [13], Li [14] 등의 방법들을 큰 차이로 앞섰다.
- **시맨틱 분할 (Lucchi)**: Res-UNet-R이 Jaccard 0.895, DSC 0.945를 기록하여 기존의 Casser [18] 등 경쟁 모델보다 우수한 성능을 보였다.
- **챌린지 결과**: ISBI 2021의 Large-scale 3D Mitochondria Instance Segmentation Challenge에서 1위를 차지하였다.

### 3. 어블레이션 연구 (Ablation Study)

- **블록 유닛**: 3D SE, 3D ECA, 3D Res 블록보다 제안한 3D ACB가 과적합을 줄이고 가장 높은 성능을 보였다.
- **네트워크 구조**: Human 샘플의 경우, 단일 디코더(Res-UNet-R)보다 듀얼 디코더(Res-UNet-H)를 사용했을 때 AP-75가 0.783에서 0.816으로 상승하였다.
- **학습 전략**: Multi-scale Training (MT) 적용 시 특히 Human 샘플에서 성능이 1.2% 향상되었다.

## 🧠 Insights & Discussion

본 논문은 단순히 모델의 깊이를 더하는 것이 아니라, 데이터의 물리적 특성(Anisotropy)과 노이즈 수준을 분석하여 이에 맞춤화된 구조(ACB, Dual-decoder)를 설계했다는 점에서 강점이 있다.

특히, 복잡한 Attention 모듈(SE, ECA)보다 단순하게 설계된 ACB가 더 좋은 성능을 낸 점은, 의료 영상 분할 작업에서 무조건적인 복잡성보다는 데이터의 기하학적 특성을 반영한 단순한 구조가 과적합 방지에 더 유리할 수 있음을 시사한다.

다만, 전처리에 사용된 Interpolation Network가 구체적으로 어떻게 학습되었는지에 대한 상세 설명이 부족하며, 테스트 셋에서만 사용되었다는 점이 명시되어 있어 학습 단계에서의 통합 가능성에 대해서는 추가 논의가 필요해 보인다.

## 📌 TL;DR

본 논문은 3D 전자 현미경 이미지의 미토콘드리아 인스턴스 분할을 위해 **비등방성 컨볼루션 블록(ACB)**과 **멀티스케일 학습 전략**, 그리고 **디노이징 전처리**를 도입한 **Res-UNet-R/H** 모델을 제안하였다. 이 방법은 데이터의 물리적 특성을 반영한 설계를 통해 ISBI 2021 챌린지에서 1위를 기록하였으며, 특히 노이즈가 많은 샘플에 대해 최적화된 구조를 제시함으로써 실질적인 분할 성능을 크게 향상시켰다.
