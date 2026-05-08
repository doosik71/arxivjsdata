# JSNet: Joint Instance and Semantic Segmentation of 3D Point Clouds

Lin Zhao, Wenbing Tao (2020)

## 🧩 Problem to Solve

본 논문은 3D 포인트 클라우드(Point Cloud) 데이터에서 **Semantic Segmentation(의미론적 분할)**과 **Instance Segmentation(인스턴스 분할)**을 동시에 해결하는 것을 목표로 한다.

- **해결하고자 하는 문제**: 기존의 3D 분할 연구들은 두 작업을 개별적으로 처리하거나, 인스턴스 분할을 의미론적 분할의 후처리 단계로 취급하는 경향이 있었다. 또한, 두 작업을 동시에 수행하려는 시도(예: ASIS)가 있었으나, k-Nearest Neighbor(kNN)과 같은 연산으로 인해 계산 복잡도가 높고 메모리 소비가 크며, 적절한 하이퍼파라미터($K$ 값 등)를 설정하기 어렵다는 한계가 있었다.
- **문제의 중요성**: 의미론적 분할은 모든 영역을 클래스별로 분류하는 것이고, 인스턴스 분할은 동일 클래스 내에서도 서로 다른 객체를 구분하는 것이다. 이 두 작업은 "다른 카테고리의 점들은 서로 다른 인스턴스에 속하며, 동일한 인스턴스의 점들은 동일한 클래스에 속한다"는 상호 보완적인 관계를 가진다.
- **논문의 목표**: 두 작업이 서로의 성능을 향상시킬 수 있도록 상호 촉진(Mutual Promotion)하는 구조의 통합 네트워크인 **JSNet**을 제안하여, 정확도를 높이면서도 연산 효율성을 확보하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **의미론적 특징(Semantic Features)과 인스턴스 특징(Instance Features) 간의 상호 교환 및 융합**을 통해 각 작업의 예측 정확도를 높이는 것이다.

1. **Point Cloud Feature Fusion (PCFF) 모듈**: 디코더의 서로 다른 계층(Layer)에서 나오는 고수준의 의미 정보와 저수준의 세부 정보를 융합하여 더 변별력 있는 특징을 추출한다.
2. **Joint Instance and Semantic Segmentation (JISS) 모듈**: 의미론적 특징을 인스턴스 임베딩 공간으로 변환하여 인스턴스 분할을 돕고, 반대로 인스턴스 특징을 의미론적 특징 공간으로 집계하여 의미론적 분할 성능을 높이는 상호 촉진 구조를 설계하였다.
3. **효율적인 백본 네트워크**: PointNet++의 집합 추상화(Set Abstraction) 모듈과 PointConv의 특징 인코딩 레이어를 결합하여, 메모리 소비는 억제하면서도 로컬 특징 추출 능력을 극대화한 하이브리드 백본을 구축하였다.

## 📎 Related Works

### 기존 연구 및 한계

- **3D 특징 추출**: PointNet은 전역 특징 추출에 강하나 로컬 특징 캡처 능력이 부족하며, PointNet++는 이를 계층적 구조로 해결했다. PointConv는 연속 필터를 통해 더 정교한 특징을 추출하지만 GPU 메모리 소비가 크다는 단점이 있다.
- **의미론적 분할**: 3D-FCNN, SEGCloud, RSNet 등이 제안되었으나, 대부분 인스턴스 임베딩의 이점을 활용하지 못했다.
- **인스턴스 분할**: SGPN, GSPN, 3D-BoNet 등이 제안되었으며, 주로 유사도 행렬이나 바운딩 박스 회귀 방식을 사용한다.
- **통합 접근 방식**: MT-PNet은 멀티태스크 네트워크와 CRF(Conditional Random Field)를 사용했으나 CRF가 후처리 단계로 분리되어 있어 최적화가 어렵다. ASIS는 두 작업을 동시에 수행하지만 kNN 연산으로 인한 메모리 비용과 계산량 문제가 심각하다.

### 차별점

JSNet은 kNN과 같은 고비용 연산 대신 **1D Convolution과 가중치 맵(Weight Map) 기반의 융합 방식**을 사용하여 메모리 효율성을 높였으며, 두 태스크가 서로의 특징 공간을 참조하도록 설계하여 상호 보완적인 학습이 가능하게 했다.

## 🛠️ Methodology

### 전체 파이프라인 구조

JSNet은 **공유 인코더(Shared Encoder) $\rightarrow$ 병렬 디코더(Parallel Decoders) $\rightarrow$ PCFF 모듈 $\rightarrow$ JISS 모듈** 순으로 구성된다.

- **Shared Encoder**: PointNet++의 Set Abstraction과 PointConv의 인코딩 레이어를 결합하여 특징을 추출한다.
- **Parallel Decoders**: 하나는 의미론적 특징($F_{SS}$)을, 다른 하나는 인스턴스 특징($F_{IS}$)을 추출하는 두 개의 가지(Branch)로 나뉜다.
- **PCFF**: 각 디코더의 마지막 3개 레이어 특징을 보간(Interpolation) 및 융합하여 특징을 정교화한다.
- **JISS**: 최종적으로 두 가지의 특징을 서로 교환하여 최종 예측 결과($P_{SSI}$, $E_{ISS}$)를 생성한다.

### 훈련 목표 및 손실 함수

전체 손실 함수 $L$은 의미론적 분할 손실($L_{sem}$)과 인스턴스 임베딩 손실($L_{ins}$)의 합으로 정의된다.
$$L = L_{sem} + L_{ins}$$

1. **Semantic Loss ($L_{sem}$)**: 일반적인 Cross Entropy Loss를 사용하여 각 포인트의 클래스를 분류한다.
2. **Instance Embedding Loss ($L_{ins}$)**: 포인트들을 임베딩 공간에 배치하여 동일 인스턴스는 가깝게, 다른 인스턴스는 멀게 만드는 판별 함수(Discriminative Function)를 사용한다.
   - **Pull Loss ($L_{pull}$)**: 각 포인트의 임베딩 $e_n$을 해당 인스턴스의 평균 임베딩 $\mu_m$으로 끌어당긴다.
     $$L_{pull} = \frac{1}{M} \sum_{m=1}^{M} \frac{1}{N_m} \sum_{n=1}^{N_m} [\|\mu_m - e_n\|_1 - \delta_v]_+^2$$
   - **Push Loss ($L_{push}$)**: 서로 다른 인스턴스의 평균 임베딩 $\mu_i$와 $\mu_j$를 일정 거리 이상으로 밀어낸다.
     $$L_{push} = \frac{1}{M(M-1)} \sum_{i=1}^{M} \sum_{j=1, j \neq i}^{M} [2\delta_d - \|\mu_i - \mu_j\|_1]_+^2$$
   - 여기서 $[x]_+ = \max(0, x)$이며, $\delta_v$와 $\delta_d$는 각각 Pull과 Push의 마진 값이다.

### JISS 모듈의 세부 절차

1. **Instance Branch (Semantic $\rightarrow$ Instance)**:
   - 의미론적 특징 $F_{SS}$를 1D Conv를 통해 인스턴스 공간($F_{SST}$)으로 변환하고 이를 $F_{IS}$와 융합한다.
   - 융합된 특징 $F_{ISSC}$에 대해 Mean과 Sigmoid 연산을 적용하여 가중치 맵 $F_{ISR}$을 생성하고, 이를 다시 곱해 중요 특징을 강조한 후 최종 임베딩 $E_{ISS}$를 출력한다.
   - 수식: $F_{ISSC} = \text{Concat}(F_{IS}, F_{IS} + \text{Conv1D}(F_{SS})) \rightarrow F_{ISSR} = F_{ISSC} \cdot \text{Sigmoid}(\text{Mean}(F_{ISSC})) \rightarrow E_{ISS} = \text{Conv1D}(\text{Conv1D}(F_{ISSR}))$

2. **Semantic Branch (Instance $\rightarrow$ Semantic)**:
   - 인스턴스 특징 $F_{ISSR}$을 1D Conv, Mean, Tile 연산을 통해 의미론적 공간($F_{ISST}$)으로 변환한다.
   - 이를 $F_{SS}$와 융합하고, 위와 동일하게 가중치 맵을 생성하여 최종 의미론적 특징 $P_{SSI}$를 출력한다.

### 추론 절차

- **Semantic Segmentation**: $P_{SSI}$에 $\text{argmax}$ 연산을 적용하여 클래스를 결정한다.
- **Instance Segmentation**: 생성된 임베딩 $E_{ISS}$에 대해 **Mean-shift Clustering**을 수행하여 인스턴스 레이블을 생성한다.

## 📊 Results

### 실험 설정

- **데이터셋**: S3DIS(실내 3D 포인트 클라우드), ShapeNet(CAD 모델 파트 분할).
- **지표**:
  - Semantic: Overall Accuracy(oAcc), Mean Accuracy(mAcc), Mean IoU(mIoU).
  - Instance: Mean Precision(mPrec), Mean Recall(mRec), Coverage(Cov), Weighted Coverage(WCov).

### 주요 결과

1. **S3DIS 인스턴스 분할**: JSNet은 ASIS, SGPN 등 기존 최신 기법보다 우수한 성능을 보였다. 특히 Area 5(다른 건물 데이터)에서 mCov 4.1, mPrec 6.8 등의 큰 폭의 향상을 기록하며 일반화 능력을 입증하였다.
2. **S3DIS 의미론적 분할**: baseline인 PointNet 대비 mIoU가 6-fold 교차 검증에서 크게 상승하였으며, ASIS 및 3P-RNN보다 높은 성능을 보였다.
3. **ShapeNet 파트 분할**: mIoU 기준 PointNet++(84.9) 및 ASIS(85.0)보다 높은 85.8을 달성하여 파트 분할 작업에서도 유효함을 확인하였다.

### Ablation Study (S3DIS Area 5)

- **백본 영향**: Base Network(PointNet++)보다 PointConv가 결합된 Backbone Network의 성능이 더 높았다.
- **PCFF 영향**: 특징 융합을 적용했을 때 분할 정밀도가 상승함을 확인했다.
- **상호 촉진(JISS) 영향**: 인스턴스 융합(IF)만 사용하거나 의미론적 인식(SF)만 사용했을 때보다, 두 가지를 모두 적용했을 때 성능 향상 폭이 가장 컸다. 이는 두 태스크가 서로 긍정적인 영향을 주고받음을 시사한다.
- **학습 전략**: Early Stopping과 Random Sampling을 적용했을 때 오버피팅이 줄어들고 일반화 성능이 향상되어, 가장 높은 성능(mPrec 62.9, mIoU 55.0)을 얻었다.

## 🧠 Insights & Discussion

### 강점

- **상호 보완적 학습**: 단순히 두 태스크를 병렬로 수행하는 것이 아니라, 서로의 특징 공간을 변환하여 융합하는 JISS 모듈을 통해 "의미론적 정보가 인스턴스 구분을 돕고, 인스턴스 정보가 클래스 분류를 돕는" 선순환 구조를 만들었다.
- **연산 효율성**: ASIS가 사용한 고비용의 kNN 연산을 1D Convolution 기반의 가중치 맵 방식으로 대체함으로써, 메모리 사용량을 줄이면서도 성능은 오히려 향상시켰다.

### 한계 및 논의사항

- **클러스터링 의존성**: 최종 인스턴스 분할을 위해 Mean-shift Clustering이라는 외부 알고리즘에 의존하고 있다. 이는 하이퍼파라미터(bandwidth) 설정에 영향을 받을 수 있다.
- **기하학적 정보 활용**: 저자들은 결론에서 향후 포인트 클라우드의 공간적 기하 구조(Spatial Geometric Topology)를 프레임워크에 추가한다면 더 좋은 결과를 얻을 수 있을 것이라고 언급하였다.

## 📌 TL;DR

JSNet은 3D 포인트 클라우드에서 **의미론적 분할과 인스턴스 분할을 동시에 수행**하는 딥러닝 네트워크이다. 핵심 기여는 하이브리드 백본과 PCFF 모듈을 통해 정교한 특징을 추출하고, **JISS 모듈을 통해 두 태스크의 특징을 상호 융합하여 서로의 성능을 끌어올린 것**이다. S3DIS와 ShapeNet 데이터셋에서 기존 SOTA 모델들을 상회하는 성능을 보였으며, 특히 메모리 효율적인 구조로 설계되어 실용성이 높다. 향후 3D 장면 이해를 위한 통합 분할 프레임워크의 기초 연구로 가치가 높다.
