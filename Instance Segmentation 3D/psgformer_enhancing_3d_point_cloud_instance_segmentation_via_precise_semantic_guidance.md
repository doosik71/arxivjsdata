# PSGformer: Enhancing 3D Point Cloud Instance Segmentation via Precise Semantic Guidance

Lei Pan, Wuyang Luan, Yuan Zheng, Qiang Fu, and Junhui Li (2023)

## 🧩 Problem to Solve

본 논문은 3D 포인트 클라우드(point cloud) 데이터에서의 인스턴스 세그멘테이션(instance segmentation) 문제를 해결하고자 한다. 3D 인스턴스 세그멘테이션은 단순히 각 포인트의 카테고리를 분류하는 세만틱 세그멘테이션을 넘어, 동일한 클래스에 속하는 서로 다른 객체(entity)들을 개별적으로 구분해내야 하는 고난도 작업이다.

기존의 접근 방식은 크게 두 가지로 나뉘는데, 각각 뚜렷한 한계점을 가지고 있다.

1. **Proposal-based 방식**: 3D 바운딩 박스를 먼저 예측한 후 내부의 인스턴스를 정제하는 Top-down 방식이다. 그러나 3D 공간에서 바운딩 박스는 자유도가 높아 피팅이 어렵고, 포인트 클라우드가 객체 표면의 일부만 표현하는 경우가 많아 기하학적 중심을 찾기 어렵다는 단점이 있다.
2. **Grouping-based 방식**: 포인트별 세만틱 라벨과 중심점 오프셋(offset)을 학습하여 그룹화하는 Bottom-up 방식이다. 최근 많은 진전이 있었으나, 세만틱 세그멘테이션 결과에 지나치게 의존하기 때문에 초기 세만틱 예측이 틀릴 경우 전체 인스턴스 분리 성능이 급격히 저하되는 문제가 있다.

따라서 본 논문의 목표는 전역(global) 및 지역(local) 세만틱 정보를 효과적으로 통합하여, 기존 방식들의 한계를 극복하고 보다 정밀한 인스턴스 세그멘테이션을 수행하는 **PSGformer** 네트워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 지역적인 세부 특징과 전역적인 장면 특징을 독립적으로 추출한 뒤, 이를 Transformer 구조를 통해 병렬적으로 융합하여 세밀한 가이드라인을 제공하는 것이다.

주요 기여 사항은 다음과 같다.

- **Multi-Level Semantic Aggregation (MSA) 모듈 제안**: Foreground 포인트 필터링과 다중 반경(multi-radius) 집계 방식을 통해 지역적인 장면 특징을 효과적으로 캡처한다. 이는 세만틱 세그멘테이션 결과에 대한 과도한 의존도를 낮추고 지역적 표현력을 높인다.
- **Parallel Feature Fusion Transformer 설계**: 슈퍼포인트(superpoint) 기반의 전역 특징과 MSA를 통해 얻은 지역 특징을 병렬적으로 처리하고 융합하는 구조를 제안한다. 이를 통해 정보 손실을 최소화하면서 전역-지역 문맥을 동시에 고려한 특징 표현을 가능하게 한다.
- **성능 입증**: ScanNetv2 데이터셋의 hidden test set에서 mAP 기준 기존 SOTA 대비 2.2% 향상된 성능을 달성하였으며, S3DIS 데이터셋에서도 경쟁력 있는 결과를 보여주었다.

## 📎 Related Works

논문에서는 기존 연구를 다음과 같이 세 가지 범주로 분류하여 설명한다.

1. **Proposal-based methods**: 3D-BoNet, 3D-SIS 등이 있으며, 2D의 Mask R-CNN 구조를 차용한다. 하지만 앞서 언급했듯 3D 바운딩 박스 예측의 정확도가 전체 성능을 좌우한다는 치명적인 한계가 있다.
2. **Grouping-based methods**: ASIS, SGPN, SoftGroup 등이 있으며, 특징 공간에서의 유사도를 이용해 포인트를 그룹화한다. 다만, 중간 집계 단계가 추가되어 연산 복잡도가 증가하고 세만틱 예측 오류에 취약하다.
3. **Instance Segmentation with Transformer**: DETR, MaskFormer 등이 2D에서 성공을 거두었으나, 3D 포인트 클라우드는 격자 구조(grid)가 아닌 비정형 데이터이므로 기존 Transformer를 그대로 적용하기 어렵다. 또한 3D 데이터의 고차원성과 볼륨 특성으로 인해 연산 효율성 문제가 발생한다.

PSGformer는 이러한 한계들을 극복하기 위해 Transformer의 장점인 장거리 의존성(long-range dependency) 모델링 능력을 활용하되, 3D 데이터 특성에 맞춘 전역/지역 특징 추출 및 융합 구조를 채택하여 차별점을 둔다.

## 🛠️ Methodology

### 전체 파이프라인

PSGformer는 크게 **Sparse 3D U-Net $\rightarrow$ 특징 추출 경로(Local/Global) $\rightarrow$ Parallel Feature Fusion Transformer $\rightarrow$ Prediction Head** 순으로 구성된다.

### 1. Feature Backbone

입력 포인트 클라우드 $P \in \mathbb{R}^{N \times 6}$에 대해 **Sparse 3D U-Net**을 사용하여 포인트별 특징 $F_p \in \mathbb{R}^{N \times C}$를 추출한다. 희소 합성곱(sparse convolution)을 통해 연산 효율성을 높이며, 스킵 연결(skip connection)을 통해 다중 스케일 정보를 유지한다.

### 2. Multi-level Semantic Aggregation (MSA)

지역 특징을 정밀하게 추출하기 위해 다음 단계를 거친다.

- **Foreground Point Filtering**: 포인트가 배경이 아닐 확률 $1-m^{(i)}(0) > \beta$를 기준으로 foreground 포인트를 필터링한다.
- **Iterative Sampling**: 이미 선택된 포인트나 배경 포인트를 제외하고 샘플링하여 모든 인스턴스를 최대한 커버하도록 한다.
- **Sphere Query**: 선택된 키포인트 $p_j$ 주변의 반경 $r$ 내에 있는 포인트 집합 $U$를 수집한다.
  $$\mathcal{U} = \{q^{(i)} \in \mathcal{U} | \text{distance}(q^{(i)}, p^{(j)}) < r\}$$
- **Feature Enrichment**: 수집된 특징을 다중 컨볼루션 블록에 통과시켜 고차원 지역 특징 $F_l \in \mathbb{R}^{M \times C}$를 생성한다.

### 3. Superpoint Pooling

전역 특징을 얻기 위해 미리 계산된 슈퍼포인트(superpoint) 구조를 활용한다. 포인트별 특징 $F_p$에 대해 슈퍼포인트 내 포인트들의 특징을 평균 풀링(average pooling)하여 $M$개의 전역 슈퍼포인트 특징 $F_s \in \mathbb{R}^{M \times D}$를 생성한다.

### 4. Parallel Feature Fusion Transformer

추출된 지역 특징 $F_l$과 전역 특징 $F_s$를 병렬로 입력받아 학습 가능한 쿼리 벡터(query vector) $Q$를 업데이트한다.

- **Superpoint Cross-Attention**: 쿼리 벡터가 슈퍼포인트 특징에 주의를 기울이도록 하며, 다음과 같은 수식으로 계산된다.
  $$\hat{Q}_l = \text{softmax}\left(\frac{QK^T}{\sqrt{D}} + A\right)V$$
  여기서 $A$는 **Superpoint Attention Mask**로, 이전 단계의 예측 결과 $M_{l-1}$이 임계값 $\tau=0.5$를 넘는 경우에만 주의를 기울이도록 제한하여 foreground 인스턴스 내에서만 attention이 일어나도록 강제한다.
  $$A_{l-1}(i, j) = \begin{cases} 0, & \text{if } M_{l-1}(i, j) \geq \tau \\ -\infty, & \text{otherwise} \end{cases}$$

### 5. Prediction Head 및 학습 절차

최종 쿼리 벡터 $Q$를 사용하여 세 가지를 예측한다.

1. **분류(Classification)**: 각 쿼리가 어떤 클래스인지 예측 ($p_i$).
2. **IoU 점수(Score)**: 예측 마스크와 GT 마스크 간의 IoU를 예측 ($s_i$).
3. **슈퍼포인트 마스크(Mask)**: 쿼리 벡터와 마스크 특징을 곱해 슈퍼포인트 마스크 $M_l$을 생성한다.

**손실 함수(Loss Function)**:
전체 손실은 다음 네 가지의 가중 합으로 구성된다.
$$\mathcal{L}_{total} = \lambda_{cls}\mathcal{L}_{cls} + \lambda_{score}\mathcal{L}_{score} + \lambda_{BCE}\mathcal{L}_{BCE} + \lambda_{Dice}\mathcal{L}_{Dice}$$

- $\mathcal{L}_{cls}$: 클래스 예측을 위한 Cross-Entropy 손실.
- $\mathcal{L}_{score}$: IoU 예측을 위한 MSE 손실.
- $\mathcal{L}_{BCE}, \mathcal{L}_{Dice}$: 마스크 정밀도를 위한 Binary Cross-Entropy 및 Dice 손실.

## 📊 Results

### 실험 설정

- **데이터셋**: ScanNetv2 (실내 장면), S3DIS (상업용 건물)
- **평가 지표**: mAP (mean Average Precision), $\text{AP}_{50}$, $\text{AP}_{25}$
- **비교 대상**: PointGroup, SoftGroup, DKNet, SSTNet 등 최신 SOTA 모델들

### 주요 결과

- **ScanNetv2 (Hidden Test Set)**: PSGformer는 **55.4% mAP**를 기록하며 기존 최고 성능 모델보다 **2.2% 향상**되었다. 특히 기존 모델들이 어려워하던 'counter' 카테고리에서 큰 성능 향상을 보였다.
- **S3DIS**: $\text{AP}_{50}$ 지표에서 새로운 SOTA를 달성하여 일반화 능력을 입증하였다.
- **추론 속도**: 단일 3080Ti GPU 기준 전체 실행 시간이 **281ms**로, 전통적인 클러스터링 기반 방식보다 훨씬 빠른 속도를 보여주었다.

### Ablation Study

- **특징 조합**: 지역 특징만 사용했을 때보다 전역 특징을 사용했을 때 성능이 좋았으며, **지역+전역 특징을 모두 사용하고 Transformer를 반복 적용(6회)했을 때 최적의 성능**이 나타났다.
- **하이퍼파라미터**: 400개의 쿼리 벡터와 1024개의 FPS 샘플링 수가 가장 좋은 mAP 성능을 보였다.
- **손실 함수**: Dice Loss가 마스크 최적화에 필수적이며, **Dice Loss + BCE Loss** 조합이 가장 효과적임이 확인되었다.

## 🧠 Insights & Discussion

**강점 및 분석**:
PSGformer의 성공 요인은 전역적 맥락과 지역적 세부 사항을 분리하여 추출한 후, 이를 Transformer의 cross-attention 메커니즘으로 정교하게 융합한 점에 있다. 특히, 예측된 마스크를 다시 attention mask로 사용하는 피드백 루프 구조가 foreground 영역에 집중하게 함으로써 정밀도를 높인 것으로 해석된다. 또한, 슈퍼포인트 기반의 풀링과 효율적인 MSA 모듈 덕분에 연산 속도를 크게 개선하면서도 성능을 높인 점이 인상적이다.

**한계 및 향후 과제**:
논문에서는 MSA 모듈과 Parallel Feature Fusion 모듈의 효율성을 더욱 증대시킬 필요가 있다고 언급한다. 또한, 실험이 ScanNetv2와 S3DIS라는 실내 데이터셋에 집중되어 있어, 실외 환경이나 더 거대한 규모의 포인트 클라우드 데이터셋에서도 동일한 성능 향상이 유지될지는 추가적인 검증이 필요하다.

## 📌 TL;DR

본 논문은 전역(Global) 및 지역(Local) 세만틱 정보를 정밀하게 통합하여 3D 인스턴스 세그멘테이션 성능을 높이는 **PSGformer**를 제안한다. **Multi-Level Semantic Aggregation**과 **Parallel Feature Fusion Transformer**를 통해 정보 손실 없이 특징을 융합하였으며, 그 결과 ScanNetv2 데이터셋에서 mAP 55.4%라는 SOTA 성능을 달성하였다. 이 연구는 기존의 grouping-based 방식이 가진 세만틱 의존성 문제를 해결하고, Transformer를 3D 포인트 클라우드 인스턴스 분리에 효율적으로 적용하는 새로운 방향을 제시하였다.
