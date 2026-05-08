# Learning 3D Semantics from Pose-Noisy 2D Images with Hierarchical Full Attention Network

Yuhang He, Lin Chen, Junkun Xie, and Long Chen (2022)

## 🧩 Problem to Solve

본 논문은 2D 다중 뷰(multi-view) 이미지 관측을 통해 3D point cloud의 시맨틱 세그멘테이션(semantic segmentation)을 수행하는 프레임워크를 제안한다.

기존의 3D point cloud 직접 학습 방식은 다음과 같은 네 가지 주요 문제로 인해 어려움이 있다. 첫째, 데이터의 양이 매우 방대하여 계산 비용이 높다. 둘째, 비정형적이고 순서가 없는(unstructured and unordered) 특성으로 인해 텍스처나 토폴로지 정보를 활용하기 어렵다. 셋째, 특정 클래스가 공간의 대부분을 차지하는 데이터 불균형(data imbalance) 문제가 심각하다. 넷째, 거리별 샘플링 밀도가 불균일하다.

이를 해결하기 위해 저자들은 3D 세그멘테이션 문제를 2D 이미지 세그멘테이션으로 전이하는 "task transfer" 패러다임을 제안한다. 그러나 이 과정에서 다음과 같은 현실적인 제약 사항이 발생한다.

- **Pose Noise**: LiDAR와 카메라 간의 캘리브레이션 오차로 인해 3D 점을 2D 픽셀로 투영할 때 정확한 대응 관계를 보장할 수 없다.
- **View-angle**: 시점의 변화로 인해 관측 대상이 왜곡되거나 가려지는(occlusion) 현상이 발생한다.
- **Void Projection**: LiDAR의 360도 시야각과 카메라의 좁은 화각 차이로 인해 일부 3D 점이 이미지 상에 투영되지 않는 문제가 발생한다.

따라서 본 논문의 목표는 이러한 pose noise와 관측의 불완전성을 극복하면서 2D의 풍부한 시맨틱 단서를 3D 공간으로 효과적으로 전이하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 3D 점 하나를 단일 픽셀이 아닌 **다중 뷰 패치(multi-view patch)** 단위로 인식하고, 이를 **계층적 풀 어텐션 네트워크(Hierarchical Full Attention Network, HiFANet)**를 통해 단계적으로 통합하는 것이다.

1. **Task Transfer 전략**: 3D point cloud를 직접 처리하는 대신, 기학습된 2D 시맨틱 세그멘테이션 네트워크의 특징(feature)을 활용하여 3D 시맨틱을 예측한다.
2. **Pose Noise 완화**: 단일 픽셀 투영 대신 $k \times k$ 크기의 패치(patch) 관측을 도입하여, 투영 오차가 발생하더라도 해당 범위 내에서 정답 정보를 찾을 수 있도록 하여 강건성을 높였다.
3. **HiFANet 설계**: 패치 $\rightarrow$ 개별 이미지 인스턴스 $\rightarrow$ 3D 점 간 상호작용으로 이어지는 3단계 계층적 어텐션 구조를 설계하여, 데이터 크기를 효율적으로 줄이면서 핵심 시맨틱 정보를 추출한다.

## 📎 Related Works

3D 시맨틱 세그멘테이션 연구는 크게 세 가지 범주로 나뉜다.

- **Point-based methods**: MLP(PointNet, PointNet++), Point Convolution(RandLA-Net), Graph Convolution 등을 사용하여 점 단위로 특징을 추출한다.
- **Voxel-based methods**: 3D 공간을 그리드로 나누어 3D CNN을 적용하지만, 씬(scene)의 규모가 커질수록 계산 비용이 급격히 증가하는 한계가 있다.
- **Projection-based methods**: 3D 점을 2D 평면으로 투영하여 2D CNN을 적용한다. 기존 방식은 합성 뷰(synthetic view)를 생성하여 사용하지만, 3D 점의 희소한 샘플링으로 인한 오해석 문제가 있다.

본 연구는 기존 투영 기반 방식과 유사하나, 합성 뷰가 아닌 **실제 RGB 이미지**와 **기학습된 2D 시맨틱 세그멘테이션 모델**의 결과를 활용한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. 전체 파이프라인

시스템은 3D 점들을 인접한 $N$개의 RGB 이미지 프레임에 투영하여 'bag-of-frames' 관측 세트를 생성한다. 각 투영 지점을 중심으로 $k \times k$ 크기의 패치를 추출하며, 각 패치 내의 픽셀들은 2D 모델로부터 얻은 카테고리 라벨 $s$와 시맨틱 특징 표현 $r$을 가진다. 이 정보들이 HiFANet의 입력으로 들어가 최종 3D 라벨을 결정한다.

### 2. HiFANet의 3단계 구조

#### Stage 1: Patch Attention (패치 내 통합)

각 이미지에서 추출된 $k \times k$ 패치 정보를 하나의 인스턴스 특징으로 압축한다.

- **작동 원리**: 패치의 중심점(principle point)을 기준으로 주변 픽셀들에 대한 어텐션 가중치를 학습한다.
- **수식**:
  $$f_{pa} = \sum_{j=1}^{k \times k} w_j \cdot V_j + f_p$$
  여기서 $f_p$는 중심점의 특징이며, $w_j$는 Scaled Dot-Product Attention을 통해 계산된 가중치이다.
  $$w = \text{softmax}\left(\frac{Q_p K^T}{\sqrt{d_1}}\right)$$
  ($Q_p$는 중심점의 쿼리, $K$는 패치 내 픽셀들의 키 값이다.)

#### Stage 2: Instance Attention (다중 뷰 통합)

한 3D 점에 대해 $N$개의 이미지에서 얻은 인스턴스 특징들을 통합하여 점 단위(point-wise) 특징을 생성한다.

- **작동 원리**: 이미지 관측 순서에 상관없이 동일한 결과가 나와야 하므로(permutation invariant), 셀프 어텐션 레이어를 적용한 후 **평균 풀링(average pooling)**을 통해 하나로 병합한다.

#### Stage 3: Inter-Point Attention (3D 점 간 상호작용)

주변 $M$개 3D 점들 사이의 공간적 구조와 특징 상호작용을 고려한다.

- **작동 원리**: Transformer의 Multi-head Self-Attention 구조를 사용한다. 특히, 두 점 사이의 상대적 Cartesian 좌표 차이를 인코딩한 **구조적 사전 정보(structural prior, $K^{pe}$)**를 Key 값에 더해준다.
- **수식**:
  $$K = K + K^{pe}$$
  이를 통해 공간적으로 가까운 점들이 유사한 라벨을 가질 가능성이 높다는 기하학적 특성을 학습에 반영한다.

## 📊 Results

### 실험 설정

- **데이터셋**: Semantic-KITTI
- **평가 지표**: mIoU (mean Intersection over Union), Average Accuracy
- **비교 대상**:
  - Point-based (PointNet, PointNet++, RandLA-Net 등)
  - 단순 이미지 집계 방식 (Voting 기반)
  - HiFANet 변형 모델 (Patch Attention 제거, Structural Prior 제거)

### 주요 결과

- **정량적 성능**: HiFANet은 **mIoU 0.620**을 달성하여, 비교 대상인 RandLA-Net(0.578) 및 기타 Point-based 방법들을 유의미하게 앞섰다.
- **데이터 효율성**: Point-based 방법들은 방대한 양의 3D 학습 데이터가 필요하지만, HiFANet은 2D 이미지의 시맨틱 정보를 활용하므로 훨씬 적은 양의 학습 데이터로도 높은 성능을 낸다.
- **Pose Noise 강건성**: 가우시안 포즈 노이즈를 추가한 실험에서, 단일 픽셀 투영 방식(patch size=1)은 성능이 급격히 하락하는 반면, 패치 기반의 HiFANet은 성능 하락 폭이 훨씬 적어 노이즈에 대한 내성이 있음을 증명하였다.

## 🧠 Insights & Discussion

### 강점 및 분석

- **2D 지식의 효율적 전이**: 3D 데이터의 희소성과 불균형 문제를 2D 이미지의 조밀한 텍스처와 성숙한 2D 세그멘테이션 네트워크를 통해 해결하였다.
- **구조적 보완**: 단순히 2D 정보를 가져오는 것에 그치지 않고, 3D 점들 간의 상대적 위치 관계(Structural Prior)를 어텐션 메커니즘에 통합함으로써 2D 정보만으로는 구분하기 어려운 객체(예: 전신주와 나무 지지대)를 정확히 구분해 낸다.

### 한계 및 향후 과제

- **Pose 의존성**: 포즈 노이즈에 강건해졌다고는 하나, 기본적으로 LiDAR-카메라 투영에 기반하므로 포즈 정확도가 극도로 낮아지면 성능이 함께 저하된다.
- **단방향 전이**: 현재는 $2\text{D} \rightarrow 3\text{D}$ 방향의 전이만 수행한다. 2D와 3D 정보를 동시에 학습하는 Joint Learning 방식을 도입한다면 추가적인 성능 향상이 가능할 것으로 보인다.

## 📌 TL;DR

본 논문은 포즈 오차가 존재하는 2D 이미지들을 활용하여 3D point cloud의 시맨틱을 학습하는 **HiFANet**을 제안한다. 다중 뷰 패치 관측과 3단계 계층적 어텐션(Patch $\rightarrow$ Instance $\rightarrow$ Inter-point) 구조를 통해 **포즈 노이즈에 강건한 3D 세그멘테이션**을 구현하였다. 특히 방대한 3D 학습 데이터 없이도 2D 시맨틱 지식을 효과적으로 전이하여 기존 3D 전용 모델보다 우수한 성능을 보였으며, 이는 실전 자율주행 센서 환경에서 매우 실용적인 접근 방식이다.
