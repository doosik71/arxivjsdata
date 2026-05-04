# 3D Bird’s-Eye-View Instance Segmentation

Cathrin Elich, Francis Engelmann, Theodora Kontogianni, and Bastian Leibe (2019)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 문제는 3D Point Cloud 데이터에서의 **Joint Semantic and Instance Segmentation**이다. 3D 장면 분석에서 개별 포인트의 클래스를 분류하는 Semantic Segmentation은 많은 진전이 있었으나, 동일 클래스 내에서 서로 다른 개체를 구분하는 Instance Segmentation은 상대적으로 덜 연구되었다.

특히 Proposal-free(제안 기반이 아닌) 방식의 Instance Segmentation을 수행하기 위해서는 전체 장면에 대해 **전역적으로 일관된 특징(Globally consistent features)**이 필요하다. 하지만 기존의 포인트 기반 방법론들은 메모리 및 연산 효율성을 위해 전체 장면을 작은 청크(Chunk) 단위로 나누어 독립적으로 처리한 후 휴리스틱하게 병합하는 방식을 사용한다. 이러한 접근법은 큰 크기의 객체가 여러 청크에 걸쳐 있을 경우 이를 하나의 인스턴스로 통합하기 어렵게 만든다는 치명적인 한계가 있다.

따라서 본 연구의 목표는 대규모 3D 포인트 클라우드에서 전역적인 문맥 정보를 유지하면서도 효율적으로 인스턴스를 구분할 수 있는 딥러닝 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **2D Bird's-Eye-View(BEV) 표현법과 3D Graph Neural Network를 결합한 하이브리드 아키텍처**를 사용하는 것이다.

전체 장면을 위에서 내려다보는 2D BEV 평면으로 투영하면, 전체 장면의 구조를 한눈에 파악할 수 있어 전역적으로 일관된 인스턴스 특징을 학습하기 용이하다. 이렇게 학습된 2D 특징을 다시 3D 공간의 포인트들에게 전파(Propagation)함으로써, 2D의 전역적 문맥 정보와 3D의 세밀한 기하학적 정보를 모두 활용하여 인스턴스 분할을 수행한다.

## 📎 Related Works

### 2D Instance Segmentation
FCN 및 U-Net과 같은 구조를 통해 픽셀 단위의 특징을 추출하고, 이를 클러스터링하여 인스턴스를 구분하는 연구들이 진행되었다. 특히 동일 인스턴스 간의 거리는 좁히고 서로 다른 인스턴스 간의 거리는 넓히는 Discriminative Loss 방식이 제안되었다.

### Deep Learning on 3D Point Clouds
PointNet과 PointNet++는 MLP와 Max-pooling을 통해 포인트 세트에서 특징을 추출하며, DGCNN(Dynamic Graph CNN)은 k-최근접 이웃(k-NN) 그래프를 통해 지역적 기하 구조를 학습하는 EdgeConv 연산을 도입하여 성능을 향상시켰다.

### 3D Instance Segmentation
기존의 SGPN(Similarity Group Proposal Network)은 포인트 클라우드를 블록 단위로 나누어 처리한 후 이를 병합하는 방식을 사용한다. 이는 전역적 일관성을 유지하기 위해 복잡한 휴리스틱 병합 알고리즘이 필요하며, 본 논문은 이러한 병합 과정 없이 전역 특징을 직접 학습함으로써 차별점을 둔다.

## 🛠️ Methodology

### 전체 시스템 구조
3D-BEVIS 프레임워크는 크게 세 단계로 구성된다: **2D Instance Feature Network $\rightarrow$ 3D Feature Propagation Network $\rightarrow$ Instance Grouping**.

### 1. 2D Instance Feature Network
먼저 3D 포인트 클라우드 $P$를 지면 평면으로 투영하여 BEV 이미지 $B \in \mathbb{R}^{H \times W \times C}$를 생성한다. 이때 각 그리드 셀에는 지면에서 가장 높은 포인트의 정보(색상, 높이 등)가 저장된다. 이 BEV 이미지는 U-Net 기반의 FCN을 통과하여 인스턴스 특징 맵 $E \in \mathbb{R}^{H \times W \times D}$와 시맨틱 라벨을 예측한다.

이 네트워크의 인스턴스 손실 함수 $L^{2D}_{inst}$는 다음과 같이 정의된다:
$$L^{2D}_{inst} = L_{var} + L_{dist}$$
여기서 $s_{i,j} = \|x_i - x_j\|^2$는 두 픽셀 특징 간의 거리이며, 각 항은 다음과 같다:
- $L_{var} = \sum_{c=1}^C \sum_{x_i, x_j \in S_c} [s_{i,j} - \delta_{var}]_+$ : 동일 인스턴스 내 포인트들의 특징 거리를 $\delta_{var}$ 이내로 좁힌다.
- $L_{dist} = \sum_{c, c'=1, c \neq c'}^C \sum_{x_i \in S_c, x_j \in S_{c'}} [\delta_{dist} - s_{i,j}]_+$ : 서로 다른 인스턴스 간의 특징 거리를 최소 $\delta_{dist}$ 이상으로 벌린다.

### 2. 3D Feature Propagation Network
BEV 투영 과정에서 가려진(occluded) 포인트들이 존재하므로, 2D에서 학습된 전역 특징 $E$를 3D 포인트 클라우드 전체로 전파해야 한다. 
- 각 포인트 $x_i$에 대해, BEV에서 대응되는 특징이 있다면 이를 결합(concatenate)하고, 없다면 0으로 설정하여 $P'$를 구성한다.
- 이후 DGCNN 아키텍처를 사용하여 이 특징들을 전파하고, 최종적인 포인트별 인스턴스 특징 $F^{inst}$와 시맨틱 라벨 $L$을 예측한다.
- 3D 인스턴스 손실 함수 $L^{3D}_{inst}$는 예측된 특징 $F^{inst}$와 타겟 특징 $F_{target}$(동일 인스턴스에 속한 BEV 특징들의 평균값) 사이의 MSE(Mean Squared Error)로 정의된다:
$$L^{3D}_{inst} = \|F^{inst} - F_{target}\|^2$$

### 3. Instance Grouping
최종적으로 예측된 $F^{inst}$를 **MeanShift** 클러스터링 알고리즘을 통해 그룹화하여 인스턴스 라벨 $I$를 생성한다. MeanShift는 클러스터의 개수를 미리 정할 필요가 없어 인스턴스 개수가 가변적인 3D 장면에 적합하다. 이후, 시맨틱 라벨이 일관되지 않은 인스턴스를 분리하는 후처리 과정을 거친다.

## 📊 Results

### 실험 설정
- **데이터셋**: S3DIS, ScanNet v2
- **비교 대상(Baselines)**: SGPN (PointNet 기반), SGPN (DGCNN 기반), PMRCNN (2D-3D 투영 방식)
- **평가 지표**: Instance Segmentation의 경우 Average Precision (AP 0.25, 0.5, 0.75), Semantic Segmentation의 경우 mIoU 및 Overall Accuracy를 사용한다.

### 주요 결과
1. **S3DIS 데이터셋**: 3D-BEVIS는 모든 overlap 임계값에서 SGPN보다 우수한 성능을 보였다. 특히 $\text{AP}_{0.5}$ 기준, SGPN(DGCNN)의 58.56%보다 높은 **65.66%**를 달성하였다.
2. **ScanNet v2 데이터셋**: 벤치마크 챌린지 결과, $\text{AP}_{0.5}$에서 22.5%를 기록하며 SGPN(14.33%) 대비 유의미한 성능 향상을 보였다.
3. **카테고리별 분석**: 벽(wall), 바닥(floor), 의자(chair) 등 대부분의 클래스에서 높은 성능을 보였으나, 천장(ceiling)의 경우 BEV 투영 시 제외되므로 성능이 낮게 측정되었다.

## 🧠 Insights & Discussion

### 강점
본 모델은 BEV라는 중간 표현법을 도입함으로써, 3D 포인트 클라우드 처리의 고질적인 문제인 '전역적 일관성 확보'를 효율적으로 해결하였다. 이를 통해 복잡한 휴리스틱 병합 과정 없이도 대규모 장면에서 일관된 인스턴스 분할이 가능함을 증명하였다.

### 한계 및 논의
- **수직 구조의 취약성**: BEV 투영 특성상 수직으로 세워진 객체나 심하게 겹쳐진 객체들은 2D 상에서 잘 구분되지 않을 수 있다.
- **천장 데이터의 손실**: 지면 투영 방식을 사용하므로 천장 포인트들에 대한 특징 학습이 불가능하다.
- **향후 방향**: 이를 해결하기 위해 단일 BEV가 아닌 다각도(Multi-view)의 2D 표현법을 도입하는 것이 대안이 될 수 있다.

## 📌 TL;DR

본 논문은 대규모 3D 포인트 클라우드의 인스턴스 분할을 위해 **2D BEV 특징 학습과 3D 특징 전파를 결합한 3D-BEVIS** 프레임워크를 제안한다. U-Net을 통해 BEV 상에서 전역적으로 일관된 인스턴스 특징을 먼저 학습하고, 이를 DGCNN으로 3D 포인트들에 전파함으로써 기존의 청크 기반 방식이 가진 한계를 극복하였다. S3DIS와 ScanNet 데이터셋에서 기존 SGPN 대비 우수한 성능을 입증하였으며, 이는 전역 문맥 정보가 3D 인스턴스 구분으로 성능 향상에 핵심적임을 시사한다.