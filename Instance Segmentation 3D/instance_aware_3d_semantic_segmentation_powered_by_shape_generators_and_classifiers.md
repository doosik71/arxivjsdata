# Instance-aware 3D Semantic Segmentation powered by Shape Generators and Classifiers

Bo Sun, Qixing Huang, Xiangru Huang

## 🧩 Problem to Solve

본 논문은 3D semantic segmentation에서 발생하는 포인트 단위 예측의 불일치 문제를 해결하고자 한다. 기존의 3D semantic segmentation 방식들은 주로 포인트(point) 또는 복셀(voxel) 단위의 특징 기술자(feature descriptor)를 학습하여 예측을 수행한다. 그러나 이러한 방식은 포인트 수준에서만 지도 학습(supervision)이 이루어지기 때문에, 동일한 객체(instance)에 속한 포인트들이 서로 다른 클래스로 예측되는 등 인스턴스 수준에서의 일관성(consistency)이 떨어지는 문제가 발생한다.

이를 해결하기 위해서는 인스턴스 수준의 지도 학습이 필요하지만, 모든 데이터에 대해 정교한 인스턴스 레이블(instance labels)을 획득하는 것은 비용이 매우 많이 든다는 현실적인 제약이 존재한다. 따라서 본 논문의 목표는 추가적인 정답 레이블 없이도 인스턴스 인식을 통해 semantic segmentation의 성능과 일관성을 높이는 **InsSeg** 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **비지도 방식의 인스턴스 클러스터링을 통해 얻은 유사-레이블(pseudo-labels)을 활용하여, 인스턴스 수준의 보조 작업(auxiliary tasks)을 수행함으로써 특징 표현력을 강화**하는 것이다.

구체적으로, 인스턴스 분류(Instance Classification)와 형상 재구성(Shape Reconstruction)이라는 두 가지 보조 작업을 도입한다. 이를 통해 네트워크가 포인트 단위의 지역적 정보뿐만 아니라 객체 전체의 구조적 및 기하학적 정보를 학습하도록 강제함으로써, 최종적으로 더 정확하고 일관된 semantic segmentation 결과를 얻도록 설계하였다.

## 📎 Related Works

**3D Semantic Segmentation**
기존 연구들은 주로 U-Net 구조를 기반으로 하여 포인트 클라우드를 다운샘플링 및 업샘플링하며 특징을 추출한다. 데이터 표현 방식에 따라 Range image 기반(SqueezeNet, RangeNet++), Sparse convolution 기반(MinkowskiNet, SparseConvNet), 또는 포인트와 복셀을 결합한 방식(SPVNAS) 등이 제안되었다. 하지만 이러한 방식들은 포인트별로 레이블을 독립적으로 예측할 뿐, 동일 객체 내의 포인트 간 관계나 인스턴스 특성을 고려하지 않는다는 한계가 있다.

**3D Multi-task Learning**
객체 검출(Object Detection)이나 장면 완성(Scene Completion)과 같은 다중 작업을 결합하여 성능을 높이려는 시도가 있었다(예: LidarMultiNet). 그러나 대부분의 연구는 추가적인 센서 데이터나 고비용의 정답 레이블을 요구한다. 반면, 본 제안 방법은 오직 semantic 레이블만을 사용하며 인스턴스 레이블은 비지도 방식으로 획득한다는 점에서 차별점을 갖는다.

**Feature Learning by Completion**
Masked Autoencoder(MAE)와 같이 일부를 마스킹하고 이를 복원하는 방식으로 특징을 학습하는 연구들이 3D 분야에서도 진행되고 있다. 본 논문은 이 개념을 차용하되, 장면 전체가 아닌 **인스턴스 수준에서 마스킹과 복원을 수행**하여 semantic segmentation에 필수적인 객체 중심의 맥락 정보를 학습하도록 한다.

## 🛠️ Methodology

### 1. 전체 파이프라인

InsSeg의 구조는 크게 세 가지 모듈로 구성된다.

1. **Descriptor Learning Module**: 입력 포인트 클라우드에서 특징 기술자를 추출하는 백본 네트워크이다.
2. **Semantic-guided Instance Clustering Module**: 학습된 기술자와 semantic 레이블을 이용하여 포인트들을 인스턴스 단위로 묶는다.
3. **Instance-level Supervision Heads**: 클러스터링된 인스턴스를 대상으로 분류 및 재구성 작업을 수행하여 백본의 학습을 돕는다.

### 2. 상세 구성 요소 및 절차

#### (1) Descriptor Learning

입력 포인트 클라우드 $P \in \mathbb{R}^{N \times 3}$를 복셀 그리드로 변환하여 초기 복셀 특징 $F_0 \in \mathbb{R}^{M \times d_0}$를 추출한다. 이후 3D U-Net 백본을 통해 멀티스케일 특징 $F \in \mathbb{R}^{M \times d}$를 생성하고, 이를 통해 각 복셀의 semantic 레이블 $\bar{S}^V$를 예측한다.

#### (2) Semantic-guided Instance Clustering

정답 semantic 레이블과 백본에서 추출된 기술자를 이용하여 Mean-shift clustering을 수행한다. 이때 각 포인트 $p$의 특징 벡터 $f = (p, \lambda_p, d)$를 사용하며, $\lambda_p$는 포인트 $p$ 주변의 기술자 분산의 역수로 정의된다. 클러스터링 반경 $r_p$는 클래스별 특성에 따라 다르게 설정한다(예: 자동차 1m, 보행자 0.5m). 이를 통해 얻어진 인스턴스 집합을 $\{O_k | k=1, \dots, K\}$라고 한다.

#### (3) Instance Classification Head ($H_c$)

인스턴스 $O_k$에 속한 복셀 특징 $F(O_k)$를 Max-pooling 하여 인스턴스 수준의 특징으로 응축한 후, MLP를 통해 클래스 $\bar{c}_k$를 예측한다.
$$\bar{c}_k = \text{MLP}(\text{max-pool}(F(O_k)))$$
손실 함수로는 온라인 어려운 예제 마이닝(OHEM) 손실 $L_c$를 사용하여, 예측이 틀린 어려운 샘플에 더 집중하여 학습하게 한다.

#### (4) Shape Reconstruction Head ($H_g$)

인스턴스의 일부를 마스킹하고 전체 형상을 복원하는 작업이다. 인스턴스 $O_k$ 내에서 임의의 복셀 $q$를 선택하고 반경 $r$ 이내의 영역을 마스킹한 특징 $F'(O_k)$를 입력으로 사용한다. PointNet Autoencoder 구조를 통해 마스킹된 영역을 포함한 전체 복셀 중심 위치 $V(O_k)$를 재구성하며, Chamfer Distance(CD)를 손실 함수로 사용한다.
$$L_g = \text{CD}(H_g(F'(O_k)), V(O_k))$$

### 3. 학습 절차 및 손실 함수

학습은 2단계로 진행된다.

- **Stage 1**: semantic segmentation 헤드만을 사용하여 백본 네트워크를 먼저 학습시킨다.
- **Stage 2**: 학습된 백본을 이용해 인스턴스를 클러스터링하고, 분류 및 재구성 헤드를 활성화하여 전체 네트워크를 공동 학습(joint training)시킨다.

전체 손실 함수는 다음과 같다.
$$L = L_s + \lambda_1 L_c + \lambda_2 L_g$$
여기서 $L_s$는 포인트별 segmentation 손실, $L_c$는 인스턴스 분류 손실, $L_g$는 형상 재구성 손실이며, $\lambda_1=0.1, \lambda_2=0.01$로 설정되었다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: SemanticKITTI (실외), Waymo Open Dataset (실외), ScanNetV2 (실내)
- **측정 지표**: mIoU (mean Intersection-over-Union) 및 인스턴스 분류 정확도 ($\text{Acc}_{\text{seg}}$)
- **비교 대상**: SPVCNN, LidarMultiNet, Cylinder3D, MinkowskiNet 등 최신 SOTA 모델들

### 2. 주요 결과

- **정량적 성능**: Waymo 데이터셋에서 mIoU $66.13\%$를 달성하여 LidarMultiNet($65.24\%$) 등을 상회하였으며, SemanticKITTI에서도 $65.0\%$의 mIoU로 최상위 성능을 기록하였다.
- **인스턴스 일관성**: $\text{Acc}_{\text{seg}}$ 지표 분석 결과, 특히 버스(Bus)나 기타 차량(Other vehicle)과 같이 예측 불일치가 자주 발생하는 희귀 클래스에서 성능 향상이 두드러졌다.
- **범용성**: Point-based, Voxel-based, Cylinder-based 등 다양한 백본 네트워크에 InsSeg 프레임워크를 결합했을 때 모두 일관된 성능 향상이 관찰되었다.
- **정성적 결과**: 시각화 결과, 기존 모델들이 객체 내부에서 서로 다른 클래스를 예측하던 불일치 현상이 InsSeg 적용 후 크게 감소하고 객체 단위의 일관된 예측이 가능해짐을 확인하였다.

## 🧠 Insights & Discussion

**강점**

- **비지도 방식의 효율성**: 정답 인스턴스 레이블 없이도 semantic-guided clustering만으로 충분히 유의미한 인스턴스 정보를 추출하여 학습에 활용할 수 있음을 증명하였다.
- **백본 독립적 구조**: 특정 아키텍처에 종속되지 않고, 기존의 다양한 3D segmentation 백본 위에 플러그인 형태로 추가하여 성능을 높일 수 있는 범용적인 프레임워크이다.
- **다각적 특징 학습**: 분류 작업은 전역적인 shape 특징을, 재구성 작업은 지역적인 geometry 특징을 학습하게 하여 상호보완적인 효과를 낸다.

**한계 및 비판적 해석**

- **객체 분리 가정**: 본 방법은 3D 공간에서 객체들이 기하학적으로 어느 정도 분리되어 있다는 가정(isolation property)에 의존한다. 따라서 객체 간 경계가 모호한 2D 이미지 데이터셋에는 직접적으로 적용하기 어렵다.
- **전역적 오답 위험**: 인스턴스 수준에서 일관성을 강제하기 때문에, 만약 인스턴스 분류 자체가 틀릴 경우 객체 전체가 잘못된 클래스로 예측될 위험이 있다. 이는 포인트 단위의 불일치보다 IoU 관점에서는 더 큰 하락을 불러올 수 있다.

## 📌 TL;DR

본 논문은 3D semantic segmentation의 고질적인 문제인 '포인트 단위 예측의 불일치'를 해결하기 위해, 비지도 기반의 인스턴스 클러스터링과 인스턴스 수준의 보조 작업(분류 및 재구성)을 결합한 **InsSeg** 프레임워크를 제안하였다. 이를 통해 추가적인 인스턴스 레이블 없이도 객체 수준의 일관성 있는 특징을 학습할 수 있게 되었으며, 실내외 다양한 데이터셋과 백본 네트워크에서 SOTA 성능 향상을 입증하였다. 이 연구는 향후 3D 장면 이해에서 고수준의 객체 개념을 효율적으로 통합하는 방향에 중요한 실마리를 제공한다.
