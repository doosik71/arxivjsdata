# DArch: Dental Arch Prior-assisted 3D Tooth Instance Segmentation with Weak Annotations

Liangdong Qiu, Chongjie Ye, Pei Chen, Yunbi Liu, Xiaoguang Han, Shuguang Cui (2022)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 3D 치아 모델의 인스턴스 분할(Instance Segmentation)을 위해 요구되는 고비용의 정밀한 어노테이션(point-wise annotations) 의존성을 줄이는 것이다. 기존의 딥러닝 기반 방법들은 모든 치아의 모든 포인트에 대해 정밀한 마스크를 생성해야 하므로, 데이터 구축에 막대한 시간이 소요되며 이는 결국 실제 환경의 복잡한 치아 모델을 충분히 커버하는 대규모 데이터셋 확보를 어렵게 만들어 모델의 일반화 성능을 제한한다.

특히, 치아가 누락되었거나, 겹쳐 있거나(crowding), 정렬이 맞지 않는(misaligned) 다양한 치아 모델에서 각 치아 객체의 위치를 정확히 찾아내는 것이 가장 큰 도전 과제이다. 따라서 본 연구의 목표는 모든 치아의 중심점(centroid)과 소수의 치아 마스크만을 사용하는 저비용의 약한 어노테이션(weak annotations) 환경에서도 높은 성능을 내는 3D 치아 인스턴스 분할 프레임워크를 개발하는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 아이디어는 치아의 기하학적 특성인 **치아궁(Dental Arch)**의 사전 정보(prior)를 활용하여 치아 중심점 검출의 정확도를 높이고, 이를 통해 최종적인 분할 성능을 향상시키는 것이다. 주요 기여 사항은 다음과 같다.

1. **저비용 어노테이션 전략 제안**: 모든 치아의 중심점과 모델당 소수의 치아 마스크만을 라벨링하는 새로운 약한 어노테이션 방식을 탐색하고 이를 처리하기 위한 DArch 프레임워크를 제안하였다.
2. **Coarse-to-Fine 치아궁 추정**: Bézier 곡선 회귀를 통해 초기 치아궁을 생성하고, 이후 Graph Convolutional Network(GCN)를 통해 이를 정밀하게 수정하는 단계적 추정 방법을 제안하였다.
3. **Arch-aware Point Sampling (APS) 모듈**: 기존의 Furthest Point Sampling(FPS)이 잇몸(gingiva)과 같은 불필요한 포인트를 샘플링하는 문제를 해결하기 위해, 추정된 치아궁을 기반으로 중심점 후보를 샘플링하는 APS 전략을 도입하였다.
4. **성능 입증**: 약한 어노테이션과 완전한 어노테이션 시나리오 모두에서 기존의 SOTA(State-of-the-Art) 방법들보다 우수한 성능을 보임을 실험적으로 증명하였다.

## 📎 Related Works

### 3D Natural Scene Understanding
PointNet, PointNet++, VoteNet 등 포인트 클라우드 기반의 객체 검출 및 분할 연구들이 진행되어 왔다. 특히 VoteNet은 허프 투표(Hough voting) 메커니즘을 통해 객체의 중심을 예측하며, 본 논문은 이 VoteNet의 아키텍처를 중심점 검출의 기본 구조로 채택하였다. 하지만 VoteNet의 FPS 샘플링 방식은 치아와 같이 세밀한 구조에서는 잇몸 등의 무관한 포인트를 선택할 가능성이 크다는 한계가 있다.

### 3D Tooth Understanding
Mask MCNet, TSegNet 등이 3D 치아 분할을 위해 제안되었다. TSegNet은 바운딩 박스보다 치아 중심점이 더 신뢰할 수 있는 신호라는 점을 발견하여 이를 활용한 파이프라인을 제안하였다. 그러나 이러한 기존 방법들은 모두 모든 치아에 대한 정밀한 포인트 단위의 어노테이션이 필요하다는 치명적인 비용 문제가 존재하며, 본 논문은 이러한 제약을 극복하기 위해 약한 어노테이션 환경을 처음으로 다루었다.

## 🛠️ Methodology

DArch 프레임워크는 **치아 중심점 검출(Tooth Centroid Detection)**과 **치아 인스턴스 분할(Tooth Instance Segmentation)**의 두 단계로 구성된 Detect-and-Segment 구조를 가진다.

### 1. Tooth Centroid Detection
이 단계는 VoteNet을 백본으로 사용하며, 여기에 치아궁 예측 브랜치를 추가하여 중심점 검출의 정확도를 높인다.

#### (1) Dental Arch Prediction (Coarse-to-Fine)
치아궁은 치아 중심점들을 지나는 곡선으로 정의되며, 두 단계로 추정한다.
- **Coarse stage (Bézier Curve Regression)**: 3차 Bézier 곡선을 사용하여 치아궁을 근사한다. MLP를 통해 4개의 제어점(control points) $\{x_{ctr}^{i}\}_{i=1}^{4}$를 예측하며, 손실 함수는 다음과 같다.
  $$L_{ctr} = \frac{1}{4} \sum_{i=1}^{4} \ell_{1}(\hat{x}_{ctr}^{i} - x_{ctr}^{i})$$
- **Fine stage (GCN Refinement)**: Bézier 곡선에서 균일하게 샘플링된 초기 포인트들을 GCN에 입력하여 오프셋(offset)을 학습하고 이를 반복적으로 적용하여 치아궁을 정밀하게 수정한다. 손실 함수는 예측된 치아궁 포인트 $\hat{x}_{i}$와 실제 치아궁 포인트 $x_{gt}^{i}$ 사이의 $\ell_{1}$ 거리로 정의된다.
  $$L_{arch} = \frac{1}{N} \sum_{i=1}^{N} \ell_{1}(\hat{x}_{i} - x_{gt}^{i})$$

#### (2) Arch-aware Point Sampling (APS)
추정된 치아궁을 기반으로 중심점 후보(proposals)를 생성한다. 단순한 KNN 방식 대신 **Hungarian method**를 사용하여 더 균일하게 샘플링하며, 이때 비용 행렬 $C$는 다음과 같이 구성된다.
$$C = \alpha D_{arch} + \beta D_{votes}$$
여기서 $D_{arch}$는 투표 포인트와 치아궁 포인트 사이의 유클리드 거리이며, $D_{votes}$는 투표 포인트의 변위 거리이다. 이를 통해 잇몸 부위의 불필요한 포인트를 제거하고 치관(crown) 주변의 유효한 포인트를 선택한다.

#### (3) Detection Loss
최종 검출 손실 함수는 오프셋 예측 손실($L_{offset}$), 제안 영역의 신뢰도 손실($L_{conf}$), 중심점 오프셋 손실($L_{centers}$)의 합으로 구성된다.
$$L_{det} = L_{offset} + L_{conf} + \gamma L_{centers}$$

### 2. Tooth Instance Segmentation
중심점이 검출되면, 각 중심점을 기준으로 주변의 $M=2048$개 포인트를 크롭하여 패치(patch)를 생성한다.
- **Patch-based Training**: 전체 모델을 한 번에 분할하는 대신, 중심점 기반의 3D 패치를 입력으로 하여 해당 치아의 마스크를 예측하도록 PointNet++ 기반의 Segmentor를 학습시킨다.
- **Weak Annotation 활용**: 학습 시에는 전체 모델 중 라벨링된 소수의 치아 마스크에서 생성된 패치들만을 사용하여 학습함으로써 데이터 효율성을 높인다.
- **Inference**: 추론 시에는 검출된 모든 중심점에 대해 패치 기반 분할을 수행하고, 이 결과들을 융합(fuse)하여 전체 모델의 분할 맵을 완성한다.

## 📊 Results

### 실험 설정
- **데이터셋**: 3,231명의 환자로부터 수집한 4,773개의 3D 치아 모델을 사용하였다. (훈련 3,973개, 테스트 800개)
- **비교 대상**: 검출 단계에서는 VoteNet, MLCVNet, Group-free 3D와 비교하였으며, 분할 단계에서는 TSegNet 및 VoteNet+PointNet++ 조합과 비교하였다.
- **평가 지표**: 검출 성능은 Accuracy(ACC), Recall, Chamfer Distance로 측정하였고, 분할 성능은 IoU와 Dice 계수를 사용하였다.

### 정량적 결과
- **분할 성능**: DArch는 완전한 어노테이션뿐만 아니라 약한 어노테이션 시나리오에서도 SOTA 방법인 TSegNet보다 우수한 성능을 보였다. 특히 약한 어노테이션 상황에서 IoU는 2.03%, Dice는 1.55% 향상되었다.
- **검출 성능**: APS 전략을 사용했을 때 ACC가 비약적으로 상승하였으며, 이는 정확한 중심점 위치 파악이 이후 분할 단계의 성능 향상으로 직결됨을 보여준다.
- **어노테이션 비용**: 실제 측정 결과, 약한 어노테이션 방식이 완전한 어노테이션 방식보다 치아 모델 하나당 라벨링 시간을 크게 단축시킴을 확인하였다 (Fig 4 참조).

## 🧠 Insights & Discussion

### 강점 및 통찰
본 연구의 가장 큰 성과는 치아의 기하학적 구조(Dental Arch)라는 도메인 지식을 딥러닝 파이프라인에 성공적으로 통합했다는 점이다. 일반적인 3D 객체 검출에서 사용하는 FPS 샘플링은 데이터의 분포만을 고려하므로 치아 모델에서는 잇몸과 같은 노이즈를 포함하기 쉽지만, APS는 치아궁이라는 제약 조건을 부여함으로써 검출의 정확도를 획기적으로 높였다. 또한, 패치 기반 학습 전략을 통해 매우 적은 양의 마스크 데이터만으로도 효과적인 분할 모델을 학습시킬 수 있음을 보여주었다.

### 한계 및 향후 과제
분할기(Segmentor)는 여전히 완전 감독 학습(fully-supervised) 방식으로 학습되었으며, 약한 어노테이션의 특성인 '중심점 정보'나 '치아궁 사전 정보'를 분할 단계에서 직접적으로 활용하지는 못했다. 저자들은 향후 이러한 정보를 분할기 설계에 직접 반영하여 일반화 성능을 더욱 높이겠다고 언급하였다. 또한, 매우 적은 양의 마스크만 사용했을 때 발생할 수 있는 실제 환경에서의 일반화 능력 부족 문제는 여전히 해결해야 할 과제로 남아 있다.

## 📌 TL;DR

본 논문은 3D 치아 인스턴스 분할을 위해 고비용의 전체 마스크 대신 **모든 중심점과 소수의 마스크만 사용하는 약한 어노테이션 기반의 DArch 프레임워크**를 제안하였다. 치아궁(Dental Arch)을 Bézier 곡선과 GCN으로 추정하고, 이를 이용한 **Arch-aware Point Sampling (APS)**을 통해 치아 중심점 검출의 정확도를 극대화하였다. 결과적으로 DArch는 데이터 라벨링 비용을 크게 낮추면서도 기존의 완전 감독 학습 기반 SOTA 모델들을 뛰어넘는 분할 성능을 달성하였으며, 이는 치과 진단 및 교정 치료의 자동화에 기여할 가능성이 크다.