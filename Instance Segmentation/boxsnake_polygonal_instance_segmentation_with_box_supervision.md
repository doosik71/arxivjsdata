# BoxSnake: Polygonal Instance Segmentation with Box Supervision

Rui Yang, Lin Song, Yixiao Ge, Xiu Li (2023)

## 🧩 Problem to Solve

인스턴스 분할(Instance Segmentation)은 객체의 정밀한 위치와 형태를 파악하는 것이 목적이며, 이는 자율 주행이나 로봇 제어와 같은 분야에서 매우 중요하다. 현재 이 분야는 픽셀 단위의 마스크를 사용하는 Mask-based 방식과 객체의 외곽선을 정점으로 표현하는 Polygon-based 방식으로 나뉜다. 하지만 두 방식 모두 정밀한 마스크나 폴리곤 어노테이션을 생성하는 데 막대한 비용과 시간이 소모된다는 치명적인 한계가 있다.

최근에는 비용이 저렴한 바운딩 박스(Bounding Box) 어노테이션만을 이용해 마스크를 예측하는 Box-supervised instance segmentation 연구가 진행되어 왔다. 그러나 기존의 박스 지도 학습 방식들은 대부분 Mask-based 프레임워크에 집중되어 있었으며, 폴리곤 기반의 표현 방식을 사용하여 박스 어노테이션만으로 학습을 수행하려는 시도는 없었다. 따라서 본 논문의 목표는 박스 어노테이션만을 활용하여 효과적인 폴리곤 인스턴스 분할을 달성하는 새로운 end-to-end 학습 기법을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 폴리곤의 정점들을 정밀하게 조정하기 위해 **Point-based Unary Loss**와 **Distance-aware Pairwise Loss**라는 두 가지 손실 함수를 설계한 것이다.

단순히 박스 정보만으로는 폴리곤의 세부 형태를 잡을 수 없으므로, 먼저 Unary Loss를 통해 폴리곤이 박스 내부에 타이트하게 위치하도록 강제하고, 이후 이미지의 픽셀 간 색상 유사성(Color Affinity)과 거리 변환(Distance Transformation)을 이용한 Pairwise Loss를 통해 폴리곤의 정점이 실제 객체의 경계선(Boundary)에 밀착되도록 유도한다. 특히, 폴리곤 정점의 좌표 회귀 문제(Regression)를 분류 문제(Classification)로 변환하여 미분 가능한 형태로 설계함으로써 end-to-end 학습이 가능하게 만들었다는 점이 주요한 기술적 기여이다.

## 📎 Related Works

인스턴스 분할 연구는 크게 세 가지 흐름으로 구분된다.

1. **Mask-based Instance Segmentation**: Mask R-CNN과 같이 픽셀 단위의 이진 마스크를 예측하는 방식이다. 최근에는 CondInst나 Query-based 방법론으로 발전하며 효율성과 정확도를 높였으나, 여전히 고비용의 마스크 라벨이 필요하다.
2. **Polygon-based Instance Segmentation**: 객체의 윤곽선을 정점들의 집합으로 표현하는 방식이다. Deep Snake나 BoundaryFormer 등이 대표적이며, 마스크 방식보다 표현력이 정교하고 효율적일 수 있으나 역시 정밀한 폴리곤 라벨이 필수적이다.
3. **Box-supervised Instance Segmentation**: BBTP, BoxInst 등 바운딩 박스만을 사용하여 마스크를 예측하는 시도들이다. 이들은 주로 다중 인스턴스 학습(MIL)이나 픽셀 간의 친화도(Affinity) 모델링을 통해 pseudo-mask를 생성하는 방식을 사용한다.

본 논문은 기존의 Box-supervised 연구들이 Mask-based에 치중했던 것과 달리, 이를 Polygon-based 프레임워크에 이식하여 박스 정보만으로 폴리곤 윤곽선을 학습하는 최초의 딥러닝 기반 방법론을 제시한다.

## 🛠️ Methodology

### 전체 시스템 구조

BoxSnake는 Mask R-CNN 프레임워크를 기반으로 하며, FPN(Feature Pyramid Network)을 통해 다중 스케일 특징 맵을 추출한다. 바운딩 박스 예측 헤드가 먼저 박스를 생성하면, Transformer 기반의 Polygon Head가 해당 박스 내부에 초기화된 폴리곤 정점들을 반복적으로 정교화(Refine)하여 최종 폴리곤을 예측한다.

### 핵심 손실 함수 (Loss Functions)

BoxSnake의 학습은 다음과 같은 통합 손실 함수를 통해 이루어진다.
$$\mathcal{L}_{polygon} = \alpha \mathcal{L}_u + \beta \mathcal{L}_{lp} + \gamma \mathcal{L}_{gp}$$

#### 1. Point-based Unary Loss ($\mathcal{L}_u$)

예측된 폴리곤 $\mathcal{C}$의 외곽을 감싸는 최소 바운딩 박스 $b_c$를 계산하고, 이를 정답 박스 $b$와 비교한다.
$$\mathcal{L}_u = 1 - \text{CIoU}(b_c, b)$$
여기서 $\text{CIoU}$는 Complete IoU를 의미한다. 이 손실 함수는 폴리곤의 정점들이 정답 박스 밖으로 벗어나지 않고 최대한 타이트하게 객체를 감싸도록 유도하여 coarse-grained한 분할을 달성한다.

#### 2. Distance-aware Pairwise Loss

폴리곤 정점의 좌표는 불연속적이므로 직접 최적화하기 어렵다. 이를 해결하기 위해 폴리곤 내부(1)와 외부(0)를 구분하는 레벨 셋(Level Set) 개념을 도입하고, 이를 미분 가능한 함수로 근사화한다.

* **Distance Relaxation (거리 완화)**: 픽셀 $(x, y)$에서 폴리곤 $\mathcal{C}$까지의 최단 거리 $D_C(x, y)$를 계산하고, 이를 Sigmoid 함수 $\sigma$를 통해 연속적인 매핑 함수 $U'_C(x, y)$로 변환한다.
    $$U'_C(x,y) = \sigma \left( \frac{2 \cdot (U_C(x,y) - 0.5) \cdot D_C(x,y)}{\tau} \right)$$
    이를 통해 폴리곤 정점의 좌표 변화가 픽셀 값의 변화로 이어지게 하여 역전파(Backpropagation)가 가능해진다.

* **Local Pairwise Term ($\mathcal{L}_{lp}$)**: 지역 윈도우 내에서 색상이 유사한 두 픽셀은 동일한 레벨 셋(둘 다 내부이거나 둘 다 외부)에 속해야 한다는 가정을 바탕으로 한다.
    $$\mathcal{L}_{lp} = \sum_{(p,q) \in \Omega} w[(i,j), (p,q)] |U'_C(i,j) - U'_C(p,q)|$$
    여기서 $w$는 색상 거리에 기반한 친화도(Affinity)이며, 색상이 비슷할수록 $w$ 값이 커져 서로 다른 레벨 셋에 있을 때 큰 페널티를 준다.

* **Global Pairwise Term ($\mathcal{L}_{gp}$)**: 객체 내부와 외부의 색상이 각각 균일(Homogeneous)해야 한다는 가정을 기반으로, 예측된 폴리곤 내부/외부의 평균 색상($u_{in}, u_{out}$)과 각 픽셀 색상 간의 분산을 최소화한다.
    $$\mathcal{L}_{gp} = \sum_{(x,y) \in \Omega} \|I(x,y) - u_{in}\|^2 \cdot U'_C(x,y) + \sum_{(x,y) \in \Omega} \|I(x,y) - u_{out}\|^2 \cdot (1 - U'_C(x,y))$$
    이는 지역적인 노이즈를 억제하고 폴리곤의 외곽선을 더욱 매끄럽게 만드는 역할을 한다.

### Clipping Strategy

전체 이미지에 대해 Pairwise Loss를 계산하는 것은 메모리 소모가 극심하다. 따라서 본 논문에서는 정답 박스를 중심으로 약간 확장된 영역만을 RoIAlign으로 크롭하여 계산하는 Clipping Strategy를 사용하여 연산 효율성을 높였다.

## 📊 Results

### 실험 설정

* **데이터셋**: COCO val2017, Cityscapes validation set.
* **백본**: ResNet-50, ResNet-101, Swin-B, Swin-L.
* **지표**: Mask AP (Average Precision).
* **비교 대상**: Fully-supervised (Mask R-CNN, BoundaryFormer 등) 및 Box-supervised (BoxInst, DiscoBox, BBTP 등) 방법론.

### 주요 결과

1. **COCO 데이터셋**: ResNet-50 백본 기준, BoxSnake는 31.1% AP를 달성하여 기존 박스 지도 학습 모델인 BoxInst(30.7%)와 DiscoBox(30.7%)를 상회하였다. 특히 예측된 박스와 실제 분할 결과 사이의 정확도 격차($\Delta AP_b$)를 약 8%까지 줄여, 마스크 기반 방식보다 더 정밀한 경계를 찾아냈음을 입증하였다.
2. **Cityscapes 데이터셋**: 26.3% AP를 기록하며 BoxInst(22.4%)와 AsyInst(24.7%) 대비 압도적인 성능 향상을 보였다. 이는 Cityscapes의 차량과 같은 강체(Rigid object)들이 폴리곤의 구조적 사전 정보(Shape Prior)를 통해 더 잘 학습되었기 때문으로 분석된다.
3. **상한선 탐색**: Swin-L 백본을 사용했을 때 최대 39.5% AP까지 성능이 향상되어, 박스 지도 학습만으로도 매우 높은 수준의 폴리곤 분할이 가능함을 보여주었다.

### Ablation Study

* **Unary Loss 영향**: $\mathcal{L}_u$가 없으면 Pairwise Loss만으로는 폴리곤이 한 점으로 수렴하거나 이미지 전체로 확장되는 Trivial solution이 발생한다. CIoU 기반의 $\mathcal{L}_u$를 사용했을 때 가장 안정적인 성능을 보였다.
* **Loss 조합**: $\mathcal{L}_u$ 단독 사용 시 23.9% $\rightarrow$ $\mathcal{L}_{lp}$ 추가 시 30.8% $\rightarrow$ $\mathcal{L}_{gp}$까지 모두 추가 시 31.1%로 AP가 상승하여, 각 손실 함수가 상호 보완적으로 작용함을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 해석

BoxSnake의 가장 큰 강점은 폴리곤 표현 방식이 갖는 **구조적 사전 정보(Structural Prior)**를 활용했다는 점이다. 마스크 기반 방식은 픽셀 하나하나를 독립적으로 예측하려 하지만, 폴리곤 방식은 정점들을 연결한 폐곡선을 예측하므로 강체 객체의 형태를 더 잘 유지한다. 이는 Cityscapes 결과에서 명확히 드러나며, 도로의 그림자와 차량의 경계를 더 명확하게 구분하는 결과로 나타났다. 또한, 폴리곤은 마스크보다 데이터 표현량이 훨씬 적어(64개 정점 vs $64 \times 64$ 픽셀) 추론 속도 면에서도 이점이 있다.

### 한계 및 향후 과제

논문에서 제시한 Bad Cases에 따르면, 다음과 같은 한계가 존재한다.

1. **오목한 형태(Concave contours)**: Pairwise Loss가 기본적으로 곡선의 길이를 짧게 유지하려는 성질이 있어, 깊게 파인 오목한 경계를 예측하는 데 어려움이 있다.
2. **유사 색상 구분**: 색상 친화도에 의존하기 때문에, 서로 다른 인스턴가 매우 유사한 색상을 가졌을 때 이를 구분하여 소유권을 결정하는 능력이 부족하다.
이를 해결하기 위해 향후 연구에서는 더 고도화된 Pairwise Loss를 설계하거나, 저수준 색상 정보 외에 고수준 시맨틱 특징(High-level features)을 활용한 관계 추론이 필요할 것으로 보인다.

## 📌 TL;DR

본 논문은 비용이 많이 드는 마스크/폴리곤 라벨 없이 **바운딩 박스 어노테이션만으로 폴리곤 인스턴스 분할을 수행하는 BoxSnake**를 제안한다. 폴리곤 정점을 박스 내에 가두는 **Unary Loss**와, 픽셀 색상 유사도를 이용해 경계를 정밀화하는 **Distance-aware Pairwise Loss**를 통해 end-to-end 학습을 구현하였다. 실험 결과, 기존의 박스 지도 학습 마스크 방식보다 우수한 성능을 보였으며, 특히 강체 객체가 많은 데이터셋에서 강력한 성능을 입증하였다. 이 연구는 향후 저비용 데이터셋 구축 및 멀티모달 모델(텍스트-폴리곤 변환 등)로의 확장에 중요한 가능성을 제시한다.
