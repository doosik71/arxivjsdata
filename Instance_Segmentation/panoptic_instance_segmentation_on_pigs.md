# Panoptic Instance Segmentation on Pigs

Johannes Brünger, Maria Gentz, Imke Traulsen, and Reinhard Koch (2020)

## 🧩 Problem to Solve

본 연구는 공장식 축산 환경에서 돼지의 행동을 자동으로 인식하고 분석하기 위한 정밀한 객체 분할(Segmentation) 문제를 해결하고자 한다. 돼지의 행동 분석은 동물의 건강과 복지 상태를 파악하는 데 매우 중요하며, 컴퓨터 비전 기반의 시스템은 동물에게 스트레스를 주지 않고 비침습적으로 관찰할 수 있다는 장점이 있다.

기존의 딥러닝 기반 접근 방식은 주로 Bounding Box를 이용한 객체 검출(Object Detection)이나 주요 신체 부위를 찾는 Keypoint Detection 방식을 사용하였다. 그러나 Bounding Box는 동물의 방향에 따라 배경 영역을 과도하게 포함하는 문제가 있으며, Keypoint 방식은 정보가 너무 희소하여 동물의 정확한 외곽선(Contour)을 추적하지 못한다는 한계가 있다. 따라서 본 논문은 픽셀 수준의 정밀도로 개별 돼지를 분할하는 Panoptic Segmentation을 통해, 동물의 크기나 무게 추정 및 보다 정밀한 행동 분류가 가능하도록 하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 돼지 대상의 다양한 분할 작업을 수행할 수 있는 다목적 신경망 프레임워크를 제안한 것이다. 중심적인 설계 아이디어는 다음과 같다.

1. **Panoptic Segmentation의 적용**: Semantic Segmentation(클래스 분류)과 Instance Segmentation(개별 객체 구분)을 결합하여, 배경과 돼지를 구분함과 동시에 겹쳐 있는 개별 돼지들을 픽셀 단위로 분리한다.
2. **다양한 Head 구조의 설계**: 동일한 Encoder-Decoder 백본을 공유하면서, 작업 목적(이진 분할, 범주형 분할, 인스턴스 분할, 방향 인식)에 따라 서로 다른 출력층(Head)을 구성하여 유연한 프레임워크를 구축하였다.
3. **Pixel Embedding과 Binary Mask의 결합**: 고차원 특징 공간으로 픽셀을 투영하는 Pixel Embedding 방식에 이진 분할(Binary Segmentation) 마스크를 결합함으로써, 연산량을 줄이면서도 효율적으로 개별 인스턴스를 클러스터링하는 기법을 제안하였다.

## 📎 Related Works

기존의 돼지 인식 연구는 크게 세 가지 방향으로 진행되어 왔다.

1. **전통적 영상 처리**: 대비 향상(Contrast Enhancement)이나 이진화 임계값(Thresholding) 기반의 배경 분리 방식을 사용하였으나, 조명 변화나 환경 오염에 취약하였다.
2. **객체 검출(Object Detection)**: Bounding Box 기반의 딥러닝 모델을 적용하여 높은 검출률을 보였으나, 동물들이 서로 겹쳐 있을 때 정확한 개체 분리가 어렵고 불필요한 배경 정보가 많이 포함되는 문제가 있었다.
3. **포즈 추정(Pose Estimation)**: 신체 주요 부위에 Keypoint를 찍어 자세를 추정하는 방식이 제안되었으나, 이는 전체적인 신체 윤곽 정보를 제공하지 못한다는 한계가 있다.

본 연구는 이러한 기존 방식들의 간극을 메우기 위해, Mask R-CNN과 같은 인스턴스 분할 개념을 도입하되, 특히 돼지 사육 환경의 특성(위에서 아래로 내려다보는 시점)을 고려하여 타원(Ellipse) 기반의 정밀한 어노테이션과 Panoptic Segmentation 접근법을 택함으로써 차별성을 둔다.

## 🛠️ Methodology

### 전체 시스템 구조

본 논문에서 제안하는 프레임워크는 기본적으로 **U-Net** 구조의 Auto-encoder를 기반으로 한다. Encoder는 입력 영상을 저차원 표현으로 변환하고, Decoder는 이를 다시 원래 해상도로 복원하며, 이 과정에서 Skip Connection을 통해 세부적인 공간 정보를 보존한다. Encoder 백본으로는 ResNet34와 Inception-ResNet-v2가 사용되었다.

### 주요 구성 요소 및 학습 절차

#### 1. Binary Segmentation (이진 분할)

가장 기본적인 단계로, 픽셀이 돼지(1)인지 배경(0)인지를 판별한다. 출력층에 Sigmoid 활성화 함수를 사용하며, 손실 함수로는 Binary Cross-Entropy Loss를 사용한다.
$$L = -\frac{1}{N} \sum_{i=1}^{N} [y_i \cdot \log(p(x_i)) + (1-y_i) \cdot \log(1-p(x_i))]$$

#### 2. Categorical Segmentation (범주형 분할)

개체 간의 경계를 명확히 하기 위해 픽셀을 '배경', '동물의 외곽(Outer edge)', '동물의 중심핵(Inner core)'의 세 가지 클래스로 분류한다. Softmax 함수를 통해 확률 분포를 생성하며, Categorical Cross-Entropy Loss를 사용한다.
$$L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{C} t_{i,j} \cdot \log(x_{i,j})$$
이 방식은 동물의 중심핵을 이용해 개별 객체의 위치를 추정하는 데 도움을 준다.

#### 3. Instance Segmentation (인스턴스 분할)

각 픽셀을 고차원 특징 공간(Pixel Embedding)으로 투영하여 동일한 객체에 속한 픽셀들은 가깝게, 서로 다른 객체에 속한 픽셀들은 멀게 배치하는 **Discriminative Loss**를 사용한다.

손실 함수는 다음 세 가지 항의 가중치 합으로 구성된다:

- **Variance term ($L_{var}$)**: 동일 객체 내 픽셀들이 중심 $\mu_c$ 근처로 모이게 하여 클러스터를 형성한다.
- **Distance term ($L_{dist}$)**: 서로 다른 객체의 중심들 간의 거리를 일정 임계값 $\delta_d$ 이상으로 유지하여 객체를 분리한다.
- **Regularization term ($L_{reg}$)**: 특징 공간의 전체적인 확장을 제한하여 수렴을 돕는다.

$$L = \alpha \cdot L_{var} + \beta \cdot L_{dist} + \gamma \cdot L_{reg}$$

추론 단계에서는 **HDBSCAN** (Hierarchical Density-Based Spatial Clustering of Applications with Noise) 알고리즘을 사용하여 임베딩 공간에서 픽셀들을 클러스터링함으로써 개별 돼지를 분리한다.

#### 4. Combined Segmentation 및 방향 인식

클러스터링의 연산량을 줄이기 위해 Binary Segmentation Head와 Embedding Head를 동시에 학습시킨다. 먼저 이진 마스크를 통해 돼지 영역만 추출한 뒤, 해당 영역의 픽셀들만 클러스터링에 투입한다. 또한, '머리'와 '몸통'을 구분하는 범주형 분할을 추가하여 타원 모델의 180도 회전 모호성을 해결하고 정확한 방향(Orientation)을 인식한다.

## 📊 Results

### 실험 설정

- **데이터셋**: 실제 돼지 사육장의 카메라 5대에서 수집한 1,000장의 이미지 (해상도 $1280 \times 800$).
- **데이터 구성**: 4대의 카메라 영상으로 학습/검증셋을 구성하고, 나머지 1대의 카메라 영상으로 테스트셋을 구성하여 일반화 성능을 측정하였다. 주간 컬러 이미지와 야간 적외선 이미지가 모두 포함되었다.
- **지표**: Panoptic Quality (PQ), F1 Score, Precision, Recall, Jaccard Index를 사용하였다.

### 주요 결과

1. **분할 성능**: 이진 분할의 경우 Jaccard Index 기준 약 $0.97$의 매우 높은 정확도를 보였다.
2. **검출 성능**: 개별 돼지를 검출하는 F1 Score는 약 $95\%$에 달했다.
3. **강건성**: 주간 이미지($F1 \approx 0.96$)와 야간 적외선 이미지($F1 \approx 0.94$) 모두에서 일관되게 높은 성능을 보여 조명 변화에 강건함을 입증하였다.
4. **백본 영향**: ResNet34와 Inception-ResNet-v2 간의 성능 차이는 미미하였다. 이는 데이터셋의 규모가 상대적으로 작아 매우 복잡한 아키텍처의 이점이 충분히 발휘되지 않았기 때문으로 분석된다.
5. **방향 인식**: 정확하게 검출된 돼지들 중 약 $94\%$의 방향을 올바르게 인식하였다.

## 🧠 Insights & Discussion

### 강점 및 성과

본 연구는 기존의 Bounding Box나 Keypoint 방식보다 훨씬 정밀한 픽셀 수준의 분할을 구현하였다. 이를 통해 동물의 외곽선을 정확히 추출할 수 있으며, 이는 향후 동물의 부피 및 무게 추정으로 이어질 수 있어 축산 관리 측면에서 실질적인 가치가 높다. 특히 HDBSCAN과 Binary Mask를 결합하여 고차원 임베딩의 연산 효율성을 높인 점이 돋보인다.

### 한계 및 비판적 해석

1. **심한 중첩 상황의 취약성**: 범주형 분할(Categorical Segmentation)의 경우, 돼지들이 심하게 겹쳐 동물의 '중심핵(Core)'이 가려지면 개체 분리가 불가능한 한계가 있다. 임베딩 방식이 이를 해결할 수 있을 것으로 기대되었으나, 실제 데이터셋에서 이러한 사례가 적어 모델이 충분히 학습되지 못한 경향이 있다.
2. **데이터셋의 규모 및 다양성**: 백본 네트워크 변경에 따른 성능 차이가 거의 없었다는 점은 현재 사용된 데이터셋의 분산(Variance)이 낮음을 시사한다. 더 다양한 환경과 대규모 데이터셋에서의 검증이 필요하다.
3. **평가 지표의 비교 가능성**: PQ(Panoptic Quality)라는 최신 지표를 사용하였으나, 기존 연구들이 이 지표를 사용하지 않아 직접적인 성능 비교가 어렵다는 점이 아쉽다.

## 📌 TL;DR

본 논문은 돼지의 정밀한 행동 분석을 위해 U-Net 기반의 **Panoptic Segmentation** 프레임워크를 제안하였다. Binary Segmentation과 Pixel Embedding을 결합한 구조를 통해 겹쳐 있는 돼지들을 픽셀 단위로 정확히 분리해냈으며, **약 95%의 F1 Score**를 달성하였다. 이 연구는 단순한 객체 검출을 넘어 동물의 정확한 체형 정보를 추출함으로써, 향후 가축의 무게 추정 및 정밀 복지 모니터링 시스템 구축에 기여할 가능성이 매우 높다.
