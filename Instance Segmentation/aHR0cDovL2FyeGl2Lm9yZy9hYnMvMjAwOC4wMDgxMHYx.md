# CASNet: Common Attribute Support Network for image instance and panoptic segmentation

Xiaolong Liu, Yuqing Hou, Anbang Yao, Yurong Chen and Keqiang Li (2020)

## 🧩 Problem to Solve

본 논문은 이미지 내의 개별 객체를 구분하여 분할하는 Instance Segmentation과, 배경(stuff)과 객체(thing)를 모두 통합하여 분할하는 Panoptic Segmentation 문제를 해결하고자 한다. 기존의 Instance Segmentation 방법론들은 주로 Mask R-CNN과 같은 2단계(two-stage) Bounding Box 검출 프레임워크에 기반하고 있다. 이러한 방식은 높은 정확도를 보이지만, 계산 비용이 크고 메모리 점유율이 높으며 학습 수렴이 어렵다는 단점이 있다. 

특히, 기존 알고리즘들은 인스턴스 간의 경계에서 픽셀이 중복되어 할당되거나(overlaps), 어느 인스턴스에도 속하지 않는 빈 공간(holes)이 발생하는 문제가 빈번하게 나타난다. 또한, 점수 기반의 Bounding Box 검출 방식은 실제 픽셀 지원이 없는 가짜 양성(false-positive) 샘플을 생성하는 경향이 있다. 따라서 본 연구의 목표는 이러한 중복과 빈 공간 문제를 해결하고, 효율적인 1단계(one-stage) Fully Convolutional 네트워크를 통해 고품질의 Instance 및 Panoptic Segmentation을 달성하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **'동일한 인스턴스에 속하는 픽셀들은 해당 인스턴스의 공통 속성(Common Attribute)을 공유한다'**는 직관에 기반한다. 여기서 공통 속성이란 기하학적 중심(geometry center)이나 Bounding Box와 같은 정보를 의미한다.

CASNet의 중심적인 설계 아이디어는 픽셀 단위로 공통 속성을 예측하고, 이를 기반으로 픽셀들을 클러스터링하여 인스턴스를 구분하는 것이다. 이를 통해 Bounding Box 기반의 제안(proposal)이나 별도의 분류기 없이도 픽셀의 소속을 명확히 정의할 수 있으며, 결과적으로 인스턴스 간의 중복이나 빈 공간이 없는 정교한 분할 결과를 얻을 수 있다. 또한, 약간의 구조 변경만으로 계산 오버헤드 거의 없이 Panoptic Segmentation으로 확장 가능하다는 점이 주요 기여 사항이다.

## 📎 Related Works

논문에서는 Instance Segmentation의 접근 방식을 크게 네 가지로 분류하여 설명한다.

1.  **CRF 및 RNN 기반 방법:** CNN으로 지역적인 라벨링을 수행한 후 Conditional Random Field(CRF)로 일관성을 맞추거나, RNN의 메모리 특성을 이용해 객체를 순차적으로 카운팅하며 분할하는 방식이다.
2.  **Instance Embedding 방식:** 각 픽셀을 임베딩 공간으로 매핑하여, 같은 인스턴스의 픽셀은 가깝게, 다른 인스턴스는 멀게 배치하는 클러스터링 방식이다. 하지만 검출 기반 방식보다 성능이 낮다는 한계가 있다.
3.  **Detection-first 방식:** Mask R-CNN과 같이 Bounding Box를 먼저 검출하고 그 내부에서 마스크를 생성하는 주류 방식으로, 정확도는 높으나 앞서 언급한 계산 효율성 및 중복 문제 등의 한계가 있다.
4.  **기타 방식:** Watershed 변환, 경계 인식(boundary-aware), 또는 YOLO와 같은 1단계 검출 방식을 응용한 방법들이 존재한다.

CASNet은 이러한 기존 방식들과 달리, 모든 픽셀이 인스턴스의 공통 속성을 직접 예측하게 함으로써 '검출 후 분할'이 아닌 '속성 기반 클러스터링'이라는 차별점을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조
CASNet은 Fully Convolutional 구조로 설계되었으며, 크게 **Backbone, Semantic Head, Common Attribute Head, Box Probability Head**의 네 가지 구성 요소로 이루어져 있다. 기본 백본으로는 고해상도 표현력을 유지하는 HRNet을 사용하며, 이는 $1/2, 1/4, 1/8$ 크기의 피처 맵 피라미드를 출력한다.

### 주요 구성 요소 및 역할
- **Semantic Head:** FCN과 동일한 구조로, 각 픽셀의 클래스 ID를 예측한다. Instance Segmentation 시에는 thing 클래스만 예측하고 나머지는 배경으로 처리하며, Panoptic Segmentation 시에는 stuff 클래스까지 모두 예측한다.
- **Common Attribute Head:** 각 픽셀이 속한 인스턴스의 Bounding Box 경계선까지의 거리(상, 하, 좌, 우 4방향 거리)를 예측한다. 이는 FCOS의 방식과 유사하지만, FCOS가 중심점 근처의 픽셀만 사용하는 것과 달리 CASNet은 모든 픽셀의 공통 속성을 예측하여 인스턴스 구분 기준으로 사용한다.
- **Box Probability Head:** 각 위치가 Bounding Box의 중심일 확률을 예측하여, 이후 포스트 프로세싱 단계에서 '시드 포인트(seed points)'를 생성하는 데 사용된다.

### 훈련 목표 및 손실 함수
전체 손실 함수 $L_{total}$은 다음과 같이 세 가지 손실의 가중치 합으로 정의된다.
$$L_{total} = \alpha L_{cls} + \beta L_{common} + \gamma L_{prob}$$

1.  **Semantic Loss ($L_{cls}$):** 클래스 분류를 위해 Cross-Entropy(CE) 손실을 사용한다.
    $$L_{cls} = -\log \left( \frac{\exp(x[class])}{\sum_{j} \exp(x[j])} \right)$$
2.  **Common Attribute Loss ($L_{common}$):** 예측된 4방향 거리와 실제 거리 사이의 $L_1$ 손실을 사용한다.
    $$L_{common} = \text{mean}(|x_n - y_n|)$$
3.  **Box Probability Loss ($L_{prob}$):** 중심점 확률 예측을 위해 Binary Cross-Entropy(BCE) 손실을 사용한다.
    $$L_{prob} = -w_n [y_n \cdot \log\sigma(x_n) + (1-y_n) \cdot \log(1-\sigma(x_n))]$$

### 추론 및 포스트 프로세싱 절차
1.  **Semantic Map 생성:** Semantic Head의 결과에서 $\text{argmax}$ 연산을 통해 클래스 맵을 얻는다.
2.  **시드 포인트 추출:** Box Probability Head에서 임계값(예: 0.5) 이상의 포인트를 뽑고, NMS(Non-Maximum Suppression)를 적용하여 최종 시드 포인트 $P^{seed}$와 그에 대응하는 $B^{seed}$를 결정한다.
3.  **Common Attribute Support (투표 메커니즘):** 특정 클래스(예: 'car')로 예측된 모든 픽셀은 Common Attribute Head의 예측값을 이용해 자신만의 Bounding Box $B^{car}$를 계산한다. 이후 시드 박스 $B^{seed}$들과의 IoU(Intersection over Union)를 계산하여 가장 높은 IoU를 가진 시드 포인트에 투표한다.
4.  **인스턴스 할당:** 투표 결과에 따라 각 픽셀에 고유한 인스턴스 ID가 부여된다. 이 과정에서 픽셀은 단 하나의 인스턴스에만 할당되므로 중복이나 빈 공간이 발생하지 않는다.

## 📊 Results

### 실험 환경
- **데이터셋:** Cityscapes (도시 도로 환경, $1024 \times 2048$ 해상도)
- **평가 지표:** Instance Segmentation은 mAP를, Panoptic Segmentation은 PQ(Panoptic Quality)를 사용하였다.
- **구현:** PyTorch, HRNet 백본, SGD 옵티마이저 사용.

### 주요 결과
실험은 모든 헤드를 같이 학습시키는 **Joint Training** 모드와, Semantic 결과는 고정된 HRNet에서 가져오는 **Separated Training** 모드로 나누어 진행되었다.

- **Instance Segmentation:** 
    - Joint Training 시 mAP 32.8%를 기록하여 Mask R-CNN(31.5%)과 AdaptIS(32.3%)보다 우수한 성능을 보였다.
    - Separated Training 시에는 mAP 36.3%로 성능이 크게 향상되었다.
- **Panoptic Segmentation:**
    - Separated Training 모드에서 PQ 66.1%를 달성하여, 기존의 UPSNet-M-COCO(61.8%)나 UTIPS(61.4%)보다 훨씬 높은 SOTA(State-of-the-art) 성능을 기록하였다.

### 분석 실험
Semantic Ground Truth(정답지)를 직접 입력했을 때, CASNet은 mAP 57.4%라는 매우 높은 성능을 보였다. 이는 본 연구가 제안한 '공통 속성 기반 분할' 방식 자체가 매우 효과적이며, 최종 성능의 병목 지점이 Semantic Segmentation의 정확도에 있음을 시사한다.

## 🧠 Insights & Discussion

### 강점 및 성과
CASNet은 1단계 Fully Convolutional 네트워크임에도 불구하고, 정교한 투표 메커니즘을 통해 2단계 검출 기반 모델들이 겪는 '인스턴스 간 중복' 및 '빈 공간' 문제를 구조적으로 해결하였다. 특히 Panoptic Segmentation에서 매우 높은 효율성과 정확도를 보인 점이 고무적이다.

### 한계 및 해석
실험 결과, Joint Training이 Semantic Segmentation의 정확도를 떨어뜨리는 현상이 관찰되었다. 이는 공통 속성 예측 작업과 세그멘테이션 작업 간의 상충 관계가 존재함을 의미하며, 두 작업을 더 효과적으로 결합할 수 있는 학습 전략이 필요함을 시사한다. 또한, 본 모델은 HRNet과 같은 고성능 백본에 크게 의존하고 있어, 경량화된 백본에서의 성능 유지 여부는 향후 과제로 남아 있다.

## 📌 TL;DR

본 논문은 픽셀들이 공유하는 공통 속성(Bounding Box 거리 등)을 예측하고 이를 통해 픽셀을 클러스터링하는 **CASNet**을 제안한다. 이 방식은 1단계 Fully Convolutional 구조로 동작하며, 기존 모델의 고질적 문제인 인스턴스 간 중복과 빈 공간 문제를 해결한다. Cityscapes 데이터셋에서 Panoptic Segmentation 기준 SOTA 성능(PQ 66.1%)을 달성하였으며, 이는 향후 실시간 자율주행 시스템의 환경 이해를 위한 고효율 분할 알고리즘으로 적용될 가능성이 높다.