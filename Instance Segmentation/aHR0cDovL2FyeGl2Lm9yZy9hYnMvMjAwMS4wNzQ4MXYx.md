# Joint Learning of Instance and Semantic Segmentation for Robotic Pick-and-Place with Heavy Occlusions in Clutter

Kentaro Wada, Kei Okada and Masayuki Inaba (2020)

## 🧩 Problem to Solve

본 논문은 물체들이 복잡하게 섞여 있는 Clutter 환경에서 로봇이 물체를 집어 옮기는 Pick-and-Place 작업을 수행할 때 발생하는 심각한 Occlusion(폐색) 문제를 해결하고자 한다.

로봇 조작을 위해서는 픽셀 단위의 객체 분할이 필수적이다. 특히, 어떤 물체가 다른 물체에 의해 가려져 있는지(Occlusion)를 정확히 파악해야만 충돌을 피하고 적절한 파지 순서를 계획할 수 있다. 기존의 Instance Segmentation 방식은 '이미지 $\rightarrow$ Bounding Box $\rightarrow$ Mask'의 2단계 예측 과정을 거치는데, 물체가 심하게 가려진 경우 1단계에서 전체 영역을 포함하는 Bounding Box를 정확히 예측하는 것이 매우 어렵다. 이로 인해 최종적으로 예측되는 Mask가 잘려나가는(Truncated) 문제가 발생하며, 이는 로봇의 파지 실패로 이어진다.

따라서 본 연구의 목표는 Instance Segmentation의 한계를 극복하기 위해 Semantic Segmentation의 이미지 수준 추론(Image-level reasoning)을 결합한 Joint Learning 모델을 구축하여, 가려진 영역(Occluded region)에 대한 예측 정확도를 높이는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Instance Occlusion Segmentation(IOS)**과 **Semantic Occlusion Segmentation(SOS)**을 함께 학습시키는 Joint Learning 구조를 제안한 것이다.

1.  **추론 수준의 결합**: Instance Segmentation의 '인스턴스 수준 추론(Instance-level reasoning)'과 Semantic Segmentation의 '이미지 수준 추론(Image-level reasoning)'을 융합하여, Bounding Box 예측 단계에서 부족했던 가시 영역 및 폐색 영역에 대한 감독(Supervision)을 보완한다.
2.  **폐색 영역 분할 학습**: CNN 기반의 픽셀 단위 스코어 회귀(Pixel-wise score regression)를 통해 물체의 가시 영역과 가려진 영역을 동시에 학습하는 모델을 구현하였다.
3.  **로봇 작업 적용**: 제안한 모델을 실제 로봇의 Random Picking 및 Target Picking 작업에 적용하여, 심한 폐색 환경에서도 효율적으로 물체를 조작할 수 있음을 증명하였다.

## 📎 Related Works

### 관련 연구 및 한계
- **Instance Segmentation**: Mask R-CNN과 같은 최신 모델들은 객체 검출 후 마스크를 예측하는 방식을 사용한다. 저자들의 이전 연구[14]에서는 이를 확장하여 가시 영역과 폐색 영역을 모두 예측하는 모델을 제안했으나, 앞서 언급한 Bounding Box 예측의 의존성 문제로 인해 한계가 있었다.
- **Semantic Segmentation**: FCN 등을 통해 픽셀 단위로 클래스를 분류한다. 하지만 동일 클래스의 서로 다른 인스턴스를 구분하지 못한다는 단점이 있다.
- **Joint Learning**: 서로 다른 비전 작업(예: Semantic Segmentation과 Depth Estimation)을 함께 학습시켜 성능을 높이는 시도들이 있었다. 그러나 본 논문처럼 유사한 출력 형태(Mask)를 가지면서 서로 다른 추론 수준(Image-level vs Instance-level)을 결합한 연구는 부족했다.

### 차별점
본 논문은 단순히 여러 작업을 함께 학습시키는 것을 넘어, Instance Segmentation의 고질적인 문제인 'Box 예측 실패'를 Semantic Segmentation의 '전역적 마스크 예측' 능력을 통해 해결하려 했다는 점에서 차별성을 가진다. 또한, Instance Mask로부터 Semantic Mask를 생성할 수 있으므로 추가적인 인간의 라벨링 없이 Joint Learning이 가능하다는 효율성을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조
본 모델은 **Shared Feature Extractor**를 중심으로 Instance Occlusion Segmentation 경로와 Semantic Occlusion Segmentation 경로가 병렬로 연결된 구조이다.

1.  **Shared Feature Extractor**: ResNet50-C4를 공유하여 공통적인 특징을 추출한다.
2.  **Instance Occlusion Segmentation (IOS)**: 
    - Mask R-CNN을 확장하여 적용하였다.
    - RPN(Region Proposal Networks) $\rightarrow$ ROI Feature Transformer $\rightarrow$ ROI Feature Extractor $\rightarrow$ Classification/Box Prediction $\rightarrow$ Mask Prediction 순으로 진행된다.
    - 특히 Mask Prediction 단계에서 가시 영역(Visible), 폐색 영역(Occluded), 배경(Background)의 세 가지 클래스를 예측하도록 확장하였다.
3.  **Semantic Occlusion Segmentation (SOS)**: 
    - FCN(Fully Convolutional Networks) 구조를 사용하여 이미지 전체에서 픽셀 단위로 클래스를 예측한다.
    - **가시 영역(Visible)**: Softmax Cross Entropy 손실 함수를 사용하여 클래스 간 경쟁적인 예측을 수행한다.
    - **폐색 영역(Occluded)**: 폐색 영역은 서로 중첩(Overlap)될 수 있으므로, 클래스별 독립적 예측이 가능한 Sigmoid Cross Entropy 손실 함수를 사용한다.

### 주요 방정식 및 학습 절차
모델은 모든 손실 함수를 합산하여 역전파(Backpropagation)를 수행한다.

인스턴스 분할 관련 손실 함수 $l_{ins}$는 다음과 같이 정의된다:
$$l_{ins} = l_{rpn\_box} + l_{rpn\_cls} + l_{ins\_box} + l_{ins\_cls} + l_{ins\_mask}$$

시맨틱 분할 관련 손실 함수 $l_{sem}$은 다음과 같이 정의된다:
$$l_{sem} = l_{sem\_vis} + l_{sem\_occ}$$

최종 통합 손실 함수 $l$은 다음과 같다:
$$l = l_{ins} + \lambda \cdot l_{sem}$$
여기서 $\lambda$는 두 작업 간의 균형을 맞추기 위한 가중치 하이퍼파라미터이다.

학습 시 ResNet50-C4까지는 가중치를 공유하지만, 이후의 $\text{res5}$ 층은 인스턴스 추론용($\text{res5}_{ins}$)과 시맨틱 추론용($\text{res5}_{sem}$)으로 분리하여 각각의 특성에 맞는 특징을 추출하도록 설계하였다.

## 📊 Results

### 실험 설정
- **데이터셋**: Amazon Robotics Challenge (ARC2017)의 40종 물체를 사용하였다. 비디오 프레임에서 사람이 한 번만 라벨링하고 이를 역투영(Backproject)하는 방식으로 가시/폐색 마스크 데이터를 효율적으로 수집하였다. (총 505장 이미지)
- **지표**: Panoptic Quality (PQ)의 평균값인 $\text{mPQ}$를 사용하여 성능을 평가하였다.
- **비교 대상**: Instance-only 학습 모델(Baseline)과 제안한 Joint Learning 모델을 비교하였다.

### 정량적 결과
- **Joint Learning의 효과**: $\lambda = 0.25$일 때 $\text{mPQ}$가 $42.2$를 기록하여, Instance-only 모델($41.0$)보다 성능이 향상됨을 확인하였다.
- **데이터 증강(Data Augmentation)**: HSV 컬러 변경, Gaussian Blur, Affine Transform을 적용했을 때 $\text{mPQ}$가 $32.3$에서 $42.2$로 대폭 상승하여 그 효과가 입증되었다.
- **백본 네트워크**: ResNet101을 사용할 경우 Joint Learning 모델의 $\text{mPQ}$는 $44.5$로, ResNet50($42.2$)보다 더 높은 성능을 보였다.

### 로봇 작업 적용 결과
1.  **Random Picking**: 
    - 가려지지 않은(Fully visible) 물체만 우선적으로 선택하는 전략을 사용하였다.
    - 총 67회 시도 중 63회 성공(성공률 94.0%)하였으며, 폐색 영역 분할 결과가 충돌 방지에 효과적임을 보였다.
2.  **Target Picking**: 
    - 심하게 가려진 타겟 물체(예: 초록색 책)를 검출하고, 이를 가리고 있는 장애물 물체들을 먼저 제거한 후 타겟을 집어 올리는 작업을 성공적으로 수행하였다.

## 🧠 Insights & Discussion

본 논문은 Instance Segmentation의 국소적 추론(Box 기반)과 Semantic Segmentation의 전역적 추론(Image 기반)이 상호 보완 관계에 있음을 실험적으로 증명하였다. 특히 폐색 영역이라는 까다로운 목표에 대해 시맨틱 분할의 감독 신호가 인스턴스 분할의 Bounding Box 예측 능력을 간접적으로 향상시켰다는 점이 인상적이다.

**강점 및 한계**:
- **강점**: 추가적인 라벨링 비용 없이 기존 데이터를 활용해 성능을 높였으며, 실제 로봇 조작 환경에서 실용적인 가치를 증명하였다.
- **한계**: 데이터셋의 규모가 505장으로 다소 작으며, 특정 환경(ARC2017 물체들)에 한정된 실험이다. 또한 $\lambda$ 값에 따라 성능 변화가 존재하므로 최적의 하이퍼파라미터를 찾는 과정이 필요하다.
- **비판적 해석**: 시맨틱 분할의 Sigmoid 손실 함수를 통해 중첩 영역을 처리한 접근은 타당하나, 물체가 극단적으로 겹쳐 있어 형체를 알아보기 힘든 경우에 대한 강건성은 추가적인 분석이 필요할 것으로 보인다.

## 📌 TL;DR

본 연구는 로봇의 Pick-and-Place 작업을 위해 **인스턴스 분할(Instance Segmentation)**과 **시맨틱 분할(Semantic Segmentation)**을 통합 학습하여, 물체의 가려진 영역(Occluded region)을 정확히 예측하는 모델을 제안하였다. 이를 통해 Bounding Box 예측 오류로 인한 마스크 손실 문제를 해결하였으며, 실제 로봇 실험에서 **94%의 Random Picking 성공률**과 **심하게 가려진 타겟 물체의 성공적인 추출**을 달성하였다. 이 연구는 복잡한 Clutter 환경에서 로봇의 파지 계획 및 충돌 회피를 위한 시각 인지 시스템 구축에 중요한 기여를 한다.