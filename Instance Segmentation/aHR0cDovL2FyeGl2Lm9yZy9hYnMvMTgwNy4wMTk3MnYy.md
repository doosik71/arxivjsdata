# Beef Cattle Instance Segmentation Using Fully Convolutional Neural Network

Aram Ter-Sarkisov, John Kelleher, Bernadette Earley, Michael Keane, Robert Ross (2018)

## 🧩 Problem to Solve

본 논문은 축사 내의 소(Beef Cattle)를 개별적으로 분리하여 검출하는 Instance Segmentation 문제를 해결하고자 한다. 특히 CCTV 영상을 통해 수집된 데이터에서 개별 객체를 정확하게 구분하는 것은 다음과 같은 이유로 매우 어렵다.

첫째, 동물은 정적인 사물과 달리 서 있거나 누워 있는 등 자세를 빈번하게 변경하므로 형태의 변화가 심하며, 이에 따라 네트워크의 높은 일반화 능력이 요구된다. 둘째, 같은 품종의 소들은 털 색깔이나 외형적 특징이 매우 유사하여, 특히 부분적인 가려짐(Partial Occlusion)이 발생했을 때 개체를 구분하는 것이 매우 어렵다. 셋째, 좁은 공간에 여러 마리가 함께 머무는 환경 특성상 객체 간의 중첩(Occlusion)이 빈번하게 발생한다. 넷째, 인공 조명과 천장 틈으로 들어오는 자연광이 혼재되어 그림자나 빛 번짐이 발생하며, 다섯째, 소의 색상 패턴이 배경과 유사하여 객체와 배경을 구분하기 어려운 배경 혼입 문제가 존재한다.

따라서 본 연구의 목표는 Region of Interest(RoI) 예측 과정 없이도 Fully Convolutional Network(FCN)를 확장하여 이러한 까다로운 환경에서도 소의 인스턴스를 효과적으로 분할할 수 있는 알고리즘을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Bounding Box 예측이나 Region Proposal Network(RPN)를 사용하지 않고, FCN의 출력과 Ground Truth 마스크를 이용하여 마스크 표현을 정교화하는 **MaskSplitter** 프레임워크를 제안한 것이다.

MaskSplitter의 중심 아이디어는 FCN이 생성한 이진 마스크(Binary Mask)를 세 가지 유형의 예측값으로 분류하여 학습시키는 것이다. 구체적으로, 단 하나의 동물과 겹치는 'Good mask', 두 마리 이상의 동물과 겹치는 'Bad mask Type 1', 그리고 한 마리의 동물에 여러 개의 마스크가 겹치는 'Bad mask Type 2'로 구분한다. 이를 통해 네트워크가 단순히 객체 영역을 찾는 것을 넘어, 개별 인스턴스를 분리하는 능력을 학습하도록 설계하였다.

## 📎 Related Works

기존의 Instance Segmentation 알고리즘, 특히 Mask R-CNN과 같은 모델들은 일반적으로 3단계 접근 방식을 취한다. 먼저 RPN을 통해 객체가 있을 법한 후보 영역(Bounding Box)을 식별하고, 해당 영역의 클래스를 분류한 뒤, 마지막으로 FCN을 통해 마스크를 추출하는 방식이다. 이러한 방식은 현재 벤치마크 데이터셋에서 가장 높은 성능을 보이는 State-of-the-art(SOTA) 방법론으로 간주된다.

또 다른 접근 방식으로는 Bounding Box를 사용하지 않고 특징 클러스터링을 통해 인스턴스의 중심을 찾는 Instance Embedding 방식이 있다. 하지만 본 논문은 이러한 복잡한 다단계 구조나 임베딩 방식 대신, 개념적으로 더 단순한 FCN 기반의 정제 방식을 제안하며, 특히 RPN의 오버헤드 없이도 유사하거나 더 뛰어난 성능을 낼 수 있음을 보여줌으로써 기존 방식과 차별점을 둔다.

## 🛠️ Methodology

### 전체 시스템 구조
본 프레임워크는 **FCN8s**(VGG16 기반)를 Backbone 네트워크로 사용하여 이미지 크기의 스코어 맵을 추출한다. FCN8s의 출력물인 객체 및 배경 스코어 맵에 pixel-wise $\text{argmax}$를 적용하면 이진 마스크(Binary Mask)가 생성되며, 이 마스크 내의 분리된 픽셀 그룹들이 개별 인스턴스의 예측값이 된다. MaskSplitter는 이 예측 마스크들을 정제하여 최종적인 인스턴스 분할을 수행한다.

### MaskSplitter 알고리즘
MaskSplitter는 예측된 마스크($B$)와 실제 정답 마스크($C$) 사이의 IoU(Intersection over Union) 행렬을 계산하여 예측 마스크의 유형을 다음과 같이 정의한다.

1. **Good Prediction**: 예측 마스크가 정답 마스크 중 정확히 하나의 동물과만 겹치는 경우이다.
2. **Bad Prediction Type 1**: 하나의 예측 마스크가 두 마리 이상의 동물과 겹치는 경우이다.
3. **Bad Prediction Type 2**: 한 마리의 동물에 대해 두 개 이상의 예측 마스크가 겹쳐 발생하는 경우이다.

### 네트워크 아키텍처 및 학습 절차
FCN8s의 최종 스코어 맵은 $3 \times 3$ 합성곱 필터를 통해 통합 스코어 맵(Unified score map)으로 연결된다. 이후 독립적인 학습을 위해 이 통합 맵에서 세 개의 서로 다른 스코어 맵(Good, Bad Type 1, Bad Type 2)으로 분기되며, 각각 $3 \times 3$ 필터를 사용한다.

각 스코어 맵은 분리된 마스크와 픽셀 단위 곱셈(Pixel-wise product)을 수행하여 희소 레이어(Sparse layer)를 형성하고, 이는 다시 단일 유닛의 완전 연결 계층(Fully connected layer)으로 연결되어 각 유형의 예측 마스크 개수를 출력한다.

### 손실 함수 (Loss Function)
전체 네트워크의 손실 함수 $L$은 다음과 같이 정의된다.

$$L = L_{gs} + L_{good} + L_{badcows} + L_{badpreds}$$

여기서 각 항의 의미는 다음과 같다.
- $L_{gs}$: 'Good' 예측 스코어 맵에 대해 시그모이드 함수를 적용한 후 계산한 픽셀 단위 교차 엔트로피 손실(Pixel-wise cross-entropy loss)이다.
- $L_{good}$: 예측된 'Good' 마스크의 개수와 실제 동물 수 사이의 유클리드 거리($L_2$) 손실이다.
- $L_{badcows}$: 한 마스크가 두 마리 이상의 동물과 겹치는 경우(Type 1)에 대한 유클리드 손실이다.
- $L_{badpreds}$: 한 마리 동물에 여러 마스크가 겹치는 경우(Type 2)에 대한 유클리드 손실이다.

## 📊 Results

### 실험 설정
- **데이터셋**: MS COCO 2017, Pascal VOC 2012, 그리고 직접 구축한 CCTV 소 데이터셋(약 5,000여 장의 크롭 이미지)을 사용하였다.
- **비교 대상**: Mask R-CNN, FCIS, MNC 등 SOTA 모델들과 비교하였다.
- **지표**: $\text{AP}@0.5$, $\text{AP}@0.7$, $\text{AP}@0.5:0.95$를 측정 지표로 사용하였다.
- **학습 환경**: Adam 옵티마이저, 학습률 $0.00001$, Tesla K40m GPU 환경에서 20,000회 반복 학습하였다.

### 주요 결과
1. **Pascal VOC 2012 (Cow)**: 제안 방법인 FCN8s+MaskSplitter가 $\text{AP}@0.5$ 기준 $0.656$을 기록하며, Mask R-CNN($0.642$)보다 약 3%p 높은 성능을 보였다.
2. **MS COCO 2017 (Cow)**: Mask R-CNN과 FCIS가 더 높은 성능을 보였다. 저자들은 이를 아키텍처의 차이보다는 ResNet101이라는 더 깊은 Backbone 네트워크를 사용했기 때문이라고 분석하였다.
3. **CCTV 소 데이터셋**:
   - **Fine-tuning 전**: 모든 SOTA 모델들이 낮은 성능을 보였으며, 이는 벤치마크 데이터셋과 실제 축사 환경의 데이터 간의 간극이 매우 크기 때문임을 시사한다.
   - **Fine-tuning 후**: FCN8s+MaskSplitter가 $\text{mAP}$ 기준 Mask R-CNN보다 8% 이상 높은 성능을 보였으며, $\text{AP}@0.5$에서도 2.5%p 더 높은 결과를 기록하였다.

## 🧠 Insights & Discussion

본 연구는 Instance Segmentation에서 필수적으로 여겨지던 RPN과 Bounding Box 예측 단계를 제거하고도, 특정 도메인(축사 환경)에서 매우 효율적인 성능을 낼 수 있음을 입증하였다. 특히 일반적인 벤치마크 데이터셋보다 실제 현장의 데이터에서 Mask R-CNN보다 뛰어난 성능을 보였다는 점은, 복잡한 다단계 구조보다 객체-배경 분리 능력을 극대화한 정제 방식이 특정 환경의 'Object-vs-Background' 문제에 더 적합할 수 있음을 시사한다.

다만, 실험에 사용된 데이터가 단 하나의 급식 구역(Single feedlot)에서 수집된 것이라는 점은 데이터 다양성 측면에서 한계로 작용한다. 또한, 현재의 구조로는 완전히 겹쳐진 객체를 분리하는 데 한계가 있을 수 있다. 향후 연구로는 비디오 데이터의 특성을 활용하기 위한 순환 신경망(Recurrent component)의 도입과 각 마스크를 개별적으로 학습하는 맵을 구축하는 방안이 제시되었다.

## 📌 TL;DR

본 논문은 RPN이나 Bounding Box 예측 없이 FCN의 출력을 정제하여 인스턴스를 분리하는 **MaskSplitter** 프레임워크를 제안하였다. 이 방법은 특히 가려짐과 배경 혼입이 심한 CCTV 축사 환경에서 Mask R-CNN과 같은 SOTA 모델보다 뛰어난 성능을 보였으며, 이는 실시간 가축 모니터링 및 복지 개선 시스템에 적용될 가능성이 높다.