# Panoptic Segmentation with a Joint Semantic and Instance Segmentation Network

Daan de Geus, Panagiotis Meletis, Gijs Dubbelman (2019)

## 🧩 Problem to Solve

본 논문은 이미지 내의 모든 요소를 인식하기 위한 Panoptic Segmentation 문제를 해결하고자 한다. 일반적으로 이미지 속의 요소는 셀 수 있는 객체인 'things'(예: 차량, 사람)와 셀 수 없는 배경 요소인 'stuff'(예: 하늘, 도로)로 구분된다. 

기존의 Instance Segmentation은 'things'의 검출과 마스크 예측에 집중하여 'stuff'를 무시하며, Semantic Segmentation은 모든 픽셀에 클래스 레이블을 부여하지만 동일 클래스 내의 개별 인스턴스를 구분하지 못한다는 한계가 있다. Panoptic Segmentation은 이 두 가지의 장점을 결합하여 모든 픽셀에 대해 클래스 레이블과 인스턴스 ID를 동시에 부여하는 것을 목표로 한다. 특히, 기존의 베이스라인 방식이 두 개의 독립된 네트워크를 사용하여 결과를 결합하는 방식이었기에 발생하는 연산 효율성 저하와 메모리 낭비 문제를 해결하고, 단일 네트워크를 통한 효율적인 학습 및 추론 구조를 제안하는 것이 본 연구의 주요 목표이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 단일 네트워크 구조인 JSIS-Net(Joint Semantic and Instance Segmentation Network)을 제안하여 Panoptic Segmentation을 수행했다는 점이다. 

가장 중심적인 설계 아이디어는 Semantic Segmentation 브랜치와 Instance Segmentation 브랜치가 공통의 Feature Extractor(ResNet-50)를 공유하도록 설계하여 Joint Training을 수행하는 것이다. 이를 통해 두 네트워크를 각각 학습시키고 예측하는 것보다 속도가 빠르고 메모리 효율적이며, 결과적으로 단일 네트워크 내에서 두 가지 작업을 동시에 최적화할 수 있는 가능성을 제시하였다.

## 📎 Related Works

Panoptic Segmentation의 개념을 정립한 Kirillov 등[8]의 연구에서는 가장 성능이 좋은 독립적인 Instance Segmentation 네트워크와 Semantic Segmentation 네트워크의 출력을 단순한 휴리스틱(Heuristics)으로 결합하는 베이스라인 방법을 제시하였다. 또한, Depth Layering이나 방향 예측을 통해 인스턴스를 구분하려는 시도[13]나, 외부 객체 검출기와 내부 세그멘테이션 네트워크를 결합하는 Dynamically Instantiated Network[1]와 같은 연구들이 존재하였다.

본 논문의 접근 방식은 이러한 기존 방식들과 달리, 공유된 백본(Backbone)을 가진 단일 네트워크 구조에서 두 작업을 동시에 수행하며, 학습 단계에서부터 Joint Learning을 적용했다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조
JSIS-Net은 크게 공통의 Feature Extractor와 두 개의 병렬 브랜치, 그리고 최종 결과를 병합하는 휴리스틱 단계로 구성된다.

1.  **Feature Extractor**: ResNet-50을 사용하여 이미지로부터 특징 맵을 추출하며, 이 특징 맵은 두 브랜치에서 공유된다.
2.  **Semantic Segmentation Branch**: 추출된 특징 맵에 Pyramid Pooling Module(PPM)을 적용하여 전역 문맥 정보를 반영한다. 이후 Deconvolution과 Bilinear Interpolation을 결합한 Hybrid Upsampling을 통해 입력 이미지 크기로 복원하여 픽셀 단위의 클래스 맵을 생성한다.
3.  **Instance Segmentation Branch**: Mask R-CNN 구조를 기반으로 한다. Region Proposal Network(RPN)가 객체 후보 영역을 제안하면, 해당 영역의 특징을 추출하여 클래스 분류(Classification), 경계 상자 회귀(Bounding Box Regression), 인스턴스 마스크(Instance Mask) 예측을 병렬로 수행한다.

### 훈련 목표 및 손실 함수
네트워크의 Joint Learning을 위해 각 브랜치에서 발생하는 손실 함수를 가중 합산한 단일 손실 함수 $L_{tot}$를 정의하여 사용한다.

$$L_{tot} = \lambda_1 L_{rpn,obj} + \lambda_2 L_{rpn,reg} + \lambda_3 L_{det,cls} + \lambda_4 L_{det,reg} + \lambda_5 L_{mask} + \lambda_6 L_{seg} + \lambda_7 R$$

여기서 각 항의 의미는 다음과 같다.
- $L_{rpn,obj}$: RPN의 객체 존재 여부(Objectness)에 대한 Softmax Cross-Entropy 손실.
- $L_{rpn,reg}$: RPN의 영역 제안 상자에 대한 Smooth L1 손실.
- $L_{det,cls}$: 객체 검출의 클래스 분류에 대한 Softmax Cross-Entropy 손실.
- $L_{det,reg}$: 검출된 객체 경계 상자의 위치에 대한 Smooth L1 손실.
- $L_{mask}$: 인스턴스 마스크 예측에 대한 Sigmoid Cross-Entropy 손실.
- $L_{seg}$: 세그멘테이션 결과에 대한 Sparse Softmax Cross-Entropy 손실.
- $R$: 모델 파라미터에 대한 $L_2$ 정규화 항.
- $\lambda_{1 \dots 7}$: 각 손실의 균형을 맞추기 위한 튜닝 파라미터이다.

### 병합 휴리스틱 (Merging Heuristics)
두 브랜치의 출력물을 하나의 Panoptic Map으로 합치기 위해 다음의 두 가지 충돌 해결 전략을 사용한다.

1.  **인스턴스 마스크 중첩 해결**: 여러 인스턴스 마스크가 동일한 픽셀을 점유할 경우, 해당 픽셀에서 가장 높은 확률값을 가진 인스턴스 마스크에 픽셀을 할당한다.
2.  **Things 클래스 예측 충돌 해결**: 
    - 먼저 Semantic Segmentation 결과에서 'things' 클래스를 모두 제거하여 'stuff' 전용 맵을 만든다.
    - 그 후, Instance Segmentation 브랜치에서 예측된 'things' 결과를 'stuff' 맵 위에 덮어씌운다. 즉, 인스턴스 예측 결과에 우선순위를 둔다.
    - 마지막으로, 예측된 'stuff' 클래스의 전체 픽셀 수가 4096개 미만인 경우, 현실적으로 가능성이 낮다고 판단하여 해당 영역을 다음으로 확률이 높은 'stuff' 클래스로 대체한다.

## 📊 Results

### 실험 설정 및 지표
- **데이터셋**: Mapillary Vistas, Microsoft COCO.
- **평가 지표**: Panoptic Quality (PQ)를 주 지표로 사용하며, 이는 Segmentation Quality (SQ)와 Recognition Quality (RQ)로 분해되어 분석된다.
- **비교 대상**: 각 브랜치를 독립적으로 학습시킨 모델과 Joint Training 모델을 비교하였다.

### 주요 결과
- **정량적 성능**: COCO test-dev set에서 27.2 PQ, Mapillary Vistas validation set에서 17.6 PQ를 달성하였다.
- **Joint Training 효과**: Mapillary Vistas 데이터셋 실험 결과, Joint Training 모델이 독립 학습 모델보다 mIoU(34.7), $mAP_{0.5}$(8.4), PQ(17.4) 모든 지표에서 우수한 성능을 보였다. 이는 공유 백본을 통한 공동 최적화가 효과적임을 시사한다.
- **RPN 성능 분석**: Mapillary Vistas 데이터셋에서 'things' 클래스의 PQ가 'stuff'보다 현저히 낮게 나타났다. 분석 결과, RPN의 Mean Recall이 COCO(0.827)에 비해 Mapillary Vistas(0.363)에서 매우 낮게 나타났으며, 이는 RPN이 본 시스템의 성능 병목(Bottleneck)임을 보여준다.

## 🧠 Insights & Discussion

본 논문은 단일 네트워크를 통한 Panoptic Segmentation의 효율성을 입증하였다. 특히 Joint Training이 독립적인 학습보다 더 나은 성능을 낼 수 있음을 정량적으로 확인한 점이 강점이다.

그러나 성능 면에서 Kirillov 등이 제시한 베이스라인보다는 낮은 수치를 기록하였다. 저자들은 이에 대해 사용된 개별 모듈(ResNet-50, Mask R-CNN 등)의 복잡도가 낮고, 하이퍼파라미터 튜닝이 최적화되지 않았을 가능성을 언급한다. 특히 Mapillary Vistas와 같은 복잡한 도로 장면 데이터셋에서는 RPN이 객체를 충분히 제안하지 못하는 문제가 심각하게 나타났다. 

비판적으로 해석하자면, 본 연구는 구조적 통합에는 성공하였으나, Panoptic Segmentation이라는 복합적인 과제를 해결하기 위해 단순히 기존의 두 모델을 합치는 수준에 머물렀다. RPN의 낮은 리콜률 문제를 해결하기 위한 추가적인 아키텍처 개선(예: FPN 도입 등)이 이루어지지 않은 점이 한계로 판단된다.

## 📌 TL;DR

본 연구는 ResNet-50 백본을 공유하는 단일 네트워크(JSIS-Net)를 통해 Semantic Segmentation과 Instance Segmentation을 동시에 수행하고, 이를 휴리스틱으로 결합하여 Panoptic Segmentation을 구현하였다. Joint Training이 독립 학습보다 효율적이고 성능이 좋음을 확인하였으나, 특히 복잡한 데이터셋에서 RPN의 낮은 검출 성능이 전체 PQ 점수를 낮추는 병목 현상으로 작용함을 발견하였다. 이 연구는 향후 완전한 End-to-End Panoptic Segmentation 네트워크로 가기 위한 효율적인 학습 구조의 기초를 제시한다.