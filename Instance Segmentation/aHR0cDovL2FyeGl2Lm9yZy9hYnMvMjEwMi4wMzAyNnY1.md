# Instance and Panoptic Segmentation Using Conditional Convolutions

Zhi Tian, Bowen Zhang, Hao Chen, Chunhua Shen (2022)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 인스턴스 분할(Instance Segmentation)에서 기존의 지배적인 방식인 Mask R-CNN과 같은 ROI(Region of Interest) 기반 방법론들이 가진 한계점을 극복하는 것이다. ROI 기반 방식은 다음과 같은 세 가지 주요 문제점을 가지고 있다.

첫째, ROI는 일반적으로 축에 정렬된(axis-aligned) 바운딩 박스 형태이므로, 불규칙한 모양의 객체를 처리할 때 배경이나 다른 인스턴스와 같은 불필요한 정보가 과도하게 포함될 수 있다. 둘째, 고정된 가중치를 가진 Mask Head가 전경과 배경을 구분해야 하므로 강력한 수용 영역(Receptive Field)이 필요하며, 이는 연산 복잡도를 증가시켜 추론 시간이 인스턴스의 개수에 따라 크게 변하는 결과를 초래한다. 셋째, 서로 다른 크기의 ROI를 배치 연산(batched computation)에 활용하기 위해 강제로 리사이징(resizing)하는 과정이 필요한데, 이 과정에서 큰 객체의 세밀한 경계 정보가 손실되어 출력 마스크의 해상도가 제한된다.

결과적으로 본 논문의 목표는 ROI 연산과 리사이징 과정 없이도 개별 인스턴스를 효과적으로 구분할 수 있는 완전히 합성곱 기반의 네트워크(Fully Convolutional Network, FCN) 프레임워크를 구축하여, 정확도와 추론 속도를 동시에 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 고정된 가중치를 사용하는 Mask Head 대신, 예측하려는 인스턴스에 따라 가중치가 동적으로 변하는 **Dynamic Conditional Convolutions(동적 조건부 합성곱)**를 도입하는 것이다.

중심적인 직관은 인스턴스마다 서로 다른 특성(모양, 위치, 외관 등)을 가지고 있으므로, 각 인스턴스에 특화된 필터를 동적으로 생성하여 적용한다면, 해당 필터가 오직 그 인스턴스의 픽셀에만 반응하게 만들 수 있다는 것이다. 이를 통해 FCN이 가지는 '유사한 외관의 객체를 구분하지 못하는 문제'를 해결하고, ROI 크롭 과정 없이 전체 피처 맵에서 직접 마스크를 예측할 수 있게 한다.

## 📎 Related Works

기존의 인스턴스 분할 연구는 크게 ROI 기반 방법과 FCN 기반 방법으로 나뉜다. Mask R-CNN으로 대표되는 ROI 기반 방법은 성능은 뛰어나지만 앞서 언급한 연산 효율성과 해상도 저하 문제가 있다. 반면, InstanceFCN과 같은 초기 FCN 기반 시도들은 유사한 외관을 가진 서로 다른 인스턴스를 구분하는 데 어려움을 겪어 성능이 ROI 기반 방법보다 낮았다. 

최근의 YOLACT나 BlendMask는 ROI 검출과 마스크 예측 피처 맵을 분리하여 속도를 높였으나, 여전히 근본적인 구조적 한계가 존재한다. 본 연구는 Dynamic Filter Networks나 CondConv의 개념을 인스턴스 분할에 도입하여, 인스턴스별로 조건화된 필터를 생성함으로써 기존 FCN의 한계를 극복하고 Mask R-CNN의 성능을 뛰어넘는 차별점을 가진다.

## 🛠️ Methodology

### 1. 전체 시스템 구조
CondInst는 기본적으로 FCOS(Fully Convolutional One-Stage object detector)를 기반으로 하며, 여기에 필터 파라미터를 생성하는 **Controller**와 이를 사용하는 **Dynamic Mask Head**를 추가한 구조이다. 전체 파이프라인은 크게 세 부분으로 구성된다.

- **Backbone & FPN**: ResNet과 FPN을 통해 다중 스케일의 피처 맵 $\{P_3, P_4, P_5, P_6, P_7\}$을 추출한다.
- **Bottom Branch**: $P_3, P_4, P_5$를 결합하여 고해상도 피처 맵 $F_{bottom}$을 생성한다. 이 피처 맵은 최종 마스크 예측의 입력으로 사용되며, 연산량 감소를 위해 채널 수를 $C_{bottom}=8$로 매우 낮게 설정한다.
- **Controller & Mask Head**: FPN의 각 레벨에서 인스턴스의 클래스, 바운딩 박스와 함께 해당 인스턴스를 위한 마스크 헤드의 필터 파라미터 $\theta_{x,y}$를 동적으로 생성한다.

### 2. 상세 구성 요소 및 절차
- **Dynamic Mask Head**: 컨트롤러가 생성한 $\theta_{x,y}$를 가중치로 사용하는 매우 가벼운 FCN이다. 3개의 $1 \times 1$ 합성곱 층(각 8채널)으로 구성되며, 마지막 층은 Sigmoid 함수를 통해 픽셀별 전경/배경 확률을 예측한다.
- **Relative Coordinates**: CNN 피처 맵은 위치 정보가 부족하므로, $F_{bottom}$에 필터가 생성된 지점을 원점으로 하는 **상대 좌표(Relative Coordinates)** 맵을 결합하여 입력한다. 이는 모델이 인스턴스의 중심으로부터의 상대적 위치를 인식하게 하여 마스크의 형태를 더 정확하게 예측하게 돕는다.
- **Panoptic Segmentation 확장**: 인스턴스 분할 브랜치에 추가적인 세만틱 분할(Semantic Segmentation) 브랜치를 결합함으로써 파놉틱 분할로 확장한다.

### 3. 주요 방정식 및 학습 목표
전체 손실 함수 $L_{overall}$은 다음과 같이 정의된다.
$$L_{overall} = L_{fcos} + \lambda L_{mask} + \mu L_{pano}$$
여기서 $L_{fcos}$는 FCOS의 분류, 박스 회귀, Center-ness 손실이며, $\lambda=1, \mu=0.5$이다.

인스턴스 마스크 손실 $L_{mask}$는 다음과 같이 계산된다.
$$L_{mask}(\{\theta_{x,y}\}) = \frac{1}{N_{pos}} \sum_{x,y} \mathbb{1}_{\{c^*_{x,y} > 0\}} L_{dice}(M_{x,y}, M^*_{x,y})$$
여기서 $M_{x,y}$는 동적 필터 $\theta_{x,y}$를 사용하여 예측한 마스크이며, $L_{dice}$는 전경-배경 불균형 문제를 해결하기 위한 **Dice Loss**를 사용한다.
$$L_{dice}(M, M^*) = 1 - \frac{2 \sum_{i,j} M_{i,j} M^*_{i,j}}{\sum_{i,j} (M_{i,j})^2 + \sum_{i,j} (M^*_{i,j})^2}$$

또한, 위치의 신뢰도를 평가하기 위한 Center-ness 점수는 다음과 같이 정의된다.
$$\text{centerness}^* = \sqrt{\frac{\min(l^*, r^*)}{\max(l^*, r^*)} \cdot \frac{\min(t^*, b^*)}{\max(t^*, b^*)}}$$

## 📊 Results

### 1. 실험 설정
- **데이터셋**: MS-COCO 및 Cityscapes.
- **기준선(Baseline)**: Mask R-CNN, YOLACT, TensorMask, SOLOv2 등.
- **지표**: Mask AP (Average Precision), PQ (Panoptic Quality), FPS (Frames Per Second).

### 2. 주요 결과
- **정확도 및 속도**: MS-COCO test-dev 세트에서 CondInst는 기본 Mask R-CNN보다 높은 정확도(35.3% vs 34.6% AP)를 보이면서도 추론 속도는 더 빠르다(49ms vs 65ms).
- **고해상도 마스크**: ROI 리사이징이 없으므로 Mask R-CNN의 $28 \times 28$ 해상도보다 훨씬 높은 해상도의 마스크를 생성하여 경계선 표현이 더 정교하다.
- **실시간 성능**: CondInst-RT 모델은 R-50 백본 기준 43 FPS의 속도를 내면서 YOLACT++보다 높은 AP(36.0% vs 34.1%)를 달성하였다.
- **파놉틱 분할**: COCO 데이터셋에서 PQ 46.1% (ResNet-101 기준)를 달성하여 기존 Panoptic-FPN(40.9%)을 크게 상회하는 SOTA 성능을 보였다.

## 🧠 Insights & Discussion

### 1. 강점 및 분석
본 논문의 가장 큰 강점은 **'인스턴스 정보의 인코딩 방식'**을 바꾼 것이다. 기존 방법이 바운딩 박스라는 기하학적 영역으로 인스턴스를 한정했다면, CondInst는 필터 파라미터 자체에 인스턴스의 외곽선(contour) 정보를 인코딩한다. 실험을 통해 상대 좌표만으로도 대략적인 윤곽을 잡을 수 있음이 확인되었으며, 여기에 $F_{bottom}$ 피처를 더해 세부 디테일을 완성하는 구조임을 밝혔다.

### 2. 한계 및 논의사항
- **메모리 문제**: 학습 시 모든 긍정 샘플(positive locations)에 대해 마스크 손실을 계산하면 메모리 소모가 극심하다. 이를 해결하기 위해 점수가 높은 상위 64개 위치만 선택하는 전략을 사용하였는데, 이는 일종의 근사치이므로 모든 샘플을 학습시키지 못하는 한계가 있다.
- **NMS 의존성**: 개념적으로는 바운딩 박스가 필요 없지만, 실제 추론 시에는 계산 효율을 위해 박스 기반 NMS를 사용한다. 마스크 기반 NMS를 사용할 경우 성능은 비슷하지만 속도가 2배 이상 느려지는 트레이드-오프가 존재한다.

## 📌 TL;DR

CondInst는 고정된 Mask Head 대신 인스턴스별로 최적화된 필터를 실시간으로 생성하는 **Conditional Convolution**을 도입하여 인스턴스 분할 문제를 해결하였다. 이를 통해 ROI 크롭과 리사이징 과정을 완전히 제거함으로써 **추론 속도 향상, 마스크 해상도 증가, 그리고 정확도 개선**이라는 세 마리 토끼를 모두 잡았다. 이 연구는 ROI 기반의 패러다임을 벗어나 FCN만으로도 SOTA 수준의 인스턴스 및 파놉틱 분할이 가능함을 입증하였으며, 향후 실시간 고정밀 세그멘테이션 연구에 중요한 베이스라인이 될 것으로 보인다.