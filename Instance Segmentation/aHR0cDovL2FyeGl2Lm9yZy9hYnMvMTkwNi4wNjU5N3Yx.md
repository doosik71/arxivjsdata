# IMP: Instance Mask Projection for High Accuracy Semantic Segmentation of Things

Cheng-Yang Fu, Tamara L. Berg, Alexander C. Berg (2019)

## 🧩 Problem to Solve

본 논문은 픽셀 수준의 정밀한 Semantic Segmentation을 달성하는 것을 목표로 한다. 특히, 배경과 같은 'Stuff' 영역보다 움직일 수 있는 전경 객체인 'Things'의 세그멘테이션 정확도를 높이는 데 집중한다.

기존의 Semantic Segmentation 방식은 고정된 스케일의 공간적 문맥을 사용하여 각 픽셀의 클래스를 독립적으로 결정해야 하므로, 객체의 전체적인 형태나 스케일을 고려하는 능력이 부족하다. 반면, Object Detection 및 Instance Segmentation은 객체를 하나의 단위로 인식하고 스케일을 명시적으로 추정하므로 객체의 존재 여부나 카테고리를 결정하는 데 더 유리하다. 하지만 Instance Segmentation 결과(특히 Mask R-CNN의 $28 \times 28$ 마스크)는 해상도가 낮아 경계선이 부정확하고 오목한(concave) 모양을 처리하는 데 한계가 있다.

따라서 본 연구의 목표는 Instance Segmentation의 '객체 단위 인식 능력'과 Semantic Segmentation의 '고해상도 픽셀 예측 능력'을 결합하여, 특히 복잡한 레이어링, 큰 변형, 작은 객체가 포함된 환경에서 높은 정확도의 Semantic Segmentation을 구현하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 **Instance Mask Projection (IMP)** 이라는 새로운 연산자를 제안한 것이다.

IMP의 핵심 직관은 Instance Segmentation 모델(예: Mask R-CNN)이 예측한 인스턴스 마스크를 단순한 결과물로 사용하는 것이 아니라, 이를 하나의 **특성 맵(Feature Map)으로 투영(Project)**하여 Semantic Segmentation 네트워크의 보조 입력으로 사용하는 것이다. 이를 통해 네트워크는 객체의 위치와 대략적인 형태에 대한 강력한 사전 정보(Strong Prior)를 제공받게 되며, 학습 과정에서 세부적인 픽셀 경계를 정교화하는 데 집중할 수 있게 된다.

## 📎 Related Works

### 관련 연구 및 한계
1. **Object Localization & Instance Segmentation**: R-CNN 시리즈와 Mask R-CNN 등이 발전하며 정교한 마스크 예측이 가능해졌으나, 이는 주로 개별 객체의 분리에 집중하며 전체 이미지의 픽셀 단위 분류(Semantic Segmentation)와는 별개로 작동하는 경우가 많았다.
2. **Semantic Segmentation**: FCN, Dilated Convolution, Encoder-Decoder 구조(U-Net, SegNet) 등이 제안되었다. 최근에는 CRF(Conditional Random Fields)나 Graphical Model을 사용하여 경계선을 날카롭게 만들려는 시도가 있었으나, 과도한 평활화(Smoothing)로 인해 작은 객체가 사라지는 문제가 발생했다.
3. **Combined Detection & Semantic Segmentation**: Panoptic Segmentation의 등장으로 두 작업을 통합하려는 시도가 늘어났다. Panoptic FPN은 FPN 구조를 통해 두 작업을 동시에 수행한다.

### 기존 접근 방식과의 차별점
본 논문은 UPSNet과 같이 마스크 투영을 사용하는 기존 연구와 차별화된다. UPSNet은 투영된 마스크를 사용하여 각 픽셀에서 인스턴스 마스크와 세그멘테이션 결과 중 어느 것을 선택할지 결정(Decision making)하는 방식으로 사용하지만, IMP는 투영된 마스크를 **학습 가능한 특성(Feature)**으로 사용하여 세그멘테이션 성능 자체를 향상시키는 직교적인(Orthogonal) 개선 방식을 취한다.

## 🛠️ Methodology

### 전체 파이프라인
전체 시스템은 크게 세 단계로 구성된다:
1. **Instance Detection**: Mask R-CNN 또는 Panoptic FPN을 통해 객체의 클래스, 점수, Bounding Box, 그리고 인스턴스 마스크를 예측한다.
2. **Instance Mask Projection (IMP)**: 예측된 마스크들을 이미지 전체 크기의 캔버스(Canvas)로 투영하여 특성 층을 생성한다.
3. **Semantic Segmentation**: IMP를 통해 생성된 캔버스를 FPN의 특성 맵들과 결합(Concatenate)하여 최종 세그멘테이션 결과를 도출한다.

### Instance Mask Projection (IMP) 상세 설명
IMP 연산자는 각 클래스 $c$에 대해 독립적인 캔버스 층을 생성한다. 각 예측된 인스턴스 마스크 $M_i$는 해당 객체의 검출 점수 $S^i$와 곱해져 가중치가 부여된다. 캔버스의 특정 위치 $p_{xy}$에 대한 값은 다음과 같은 방정식으로 계산된다.

$$\text{canvas}(c, p_{xy}) = \max (\text{canvas}(c, p_{xy}), S^i M^i(\text{pre}_i(p_{xy})))$$

여기서 $\text{pre}_i$는 캔버스의 좌표를 인스턴스 마스크 $M_i$의 좌표로 매핑하는 함수이다. $\max$ 연산을 통해 여러 인스턴스가 겹칠 경우 가장 신뢰도가 높은(점수가 높은) 마스크 값이 유지된다. 최종적으로 이 캔버스는 $\frac{1}{4}$ 스케일(또는 설정된 스케일)의 특성 맵 형태로 생성된다.

### 모델 아키텍처 변형
논문은 IMP를 적용한 여러 모델 변형을 제시한다:
- **Mask R-CNN-IMP**: 추가 학습 없이 IMP 결과만으로 세그멘테이션을 수행한다.
- **Panoptic-P2-IMP**: FPN의 $P2$ 레이어 특성과 IMP 캔버스를 결합한다.
- **Panoptic-FPN-IMP (최종 모델)**: FPN의 $P2, P3, P4, P5$ 레이어를 모두 업샘플링하여 합산(Sum)한 후, IMP 캔버스와 결합하여 최종 예측 헤드로 전달한다.

### 학습 절차
학습은 두 단계로 진행된다:
1. **1단계**: Mask R-CNN을 사용하여 객체 검출 및 인스턴스 세그멘테이션 모델을 먼저 학습시킨다.
2. **2단계**: 1단계 모델을 초기값으로 사용하여 전체 네트워크(IMP 및 Semantic Segmentation 헤드 포함)를 End-to-End로 학습시킨다. 이는 IMP 결과가 학습 초기 단계에서 너무 크게 변하여 수렴 속도가 느려지는 것을 방지하기 위함이다.

## 📊 Results

### 실험 설정
- **데이터셋**: Varied Clothing Dataset (VCP), ModaNet, Cityscapes.
- **지표**: mean IOU (mIOU), mean Accuracy (mAcc).
- **비교 대상**: DeepLabV3+, Panoptic FPN, Mask R-CNN 및 그 변형 모델들.

### 주요 결과
1. **Varied Clothing Dataset (VCP)**:
   - Panoptic-FPN-IMP는 기존 Panoptic-FPN 대비 mIOU를 약 2.02 포인트 향상시켰다.
   - 특히 단순한 Mask R-CNN-IMP만으로도 Semantic-FPN과 유사한 성능을 보였다.
2. **ModaNet**:
   - 가장 극적인 성능 향상이 나타났다. 기존 DeepLabV3+의 mIOU 51%에서 **71.4%로 절대적인 성능 향상(20.4%)**을 달성했다.
   - 특히 벨트, 선글라스, 모자 등 크기가 작은 객체에서 매우 높은 성능 향상을 보였다.
3. **Cityscapes**:
   - 'Thing' 클래스(사람, 자동차, 트럭 등)에 집중하여 분석한 결과, mIOU가 약 3.2 ~ 4.2 포인트 향상되었다.
   - 특히 데이터 샘플 수가 적거나 크기가 작은 클래스(Train, Bus, Motorcycle 등)에서 효과가 뚜렷했다.

### 추론 속도
IMP 연산은 매우 효율적이며, 베이스라인 모델 위에 추가했을 때 추론 시간에 미치는 영향은 **약 1~2ms**에 불과하여 실시간 성능에 거의 영향을 주지 않는다.

## 🧠 Insights & Discussion

### 강점 및 분석
- **작은 객체 및 모호한 클래스 해결**: Semantic Segmentation은 고정 스케일 윈도우를 사용하므로 작은 객체를 놓치기 쉽지만, IMP는 Object Detection의 가변 스케일 컨텍스트를 활용하므로 작은 객체 인식률을 획기적으로 높였다. 또한, 드레스 하단과 스커트를 구분하는 것과 같이 모호한 클래스 구분에서도 객체 단위의 컨텍스트가 결정적인 도움을 준다.
- **일관성 유지**: 객체 내부의 픽셀들이 서로 다른 클래스로 예측되는 '카테고리 혼동' 문제를 해결하여 더 깨끗한(clean) 결과를 생성한다.

### 한계 및 비판적 해석
- **경계선 정밀도 문제**: 분석 결과, 객체의 중심부에서는 성능 향상이 뚜렷하지만 경계선(boundary) 근처에서는 IMP의 효과가 감소한다. 이는 Mask R-CNN이 생성하는 마스크의 해상도가 $28 \times 28$로 매우 낮아, 이를 투영했을 때 경계선이 뭉개지기 때문이다. 이를 보완하기 위해 최종 모델에서는 FPN 특성을 결합하여 고해상도 경계선을 복원하도록 설계하였다.
- **Stuff 클래스에 대한 낮은 효과**: 본 방법론은 'Things'에 최적화되어 있어, 도로(road)나 하늘(sky)과 같은 'Stuff' 클래스에서는 성능 향상이 미미하다. 이는 방법론의 설계 의도와 일치하지만, 범용적인 세그멘테이션 도구로서의 확장성에는 제약이 있다.

## 📌 TL;DR

본 논문은 인스턴스 세그멘테이션 마스크를 세그멘테이션 네트워크의 특성 맵으로 투영하는 **Instance Mask Projection (IMP)** 연산자를 제안하였다. 이 방법은 Object Detection의 '객체 단위 인식' 능력을 Semantic Segmentation에 주입함으로써, 특히 **작은 객체의 인식률을 높이고 클래스 간의 일관성을 확보**하는 데 탁월한 효과를 보인다. ModaNet에서 mIOU를 20.4%나 향상시키는 획기적인 결과를 보였으며, 추론 시간 증가가 거의 없어 실제 자율주행이나 패션 분석 시스템 등 다양한 실무 환경에 즉시 적용 가능성이 매우 높은 연구이다.