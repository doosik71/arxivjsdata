# MaskUno: Switch-Split Block For Enhancing Instance Segmentation

Jawad Haidar, Marc Mouawad, Imad Elhajj, Daniel Asmar (2024)

## 🧩 Problem to Solve

본 논문은 Instance Segmentation에서 발생하는 **Competing Kernels(경쟁 커널)** 문제를 해결하고자 한다. Instance Segmentation은 단순히 픽셀을 분류하는 것을 넘어, 동일한 클래스의 개별 객체(instance)를 구분해야 하는 고도의 작업이다.

기존의 Mask R-CNN 기반 아키텍처들은 여러 클래스를 동시에 학습할 때, 각 클래스를 위한 커널들이 자신의 정확도를 최대화하기 위해 서로 경쟁하는 양상을 보인다. 즉, 단일한 Mask Prediction Branch 내에서 여러 클래스의 특징을 동시에 학습하려다 보니, 클래스 간의 상충 관계(trade-off)가 발생하여 각 클래스에 최적화된 풍부한 표현력을 갖추기 어렵다는 점이 문제로 지적된다.

논문의 목표는 Mask Prediction 단계를 클래스별로 전문화된 구조로 분리함으로써 이러한 커널 경쟁 문제를 완화하고, 결과적으로 Instance Segmentation의 정확도(mAP)를 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Switch-Split block**이라는 모듈형 구조를 제안하는 것이다.

기존의 다중 클래스 예측 헤드(multi-class prediction head)를 제거하고, 대신 분류 결과에 따라 해당 객체의 클래스에 맞는 전용 마스크 예측기로 경로를 배정하는 '스위치' 구조를 도입한다. 이를 통해 각 클래스는 독립적인 가중치를 가진 전용 커널을 통해 학습되므로, 클래스 간의 간섭 없이 각 객체의 특성에 최적화된 세그멘테이션 마스크를 생성할 수 있다.

## 📎 Related Works

논문에서는 Instance Segmentation의 성능 향상을 위한 기존 연구들을 세 가지 방향으로 분류하여 설명한다.

1. **Backbones**: 특징 추출 단계에서 공간 해상도(spatial resolution)를 유지하기 위한 연구들이다. SpineNet은 Neural Architecture Search를 통해 스케일 변환 특징을 학습하며, DetectoRS는 recursive FPN과 Switchable Atrous Convolutions를 통해 거시적/미시적 관점에서 특징을 재탐색한다.
2. **Object Detection**: 객체의 경계 박스(Bounding Box)를 찾는 단계이다. One-stage detector(YOLO, SSD, RetinaNet 등)는 속도가 빠르며, Two-stage detector(Faster R-CNN, Mask R-CNN 등)는 정확도가 높다. Mask R-CNN은 여기에 Mask prediction head와 ROI-Align 레이어를 추가하여 Instance Segmentation을 가능하게 했다.
3. **Cascade**: 결과물을 반복적으로 정제(refinement)하는 방식이다. Cascade Mask R-CNN이나 Hybrid Task Cascade(HTC)는 단계적인 정제를 통해 더 정확한 Bounding Box와 Mask를 생성한다.

**차별점**: 기존 연구들이 주로 백본 네트워크, 손실 함수, 혹은 캐스케이드 구조의 개선에 집중한 반면, MaskUno는 최종 예측 레이어에서 클래스 간의 그래디언트 충돌을 막기 위해 예측 경로를 분리하는 접근 방식을 취한다.

## 🛠️ Methodology

### 전체 시스템 구조 및 Switch-Split block

MaskUno는 기존 Instance Segmentation 모델의 터미널 블록(Terminal Block)을 수정한다. 일반적인 모델은 분류(Classifier), 경계 박스 회귀(BBox Regressor), 마스크 예측(Mask Predictor)의 세 가지 브랜치가 병렬적으로 작동하지만, MaskUno는 이를 다음과 같은 순차적/분기적 구조로 변경한다.

1. **Bounding Box Refinement**: 먼저 경계 박스 헤드가 정제된 ROI(Region of Interest)를 생성한다.
2. **ROI Align**: 정제된 ROI를 바탕으로 ROI-Align 레이어를 통과시켜 특징을 추출한다.
3. **Switch**: 분류기(Classifier)가 해당 ROI의 클래스를 결정하면, 이 정보가 '스위치' 역할을 하여 해당 클래스 전용의 Mask Head로 데이터를 전달한다.
4. **Split Block**: $N$개의 클래스가 있다면 $N+1$개의 전문화된 블록이 존재하며, 각 블록은 특정 클래스 하나만을 위해 학습된 전용 Mask Head를 가진다.

### 주요 방정식 및 손실 함수

각 블록은 다음의 세 가지 손실 함수를 최소화하도록 학습된다.

1. **분류 손실 (Categorical Cross-Entropy Loss)**:
    $$\text{L}_{\text{cls}} = -\sum_{i=1}^{N} y_i \log(\hat{y}_i)$$
    여기서 $y_i$는 실제 클래스 레이블, $\hat{y}_i$는 예측된 클래스 레이블이며, $N$은 클래스의 수이다.

2. **경계 박스 손실 (Smooth L1 Loss)**:
    $$\text{L}_1(x) = \begin{cases} 0.5x^2, & \text{if } |x| < 1 \\ |x| - 0.5, & \text{otherwise} \end{cases}$$
    예측값과 실제값의 차이 $x$에 대해 L1 손실을 완화하여 학습의 안정성을 높인다.

3. **마스크 손실 (Pixel-wise Binary Cross Entropy Loss)**:
    $$\text{L}_{\text{mask}} = -\sum_{i,j} [y_{\text{true}}(i, j) \log(y_{\text{pred}}(i, j)) + (1 - y_{\text{true}}(i, j)) \log(1 - y_{\text{pred}}(i, j))]$$
    각 픽셀이 객체에 속하는지 여부를 이진 분류하는 손실 함수이다.

### 학습 절차

중요한 점은 각 클래스의 Mask Head가 자신만의 독립적인 $\text{L}_{\text{mask}}^i$ 손실 함수를 가진다는 것이다. 이를 통해 각 클래스의 가중치가 비용(cost) 관점에서도 서로 독립적이게 되어, 커널 간의 경쟁을 근본적으로 차단한다. 모든 블록의 학습 스케줄과 하이퍼파라미터는 동일하게 유지하여 특정 클래스에 대한 오버피팅을 방지한다.

## 📊 Results

### 실험 설정

- **데이터셋**: COCO (Common Objects in Context) 데이터셋의 `train2017` 및 `val2017`을 사용한다.
- **측정 지표**: mAP(mean Average Precision)를 기본으로 하며, IoU 임계값 0.5 및 0.75($AP_{50}, AP_{75}$), 그리고 객체 크기별($AP_s, AP_m, AP_l$) 지표를 측정한다.
- **비교 모델**: Mask R-CNN, Cascade Mask R-CNN, Hybrid Task Cascade (HTC), DetectoRS.

### 주요 결과

1. **소규모 클래스 실험 (10개 클래스)**:
    - 네 가지 모델 모두에서 mAP가 **0.5%에서 5%까지 유의미하게 상승**하였다.
    - 다만, 'sheep' 클래스처럼 학습 샘플 수가 적은 경우(765개)에는 성능 향상이 미미하거나 약간 감소하는 경향을 보였다. 이는 데이터 양이 충분해야 전용 헤드의 효과가 나타남을 시사한다.

2. **대규모 클래스 실험 (80개 클래스)**:
    - **Mask R-CNN**: mAP가 **4.8% 상승**하여, 특정 클래스 선택에 관계없이 일반적인 성능 향상이 있음을 입증하였다.
    - **DetectoRS**: mAP가 **2.03% 상승**하였다. 이는 백본 개선이나 캐스케이드 구조와 MaskUno의 커널 경쟁 해결 방식이 상호 보완적임을 보여준다.

## 🧠 Insights & Discussion

### 강점

MaskUno는 모델 아키텍처에 구애받지 않고 적용 가능한 모듈형 구조이다. 특히, 기존의 고성능 모델(DetectoRS 등)에 적용했을 때 추가적인 성능 향상을 이끌어냈다는 점은, 그동안 Instance Segmentation 연구가 주로 백본이나 정제 과정에만 집중했지, 클래스 간의 커널 경쟁이라는 근본적인 학습 효율 문제를 간과했음을 보여준다.

### 한계 및 가정

본 논문의 실험 설정은 각 클래스별 양성 이미지(positive images)를 기준으로 평가되었다. 따라서 **음성 프레임(negative frames)**, 즉 객체가 없는 이미지에서 발생하는 False Positive(거짓 양성) 문제에 대해서는 충분히 다루지 않았다. 만약 스위치 단계에서 분류가 잘못될 경우, 엉뚱한 전문 헤드로 데이터가 전달되어 잘못된 마스크가 생성될 위험이 있다.

### 비판적 해석

제안된 방법은 성능을 높이지만, 클래스 수 $N$이 증가함에 따라 $N$개의 전용 Mask Head를 유지해야 하므로 모델의 파라미터 수와 메모리 사용량이 증가한다. 실시간 추론 환경에서의 오버헤드에 대한 분석이 누락된 점은 아쉬운 부분이다.

## 📌 TL;DR

MaskUno는 Instance Segmentation 모델에서 여러 클래스가 하나의 예측 헤드를 공유하며 발생하는 **'커널 경쟁(Competing Kernels)'** 문제를 해결하기 위해, 분류 결과에 따라 전용 마스크 예측기로 경로를 분기하는 **Switch-Split block**을 제안한다. 이 방법은 Mask R-CNN 기반의 다양한 모델에 적용 가능하며, COCO 데이터셋 기준 DetectoRS 모델에서 mAP를 **2.03%**, 기본 Mask R-CNN에서 **4.8%** 향상시키는 성과를 거두었다. 향후 Transformer 기반 모델로 확장하거나 경계 박스 회귀 단계까지 분리 적용한다면 더 높은 성능을 기대할 수 있을 것이다.
