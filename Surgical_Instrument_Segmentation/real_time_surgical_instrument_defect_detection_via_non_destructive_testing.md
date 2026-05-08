# Real-Time Surgical Instrument Defect Detection via Non-Destructive Testing

Qurrat Ul Ain, Atif Aftab Ahmed Jilani, Zunaira Shafqat, Nigar Azhar Butt (2024)

## 🧩 Problem to Solve

수술 도구의 결함은 멸균 상태 유지, 기계적 무결성 및 환자의 안전에 심각한 위험을 초래하며, 수술 중 합병증 가능성을 높인다. 그러나 현재 수술 도구 제조 공정의 품질 관리는 주로 수동 육안 검사에 의존하고 있다. 수동 검사는 검사자의 숙련도나 상태에 따라 주관적이고 일관성이 없으며, 특히 미세 균열(micro-cracks), 작은 기공(tiny pores), 초기 단계의 부식(early-stage corrosion)과 같은 미세한 결함을 탐지하는 데 한계가 있어 인적 오류가 발생할 가능성이 매우 높다.

따라서 본 논문의 목표는 수동 검사의 주관성과 비효율성을 극복하고, ISO 13485 및 FDA 표준을 준수할 수 있도록 하는 자동화된 실시간 결함 탐지 프레임워크인 **SurgScan**을 개발하는 것이다. 이를 통해 제조 공정의 확장성을 확보하고, 검사 오류로 인한 경제적 손실과 의료 사고 위험을 최소화하고자 한다.

## ✨ Key Contributions

본 연구의 핵심 기여는 다음과 같다.

1. **고해상도 수술 도구 결함 데이터셋 구축**: 11종의 수술 도구와 5가지 주요 결함 카테고리를 포함하는 대규모 데이터셋을 구축하였다. 원본 이미지 8,573장을 데이터 증강(Data Augmentation)을 통해 총 102,876장으로 확장하여 모델의 강건성과 일반화 성능을 높였다.
2. **YOLOv8 기반의 실시간 탐지 프레임워크 설계**: 실시간 추론 속도와 정확도를 동시에 확보하기 위해 YOLOv8 아키텍처를 채택하였으며, 이를 수술 도구 및 결함 분류 작업에 최적화하였다.
3. **전처리 및 증강 기법의 통계적 검증**: Chi-Square 테스트와 ANOVA 분석을 통해 데이터 증강이 결함 분포의 균형을 맞추는 데 기여했음을 증명하고, 특히 Contrast-enhanced 전처리가 결함 탐지 정확도를 유의미하게 향상시킨다는 점을 통계적으로 입증하였다.
4. **산업적 적용 가능성 제시**: 기존의 CNN 기반 모델들과 비교하여 압도적인 추론 속도($4.2\text{--}5.8 \text{ ms}$)와 높은 정확도($99.3\%$)를 달성함으로써, 실제 산업 현장의 고속 생산 라인에 배포 가능한 솔루션을 제시하였다.

## 📎 Related Works

### 기존 접근 방식 및 한계

1. **전통적 이미지 처리 (Traditional Image Processing)**: Canny edge detection, Hough transforms, contour analysis 등의 규칙 기반 방식이 사용되었다. 하지만 이러한 방법은 조명 변화, 표면 반사, 결함 모양의 복잡성에 매우 취약하다.
2. **머신러닝 기반 방식 (Machine Learning)**: SVM, Random Forest, KNN 등의 분류기가 제안되었으나, 이는 수동으로 설계된 특징 추출(Handcrafted feature extraction)에 크게 의존하며 실시간 대규모 검사에 필요한 확장성이 부족하다.
3. **딥러닝 기반 CNN (Convolutional Neural Networks)**: ResNet, EfficientNet 등이 등장하며 특징 자동 추출이 가능해졌고 정확도가 향상되었다. 그러나 대부분의 CNN 모델은 추론 시 여러 번의 forward pass가 필요하거나 연산량이 많아, 실시간 제조 환경에 적용하기에는 지연 시간(Latency)이 너무 길다는 단점이 있다.
4. **객체 탐지 모델 (Object Detection)**: YOLO, SSD와 같은 Single-shot detector들이 등장하여 추론 속도를 획기적으로 줄였다. 하지만 수술 도구와 같이 반사율이 높은 금속 표면에서의 미세 결함을 정밀하게 탐지하는 연구는 여전히 부족한 실정이다.

### SurgScan의 차별점

SurgScan은 범용 산업 데이터셋이 아닌, 실제 수술 도구에 특화된 고해상도 전문 데이터셋을 사용하며, 단순한 분류(Classification)를 넘어 실시간 탐지 및 국지화(Detection and Localization)를 목표로 하여 산업적 실용성을 극대화하였다.

## 🛠️ Methodology

### 전체 시스템 구조

SurgScan은 두 단계의 순차적 파이프라인으로 구성된 모듈형 구조를 가진다.

1. **수술 도구 분류 (Instrument Classification)**: 입력 이미지에서 어떤 종류의 수술 도구인지 먼저 식별한다.
2. **결함 분류 (Defect Classification)**: 식별된 도구 종류에 특화된 결함 탐지 모델을 선택하여 해당 도구의 결함 여부와 종류를 판별한다.

### 주요 구성 요소 및 절차

#### 1. 전처리 (Preprocessing)

입력 이미지의 품질을 최적화하기 위해 다음 과정을 거친다.

- **Unsharp Masking**: 에지와 텍스처를 강조하여 스크래치나 부식 같은 미세 결함의 가시성을 높인다.
- **Resizing**: 모든 이미지를 $1024 \times 1024$ 픽셀로 통일한다.
- **Normalization**: 픽셀 값을 $[0, 1]$ 범위로 스케일링하여 조명 변화의 영향을 줄이고 학습 안정성을 높인다.

#### 2. 모델 아키텍처 및 학습 설정

- **백본**: ImageNet으로 사전 학습된 **YOLOv8**을 사용한다.
- **Fine-tuning 전략**: 저수준 특징(low-level features)을 유지하기 위해 첫 9개 레이어를 동결(freeze)하고, 나머지 심층 레이어를 수술 도구 및 결함 특성에 맞게 미세 조정한다.
- **학습 파라미터**:
  - Optimizer: Adam ($\eta = 0.001$)
  - Loss Function: 분류를 위한 Cross-Entropy Loss를 사용한다.
  - Batch Size: 16
  - Epochs: 30 (Early stopping patience = 5 적용)
  - Regularization: Dropout (0.3), L2 weight decay (0.0005), Batch Normalization을 적용하여 과적합을 방지한다.

#### 3. 추론 및 후처리 (Inference & Post-processing)

모델의 신뢰도를 높이기 위해 **Confidence-based Filtering**을 적용한다.

- **도구 분류**: 신뢰도 점수가 $50\%$ 미만인 경우, 자동 분류를 중단하고 '수동 검토(manual review)' 대상으로 분류한다.
- **결함 분류**: 결함 탐지 신뢰도가 낮을 경우, False Positive(오탐)를 줄이기 위해 기본적으로 '결함 없음(No Defect Detected)'으로 처리한다.

## 📊 Results

### 실험 설정

- **데이터셋**: 11종의 도구(Scissors, Scalpel, Forceps 등)와 5종의 결함(Crack, Cuts, Pores, Scratches, Corrosion)으로 구성된 102,876장의 이미지.
- **비교 모델**: ResNet152, ResNext101, EfficientNet-b4, YOLOv5.
- **평가 지표**: Accuracy, Precision, Recall, F1-score, mAP (mean Average Precision), Inference Time (ms).

### 주요 결과

1. **정량적 성능**:
    - SurgScan(YOLOv8)은 테스트 셋에서 **$99.39\%$의 높은 정확도**를 기록하였으며, Precision($99.36\%$)과 F1-score($99.32\%$)에서도 다른 모델들을 압도하였다.
    - 특히 EfficientNet-b4가 $99.07\%$로 근소하게 뒤따랐으나, Recall 측면에서 YOLOv8이 더 우수한 성능을 보였다.
2. **연산 효율성**:
    - **추론 속도**: 이미지당 **$4.2\text{--}5.8 \text{ ms}$**로 매우 빠르게 처리하여 실시간 산업 배포에 적합함을 보였다.
    - 비교 모델인 ResNet152($15.3 \text{ ms}$)나 EfficientNet($8.1 \text{ ms}$)보다 월등히 빠른 속도를 기록하였다.
3. **통계적 분석**:
    - **Chi-Square Test**: 데이터 증강 후 결함 분포의 균형이 통계적으로 유의미하게 개선되었음을 확인하였다 ($p < 0.001$).
    - **ANOVA Test**: 밝기(Brightness)나 선명도(Sharpness)보다 **대비(Contrast) 조정**이 결함 분류 정확도 향상에 가장 결정적인 영향을 미친다는 것을 입증하였다 ($p < 0.05$).

## 🧠 Insights & Discussion

### 강점

- **속도와 정확도의 균형**: YOLOv8을 활용하여 실시간성(Real-time)과 고정밀도(High-precision)라는 두 마리 토끼를 모두 잡았다.
- **데이터 기반의 체계적 접근**: 단순히 모델을 돌려본 것이 아니라, 전처리 기법의 효과를 ANOVA 등 통계적 방법으로 검증하여 방법론의 근거를 확실히 하였다.
- **산업적 실용성**: 2단계 분류 구조(도구 $\rightarrow$ 결함)를 통해 각 도구별 특화된 결함 탐지가 가능하도록 설계하여 실제 공정 적용 가능성을 높였다.

### 한계 및 비판적 해석

- **저대비 결함의 탐지**: 결과 분석에서 언급되었듯, 아주 희미한 스크래치(low-contrast scratches)는 여전히 '결함 없음'으로 오분류되는 경우가 있다. 이는 전처리 단계에서 Contrast 조정을 더 정교하게 하거나, 더 고해상도의 센서가 필요함을 시사한다.
- **통제된 환경의 한계**: 데이터셋이 통제된 포토박스(photo box) 환경에서 촬영되었으므로, 실제 공장의 가변적인 조명이나 배경 소음이 존재하는 환경에서는 성능 하락이 발생할 수 있다. (논문의 'External Threats' 섹션에서 언급됨)
- **연속 공정 통합 문제**: 현재는 단일 이미지 기반의 검사이며, 컨베이어 벨트와 같은 연속 흐름 시스템에 통합하기 위한 실시간 비디오 스트림 처리 최적화에 대한 구체적인 구현 내용은 부족하다.

## 📌 TL;DR

본 논문은 수술 도구 제조 공정의 수동 검사 한계를 극복하기 위해 YOLOv8 기반의 실시간 결함 탐지 프레임워크인 **SurgScan**을 제안한다. 10만 장 규모의 전문 데이터셋을 구축하고, 2단계 분류 파이프라인과 Contrast 최적화 전처리를 통해 **$99.3\%$의 정확도와 $5 \text{ ms}$ 내외의 빠른 추론 속도**를 달성하였다. 이는 기존 CNN 모델 대비 효율성이 극대화된 결과이며, 향후 의료 기기 제조 분야의 자동화된 품질 관리 시스템(QC) 구축에 핵심적인 역할을 할 것으로 기대된다.
