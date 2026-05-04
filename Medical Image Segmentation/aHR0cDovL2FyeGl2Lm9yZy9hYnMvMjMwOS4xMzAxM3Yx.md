# Performance Analysis of UNet and Variants for Medical Image Segmentation

Walid Ehab, Yongmin Li (2023)

## 🧩 Problem to Solve

본 연구는 의료 영상 분석의 핵심 단계인 의료 영상 분할(Medical Image Segmentation)에서 딥러닝 모델, 특히 UNet 및 그 변형 모델들의 성능을 종합적으로 분석하는 것을 목표로 한다. 

의료 영상 분할은 장기, 종양, 혈관과 같은 유의미한 영역을 식별하여 픽셀 단위로 라벨링하는 작업이다. 이는 조기 질병 발견과 정확한 진단 및 치료 계획 수립에 필수적이지만, 다음과 같은 여러 기술적 난제가 존재한다:
- **영상 모달리티의 다양성**: CT, MRI, 초음파, PET, X-ray 등 각 촬영 방식마다 조직 강도, 노이즈, 아티팩트 등의 특성이 달라 맞춤형 접근 방식이 필요하다.
- **클래스 불균형(Class Imbalance)**: 특히 종양 분할의 경우, 전체 영상에서 타겟 영역이 차지하는 비중이 매우 작아 모델이 배경으로 오분류하기 쉽다.
- **정밀한 경계 획정**: 복잡한 형태의 경계나 미세한 세부 사항을 정확하게 포착해야 하는 정밀도가 요구된다.

따라서 본 논문은 UNet, Res-UNet, Attention Res-UNet 세 가지 아키텍처를 뇌종양, 폴립, 심장 분할이라는 세 가지 서로 다른 의료 영상 작업에 적용하여 그 성능을 정량적으로 비교 평가하고자 한다.

## ✨ Key Contributions

본 연구의 중심적인 아이디어는 표준 UNet 아키텍처에 **Residual Connection(잔차 연결)**과 **Attention Mechanism(주의 집중 메커니즘)**을 단계적으로 추가했을 때, 복잡한 의료 영상 분할 성능이 어떻게 향상되는지를 실험적으로 검증하는 것이다.

- **Res-UNet**: Residual Block을 도입하여 네트워크의 층이 깊어짐에 따라 발생하는 Vanishing Gradient 문제를 완화하고, 더 복잡한 특징을 효과적으로 학습하도록 설계하였다.
- **Attention Res-UNet**: Attention Gate를 추가하여 입력 이미지에서 중요한 영역에 선택적으로 집중하고 노이즈나 무관한 특징을 억제함으로써, 특히 미세한 세부 사항과 클래스 불균형 문제를 해결하고자 하였다.

## 📎 Related Works

### 기존 접근 방식 및 한계
1. **저수준 영상 처리 방식**: Thresholding, Region-based segmentation, Watershed transform 등이 사용되었으나, 이는 단순한 작업에는 적합하지만 복잡한 의료 영상에서는 한계가 명확하다.
2. **통계적 방법**: K-means clustering, Active contours(Snakes), Probabilistic modelling, Graph cut, Markov Random Field(MRF), Level-set method 등이 제안되었으나, 여전히 복잡한 해부학적 구조를 처리하는 데 어려움이 있다.
3. **딥러닝 기반 방법**: CNN, FCN, SegNet 등이 등장하며 비약적인 발전을 이루었으며, 특히 UNet은 U-자형 구조와 Skip Connection을 통해 의료 영상 분할의 표준이 되었다.

### 기존 연구와의 차별점
본 논문은 특정 새로운 아키텍처를 제안하는 것보다, 기존의 검증된 모델들(UNet $\rightarrow$ Res-UNet $\rightarrow$ Attention Res-UNet)을 다양한 데이터셋과 손실 함수 환경에서 비교 분석함으로써, 실제 구현 시 어떤 모델이 어떤 상황(예: 극심한 클래스 불균형 상황)에서 더 유리한지에 대한 가이드라인을 제공한다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 1. 네트워크 아키텍처

#### (1) UNet
- **구조**: Encoder(Contracting Path)와 Decoder(Expansive Path)의 대칭 구조로 이루어져 있다.
- **Encoder**: 4개의 인코딩 레이어로 구성되며, 각 레이어는 2개의 Convolution $\rightarrow$ Batch Normalization $\rightarrow$ ReLU 활성화 함수로 이루어진 블록과 Max-pooling 다운샘플링 층으로 구성된다.
- **Decoder**: Up-sampling과 Transposed Convolution을 통해 원래 해상도를 복구한다.
- **Skip Connection**: Encoder의 특징 맵을 Decoder에 직접 연결하여, 다운샘플링 과정에서 손실된 공간 정보와 세부 사항을 보존한다.

#### (2) Res-UNet
- **구조**: UNet의 기본 Convolution Block을 **Residual Block**으로 대체하였다.
- **핵심 기법**: 입력값과 마지막 $3 \times 3$ Convolution 층의 출력을 더하는 덧셈 층(Addition layer)을 추가하여 Gradient가 더 원활하게 흐르도록 함으로써 더 깊은 네트워크 학습을 가능하게 한다.

#### (3) Attention Res-UNet
- **구조**: Res-UNet 기반에 **Attention Block**과 **Gating Signal**을 추가하였다.
- **작동 원리**:
    - **Gating Signal**: 하위 레이어의 출력을 현재 레이어의 차원과 맞춘 신호이다.
    - **Attention Block**: 입력 특징 맵 $x$와 Gating signal $g$를 결합하여 공간적 주의 가중치(Attention weights)를 계산한다.
    - **연산 과정**: $\text{Spatially transformed } x \rightarrow \text{Gating signal transformation } \phi_g \rightarrow \text{ReLU} \rightarrow \text{Sigmoid} \rightarrow \text{Upsampling} \rightarrow \text{Element-wise multiplication with } x$.
    - 이를 통해 모델은 중요한 지역에 집중하고 불필요한 배경 노이즈를 억제한다.

### 2. 훈련 목표 및 손실 함수

분할 작업의 특성에 따라 두 가지 손실 함수를 사용하였다.

- **Binary Focal Loss (BFL)**: 뇌종양 및 폴립 분할과 같은 이진 분류 작업에서 클래스 불균형을 해결하기 위해 사용한다.
$$BFL = -(1 - p_t)^\gamma \cdot \log(p_t)$$
여기서 $\gamma$는 Focusing parameter로, $\gamma$가 높을수록 모델이 분류하기 어려운(Hard) 샘플에 더 많은 가중치를 두어 학습하게 한다.

- **Categorical Focal Cross-Entropy (CFC)**: 심장 분할과 같은 다중 클래스 작업에서 사용한다.
$$CFC(y, p) = -\sum_{i=1}^N \alpha_i \cdot (1 - p_i)^\gamma \cdot y_i \cdot \log(p_i)$$
표준 Categorical Cross-entropy에 Focal loss의 개념을 결합하여, 소수 클래스(Minority class)에 더 집중하도록 설계되었다.

### 3. 성능 측정 지표
- **Dice Similarity Coefficient (DSC)**: 예측 마스크 $A$와 정답 마스크 $B$ 사이의 겹침 정도를 측정한다.
$$DSC = \frac{2 \times |A \cap B|}{|A| + |B|}$$
- **Intersection over Union (IoU)**: 합집합 대비 교집합의 비율을 측정한다.
$$IoU = \frac{|A \cap B|}{|A \cup B|}$$
- 기타 지표: Precision, Recall, Accuracy, Execution Time 등을 함께 측정하였다.

## 📊 Results

본 연구는 세 가지 서로 다른 데이터셋을 통해 모델을 검증하였다.

### 1. Brain Tumor Segmentation (LGG MRI)
- **특징**: 극심한 클래스 불균형 존재 (종양 영역이 매우 작음).
- **결과**: **Res-UNet**이 Dice(0.931)와 IoU(0.870)에서 가장 높은 성능을 보였으며, **Attention Res-UNet**은 Recall(0.946)에서 가장 우수한 성적을 거두어 종양 영역을 놓치지 않고 포착하는 능력이 뛰어남을 보였다. UNet은 종양을 배경으로 오분류하는 경향(False Negative)이 강했다.

### 2. Polyp Segmentation (Colonoscopy)
- **특징**: 폴립의 크기와 모양이 매우 불규칙함.
- **결과**: 역시 **Res-UNet**이 Dice(0.838)와 IoU(0.721)에서 최고 성능을 기록했다. Attention Res-UNet은 Recall(0.788)에서 강점을 보였다. 전반적으로 UNet보다 변형 모델들이 정밀한 경계 획정에 우수하였다.

### 3. Heart Segmentation (ACDC MRI)
- **특징**: 다중 클래스(Background, RV, Myocardium, LV) 분할. 클래스 불균형이 상대적으로 적음.
- **결과**: 클래스별로 강점이 달랐다.
    - **RV Cavity (Class 1)**: Res-UNet이 가장 우수한 Precision/Recall 및 Dice를 기록했다.
    - **LV Cavity (Class 3)**: Attention Res-UNet이 가장 높은 Precision/Recall을 기록했다.
    - **전체 정확도 및 손실**: UNet이 Accuracy(98.41%)와 Loss(1.00%) 면에서 소폭 우세하거나 유사한 성능을 보였는데, 이는 데이터셋의 불균형이 덜해 표준 UNet으로도 충분한 성능이 나왔음을 시사한다.

## 🧠 Insights & Discussion

### 강점 및 해석
- **Residual Connection의 효과**: Res-UNet과 Attention Res-UNet이 전반적으로 UNet보다 우수한 성능을 보인 것은 Residual Connection이 Vanishing Gradient 문제를 해결하여 더 깊고 복잡한 특징을 학습할 수 있게 했기 때문이다.
- **Attention Mechanism의 효과**: Attention Res-UNet이 일관되게 높은 Recall 값을 기록한 것은, 주의 집중 메커니즘이 클래스 불균형 상황에서 타겟 영역을 더 정밀하게 찾아내도록 돕기 때문으로 분석된다.
- **지표의 함정**: 저자는 Accuracy가 매우 높게 나오더라도(예: 99% 이상), 타겟 영역이 매우 작은 의료 영상에서는 실제 분할 성능을 대표하지 못하므로 Dice나 IoU와 같은 지표가 더 중요함을 강조하였다.

### 한계 및 비판적 해석
- **데이터의 한계**: 심장 분할 작업에서 RV cavity(Class 1)의 성능이 다른 클래스보다 낮게 나타난 것은 해당 클래스의 학습 데이터가 부족했기 때문이며, 이는 모델 구조의 문제라기보다 데이터셋의 불균형 문제임을 명시하였다.
- **차원 확장 필요성**: 본 실험은 2D 슬라이스 기반으로 진행되었으나, 의료 영상은 본질적으로 3D 데이터이다. 따라서 3D UNet이나 3D Convolution의 적용 여부에 대한 분석이 추가되었다면 더 실무적인 가이드라인이 되었을 것이다.

## 📌 TL;DR

본 논문은 의료 영상 분할에서 **UNet $\rightarrow$ Res-UNet $\rightarrow$ Attention Res-UNet**으로 이어지는 아키텍처 확장이 성능에 미치는 영향을 세 가지 서로 다른 의료 작업(뇌종양, 폴립, 심장)을 통해 분석하였다.

- **결론**: 단순 UNet은 작은 타겟이나 복잡한 경계에서 한계가 있으며, **Res-UNet은 정밀도(Precision, Dice)** 면에서, **Attention Res-UNet은 재현율(Recall) 및 클래스 불균형 해결** 면에서 탁월한 성능을 보인다.
- **실무적 시사점**: 타겟 영역이 매우 작고 클래스 불균형이 심한 데이터셋을 다룰 때는 **Attention Res-UNet** 구조와 **Focal Loss**의 조합이 가장 권장되는 전략이다.