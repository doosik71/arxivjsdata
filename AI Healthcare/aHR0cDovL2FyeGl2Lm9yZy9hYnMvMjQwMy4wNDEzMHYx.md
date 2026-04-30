# An Explainable AI Framework for Artificial Intelligence of Medical Things

Al Amin, Kamrul Hasan, Saleh Zein-Sabatto, Deo Chimba, Imtiaz Ahmed, Tariqul Islam (2024)

## 🧩 Problem to Solve

본 논문은 의료 사물 인터넷 인공지능(Artificial Intelligence of Medical Things, AIoMT) 환경에서 딥러닝 모델의 복잡성으로 인해 발생하는 '블랙박스' 문제, 즉 의사결정 과정의 불투명성을 해결하고자 한다. 의료 분야, 특히 뇌종양 진단과 같은 치명적인 질환의 경우, AI가 내린 진단 결과에 대한 근거가 명확하지 않으면 의료진이 이를 신뢰하기 어렵고 실제 임상에 적용하는 데 한계가 있다. 

따라서 본 연구의 목표는 뇌종양 검출을 위한 고성능의 딥러닝 앙상블 모델을 구축하는 동시에, 이를 해석 가능하게 만드는 설명 가능한 AI(Explainable AI, XAI) 프레임워크를 설계하여 진단의 정확성과 투명성을 동시에 확보하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같이 요약할 수 있다.

1. **AIoMT 전용 XAI 프레임워크 설계**: 의료 데이터의 특성을 고려하여 LIME, SHAP, Grad-CAM과 같은 다양한 XAI 기법을 통합한 프레임워크를 제안하였다.
2. **앙상블 기반의 고정밀 진단 모델**: VGG16, VGG19, InceptionV3, ResNet50, DenseNet121 등 5가지의 서로 다른 CNN 모델을 결합한 Maximum Voting Classifier(다수결 투표 분류기)를 통해 진단 성능을 극대화하였다.
3. **Cloud-Edge 협업 구조 제안**: 대규모 데이터 학습 및 저장을 위한 Cloud 계층과 실시간 진단을 수행하는 Edge 계층으로 구성된 2단계 계층 구조를 통해 실제 의료 환경에서의 적용 가능성을 높였다.

## 📎 Related Works

기존의 뇌종양 진단 연구들은 주로 전이 학습(Transfer Learning)이나 다양한 CNN 아키텍처를 활용하여 정확도를 높이는 데 집중해 왔다. 예를 들어, 일부 연구에서는 VGG16을 미세 조정하여 98.69%의 높은 정확도를 달성하거나, Vision Transformer와 앙상블 모델을 결합하여 96.94%의 성능을 보인 사례가 있다.

그러나 이러한 기존 접근 방식들은 다음과 같은 한계점을 가진다.
- **설명 가능성의 부족**: 높은 정확도를 보임에도 불구하고, 모델이 왜 그런 판단을 내렸는지에 대한 근거를 제시하지 못하는 경우가 많았다.
- **실시간 적용의 어려움**: 대용량 데이터를 처리하는 과정에서 계산 복잡도가 증가하여 실시간 구현에 제약이 있었다.
- **데이터 의존성**: 특정 데이터셋에 과적합되거나 데이터 확장 시 성능 향상의 한계가 명시되었다.

본 논문은 이러한 한계를 극복하기 위해 Cloud-Edge 기반의 인프라를 도입하고, 여러 XAI 기법을 동시에 적용하여 성능과 투명성을 모두 잡고자 하였다.

## 🛠️ Methodology

### 1. 시스템 아키텍처 (Cloud-Edge Structure)
시스템은 크게 두 개의 계층으로 구성된다.
- **Cloud 계층**: 원시 MRI 데이터셋을 안전하게 저장하고, PCA(Principal Component Analysis)를 이용한 차원 축소 및 데이터 전처리를 수행한다. 또한, 연산 자원이 풍부한 클라우드에서 앙상블 모델의 학습과 검증이 이루어진다.
- **Edge 계층**: MRI 기기와 직접 연결되어 실시간으로 영상을 획득한다. 클라우드에서 학습된 모델을 배포받아 실시간으로 종양 유무를 분류한다.

### 2. Maximum Voting Classifier (앙상블 모델)
본 논문은 5개의 서로 다른 CNN 모델($M_1, \dots, M_n$)을 사용하는 앙상블 기법을 채택하였다. 각 모델의 예측 결과 중 가장 많이 등장한 클래스를 최종 결과로 선택하는 다수결 방식을 사용한다.

- **수학적 정의**: 
  입력 데이터 $X$에 대해 $M$개의 분류 규칙 $h(X)$가 있을 때, 최종 결정 $C_X$는 다음과 같다.
  $$C_X = \text{mode}\{h_1(X), \dots, h_M(X)\}$$
  가중치를 적용한 경우, 다음과 같이 가중치 합이 최대인 클래스를 선택한다.
  $$C_X = \arg \max_i \sum_{j=1}^B w_j I(h_j(X) = i)$$

- **손실 함수**: 모델 최적화를 위해 Binary Cross-Entropy (BCE) 손실 함수를 사용하였다.
  $$L = -\frac{1}{n} \sum_{i=1}^n [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)]$$
  여기서 $y_i$는 실제 라벨, $\hat{y}_i$는 모델이 예측한 확률이다.

- **사용된 베이스 모델**: VGG16, VGG19, InceptionV3, ResNet50, DenseNet121.

### 3. XAI 프레임워크의 구성
진단 결과의 근거를 시각화하기 위해 세 가지 기법을 사용한다.

- **LIME (Local Interpretable Model-Agnostic Explanations)**: 특정 입력 샘플 주변에 섭동(perturbation)을 주어 국소적인 선형 모델을 학습시킴으로써, 어떤 픽셀이 결정에 영향을 주었는지 설명한다.
  $$y' = \beta_0 + \beta_1 x'_1 + \dots + \beta_n x'_n + \varepsilon$$
- **SHAP (SHapley Additive exPlanations)**: 게임 이론의 Shapley value를 활용하여 각 특징(픽셀)이 최종 예측값에 기여한 정도를 정량적으로 계산한다.
  $$\phi_i(f) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(|N|-|S|-1)!}{|N|!} [f(S \cup \{i\}) - f(S)]$$
- **Grad-CAM (Gradient-weighted Class Activation Mapping)**: 마지막 컨볼루션 레이어의 그래디언트 정보를 활용하여, 모델이 주목한 영역을 히트맵(Heatmap) 형태로 시각화한다.
  $$L_c = \sum_{i,j} \alpha_k^c A_{i,j}^k$$

## 📊 Results

### 1. 실험 환경
- **데이터셋**: Kaggle의 Br35H 데이터셋 (종양 2590장, 정상 500장).
- **데이터 분할**: 학습 데이터 2472장, 검증 데이터 618장.
- **평가 지표**: Precision, Recall, F1-Score, Accuracy, Loss.

### 2. 정량적 결과
앙상블 모델은 개별 모델들보다 전반적으로 우수한 성능을 보였다.
- **정확도**: 학습 정확도 $99\%$, 검증 정확도 $98\%$를 달성하였다.
- **성능 지표**: Precision $98\%$, Recall $97\%$, F1-Score $99\%$로 매우 높은 수치를 기록하였다.
- **Confusion Matrix 분석**: 검증 데이터셋에서 True Positive 514건, True Negative 93건을 기록하였으며, False Positive 7건, False Negative 4건으로 오분류가 매우 적음을 확인하였다.

### 3. 정성적 결과
SHAP, LIME, Grad-CAM을 통해 생성된 시각화 결과, 모델이 뇌종양의 실제 위치와 특징적인 영역을 정확히 포착하여 진단하고 있음을 확인하였다. 이는 AI의 판단 근거가 의학적 소견과 일치함을 시사하며, 의료진의 신뢰도를 높일 수 있는 결과이다.

## 🧠 Insights & Discussion

본 논문은 단순한 모델 성능 향상을 넘어, 의료 AI에서 가장 중요한 **'신뢰성(Trust)'**과 **'해석 가능성(Interpretability)'** 문제를 정면으로 다루었다. 

**강점:**
- 여러 CNN 모델의 장점을 결합한 앙상블 기법을 통해 단일 모델의 편향(Bias)을 줄이고 일반화 성능을 높였다.
- 서로 다른 원리로 작동하는 세 가지 XAI 기법을 통합 적용하여, 상호 보완적인 설명력을 제공하였다.
- Cloud-Edge 구조를 통해 실제 의료 현장에서 발생할 수 있는 데이터 처리 지연 및 보안 문제를 고려한 아키텍처를 제시하였다.

**한계 및 비판적 해석:**
- **데이터 불균형**: 사용된 Br35H 데이터셋의 클래스 비율(종양 vs 정상)이 약 5:1로 불균형하다. 앙상블 모델이 높은 정확도를 보였으나, 불균형 데이터에 대한 구체적인 처리 기법(예: Oversampling 등)에 대한 설명이 부족하다.
- **실제 환경 검증 부재**: 제안된 Cloud-Edge 구조가 개념적으로는 훌륭하나, 실제 하드웨어 환경에서 지연 시간(Latency)이나 처리 속도를 측정한 정량적 지표가 제시되지 않았다.
- **종양 종류의 단순성**: 본 연구는 '종양 유무'라는 이진 분류(Binary Classification)에 집중하고 있다. 실제 임상에서는 종양의 종류(Malignant vs Benign)나 등급을 나누는 것이 더 중요하므로 이에 대한 확장이 필요하다.

## 📌 TL;DR

본 연구는 뇌종양 진단의 정확성과 투명성을 높이기 위해 **5종의 CNN 모델을 결합한 앙상블 분류기**와 **LIME, SHAP, Grad-CAM 기반의 XAI 프레임워크**를 Cloud-Edge 구조에 통합한 AIoMT 시스템을 제안하였다. 실험 결과 **검증 정확도 98%**라는 뛰어난 성능과 함께 진단 근거를 시각적으로 제시함으로써 의료진이 신뢰할 수 있는 AI 진단 도구의 가능성을 입증하였다. 이 연구는 향후 다양한 질환으로의 확장 및 실시간 의료 모니터링 시스템 구축에 중요한 기반이 될 수 있다.