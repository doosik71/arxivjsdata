# Multi-Stage Segmentation and Cascade Classification Methods for Improving Cardiac MRI Analysis

Vitalii Slobodzian, Pavlo Radiuk, Oleksander Barmak and Iurii Krak (2024)

## 🧩 Problem to Solve

본 연구는 심장 자기공명영상(Cardiac Magnetic Resonance Imaging, MRI) 분석에서 발생하는 심장 구조의 정확한 분할(Segmentation)과 질환 분류(Classification)의 어려움을 해결하고자 한다.

심장 MRI는 비침습적이고 고해상도 영상을 제공하여 심장 진단의 '골드 표준'으로 여겨지지만, 다음과 같은 기술적 난관이 존재한다. 첫째, 호흡 및 심장 박동으로 인한 지속적인 움직임이 영상에 아티팩트(Artifact)를 유발하여 이미지 선명도를 저하시킨다. 둘째, 심장의 복잡한 해부학적 구조와 금속 임플란트 등으로 인한 왜곡이 정확한 해석을 방해한다.

기존의 딥러닝 기반 접근 방식들은 이러한 아티팩트에 취약하거나, 복잡한 케이스에서 분할 성능이 떨어지는 한계가 있으며, 이는 결국 최종적인 질환 분류의 정확도 저하로 이어진다. 따라서 본 논문의 목표는 다단계 분할 프로세스와 계층적 분류 구조를 도입하여, 임상 의사결정을 지원할 수 있는 수준의 높은 정확도를 가진 심장 MRI 분석 방법을 설계하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 설계 아이디어는 분할과 분류 과정을 세분화하여 단계적으로 정확도를 높이는 것이다.

1. **다단계 분할 전략(Multi-stage Segmentation):** 단순한 단일 모델 분할이 아니라, 구조별 위치 찾기(Localization) $\rightarrow$ 마스크 생성(Mask Generation) $\rightarrow$ 후처리(Post-processing)의 3단계 과정을 거쳐 정밀도를 극대화한다. 특히 Gaussian smoothing을 통해 경계선을 정밀하게 다듬어 아티팩트를 줄인다.
2. **계층적 분류 모델(Cascade Classification):** 클래스 불균형 문제를 해결하기 위해 여러 개의 이진 분류기(Binary Classifier)를 직렬로 연결한 캐스케이드 구조를 제안한다. 이를 통해 복잡한 다중 클래스 분류 문제를 단순한 이진 결정의 연속으로 변환하여 일반화 성능을 높였다.

## 📎 Related Works

기존 연구들은 주로 U-Net과 ResNet과 같은 딥러닝 아키텍처를 사용하여 심장 분할을 수행하였다. U-Net의 인코더-디코더 구조는 전역적 및 지역적 특징을 잘 포착하지만, 최적의 성능을 위해 방대한 양의 학습 데이터와 높은 계산 자원이 필요하다는 단점이 있다.

최근에는 Attention 메커니즘이나 Residual connection을 추가한 개선된 U-Net 모델들이 등장하였으나, 여전히 영상 아티팩트에 대한 강건함(Robustness)과 계산 효율성 측면에서 과제가 남아 있다. 또한, 분할과 분류를 별개의 작업으로 처리하는 경향이 있어, 분할 단계의 오류가 분류 단계로 전이되는 문제가 발생한다.

본 연구는 이러한 한계를 극복하기 위해 분할 단계에서 위치 기반의 세분화된 접근 방식을 취하고, 분류 단계에서는 이진 분류기를 계층적으로 배치함으로써 기존의 단일 분류 모델보다 정밀한 진단이 가능하도록 설계되었다.

## 🛠️ Methodology

### 1. MRI 분할 방법 (Method of MRI Segmentation)

심장 구조 분할은 다음의 3단계 파이프라인으로 구성된다.

**Step 1: 위치 찾기 (Localization)**
전체 마스크를 심근(Myocardium), 좌심실(LV), 우심실(RV)의 세 가지 이진 마스크로 분해한다. 각 구조별로 최적화된 U-Net 및 ResNet-34 기반의 모델을 사용하여 MRI 영상 내에서 해당 구조가 위치한 관심 영역(ROI)을 탐지한다. 이는 객체 탐지기(Object Detector)와 유사하게 작동하며, 불필요한 배경 노이즈를 제거하여 이후 단계의 계산 복잡도를 줄이고 정확도를 높인다.

**Step 2: 심장 마스크 생성 (Cardiac Mask Generation)**
Step 1에서 추출된 지역화된 영상들을 입력으로 받아 각 구조의 정밀한 경계를 획정한다. 이미 지역화된 작은 영역을 다루기 때문에 모델이 더 세밀한 디테일을 포착할 수 있다.

**Step 3: 후처리 (Post-processing)**
학습을 위해 리사이징된 영상들을 다시 원래 크기로 되돌릴 때 발생하는 디테일 손실과 아티팩트를 방지하기 위해 Gaussian smoothing을 적용한다. 가우시안 필터의 수식은 다음과 같다.

$$G(x,y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}$$

여기서 $\sigma$는 smoothing의 강도를 결정하는 표준편차이며, 본 논문에서는 선형 회귀(Linear Regression)를 통해 각 이미지 크기에 최적화된 $\sigma$ 값을 자동으로 결정한다.

### 2. MRI 분류 방법 (Method of MRI Classification)

분류 모델은 분할된 마스크와 원본 MRI 영상을 결합하여 심장 질환을 진단한다.

**데이터 구성:** 
수축기(Systolic)와 이완기(Diastolic) 단계의 영상을 모두 사용하며, RV, LV, 심근의 각 세그먼트를 서로 다른 RGB 채널에 할당하여 딥러닝 모델이 구조적, 조직적 이질성을 분석할 수 있도록 한다.

**캐스케이드 분류 구조 (Cascade Classification):**
클래스 불균형 문제를 해결하기 위해 4개의 이진 분류기를 다음과 같이 배치한다.
1. **Classifier 1:** LV 병변 vs (RV 병변 + 정상)
2. **Classifier 2:** RV 이상 vs 정상
3. **Classifier 3:** 비후성 심근증(HCM) vs 기타 LV 병변
4. **Classifier 4:** 심근경색(MINF) vs 확장성 심근증(DCM)

**모델 아키텍처 및 학습:**
각 분류기는 50개 층으로 구성된 CNN 모델을 사용한다. 초기 Conv 레이어에서 기본적인 엣지와 텍스처를 추출하고, 이후 Conv 레이어들을 통해 추상적인 특징을 학습한다. 마지막으로 Global Average Pooling을 거쳐 이진 분류 결과가 출력된다. 학습에는 Adam optimizer와 Categorical Cross-Entropy 손실 함수가 사용되었으며, 과적합 방지를 위해 Early Stopping을 적용하였다.

## 📊 Results

### 1. 분할 성능 평가
ACDC 데이터셋을 사용하여 평가한 결과, 제안 방법의 각 단계가 성능 향상에 기여함을 확인하였다. Dice coefficient(다이스 계수)를 통해 측정하였으며, 수식은 다음과 같다.

$$Dice = \frac{2 \times |A \cap B|}{|A| + |B|}$$

- **정량적 결과:** 최종 제안 방법(L + D + PP)은 좌심실(LV) 0.974, 우심실(RV) 0.947의 Dice 계수를 달성하였다.
- **비교 분석:** 기존 연구인 Hu et al., da Silva et al. 등과 비교했을 때 모든 지표에서 더 높은 성능을 보였다. 특히 단순 이미지 분할보다 위치 찾기(L)와 분해(D), 후처리(PP)가 결합되었을 때 성능이 비약적으로 상승함을 입증하였다.

### 2. 분류 성능 평가
분류 성능은 정밀도(Precision), 재현율(Recall), F1-score 및 전체 정확도로 측정하였다.

- **단계별 정확도:** Classifier 1(0.96), Classifier 2(1.0), Classifier 3(1.0), Classifier 4(0.90)로 나타났다. 특히 정상 상태와 HCM의 구분 능력이 매우 뛰어났다.
- **전체 정확도:** 제안 모델의 평균 분류 정확도는 $97.2\%$를 기록하였다. 이는 Ammar et al. (0.923), Zheng et al. (0.941)보다 높으며, Mahendra et al. (0.998)과 근접한 수준의 경쟁력 있는 성능이다.
- **ROC 곡선:** AUC 값이 Classifier 1~3에서는 거의 1.0에 도달하였고, Classifier 4는 0.91로 나타나 전반적으로 매우 높은 판별력을 보였다.

## 🧠 Insights & Discussion

본 연구의 강점은 분할 단계에서 **'지역화 $\rightarrow$ 정밀 분할 $\rightarrow$ 가우시안 스무딩'**으로 이어지는 파이프라인을 통해 하드웨어적 제약이 있는 MRI 영상에서도 높은 기하학적 정확도를 확보했다는 점이다. 또한, 복잡한 다중 클래스 진단을 이진 분류기의 캐스케이드 구조로 설계함으로써 의료 데이터 특유의 클래스 불균형 문제를 효과적으로 완화하였다.

하지만 다음과 같은 한계점이 존재한다.
- **영상 품질 의존성:** 입력 영상의 품질이 낮거나 심근/심실의 일부가 보이지 않는 경우 성능이 크게 저하된다.
- **밝기 민감도:** 영상의 밝기가 너무 낮거나 높을 때 경계선 식별에 어려움을 겪는다.
- **희귀 사례 부족:** 특정 병리 조건(예: 스펀지 심근증 등)에 대한 학습 데이터가 부족하여 복잡한 희귀 질환에 대한 일반화 능력이 제한적일 수 있다.

결과적으로 본 모델은 이상적인 조건에서는 매우 강력하지만, 실제 임상 환경의 저품질 영상이나 매우 희귀한 케이스에 적용하기 위해서는 추가적인 강건성 확보가 필요하다.

## 📌 TL;DR

이 논문은 심장 MRI 분석을 위해 **다단계 분할(Localization $\rightarrow$ Generation $\rightarrow$ Smoothing)**과 **이진 분류기 캐스케이드(Cascade of Binary Classifiers)** 구조를 제안하였다. 이를 통해 LV Dice 0.974, RV Dice 0.947의 높은 분할 정확도와 97.2%의 질환 분류 정확도를 달성하였다. 이 연구는 심장 질환의 자동 진단 시스템을 구축하는 데 있어 매우 효과적인 파이프라인을 제시하였으며, 향후 임상 의사결정 지원 시스템의 핵심 모듈로 활용될 가능성이 높다.