# A Survey of Explainable AI in Deep Visual Modeling: Methods and Metrics

Naveed Akhtar (2023)

## 🧩 Problem to Solve

최근 딥러닝 기반의 시각 모델(Deep Visual Models)은 의료 진단, 자율 주행, 법의학 및 감시 시스템과 같이 실패 시 위험 부담이 큰 고위험 도메인(high-stake domains)에서 널리 활용되고 있다. 그러나 이러한 모델들은 내부 작동 원리를 알 수 없는 '블랙박스(black-box)' 특성을 가지고 있어, 인공지능 시스템의 공정성(fairness), 투명성(transparency) 및 책임성(accountability)을 확보하는 데 큰 걸림돌이 되고 있다.

그동안 설명 가능한 AI(Explainable AI, XAI)에 관한 전반적인 리뷰 논문들은 존재해 왔으나, 딥러닝 기반 시각 모델의 해석을 위한 방법론과 평가 지표를 체계적으로 분류하고 정리한 전문적인 서베이 연구는 부족한 실정이다. 따라서 본 논문의 목표는 딥 시각 모델의 해석을 위한 기존 기술들을 방법론적으로 분류(taxonomy)하고, 모델 설명의 다양한 속성을 측정하는 평가 지표들을 체계적으로 정리하여 연구 커뮤니티에 명확한 가이드라인을 제공하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 딥 시각 모델의 투명성을 높이기 위한 기술적 접근 방식을 체계적으로 구조화하고, 그 성능을 측정하는 정량적 지표들을 통합적으로 분석한 점에 있다. 구체적인 기여 사항은 다음과 같다.

1.  **방법론적 분류 체계(Taxonomy) 제시**: 딥 시각 모델을 위한 XAI 기술을 설계 시점과 접근 방식에 따라 Ante-hoc(사전 설계) 방법과 Post-hoc(사후 해석) 방법으로 나누고, 이를 다시 세부 기술군으로 분류하여 체계적인 지도를 제공한다.
2.  **평가 지표의 체계적 정리**: 설명의 품질을 측정하는 지표들이 파편화되어 있다는 점에 주목하여, 모델 충실도(Model Fidelity), 지역화 능력(Localisation Ability), 안정성(Stability) 등 측정하고자 하는 속성별로 지표들을 수집하고 정리하였다.
3.  **용어의 표준화 및 명확화**: '설명(Explanation)'과 '해석(Interpretation)', 'Saliency map'과 'Attribution map' 등 혼용되어 사용되던 용어들을 정리하여 학술적 일관성을 부여하였다.
4.  **향후 연구 방향 제시**: 현재 XAI 연구가 가진 한계점(예: 상관관계와 인과관계의 혼동, 도메인 불일치 등)을 분석하고 미래의 도전 과제들을 제시하였다.

## 📎 Related Works

기존의 XAI 문헌들은 광범위한 도메인을 다루는 일반적인 리뷰에 치중되어 있었으며, 시각 모델이라는 특수한 데이터 형태와 모델 구조에 특화된 방법론적 분류는 미비하였다. 특히 시각적 데이터는 해석이 비교적 용이하다는 특성이 있음에도 불구하고, 이를 체계적으로 프레임워크화한 연구가 부족했다.

본 논문은 기존 연구들이 단순히 최신 기법을 나열하는 수준을 넘어, 시각 모델의 투명성을 높이기 위해 어떤 기술적 전략이 사용되었는지, 그리고 그 결과물이 실제로 모델의 동작을 정확히 반영하는지를 어떻게 정량적으로 측정할 수 있는지를 중점적으로 다룸으로써 기존 서베이들과 차별화를 둔다.

## 🛠️ Methodology

본 논문은 딥 시각 모델의 해석 문제를 다음과 같이 정식화하고 분류한다. 시각 모델을 $M: M(I) \to y$라고 정의할 때, 여기서 $I \in \mathbb{R}^{h \times w \times c}$는 입력 이미지이며 $y \in \mathbb{R}^K$는 $K$개 클래스에 대한 예측 벡터이다.

### 1. Ante-hoc Methods (내재적 설명 가능 모델)
모델의 설계 단계부터 투명성을 고려하는 방법으로, 모델을 함수들의 계층 구조 $M(I) = f_L(\Theta_L, f_{L-1}(\Theta_{L-1}, \dots, f_1(\Theta_1, I) \dots)) \to y$로 본다.

*   **Explainable Model Design from Scratch**: 처음부터 해석 가능한 구조로 설계한다. Generalized Additive Models(GAMs)를 활용하여 입력을 개별 특징으로 분리해 처리하거나, SVM 기반 뉴런을 사용하는 Essence Neural Networks 등이 이에 해당한다.
*   **Augmenting Neural Models**: 기존 블랙박스 모델에 해석 가능한 컴포넌트를 추가한다.
    *   **Internal Component Augmentation**: 모델 내부에 Concept Whitening 모듈을 추가하여 잠재 공간의 축을 기정의된 개념과 정렬시키거나, Concept Bottleneck Models(CBMs)를 통해 인간이 이해할 수 있는 중간 개념층을 도입한다.
    *   **External Component Augmentation**: 모델 외부의 별도 브랜치를 통해 설명을 생성한다. 예를 들어, 외부 Concept Decoder를 학습시키거나 StyleGAN을 설명기로 활용하는 방식이 있다.

### 2. Post-hoc Methods (사후 해석 방법)
학습이 완료된 고정된 모델 $M$을 대상으로 하며, 입력 픽셀 집합 $P^I$에 대해 각 픽셀의 중요도를 나타내는 가중치 배열 $W^I$를 생성하는 함수 $S: S(M, I) \to W^I$를 구현하는 것이 목표이다.

*   **Model-agnostic Methods (Black-box)**: 모델 내부 구조에 접근하지 않고 입력-출력 관계만을 쿼리한다. LIME(국소적 근사 모델 피팅), RISE(무작위 입력 샘플링), SHAP(Shapley Value 기반 기여도 계산) 등이 대표적이다.
*   **Model-specific Methods (White-box)**: 모델 내부의 신호를 활용한다.
    *   **Neuron Activation-based**: 내부 뉴런의 활성화 값을 이용한다. CAM, Grad-CAM 및 그 변형들(Grad-CAM++, Score-CAM)이 있으며, 결과물로 **Saliency map**을 생성한다.
    *   **Backpropagation-based**: 입력에 대한 모델의 기울기(Gradient)를 역전파하여 중요도를 계산한다. Guided Backpropagation, SmoothGrad 등이 있으며, 결과물로 **Attribution map**을 생성한다. 특히 Integrated Gradients와 같은 경로 속성(Path Attribution) 방법은 입력과 참조 이미지 사이의 경로를 적분하여 계산한다.

## 📊 Results

본 논문은 제안된 방법론들을 평가하기 위한 지표들을 다음과 같이 분류하여 분석하였다.

### 1. 모델 충실도 (Model Fidelity)
설명이 모델의 실제 동작을 얼마나 정확하게 반영하는지를 측정한다. 주로 입력 특징을 제거하거나 섭동(perturbation)을 주었을 때 출력값이 어떻게 변하는지를 분석한다.
*   **Pixel Flipping / Region Perturbation**: 중요도가 높은 픽셀을 제거했을 때 예측 점수의 변화를 측정한다.
*   **ROAR (Remove and Retrain)**: 픽셀 제거 후 모델을 다시 학습시켜 분포 변화(distribution shift) 문제를 해결하려 한 지표이다.
*   **ROAD (Remove and Debias)**: 재학습 없이 디바이아싱(de-biasing) 단계를 통해 ROAR의 계산 비용 문제를 해결하고 일관성을 높인 지표이다.

### 2. 지역화 능력 (Localisation Ability)
설명이 실제 객체가 있는 영역(Foreground)에 집중되었는지를 Ground Truth(GT)와 비교하여 측정한다.
*   **Pointing Game**: 가장 중요도가 높은 픽셀이 객체 영역 내에 있는지 확인한다.
*   **Relevance Mass / Rank**: 객체 마스크 영역에 할당된 누적 중요도나 상위 $K$개 픽셀의 분포를 측정한다.

### 3. 안정성 (Stability)
입력에 미세한 변화가 있을 때 설명이 급격하게 변하지 않는지를 측정한다.
*   **Lipschitz Continuity**: 입력 변화량 대비 설명 변화량의 최대비를 측정하여 국소적 안정성을 평가한다.
*   **Relative Stability**: 입력, 임베딩, 출력 로짓의 변화에 따른 설명의 상대적 변화를 분석한다.

### 4. 기타 속성
*   **Conciseness (간결성)**: Gini Index 등을 사용하여 설명이 얼마나 희소(sparse)하고 간결한지 측정한다.
*   **Sanity Preservation (건전성 유지)**: 모델 가중치를 무작위화(randomization)했을 때 설명이 함께 변하는지를 확인하여, 설명 도구가 모델의 가중치에 실제로 의존하는지 검증한다 (Cascading Randomization).
*   **Axiomatic Properties**: Completeness, Linearity 등 수학적으로 정의된 공리적 성질을 만족하는지 평가한다.

## 🧠 Insights & Discussion

본 논문은 시각 XAI 분야의 현재 상태를 비판적으로 분석하며 다음과 같은 통찰을 제시한다.

첫째, **사후 해석(Post-hoc) 방법의 한계**이다. 대부분의 Post-hoc 방법은 개별 샘플에 대한 설명(input-specific)에 집중하고 있어, 정작 모델 전체의 불투명성(opacity)은 해결하지 못하고 있다. 모델의 일반적인 동작을 설명하는 input-agnostic 방향의 연구가 필요하다.

둘째, **상관관계와 인과관계의 혼동**이다. 현재의 Saliency map이나 Fidelity 지표들은 주로 입력 특징과 출력 사이의 '상관관계'를 측정한다. 하지만 단순한 상관관계는 진정한 의미의 설명이 될 수 없으며, 인과관계를 분석할 수 있는 정교한 방법론과 지표가 도입되어야 한다.

셋째, **도메인 불일치(Domain Mismatch) 문제**이다. 게임 이론(Shapley Value)이나 경제학 개념을 시각 도메인에 그대로 적용하면서 문제가 발생한다. 예를 들어, 이미지에서 특정 특징의 '부재(absence)'를 정의하여 참조 이미지를 설정하는 것은 매우 모호하며, 이로 인해 공리적 성질을 만족한다고 주장하는 방법들이 실제로는 그렇지 않은 경우가 많다.

넷째, **성능과 투명성의 딜레마**이다. 일반적으로 해석 가능성이 높은 모델은 성능이 낮다는 믿음이 지배적이지만, 이를 분석적으로 증명한 연구는 부족하다. 규제 당국이 투명성을 요구하는 상황에서 이 딜레마를 해결하는 것이 향후 핵심 과제가 될 것이다.

## 📌 TL;DR

본 논문은 딥 시각 모델의 설명 가능성(XAI)을 위한 방법론을 **Ante-hoc(내재적)**과 **Post-hoc(사후적)**으로 체계적으로 분류하고, 이를 평가하기 위한 지표들을 **충실도, 지역화, 안정성, 간결성** 등의 관점에서 통합 정리한 종합 서베이 보고서이다. 특히 단순한 기법 나열을 넘어, 시각 데이터의 특성과 이론적 배경(게임 이론 등) 사이의 불일치 문제를 지적하고, 단순 상관관계를 넘어선 인과적 설명의 필요성을 강조하였다. 이 연구는 향후 시각 XAI 모델을 설계하고 객관적으로 평가하려는 연구자들에게 필수적인 가이드라인과 분류 체계를 제공한다.