# Instance-level quantitative saliency in multiple sclerosis lesion segmentation

Federico Spagnolo et al. (2024)

## 🧩 Problem to Solve

본 연구는 다발성 경화증(Multiple Sclerosis, MS)과 같은 다발성 병변 질환의 의료 영상 분석에서 딥러닝 모델의 '블랙박스' 특성을 해결하고자 한다. 특히, 기존의 설명 가능한 AI(XAI) 방법론들은 주로 분류(Classification) 작업이나 시맨틱 세그멘테이션(Semantic Segmentation)의 클래스 수준(Class-level) 설명에 집중되어 있었다.

그러나 의료 영상에서는 동일한 클래스에 속하는 여러 개의 개별 병변(Instance)이 존재하며, 임상의는 특정 병변이 왜 그렇게 검출되고 외곽선이 그려졌는지에 대한 개별 인스턴스 수준의 설명을 필요로 한다. 따라서 본 논문의 목표는 시맨틱 세그멘테이션 모델에서 특정 인스턴스에 대한 설명 맵(Explanation map)을 생성하고, 이를 통해 모델의 결정 근거를 정량적으로 분석하는 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 기존의 XAI 기법인 SmoothGrad(SG)와 Grad-CAM++를 시맨틱 세그멘테이션의 인스턴스 수준으로 확장한 것이다. 

주요 설계 아이디어는 다음과 같다.
1. **Instance-level Saliency**: 모든 클래스 픽셀을 통합하여 설명하는 대신, 관심 대상인 특정 병변 영역($\Omega$)에 해당하는 로짓(Logits)만을 사용하여 해당 인스턴스에 특화된 Saliency map을 생성한다.
2. **Quantitative Saliency**: 병변의 크기에 따라 Saliency 값의 강도가 달라지는 문제를 해결하기 위해, 단순 평균(Average) 방식이 아닌 부호가 포함된 최대값(Maximum with sign) 집계 방식을 도입하여 서로 다른 크기의 병변 간에도 정량적 비교가 가능하게 하였다.

## 📎 Related Works

기존의 XAI 방법론으로 Vanilla Gradients, SmoothGrad, Grad-CAM 및 Grad-CAM++ 등이 널리 사용되어 왔다. Vanilla Gradients는 입력 이미지의 작은 변화에 민감하고 노이즈가 많은 한계가 있으며, SmoothGrad는 입력에 노이즈를 추가하고 평균을 내어 이를 완화한다. Grad-CAM 계열은 특정 레이어의 활성화 맵(Activation map)과 그래디언트를 결합하여 시각화하지만, 이미지 내에 동일 클래스의 인스턴스가 여러 개 존재할 경우 정확도가 떨어진다는 단점이 있다.

세그멘테이션 분야에 XAI를 적용하려는 최근 시도들이 있었으나, 대부분은 특정 클래스의 모든 예측 픽셀을 합산하여 클래스 전체에 대한 설명을 제공하는 수준에 그쳤다. 이는 서로 다른 인스턴스 간의 영향력을 구분하지 못하며, 특히 의료 영상처럼 개별 병변의 특성이 중요한 경우 해석력이 떨어진다는 한계가 있다.

## 🛠️ Methodology

### 전체 파이프라인 및 시스템 구조
연구팀은 687명의 MS 환자로부터 수집한 FLAIR 및 MPRAGE MRI 스캔 데이터를 사용하였다. 세그멘테이션 모델로는 3D U-Net, nnU-Net, Swin UNETR 세 가지 아키텍처를 학습시켜 성능을 평가하고, 제안한 XAI 방법론을 적용하여 각 모델의 결정 기전과 입력 모달리티의 기여도를 분석하였다.

### 상세 방법론 및 방정식

#### 1. Instance-level SmoothGrad (SG)
기존 SG는 분류 작업에서 스칼라 값의 그래디언트를 계산하지만, 세그멘테이션에서는 각 픽셀마다 출력이 존재한다. 이를 해결하기 위해 특정 병변 영역 $\Omega$ 내의 모든 출력 픽셀에 대한 그래디언트를 집계한다.

**평균 집계 방식 (Average Aggregation):**
$$M_{\text{gradient}}^{\Omega}[v] = \frac{1}{N|\Omega|} \sum_{n=1}^{N} \sum_{v' \in \Omega} \frac{\partial y(x_n)[v']}{\partial x_n[v]} \quad (2)$$
여기서 $N$은 노이즈가 추가된 입력의 횟수, $\Omega$는 관심 병변 영역, $y(x_n)[v']$는 출력 로짓이다.

**정량적 집계 방식 (Maximum Aggregation):**
병변 크기에 따른 값의 편차를 줄이기 위해, 영역 $\Omega$ 내에서 부호를 유지한 채 최대 절대값을 가진 그래디언트를 선택한다.
$$M_{\text{gradient}}^{\Omega}[v] = \frac{1}{N} \sum_{n=1}^{N} \text{argmax}_{v' \in \Omega} |D_{n}^{v'}|, \quad \text{where } D_{n}^{v'} = \frac{\partial y(x_n)[v']}{\partial x_n[v]} \quad (3)$$

#### 2. Instance-level Grad-CAM++
Grad-CAM++를 인스턴스 수준으로 확장하기 위해, 특정 병변 영역 $\Omega$의 로짓 합만을 타겟으로 설정하여 $y'$를 정의한다.
$$y' = \sum_{v \in \Omega} y[v] \quad (9)$$
또한, 기존의 전역 평균 풀링(Global Average Pooling) 대신 각 픽셀 위치에 따른 가중치 $\omega_k[v]$를 개별적으로 계산하여, 다른 인스턴스의 활성화 값이 섞이지 않도록 제한한다.
$$M_{\text{GradCAM}}^{\Omega}[v] = \text{Relu} \left( \sum_k \omega_k[v] \cdot A_k[v] \right) \quad (8)$$
$$\omega_k[v] = \alpha_k[v] \cdot \text{Relu} \left( \frac{\partial y'}{\partial A_k[v]} \right) \quad (10)$$
여기서 $A_k$는 $k$번째 활성화 맵이며, $\alpha_k[v]$는 고차 미분을 통해 계산된 가중치 계수이다.

## 📊 Results

### 모델 성능 및 실험 설정
- **성능 지표**: Swin UNETR가 Normalized Dice score 0.80으로 가장 우수한 성능을 보였으며, nnU-Net(0.78), 3D U-Net(0.71) 순으로 나타났다.
- **분석 대상**: U-Net 모델을 중심으로 TP(True Positive), FP(False Positive), FN(False Negative), TN(True Negative) 사례에 대해 Saliency map을 분석하였다.

### 주요 정량적/정성적 결과
1. **입력 모달리티 기여도**: Saliency map 분석 결과, 모델은 MPRAGE보다 FLAIR 영상에 더 크게 의존하였다. FLAIR의 경우 병변 내부와 경계에서 양(+)의 Saliency 값이, 주변부에서 음(-)의 Saliency 값이 나타났다.
2. **정량적 값의 분포**: peak saliency 값의 분포를 분석한 결과, TP > FP > FN > TN 순으로 값이 높게 나타났다. 특히 Mann-Whitney U 테스트 결과, 각 그룹 간의 통계적 유의미한 차이($p < 0.001$)가 확인되어, Saliency의 절대값이 오류 검출의 지표로 사용될 수 있음을 시사한다.
3. **문맥 정보(Contextual Information)의 영향**: 병변 주변의 건강한 조직을 점진적으로 확대하며 관찰한 결과, 모델이 병변을 정확히 검출하기 위해서는 병변 경계로부터 최소 7mm에서 최대 15mm 정도의 주변 문맥 정보가 필요함이 확인되었다.

## 🧠 Insights & Discussion

본 연구는 XAI를 통해 딥러닝 모델이 단순히 픽셀의 강도만을 보는 것이 아니라, 병변과 주변 조직 간의 대비(Contrast)라는 문맥적 정보를 활용하고 있음을 밝혀냈다.

**강점 및 통찰:**
- **임상적 일관성**: FLAIR 영상이 MS 병변 검출에 더 중요하다는 결과는 실제 임상 진료 지침과 일치하며, 이는 모델의 결정 기전이 타당함을 입증한다.
- **설계 최적화 가능성**: 주변 문맥 정보가 약 12-15mm 지점에서 포화(Plateau)된다는 점은, 모델 학습 시 사용되는 패치 크기(Patch size)를 최적화하여 연산 효율을 높일 수 있는 근거가 된다.
- **오류 감소 가능성**: Saliency map의 정량적 특성을 활용해 FP를 줄이는 방법론(Spagnolo et al. 2025)으로 확장될 수 있음을 보여주었다.

**한계 및 비판적 해석:**
- **데이터 특성**: 학습 데이터에 Skull stripping(두개골 제거)이 적용되지 않아 일부 FP가 뇌 외부 조직에서 발생하였다는 점이 언급되었다.
- **계산 비용**: Swin UNETR와 같은 트랜스포머 기반 모델은 Saliency map 계산 비용이 CNN 기반 모델보다 훨씬 높아 실시간 적용에 제약이 있을 수 있다.

## 📌 TL;DR

본 논문은 시맨틱 세그멘테이션 모델에서 특정 개별 인스턴스(병변)에 대한 설명을 제공하는 **인스턴스 수준의 정량적 Saliency map 생성 방법**을 제안하였다. SmoothGrad와 Grad-CAM++를 확장하여 특정 병변 영역($\Omega$)에 집중한 설명을 생성하고, 최대값 집계 방식을 통해 병변 크기와 무관하게 정량적 비교가 가능하게 하였다. 실험 결과, 모델이 FLAIR 영상과 주변 조직의 문맥 정보(7-15mm)를 중요하게 사용함을 확인하였으며, 생성된 Saliency 값의 분포를 통해 TP/FP/FN/TN을 구분할 수 있는 가능성을 제시하였다. 이는 향후 모델 성능 최적화 및 의료 현장에서의 AI 신뢰성 확보에 중요한 기여를 할 것으로 기대된다.