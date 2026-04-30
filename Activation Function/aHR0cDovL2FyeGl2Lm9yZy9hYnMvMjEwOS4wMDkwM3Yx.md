# Effect of the output activation function on the probabilities and errors in medical image segmentation

Lars Nieradzik, Gerik Scheuermann, Dorothee Saur, and Christina Gillmann (2021)

## 🧩 Problem to Solve

본 논문은 의료 영상 분할(Medical Image Segmentation) 작업에서 신경망의 마지막 단계에 사용되는 출력 활성화 함수(Output Activation Function)의 선택이 예측 확률과 분할 오류에 미치는 영향을 분석한다.

일반적으로 이진 분류 및 분할 작업에서는 Sigmoid 함수가 표준적으로 사용되지만, 의료 영상은 데이터 자체에 내재된 불확실성(Uncertainty)이 크다는 특징이 있다. 따라서 단순한 정확도 향상을 넘어, 예측의 불확실성을 적절히 표현하면서도 분할 성능을 높일 수 있는 최적의 활성화 함수와 손실 함수(Loss Function)의 조합을 찾는 것이 본 연구의 핵심 목표이다.

## ✨ Key Contributions

본 연구의 중심적인 아이디어는 활성화 함수의 **점근적 동작(Asymptotic Behavior)**, 특히 함수가 0과 1이라는 점근선에 도달하는 **변화율(Rate of Change)**이 예측 확률의 분포와 모델의 캘리브레이션(Calibration)에 어떤 영향을 주는지를 분석하는 것이다.

이를 위해 저자들은 다음과 같은 기여를 수행하였다:
- 의료 영상 분할에 적용 가능한 7가지 출력 활성화 함수와 3가지 손실 함수의 조합을 체계적으로 평가하였다.
- 활성화 함수의 '유효 도메인(Effective Domain)' 개념을 도입하여, 변화율이 빠른 함수(좁은 도메인)와 느린 함수(넓은 도메인)가 성능 및 확률 분포에 미치는 영향을 분석하였다.
- 4가지 서로 다른 의료 영상 데이터셋을 통해 제안하는 방법론의 일반성을 검증하였다.
- 다양한 활성화 함수를 실험할 수 있는 오픈 테스트 공간(Open Test Space)을 제공하였다.

## 📎 Related Works

기존의 딥러닝 연구들은 주로 ReLU와 같은 중간층(Intermediate Layers)의 활성화 함수를 개선하는 데 집중해 왔으며, 출력층의 활성화 함수에 대한 연구는 상대적으로 부족하였다. 

통계학 분야에서는 Logistic 모델(Sigmoid) 외에도 Probit 모델(Normal CDF)이나 Linear 모델이 사용되어 왔으며, 일부 연구에서 이들을 비교 분석한 사례가 있다. 그러나 이러한 통계적 모델들의 활성화 함수를 현대의 딥러닝 아키텍처인 출력층에 적용하여 의료 영상 분할 작업에서 심층적으로 분석한 연구는 거의 없었다. 본 논문은 이러한 통계학적 기반의 함수들을 딥러닝의 출력층에 도입하여 기존 Sigmoid 중심의 관행에 도전한다.

## 🛠️ Methodology

### 전체 파이프라인
본 연구는 $\text{ResNet-34}$를 Encoder로, $\text{U-Net}$ 구조를 Decoder로 사용하는 아키텍처를 기반으로 한다. Decoder의 출력값에 서로 다른 출력 활성화 함수를 적용하고, 이를 특정 손실 함수로 학습시킨 후 그 결과를 평가하는 구조이다.

### 출력 활성화 함수 (Output Activation Functions)
저자들은 함수가 $0$과 $1$ 사이의 값을 가지며, $f(0) = 0.5$인 단조 증가 대칭 함수들을 선택하였다. 특히 **유효 도메인(Effective Domain)** $X=\{x|f(x) \le 1-\epsilon, f(x) \ge \epsilon\}$ (단, $\epsilon=0.0025$)을 기준으로 함수들을 분류하였다.

1. **변화율이 빠른 함수 (좁은 유효 도메인):** $\text{Normal CDF}$ ($\Phi(x)$). 이는 네트워크가 더 날카로운 결정(Sharper Decisions)을 내리게 하며 불확실성을 줄이는 경향이 있다.
2. **변화율이 느린 함수 (넓은 유효 도메인):** $\text{Inverse Square Root}$, $\text{Arctangent}$, $\text{Softsign}$. 이는 확률값이 $0$이나 $1$로 너무 빨리 치우치는 것을 방지하여 확률 추정의 유연성을 제공한다.
3. **기타:** $\text{Sigmoid}$ (기준점), $\text{Linear}$ (이미지/배치별로 최소-최대값 리스케일링 적용), $\text{HardTanh}$ (불연속 지점이 존재함).

### 손실 함수 (Loss Functions)
다음 세 가지 손실 함수를 조합하여 실험하였다:
- **Binary Cross-Entropy (BCE):** 
  $$\text{L}(\beta; Y, X) = -\sum_{i=1}^{n} [y_i \log f(x_i^T \beta) + (1-y_i) \log(1-f(x_i^T \beta))]$$
- **Mean Squared Error (MSE):** 
  $$\text{L}(\beta; Y, X) = \sum_{i=1}^{n} (\hat{y}_i - y_i)^2$$
- **Dice Loss:** 
  $$\text{L}(\beta; Y, X) = 1 - \frac{2 \sum_i \hat{y}_i y_i}{\sum_i \hat{y}_i + y_i}$$

### 학습 및 평가 절차
- **데이터셋:** $\text{ACDC}$ (심장), $\text{ISLES}$ (뇌), $\text{Kvasir-SEG}$ (결장), $\text{MSD}$ (전립선) 총 4종.
- **평가 지표:** $\text{Negative Log-Likelihood (NLL)}$, $\text{Dice Coefficient}$, 그리고 예측 확률의 신뢰도를 측정하는 $\text{Reliability Diagram}$을 사용하였다.

## 📊 Results

### 정량적 결과
- **BCE 및 MSE 손실 함수 사용 시:** 일반적으로 유효 도메인이 좁은 함수(변화율이 빠른 함수)가 더 낮은 분할 오류를 보였다. 특히 $\text{Normal CDF}$는 $\text{Sigmoid}$보다 우수하거나 대등한 성능을 보였다.
- **Dice Loss 사용 시:** BCE/MSE와 반대로, 유효 도메인이 넓은 함수(변화율이 느린 함수, 예: $\text{Arctangent}$)가 $\text{NLL}$ 관점에서 더 좋은 결과를 내는 경향이 있었다.
- **데이터셋별 특성:**
    - $\text{ISLES}$ 데이터셋에서는 $\text{BCE} + \text{Normal CDF}$ 조합이 가장 낮은 $\text{NLL}$과 높은 $\text{Dice}$ 계수를 기록하였다.
    - $\text{Kvasir-SEG}$에서는 $\text{Linear}$나 $\text{HardTanh}$ 같은 단순한 함수들이 $\text{Sigmoid}$보다 높은 $\text{Dice}$ 계수를 보이기도 하였다.
    - $\text{MSD}$에서는 $\text{BCE} + \text{Arctangent}$ 조합이 $\text{Dice}$ 계수 면에서 가장 우수하였다.

### 정성적 및 확률적 결과
- **Reliability Diagram 분석:** 유효 도메인이 넓은 함수(예: $\text{Softsign}$, $\text{Arctangent}$)가 $\text{Sigmoid}$나 $\text{Normal CDF}$보다 실제 확률 분포와 더 잘 일치하는(Better Calibration) 경향을 보였다.
- **불확실성 시각화:** 유효 도메인이 좁은 함수는 경계 영역에서 확률값이 급격히 변하여 확신이 강한 예측을 하는 반면, 넓은 도메인의 함수는 경계 영역에서 더 완만한 확률 변화를 보여 불확실성을 더 잘 표현하였다.

## 🧠 Insights & Discussion

### 강점 및 분석
본 연구는 활성화 함수의 수학적 성질(변화율)이 딥러닝 모델의 결과물인 확률 맵의 특성을 결정짓는다는 점을 실험적으로 입증하였다. 
- **정확도 vs 신뢰도:** 변화율이 빠른 함수($\text{Normal CDF}$)는 네트워크의 자유도를 제한하여 더 나은 로컬 미니마(Local Minima)에 도달하게 함으로써 분할 정확도를 높이는 경향이 있다. 반면, 변화율이 느린 함수는 확률값이 양 극단으로 쏠리는 것을 방지하여 예측의 신뢰도(Calibration)를 높인다.
- **Dice Loss의 특이성:** $\text{Dice Loss}$는 개별 픽셀의 확률보다 전체적인 겹침(Overlap)을 최적화하므로, 활성화 함수의 변화율이 느릴 때 오히려 개별 픽셀의 확률 추정치가 더 안정화되는 역설적인 결과가 나타났다.

### 한계 및 비판적 해석
- **손실 함수의 제한:** $\text{Lovász-Softmax}$와 같은 최신 대리 손실 함수(Surrogate Loss)를 모두 테스트하지 못한 점이 한계로 지적된다.
- **다중 클래스 확장성:** 본 연구는 이진 분할(Binary Segmentation)에 집중하였으며, $\text{Softmax}$를 사용하는 다중 클래스 분할 작업으로의 확장 가능성에 대해서는 이론적인 제언만 이루어졌다.
- **Hyperparameter 영향:** $\text{HardTanh}$의 낮은 성능은 불연속성으로 인한 수렴 문제로 해석되는데, 이는 학습률(Learning Rate) 조절 등을 통해 개선될 여지가 있다.

## 📌 TL;DR

본 논문은 의료 영상 분할에서 관습적으로 사용되는 $\text{Sigmoid}$ 출력 활성화 함수가 최선이 아님을 밝히고, 활성화 함수의 **변화율(Rate of Change)**에 따른 성능 차이를 분석하였다. 실험 결과, **정확한 분할(Segmentation Error 감소)을 위해서는 $\text{Normal CDF}$와 같이 변화율이 빠른 함수**가 유리하며, **예측 확률의 신뢰도(Calibration)를 높이기 위해서는 $\text{Arctangent}$나 $\text{Softsign}$ 같이 변화율이 느린 함수**가 유리하다는 트레이드-오프 관계를 발견하였다. 이 연구는 향후 의료 AI 모델 설계 시 목적(정확도 우선 vs 불확실성 표현 우선)에 따라 출력 활성화 함수를 선택해야 한다는 중요한 가이드라인을 제시한다.