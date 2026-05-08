# Effect of the output activation function on the probabilities and errors in medical image segmentation

Lars Nieradzik, Gerik Scheuermann, Dorothee Saur, and Christina Gillmann (2021)

## 🧩 Problem to Solve

이 논문은 의료 영상 분할(Medical Image Segmentation) 작업에서 신경망의 최종 출력층에 사용되는 활성화 함수(Output Activation Function)의 선택이 예측 확률과 분할 오류에 미치는 영향을 분석한다.

일반적으로 이진 분류 및 분할 작업에서는 Sigmoid 함수가 표준적으로 사용된다. 그러나 의료 영상은 데이터 자체의 불확실성(Uncertainty)이 크기 때문에, 단순한 정확도 향상뿐만 아니라 예측의 불확실성을 적절히 표현할 수 있는 활성화 함수의 선택이 매우 중요하다. 기존 연구들은 주로 중간층(Hidden layers)의 활성화 함수(예: ReLU의 대체제)에 집중해 왔으며, 출력층의 활성화 함수가 성능과 확률 보정(Calibration)에 미치는 영향을 체계적으로 분석한 연구는 부족한 실정이다.

따라서 본 연구의 목표는 다양한 출력 활성화 함수와 손실 함수(Loss function)의 조합을 실험하여, 의료 영상 분할 작업에 가장 적합한 조합을 식별하고 그 수학적 근거를 제시하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 출력 활성화 함수의 **점근적 동작(Asymptotic behavior)**과 **변화율(Rate of change)**이 예측 결과에 미치는 영향을 분석한 것이다.

1. **출력 활성화 함수의 체계적 제안**: 통계학의 Probit 및 Linear 모델에서 영감을 얻어, Sigmoid 외에 Normal CDF, Arctangent, Softsign 등 다양한 함수를 제안하고 이를 의료 영상 분할에 적용하였다.
2. **효과적 도메인(Effective Domain) 개념 도입**: 활성화 함수가 $0$과 $1$이라는 점근선에 얼마나 빨리 도달하는지를 정의하여, 변화율에 따른 예측의 확실성(Certainty)과 확률 보정의 상관관계를 분석하였다.
3. **포괄적인 벤치마크 실험**: 4가지 서로 다른 의료 데이터셋, 7가지 활성화 함수, 3가지 손실 함수의 조합(총 21개 모델)을 통해 정량적 성능과 정성적 확률 분포를 비교 분석하였다.
4. **오픈 테스트 스페이스 제공**: 다른 연구자들이 다양한 활성화 함수를 쉽게 테스트할 수 있도록 GitHub를 통해 소스 코드를 공개하였다.

## 📎 Related Works

기존의 딥러닝 연구들은 주로 ReLU를 대체할 Swish나 Pade Activation Unit과 같은 중간층 활성화 함수 최적화에 집중하였다. 반면, 통계학 분야에서는 Logit(Sigmoid), Probit(Normal CDF), Linear 모델 간의 성능 비교 연구가 존재해 왔다.

본 논문은 이러한 통계학적 모델들이 딥러닝의 출력층 활성화 함수로 사용될 가능성에 주목한다. 특히, 기존 딥러닝 문헌에서는 출력층에 Sigmoid나 Softmax를 사용하는 것이 당연시되었으나, 본 저자들은 이것이 최적이 아닐 수 있다는 가설을 세우고 통계학적 근거를 바탕으로 대안을 탐색한다.

## 🛠️ Methodology

### 전체 시스템 구조

본 연구에서는 표준적인 2D U-Net 아키텍처를 사용하며, 인코더로는 ImageNet으로 사전 학습된 ResNet-34를 채택하여 견고함을 높였다. 전체 파이프라인은 $\text{Encoder} \rightarrow \text{Decoder} \rightarrow \text{Output Activation} \rightarrow \text{Loss Function}$ 순으로 구성된다.

### 출력 활성화 함수 (Output Activation Functions)

저자들은 함수 $f: \mathbb{R} \rightarrow [0, 1]$가 단조 증가하며 $f(0) = 0.5$인 대칭 함수들을 고려하였다. 특히 함수가 점근선 $0$과 $1$에 도달하는 속도를 정의하기 위해 **Effective Domain** $X$를 다음과 같이 정의한다.
$$X = \{x \mid f(x) \le 1-\epsilon, f(x) \ge \epsilon\}, \quad (\epsilon = 0.0025)$$

분석 대상이 된 주요 함수들은 다음과 같다.

- **Faster rate of change (작은 Effective Domain)**: Normal CDF ($\Phi(x)$). 예측이 더 날카롭고(Sharper) 불확실성이 적게 표현된다.
- **Slower rate of change (큰 Effective Domain)**: Inverse Square Root, Arctangent, Softsign. 확률값이 $0$과 $1$로 너무 빠르게 쏠리는 것을 방지하여 확률 보정(Calibration)에 유리할 수 있다.
- **특수 사례**: Linear(이미지별 최소/최대값으로 리스케일링), HardTanh(불연속성 존재).

### 손실 함수 (Loss Functions)

다음 세 가지 손실 함수를 조합하여 실험하였다.

1. **Binary Cross Entropy (BCE)**: 최대 가능도 추정(MLE)에 기반하며, 개별 픽셀의 확률 오류를 최적화한다.
    $$L_{BCE} = -\sum_{i=1}^{n} [y_i \log \hat{y}_i + (1-y_i) \log(1-\hat{y}_i)]$$
2. **Mean Squared Error (MSE)**: 예측값과 실제값의 제곱 차이를 최소화한다.
    $$L_{MSE} = \sum_{i=1}^{n} (\hat{y}_i - y_i)^2$$
3. **Dice Loss**: 예측 영역과 정답 영역의 겹침(Overlap) 정도를 직접 최적화한다.
    $$L_{Dice} = 1 - \frac{2 \sum \hat{y}_i y_i}{\sum \hat{y}_i + \sum y_i}$$

### 학습 및 추론 절차

- **데이터셋**: ACDC(심장), ISLES(뇌), Kvasir-SEG(용종), MSD(전립선).
- **최적화**: Adam optimizer (learning rate $10^{-3}$), 5-fold cross-validation 사용.
- **평가 지표**: Negative Log-Likelihood (NLL), Dice Coefficient, Reliability Diagram(확률 보정 측정).

## 📊 Results

### 정량적 결과

실험 결과, 활성화 함수의 변화율과 손실 함수의 종류에 따라 상충하는 결과가 나타났다.

1. **BCE 및 MSE 손실 함수 사용 시**:
    - **변화율이 빠른 함수(예: Normal CDF)**가 Sigmoid보다 더 낮은 NLL(분할 오류 감소)을 기록하는 경향이 있다.
    - 이는 네트워크의 자유도를 제한하여 더 나은 지역 최적점(Local minima)에 도달하게 하기 때문으로 분석된다.
2. **Dice Loss 사용 시**:
    - **변화율이 느린 함수(예: Arctangent, Softsign)**가 오히려 더 나은 NLL 성능을 보였다.
    - Dice Loss는 개별 픽셀보다 전체 겹침을 최적화하므로, 넓은 도메인을 가진 함수가 개별 픽셀의 오류를 더 유연하게 조정할 수 있게 한다.

### 정성적 결과 및 확률 보정

- **Reliability Diagram 분석**: 변화율이 느린 함수(Slower rate of change)를 사용할수록 예측 확률이 실제 정답 빈도와 더 일치하는 경향을 보였다. 즉, **확률 보정(Calibration) 측면에서는 도메인이 넓은 함수가 유리**하다.
- **시각적 분석**: Normal CDF는 경계 영역에서 더 확실한(Certain) 예측을 하는 반면, Softsign 등은 더 많은 불확실성을 유지한다.

### 종합 비교 (vs. Sigmoid)

- **Normal CDF**는 여러 데이터셋에서 Sigmoid보다 최적 Dice Coefficient 성능이 우수하거나 대등하였다.
- **Linear** 활성화 함수 또한 Sigmoid 대비 경쟁력 있는 성능을 보였다.

## 🧠 Insights & Discussion

### 핵심 통찰

본 연구는 출력 활성화 함수의 **'변화율'**이 모델의 **'확실성(Certainty)'**과 **'정확도(Accuracy)'** 사이의 트레이드오프를 결정한다는 점을 밝혀냈다.

- **빠른 변화율 $\rightarrow$ 높은 확실성 $\rightarrow$ 낮은 분할 오류(BCE 기준)**
- **느린 변화율 $\rightarrow$ 낮은 확실성 $\rightarrow$ 우수한 확률 보정**

### 비판적 해석 및 한계

1. **손실 함수와의 상호작용**: 단순히 "어떤 활성화 함수가 좋다"라고 결론 내릴 수 없으며, 반드시 사용 중인 손실 함수(BCE vs Dice)와의 조합을 고려해야 한다.
2. **Dice Loss의 한계**: Dice Loss는 확률값의 불확실성을 생성하지 않는 경향이 있으며, 본 실험에서도 BCE/MSE보다 전반적인 성능이 낮게 나타났다. 이는 Dice Loss가 정답을 $\{0, 1\}$로 제한하는 제약 조건을 $[0, 1]$로 완화한 대리 손실 함수(Surrogate loss)이기 때문으로 풀이된다.
3. **범위의 한계**: 본 연구는 이진 분할(Binary segmentation)에 집중하였다. 다중 클래스 분할(Multi-label segmentation)을 위해서는 Softmax와 같은 범주형 분포를 생성하는 함수로의 확장이 필요하다.

## 📌 TL;DR

이 논문은 의료 영상 분할에서 표준적으로 쓰이는 Sigmoid 활성화 함수가 항상 최선은 아님을 증명하였다. **Normal CDF**와 같이 변화율이 빠른 함수는 분할 정확도를 높이는 데 유리하고, **Arctangent**와 같이 변화율이 느린 함수는 확률 보정(Calibration)을 개선하는 데 유리하다. 결론적으로, 사용자의 목적(정확도 중심 vs 불확실성 표현 중심)과 선택한 손실 함수에 따라 적절한 출력 활성화 함수를 선택해야 하며, 특히 Normal CDF가 Sigmoid의 강력한 대안이 될 수 있음을 제시하였다.
