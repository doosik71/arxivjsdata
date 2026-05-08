# Is Mamba Capable of In-Context Learning?

Riccardo Grazzi, Julien Siems, Simon Schrodi, Thomas Brox, Frank Hutter (2024)

## 🧩 Problem to Solve

본 논문은 대규모 언어 모델(LLM)에서 나타나는 핵심 능력 중 하나인 In-Context Learning(ICL)이 Transformer 외의 아키텍처, 특히 최근 주목받는 Mamba 모델에서도 가능한지를 탐구한다.

ICL은 모델이 가중치 업데이트(fine-tuning) 없이 입력으로 제공된 몇 가지 예시(context)만을 통해 새로운 태스크를 수행하는 메타 학습(meta-learning)의 일종이다. 현재 Transformer 기반 모델들이 ICL에서 압도적인 성능을 보이고 있으나, 입력 시퀀스 길이에 따라 연산 복잡도가 제곱으로 증가하는 $\mathcal{O}(L^2)$의 한계를 가지고 있다.

반면, Mamba와 같은 State Space Model(SSM)은 시퀀스 길이에 대해 선형 복잡도 $\mathcal{O}(L)$를 가지며 효율적인 추론이 가능하다. 따라서 본 연구의 목표는 Mamba가 Transformer 수준의 ICL 능력을 갖추고 있는지 실험적으로 검증하고, 만약 가능하다면 어떤 메커니즘을 통해 이를 수행하는지 분석하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같다.

1. **Mamba의 ICL 능력 검증**: 단순 함수 근사(simple function approximation) 및 복잡한 자연어 처리(NLP) 태스크 모두에서 Mamba가 Transformer와 대등한 ICL 성능을 보임을 입증하였다.
2. **효율적 대안 제시**: Mamba가 S4나 RWKV와 같은 기존 선형 모델보다 우수한 성능을 보임을 확인하였으며, 긴 시퀀스를 다루는 ICL 태스크에서 Transformer를 대체할 수 있는 효율적인 대안이 될 수 있음을 시사하였다.
3. **내부 메커니즘 분석**: Probing 기법을 통해 Mamba가 Transformer와 유사하게 층(layer)을 거듭하며 내부 표현을 점진적으로 최적화하는 '반복적 최적화(iterative optimization)' 방식을 통해 ICL 문제를 해결한다는 가설을 제시하였다.

## 📎 Related Works

### 기존 연구 및 한계

- **Transformer 기반 ICL**: GPT-3 이후 Transformer 모델들이 사전 학습을 통해 ICL 능력을 습득함이 밝혀졌으며, 일부 연구에서는 이것이 내부적으로 Gradient Descent와 유사한 최적화 과정을 수행하는 것이라고 분석하였다.
- **State Space Models (SSM)**: S4, H3와 같은 모델들은 시퀀스 처리를 위해 순환 신경망(RNN)과 합성곱 신경망(CNN)의 특성을 결합하여 효율성을 높였으나, 입력 내용에 따라 정보를 선택적으로 저장하는 능력이 부족하여 복잡한 추론 태스크(예: selective copying)에서 Transformer보다 성능이 낮았다.

### 차별점

본 논문은 기존의 SSM 연구들이 주로 단순한 언어 학습이나 특정 합성 태스크에 집중했던 것과 달리, 실제 사전 학습된 Mamba 모델을 사용하여 복잡한 NLP 태스크와 다양한 함수 클래스에 대한 ICL 능력을 종합적으로 평가하였다. 또한, 단순히 성능 측정에 그치지 않고 내부 레이어의 표현을 분석하는 Probing을 통해 작동 원리를 규명하려 했다는 점이 차별화된다.

## 🛠️ Methodology

### 1. State Space Models 및 Mamba 아키텍처

기본적인 SSM은 입력 시퀀스 $(x_1, \dots, x_L)$를 잠재 상태(latent states) $(h_1, \dots, h_L)$를 거쳐 출력 시퀀스 $(y_1, \dots, y_L)$로 매핑한다. 선형 시불변(Linear Time Invariant, LTI) SSM의 기본 식은 다음과 같다.

$$h_t = \bar{A}h_{t-1} + \bar{B}x_t$$
$$y_t = Ch_t$$

여기서 $\bar{A}, \bar{B}, C$는 학습 가능한 파라미터이다. 하지만 LTI-SSM은 입력 내용에 관계없이 동일한 전이 행렬을 사용하므로, 특정 정보를 선택적으로 유지하는 능력이 부족하다.

Mamba는 이를 해결하기 위해 **Selection Mechanism**을 도입한 선형 시변(Linear Time Varying) SSM이다. Mamba에서는 $\bar{A}_t, \bar{B}_t, C_t$가 현재 입력 $x_t$의 함수로 정의된다.

- **가변 파라미터**: $\bar{A}_t = \exp(A\Delta_t)$, $\bar{B}_t = \dots$, $\Delta_t = \text{softplus}(W_3x_t + b_3)$ 와 같이 입력 $x_t$에 따라 게이팅 메커니즘이 작동하여 정보를 선택적으로 무시하거나 상태를 리셋할 수 있다.
- **효율성**: 선택 메커니즘으로 인해 CNN의 병렬 처리는 불가능해졌지만, Hardware-aware algorithm(parallel scan)을 통해 GPU의 SRAM을 효율적으로 활용함으로써 $\mathcal{O}(L)$의 시간 복잡도를 유지하며 빠르게 학습하고 추론한다.

### 2. 실험 설계 및 Probing 방법론

- **함수 클래스 테스트**: Linear, Sparse Linear, 2-layer ReLU NN, Decision Tree 등 4가지 함수 분포에서 모델을 학습시키고, context로 주어진 예시들을 통해 쿼리 포인트의 값을 예측하게 하였다.
- **NLP 태스크 테스트**: Pile 데이터셋으로 사전 학습된 다양한 크기의 Mamba 모델을 사용하여 번역, 지식 추출, 언어적 변환 등 27가지 NLP 태스크를 평가하였다.
- **Probing 분석**: 모델의 중간 레이어 $l$에서의 표현 $z_{l,i}$에 대해 선형 프로브(linear probe)를 적용하여 중간 예측값 $\hat{y}_{l,i}$를 계산한다.

$$\hat{y}_{l,i} = a_l f(z_{l,i}) + b_l$$

여기서 $a_l, b_l$은 검증 셋을 통해 최소제곱법(least squares)으로 구한 스케일 및 시프트 파라미터이다. 이를 통해 레이어가 깊어질수록 예측 오차가 줄어드는지 확인하여 '반복적 최적화' 여부를 판단한다.

## 📊 Results

### 1. 단순 함수 클래스 성능

- **결과**: Mamba는 Skewed Linear Regression 및 Sparse Linear Regression 태스크에서 Transformer와 거의 동일한 성능을 보였으며, Least Squares(최적의 이론적 기준선)에 근접하였다.
- **S4와의 비교**: LTI-SSM인 S4는 모든 설정에서 현저히 낮은 성능을 보였다. 이는 Mamba의 Selection Mechanism이 ICL 수행에 필수적임을 시사한다.
- **외삽(Extrapolation)**: 학습 시 사용한 context 길이보다 더 긴 입력이 들어왔을 때, Mamba는 ReLU NN이나 Decision Tree에서는 성능이 유지되거나 개선되는 경향을 보였으나, Linear Regression에서는 Transformer보다 더 빠르게 성능이 저하되는 모습을 보였다.

### 2. NLP 태스크 성능

- **정량적 결과**: Mamba 2.8B 모델은 ICL 정확도 면에서 Llama 7B에 근접하였으며, GPT-J나 Pythia 모델들과 대등한 성능을 기록하였다.
- **RWKV와의 비교**: 유사한 구조의 RWKV 모델보다 모든 파라미터 규모에서 일관되게 높은 성능을 보였다.
- **Context 길이 확장**: 입력 예시(in-context examples)의 수가 증가함에 따라 Mamba의 정확도가 안정적으로 향상됨을 확인하였다.

### 3. Probing 분석 결과

- **학습 곡선**: Skewed/Sparse Linear Regression 태스크에서 레이어 인덱스가 증가함에 따라 $\log(\text{MSE})$가 거의 선형적으로 감소하였다. 이는 Mamba가 Transformer와 마찬가지로 레이어를 거치며 정답에 가까워지는 반복적 최적화 과정을 수행함을 의미한다.
- **특이사항**: Decision Tree 태스크에서는 초기 레이어 절반 정도까지는 오차가 높게 유지되다가 후반부에 급격히 감소하는 양상을 보였다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 연구는 Mamba가 단순히 시퀀스 모델링을 잘하는 것을 넘어, Transformer의 전유물로 여겨졌던 ICL 능력을 갖추고 있음을 입증하였다. 특히 Probing 분석을 통해 Mamba의 내부 동작 방식이 Transformer의 '내부적 경사 하강법(in-context gradient descent)' 가설과 궤를 같이한다는 점을 밝혀낸 것은 학술적으로 매우 중요한 발견이다.

### 한계 및 비판적 해석

- **외삽 성능의 불안정성**: Linear Regression 태스크에서 context 길이가 길어질 때 성능이 급격히 하락하는 'U-shape' 곡선이 관찰되었다. 이는 Mamba가 매우 긴 context를 처리할 때 내부 상태의 수치적 불안정성이나 정보 손실이 발생할 수 있음을 암시한다.
- **Probing의 단순성**: 본 논문에서 사용한 선형 프로브는 매우 단순한 형태이다. Mamba의 복잡한 최적화 과정을 완전히 설명하기에는 부족할 수 있으며, 더 정교한 비선형 프로빙 기법이 필요할 수 있다.
- **범위의 제한**: 텍스트와 단순 수치 함수에 국한된 실험이므로, 이미지나 오디오와 같은 다른 도메인에서도 동일한 ICL 특성이 나타날지는 미지수이다.

## 📌 TL;DR

본 논문은 선형 복잡도를 가진 **Mamba 아키텍처가 Transformer 수준의 In-Context Learning(ICL) 능력을 보유하고 있음**을 실험적으로 증명하였다. 단순 함수 근사와 복잡한 NLP 태스크 모두에서 우수한 성능을 보였으며, 특히 **레이어를 거치며 점진적으로 최적화하는 메커니즘**이 Transformer와 유사함을 확인하였다. 이 결과는 Mamba가 긴 시퀀스를 처리해야 하는 ICL 기반의 AutoML이나 LLM 시스템에서 **Transformer를 대체할 수 있는 매우 효율적인 대안**이 될 수 있음을 시사한다.
