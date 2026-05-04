# The Hidden Attention of Mamba Models

Ameen Ali, Itamar Zimerman, and Lior Wolf (2024)

## 🧩 Problem to Solve

Mamba 모델의 기반이 되는 Selective State Space Model (SSM)은 NLP, 컴퓨터 비전 등 다양한 도메인에서 Transformer에 필적하는 성능을 보이면서도, 시퀀스 길이에 대해 선형 복잡도를 가지며 추론 속도가 빠르다는 장점이 있다. 그러나 이러한 성공에도 불구하고 Mamba 모델 내부에서 토큰 간의 정보 흐름(information-flow)이 어떻게 이루어지는지, 그리고 모델이 구체적으로 어떻게 학습하는지에 대한 메커니즘은 거의 알려지지 않았다.

특히 Mamba 모델은 RNN, CNN 또는 Attention 메커니즘과 어떤 유사성을 가지는지에 대한 분석이 부족하며, Transformer에서 널리 사용되는 설명 가능성(Explainability) 방법론들이 Mamba에는 적용되지 않아 모델의 디버깅이나 사회적으로 민감한 도메인에서의 적용에 한계가 있다. 따라서 본 논문의 목표는 Mamba 모델을 Attention 관점에서 재해석하여 모델의 내부 동작을 이해할 수 있는 해석 가능성(Interpretability) 도구를 제공하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Mamba의 Selective SSM 레이어를 **데이터 제어 선형 연산자(data-control linear operator)** 형태로 재정의함으로써, Mamba 내부에 숨겨진 **Implicit Attention Matrix(암시적 어텐션 행렬)**가 존재함을 이론적으로 증명하는 것이다.

주요 기여 사항은 다음과 같다:
- Mamba 모델이 암시적 어텐션에 의존하고 있음을 보여줌으로써, Mamba를 Attention 기반 모델의 관점에서 바라볼 수 있는 새로운 시각을 제시한다.
- Mamba 모델이 Transformer보다 수백 배 더 많은 수의 어텐션 행렬을 생성한다는 사실을 밝혀낸다.
- 추출된 hidden attention matrices를 이용하여 Mamba 모델을 위한 최초의 설명 가능성(XAI) 도구(Attention-Rollout 및 Mamba-Attribution)를 제안한다.
- Mamba 모델이 Transformer와 유사한 수준의 설명 가능성 지표를 가짐을 실험적으로 입증한다.
- SSM의 표현력(expressiveness)에 대한 이론적 분석을 통해, Selective SSM이 Transformer 헤드가 표현할 수 있는 모든 함수를 표현할 수 있으며 그 이상의 능력을 갖추었음을 증명한다.

## 📎 Related Works

**Transformers**
Transformer는 Self-Attention 메커니즘을 통해 토큰 간의 의존성을 캡처하며, $\text{softmax}$를 통해 각 토큰의 중요도를 가중치로 계산한다. 이는 모델의 해석 가능성을 높이는 핵심 요소로 작용해 왔다.

**State-Space Layers (SSMs)**
기존의 SSM(S4, S5 등)은 재귀(recurrence) 또는 합성곱(convolution) 형태로 구현되어 효율적인 연산을 가능하게 했다. 하지만 기존 SSM들은 고정된 믹싱 요소(fixed mixing elements)를 사용하여 입력 데이터에 따라 동적으로 가중치를 조절하는 능력이 부족했다.

**Selective SSMs (S6/Mamba)**
Mamba는 입력 시퀀스에 따라 파라미터 $\bar{A}, \bar{B}, C$가 변하는 Time-variant SSM을 도입하여 표현력을 높였다. 하지만 이러한 동적 특성으로 인해 기존의 고정 커널 기반 SSM 분석법으로는 내부 동작을 설명하기 어려웠으며, Transformer와 같은 명시적인 어텐션 맵이 존재하지 않는다는 특징이 있다.

## 🛠️ Methodology

### 1. Hidden Attention Matrices의 도출
논문은 Mamba의 재귀 식을 전개하여 입력 $x$와 출력 $y$ 사이의 관계를 행렬 형태로 재구성한다. 단일 채널에 대해 초기 상태 $h_0 = 0$이라고 가정할 때, 출력 $y$는 다음과 같이 표현될 수 있다:
$$y = \tilde{\alpha}x$$
여기서 $\tilde{\alpha} \in \mathbb{R}^{L \times L}$는 **Hidden Attention Matrix**이며, 각 원소 $\tilde{\alpha}_{i,j}$는 토큰 $x_j$가 $y_i$에 미치는 영향을 나타낸다. 이 값은 다음과 같이 계산된다:
$$\tilde{\alpha}_{i,j} = C_i \left( \prod_{k=j+1}^{i} \bar{A}_k \right) \bar{B}_j$$

### 2. Mamba Attention의 수식적 단순화
$\bar{A}_t$가 대각 행렬임을 이용하여, $\tilde{\alpha}_{i,j}$를 다음과 같은 Query, Key, History의 형태로 단순화하여 해석한다:
$$\tilde{\alpha}_{i,j} \approx \tilde{Q}_i \tilde{H}_{i,j} \tilde{K}_j$$
- $\tilde{Q}_i$: $S_C(\hat{x}_i)$로 정의되며, 현재 토큰의 쿼리 역할을 한다.
- $\tilde{K}_j$: $\text{ReLU}(S_\Delta(\hat{x}_j) S_B(\hat{x}_j))$로 정의되며, 과거 토큰의 키 역할을 한다.
- $\tilde{H}_{i,j}$: $\exp(\sum_{k=j+1}^{i} S_\Delta(\hat{x}_k) A)$ 형태로, $x_j$부터 $x_i$까지의 연속적인 역사적 맥락(continuous aggregated historical context)을 제어한다.

이 구조는 Transformer의 점곱(dot-product) 어텐션과 달리, $\tilde{H}_{i,j}$라는 항을 통해 시퀀스의 연속적인 맥락을 더 효율적으로 활용할 수 있음을 시사한다.

### 3. 설명 가능성 도구 적용
추출된 $\tilde{\alpha}$를 바탕으로 두 가지 XAI 기법을 제안한다.

- **Attention-Rollout (Class-agnostic):** 모든 채널 $d$의 어텐션 행렬을 합산하고, 레이어 $\lambda$를 거치며 행렬 곱을 통해 최종 맵 $\rho$를 생성한다.
  $$\tilde{\alpha}_\lambda = I + \sum_{d \in [D]} \tilde{\alpha}_{\lambda,d}, \quad \rho = \prod_{\lambda=1}^{\Lambda} \tilde{\alpha}_\lambda$$
- **Mamba-Attribution (Class-specific):** 특정 클래스에 대한 그래디언트 $\nabla \hat{y}'_{\lambda,d}$와 어텐션 행렬 $\tilde{\alpha}_{\lambda,d}$를 결합하여 클래스 특화 맵을 생성한다.
  $$\tilde{\beta}_\lambda = I + \left( \sum_{d \in D} \nabla \hat{y}'_{\lambda,d} \right) \odot \left( \sum_{d \in D} \tilde{\alpha}_{\lambda,d} \right)$$

## 📊 Results

### 1. 시각화 및 정성적 분석
- **Attention Map 비교:** Vision Mamba(ViM)와 ViT, Mamba-130M과 Pythia-160M의 어텐션 맵을 비교한 결과, 두 모델 모두 깊은 레이어로 갈수록 원거리 토큰 간의 의존성을 더 많이 캡처하는 유사한 패턴을 보였다.
- **CLS 토큰 위치 영향:** ViM에서 CLS 토큰의 위치(중간 vs 끝)에 따라 주변 패치들의 영향력이 달라짐을 확인하였으며, 이는 전역 CLS 토큰 사용의 필요성을 시사한다.

### 2. 정량적 평가 (ImageNet)
- **Perturbation Test (섭동 테스트):** 가장 중요한 픽셀부터 제거했을 때(Positive)와 덜 중요한 픽셀부터 제거했을 때(Negative)의 정확도 변화(AUC)를 측정하였다. Raw-Attention과 Attn-Rollout에서는 Mamba가 ViT보다 우수하거나 대등한 성능을 보였으나, Attribution 방법에서는 ViT가 더 높은 성능을 보였다.
- **Segmentation Test:** ground-truth 세그멘테이션 맵과 비교하여 Pixel Accuracy, mIoU, mAP를 측정했다. Raw-Attention과 Attn-Rollout에서 Mamba가 ViT를 능가하는 결과를 보였으며, 이는 Mamba의 어텐션 메커니즘이 공간적 위치 정보를 잘 파악하고 있음을 의미한다.

### 3. NLP 실험 (IMDb)
- BERT-large와 비교하여 Mamba-Attr의 성능을 측정하였다. Pruning(불필요한 단어 제거) 및 Activation(중요 단어 추가) 태스크 모두에서 Mamba-Attr이 BERT의 Transformer-Attr과 대등하거나 더 우수한 성능을 보였다. 특히 Mamba-Attr이 생성하는 설명이 더 희소(sparse)하여 핵심 단어를 더 명확하게 짚어내는 경향이 있었다.

## 🧠 Insights & Discussion

**S6의 본질: Data-controlled Non-diagonal Mixer**
논문은 SSM의 진화 과정을 분석하여, S4, S5 등 기존 모델은 고정된 믹싱 요소(fixed mixing elements)를 가진 반면, Selective SSM(Mamba)은 **데이터 제어 비대각 믹서(data-controlled non-diagonal mixer)**를 가지고 있음을 이론적으로 증명(Theorem 1)한다. 저자들은 이 특성이 Transformer의 In-Context Learning (ICL) 능력과 유사한 능력을 Mamba가 갖게 된 핵심 이유라고 주장한다.

**표현력 분석 (Expressiveness)**
Theorem 2/3를 통해, 단일 채널의 Selective SSM 레이어가 단일 Transformer 헤드가 표현할 수 있는 모든 함수를 표현할 수 있음을 증명하였다. 반면, 단일 Transformer 레이어는 Selective SSM 레이어가 표현할 수 있는 모든 함수(예: 'count in row' 문제)를 표현할 수 없다. 이는 Mamba가 이론적으로 Transformer보다 더 높은 표현력을 가질 수 있음을 시사한다.

**Oversmoothing 문제**
Transformer는 $\text{softmax}$ 연산으로 인해 토큰 간의 표현이 지나치게 유사해지는 Oversmoothing 문제가 발생한다. 반면 Mamba는 $\text{softmax}$를 사용하지 않기 때문에 토큰 간의 코사인 유사도가 더 낮게 유지되며, 결과적으로 더 세밀한 특징을 유지할 수 있다.

## 📌 TL;DR

본 논문은 Mamba 모델의 Selective SSM 레이어를 어텐션 메커니즘으로 재정의하여, 내부에 숨겨진 **Implicit Attention Matrix**를 추출하는 방법을 제안한다. 이를 통해 Mamba 모델을 위한 최초의 XAI 도구를 개발하였으며, 실험을 통해 Mamba가 Transformer와 대등하거나 일부 지표에서 더 뛰어난 해석 가능성을 가짐을 입증하였다. 이론적으로는 Mamba가 '데이터 제어 비대각 믹서'를 통해 Transformer보다 더 높은 표현력을 가질 수 있음을 증명함으로써, SSM 모델의 성능 향상 원인을 규명하였다. 이 연구는 향후 Mamba 기반 모델의 신뢰성과 강건성을 분석하는 중요한 분석 틀을 제공한다.