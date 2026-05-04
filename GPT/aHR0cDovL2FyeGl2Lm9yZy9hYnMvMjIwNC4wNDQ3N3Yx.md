# FoundationLayerNorm: Scaling BERT and GPT to 1,000 Layers

Dezhou Shen(2022)

## 🧩 Problem to Solve

본 연구가 해결하고자 하는 핵심 문제는 Transformer 기반 모델인 BERT와 GPT의 층(layer) 수를 극단적으로 늘렸을 때 발생하는 학습 불안정성이다. 일반적인 BERT와 GPT 모델은 보통 10층에서 20층 사이의 깊이를 가지며, 매우 깊은 층을 가진 모델에 대한 연구와 학습 사례는 매우 드물다.

모델의 깊이가 깊어질수록 기울기 소실(Gradient Vanishing)이나 기울기 폭주(Gradient Exploding)와 같은 문제가 심화되며, 특히 이전 층의 출력값에 대한 의존성이 높아져 학습이 어려워진다. 따라서 본 논문의 목표는 1,000층이라는 전례 없는 깊이까지 BERT와 GPT를 확장하고, 이를 안정적으로 학습시킬 수 있는 새로운 정규화(Normalization) 방법을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 아이디어는 잔차 연결(Residual Connection) 부분에 특수한 정규화 함수를 도입하여 모델 업데이트 시의 상수를 제약함으로써 학습의 안정성을 확보하는 것이다.

가장 핵심적인 기여는 BERT를 위한 **Upscale Layer Normalization**과 GPT를 위한 **Foundation Layer Normalization**이라는 두 가지 정규화 기법을 제안한 것이다. 이를 통해 기존 모델들보다 한 차원 더 깊은 1,000층 규모의 BERT 및 GPT 모델을 성공적으로 학습시켰으며, 이는 해당 아키텍처들 중 가장 깊은 모델임을 주장한다.

## 📎 Related Works

논문에서는 다음과 같은 기존의 정규화 및 안정화 기법들을 언급한다.

- **Batch Normalization**: 데이터 분포의 분산에 따라 입력을 조정하여 학습을 가속화하지만, Transformer 구조에서는 한계가 있다.
- **Layer Normalization**: 각 뉴런의 활동을 정규화하여 은닉 상태의 역학을 안정화하며, 특히 순환 신경망(RNN)에서 효과적이다.
- **Pre-LN vs Post-LN**: Pre-LN(Pre-norm residual connection)은 Post-LN보다 안정적이지만, 하단 층의 기울기가 상단 층보다 커지는 경향이 있어 성능 저하가 발생할 수 있다.
- **DeepNorm**: Transformer를 1,000층까지 확장하기 위해 잔차 연결을 스케일링하고, 피드포워드 네트워크와 어텐션 레이어의 특정 가중치들을 $\beta$ 값으로 스케일링하는 방법을 제안하였다.

본 논문은 이러한 DeepNorm의 아이디어를 계승하되, 가중치 스케일링 부분을 제거하거나 상수를 조정함으로써 더 단순하면서도 효과적인 구조를 추구한다는 점에서 차별점을 갖는다.

## 🛠️ Methodology

본 논문은 모델의 종류에 따라 두 가지 서로 다른 정규화 접근 방식을 사용한다.

### 1. Upscale Layer Normalization (for BERT)

BERT 모델의 안정적인 학습을 위해 DeepNorm의 수식을 수정하여 적용하였다. 기존 DeepNorm이 잔차 브랜치 내부의 가중치 $\theta$를 $\beta$로 스케일링한 것과 달리, Upscale Layer Normalization은 가중치 $\theta$를 그대로 유지하고 상수 $\alpha$만을 적용한다.

전체적인 수식은 다음과 같다.
$$x_{i+1} = \text{LN}(\alpha x_i + G_i(x_i, \theta_i))$$

여기서 $x_i$는 $i$번째 층의 입력, $\text{LN}$은 Layer Normalization, $G_i$는 $i$번째 Transformer 서브레이어(Attention 또는 Feedforward Network)를 의미한다. 이때 상수 $\alpha$는 모델의 총 층 수 $N$에 따라 다음과 같이 결정된다.
$$\alpha = (2N)^{1/4}$$

### 2. Foundation Layer Normalization (for GPT)

GPT 모델의 경우, 경험적인 수치를 바탕으로 한 더 단순한 형태의 정규화를 적용하였다.

수식은 다음과 같다.
$$x_{i+1} = \text{LN}(0.974 x_i + G_i(x_i, \theta_i))$$

여기서 $0.974$는 저자가 경험적으로 찾아낸 상수 파라미터이다. 이 방식을 통해 GPT 모델을 1,000층까지 확장하여 학습시키는 것이 가능해졌다.

## 📊 Results

### 실험 환경 및 설정

- **BERT-1k**: The Pile 데이터셋에서 9G의 데이터를 사용하였으며, Nvidia 3090 GPU로 100k step(약 4일) 동안 학습하였다. Hidden size는 64, Attention head는 2개로 설정되었다.
- **GPT-1k**: The Pile 데이터셋에서 200G의 데이터를 사용하였으며, Nvidia 3090 GPU로 150k step(약 7일) 동안 학습하였다. Hidden size는 256, Attention head는 1개로 설정되었다.

### 정량적 결과

- **BERT 평가**: Quora Question Pairs(QQP) 데이터셋에서 Precision 72%, Recall 69%, F1-score 70%, Accuracy 73%의 성능을 기록하였다.
- **GPT 평가**: 다양한 벤치마크에서 평가를 진행하였다.
  - PIQA Accuracy: $55.17\%$
  - Winogrande Accuracy: $50.36\%$
  - Hellaswag Accuracy: $25.54\%$
  - LAMBADA Accuracy: $0.72\%$ (매우 낮음)

### 모델 비교 분석

GPT-1k 모델(파라미터 약 815.5M)을 다른 거대 모델들과 비교했을 때, GPT-J(6B)의 약 1/75 수준의 파라미터 크기임에도 불구하고 PIQA와 Winogrande 데이터셋에서 경쟁력 있는 성능을 보여주었다. 특히 연산량(FLOPs) 측면에서 GPT-1k($3.72 \times 10^{19}$)는 GPT-J($1.5 \times 10^{22}$)보다 훨씬 효율적임을 확인하였다.

## 🧠 Insights & Discussion

본 논문은 적절한 정규화 기법(FoundationLayerNorm)만 있다면 Transformer 모델의 깊이를 1,000층까지 확장하는 것이 기술적으로 가능하다는 것을 입증하였다. 이는 모델의 용량을 키우기 위해 단순히 너비(hidden size)를 넓히는 것뿐만 아니라 깊이를 극단적으로 늘리는 방향이 유망한 전략이 될 수 있음을 시사한다.

그러나 몇 가지 비판적 해석과 한계점이 존재한다.
첫째, 모델의 깊이는 1,000층으로 매우 깊지만, hidden size가 BERT의 경우 64, GPT의 경우 256으로 현대적인 거대 언어 모델(LLM)에 비해 매우 작다. 즉, '매우 깊지만 매우 얇은' 구조이다.
둘째, LAMBADA와 같은 일부 데이터셋에서 성능이 극도로 낮게 나타난 점은, 단순히 층을 깊게 쌓는 것만으로는 모든 자연어 처리 작업에서 성능 향상을 꾀할 수 없음을 보여준다.
셋째, 제안된 상수 $0.974$가 경험적(empirical)으로 도출되었다는 점은 이론적 일반화 가능성에 대한 추가 연구가 필요함을 의미한다.

## 📌 TL;DR

본 연구는 BERT와 GPT를 1,000층까지 확장하기 위해 잔차 연결 부위의 상수를 조정하는 **Upscale Layer Normalization** 및 **Foundation Layer Normalization**을 제안하였다. 실험 결과, 매우 적은 파라미터 수와 연산량으로도 일부 벤치마크에서 거대 모델과 경쟁 가능한 성능을 확인하였으며, 이는 Transformer 모델의 확장 방향으로서 '극단적인 깊이'의 가능성을 제시한 연구이다.
