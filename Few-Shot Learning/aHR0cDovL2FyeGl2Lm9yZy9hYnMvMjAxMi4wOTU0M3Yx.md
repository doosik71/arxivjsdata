# Few-shot Sequence Learning with Transformers

Lajanugen Logeswaran, Ann Lee, Myle Ott, Honglak Lee, Marc’Aurelio Ranzato, Arthur Szlam (2020)

## 🧩 Problem to Solve

본 논문은 이산 시퀀스(discrete sequences) 데이터에 대한 Few-shot learning 문제를 해결하고자 한다. 일반적으로 Few-shot learning은 컴퓨터 비전 분야에서 활발히 연구되어 왔으나, 자연어 처리(NLP)나 강화 학습(RL)과 같이 토큰의 시퀀스로 이루어진 데이터를 다루는 영역에서의 연구는 상대적으로 부족한 실정이다.

특히, 새로운 작업(task)이 주어졌을 때 아주 적은 양의 훈련 예시(handful of training examples)만으로 모델을 빠르게 적응시키는 것이 핵심이다. 본 연구의 목표는 Transformer 아키텍처를 기반으로 하여, 복잡한 모델 구조의 변경이나 계산 비용이 높은 고차 미분(second order derivatives) 없이도 효율적으로 시퀀스 분류(sequence classification) 및 변환(transduction) 작업을 수행할 수 있는 학습 알고리즘을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 모델의 모든 파라미터를 업데이트하는 대신, 해당 작업의 특성을 함축하는 소규모의 **Task Embedding** 벡터 $z$만을 최적화하여 모델을 적응시키는 것이다.

주요 기여 사항은 다음과 같다:
1. **TAM (Transformer trained with Alternating Minimization)** 알고리즘 제안: 공유 파라미터 $\theta$와 작업별 파라미터 $z$를 교대로 최적화하는 단순하고 효율적인 학습 방식을 도입하였다.
2. **Input Conditioning** 방식의 효용성 입증: Task embedding을 모델의 입력 시퀀스에 토큰 형태로 추가하는 것만으로도 충분한 작업 조건화(conditioning)가 가능함을 보여주었다.
3. **Compositional Task Descriptors**의 도입: 작업을 구성하는 기본 단위(primitives) 정보를 제공함으로써, 학습 시 보지 못한 새로운 작업에 대해서도 더 나은 일반화 성능을 보임을 입증하였다.
4. **통제된 벤치마크 구축**: 시퀀스 분류, 시퀀스 변환, 경로 찾기(Path finding)라는 세 가지 합성 데이터셋을 통해 다양한 Few-shot baseline 모델들의 성능을 정량적으로 분석하였다.

## 📎 Related Works

논문에서는 Few-shot learning 및 Meta-learning과 관련된 기존 연구들을 다음과 같이 분류하고 차별점을 제시한다.

1. **Meta-learners**: MAML(Model-Agnostic Meta-Learning)과 같이 학습 알고리즘 자체를 학습하여 빠르게 적응하는 방식이다. 이러한 방식은 주로 고차 미분을 사용하여 계산 비용이 매우 높다는 한계가 있다.
2. **Sample-efficient Architectures**: Matching Networks나 SNAIL과 같이 특정 아키텍처를 통해 샘플 효율성을 높이는 방식이다. 하지만 이러한 메모리 기반 방식은 시퀀스 길이가 길어질 때 성능이 저하되는 경향이 있다.
3. **Task Transfer for Transformers**: Transformer의 입력에 특수 토큰을 추가하여 작업을 전환하는 방식들이 존재한다. 본 논문은 이 직관을 meta-learning 설정으로 확장하여, 이 토큰(Task embedding)을 최적화 대상인 파라미터로 취급하였다.

TAM은 CAVIA와 유사하게 '빠른 가중치(fast weights)'와 '느린 가중치(slow weights)'를 구분하지만, 고차 미분을 사용하지 않고 교대 최소화(Alternating Minimization) 방식을 사용하여 계산 효율성을 극대화했다는 점에서 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조
본 연구에서는 Transformer 아키텍처를 기반으로 하며, 작업의 종류에 따라 다음과 같이 구성한다.
- **시퀀스 분류**: BERT와 유사한 Transformer Encoder를 사용한다. 입력 시퀀스 앞에 Task embedding $z$를 추가하며, 최종 레이어에서 $z$ 위치의 표현물을 사용하여 분류 헤드(classification head)를 통해 결과를 출력한다.
- **시퀀스 변환**: Transformer Decoder를 사용하며, 입력 시퀀스에 $z$를 추가하여 조건부 확률을 모델링한다.

### 주요 방정식 및 학습 목표
모델은 입력 시퀀스 $x$, 작업 임베딩 $z$, 공유 파라미터 $\theta$가 주어졌을 때 출력 $y$의 로그 가능도(log-likelihood)를 최대화하는 것을 목표로 한다.
$$\log p(y|x, z; \theta)$$
시퀀스 변환의 경우, 연쇄 법칙(chain rule)에 의해 다음과 같이 분해된다.
$$\log p(y|x, z; \theta) = \sum_{i} \log p(y_i | y_{i-1}, \dots, y_1, x, z; \theta)$$

### TAM 학습 절차 (Alternating Minimization)
TAM은 공유 파라미터 $\theta$와 작업별 임베딩 $z$를 다음과 같은 단계로 학습한다 (Algorithm 1 참조).

1. **작업 샘플링**: 훈련 데이터셋에서 특정 작업 $T^{train}_i$를 샘플링한다.
2. **Inner Loop (Task Embedding 최적화)**: $\theta$를 고정한 상태에서, 해당 작업의 소수 예시들에 대해 $z$를 경사 하강법(gradient descent)으로 최적화한다.
   $$z^{T^{train}_i} \leftarrow z^{T^{train}_i} - \nabla_{z^{T^{train}_i}} \sum_{j} -\log p(y_j | x_j, z^{T^{train}_i}; \theta)$$
3. **Outer Loop (공유 파라미터 최적화)**: $z$가 최적화되는 과정에서 누적된 $\theta$에 대한 그래디언트 $\Delta \theta$를 사용하여 $\theta$를 업데이트한다.
   $$\theta \leftarrow \theta + \Delta \theta, \quad \text{where } \Delta \theta \text{ is accumulated during the inner loop.}$$

테스트 시에는 고정된 $\theta$를 유지한 채, 주어진 $k$개의 예시를 통해 새로운 작업의 $z^{test}$만을 빠르게 최적화하여 추론에 사용한다.

## 📊 Results

### 실험 설정
- **데이터셋**: 합성 시퀀스 분류(Sequence Classification), 합성 시퀀스 변환(Sequence Transduction), 그리드 월드 경로 찾기(Path Finding) 벤치마크.
- **지표**: 분류 작업은 정확도(Accuracy), 변환 및 경로 찾기 작업은 Perplexity를 측정한다.
- **비교 대상**: Task-Agnostic Transformer, Multitask Transformer, Matching Networks, SNAIL, MAML, CAVIA.

### 주요 결과
1. **정량적 성능**: $k > 1$인 설정에서 TAM은 MAML, CAVIA를 포함한 모든 베이스라인과 비슷하거나 더 우수한 성능을 보였다. 특히 $k$가 증가할수록 TAM의 성능 향상 폭이 뚜렷하였다.
2. **계산 효율성**: Table 5에 따르면, TAM은 CAVIA보다 훨씬 빠른 학습 속도를 보였다 (예: 분류 작업에서 CAVIA는 3시간, TAM은 2시간 소요되며 성능은 대등함).
3. **Compositional 성능**: 작업의 기본 구성 요소(primitives) 정보를 제공했을 때, 특히 1-shot과 같은 극단적인 Few-shot 상황에서 정확도가 크게 향상되었다. 이는 모델이 구성적 추론(compositional reasoning)을 통해 새로운 작업을 더 잘 일반화함을 시사한다.
4. **Ablation Study**: Task embedding을 입력 토큰으로 넣는 방식이 Adapter 레이어를 사용하는 것과 유사한 성능을 보였으며, Layer Normalization 파라미터를 조정하는 것보다 우수하였다. 또한, LSTM보다 Transformer 아키텍처가 TAM 알고리즘과 결합했을 때 더 높은 성능을 냈다.

## 🧠 Insights & Discussion

### 강점 및 해석
TAM은 복잡한 Meta-learning 알고리즘이 반드시 고차 미분을 필요로 하지 않는다는 점을 시사한다. 단순한 교대 최소화(Alternating Minimization)와 Transformer의 입력 조건화(Input Conditioning)만으로도 충분히 강력한 Few-shot 적응 능력을 갖출 수 있음을 입증하였다. 특히, Task embedding이 작업의 구조적 특징(예: 경로 찾기에서 목적지 좌표)을 학습하여 공간적으로 군집화된다는 PCA 시각화 결과는 모델이 내부적으로 작업의 의미론적 구조를 파악하고 있음을 보여준다.

### 한계 및 미해결 질문
- **극단적 Few-shot의 어려움**: $k=1$인 상황에서는 여전히 Matching Networks나 SNAIL 같은 메모리 기반 모델들이 상대적으로 우세한 경향이 있다. 이는 매우 적은 데이터로는 최적의 $z$를 찾아내는 것 자체가 어렵기 때문으로 해석된다.
- **데이터 의존성**: 논문에서는 Outer loop 학습을 위해 작업당 300개 이상의 예시가 필요하다고 언급하였다. 이는 실질적인 Few-shot 상황보다는 Meta-training 단계에서의 데이터 확보가 중요하다는 것을 의미한다.

## 📌 TL;DR

본 논문은 Transformer 모델을 Few-shot 시퀀스 학습에 적응시키기 위해, 작업별 특성을 담은 **Task Embedding $z$**를 입력 토큰으로 추가하고 이를 공유 파라미터 $\theta$와 교대로 최적화하는 **TAM 알고리즘**을 제안한다. 실험 결과, TAM은 MAML이나 CAVIA 같은 고차 미분 기반 모델보다 계산 효율성이 뛰어나면서도 대등하거나 더 높은 성능을 보였으며, 특히 작업의 구성적(compositional) 정보를 활용할 때 그 효과가 극대화됨을 확인하였다. 이 연구는 복잡한 아키텍처 변경 없이 단순한 입력 조건화와 최적화 전략만으로도 효율적인 시퀀스 기반 Few-shot learning이 가능함을 보여주어, 향후 NLP 및 RL 분야의 빠른 작업 적응 연구에 중요한 기초를 제공한다.