# Spurious Rewards: Rethinking Training Signals in RLVR

Rulin Shao, Shuyue Stella Li, Rui Xin, Scott Geng, Yiping Wang, Sewoong Oh, Simon Shaolei Du, Nathan Lambert, Sewon Min, Ranjay Krishna, Yulia Tsvetkov, Hannaneh Hajishirzi, Pang Wei Koh, Luke Zettlemoyer (2025)

## 🧩 Problem to Solve

본 논문은 검증 가능한 보상(Verifiable Rewards)을 이용한 강화학습(RLVR, Reinforcement Learning with Verifiable Rewards)이 언어 모델의 수학적 추론 능력을 향상시키는 정확한 메커니즘이 무엇인지 탐구한다.

기존의 RLVR 연구들은 정답 여부에 기반한 정확한 보상 신호가 모델의 추론 능력을 '학습'시킨다고 가정한다. 그러나 본 연구는 이러한 보상 신호가 매우 약하거나, 심지어 정답과 상관관계가 없는 '가짜 보상(Spurious Rewards)'일 때조차 특정 모델에서 성능 향상이 일어나는 현상에 주목한다. 이는 RLVR이 새로운 능력을 가르치는 것이 아니라, 사전 학습(Pretraining) 단계에서 이미 습득한 잠재적 능력을 표면으로 끌어올리는(Elicit) 역할만 수행할 가능성이 있음을 시사한다. 따라서 본 논문의 목표는 가짜 보상이 어떻게 성능 향상을 유도하는지 분석하고, 이것이 모델의 사전 학습 상태에 따라 어떻게 달라지는지 밝히는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 다음과 같다.

1. **가짜 보상의 효과 발견**: Qwen2.5-Math 모델의 경우, 무작위 보상(Random Reward)이나 틀린 정답에 보상을 주는 방식(Incorrect Reward)과 같은 가짜 보상만으로도 정답 기반 보상(Ground Truth Reward)에 근접하는 상당한 성능 향상을 얻을 수 있음을 발견하였다.
2. **Code Reasoning의 역할 규명**: Qwen 모델이 보여주는 성능 향상의 핵심이 'Code Reasoning'(실제 실행은 하지 않지만 추론 과정에서 파이썬 코드를 작성하는 행위)이라는 특정 추론 전략의 빈도를 높이는 데 있음을 밝혀냈다.
3. **GRPO Clipping Bias 메커니즘 제시**: 무작위 보상에서도 성능이 오르는 이유를 GRPO(Group Relative Policy Optimization) 알고리즘의 Clipping 메커니즘이 가진 편향(Bias)에서 찾았다. 이 편향이 모델이 기존에 가지고 있던 고확률의 행동 패턴(Prior)을 강화하는 방향으로 작용하여, 결과적으로 유용한 추론 전략인 Code Reasoning을 강화시킨다는 가설을 제시하고 실험적으로 검증하였다.
4. **모델 의존성 경고**: 이러한 현상이 Qwen 모델군에서는 뚜렷하지만 Llama3나 OLMo2와 같은 다른 모델군에서는 나타나지 않음을 보여줌으로써, RLVR 연구 시 특정 모델에만 의존하여 결론을 내리는 것의 위험성을 경고하였다.

## 📎 Related Works

본 논문은 다음과 같은 관련 연구들을 다룬다.

- **RLVR 및 추론 향상**: DeepSeek-Math, Tulu 3 등 검증 가능한 보상을 통해 수학적 추론을 개선하려는 시도들이 있었다. 최근 일부 연구에서는 RL이 모델의 새로운 능력을 창조하기보다 내재된 능력을 증폭시킨다는 가설을 제기하였다.
- **비지도 강화학습(Unsupervised RL)**: ScPO나 TTRL(Test-Time RL)과 같이 정답 없이 모델 내부의 일관성(Consistency)이나 다수결(Majority Vote)을 통해 보상을 생성하는 방식들이 제안되었다.

**기존 연구와의 차별점**: 기존 연구들이 '정답이 없는 상황에서 어떻게 유용한 보상을 만들 것인가'에 집중했다면, 본 논문은 '완전히 무의미하거나 틀린 보상을 주어도 성능이 오르는가'라는 극단적인 설정을 통해 RLVR의 본질적인 작동 방식과 모델 사전 학습의 중요성을 분석했다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 전체 파이프라인 및 학습 구조

본 연구는 GRPO(Group Relative Policy Optimization) 알고리즘을 사용하여 Qwen2.5-Math 및 기타 모델들을 미세 조정한다. GRPO는 별도의 가치 함수(Value Function) 모델 없이 그룹 내 상대적 보상을 사용하여 정책을 업데이트하는 효율적인 알고리즘이다.

### 보상 함수(Reward Functions) 설계

연구진은 보상 신호의 강도와 정확도에 따라 다음과 같이 다섯 가지 보상 함수를 설계하였다.

1. **Ground Truth Rewards**: 정답과 일치하는 경우에만 보상 $1$을 부여하는 표준 방식이다.
2. **Majority Vote Rewards**: 학습 전 모델이 생성한 64개의 답변 중 가장 많이 나온 답변을 정답으로 간주하고 보상을 부여하는 방식이다.
3. **Format Rewards**: 정답 여부와 상관없이 답변에 $\boxed{}$ 형식이 포함되어 있는지만 확인하여 보상을 부여한다.
4. **Random Rewards**: 정답과 무관하게 $\gamma$의 확률(기본값 0.5)로 보상 $1$을, 그 외에는 $0$을 부여하는 완전한 노이즈 신호이다.
5. **Incorrect Rewards**: 다수결로 뽑은 오답 중 하나를 정답으로 설정하고, 모델이 그 오답을 맞혔을 때 보상을 부여하는 역방향 보상 방식이다.

### 주요 방정식 및 학습 절차

GRPO의 핵심은 그룹 내 상대적 이점(Advantage)을 계산하는 것이다. 특정 프롬프트 $x$에 대해 생성된 그룹 내 답변들의 보상 평균 $\bar{r}_x$와 표준편차 $\sigma_x$를 이용하여 다음과 같이 Advantage $\hat{A}$를 정의한다.

$$\hat{A}(x,y) = \frac{r(x,y) - \bar{r}_x}{\sigma_x}$$

최적화 목표 함수 $J(\theta)$는 다음과 같이 PPO 스타일의 Clipping이 적용된 surrogate objective를 최대화하는 것이다.

$$J(\theta) = \mathbb{E} \left[ \sum_{t=1}^{|y|} \min \left( \rho_t(y; \theta) \hat{A}(x,y), \text{clip}(\rho_t(y; \theta), 1-\varepsilon^c, 1+\varepsilon^c) \hat{A}(x,y) \right) \right]$$

여기서 $\rho_t(y; \theta)$는 현재 정책과 이전 정책의 확률 비율이다. 본 연구에서는 분석의 단순화를 위해 KL 발산(KL divergence) 패널티 항을 제거($\lambda=0$)하고 학습을 진행하였다.

## 📊 Results

### 실험 설정

- **데이터셋**: MATH-500, AMC, AIME 2024/2025.
- **평가 지표**: Pass@1 accuracy 및 average@8 accuracy.
- **대상 모델**: Qwen2.5-Math (7B, 1.5B), Qwen2.5 (7B, 1.5B), Llama3.1 (8B), Llama3.2 (3B), OLMo2 (7B).

### 주요 결과

1. **Qwen 모델의 놀라운 강건성**: Qwen2.5-Math-7B 모델은 무작위 보상(Random)으로 학습했을 때 MATH-500 정확도가 $21.4\%$ 상승하였으며, 틀린 정답 보상(Incorrect)으로는 $24.1\%$ 상승하였다. 이는 정답 보상(Ground Truth)으로 얻은 $29.1\%$ 상승분과 비교했을 때 매우 근접한 수치이다.
2. **모델군별 차이**: 위와 같은 가짜 보상의 효과는 Llama3나 OLMo2 모델에서는 거의 나타나지 않았으며, 일부의 경우 오히려 성능이 하락하였다. 이는 RLVR의 효과가 모델의 사전 학습 상태에 극도로 의존함을 보여준다.
3. **Code Reasoning과의 상관관계**: Qwen 모델에서 RLVR 학습이 진행됨에 따라, 정답 여부와 상관없이 파이썬 코드를 사용하여 추론하는 'Code Reasoning'의 빈도가 $65\%$에서 $90\%$ 이상으로 급증하였으며, 이는 정확도 향상 곡선과 강하게 일치하였다.
4. **인위적 유도 실험**: 프롬프트를 통해 강제로 "파이썬을 사용하여 풀자"라고 지시하거나, 코드 작성 여부에만 보상을 주는 'Python Reward'를 적용했을 때, Qwen 모델의 성능이 비약적으로 향상되었다.

## 🧠 Insights & Discussion

### RLVR의 본질: 학습인가, 인출인가?

본 논문은 RLVR이 모델에게 새로운 수학적 원리를 가르치는 것이 아니라, 사전 학습 중에 이미 습득했지만 명시적으로 드러나지 않았던 '유용한 추론 전략(예: Code Reasoning)'을 더 자주 사용하도록 만드는 **인출(Elicitation)** 과정임을 시사한다.

### 무작위 보상이 작동하는 이유: Clipping Bias

가장 의아한 점인 '무작위 보상'의 효과는 GRPO의 Clipping 메커니즘으로 설명된다.

- 이론적으로 무작위 보상의 기대 Advantage는 $0$이다.
- 하지만 Clipping 항은 확률 비율 $\rho$가 특정 범위 $(1-\varepsilon^c, 1+\varepsilon^c)$를 벗어나지 않게 제한한다.
- 사전 확률이 높은 토큰일수록 Clipping 범위가 상대적으로 넓어 패널티를 덜 받게 되며, 결과적으로 모델은 새로운 탐색(Exploration)보다는 기존의 고확률 패턴(Prior)을 강화하는 방향으로 업데이트된다.
- Qwen 모델은 이미 유용한 Code Reasoning 패턴을 Prior로 가지고 있었기에, 이 Bias가 결과적으로 성능 향상을 이끌어낸 것이다.

### 한계 및 비판적 해석

- **데이터 오염 가능성**: AIME 2025(학습 컷오프 이후 데이터)에서는 가짜 보상의 효과가 사라진 것을 보아, 가짜 보상이 작동하려면 모델이 해당 유형의 문제에 대한 어느 정도의 사전 지식을 가지고 있어야 함이 확인되었다.
- **일반화의 문제**: 본 연구는 Code Reasoning을 주요 사례로 들었으나, 저자들도 언급했듯이 '반복 없는 생성(No-repetition)'과 같은 다른 패턴들도 성능에 영향을 줄 수 있다. 따라서 Code Reasoning이 유일한 요인은 아닐 수 있다.

## 📌 TL;DR

본 논문은 RLVR이 정답과 무관한 가짜 보상(무작위, 오답, 형식 보상)만으로도 Qwen2.5-Math 모델의 성능을 크게 향상시킬 수 있음을 발견하였다. 이는 GRPO의 Clipping Bias가 모델의 내재된 유용한 추론 전략(특히 Code Reasoning)을 강화하기 때문에 발생하는 현상이다. 결론적으로 RLVR의 성능 향상은 보상 신호 자체의 품질보다 **모델의 사전 학습 상태와 최적화 알고리즘의 상호작용**에 더 크게 의존하며, 따라서 새로운 RLVR 방법론을 검증할 때 Qwen 모델 하나에만 의존하는 것은 매우 위험하다는 실무적 경고를 전달한다.
