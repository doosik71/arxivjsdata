# Why Can Large Language Models Generate Correct Chain-of-Thoughts?

Rasul Tutunov, Antoine Grosnit, Juliusz Ziomek, Jun Wang, Haitham Bou-Ammar (2024)

## 🧩 Problem to Solve

본 논문은 대규모 언어 모델(LLM)에서 나타나는 **Chain-of-Thought(CoT) 프롬프팅의 이론적 근거를 규명**하고자 한다. 

많은 연구를 통해 CoT 프롬프팅이 수학적 추론이나 일반적인 추론 작업에서 성능을 비약적으로 향상시킨다는 점이 실증적으로 증명되었다. 특히, 모델에게 중간 단계의 추론 과정을 생성하도록 유도하면 직접 정답을 출력하게 하는 것보다 훨씬 높은 정답률을 보인다는 점이 확인되었다. 그러나 이러한 현상이 왜 발생하는지에 대한 심층적인 이론적 분석은 부족한 상태였다. 

따라서 본 연구의 목표는 LLM이 어떻게 정교하고 일관된 생각의 사슬(CoT)을 생성할 수 있는지, 그리고 **Few-shot CoT 예시들이 어떻게 모델로 하여금 정답으로 이끄는 추론 경로를 찾게 하는지를 수학적으로 정당화**하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 LLM의 CoT 생성 과정을 **2단계 계층적 그래픽 모델(Two-level Hierarchical Graphical Model)**로 정식화하고, 이를 통해 LLM이 생성한 CoT의 확률 분포와 실제 언어(True Language)의 확률 분포 사이의 거리가 지수적으로 수렴한다는 것을 이론적으로 증명한 것이다.

주요 아이디어는 다음과 같다:
1. **계층적 잠재 변수 모델**: 언어 생성 과정을 '문맥(Context) $\rightarrow$ 의도(Intention) $\rightarrow$ 메시지(Message)'의 계층 구조로 설계하여, CoT의 핵심인 '일관성(Coherence)'과 '관련성(Relevance)'을 모델링하였다.
2. **Marginal Approximator로서의 LLM**: LLM이 충분히 크고 많은 데이터로 학습되었다면, 실제 언어의 주변 확률 분포(Marginal Distribution)를 매우 정확하게 근사한다는 가정을 기반으로 분석을 진행하였다.
3. **모호성(Ambiguity)의 정량화**: 제공된 CoT 예시들이 얼마나 명확하게 잠재적 문맥을 드러내는지를 '모호성($\epsilon$)'이라는 지표로 정의하고, 이 지표가 낮을수록 LLM이 정답에 가까운 CoT를 생성할 확률이 높아짐을 보였다.

## 📎 Related Works

논문은 CoT와 In-Context Learning(ICL)에 관한 기존 연구들을 다음과 같이 분류하고 차별점을 제시한다.

- **실증적 연구**: Training data의 특성이나 레이블 공간의 커버리지가 ICL 성능에 영향을 준다는 연구들이 있었으나, 이는 이론적 메커니즘을 설명하지는 못한다.
- **ICL의 이론적 접근**: Xie et al. (2022)과 Jiang (2023)은 ICL을 암시적 베이지안 추론(Implicit Bayesian Inference) 또는 잠재 개념 발견(Latent Concept Discovery)으로 해석하였다. 본 논문은 이들의 접근 방식을 확장하여 CoT라는 순차적 생성 과정에 적용하였다.
- **표현력(Expressivity) 관점**: Feng et al. (2023)은 Transformer 모델이 정답을 직접 출력하려면 깊이가 매우 깊어야 하지만, CoT를 생성하면 상수 깊이(Constant depth)로도 해결 가능함을 보였다. 

**본 논문의 차별점**: 기존 연구가 "모델이 CoT를 생성할 능력이 있는가(Expressivity)"에 집중했다면, 본 논문은 "어떻게 프롬프팅을 통해 모델이 실제 정답에 맞는 CoT를 생성하도록 유도(Trigger)되는가"라는 생성 프로세스의 확률적 정당성을 다룬다.

## 🛠️ Methodology

### 1. 계층적 확률 그래픽 모델 (Hierarchical Probabilistic Graphical Model)
본 논문은 자연어 생성 과정을 다음과 같은 확률적 흐름으로 정의한다.

1. **문맥(Context, $c$)**: 가장 상위 수준의 잠재 변수로, 특정 논리 체계(예: 산술 연산, 상식 추론 등)를 결정한다. $c \sim q(c)$에서 샘플링된다.
2. **의도(Intention, $\theta_i$)**: 각 단계에서 전달하고자 하는 추상적인 목표이다. 
   - 초기 의도 $\theta_0$는 문맥 $c$에 의해 결정된다 ($\theta_0 \sim q(\cdot|c)$).
   - 이후 단계의 의도 $\theta_i$는 이전의 모든 메시지($x_{0:i-1}$), 이전의 모든 의도($\theta_{0:i-1}$), 그리고 문맥 $c$에 의존하여 생성된다. 이는 CoT의 **관련성**과 **일관성**을 보장하기 위함이다.
3. **메시지(Message, $x_i$)**: 실제로 출력되는 텍스트 토큰 시퀀스이다. 각 메시지는 해당 단계의 의도 $\theta_i$에 의해서만 결정된다 ($x_i \sim q(\cdot|\theta_i)$).

이 전체 프로세스는 다음과 같은 수식으로 표현된다:
$$c \sim q(c), \quad \theta_0 \sim q(\cdot|c), \quad x_0 \sim q(\cdot|\theta_0)$$
$$\theta_i \sim q(\cdot|x_{0:i-1}, \theta_{0:i-1}, c), \quad x_i \sim q(\cdot|\theta_i) \quad (\forall i \ge 1)$$

### 2. 자연어 모호성 (Natural Language Ambiguity)
메시지 시퀀스를 통해 원래의 문맥 $c^*$와 의도 $\theta^*$를 완벽하게 복원하지 못하는 정도를 모호성 $\epsilon$으로 정의한다.
$$q_{\text{True}}(c^*, (\theta^*_i)_{0 \le i \le m} | (x_i)_{0 \le i \le m}) = 1 - \epsilon((x_i)_{0 \le i \le m})$$
즉, $\epsilon$이 낮을수록 해당 텍스트를 통해 작성자의 의도와 문맥을 명확히 알 수 있음을 의미한다.

### 3. 핵심 정리 (Main Theorem)
LLM이 학습을 통해 실제 언어의 주변 분포 $q_{\text{True}}$를 근사한다고 가정할 때, $N$개의 CoT 예시 $Z_k$와 입력 $x_0$가 주어졌을 때 LLM이 생성하는 CoT의 확률 $p_{\text{LLM}}$와 실제 문맥 $c^*$를 알고 있을 때의 확률 $q_{\text{True}}$의 차이는 다음과 같은 상한선을 가진다.

$$\left| p_{\text{LLM}}((x_r)_{1 \le r \le m} | x_0, (Z_k)_{1 \le k \le N}) - q_{\text{True}}((x_r)_{1 \le r \le m} | x_0, c^*) \right| \le \eta \prod_{k=1}^N \frac{\epsilon(Z_k)}{1 - \epsilon(Z_k)}$$

여기서 $\eta$는 입력 작업 $x_0$의 모호성에 영향을 받는 상수이다. 이 식은 **제공된 예시들의 모호성 $\epsilon(Z_k)$이 낮고 예시의 개수 $N$이 많아질수록, LLM의 생성 분포가 실제 정답 분포로 지수적으로 수렴함**을 수학적으로 보여준다.

## 📊 Results

본 논문은 이론적 분석 논문으로, 별도의 벤치마크 데이터셋 실험 결과보다는 **수학적 증명과 조건 분석**을 통해 결과를 제시한다.

- **지수적 수렴성**: 예시 $Z_k$의 모호성이 $\epsilon(Z_k) \le \delta < 1/2$인 경우, 두 분포의 차이는 $\rho^N$ (단, $\rho = \delta/(1-\delta) < 1$)의 형태로 감소한다. 이는 $N$이 증가함에 따라 LLM이 정답 CoT를 생성할 가능성이 매우 빠르게 높아짐을 의미한다.
- **시퀀스 길이의 영향 (Lemma 4.3)**: 예시의 길이가 길어질수록(더 상세한 추론 단계가 포함될수록) 모호성 $\epsilon$이 0으로 수렴한다는 조건 하에, 충분히 긴 CoT 예시를 제공하는 것만으로도 낮은 모호성 조건을 만족시켜 성능을 높일 수 있음을 보였다.
- **비균등 사전 분포(Non-Uniform Prior) 분석**: 문맥 $c$의 사전 분포 $q(c)$가 균등하지 않을 때, 왜곡 파라미터(Skewness parameter) $\gamma(c^*)$를 도입하여 분석하였다. 이 경우 $\gamma$ 값이 클수록(데이터 불균형이 심할수록) 더 많은 예시나 더 낮은 모호성의 예시가 필요함을 수학적으로 도출하였다.

## 🧠 Insights & Discussion

### 강점 및 시사점
본 연구는 "왜 CoT 프롬프팅이 작동하는가"라는 질문에 대해, **LLM이 예시들을 통해 잠재적인 '추론 문맥(Reasoning Context)'을 추론하는 베이지안 프로세스**라는 명확한 이론적 프레임워크를 제공하였다. 특히, 단순히 예시의 양($N$)뿐만 아니라 예시의 질(모호성 $\epsilon$)과 상세함(길이 $m$)이 왜 중요한지를 수학적으로 연결했다는 점이 매우 인상적이다.

### 한계 및 논의사항
1. **가정의 현실성**: LLM이 실제 언어의 주변 분포를 완벽하게 근사한다는 가정($p_{\text{LLM}} \approx q_{\text{True}}$)은 매우 강력한 가정이다. 실제 모델에서는 근사 오차가 존재하며, 이 오차가 전체 bound에 어떤 영향을 미치는지에 대한 분석이 추가될 필요가 있다.
2. **모호성의 측정 가능성**: 이론적으로는 $\epsilon$을 정의했지만, 실제 텍스트 데이터에서 특정 CoT 예시의 모호성을 어떻게 정량적으로 측정할 수 있을지에 대한 방법론은 제시되지 않았다.
3. **실증적 검증 부족**: 본 논문은 이론적 증명에 집중하고 있어, 실제 LLM(예: GPT-4, Llama 등)에서 예시의 개수나 길이에 따른 성능 변화가 본 논문의 지수적 수렴 곡선과 일치하는지 확인하는 실험적 검증이 수반되지 않았다.

## 📌 TL;DR

본 논문은 LLM이 CoT 프롬프팅을 통해 정답을 맞히는 이유를 **계층적 확률 모델과 베이지안 추론** 관점에서 분석하였다. 결론적으로, LLM은 제공된 Few-shot 예시들을 통해 문제 해결에 필요한 **잠재적 문맥(Context)을 추론**하며, 예시의 개수가 많아지고 모호성이 낮을수록(즉, 예시가 명확하고 상세할수록) 실제 정답의 추론 경로를 생성할 확률이 지수적으로 증가함을 이론적으로 증명하였다. 이 연구는 향후 더 효율적인 프롬프트 설계 및 고품질 CoT 데이터셋 구축을 위한 이론적 토대를 제공한다.