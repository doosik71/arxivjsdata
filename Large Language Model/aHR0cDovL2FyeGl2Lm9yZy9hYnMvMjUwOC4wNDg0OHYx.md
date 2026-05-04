# Large Language Models Reasoning Abilities Under Non-Ideal Conditions After RL-Fine-Tuning

Chang Tian et al. (2025)

## 🧩 Problem to Solve

본 논문은 강화학습(Reinforcement Learning, RL)을 통해 미세 조정(Fine-tuning)된 대규모 언어 모델(LLM)들이 실제 환경과 같은 **비이상적인 조건(Non-ideal conditions)**에서도 강건한 추론 능력을 유지하는지를 분석한다.

현재 대부분의 LLM 추론 벤치마크는 노이즈가 없는 이상적인 설정에서 평가된다. 하지만 실제 세계의 추론은 다양한 가능성을 종합하여 결론을 내리거나, 세밀한 노이즈를 억제하고, 불필요한 문맥을 필터링하는 등의 **고급 추론 능력(Advanced Reasoning Abilities)**을 요구한다. 저자들은 RL-fine-tuning이 기본적인 추론 성능은 높일 수 있으나, 이러한 복잡하고 비이상적인 시나리오에서는 성능이 급격히 저하될 수 있다는 문제 제기를 한다.

따라서 본 연구의 목표는 뇌과학적 통찰을 바탕으로 세 가지 비이상적 시나리오를 정의하고, RL-fine-tuned 모델들의 한계를 정량적으로 평가하며, 이를 해결하기 위한 완화 전략(Remediation strategy)을 탐색하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같다.

1.  **비이상적 시나리오의 정의**: 뇌과학의 연구 결과(인간의 추론은 불완전한 입력 하에서도 신뢰성을 유지함)에서 영감을 얻어, **요약 추론(Summary Inference)**, **세밀한 노이즈 억제(Fine-grained Noise Suppression)**, **문맥 필터링(Contextual Filtering)**이라는 세 가지 도전적인 추론 시나리오를 제안하였다.
2.  **성능 저하의 실증적 증명**: RL-fine-tuning이 이상적인 조건에서는 성능을 향상시키지만, 비이상적인 조건에서는 성능이 유의미하게 하락함을 여러 LLM과 LVLM(Large Vision-Language Model) 실험을 통해 증명하였다.
3.  **완화 전략 제안**: 포맷 보상(Format reward)을 통한 학습 방식의 변경(Stage C)과 가이드 예시(Example guidance) 제공(Stage G, H)을 통해 비이상적 조건에서의 성능 저하를 일부 완화할 수 있는 방법을 제시하였다.
4.  **새로운 평가 데이터셋 공개**: 세밀한 방해 요소(Distractors)와 불필요한 문맥 정보가 포함된 `FineTest` 및 `FilterTest` 데이터셋을 구축하여 공개하였다.

## 📎 Related Works

### RL for Enhancing LLM Reasoning
LLM의 추론 능력을 높이기 위한 RL 기법은 크게 두 가지로 나뉜다.
- **Monte Carlo 기반 방법**: MCTS-DPO나 ReST-MCTS와 같이 MCTS를 통해 우수한 추론 궤적을 탐색하고 이를 학습에 활용하는 방식이다.
- **Policy Gradient 기반 방법**: PPO, GRPO, DAPO와 같이 보상 피드백을 통해 정책 파라미터를 직접 최적화하는 방식이다. 본 논문에서는 계산 효율성이 높은 **GRPO**를 채택하여 모델을 미세 조정하였다.

### Assessing the Performance of LLMs
MMLU와 같은 표준 벤치마크가 널리 사용되고 있으나, 데이터 오염(Data Contamination)으로 인해 성능이 과대평가되는 문제가 제기되어 왔다. MMLU-Pro와 같은 더 어려운 벤치마크가 등장했음에도 불구하고, 본 논문은 단순한 난이도 상승이 아닌 '비이상적 입력 조건'에서의 강건성을 평가한다는 점에서 기존 접근 방식과 차별화된다.

## 🛠️ Methodology

### GRPO (Group Relative Policy Optimization)
본 연구에서는 GPU 메모리 효율성이 높고 가치 모델(Value model)이 필요 없는 GRPO 알고리즘을 사용하여 모델을 학습시켰다.

GRPO의 목적 함수는 다음과 같이 정의된다.
$$J^{GRPO}(\theta) = \mathbb{E}_{q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{old}}(O|q)} \left[ \frac{1}{G} \sum_{i=1}^G \left( \min \left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)} A_i, \text{clip} \left( \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}, 1-\epsilon, 1+\epsilon \right) A_i \right) - \beta D_{KL}(\pi_\theta \| \pi_{ref}) \right) \right]$$

여기서 $D_{KL}$은 참조 모델과의 거리 제한을 위한 KL 발산 항이며, $A_i$는 그룹 내 상대적 보상을 이용한 Advantage이다. Advantage는 다음과 같이 계산된다.
$$A_i = \frac{r_i - \text{mean}(\{r_1, r_2, \dots, r_G\})}{\text{std}(\{r_1, r_2, \dots, r_G\})}$$
즉, 개별 응답의 보상 $r_i$를 그룹의 평균과 표준편차로 정규화하여, 상대적으로 더 좋은 응답에 더 높은 가중치를 부여한다.

### 세 가지 비이상적 시나리오 및 학습/평가 단계

#### 1. 요약 추론 (Summary Inference)
여러 선택지가 있을 때 각각을 가정하여 분석하고 이를 종합하여 최종 결론을 내리는 능력이다.
- **Stage A (표준 학습)**: $\text{<reasoning>}$과 $\text{<answer>}$ 태그를 사용하는 기본 포맷 학습.
- **Stage C (요약 학습)**: 각 선택지가 정답이라고 가정하고 분석한 뒤, 이를 요약하여 최종 답을 내도록 학습하며 이에 대한 포맷 보상을 제공한다.
- **Stage B $\rightarrow$ D (평가)**: Stage B는 일반적인 평가 방식이며, Stage D는 명시적으로 모든 옵션을 분석하고 요약하라는 지시를 추가한 비이상적 평가 방식이다.

#### 2. 세밀한 노이즈 억제 (Fine-grained Noise Suppression)
질문 텍스트 내에 삽입된 미세한 방해 요소(예: 수학 식 중간의 무의미한 기호)를 제거하고 핵심 정보에 집중하는 능력이다.
- **FineTest**: 원본 테스트셋에 세밀한 방해 요소를 합성하여 구축하였다.
- **완화 전략**: 학습(Stage G) 또는 평가(Stage H) 시에 노이즈를 식별하고 무시하는 방법을 보여주는 **가이드 예시(Example guidance)**를 제공한다.

#### 3. 문맥 필터링 (Contextual Filtering)
질문 앞부분에 추가된 무관한 문맥(예: 날씨, 뉴스 등)을 무시하고 질문의 본질을 파악하는 능력이다.
- **FilterTest**: 질문 앞에 무관한 텍스트를 추가하여 구축하였다.
- **완화 전략**: 세밀한 노이즈 억제와 마찬가지로 가이드 예시(Stage G, H)를 통해 모델의 잠재적 필터링 능력을 활성화한다.

## 📊 Results

### 실험 설정
- **모델**: Llama 3.1-8B, Qwen 3-14B, Mistral-Small-24B (LLM) 및 Qwen 2.5-VL-7B (LVLM).
- **데이터셋**: CommonsenseQA, Ceval-exam, Math12k, MathReasoning, Mathverse, MathVision 등 총 8개 데이터셋.
- **지표**: Pass@1 Accuracy ($\text{ACC}$).

### 주요 결과 분석
1.  **요약 추론**:
    - RL-fine-tuning(Stage AB)은 이상적 조건(Stage B)에서 성능을 크게 향상시켰다.
    - 그러나 비이상적 조건(Stage AD)으로 변경하면 모든 모델에서 성능이 하락하였다.
    - 요약 추론 전용 학습(Stage CD)을 진행하면 성능이 회복되지만, 여전히 기본 모델 대비 격차가 존재한다.
2.  **노이즈 억제 및 필터링**:
    - **이상적 환경 $\rightarrow$ 비이상적 환경**: `FineTest`와 `FilterTest`에서 성능이 눈에 띄게 감소하였다. 특히 Mistral 모델이 세밀한 노이즈에 취약한 모습을 보였다.
    - **Llama 3.1의 특이점**: Llama 3.1은 RL 학습 후 오히려 이상적 조건에서의 성능이 하락하는 '정책 붕괴(Policy collapse)' 현상이 관찰되었다. 이는 보상이 희소한(Sparse reward) 환경에서 Advantage $A_i$가 0에 수렴하며 KL 항이 모델을 잘못된 방향으로 업데이트했기 때문으로 분석된다.
    - **완화 전략의 효과**: 가이드 예시를 제공하는 Stage EH와 GH는 성능을 개선시켰다. 특히 Qwen3는 평가 시 예시만 주어도(Stage EH) 성능이 올랐으나, Llama 3.1은 학습 단계부터 예시가 포함되어야(Stage GH) 성능 향상이 나타났다.

## 🧠 Insights & Discussion

본 논문은 RL-fine-tuning이 가져오는 성능 향상이 상당 부분 '이상적인 벤치마크'에 최적화된 결과일 수 있음을 시사한다.

**강점 및 발견**:
- **모델 크기의 영향**: Mistral(24B)과 Qwen3(14B)가 Llama 3.1(8B)보다 우수한 성능을 보였으며, 이는 파라미터 수가 많을수록 지식 용량과 표현력이 증가하여 추론 강건성이 높아짐을 의미한다.
- **잠재 능력의 활성화**: 가이드 예시를 통해 성능이 향상된 것은 모델이 이미 기본적인 추론 능력을 갖추고 있으나, 이를 비이상적 시나리오에서 인출(Retrieve)하는 능력이 부족함을 보여준다.

**한계 및 비판적 해석**:
- RL-fine-tuned 모델들이 여전히 비이상적 조건에서 취약하다는 점은, 현재의 RL 학습 방식이 정답 도출이라는 결과(Outcome)에만 집중할 뿐, 인간처럼 입력을 정제하고 처리하는 **과정의 강건성(Robustness of process)**을 충분히 학습시키지 못하고 있음을 드러낸다.
- 제안된 완화 전략(예시 제공)은 일종의 '힌트'를 주는 방식이므로, 모델 자체의 내재적 추론 능력이 완전히 해결되었다고 보기 어렵다.

## 📌 TL;DR

본 논문은 RL로 미세 조정된 LLM들이 노이즈가 없는 이상적 환경에서는 뛰어나지만, 실제 세계의 비이상적 조건(요약 추론, 노이즈 억제, 문맥 필터링)에서는 성능이 급격히 떨어진다는 것을 밝혀냈다. 저자들은 뇌과학 기반의 평가 시나리오와 노이즈 데이터셋을 제안하였으며, 포맷 보상과 가이드 예시를 통해 이를 완화할 수 있음을 보였다. 이 연구는 LLM의 추론 능력이 과대평가 되었을 가능성을 제기하며, 향후 모델 평가 시 강건성 테스트의 중요성을 강조한다.