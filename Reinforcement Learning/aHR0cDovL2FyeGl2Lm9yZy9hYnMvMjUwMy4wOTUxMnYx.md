# REINFORCEMENTLEARNING IS ALL YOU NEED
Yongsheng Lian

## Problem to Solve
이 논문은 **DeepSeek R1**의 성공에 영감을 받아, 인간 피드백 없이 **순수 강화 학습**(**pure reinforcement learning**, **RL**)만으로 언어 모델의 **추론 능력**(**reasoning capabilities**)을 향상시킬 수 있는지 탐구합니다. 구체적으로는 강화 학습만으로 훈련된 3B 언어 모델이 훈련 데이터를 넘어선 **일반화**(**generalization**) 능력을 보이는지, 그리고 **"아하 모먼트"**(**aha moments**)와 응답 길이(response length)가 추론 품질에 어떻게 영향을 미치는지 밝히는 것을 목표로 합니다.

## Key Contributions
*   **RL-Only 훈련의 효과 입증**: **순수 RL 훈련**만으로 언어 모델의 추론 능력을 크게 향상시킬 수 있음을 입증했습니다.
*   **벤치마크 성능 향상**: 훈련된 모델이 5개 중 4개 벤치마크에서 **기준 모델**(**Base Model**)을 능가하며, 훈련 데이터 범위를 넘어선 **일반화**(**generalization**) 능력을 성공적으로 시연했습니다.
*   **"아하 모먼트" 분석**: 모델에서 **"아하 모먼트"**(**aha moments**)라는 통찰의 순간이 재현되었으나, 이러한 순간이 항상 **정확한 답변**(**correct answers**)으로 이어지는 것은 아님을 발견했습니다. 이는 모델이 **내부 디버깅 프로세스**(**internal debugging process**)를 거치지만, **강력한 검증 메커니즘**(**robust verification mechanism**)이 필요함을 시사합니다.
*   **응답 길이와 추론 품질**: 응답 길이가 추론 품질과 직접적인 상관관계가 없으며, 훈련이 진행될수록 모델이 **무차별 대입 방식**(**brute-force approach**)에서 더 간결하고 **인간 유사 추론 방식**(**human-like reasoning approaches**)으로 전환하는 경향을 보였습니다.
*   **`GRPO` 알고리즘의 활용**: `DeepSeek R1`에서 사용된 `GRPO`(**Group Relative Policy Optimization**) 알고리즘을 활용하여 **계산 효율성**(**computational efficiency**)을 확보했지만, `PPO`(**Proximal Policy Optimization**) 대비 정확도 및 **길이 해킹**(**length hacking**) 문제에 대한 통찰을 제공했습니다.

## Methodology
*   **훈련 데이터셋**: 언어 모델 훈련을 위해 잘 알려진 숫자 퍼즐인 `Countdown Game`을 활용했습니다. 이 게임은 무작위 숫자와 기본 산술 연산(`+`, `-`, `×`, `÷`)을 사용하여 목표 숫자에 도달하는 문제로, **수리 추론**(**numerical reasoning**) 훈련에 이상적인 데이터셋을 제공합니다.
*   **규칙 기반 보상 모델링**(**Rule-Based Reward Modeling**):
    *   **형식 보상**(**Format Reward**): 응답이 ` <think> </think> ` 태그로 추론 과정을 명시하고, ` <answer> </answer> ` 태그로 최종 방정식과 결과를 명확히 제시하는 등 특정 형식을 따르는지 **정규 표현식 기반**(**regular expression-based**)으로 검증합니다.
    *   **답변 보상**(**Answer Reward**): ` <answer> </answer> ` 태그 내의 최종 결과가 목표 숫자와 정확히 일치하는 경우 1, 그렇지 않은 경우 0을 부여하는 **이진 보상 함수**(**binary reward function**)를 사용합니다.
*   **강화 학습 알고리즘**: `DeepSeek-R1` 훈련에서 사용된 `GRPO`(**Group Relative Policy Optimization**) 방법을 사용했습니다. `GRPO`는 `PPO`와 달리 명시적인 **가치 함수**(**value function**) 없이도 정책을 최적화하며, 다음과 같이 **어드밴티지 함수**(**advantage function**)를 계산하여 **계산 복잡성**(**computational complexity**)을 줄입니다.
    $$J_{\text{GRPO}}(\theta) = E\left[ \sum_{i=1}^{G} \min\left( \frac{\pi_{\theta}(o_i)}{\pi_{\theta_{\text{old}}}(o_i)} A_i, \text{clip}\left( \frac{\pi_{\theta}(o_i)}{\pi_{\theta_{\text{old}}}(o_i)}, 1-\epsilon, 1+\epsilon \right) A_i \right) - \beta D_{\text{KL}}(\pi_{\theta} \| \pi_{\text{ref}}) \right]$$
    $$A_i = \frac{r_i - \text{mean}(\{r_1,r_2,\cdots,r_G\})}{\text{std}(\{r_1,r_2,\cdots,r_G\})}$$
    여기서 $A_i$는 해당 그룹 내에서 정규화된 보상으로, 동적인 "스코어 라인" 역할을 합니다.
*   **벤치마크 평가**: 훈련된 모델의 성능을 `GSM8K`, `IFEval`, `BBH`, `MATH`, `MMLU-Pro` 등 5가지 벤치마크에서 **기준 모델**(**Base Model**)과 **R1 프롬프트**(**R1 Prompt**)를 적용한 기준 모델과 비교 평가했습니다. 모든 평가는 **제로샷**(**zero-shot**) 설정에서 진행되었습니다.

## Results
*   **초기 훈련 단계**: 모델은 ` <think> ` 태그를 여러 번 중첩하는 등 예상된 응답 형식을 위반하는 경향을 보였습니다.
*   **인간 유사 사고의 초기 징후**: 약 80단계 훈련 후, 모델은 올바른 응답 구조를 따르기 시작했으나, 논리적인 단계별 추론보다는 **무차별 대입 방식**(**brute-force, trial-and-error approach**)으로 해답을 찾았습니다. "too high"와 같은 반응으로 **검증 기반 사고**(**verification-based thinking**)를 보였지만, 진정한 **성찰**(**reflection**)이나 **전략적 적응**(**strategic adaptation**)은 부족했습니다.
*   **인간 유사 사고의 발현**: 300단계 훈련 후, 모델은 체계적인 탐색, 패턴 인식, 그리고 잘못된 추론 경로를 **되돌아가는**(**backtracking**) 능력을 보여주며 **인간 유사 사고**(**human-like thinking**)를 발현했습니다. 이는 효율적이고 직접적인 문제 해결 접근 방식이었습니다.
*   **"아하 모먼트"**: **DeepSeek R1**에서 보고된 **"아하 모먼트"**(**aha moments**)가 재현되었고, 모델이 "But wait"과 같은 문구를 통해 내부 디버깅 과정을 보여주었지만, 최종 답변의 정확도를 항상 보장하지는 않았습니다. 이는 추론의 돌파구와 정확성 사이에 **강력한 검증 메커니즘**(**robust verification mechanism**)이 필요함을 시사합니다.
*   **응답 길이 대 추론 능력**: 훈련이 진행됨에 따라 응답 길이가 초기 **무차별 대입 방식**(**brute-force approach**)에서 **추론 기반 방식**(**reasoning-based approach**)으로 전환되면서 점차 짧아지는 경향을 보였습니다. 때로는 추론 과정 없이 바로 정답을 제공하기도 했습니다.
*   **훈련 모델의 성능**:
    *   **GSM8K**: 훈련된 모델은 `Flexible-Extract`에서 69.7%, `Strict-Match`에서 64.8%의 성능을 달성하여 **기준 모델**(**Base Model**)을 크게 능가했습니다. 이는 **강화 학습**(**reinforcement learning**, **RL**)이 **추론 깊이**(**reasoning depth**)와 **일관성**(**consistency**)을 향상시킴을 보여줍니다.
    *   **MATH**: `Math Verify` 지표에서 훈련된 모델은 **기준 모델**(**Base Model**)의 13%에서 27%로 **100% 이상 상대적 향상**을 보였습니다. 이는 출력 형식과 무관하게 **수학적 추론 능력**(**mathematical reasoning ability**)이 크게 향상되었음을 의미합니다.
    *   **BBH**: 훈련된 모델은 44.6%의 `Extract Match` 점수를 달성하여 **기준 모델**(**Base Model**) 대비 **7% 포인트 개선**을 보였습니다. 특히 `Date Understanding`, `Disambiguation QA`, `Logical Deduction` 등 복잡한 추론 작업에서 큰 성능 향상을 기록했습니다.
    *   **MMLU-Pro**: 훈련된 모델은 `Extract Match`에서 22.4%를 달성하여 **기준 모델**(**Base Model**) 대비 **6.2% 포인트 향상**을 보였습니다. `Psychology`(**+68.5%**), `Biology`(**+60.5%**), `Mathematics`(**+38.6%**) 분야에서 특히 두드러진 개선을 보여, **도메인 특화 지식**(**domain-specific knowledge**)과 **논리적 추론**(**logical inference**) 통합 능력이 강화되었음을 시사합니다.
    *   **IFEval**: 훈련된 모델의 성능은 **기준 모델**(**Base Model**) 대비 소폭 감소($\sim 1\%$)했습니다. 이는 `Countdown Game`을 통한 훈련이 **명령 따르기 능력**(**instruction-following capabilities**)에는 직접적인 긍정적 영향을 미치지 않았음을 보여줍니다.