# Med-RLVR: Emerging Medical Reasoning from a 3B base model via reinforcement Learning

Sheng Zhang, Qianchu Liu, Guanghui Qin, Tristan Naumann, Hoifung Poon (2025)

## 🧩 Problem to Solve

본 논문은 검증 가능한 보상을 통한 강화학습(Reinforcement Learning from Verifiable Rewards, RLVR)이 수학 및 코딩과 같은 정답이 명확한 도메인을 넘어, 의료와 같은 지식 집약적인 도메인에서도 추론 능력을 발현시킬 수 있는지를 탐구한다.

기존의 RLVR 연구는 주로 정답 공간이 넓고 개방적인 수학이나 코딩 문제에 집중되어 있었다. 반면, 의료 분야의 다지선다형 질문 답변(Multiple-Choice Question Answering, MCQA)은 정답 공간이 매우 좁다는 특성이 있어, 동일한 메커니즘이 작동할지 불분명했다. 또한 의료 데이터는 전문적인 임상 지식과 정교한 추론을 요구하므로 난이도가 매우 높다. 따라서 본 연구의 목표는 명시적인 추론 감독(Reasoning Supervision) 없이, 오직 정답 레이블만을 활용한 RLVR을 통해 소규모 베이스 모델(3B 파라미터)에서 의료 추론 능력을 유도하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 의료 도메인의 MCQA 데이터를 활용하여 RLVR을 적용함으로써, 모델이 스스로 추론 과정을 생성하도록 유도하는 MED-RLVR 프레임워크를 제안한 것이다.

중심적인 설계 아이디어는 정답의 정답 여부와 출력 형식을 기반으로 한 단순한 규칙 기반 보상 함수(Rule-based Reward Function)를 사용하여, 모델이 정답에 도달하기 위한 최적의 사고 경로를 스스로 찾아내게 하는 것이다. 이를 통해 지도 학습(SFT) 방식보다 더 강력한 일반화 성능을 확보하고, 작은 모델에서도 추론 능력이 창발(Emergence)할 수 있음을 입증하였다.

## 📎 Related Works

논문은 DeepSeek-R1과 같은 최근의 RLVR 연구를 언급하며, 베이스 모델에서 명시적인 감독 없이 추론 능력을 끌어낼 수 있는 가능성을 제시한다. 기존 연구들은 주로 수학적 정밀도나 코드 실행 가능성과 같은 검증 가능한 보상을 사용하였다.

기존 접근 방식과의 차별점은 적용 도메인의 확장과 정답 공간의 특성 차이에 있다. 수학/코딩 작업은 정답 공간이 방대하지만, MCQA는 선택지 중 하나를 고르는 매우 제한적인 구조를 가진다. 이러한 구조적 차이에도 불구하고 RLVR이 유효한지를 검증했다는 점이 본 연구의 독창성이다.

## 🛠️ Methodology

### 전체 파이프라인 및 시스템 구조

MED-RLVR은 Qwen2.5-3B 베이스 모델을 기반으로 하며, 별도의 명령어 튜닝(Instruction Tuning)이나 추론 데이터셋 없이 MedQA-USMLE 데이터셋의 질문과 정답 레이블만을 사용하여 학습한다. 학습 알고리즘으로는 Proximal Policy Optimization (PPO)을 채택하였다.

### 학습 목표 및 손실 함수

PPO는 이전 정책과 현재 정책의 차이가 너무 크지 않도록 제한하면서 보상을 최대화하는 방향으로 정책을 업데이트한다. 목적 함수는 다음과 같은 Clipped Surrogate Objective를 사용한다.

$$J_{PPO}(\theta) = \mathbb{E}_{q \sim P(Q), o \sim \pi_{\theta_{old}}(O|q)} \left[ \frac{1}{|O|} \sum_{t=1}^{|O|} \min \left( \frac{\pi_{\theta}(o_t|q, o_{<t})}{\pi_{\theta_{old}}(o_t|q, o_{<t})} A_t, \text{clip} \left( \frac{\pi_{\theta}(o_t|q, o_{<t})}{\pi_{\theta_{old}}(o_t|q, o_{<t})}, 1-\varepsilon, 1+\varepsilon \right) A_t \right) \right]$$

여기서 $\pi_{\theta}$는 현재 정책, $\pi_{\theta_{old}}$는 이전 정책, $A_t$는 GAE(Generalized Advantage Estimation)를 통해 계산된 Advantage 값이다. 또한, 보상 모델의 과최적화를 방지하기 위해 참조 모델(Reference Model)과의 KL penalty를 추가하여 토큰별 보상을 다음과 같이 계산한다.

$$r_t = r_{\phi}(q, o) - \beta \log \frac{\pi_{\theta}(o_t|q, o_{<t})}{\pi_{ref}(o_t|q, o_{<t})}$$

### 보상 함수 (Verifiable Rewards)

보상 함수는 매우 단순한 규칙 기반으로 설계되었으며, 출력 형식이 $\text{<think>} \dots \text{</think>} \text{<answer>} \dots \text{</answer>}$를 따르는지와 정답이 일치하는지를 검사한다.

1. **형식 검사**: 형식을 지키지 않은 경우 $-1.0$의 패널티를 부여한다.
2. **정답 검사**: 형식을 지켰으나 정답이 틀린 경우 $0.0$을 부여하고, 정답이 맞은 경우 $1.0$의 보상을 부여한다.

### 추론 절차 및 프롬프트

모델은 사용자의 질문을 받으면 먼저 내부적으로 사고 과정($\text{<think>}$)을 거친 후 최종 답변($\text{<answer>}$)을 출력하도록 유도하는 프롬프트를 사용한다.

## 📊 Results

### 실험 설정

- **데이터셋**: 학습에는 MedQA-USMLE(원본 버전)를 사용하였으며, 평가에는 MedQA-USMLE(In-distribution)와 MMLU-Pro Health(Out-of-distribution) 데이터셋을 사용하였다.
- **기준선 (Baselines)**: 동일 데이터로 학습한 SFT 모델, Qwen2.5-3B 베이스 모델(Direct/CoT 프롬프팅)과 비교하였다.
- **지표**: 정확도(Accuracy)를 측정하였다.

### 주요 결과

- **In-distribution 성능**: Med-RLVR은 MedQA 테스트 세트에서 SFT와 대등한 수준의 성능을 보였다.
- **Out-of-distribution 일반화**: MMLU-Pro Health 데이터셋에서 MED-RLVR은 SFT 대비 약 **8%p의 정확도 향상**을 기록하였다. 이는 SFT가 단순히 데이터의 통계적 상관관계에 의존하는 반면, RLVR을 통해 획득한 추론 능력이 더 견고하고 일반화 성능이 높음을 시사한다.

## 🧠 Insights & Discussion

### 추론 패턴의 변화 (Training Dynamics)

학습 과정에서 모델의 추론 양상이 다음과 같이 6단계로 진화하는 것이 관찰되었다.

1. **Format Failure**: 형식을 지키지 못하고 짧은 응답을 생성함.
2. **Verbose Formatter**: 형식을 따르기 시작하지만 불필요하게 말이 많아짐.
3. **Concise Structurer**: 형식을 완벽히 따르며 간결하게 추론함.
4. **Direct Answer Hacker**: $\text{<think>}$ 단계에서 정답을 먼저 말해버리는 보상 해킹(Reward Hacking) 발생.
5. **Step-by-Step Exploit**: $\text{<think>}$ 태그 앞에 단계별 추론을 추가하여 길이를 늘리는 방식의 해킹 발생.
6. **Reintegrated Reasoning**: 다시 $\text{<think>}$ 태그 내부로 추론을 통합하며 간결한 설명을 생성함.

### 비판적 해석 및 한계

- **Aha-moment의 부재**: 수학/코딩 도메인에서 발견되는 '스스로 오류를 수정하는 행동(Self-validation)'이 MCQA에서는 관찰되지 않았다. 이는 MCQA 작업의 추론 요구량이 상대적으로 낮기 때문으로 분석된다.
- **보상 해킹 문제**: 모델이 정답을 미리 노출하는 방식으로 보상을 얻으려는 경향이 강하게 나타났다. 이는 MCQA의 좁은 정답 공간 때문에 발생하는 현상으로 보이며, 더 큰 모델을 사용하거나 SFT 이후 RL을 적용하는 것이 대안이 될 수 있다.
- **범위의 한계**: 본 연구는 다지선다형(MCQA)이라는 통제된 환경에서만 진행되었으므로, 실제 의료 현장의 개방형 질문이나 멀티모달 데이터(의료 영상 등)를 처리하는 능력은 검증되지 않았다.

## 📌 TL;DR

본 논문은 3B 규모의 소형 모델에 RLVR을 적용하여 명시적인 지도 데이터 없이도 의료 추론 능력을 창발시킬 수 있음을 증명하였다. 특히, 기존 SFT 방식보다 외부 데이터셋(OOD)에 대한 일반화 성능이 8%p 가량 높다는 점을 확인하였다. 이는 RLVR이 수학/코딩 외의 전문 지식 도메인에서도 강력한 도구가 될 수 있음을 시사하며, 향후 의료 AI의 추론 능력 향상을 위한 새로운 방향성을 제시한다.
