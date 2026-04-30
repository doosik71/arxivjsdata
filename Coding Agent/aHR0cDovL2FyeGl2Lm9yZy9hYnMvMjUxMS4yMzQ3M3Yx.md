# ThetaEvolve: Test-time Learning on Open Problems

Yiping Wang, Shao-Rong Su, Zhiyuan Zeng, Eva Xu, Liliang Ren, Xinyu Yang, Zeyi Huang, Xuehai He, Luyao Ma, Baolin Peng, Hao Cheng, Pengcheng He, Weizhu Chen, Shuohang Wang, Simon Shaolei Du, Yelong Shen (2025)

## 🧩 Problem to Solve

본 논문은 수학적 발견, 특히 개방형 최적화 문제(Open Optimization Problems)의 경계값(Bounds)을 개선하기 위한 프로그램 진화(Program Evolution) 시스템의 한계를 해결하고자 한다. 기존의 AlphaEvolve와 같은 시스템은 전면적인 성능 향상을 이루었으나, 다음과 같은 세 가지 주요 문제점을 가지고 있다.

첫째, 시스템이 폐쇄 소스(Closed-source)로 운영되어 체계적인 연구와 재현이 어렵다. 둘째, 성능 달성을 위해 거대 모델들의 앙상블(Ensemble of frontier LLMs)에 의존하므로, 소규모 오픈 소스 모델로는 이러한 도전적인 과제를 해결할 수 없다는 인식이 지배적이다. 셋째, 기존 시스템들은 순수하게 추론 단계의 파이프라인(Inference-only pipeline)으로 작동하여, 모델이 진화 과정에서 획득한 전략이나 탐색 능력을 자체적으로 내재화(Internalize)하지 못하고 매번 새로운 추론 과정에 의존해야 한다는 한계가 있다.

따라서 본 연구의 목표는 오픈 소스 프레임워크인 ThetaEvolve를 통해 테스트 시간(Test-time)에 In-context Learning과 강화학습(Reinforcement Learning, RL)을 효율적으로 확장하여, 모델이 스스로 진화 전략을 학습하고 소형 모델만으로도 수학적 경계값을 갱신하도록 하는 것이다.

## ✨ Key Contributions

ThetaEvolve의 핵심 아이디어는 프로그램 진화 파이프라인을 하나의 '적응형 검증 가능 환경(Adaptive Verifiable Environment)'으로 간주하고, 이를 강화학습 루프와 결합하는 것이다. 주요 기여 사항은 다음과 같다.

1. **효율적인 테스트 시간 연산 확장:** 단일 LLM 사용, 대규모 프로그램 데이터베이스 구축, 배치 샘플링(Batch Sampling) 도입을 통해 추론 처리량을 극대화하고 탐색 효율을 높였다.
2. **소형 모델의 한계 돌파:** DeepSeek-R1-0528-Qwen3-8B와 같은 8B 규모의 소형 오픈 소스 모델만으로도 Circle Packing 및 First Auto-correlation Inequality와 같은 난제에서 기존 SOTA 경계값을 경신하는 성과를 거두었다.
3. **진화 능력의 내재화:** 테스트 시간 RL을 통해 모델이 프로그램 개선 전략을 학습하게 함으로써, RL을 거친 체크포인트가 순수 추론 모델보다 더 빠르게 정답에 도달하며, 이러한 능력이 학습하지 않은 다른 과제(Unseen tasks)로도 전이(Transfer)됨을 입증하였다.

## 📎 Related Works

본 논문은 다음과 같은 관련 연구들을 배경으로 한다.

- **프롬프트 최적화 및 에이전틱 LLM:** DSPy와 같은 프롬프트 최적화 연구나 피드백을 통해 문맥 정보를 업데이트하는 에이전트 연구들이 존재하지만, 이는 주로 다운스트림 성능 향상에 초점을 맞춘다.
- **프로그램 탐색 기반 발견:** FunSearch와 AlphaEvolve는 특정 수학적 목적 함수를 최적화하기 위한 프로그램 탐색에 집중한다. 그러나 이들은 대부분 추론 시간의 파이프라인으로, 모델 자체의 가중치가 업데이트되지 않는다.
- **Test-Time RL:** AlphaProof는 Lean과 같은 정형 검증기를 통해 테스트 시간에 RL을 수행하여 성능을 높였다. ThetaEvolve는 이러한 아이디어를 수치적 최적화 문제(Continuous-reward objectives)로 확장하여 적용하였다.

기존 접근 방식과의 차별점은 단순한 추론 확장(Inference-time scaling)을 넘어, 동적인 프로그램 데이터베이스 환경에서 RL을 수행함으로써 모델이 '탐색하는 방법' 자체를 배우게 한다는 점에 있다.

## 🛠️ Methodology

### 1. 전체 시스템 구조
ThetaEvolve는 프로그램 데이터베이스에서 부모 프로그램을 샘플링하고, LLM이 이를 수정하여 자식 프로그램을 생성하며, 검증기(Verifier)를 통해 점수를 측정하고 다시 데이터베이스에 저장하는 루프를 가진다.

### 2. 주요 구성 요소 및 최적화
- **Single LLM & Large Database:** 앙상블 대신 단일 모델을 사용하며, 데이터베이스 크기를 10,000개로 대폭 확장하여 테스트 시간 연산이 증가함에 따라 더 다양한 후보군을 유지하도록 했다.
- **Batch Sampling:** 매 단계에서 $B$개의 부모 프로그램을 샘플링하고, 각 프롬프트당 $n$개의 응답을 생성하여 총 $B \times n$개의 자식 프로그램을 한 번에 생성한다. 이는 vLLM이나 SGLang 같은 배치 추론 엔진을 활용하여 처리량을 극대화한다.
- **Early Check & Lazy Penalty:** 불필요한 평가 비용을 줄이고 모델의 정체(Stagnation)를 막기 위해 다음과 같은 페널티 함수 $s(pp, r)$를 정의한다.
$$s(pp,r) = \begin{cases} -0.4, & \text{if no diff blocks found} \\ -0.3, & \text{elif no valid changes (} cp \equiv pp \text{)} \\ -0.2, & \text{elif no solution} \\ -0.1, & \text{elif invalid solution} \\ E(cp), & \text{otherwise} \end{cases}$$
특히, 데이터베이스에 이미 존재하는 프로그램과 동일한 출력을 내놓는 경우 'Lazy Penalty'를 부여하여 모델이 단순 반복이 아닌 실질적인 개선을 시도하도록 유도한다.

### 3. RL Reward Shaping (선택 사항)
목적 함수의 값 범위가 너무 좁아 학습 신호가 약한 경우, 보상 함수 $R(s)$를 다음과 같이 정규화한다.
$$R(s) = \begin{cases} s, & \text{if } s < 0 \text{ or disabled} \\ k \cdot F(s), & \text{otherwise} \end{cases}$$
여기서 $F(s)$는 선형 매핑 $H(s)$를 기반으로 하며, $\alpha$ 계수를 통해 상위 점수에 더 공격적인 보상을 주는 구조이다.
$$F(s) = \frac{\text{clip}(H(s), 0, 1)}{\alpha}$$
$$H(s) = \begin{cases} (s - L)/(U - L), & \text{if maximizing} \\ (U - s)/(U - L), & \text{otherwise} \end{cases}$$
($L, U$는 목적 함수의 하한 및 상한값)

### 4. 학습 절차
RL 알고리즘으로는 GRPO(Group Relative Policy Optimization)를 사용하며, 비대칭 클리핑(Asymmetric clipping)과 절단된 중요도 샘플링(Truncated importance sampling)을 적용하여 학습 안정성을 높였다.

## 📊 Results

### 1. 실험 설정
- **모델:** ProRL-1.5B-v2, DeepSeek-R1-0528-Qwen3-8B (Distill-Qwen3-8B).
- **과제:** Circle Packing(CP), First/Second/Third Auto-correlation Inequality, Hadamard Matrix.
- **지표:** 각 수학적 문제의 목적 함수 값(경계값).

### 2. 주요 결과
- **SOTA 경신:** Distill-Qwen3-8B 모델은 Circle Packing과 First Auto-correlation Inequality에서 AlphaEvolve(Gemini-2.0-Flash/Pro 앙상블 사용)의 결과를 능가하는 새로운 최적값을 발견하였다. 특히 Circle Packing의 경우, ShinkaEvolve(Claude-sonnet-4 등 6개 모델 앙상블)가 75초 걸려 찾은 해를 단 3초 만에 동일하게 찾아냈다.
- **RL의 효과:** 모든 과제에서 RL을 적용한 경우가 순수 추론(w/o RL)보다 일관되게 높은 성능을 보였다.
- **능력의 내재화 및 전이:** Circle Packing 과제로 RL 학습을 시킨 체크포인트를 다른 과제(Hadamard Matrix 등)에 적용했을 때, 베이스 모델보다 훨씬 빠른 속도로 성능이 향상되었다. 이는 모델이 '최적화 프로그램을 진화시키는 일반적인 능력'을 학습했음을 시사한다.
- **동적 환경의 중요성:** 항상 초기 프로그램에서 시작하는 정적 환경(Static Environment)에서의 RL은 ThetaEvolve의 동적 환경 대비 현저히 낮은 성능을 보였다.

## 🧠 Insights & Discussion

본 논문은 단순한 추론 시간의 확장이 아니라, 모델이 스스로 학습하는 '테스트 시간 학습(Test-time Learning)'의 가능성을 보여주었다. 특히 부록 D에서 제시한 수학적 직관에 따르면, 정답 프로그램 $P$에 도달할 확률 $\epsilon_\theta$는 매우 낮지만, 동적 데이터베이스를 통해 중간 단계 $P_i$를 거쳐 갈 때의 확률은 훨씬 높다. 따라서 모델은 동적 환경에서 더 풍부한 학습 신호를 얻을 수 있다.

**강점:** 소형 모델만으로도 거대 모델 앙상블의 성과를 뛰어넘을 수 있음을 입증했으며, RL을 통해 진화 능력을 내재화하고 이를 타 과제로 전이시킬 수 있음을 보였다.

**한계 및 논의:** 보상 형성(Reward Shaping) 파라미터($U, L, \alpha$)가 모델의 성능에 따라 다르게 작용한다는 점이 발견되었다. 이는 새로운 문제에 적용할 때 세심한 튜닝이 필요함을 의미한다. 또한, 현재는 단일 과제 중심의 RL이지만, 여러 과제를 동시에 학습시키는 멀티 타겟 학습으로 확장될 가능성이 크다.

## 📌 TL;DR

ThetaEvolve는 수학적 난제 해결을 위한 프로그램 진화 파이프라인에 테스트 시간 RL을 결합한 오픈 소스 프레임워크이다. 이를 통해 8B 규모의 소형 모델이 거대 모델 앙상블보다 더 나은 수학적 경계값을 발견하게 했으며, 모델이 스스로 프로그램을 개선하는 전략을 내재화하여 다른 과제로 전이시킬 수 있음을 증명하였다. 이 연구는 LLM이 단순한 추론 도구를 넘어, 동적 환경에서의 학습을 통해 과학적 발견 능력을 스스로 키울 수 있음을 보여준 중요한 사례이다.