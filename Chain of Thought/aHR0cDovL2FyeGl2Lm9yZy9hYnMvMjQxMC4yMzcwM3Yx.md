# OCEAN: Offline Chain-of-Thought Evaluation and Alignment in Large Language Models via Knowledge Graph Exploration

Junda Wu, Xintong Li, Ruoyu Wang, Yu Xia, Yuxin Xiong, Jianing Wang, Tong Yu, Xiang Chen, Branislav Kveton, Lina Yao, Jingbo Shang, Julian McAuley (2024)

## 🧩 Problem to Solve

본 논문은 거대 언어 모델(LLM)이 생성하는 생각의 사슬(Chain-of-Thought, CoT) 추론 과정에서 발생하는 불충실성(unfaithfulness)과 사실적 오류 문제를 해결하고자 한다. LLM은 복잡한 문제 해결을 위해 중간 추론 단계를 생성하지만, 이 과정에서 파라미터 내부 지식의 한계나 환각 현상으로 인해 사실적으로 틀린 근거를 제시하는 경우가 빈번하다.

이러한 문제를 해결하기 위해 일반적으로 인간의 피드백을 통한 강화학습(RLHF)이 사용되지만, CoT와 같은 다단계 추론 과정에 대해 포괄적이고 정확한 인간 피드백을 수집하는 것은 비용이 매우 높고 확장성이 떨어진다는 한계가 있다. 따라서 본 연구의 목표는 지식 그래프(Knowledge Graph, KG)를 활용하여 인간의 개입 없이도 LLM의 CoT 추론 경로를 오프라인에서 평가하고, 이를 통해 모델의 추론 능력을 사실적으로 정렬(Alignment)하는 프레임워크인 OCEAN을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 구조화된 지식 그래프의 추론 경로를 자연어로 구체화(Verbalization)하여 LLM의 텍스트 기반 추론 방식과 지식 그래프의 구조적 추론 방식 사이의 이질성을 극복하는 것이다.

구체적으로, 지식 그래프의 경로를 텍스트로 변환한 뒤 이를 학습한 '지식 그래프 선호 정책(KG Preference Policy)'을 구축하고, 이를 기반으로 오프라인 정책 평가(Offline Policy Evaluation) 기법인 Inverse Propensity Scores(IPS)를 확장한 KG-IPS 추정량을 제안하였다. 이를 통해 모델을 온라인 환경에서 직접 상호작용시키지 않고도, 수집된 오프라인 데이터를 통해 추론 경로의 정밀도를 평가하고 직접적으로 최적화할 수 있는 메커니즘을 설계하였다.

## 📎 Related Works

기존의 오프라인 정책 평가(OPE) 연구들은 주로 추천 시스템이나 헬스케어와 같이 온라인 실험 비용이 크거나 위험한 분야에 집중되어 왔으며, LLM의 다단계 추론 정렬에 적용된 사례는 부족했다. 또한, RLHF나 DPO와 같은 정렬 기법들은 인간의 선호도에 의존하므로 데이터 수집 비용이 높고, 지식의 포괄성이 부족할 수 있다는 한계가 있다.

CoT 추론을 개선하기 위해 외부 지식을 프롬프트에 추가하거나(Knowledge Augmentation), 사후에 수정하는(Self-correction) 방법들이 제안되었으나, 이는 주로 프롬프트 엔지니어링 수준에 머물러 있어 모델 내부의 파라미터 지식과 추론 과정 사이의 근본적인 불일치를 해결하지 못한다. 반면, OCEAN은 지식 그래프를 활용하여 모델의 내부 추론 경로 자체를 사실적 근거에 맞게 직접 정렬한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. 문제 정의: CoT의 MDP 모델링
본 논문은 LLM의 CoT 생성 과정을 마르코프 결정 과정(Markov Decision Process, MDP)으로 정의한다.
- **상태($s_t$):** 입력 프롬프트 $q$와 이전에 생성된 추론 경로 $(c_i)_{i=0}^{t-1}$의 집합이다.
- **행동($a_t$):** 어휘 집합 $V$에서 샘플링된 토큰 시퀀스로, 추론 단계 $c_t$의 일부가 된다.
- **전이:** 현재 상태에 생성된 추론 경로를 이어 붙여 다음 상태 $s_{t+1} = [s_t, c_t]$가 된다.

### 2. 구체화된 지식 그래프 추론 (Verbalized KG Reasoning)
구조화된 KG(예: Wikidata5M)의 엔티티-관계 쌍으로 구성된 경로 $h = (e_0, r_1, e_1, \dots, r_T, e_T)$를 GPT-4를 이용해 자연어 형태의 CoT $c = f(h)$로 변환한다. 이를 통해 KG의 정교한 지식 구조를 LLM이 이해할 수 있는 텍스트 형태로 변환하여 지식 그래프 선호 정책 $\mu_\phi$를 학습시킨다. $\mu_\phi$는 작은 언어 모델(GPT-2 Medium)을 백본으로 하여 학습된다.

### 3. KG-IPS 평가 추정량
대상 정책 $\pi_\theta$의 성능을 오프라인에서 평가하기 위해 다음과 같은 KG-IPS 추정량 $\hat{V}_{KG-IPS}(\theta)$를 제안한다.

$$\hat{V}_{KG-IPS}(\theta) = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{T_i |c_t^{(i)}|} \sum_{t=1}^{T_i} \sum_{e \in c_t} \frac{\pi_\theta(e|s_t^{(i)})}{\lambda(e|s_t^{(i)})} \log \pi_0(e|s_t^{(i)}) \quad (3)$$

여기서 $\lambda(e|s_t^{(i)})$는 토큰 $e$가 엔티티인 경우 KG 선호 정책 $\mu_\phi$를 따르고, 엔티티가 아닌 경우 베이스 모델 정책 $\pi_0$를 따르도록 설계되었다.
$$\lambda(e|s_t^{(i)}) = \mathbb{1}\{e \in a_t^{(i)}\} \cdot \mu_\phi(e|s_t^{(i)}) + \mathbb{1}\{e \in c_t^{(i)} \setminus a_t^{(i)}\} \cdot \pi_0(e|s_t^{(i)})$$

이 설계는 엔티티 토큰에 대해서는 KG의 사실적 정밀도를 강제하고, 일반 토큰에 대해서는 베이스 모델의 언어 생성 능력을 유지함으로써 모델의 퇴화(degeneration)를 방지하고 분산을 줄이기 위함이다.

### 4. 정책 최적화 및 이론적 분석
추정된 가치 함수 $\hat{V}_{KG-IPS}(\theta)$를 최대화하도록 정책 그래디언트(Policy Gradient)를 통해 모델을 업데이트한다.
$$\theta \leftarrow \theta + \nabla_\theta \hat{V}_{KG-IPS}(\theta) \quad (4)$$

본 논문은 이론적으로 KG-IPS 추정량이 편향되지 않은(unbiased) 추정치임을 증명하였으며, 분산의 하한선을 $\frac{M^2}{4n}$으로 정의하고 sub-Gaussian 집중 부등식을 통해 신뢰 구간을 설정하였다.

## 📊 Results

### 1. 실험 설정
- **백본 모델:** Llama-3 (8B), Gemma-2 (2B), Phi-3.5-mini (3.8B), Mistral-0.2 (7B)
- **데이터셋:**
    - 다단계 추론(Multi-hop QA): HotpotQA, MuSiQue, StrategyQA
    - 지식 집약적 추론(Knowledge-intensive QA): ARC, PubMedQA, SciQA
    - 상식 추론(Commonsense Reasoning): CSQA, OpenBookQA, WinoGrande
- **기준선(Baseline):** Base LLM, 지도 학습 기반 미세 조정(SFT)

### 2. 주요 결과
- **다단계 추론:** OCEAN은 대부분의 모델에서 베이스 모델보다 높은 성능을 보였으며, 특히 HotpotQA와 StrategyQA(Context 없음)에서 최적의 성능을 달성하였다.
- **지식 집약적 추론:** SFT의 경우 특정 도메인 지식과의 충돌로 인해 성능이 급격히 하락하는 '지식 불일치' 현상이 관찰되었으나, OCEAN은 이러한 위험 없이 전반적인 정확도를 향상시켰다.
- **상식 추론:** SFT는 상식 지식을 망각하는 '치명적 망각(Catastrophic Forgetting)' 현상이 심각하게 나타난 반면, OCEAN은 베이스 모델의 성능을 유지하거나 오히려 향상시키는 견고함을 보였다.
- **생성 품질:** Self-BLEU, Distinct-2, AlignScore 지표를 통해 OCEAN이 정렬 후에도 텍스트 생성의 다양성과 충실도를 유지함을 확인하였다.

### 3. 사례 분석 (Case Study)
사례 분석 결과, 베이스 모델은 "기타를 칠 때 주로 하는 행동"에 대해 "노래하기는 주된 행동이 아니다"라고 잘못 판단하여 오답을 냈으나, OCEAN이 적용된 모델은 노래하기와 음악 만들기의 관계를 정확히 파악하여 정답을 도출하였다. 또한, 불필요하게 긴 추론 과정을 생략하고 더 간결하고 정확한 CoT를 생성하는 경향이 확인되었다.

## 🧠 Insights & Discussion

본 연구의 가장 큰 강점은 구조화된 지식 그래프와 비구조화된 LLM 텍스트 사이의 간극을 '구체화(Verbalization)'와 '선호 정책'이라는 개념으로 해결하여, 오프라인 환경에서도 효과적인 CoT 정렬을 가능하게 했다는 점이다. 특히, 모든 토큰이 아닌 '엔티티' 토큰에 대해서만 KG 가중치를 적용한 전략은 모델의 일반적인 언어 생성 능력을 훼손하지 않으면서 사실적 정확도만 선택적으로 높일 수 있음을 보여주었다.

다만, 본 방법론은 지식 그래프(Wikidata5M)의 품질과 범위에 의존하며, KG에서 텍스트로 변환하는 과정에서 GPT-4와 같은 고성능 모델의 보조가 필요하다는 가정이 전제되어 있다. 또한, KG에 존재하지 않는 최신 지식이나 매우 희귀한 지식에 대해서는 여전히 한계가 있을 수 있다.

비판적으로 해석하자면, SFT보다 뛰어난 일반화 성능을 보인 이유는 직접적인 정답 매핑이 아니라 MDP 기반의 가치 함수 최적화를 통해 추론의 '경로'를 학습했기 때문으로 판단된다. 이는 단순한 지식 주입보다 추론 프로세스의 정렬이 모델의 견고성에 더 중요하다는 점을 시사한다.

## 📌 TL;DR

본 논문은 LLM의 CoT 추론 경로를 지식 그래프(KG)를 이용해 오프라인에서 평가하고 정렬하는 **OCEAN** 프레임워크를 제안한다. KG의 경로를 자연어로 변환해 선호 정책을 만들고, 이를 **KG-IPS 추정량**에 적용하여 모델을 최적화함으로써 사실적 오류를 줄이고 추론의 충실도를 높였다. 실험 결과, OCEAN은 기존 SFT 방식과 달리 치명적 망각 없이 다단계 및 지식 집약적 추론 성능을 유의미하게 향상시켰으며, 이는 향후 외부 지식을 활용한 LLM의 신뢰성 있는 추론 최적화 연구에 중요한 기여를 할 것으로 보인다.