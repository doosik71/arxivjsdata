# ON A CONNECTION BETWEEN IMITATION LEARNING AND RLHF

Teng Xiao, Yige Yuan, Mingxiao Li, Zhengyu Chen, Vasant G Honavar (2025)

## 🧩 Problem to Solve

본 논문은 거대 언어 모델(LLM)을 인간의 선호도 데이터에 정렬(Alignment)시키는 문제에 집중한다. 기존의 표준적인 방법론인 Reinforcement Learning from Human Feedback (RLHF)는 보상 모델(Reward Model) 학습과 강화학습(RL) 최적화라는 두 단계의 과정을 거치는데, 이는 계산 비용이 높고 학습 과정이 불안정하다는 치명적인 단점이 있다.

이를 해결하기 위해 Direct Preference Optimization (DPO)와 같은 단일 단계(one-step) 방식들이 제안되었으나, 이들은 근본적으로 Bradley-Terry (BT) 선호도 모델이라는 파라미터 모델에 의존한다. 이로 인해 모델이 과적합(Overfitting)되기 쉽고, 결과적으로 선호도 데이터에 대한 최적의 정렬을 달성하지 못하는 한계가 존재한다. 따라서 본 논문의 목표는 RLHF와 선호도 최적화를 새로운 관점에서 해석하고, 기존의 한계를 극복하는 보다 원칙적인 정렬 프레임워크를 설계하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **RLHF를 Imitation Learning (IL, 모방 학습)의 관점에서 재해석**하는 것이다. 연구진은 RLHF가 본질적으로 선택된 응답(Chosen response)의 분포를 모방하는 문제와 이론적으로 밀접하게 연결되어 있음을 증명하였다.

이러한 통찰을 바탕으로, 본 논문은 **DIL (Direct Imitation Learning)**이라는 프레임워크를 제안한다. DIL은 복잡한 보상 모델링이나 강화학습 루프 없이, 밀도 비율 추정(Density Ratio Estimation)을 통해 모방 학습 목적 함수를 직접 최적화한다. 이는 기존의 다양한 정렬 알고리즘들을 특수 사례로 포괄하는 통합적인 관점을 제공하며, 특히 BT 모델의 가정 없이도 효과적인 정렬이 가능함을 보여준다.

## 📎 Related Works

### 1. RLHF 및 Offline Preference Optimization

기존의 RLHF는 PPO와 같은 알고리즘을 사용하여 보상을 최대화하며, DPO는 이를 단순화하여 명시적인 보상 모델 없이 정책을 직접 학습한다. IPO, KTO, SimPO 등 다양한 변형들이 제안되었으나, 대부분 보상 최대화 목적 함수에 기반하고 있어 과적합 및 추론 능력 저하 문제가 보고되었다.

### 2. Imitation Learning (IL)

전통적인 IL은 전문가의 행동을 모방하는 것으로, GAIL과 같은 온라인 방식과 IQ-Learn 같은 오프라인 방식이 있다. 최근 LLM 정렬에 IL을 적용하려는 시도가 있었으나, 본 논문은 RLHF와 IL 사이의 명확한 이론적 연결 고리를 밝힘으로써 차별성을 가진다.

## 🛠️ Methodology

### 1. RLHF와 모방 학습의 이론적 연결

논문은 RLHF의 각 단계가 다음과 같은 모방 학습 과정과 동일함을 이론적으로 증명한다.

* **보상 학습 단계:** Energy-Based Models (EBMs)를 사용한 정책 $\pi_\phi$가 전문가 정책 $\pi_{\text{chosen}}$을 모방하도록 Forward KL Divergence를 최소화하는 과정은, 결과적으로 BT 모델 기반의 보상 함수 학습과 동일하다.
* **RL 최적화 단계:** 학습된 EBM 정책 $\pi_\phi$를 분석적인 정책 $\pi_\theta$로 증류(Distillation)하는 과정은 Reverse KL Divergence를 최소화하는 역 지식 증류(Reverse Knowledge Distillation) 과정으로 해석될 수 있다.

결론적으로, KL-정규화된 RLHF는 다음과 같은 최적화 문제로 정의된다.
$$\min_{\pi_\theta} \text{KL}(\pi_\theta \| \pi^*_\phi) \quad \text{s.t.} \quad \pi^*_\phi = \arg \min_{\pi_\phi} \text{KL}(\pi_{\text{chosen}} \| \pi_\phi)$$

### 2. Direct Imitation Learning (DIL) 프레임워크

DIL은 $\pi_{\text{chosen}}$의 분포를 직접 모방하기 위해 Reverse KL Divergence를 최소화하는 것을 목표로 한다.
$$\min_\theta \text{KL}(\pi_\theta(y|x) \| \pi_{\text{chosen}}(y|x))$$

하지만 $\pi_{\text{chosen}}$은 알 수 없는 분포이므로, 논문은 **밀도 비율(Density Ratio)** $r(x,y) = \frac{\pi_{\text{chosen}}(y|x)}{\pi_{\text{ref}}(y|x)}$를 도입하여 목적 함수를 재구성한다.

### 3. 밀도 비율 추정과 Bregman Divergence

밀도 비율을 효율적으로 추정하기 위해 Bregman Divergence를 사용한다. 특히 Least-Squared Importance Fitting (LSIF)를 적용하면 다음과 같은 손실 함수가 유도된다.
$$\mathcal{L}(\phi; D) = \mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \frac{1}{2} r_\phi(x, y_l)^2 - r_\phi(x, y_w) \right]$$
여기서 $y_w$는 선택된 응답(chosen), $y_l$은 거부된 응답(rejected)이다.

### 4. 최종 DIL 목적 함수

RL 루프를 제거하고 정책 파라미터 $\theta$를 직접 최적화하기 위해, 밀도 비율의 폐형 솔루션(closed-form solution)을 정책에 대입한다. LSIF 기반의 DIL 최종 손실 함수는 다음과 같다.
$$\mathcal{L}_{\text{DIL}}(\theta; D) = \mathbb{E}_{(x, y_w, y_l) \sim D} \left[ -\frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} + \frac{1}{2} \left( \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right)^2 \right]$$

또한, DPO는 이 DIL 프레임워크에서 밀도 비율 추정 방법으로 Contrastive Predictive Coding (CPC)를 사용한 특수 사례임을 수학적으로 증명하였다.

## 📊 Results

### 1. 실험 설정

* **데이터셋:** UltraFeedback Binarized, Reddit TL;DR (요약), Anthropic-HH (대화).
* **모델:** Mistral-7B, Llama3-8B, Pythia-2.8B.
* **비교 대상:** DPO, f-DPO, IPO, SLiC, CPO, SimPO.
* **평가 지표:** Open LLM Leaderboard (MMLU-PRO, BBH, MATH, GSM8K 등) 및 GPT-4 기반의 Win-rate 측정.

### 2. 정량적 결과

* **벤치마크 성능:** Table 2에 따르면, DIL(LSIF)은 거의 모든 벤치마크에서 DPO와 SimPO를 능가한다. 특히 Llama3 모델의 수학(MATH, GSM8K) 벤치마크에서 SimPO 대비 7.5% 이상의 상대적 성능 향상을 보였다.
* **인간 선호도 정렬:** Table 3에서 DIL은 요약 및 대화 생성 작업에서 SFT 및 Chosen 응답 대비 가장 높은 Win-rate를 기록하며, 유용성(Helpfulness)과 무해성(Harmlessness) 모두에서 우위를 점했다.

### 3. 정성적 분석 및 학습 동역학

Figure 2의 학습 곡선 분석 결과, DPO와 SimPO는 학습이 진행됨에 따라 선택된 응답(Chosen)의 가능성(Likelihood)이 급격히 감소하는 경향을 보였다. 반면, **DIL은 선택된 응답의 가능성을 보존하면서 거부된 응답과의 마진을 넓히는 특성**을 보였다. 이는 DIL이 추론 능력을 훼손하지 않으면서 정렬을 달성함을 시사한다.

## 🧠 Insights & Discussion

### 강점 및 기여

본 논문은 RLHF를 단순한 보상 최대화가 아닌 모방 학습의 관점에서 재정의함으로써, 정렬 알고리즘 설계에 새로운 이론적 토대를 제공하였다. 특히 DPO와 같은 기존 방법론들이 왜 추론 능력을 저하시키는지(선택된 응답의 likelihood 감소)를 분석하고, 이를 해결하기 위한 DIL의 구조적 이점을 입증한 점이 매우 뛰어나다.

### 한계 및 논의사항

* **오프라인 설정의 한계:** 본 연구는 오프라인 데이터셋만을 사용하였으며, 모델이 생성한 데이터를 다시 학습에 사용하는 On-policy 학습 환경에서의 효과는 검증되지 않았다.
* **밀도 비율 추정 함수:** LSIF 외에도 BCE, UKL 등 다양한 Bregman Divergence 함수를 사용할 수 있음을 보여주었으나, 각 작업별로 어떤 함수가 최적인지에 대한 명확한 가이드라인은 부족하다.

### 비판적 해석

DIL은 하이퍼파라미터에 대한 의존도를 낮추면서도 성능을 높였다는 점에서 실용적 가치가 크다. 다만, $\pi_{\text{ref}}$에 대한 의존성이 여전히 존재하므로, Reference-free 방식(예: SimPO)이 가진 장점을 어떻게 통합할 수 있을지가 향후 중요한 연구 과제가 될 것으로 보인다.

## 📌 TL;DR

본 논문은 RLHF가 본질적으로 선택된 응답 분포를 모방하는 모방 학습(Imitation Learning) 과정임을 이론적으로 증명하고, 이를 직접 최적화하는 **DIL (Direct Imitation Learning)** 프레임워크를 제안한다. DIL은 밀도 비율 추정(Density Ratio Estimation)을 통해 BT 모델의 제약을 벗어나며, 기존 DPO/SimPO가 겪던 '선택된 응답의 가능성 감소' 문제를 해결하여 **정렬 성능 향상과 추론 능력 보존**이라는 두 마리 토끼를 잡았다. 이는 향후 LLM 정렬 연구가 단순한 보상 최대화를 넘어 분포 모방의 관점으로 확장될 가능성을 보여준다.
