# RLVR-World: Training World Models with Reinforcement Learning

Jialong Wu, Shaofeng Yin, Ningya Feng, Mingsheng Long (2025)

## 🧩 Problem to Solve

본 논문은 상태 전이(state transitions)를 예측하는 월드 모델(World Models)의 학습 목표와 실제 평가 지표 사이의 불일치 문제를 해결하고자 한다. 일반적으로 월드 모델은 최대 가능도 추정(Maximum Likelihood Estimation, MLE)과 같은 대리 목표(surrogate objectives)를 통해 학습된다. 하지만 MLE는 언어 모델에서의 환각(hallucination)이나 반복 생성 문제, 그리고 비디오 모델에서의 흐릿한 예측(blurry predictions)과 같은 부작용을 야기하며, 이는 모델의 최종 목표인 예측 정확도나 지각적 품질(perceptual quality)과 직접적으로 일치하지 않는다.

특히, 최근의 월드 모델들은 이산적 토크나이저 기반의 자기회귀 모델(autoregressive models)이나 확산 모델(diffusion models)과 같은 비종단간(non-end-to-end) 구조를 채택하는 경우가 많아, 미분 불가능한 최종 평가 지표를 직접적으로 최적화하는 것이 불가능했다. 따라서 본 연구의 목표는 검증 가능한 보상(verifiable rewards)을 이용한 강화학습(RLVR)을 통해, 월드 모델이 타겟 지표를 직접 최적화하도록 하는 통합 프레임워크인 RLVR-World를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 월드 모델의 학습 과정을 '사전 학습(Pre-training) $\rightarrow$ 지도 미세 조정(SFT) $\rightarrow$ 강화학습 기반 사후 학습(RLVR Post-training)'의 단계로 구성하고, 마지막 단계에서 모델이 생성한 결과물을 디코딩하여 실제 지표(정확도, LPIPS 등)를 계산하고 이를 보상으로 사용하는 것이다.

중심적인 설계 직관은 언어 모델의 추론 능력을 향상시킨 RLVR 패러다임을 월드 모델링에 적용하는 것이다. 이를 위해 다양한 모달리티의 상태 전이 문제를 공통의 자기회귀 시퀀스 생성 문제로 통합하고, 각 모달리티에 특화된 검증 가능한 보상 함수를 설계하여 모델이 데이터 분포를 단순히 모사하는 것을 넘어 실제 예측 성능을 극대화하도록 유도한다.

## 📎 Related Works

기존의 월드 모델 연구들은 주로 VAE, Diffusion, 또는 Transformer 기반의 자기회귀 모델을 사용하여 환경의 역학(dynamics)을 학습해 왔다. 최근에는 비디오 토큰화를 통해 대규모 Transformer를 적용하는 추세이나, 여전히 MLE 기반 학습에 의존하여 누적 오차(accumulation errors)나 지각적 품질 저하 문제를 겪고 있다.

생성 모델을 위한 강화학습 분야에서는 인간의 선호도를 학습하는 RLHF가 널리 쓰였으나, 이는 보상 모델의 과최적화(overoptimization) 위험이 있다. 이에 반해 DeepSeek-R1 등에서 사용된 RLVR은 규칙 기반의 명확한 보상을 사용하여 수학이나 코드와 같은 추론 작업에서 큰 성과를 거두었다. 본 논문은 이러한 RLVR이 '상태 전이 예측'이라는 명확한 정답이 존재하는 월드 모델링 작업에도 매우 적합하다는 점에 착안하여 기존 접근 방식과 차별화를 꾀했다.

## 🛠️ Methodology

### 전체 파이프라인 및 구조

RLVR-World는 서로 다른 모달리티(언어, 비디오)를 통합된 시퀀스 모델링 프레임워크로 처리한다. 전체 과정은 다음과 같다:

1. **토큰화(Tokenization):** 현재 상태 $s$와 액션 $a$를 입력 시퀀스(질문 $q$)로, 다음 상태 $s'$를 출력 시퀀스(응답 $o$)로 변환한다.
2. **사전 학습(Pre-training):** MLE를 통해 기본 예측 능력을 학습시킨다.
3. **RLVR 사후 학습:** GRPO(Group Relative Policy Optimization) 알고리즘을 사용하여 타겟 지표를 직접 최적화한다.

### 상세 학습 절차 및 알고리즘

본 연구는 가치 함수(value function)가 필요 없는 GRPO를 사용하여 학습 효율을 높였다.

**1. GRPO를 통한 최적화:**
질문 $q$에 대해 모델 $\pi_{\theta_{old}}$로부터 그룹 응답 $\{o_i\}_{i=1}^G$를 샘플링하고, 각 응답의 보상 $R_i$를 그룹 내에서 정규화하여 어드밴티지 $\hat{A}_{i,t}$를 계산한다.
$$\hat{A}_{i,t} = \frac{R_i - \text{mean}(\{R_i\}_{i=1}^G)}{\text{std}(\{R_i\}_{i=1}^G)}$$

최종 목적 함수 $J_{GRPO}(\theta)$는 다음과 같이 클리핑된 목적 함수와 KL 발산 패널티의 합으로 정의된다.
$$J_{GRPO}(\theta) = \mathbb{E}_{q \sim \mathcal{D}, \{o_i\} \sim p_{\theta_{old}}} \left[ \frac{1}{G} \sum_{i=1}^G \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \left( \min \left( \frac{p_{i,t}^\theta}{p_{i,t}^{\theta_{old}}} \hat{A}_{i,t}, \text{clip} \left( \frac{p_{i,t}^\theta}{p_{i,t}^{\theta_{old}}}, 1-\epsilon, 1+\epsilon \right) \hat{A}_{i,t} \right) - \beta D_{KL}[p_\theta || p_{ref}] \right) \right]$$

**2. 검증 가능한 보상(Verifiable Rewards) 설계:**
모델이 생성한 토큰 시퀀스를 다시 상태 $\hat{s}'_i$로 디코딩한 후, 정답 상태 $s'$와 비교하여 보상을 계산한다.
$$R_i = \text{sign}(D) \cdot D(\hat{s}'_i, s')$$
여기서 $D$는 모달리티별 지표이다:

- **언어 월드 모델:** 예측 정확도(Accuracy) 또는 F1-score를 사용한다.
- **비디오 월드 모델:** $L^1$ 손실과 LPIPS(지각적 유사도)의 합을 음수화하여 보상으로 사용한다.
  $$R = -\sum_{\tau=t+1}^{t+T} [L^1(\hat{s}_\tau, s_\tau) + \text{LPIPS}(\hat{s}_\tau, s_\tau)]$$

## 📊 Results

### 실험 설정

- **언어 모델:** ByteSized32(텍스트 게임)와 WebArena(웹 내비게이션) 데이터셋을 사용하였다. 모델은 DeepSeek-R1-Distill-Qwen (1.5B, 7B)을 기반으로 한다.
- **비디오 모델:** RT-1 로봇 조작 데이터셋과 PushT, Rope, Granular 데이터셋을 사용하였다. iVideoGPT 기반의 autoregressive Transformer를 사용하였다.

### 주요 결과

**1. 언어 월드 모델:**

- 텍스트 게임 상태 예측에서 1.5B 모델 기준 SFT 대비 정확도가 대폭 향상되었으며, 7B 모델의 경우 GPT-4의 성능을 상회하는 결과를 보였다.
- 웹 페이지 상태 예측에서 F1-score가 SFT 대비 $+15.1\%$ 상승하였으며, 이를 적용한 모델 예측 제어(MPC) 에이전트의 성공률이 $12.06\%$에서 $14.29\%$로 증가하였다.

**2. 비디오 월드 모델:**

- RT-1 데이터셋에서 LPIPS 지표가 상대적으로 $+9.2\%$ 개선되었으며, 이는 수백 번의 RLVR 그래디언트 스텝만으로 달성되었다. (MLE 학습 시에는 수십만 번의 스텝이 필요함)
- 특히 다단계 예측(multi-step prediction)에서 심각한 문제였던 반복 생성 현상(repetition rate)을 $48.6\%$에서 $9.9\%$로 획기적으로 낮추었다.
- PushT, Granular 등 벤치마크에서 SOTA 수준의 성능을 보였으며, 특히 입자 기반의 복잡한 Granular 데이터셋에서 강점을 보였다.

**3. 다운스트림 적용:**

- **Real2Sim 정책 평가:** RLVR로 학습된 비디오 월드 모델을 시뮬레이터로 사용했을 때, 실제 환경의 성공률과 시뮬레이션 상의 성공률 사이의 간극이 기존 SIMPLER 시뮬레이터보다 더 좁아짐을 확인하였다.

## 🧠 Insights & Discussion

본 연구는 MLE와 같은 대리 목표가 생성 모델의 최종 품질과 일치하지 않는다는 점을 지적하고, RLVR을 통해 이를 직접 최적화할 수 있음을 입증하였다. 특히 비디오 생성에서 발생하는 반복 생성 문제는 모델이 픽셀의 단순 유사성에 의존하는 MLE의 특성 때문에 발생하는데, LPIPS와 같은 지각적 지표를 보상으로 사용함으로써 이를 효과적으로 억제할 수 있었다.

**강점 및 한계:**

- **강점:** 모달리티에 관계없이 적용 가능한 통합 프레임워크를 제안하였으며, 매우 적은 학습 스텝으로도 상당한 성능 향상을 이끌어냈다.
- **한계:** RLVR 학습이 수백 스텝 내에 빠르게 수렴하는 경향이 있어, 그 이상의 지속적인 성능 향상을 위해서는 데이터나 알고리즘의 추가적인 분석이 필요하다. 또한, 현재는 특정 데이터셋에 맞춰 튜닝되었으므로, 일반 목적의 파운데이션 월드 모델에 적용했을 때의 OOD(Out-of-Distribution) 일반화 능력에 대한 연구가 더 필요하다.

비판적으로 해석하자면, 본 연구는 사후 학습(post-training)의 효과를 입증했으나, 결국 성능의 상한선은 사전 학습된 베이스 모델의 용량(capacity)에 의존한다. 따라서 진정한 의미의 일반 월드 모델을 구축하기 위해서는 대규모 다중 도메인 데이터에 대한 사전 학습과 RLVR의 결합이 필수적일 것이다.

## 📌 TL;DR

RLVR-World는 MLE 기반 학습의 한계를 극복하기 위해 **검증 가능한 보상(Verifiable Rewards)**을 이용한 강화학습으로 월드 모델을 최적화하는 프레임워크이다. 언어와 비디오 모달리티 모두에서 타겟 지표(정확도, LPIPS 등)를 직접 최적화함으로써 예측 정확도를 높이고 반복 생성과 같은 아티팩트를 제거하였다. 이 연구는 RLVR이 단순한 추론 모델을 넘어, 생성 모델 전반의 유용성을 높이는 일반적인 사후 학습 패러다임이 될 수 있음을 시사한다.
