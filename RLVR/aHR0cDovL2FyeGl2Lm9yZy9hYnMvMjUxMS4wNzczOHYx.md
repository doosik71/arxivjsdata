# From Exploration to Exploitation: A Two-Stage Entropy RLVR Approach for Noise-Tolerant MLLM Training

Donglai Xu et al. (2025)

## 🧩 Problem to Solve

본 논문은 Multimodal Large Language Models(MLLMs)를 위한 Reinforcement Learning with Verifiable Rewards(RLVR) 학습 시 발생하는 **데이터 어노테이션 노이즈(Annotation Noise)** 문제에 집중한다. RLVR은 검증 가능한 보상을 통해 모델의 추론 능력을 향상시키지만, 실제 환경의 데이터셋은 레이블링 오류가 포함된 경우가 많아 학습의 안정성을 저해한다.

기존의 비지도 RLVR 방식이나 단순한 Entropy Minimization 기법은 다음과 같은 한계를 가진다. 첫째, 잘못된 레이블(Incorrect labels)에 과적합(Overfitting)될 위험이 크다. 둘째, Group-Relative Policy Optimization(GRPO)의 핵심인 그룹 내 보상 랭킹 신호(Reward ranking signal)를 제한하여 학습 효율을 떨어뜨린다. 따라서 본 연구의 목표는 노이즈가 섞인 데이터셋에서도 강건하게 작동하며, 모델이 탐색(Exploration)에서 활용(Exploitation) 단계로 효율적으로 전이될 수 있도록 돕는 새로운 엔트로피 최적화 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **토큰 수준의 엔트로피(Token-level Entropy)를 학습 단계에 따라 동적으로 조절**하는 2단계 최적화 전략이다.

1. **탐색 단계(Exploration Phase):** 학습 초기에는 엔트로피를 최대화($\text{Entropy Maximization}$)하여 모델이 다양하고 확률적인 출력을 생성하도록 유도한다. 이는 노이즈 섞인 레이블로의 조기 수렴을 방지하는 정규화(Regularizer) 역할을 하며, GRPO가 더 신뢰할 수 있는 보상 그래디언트를 추정할 수 있도록 그룹 내 변동성을 확보한다.
2. **활용 단계(Exploitation Phase):** 학습 후반부에는 엔트로피를 최소화($\text{Entropy Minimization}$)하여 모델이 더 확신 있고 결정론적인(Deterministic) 출력을 내도록 유도한다. 이를 통해 탐색 단계에서 습득한 지식을 공고히 하고 예측 정확도를 정교화한다.

## 📎 Related Works

논문에서는 RLVR의 한계를 극복하기 위한 기존 접근 방식을 세 가지 카테고리로 분류하여 설명한다.

1. **External-Signal-Based Methods:** 컴파일러나 LLM-as-a-Judge와 같은 외부 신호를 활용하는 방식이다. 하지만 도메인마다 LLM의 능력이 상이하고 컴파일러와 같은 도구는 특정 작업에만 국한된다는 한계가 있다.
2. **Internal-Signal-Based Methods:** 포맷 보상(Format rewards)이나 랜덤 보상과 같이 모델 출력 자체에서 보상을 도출하는 방식이다. 유연하지만, 보상 함수가 실제 작업 목표와 밀접하게 정렬되지 않는 경우가 많아 효과가 제한적이다.
3. **Entropy-Based Methods:** 생성의 불확실성(Uncertainty)을 활용하는 방식으로, 주로 엔트로피 최소화를 통해 신뢰도를 높이는 연구들이 진행되었다. 그러나 이러한 방식은 엔트로피의 동적인 역할을 간과하여, 노이즈가 있는 데이터에서 하위 최적(Sub-optimal)의 결정론적 행동에 갇힐 위험이 있다.

본 논문은 이러한 단일 방향의 엔트로피 조절 대신, 학습 스케줄에 따른 방향 전환(Switching)을 통해 탐색과 활용의 트레이드-오프를 제어함으로써 차별성을 가진다.

## 🛠️ Methodology

### 1. Group Relative Policy Optimization (GRPO)

본 연구는 기본 알고리즘으로 GRPO를 사용한다. GRPO는 각 입력 $x$에 대해 $K$개의 응답 $\{y_1, y_2, \dots, y_K\}$을 샘플링하고, 그룹 내 보상을 정규화하여 어드밴티지(Advantage) $A_i$를 계산한다.

$$A_i = \frac{r(y_i) - \text{mean}(r(y_{1:K}))}{\text{std}(r(y_{1:K}))}$$

최종적으로 클리핑(Clipping) 메커니즘이 포함된 서로게이트 손실 함수 $L_{\text{GRPO}}$를 통해 정책을 업데이트한다.

### 2. Token-Level Entropy

모델의 불확실성을 세밀하게 측정하기 위해 토큰 수준의 엔트로피를 정의한다. 어휘집 $V$에 대한 조건부 확률 분포 $\pi_\theta(v | x, y_{<t})$가 있을 때, $t$번째 토큰의 엔트로피 $H_t$는 다음과 같다.

$$H_t(x, y) = -\sum_{v \in V} \pi_\theta(v | x, y_{<t}) \log \pi_\theta(v | x, y_{<t})$$

전체 시퀀스의 토큰 수준 엔트로피 $H_{\text{token}}$은 모든 토큰 $T$에 대해 평균을 내어 계산하며, 이에 따른 엔트로피 손실 함수는 다음과 같이 정의된다.

$$L_{\text{entropy}} = -\mathbb{E}_{x \sim D} \left[ \frac{1}{K} \sum_{i=1}^K H_{\text{token}}(x, y_i) \right]$$

### 3. Two-Stage Entropy-Guided GRPO

제안하는 방법론은 GRPO 손실에 스케줄링된 엔트로피 항을 추가한 통합 목적 함수를 사용한다.

$$L_{\text{total}} = L_{\text{GRPO}} + \lambda(\tau) L_{\text{entropy}}$$

여기서 계수 $\lambda(\tau)$는 학습 단계 $\tau$에 따라 다음과 같이 변하는 조각 함수(Piecewise function)로 정의된다.

$$\lambda(\tau) = \begin{cases} \lambda_{\max}, & \text{if } \tau \le \tau_{\text{switch}} \quad (\text{Stage 1: Exploration}) \\ -\lambda_{\min}, & \text{otherwise} \quad (\text{Stage 2: Exploitation}) \end{cases}$$

$\tau_{\text{switch}}$는 일반적으로 전체 학습 단계의 약 80% 지점으로 설정하며, 이 시점은 토큰 엔트로피의 이동 평균이 포화(Saturate)되는 시점과 일치한다.

## 📊 Results

### 실험 설정

- **모델:** Qwen2-VL-2B, Qwen2.5-VL-3B, Qwen2-VL-7B.
- **작업 및 데이터셋:** GUI Grounding (ScreenSpot), Fine-grained Classification (Pets37), Open-Vocabulary Object Detection (COCO).
- **노이즈 설정:** 정답 레이블의 비율을 100%부터 0%까지 다양하게 설정하여 노이즈 내성을 측정하였다.
- **비교 대상:** Base Model, Vanilla GRPO, GRPO w. Min (엔트로피 최소화만 적용), GRPO w. Max (엔트로피 최대화만 적용).

### 주요 결과

1. **노이즈 내성:** GUI Grounding 작업에서 50%의 노이즈가 포함된 데이터로 학습했을 때, 제안 방법(GRPO w. Two)은 80.2%의 정확도를 달성하였다. 이는 클린 데이터 성능과 단 2% 차이밖에 나지 않는 수치이며, Vanilla GRPO보다 월등히 높다.
2. **모델 스케일링:** 모델 크기가 커질수록(2B $\rightarrow$ 3B $\rightarrow$ 7B) 2단계 엔트로피 스케줄링의 이점이 더 크게 나타났다. 특히 Qwen2-VL-7B 모델은 50% 노이즈 설정에서 baseline 대비 8.6%의 성능 향상을 보였다.
3. **OOD 일반화:** ScreenSpot-Pro 벤치마크를 통한 Out-of-Distribution(OOD) 평가 결과, 2단계 방법론이 가장 높은 성능(20.7%)을 기록하였다. 이는 1단계의 탐색 과정이 더 일반화 가능한 특징을 학습하도록 돕는다는 가설을 뒷받침한다.
4. **순서의 중요성:** '최대화 $\rightarrow$ 최소화' 순서가 '최소화 $\rightarrow$ 최대화' 순서보다 모든 노이즈 레벨에서 일관되게 우수한 성능을 보였다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 연구는 GRPO의 그룹 내 정규화(Group-relative normalization)가 어느 정도 자체적인 노이즈 내성을 가지고 있음을 발견하였다. 모든 샘플이 동일한 보상을 받으면 어드밴티지가 0이 되어 해로운 그래디언트가 생성되지 않는 'Self-gating' 효과가 발생한다. 제안된 2단계 엔트로피 전략은 이 견고한 베이스라인 위에 얹어져, 초기에는 노이즈에 의한 과적합을 막고 후기에는 예측을 정교화함으로써 성능을 극대화한다.

### 한계 및 비판적 해석

논문에서 명시한 한계점은 **Base Model의 초기 능력(Zero-shot ability)**에 의존적이라는 점이다. 베이스 모델의 초기 능력이 너무 낮을 경우, 1단계 엔트로피 최대화 과정에서 정답 궤적(Correct trajectory)을 찾기 전 잘못된 모드(Erroneous modes)가 증폭될 위험이 있다. 이는 Pets37 데이터셋의 100% 노이즈 설정 결과에서 일부 관찰된다.

또한, $\tau_{\text{switch}}$를 80%로 고정한 단순한 조각 함수를 사용하였으나, 실제 적용 시에는 모델의 수렴 속도에 따라 이 지점을 동적으로 결정하는 더 정교한 스케줄러가 필요할 것으로 보인다.

## 📌 TL;DR

본 논문은 MLLM의 RLVR 학습 시 데이터 노이즈 문제를 해결하기 위해, 학습 초반에는 엔트로피를 최대화하여 탐색을 촉진하고 후반에는 엔트로피를 최소화하여 지식을 공고히 하는 **2단계 엔트로피 최적화 방법론**을 제안한다. 실험 결과, 이 방법은 다양한 모델 크기와 작업(GUI Grounding, Classification, OVOD)에서 노이즈에 대한 강건성을 크게 향상시켰으며, 특히 높은 노이즈 환경에서 탁월한 성능을 보였다. 이 연구는 불완전한 데이터셋을 활용한 MLLM 사후 학습(Post-training)에 있어 매우 실용적인 가이드라인을 제공한다.
