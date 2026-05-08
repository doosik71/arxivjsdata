# Making Large Language Models Better Reasoners with Alignment

Peiyi Wang, Lei Li, Liang Chen, Feifan Song, Binghuai Lin, Yunbo Cao, Tianyu Liu, Zhifang Sui (2023)

## 🧩 Problem to Solve

본 논문은 거대 언어 모델(LLM)의 추론 능력을 향상시키기 위해 널리 사용되는 Chain-of-Thought(COT) 파인튜닝 과정에서 발생하는 **Assessment Misalignment(평가 미정렬)** 문제를 해결하고자 한다.

일반적인 Vanilla Fine-Tuning(VFT) 방식은 최대 가능도 추정(MLE) 목적 함수를 사용하여 모델이 정답지(Reference Answer)의 COT 경로만을 학습하도록 강제한다. 이 과정에서 모델은 정답에 도달하는 다양한 다른 올바른 경로들까지도 부정적인 예시(negative examples)로 처리하게 된다. 결과적으로 VFT를 거친 모델은 어떤 COT가 실제로 고품질인지 저품질인지 판단하는 능력이 부족해지며, 심지어 정답이 틀린 COT에 더 높은 점수(낮은 Perplexity)를 부여하는 현상이 발생한다. 이러한 평가 능력의 결여는 모델의 전반적인 추론 성능을 제한하는 핵심 원인이 된다.

따라서 본 연구의 목표는 모델이 고품질의 COT와 저품질의 COT를 정확하게 구분하여 점수를 매길 수 있도록 정렬(Alignment)함으로써, LLM의 추론 능력을 근본적으로 개선하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 LLM이 스스로 생성한 다양한 COT 응답들 사이의 품질 차이를 인식하게 만드는 **Alignment Fine-Tuning(AFT)** 패러다임을 제안한 것이다.

중심 아이디어는 단순히 정답을 맞히는 것뿐만 아니라, **'정답을 맞힌 COT(Positive)의 점수가 틀린 COT(Negative)의 점수보다 항상 높아야 한다'**는 제약 조건을 학습시키는 것이다. 특히, 저품질 응답의 점수를 무작정 낮추는 것이 아니라 적절한 범위 내에서 유지하도록 하는 **Constraint(제약)** 개념을 도입하여, 정렬 과정에서 발생할 수 있는 모델의 생성 능력 저하(Degradation)를 방지한 것이 이 연구의 핵심적인 설계 직관이다.

## 📎 Related Works

### 1. LLM 추론 능력 향상 연구

기존 연구들은 크게 세 가지 방향으로 진행되었다.

- **Pre-training**: 방대한 데이터셋을 통해 기초 추론 능력을 확보한다.
- **Fine-tuning**: COT 데이터셋을 통해 추론 과정을 학습시킨다.
- **Prompting**: COT 프롬프팅이나 Self-consistency 전략을 통해 파라미터 변경 없이 성능을 높인다.
본 논문은 이 중 파인튜닝 방식에 집중하며, 기존 VFT가 가진 평가 미정렬 문제를 지적하며 차별점을 둔다.

### 2. LLM 정렬(Alignment) 연구

- **RLHF**: 인간의 피드백을 통해 보상 모델을 학습시키고 PPO 등의 강화학습으로 최적화하지만, 학습 효율과 복잡도가 높다는 한계가 있다.
- **SFT with Ranking**: 순위 기반 손실 함수를 사용하여 인간의 선호도에 맞게 정렬한다.
기존의 정렬 연구는 주로 모델의 안전성(Safety)에 집중해 왔으며, 추론 능력 향상을 위한 정렬은 간과되었다. 또한, DPO, RRHF, PRO와 같은 기존 랭킹 기반 방법론들은 저품질 예시의 점수를 낮출 때 적절한 제약 조건을 설정하지 않아 모델 성능이 오히려 하락하는 경향이 있음을 본 논문은 분석하였다.

## 🛠️ Methodology

### 1. 전체 파이프라인 (AFT Paradigm)

AFT는 다음과 같은 3단계 절차로 구성된다.

1. **VFT 단계**: COT 학습 데이터를 사용하여 모델을 기본적으로 파인튜닝한다.
2. **데이터 생성 및 분류**: 각 질문에 대해 모델이 여러 개의 COT 응답을 생성하게 하고, 최종 정답의 정오 여부에 따라 이를 Positive 그룹($G^P$)과 Negative 그룹($G^N$)으로 분류한다.
3. **점수 교정(Calibration)**: 제안된 **Constraint Alignment (CA) loss**를 통해 두 그룹 간의 점수 체계를 정렬한다.

### 2. 주요 구성 요소 및 방정식

#### 점수 정의

모델이 생성한 COT $c$에 대한 점수 $s_c^\theta$는 토큰 평균 로그 가능도(token-averaged log-likelihood)로 계산한다.
$$s_c^\theta = \frac{1}{|c|} \sum_{j=1}^{|c|} \log P(c_j | c_{<j}, q; \theta)$$

#### Alignment Term

먼저, Positive 샘플의 점수가 Negative 샘플보다 높도록 유도하는 InfoNCE 기반의 손실 함수 $L_A$를 정의한다.
$$L_A = \log \left[ 1 + \sum_{c^p \in G^P} \sum_{c^n \in G^N} \exp(s_{c^n}^\theta - s_{c^p}^\theta) \right]$$

#### Constraint Methods (모델 저하 방지)

저품질 COT라 하더라도 어느 정도 수준의 품질을 유지하고 있으므로, 점수를 무분별하게 낮추면 모델이 망가질 수 있다. 이를 해결하기 위해 두 가지 제약 방법을 제안한다.

**1) Detached Constraint (DC)**
Negative 샘플의 점수에 대한 그래디언트를 끊어(detach), 오직 Positive 샘플의 점수를 높이는 방향으로만 학습한다.
$$L_{DC}^A = \log \left[ 1 + \sum_{c^p \in G^P} \sum_{c^n \in G^N} \exp(D(s_{c^n}^\theta) - s_{c^p}^\theta) \right]$$
여기서 $D(\cdot)$는 detach 연산을 의미한다.

**2) Boundary Constraint (BC)**
Negative 샘플의 점수가 특정 경계 $B$보다 낮아질 경우에만 점수를 높이도록 유도하는 제약 항을 추가한다.
$$L_{BC}^A = \log \left\{ 1 + \sum_{c^p \in G^P} \sum_{c^n \in G^N} \left[ \exp(s_{c^n}^\theta - s_{c^p}^\theta) + \exp(T - s_{c^n}^\theta) \right] \right\}$$

- **Alignment term**: $\exp(s_{c^n}^\theta - s_{c^p}^\theta)$는 $s_{c^p}^\theta$를 높이고 $s_{c^n}^\theta$를 낮춘다.
- **Constraint term**: $\exp(T - s_{c^n}^\theta)$는 $s_{c^n}^\theta$가 너무 낮아지면 이를 다시 높이는 역할을 한다.
- **경계 설정**: 경계 $B$는 $\min(s_{c^p}^\theta) - \beta$로 설정하며, 여기서 $\beta$는 하이퍼파라미터이다. 이를 만족하는 $T$ 값은 $T = 2s_{c^p^*}^\theta - 2\beta - s_{c^p}^\theta$로 도출된다.

### 3. Ranking Alignment로의 확장

이진 피드백(정답/오답)을 넘어, COT들 간의 상세한 순위 정보($c_1 \succ c_2 \succ \dots \succ c_k$)가 있을 경우 이를 $L_{RBC}^A$로 확장하여 더 세밀한 학습 신호를 제공할 수 있다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: GSM8K(수학), AQUA-RAT(대수학), ECQA(상식 추론), GSM8K-RANK(순위 기반 평가용).
- **모델**: Llama, Llama2 (7B, 13B).
- **비교 대상**: VFT, RFT(Rejective Sampling FT), RRHF, PRO.
- **평가 지표**: 정답 정확도(Accuracy).

### 2. 주요 결과

- **이진 피드백 상황**: AFT는 모든 모델과 데이터셋에서 VFT보다 유의미하게 높은 성능을 보였으며, RFT보다도 약간 더 좋은 성능을 기록하였다. 평균적으로 VFT 대비 약 $1.91\% \sim 2.57\%$의 정확도 향상을 보였다.
- **순위 피드백 상황**: GSM8K-RANK 데이터셋에서 AFT($L_{RBC}^A$)는 RFT를 포함한 모든 베이스라인을 압도하였다. 특히 RRHF나 PRO 같은 기존 랭킹 기반 방법들은 VFT보다 오히려 성능이 하락하는 경향을 보였는데, AFT는 이를 극복하고 큰 폭의 향상을 이루었다.
- **OOD 및 멀티태스크 성능**: 세 가지 데이터셋을 동시에 학습시킨 멀티태스크 환경과 외부 데이터셋인 MMLU(Zero-shot) 평가에서도 AFT가 VFT보다 뛰어난 일반화 성능을 보였다.

## 🧠 Insights & Discussion

### 1. Constraint의 중요성

본 논문의 가장 중요한 발견은 **정렬 학습 시 '제약 조건(Constraint)'이 필수적**이라는 점이다. 실험 결과, 제약 조건 없이 랭킹 손실 함수만 적용했을 경우 모델의 생성 능력이 심각하게 훼손되는 현상이 발견되었다. 케이스 스터디(Case Study)에 따르면, 제약 조건이 없을 때 모델은 의미 없는 토큰을 반복해서 출력하는 등 붕괴 현상을 보였다. 이는 저품질 샘플의 점수를 무조건 낮추는 것이 모델의 전반적인 언어 생성 확률 분포를 왜곡시키기 때문으로 해석된다.

### 2. 기존 방법론(DPO, RRHF, PRO)에 대한 비판적 분석

저자들은 기존의 정렬 방법들이 모델 붕괴를 막기 위해 그래디언트 가중치를 줄이거나(DPO), 특정 조건에서 손실을 0으로 만드는(RRHF) 등의 전략을 썼지만, 이는 근본적인 해결책이 아니라고 분석한다. 단순히 가중치 크기를 조절하는 것만으로는 부족하며, 본 논문에서 제안한 것처럼 **점수의 하한선을 정하는 방향성 제어(Boundary Constraint)**가 필요함을 수학적 그래디언트 분석을 통해 입증하였다.

### 3. 한계점 및 향후 과제

- **모델 크기**: 리소스 제한으로 인해 65B 이상의 거대 모델에 적용하지 못했다.
- **하이퍼파라미터 의존성**: $\beta$ 값에 따라 성능 변화가 크기 때문에, 검증 세트를 통한 탐색 비용이 발생한다. 이를 자동화하거나 동적으로 조절하는 방법이 향후 연구 과제로 남았다.

## 📌 TL;DR

본 논문은 LLM이 정답을 맞힌 COT와 틀린 COT를 제대로 구분하지 못하는 **Assessment Misalignment** 문제를 정의하고, 이를 해결하기 위한 **Alignment Fine-Tuning(AFT)** 방법을 제안한다. 특히 저품질 응답의 점수가 일정 수준 이하로 떨어지지 않도록 하는 **Boundary Constraint**를 도입하여, 모델의 생성 능력을 보존하면서 추론 능력을 향상시켰다. 이 방법은 수학 및 상식 추론 벤치마크에서 기존 VFT 및 랭킹 기반 방법론들보다 우수한 성능을 보였으며, 기존 정렬 방법론들이 간과했던 '제약 조건'의 중요성을 학술적으로 규명하였다.
