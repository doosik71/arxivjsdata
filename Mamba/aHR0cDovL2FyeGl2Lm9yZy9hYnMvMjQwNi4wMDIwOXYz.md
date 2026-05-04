# Mamba State-Space Models Are Lyapunov-Stable Learners

John T. Halloran, Manbir Gulati, Paul Roysdon (2025)

## 🧩 Problem to Solve

본 논문은 Mamba State-Space Models (SSMs)의 학습 안정성, 특히 재귀적 역학(recurrent dynamics)의 민감도 문제를 다룬다. 일반적으로 SSM과 같은 재귀 기반 딥러닝 모델은 작은 변화에도 출력값이 기하급수적으로 변할 수 있는 민감성 문제가 존재하며, 이는 학습의 불안정성으로 이어진다.

특히 최근 대규모 언어 모델(LLM)의 효율적인 적응을 위해 Mixed-Precision Fine-Tuning (MPFT)과 Parameter-Efficient Fine-Tuning (PEFT)이 널리 사용되고 있으나, Mamba 모델이 이러한 정밀도 감소나 파라미터 효율적 튜닝 환경에서 얼마나 안정적으로 동작하는지에 대해서는 연구된 바가 없다. 기존의 Transformer 기반 LLM들은 MPFT 및 PEFT 적용 시 Full-precision(FP32) 모델과 비교하여 성능이 급격히 하락하는 'Large deviation spikes' 현상이 발생하는데, Mamba가 이러한 문제로부터 자유로운지를 규명하는 것이 본 연구의 주요 목표이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Mamba 및 Mamba-2 모델의 재귀적 역학이 역학 시스템 이론(Dynamical Systems Theory), 그 중에서도 **Lyapunov stability(리야푸노프 안정성)** 관점에서 안정적임을 이론적으로 증명하고 이를 실험적으로 검증한 것이다.

중심적인 아이디어는 Mamba의 상태 공간 방정식에서 최대 리야푸노프 지수(maximal Lyapunov exponent, $\lambda_{\max}$)가 0 이하임을 보임으로써, 입력값의 작은 변화($\epsilon$)가 시간이 지남에 따라 기하급수적으로 증폭되지 않고 오히려 감쇠하거나 일정하게 유지됨을 입증하는 것이다. 이를 통해 Mamba가 Transformer보다 저정밀도 튜닝 환경에서 훨씬 견고한(robust) 학습 성능을 보임을 입증하였다.

## 📎 Related Works

본 논문은 기존 연구의 한계를 다음과 같이 지적하며 차별점을 제시한다.

1. **MPFT 발산 연구의 한계**: 기존 Transformer 기반 LLM의 저정밀도 추론 및 튜닝 연구(Dettmers et al.)들은 주로 단일 샷(zero-shot 또는 5-shot) 성능만을 측정하였다. 본 논문은 이를 확장하여 $\{0, 1, 3, 5\}$-shot 성능의 평균 발산도를 측정함으로써, 기존에 발견되지 않았던 심각한 성능 저하 구간(deviation spikes)을 포착해냈다.
2. **SSM 안정성 분석의 한계**: 기존 S4 모델의 안정성 연구들은 선형 시불변(Linear Time-Invariant, LTI) 시스템 가정을 바탕으로 Hurwitz 행렬이나 고유값 분해를 사용하였다. 그러나 Mamba는 시변(Time-Varying, LTV) 시스템이므로 LTI 기반의 분석법을 적용할 수 없다. 본 논문은 LTV 시스템에서도 적용 가능한 리야푸노프 지수를 도입하여 분석하였다.
3. **ICL 연구의 확장**: 기존 Mamba의 In-Context Learning (ICL) 연구들은 주로 합성 데이터셋(synthetic tasks)에 치중되어 있었으나, 본 논문은 실제 자연어 처리 벤치마크에서 instruction tuning을 통한 ICL 능력 향상을 분석하였다.

## 🛠️ Methodology

### 전체 파이프라인 및 구조

Mamba 블록은 다음과 같은 상태 공간 방정식을 통해 입력을 출력으로 변환한다.

$$x_t = \bar{A}_t x_{t-1} + \bar{B}_t u_t$$
$$y_t = C_t x_t$$

여기서 $u_t$는 입력, $x_t$는 잠재 상태(latent state), $y_t$는 출력이며, $\bar{A}_t, \bar{B}_t, C_t$는 시변 파라미터이다. 구체적으로 $\bar{A}_t = \exp(\bar{\Delta}_t A)$이며, $\bar{\Delta}_t$는 $\text{softplus}$ 함수를 통해 계산된 이산 단계 크기(discrete step-size)를 제어한다.

### 이론적 분석: Lyapunov Stability

논문은 $\epsilon$만큼 차이가 나는 두 입력 궤적 사이의 발산 속도가 최대 리야푸노프 지수 $\lambda_{\max}$에 의해 결정된다는 점에 주목한다.

$$\max |F^N_\theta(x_0, u_1) - F^N_\theta(x_0 + \epsilon, u_1 + \epsilon)| \mathcal{O}(|\epsilon| \exp(N\lambda_{\max}))$$

- **Theorem 1 & 2**: Mamba의 $\lambda_{\max}$가 $0$ 이하임을 증명하여, 잠재 상태 $x_t$와 최종 출력 $y_t$의 변화가 시간이 지나도 기하급수적으로 증가하지 않고 비증가(non-increasing)함을 보였다.
- **Theorem 3 & 4**: 이 결과를 MPFT(정밀도 변환으로 인한 오차)와 PEFT(LoRA 등을 통한 가중치 변화) 상황으로 확장하여, 이러한 환경에서 발생하는 입력 및 상태의 변화가 출력값의 기하급수적 발산을 일으키지 않음을 증명하였다.

### 실험 절차

- **발산 측정**: FP32 Full fine-tuning 모델과 MPFT/PEFT 적용 모델 간의 성능 차이(Mean divergence)를 MMLU와 Winogrande 벤치마크에서 측정하였다.
- **비교 대상**: Mamba 및 Mamba-2 모델을 Transformer 기반 모델인 Pythia와 OpenELM과 비교하였다.
- **ICL 평가**: Instruction tuning 전후의 AIPSS(Average Improvement Percentage of Shot-vs-zero-shot)를 측정하여 ICL 능력을 평가하였다.

## 📊 Results

### 1. Mamba 블록의 역학적 안정성

무작위로 생성된 100개의 Mamba 블록에 초기 상태 변화 $\epsilon$을 주었을 때, 출력 상태의 최대 편차가 시간이 흐름에 따라 기하급수적으로 감소하며 원래의 궤적으로 수렴하는 것을 확인하였다.

### 2. MPFT 및 PEFT 하에서의 안정성

- **발산 스파이크**: Pythia는 9회, OpenELM은 4회의 'Large deviation spikes'(발산도 $> 1$)를 보인 반면, **Mamba 모델은 단 한 건의 스파이크도 발생하지 않았다.**
- **평균 발산도**: Winogrande 벤치마크에서 Mamba의 평균 발산도는 Pythia보다 약 2.9배, OpenELM보다 약 2.4배 낮았다.
- **하이퍼파라미터 강건성**: LoRA의 학습률(learning rate)이나 차원(dimension) 변화에 따른 성능 변동 폭이 Pythia($4.7\%$)보다 Mamba($1.5\%$)에서 훨씬 작게 나타나, 하이퍼파라미터 선택에 덜 민감함을 보였다.

### 3. Instruction Tuning과 ICL 능력

- **사전 학습 상태**: 사전 학습된 Mamba 모델의 ICL 능력은 Pythia 대비 38%~82% 수준으로 낮았다.
- **튜닝 후 상태**: Instruction tuning을 적용한 후, Mamba의 ICL 성능은 Pythia의 81.5%~132% 수준으로 크게 향상되었다. 특히 Mamba-2 2.7B 모델은 유사한 크기의 Transformer 모델들보다 뛰어난 ICL 성능을 보였다.

### 4. 하드웨어 효율성

MPFT와 PEFT를 적용했을 때, Mamba 790M 모델 기준 FP32 Full fine-tuning 대비 평균 학습 속도(Tokens per second)는 **1.74배 향상**되었으며, 토큰당 메모리 점유율은 **47.2% 감소**하였다.

## 🧠 Insights & Discussion

본 연구는 Mamba 모델이 단순한 성능 향상을 넘어, 수치적 안정성 측면에서 Transformer보다 우월한 특성을 가졌음을 이론과 실험으로 입증하였다.

**강점 및 해석**:

- **이론적 뒷받침**: 단순한 실험 결과 나열이 아니라 리야푸노프 안정성 이론을 통해 Mamba의 구조적 안정성을 증명함으로써, 왜 Mamba가 저정밀도 튜닝에서도 안정적인지를 설명하였다.
- **실용적 가치**: Transformer에서 빈번하게 발생하는 성능 급락(deviation spikes) 현상이 Mamba에서는 나타나지 않는다는 점은, 향후 Mamba 모델을 더 공격적인 저정밀도(FP8, NF4 등) 환경에서 튜닝할 수 있는 가능성을 제시한다.

**한계 및 논의**:

- **모델 크기의 제약**: 메모리 한계로 인해 가장 큰 Mamba 모델(1.4B, 2.8B)에 대한 FP32 Full fine-tuning 비교 실험을 수행하지 못했다.
- **ICL의 창발성**: Mamba의 ICL 능력이 단순한 사전 학습만으로는 부족하며, Instruction tuning이라는 추가적인 정렬 과정이 있어야 비로소 Transformer 수준의 능력이 창발(emergence)한다는 점은 시사하는 바가 크다.

## 📌 TL;DR

본 논문은 Mamba SSM이 리야푸노프 안정성(Lyapunov stability)을 가짐을 이론적으로 증명하고, 이를 통해 **Mixed-Precision 및 PEFT 튜닝 환경에서 Transformer보다 훨씬 안정적으로 학습됨**을 보였다. 특히 Transformer에서 나타나는 성능 급락 현상이 Mamba에서는 전혀 발견되지 않았으며, Instruction tuning을 통해 Transformer를 능가하는 In-Context Learning 능력을 확보할 수 있음을 입증하였다. 이는 Mamba가 매우 효율적이고 견고한 학습자(Stable Learner)임을 의미하며, 향후 초저정밀도 학습 및 거대 모델 최적화 연구에 중요한 근거가 될 것이다.
