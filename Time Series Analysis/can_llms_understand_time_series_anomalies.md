# CAN LLMS UNDERSTAND TIME SERIES ANOMALIES?

Zihao Zhou, Rose Yu (2025)

## 🧩 Problem to Solve

본 논문은 대규모 언어 모델(Large Language Models, LLMs)이 시계열 데이터의 이상치 탐지(Anomaly Detection) 작업을 수행함에 있어 실제로 시계열의 패턴을 '이해'하고 있는지, 아니면 단순한 통계적 편향이나 산술 능력에 의존하는지를 분석한다.

기존의 시계열 예측(Forecasting) 연구에서는 LLM의 제로샷(Zero-shot) 능력이 강조되었으나, 예측 작업의 평가지표인 MSE(Mean Squared Error)는 모델이 데이터의 역동성을 깊이 있게 이해하지 못하고 단순히 평균적인 선을 출력하더라도 양호한 점수를 얻을 수 있다는 한계가 있다. 반면, 이상치 탐지는 불규칙한 행동을 정확히 짚어내야 하므로 LLM이 시계열의 기저 패턴을 실제로 파악하고 있는지 테스트하기에 더 적합한 과제이다. 따라서 본 연구의 목표는 제로샷 및 퓨샷(Few-shot) 시나리오에서 LLM과 멀티모달 LLM(M-LLMs)의 이상치 탐지 능력을 체계적으로 조사하고, 모델의 동작 원리에 관한 가설들을 검증하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 현대적인 LLM 및 M-LLM이 시계열 이상치 탐지에서 보이는 행동 양식을 최초로 포괄적으로 분석했다는 점이다. 주요 발견 사항은 다음과 같다.

1. **시각적 이점(Visual Advantage):** LLM은 시계열 데이터를 텍스트 토큰으로 처리할 때보다 이미지 형태로 처리할 때 훨씬 더 뛰어난 성능을 보인다.
2. **추론 능력의 한계(Limited Reasoning):** 시계열 분석 시 Chain-of-Thought(CoT)와 같은 명시적 추론 유도 프롬프팅이 성능 향상에 기여하지 않으며, 오히려 성능이 저하되는 경우가 많다.
3. **처리 메커니즘의 재정의:** LLM의 시계열 이해 능력이 기존의 믿음과 달리 토큰 반복 편향(Repetition Bias)이나 단순 산술 능력(Arithmetic Ability)에서 기인하지 않음을 입증하였다.
4. **모델 간 이질성(Model Heterogeneity):** 모델 아키텍처에 따라 시계열 이해 및 이상치 탐지 능력이 크게 다르므로, 특정 작업에 맞는 모델 선택이 중요함을 보였다.

## 📎 Related Works

### 관련 연구 및 한계

- **시계열 분석을 위한 LLM:** Gruver et al. (2023) 등은 LLM이 제로샷으로 시계열 예측을 수행할 수 있으며, 이는 모델의 패턴 외삽(Extrapolation) 능력 덕분이라고 주장했다. 그러나 Tan et al. (2024)은 LLM 구성 요소를 제거하거나 단순한 어텐션 레이어로 대체해도 성능이 비슷하거나 오히려 향상된다고 주장하며 LLM의 실질적 유용성에 의문을 제기했다.
- **시계열 이상치 탐지:** 전통적인 통계 방법론과 최근의 딥러닝 기반 접근법(Transformer, Variational Graph Convolutional Recurrent Network 등)이 존재한다. 하지만 많은 벤치마크 데이터셋이 trivial(너무 쉬움)하거나 레이블링 오류가 있다는 지적이 있어, 통제된 환경에서의 실험이 필요하다.
- **멀티모달 LLM (M-LLMs):** Qwen-VL, Phi-3-Vision과 같은 모델들이 등장했으나, 인간이 시계열 이상치를 주로 시각적 검사를 통해 찾아낸다는 점에도 불구하고 시계열 데이터를 이미지로 입력하여 분석한 연구는 부족한 상태였다.

### 기존 접근 방식과의 차별점

본 연구는 단순한 성능 측정에 그치지 않고, LLM의 내부 동작 원리에 대한 6가지 구체적인 과학적 가설을 세우고 이를 검증하기 위한 통제된 실험(Controlled Experiments)을 설계했다는 점에서 기존 연구와 차별화된다.

## 🛠️ Methodology

### 이상치 정의 및 분류

본 논문에서는 시계열 $X=\{x_1, x_2, \dots, x_T\}$에 대해 두 가지 관점에서 이상치를 정의한다.

1. **결정론적 생성 함수(Deterministic Generating Function):** 데이터 포인트 $x_t$가 예측값 $G$에서 $\delta$ 이상 벗어날 때 이상치로 간주한다.
    $$|x_t - G(x_t | x_{t-1}, x_{t-2}, \dots, x_{t-n})| > \delta$$
2. **조건부 확률(Conditional Probability):** 데이터 포인트 $x_t$의 조건부 확률이 임계값 $\epsilon$보다 낮을 때 이상치로 간주한다.
    $$P(x_t | x_{t-1}, x_{t-2}, \dots, x_{t-n}) < \epsilon$$

이상치 패턴은 크게 **범위 외 이상치(Out-of-range anomalies)**와 **문맥적 이상치(Contextual anomalies)**로 분류하며, 문맥적 이상치는 다시 추세(Trend), 주파수(Frequency), 문맥적 포인트(Contextual Point) 이상치로 세분화한다.

### 검증 가설 (Hypotheses)

- **H1 (CoT Reasoning):** LLM은 시계열 데이터에 대해 단계별 추론을 수행해도 이득이 없다.
- **H2 (Repetition Bias):** LLM의 반복 편향이 주기적 구조 인식 능력의 근거가 된다.
- **H3 (Arithmetic Reasoning):** LLM의 산술 능력이 선형/지수 추세 외삽 능력과 연결된다.
- **H4 (Visual Reasoning):** 시계열 이상치는 텍스트보다 시각적 입력으로 더 쉽게 탐지된다.
- **H5 (Visual Perception Bias):** LLM은 인간의 시각적 인지 편향(예: 가속도 인지 저하)과 유사한 한계를 보인다.
- **H6 (Long Context Bias):** 토큰 수가 적을수록(정보 손실이 있더라도) 성능이 향상된다.

### 실험 설계

- **모델:** Qwen-VL-Chat, InternVL2-Llama3-76B (오픈소스), GPT-4o-mini, Gemini-1.5-Flash (상용).
- **입력 표현:**
  - **텍스트:** Original, CSV, Prompt as Prefix (PAP, 통계치 포함), Token per Digit (TPD, 숫자 분리).
  - **시각:** Matplotlib을 이용해 생성한 시계열 그래프 이미지.
- **프롬프팅:** Zero-shot, Few-shot (FSL), Chain-of-Thought (CoT).
- **출력 형식:** 분석의 일관성을 위해 JSON 리스트 형태(`[{"start": 10, "end": 25}, ...]`)로 출력을 제한했다.
- **평가지표:** 시계열의 시간적 특성을 반영하기 위해 단순 F1-score 대신 **Affinity F1 score**를 주 지표로 사용한다.

## 📊 Results

### 가설 검증 결과

- **H1 (CoT Reasoning) $\rightarrow$ 유지 (Retained):** CoT 프롬프팅을 사용했을 때 오히려 성능이 꾸준히 하락했다. 이는 LLM이 시계열 분석 시 인간과 같은 단계적 논리 추론을 사용하지 않음을 시사한다.
- **H2 (Repetition Bias) $\rightarrow$ 기각 (Rejected):** 텍스트 데이터에 미세한 노이즈를 섞어 토큰 반복성을 깨뜨렸음에도 성능 저하가 크지 않았다. 즉, 주기성 인식은 단순 토큰 반복 편향 때문이 아니다.
- **H3 (Arithmetic Reasoning) $\rightarrow$ 기각 (Rejected):** 인컨텍스트 러닝을 통해 모델의 산술 능력을 의도적으로 저하시켰음에도(DysCalc), 이상치 탐지 성능은 일정하게 유지되었다.
- **H4 (Visual Reasoning) $\rightarrow$ 유지 (Retained):** 거의 모든 모델과 이상치 유형에서 이미지 입력 시의 성능이 텍스트 입력보다 월등히 높았다.
- **H5 (Visual Perception Bias) $\rightarrow$ 기각 (Rejected):** 인간이 시각적으로 감지하기 어려운 'Flat Trend' 이상치를 LLM은 일반 추세 이상치와 비슷하게 잘 탐지해냈다. LLM은 인간의 시각적 인지 편향을 공유하지 않는다.
- **H6 (Long Context Bias) $\rightarrow$ 유지 (Retained):** 데이터를 보간법(Interpolation)으로 30% 수준으로 축소하여 토큰 수를 줄였을 때 성능이 일관되게 향상되었다.

### 종합 성능 및 모델 비교

- **모델별 특성:** Gemini-1.5-Flash가 전반적으로 가장 우수한 성능을 보였으며, Qwen은 텍스트보다 이미지 입력에서 훨씬 강한 모습을 보였다. GPT-4o-mini는 긴 시퀀스에서도 성능 저하가 상대적으로 적었다.
- **전통적 방법론과의 비교:** 적절한 프롬프트와 시각적 입력을 사용한 LLM은 포인트, 범위, 추세 이상치 탐지에서 Isolation Forest나 단순 임계값 방식(Thresholding)보다 우수한 성능을 보였다. 다만, 주파수 이상치는 여전히 LLM에게 어려운 과제이다.

## 🧠 Insights & Discussion

### 강점 및 가능성

LLM, 특히 M-LLM은 시각적 정보를 통해 시계열의 거시적 패턴을 파악하는 능력이 뛰어나다. 이는 인간 전문가가 그래프를 통해 이상치를 찾는 방식과 유사하며, 제로샷 상황에서 전통적인 통계 모델보다 더 유연한 탐지가 가능함을 보여준다.

### 한계 및 비판적 해석

1. **추론의 부재:** CoT의 실패는 LLM이 시계열 데이터를 '수치적 논리'로 처리하는 것이 아니라, 일종의 '패턴 매칭'이나 '시각적 특징 추출' 방식으로 처리하고 있음을 암시한다.
2. **컨텍스트 길이 문제:** 토큰 수 감소 시 성능이 향상되는 점은 현재의 LLM 토크나이저가 수치 데이터를 효율적으로 처리하지 못하고 있음을 보여준다.
3. **실제 적용의 주의점:** LLM이 단순한(trivial) 이상치는 잘 찾지만, 아주 미세하고 복잡한 실제 세계의 이상치를 이해한다는 증거는 부족하다. 특히 주파수 분석과 같은 영역에서는 푸리에 변환(Fourier Analysis) 같은 전처리를 거친 후 LLM에 입력하는 파이프라인이 필요할 것이다.

## 📌 TL;DR

본 연구는 LLM의 시계열 이상치 탐지 능력을 체계적으로 분석하여, **"LLM은 텍스트보다 이미지로 시계열을 더 잘 이해하며, 이는 산술 능력이나 토큰 반복 편향, 혹은 명시적 논리 추론(CoT)과는 무관한 메커니즘에 기반한다"**는 것을 밝혔다. 또한, 모델별 성능 차이가 크고 긴 시퀀스 처리에서 한계를 보임을 확인하였다. 이 결과는 향후 시계열 분석 시스템 설계 시 단순 텍스트 입력보다는 시각화 및 데이터 축소 전략을 사용하고, 모델 선택에 신중해야 함을 시사한다.
