# SPEAR: Soft Prompt Enhanced Anomaly Recognition for Time Series Data

Hanzhe Wei, Jiajun Wu, Jialin Yang, Henry Leung, and Steve Drew (2025)

## 🧩 Problem to Solve

본 연구는 시계열 데이터(Time Series Data)에서의 이상치 탐지(Anomaly Detection) 문제를 해결하고자 한다. 시계열 이상치 탐지는 헬스케어, 인터넷 트래픽 모니터링, 산업 시스템 감시 등 광범위한 분야에서 매우 중요하다.

기존의 접근 방식들은 다음과 같은 한계점을 가지고 있다:

- **가변 길이 및 문맥 의존성:** 시계열 데이터의 길이가 다양하고, 단순한 수치적 이상치뿐만 아니라 문맥적(Context-based) 이상치를 탐지하는 데 어려움이 있다.
- **LLM 활용의 비용 및 리소스 문제:** 최근 Large Language Models(LLMs)가 시계열 분석에 도입되고 있으나, 전체 모델을 Fine-tuning하는 것은 계산 비용이 매우 높고 사전 학습된 특징(Pretrained features)을 왜곡시킬 위험이 있다.
- **Zero-shot 접근법의 한계:** Zero-shot 방식은 GPT-4와 같은 초거대 모델과 정교하게 설계된 Prompt Template에 의존하며, 이는 높은 비용과 개인정보 보호 문제를 야기한다.

따라서 본 논문의 목표는 계산 효율성을 높이면서도 소규모 LLM(Smaller LLMs)을 활용하여 고성능의 시계열 이상치 탐지를 수행할 수 있는 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Soft Prompt**와 **Quantization(양자화)**을 결합하여, 모델의 가중치를 고정(Frozen)한 상태에서 소규모 LLM을 시계열 이상치 탐지 작업에 효율적으로 적응시키는 것이다.

- **Soft Prompt 활용:** 텍스트 기반의 하드 프롬프트 대신, 학습 가능한 연속적인 벡터 형태의 Soft Prompt를 사용하여 모델이 시계열 데이터의 특성을 더 잘 파악하도록 유도한다.
- **데이터 양자화:** 연속적인 시계열 수치 데이터를 LLM이 처리할 수 있도록 이산적인 토큰(Discrete tokens)으로 변환하는 양자화 과정을 도입하였다.
- **효율적 적응:** LLM 본체는 고정하고 Soft Prompt와 분류 헤드(Classification Head)만 학습시킴으로써, 적은 메모리와 계산 자원으로도 SOTA 모델에 근접한 성능을 달성하고자 하였다.

## 📎 Related Works

### 1. Soft Prompts

Lester 등이 제안한 Prompt Tuning은 전체 모델을 튜닝하는 대신 연속적인 프롬프트 임베딩만을 최적화하여 Full Fine-tuning에 근접한 성능을 낼 수 있음을 보였다. 이는 NLP 분야에서는 널리 활용되었으나, 시계열 데이터 분야에서의 탐색은 상대적으로 부족했다.

### 2. LLMs for Anomaly Detection

최근 LLMAD와 같이 Few-shot learning이나 Chain-of-Thought를 활용한 연구들이 등장하였다. 또한, 신호를 텍스트로 변환하여 ARIMA나 LSTM보다 우수한 성능을 보인 사례가 보고되었다. 그러나 이러한 방법들은 정교한 채팅 템플릿 설계가 필요하거나, 계산 비용이 높은 Fine-tuning에 의존한다는 한계가 있다.

SPEAR는 이러한 기존 방식과 달리, 양자화와 Soft Prompt를 통해 프롬프트 엔지니어링의 부담을 줄이고 소규모 모델의 효율성을 극대화했다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 전체 파이프라인

SPEAR의 전체 구조는 **데이터 전처리 $\rightarrow$ 양자화 및 임베딩 $\rightarrow$ Soft Prompt 결합 $\rightarrow$ Frozen LLM 통과 $\rightarrow$ 이진 분류** 순으로 진행된다.

### 1. 데이터 전처리 및 양자화

연속적인 시계열 데이터 $X = \{x_1, x_2, ..., x_T\}$를 LLM의 입력 형식으로 변환하기 위해 다음 과정을 거친다:

- **Scaling:** Min-Max Scaling을 통해 데이터를 $[0, 1]$ 범위로 정규화한다.
- **Quantization:** 정규화된 데이터를 $N$개의 이산적인 빈(Bins)으로 나누어 각 시점 $t$의 값을 $q_t = Q(x_t)$로 변환한다.
- **Embedding:** 양자화된 토큰 $q_t$를 임베딩 행렬 $E \in \mathbb{R}^{N \times d}$를 통해 고차원 벡터 $e_t = E_{q_t}$로 매핑한다.

### 2. Soft Prompt 및 입력 구성

학습 가능한 소프트 프롬프트 임베딩 $P = \{p_1, p_2, ..., p_m\}$을 정의한다. 이 벡터들은 LLM의 어텐션 메커니즘이 시계열의 이상치 탐지 작업에 집중하도록 가이드하는 역할을 한다.
최종 입력 시퀀스 $S$는 다음과 같이 결합된다:
$$S = [p_1, p_2, ..., p_m, e_1, e_2, ..., e_T]$$

### 3. 추론 및 학습 절차

- **Frozen LLM:** 결합된 임베딩 $S$는 BERT나 Gemma와 같은 사전 학습된 LLM에 입력된다. 이때 LLM의 가중치는 고정된다.
- **Classification Head:** LLM의 출력값 $h_{m+t}$에 선형 레이어를 추가하여 로짓 $z_t$를 생성하고, 시그모이드 함수를 통해 확률 $\hat{y}_t$를 계산한다.
  $$z_t = Wh_{m+t} + b$$
  $$\hat{y}_t = \sigma(z_t) = \frac{1}{1 + e^{-z_t}}$$
- **손실 함수:** 예측값 $\hat{y}_t$와 실제 라벨 $y_t$ 사이의 **Binary Cross-Entropy (BCE) Loss**를 사용하며, AdamW 옵티마이저를 통해 Soft Prompt 임베딩과 분류 헤드의 가중치만을 업데이트한다.

### 4. 데이터 보강 및 라벨링

- **T-SMOTE:** 클래스 불균형 문제를 해결하기 위해 시계열 데이터에 특화된 SMOTE인 T-SMOTE를 사용하여 소수 클래스(이상치)의 합성 샘플을 생성한다.
- **Context Anomalies Labeling:** MIMIC-IV 데이터셋의 풍부함을 위해 다음 4가지 통계적 방법으로 문맥적 이상치를 정의하고 라벨링하였다:
    1. **Monotonic Trend:** 선형 회귀의 기울기 $|\beta_1| > 0.01$ 및 $p\text{-value} < 0.05$일 때 탐지.
    2. **Sudden Spike:** 1차 차분 $d_t = |x_t - x_{t-1}|$가 $\mu_d + 3\sigma_d$를 초과할 때 탐지.
    3. **Sudden Shift:** 두 구간으로 나누어 t-test를 수행하여 $p\text{-value} < 0.05$일 때 탐지.
    4. **Volatility Change:** Levene’s test를 통해 두 구간의 분산 차이가 유의미할 때($p\text{-value} < 0.05$) 탐지.

## 📊 Results

### 실험 설정

- **데이터셋:** MIMIC-IV (의료 데이터), NASA (위성 텔레메트리), NAB AWS (IT 인프라).
- **비교 대상:** LSTM, Zero-shot Gemma, Zero-shot BERT, Zero-shot GPT-4.
- **SPEAR 모델:** BERT-Base(110M) 및 Gemma-2B를 백본으로 사용.
- **평가 지표:** Accuracy, F1-Score, Recall, Precision, AUROC, AUPR.

### 주요 결과

1. **SPEAR-BERT의 우수성:** 모든 데이터셋에서 SPEAR-BERT가 가장 우수한 성능을 보였다. 특히 MIMIC-IV에서는 Accuracy 0.93, F1-Score 0.9285를 기록하며 LSTM과 Zero-shot 모델들을 압도하였다.
2. **Zero-shot 모델의 한계:** GPT-4, Gemma 등의 Zero-shot 모델들은 Recall은 높으나 Precision이 매우 낮게 나타났다. 이는 작업 특화 가이드(Soft Prompt)가 없을 때 모델이 대부분의 포인트를 이상치로 과잉 예측(Overpredict)하는 경향이 있음을 보여준다.
3. **SPEAR-BERT vs SPEAR-Gemma:** BERT 기반 모델이 Gemma 기반 모델보다 일관되게 높은 성능을 보였다. 이는 BERT의 Masked Language Modeling 사전 학습 방식이 Soft Prompt 수정에 더 유연하게 반응하거나, Gemma와 같은 더 큰 모델은 Soft Prompt만으로는 최적화가 부족할 수 있음을 시사한다.
4. **LSTM과의 비교:** 가변 길이 시퀀스가 많은 MIMIC-IV에서 SPEAR는 LSTM보다 뛰어난 성능을 보였다. 이는 LSTM의 고정 입력 및 패딩으로 인한 노이즈 문제 없이, 트랜스포머의 유연한 길이 처리를 활용했기 때문이다.
5. **효율성:** Soft Prompt의 메모리 사용량은 매우 적다 (BERT 기준 0.06 MB). 이는 모델 전체를 튜닝하는 것보다 훨씬 효율적이며, 하드웨어 제약이 있는 환경에서도 배포 가능함을 입증한다.

## 🧠 Insights & Discussion

### 강점

- **효율적인 적응:** LLM의 가중치를 고정하고 극소수의 파라미터(Soft Prompt)만 학습시켜도 시계열 데이터라는 이종 도메인에 성공적으로 적응시켰다.
- **과잉 예측 해결:** 단순 Zero-shot 모델들이 겪는 낮은 Precision 문제를 Soft Prompt를 통한 Calibration으로 해결하여, 실제 시스템 적용 시 발생할 수 있는 과도한 False Positive를 줄였다.
- **유연성:** 양자화 과정을 통해 연속적 수치 데이터를 토큰화함으로써, 텍스트 기반 LLM을 시계열 분석에 범용적으로 사용할 수 있는 경로를 제시했다.

### 한계 및 비판적 해석

- **양자화의 손실:** 연속적인 수치를 $N$개의 빈으로 나누는 양자화 과정에서 필연적으로 정보 손실이 발생한다. $N$의 설정값이 성능에 미치는 영향에 대한 더 깊은 분석이 필요하다.
- **프록시 이상치의 한계:** MIMIC-IV에 도입한 문맥적 이상치들이 통계적 휴리스틱에 기반한 '프록시(Proxy)'일 뿐, 실제 의료적 유효성이 검증된 이벤트는 아니라는 점이 명시되어 있다.
- **모델 크기 역설:** 더 큰 모델(Gemma)보다 작은 모델(BERT)이 더 좋은 성능을 낸 점은 흥미로우나, 이는 Soft Prompt의 표현력 한계일 수 있다. 더 큰 모델을 위한 다른 효율적 튜닝 기법(LoRA 등)과의 비교 연구가 추가되어야 할 것이다.

## 📌 TL;DR

본 논문은 소규모 LLM을 시계열 이상치 탐지에 활용하기 위해 **양자화(Quantization)**와 **학습 가능한 Soft Prompt**를 결합한 **SPEAR** 프레임워크를 제안한다. 실험 결과, SPEAR-BERT는 계산 비용을 최소화하면서도 Zero-shot 거대 모델(GPT-4 등)이나 전통적인 LSTM보다 뛰어난 성능을 보였으며, 특히 불균형 데이터셋에서 정밀한 탐지 능력을 입증하였다. 이 연구는 고비용의 Fine-tuning 없이도 LLM을 특정 도메인 분석 작업에 효율적으로 전이시킬 수 있음을 보여주며, 향후 실시간 모니터링 및 엣지 컴퓨팅 환경의 이상치 탐지 시스템에 적용될 가능성이 높다.
