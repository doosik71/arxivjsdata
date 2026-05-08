# Can LLMs Serve As Time Series Anomaly Detectors?

Manqing Dong, Hao Huang, and Longbing Cao (2024)

## 🧩 Problem to Solve

최근 대규모 언어 모델(LLM)을 시계열 예측(Time Series Forecasting)에 적용하려는 시도가 많았으나, 시계열 이상치 탐지(Time Series Anomaly Detection)에 대한 탐구는 상대적으로 부족했다. 시계열 예측은 주로 데이터의 주류 패턴과 규칙적인 특성을 학습하는 반면, 이상치 탐지는 단순한 점(Point) 형태의 예외뿐만 아니라 문맥적(Contextual) 예외까지 처리해야 하므로 훨씬 더 도전적인 과제이다.

본 논문의 목표는 LLM이 시계열 데이터에서 이상치를 탐지하고, 그 이유를 텍스트로 설명할 수 있는 '설명 가능한 이상치 탐지기(Explainable Anomaly Detector)'로서 작동할 수 있는지 조사하는 것이다. 특히 GPT-4와 LLaMA3를 대상으로 제로샷(Zero-shot) 성능, 프롬프트 엔지니어링의 효과, 그리고 명령어 미세 조정(Instruction Fine-tuning)을 통한 성능 향상 가능성을 분석한다.

## ✨ Key Contributions

본 연구의 핵심 기여는 다음과 같다.

1. **LLM의 제로샷 이상치 탐지 능력 조사**: LLM이 시계열의 일반적인 이상 패턴을 이해하고 이를 설명할 수 있는지 실험적으로 분석하였다.
2. **전략적 프롬프트 엔지니어링 제안**: Multi-modal Instruction, In-context Learning, Chain-of-Thought(CoT) 등의 전략을 통해 LLM이 기존 베이스라인 모델들과 경쟁 가능한 수준의 탐지 성능을 낼 수 있음을 보였다.
3. **TTGenerator 및 합성 데이터셋 구축**: 이상치와 그에 대한 텍스트 설명이 포함된 데이터를 자동으로 생성하는 TTGenerator를 제안하고, 이를 통해 LLaMA3를 미세 조정하여 이상치 탐지 성능을 향상시켰다.

## 📎 Related Works

기존의 시계열 분석 연구는 크게 두 가지 방향으로 진행되었다. 첫째는 시계열을 토큰 시퀀스로 처리하여 LLM에 직접 입력하거나 프롬프트와 결합하는 Prompt Engineering 방식이고, 둘째는 LLM을 백본으로 사용하되 인코더/디코더를 추가하여 시계열 임베딩을 학습시키는 Aligning 방식이다.

이상치 탐지 분야에서도 LLM을 활용한 시도가 있었으나, 기존 연구들은 주로 이상치 탐지를 단순한 이진 분류(Binary Classification) 문제로 취급하거나, LLM을 교사 모델로 사용하여 학생 모델을 학습시키는 지식 증류(Knowledge Distillation) 방식에 치중하였다. 반면, LLM의 핵심 강점인 '텍스트 추론 능력'을 활용하여 왜 특정 지점이 이상치인지에 대한 상세한 설명을 생성하는 연구는 거의 이루어지지 않았다.

## 🛠️ Methodology

### 1. 프롬프트 전략 (Prompting Strategies)

LLM의 성능을 극대화하기 위해 다음과 같은 세 가지 전략을 사용한다.

- **Multi-modal Instruction**: LLM이 시계열 데이터의 수치뿐만 아니라 '시각적 표현'을 상상하며 분석하도록 유도하는 프롬프트를 추가한다.
- **In-context Learning**: 전역 점 이상치(Global Point Anomaly) 등 5가지 이상치 유형에 대한 정의, 예시 데이터, 그리고 이상적인 설명이 포함된 $n$-shot 예시를 제공한다.
- **Chain-of-Thought (CoT)**: "이상치 존재 여부 확인 $\rightarrow$ 인덱스 추출 $\rightarrow$ 이유 설명"의 순서로 단계별 추론을 수행하도록 가이드한다.

### 2. TTGenerator (시계열 및 텍스트 설명 생성기)

명령어 미세 조정을 위해 이상치와 설명이 쌍을 이루는 데이터셋이 필요하며, 이를 위해 TTGenerator를 설계하였다.

**기초 시계열 생성**: 시계열 $X$를 추세($\tau$), 계절성($s$), 노이즈($\epsilon$)의 합으로 정의한다.
$$X = s(T) + \tau(T) + \epsilon$$

- $s(T)$: 사인 함수(Sine wave), 사인 함수의 조합, 또는 Inverse Fast Fourier Transform(IFFT)를 통해 생성한다.
- $\tau(T)$: 선형(Linear) 또는 다항식(Polynomial) 추세를 적용한다.
- $\epsilon$: 평균 0, 분산 1인 백색 잡음(White noise)을 추가한다.

**이상치 생성**:

- **Point-wise Anomalies**: $|x_t - \hat{x}_t| > \delta$ 조건을 만족하도록 생성한다. 전역(Global) 이상치는 전체 표준편차 $\sigma(X)$를 기준으로, 국소(Local) 이상치는 주변 윈도우의 표준편차를 기준으로 $\delta$를 결정한다.
- **Pattern-wise Anomalies**: $sim(X_{i,j}, \hat{X}_{i,j}) > \delta$ 조건을 통해 계절성(진폭/주기 변화), 추세(변곡점/단절), 형태(패턴 변화/단절) 이상치를 생성한다.

**설명 생성**: 템플릿 기반으로 기초 패턴과 이상치 특성을 기술한 후, GPT-4를 이용해 문장을 자연스럽게 재작성(Rewrite)하여 최종 학습 데이터를 구축한다.

### 3. LLaMA3 미세 조정 (Instruction Fine-tuning)

TTGenerator로 생성된 데이터를 사용하여 LLaMA3-8B-Instruct 모델을 미세 조정한다. 모든 파라미터를 학습시키는 대신 LoRA(Low-Rank Adaptation)를 적용하여 효율적인 학습을 수행하였으며, 출력 형식을 JSON(anomaly 인덱스 리스트, reason 설명 문자열)으로 고정하여 일관성을 확보하였다.

## 📊 Results

### 1. GPT-4의 성능 및 베이스라인 비교

YAHOO, ECG, SVDB, IOPS 등 4개의 벤치마크 데이터셋에서 F-score와 Range-F(윈도우 내 탐지 시 정답 처리) 지표로 평가하였다.

- **결과**: GPT-4는 IForest, Autoencoder, LSTM, TimesNet 등 전통적인 모델들과 비교했을 때 평균 랭킹 3위를 기록하며 경쟁력 있는 성능을 보였다. 특히 ECG, SVDB, IOPS 데이터셋에서 매우 안정적인 성능을 나타냈다.
- **한계**: 시계열 길이가 길어질수록 실제 인덱스가 아닌 엉뚱한 값을 생성하는 Hallucination(환각) 현상이 발생하였다. 특히 이상치 비율이 높은 데이터셋(ECG, SVDB)에서 이러한 경향이 강하게 나타났다.

### 2. LLaMA3 미세 조정 결과

합성 데이터셋을 통해 미세 조정된 LLaMA3의 성능을 분석하였다.

- **결과**: 원래의 LLaMA3보다 미세 조정 후의 모델이 전반적인 F-score와 Range-F에서 향상을 보였으며, 특히 계절성(Seasonality) 이상치 탐지 능력이 크게 개선되었다.
- **특징**: LLaMA3는 GPT-4보다 설명의 구체성은 떨어지며 보다 일반적인 설명을 제공하는 경향이 있었다.

### 3. 설명 능력 분석

GPT-4의 생성 설명을 분석한 결과, 단순한 스파이크(Spike)나 딥(Dip) 형태의 점 이상치는 매우 정확하게 설명하였다. 하지만 복잡한 문맥적 이상치의 경우, 이를 단순한 점 이상치로 오해하거나 수치적 환각을 일으키는 경우가 관찰되었다.

## 🧠 Insights & Discussion

**강점 및 가능성**:
본 연구는 LLM이 단순한 수치 예측을 넘어, 시계열의 구조적 이상을 탐지하고 이를 인간이 이해할 수 있는 언어로 설명할 수 있는 잠재력이 있음을 입증하였다. 특히 GPT-4의 경우 복잡한 튜닝 없이 적절한 프롬프트만으로도 '활성화(Activate)'되어 높은 성능을 내는 Emergent Ability를 보여주었다.

**한계 및 비판적 해석**:

1. **인덱스 환각(Index Hallucination)**: LLM이 텍스트 기반으로 수치를 처리하기 때문에 발생하는 근본적인 문제이다. 시계열 길이가 길어질수록 정확한 위치를 짚어내는 능력이 급격히 저하된다.
2. **모델 크기의 영향**: LLaMA3-8B와 같은 소형 모델은 프롬프트 엔지니어링만으로는 성능 향상이 제한적이며, 반드시 데이터 기반의 미세 조정이 필요함을 확인하였다.
3. **데이터 표현의 한계**: 본 논문은 시계열을 순수 텍스트 토큰으로 처리하였다. 저자들은 이미지나 임베딩 방식이 인덱스 정확도를 더 떨어뜨린다고 언급하였으나, 향후 더 정교한 인코딩 방식에 대한 연구가 필요할 것으로 보인다.

## 📌 TL;DR

본 논문은 LLM을 시계열 이상치 탐지 및 설명 도구로 활용할 수 있는지 연구하였다. GPT-4는 프롬프트 전략을 통해 기존 통계/딥러닝 모델에 필적하는 성능을 보였으며, LLaMA3는 제안된 합성 데이터 생성기(TTGenerator)를 통한 미세 조정 후 성능이 향상되었다. 비록 인덱스 환각이라는 한계가 존재하지만, '설명 가능한 이상치 탐지'라는 새로운 가능성을 제시하며 향후 시계열 분석의 패러다임을 텍스트 기반 추론으로 확장할 수 있는 중요한 기초 연구가 될 것으로 평가된다.
