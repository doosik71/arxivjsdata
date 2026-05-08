# When LLM Meets Time Series: Can LLMs Perform Multi-Step Time Series Reasoning and Inference

Wen Ye, Jinbo Liu, Defu Cao, Wei Yang, Yan Liu (2025)

## 🧩 Problem to Solve

본 연구는 대규모 언어 모델(Large Language Models, LLMs)이 시계열 데이터(Time Series Data)에 대해 복잡한 추론 및 분석을 수행할 수 있는 '범용 시계열 AI 어시스턴트'로서 기능할 수 있는지를 평가하는 것을 목표로 한다.

에너지, 금융, 기후 과학, 헬스케어와 같은 실제 도메인에서 시계열 워크플로우는 단순히 값을 예측하는 것을 넘어, 다단계 추론(Multi-step reasoning), 정밀한 수치 계산, 도메인 지식의 통합, 그리고 운영 제약 조건(Operational constraints)의 준수를 요구하는 매우 복잡한 특성을 가진다. 그러나 기존의 LLM 평가 방식은 단일 작업에 집중하거나, 시계열 데이터 없이 단순한 질의응답(QA) 형태에 그치며, 실제 운영 제약 조건을 반영하지 못한다는 한계가 있다. 따라서 본 논문은 LLM이 실제 데이터 분석가처럼 복잡한 제약 조건이 포함된 분석 워크플로우를 스스로 구축하고 실행할 수 있는지 검증하기 위한 엄격한 벤치마크의 필요성을 제기한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 LLM을 시계열 AI 어시스턴트로 평가하기 위한 최초의 벤치마크인 **TSAIA(TimeSeries Artificial Intelligence Assistant) Benchmark**를 제안한 것이다.

TSAIA의 중심 아이디어는 실제 도메인의 실용성을 확보하고, 동적으로 확장 가능하며, 이질적인 작업 유형에 대해 통합된 평가 체계를 구축하는 것이다. 이를 위해 연구진은 20개 이상의 학술 문헌을 조사하여 33가지의 실제 작업 유형을 정의하고, 이를 4가지 상위 범주(Predictive, Diagnostic, Analytical, Decision-making)로 체계화하였다. 또한, 정적인 데이터셋이 아닌 '질문 생성기(Question Generator)'를 통해 다양한 제약 조건과 파라미터를 가진 작업 인스턴스를 동적으로 생성함으로써 벤치마크의 확장성을 확보하였다.

## 📎 Related Works

기존의 시계열 관련 벤치마크는 크게 세 그룹으로 나뉘며, 각각 다음과 같은 한계를 가진다.

1. **순수 QA 작업 (예: Test of Time, TRAM):** 논리적 추론 능력은 평가하지만, 실제 시계열 수치 데이터가 포함되지 않아 모델이 수치 신호를 처리하는 능력을 측정할 수 없다.
2. **단일 정적 분석 작업 (예: TSI-Bench, TSB-AD, GIFT-Eval):** 보간(Imputation), 이상치 탐지, 예측 등 특정 작업에만 집중하며, 고정된 데이터셋과 윈도우 크기를 사용하여 동적인 시나리오 대응 능력을 평가하기 어렵다. 또한 추론(Reasoning) 과정이 결여되어 있다.
3. **하이브리드 QA 및 분석 (예: MTBench, ChatTime):** 시계열 데이터와 텍스트 입력을 결합하고 추론 요소를 포함하지만, 여전히 설정이 고정적이며 주로 문맥 기반의 예측 작업에 치중되어 있다.

TSAIA는 이러한 기존 연구와 달리 **동적 생성(Dynamic)**, **시계열 데이터 포함(TS involved)**, **다단계 추론(Reasoning)**이라는 세 가지 요소를 모두 충족하는 최초의 벤치마크라는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 1. 작업 범주화 (Task Categorization)

TSAIA는 작업을 다음 네 가지 범주로 구분하여 모델의 다각적인 능력을 평가한다.

- **Predictive Tasks:** 공변량(Covariates) 유무에 따른 예측을 수행하며, 특히 최대/최소 부하 제한, 램프 속도(Ramp rate) 제약 등 실제 운영 제약 조건을 준수해야 한다.
- **Diagnostic Tasks:** 데이터 내의 비정상 패턴이나 잠재 구조를 식별한다. 기준 샘플을 이용한 이상치 탐지(Anomaly Detection)와 도메인 지식을 활용한 인과 관계 발견(Causal Discovery)이 포함된다.
- **Analytical Tasks:** 금융 도메인을 중심으로 리스크-수익 분석(Risk-return analysis)이나 과거 자산 가격을 이용한 트레이딩 전략 생성을 수행한다.
- **Decision-making Tasks:** 객관식 문제 형태로 제공되며, 포트폴리오 성능 지표를 분석하여 최적의 선택지를 고르거나 주식과 시장 지수의 성능을 비교하는 능력을 평가한다.

### 2. 질문 생성 파이프라인 (Question Generator)

작업 인스턴스는 다음의 5단계 프로세스를 통해 생성된다.
$$\text{Task Type Selection} \rightarrow \text{Data Source Selection} \rightarrow \text{Context Parameterization} \rightarrow \text{Adding Complexity} \rightarrow \text{Ground Truth Construction}$$
이 과정에서 입력 시퀀스 길이, 예측 호라이즌, 타겟 변수 등이 무작위로 설정되며, 도메인별 제약 조건이 추가되어 문제의 난이도가 조절된다.

### 3. 실행 및 평가 프레임워크

LLM의 수치 처리 한계를 극복하기 위해 **CodeAct** 에이전트 프레임워크를 도입하였다. LLM은 직접 정답을 내놓는 대신, 작업을 수행하기 위한 **Python 코드**를 생성한다.

- **추론 절차:** $\text{LLM} \rightarrow \text{Python 코드 생성} \rightarrow \text{코드 실행} \rightarrow \text{실행 결과 피드백} \rightarrow \text{코드 수정 및 최종 답변 도출}$
- **평가 프로토콜:** 3단계 검증 과정을 거친다.
    1. **구조적 정확성:** 출력 형식이 맞는지, 텐서의 형태(Shape)가 올바른지 확인한다.
    2. **제약 조건 준수:** 주입된 도메인 제약 조건 및 지식을 반영했는지 확인한다.
    3. **추론 품질 측정:** 정답(Ground Truth)과 비교하여 최종 메트릭을 계산한다.

### 4. 주요 평가지표

작업별로 서로 다른 지표를 사용하여 평가의 유효성을 높였다.

- **Predictive:** $\text{MAPE (Mean Absolute Percentage Error)}$
- **Diagnostic:** $\text{F1-score}$ 또는 $\text{Accuracy}$
- **Analytical:** $\text{Absolute Error}$ 또는 금융 지표($\text{CR: Cumulative Return, AR: Annualized Return, MDD: Maximum Drawdown}$)
- **공통 지표:** $\text{Success Rate}$ (성공률). 이는 형식이 맞고 제약 조건을 만족하며, 성능 메트릭이 임계값(예: $\text{MAPE} < 1$)을 통과한 비율이다.

## 📊 Results

### 1. 실험 설정

GPT-4o, Qwen2.5-Max, Llama-3.1 70B, Claude-3.5 Sonnet, DeepSeek, Gemini-2.0, Codestral, DeepSeek-R 등 8가지 SOTA 모델을 대상으로 평가를 수행하였다. 모든 모델은 $\text{temperature}=0$으로 설정하여 결정론적인 출력을 유도하였다.

### 2. 주요 결과

- **예측 작업 (Predictive):** 단순한 최대/최소 부하 제약 조건은 잘 처리하지만, 램프 속도나 변동성 제어와 같이 시간적 매끄러움(Temporal smoothness)을 요구하는 작업에서는 성공률이 급격히 떨어진다. 다중 그리드(Multiple grids) 예측 시 데이터 차원이 증가함에 따라 모든 모델의 성능이 하락하였다.
- **진단 작업 (Diagnostic):** 사전 지식이 주어진 인과 발견 작업에서는 높은 성공률을 보였으나, 참조 샘플을 통해 임계값을 스스로 캘리브레이션(Calibration)해야 하는 이상치 탐지 작업에서는 매우 취약한 모습을 보였다. 많은 모델이 모든 값을 0으로 예측하는 '사소한 예측(Trivial prediction)' 오류를 범했다.
- **분석 및 의사결정 작업 (Analytical & Decision-making):** 금융 지표의 공식이 단순할수록 성능이 높았으며, 복잡한 전략 수립이나 시장 비교 분석에서는 대부분의 모델이 무작위 선택(Random chance) 수준의 정확도를 보였다. 단, **DeepSeek-R**은 의사결정 작업에서 유일하게 일관되게 무작위 수준 이상의 성능을 기록하였다.

### 3. 모델별 특성 분석

- **DeepSeek-R:** 가장 높은 추론 성능을 보였으나, 정답에 도달하기까지 가장 많은 상호작용 턴(Turn)과 토큰을 사용하였다. 이는 더 끈기 있고 탐색적인 문제 해결 전략을 사용함을 시사한다.
- **GPT-4o 및 DeepSeek-Chat:** 토큰 효율성 측면에서 가장 우수한 모습을 보였다.
- **Gemini-2.0 및 Codestral:** 대부분의 작업 범주에서 실행 에러(Execution Error) 빈도가 매우 높아, 구조화된 다단계 시계열 추론에 부적합한 것으로 나타났다.

## 🧠 Insights & Discussion

### 1. 구성적 추론(Compositional Reasoning)의 부재

본 연구의 가장 중요한 발견은 현재의 LLM들이 복잡한 분석 워크플로우를 스스로 조립하는 **구성적 추론 능력**이 부족하다는 점이다. 특히 참조 샘플을 이용해 임계값을 설정하고 이를 다시 탐지에 적용하는 다단계 프로세스를 구축하지 못하고, 단순한 패턴 매칭에 의존하는 경향이 강하다.

### 2. 오류 분석을 통한 한계점

GPT-4o의 사례 분석 결과, 작업의 난이도가 높아질수록 오류의 양상이 다양해진다.

- 단순 예측 $\rightarrow$ 실행 에러(Execution Error) 중심
- 다중 시계열 제약 예측 $\rightarrow$ 제약 조건 위반(Constraint Violation) 중심
- 참조 샘플 기반 진단 $\rightarrow$ 사소한 예측(Trivial Prediction) 중심
이는 모델이 외부 문맥을 통합하고 추상적인 추론을 수행하는 능력이 부족함을 보여준다.

### 3. 도메인 특화의 필요성

범용 모델들이 금융이나 에너지와 같은 특수 도메인의 고유한 제약 조건과 수치적 특성을 처리하는 데 어려움을 겪는 것을 통해, 단순한 일반 성능 향상보다는 **도메인 특화(Domain specialization)** 및 심볼릭 추론(Symbolic reasoning)과의 결합이 필수적임을 시사한다.

## 📌 TL;DR

본 논문은 LLM이 실제 시계열 데이터 분석가처럼 동작할 수 있는지 평가하기 위해, 동적 생성 가능하고 다단계 추론을 요구하는 벤치마크인 **TSAIA**를 제안하였다. 8종의 SOTA 모델을 평가한 결과, 대부분의 모델이 단순 수치 계산은 가능하지만 **복잡한 분석 워크플로우를 설계하고 제약 조건을 정밀하게 준수하는 구성적 추론 능력은 현저히 떨어진다**는 것을 확인하였다. 특히 DeepSeek-R이 추론 능력에서 강점을 보였으나 비용(토큰/턴)이 높았으며, 전반적으로 시계열 AI 어시스턴트를 구현하기 위해서는 실행 피드백과 도메인 지식이 밀접하게 통합된 하이브리드 접근 방식이 필요함을 시사한다.
