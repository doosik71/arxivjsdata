# ARGOS: Agentic Time-Series Anomaly Detection with Autonomous Rule Generation via Large Language Models

Yile Gu, Yifan Xiong, Jonathan Mace, Yuting Jiang, Yigong Hu, Baris Kasikci, and Peng Cheng (2025)

## 🧩 Problem to Solve

클라우드 인프라의 관측 가능성(Observability)을 확보하기 위해 시계열 데이터의 이상 탐지(Anomaly Detection) 시스템은 필수적이다. 하지만 기존의 시스템들은 실제 운영 환경에서 반드시 갖춰야 할 세 가지 핵심 속성인 설명 가능성(Explainability), 재현성(Reproducibility), 그리고 자율성(Autonomy)을 동시에 만족시키지 못하는 한계가 있다.

설명 가능성은 장애 발생 시 엔지니어가 원인을 즉각 파악할 수 있게 하며, 재현성은 동일한 데이터에 대해 항상 동일한 결과를 보장하여 불필요한 리소스 낭비를 방지한다. 자율성은 데이터 분포가 변하는 동적인 클라우드 환경에서 인간의 개입 없이 시스템이 스스로 업데이트되는 능력을 의미한다. 본 논문의 목표는 LLM(Large Language Model)을 활용하여 이 세 가지 속성을 모두 충족하는 자율적인 시계열 이상 탐지 시스템을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 LLM을 런타임의 탐지기로 사용하는 것이 아니라, **훈련 단계에서 설명 가능하고 재현 가능한 '탐지 규칙(Detection Rules)'을 생성하는 도구로 사용하는 것**이다.

중심적인 설계 직관은 구조화된 탐지 규칙을 중간 표현체(Intermediate Representation)로 활용하는 것이다. 규칙을 실행 가능한 Python 코드로 작성하면, 결과적으로 런타임에서는 LLM 없이 결정론적인 코드만 실행하므로 재현성과 설명 가능성을 확보할 수 있으며, 규칙 생성 과정은 LLM의 에이전트 기반 파이프라인에 맡김으로써 자율성을 달성한다.

## 📎 Related Works

기존의 시계열 이상 탐지 접근 방식은 다음과 같은 한계가 존재한다.

1. **딥러닝 기반 방법 (DL-based):** 높은 정확도를 보이지만 블랙박스 모델 특성상 설명 가능성이 부족하며, 하이퍼파라미터 튜닝에 많은 수작업이 필요하여 자율성이 낮다.
2. **LLM 기반 방법 (LLM-based):** 자연어 설명을 통해 설명 가능성과 자율성을 높였으나, LLM 고유의 확률적 특성(Stochastic nature)으로 인해 동일한 입력에 대해서도 결과가 달라지는 재현성 결여 문제가 심각하다.
3. **규칙 기반 방법 (Rule-based):** 명시적인 로직을 사용하여 설명 가능성과 재현성이 매우 뛰어나며 산업계에서 널리 쓰이지만, 새로운 패턴의 이상치를 탐지하기 위해 엔지니어가 수동으로 규칙을 작성하고 임계값(Threshold)을 조정해야 하므로 자율성이 없다.

ARGOS는 LLM의 코드 생성 능력과 규칙 기반 방법의 결정론적 특성을 결합하여 이러한 트레이드-오프를 해결한다.

## 🛠️ Methodology

### 전체 시스템 구조

ARGOS는 데이터 전처리(Data Preprocessing), 규칙 훈련(Rule Training), 배포(Deployment)의 세 단계로 구성된다.

### 훈련 엔진 (Training Engine) 및 에이전트 파이프라인

LLM이 생성한 코드의 구문 오류와 정확도 문제를 해결하기 위해 세 가지 에이전트가 협력하는 피드백 루프를 구축한다.

1. **Detection Agent:** 입력 데이터와 레이블을 바탕으로 이상 징후를 정의하는 자연어 규칙을 먼저 작성하고, 이를 Python 함수 `inference(sample: np.ndarray) -> np.ndarray` 형태로 구현한다.
2. **Repair Agent:** 생성된 코드를 더미 데이터로 실행하여 구문 오류(Syntax Error)를 체크한다. 오류 발생 시 스택 트레이스와 에러 메시지를 전달받아 코드를 수정한다.
3. **Review Agent:** 검증 데이터셋을 통해 규칙의 정확도를 평가한다. 이전 버전보다 성능이 하락한 경우, 오답 샘플과 코드 차이점을 분석하여 성능 저하가 없을 때까지 규칙을 개선한다.

### 모델 융합 및 정확도 보장 (Model Fusion)

LLM이 생성한 규칙이 기존의 성숙한 프로덕션 시스템보다 성능이 낮을 가능성을 방지하기 위해, 기존 탐지기(Base Detector)와 LLM 규칙을 결합하는 융합 방식을 제안한다.

- **False Negative (FN) 규칙:** Base Detector가 정상으로 오분류한 이상치 샘플을 학습하여 탐지 능력을 보완한다.
- **False Positive (FP) 규칙:** Base Detector가 이상으로 오분류한 정상 샘플을 학습하여 오탐을 줄인다.

최종 예측 결과는 다음과 같은 **Aggregation 알고리즘**을 통해 결정된다.
$$
L_{agg}[t] =
\begin{cases}
\text{abnormal}, & \text{if } L_{base}[t] = \text{normal and } L_{fn}[t] = \text{abnormal} \\
\text{normal}, & \text{if } L_{base}[t] = \text{abnormal and } L_{fp}[t] = \text{normal} \\
L_{base}[t], & \text{otherwise}
\end{cases}
$$

### 효율성 향상 (Efficiency Enhancement)

LLM의 생성 비용을 줄이기 위해 **Top-k selection** 전략을 사용한다. 매 반복마다 $n$개의 규칙 후보를 생성하고, 그 중 검증 성능이 가장 좋은 $k$개의 규칙만을 선택하여 다음 반복 단계의 정제 과정으로 전달함으로써 수렴 속도를 높인다.

## 📊 Results

### 실험 설정

- **데이터셋:** KPI, Yahoo, 그리고 Microsoft 내부 데이터셋(Internal)을 사용한다.
- **평가 지표:** 시계열 특성을 고려한 $\text{Event-}F_1^{PA}$ (Event-Based $F_1$ score with Point Adjustment)를 주 지표로 사용한다.
- **비교 대상:** AnomalyTransformer, FCVAE, LSTMAD, TFAD 등의 DL 모델과 LLMAD, SigLLM 등의 LLM 기반 모델을 baseline으로 설정한다.

### 주요 결과

1. **정확도 향상:** 모든 데이터셋에서 SOTA 방법론보다 높은 성능을 보였다. 특히 내부 데이터셋에서는 $F_1$ 스코어를 최대 $28.3\%$ 향상시켰으며, KPI($+9.5\%$), Yahoo($+4.8\%$) 데이터셋에서도 우수한 성적을 거두었다.
2. **재현성 및 정확도 보장:** Aggregator를 사용하지 않았을 때는 일부 메트릭에서 성능 하락이 발생했으나, 모델 융합(Model Fusion)을 적용한 결과 모든 메트릭에서 기존 베이스라인 대비 성능 저하가 없음을 확인하였다.
3. **추론 속도:** 런타임에 무거운 DL 모델이나 LLM API를 호출하지 않고 경량 Python 코드만 실행하므로 추론 속도가 비약적으로 향상되었다. (KPI $3.0\times$, Yahoo $34.3\times$, Internal $1.5\times$ 가속)
4. **훈련 비용:** GPT-4o를 사용한 규칙 훈련 비용은 데이터셋당 50회 반복 기준으로 총 10달러 미만으로 매우 경제적이다.

## 🧠 Insights & Discussion

ARGOS는 LLM을 '추론기'가 아닌 '컴파일러/최적화 도구'로 재정의함으로써 LLM의 고질적인 문제인 비결정론적 특성(Non-determinism)을 완전히 제거하였다.

특히 주목할 점은 **Model Fusion** 전략이다. 무조건 LLM의 규칙만 믿는 것이 아니라, 기존 시스템의 오류(FN, FP)만을 집중적으로 학습하여 보완하는 방식을 통해 "최소한 기존 시스템보다는 성능이 좋음"을 보장하는 안전장치를 마련한 점이 실무적인 관점에서 매우 강력한 강점이다.

다만, 본 연구에서는 훈련 단계에서 사용되는 샘플의 분포가 런타임 데이터의 분포를 충분히 대표한다는 가정을 전제로 한다. 만약 런타임에 완전히 새로운 형태의 이상 패턴이 등장한다면, 다시 훈련 엔진을 가동하여 규칙을 업데이트해야 하는 주기적인 재학습 비용이 발생할 수 있다.

## 📌 TL;DR

ARGOS는 LLM 기반의 에이전트 파이프라인을 통해 시계열 이상 탐지를 위한 **실행 가능한 Python 규칙을 자율적으로 생성**하는 시스템이다. 런타임에는 생성된 결정론적 규칙만을 실행하여 **설명 가능성, 재현성, 자율성**을 동시에 확보하였으며, 기존 탐지기와의 모델 융합을 통해 정확도 하락 없는 성능 향상을 달성하였다. 이 연구는 LLM을 이용해 전통적인 규칙 기반 시스템의 유지보수 문제를 해결하고, 실제 운영 환경에 즉시 적용 가능한 고효율 탐지 체계를 구축했다는 점에서 큰 의의가 있다.
