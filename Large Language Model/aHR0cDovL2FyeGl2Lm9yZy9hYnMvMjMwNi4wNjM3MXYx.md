# A Comprehensive Review of State-of-The-Art Methods for Java Code Generation from Natural Language Text

Jessica López Espejel, Mahaman Sanoussi Yahaya Alassan, El Mehdi Chouham, Walid Dahhane, El Hassane Ettifouri (2023)

## 🧩 Problem to Solve

본 논문은 자연어(Natural Language, NL) 텍스트를 Java 소스 코드로 자동 변환하는 **Java Code Generation** 작업에 대한 최신 기술들을 종합적으로 분석하고 검토하는 것을 목표로 한다.

자연어 기반의 코드 생성은 프로그래머가 단순하고 반복적인 작업에서 벗어나 더 복잡하고 창의적인 설계에 집중할 수 있게 함으로써 소프트웨어 엔지니어링의 생산성을 크게 향상시킬 수 있다는 점에서 매우 중요하다. 그러나 이 작업은 다음과 같은 기술적 난제들을 포함하고 있다.

1. **엄격한 구문 규칙**: 프로그래밍 언어는 자연어와 달리 매우 엄격한 문법적, 어휘적 제약 조건을 가진다.
2. **심층적 의미 이해**: 자연어 지시문의 의도를 정확히 파악하여 이를 실행 가능한 논리 구조로 변환하는 깊은 수준의 의미론적(Semantic) 이해가 필요하다.
3. **낮은 오류 허용치**: 점 하나, 콜론 하나와 같은 아주 작은 실수만으로도 코드의 의미가 완전히 변하거나 컴파일 오류가 발생하여 결과물이 완전히 잘못될 수 있다.

## ✨ Key Contributions

본 논문은 Java 코드 생성 분야의 발전 과정을 체계적으로 정리한 첫 번째 종합 리뷰 논문으로서 다음과 같은 핵심적인 기여를 한다.

- **모델 아키텍처의 분류 및 분석**: RNN 기반 모델부터 최신 Transformer 기반 모델(Encoder-only, Decoder-only, Encoder-Decoder)까지의 진화 과정을 분류하고 각 모델의 장단점을 분석하였다.
- **학습 목표(Learning Objectives)의 체계화**: Masked Language Modeling(MLM), Replaced Token Detection(RTD), Unidirectional Language Modeling(ULM) 등 코드 생성 모델에서 사용되는 다양한 손실 함수와 학습 목표를 수학적으로 정의하고 설명하였다.
- **데이터셋 및 평가 지표 정리**: CONCODE, CodeSearchNet 등 주요 Java 데이터셋의 특성과 EM, BLEU, CodeBLEU와 같은 평가 지표의 한계 및 유용성을 상세히 기술하였다.
- **실험적 비교 분석**: CONCODE 데이터셋을 기준으로 다양한 SOTA(State-of-the-art) 모델들의 성능을 정량적으로 비교하여, 사전 학습(Pre-training)의 중요성과 아키텍처별 성능 차이를 입증하였다.

## 📎 Related Works

논문에서는 딥러닝 이전의 초기 접근 방식들을 다음과 같이 소개하며, 이들이 현대의 복잡한 Java 코드 생성 작업을 수행하기에는 한계가 있음을 명시한다.

1. **Regular Expressions**: 자연어를 정규 표현식으로 매핑하는 규칙 기반 기법들이 사용되었으나, 복잡한 쿼리를 처리하는 데 한계가 있었다.
2. **Logical Forms**: 확률적 범주 문법(PCGs)을 사용하여 자연어를 논리 형태로 변환하였으나, 파서의 복잡도로 인해 대규모 데이터셋으로 확장하기 어려웠다.
3. **Agent-specific Language & Sequence of Instructions**: 특정 환경의 명령어나 액션 시퀀스로 변환하는 방식이 제안되었으나, 모호한 지시문을 처리하는 능력이 부족하였다.
4. **Semantic Parsing & SQL Generation**: 자연어를 의미론적으로 분석하여 SQL 쿼리를 생성하는 방식이 연구되었으며, 이는 현대의 코드 생성 작업과 가장 유사한 초기 형태이다.

이러한 기존 방식들은 주로 규칙 기반(Rule-based)이거나 단순 통계 방식에 의존하였기에, Java와 같이 복잡한 구조를 가진 언어의 전체 프로그램을 생성하는 데는 부적합하며, 이를 극복하기 위해 RNN과 Transformer 기반의 딥러닝 모델로 패러다임이 전환되었다.

## 🛠️ Methodology

본 논문은 개별 모델을 제안하는 것이 아니라, 기존 모델들의 방법론을 분석하는 리뷰 논문이다. 분석 대상이 된 모델들은 크게 세 가지 아키텍처로 구분된다.

### 1. RNN-based Methods

초기에는 LSTM, GRU와 같은 순환 신경망과 Seq2Seq 구조가 사용되었다. 입력을 고정 길이 벡터로 인코딩하고 디코더가 토큰을 예측하는 방식이다. 하지만 기울기 소실/폭주(Vanishing/Exploding Gradient) 문제와 긴 시퀀스 처리의 한계로 인해 현재는 Transformer 모델로 대체되었다.

### 2. Transformer-based Methods

Transformer 모델은 사전 학습(Pre-training) 후 미세 조정(Fine-tuning) 단계를 거치며, 아키텍처에 따라 다음과 같이 나뉜다.

#### (1) Encoder-only Models (예: CodeBERT, GraphCodeBERT)

자연어 $\text{NL}$과 소스 코드 $\text{PL}$을 결합하여 입력 $\text{x} = \{\text{NL, PL}\}$로 사용한다.

- **MLM (Masked Language Modeling)**: 무작위로 마스킹된 토큰을 예측한다.
    $$\mathcal{L}_{MLM}(\theta) = -\sum_{i \in m_{nl} \cup m_{pl}} \log P_{\theta}^{1}(\text{x}_i | \text{NL}_{masked}, \text{PL}_{masked})$$
- **RTD (Replaced Token Detection)**: 생성기가 교체한 토큰이 원래 토큰인지 아닌지를 판별한다.

#### (2) Decoder-only Models (예: GPT-2, GPT-3/4, LLaMA, Chinchilla)

이전 토큰들을 기반으로 다음 토큰을 예측하는 단방향 언어 모델링(ULM)을 수행한다.

- **ULM (Unidirectional Language Modeling)**:
    $$\mathcal{L}_{ULM}(\theta) = -\sum_{i=0}^{k-1} \log P_{\theta}(\text{x}_i | \text{x}_{<i})$$
- **PAR (Parametric Loss Function)**: Chinchilla 모델에서 사용되며, 모델 파라미터 수와 학습 토큰 수에 따른 성능 갭을 최적화하는 매개변수 손실 함수를 사용한다.

#### (3) Encoder-Decoder Models (예: PLBART, CodeT5, StructCoder)

인코더가 NL을 처리하고 디코더가 PL을 생성하는 구조이다.

- **MSP (Masked Span Prediction)**: 임의 길이의 텍스트 구간(Span)을 마스킹하고 이를 예측한다.
    $$\mathcal{L}_{MSP}(\theta) = \sum_{i=0}^{k-1} -\log P_{\theta}(\text{NL}_{masked, i} | \text{NL}_{\setminus masked}, \text{NL}_{masked, <i})$$
- **Identifier-aware Objectives**: CodeT5 등에서 사용하며, 식별자 태깅(IT) 및 식별자 예측(MIP)을 통해 코드의 의미적 특징을 강화한다.

### 3. 구조적 정보 활용 (AST & DFG)

최신 모델들은 단순 텍스트가 아닌 코드의 구조적 정보를 활용한다.

- **AST (Abstract Syntax Tree)**: 코드의 계층적 구조를 그래프 형태로 반영한다.
- **DFG (Data Flow Graph)**: 변수 간의 데이터 흐름을 추적하여 semantic 정보를 보강한다 (예: GraphCodeBERT).

## 📊 Results

### 실험 설정

- **데이터셋**: CONCODE (Github에서 수집된 JavaDoc-코드 쌍)
- **평가 지표**:
  - **Exact Match (EM)**: 정답과 완전히 일치하는지 측정 (매우 엄격함).
  - **BLEU**: n-gram 중첩도를 측정 (의미론적 분석 부족).
  - **CodeBLEU**: BLEU에 AST 및 DFG 일치도를 추가하여 코드 특성을 반영 (가장 정확함).

### 주요 결과 (Table 4 기준)

- **아키텍처별 성능**: RNN 기반 모델 $\ll$ Transformer 기반 모델 순으로 성능이 높게 나타났다.
- **최상위 모델**:
  - **BLEU 및 CodeBLEU**: `CodeT5-large`가 가장 높은 성능을 보였다. 이는 방대한 파라미터 수(770M)에 기인한 것으로 분석된다.
  - **Exact Match (EM)**: `REDCODER`가 가장 높은 점수를 기록하였다. 이는 검색 모듈(SCODER-R)과 생성 모듈(SCODER-G)을 결합한 Retrieval-Augmented 방식의 효율성을 보여준다.
- **사전 학습의 효과**: `CodeGPT-adapted`(사전 학습 가중치 사용)가 `CodeGPT-2`(처음부터 학습)보다 월등히 높은 성능을 보였으며, 이는 사전 학습된 표현(Representation)이 다운스트림 작업의 학습을 크게 돕는다는 것을 입증한다.

## 🧠 Insights & Discussion

### 강점 및 발견

- **구조적 정보의 중요성**: AST와 DFG를 학습 목표에 통합한 모델(CodeT5, StructCoder)들이 단순 텍스트 기반 모델보다 우수한 성능을 보였다.
- **모델 크기와 성능의 트레이드오프**: `CodeT5-large`가 성능은 가장 좋으나 메모리 및 학습 시간 비용이 매우 크다. 반면 `REDCODER`는 상대적으로 적은 파라미터로도 높은 EM 점수를 기록하여 효율적인 대안이 될 수 있음을 보여주었다.

### 한계 및 비판적 해석

- **평가 지표의 한계**: 현재 사용되는 EM, BLEU, CodeBLEU 모두 구문적 유사성(Syntactic Similarity)에 크게 의존한다. 서로 다른 구문을 가졌더라도 동일한 기능을 수행하는 코드가 존재할 수 있는데, 현재의 지표로는 이를 완전히 평가할 수 없다.
- **LLM의 벤치마크 부재**: ChatGPT, LLaMA와 같은 거대 언어 모델(LLM)들이 뛰어난 성능을 보인다고 알려져 있으나, 정작 Java 코드 생성의 표준 벤치마크인 CONCODE 데이터셋에서의 정량적 결과는 아직 충분히 보고되지 않았다.

## 📌 TL;DR

본 논문은 자연어로부터 Java 코드를 생성하는 딥러닝 모델들의 변천사를 종합적으로 분석한 리뷰 보고서이다. 분석 결과, **RNN보다는 Transformer 기반 모델이 압도적**이며, 특히 **Encoder-Decoder 구조(예: CodeT5)와 Retrieval-Augmented 구조(예: REDCODER)**가 현재 SOTA 성능을 달성하고 있다. 또한, 모델의 단순 크기 확장보다는 **AST/DFG와 같은 코드 구조 정보의 활용**과 **효율적인 사전 학습 전략**이 성능 향상의 핵심임을 밝히고 있다. 향후 연구는 구문적 일치도를 넘어선 **의미론적 평가 지표의 개발**과 거대 모델의 **경량화 및 효율적 학습** 방향으로 나아가야 한다.
