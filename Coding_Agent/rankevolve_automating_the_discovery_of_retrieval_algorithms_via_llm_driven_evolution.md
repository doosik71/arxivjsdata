# RankEvolve: Automating the Discovery of Retrieval Algorithms via LLM-Driven Evolution

Jinming Nian, Fangchen Li, Dae Hoon Park, Yi Fang (2026)

## 🧩 Problem to Solve

전통적인 Lexical Retrieval 알고리즘인 BM25나 Query Likelihood(QL) 등은 여전히 효율적인 1단계 랭커(first-stage ranker)로 사용되고 있다. 그러나 이러한 알고리즘의 성능 개선은 주로 하이퍼파라미터 튜닝이나 개별 스코어링 컴포넌트에 대한 인간의 직관에 의존해 왔다. 즉, 새로운 랭킹 함수를 설계하는 과정이 체계적인 자동화보다는 수동적인 시행착오에 치우쳐 있다는 점이 문제이다.

본 논문의 목표는 대규모 언어 모델(LLM)이 평가자(evaluator)와 진화적 탐색(evolutionary search)의 가이드를 받아, 인간의 개입 없이도 성능이 향상된 새로운 Lexical Retrieval 알고리즘을 자동으로 발견할 수 있는지 탐구하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 랭킹 알고리즘을 실행 가능한 Python 코드로 표현하고, LLM을 돌연변이 연산자(mutation operator)로 사용하여 알고리즘을 반복적으로 진화시키는 것이다.

1. **LLM 기반 프로그램 진화 프레임워크 제안**: RankEvolve라는 시스템을 통해 검색 알고리즘 전체를 LLM이 직접 수정하고 최적화하는 자동화된 발견 경로를 제시하였다.
2. **시드 프로그램 구조의 영향 분석**: 알고리즘의 구조적 자유도(structural freedom)가 진화 결과에 미치는 영향을 분석하여, 제약이 적은 Freeform 구조가 더 높은 성능의 알고리즘을 발견함을 입증하였다.
3. **새로운 스코어링 모티프 발견**: 진화된 알고리즘들이 기존 문헌에 명시되지 않은 독창적인 스코어링 메커니즘을 생성하며, 이것이 학습에 사용되지 않은 외부 데이터셋에서도 일반화되는 성능 향상을 보임을 확인하였다.

## 📎 Related Works

기존에 랭킹 함수를 자동으로 발견하려는 시도는 주로 유전 프로그래밍(Genetic Programming, GP)을 통해 이루어졌다. ARRANGER와 같은 프레임워크는 산술 연산자($+, \times, \log$)를 조합하여 TF-IDF 특징들을 연결하는 방식으로 함수를 진화시켰다. 그러나 고전적 GP는 표현식 트리(expression tree)를 무작위로 교체하는 방식이므로, 해당 식이 실제로 무엇을 계산하는지에 대한 의미론적 이해 없이 탐색을 수행한다는 한계가 있다.

반면 RankEvolve는 LLM을 돌연변이 연산자로 사용함으로써, 코드의 의미를 이해하고 추론에 기반한 수정(reason-informed edits)을 수행할 수 있다. 예를 들어, 쿼리의 문서 커버리지 신호가 부족하다는 점을 인지하고 이를 보완하는 코드를 추가하는 식의 고차원적 탐색이 가능하다. 또한 Learning-to-Rank(LTR) 방식이 기존 특징들의 가중치를 학습하는 것과 달리, RankEvolve는 완전히 새로운 특징과 스코어링 함수 자체를 제안한다는 점에서 차별화된다.

## 🛠️ Methodology

### 전체 파이프라인

RankEvolve는 LLM 가이드 하에 프로그램 합성(program synthesis)을 수행하는 진화적 탐색 구조를 가진다. 전체 시스템은 다음 네 가지 구성 요소로 이루어져 있다.

1. **Search Space (시드 프로그램 및 시스템 프롬프트)**: LLM이 수정 가능한 코드 영역(interface)을 정의한다. BM25 시드의 경우 문서 표현, 쿼리 표현, 스코어링 함수 세 부분으로 분해하여 자유도를 높였다.
2. **Population Management**: 섬 기반 진화(island-based evolution)와 MAP-Elites를 결합하여 다양성을 유지한다. 인구 집단을 $K$개의 독립적인 섬으로 나누고, 각 섬 내부에서는 코드 길이(complexity)와 편집 거리(diversity)라는 두 가지 차원의 그리드에 프로그램을 배치하여 지역 최적점(local minima)에 빠지는 것을 방지한다.
3. **Mutation Proposal**: 부모 프로그램을 선택(탐색, 활용, 가중치 기반 샘플링 중 하나)하고, 이를 LLM에게 제공하여 `SEARCH/REPLACE` diff 형식의 수정을 제안하게 한다. 이때 상위 성능 프로그램들과 이전 시도 기록들이 프롬프트에 포함된다.
4. **Evaluator**: 제안된 프로그램을 실행하여 12개의 데이터셋에서 성능을 측정한다.

### 최적화 목표 및 손실 함수

본 연구에서는 별도의 학습 가능한 파라미터가 없으므로 손실 함수 대신 fitness score를 최적화 목표로 설정한다. 최적화 타겟은 다음과 같이 정의된다.

$$\text{Fitness Score} = 0.8 \times \text{Avg Recall@100} + 0.2 \times \text{Avg nDCG@10}$$

이 가중치는 1단계 검색(first-stage retrieval)의 목적이 하위 리랭커(reranker)를 위해 관련 문서의 재현율(Recall)을 극대화하는 것에 있음을 반영하며, nDCG@10은 보조적인 랭킹 품질 신호로 사용된다.

### 진화된 알고리즘의 상세 구조

#### 1. Evolved BM25

최종 진화된 BM25는 4개의 병렬 토큰 공간(Base, Prefix, Bigram, Micro)을 사용하는 멀티 채널 스코어링 구조를 가진다.

- **Core Scoring Function**:
$$R(q, d) = \ln(1 + \mathcal{E}) \cdot \mathcal{B}_{cov} \cdot \mathcal{B}_{spec} \cdot \mathcal{B}_{coord} \cdot \mathcal{B}_{anc} / \mathcal{B}_{len}$$
여기서 $\mathcal{E}$는 가중치 $\omega(t)$가 적용된 $\log\text{-TF}$의 합이며, $\mathcal{B}$ 시리즈는 커버리지, 특이성(PMI 기반), 조정(coordination), 앵커(희귀 단어 보너스), 길이 감쇠(로그 형태)를 조절하는 multiplier들이다.
- **특이점**: $\omega(t)$ 함수를 통해 불용어(stopword)를 자동으로 억제하는 소프트 필터를 학습하였으며, BM25의 선형 길이 정규화를 더 완만한 로그 형태로 변경하였다.

#### 2. Evolved Query Likelihood (QL)

QL 시드에서 진화된 알고리즘은 가산적(additive) 구조를 유지하면서도 다음과 같은 고도화된 메커니즘을 도입하였다.

- **Enriched Collection LM**: 배경 언어 모델의 분포를 평탄화($\tau=0.85$ 제곱)하고 문서 빈도 모델과 보간(interpolation)하여 희귀 단어의 변별력을 높였다.
- **Adaptive TF Saturation**: IDF에 따라 TF 포화도를 다르게 적용한다. 일반 단어는 강하게 포화시키고($\beta \approx 0.7$), 희귀 단어는 신호를 보존한다($\beta \approx 1.0$).
- **Penalty Architecture**: Leaky Rectifier를 통해 음수 점수를 12% 강도로 유지하고, 문서에 완전히 누락된 단어에 대해 명시적인 Penalty를 부여하는 계층적 구조를 설계하였다.

## 📊 Results

### 실험 설정

- **데이터셋**: BEIR, BRIGHT, TREC DL 19/20 등 총 28개 데이터셋 사용. 진화 과정에는 12개만 사용하고 나머지 16개는 일반화 성능 측정(held-out)을 위해 사용하였다.
- **기준선**: BM25, BM25+, BM25-adapt, QL-Dir, QL-JM.
- **LLM**: GPT-5.2 (API) 사용.

### 정량적 결과

표 1에 따르면, 진화된 $\text{BM25}^\star$와 $\text{QL-Dir}^\star$는 모든 벤치마크에서 기존 시드 및 변형 알고리즘보다 우수한 성능을 보였다. 특히 학습에 사용되지 않은 16개 데이터셋에서도 유의미한 성능 향상이 관찰되어, 단순한 오버피팅이 아닌 일반적인 검색 알고리즘의 개선이 이루어졌음을 입증하였다.

### 구조적 자유도에 따른 Ablation Study

시드 프로그램의 제약 수준을 세 단계로 나누어 실험한 결과(표 2), 자유도가 높을수록 성능이 단조 증가하였다.

- **Constrained**: 하이퍼파라미터 튜닝 수준 $\to$ 가장 낮은 성능 향상.
- **Composable**: 모듈형 프리미티브 수정 $\to$ 중간 성능 향상.
- **Freeform**: 전체 파이프라인 구조 변경 가능 $\to$ 가장 높은 성능 및 일반화 능력 확보.

### 지연 시간(Latency) 분석

알고리즘이 복잡해짐에 따라 쿼리 처리 시간은 증가하였다. $\text{BM25}^\star$는 기본 BM25보다 약 11배 느려졌는데, 이는 LLM이 효율성 제약 없이 성능(Recall/nDCG) 극대화에만 집중했기 때문이다.

## 🧠 Insights & Discussion

### 수렴된 원칙 (Convergent Principles)

가장 흥미로운 점은 BM25와 QL이라는 완전히 다른 시작점(시드)에서 출발했음에도 불구하고, 두 알고리즘이 유사한 고수준 전략으로 수렴했다는 것이다.

- 두 경로 모두 **TF 포화(TF saturation), 소프트 불용어 필터링, 명시적 조정 메커니즘, 완만한 길이 정규화**를 독립적으로 발견하였다.
- 다만 이를 구현하는 방식은 달랐다(BM25는 곱셈 기반 모듈레이션, QL은 가산 기반 패널티). 이는 이러한 요소들이 Lexical Retrieval의 성능 향상을 위한 근본적인 요구사항임을 시사한다.

### 구조적 자유도의 중요성

Ablation 연구를 통해 시드 프로그램의 설계가 발견 가능한 알고리즘의 상한선(upper bound)을 결정한다는 것을 확인하였다. 제약이 많은 시드는 기존 공식 근처의 지역 최적점에 머물게 하지만, Freeform 구조는 비직관적인 구조적 개선을 가능하게 한다.

### 한계 및 비판적 해석

본 연구의 결과물은 성능은 뛰어나지만 코드가 매우 복잡하고 실행 시간이 느리다는 치명적인 단점이 있다. 이는 실제 서비스 환경에 적용하기 위해서는 지연 시간(latency)을 최적화 목표에 포함하는 추가적인 연구가 필수적임을 의미한다. 또한, LLM이 학습 데이터에 존재하던 기존 IR 논문의 아이디어들을 파라메트릭 지식 형태로 꺼내어 조합했을 가능성이 높으며, 이것이 순수한 '발견'인지 '재조합'인지에 대한 논의가 필요하다.

## 📌 TL;DR

본 논문은 LLM을 돌연변이 연산자로 활용하여 새로운 검색 알고리즘을 자동으로 생성하는 **RankEvolve** 프레임워크를 제안하였다. BM25와 QL 시드로부터 진화된 알고리즘들은 기존의 인간 설계 알고리즘보다 우수한 성능을 보였으며, 특히 서로 다른 시드에서도 유사한 최적화 원칙으로 수렴하는 경향을 보였다. 이 연구는 LLM 기반의 프로그램 진화가 정보 검색(IR) 분야의 알고리즘 자동 발견을 위한 실용적인 경로가 될 수 있음을 시사한다.
