# Emissions and Performance Trade-off Between Small and Large Language Models

Anandita Garg, Uma Gaba, Deepan Muthirayan, Anish Roy Chowdhury (2025)

## 🧩 Problem to Solve

본 논문은 거대 언어 모델(Large Language Models, LLMs)의 확산으로 인해 발생하는 막대한 탄소 발자국(Carbon Footprint) 문제를 해결하고자 한다. LLM은 에너지 집약적인 학습 단계뿐만 아니라, 매일 수십억 번 수행되는 추론(Inference) 단계에서도 상당한 양의 이산화탄소($\text{CO}_2$)를 배출한다. 최근 연구에 따르면 추론 단계가 모델 전체 생애 주기 에너지 소비의 최대 90%를 차지할 수 있다는 점이 지적되고 있다.

따라서 본 연구의 목표는 특정 predefined task에 대해 파인튜닝(Fine-tuning)된 소형 언어 모델(Small Language Models, SLMs)이 LLM의 지속 가능한 대안이 될 수 있는지 분석하는 것이다. 특히, 성능 손실을 최소화하면서 추론 시 발생하는 탄소 배출량을 획기적으로 줄일 수 있는 성능-배출량 간의 트레이드-오프(Trade-off) 관계를 규명하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 모든 작업에 거대한 모델을 사용하는 대신, 특정 작업에 최적화된 초소형 모델($\le 250\text{M}$ parameters)을 파인튜닝하여 사용함으로써 환경적 영향과 비용을 줄이는 것이다.

주요 기여 사항은 다음과 같다.

- **초소형 모델의 가능성 검증**: 기존 연구들이 보통 7B 파라미터 수준의 모델을 다룬 것과 달리, 본 연구는 $250\text{M}$ 이하의 초소형 SLM을 대상으로 하여 LLM과의 성능 및 배출량을 비교 분석하였다.
- **추론 중심의 탄소 배출 분석**: 학습 단계보다 누적 영향이 더 큰 추론 단계의 탄소 배출량에 집중하여 실질적인 환경 영향을 평가하였다.
- **작업별 효율성 도출**: NLP, Reasoning, Programming의 세 가지 도메인에서 어떤 작업이 SLM으로 대체 가능하며, 이때 얻을 수 있는 환경적 이득이 어느 정도인지 정량적으로 제시하였다.

## 📎 Related Works

기존의 언어 모델 스케일링 법칙(Scaling Laws)에 따르면, 모델의 파라미터 수, 학습 데이터셋의 크기, 컴퓨팅 자원이 증가할수록 모델의 성능(Loss 기준)이 예측 가능한 멱함수(Power-law) 형태로 향상된다. 이로 인해 일반적인 언어 이해 및 생성 능력에서는 LLM이 SLM보다 압도적인 우위를 점한다.

하지만 최근 연구들은 파인튜닝을 통해 소형 모델의 성능을 특정 작업에서 끌어올릴 수 있음을 보여주었다. 예를 들어, Kumar 등은 7B 이하의 모델이 이미지 캡셔닝이나 텍스트-SQL 변환 등에서 LLM과 유사한 성능을 낼 수 있음을 입증하였다.

본 논문은 이러한 선행 연구에서 더 나아가, 모델 크기를 $250\text{M}$ 이하로 대폭 낮추어 분석했다는 점과, 많은 연구가 간과하는 '추론 단계'의 탄소 배출량을 집중적으로 분석했다는 점에서 차별점을 가진다. 또한, 모델 제공업체들의 환경 데이터 공개 부족과 EU AI Act의 규제적 한계(학습 단계의 에너지 소비만 보고하도록 되어 있음)를 지적하며 표준화된 측정 지표의 필요성을 강조한다.

## 🛠️ Methodology

### 전체 파이프라인

연구진은 NLP, Reasoning, Programming의 세 가지 카테고리에서 총 6개의 세부 작업을 선정하고, 각 작업에 적합한 SLM을 선정하여 파인튜닝한 후, 오픈소스 LLM(Mistral 7B, Qwen3 235B, DeepSeek R1 671B)과 성능 및 탄소 배출량을 비교하였다.

### 탄소 배출량 계산 방법

LLM의 공개된 추론 배출 데이터가 부족하여, 본 논문은 GPT-3의 추정치(쿼리당 $4.32\text{g } \text{CO}_2$, 평균 100 토큰 가정)를 기반으로 파라미터 수에 따라 선형적으로 스케일링하는 방식을 채택하였다. `eco2AI` 라이브러리를 사용하여 하드웨어 및 지역적 변수를 반영하였다.

1. **파라미터 및 토큰당 탄소 배출 계수 ($K$) 계산**:
   $$K = \frac{4.32\text{ g} / 100\text{ tokens}}{175\text{ billion parameters}} = 0.0002469\text{ g}/(\text{billion-param} \cdot \text{token})$$

2. **모델별 토큰당 탄소 배출량 ($\text{Carbon}_{\text{token}}$)**:
   $$\text{Carbon}_{\text{token}} = K \times P_{\text{active}}$$
   여기서 $P_{\text{active}}$는 모델의 활성 파라미터 수(Billion 단위)이다. (예: MoE 구조인 DeepSeek-R1은 전체 파라미터가 아닌 활성 파라미터 $37\text{B}$를 적용)

3. **추론 1회당 탄소 배출량 ($\text{Carbon}_{\text{inference}}$)**:
   $$\text{Carbon}_{\text{inference}} = \text{Carbon}_{\text{token}} \times \text{Tokens}_{\text{task}}$$
   $\text{Tokens}_{\text{task}}$는 작업별 예상 토큰 소비량(예: Sentiment Analysis는 150토큰)을 곱하여 계산한다.

## 📊 Results

### 실험 설정

- **SLM 범위**: $\le 250\text{M}$ parameters
- **LLM 범위**: $\ge 1\text{B}$ parameters
- **평가 지표**: Accuracy, Perplexity, BERT Score F1, BLEU, ROUGE-L, pass@1 등

### 주요 결과

분석 결과, 6개 작업 중 4개 작업에서 파인튜닝된 SLM이 LLM과 대등하거나 더 우수한 성능을 보였으며, 탄소 배출량은 획기적으로 낮았다.

1. **NLP (성공적)**:
   - **Sentiment Analysis**: ELECTRA($14\text{M}$) 등의 SLM이 LLM보다 높은 정확도를 보였으며, 배출량은 LLM 대비 최대 13,000배 낮았다.
   - **Content Creation**: 성능 차이는 미미했으나, 배출량은 최대 1,200배 감소하였다.

2. **Reasoning (부분적 성공)**:
   - **Natural Language Inference (NLI)**: FLAN-T5 Base($250\text{M}$)와 deBERTa v3 Base($184\text{M}$)가 LLM들을 압도하는 성능(정확도 $\approx 88\%$)을 보였으며, 배출량은 최대 1,300배 감소하였다.
   - **CoT Reasoning**: SLM의 성능이 매우 낮게 나타났다. 이는 복잡한 추론 능력이 수십억 개 이상의 파라미터를 가진 모델에서 발현된다는 기존 이론과 일치한다.

3. **Programming (부분적 성공)**:
   - **Code Summarization**: CodeParrot-Small 및 TinyCodeLM이 LLM들보다 우수한 BLEU/ROUGE-L 점수를 기록하였으며, 배출량은 최대 660배 낮았다.
   - **Code Generation**: LLM(특히 DeepSeek R1)이 압도적인 성능을 보였다. SLM은 기능적 코드 생성 능력이 현저히 떨어졌다.

## 🧠 Insights & Discussion

### 강점 및 시사점

본 연구는 'LLM 에너지 딜레마'에 대한 실질적인 해답을 제시한다. 모든 작업에 LLM을 사용하는 대신, 작업의 성격에 따라 SLM을 선택적으로 배치하는 전략이 환경적으로나 경제적으로 매우 이득임을 증명하였다. 특히 **Total Cost of Ownership (TCO)** 관점에서, SLM은 저렴한 하드웨어 사용이 가능하므로 자본 지출(CapEx)과 운영 지출(OpEx)을 동시에 낮출 수 있다.

### 한계 및 비판적 해석

- **모델 규모의 한계**: 본 연구에서 SLM의 상한선을 $250\text{M}$으로 매우 낮게 잡았다. 따라서 SLM과 LLM 사이의 '스위트 스팟(Sweet Spot, 최적의 균형점)'이 어디인지 구체적으로 특정하지 못했다.
- **비교 대상의 제한**: 리소스 제약으로 인해 오픈소스 LLM만을 비교 대상으로 삼았으며, GPT-4나 GPT-5와 같은 최상위 폐쇄형 모델과의 비교는 이루어지지 않았다.
- **추정치의 불확실성**: LLM의 배출량을 GPT-3 기준으로 선형 스케일링하여 추정했기 때문에, 실제 하드웨어 가속기나 최적화 기법(Quantization 등)이 적용된 실제 환경과는 차이가 있을 수 있다.

## 📌 TL;DR

본 논문은 초소형 언어 모델($\le 250\text{M}$)을 파인튜닝하여 사용했을 때, 6개 주요 작업 중 4개(감성 분석, 콘텐츠 생성, 자연어 추론, 코드 요약)에서 LLM과 대등한 성능을 내면서도 탄소 배출량을 수백에서 수만 배까지 줄일 수 있음을 입증하였다. 다만, 복잡한 추론(CoT)과 코드 생성 작업은 여전히 대규모 모델이 필수적이다. 이 연구는 특정 작업에 최적화된 SLM 배치가 환경 보호뿐만 아니라 TCO 절감이라는 경제적 이득을 제공함을 시사하며, 향후 'Green AI' 구현을 위한 구체적인 근거를 제시한다.
