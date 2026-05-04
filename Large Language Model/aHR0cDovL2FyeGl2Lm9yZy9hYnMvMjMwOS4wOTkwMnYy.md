# Speaker Attribution in German Parliamentary Debates with QLoRA-adapted Large Language Models

Tobias Bornheim, Niklas Grieger, Patrick Gustav Blaneck, and Stephan Bialonski (2024)

## 🧩 Problem to Solve

본 논문은 독일 의회 토론 데이터와 같은 정치적 텍스트에서 **Speaker Attribution**(화자 귀속)을 자동화하는 문제를 해결하고자 한다. Speaker Attribution이란 특정 발화 사건에서 '누가, 무엇을, 누구에게' 말했는지를 탐지하는 작업으로, 이는 자연어 처리의 **Semantic Role Labeling**(SRL, 의미역 표지)과 밀접한 관련이 있다.

정치적 텍스트는 대개 비정형 구조를 가지고 있어 수동 분석에 막대한 비용과 시간이 소요된다. 특히 독일어와 같은 언어는 영어에 비해 주석이 달린 학습 데이터셋이 부족한 저자원 언어(low-resource language) 특성이 있어, 기존의 end-to-end 모델을 학습시키기에 어려움이 있다. 따라서 본 연구의 목표는 최신 대형 언어 모델(LLM)인 Llama 2를 활용하여 독일 의회 토론 데이터에서 화자 귀속을 효율적으로 수행하는 시스템을 구축하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 **Llama 2 70B** 모델을 **QLoRA**(Quantized Low-Rank Adaptation) 기법으로 미세 조정(fine-tuning)하여, 화자 귀속의 핵심 요소인 **Cue**(단서)와 **Role**(역할)을 단계적으로 추출하는 파이프라인을 설계한 것이다. 

특히, 모델이 단순히 텍스트를 생성하는 것을 넘어 정교한 구조의 정보를 추출할 수 있도록 영어로 작성된 지시문(Instruction) 기반의 프롬프트 템플릿을 설계하였으며, Cue 모델과 Role 모델을 분리하여 순차적으로 적용하는 전략을 취했다.

## 📎 Related Works

기존의 Semantic Role Labeling(SRL) 연구들은 초기에는 구문론적 특징(syntactic features)에 의존하는 방식이었다. 이후 BERT와 같은 encoder-only 모델들이 등장하며 원시 토큰에서 특징을 학습하는 end-to-end 방식으로 전환되었으나, 이는 대규모의 주석 데이터셋을 필요로 한다는 한계가 있었다.

최근에는 GPT-4, Claude 2, Llama 2와 같은 decoder-only 기반의 LLM이 등장하며 자연어 지시를 이해하고 수행하는 능력이 비약적으로 향상되었다. 본 논문은 이러한 LLM의 능력이 Speaker Attribution 작업에 적용될 가능성을 탐구하며, 기존의 특징 공학(feature engineering) 기반 접근 방식이나 단순한 encoder-only 모델의 한계를 넘어 지시어 기반 미세 조정을 통해 성능을 높이고자 하였다.

## 🛠️ Methodology

### 전체 파이프라인 및 시스템 구조
본 연구는 두 개의 서로 다른 Llama 2 70B 모델을 사용하여 2단계 추론 과정을 거친다.
1. **Cue Model**: 입력 문장에서 발화 사건을 일으키는 단서(Cue)를 탐지한다.
2. **Role Model**: 탐지된 Cue를 바탕으로 관련된 역할(Source, Message, Addressee 등)을 추출한다.

### 데이터 전처리 및 프롬프트 설계
- **문맥 확장**: 역할(Role) 정보가 Cue가 포함된 샘플 이후의 문장에 걸쳐 나타날 수 있으므로, 현재 샘플에 다음 두 개의 샘플을 연결하여 입력으로 제공한다.
- **프롬프트 언어**: Llama 2와 같은 다국어 모델은 영어 프롬프트를 사용할 때 성능이 향상된다는 선행 연구에 따라, 지시문은 영어로 작성하였다.
- **특수 토큰**: Cue나 Role이 존재하지 않는 경우에는 `#UNK#`라는 마커를 사용하도록 학습시켰다.

### 훈련 목표 및 방법 (QLoRA)
모델의 파라미터 수가 매우 크기 때문에 메모리 효율성을 위해 **QLoRA** 기법을 적용하였다. QLoRA의 핵심은 다음과 같다.
- 모델의 가중치를 4비트로 양자화(Quantization)하여 메모리 점유율을 낮춘다.
- 모든 선형 트랜스포머 블록에 **Low Rank Adapters (LoRA layers)**를 추가한다.
- 학습 시에는 양자화된 기본 가중치는 고정하고, 추가된 LoRA 레이어만 학습시킨다.

**학습 하이퍼파라미터**:
- 학습률(Learning rate): $\eta = 0.0001$ (상수)
- Warmup: 초기 학습 단계의 3% 동안 linear warmup 적용
- Dropout: LoRA 레이어에 0.05 적용
- 최적화: Cue 모델은 2000 step, Role 모델은 2500 step 동안 학습을 진행하였다.

### 추론 및 후처리 절차
추론은 결정론적(deterministic)인 결과를 얻기 위해 **Greedy Decoding**(가장 확률이 높은 토큰을 항상 선택) 방식을 사용하였다. 출력된 결과는 다음과 같은 후처리 과정을 거친다.
- **형식 강제**: 설계된 출력 형식을 따르지 않을 경우 `#UNK#`로 처리한다.
- **중복 제거**: 겹치는 Cue가 발견되면 하나로 통합한다.
- **환각 제거**: 입력 문장에 존재하지 않는 단어가 생성된 경우(Levenshtein distance 1 초과) 이를 무시한다.
- **모호성 해결**: 동일 단어가 여러 번 등장할 경우, 주변 단어 중 Cue/Role에 해당하는 단어가 더 많은 위치를 선택한다.

## 📊 Results

### 실험 설정
- **데이터셋**: GermEval 2023 Shared Task의 독일 연방의회(Bundestag) 연설 데이터.
- **평가 지표**: **Proportional F1 score**를 사용하였다. 이는 예측된 구간과 실제 구간의 겹침 정도를 측정하는 Precision과 Recall의 조화 평균이다.

### 주요 결과
실험 결과는 다음과 같다 (Table 3 참조).

| Task | Precision | Recall | F1 |
| :--- | :---: | :---: | :---: |
| **Subtask 1: Cues** | 0.889 | 0.889 | 0.889 |
| **Subtask 1: Roles** | 0.787 | 0.822 | 0.804 |
| **Subtask 1: Joint (Cues & Roles)** | 0.798 | 0.829 | 0.813 |
| **Subtask 2: Roles (Gold Cues)** | 0.910 | 0.873 | 0.891 |

### 결과 분석
1. **Cue 예측 성능**: Cue 모델은 0.889라는 높은 F1 스코어를 기록하며 안정적인 성능을 보였다.
2. **에러 전파(Error Propagation)**: Subtask 1(예측된 Cue 사용)의 Role F1 스코어(0.804)보다 Subtask 2(정답 Cue 제공)의 Role F1 스코어(0.891)가 훨씬 높게 나타났다.
3. **Precision의 변화**: 특히 정답 Cue를 제공했을 때 Precision이 $0.787 \rightarrow 0.910$으로 크게 상승하였다. 이는 Cue 모델이 실제로는 Cue가 없는 문장을 Cue가 있다고 과잉 예측(False Positive)했고, 이것이 Role 모델의 잘못된 예측으로 이어졌음을 시사한다.

## 🧠 Insights & Discussion

### 강점 및 성과
본 연구는 전통적인 언어학적 특징(linguistic features)을 수동으로 설계하지 않고도, LLM의 미세 조정만으로 독일어 화자 귀속 작업에서 경쟁력 있는 성능을 달성할 수 있음을 입증하였다. 이는 복잡한 의미론적 분석 작업에서도 적절한 프롬프트 설계와 효율적인 튜닝 기법(QLoRA)이 유효함을 보여준다.

### 한계 및 비판적 해석
- **에러 전파 문제**: Cue 모델의 작은 실수가 Role 모델의 성능을 크게 떨어뜨리는 구조적 취약점이 확인되었다. 두 모델을 개별적으로 학습시키는 대신, End-to-End로 통합 학습시키거나 Cue 예측 결과의 신뢰도를 Role 모델에 전달하는 메커니즘이 필요할 것으로 보인다.
- **범용성 검증 부족**: 독일 의회 데이터라는 특수한 도메인에 한정되어 실험이 진행되었으므로, 일반적인 뉴스 기사나 다른 국가의 정치 텍스트에서도 동일한 성능이 나올지는 미지수이다.
- **보안 및 리스크**: 저자들은 생성된 텍스트를 직접 노출하지 않고 원문 단어로 매핑하여 사용하므로 리스크가 적다고 주장하지만, LLM의 특성상 예기치 못한 환각(Hallucination)이나 편향이 발생할 가능성이 여전히 존재한다.

## 📌 TL;DR

본 논문은 **Llama 2 70B 모델에 QLoRA를 적용**하여 독일 의회 토론의 **Speaker Attribution**(화자 귀속)을 자동화하는 시스템을 제안하였다. Cue 탐지와 Role 추출을 분리한 2단계 파이프라인을 통해 높은 성능을 거두었으며, 특히 **정교한 프롬프트 설계와 양자화 기반 미세 조정**이 저자원 언어의 의미역 표지 작업에 효율적임을 보였다. 이 연구는 향후 LLM 기반의 복잡한 Semantic Role Labeling 시스템 개발에 중요한 기초 자료가 될 것으로 기대된다.