# Clinical information extraction for Low-resource languages with Few-shot learning using Pre-trained language models and Prompting

Phillip Richter-Pechanski, Philipp Wiesenbach, Dominic M. Schwab, Christina Kiriakou, Nicolas Geis, Christoph Dieterich, and Anette Frank (2024)

## 🧩 Problem to Solve

본 논문은 독일어와 같은 저자원 언어(Low-resource languages) 환경에서 의료 문서, 특히 의사 소견서(Doctor's letters)의 비정형 텍스트로부터 임상 정보를 자동으로 추출하는 문제를 해결하고자 한다. 구체적인 작업은 의사 소견서의 단락을 9개의 섹션 클래스(예: Anamnese, Medikation, Befunde 등)로 분류하는 **Section Classification**이다.

이 문제는 다음과 같은 네 가지 주요 도전 과제를 안고 있다.

1. **도메인 및 전문가 의존성(Domain-and-Expert-dependent):** 데이터 어노테이션과 모델 평가를 위해 고도의 임상 전문 지식을 갖춘 전문가의 참여가 필수적이다.
2. **자원 제약(Resource-constrained):** 전문가의 시간과 비용이 매우 비싸며, 엄격한 데이터 보호 규정으로 인해 외부 전문가의 참여가 어렵다.
3. **온프레미스 환경(On-premise):** 민감한 개인 정보 보호를 위해 병원 내부 IT 인프라에서 모델을 구축해야 하며, 이로 인해 계산 자원이 제한적이다.
4. **투명성(Transparency):** 의료 분야의 특성상 모델의 예측 결과가 안전하고 신뢰할 수 있어야 하며, 예측 근거가 설명 가능(Explainable)해야 한다.

결과적으로 본 연구의 목표는 최소한의 학습 데이터(Few-shot)와 제한된 컴퓨팅 자원만으로도 높은 성능과 설명 가능성을 동시에 확보할 수 있는 최적의 임상 정보 추출 파이프라인을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **경량화된 Masked Language Model(MLM)에 도메인 적응(Domain Adaptation)을 적용하고, 이를 Prompt-based learning 방법론인 PET(Pattern-Exploiting Training)와 결합**하는 것이다.

주요 기여 사항은 다음과 같다.

- **Few-shot Learning 적용:** 대규모 데이터 없이 소량의 샘플(Shot)만으로 성능을 극대화하기 위해 PET를 도입하여 전통적인 Sequence Classification(SC) 모델보다 우수한 성능을 입증하였다.
- **체계적인 도메인 적응 전략:** 일반 언어 모델(gbert)과 의료 특화 모델(medbert-de)을 대상으로 task-adaptation, domain-adaptation, 그리고 이를 결합한 combined-adaptation의 효과를 정량적으로 분석하였다.
- **설명 가능성 확보:** Shapley value(SHAP)를 사용하여 모델이 어떤 토큰을 근거로 예측했는지 분석함으로써, 학습 데이터의 품질을 개선하고 모델 선택의 근거를 마련하였다.
- **실무적 가이드라인 제공:** 저자원 언어의 임상 정보 추출 프로젝트에서 모델 크기, 프롬프트 설계, 컨텍스트 활용 방안에 대한 프로세스 중심의 가이드라인을 제시하였다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구와 그 한계를 언급한다.

- **Fine-tuning $\rightarrow$ Prompting:** 기존의 'Pretrain-then-finetune' 패러다임은 대량의 라벨링 데이터가 필요하며, 데이터가 부족한 few-shot 상황에서는 성능이 급격히 떨어진다. 최근의 Prompting 방식은 모델이 컨텍스트를 통해 정답을 추론하게 함으로써 이 문제를 해결한다.
- **Domain Adaptation:** 일반 PLM은 도메인 외(Out-of-domain) 설정에서 성능이 하락한다. 이를 위해 도메인 특화 데이터로 추가 사전 학습(Further-pretraining)을 진행하는 방식이 제안되었으나, 임상 루틴 텍스트의 특수성으로 인해 적용 결과가 다양하다.
- **Medical PLMs:** English 기반의 BioBERT 등은 많으나, 독일어와 같은 저자원 언어의 임상 PLM은 데이터 보호 규정으로 인해 매우 부족한 실정이다.
- **Interpretability:** 딥러닝의 블랙박스 특성을 해결하기 위해 Saliency feature 기반의 방법들이 제안되었으며, 특히 Shapley value는 이론적 근거가 탄탄한 특성 기여도 분석 방법으로 알려져 있다.

## 🛠️ Methodology

### 1. Pattern-Exploiting Training (PET)

PET는 텍스트 분류 작업을 언어 모델링 문제(Cloze-style problem)로 변환하여 학습하는 semi-supervised 방법론이다. 전체 프로세스는 세 단계로 구성된다.

1. **Pattern-based Fine-tuning:** 입력 인스턴스 $x$를 패턴 함수 $P(x)$를 통해 빈칸 채우기(Cloze) 문장으로 변환하고, MLM을 사용하여 [MASK] 토큰에 올 가능성이 가장 높은 토큰 $v(y)$를 예측하도록 미세 조정한다.
2. **Soft Labeling:** 위에서 학습된 여러 패턴 모델들의 앙상블을 사용하여 라벨이 없는 대규모 데이터셋 $D$에 대해 소프트 라벨을 부여한다.
3. **Final Classification:** 소프트 라벨이 부여된 $D$를 사용하여 전통적인 Sequence Classification 헤드를 가진 최종 분류기 $C$를 학습시킨다.

**프롬프트 템플릿 및 Verbalizer:**

- **템플릿:** Null prompt(단순 [MASK]), Punctuation(콜론/하이픈 추가), Prompt(특정 단어 추가), Q&A(질문-답변 형식)의 네 가지 타입을 사용하였다.
- **Verbalizer:** 레이블을 특정 토큰으로 매핑하는 과정에서 전문가의 수작업을 줄이기 위해 **PETAL (Automatic Labels)** 방식을 사용하여 데이터와 모델에 기반해 최적의 토큰을 자동으로 계산하였다.

### 2. Pre-trained Language Models & Domain Adaptation

BERT 아키텍처 기반의 `gbert-base`, `gbert-large`, `medbert-de`를 사용하였으며, 다음과 같은 추가 사전 학습(Further-pretraining) 전략을 적용하였다.

- **Task-adaptation:** 대상 작업과 동일한 소스의 비라벨링 데이터(CARDIO:DE)로 학습.
- **Domain-adaptation:** 더 넓은 범위의 심혈관 도메인 데이터(의사 소견서 17.9만 건 및 암 가이드라인 GGPONC)로 학습.
- **Combined:** Domain-adaptation 이후 Task-adaptation을 순차적으로 진행.

### 3. Interpretability with Shapley Values

모델의 예측 근거를 분석하기 위해 게임 이론 기반의 Shapley value를 도입하였다. 특정 특징 $i$의 Shapley value $\phi_i(f)$는 다음과 같이 정의된다.
$$\phi_i(f) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|! (|N| - |S| - 1)!}{|N|!} [f(S \cup \{i\}) - f(S)]$$
여기서 $f$는 예측 함수, $S$는 특징 $i$를 제외한 특징들의 부분 집합, $N$은 전체 특징 집합을 의미한다. 본 논문에서는 계산 효율성을 위해 이를 근사한 **SHAP** 라이브러리를 사용하였다.

### 4. Data & Contextualization

- **데이터셋:** 독일어 심혈관 도메인 말뭉치인 `CARDIO:DE`를 사용하였으며, 9개의 섹션 클래스로 구성된다.
- **컨텍스트 구성:** 단일 단락만 사용하는 `no-context`, 이전/다음 단락을 함께 제공하는 `context`, 이전 단락만 제공하는 `prev-context` 세 가지 설정을 비교하였다.

## 📊 Results

### 1. 핵심 실험 결과 (Core Experiments)

- **PET vs. SC:** Few-shot 설정(Shot $\le 100$)에서 PET가 전통적인 SC 모델을 압도적으로 능가하였다. 특히 20-shot 설정에서 PET는 SC 대비 정확도를 **48.6%에서 79.1%로 크게 향상**시켰다.
- **사전 학습의 영향:** `gbert` 모델의 경우 combined-adaptation $\rightarrow$ domain-adaptation $\rightarrow$ task-adaptation 순으로 성능이 향상되었다. 반면 `medbert-de`는 추가 사전 학습이 일관된 성능 향상을 보이지 않았는데, 이는 기존 모델이 이미 특정 도메인(종양학)에 특화되어 있어 추가 학습이 오히려 방해가 되었을 가능성이 제기되었다.
- **최적 모델:** `gbert-base-comb` 모델이 few-shot 환경에서 가장 우수한 성능을 보였다.

### 2. 추가 실험 결과 (Additional Experiments)

- **모델 크기:** `gbert-large`가 `gbert-base`보다 전반적으로 우수하며, 특히 `Anamnese`와 같은 복잡한 자유 텍스트 섹션 분류에서 F1-score가 유의미하게 상승하였다.
- **Null Prompts:** 수작업 템플릿 없이 [MASK]만 사용하는 Null prompt가 100-shot 이상의 설정에서는 정교한 템플릿과 대등한 성능을 보여, 엔지니어링 비용을 크게 줄일 수 있음을 확인하였다.
- **컨텍스트 활용:** 컨텍스트를 추가했을 때 정확도가 평균 2.4pp 향상되었으며, 특히 `Anamnese` 클래스의 F1-score가 크게 개선되었다.
- **최종 조합:** `gbert-large-comb` 모델에 컨텍스트를 적용하고 모든 템플릿을 사용했을 때, full-dataset으로 학습한 SC 모델과의 격차를 최소 $-5.2$pp까지 좁혔다.

### 3. Shapley Value 분석

- **오분류 분석:** `Anamnese` 클래스의 오분류 샘플을 분석한 결과, 모델이 'Aufnahme', 'Patient'와 같은 특정 토큰에 과하게 의존하고 있음을 발견하였다.
- **모델 비교:** `gbert-large` 모델은 `gbert-base`보다 컨텍스트보다 분류 대상이 되는 메인 단락의 토큰에 더 높은 가중치를 두어, 더 신뢰할 수 있는 예측을 수행함을 확인하였다.

## 🧠 Insights & Discussion

### 1. 강점 및 발견점

- **일반 모델의 잠재력:** 대규모 일반 언어 데이터로 학습된 모델(`gbert`)을 도메인 특화 데이터로 추가 학습시키는 것이, 처음부터 작은 규모의 도메인 데이터로 학습된 모델(`medbert-de`)보다 few-shot 상황에서 더 유연하게 적응할 수 있음을 보여주었다.
- **컨텍스트의 중요성:** 의료 문서의 섹션은 앞뒤 내용과의 유기적 관계가 강하므로, 단순 단락 분류보다 주변 컨텍스트를 포함하는 것이 성능 향상의 핵심이다.
- **설명 가능성의 실용성:** SHAP 분석을 통해 단순히 성능 지표만 보는 것이 아니라, 모델이 잘못된 특징(예: 단순 빈출 단어)에 의존하고 있는지 파악하여 학습 데이터를 최적화할 수 있는 경로를 제시하였다.

### 2. 한계 및 논의사항

- **단순 섹션의 성능 저하:** `Anrede`나 `Mix`와 같이 매우 짧은 단락으로 구성된 섹션의 경우, 컨텍스트를 추가하면 오히려 모델이 대상 단락과 컨텍스트를 구분하지 못해 성능이 떨어지는 현상이 관찰되었다.
- **데이터 의존성:** 20-shot과 같은 극소량의 데이터에서는 여전히 Full-dataset 모델과의 격차가 존재하며, 이는 의료 도메인의 높은 복잡성으로 인해 최소한의 임계치 이상의 데이터가 필요함을 시사한다.

## 📌 TL;DR

본 논문은 저자원 언어(독일어)의 임상 텍스트 섹션 분류를 위해 **도메인 적응된 경량 PLM과 PET(Prompt-based learning)를 결합한 프레임워크**를 제안한다. 20-shot의 매우 적은 데이터만으로도 정확도를 **48.6%에서 79.1%로 끌어올렸으며**, 특히 `gbert` 모델에 도메인/태스크 적응 학습을 결합하고 컨텍스트를 추가했을 때 최적의 성능을 얻었다. 또한 Shapley value를 통해 모델의 예측 근거를 투명하게 분석함으로써 의료 현장에서 요구되는 신뢰성을 확보하였다. 이 연구는 컴퓨팅 자원과 전문가 라벨링 데이터가 부족한 의료 환경에서 실무적으로 적용 가능한 효율적인 정보 추출 가이드라인을 제공한다.
