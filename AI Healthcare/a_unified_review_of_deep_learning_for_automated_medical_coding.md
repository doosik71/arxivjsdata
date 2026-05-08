# A Unified Review of Deep Learning for Automated Medical Coding

SHAOXIONG JI, XIAOBO LI, WEI SUN, HANG DONG, ARA TAALAS, YIJIA ZHANG, HONGHAN WU, ESA PITKÄNEN, PEKKA MARTTINEN

## 🧩 Problem to Solve

자동화된 의료 코딩은 임상 문서를 기반으로 의료 코드를 예측하여 비정형 데이터를 관리 가능한 형태로 변환하는 의료 운영 및 제공에 필수적인 작업입니다. 최근 딥러닝과 자연어 처리(NLP)의 발전이 이 분야에 널리 적용되었지만, 딥러닝 기반 의료 코딩 모델의 신경망 아키텍처 설계에는 **통합된 관점(unified view)이 부족**합니다.

이러한 배경 속에서 연구는 다음 세 가지 주요 과제에 직면해 있습니다:

1. **잡음 많고 긴 임상 기록 (Noisy and Lengthy Clinical Notes)**: 임상 기록은 전문 용어, 비표준 동의어, 오타가 많고, 수백에서 수천 단어에 이르는 긴 문서인 경우가 많습니다. 또한 작성 스타일이 다양하고 시간 경과에 따라 코딩 시스템과 표기법이 변합니다.
2. **고차원 의료 코드 (High-dimensional Medical Codes)**: 의료 기록은 수많은 진단 코드와 연결될 수 있어, 수천에서 수만에 이르는 레이블을 가진 다중 레이블 극한 분류(multi-label extreme classification) 문제로 간주됩니다.
3. **불균형한 클래스 (Imbalanced Classes)**: 환자 한 명에게 할당되는 코드는 소수에 불과하며, 흔한 질병과 희귀 질병의 분포가 불균형하여 `long-tail 현상`이 나타납니다.

## ✨ Key Contributions

이 논문은 딥러닝 기반 자동화 의료 코딩 연구의 통합된 관점을 제시하며, 다음과 같은 핵심 기여를 합니다:

- **통합 인코더-디코더 프레임워크 제안**: 의료 코딩 모델의 구성 요소를 이해하기 위한 `통합 프레임워크`를 제안하여 신경망 아키텍처 설계에 대한 일반적인 이해를 제공합니다.
- **최신 모델 요약 및 분류**: 제안된 프레임워크 하에 최근의 고급 딥러닝 기반 의료 코딩 모델들을 인코더 모듈, 깊은 아키텍처 구성 메커니즘, 디코더 모듈, 보조 정보 활용의 네 가지 핵심 구성 요소로 분해하고 요약합니다.
- **벤치마크 및 실제 활용 논의**: 널리 사용되는 벤치마크 데이터셋, 평가 지표 및 실제 적용 사례를 소개합니다.
- **주요 연구 과제 및 미래 방향 제시**: 현재의 한계점을 논의하고, 장기 의존성, 확장성, 클래스 불균형, 해석 가능성, 인간 개입 시스템, LLMs 활용 등 향후 연구 방향을 제시합니다.

## 📎 Related Works

이 분야에는 여러 체계적 및 서술적 리뷰가 존재합니다.

- **초기 리뷰**: Campbell et al. [24]은 퇴원 코딩의 정확도를, Stanfill et al. [160]은 기존 분류 방법과 자동 코딩 시스템을 소개했습니다. Burns et al. [19]는 데이터 정확도에 대한 업데이트된 리뷰를 수행했습니다.
- **최근 체계적 리뷰**: Kaur et al. [88, 89]은 2010년부터 2021년까지의 기계 학습 및 NLP 기술을 활용한 연구들을 검토했습니다. Khope and Elias [91]는 MIMIC-III 데이터셋을 사용한 연구에 초점을 맞추었습니다.
- **기술적 리뷰**: Teng et al. [164]은 초기 2022년에 출판되어 특징 엔지니어링 기반 분류기와 딥러닝 방법에 초점을 맞춘 기술적 리뷰를 제공했으나, 모든 딥러닝 아키텍처를 일반화하는 통합된 관점을 제시하지는 못했습니다.

본 논문은 이러한 기존 리뷰의 한계를 극복하고, 딥러닝 아키텍처의 미묘한 다양성을 일반화하는 **통합된 관점을 제공**하고, `multitask learning`, `few-shot/zero-shot learning`, `contrastive learning`, `adversarial generative learning`, `reinforcement learning`과 같은 `supervised learning`을 넘어선 **최신 학습 패러다임을 포함**한다는 점에서 차별점을 가집니다.

## 🛠️ Methodology

본 논문은 자동화된 의료 코딩을 위한 `통합 인코더-디코더 프레임워크`를 제안합니다. 이 프레임워크는 크게 네 가지 구성 요소로 나뉩니다: `인코더 모듈 (Encoder Modules)`, `깊은 아키텍처 구성 메커니즘 (Building Deep Architectures)`, `디코더 모듈 (Decoder Modules)`, 그리고 `보조 정보 활용 (Usage of Auxiliary Information)`. `Human-in-the-loop` 시스템도 중요한 방법론으로 다룹니다.

### 인코더 모듈 (Encoder Modules)

임상 텍스트 데이터에서 관련 특징을 추출하는 역할을 합니다.

- **순환 신경망 인코더 (Recurrent Neural Encoders, RNN)**: 시퀀스 데이터를 모델링하여 `Attentive LSTM` [157], `HA-GRU` [11] 와 같이 시퀀스 의존성을 캡처하는 데 활용됩니다.
- **합성곱 신경망 인코더 (Convolutional Neural Encoders, CNN)**: 텍스트에서 지역 특징과 패턴을 추출하는 데 사용됩니다. `TextCNN` [96], `CAML` [131], `DCAN` [77], `MultiResCNN` [104], `Gated CNN` [82] 등이 있습니다.
- **신경망 어텐션 및 트랜스포머 인코더 (Neural Attention and Transformer Encoders)**: `self-attention` 메커니즘을 사용하여 문맥화된 단어 임베딩을 학습합니다. `BERT` [211], `Longformer` [12], `BigBird` [205], `FLASH` [74] 등이 긴 문서 처리에 사용됩니다. `BERT` 기반 인코더는 512 토큰 길이 제한이 있어 긴 문서 처리를 위해 `계층적 BERT` [78]와 같은 접근 방식이 연구됩니다.
- **그래프 인코더 (Graph Encoders)**: 의료 그래프를 구축하고 `Graph Convolutional Network (GCN)` [98] 또는 `Graph Attention Network (GAT)` [171]를 사용하여 질병 계층 구조나 환자 정보 등 구조적 정보를 캡처합니다 (`GMAN` [203], `CMGE` [188]).
- **계층적 인코더 (Hierarchical Encoders)**: 텍스트의 계층적 구조(문자, 단어, 문장, 청크)를 인코딩에 활용하여 긴 임상 문서 처리의 어려움을 완화합니다 (`HA-GRU` [11], `BERT-hier` [78], `MD-BERT` [209]).

### 깊은 아키텍처 구성 메커니즘 (Building Deep Architectures)

- **스태킹 (Stacking)**: 여러 신경망 레이어를 단순히 쌓는 가장 기본적인 방법입니다.
- **임베딩 주입 (Embedding Injection)**: 원본 단어 임베딩을 각 중간 레이어에 연결하여 정보 손실을 완화합니다 (`J_l = concat(X, H_l)$ [82]).
- **잔여 연결 (Residual Connections)**: `skip connection`을 사용하여 `vanishing gradient` 문제를 방지하고 매우 깊은 신경망 아키텍처를 구축합니다 ($\mathrm{H}_{l+1} = \sigma(\mathrm{H}_{l} + G(\mathrm{H}_{l}))$ [104, 77, 123]).
- **하이웨이 네트워크 (Highway Networks)**: `gating mechanism`을 사용하여 정보 흐름을 제어합니다 (의료 코딩 모델에서는 아직 널리 채택되지 않음).

### 디코더 모듈 (Decoder Modules)

학습된 은닉 표현을 의료 코드로 매핑하여 최종 분류 결과를 생성합니다.

- **완전 연결 레이어 (Fully Connected Layer)**: 가장 간단한 디코더로, 선형 투영 후 `Sigmoid` 활성화 함수를 사용하여 예측 로짓을 생성합니다 ($\hat{y} = \mathrm{Sigmoid}(\mathrm{Pooling}(\mathrm{HW}^T))$ [157, 147]).
- **신경망 어텐션 디코더 (Neural Attention Decoders)**: `Label-wise Attention Network (LAN)` [131]와 같이 의료 코드와 관련된 중요한 정보에 초점을 맞춰 코드 예측을 강화합니다 ($A = \mathrm{Softmax}(\mathrm{HU})$ [131, 77, 104, 123]). `Structured self-attention` [110, 14]도 활용됩니다.
- **계층적 디코더 (Hierarchical Decoders)**: `ICD` 코드의 계층적 구조를 활용하여 보다 구조화된 예측을 수행합니다 (`JointLAAT` [173], `RPGNet` [180]).
- **멀티태스크 디코더 (Multitask Decoders)**: 여러 코딩 시스템을 동시에 처리하며 `multitask learning`을 통해 예측 정확도를 높입니다 (`MT-RAM` [161], `MARN` [162]).
- **Few-shot/Zero-shot 디코더 (Few-shot/Zero-shot Decoders)**: 학습 데이터에 거의 없거나 전혀 없는 코드를 예측하는 것을 목표로 합니다. 임상 문서와 레이블 벡터 간의 의미 일치를 계산하는 검색 태스크로 정의됩니다 ($\tilde{y}_i = \mathrm{Sigmoid}(e_i^\top v_i)$ [151, 120, 159]).
- **자동회귀 생성 디코더 (Autoregressive Generative Decoders)**: 사전 훈련된 언어 모델의 `prompt tuning` 패러다임을 활용하여 다중 레이블 분류를 자동회귀 생성 태스크로 전환합니다 (`Yang et al. [199, 200]`).

### 보조 정보 활용 (Usage of Auxiliary Information)

표현 학습을 강화하고 의료 코딩 성능을 향상시키기 위해 외부 데이터를 활용합니다.

- **Wikipedia 문서**: 의료 진단에 대한 상세한 설명을 제공하여 모델의 임상 텍스트 이해를 돕습니다 (`C-MemNN` [147], `KSI` [9], `MCDA` [183]).
- **코드 설명 (Code Description)**: 의료 코드의 정확한 의미를 설명하는 텍스트 정보를 활용하여 표현 학습을 강화합니다 (`DR-CAML` [131], `CAIC` [165], `BiCapsNetLE` [10], `DLAC` [61]).
- **코드 계층 구조 (Code Hierarchy)**: 코드 간의 관계를 구조화된 지식 그래프로 간주하여 문서 표현에 통합합니다 (`MSATT-KG` [193], `HyperCore` [26], `HieNet` [181]).
- **의료 온톨로지 (Medical Ontology)**: `UMLS (Unified Medical Language System)`와 `SNOMED CT`와 같은 포괄적인 지식 소스에서 개념을 추출하고 동의어를 활용하여 모델 성능을 향상시킵니다 (`MSMN` [204], `Dong et al. [52]`, `Li et al. [109]`).
- **차트 데이터 (Chart Data)**: 환자의 생리적 상태를 기록하는 구조화된 데이터(예: 입원 기록, 실험실 결과, 처방)를 텍스트 데이터와 결합하여 `multimodal learning`을 수행합니다 (`Wang et al. [178]`, `Xu et al. [195]`, `TreeMAN` [118]).
- **개체 및 개념 (Entities and Concepts)**: 임상 기록에서 의료 코드에 대한 텍스트 언급을 추상화한 개체 및 개념을 활용하여 텍스트 특징을 강화하거나 추가 감독 신호를 제공합니다 (`cTAKES` [185], `SemEHR` [189], `MedCAT` [101]).

### Human-in-the-loop

인간 코더의 전문성을 자동화된 시스템에 통합하여 코딩 효율성과 정확도를 높입니다.

- **컴퓨터 지원 임상 코딩 (Computer-assisted Clinical Coding, CAC)**: 임상 코딩의 정확도와 품질을 향상시키고 인력의 부담을 줄이는 기술입니다.
- **능동 학습 (Active Learning)**: 인간 주석자가 가장 유용한 데이터 샘플에 집중하도록 하여 수동 라벨링 비용을 줄입니다 [60].
- **설명 가능성 (Explainability)**: 의료 전문가가 모델의 추론을 이해하고 예측을 신뢰할 수 있도록 모델이 특정 예측에 도달한 이유를 설명합니다 [166].
- **인간 평가 (Human Evaluation)**: 자동화된 의료 코딩 시스템의 성능을 인간 코더의 성능과 비교하여 평가합니다 [92].

## 📊 Results

의료 코딩 모델의 예측 성능은 `AUC-ROC`, `F1-score` (micro 및 macro 평균), `P@k (Precision@k)`와 같은 다중 레이블 다중 클래스 분류의 표준 평가 지표를 사용하여 평가됩니다. `MIMIC-III` 데이터셋은 의료 코딩 연구에서 가장 널리 사용되는 공개 데이터 소스이며, 특히 상위 50개 빈번 코드(`MIMIC-III top-50`)와 전체 코드 데이터셋(`MIMIC-III-full`)이 사용됩니다.

Table 4는 `MIMIC-III-full` 데이터셋에서 여러 대표적인 모델의 성능 (AUC-ROC, F1-score, P@k)을 보여줍니다.

- `CNN`과 `BiGRU` 같은 초기 모델들은 `AUC-ROC` 80% 초반, `Micro F1` 40% 초반의 성능을 보였습니다.
- `CAML`과 `DR-CAML`은 `AUC-ROC`를 89%대로 향상시켰습니다.
- `MultiResCNN`, `MSAAT-KG`, `LAAT`, `JointLAAT`, `HyperCore` 등 더욱 발전된 모델들은 `AUC-ROC` 91~93%, `Micro F1` 55~58% 범위로 성능을 개선했습니다.
- `ISD`, `RAC`, `MDBERT`, `MSMN`, `HieNet`과 같은 최신 모델들은 `AUC-ROC` 93% 이상, `Micro F1` 56% 이상을 달성하며 상당한 성능 향상을 보여주었습니다. 특히 `RAC`와 `MSMN`은 `AUC-ROC` 94%~95%에 근접하는 높은 성능을 기록했습니다.

평가 지표 측면에서는 각 코드를 독립적으로 처리하는 `flat evaluation` 대신, 계층적 코드 구조를 반영하는 `계층적 평가`의 필요성이 강조됩니다. `CoPHE` [55]는 계층 내 노드의 깊이를 나타내는 지표를 제안하여 잘못되었지만 관련성 있는 코드를 정량화하고, `Weak Hierarchical Confusion Matrix (WHCM)` [56]는 코드 "패밀리" 가정을 기반으로 `document-level` 다중 레이블 설정에서 혼동 행렬을 조정합니다.

## 🧠 Insights & Discussion

딥러닝은 자동화된 의료 코딩의 성능을 크게 향상시켰지만, 여전히 많은 과제가 남아 있습니다.

- **리더보드 중심 연구의 한계**: 공개 벤치마크의 점수 추구에만 집중하면 다른 중요한 문제를 간과할 수 있습니다. 예를 들어, `MIMIC-III` 데이터셋은 불완전하게 라벨링되었을 수 있으며, 10년 이상 된 과거 데이터만을 반영합니다.
- **장기 의존성 및 확장성**: 임상 기록이 매우 긴 문서일 때 `neural encoder`가 장기 의존성을 캡처하는 것은 어렵습니다. `self-attention` 기반 모델은 `quadratic complexity`로 인해 확장성 문제가 있습니다.
- **임상 관련성**: 모델이 다른 텍스트 언급 간의 임상 관련성을 얼마나 잘 캡처하는지는 불분명합니다. `Multimodal deep learning`과 `지식 그래프 (knowledge graphs)` [30, 81]를 통한 `임상 지식`의 심층 주입이 필요합니다.
- **클래스 불균형 및 계층적 디코딩**: `long-tail` 분포로 인한 클래스 불균형 문제는 여전히 중요한 과제입니다. 코드 계층 구조를 활용한 `계층적 디코더` 개발과 `few-shot`, `zero-shot learning`이 필요합니다.
- **해석 가능성 (Interpretability)**: 딥러닝 모델은 여전히 `black box` 특성을 가지고 있어 예측의 투명성, 정당성, 책임성을 확보하는 `해석 가능성` 연구가 중요합니다. `attention weights` 시각화 [51]는 후처리 연구의 일환이며, `지식 기반 추론` 도입이 필요합니다.
- **Human-in-the-loop 시스템**: 인간 전문가가 모델 훈련 과정에 개입하고 모델 성능을 향상시킬 수 있는 `human-in-the-loop systems` [206]의 통합이 중요합니다. `Active learning`, 코딩 지침 주입 등이 포함됩니다.
- **업데이트된 가이드라인 및 데이터 변화**: 코딩 가이드라인은 자주 업데이트되며, 임상 관행도 시간에 따라 변할 수 있습니다 (예: 새로운 팬데믹). 모델은 `데이터 변화`에 강건해야 하며 `incremental learning` 또는 `lifelong learning` [140]이 필요합니다.
- **새로운 인코더-디코더 아키텍처 및 대규모 언어 모델 (LLMs)**: `seq2seq` 모델 [196, 6]은 의료 코드 간의 의존성을 더 잘 포착할 수 있으며, `ChatGPT` [11]와 같은 `대규모 사전 훈련 언어 모델 (LLMs)` [197, 124]의 생성 능력과 `prompt-based learning` [116, 199]을 활용하는 것은 흥미로운 방향입니다. 그러나 `hallucinated generation` 문제와 `fine-tuning`의 어려움이 있습니다 [57].
- **프라이버시 및 보안 문제**: 의료 텍스트는 민감한 환자 정보를 포함하므로, `익명화`를 통한 `프라이버시`와 `보안`이 항상 고려되어야 합니다. 모델 공유 시 `환자 특정 정보`가 역공학될 가능성을 배제할 수 없습니다.

## 📌 TL;DR

자동화된 의료 코딩은 방대한 임상 기록에서 의료 코드를 정확하게 예측하는 중요한 작업이지만, 기존 딥러닝 모델들은 `긴/잡음 있는 문서`, `고차원/불균형한 코드` 등의 문제와 `통합된 아키텍처 설계 관점 부족`에 직면했습니다. 이 논문은 이러한 과제를 해결하기 위해 `인코더-디코더`를 중심으로 하는 `통합 프레임워크`를 제안하고, `RNN`, `CNN`, `Transformer`, `그래프 신경망` 등의 `인코더`, `잔여 연결` 등의 `깊은 아키텍처 구성`, `레이블 어텐션`, `계층적/멀티태스크/few-shot/자동회귀 생성` `디코더`, 그리고 `코드 설명`, `계층 구조`, `의료 온톨로지` 등의 `보조 정보` 활용을 체계적으로 검토했습니다. 또한 `컴퓨터 지원 코딩`, `능동 학습`, `설명 가능성` 등 `인간 개입 시스템`의 중요성도 강조했습니다. `MIMIC-III` 벤치마크에서의 성능 개선과 함께, `장기 의존성`, `해석 가능성`, `클래스 불균형`, `데이터 변화`, `LLMs` 활용의 미래 연구 과제를 제시하며 이 분야의 발전을 위한 포괄적인 지침을 제공합니다.
