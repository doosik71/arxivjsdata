# Representation Learning for Electronic Health Records

Wei-Hung Weng, Peter Szolovits (2019)

## 🧩 Problem to Solve

전자 건강 기록(Electronic Health Records, EHR)은 임상 서술(clinical narratives), 검사 보고서, 실험실 측정값, 인구통계학적 정보 등 매우 다양하고 이질적인(heterogeneous) 데이터 소스로 구성되어 있다. 이러한 데이터는 본질적으로 비정형적이고 희소(sparse)하며, 시간적 불규칙성을 띠는 특성이 있다.

전통적으로는 도메인 전문가가 직접 특징을 추출하는 Expert-curated 방식(예: APACHE, SOFA score 등)을 사용해 왔으나, 이는 다음과 같은 한계가 있다:
1. **확장성 부족**: 수동으로 특징을 설계하므로 대규모 데이터셋으로 확장하기 어렵고 일반화 능력이 떨어진다.
2. **숨겨진 패턴 간과**: 복잡하고 이질적인 데이터 내부에 존재하는 잠재적인 패턴이나 새로운 지식을 포착하는 데 한계가 있다.

따라서 본 논문의 목표는 EHR 데이터를 기계 학습 알고리즘이 효율적으로 학습할 수 있는 저차원의 밀집 벡터 형태인 Representation으로 변환하는 방법론을 체계적으로 분석하고, 이를 통해 진단 지원, 위험 예측, 환자 표현형 분석(phenotyping) 등 다운스트림(downstream) 임상 과업의 성능을 향상시키는 것이다.

## ✨ Key Contributions

본 보고서(리뷰 논문)의 중심적인 기여는 EHR의 Representation Learning을 위한 전반적인 프레임워크를 제시하고, 최신 딥러닝 기법들이 어떻게 적용되는지를 분석한 점이다. 핵심 아이디어는 다음과 같다:

- **분산 표현(Distributed Representation)의 도입**: 이산적인(discrete) 데이터를 연속적인 벡터 공간으로 매핑하여 차원의 저주와 희소성 문제를 해결하고, 의미적으로 유사한 데이터가 벡터 공간에서 가깝게 위치하도록 한다.
- **계층적 및 시간적 특성 반영**: EHR의 핵심인 시간적 선후 관계(temporality)와 질병-처방 간의 계층 구조(hierarchy)를 반영하는 아키텍처(RNN, CNN, Graph-based models)를 제안한다.
- **도메인 지식 주입**: 단순한 데이터 기반 학습을 넘어, UMLS와 같은 의학 온톨로지(ontology)를 통해 전문가의 지식을 모델의 정규화(regularization)나 제약 조건으로 주입하여 해석 가능성과 강건성을 높인다.
- **임상 언어 모델의 특수성 강조**: 일반 도메인의 NLP 모델을 그대로 사용하는 대신, Clinical BERT와 같이 임상 데이터로 pre-training된 모델이 필요함을 역설한다.

## 📎 Related Works

논문에서는 Representation Learning의 발전 과정을 크게 두 단계로 구분하여 설명한다.

1. **Expert-curated Representations**:
   - **설명**: 도메인 전문가가 중요한 임상 변수를 식별하여 점수 체계를 만드는 방식이다.
   - **한계**: 해석력은 매우 높으나, 복잡한 데이터 내의 숨겨진 패턴을 찾지 못하며 새로운 데이터셋에 적용하기 위한 일반화 능력이 낮다.

2. **Learning-based Representations**:
   - **설명**: 데이터로부터 직접 특징을 학습하는 방식으로, NLP의 Bag-of-words나 n-gram부터 시작하여 최신 딥러닝 기반의 임베딩까지 발전하였다.
   - **차별점**: 전문가의 개입을 최소화하면서 데이터 내의 통계적 특성을 통해 잠재 표현을 추출하며, 특히 Deep Learning을 통해 비선형 변환을 거친 추상적인 표현 학습이 가능해졌다.

## 🛠️ Methodology

본 논문은 특정 하나의 알고리즘이 아니라 EHR Representation Learning의 전반적인 방법론을 다룬다.

### 1. Deep Learning 기반 기본 메커니즘
- **Encoder-Decoder 구조**: 입력을 잠재 공간(latent space)으로 매핑하는 Encoder와 이를 다시 타겟 공간으로 복원/변환하는 Decoder로 구성된다. Autoencoder의 경우, 입력 $x$와 출력 $\hat{x}$ 사이의 재구성 손실(reconstruction loss)을 최소화하여 핵심 특징을 추출한다.
- **시퀀스 학습**: Word2vec의 Skip-gram(중심 단어로 주변 단어 예측)과 CBOW(주변 단어로 중심 단어 예측)와 같은 메커니즘을 통해 임상 토큰 간의 의미적 관계를 학습한다.

### 2. 환자 상태 표현 학습 (Patient State Representation)
- **시간적 특성 처리**: 
  - **RNN/LSTM/GRU**: 환자의 방문 기록을 시퀀스로 처리한다.
  - **T-LSTM**: 시간 간격의 불규칙성을 해결하기 위해 Time-decay 메커니즘을 forget gate에 도입하여, 오래된 기억의 영향력을 감소시킨다.
- **계층 및 구조 학습**: 
  - **MiME**: 진단-처방 간의 다층적 구조와 상호작용을 반영하여 임베딩을 생성한다.
  - **GRAM**: ICD-9 온톨로지를 Graph-based attention 모델로 구현하여 계층적 관계를 벡터화한다.
- **도메인 지식 주입**: Graph Laplacian Regularization 등을 통해 온톨로지의 관계 정보를 모델의 가중치 학습 시 제약 조건으로 사용한다.

### 3. 임상 언어 표현 학습 (Clinical Language Representation)
- **토큰 및 개념 표현**:
  - **CUI(Concept Unique Identifier) 추출**: cTAKES나 MetaMap을 사용하여 비정형 텍스트에서 표준화된 개념(CUI)을 추출하고, 이를 기반으로 `cui2vec`와 같은 개념 레벨 임베딩을 학습한다.
- **Pre-training 및 Transfer Learning**:
  - **Clinical BERT**: 일반 BERT $\to$ BioBERT(PubMed 학습) $\to$ Clinical BERT(MIMIC-III 등 임상 노트 학습) 순으로 미세 조정(fine-tuning)하여 임상 도메인에 특화된 언어 표현을 획득한다.

## 📊 Results

본 논문은 여러 선행 연구의 결과를 종합하여 Representation Learning의 효용성을 입증한다.

- **정량적 결과**:
  - **Deep Patient**: DAE(Denoising Autoencoder) 기반의 표현이 PCA나 k-means 기반 표현보다 질병 분류 및 태깅 과업에서 더 높은 성능을 보였다.
  - **Doctor AI**: GRU 기반의 환자 방문 표현 학습이 Logistic Regression 및 MLP보다 다중 레이블 진단 예측(recall@30 지표)에서 우수한 성능을 기록하였다.
  - **Clinical BERT**: 일반 도메인 모델보다 임상 노트 기반으로 추가 학습된 모델이 임상 NLP 벤치마크 테스트에서 더 높은 성능을 나타냈다.

- **평가 지표**: 
  - 예측 과업에서는 AUROC, PR-AUC, F1-score 등을 사용하며, 정보 검색 과업에서는 MRR, MAP, nDCG 등을 활용한다.
  - 정성적 평가는 t-SNE나 UMAP과 같은 차원 축소 시각화 및 유사 사례 검색(similar case retrieval)을 통해 수행한다.

## 🧠 Insights & Discussion

### 강점 및 가치
본 논문은 EHR 데이터의 특수성(희소성, 불규칙한 시간성, 전문 용어)을 정확히 짚어내고, 이를 해결하기 위한 딥러닝 기법들을 체계적으로 정리하였다. 특히 단순한 성능 향상을 넘어, 의료 현장에서 필수적인 **해석 가능성(Interpretability)**을 위해 Attention mechanism, LIME, SHAP 등의 기법을 결합해야 함을 강조한 점이 고무적이다.

### 한계 및 미해결 과제
- **데이터 부족 및 편향**: 의료 데이터는 수집이 어렵고, 학습 데이터 자체에 내재된 모델 편향(bias)과 공정성(fairness) 문제가 존재한다.
- **프라이버시**: 적대적 공격(adversarial attack)이나 모델 스틸링으로 인한 환자 정보 유출 위험이 상존한다.
- **인과관계 결여**: 대부분의 Representation Learning은 상관관계(correlation)에 기반하며, 실제 임상 의사결정에 핵심적인 인과관계(causality) 추론은 충분히 다뤄지지 않았다.

### 비판적 해석
논문에서 제시된 수많은 모델이 높은 성능을 보였으나, 실제 임상 현장(Clinical Deployment)에 적용되기 위해서는 단순한 AUROC 수치보다 '왜 이런 예측이 나왔는가'에 대한 임상적 근거가 더 중요하다. 따라서 앞으로의 연구는 단순한 임베딩 성능 향상보다는, 전문가의 판단 과정과 유사한 Representation을 어떻게 구축할 것인가에 집중해야 할 것이다.

## 📌 TL;DR

본 논문은 비정형적이고 복잡한 **EHR 데이터를 효율적인 저차원 벡터로 변환하는 Representation Learning 기법들을 종합적으로 분석**한 리뷰 보고서이다. 수동 특징 추출에서 벗어나 **Autoencoder, RNN/Transformer, Graph-based 모델** 등을 통해 환자의 상태와 임상 언어를 학습하는 방법론을 제시하며, 특히 **임상 도메인 특화 Pre-training과 해석 가능성**의 중요성을 강조한다. 이 연구는 향후 정밀 의료 및 자동화된 임상 의사결정 지원 시스템 구축을 위한 핵심적인 기술적 토대를 제공한다.