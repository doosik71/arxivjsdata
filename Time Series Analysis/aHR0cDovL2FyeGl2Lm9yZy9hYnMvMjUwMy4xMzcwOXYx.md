# Multi-modal Time Series Analysis: A Tutorial and Survey

Yushan Jiang, Kanghui Ning, Zijie Pan, Xuyang Shen, Jingchao Ni, Wenchao Yu, Anderson Schneider, Haifeng Chen, Yuriy Nevmyvaka, Dongjin Song (2025)

## 🧩 Problem to Solve

현대 데이터 마이닝 분야에서 시계열 데이터는 텍스트, 이미지, 정형 테이블 데이터 등 다양한 모달리티(modality)와 결합되어 수집되는 경우가 많다. 이러한 Multi-modal Time Series(다중 모달 시계열) 데이터는 단순한 시간적 동역학 이상의 풍부한 문맥 정보를 제공하며, 이를 통해 시스템에 대한 포괄적인 이해와 정확한 분석이 가능하다.

그러나 다중 모달 시계열 데이터를 효과적으로 분석하는 데에는 다음과 같은 핵심적인 문제들이 존재한다.

1. **데이터 이질성(Data Heterogeneity):** 각 모달리티는 서로 다른 통계적 특성, 구조, 차원을 가지므로, 이들을 하나의 통일된 표현 공간으로 정렬하는 것이 어렵다.
2. **모달리티 갭 및 정렬 문제(Modality Gap & Misalignment):** 텍스트나 이미지 같은 문맥 데이터가 시계열 데이터와 서로 다른 타임스텝이나 시간적 입도(granularity)에서 나타나기 때문에 유의미한 상호작용을 이끌어내기 어렵다.
3. **고유 노이즈(Inherent Noise):** 실제 데이터에는 분석 목표와 무관한 불필요한 정보가 포함되어 있어, 모델이 잘못된 상관관계를 학습하게 만들 가능성이 크다.

본 논문의 목표는 이러한 도전 과제들을 해결하기 위해 제안된 최신 다중 모달 시계열 분석 방법론들을 체계적으로 정리하고, 이를 관통하는 통합된 교차 모달 상호작용 프레임워크(Unified Cross-modal Interaction Framework)를 제시하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다중 모달 시계열 분석의 복잡한 방법론들을 **상호작용 단계(Interaction Stage)**와 **상호작용 전략(Interaction Strategy)**이라는 두 가지 축으로 분류하여 체계화한 점에 있다.

1. **통합 상호작용 프레임워크 제안:** 기존의 파편화된 방법론들을 입력(Input), 중간(Intermediate), 출력(Output) 단계에서 일어나는 **융합(Fusion), 정렬(Alignment), 전이(Transference)**라는 세 가지 전략으로 범주화하였다.
2. **광범위한 방법론 및 데이터셋 카탈로그화:** 40개 이상의 다중 모달 시계열 방법론과 이에 대응하는 오픈소스 데이터셋을 체계적으로 정리하여 제공한다.
3. **도메인별 응용 사례 분석:** 헬스케어, 금융, 교통, 환경 등 다양한 실제 도메인에서 다중 모달 시계열 분석이 어떻게 적용되는지를 분석하고, 향후 연구 방향을 제시한다.

## 📎 Related Works

논문에서는 다중 모달 기계 학습(Multi-modal Machine Learning)의 전반적인 발전과 함께 시계열 특화 연구들의 한계를 지적한다.

- **일반 다중 모달 학습:** 다양한 모달리티의 통합 표현 학습과 지식 전이에 집중하지만, 시계열 데이터 특유의 시간적 의존성(temporal dependencies)을 모델링하는 데는 한계가 있다.
- **기존 시계열 관련 서베이와의 차별점:**
  - **Imaging-based transformation:** 일부 연구는 시계열을 이미지로 변환하여 시각적 모델을 적용하는 데 집중하지만, 이는 특정 모달리티에 국한된 접근이다.
  - **LLM-based reasoning:** 최신 연구들은 LLM을 이용한 추론 능력 향상에 집중하고 있으나, 전반적인 파이프라인과 통합된 원칙을 제시하는 체계적인 리뷰는 부족한 실정이다.

본 논문은 특정 모달리티나 작업에 국한되지 않고, 시계열과 다른 모달리티 간의 상호작용 원리를 일반화하여 설명한다는 점에서 기존 연구들과 차별화된다.

## 🛠️ Methodology

본 논문은 다중 모달 시계열 분석의 전체 파이프라인을 **Fusion, Alignment, Transference** 세 가지 핵심 상호작용으로 정의한다.

### 1. Fusion (융합)

이질적인 모달리티를 통합하여 상호 보완적인 정보를 캡처하는 과정이다.

- **Input-level Fusion:** 시계열, 테이블, 텍스트 데이터를 하나의 텍스트 프롬프트(Prompt)로 통합하여 LLM에 입력하는 방식이다.
- **Intermediate-level Fusion:** 각 모달리티 인코더가 데이터를 잠재 공간(latent space)으로 매핑한 후, 이를 더하거나(Addition) 연결(Concatenation)하는 방식이다.
- **Output-level Fusion:** 각 모달리티가 독립적으로 예측값을 생성하고, 이후 MLP 등을 통해 최종 출력값을 동적으로 합성하는 방식이다.

### 2. Alignment (정렬)

서로 다른 모달리티 간의 관계를 보존하고 의미적 일관성을 확보하는 과정이다.

- **Input-level Alignment:** 결측치 처리, 불규칙한 샘플링 간격 조정 등 데이터 전처리 단계에서의 시간적 정렬을 의미한다.
- **Intermediate-level Alignment:**
  - **Self-Attention:** 모든 모달리티 토큰 간의 관계를 무방향적으로 학습한다.
    $$ \text{Attention}(X_{mm}) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V $$
  - **Cross-Attention:** 시계열 데이터를 Query($Q$)로 사용하여 다른 모달리티($K, V$)로부터 관련 문맥 정보를 가져오는 방향성 정렬 방식이다.
    $$ \text{CrossAttention}(X_{ts}, X_c) = \text{softmax}\left(\frac{Q_{ts}K_c^\top}{\sqrt{d_k}}\right)V_c $$
  - **Gating Mechanism:** 파라미터화된 필터링을 통해 각 모달리티의 영향력을 조절한다.
    $$ \Gamma = \sigma(W_g[X_{ts}; X_c] + b_g), \quad X = \Gamma \odot X_{ts} + (1-\Gamma) \odot X_c $$
- **Learning Objectives:** Contrastive Learning을 통해 모달리티 간의 불변 표현(invariant representation)의 유사도를 최대화함으로써 정렬을 수행한다.

### 3. Transference (전이)

한 모달리티를 다른 모달리티로 매핑하거나 변환하는 과정이다.

- **Input-level Transference:** 시계열 데이터를 이미지나 테이블 형태로 변환하거나, LLM을 이용해 시계열에 대한 텍스트 설명을 생성하여 데이터 증강(Augmentation)으로 활용한다.
- **Intermediate/Output-level Transference:** 특정 작업을 위해 모달리티를 변환하는 것으로, 예를 들어 EEG 신호를 텍스트로 변환(EEG-to-text)하거나 시계열 데이터를 기반으로 인과 그래프(Causal Graph)를 생성하는 것이 이에 해당한다.

## 📊 Results

본 논문은 실험적 수치보다는 기존 방법론들의 분석 결과와 적용 사례를 중심으로 기술한다.

### 분석 대상 데이터셋 및 작업

- **헬스케어:** MIMIC-III, MIMIC-IV (TS, Text, Tabular) $\rightarrow$ 사망률 예측, 환자 모니터링.
- **금융:** FNSPID, DOW30 (TS, Text) $\rightarrow$ 주가 예측, 뉴스 기반 분석.
- **교통/모빌리티:** NYC-taxi, UrbanGPT (ST, Text) $\rightarrow$ 교통량 예측.
- **환경:** Terra, VIMTS (ST, Text, Image) $\rightarrow$ 결측치 보간, 환경 예측.

### 주요 분석 결과

- **LLM의 도입 효과:** 최근 연구들(Time-LLM, UniTime 등)은 시계열 데이터를 텍스트 프롬프트로 변환하거나 LLM의 reasoning 능력을 활용함으로써, 도메인 일반화 성능과 예측 해석력을 크게 향상시켰다.
- **상호작용 전략의 유효성:** 단순한 Concatenation보다는 Cross-attention이나 Gating mechanism을 적용한 모델이 모달리티 간의 불필요한 노이즈를 효과적으로 제거하고 유의미한 상관관계를 더 잘 포착하는 경향을 보인다.
- **전이 학습의 가치:** 시계열을 이미지로 변환하거나 텍스트로 설명하는 Transference 기법은 데이터가 부족한 상황에서 강력한 시맨틱 앵커(semantic anchor) 역할을 하여 성능을 높이는 것으로 나타났다.

## 🧠 Insights & Discussion

### 강점 및 시사점

본 논문은 파편화되어 있던 다중 모달 시계열 연구들을 **'단계 $\times$ 전략'**이라는 명확한 기준으로 체계화함으로써, 후속 연구자들이 자신의 문제 정의에 맞는 적절한 아키텍처를 선택할 수 있는 가이드라인을 제공했다. 특히 LLM의 등장으로 인해 시계열 분석이 단순한 수치 예측에서 문맥적 이해와 추론의 영역으로 확장되고 있음을 잘 보여준다.

### 한계 및 향후 과제

- **도메인 일반화(Domain Generalization):** 특정 도메인에서 학습된 다중 모달 모델이 다른 도메인의 분포 변화(distribution shift)에 어떻게 대응할 것인가에 대한 연구가 여전히 부족하다.
- **노이즈 및 결측치 강건성:** 실제 환경에서는 특정 모달리티의 데이터가 누락되거나 심한 노이즈가 포함된 경우가 많으며, 이를 효과적으로 처리하기 위한 모달리티별 보간 및 정제 기법이 더 필요하다.
- **윤리적 고려 및 편향:** 다중 모달 데이터(특히 텍스트)에 포함된 사회적 편향이 시계열 예측 결과에 영향을 미칠 수 있으므로, Fairness-aware 기술의 도입이 시급하다.

## 📌 TL;DR

본 논문은 다중 모달 시계열 분석 방법론을 **입력/중간/출력 단계**에서의 **융합(Fusion), 정렬(Alignment), 전이(Transference)**라는 통합 프레임워크로 체계화한 서베이 논문이다. 40개 이상의 최신 방법론과 데이터셋을 정리하였으며, 특히 LLM을 활용한 시계열 문맥 이해의 가능성을 제시하였다. 이 연구는 향후 다중 모달 시계열 모델의 설계 표준을 제시하고, 추론 가능하고 강건한 시계열 분석 시스템을 구축하는 데 중요한 기초 자료가 될 것이다.
