# A review of domain adaptation without target labels

Wouter M. Kouw, Marco Loog

## 🧩 Problem to Solve

기계 학습 모델은 훈련 데이터(소스 도메인)와 테스트 데이터(타겟 도메인)의 분포가 다를 때(데이터셋 시프트) 일반화 성능이 저하되는 문제에 직면합니다. 특히 타겟 도메인의 레이블이 전혀 없는 상황에서(비지도 도메인 적응) 어떻게 소스 도메인에서 학습한 분류기가 타겟 도메인에 잘 일반화될 수 있는지가 핵심 연구 문제입니다. 이러한 데이터 편향은 임상 연구, 의료 영상, 컴퓨터 비전, 로봇 공학, 자연어 처리 등 다양한 분야에서 나타납니다.

## ✨ Key Contributions

- 타겟 레이블이 없는 도메인 적응 접근 방식을 **샘플 기반(sample-based)**, **특징 기반(feature-based)**, **추론 기반(inference-based)**의 세 가지 주요 범주로 체계적으로 분류하고 검토합니다.
- 교차 도메인 일반화 오류에 대한 경계를 공식화하는 데 필요한 다양한 조건들을 분석하고 논의합니다.
- 도메인 적응 연구의 핵심 아이디어들을 재조명하고, 추가 연구를 위한 중요한 질문들을 제시합니다.
- 방법론의 가정을 검증하는 가설 검정의 중요성, 해석 가능성, 인과 관계의 역할, 그리고 실제 적용에서의 한계점들을 강조합니다.

## 📎 Related Works

- 통계학 및 계량 경제학의 **샘플 선택 편향(sample selection bias)** 연구를 언급합니다 [1], [2].
- 데이터셋 시프트의 유형 및 원인에 대한 연구 [30], 전이 학습(transfer learning)의 변형에 대한 연구 [32], [33]를 참조합니다.
- 시각 도메인 적응(visual domain adaptation) [24], [25], 딥러닝 기반 접근 [26], 유전체 서열 분석 [27], 생체 의료 영상 분야 [28]의 기존 검토 논문들을 언급합니다.
- **중요도 가중치 부여(importance-weighting)** [6], [7], **부분공간 정렬(subspace alignment)** [10], **최적 수송(optimal transport)** [11], **도메인 불변 표현(domain-invariant representations)** [12], **적대적 신경망(adversarial neural networks)** [119] 등 다양한 기존 기법들을 포함합니다.

## 🛠️ Methodology

본 논문은 레이블 없는 도메인 적응 방법들을 다음과 같이 세 가지 주요 범주로 분류하고 세부적인 방법들을 설명합니다.

- **샘플 기반(Sample-based) 접근 방식:**

  - **데이터 중요도 가중치(Data importance-weighting):**
    - 공변량 이동($p_S(y|x) = p_T(y|x)$) 가정을 기반으로, 타겟 분포에 대한 소스 분포의 비율인 $w(x) = p_T(x)/p_S(x)$를 추정하여 개별 샘플에 가중치를 부여합니다.
    - $R_T(h) = \sum_{y \in \mathcal{Y}} \int_{\mathcal{X}} \ell(h(x),y) \frac{p_T(x,y)}{p_S(x,y)} p_S(x,y)dx$
    - 간접 추정기(parametric, non-parametric 커널 밀도 추정)와 직접 추정기(KLIEP, LSIE, KMM, 로지스틱 회귀, Nearest-Neighbour Weighting)로 나뉩니다.
  - **클래스 중요도 가중치(Class importance-weighting):**
    - 사전 이동($p_S(x|y) = p_T(x|y)$) 가정을 기반으로, 클래스 사전 확률의 비율 $w(y) = p_T(y)/p_S(y)$를 추정하여 클래스에 가중치를 부여합니다.
    - Black Box Shift Estimation (BBSE) 및 Regularized Learning of Label Shift와 같은 방법들이 있습니다.

- **특징 기반(Feature-based) 접근 방식:**

  - **부분공간 매핑(Subspace mappings):**
    - 소스와 타겟 도메인의 공통 부분공간을 찾아 정렬하는 기법 (Subspace Alignment, Geodesic Flow Kernel, Grassmann manifold).
    - 데이터 구조를 그래프 기반 방식으로 모델링하거나 통계적 매니폴드 상의 경로를 찾는 방법도 포함됩니다.
  - **최적 수송(Optimal transport):**
    - 두 도메인의 확률 측정값(probability measures) 사이의 최소 비용 변환 맵을 찾아 소스 데이터를 타겟 데이터와 일치시킵니다.
    - Wasserstein distance ($D_W[p_S(x), p_T(x)] = \inf_{\gamma \in \Gamma} \int_{\mathcal{X} \times \mathcal{X}} d(x,z)d\gamma(x,z)$)를 최소화하는 계획을 찾습니다.
  - **도메인 불변 공간(Domain-invariant spaces):**
    - 두 도메인의 데이터를 구별할 수 없는 도메인 불변 특징 공간으로 매핑하는 방법을 학습합니다 (Transfer Component Analysis (TCA), Distribution-Matching Embedding (DME), Domain-Invariant Component Analysis (DICA)).
  - **딥 도메인 적응(Deep domain adaptation):**
    - 오토인코더(autoencoder) 또는 Domain-Adversarial Neural Network (DANN)와 같은 딥러닝 모델을 사용하여 도메인 불변 표현을 학습합니다. DANN은 레이블 분류 손실을 최소화하면서 도메인 분류 손실을 최대화합니다.
    - $D_A[x,z] = 2(1-2\hat{e}(x,z))$와 같은 프록시 A-거리(proxy A-distance)를 최소화하는 방향으로 학습합니다.
  - **대응 학습(Correspondence learning):**
    - 자연어 처리에서 "피벗 단어(pivot word)"와 같이 두 도메인 모두에서 자주 나타나는 단어를 통해 특징 간의 상호 의존성을 학습하여 공통 특징을 구축합니다.

- **추론 기반(Inference-based) 접근 방식:**
  - **알고리즘 견고성(Algorithmic robustness):**
    - 데이터 시프트에 강인하도록 분류기의 손실 변화가 제한되는 방식으로 특징 공간을 분할하는 알고리즘을 설계합니다 (예: $\lambda$-shift SVM Adaptation ($\lambda$-SVMA)).
  - **미니맥스 추정기(Minimax estimators):**
    - 최악의 시나리오(예: 불확실한 타겟 도메인의 사후 분포 또는 중요도 가중치)에서 위험을 최소화하도록 분류기를 학습합니다 (Robust Bias-Aware classifier, Target Contrastive Robust risk estimator).
  - **자기 학습(Self-learning):**
    - 소스 데이터로 분류기를 훈련한 다음, 타겟 샘플에 가상의 레이블(pseudo-labels)을 할당하고 이 가상 레이블을 사용하여 분류기를 재훈련하는 반복적인 절차입니다 (Co-training, Domain Adaptation Support Vector Machine (DASVM), Balanced Distribution Adaptation (BDA)).
  - **경험적 베이즈(Empirical Bayes):**
    - 소스 도메인 데이터를 사용하여 타겟 모델의 매개변수에 대한 정보성 사전 분포(informative prior distribution)를 형성하여 추론 과정에 도메인 지식을 통합합니다.
  - **PAC-베이즈(PAC-Bayes):**
    - PAC-학습과 베이즈 추론을 결합하여, 일반화 오류 바운드를 사용하여 학습을 안내합니다 (Gibbs classifier, Domain Adaptation for Linear Classifiers (DALC)).

## 📊 Results

- 본 검토 논문은 도메인 적응에 대한 광범위한 접근 방식들을 체계적으로 분류하고 각 범주 내의 세부 방법들을 제시함으로써, 이 분야의 복잡성을 명확히 합니다.
- 특히, 동등한 사후 분포($p_S(y|x) = p_T(y|x)$)나 조건부 불변 구성 요소($p_S(t(x)|y) = p_T(t(x)|y)$)와 같은 특정 조건들이 충족될 때 일반화 오류에 대한 이론적인 경계가 존재함을 밝힙니다.
- 이러한 경계는 도메인 불일치 측정치($D_{\mathcal{H}\Delta\mathcal{H}}$, Rényi divergence 등)와 샘플 크기, 분류기 복잡성 등에 따라 변화함을 보여줍니다.
- 예시 시나리오(심장 질환 진단)를 통해 중요도 가중치, 특징 변환, 추론 기반 방법이 분류기의 결정 경계를 어떻게 변화시키는지 시각적으로 보여줍니다.

## 🧠 Insights & Discussion

- **가정의 검증과 No-Free-Lunch:** 공변량 이동이나 사전 이동과 같은 도메인 간 관계에 대한 가정은 타겟 레이블 없이는 검증하기 어렵습니다. 따라서 특정 방법이 주어진 데이터셋에서 얼마나 잘 작동할지 예측하는 것은 불가능하며, 도메인 적응에도 "No-Free-Lunch" 정리가 적용됩니다 [203].
- **해석 가능성(Interpretability):** 방법의 내부 메커니즘을 쉽게 검사하고 성공 또는 실패 이유에 대한 직관을 얻는 것이 중요합니다. 이는 도메인 적응에 대한 이해를 깊게 하고 새로운 통찰력을 제공합니다.
- **탐색 공간 축소:** 소스 도메인 지식은 타겟 도메인 학습을 위한 매개변수 탐색 공간을 줄이는 데 도움이 되지만, 부적절한 경우 "음성 전이(negative transfer)"를 유발할 수 있습니다 [179].
- **다중 사이트 연구(Multi-site studies):** 도메인 적응 방법은 다양한 연구 그룹이나 환경에서 수집된 데이터의 "배치 효과(batch effects)"를 제거하고 데이터를 통합하는 데 유용한 도구입니다.
- **인과 관계(Causality):** 변수 간의 인과 구조에 대한 지식은 도메인 시프트의 유형을 명확히 하고, 보다 합리적인 가정을 세우며, 궁극적으로 적응 전략 선택을 자동화하는 데 기여할 수 있습니다. 그러나 현재 인과 추론 절차는 대규모 변수에 대해 확장성이 제한적입니다.
- **제한 사항:**
  - 중요도 가중치 부여는 고차원 환경에서 "차원의 저주(curse of dimensionality)"로 인해 어려움을 겪습니다.
  - 많은 방법이 타겟 분포의 서포트(support)가 소스 분포의 서포트 안에 포함되어야 한다고 가정합니다. 그렇지 않을 경우 병리학적 해가 발생할 수 있습니다.
  - 분포의 서포트를 가깝게 만드는 변환과 서포트 요구 사항을 포함하는 방법을 결합하는 것은 복잡하며, 후방 분포 또는 조건부 분포의 등가성을 깨뜨릴 수 있습니다.
  - 고차원 환경에서는 데이터 분포를 정렬하는 많은 변환이 있을 수 있지만, 후방 또는 클래스 조건부 분포는 정렬하지 못하여 특정 응용 분야에 대한 휴리스틱과 도메인 지식이 필요합니다.
  - 문헌의 불일치하는 용어 사용(예: "unsupervised domain adaptation", "transductive transfer learning", "covariate shift")은 연구를 어렵게 만듭니다.

## 📌 TL;DR

**문제:** 기계 학습 모델은 훈련 데이터(소스 도메인)와 테스트 데이터(타겟 도메인)의 분포가 다를 때 성능 저하를 겪으며, 특히 타겟 도메인의 레이블이 없는 경우 일반화가 어렵습니다.

**해결책:** 이 검토 논문은 레이블 없는 도메인 적응 방법들을 **샘플 기반(개별 관측치 가중치 부여)**, **특징 기반(특징 공간 변환/매핑)**, **추론 기반(매개변수 추정 과정에 적응 통합)**의 세 가지 핵심 범주로 분류하여 설명합니다. 각 범주 내에는 중요도 가중치, 부분공간 매핑, 최적 수송, 딥 도메인 적응, 자기 학습, 미니맥스 추정기 등 다양한 세부 기법들이 포함됩니다.

**주요 발견:** 이 논문은 적응형 분류기의 일반화 오류 경계를 설정하는 데 필요한 조건들(예: 동등 사후 분포)을 분석하고, 방법론적 가정을 검증하는 가설 검정의 중요성, 해석 가능성, 그리고 인과 관계가 도메인 적응 전략 선택에 미칠 수 있는 영향에 대해 논의합니다. 궁극적으로, 도메인 적응은 데이터가 독립적이고 동일하게 분포되어 있다는 표준 기계 학습 가정을 벗어나, 다양한 과학 및 공학 분야에 필수적인 중요한 연구 분야임을 강조합니다.
