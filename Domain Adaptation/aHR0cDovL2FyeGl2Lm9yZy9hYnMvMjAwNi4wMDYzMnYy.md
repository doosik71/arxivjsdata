# Neural Unsupervised Domain Adaptation in NLP—A Survey

Alan Ramponi, Barbara Plank

## 🧩 Problem to Solve

자연어 처리(NLP) 분야에서 딥 뉴럴 네트워크는 레이블된 데이터 학습에 뛰어나지만, 훈련 데이터와 테스트 데이터의 분포가 다른 **도메인 불일치(domain shift)** 상황에서는 성능이 급격히 저하됩니다. 특히 타겟 도메인의 레이블된 데이터가 전혀 없는 **비지도 도메인 적응(Unsupervised Domain Adaptation, UDA)** 환경에서 모델의 일반화 능력 부족은 큰 도전 과제입니다. 실제 시나리오에서는 레이블링 비용과 시간으로 인해 레이블된 데이터가 희소한 경우가 많으며, 이는 궁극적으로 모델이 훈련 분포를 넘어선 **분포 외 일반화(out-of-distribution generalization)**를 달성하지 못하게 하는 근본적인 문제입니다.

## ✨ Key Contributions

- NLP 분야의 신경망 기반 비지도 도메인 적응(UDA) 기법들을 포괄적으로 검토하고 분류했습니다.
- 다양한 UDA 접근 방식들의 장점과 약점을 분석하고 비교했습니다.
- 해당 분야의 현재 도전 과제와 미래 연구 방향을 제시했습니다.
- "도메인(domain)"이라는 개념을 재검토하고, 언어 데이터의 이질성을 더 잘 포착하기 위해 "다양성(variety)"이라는 용어 사용을 제안했습니다.
- 기존 NLP UDA 연구가 특정 작업(예: 감성 분석)에 편향되어 있음을 발견하고, 포괄적인 벤치마크의 필요성을 강조했습니다.

## 📎 Related Works

- **기존 도메인 적응 서베이**: 시각 애플리케이션(Csurka, 2017; Patel et al., 2015), 기계 번역(Chu and Wang, 2018), NLP의 비신경망 DA 방법(Jiang, 2008; Margolis, 2011) 등.
- **전이 학습 서베이**: Pan and Yang (2009), Weiss et al. (2016), Yang et al. (2020) 등.
- **도메인 개념**: Plank (2016)의 '다양성 공간(variety space)' 개념을 인용하여 도메인 정의의 모호성을 지적했습니다.
- **관련 문제**: 교차 언어 학습(cross-lingual learning), 도메인 일반화(domain generalization), 분포 외 일반화(out-of-distribution generalization) 등.

## 🛠️ Methodology

본 서베이는 신경망 기반 비지도 도메인 적응 기법을 크게 **모델 중심(model-centric)**, **데이터 중심(data-centric)**, 그리고 이 둘을 결합한 **하이브리드(hybrid)** 세 가지 범주로 분류하여 설명합니다.

### 모델 중심 접근 방식

모델의 특징 공간, 손실 함수, 아키텍처 또는 모델 파라미터를 수정하여 도메인 불일치를 해결합니다.

- **특징 중심 방법**:
  - **피벗 기반 DA ($\text{Pivots-based DA}$)**: 도메인 간 공통된 특징($\text{pivots}$)을 사용하여 정렬된 특징 공간을 구축합니다.
    - `SCL` (Structural Correspondence Learning) 및 `SFA` (Spectral Feature Alignment)와 같은 초기 방법이 있으며, 신경망에서는 `AE-SCL`, `PBLM`, `TRL-PBLM` 등이 제안되었습니다.
  - **오토인코더 기반 DA ($\text{Autoencoder-based DA}$)**: 오토인코더를 사용하여 도메인 간 전이가 잘 되는 잠재 표현을 비지도 방식으로 학습합니다.
    - `SDA` (Stacked Denoising Autoencoder)와 `MSDA` (Marginalized SDA)가 초기 신경망 UDA 접근 방식에 해당합니다.
- **손실 중심 방법**:
  - **도메인 적대자 ($\text{Domain Adversaries}$)**: $\text{GAN}$에서 영감을 받아 도메인 분류기의 혼란을 최대화하고, 동시에 주된 작업에 대한 정확한 예측기를 학습하여 도메인 불변 특징을 얻습니다.
    - `DANN` (Domain-Adversarial Neural Network)는 $\text{gradient reversal layer}$를 통해 도메인 불변 특징을 학습하는 가장 널리 사용되는 방법입니다.
    - `Wasserstein distance` 기반 방법은 $\text{DANN}$보다 안정적인 훈련을 제공합니다.
    - `DSN` (Domain Separation Networks)은 공유 인코더와 도메인별 사설 인코더를 분리하여 특징을 학습합니다. `GSN` (Genre Separation Networks)은 `DSN`의 변형입니다.
  - **재가중치 ($\text{Reweighting}$)**: 각 훈련 인스턴스에 타겟 도메인과의 유사도에 비례하는 가중치를 할당하여 도메인 불일치를 줄입니다.
    - `MMD` (Maximum Mean Discrepancy)와 `KMM` (Kernel Mean Matching) 등이 이 범주에 속합니다.

### 데이터 중심 접근 방식

데이터 자체를 활용하여 도메인 적응을 수행합니다.

- **가상 레이블링 ($\text{Pseudo-labeling}$)**: 훈련된 분류기가 타겟 도메인의 비레이블 데이터에 가상 레이블을 부여하고, 이 가상 레이블을 추가적인 훈련 데이터로 활용합니다.
  - `Self-training`, `Co-training`, `Tri-training` 및 `Temporal ensembling`과 같은 준지도 학습 기법이 여기에 포함됩니다.
- **데이터 선택 ($\text{Data Selection}$)**: 특정 타겟 도메인에 가장 적합한 소스 데이터를 선택하여 모델 성능을 향상시킵니다.
  - $\text{Perplexity}$나 $\text{Jensen-Shannon divergence}$와 같은 도메인 유사도 측정값을 활용합니다.
- **사전 학습 ($\text{Pre-training}$)**: 대규모 비레이블 데이터를 사용하거나 보조 작업을 통해 모델을 사전 학습시킵니다.
  - **일반 사전 학습**: `BERT`와 같은 모델을 대규모 일반 코퍼스에 훈련시킵니다.
  - **적응형 사전 학습**:
    - **다단계 사전 학습 ($\text{Multi-phase pre-training}$)**: 일반적인 데이터로 시작하여 점진적으로 도메인 특정, 나아가 작업 특정 데이터로 사전 학습을 확장합니다 (`BioBERT`, `AdaptaBERT`, $\text{DAPT}$, $\text{TAPT}$).
    - **보조 작업 사전 학습 ($\text{Auxiliary-task pre-training}$)**: 레이블된 보조 작업(intermediate tasks)을 통해 모델을 추가로 훈련시킵니다 (`STILT`).

### 하이브리드 접근 방식

모델 중심과 데이터 중심 방법의 요소를 결합하여 시너지를 창출합니다.

- 준지도 목표와 적대적 손실의 결합.
- 피벗 기반 접근 방식과 가상 레이블링 또는 문맥화된 워드 임베딩의 결합 (`SelfAdapt`, `PERL`).
- 다중 작업 학습(multi-task learning)과 도메인 불일치 적응의 결합 (`Multi-task-DA`, `Adaptive ensembling`).

## 📊 Results

- **작업 편향성**: 현재 UDA 연구는 감성 분석(sentiment analysis) 작업에 과도하게 집중되어 있으며, 이는 다른 NLP 작업에 대한 일반화 가능성 평가를 어렵게 만듭니다 (Table 1에서 "column bias"로 지적).
- **테스트 다양성 부족**: 다양한 작업 및 여러 적응 방법론에 대한 포괄적이고 체계적인 테스트가 부족합니다 (Table 1에서 "row sparsity"로 지적).
- **사전 학습 모델의 한계**: 대규모 사전 학습 모델도 도메인 외 데이터에 대해서는 성능 저하를 보이며, 최대 우도 학습(maximum likelihood training)은 모델을 과신(over-confident)하게 만들 수 있습니다. 또한, 미세 조정(fine-tuning) 과정이 때때로 취약하고($\text{brittleness}$), 훈련 데이터 순서나 초기 시드 선택에 따라 결과가 크게 달라질 수 있습니다.
- **도메인의 중요성**: "도메인" 또는 "다양성" 개념은 여전히 중요하며, 도메인 관련 데이터는 사전 학습 모델의 적응에 필수적입니다.

## 🧠 Insights & Discussion

- **"도메인" 개념 재고 및 "다양성" 제안**: 이 연구는 "도메인"이라는 느슨하게 사용되는 용어 대신, 언어 데이터의 숨겨진 다양한 언어적, 주석적 측면을 더 잘 포착하기 위해 "다양성(variety)"이라는 용어를 사용할 것을 제안합니다. 각 코퍼스는 장르, 하위 도메인, 사회인구학적 요소 등 다양한 잠재적 요인에 의해 형성되는 고차원 "다양성 공간(variety space)"의 부분 공간으로 이해되어야 합니다.
- **미래를 위한 도전 과제**:
  - **포괄적인 UDA 벤치마크**: 다양한 작업, 1:1 적응을 넘어서는 복잡한 설정, 데이터의 알려진 "다양성" 측면을 명확히 문서화한 새로운 벤치마크 개발이 시급합니다.
  - **지식 전이 메커니즘**: 고전적인 UDA 기법들이 신경망 시대에 어떻게 일반화되는지, 그리고 모델이 어떤 지식을 포착하고 전이하는지에 대한 심층적인 연구가 필요합니다.
  - **비레이블 데이터 희소성 ($\text{X scarcity}$)**: 특히 임상 데이터와 같이 매우 전문화된 언어 변이에서는 비레이블 데이터조차 구하기 어려울 수 있습니다. 이러한 극한의 데이터 희소성($\text{X scarcity}$) 또는 데이터 부재 상황에서의 적응 방법론 연구가 중요합니다.
- **분포 외 일반화**: 궁극적으로 1:1 도메인 적응 시나리오를 넘어, 알 수 없는 도메인에 대한 강력한 **분포 외 일반화** 능력 달성이 NLP의 핵심 목표가 되어야 합니다.

## 📌 TL;DR

**문제**: NLP 모델은 훈련 데이터와 다른 도메인(도메인 불일치)에서 성능이 저하되며, 특히 레이블된 타겟 데이터가 없는 비지도 도메인 적응(UDA) 환경에서의 일반화 능력이 부족합니다.
**방법**: 본 서베이는 신경망 기반 UDA 기법들을 모델 중심(특징 학습, 적대적 학습), 데이터 중심(가상 레이블링, 데이터 선택, 사전 학습), 하이브리드 접근 방식으로 포괄적으로 검토하고 분류합니다. 또한, "도메인" 개념을 "다양성"으로 재정의할 것을 제안하며, 대규모 사전 학습 모델의 적응 전략을 분석합니다.
**발견**: UDA 연구가 감성 분석에 편향되어 있고, 다양한 작업 및 방법론에 대한 포괄적인 평가가 부족함을 지적합니다. 사전 학습된 모델도 분포 외(OOD) 데이터에 취약하며, 데이터의 "다양성"을 고려하는 것이 여전히 중요함을 강조합니다. 미래에는 더 포괄적인 벤치마크와 데이터 희소성 문제 해결, 궁극적으로 분포 외 일반화가 중요함을 제시합니다.
