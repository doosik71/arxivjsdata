# Explainable Anomaly Detection in Images and Videos: A Survey

Yizhou Wang, Dongliang Guo, Sheng Li, Octavia Camps, Yun Fu (2024)

## 🧩 Problem to Solve

본 논문은 이미지 및 비디오 데이터에서 발생하는 이상 탐지(Anomaly Detection) 모델의 '블랙박스(Black-box)' 특성으로 인해 발생하는 해석 가능성의 부재 문제를 해결하고자 한다. 딥러닝 기반의 시각적 이상 탐지 기술은 최근 급격히 발전하였으나, 모델이 왜 특정 샘플을 이상치로 판단했는지에 대한 합리적인 근거를 제시하는 설명 가능한 AI(Explainable AI, XAI) 연구는 상대적으로 부족한 실정이다.

이러한 문제의 해결은 특히 의료 영상 진단이나 산업 현장의 품질 검사, 공공 안전을 위한 보안 감시와 같이 인간의 안전과 직결된 안전 필수 도메인(Safety-critical domains)에서 매우 중요하다. 단순히 이상 여부를 판단하는 것을 넘어, 이상 부위를 정확히 국소화(Localization)하고 그 이유를 설명할 수 있어야만 사용자가 모델의 결과를 신뢰하고 실제 문제 해결(Troubleshooting)에 활용할 수 있기 때문이다. 따라서 본 논문의 목표는 이미지 및 비디오 수준의 설명 가능한 이상 탐지 방법론들을 체계적으로 분류하고 분석하여, 연구자들에게 포괄적인 가이드라인을 제공하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 시각적 이상 탐지 분야에서 '설명 가능성'에 집중한 최초의 서베이 논문이라는 점에 있다. 주요 기여 사항은 다음과 같다.

- **설명 가능한 2D 이상 탐지의 체계적 분류(Taxonomy) 제안**: 이미지 이상 탐지(IAD)와 비디오 이상 탐지(VAD)를 구분하고, 각 영역에서 사용되는 설명 방식(Attention, Perturbation, Generative model, Reasoning 등)에 따라 방법론을 분류하였다.
- **방법론 간의 상호 운용성 분석**: 이미지 기반의 설명 가능 방법론이 비디오 데이터에 적용 가능한 조건과, 반대로 비디오 전용 방법론이 이미지에 적용되기 어려운 이유를 기술적 관점에서 분석하였다.
- **데이터셋 및 평가 지표의 종합적 정리**: IAD와 VAD에서 널리 사용되는 벤치마크 데이터셋과 AUROC, IoU, PRO 등 정량적 평가 지표를 상세히 정리하였다.
- **향후 연구 방향 제시**: 설명 가능성의 정량화, 자기지도학습(Self-supervised learning) 기반 모델의 해석, 파운데이션 모델(Foundation Models)의 활용 가능성 등 미래 연구 과제를 도출하였다.

## 📎 Related Works

기존의 이상 탐지 관련 서베이 논문들은 주로 탐지 성능 향상이나 일반적인 방법론의 나열에 집중해 왔다. 일부 XAI 관련 서베이가 존재하지만, 이는 모든 데이터 타입(정형 데이터 포함)을 광범위하게 다루고 있어 2D 시각 데이터만이 가지는 특수성(공간적 구조, 시간적 연속성 등)에 기반한 심도 있는 통찰을 제공하지 못했다.

본 논문은 기존 연구들과 달리 '시각적 데이터'와 '설명 가능성'이라는 두 가지 핵심 축을 결합하여 분석한다. 특히 단순히 모델의 결과물(Attention map 등)을 보여주는 것을 넘어, 파운데이션 모델의 추론 능력을 이용한 텍스트 기반 설명 등 최신 트렌드를 반영하여 기존 서베이들의 공백을 메우고 있다.

## 🛠️ Methodology

본 논문은 설명 가능한 이상 탐지 방법론을 크게 이미지(IAD)와 비디오(VAD) 두 가지 모달리티로 나누어 분석한다.

### 1. Explainable Image Anomaly Detection (IAD)

IAD의 설명 가능 방법론은 다음과 같은 네 가지 범주로 분류된다.

- **Attention-based IAD**: 모델이 이미지의 어느 부분에 집중했는지를 시각화한다. Grad-CAM과 유사한 방식으로, 활성화 맵 $A_k$와 출력 $y$ 사이의 기울기를 이용하여 다음과 같이 어텐션 스코어 $a_k$를 계산한다.
  $$\alpha_k = \frac{1}{M} \sum_{i} \sum_{j} \frac{\partial y}{\partial A_{kij}}$$
- **Input-perturbation-based IAD**: 입력 데이터 $x$에 미세한 섭동(Perturbation)을 가했을 때 출력 스코어의 변화를 관찰하여 이상치를 설명한다. 기본 식은 다음과 같다.
  $$x' = x + \text{perturbation term}$$
- **Generative-model-based IAD**: GAN, VAE, Diffusion Model 등을 이용하여 정상 데이터의 분포를 학습한다. 정상 샘플은 잘 재구성(Reconstruction)되지만 이상치는 재구성 오차가 크게 발생한다는 점을 이용해, 입력과 출력의 차이를 통해 이상 부위를 설명한다. 특히 Likelihood Ratio(LLR)와 Likelihood Regret(LR)과 같은 확률적 지표를 사용하여 설명력을 높인다.
- **Foundation-model-based IAD**: CLIP, GPT-4V와 같은 거대 모델의 제로샷(Zero-shot) 추론 능력을 활용한다. 텍스트 프롬프트를 통해 이상 징후를 묻고 답하는 형식으로 자연어 설명을 제공하며, 이는 가장 높은 수준의 해석 가능성을 제공한다.

### 2. Explainable Video Anomaly Detection (VAD)

VAD는 시간적 차원이 추가되어 다음과 같이 분류된다.

- **Attention-based VAD**: 프레임별 이상 스코어 예측값 $S$로부터 역추적하여 원본 프레임의 활성화 맵을 생성한다.
  $$\phi(x) = \sum_{i,j} \sum_{k} w_k \cdot p_k(i,j)$$
- **Reasoning-based VAD**: 지식 그래프(Knowledge Graph), 씬 그래프(Scene Graph), 인과 추론(Causal Inference)을 활용한다. 예를 들어, (주체, 서술어, 대상)의 트리플 형태로 객체 간의 관계를 정의하여 "사람이 가방을 던졌다"와 같은 고수준의 시맨틱 설명을 제공한다.
- **Intrinsically Interpretable VAD**:
  - **Post-model interpretable**: 예측 프레임 $f(x)$와 실제 프레임 $x$의 차이를 계산하는 오차 맵을 생성한다. $\phi(x) = \theta(f(x) - x)$
  - **Attribute-based**: 속도, 포즈, 동작 등 인간이 이해할 수 있는 구체적인 속성(Attribute)을 추출하여 이 값이 정상 범위를 벗어났는지를 통해 설명한다.

## 📊 Results

### 1. 이미지 이상 탐지 결과

MVTec AD 데이터셋을 기준으로 분석한 결과, **Generative-model-based 방법론**이 가장 우수한 성능을 보였다. 특히 Diffusion 기반 모델들이 정교한 재구성 능력을 바탕으로 높은 AUROC 수치를 기록하였다.

### 2. 비디오 이상 탐지 결과

Ped2, Avenue, STC 데이터셋에서 **Intrinsically interpretable 방법론**들이 강력한 경험적 성능을 보였으며, 특히 객체의 속성을 직접 이용하는 방식이 실제 이상 상황을 효과적으로 포착하는 것으로 나타났다.

### 3. 데이터셋 및 지표 요약

- **IAD 데이터셋**: MVTec AD, VisA 등 산업용 데이터셋과 CheXpert, BRATS 등 의료 영상 데이터셋이 주로 사용된다.
- **VAD 데이터셋**: UCSD Pedestrian, Avenue, ShanghaiTech 등 보안 감시 데이터셋이 주를 이루며, 최근에는 NWPU Campus와 같이 대규모의 복잡한 데이터셋이 등장하고 있다.
- **평가 지표**: 샘플 수준에서는 AUROC, AUPR을 사용하며, 픽셀 수준에서는 IoU와 PRO(Per-Region Overlap)를 사용하여 국소화 성능을 평가한다.

## 🧠 Insights & Discussion

본 논문은 설명 가능한 이상 탐지 분야의 현주소를 진단하며 다음과 같은 통찰을 제시한다.

**강점 및 기회**:

- 파운데이션 모델(LLM/LMM)의 도입으로 인해, 과거의 단순한 히트맵(Heatmap) 방식에서 벗어나 구체적인 자연어 설명이 가능해졌다.
- 에러 기반(Error-based) 방법론은 이미지와 비디오 모두에 유연하게 적용될 수 있는 공통 분모를 가지고 있다.

**한계 및 비판적 해석**:

- **정량적 평가의 부재**: 현재 대부분의 '설명'은 시각적 예시(Qualitative examples)에 의존하고 있다. 설명이 얼마나 정확한지를 측정할 수 있는 정량적 지표와 그에 맞는 그라운드 트루스(Ground Truth) 데이터셋이 절실하다.
- **자기지도학습의 불투명성**: 최신 SOTA 모델들이 자기지도학습(Self-supervised learning)을 사용하지만, 정작 어떤 pretext task가 왜 이상 탐지 성능을 높이는지에 대한 이론적/설명적 분석은 부족하다.
- **맥락 의존성 문제**: 동일한 행동(예: 격투)이라도 장소(예: 권투 경기장 vs 교실)에 따라 정상과 이상이 갈리는 '맥락적 이상'을 처리하기 위해서는 단순한 특징 추출을 넘어선 고차원적 추론 능력이 필요하다.

## 📌 TL;DR

이 논문은 시각적 이상 탐지(IAD, VAD) 분야의 **설명 가능성(Explainability)을 다룬 최초의 종합 서베이**이다. 어텐션, 섭동, 생성 모델, 추론 기반 방법론으로 이어지는 체계적인 분류 체계(Taxonomy)를 제안하였으며, 특히 최신 파운데이션 모델을 이용한 해석 방식의 잠재력을 강조하였다. 향후 연구는 단순한 성능 향상을 넘어 **설명 가능성의 정량적 측정**과 **맥락 기반의 고차원적 추론** 방향으로 나아가야 함을 시사한다.
