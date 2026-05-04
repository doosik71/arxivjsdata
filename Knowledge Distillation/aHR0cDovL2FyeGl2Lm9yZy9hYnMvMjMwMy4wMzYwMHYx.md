# ADAPTIVE KNOWLEDGE DISTILLATION BETWEEN TEXT AND SPEECH PRE-TRAINED MODELS

Jinjie Ni, Yukun Ma, Wen Wang, Qian Chen, Dianwen Ng, Han Lei, Trung Hieu Nguyen, Chong Zhang, Bin Ma, Erik Cambria (2023)

## 🧩 Problem to Solve

본 연구는 음성 사전 학습 모델(Speech Pre-trained Model)이 텍스트 사전 학습 모델(Text Pre-trained Model)이 보유한 풍부한 언어적 지식을 전수받을 수 있도록 하는 지식 증류(Knowledge Distillation) 방법을 다룬다.

최근 자가 지도 학습(Self-supervised Learning)의 발전으로 강력한 음성 모델들이 등장했으나, 음성 데이터는 텍스트와 달리 상징적 표현(Symbolic representation)이 부족하여 언어적 정보 인코딩 능력이 상대적으로 떨어진다. 이를 해결하기 위해 텍스트 모델의 지식을 음성 모델로 전이하려는 시도가 있었으나, 다음과 같은 두 가지 핵심적인 문제로 인해 효율적인 증류가 어려웠다.

1. **Semantic Gap (의미적 격차):** 음성 신호에는 텍스트에 존재하지 않는 무음(blank)이나 노이즈가 포함되어 있으며, 모든 프레임이 동일한 의미적 중요도를 가지지 않는다.
2. **Granularity Gap (입도 격차):** 텍스트의 토큰 하나가 여러 개의 음성 프레임(Phonemes 등)에 대응되므로, 두 모달리티 간의 1:1 매핑이 불가능하다.

따라서 본 논문의 목표는 모델 구조를 변경하지 않고 소량의 데이터만으로 두 모달리티의 임베딩 공간을 정렬(Alignment)하는 Metric-based Distillation 기법인 **PAD (Prior-informed Adaptive knowledge Distillation)**를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 텍스트와 음성 간의 의미적·입도적 격차를 줄이기 위해 '중요도 기반의 사전 정보'와 '적응형 구간 집계'를 도입하는 것이다.

1. **Attention-based Significance Priors (ASP):** 트랜스포머 모델의 Self-attention map을 활용하여 각 토큰이나 프레임의 의미적 중요도를 계산한다. 이를 통해 무의미한 노이즈 프레임의 영향을 줄이고 핵심적인 의미 정보를 가진 유닛에 집중하여 정렬을 수행한다.
2. **Anchor-based Adaptive Span Aggregation (AASA):** 음성-텍스트 간의 입도 차이를 해결하기 위해, 중요도가 높은 지점을 앵커(Anchor)로 설정하고 이를 중심으로 다양한 크기의 구간(Span)을 적응적으로 생성하여 집계한다. 이를 통해 텍스트 토큰과 대응되는 적절한 길이의 음성 표현을 찾을 수 있게 한다.
3. **Metric-based Distillation 전략 분석:** Global 및 Local 정렬 전략에 PAD를 적용하여, SLU(Spoken Language Understanding) 벤치마크에서 단순한 정렬 방식보다 뛰어난 성능을 보임을 입증하였다.

## 📎 Related Works

기존의 음성-텍스트 지식 활용 방식은 크게 두 가지로 나뉜다.

1. **One-tower 접근 방식:** 단일 인코더를 통해 두 모달리티를 처리하는 방식으로, 유연하지만 대량의 데이터가 필요하거나 구조적 제약이 있을 수 있다.
2. **Two-tower 접근 방식:** 각각의 전용 인코더를 사용하며, 추가적인 레이어/코드북을 도입하거나 정렬 손실 함수(Alignment loss)를 통해 임베딩 공간을 일치시킨다.

본 연구는 Two-tower 방식 중에서도 모델 구조 변경 없이 거리 기반의 목적 함수만을 사용하는 **Metric-based Distillation**에 집중한다. 기존 연구(예: SPLAT)에서는 $L_2$ 거리나 코사인 유사도와 같은 단순한 지표를 사용했으나, 이는 앞서 언급한 의미적·입도적 격차를 무시함으로써 증류 효율을 떨어뜨리는 한계가 있었다. PAD는 이러한 한계를 ASP와 AASA를 통해 극복하며 차별점을 가진다.

## 🛠️ Methodology

### 1. 기본 정렬 전략 (Global & Local Alignment)

PAD의 기반이 되는 두 가지 정렬 방식은 다음과 같다.

* **Global-level Alignment:** 문장 전체를 대표하는 벡터 $\hat{s}$ (음성)와 $\hat{t}$ (텍스트) 사이의 $L_1$ 거리를 최소화한다.
    $$\mathcal{L}_{Glob} = ||\hat{s} - \hat{t}||_1$$
* **Local-level Alignment:** 각 텍스트 유닛 $t_j$에 대해 가장 유사한 음성 유닛 $s_i$를 찾아 그 유사도의 합을 최대화한다.
    $$\mathcal{L}_{Loc} = -\frac{1}{n} \sum_{j=0}^{n} \max_{i} \phi(s_i, t_j)$$
    여기서 $\phi$는 코사인 유사도와 같은 유사도 측정 함수이다.

### 2. Attention-based Significance Priors (ASP)

모든 유닛을 동일하게 취급하는 대신, Self-attention map에서 추출한 중요도 분포 $P^{sig}$를 사용하여 정렬을 가이드한다. 특정 유닛 $h_m$의 중요도는 모든 레이어의 어텐션 가중치의 평균으로 계산된다.
$$P^{sig}(H) = \frac{1}{L_0} \sum_{l=1}^{L_0} \frac{\mathbf{1}^\top A(H)_l^m}{\sum_{m=1}^{n} A(H)_l^m}$$

* **Global Alignment 적용:** $\hat{s}$와 $\hat{t}$를 계산할 때 이 중요도 가중치를 곱하여 의미적으로 중요한 부분만 반영한다.
* **Local Alignment 적용:** 유사도 시퀀스 $\Phi$에 중요도 가중치를 적용하여, 의미 없는 구간의 정렬 시도를 억제한다.

### 3. Anchor-based Adaptive Span Aggregation (AASA)

음성 프레임들을 적응적으로 묶어 텍스트 토큰의 입도와 맞추는 과정이다.

1. **Anchor 설정:** ASP를 통해 계산된 중요도가 높은 지점들을 앵커 포인트 $\Gamma$로 선정한다.
2. **Span 생성:** 각 앵커를 중심으로 기준 척도 $\xi$의 배수($\xi/2, \xi, 2\xi, \dots$)로 다양한 크기의 구간(Span)을 설정한다.
3. **Aggregation:** 설정된 각 구간 내의 음성 임베딩들을 풀링(Pooling)하여 집계된 표현 $\tilde{s}$를 생성한다.
4. **정렬:** 텍스트 토큰은 이제 개별 음성 프레임이 아닌, 이 적응형 스팬 풀(Span pool) 내의 최적의 스팬과 정렬된다.

## 📊 Results

### 실험 설정

* **모델:** Student는 `wav2vec 2.0-base`, Teacher는 `bert-base-uncased`를 사용하였다.
* **데이터:** 정렬 학습에는 Librispeech의 10시간 분량 데이터를 사용하였으며, 평가는 SUPERB 벤치마크를 따랐다.
* **태스크:** 의도 분류(Intent Classification, IC), 감정 인식(Emotion Recognition, ER), 슬롯 필링(Slot Filling, SF).

### 주요 결과

* **정량적 성능:** Table 1 결과에 따르면, PAD가 적용된 모든 변형(PAD-Glob, PAD-TLocal, PAD-SLocal)이 기존의 Baseline들보다 우수한 성능을 보였다.
* **태스크별 특성:**
  * **Global Alignment (PAD-Glob):** IC, ER과 같은 분류 태스크에서 강점을 보였다. 이는 분류 작업이 문장 전체의 전역적 의미를 필요로 하기 때문이다.
  * **Local Alignment (PAD-TLocal, PAD-SLocal):** SF와 같은 시퀀스 생성 태스크에서 더 좋은 성능을 보였다.
  * **Span-level Alignment (PAD-SLocal):** AASA를 통한 스팬 기반 정렬은 전역적 특성과 지역적 특성을 모두 잘 포착하여 두 종류의 태스크 모두에서 준수한 성능을 기록하였다.

### 분석 및 절제 연구 (Ablation Study)

* ASP(Significance Prior)를 제거했을 때 전반적인 성능 하락이 관찰되어, 의미적 격차를 줄이는 ASP의 효과가 입증되었다.
* AASA에서 스팬 풀을 제거하거나 앵커 포인트 대신 고정 간격(Even-stride)을 사용했을 때 성능이 하락하여, 적응형 구간 집계의 필요성이 증명되었다.

## 🧠 Insights & Discussion

본 논문은 음성과 텍스트라는 서로 다른 모달리티를 정렬할 때 단순한 거리 측정만으로는 부족하며, **의미적 중요도(Semantic Significance)**와 **입도(Granularity)**라는 두 가지 핵심 요소를 고려해야 함을 시사한다.

특히 흥미로운 점은 **Global 정렬과 Local 정렬 사이의 트레이드오프(Trade-off)**이다. 실험 결과, 두 정렬을 동시에 수행(Joint alignment)했을 때 오히려 성능이 떨어지는 경향이 발견되었다. 이는 전역적 의미 정렬과 세부 토큰 정렬이 본질적으로 서로 충돌할 수 있음을 의미한다. 다만, AASA를 통한 스팬 레벨 정렬은 두 방식의 중간적 성격을 띠어 비교적 조화로운 성능을 낸다는 점이 확인되었다.

한계점으로는, 이러한 정렬의 충돌을 완전히 해결할 수 있는 통합적인 프레임워크에 대한 논의가 부족하며, 이는 향후 연구 과제로 남겨져 있다.

## 📌 TL;DR

본 논문은 음성-텍스트 사전 학습 모델 간의 지식 증류를 위해 **PAD (Prior-informed Adaptive knowledge Distillation)**를 제안한다. Self-attention 기반의 **ASP**로 의미적 중요도를 반영하고, **AASA**를 통해 음성-텍스트 간의 입도 차이를 적응적으로 해결함으로써, 모델 구조 변경 없이도 SLU 태스크 성능을 크게 향상시켰다. 이 연구는 모달리티 간의 간극을 줄이는 정교한 정렬 전략이 음성 모델의 언어 이해 능력을 높이는 데 필수적임을 보여준다.
