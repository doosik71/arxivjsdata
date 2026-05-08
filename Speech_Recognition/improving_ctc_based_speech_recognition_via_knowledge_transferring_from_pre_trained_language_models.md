# IMPROVING CTC-BASED SPEECH RECOGNITION VIA KNOWLEDGE TRANSFERRING FROM PRE-TRAINED LANGUAGE MODELS

Keqi Deng, Songjun Cao, Yike Zhang, Long Ma, Gaofeng Cheng, Ji Xu, Pengyuan Zhang (2022)

## 🧩 Problem to Solve

본 논문은 Connectionist Temporal Classification (CTC) 기반의 자동 음성 인식(Automatic Speech Recognition, ASR) 모델이 가진 구조적 한계를 해결하고자 한다.

CTC 기반 모델은 조건부 독립 가정(conditional independence assumption)으로 인해 디코딩 속도가 매우 빠르고 병렬 처리가 가능하다는 장점이 있다. 그러나 이러한 가정 때문에 문맥적 정보를 충분히 활용하지 못하며, 이로 인해 Attention 기반의 Encoder-Decoder (AED) 모델보다 인식 성능이 낮다는 치명적인 단점이 존재한다.

일반적으로 이를 해결하기 위해 외부 언어 모델(External Language Model, LM)을 사용하지만, 이는 빔 서치(beam search) 과정에서 모델을 자기회귀(Auto-Regressive, AR) 방식으로 동작하게 만들어, CTC 모델 본연의 장점인 빠른 비자기회귀(Non-Auto-Regressive, NAR) 디코딩 속도를 상실하게 만든다. 따라서 본 연구의 목표는 추론 단계에서 추가적인 파라미터를 도입하지 않으면서, 사전 학습된 언어 모델(Pre-trained LM)의 지식을 CTC 모델에 전이(Knowledge Transferring)하여 성능을 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 BERT와 GPT2 같은 강력한 사전 학습 언어 모델이 가진 텍스트 모달리티의 문맥 지식을 CTC 기반 ASR 모델에 이식하는 것이다. 이를 위해 저자들은 두 가지 지식 전이 방법을 제안한다.

1. **KT-RL (Knowledge Transferring based on Representation Learning):** BERT의 양방향 문맥 표현(Representation)을 보조 학습 목표로 설정하여, ASR 모델의 인코더가 텍스트의 특징을 학습하도록 유도하는 표현 학습 기반 방법이다.
2. **KT-CL (Knowledge Transferring based on Classification Learning):** GPT2를 활용하여 텍스트 모델링을 수행하고, 이를 하이브리드 CTC/Attention 구조와 결합하여 공동 훈련(Joint Training)을 통해 지식을 전이하는 분류 학습 기반 방법이다.

## 📎 Related Works

최근 wav2vec2.0과 같은 자기지도 학습(Self-supervised learning) 방식이 ASR 성능을 크게 향상시켰으나, 여전히 CTC 모델의 독립성 가정 문제는 남아 있다. 기존 연구들 중 일부는 BERT의 지식을 AED 기반 ASR 시스템으로 전이하려는 시도를 하였으나, AED 모델은 이미 디코더를 통해 텍스트 정보를 어느 정도 활용할 수 있기 때문에 성능 향상 폭이 제한적이었다.

반면, CTC 기반 모델은 텍스트 모달리티를 활용할 수 있는 디코더가 아예 없으므로, 사전 학습된 LM의 지식을 전이하는 것이 훨씬 더 효과적일 것이라는 것이 본 논문의 차별점이다.

## 🛠️ Methodology

본 연구는 wav2vec2.0을 기본 CTC ASR 시스템으로 사용하며, 구체적인 두 가지 전이 방법론은 다음과 같다.

### 1. KT-RL (Representation Learning 기반 지식 전이)

BERT의 고정된 파라미터를 사용하여 텍스트 표현을 추출하고, 이를 ASR 모델의 출력과 정렬시켜 학습시킨다. 텍스트와 음성의 길이는 서로 다르므로, 이를 맞추기 위해 두 가지 정렬 방식을 제안한다.

**A. KT-RL-CIF (CIF 기반 정렬)**
Continuous Integrate-and-Fire (CIF) 메커니즘을 사용하여 음성과 텍스트 간의 단조 정렬(Monotonic Alignment)을 구현한다.

- 먼저 FC 레이어 이후의 최대값에 시그모이드 함수를 적용하여 가중치 $w_m$을 구한다.
- $$w_m = \text{sigmoid}(\max(\text{FC}(h_m)))$$
- 이를 전체 시간 축에 대해 누적하며, 누적 값이 1을 넘을 때마다 하나의 언어적 표현 $l_n$을 출력한다.
- BERT에서 추출한 타겟 임베딩 $e_n$과 추출된 $l_n$ 사이의 코사인 유사도 손실(Cosine Embedding Loss)을 계산한다.
- $$\mathcal{L}_{cos} = k \cdot \sum_{n=0}^{N} (1 - \cos(l_n, e_n))$$
- 최종 학습 목표는 $\mathcal{L}_{mtl} = \lambda \mathcal{L}_{ctc} + (1-\lambda) \mathcal{L}_{cos}$ 이다.

**B. KT-RL-ATT (Attention 기반 정렬)**
병렬 효율성을 높이기 위해 Multi-head Cross Attention을 사용한다.

- 타겟 길이 $N$에 맞는 위치 임베딩(Positional Embedding) $P$를 쿼리(Query)로 사용하고, wav2vec2.0의 출력 $H$를 키(Key)와 값(Value)으로 사용하여 언어적 표현 $L$을 추출한다.
- 이후 손실 함수 계산 방식은 KT-RL-CIF와 동일하다.

### 2. KT-CL (Classification Learning 기반 지식 전이)

GPT2를 사용하여 이전 토큰들의 텍스트 표현 $G$를 추출하고, 이를 ASR 인코더 출력 $H$와 Cross Attention으로 결합한다.

- GPT2의 출력 $G$가 쿼리가 되고, 음성 인코더 출력 $H$가 키와 값이 된다.
- 미래 토큰을 보지 못하도록 Subsequent Mask를 적용한다.
- Cross Attention의 결과물 $O$를 선형 분류기에 통과시켜 타겟과 비교하는 Cross Entropy Loss $\mathcal{L}_{ce}$를 계산한다.
- 최종 학습 목표는 $\mathcal{L}_{mtl} = \beta \mathcal{L}_{ctc} + (1-\beta) \mathcal{L}_{ce}$ 이다.

## 📊 Results

### 실험 설정

- **데이터셋:** Mandarin AISHELL-1 코퍼스.
- **베이스라인:** Vanilla wav2vec2.0 CTC 모델.
- **평가 지표:** Character Error Rate (CER).
- **학습 환경:** Adam optimizer, 20 epochs, 8s M40 GPUs.

### 주요 결과

실험 결과, 제안된 방법론이 베이스라인 대비 상당한 성능 향상을 보였다. 특히 외부 LM을 사용하지 않았을 때의 효과가 두드러졌다.

- **정량적 결과:** KT-RL-CIF 방법이 테스트 셋에서 **4.2% CER**을 달성하였다.
- **상대적 향상:** 외부 LM 없이 wav2vec2.0 CTC 베이스라인과 비교했을 때 **CER이 상대적으로 16.1% 감소**하였다.
- **비교 분석:** 외부 LM을 사용한 베이스라인보다, 제안 방법(KT-RL-CIF)을 적용하고 외부 LM을 사용하지 않은 NAR 시스템이 더 높은 성능을 보였다. 이는 모델 자체에 문맥 지식이 성공적으로 내재화되었음을 의미한다.

## 🧠 Insights & Discussion

**1. 양방향 vs 단방향 지식의 영향**
KT-RL(BERT 기반)이 KT-CL(GPT2 기반)보다 우수한 성능을 보였다. 이는 ASR 모델이 텍스트의 양방향 문맥 정보를 학습하는 것이 단방향 정보보다 훨씬 유리함을 시사한다.

**2. CIF vs Attention 정렬**
KT-RL-CIF가 KT-RL-ATT보다 정확도가 높았다. Attention 메커니즘은 너무 유연하여 특정 단어의 지식이 해당 단어에 속하지 않는 프레임으로 전이될 가능성이 있는 반면, CIF는 단조 정렬을 강제함으로써 더 정확한 지식 전이가 가능했다. 다만, 학습 속도는 병렬 처리가 가능한 KT-RL-ATT가 훨씬 빨랐다.

**3. 손실 함수의 선택**
MSE Loss보다 Cosine Embedding Loss가 더 효과적이었다. 이는 표현 학습 과정에서 벡터의 절대적인 거리보다 벡터의 방향(Angle)이 더 중요한 정보임을 의미한다.

**4. 한계 및 해석**
본 연구는 추론 시 추가 파라미터를 사용하지 않는다는 점을 강조했으나, 학습 단계에서는 BERT/GPT2와 같은 거대 모델을 사용하므로 학습 비용이 증가한다. 하지만 한 번 학습된 후에는 순수 CTC 구조만 사용하므로 추론 효율성은 그대로 유지된다는 강점이 있다.

## 📌 TL;DR

본 논문은 CTC 기반 ASR 모델의 고질적인 문제인 문맥 정보 부족을 해결하기 위해, 사전 학습된 언어 모델(BERT, GPT2)의 지식을 전이하는 두 가지 방법(KT-RL, KT-CL)을 제안하였다. 특히 BERT의 양방향 표현을 CIF 메커니즘으로 정렬하여 전이하는 **KT-RL-CIF** 방식이 가장 뛰어난 성능을 보였으며, 이를 통해 외부 LM 없이도 CER을 상대적으로 16.1% 낮추는 성과를 거두었다. 이 연구는 추론 속도를 유지하면서도 AED 모델에 근접하는 성능을 내는 고속 NAR ASR 시스템 구축 가능성을 제시하였다.
