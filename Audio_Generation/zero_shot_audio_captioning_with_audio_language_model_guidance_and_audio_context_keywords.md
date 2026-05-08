# Zero-shot audio captioning with audio-language model guidance and audio context keywords

Leonard Salewski, Stefan Fauth, A. Sophia Koepke, Zeynep Akata (2023)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 문제는 사전 훈련 없이 오디오 콘텐츠에 대한 설명적인 텍스트 캡션을 자동으로 생성하는 **Zero-shot audio captioning**이다. 일반적인 오디오 캡셔닝은 말소리를 텍스트로 변환하는 Speech Recognition과 달리, 주변 소음이나 인간의 행동으로 인해 발생하는 환경음 등을 묘사하는 것을 목표로 한다.

기존의 오디오 캡셔닝 방법들은 대부분 지도 학습(Supervised Learning)에 의존하며, 이는 방대한 양의 오디오-텍스트 쌍 데이터셋을 필요로 한다. 이러한 방식은 새로운 오디오 문맥에 대한 적응력이 떨어지며, 계속해서 증가하는 오디오 콘텐츠의 다양성을 수용하기 어렵다는 한계가 있다. 따라서 본 연구의 목표는 작업 특화 훈련(Task-specific training) 없이도 사전 훈련된 모델들을 활용하여 고품질의 오디오 캡션을 생성하는 **ZerAuCap** 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 사전 훈련된 **Large Language Model (LLM)**의 텍스트 생성 능력과 **Audio-Language Model**의 오디오-텍스트 정렬 능력을 결합하여, LLM의 생성 과정을 두 단계로 가이드하는 것이다.

1. **Audio Context Keywords**: 오디오 신호와 유사도가 높은 키워드를 추출하여 LLM의 입력 프롬프트에 포함함으로써, LLM이 오디오 내용과 관련된 텍스트를 생성하도록 유도한다.
2. **Audio-Relevancy Guiding**: LLM이 토큰을 하나씩 생성할 때마다, 후보 토큰들이 실제 오디오 신호와 얼마나 일치하는지를 Audio-Language Model을 통해 평가하고 이를 바탕으로 최종 토큰을 선택한다.

이러한 접근 방식은 LLM의 내부 파라미터를 최적화하거나 미세 조정(Fine-tuning)하지 않고도, 추론 단계에서 가이드를 제공함으로써 계산 비용을 낮추면서 성능을 높이는 전략을 취한다.

## 📎 Related Works

오디오 캡셔닝은 주로 오디오와 텍스트를 공동 임베딩 공간으로 매핑하는 Audio-text retrieval 연구와 밀접한 관련이 있다. 기존의 많은 프레임워크들은 AudioCaps나 Clotho와 같은 데이터셋을 사용하여 지도 학습 방식으로 구축되었으나, 이는 앞서 언급한 데이터 의존성 문제를 가진다.

Zero-shot 캡셔닝 분야에서는 최근 CLIP과 같은 시각-언어 모델을 LLM과 결합한 Zero-shot image captioning 연구들이 활발히 진행되었다. 일부 연구는 LLM의 은닉 활성화 값(Hidden activations)을 최적화하거나, 디코딩 과정에서 CLIP 유사도를 통해 다음 토큰을 선택하는 방식을 제안했다.

본 논문과 동시기에 발표된 Shaharabany et al. [32]의 연구는 오디오-텍스트 매칭 점수를 통해 LLM의 Key-Value 쌍을 반복적으로 최적화하는 Zero-shot audio captioning을 제안했다. 반면, **ZerAuCap**은 모델의 내부 상태를 최적화하는 대신, 단순한 키워드 프롬프팅과 토큰 수준의 가이던스를 사용하여 훨씬 적은 계산 비용으로 더 나은 성능을 달성한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

ZerAuCap 프레임워크는 사전 훈련된 Audio-Language Model(오디오 인코더 $g_a$, 텍스트 인코더 $g_t$)과 LLM을 사용하여 훈련 없이 캡션을 생성한다. 전체 프로세스는 다음과 같은 두 가지 주요 단계로 구성된다.

### 1. Zero-shot Keyword Selection

먼저 주어진 오디오 클립 $\alpha_i$와 미리 정의된 키워드 집합 $K$ 사이의 유사도를 측정하여 가장 관련성이 높은 $l$개의 키워드 $K^*$를 선택한다. 이때 코사인 유사도(Cosine Similarity)를 사용하며 방정식은 다음과 같다.

$$K^* = \arg \text{top}-l_{k \in K} \text{CosSim}(g_a(\alpha_i), g_t(k)) = \arg \text{top}-l_{k \in K} \frac{g_a(\alpha_i) \cdot g_t(k)}{\|g_a(\alpha_i)\|_2 \cdot \|g_t(k)\|_2}$$

이렇게 선택된 키워드들은 LLM의 프롬프트에 다음과 같은 구조로 삽입된다:
`"OBJECTS: [K*]. THIS IS A SOUND OF"`
이를 통해 LLM은 오디오에 포함된 핵심 개념을 인지한 상태에서 텍스트 생성을 시작하게 된다.

### 2. Audio-Relevancy Guiding

LLM이 텍스트를 생성하는 과정에서, 단순하게 확률이 가장 높은 토큰을 선택하는 것이 아니라 오디오와의 일치성을 고려하여 재가중치(Re-weighting)를 부여한다.

1. LLM이 예측한 확률 분포에서 상위 $m$개의 후보 토큰 $c_i$를 선택한다.
2. 현재까지 생성된 문장에 후보 토큰 $c_i$를 추가한 시퀀스 $\hat{c}_i$를 구성하고, 이를 오디오 클립과 매칭하여 유사도 $f(g_a(\alpha_i), \hat{c}_i)$를 계산한다.
    $$f(g_a(\alpha_i), \hat{c}_i) = \frac{e^{\kappa \cdot \text{CosSim}(g_a(\alpha_i), g_t(\hat{c}_i))}}{\sum_{j=1}^{k} e^{\kappa \cdot \text{CosSim}(g_a(\alpha_i), g_t(\hat{c}_j))}}$$
    여기서 $\kappa$는 온도(Temperature) 하이퍼파라미터이다.
3. 최종적으로 LLM이 부여한 확률 $p_\theta$와 오디오 유사도의 가중 합이 최대가 되는 토큰 $x_t$를 선택한다.
    $$x_t = \arg \max_{i \in 1, \dots, m} [p_\theta(c_i | x_{<t}) + \beta \cdot f(g_a(\alpha_i), \hat{c}_i)]$$
    여기서 $\beta$는 오디오 가이던스의 영향력을 조절하는 가중치 계수이다. 이 과정은 문장이 마침표(.)로 끝날 때까지 반복된다.

## 📊 Results

### 실험 설정

- **LLM**: OPT (1.3B parameter) 모델을 기본으로 사용하였다.
- **Audio-Language Model**: WavCaps 사전 훈련 모델을 사용하여 유사도를 측정하였다.
- **키워드 소스**: AudioSet 클래스 리스트에서 추출한 614개의 키워드를 사용하였다.
- **데이터셋 및 지표**: AudioCaps와 Clotho 데이터셋을 사용하였으며, BLEU, METEOR, ROUGE-L, CIDEr, SPICE, SPIDER 등 표준 NLG 지표로 평가하였다.

### 정량적 결과

ZerAuCap은 Zero-shot 설정에서 기존의 Shaharabany et al. [32] 모델을 상당한 차이로 앞질렀으며, 특히 AudioCaps 데이터셋에서 새로운 SOTA(State-of-the-art) 성능을 기록하였다. 오디오 입력 없이 텍스트만 생성하는 Baseline보다 모든 지표에서 우수한 성능을 보였으며, 이는 본 모델이 오디오의 실제 내용을 효과적으로 캡처하고 있음을 시사한다. 지도 학습 기반 모델(Upper bound)과 비교했을 때도 상당 수준의 성능을 보여, 훈련 없이도 유의미한 캡션 생성이 가능함을 입증하였다.

### 정성적 결과 및 Ablation Study

정성적 평가 결과, 모델은 다양한 환경음(예: 교회 종소리, 아기 울음소리 등)을 정확하게 묘사하였다. 다만, 일부 사례에서는 캡션이 너무 짧게 생성되어 상세한 묘사가 부족한 한계가 관찰되었다.
Ablation Study를 통해 다음을 확인하였다:

- **키워드 제거 시**: 성능이 급격히 저하되어, LLM에게 오디오 개념을 미리 제공하는 것이 매우 중요함을 알 수 있다.
- **Audio-relevancy guiding 제거 시 ($\beta=0$)**: 성능이 감소하며, 토큰 생성 단계에서의 가이드가 품질 향상에 기여함을 확인하였다.

## 🧠 Insights & Discussion

본 연구는 사전 훈련된 모델들의 능력을 적절히 결합함으로써, 추가적인 학습 데이터나 파라미터 업데이트 없이도 복잡한 멀티모달 작업인 오디오 캡셔닝을 수행할 수 있음을 보여주었다. 특히 LLM의 일반적인 언어 생성 능력(World Knowledge)과 Audio-Language Model의 도메인 특화 정렬 능력을 상호 보완적으로 사용한 점이 주효했다.

다만, 본 모델은 LLM의 생성 확률과 외부 유사도 점수를 단순 합산하는 방식을 사용하므로, $\beta$ 값과 같은 하이퍼파라미터에 민감할 수 있다. 또한, 정성적 결과에서 나타났듯이 생성된 문장이 지나치게 간결한 경향이 있는데, 이는 LLM이 오디오-텍스트 매칭 점수를 극대화하기 위해 가장 확실하고 짧은 단어 위주로 선택하기 때문일 가능성이 크다. 향후 연구에서는 문장의 다양성과 상세도를 높이는 전략이 필요할 것으로 보인다.

## 📌 TL;DR

본 논문은 훈련 없이 오디오 캡션을 생성하는 **ZerAuCap** 프레임워크를 제안한다. 이 프레임워크는 (1) 오디오와 유사한 키워드를 추출하여 LLM 프롬프트에 넣는 **Keyword Selection**과 (2) 생성되는 토큰의 오디오 유사도를 실시간으로 반영하는 **Audio-Relevancy Guiding**이라는 두 단계의 가이드 방식을 사용한다. 이를 통해 AudioCaps와 Clotho 데이터셋에서 Zero-shot SOTA 성능을 달성하였으며, 이는 대규모 언어 모델과 오디오-언어 모델의 결합이 효율적인 제로샷 학습의 대안이 될 수 있음을 시사한다.
