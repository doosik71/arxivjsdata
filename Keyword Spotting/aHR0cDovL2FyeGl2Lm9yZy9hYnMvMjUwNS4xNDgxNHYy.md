# GraphemeAug: A Systematic Approach to Synthesized Hard Negative Keyword Spotting Examples

Harry Zhang, Kurt Partridge, Pai Zhu, Neng Chen, Hyun Jin Park, Dhruuv Agarwal, Quan Wang (2025)

## 🧩 Problem to Solve

본 논문은 음성 기반의 Keyword Spotting (KWS) 시스템에서 발생하는 'Confusable' 문구로 인한 오작동 문제를 해결하고자 한다. KWS의 핵심 성능은 키워드와 매우 유사하게 들리지만 실제로는 키워드가 아닌 경계 지역(decision boundary)의 데이터들을 얼마나 정확하게 분류하느냐에 달려 있다.

그러나 실제 환경에서 이러한 Confusable 사례(예: "Alexa"를 "All exhausted..."로 오인하는 경우)는 훈련 데이터셋에서 매우 희소하게 나타난다. 이로 인해 모델은 키워드와 음향적으로 유사한 'Hard Negative' 샘플에 취약해지며, 결과적으로 False Accept Rate(오인식률)가 높아지는 문제가 발생한다. 따라서 본 연구의 목표는 결정 경계 근처의 적대적 예제(adversarial examples)를 체계적으로 생성하여 KWS 모델의 강건성(robustness)과 정확도를 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 키워드의 자소(Grapheme) 수준에서 체계적인 변이를 가함으로써, 음향적으로 유사한 Confusable 문구를 대량으로 생성하는 **GraphemeAug** 알고리즘을 제안한 것이다. 

단순히 기존 데이터에서 유사한 단어를 찾는 것이 아니라, 텍스트 기반의 변이 후 TTS(Text-to-Speech)를 통해 오디오를 생성함으로써, 실제 데이터셋에서 발견하기 어려운 '롱테일(long-tail)' 영역의 Confusable 사례까지 포괄적으로 커버하려는 전략을 취한다. 또한, TTS 생성 시 Style Transfer 기술을 적용하여 합성 데이터의 자연스러움과 다양성을 확보하여 모델의 성능을 극대화하였다.

## 📎 Related Works

기존의 Confusable 문제 해결 방식은 크게 두 가지 방향으로 나뉜다. 첫 번째는 기존 오디오 샘플로부터 적대적 예제를 생성하는 방식이며, 두 번째는 사전에 정의된 Confusable 리스트를 바탕으로 합성 오디오를 만드는 방식이다.

본 논문에서 언급하는 기존 방식들의 한계는 다음과 같다.
1. **데이터 마이닝 방식:** ASR(Automatic Speech Recognition) 전사 데이터에서 어휘적으로 유사한 단어를 찾는 방식은 실제 오디오 데이터의 가용성에 의존하며, 드문 단어나 고유 명사의 경우 커버리지가 부족하다.
2. **자연 발생 데이터셋 활용:** 기존 데이터셋에서 자연스럽게 발생하는 Confusable을 찾는 방식은 데이터의 양이 제한적이며, 모든 발음 가능성을 체계적으로 다루지 못한다.

GraphemeAug는 TTS를 활용하여 어휘적으로 유사한 모든 가능한 조합을 생성함으로써 위와 같은 데이터 의존성 문제를 해결하고, 체계적인 커버리지를 보장한다는 점에서 차별점을 갖는다.

## 🛠️ Methodology

### 1. GraphemeAug 알고리즘
GraphemeAug는 타겟 키워드의 자소(grapheme)에 대해 다음 세 가지 핵심 편집 연산을 재귀적으로 적용하여 Confusable 텍스트를 생성한다.

- **Grapheme Addition (추가):** 키워드 내 특정 위치에 자소 하나를 삽입한다.
- **Grapheme Removal (제거):** 키워드에서 자소 하나를 제거한다.
- **Grapheme Substitution (교체):** 자소를 동일한 클래스(모음은 모음으로, 자음은 자음으로)의 다른 자소로 교체한다.

이때, 생성된 문구와 원래 키워드 간의 거리는 Levenshtein distance(편집 거리)로 측정된다. 이 알고리즘은 언어적 규칙을 따르지 않고 모든 조합을 생성하므로, 원어민이 발음하기 어려운 생소한 문구까지 포함하게 된다.

### 2. 오디오 생성 파이프라인
생성된 텍스트는 다음과 같은 과정을 거쳐 학습 데이터로 변환된다.
- **AudioLM TTS Engine:** AudioLM 기반의 모델을 사용하여 텍스트를 음성으로 변환한다. 특히 **Style Transfer** 모드를 사용하여 원본 소스 오디오의 운율(prosody)과 화자 특성을 모방함으로써 합성 데이터의 다양성을 높인다.
- **환경 시뮬레이션:** 생성된 오디오에 Room Simulation과 Noise Mixing을 적용하여 25가지의 다양한 환경 변수를 생성한다.
- **특징 추출:** 25ms 윈도우, 10ms 프레임 간격으로 Filterbank energies를 계산하며, 3개의 연속된 프레임을 쌓아 $120$차원의 입력 특징 벡터를 생성한다.

### 3. 모델 아키텍처 및 학습
- **모델 구조:** 스트리밍 추론에 최적화된 2단계 Encoder-Decoder 구조를 사용한다. 구체적으로 7개의 Factored Convolution layers (SVDF)와 3개의 Bottleneck Projection layers로 구성되며, 총 파라미터 수는 약 $320,000$개이다.
- **학습 절차:** 기본 데이터셋(Positive, Negative) 외에, Negative 샘플의 $10\%$를 GraphemeAug로 생성한 Confusable 샘플로 대체하여 학습시킨다.

## 📊 Results

### 1. 실험 설정
- **데이터셋:** 13개의 영어 소스 데이터셋(각 약 60만 개 샘플)을 활용하여 합성 데이터를 생성하였다. 
- **평가 지표:** AUC(Area Under the ROC Curve)를 주요 지표로 사용하였다.
- **평가 데이터:** 실제 사람이 말한 데이터(eval-real-pos, eval-real-neg)와 실제 Confusable 문구 데이터(eval-real-conf), 그리고 편집 거리 3으로 생성된 합성 데이터(eval-ed3)를 사용하였다.

### 2. 주요 결과
- **Style Transfer의 효과:** Style Transfer를 적용한 TTS 데이터로 학습했을 때, 표준 TTS 대비 AUC가 $22\%$ 향상되었다.
- **Confusable 학습의 효과:** 편집 거리 3의 Confusable 샘플(10,000개 유니크 문구)을 학습에 포함시킨 경우, Confusable 평가셋에서 AUC가 $61\%$ 향상되는 괄목할 만한 성과를 보였다. 또한, 이러한 학습이 일반적인 Positive/Negative 데이터에 대한 성능을 저하시키지 않음이 확인되었다.
- **데이터 다양성의 영향:** 유니크한 Confusable 문구의 수가 많을수록 성능이 크게 향상되었다. 10개를 사용했을 때보다 10,000개를 사용했을 때 AUC가 $58\%$ 더 높게 나타나, 모델이 일반화 능력을 갖추기 위해서는 대규모의 다양한 Confusable 데이터가 필수적임을 시사한다.
- **편집 거리의 영향:** 편집 거리가 너무 짧으면 키워드와 거의 동일하게 들려 False Reject가 발생할 수 있으나, 적절히 큰 편집 거리를 설정하는 것이 음향적으로 유사하면서도 구별 가능한 문구를 생성하는 데 도움이 된다.

### 3. 합성 데이터 vs 실제 데이터
- GraphemeAug로 생성한 합성 데이터로 학습한 모델은 실제 사람이 말한 Confusable 오디오(eval-real-conf)에 대해서도 AUC가 $54\%$ 향상되었다.
- 반면, 실제 Confusable 데이터를 TTS로 합성하여 학습시킨 모델은 실제 데이터에 대해서는 높은 성능을 보였으나, 합성된 Confusable 데이터(eval-real-conf 외의 조합)에 대해서는 낮은 성능($91.7\%$)을 보였다. 이는 체계적인 합성 데이터 생성이 실제 데이터가 제공하지 못하는 광범위한 커버리지를 제공함을 의미한다.

## 🧠 Insights & Discussion

본 논문은 KWS 모델의 취약점인 '음향적 유사 문구' 문제를 해결하기 위해, 텍스트 수준의 체계적 변이와 고품질 TTS 합성을 결합한 접근법이 매우 효과적임을 입증하였다.

**강점 및 시사점:**
1. **롱테일 문제 해결:** 실제 데이터 수집으로는 불가능한 방대한 양의 유사 문구를 생성함으로써, 모델이 결정 경계를 훨씬 더 정교하게 학습할 수 있게 하였다.
2. **합성 데이터의 일반화 능력:** 합성 데이터로 학습한 모델이 실제 오디오 Confusable까지 방어할 수 있다는 점은, TTS 기반 증강이 실제 환경의 강건성을 높이는 유효한 수단임을 보여준다.
3. **규모의 중요성:** 단순히 몇 가지 유사 단어를 넣는 것이 아니라, 수천 개 이상의 유니크한 변이를 제공하는 '스케일'이 모델 성능의 핵심임을 밝혀냈다.

**한계 및 논의사항:**
- 본 연구는 자소(Grapheme) 기반의 편집을 수행하였는데, 이는 언어적 규칙을 무시하므로 실제 발음과 괴리가 있는 문구가 생성될 수 있다. 논문에서도 언급되었듯, Phoneme(음소) 기반의 편집이 더 직접적인 음향적 변화를 유도할 수 있으나, 이를 위해서는 G2P(Grapheme-to-Phoneme) 모델과 음소 기반 TTS 엔진이 필요하다는 전제 조건이 있다.

## 📌 TL;DR

본 논문은 키워드의 자소를 체계적으로 변이(추가, 삭제, 교체)시켜 Hard Negative 샘플을 생성하는 **GraphemeAug** 알고리즘을 제안하였다. 이를 통해 생성된 대규모 합성 데이터를 Style Transfer TTS로 구현하여 학습시킨 결과, 키워드 오인식(False Accept)을 획기적으로 줄이면서도 기존 성능을 유지할 수 있었다. 특히, 실제 데이터보다 체계적인 합성 데이터가 더 넓은 커버리지를 제공하여 모델의 일반화 성능을 높인다는 점을 확인하였으며, 이는 향후 KWS 시스템의 강건성을 높이는 효율적인 데이터 증강 전략이 될 가능성이 높다.