# Disentangled-Transformer: An Explainable End-to-End Automatic Speech Recognition Model with Speech Content-Context Separation

Pu Wang, Hugo Van hamme (2024)

## 🧩 Problem to Solve

현대의 End-to-End(E2E) Transformer 기반 자동 음성 인식(Automatic Speech Recognition, ASR) 시스템은 높은 인식 정확도를 보이지만, 내부 표현(representation)이 '블랙박스' 형태로 학습되어 해석 가능성(interpretability)이 떨어진다는 문제가 있다. 특히, Transformer의 인코더 층에서 학습된 표현들은 언어적 내용(speech content)뿐만 아니라 화자의 정체성(speaker identity), 방언, 억양, 감정, 배경 소음 등 다양한 특성들이 서로 복잡하게 얽혀 있는 **Entanglement(얽힘)** 현상이 발생한다.

이러한 얽힘 현상은 데이터 불일치(data mismatch)가 발생했을 때 모델의 결함이나 편향을 탐지하는 것을 어렵게 만든다. 예를 들어, 표준 화자 데이터로만 학습된 모델은 구어 장애(dysarthric) 화자나 다양한 사회언어학적 배경을 가진 화자의 음성을 처리할 때 인식 성능이 크게 요동치는데, 이는 모델이 언어적 내용과 화자 특성을 분리하지 못하고 함께 처리하기 때문이다. 따라서 본 논문의 목표는 ASR 성능을 저하시키지 않으면서, 내부 표현을 명시적인 하위 임베딩(sub-embeddings)으로 분리하여 모델의 해석 가능성을 높이는 **Disentangled-Transformer**를 제안하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 음성 신호의 서로 다른 특성이 시간적 해상도(temporal resolution)에 따라 다르게 변화한다는 직관에 기반한다. 구체적으로, '무엇을 말하는가'에 해당하는 **언어적 내용(linguistic content)**은 수십 밀리초(ms) 단위로 빠르게 변화하는 반면, '누가 말하는가'에 해당하는 **화자 정체성(speaker identity)**이나 억양 등의 특성은 상대적으로 느리게 변화한다.

이러한 시간적 행동의 차이를 이용하여, Transformer 인코더 내의 특정 Attention Head가 빠르게 변화하는 내용 임베딩(content embedding)을 캡처하게 하고, 또 다른 특정 Head는 느리게 변화하는 화자 임베딩(speaker embedding)을 캡처하도록 강제함으로써 표현의 얽힘을 해소하고자 한다.

## 📎 Related Works

기존의 음성 표현 분리(speech representation disentanglement) 연구는 주로 음성 편집이나 합성(synthesis) 작업에서 다루어졌다. 예를 들어, 화자의 성별 속성을 분리하여 개인정보를 보호하거나, 상호 정보량 학습(mutual information learning)을 통해 화자 정보를 분리하여 목소리 변환(Voice Conversion)을 수행하는 방식이 제안되었다.

그러나 기존 방식들은 주로 Adversarial VQ-VAE(Vector-Quantized Variational Autoencoder) 학습과 재구성(reconstruction) 구조를 사용하는데, 이는 음성 내용의 해상도를 저하시키거나 학습 과정이 불안정하다는 한계가 있다. 본 논문은 ASR 성능 향상을 주 목적으로 하므로, 추가적인 신호 재구성기(reconstructor)를 도입하지 않고 ASR 태스크 내에서 단순하면서도 효과적인 정규화 방식을 통해 분리를 달성한다는 점에서 기존 연구와 차별화된다.

## 🛠️ Methodology

### 1. 전체 구조 및 파이프라인

제안된 Disentangled-Transformer는 표준 Transformer 인코더의 Multi-Head Self-Attention(MHSA) 구조를 수정하여, 각 Head의 역할을 내용 캡처와 화자 캡처로 명시적으로 나눈다. 인코더의 특정 층 또는 모든 층에서 일부 Head는 내용 임베딩 $c$를, 하나의 Head는 화자 임베딩 $s$를 생성하도록 설계된다.

### 2. Time-Invariant Regularization (시간 불변 정규화)

화자 임베딩 $s$가 시간적으로 안정적으로 유지되도록 하기 위해, 학습 과정에서 다음과 같은 시간 불변 정규화 손실 함수 $L_s$를 추가한다.

$$L_s = \lambda_s \frac{1}{L} \sum_{l} \frac{1}{\sqrt{d_s}} \sum_{t} ( \sqrt{||s_{t+1} - s_t||^2} + \sqrt{||s_{t+5} - s_t||^2} )$$

여기서:

- $l$: 정규화가 적용되는 인코더 층의 인덱스이다.
- $d_s$: 화자 임베딩의 차원이다.
- $s_t$: $t$번째 프레임에서의 화자 임베딩이다.
- $\lambda_s$: 페널티 스케일(기본값 0.1)이다.

이 식은 인접한 프레임($s_{t+1}-s_t$)과 약간의 간격을 둔 프레임($s_{t+5}-s_t$) 사이의 거리(L2 norm)를 최소화함으로써, 화자 임베딩이 급격하게 변하지 않고 일정하게 유지되도록 강제한다. 특히 5프레임(약 50ms) 간격의 제약은 음소(phoneme)의 평균 발화 시간보다 길기 때문에, 내용 변화에는 영향을 주지 않으면서 화자 특성만을 안정적으로 추출하게 한다.

### 3. 학습 절차 및 손실 함수

ASR 모델은 Hybrid CTC/attention 아키텍처를 사용하며, 전체 손실 함수는 다음과 같이 정의된다.

$$L_{asr} = \alpha L_{ctc} + (1-\alpha) L_{attn} + L_s$$

여기서 $\alpha = 0.3$으로 설정되었으며, $L_{ctc}$는 CTC 손실, $L_{attn}$은 Attention 기반 교차 엔트로피 손실이다.

### 4. 화자 분리(Speaker Diarization) 모델과의 연동

분리된 임베딩의 해석 가능성을 검증하기 위해, ASR 모델의 Disentangled-Transformer 인코더를 화자 분리 모델과 공유한다. 화자 임베딩 $s$를 입력으로 받는 단순한 선형 디코더(linear decoder) 층을 추가하여 '누가 언제 말하는가'를 예측하는 이진 다중 클래스 분류 태그를 생성한다.

## 📊 Results

### 1. ASR 성능 (WER)

LibriSpeech100h 데이터셋을 사용하여 실험한 결과, Disentangled-Transformer는 Baseline Transformer와 비교하여 단어 오류율(Word Error Rate, WER) 면에서 대등하거나 오히려 향상된 성능을 보였다.

| Method | dev-clean | dev-other | test-clean | test-other |
| :--- | :---: | :---: | :---: | :---: |
| Baseline Transformer | 8.0 | 20.1 | 8.3 | 20.6 |
| Disentangled-Transformer | 7.8 | 19.6 | 8.1 | 20.0 |

이는 제안된 분리 기법이 ASR의 본래 성능을 해치지 않으면서 일종의 정규화(regularization) 효과를 제공함을 시사한다.

### 2. 화자 분리 성능 (DER)

LibriMix 데이터셋을 통해 화자 분리 성능을 평가한 결과, Disentangled-Transformer 기반 모델이 벤치마크 모델 및 단순 ASR-Transformer보다 현저히 낮은 화자 오류율(Diarization Error Rate, DER)을 기록하였다. 특히 두 화자의 음성이 완전히 겹치는 LibriMix 4.0 환경에서도 강건한 성능을 보였다.

### 3. 정성적 분석 (T-SNE 시각화)

T-SNE를 통해 임베딩을 시각화했을 때, Baseline 모델은 특정 층의 특정 Head에서만 우연히 화자 군집이 나타나는 반면, Disentangled-Transformer는 정규화가 적용된 4번째 Head에서 모든 층에 걸쳐 명확하고 일관된 화자별 군집(cluster)이 형성됨을 확인하였다.

## 🧠 Insights & Discussion

본 논문은 복잡한 VQ-VAE나 적대적 학습 없이, 단순한 시간적 제약(temporal constraint)만으로도 음성 표현의 얽힘을 효과적으로 해소할 수 있음을 보여주었다. 특히 ASR 성능 향상과 해석 가능성 확보라는 두 마리 토끼를 동시에 잡았다는 점이 강점이다.

실험 과정에서 흥미로운 점은 인코더의 깊은 층(예: 15번째 층) 하나만 교체했을 때 일부 성능 저하가 발생했다는 점이다. 이는 네트워크의 깊은 층일수록 화자 정보보다는 언어적 내용에 집중하며 화자 정보를 폐기하는 경향이 있기 때문으로 분석된다. 따라서 상위 층에 분리 기법을 적용할 때는 페널티 스케일 $\lambda_s$를 더 작게 설정하는 것이 적절할 수 있다는 통찰을 제공한다.

다만, 본 연구는 '화자 정체성'이라는 하나의 특성에 집중하여 분리를 수행하였다. 실제 음성에는 억양, 감정, 소음 등 더 다양한 컨텍스트 특성이 존재하므로, 향후 연구에서는 이러한 다양한 속성들을 어떻게 개별적으로 분리하고 제어할 수 있을지에 대한 논의가 필요할 것으로 보인다.

## 📌 TL;DR

이 논문은 Transformer 기반 ASR 모델의 내부 표현이 얽혀 있어 발생하는 블랙박스 문제를 해결하기 위해, **시간적 변화 속도가 다른 음성 내용과 화자 특성을 분리하는 Disentangled-Transformer**를 제안한다. 화자 임베딩 Head에 시간 불변 정규화(time-invariant regularization)를 적용함으로써 ASR 성능을 유지/향상시키는 동시에, 명시적인 화자 표현을 추출하여 화자 분리(Diarization) 성능을 획기적으로 높였다. 이는 향후 더 설명 가능하고 편향 없는(fair) 음성 인식 시스템을 구축하는 데 중요한 기초 연구가 될 가능성이 높다.
