# Multilingual Query-by-Example Keyword Spotting with Metric Learning and Phoneme-to-Embedding Mapping

Paul M. Reuter, Christian Rollwage, Bernd T. Meyer (2023)

## 🧩 Problem to Solve

본 논문은 사용자 정의 키워드를 인식할 수 있는 다국어 Query-by-Example (QbE) 기반의 Keyword Spotting (KWS) 시스템 구축을 목표로 한다. 일반적인 KWS 시스템은 특정 키워드 집합을 미리 정의하고 학습하는 closed-set 분류 문제로 접근하는데, 이는 사용자가 새로운 키워드를 설정하고 싶을 때마다 모델을 재학습시켜야 한다는 치명적인 한계가 있다. 특히 온디바이스(on-device) 환경에서 작동해야 하는 음성 비서의 경우, 계산 자원의 제약으로 인해 대규모 어휘 연속 음성 인식(LVCSR) 시스템을 사용하기 어렵다. 따라서 본 연구는 사용자가 몇 가지 예시 오디오(examples)만으로 새로운 키워드를 설정할 수 있도록 하며, 다양한 언어에 대해 일반화 성능을 갖는 소형 모델을 개발하고자 한다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 다국어 크라우드 소싱 음성 데이터와 Metric Learning을 결합하여, 재학습 없이도 높은 분별력을 가진 단어 임베딩(word embedding) 공간을 학습시킨 점이다. 구체적인 설계 아이디어는 다음과 같다.

1. **Metric Learning의 도입**: 단순 분류기가 아닌, Circle Loss를 활용한 메트릭 학습을 통해 임베딩 공간에서의 클래스 내 유사도는 높이고 클래스 간 유사도는 낮추어, 학습 시 보지 못한 단어(unseen words)에 대해서도 강건한 유사도 비교가 가능하게 하였다.
2. **Phoneme-to-Embedding (P2E) 매핑**: 사용자가 오디오 예시를 직접 녹음해야 하는 불편함을 줄이기 위해, 단어의 음소(phoneme) 시퀀스를 입력받아 KWS 모델이 학습한 임베딩 벡터를 예측하는 LSTM 모델을 제안하였다.
3. **효율적인 아키텍처 채택**: Fast-ResNet-34를 사용하여 파라미터 수를 1.4M 수준으로 낮춤으로써 온디바이스 적용이 가능한 small-footprint 시스템을 구현하였다.

## 📎 Related Works

기존의 KWS 접근 방식은 크게 LVCSR 기반, HMM 기반, 그리고 DNN 분류기 기반으로 나뉜다. LVCSR은 계산 비용과 지연 시간이 커서 온디바이스에 부적합하며, DNN 분류기는 predefined 키워드만 인식할 수 있는 폐쇄적 구조를 가진다. 이를 해결하기 위해 제안된 QbE 방식 중에는 음소 사후 확률(phoneme posterior probabilities)과 Dynamic Time Warping (DTW)을 결합한 방법이나, LSTM을 사용하여 고정 길이 벡터를 추출하는 방법이 존재한다. 또한, 최근에는 소량의 데이터로 특정 키워드에 최적화하는 Few-shot learning 기법이 제안되었으나, 본 논문은 이러한 재학습 과정 없이 임베딩 공간에서의 거리 측정만으로 작동하는 Metric Learning 방식이 더 효율적이고 일반화 성능이 높음을 강조하며 기존 방식과 차별화한다.

## 🛠️ Methodology

### 전체 파이프라인 및 시스템 구조

시스템은 크게 **특징 추출 $\rightarrow$ Fast-ResNet-34 기반 임베딩 추출 $\rightarrow$ 유사도 측정**의 단계로 구성된다. 입력 신호는 1초 길이의 오디오에서 추출된 40차원 Mel-filterbanks이며, 이를 모델에 통과시켜 고정 길이의 단어 임베딩 벡터를 생성한다.

### 주요 구성 요소 및 학습 절차

1. **모델 아키텍처**: Fast-ResNet-34를 사용한다. 이는 ResNet-34의 채널 수를 줄인 형태로, 마지막 단계에서 Temporal Average Pooling (TAP)을 통해 시간 축을 압축하여 $256 \times 1$ 크기의 최종 임베딩 벡터를 생성한다.
2. **2단계 학습 전략**:
    * **1단계 (Classification)**: 먼저 3,917개의 단어에 대해 Cross-entropy loss를 사용하여 분류기로 학습시킨다. 이는 모델이 기본적인 음성 특징을 파악하도록 하는 pre-training 단계이다.
    * **2단계 (Metric Learning)**: 분류 층을 제거하고 `conv1`부터 `conv4`까지의 가중치를 동결(freeze)한 후, `conv5`와 `fc` 층을 **Circle Loss**로 미세 조정(fine-tuning)한다.
3. **Circle Loss**: 본 논문에서 사용한 Circle Loss는 Triplet loss의 일반화된 형태로, 클래스 내 유사도와 클래스 간 유사도의 가중치를 조절하여 더 효율적인 임베딩 공간을 학습한다. 손실 함수는 하이퍼파라미터 $\gamma = 80, m = 0.4$로 설정되었다.
4. **Phoneme-to-Embedding (P2E) 모델**:
    * 입력: 단어의 음소 시퀀스 $\rightarrow$ Embedding Layer (128-dim) $\rightarrow$ 2-layer LSTM $\rightarrow$ Mean Pooling $\rightarrow$ FC Layer.
    * 목표: KWS 모델이 생성한 실제 오디오 임베딩 벡터를 예측하는 것이며, Cosine loss를 통해 학습한다.

## 📊 Results

### 실험 설정

* **데이터셋**: Common Voice (영어, 독일어, 프랑스어, 카탈루냐어) 및 Hey-Snips 데이터셋.
* **비교 대상**: 5개의 예시로 특정 키워드에 대해 미세 조정을 수행하는 Classifier-based baseline [13].
* **평가 지표**: Equal Error Rate (EER), False Negative Rate (FNR), False Alarms per hour (FA/h).

### 주요 결과

1. **분류 정확도**:
    * 학습에 사용된 언어의 새로운 단어(Out-of-vocabulary, OOV)에 대해 baseline 대비 EER을 $1.96\%$에서 $0.82\%$로 **$59.2\%$ 감소**시켰다.
    * 학습하지 않은 언어(Out-of-embedding, OOE)에 대해서도 EER을 $3.76\%$에서 $2.00\%$로 **$47.9\%$ 감소**시켜 우수한 일반화 성능을 보였다.
2. **P2E 성능**: 음소 기반 임베딩 예측 모델(P2E)의 정확도는 실제 오디오 예시 5개를 사용하여 유사도를 측정했을 때의 정확도와 매우 유사하게 나타났다. 이는 KWS 모델이 임베딩 공간에 음소 수준의 정보를 효과적으로 인코딩하고 있음을 시사한다.
3. **스트리밍 성능**: 1초 길이의 슬라이딩 윈도우(stride 0.1s)를 사용하여 테스트한 결과, Hey-Snips 데이터셋의 clean audio 환경에서 $0.1$ FA/h일 때 $5.4\%$의 FNR을 기록하였다.

## 🧠 Insights & Discussion

본 연구는 KWS 시스템에서 타겟이 아닌 단어들을 하나의 'unknown' 클래스로 묶어 학습시키는 기존의 방식보다, Metric Learning을 통해 임베딩 공간 자체를 최적화하는 것이 unseen words에 대해 훨씬 더 강력한 성능을 보인다는 점을 입증하였다. 특히 P2E 모델의 성공은 딥러닝 기반의 KWS 임베딩이 단순한 오디오 패턴 매칭을 넘어 언어학적인 음소 구조를 반영하고 있음을 보여준다.

다만, 몇 가지 한계점이 존재한다. 첫째, P2E 모델은 학습 시 사용된 음소 집합 내에서만 작동하므로, 학습되지 않은 새로운 언어의 음소에 대해서는 적용이 어렵다. 둘째, 실험에 사용된 Common Voice 데이터셋에 정렬 오류(alignment error)나 복수형 단어 등이 포함되어 있어 False Alarm rate가 실제보다 높게 측정되었을 가능성이 있다. 마지막으로, 실제 일상적인 소음 환경에서의 성능 검증이 추가적으로 필요하다.

## 📌 TL;DR

본 논문은 **Fast-ResNet-34** 아키텍처와 **Circle Loss** 기반의 메트릭 학습을 결합하여, 재학습 없이 몇 가지 예시만으로 작동하는 **다국어 Query-by-Example KWS 시스템**을 제안하였다. 특히 음소 시퀀스로부터 임베딩을 예측하는 **P2E 매핑**을 통해 오디오 녹음 없이도 키워드 설정이 가능함을 보였으며, 기존 분류 기반 방식보다 훨씬 낮은 EER을 달성하여 실용적인 온디바이스 키워드 검출 가능성을 제시하였다.
