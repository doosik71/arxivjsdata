# TO WAKE-UP OR NOT TO WAKE-UP: REDUCING KEYWORD FALSE ALARM BY SUCCESSIVE REFINEMENT

Yashas Malur Saidutta, Rakshith Sharma Srinivasa, Ching-Hua Lee, Chouchang Yang, Yilin Shen, Hongxia Jin (2023)

## 🧩 Problem to Solve

본 논문은 Keyword Spotting (KWS) 시스템에서 발생하는 False Alarm (FA), 즉 사용자가 키워드를 말하지 않았음에도 시스템이 이를 키워드로 잘못 인식하는 문제를 해결하고자 한다.

KWS 시스템은 항상 켜져 있는(always-on) 상태로 오디오 스트림을 처리하는데, 실제 키워드가 발화되는 빈도는 매우 낮기 때문에 대부분의 시간 동안은 키워드가 아닌 일반 음성이나 소음(non-speech)을 처리하게 된다. 이러한 상황에서 FA가 빈번하게 발생할 경우 다음과 같은 심각한 문제가 야기된다:

1. **프라이버시 문제**: KWS가 키워드를 감지하면 일반적으로 더 큰 규모의 클라우드 기반 자동 음성 인식(ASR) 시스템을 트리거하는데, FA로 인해 불필요한 오디오가 클라우드로 업로드될 수 있다.
2. **전력 소모 증가**: 하위 태스크의 불필요한 실행은 스마트폰이나 스마트워치와 같은 배터리 기반 장치에서 전력 효율을 저하시킨다.

따라서 본 논문의 목표는 기존 딥러닝 기반 KWS 모델의 구조를 크게 변경하지 않으면서도 FA율을 획기적으로 낮추는 범용적인 방법론을 제시하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 오디오 데이터가 가진 자연스러운 계층적 구조(Hierarchical Structure)를 활용하는 **Successive Refinement (SR)** 기법이다.

기존의 KWS 모델들은 '키워드', '키워드가 아닌 음성', '비음성(소음)'을 서로 완전히 독립적이고 동일하게 거리가 떨어진 클래스로 취급하여 다중 클래스 분류(Multi-class Classification) 문제로 접근했다. 그러나 실제로는 **[전체 오디오 $\supset$ 음성(Speech) $\supset$ 키워드(Keyword)]**의 포함 관계가 성립한다.

저자들은 이 직관을 바탕으로, 한 번에 키워드를 판별하는 것이 아니라 **(1) 음성 여부 판별 $\rightarrow$ (2) 키워드 유사성 판별 $\rightarrow$ (3) 구체적인 키워드 종류 판별**의 단계적 정제 과정을 거침으로써 FA를 효과적으로 억제할 수 있다고 제안한다.

## 📎 Related Works

기존의 딥러닝 기반 KWS 연구들은 주로 raw audio에서 특징을 추출하는 파라미터화된 방식이나, 모델의 크기를 줄여 온디바이스(on-device) 환경에 최적화하는 아키텍처 설계에 집중해 왔다. 특히 BC-ResNet, Keyword Transformer (KWT), TDNN-SWSA와 같은 모델들이 대표적이다.

FA를 줄이기 위한 일부 기존 시도들은 발화 이후의 오디오를 통해 문맥 정보를 얻거나, 정교한 비음성 Hidden Markov Model (HMM)을 사용하는 방식이 있었다. 하지만 본 논문이 제안하는 SR 방식은 특정 아키텍처에 종속되지 않고 어떤 딥러닝 KWS 모델에도 적용 가능한 'Plug-and-play' 형태의 보완적 방법론이라는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조 및 파이프라인

제안된 시스템은 공통의 **Backbone model**을 공유하고, 그 위에 세 개의 특화된 분류 브랜치(Branch)를 얹은 구조이다. 각 브랜치는 모델의 마지막 1~2개 레이어를 복제하여 구현하며, 메모리와 연산량 증가를 최소화한다.

1. **Speech Branch**: 입력 신호가 인간의 음성인지 아니면 소음인지 분류한다.
2. **Keyword-like Branch**: 입력이 음성일 때, 이것이 키워드 형태의 발화인지 아니면 일반 음성인지 분류한다.
3. **Keyword Classification Branch**: 입력이 키워드 형태일 때, 구체적으로 어떤 키워드($c_n$)인지 분류한다.

### 수학적 공식 및 확률 모델

본 방법론은 전확률 법칙(Law of Total Probability)에 근거한다. $N$개의 키워드가 있을 때, 특정 키워드 $c_n$이 나타날 확률은 다음과 같이 정의된다:

$$p(c_n|x) = p(c_n|k=1, s=1, x) \cdot p(k=1|s=1, x) \cdot p(s=1|x)$$

여기서 각 변수의 의미는 다음과 같다:

- $s=1$: 입력 $x$가 음성(Speech)일 확률
- $k=1$: 입력 $x$가 키워드 형태(Keyword-like)일 확률
- $p(c_n|...)$: 음성이며 키워드 형태일 때, 그것이 구체적으로 $c_n$일 확률

### 학습 절차 및 손실 함수

학습 시에는 각 브랜치에 해당하는 데이터만 통과시켜 계층적 감독(Hierarchical Supervision)을 수행한다.

- **Keyword Branch**: 키워드가 포함된 데이터만 입력 $\rightarrow$ $p(c_n|k=1, s=1, x)$ 학습
- **Keyword-like Branch**: 모든 음성 데이터 입력 $\rightarrow$ $p(k=1|s=1, x)$ 학습
- **Speech Branch**: 모든 데이터 입력 $\rightarrow$ $p(s=1|x)$ 학습

최종 손실 함수 $L$은 다음과 같이 세 가지 손실의 가중합으로 구성된다:
$$L = L_{softmax} + \lambda_1 L_{keyword\_branch} + \lambda_2 L_{speech\_branch}$$

- $L_{softmax}$: 키워드 종류 분류를 위한 Softmax Loss
- $L_{keyword\_branch}, L_{speech\_branch}$: 이진 분류를 위한 Weighted Focal Loss

### 추론(Inference) 절차

추론 시에는 세 브랜치의 결과값을 모두 곱하여 최종 확률 분포 $p$를 계산한다:
$$p = [p_{c_1}p_{K=1}p_{S=1}, \dots, p_{c_N}p_{K=1}p_{S=1}, p_{K=0}p_{S=1}, p_{S=0}]$$
이 분포에서 가장 높은 확률을 가진 인덱스를 선택하여 최종 결과를 결정한다.

## 📊 Results

### 실험 설정

- **데이터셋**: Google Speech Commands V2 (GSC-V2)를 기반으로 하며, 비음성 데이터를 위해 Audioset을, 일반 음성 데이터를 위해 Child-speech 및 AV-Speech 데이터셋을 사용하였다.
- **OOD 실험**: 모델이 학습 과정에서 보지 못한 환경에서의 강건성을 테스트하기 위해 MUSAN 데이터셋을 사용하여 Out-Of-Domain (OOD) FA를 측정하였다.
- **비교 모델**: BC-ResNet, Keyword Transformer (KWT), TSWSA 등 다양한 크기(13K $\sim$ 2.41M 파라미터)의 모델에 SR을 적용하였다.

### 주요 결과

- **FA 감소 효과**: SR을 적용했을 때, In-domain 데이터에서 FA율이 최대 8배, OOD 데이터에서 최대 7배까지 감소하였다.
- **정확도 유지**: FA를 획기적으로 줄이면서도 전체적인 분류 정확도(Accuracy)와 F1-score는 Baseline 모델과 비슷하거나 오히려 향상되는 결과를 보였다.
- **확장성**: 키워드의 개수가 많아질수록(10개 $\rightarrow$ 35개) SR이 FA를 억제하는 효과가 더 크게 나타났다.
- **시각화 분석**: t-SNE 분석 결과, 각 브랜치가 단계적으로 특화된 임베딩을 학습함을 확인하였다. (음성/비음성 분리 $\rightarrow$ 키워드 유사성 분리 $\rightarrow$ 개별 키워드 클러스터링)

## 🧠 Insights & Discussion

본 논문은 복잡한 다중 클래스 분류 문제를 단순한 이진 분류의 연속적인 단계로 분해함으로써 모델의 판별력을 높였다. 특히, 키워드와 일반 음성은 서로 유사하지만 비음성 소음과는 확연히 다르다는 도메인 지식을 확률 모델에 직접 반영한 점이 주효했다.

**강점**:

- **범용성**: 특정 모델 구조에 구애받지 않고 적용 가능한 'Plug-and-play' 방식이다.
- **효율성**: 추가되는 파라미터가 매우 적어 연산 오버헤드가 거의 없다.
- **강건성**: 특히 OOD 데이터에서 FA 감소 폭이 커, 실제 환경에서의 유용성이 높다.

**한계 및 논의**:

- 본 논문에서는 $\lambda_1, \lambda_2$와 같은 하이퍼파라미터를 사용하는데, 이에 대한 최적화 방법이 명시적으로 제시되지 않았다.
- 모든 모델에서 일관된 성능 향상이 있었으나, KWT-3 모델의 경우 baseline 자체가 수렴하지 않아 실험에서 제외된 점이 아쉽다.

## 📌 TL;DR

이 논문은 KWS 시스템의 고질적인 문제인 False Alarm을 줄이기 위해, 오디오의 계층적 구조(비음성 $\rightarrow$ 음성 $\rightarrow$ 키워드)를 반영한 **Successive Refinement (SR)** 기법을 제안한다. 이 방법은 기존 모델의 Backbone을 공유하면서 세 개의 특화된 분류 브랜치를 통해 단계적으로 필터링하는 구조로, 정확도 손실 없이 In-domain 및 OOD 환경에서 FA율을 최대 7~8배 감소시켰다. 이는 프라이버시 보호와 전력 효율이 중요한 온디바이스 AI 시스템에 매우 실용적인 솔루션이 될 가능성이 높다.
