# METRIC LEARNING FOR USER-DEFINED KEYWORD SPOTTING

Jaemin Jung, Youkyum Kim, Jihwan Park, Youshin Lim, Byeong-Yeol Kim, Youngjoon Jang, Joon Son Chung (2022)

## 🧩 Problem to Solve

본 논문은 사용자가 임의로 정의한 새로운 단어를 인식하는 User-defined Keyword Spotting (UD-KWS) 문제를 해결하고자 한다. 기존의 대부분의 Keyword Spotting (KWS) 연구들은 미리 정의된 키워드 집합 내에서 정답을 찾는 closed-set classification 문제로 접근해 왔다. 이러한 방식은 학습 단계에서 보지 못한(unseen) 새로운 단어에 대해서는 대응할 수 없다는 치명적인 한계가 있으며, 이는 실제 사용자 경험 측면에서 매우 제한적이다.

사용자가 직접 키워드를 설정할 수 있는 기능은 기기마다 다른 호출어를 설정하여 오작동을 방지하거나, 허가되지 않은 사용자의 접근을 막는 보안 계층을 형성하는 등 사용자 경험과 보안성을 크게 향상시킬 수 있다. 따라서 본 연구의 목표는 새로운 키워드에 대해 추가적인 학습(incremental training) 없이도 효과적으로 인식할 수 있도록, 단어의 특징을 변별력 있는 임베딩 공간으로 매핑하는 Metric Learning 기반의 학습 전략을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 KWS를 단순한 분류 문제가 아닌, 얼굴 인식이나 화자 인증과 같은 Metric Learning 문제로 재정의하는 것이다. 주요 기여 사항은 다음과 같다.

첫째, 기존의 음성 코퍼스를 활용하여 대규모 키워드 데이터셋인 LibriSpeech Keywords (LSK)를 구축하였다. 특히, Forced Aligner를 통해 추출된 데이터의 품질 저하 문제를 해결하기 위해 pre-trained speech recognition 모델을 이용한 Character Error Rate (CER) 기반의 필터링 방법을 제안하여 학습 데이터의 신뢰성을 확보하였다.

둘째, Metric Learning 기반의 2단계 학습 전략(Two-stage training strategy)을 제안하였다. 대규모의 Out-of-domain 데이터셋으로 사전 학습(Pre-training)을 수행하여 일반적인 단어 표현력을 학습시킨 후, 소규모의 In-domain 데이터셋으로 미세 조정(Fine-tuning)함으로써 사용자 정의 키워드에 대한 일반화 성능을 극대화하였다.

셋째, UD-KWS 분야의 공정한 성능 비교를 위해 Detection Error Tradeoff (DET) 곡선, Equal Error Rate (EER) 등 검출 작업에 적합한 통일된 평가 프로토콜과 지표를 제시하였다.

## 📎 Related Works

기존의 KWS 연구는 크게 분류(Classification) 방식과 검출(Detection) 방식으로 나뉜다. 분류 기반 방식은 타겟 키워드와 비타겟 소음을 구분하는 네트워크를 설계하며, 일부 연구는 마지막 선형 레이어를 교체하거나 다국어 데이터셋으로 사전 학습하여 표현력을 높이려 했다. 검출 기반 방식은 Metric Learning을 도입하여 특징 공간에서의 거리를 측정하는 방식으로, Prototypical Network 등을 활용해 소수의 예시만으로 키워드를 인식하려 했다. 하지만 기존의 검출 기반 방식들은 여전히 타겟 키워드에 맞추기 위한 추가적인 증분 학습(incremental training)이 필요하다는 한계가 있다.

또한, 데이터 부족 문제를 해결하기 위해 LibriSpeech나 Common Voice 같은 ASR 데이터셋에서 키워드를 추출하여 사용해 왔으나, 기존 연구들은 Forced Aligner의 결과물을 검증 없이 그대로 사용하거나 학습/테스트 셋 간의 키워드 중복 문제를 고려하지 않았다는 점에서 한계가 있다. 본 논문은 CER 기반 필터링을 통해 이 문제를 해결하고 엄격한 데이터 분리를 통해 실제 UD-KWS 시나리오를 더 정확하게 반영하였다.

## 🛠️ Methodology

### 전체 시스템 구조 및 학습 절차
본 시스템은 크게 데이터 구축, 사전 학습(Pre-training), 미세 조정(Fine-tuning), 그리고 추론(Inference) 단계로 구성된다. 

1. **데이터 구축**: LibriSpeech 코퍼스에서 wav2vec 2.0 모델을 사용하여 개별 단어를 Forced-align하고, 이를 통해 1,000개의 클래스를 가진 LSK 데이터셋을 구축하였다. 이때 CER 점수를 측정하여 잘못 정렬된 데이터를 제거하고, 빈도가 너무 높은 관사나 전치사, 그리고 테스트셋에 포함될 키워드를 제외하는 필터링 과정을 거쳤다.
2. **사전 학습**: 구축된 LSK 데이터셋(Out-of-domain)을 사용하여 모델이 음성 단어들을 변별력 있는 임베딩 공간에 배치하도록 학습시킨다.
3. **미세 조정**: Google Speech Commands (GSC) 데이터셋의 일부 키워드(In-domain)를 사용하여 모델을 미세 조정한다. 이는 LSK와 GSC 간의 음향적 특성 및 단어 격리 특성의 차이를 줄이기 위함이다.
4. **추론**: 등록 단계(Enrollment)에서 사용자 정의 키워드의 샘플들을 통해 프로토타입(Centroid)을 생성하고, 쿼리 입력값이 들어오면 임베딩 공간에서의 거리를 측정하여 해당 키워드인지 판별한다.

### 목적 함수 (Objective Functions)
본 논문은 세 가지 손실 함수를 비교 분석하였다.

**1. Softmax Loss**
가장 기본적인 분류 손실 함수로, 다음과 같이 정의된다.
$$L_S = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{e^{W^T x_i + b_{y_i}}}{\sum_{j=1}^{C} e^{W^T x_i + b_j}}$$
여기서 $W$와 $b$는 학습 가능한 파라미터, $x_i$는 특징 벡터, $y_i$는 클래스 레이블이다. 이 함수는 클래스 내부의 응집력이나 클래스 간의 분리도를 명시적으로 강제하지 않는다.

**2. AM-Softmax (Additive Margin Softmax)**
Softmax에 마진을 추가하여 변별력을 높인 함수이다. 먼저 가중치와 입력 벡터를 정규화하여 코사인 유사도만 고려하게 하며, 정답 클래스에 마진 $m$을 적용한다.
$$L_C = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{e^{s(\cos(\theta_{y_i,i}) - m)}}{e^{s(\cos(\theta_{y_i,i}) - m)} + \sum_{j \neq y_i} e^{s(\cos(\theta_{j,i}))}}$$
여기서 $s$는 그래디언트 소실을 방지하기 위한 스케일 팩터이다.

**3. Angular Prototypical (AP) Loss**
쿼리와 프로토타입 간의 거리를 직접 최적화하는 방식이다. 각 클래스의 프로토타입 $c_k$를 다음과 같이 계산한다.
$$c_k = \frac{1}{M-1} \sum_{i=1}^{M-1} e_{k,i}$$
이후 코사인 기반 유사도 $S_{j,k} = w \cdot \cos(e_{j,M}, c_k) + b$를 계산하고, 이를 통해 쿼리 샘플을 분류하는 손실 함수를 구성한다.
$$L_{AP} = -\frac{1}{B} \sum_{j=1}^{B} \log \frac{e^{S_{j,j}}}{\sum_{k=1}^{B} e^{S_{j,k}}}$$

### 배치 구성 (Batch Configuration)
학습 시 각 미니배치 내에서는 동일한 키워드에서 추출된 서로 다른 오디오 데이터가 Positive pair가 되며, 서로 다른 키워드 간의 쌍은 모두 Negative pair로 처리된다. Prototypical-based 네트워크의 경우, 프로토타입 계산을 위해 클래스당 최소 2개 이상의 샘플이 배치에 포함되어야 한다.

## 📊 Results

### 실험 설정
- **데이터셋**: Google Speech Commands (GSC)를 타겟 도메인으로 설정하고, 이를 Pre-defined(10개), Unknown(15개), User-defined(10개) 클래스로 나누어 사용하였다.
- **평가 지표**: False Rejection Rate (FRR)와 False Alarm Rate (FAR)의 균형점인 Equal Error Rate (EER), 특정 FAR에서의 FRR, F1-score, Accuracy를 사용하였다.
- **등록 설정**: 1-shot, 5-shot, 10-shot enrollment 환경에서 성능을 측정하였다.
- **구현**: `res15` 아키텍처를 기반으로 하며, 40차원의 MFCC를 입력으로 사용하였다.

### 주요 결과
1. **손실 함수의 영향**: 단순 Softmax나 AM-Softmax보다 AP Loss를 사용하고 2단계 학습을 적용했을 때 가장 우수한 성능을 보였다. 특히 1-shot enrollment 성능이 기존의 증분 학습 기반 베이스라인(10-shot 사용)보다 더 뛰어난 결과를 나타냈다.
2. **사전 학습의 효과**: LSK 데이터셋으로만 학습하거나 GSC로만 학습한 경우보다, LSK 사전 학습 후 GSC 미세 조정을 거친 모델의 성능이 월등히 높았다. 이는 t-SNE 시각화 결과에서도 확인되며, 2단계 학습을 거친 모델이 unseen 키워드들을 임베딩 공간에서 훨씬 더 명확하게 군집화하는 것을 보여준다.
3. **데이터 양의 영향**: 사전 학습 시 샘플 수보다 클래스의 수가 일반화 성능에 더 중요한 영향을 미친다는 점을 확인하였다. LSK에 한국어 데이터셋(KSK)을 추가하여 클래스 수를 2,000개로 늘렸을 때, 언어가 다름에도 불구하고 성능이 추가로 향상되었다.
4. **CER 필터링의 효과**: CER 기반 필터링을 적용하여 잘못 정렬된 데이터를 제거했을 때 EER이 $73.47\%$에서 $33.24\%$로 대폭 감소하여, 데이터 정제의 중요성을 입증하였다.

## 🧠 Insights & Discussion

본 논문은 UD-KWS 문제를 해결하기 위해 Metric Learning과 대규모 데이터셋을 통한 일반화 전략을 성공적으로 결합하였다. 특히 주목할 점은 **학습 데이터의 다양성(Number of Classes)**이 소수의 샘플 수보다 훨씬 더 중요한 역할을 한다는 점이다. 이는 모델이 특정 단어를 외우는 것이 아니라, "단어를 구분하는 법" 자체를 학습해야 하기 때문에 가능한 결과이다.

또한, Out-of-domain 데이터로 사전 학습을 하고 In-domain 데이터로 미세 조정을 하는 전략은, 대규모 데이터의 일반적 특징과 특정 도메인의 음향적 특성을 모두 확보할 수 있는 효율적인 방법임을 보여주었다.

다만, 본 연구는 1초로 고정된 오디오 길이를 사용하였으며, 실제 환경에서 발생할 수 있는 가변적인 단어 길이와 매우 복잡한 배경 소음 환경에 대한 강건성은 추가적인 논의가 필요해 보인다. 또한, 제안된 프로토콜이 다양한 언어 환경에서도 동일한 효율성을 갖는지에 대한 더 폭넓은 검증이 요구된다.

## 📌 TL;DR

본 연구는 사용자가 정의한 키워드를 인식하는 UD-KWS 문제를 해결하기 위해, **CER 기반 필터링을 거친 대규모 데이터셋(LSK)**과 **Angular Prototypical Loss 기반의 2단계 학습 전략**을 제안하였다. 이를 통해 추가적인 증분 학습 없이도 새로운 키워드를 효과적으로 검출할 수 있는 변별력 있는 임베딩 공간을 학습시켰으며, 기존 SOTA 모델 대비 뛰어난 성능을 달성하였다. 이 연구는 향후 개인화된 음성 인터페이스나 보안 강화형 호출어 시스템 구축에 있어 핵심적인 기반 기술이 될 가능성이 높다.