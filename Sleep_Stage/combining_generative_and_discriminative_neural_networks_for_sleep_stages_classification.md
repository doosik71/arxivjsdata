# Combining Generative and Discriminative Neural Networks for Sleep Stages Classification

Endang Purnama Giri, Mohamad Ivan Fanany, Aniati Murni Arymurthy (2016)

## 🧩 Problem to Solve

본 연구가 해결하고자 하는 문제는 수면 다원 검사(Polysomnography, PSG) 데이터로부터 수면 단계(Sleep Stages)를 자동으로 분류하는 것이다. 수면 단계 패턴은 수면 장애의 존재를 진단하는 데 중요한 단서를 제공하지만, EEG(뇌전도), EOG(안전도), EMG(근전도) 신호를 수동으로 모니터링하고 분석하는 것은 매우 어렵고 비실용적이다.

따라서 본 논문의 목표는 생성 모델(Generative Model)의 특징 추출 능력과 판별 모델(Discriminative Model)의 시퀀스 패턴 인식 능력을 결합하여, 기존의 수동 분석이나 단순한 자동 분류 방식보다 높은 정확도를 가진 수면 단계 분류 모델을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Deep Belief Network(DBN)의 생성 능력**과 **Long Short-Term Memory(LSTM)의 판별 및 시퀀스 인식 능력**을 결합한 하이브리드(Hybrid) 모델을 구축하는 것이다.

- **DBN의 역할**: 입력된 특징 데이터로부터 고수준의 계층적 특징을 자동으로 생성하는 특징 생성기(Feature Generator) 역할을 수행한다.
- **LSTM의 역할**: DBN이 추출한 특징들의 시퀀스(Sequence) 데이터를 분석하여 최종 수면 단계를 예측하는 판별기 역할을 수행한다.
- **시퀀스 특성 활용**: 수면 단계의 전환은 특정 패턴을 가지는 시계열 데이터라는 점에 착안하여, 단일 인스턴스 분류가 아닌 시퀀스 기반의 분류 체계를 적용하였다.

## 📎 Related Works

기존의 수면 단계 분류 연구들은 다음과 같은 접근 방식을 취해왔다.

- **얕은 분류기(Shallow Classifiers)**: SVM이나 일반적인 Neural Network를 사용한 연구들이 있었으나, 데이터의 양이 증가함에 따라 정확도가 저하되는 경향을 보였다.
- **DBN 및 HMM 결합**: 일부 연구에서 DBN을 통해 특징을 학습하고 Hidden Markov Model(HMM)을 결합하여 시퀀스를 처리하려 했다.
- **최신 기술(State of the Art)**: Sparse DBN과 여러 분류기를 조합하여 약 91.31%의 정확도를 달성한 연구가 존재한다.

본 논문은 기존 연구들이 주로 단일 인스턴스 라벨링에 의존했다는 한계를 지적하며, LSTM을 통해 수면 데이터의 시계열적 특성을 더 효과적으로 반영함으로써 차별성을 둔다.

## 🛠️ Methodology

### 전체 파이프라인

본 모델은 $\text{Handcrafted Features} \rightarrow \text{DBN (Generative)} \rightarrow \text{LSTM (Discriminative)} \rightarrow \text{Classification}$ 순의 파이프라인을 가진다.

### 주요 구성 요소 및 상세 설명

**1. DBN (Deep Belief Network) 모델**

- **입력**: 이전 연구들에서 사용된 28개의 "Handcrafted" 특징값들을 입력으로 사용한다.
- **구조**: 2개의 층으로 구성된 DBN을 사용하며, 가시 유닛(Visible unit)과 은닉 유닛(Hidden unit)은 각각 200개로 설정되었다.
- **절차**: 비지도 학습 기반의 Pre-training 과정을 거치며, 최종 출력층은 5개의 뉴런으로 구성되어 LSTM의 입력으로 전달된다.

**2. LSTM (Long Short-Term Memory) 모델**

- **구조**: 3개의 LSTM 층을 쌓은 Stacked LSTM 구조이다.
- **입력 형태**: DBN의 출력값 중 이전 5개(또는 10, 15개)의 시퀀스를 입력으로 받는다.
- **상세 레이어**:
  - Layer 1: 5개의 시퀀스 $\times$ 5개의 특징 $\rightarrow$ 128개 출력
  - Layer 2: 128개 입력 $\rightarrow$ 64개 출력
  - Layer 3: 64개 입력 $\rightarrow$ 32개 출력
- **최종 출력**: Softmax 함수를 통해 다중 클래스 분류(Wake, S1, S2, S3, REM)를 수행한다.

### 학습 설정 및 손실 함수

- **손실 함수**: 다중 클래스 분류를 위해 Categorical Crossentropy를 사용한다.
- **최적화 알고리즘**: RMSprop을 사용한다.
- **하이퍼파라미터**: Epoch 100, Batch size 500으로 설정하였다.

## 📊 Results

### 실험 설정

- **데이터셋**: 공개 벤치마크 데이터셋(Benchmark dataset)과 저자들의 MKG 데이터셋 두 가지를 사용하였다.
- **비교 대상**: DBN 단독, LSTM 단독, DBN+HMM 결합 모델과 성능을 비교하였다.
- **평가 지표**: Accuracy(정확도) 및 F-score를 사용하였다.

### 정량적 결과

- **벤치마크 데이터셋**: DBN+LSTM 모델이 평균 **98.75%** (F-score = 0.9875)의 정확도를 기록하였다. 이는 기존 SOTA(91.31%)보다 월등히 높은 수치이다.
- **MKG 데이터셋**: DBN+LSTM 모델이 평균 **98.94%** (F-score = 0.9894)의 정확도를 달성하였다.
- **비교 분석**:
  - DBN 단독 모델의 정확도가 가장 낮았으며($\approx 51.5\%$), LSTM 단독 모델($\approx 63.6\%$)보다 DBN과 결합했을 때 성능이 비약적으로 상승하였다.
  - DBN+HMM 모델(평균 $\approx 72.3\%$)과 비교했을 때도 DBN+LSTM이 훨씬 뛰어난 성능과 안정성을 보였다.

### 정성적 결과 (Confusion Matrix 분석)

- **DBN+LSTM** 모델에서 가장 분류하기 어려운 단계는 S2였으나, 그럼에도 불구하고 97.69%의 높은 정확도를 보였다.
- **DBN+HMM** 모델에서는 S1 단계의 예측 성공률이 42.97%로 매우 낮아, S1과 Wake 단계를 구분하는 데 어려움이 있음이 확인되었다.

## 🧠 Insights & Discussion

### 강점 및 성과

본 연구는 생성 모델인 DBN을 통해 입력 데이터의 고수준 특징을 먼저 추출하고, 이를 시퀀스 모델인 LSTM에 전달함으로써 수면 단계 분류의 정확도를 극대화하였다. 특히 LSTM이 HMM보다 수면 단계의 전이 패턴을 훨씬 더 지능적으로 학습하여, 데이터의 변동성이 큰 구간에서도 강건한(Robust) 성능을 보였다는 점이 입증되었다.

### 한계 및 비판적 해석

- **특징 추출의 의존성**: 본 모델은 여전히 사람이 직접 설계한 "Handcrafted features"에 의존하고 있다. 반면, 최신 SOTA 연구들은 raw data를 직접 처리하는 방식을 취하고 있어, 유연성 측면에서는 raw data 접근 방식이 더 유리할 수 있다.
- **계산 복잡도**: DBN의 Pre-training 과정에서 가장 많은 계산 시간이 소요된다(벤치마크 데이터셋 기준 약 4시간, MKG 기준 17시간). 비록 LSTM의 추론 및 학습 속도는 GPU 가속으로 매우 빠르나, 전체 파이프라인의 초기 학습 비용이 존재한다.
- **클래스 간 유사성**: S1과 S2 단계, 또는 WAKE와 S1 단계의 분류가 상대적으로 어려운 이유는 EOG 및 EMG 파형의 특성이 유사하기 때문인 것으로 분석된다.

## 📌 TL;DR

본 논문은 수면 단계 분류를 위해 **DBN의 생성적 특징 추출 능력**과 **LSTM의 시계열 판별 능력**을 결합한 하이브리드 딥러닝 모델을 제안하였다. 실험 결과, 벤치마크 데이터셋에서 **98.75%**라는 매우 높은 정확도를 달성하며 기존 SOTA 성능을 크게 상회하였다. 비록 수동으로 설계된 특징(Handcrafted features)을 사용했다는 한계가 있으나, 시퀀스 데이터를 활용한 수면 단계 분석의 효용성을 입증하였으며 향후 Raw data를 직접 처리하는 방향으로의 확장 가능성을 제시하였다.
