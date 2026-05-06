# Multi-Task Learning for Arousal and Sleep Stage Detection Using Fully Convolutional Networks

Hasan Zan, Abdulnasır Yildiz (2023)

## 🧩 Problem to Solve

본 논문은 수면 무호흡증이나 수면 장애 진단에 필수적인 **수면 각성(Sleep Arousal)** 탐지와 **수면 단계(Sleep Stage)** 분류를 자동화하는 문제를 다룬다. 전통적으로 수면 분석은 수면다원검사(Polysomnography, PSG)를 통해 이루어지며, 전문가가 AASM(American Academy of Sleep Medicine) 가이드라인에 따라 수작업으로 스코어링하는 방식을 사용한다.

그러나 이 전통적인 방식은 다음과 같은 심각한 문제점을 가지고 있다. 첫째, 전체 밤의 신호를 분석해야 하므로 매우 많은 시간이 소요된다. 둘째, 동일한 신호를 두고도 전문가마다 해석이 달라 높은 변동성(variability)이 존재한다. 특히 수면 각성은 수면 단계의 패턴을 방해하여 수면의 질을 떨어뜨리고 신체적, 정신적 건강에 악영향을 미치므로, 이를 정확하고 효율적으로 탐지하는 것은 매우 중요하다. 따라서 본 연구의 목표는 raw EEG 신호를 입력으로 하여 수면 각성과 수면 단계를 동시에 정확하게 예측할 수 있는 통합된 계산 모델을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 수면 각성과 수면 단계라는 두 가지 상호 연관된 과제를 **Multi-Task Learning (MTL)** 프레임워크 내에서 **세그멘테이션(Segmentation)** 문제로 정의하고, 이를 해결하기 위해 **Fully Convolutional Network (FCN)** 기반의 모델인 **FullSleepNet**을 제안한 것이다.

주요 기여 사항은 다음과 같다.

- **통합 세그멘테이션 접근법**: 수면 단계 분류(Classification)와 각성 탐지(Detection)를 별개의 과제가 아닌, 하나의 1차원 세그멘테이션 문제로 통합하여 처리한다.
- **Full-night 신호 처리**: 기존의 슬라이딩 윈도우(Sliding Window) 방식 대신, 밤새 기록된 전체 EEG 신호를 한 번에 입력받아 처리함으로써 윈도우 경계에서 발생하는 정보 손실을 방지하고 전역적인 문맥(Global Context)을 유지한다.
- **효율적인 아키텍처 설계**: 불필요한 Upsampling이나 Deconvolution 레이어를 제거하여 계산 리소스를 줄이면서도 2초 해상도의 예측 마스크를 생성하도록 최적화하였다.
- **Raw 데이터 활용**: 복잡한 전처리나 수작업 특징 추출 없이, 표준화된 raw EEG 신호만을 사용하여 End-to-End 학습을 수행한다.

## 📎 Related Works

기존의 자동 수면 스코어링 연구는 크게 두 가지 방향으로 진행되었다. 초기에는 시간-주파수 도메인 특징이나 비선형 파라미터를 수작업으로 추출하여 SVM, Decision Tree, ANN 등의 머신러닝 알고리즘으로 분류하는 방식이 주를 이루었다. 하지만 이러한 방식은 특정 데이터셋에 과적합되기 쉽고 새로운 시스템에 적용할 때마다 수동 튜닝이 필요하다는 한계가 있다.

최근에는 CNN, RNN, Transformer 등 딥러닝 모델이 도입되어 특징 추출 과정을 자동화하였다. 수면 단계 분류를 위해 여러 에포크(epoch)를 입력으로 사용하는 하이브리드 네트워크들이 제안되었으며, 각성 탐지를 위해 BiLSTM이나 Attention 메커니즘을 결합한 모델들이 연구되었다.

그럼에도 불구하고 기존 연구들의 한계점은 다음과 같다.

1. **과제의 분리**: 각성 탐지와 수면 단계 분류는 임상적으로 밀접하게 연관되어 있음에도 불구하고, 대부분의 연구가 한 가지 과제에만 집중하였다.
2. **입력 방식의 한계**: 많은 모델이 짧은 윈도우 단위로 신호를 처리하여 수면 단계 간의 긴 의존성을 충분히 활용하지 못하였다.
3. **데이터 효율성**: 특히 각성 탐지의 경우 소규모 데이터셋에서 학습된 경우가 많아 일반화 성능이 부족하였다.

## 🛠️ Methodology

### 전체 시스템 구조

FullSleepNet은 단일 채널 EEG 신호를 입력받아 수면 각성 마스크와 수면 단계 마스크를 동시에 출력하는 구조이다. 전체 파이프라인은 **Convolution $\rightarrow$ Recurrent $\rightarrow$ Attention $\rightarrow$ Segmentation**의 네 가지 모듈로 구성된다.

### 주요 구성 요소 및 역할

1. **Convolutional Module**: 8개의 컨볼루션 블록으로 구성된다. 각 블록은 서로 다른 커널 크기를 가진 두 개의 레이어를 가져, 작은 필터로는 시간적 정보(temporal information)를, 큰 필터로는 주파수 정보(frequency information)를 캡처한다. 이후 Max-pooling을 통해 차원을 축소하여 연산량을 줄이고 과적합을 방지한다.
2. **Recurrent Module**: 3개의 BiLSTM(Bidirectional LSTM) 레이어를 사용하여 추출된 특징들 사이의 장기 의존성(long-range dependencies)을 학습한다. 정방향과 역방향의 은닉 상태(hidden state)를 결합하여 전후 문맥을 모두 고려한다.
3. **Attention Module**: 모델이 입력 신호의 중요한 부분에 집중하고 노이즈를 억제하도록 돕는다. BiLSTM의 출력 $\mathbf{h}_t$를 바탕으로 정렬 점수(alignment score) $e_t$와 어텐션 가중치 $\alpha_t$를 계산하고, 이를 통해 문맥 벡터(context vector) $\mathbf{c}$를 생성한다.
4. **Segmentation Module**: 최종 예측을 수행하는 두 개의 브랜치로 나뉜다.
    - **각성 탐지 브랜치**: Sigmoid 활성화 함수를 사용하여 각 시점의 각성 확률 $\hat{y}_a$를 출력한다.
    - **수면 단계 분류 브랜치**: Softmax 활성화 함수를 사용하여 5개 단계(W, N1, N2, N3, REM)에 대한 확률 $\hat{y}_s$를 출력한다.

### 주요 방정식 및 학습 절차

**1. Attention 메커니즘**:
정렬 점수 $e_t$는 다음과 같이 계산된다.
$$e_t = \tanh(\mathbf{h}_t \cdot \mathbf{W}_{attn} + \mathbf{b}_{attn})$$
이후 Softmax를 통해 가중치 $\alpha_t$를 구하고, 최종 문맥 벡터 $\mathbf{c}$를 생성한다.
$$\alpha_t = \text{softmax}(e_t), \quad \mathbf{c} = \sum_{t=1}^{T} \alpha_t \mathbf{h}_t$$

**2. 손실 함수 (Loss Function)**:
각성 탐지에는 Binary Cross-Entropy(BCE)를, 수면 단계 분류에는 Categorical Cross-Entropy(CCE)를 사용하며, 두 손실의 가중 합으로 최종 손실 $\mathcal{L}$을 정의한다.
$$\mathcal{L} = \omega_1 \left[ -\frac{1}{P} \sum_{i=1}^{P} (y_i^a \log \hat{y}_i^a + (1-y_i^a) \log(1-\hat{y}_i^a)) \right] + \omega_2 \left[ -\sum_{i=1}^{P} y_i^s \log \hat{y}_i^s \right]$$
여기서 $\omega_1 = \omega_2 = 1$로 설정하여 두 과제에 동일한 가중치를 부여하였다.

**3. 학습 과정**:

- 데이터는 Train/Validation/Test 세트로 $0.5:0.2:0.3$ 비율로 분할한다.
- EEG 신호는 평균을 빼고 표준편차로 나누는 표준화(Standardization)를 거친다.
- Adam Optimizer를 사용하며, 학습률은 $10^{-4}$로 설정하였다.
- 신호에 $[0.9, 1.1]$ 범위의 랜덤 스칼라를 곱하는 단순 데이터 증강(Data Augmentation)을 적용하였다.

## 📊 Results

### 실험 설정

- **데이터셋**: SHHS(Sleep Heart Health Study)와 MESA(Multi-Ethnic Study of Atherosclerosis)라는 두 개의 대규모 PSG 데이터셋을 사용하였다.
- **측정 지표**:
  - 각성 탐지: Sample-level(AUPRC, AUROC) 및 Epoch-level(Precision, Recall, F1, Accuracy, Kappa) 지표를 사용한다.
  - 수면 단계 분류: Accuracy, F1 score, Cohen's kappa coefficient를 사용한다.

### 주요 결과

1. **수면 각성 탐지 (Arousal Detection)**:
    - SHHS와 MESA 데이터셋 모두에서 AUPRC 약 $0.70$, AUROC 약 $0.96 \sim 0.97$의 높은 성능을 기록하였다.
    - 특히 기존 SOTA 모델들과 비교했을 때, 단일 채널 EEG만을 사용했음에도 불구하고 AUPRC를 SHHS에서 17%, MESA에서 13% 향상시키는 괄목할 만한 성과를 거두었다.

2. **수면 단계 분류 (Sleep Stage Classification)**:
    - SHHS에서는 Accuracy $0.875$, MESA에서는 $0.829$를 달성하였다.
    - 기존의 최상위 모델(SOTA)들과 비교했을 때 매우 유사한 성능(Accuracy 차이 약 $0.2\%$)을 보였으며, 각성과 단계를 동시에 처리하는 다른 모델(Zhang et al.)보다는 훨씬 뛰어난 성능을 보였다.

3. **Ablation Study**:
    - 컨볼루션(C), 리커런트(R), 어텐션(A) 모듈을 조합하여 실험한 결과, 모든 모듈이 포함된 **Model-CRA**가 가장 높은 성능을 보였다.
    - 특히 BiLSTM 기반의 Recurrent 모듈이 각성 탐지 성능 향상에 결정적인 역할을 하는 것으로 나타났다.

## 🧠 Insights & Discussion

### 강점 및 분석

FullSleepNet의 가장 큰 강점은 **전체 밤의 신호를 한 번에 처리**한다는 점이다. 이는 기존의 슬라이딩 윈도우 방식이 가진 경계 정보 손실 문제를 해결하고, 수면 전문가가 전체 기록을 훑어보며 전후 맥락을 고려해 스코어링하는 실제 임상 방식을 모방한 결과라고 볼 수 있다. 또한, Multi-Task Learning을 통해 각성과 수면 단계라는 서로 영향을 주고받는 두 과제를 동시에 학습함으로써 상호 보완적인 특징을 학습할 수 있었다.

### 한계 및 비판적 해석

1. **단계별 분류 난이도**: 결과 분석에서 N1 단계의 F1 score가 다른 단계에 비해 현저히 낮게 나타났다. 이는 N1 단계가 Wake나 N2 단계와 패턴이 유사하여 전문가들 사이에서도 합의율(inter-rater agreement)이 낮기 때문으로 분석된다.
2. **어텐션 모듈의 효율성**: Ablation study 결과, 어텐션 모듈의 추가적인 성능 향상 폭이 리커런트 모듈에 비해 상대적으로 적었다. 저자들은 이를 각성 이벤트의 돌발적인 특성 때문이거나 BiLSTM과 기능적으로 중복되기 때문이라고 추측하였다.
3. **데이터 분포의 영향**: MESA 데이터셋에서 N3 단계의 성능이 SHHS보다 낮게 나왔는데, 이는 MESA 데이터셋 내 N3 에포크의 비율이 낮아 학습 데이터가 부족했거나 전문가들의 스코어링 일관성이 부족했기 때문일 가능성이 크다.

## 📌 TL;DR

본 논문은 Raw EEG 신호를 입력으로 하여 수면 각성 탐지와 수면 단계 분류를 동시에 수행하는 **Multi-Task Learning 기반의 FCN 모델(FullSleepNet)**을 제안한다. 이 모델은 전체 밤의 신호를 한 번에 처리하는 세그멘테이션 방식을 채택하여, **수면 각성 탐지에서 SOTA 성능을 달성**하였으며 **수면 단계 분류에서도 기존 전문 모델들에 필적하는 성능**을 보였다. 전처리를 최소화하고 단일 채널 EEG만을 사용하여 효율성을 극대화했으므로, 향후 수면 의학의 임상 의사결정 지원 시스템(Clinical Decision Support System)으로 확장될 가능성이 매우 높은 연구이다.
