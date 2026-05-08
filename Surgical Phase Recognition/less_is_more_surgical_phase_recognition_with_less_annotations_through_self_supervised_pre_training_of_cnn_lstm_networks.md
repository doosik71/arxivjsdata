# Less is More: Surgical Phase Recognition with Less Annotations through Self-Supervised Pre-training of CNN-LSTM Networks

Gaurav Yengera, Didier Mutter, Jacques Marescaux, Nicolas Padoy (2018)

## 🧩 Problem to Solve

본 논문은 복강경 수술 비디오를 이용한 실시간 수술 단계 인식(Surgical Phase Recognition) 시스템의 구축과 그 과정에서 발생하는 데이터 라벨링 비용 문제를 해결하고자 한다. 수술 단계 인식 기술은 수술 중 보조 시스템 개발, 수술실 자원 관리 최적화, 그리고 전반적인 환자 안전성 향상을 위해 필수적이다.

하지만 기존의 최신 알고리즘들은 완전 지도 학습(Fully Supervised Learning)에 의존하고 있으며, 이는 숙련된 전문가가 수많은 수술 비디오를 직접 수동으로 라벨링해야 함을 의미한다. 이러한 작업은 매우 많은 시간과 비용이 소요될 뿐만 아니라, 수술의 종류가 다양하여 모든 수술 타입에 대해 지도 학습 기반 모델을 확장 적용하는 데 큰 한계가 있다. 따라서 본 연구의 목표는 수동 라벨링에 대한 의존도를 낮추면서도 높은 인식 성능을 유지할 수 있는 준지도 학습(Semi-supervised) 접근 방식을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 수술 비디오 자체에 내재된 정보를 이용하는 **자기지도 사전 학습(Self-supervised Pre-training)**을 도입하는 것이다. 구체적으로, 수술 비디오의 타임스탬프를 통해 자동으로 추출할 수 있는 **잔여 수술 시간(Remaining Surgery Duration, RSD)** 예측 작업을 사전 학습 과제로 설정하였다.

연구진은 수술의 시간적 진행 과정이 수술 단계(Phase)와 밀접하게 연관되어 있으며, 각 단계의 변화가 잔여 시간의 변화로 나타난다는 직관에 기반하였다. 이를 통해 CNN과 LSTM 네트워크가 수술 워크플로우의 시공간적 특징을 먼저 학습하게 함으로써, 이후 적은 양의 라벨링 데이터만으로도 효율적으로 수술 단계를 인식할 수 있도록 설계하였다. 또한, CNN과 LSTM을 통합하여 최적화하는 **EndoN2N**이라는 end-to-end 학습 모델을 제안하여 학습 효율을 높였다.

## 📎 Related Works

기존의 수술 단계 인식 연구들은 주로 Hidden Markov Models (HMMs), Conditional Random Fields (CRFs), 또는 CNN-RNN 결합 모델을 사용해 왔다. 특히 CNN-LSTM 구조는 시공간적 특징을 모두 추출할 수 있어 널리 쓰였으나, 대개 CNN과 LSTM을 별도로 학습시키는 2단계 방식(예: EndoLSTM)을 취하거나, 메모리 문제로 인해 비디오 전체가 아닌 짧은 세그먼트 단위로만 end-to-end 학습을 진행하는 한계가 있었다.

자기지도 학습 분야에서는 이미지의 상대적 위치 예측이나 템포럴 컨텍스트 학습(Temporal Context Learning) 등이 연구되었다. 수술 단계 인식에 이를 적용한 Bodenstedt et al. (2017)의 연구(TempCon)가 있었으나, 이는 무작위로 샘플링된 프레임 쌍의 순서만을 예측하므로 비디오 전체의 긴 흐름(Long-range temporal structure)을 학습하지 못하며, 오직 CNN만을 사전 학습시킨다는 한계가 있다. 본 논문은 RSD 예측을 통해 비디오 전체 시퀀스를 활용하고 CNN과 LSTM을 동시에 사전 학습시킨다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 1. EndoN2N 아키텍처 및 학습 절차

EndoN2N 모델은 CNN(CaffeNet 기반)과 LSTM을 결합한 구조이다. 모델의 전체 흐름은 다음과 같다:

1. **CNN Fine-tuning**: 초기 단계에서 CNN을 수술 단계 인식 작업에 대해 먼저 미세 조정하여 정보력 있는 특징 추출기를 만든다.
2. **CNN-LSTM 통합**: 미세 조정된 CNN 층을 LSTM에 연결하고, 마지막에 수술 단계 수($M=7$)만큼의 뉴런을 가진 Fully Connected (FC) 레이어를 배치한다.
3. **손실 함수**: 다중 클래스 분류 문제로 정의하며, 다음의 multinomial logistic loss를 사용한다.
   $$L = -\frac{1}{T} \sum_{t=1}^{T} \sum_{p=1}^{M} y_{t}^{p} \log(\sigma(z_{t})^{p})$$
   여기서 $T$는 프레임 수, $y_{t}^{p}$는 정답 라벨, $\sigma(z_{t})^{p}$는 소프트맥스 함수를 통한 예측 확률이다.

### 2. End-to-End 학습을 위한 BPTT 근사 (Approximate BPTT)

수술 비디오는 매우 길기 때문에 전체 시퀀스에 대해 BPTT(Backpropagation Through Time)를 수행하면 메모리 복잡도가 너무 높아진다. 이를 해결하기 위해 비디오를 $\ell$개의 연속적인 서브시퀀스로 나누어 손실을 계산하고 그래디언트를 누적하는 방식을 취한다.

- 서브시퀀스 경계에서 LSTM의 셀 상태와 은닉 상태는 전방 전달(Forward propagate)하되, 역전파(BPTT)는 절단(Truncated)한다.
- 이를 통해 메모리 사용량을 줄이면서도 전체 비디오에 대한 가중치 업데이트를 수행한다.

### 3. RSD 사전 학습 (RSD Pre-training)

본 논문의 핵심인 RSD 사전 학습은 수동 라벨 없이 타임스탬프만으로 학습하는 회귀(Regression) 작업이다.

- **사전 학습 과제**: 현재 프레임에서 수술 종료까지 남은 시간($t_{rsd}$)과 수술 진행률($prog$)을 동시에 예측하는 Multi-task 학습을 수행한다.
- **손실 함수**: 예측값과 실제값의 차이를 줄이기 위해 Smooth L1 Loss($\Omega$)를 사용한다.
  $$L = -\frac{1}{T} \sum_{t=1}^{T} [y_{t}^{rsd} \Omega(z_{t}^{rsd}) + y_{t}^{prog} \Omega(\rho(z_{t}^{prog}))]$$
  $$\Omega(x) = \begin{cases} 0.5x^2, & \text{if } |x| < 1 \\ |x| - 0.5, & \text{otherwise} \end{cases}$$
- **모델 업데이트**: RSD 사전 학습의 효과를 극대화하기 위해, LSTM의 입력 특징으로 **경과 시간(Elapsed Time)**과 **예측된 진행률(Predicted Progress)**을 추가로 입력하는 구조로 수정하였다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: Cholec120 (120개의 담낭 절제술 비디오, 7개 단계로 구분).
- **비교 대상**: EndoLSTM (2단계 학습), Vanilla EndoN2N (사전 학습 없음), TempCon (기존 자기지도 학습).
- **평가 지표**: Accuracy, Precision, Recall, F1-score.

### 2. 주요 결과

- **EndoN2N vs EndoLSTM**: End-to-end 학습을 수행한 EndoN2N이 모든 지표에서 EndoLSTM을 능가하였다. 특히 가장 인식이 어려운 'Clipping and Cutting' 단계에서 현저한 성능 향상을 보였다.
- **RSD 사전 학습의 효과**:
  - RSD 사전 학습을 적용했을 때, 라벨링된 비디오 수를 **20% 줄여도** 기존 모델과 비슷하거나 오히려 더 나은 성능을 보였다.
  - 라벨링 데이터를 **50%만 사용하더라도** 성능 하락폭이 5% 이내로 유지되었다.
  - 흥미롭게도, 모든 데이터를 라벨링하여 학습시킨 경우에도 RSD 사전 학습을 거친 모델이 더 높은 성능을 기록하였다.
- **RSD vs TempCon**: RSD 기반 사전 학습이 TempCon 방식보다 훨씬 우수한 성능을 보였으며, 특히 라벨링 데이터가 적을수록 그 격차가 커졌다.

## 🧠 Insights & Discussion

본 연구는 수술 비디오의 '시간적 흐름'이라는 내재적 속성을 활용하여 데이터 희소성 문제를 효과적으로 해결하였다. 특히 RSD 예측이라는 단순한 보조 작업이 수술 단계라는 복잡한 의미론적 구조를 학습하는 데 매우 유용한 가이드 역할을 한다는 점을 입증하였다.

또한, end-to-end 학습이 CNN의 특징 추출 과정과 LSTM의 시계열 분석 과정을 상호 최적화함으로써 일반화 성능을 높인다는 것을 확인하였다. 분석 결과, RSD 사전 학습 모델은 실제 임상 적용 시 중요한 **첫 번째 단계 경계 탐지(First phase boundary detection)** 능력이 뛰어나고, 예측 결과의 **노이즈(Noise)**가 적어 실시간 알림 시스템 등에 적용 가능성이 높음을 시사한다.

다만, 본 연구는 담낭 절제술이라는 단일 수술 타입에 국한되어 검증되었다는 한계가 있다. 향후 다양한 수술 타입으로 확장하여 제안 방법론의 범용성을 검증할 필요가 있으며, GAN이나 합성 데이터를 활용한 추가적인 준지도 학습 방법과의 비교 연구가 필요하다.

## 📌 TL;DR

본 논문은 수동 라벨링 비용이 높은 수술 단계 인식 문제를 해결하기 위해, **잔여 수술 시간(RSD) 예측**이라는 자기지도 사전 학습 과제와 **EndoN2N**이라는 end-to-end CNN-LSTM 학습 구조를 제안하였다. 실험 결과, RSD 사전 학습을 통해 **라벨링 데이터를 20%~50% 적게 사용하고도 기존 지도 학습 모델과 대등하거나 더 뛰어난 성능**을 달성하였으며, 이는 데이터 확보가 어려운 의료 영상 분석 분야에서 모델의 확장성을 크게 높일 수 있는 실용적인 접근 방식이다.
