# Audio Adversarial Examples for Robust Hybrid CTC/Attention Speech Recognition

Ludwig Kürzinger, Edgar Ricardo Chavez Rosas, Lujun Li, Tobias Watzel, and Gerhard Rigoll (2020)

## 🧩 Problem to Solve

본 논문은 자동 음성 인식(Automatic Speech Recognition, ASR) 시스템, 특히 최신의 end-to-end 하이브리드 CTC/Attention 구조가 가진 보안 취약성과 강건성(Robustness) 문제를 해결하고자 한다.

최근 ASR 시스템은 더 깊은 신경망을 사용하여 성능을 높이는 추세이나, 이는 동시에 특수하게 설계된 노이즈인 Audio Adversarial Examples (AAEs)에 더 취약해지는 결과를 초래한다. AAE는 인간이 듣기에는 거의 느껴지지 않는 미세한 노이즈임에도 불구하고, 모델이 완전히 잘못된 텍스트로 전사(transcription)하게 만들어 시스템의 보안 위협(예: 개인 비서에게 숨겨진 음성 명령 전달)을 야기할 수 있다.

따라서 본 연구의 목표는 하이브리드 CTC/Attention ASR 시스템을 겨냥한 효과적인 AAE 생성 알고리즘을 제안하고, 이를 역으로 활용한 Adversarial Training을 통해 모델의 강건성과 일반적인 인식 성능을 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 하이브리드 CTC/Attention 구조의 특성을 반영하여, 기존의 단일 구조 대상 공격보다 더 강력한 AAE 생성 방법론을 제안한 것이다.

1. **Attention 기반의 윈도우 기법 제안**: Attention 메커니즘의 자기회귀(auto-regressive) 특성을 이용하여, 전체 시퀀스가 아닌 특정 부분에 집중해 노이즈를 생성하는 Static Window 및 Moving Window 방식을 제안하였다.
2. **Joint CTC/Attention 공격 방법론**: CTC 손실 함수 기반의 공격과 Attention 기반의 공격을 결합하여, 하이브리드 모델의 두 가지 경로를 모두 교란하는 통합 그래디언트 방법을 제시하였다.
3. **Adversarial Training을 통한 강건성 확보**: 생성된 AAE를 학습 과정에 포함시켜, adversarial noise뿐만 아니라 일반적인 화이트 노이즈에 대해서도 더 강건하며 일반적인 인식 성능까지 개선된 모델을 구현하였다.

## 📎 Related Works

본 연구는 다음과 같은 기존 연구들의 한계점과 차별점을 가진다.

- **ASR 아키텍처**: CTC 기반 모델과 Attention 기반 Encoder-Decoder 모델이 각각 존재하며, 이를 결합한 Hybrid CTC/Attention 구조가 제안된 바 있다.
- **기존 AAE 연구**:
  - 이미지 인식 분야의 Fast Gradient Sign Method (FGSM)가 음성 분야로 확장되어 적용되었다.
  - 기존의 AAE 연구들은 주로 단순한 RNN이나 CNN 구조를 대상으로 하였으며, 본 논문의 핵심인 Attention 메커니즘이 포함된 복잡한 구조에 대한 분석은 부족했다.
  - CTC 기반 ASR에 대한 공격(Carlini et al.)이나 Attention 기반 LAS 모델에 대한 공격(Sun et al.)은 개별적으로 연구되었으나, 이 두 가지가 결합된 하이브리드 구조에 대한 통합적인 공격 방법은 제시되지 않았다.
- **차별점**: 본 논문은 단순한 전역적 그래디언트 합산이 아닌, '윈도우' 개념을 도입하여 Attention 메커니즘의 국소적 취약점을 공략한다는 점에서 기존 연구와 차별화된다.

## 🛠️ Methodology

본 논문에서 제안하는 모든 AAE 생성 방법은 화이트박스(whitebox) 모델을 가정하며, 입력 특징 시퀀스 $X = x_{1:T}$에 가산적 노이즈 $\delta(x_t)$를 더해 적대적 예제 $\hat{x}_t$를 생성한다.

$$\hat{x}_t = x_t + \delta(x_t), \forall t \in [1, T]$$

### 1. Attention-based AAEs

Attention 디코더의 교차 엔트로피 손실 $J(X, y_l; \theta)$를 이용한다.

- **Static Window (SW)**: 전체 시퀀스가 아닌, 시작 토큰 $\gamma$부터 $l_w$개의 토큰까지만 손실을 합산하여 노이즈를 생성한다. 이는 특정 부분의 attention을 방해함으로써 이후의 디코딩 전체를 무너뜨리려는 전략이다.
$$\delta^{SW}(x_t) = \epsilon \cdot \text{sgn}(\nabla_{x_t} \sum_{l=\gamma}^{\gamma+l_w} J(X, y^*_l; \theta))$$

- **Moving Window (MW)**: 정적 윈도우의 위치 의존성 문제를 해결하기 위해, 고정된 길이 $l_w$와 스트라이드 $\nu$를 가진 슬라이딩 윈도우를 사용한다. 각 윈도우의 그래디언트를 정규화하여 누적함으로써 더 광범위하고 강력한 공격을 수행한다.
$$\nabla^{MW}(x_t) = \sum_{i=0}^{\lceil L/\nu \rceil} \frac{\nabla_{x_t} \sum_{l=i \cdot \nu}^{i \cdot \nu + l_w} J(X, y^*_l; \theta)}{\|\nabla_{x_t} \sum_{l=i \cdot \nu}^{i \cdot \nu + l_w} J(X, y^*_l; \theta)\|_1}$$
$$\delta^{MW}(x_t) = \epsilon \cdot \text{sgn}(\nabla^{MW}(x_t))$$

### 2. CTC-based AAEs

전체 재구성된 라벨 문장 $y^*$에 대한 CTC 손실 $L^{CTC}$를 기반으로 FGSM을 적용한다.
$$\delta^{CTC}(x_t) = \epsilon \cdot \text{sgn}(\nabla_{x_t} L^{CTC}(X, y^*; \theta))$$

### 3. Hybrid CTC/Attention AAEs

위의 두 가지 방식(Attention 기반 $\delta^{att}$와 CTC 기반 $\delta^{CTC}$)을 가중치 $\xi \in [0, 1]$를 이용하여 결합한다.
$$\hat{x}_t = x_t + (1-\xi) \cdot \delta^{att}(x_t) + \xi \cdot \delta^{CTC}(x_t)$$

### 4. Adversarial Training (AT)

학습 시 미니배치의 샘플을 확률 $p_a$로 선택하여 AAE로 대체하거나 추가한다. 특히 오버피팅을 방지하기 위해 섭동 크기 $\epsilon$을 $[0.0, 0.3]$ 범위에서 무작위로 샘플링한다. 학습 목표 함수는 다음과 같이 확장된다.
$$\hat{J}(X, y; \theta) = \sum_{i} (J(X, y_i; \theta) + J(\hat{X}, y_i; \theta))$$

## 📊 Results

### 실험 설정

- **데이터셋**: TEDlium v2 (200시간 이상의 음성 데이터).
- **모델**: LSTM 인코더와 Location-aware Attention 디코더를 갖춘 하이브리드 구조.
- **지표**: Character Error Rate (CER) 및 Word Error Rate (WER).
- **비교 대상**: ESPnet 툴킷 기반의 Baseline 모델.

### 주요 결과

1. **AAE 생성 효율성 (Case Study)**:
    - TTS로 생성한 깨끗한 문장("Peter", "Ani")을 대상으로 실험한 결과, Moving Window 방식이 Static Window나 CTC 기반 방식보다 훨씬 높은 CER을 유도하였다.
    - 하이브리드 공격($\xi=0.5$) 시, 단일 공격보다 오류율이 낮아지는 경향이 있었으나, Moving Window 기반 공격은 여전히 강력한 성능을 보였다.

2. **Adversarial Training의 효과**:
    - **강건성 향상**: Baseline 모델은 AAE 테스트 셋에서 거의 100%에 가까운 WER을 보였으나, AT를 적용한 모델은 이를 약 55~60% 수준으로 크게 낮추었다.
    - **일반 성능 향상**: 정규 테스트 셋(Clean data)에서도 Baseline의 WER 18.3% 대비 하이브리드 AAE로 학습시킨 모델은 16.5%를 기록하여, 절대적으로 1.8%의 성능 향상을 보였다.
    - **노이즈 내성**: 화이트 노이즈(5dB, 30dB)가 섞인 데이터에서도 AT 모델이 Baseline보다 더 낮은 WER을 기록하여 일반적인 노이즈에 대한 강건성도 함께 향상되었음을 확인하였다.

## 🧠 Insights & Discussion

본 연구는 하이브리드 ASR 모델이 단순히 두 구조의 합이 아니라, 각각의 취약점이 공존함을 보여주었다. 특히 Attention 메커니즘의 자기회귀적 성질을 이용하여 국소적인 영역을 공격하는 Moving Window 방식이 매우 효과적이라는 점을 밝혀냈다.

가장 흥미로운 점은 Adversarial Training이 단순히 공격에 대한 방어 수단에 그치지 않고, 모델의 일반적인 일반화 성능(Generalization)을 높였다는 것이다. 이는 AAE가 모델이 간과하고 있던 데이터의 경계 영역을 탐색하게 함으로써, 일종의 강력한 데이터 증강(Data Augmentation) 역할을 수행했기 때문으로 해석할 수 있다.

다만, 본 실험은 특징 공간(feature space) 내에서의 섭동만을 다루었으며, 이를 다시 오디오 신호로 복원했을 때 인간이 느끼는 인지적 영향이나 실제 물리적 환경에서의 공격 성공률에 대해서는 명시적으로 분석하지 않았다.

## 📌 TL;DR

본 논문은 하이브리드 CTC/Attention ASR 모델을 무력화할 수 있는 새로운 적대적 예제(AAE) 생성 알고리즘(특히 Moving Window 방식)을 제안하였다. 이를 통해 모델의 취약성을 분석하였으며, 생성된 AAE를 학습 과정에 도입하는 Adversarial Training을 통해 적대적 공격에 대한 방어력은 물론, 일반적인 음성 인식 성능까지 향상시킬 수 있음을 입증하였다. 이 연구는 향후 보안성이 강화된 음성 인식 시스템 구축 및 강건한 딥러닝 모델 학습 전략 수립에 중요한 기초 자료가 될 것으로 보인다.
