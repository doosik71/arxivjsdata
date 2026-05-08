# Ubi-SleepNet: Advanced Multimodal Fusion Techniques for Three-stage Sleep Classification Using Ubiquitous Sensing

Bing Zhai, Yu Guan, Michael Catt, Thomas Plötz (2021)

## 🧩 Problem to Solve

수면 모니터링의 임상적 표준(Gold Standard)은 수면다원검사(Polysomnography, PSG)이다. PSG는 뇌파(EEG), 근전도(EMG), 안구전도(EOG) 등을 통해 수면 단계를 5단계(Wake, REM, N1, N2, N3)로 정밀하게 구분할 수 있으나, 장비의 고비용, 착용의 불편함, 그리고 통제된 실험실 환경이 필요하다는 점 때문에 일상적인 장기 모니터링에는 부적합하다.

이에 대한 대안으로 스마트워치와 같은 유비쿼터스 센싱(Ubiquitous Sensing) 기술이 주목받고 있으며, 특히 심장 신호(Cardiac sensing)와 움직임 신호(Movement sensing)를 이용한 3단계 수면 분류(Wake, REM, NREM)가 유망한 접근법으로 제시되었다. 그러나 서로 다른 특성을 가진 두 모달리티의 데이터를 어떻게 효과적으로 융합(Fusion)하여 분류 정확도를 극대화할 것인가에 대한 체계적인 연구가 부족한 상황이다. 기존 연구들은 주로 단순한 특징 결합(Feature Concatenation) 방식에 의존하여 NREM 수면 시간을 과다 추정하고 Wake 시간을 과소 추정하는 한계를 보였다.

본 논문의 목표는 심장 및 움직임 센싱 데이터를 활용하여 3단계 수면 분류 성능을 높이기 위한 다양한 딥러닝 기반 다중 모달 융합 전략(Fusion Strategy)과 융합 방법(Fusion Method)을 체계적으로 분석하고 최적의 구조를 제안하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 융합이 일어나는 **시점(Strategy)**과 융합하는 **방식(Method)**을 분리하여 다각도로 실험하는 것이다.

1. **세 가지 융합 전략(Fusion Strategies) 제안**: 데이터가 어느 단계에서 병합되어야 하는지를 결정하는 Early-stage, Late-stage, Hybrid fusion 전략을 연구하였다.
2. **세 가지 융합 방법(Fusion Methods) 적용**: 어떻게 결합하는 것이 최적인지를 결정하기 위해 Simple operations(Concatenation, Addition), Attention mechanism, Tensor-based method(Bilinear pooling)를 적용하였다.
3. **실질적 벤치마크 제공**: 소비자 수준의 Apple Watch 데이터셋과 연구 수준의 MESA 데이터셋 모두에서 실험을 진행하여, 제안하는 융합 기법들이 기존 벤치마크 모델보다 통계적으로 유의미하게 성능이 향상됨을 입증하였다.
4. **해석 가능성 탐구**: Grad-CAM을 활용하여 수면 단계 분류 과정에서 모델이 어떤 특징과 시간대에 주목하는지를 시각화하고, 이를 통해 딥러닝 모델의 투명성을 높이는 사용자 연구를 수행하였다.

## 📎 Related Works

기존의 수면 모니터링 연구는 크게 세 가지 방향으로 진행되었다.

- **PSG 기반 모니터링**: 가장 정확하지만 일상생활 적용이 불가능하다.
- **Actigraphy**: 가속도계를 이용하여 수면/각성(Wake/Sleep)은 잘 구분하지만, 구체적인 수면 단계(REM/NREM)를 구분하는 데는 한계가 있다.
- **심장 활동 기반 모니터링**: 심박 변이도(Heart Rate Variability, HRV) 분석을 통해 자율신경계의 상태를 파악함으로써 수면 단계를 추론한다. 특히 REM 수면은 교감신경 활성도가 높아 NREM 수면과 구별되는 특성을 가진다.

본 논문은 기존의 단순한 특징 결합 방식에서 벗어나, 모달리티 간의 상호작용을 더 깊게 학습할 수 있는 고급 융합 기법을 도입함으로써 기존 연구들이 가졌던 분류 성능의 한계를 극복하고자 한다.

## 🛠️ Methodology

### 1. 전체 시스템 구조 및 데이터 처리

시스템은 슬라이딩 윈도우(Sliding Window) 방식을 사용하여 수면 기록을 세그먼트로 분할한다. 윈도우 길이 $T=101$, 스트라이드 $S=1$로 설정하여 각 수면 에포크(Sleep epoch, 30초 단위)에 대해 예측을 수행한다.

- **입력 데이터**:
  - Cardiac Sensing: 심박수(HR) 통계값 또는 HRV 특징.
  - Movement Sensing: 가속도계 기반의 활동량(Activity counts).
- **목표 함수**: 모델 $f(\cdot)$의 예측값 $\hat{y}^{(i)}$와 실제 라벨 $y^{(i)}$ 사이의 경험적 손실(Empirical loss) $L$을 최소화하는 것이다.
$$\min_{f} \frac{1}{N} \sum_{i=1}^{N} L(\hat{y}^{(i)} = f(X_{mov}^{(i)}, X_{car}^{(i)}), y^{(i)})$$

### 2. 융합 전략 (Fusion Strategies)

- **Early-stage Fusion**: 입력 단계에서 두 모달리티의 특징을 단순히 결합(Concatenate)하여 신경망에 입력한다.
$$\hat{y}^{(i)} = h(\text{Concatenate}(X_{car}^{(i)}, X_{mov}^{(i)}))$$
- **Late-stage Fusion**: 각 모달리티를 독립적인 네트워크 $q$로 처리하여 고차원 표현(High-level representation)을 얻은 후, 이를 집계 함수(Agg)로 합쳐 분류기에 전달한다.
$$\hat{y}^{(i)} = \phi(\text{Agg}(q(x_{mov}^{(i),1}), \dots, q(x_{car}^{(i),C_{car}})))$$
- **Hybrid Fusion**: 모달리티별 전용 네트워크 $f_{mov}, f_{car}$를 통해 각각의 표현을 학습한 뒤, 후반 단계에서 이를 융합한다.
$$\hat{y}^{(i)} = \phi(\text{Agg}(f_{mov}(X_{mov}^{(i)}), f_{car}(X_{car}^{(i)})))$$

### 3. 융합 방법 (Fusion Methods)

- **Simple Operations**: 가장 기본적인 방식으로, 벡터를 그대로 붙이는 **Concatenation**과 요소별로 더하는 **Addition**을 사용한다.
- **Attention Mechanism**: 한 모달리티의 컨텍스트를 기반으로 다른 모달리티의 가중치를 동적으로 계산한다.
  - $\tanh$와 $\text{softmax}$를 이용하여 시간 축에 대한 어텐션 분포 $P_{att}^{(i)}$를 생성하고, 이를 특정 모달리티 표현에 곱하여 중요한 정보만 필터링한다.
  - **Attention-on-Mov**와 **Attention-on-Car** 두 가지 시나리오를 실험하였다.
- **Bilinear Pooling**: 두 표현 행렬의 외적(Outer product)을 계산하여 모든 요소 간의 곱셈 상호작용을 캡처한다.
$$k_{bi}^{(i)} = \text{vec}(X_{car}'^{(i)} \otimes X_{mov}'^{(i)})$$
이후 부호 유지 제곱근(Signed square-root) 처리와 $L_2$ 정규화를 거쳐 차원을 축소한 뒤 분류기에 입력한다.

### 4. 백본 네트워크 (Backbone Network)

본 연구에서는 **DeepCNN**과 스킵 연결(Skip connection)이 추가된 **ResDeepCNN**을 사용하였다. 합성곱 층(Convolutional layers)과 맥스 풀링(Max-pooling), 완전 연결 층(FC layers)으로 구성되며, 가중치 업데이트에는 Adam 옵티마이저를 사용하였다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: Apple Watch 데이터셋(31명, 소비자급), MESA 데이터셋(1,743명, 연구급).
- **지표**: Accuracy, Cohen’s $\kappa$, Mean F1, Time Deviation (예측된 수면 시간과 실제 시간의 편차).

### 2. 주요 결과

- **Apple Watch 데이터셋**: Late-stage Fusion에서 **Addition** 방법을 사용한 **ResDeepCNN** 모델이 가장 높은 성능을 보였다 (Mean F1 66.5%, Accuracy 78.2%).
- **MESA 데이터셋**: Hybrid Fusion에서 **Attention-on-Mov** 방법을 사용한 **ResDeepCNN** 모델이 최적의 성능을 기록하였다 (Mean F1 73.3%, Accuracy 79.6%).
- **추론 효율성**: Early-stage Fusion이 가장 빠른 추론 속도를 보였으며, Bilinear pooling 방식은 파라미터 수가 가장 많아 메모리 부담이 컸다.
- **정성적 결과**: Grad-CAM 시각화 연구 결과, 기계 학습 기반의 시각화 보조 도구가 인간의 수면 단계 인식 정확도를 향상시킴을 확인하였다.

## 🧠 Insights & Discussion

### 1. 융합 전략 및 방법의 효과

단순한 결합(Concatenation)보다 고차원 특징을 분리하여 학습한 후 융합하는 Late/Hybrid 전략이 전반적으로 우수하였다. 특히 **Attention-on-Mov**가 효과적이었는데, 이는 움직임 센싱만으로는 REM과 NREM을 구분하기 어렵지만, 심장 신호의 컨텍스트를 통해 움직임 데이터의 가중치를 조절함으로써 더 정교한 구분이 가능해졌기 때문으로 해석된다.

### 2. 하드웨어 제약과 모델 선택의 트레이드오프

유비쿼터스 컴퓨팅 환경에서는 성능뿐만 아니라 추론 속도와 모델 크기가 중요하다.

- **스마트폰 전송 가능 시**: 성능이 가장 좋은 Late-stage Fusion (Addition) 모델이 권장된다.
- **스마트워치 내장 처리 시**: 추론 속도가 월등히 빠른 Early-stage Fusion 모델이 현실적인 선택지가 된다.

### 3. 해석 가능성 (Explainability)

Grad-CAM을 통해 모델이 특정 수면 단계에서 어떤 특징(예: HRV의 HF, LF 성분 또는 활동량)에 주목하는지 시각화할 수 있었다. 이는 딥러닝의 '블랙박스' 문제를 완화하고, 의료 전문가나 사용자가 시스템의 결과에 대해 신뢰를 가질 수 있게 하는 중요한 장치가 된다.

### 4. 한계점

본 연구에서는 윈도우 길이를 101로 설정하였는데, 이로 인해 예측 시점이 윈도우 중심에 위치하게 되어 Grad-CAM 시각화 시 하이라이트 영역이 예측 시점보다 뒤로 밀리는 '윈도우 바이어스(Window bias)'가 발생하였다. 또한, 일부 사용자들은 시각화 도구가 오히려 이해를 방해한다는 피드백을 주어, 개인화된 설명 방식에 대한 추가 연구가 필요함을 시사한다.

## 📌 TL;DR

본 논문은 심장 및 움직임 센싱 데이터를 활용한 3단계 수면 분류를 위해 3가지 융합 전략과 3가지 융합 방법을 체계적으로 분석한 연구이다. 실험 결과, 단순히 데이터를 합치는 것보다 **Attention 메커니즘이나 Late-stage/Hybrid 융합**을 사용하는 것이 성능을 크게 향상시킴을 보였으며, 특히 **Attention-on-Mov** 방식이 효과적임을 입증하였다. 이 연구는 소비자용 웨어러블 기기를 통한 대규모 장기 수면 모니터링의 기술적 기반을 마련하였으며, Grad-CAM을 통한 모델 해석 가능성을 제시하여 실용성을 높였다.
