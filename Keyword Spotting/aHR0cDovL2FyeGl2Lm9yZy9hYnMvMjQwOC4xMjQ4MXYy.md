# Self-Learning for Personalized Keyword Spotting on Ultra-Low-Power Audio Sensors

Manuele Rusci, Francesco Paci, Marco Fariselli, Eric Flamand, Tinne Tuytelaars (2024)

## 🧩 Problem to Solve

본 논문은 초저전력(Ultra-Low-Power) 스마트 오디오 센서에 배포된 후, 사용자의 환경과 음성에 맞게 모델을 개인화(Personalization)하는 Keyword Spotting(KWS) 시스템의 한계를 해결하고자 한다.

일반적으로 KWS 모델은 서버에서 대규모 데이터로 사전 학습된 후 MCU(Microcontroller Unit)에 배포되어 고정된 상태로 사용된다. 이러한 방식은 배포 후 새로운 키워드를 추가하거나, 특정 소음 환경에서 정확도를 높이는 등의 온디바이스 개인화가 불가능하다는 경직성을 가진다. 최근 제안된 Few-Shot Learning 기반의 개인화 방식은 소수의 예시(Few examples)만을 사용하여 임베딩 공간에서 프로토타입(Prototype)을 생성하고 거리 기반으로 분류하지만, 특징 추출기(Feature Extractor)가 고정되어 있어 달성 가능한 정확도에 명확한 상한선(Upper bound)이 존재한다는 치명적인 문제가 있다.

결과적으로 본 연구의 목표는 레이블이 없는 데이터만을 사용하는 Self-learning 방법을 통해 배포 후 온디바이스에서 모델을 점진적으로 미세 조정(Fine-tuning)함으로써, 제한된 자원 환경에서도 개인화된 KWS의 인식 정확도를 극대화하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **의사 레이블링(Pseudo-labeling) 기반의 자기 학습(Self-learning) 프레임워크**를 초저전력 MCU 시스템에 구현하는 것이다.

핵심 직관은 사용자가 제공한 아주 적은 수의 샘플을 통해 생성한 프로토타입과 새로운 입력 데이터 간의 거리를 측정하고, 이를 통해 신뢰할 수 있는 데이터에만 의사 레이블을 부여하여 특징 추출기를 다시 학습시키는 것이다. 특히, 모든 데이터를 학습에 사용하는 대신, 거리 기반의 임계값(Threshold)을 적용하여 확실한 양성(Positive)과 음성(Negative) 샘플만을 선택적으로 추출함으로써 레이블 노이즈를 최소화하고 학습의 안정성을 확보하였다.

## 📎 Related Works

**1. 개인화된 KWS (Personalized KWS):**
기존 연구들은 DS-CNN이나 ResNet15와 같은 모델을 사용하여 MFCC 특징을 추출하고, Triplet Loss나 Prototypical Networks를 통해 클래스 간 거리를 멀게 하는 Metric Learning을 적용하였다. 하지만 대부분의 접근 방식은 배포 후 특징 추출기를 고정(Frozen)시킨 채 거리 기반 분류기만을 사용하므로, 모델 자체가 가진 표현력의 한계를 극복하지 못한다.

**2. 자기 학습 시스템 (Self-Learning Systems):**
ASR(Automatic Speech Recognition) 분야에서는 모델이 생성한 전사(Transcription) 데이터를 의사 레이블로 사용하여 모델을 개선하는 방식이 제안되었다. 그러나 이러한 방식은 대개 수백만 개의 샘플로 학습된 거대 모델(Teacher model)을 전제로 한다. 본 논문은 이와 달리 아주 적은 샘플만 존재하는 Few-shot 상황에서 거리 기반 결정 규칙을 통해 의사 레이블을 생성한다는 점에서 차별점을 가진다.

**3. 온디바이스 학습 (On-Device Learning):**
최근 MCU 상에서 역전파(Backpropagation) 비용을 줄이기 위해 일부 레이어만 학습시키거나 Bias 항만 학습시키는 연구들이 진행되었다. 본 논문은 이러한 하드웨어 제약 조건을 고려하여 GAP9 MCU와 PULP-TrainLib 라이브러리를 활용, 실제 시스템에서 학습 가능성을 검증하였다.

## 🛠️ Methodology

### 전체 시스템 구조
제안된 self-learning 프레임워크는 크게 세 단계의 파이프라인으로 구성된다: **온디바이스 캘리브레이션 $\rightarrow$ 레이블링 $\rightarrow$ 점진적 학습**.

### 1. 온디바이스 캘리브레이션 (On-Device Calibration)
사용자가 제공한 소수의 양성 샘플 $\{x^p_i\}_{i=1}^K$와 음성 샘플 $\{x^n_i\}_{i=1}^K$를 사용하여 레이블링 작업에 필요한 파라미터를 설정한다.
- **프로토타입 계산:** 양성 샘플들의 임베딩 벡터 $z = f(x)$의 평균을 구하여 프로토타입 $c^p$를 생성한다.
  $$c^p = \frac{1}{K} \sum_{i=1}^K z^p_i$$
- **임계값 설정:** 양성/음성 샘플과 프로토타입 사이의 평균 거리 $dist^p, dist^n$을 계산하고, 이를 바탕으로 하한 임계값 $Th_L$과 상한 임계값 $Th_H$를 설정한다.
  $$Th(\tau) = dist^p + \tau \cdot (dist^n - dist^p)$$
  여기서 $\tau_L$은 양성 샘플을 선택하는 보수적인 기준이 되며, $\tau_H$는 확실한 음성 샘플을 구분하는 기준이 된다.

### 2. 레이블링 (Labeling)
실시간으로 입력되는 오디오 스트림에 대해 슬라이딩 윈도우($T=1\text{s}$)를 적용하여 거리를 측정한다.
- **거리 측정 및 필터링:** 현재 프레임 $x(t)$의 임베딩과 프로토타입 $c^p$ 사이의 유클리드 거리를 계산하고, 저역통과필터(LPF)를 적용하여 $dist^f(t)$를 얻는다.
- **의사 레이블 부여:**
  - $\min_t dist^f(t) < Th_L$: **Pseudo-positive** (양성으로 간주)
  - $\min_t dist^f(t) > Th_H$: **Pseudo-negative** (음성으로 간주)
  - 그 외 범위: 레이블을 부여하지 않고 버림으로써 오분류 위험을 방지한다.

### 3. 점진적 학습 (Incremental Training)
수집된 의사 레이블 데이터셋을 사용하여 특징 추출기 $f(\cdot)$를 미세 조정한다.
- **손실 함수:** Triplet Loss를 사용하여 양성 샘플 간의 거리는 좁히고, 음성 샘플과의 거리는 멀게 학습시킨다.
  $$\text{loss} = \frac{1}{N_{tr}} \sum_{i=1}^{N_{tr}} \max(d(z^p_{1i}, z^p_{2i}) - d(z^p_{1i}, z^n_i) + m, 0)$$
  여기서 $z^p_1$은 의사 양성 샘플, $z^p_2$는 사용자가 제공한 실제 양성 샘플, $z^n$은 의사 음성 샘플이다.
- **학습 절차:** Adam 옵티마이저를 사용하며, 고정된 에포크(Epoch) 동안 학습을 진행한 후 새로운 프로토타입을 재계산하여 분류기를 업데이트한다.

### 하드웨어 구현
- **시스템 구성:** Vesper VM3011 마이크와 GreenWaves GAP9 MCU, 외부 RAM(64MB)으로 구성된다.
- **에너지 효율화:** MFCC 계산과 DNN 추론은 GAP9의 Cluster(멀티코어 및 가속기)에서 처리하고, 전체 제어는 FC(Fabric Controller) 코어가 담당하여 불필요한 전력 소모를 줄였다.

## 📊 Results

### 실험 설정
- **모델:** DS-CNN-S, DS-CNN-M, DS-CNN-L, ResNet15 (파라미터 수 최대 0.5M)
- **데이터셋:** HeySnips, HeySnapdragon (공공 데이터셋) 및 HeySnips-REC (실제 수집 데이터)
- **평가 지표:** FAR(False Acceptance Rate) = 0.5/hour 기준 인식 정확도

### 주요 결과
**1. 인식 정확도 향상:**
- 사전 학습된 모델(Pretrained) 대비 self-learning 적용 시 모든 모델에서 정확도가 향상되었다.
- 특히 가장 작은 모델인 **DS-CNN-S**에서 가장 큰 폭의 성능 향상이 나타났다 (HeySnips +19.2%, HeySnapdragon +16.0%).
- **ResNet15**는 가장 높은 최종 정확도(최대 94.6%)를 달성하였다.

**2. 실제 수집 데이터(HeySnips-REC) 결과:**
- 리버브와 소음이 존재하는 실제 환경에서도 self-learning은 효과적이었으며, DS-CNN-L 모델의 경우 사전 학습 모델 대비 평균 **+13.3%**의 정확도 향상을 보였다.

**3. 시스템 전력 및 에너지 소모:**
- **레이블링 작업:** 평균 전력 소모는 모델에 따라 $6.1\text{mW}$(DS-CNN-S)에서 $8.2\text{mW}$(ResNet15) 사이이며, 실시간 처리가 가능하다.
- **학습 작업:** DS-CNN-L 기준 학습 시간은 약 2.9분이며, 샘플링 간격을 적절히 조절할 경우 학습에 드는 에너지가 레이블링 에너지의 $10\times$ 이하가 될 수 있음을 확인하였다.

## 🧠 Insights & Discussion

**1. Triplet Loss의 강건성:**
의사 레이블링 과정에서 일부 오류(False Positive)가 발생함에도 불구하고 성능이 향상된 이유는 Triplet Loss의 특성 때문이다. Triplet Loss는 절대적인 분류보다는 상대적인 거리 관계를 최적화하므로, 일부 잘못된 샘플이 포함되어도 전체적인 임베딩 공간의 구조를 개선하는 효과가 있다.

**2. 모델 용량과 레이블 품질의 상관관계:**
모델의 용량이 클수록(예: ResNet15) 초기 프로토타입의 품질이 좋아 의사 레이블의 정확도가 높았다. 반면 작은 모델은 초기 레이블 품질은 낮지만, self-learning을 통한 성능 향상 폭이 훨씬 컸다. 이는 self-learning이 작은 모델의 표현력 한계를 보완하는 데 매우 효과적임을 시사한다.

**3. 하드웨어 제약과 트레이드-오프:**
윈도우 스트라이드($T_S$)를 줄이면 정확도는 올라가지만 전력 소모가 증가한다. 본 논문은 $T_S = 0.125\text{s}$를 에너지와 정확도 사이의 최적의 타협점으로 제시하였다.

**4. 한계점:**
매우 적은 데이터 환경(Extreme low-data regime)에서는 일부 사용자에게 대해 오히려 성능이 하락하는 경우가 관찰되었으며, 이는 향후 연구 과제로 남겨두었다.

## 📌 TL;DR

본 논문은 초저전력 MCU 환경에서 레이블 없는 데이터를 활용해 KWS 모델을 스스로 개선하는 **Self-learning 프레임워크**를 제안한다. 거리 기반의 **의사 레이블링(Pseudo-labeling)** 전략과 **Triplet Loss**를 이용한 점진적 학습을 통해, 고정된 모델의 성능 한계를 극복하고 인식 정확도를 최대 **19.2%** 향상시켰다. 특히 GAP9 MCU 상에서 실시간 레이블링($<8.2\text{mW}$)과 효율적인 온디바이스 학습이 가능함을 입증하여, 배터리 기반 스마트 센서의 실질적인 개인화 가능성을 제시하였다.