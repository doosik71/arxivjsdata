# Imitation Learning for Generalizable Self-Driving Policy with Sim-to-Real Transfer

Zoltán Lőrincz, Márton Szemenyei, Róbert Moni (2022)

## 🧩 Problem to Solve

본 연구는 자율주행 로봇의 제어 정책(Policy)을 학습시킬 때 발생하는 실제 환경과 시뮬레이션 환경 간의 간극, 즉 Sim-to-Real gap 문제를 해결하고자 한다. 일반적으로 로봇 학습을 실제 환경에서 직접 수행하는 것은 안전성, 비용, 그리고 시간적 제약으로 인해 매우 어렵다. 따라서 시뮬레이션 환경에서 먼저 학습을 진행한 후 이를 실제 환경에 적용하는 방식이 선호되지만, 두 환경의 시각적·물리적 차이로 인해 시뮬레이션에서 학습된 모델의 성능이 실제 환경에서는 급격히 저하되는 문제가 발생한다.

본 논문의 구체적인 목표는 Duckietown 환경에서 단일 전방 카메라 영상만을 이용하여 우측 차선을 따라 주행하는(Right-lane following) 일반화된 주행 정책을 개발하고, 이를 실제 물리적 로봇에 성공적으로 전이(Transfer)시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다양한 Imitation Learning(IL) 알고리즘과 Sim-to-Real 전이 기법들을 조합하여 실험하고, 그 효과를 정량적으로 비교 분석했다는 점에 있다. 구체적으로 세 가지의 IL 알고리즘(BC, DAgger, GAIL)과 두 가지의 Sim-to-Real 방법론(Domain Randomization, Visual Domain Adaptation)을 적용하여, 어떤 조합이 시뮬레이션에서 실제 환경으로의 전이에 가장 효과적인지를 검증하였다.

## 📎 Related Works

논문은 모방 학습의 초기 모델인 ALVINN부터, 이를 개선한 DAgger, 그리고 역강화학습(Inverse Reinforcement Learning) 기반의 GAIL까지의 흐름을 소개한다. 또한, Sim-to-Real 문제를 해결하기 위해 시뮬레이터의 파라미터를 무작위로 변경하여 모델이 일반적인 특징을 학습하게 하는 Domain Randomization(DR)과, 이미지 간 변환 네트워크를 통해 두 도메인의 관측치를 공통 영역으로 매핑하는 Visual Domain Adaptation(VDA) 기법들을 언급한다. 기존 연구들이 객체 인식이나 단순한 로봇 팔 제어에 집중했다면, 본 연구는 이를 Duckietown이라는 특정 자율주행 환경의 차선 유지 작업에 적용하여 실효성을 분석했다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. Imitation Learning 알고리즘
본 연구에서는 시뮬레이션 내에서 정책을 학습하기 위해 다음과 같은 세 가지 알고리즘을 사용하였다.

- **Behavioral Cloning (BC):** 전문가의 상태-행동 쌍(state-action pairs)을 독립적인 데이터로 간주하고, 지도 학습(Supervised Learning)을 통해 전문가의 정책을 그대로 복제하는 가장 단순한 형태의 IL이다.
- **Dataset Aggregation (DAgger):** BC의 한계를 극복하기 위해 대화형 전문가(Interactive Demonstrator)를 활용한다. 현재 정책으로 주행하며 수집한 상태에 대해 전문가의 피드백을 받아 데이터를 지속적으로 추가하며 학습함으로써, 에이전트가 과거에 저지른 오류를 수정할 수 있도록 한다.
- **Generative Adversarial Imitation Learning (GAIL):** GAN 구조를 채택한 역강화학습 알고리즘이다. 정책 네트워크(Generator)와 판별자(Discriminator)로 구성되며, 판별자는 에이전트의 궤적과 전문가의 궤적을 구분하려 하고, 정책 네트워크는 판별자를 속이도록 학습함으로써 보상 함수를 내재적으로 학습한다.

### 2. 전문가 데몬스트레이터 및 데이터 전처리
전문가 데이터는 수정된 Pure Pursuit PD 제어기를 통해 생성되었다. 단순한 Pure Pursuit에 직선과 곡선 구간의 속도 및 조향 이득(gain) 값을 다르게 설정하고 미분(Derivative) 항을 추가하여 성능을 높였다.

입력 데이터인 RGB 이미지($480 \times 640$)는 차원 축소와 학습 속도 향상을 위해 다음과 같이 전처리되었다.
- **Downscaling:** $60 \times 80$ 해상도로 축소한다.
- **Normalization:** 픽셀 값을 $[0.0, 1.0]$ 범위로 정규화한다.
- GAIL의 경우, 추가적으로 ImageNet으로 사전 학습된 ResNet을 특징 추출기로 사용하였다.

출력 행동(Action)은 PWM 신호를 직접 예측하는 대신, 가속도(Throttle, $[0, 1]$)와 조향각(Steering angle, $[-1, 1]$)이라는 두 개의 스칼라 값을 예측하도록 설계하였으며, 이를 사후에 PWM 신호로 변환하여 로봇에 적용하였다.

### 3. Sim-to-Real 전이 방법론
시뮬레이션에서 학습된 모델을 실제 환경에 적용하기 위해 두 가지 기법을 도입하였다.
- **Domain Randomization (DR):** 조명 조건, 텍스처, 카메라 파라미터, 로봇 크기 및 물리적 파라미터 등을 매 리셋마다 무작위로 변경하여 학습시킨다. 이를 통해 모델이 특정 환경에 오버피팅되지 않고 일반적인 특징을 학습하게 한다.
- **Visual Domain Adaptation (VDA-UNIT):** UNIT(Unsupervised Image-to-Image Translation) 네트워크를 사용하여 시뮬레이션 도메인 $X_{sim}$과 실제 도메인 $X_{real}$을 공통 잠재 공간(Common latent space) $Z$로 매핑한다. 이 공통 공간 $Z$에서 제어 정책을 학습함으로써, 실제 환경의 이미지가 들어와도 시뮬레이션에서 학습한 정책을 그대로 사용할 수 있게 한다.

## 📊 Results

### 1. 시뮬레이션 환경 평가
시뮬레이션에서는 주행 거리(Traveled distance), 생존 시간(Survival time), 횡방향 편차(Lateral deviation), 주요 위반 횟수(Major infractions)의 4가지 지표를 사용하였다.
- **BC와 DAgger**는 베이스라인 모델보다 주행 거리와 생존 시간 면에서 우수한 성능을 보였다.
- **GAIL**은 상대적으로 낮은 성능을 보였는데, 이는 학습 절차의 복잡성과 하이퍼파라미터 최적화의 어려움 때문으로 분석된다.
- 베이스라인 모델은 횡방향 편차가 가장 낮았으나, 이는 단순히 주행 속도가 매우 느렸기 때문인 것으로 해석된다.

### 2. 실제 환경 평가
실제 환경에서는 생존 시간과 방문한 도로 타일 수(Visited road tiles)를 측정하였다.
- **DR**과 **VDA-UNIT**을 적용한 모델은 모두 성공적으로 우측 차선을 따라 주행하며 Sim-to-Real 문제를 해결하였다.
- 반면, Sim-to-Real 기법을 적용하지 않은 **DAgger** 모델은 실제 환경에서 완전히 실패하였다. 이는 시뮬레이션과 실제 환경의 시각적 차이가 매우 크다는 것을 시사한다.
- VDA-UNIT의 경우, 시뮬레이션 이미지를 실제 이미지처럼, 혹은 그 반대로 변환하는 품질이 매우 높음을 확인하였다.

## 🧠 Insights & Discussion

본 연구는 자율주행 정책 학습에 있어 DAgger가 BC보다 약간 더 많은 학습 시간이 소요되지만, 최종 성능 면에서 가장 유리하다는 점을 보여주었다. 특히 GAIL의 경우 이론적 잠재력에도 불구하고 하이퍼파라미터 튜닝이 까다로워 실용적인 성능을 내기 어렵다는 한계가 명시되었다.

가장 중요한 통찰은 **Sim-to-Real 전이 기법의 필수성**이다. 단순한 모방 학습만으로는 실제 환경의 변수를 극복할 수 없으며, DR이나 VDA-UNIT과 같은 도메인 적응 기법이 동반되어야만 실제 로봇에 배포 가능한 수준의 정책을 얻을 수 있음을 입증하였다. 다만, 실제 환경 평가 시 정밀한 위치 측정 시스템의 부재로 인해 커스텀 지표(방문 타일 수 등)를 사용해야 했던 점은 한계로 남는다.

## 📌 TL;DR

본 논문은 Duckietown 환경에서 BC, DAgger, GAIL과 같은 모방 학습 알고리즘을 비교하고, 이를 실제 로봇에 적용하기 위해 Domain Randomization 및 UNIT 기반의 시각적 도메인 적응 기법을 적용하였다. 실험 결과, DAgger 알고리즘과 Sim-to-Real 전이 기법의 조합이 가장 효과적이었으며, 특히 전이 기법 없이는 실제 환경 주행이 불가능함을 확인하였다. 이 연구는 향후 더 복잡한 자율주행 과제에서 시뮬레이션 기반 학습 후 실제 환경으로 전이하는 파이프라인의 기초 틀을 제공한다.