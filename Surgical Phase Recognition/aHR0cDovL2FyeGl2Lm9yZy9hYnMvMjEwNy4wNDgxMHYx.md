# Not End-to_End: Explore Multi-Stage Architecture for Online Surgical Phase Recognition

Fangqiu Yi and Tingting Jiang (2021)

## 🧩 Problem to Solve

본 논문은 수술 비디오의 각 프레임에서 현재 어떤 수술 단계(phase)가 진행 중인지 실시간으로 예측하는 **Online Surgical Phase Recognition** 문제를 다룬다. 수술 단계 인식은 수술 과정 모니터링, 외과의 일정 관리, 수술 팀 간의 협업 강화 등 컴퓨터 보조 수술 시스템(computer assisted surgery systems)에서 매우 중요한 역할을 한다.

기존의 많은 컴퓨터 비전 작업에서는 초기 예측 단계(predictor stage)와 이를 보정하는 정제 단계(refinement stage)로 구성된 **Multi-stage Architecture**가 효과적으로 사용되어 왔다. 하지만 저자들은 수술 단계 인식 작업에 단순히 Multi-stage Architecture를 적용하고 이를 **End-to-End(E2E)** 방식으로 학습시켰을 때, 정제 단계의 성능 향상이 매우 제한적이라는 문제점을 발견하였다. 따라서 본 연구의 목표는 Multi-stage Architecture가 수술 단계 인식에서 제대로 작동하지 않는 원인을 분석하고, 이를 해결하기 위한 **Non end-to-end 학습 전략**을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 예측 단계와 정제 단계를 분리하여 학습시키는 **Non end-to-end training strategy**이다. 저자들은 E2E 학습이 실패하는 두 가지 주요 원인으로 '학습과 추론 단계 사이의 입력 데이터 분포 격차(distribution gap)'와 '소규모 데이터셋으로 인한 과적합(overfitting)'을 꼽았다.

이를 해결하기 위해 정제 단계의 학습 시, 실제 추론 과정에서 발생할 법한 오류가 포함된 **Disturbed Prediction Sequences(방해된 예측 시퀀스)**를 인위적으로 생성하여 학습에 사용함으로써, 정제 모델이 예측 모델의 실수를 효과적으로 보정하는 방법을 배우도록 설계하였다.

## 📎 Related Works

수술 단계 인식 모델은 크게 두 가지 그룹으로 나뉜다.

1. **Single-stage 모델**: 시각적 특징(visual features)을 입력받아 즉시 예측 결과를 출력한다. Dynamic Time Warping, Conditional Random Field(CRF), Hidden Markov Model(HMM)의 변형이나 RNN/LSTM 기반의 모델들이 이에 해당한다.
2. **Multi-stage 모델**: 예측 단계 위에 정제 단계를 추가로 쌓아 결과를 보정한다. 대표적으로 Czempiel 등이 제안한 TeCNO가 있으며, Causal TCN을 사용하여 초기 예측을 수행하고 또 다른 Causal TCN으로 이를 정제하는 구조를 가진다.

기존 Multi-stage 접근 방식의 한계는 앞서 언급했듯이 E2E 학습 방식으로는 정제 단계가 실제 추론 시의 노이즈 섞인 입력에 적응하지 못해 성능 향상이 미비하다는 점이다.

## 🛠️ Methodology

본 논문은 전체 시스템을 예측 단계(Predictor Stage)와 정제 단계(Refinement Stage)로 나누고, 이를 각각 독립적으로 학습시키는 구조를 제안한다.

### 2.1 Predictor Stage

예측 단계에서는 효율성과 성능이 검증된 **Causal TCN**을 사용한다. 입력 데이터는 사전 학습된 CNN에서 추출된 프레임별 시각적 특징이다. 출력값 $\hat{y}_p \in \mathbb{R}^{C \times T}$는 각 프레임에 대한 $C$개 클래스의 분류 확률 벡터이다.

학습을 위한 손실 함수 $L_p$는 교차 엔트로피(Cross-Entropy) 손실과 인접한 두 프레임 간의 예측값 차이를 줄이는 Smoothing Loss의 합으로 정의된다.

$$L_p = \frac{1}{T} \sum_{t=1}^{T} -\log(\hat{y}_p(c,t)) + \frac{1}{TC} \sum_{m=1}^{C} \sum_{t=1}^{T-1} |\hat{y}_p(m,t) - \hat{y}_p(m,t+1)|^2$$

### 2.2 Refinement Stage

정제 단계는 예측 단계의 출력값 $\hat{y}_p$를 입력으로 받아 최종 결과 $\hat{y}_r$을 출력한다. 추론 시에는 예측 모델의 불완전함으로 인해 $\hat{y}_p$에 많은 오류가 포함되어 있으므로, 학습 단계에서도 이를 모사한 **Disturbed Prediction Sequences**를 생성하여 학습시킨다.

**Disturbed Sequence 생성 방법:**

1. **Mask-Hard-Frame Type**: 시각적으로 구분이 어려운 'Hard frames'를 식별하고, 해당 프레임의 이미지에 블랙 마스크를 씌워 정보를 제거한 뒤 예측 모델에 통과시켜 의도적으로 오류가 섞인 예측 시퀀스를 생성한다.
2. **Cross-Validate Type**: 학습 데이터를 $K$개 그룹으로 나누어, $K-1$개 그룹으로 학습시킨 예측 모델이 나머지 1개 그룹(unseen data)에 대해 예측하게 함으로써 실제 추론 상황과 유사한 오류 시퀀스를 확보한다.

정제 단계의 학습에는 Cross-Entropy Loss를 사용하며, 모델 구조로는 TCN, Causal TCN, GRU 세 가지를 평가하였다. 추론 시에는 두 단계가 순차적으로 연결되어 End-to-End 형태로 동작한다.

## 📊 Results

### 실험 설정

- **데이터셋**: M2CAI16 Workflow Challenge (41개 비디오), Cholec80 (80개 비디오)
- **평가 지표**: Accuracy (Acc), Jaccard Index (JACC), Recall (Rec)
- **비교 대상**: Single-stage 모델, E2E 학습 기반 Multi-stage 모델, 제안하는 Non E2E Multi-stage 모델

### 주요 결과

1. **E2E vs Non E2E**: Table 1에 따르면, E2E 방식으로 학습한 Multi-stage 모델은 Single-stage 모델과 비슷하거나 오히려 낮은 성능(과적합 때문)을 보였다. 반면, 제안한 Non E2E 전략을 사용했을 때 모든 정제 모델(GRU, Causal TCN, TCN)에서 성능이 크게 향상되었다.
2. **Stacked GRU 효과**: 정제 단계에 GRU를 여러 층 쌓았을 때(Stacked GRU), Cholec80에서는 3층, M2CAI16에서는 2층을 쌓았을 때 최적의 성능을 보였다. 이는 점진적인 정제(incremental refinement)가 효과적임을 시사한다.
3. **Disturbed Sequence의 영향**: Mask-Hard-Frame(mhf)과 Cross-Validate(cv) 타입을 함께 사용했을 때 가장 높은 성능을 기록하였다. 단순 랜덤 마스킹(random-mask)보다 Hard frame을 타겟팅한 마스킹이 더 효과적이었다.
4. **SOTA 비교**: 제안 모델은 기존 SOTA 모델인 SV-RCNet, TeCNO 등을 상회하는 성능을 기록하였다. 특히 TeCNO와 동일한 구조임에도 학습 전략의 차이만으로 큰 성능 향상을 이끌어냈다.

## 🧠 Insights & Discussion

본 연구는 Multi-stage 구조가 이론적으로는 타당함에도 불구하고 왜 실제 수술 단계 인식에서 효과가 없었는지를 분석하여, **학습-추론 간의 데이터 분포 불일치**라는 핵심 원인을 찾아냈다. E2E 학습 시 예측 단계가 빠르게 수렴하면 정제 단계는 '거의 정답인' 데이터만 보게 되어, 정작 추론 시의 '틀린' 데이터를 처리하는 능력을 기르지 못한다는 통찰은 매우 중요하다.

또한, 수술 데이터셋의 규모가 작기 때문에 모델의 파라미터가 증가하는 Multi-stage 구조에서 과적합이 발생하기 쉽다는 점을 지적하며, 단계를 분리해 학습함으로써 이를 완화할 수 있음을 보였다.

다만, 본 논문에서는 정제 모델로 TCN, GRU 등 기본적인 시퀀스 모델만을 탐색하였으며, 더 복잡한 Transformer 기반의 정제 모델이나 다른 형태의 데이터 증강 기법이 적용되었을 때의 결과는 제시되지 않았다.

## 📌 TL;DR

이 논문은 수술 단계 인식에서 Multi-stage Architecture의 성능 저하 원인이 End-to-End 학습 시 발생하는 **학습-추론 간 입력 분포 격차**와 **과적합**에 있음을 밝히고, 이를 해결하기 위해 예측/정제 단계를 분리 학습시키는 **Non end-to-end 전략**을 제안하였다. 특히 실제 오류를 모사한 **Disturbed Prediction Sequences**를 통해 정제 모델을 학습시킴으로써 SOTA 수준의 성능 향상을 달성하였다. 이 연구는 데이터셋이 제한적인 도메인에서 Multi-stage 모델을 어떻게 효과적으로 학습시킬 수 있는지에 대한 중요한 가이드라인을 제공한다.
