# Multi Agent System for Machine Learning Under Uncertainty in Cyber Physical Manufacturing System

Bang Xiang Yong and Alexandra Brintrup (2021)

## 🧩 Problem to Solve

현대 제조 산업에서 머신러닝(Machine Learning, ML)의 적용이 증가하고 있으나, 대부분의 연구는 예측 정확도(predictive accuracy)를 최대화하는 것에만 집중하고 있다. 하지만 사이버 물리 제조 시스템(Cyber-Physical Manufacturing System, CPMS)과 같이 동적이고 불확실한 환경에서는 정확도만을 추구할 경우 과적합(overfitting)의 위험이 있으며, 이는 센서 고장이나 모델의 과신(overconfidence)으로 인한 잘못된 예측으로 이어질 수 있다. 이러한 불확실성을 처리하지 못하는 시스템은 결국 제조 현장에서의 신뢰도를 떨어뜨려 기술 도입을 저해하는 요소가 된다. 따라서 본 논문의 목표는 CPMS 내 머신러닝 시스템이 불확실성 하에서도 안정적으로 작동하기 위한 성공 기준을 수립하고, 확률론적 머신러닝(probabilistic ML)을 활용하여 이러한 기준을 달성할 수 있는 Multi-Agent System(MAS) 아키텍처를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 머신러닝 모델의 예측 결과와 함께 그에 따른 불확실성을 정량화하고, 이를 자율적으로 처리할 수 있는 분산형 에이전트 구조를 설계하는 것이다. 구체적으로는 Bayesian Neural Networks(BNN)를 통해 예측의 확실성(certainty)을 측정하고, 이를 기반으로 의사결정을 내리거나 모델을 업데이트하는 체계를 제안한다. 이를 통해 사용자는 모델이 '단순한 추측'을 하는지 아니면 '근거 있는 예측'을 하는지를 판단할 수 있으며, 시스템은 불확실성이 높은 상황에서 스스로 재구성(reconfiguration)하거나 경고를 보냄으로써 시스템의 안전성(Safety)과 해석 가능성(Interpretability)을 높일 수 있다.

## 📎 Related Works

기존 연구들에서는 Bayesian Networks(BN)나 Gaussian Process(GP) 등을 사용하여 제조 공정의 불확실성을 정량화하려는 시도가 있었다. 예를 들어, BN은 다단계 제조 공정의 모니터링 및 결함 진단에 사용되었으며, GP는 공구의 에너지 예측이나 베어링의 잔존 수명 예측 등에 활용되었다. 그러나 기존 접근 방식들은 주로 합성 데이터셋(synthetic datasets)을 사용하거나 고차원 입력 데이터에 대해 확장성(scalability)이 부족하다는 한계가 있다. 또한, 많은 연구가 중앙 집중식 시스템에서 수행되어 실제 제조 환경의 분산된 특성을 반영하지 못했다. 본 논문은 단순히 불확실성을 측정하는 것을 넘어, CPMS라는 복잡한 환경 내에서 이를 어떻게 자율적으로 처리하고 행동(acting upon)으로 옮길 것인가에 대한 프레임워크를 제시함으로써 기존 연구와 차별점을 둔다.

## 🛠️ Methodology

### 전체 시스템 구조 (Multi-Agent System Architecture)

본 논문은 유연성, 확장성, 이질성 및 자가 치유 능력을 확보하기 위해 다음과 같은 6가지 역할을 가진 에이전트 기반 아키텍처를 제안한다.

1. **Sensor Agent**: 물리적 환경의 센서와 인터페이스하며 데이터를 수집하여 Aggregator Agent에게 전달한다.
2. **Aggregator Agent**: 서로 다른 소스에서 오는 데이터를 수집하고 통합하여 Predictor Agent와 데이터베이스로 전송한다.
3. **Predictor Agent**: 배포된 확률론적 모델을 실행하여 예측값과 불확실성을 계산한다.
4. **Model Trainer Agent**: 모델을 학습, 업데이트 및 배포하며 필요 시 Predictor Agent를 생성하거나 제거한다.
5. **Decision Maker Agent**: 예측값과 불확실성을 전달받아 설정된 임계값(threshold)을 기준으로 최종 결정을 내리며, 모델 업데이트 요청을 Model Trainer Agent에게 보낸다.
6. **User Interface Agent**: 사용자에게 시스템 상태, 센서 데이터, 예측 결과 및 불확실성을 시각화하여 제공한다.

### 머신러닝 모델 및 학습 절차

본 연구에서는 불확실성 정량화를 위해 Bayesian Neural Network(BNN)를 사용한다.

- **데이터 전처리**: 유압 시스템 데이터셋을 사용하여 시간 및 주파수 도메인에서 평균, 표준편차, 왜도(skewness), 첨도(kurtosis) 등 272개의 특징(feature)을 추출하고 정규화를 수행한다.
- **네트워크 구조**: 3개의 은닉층을 가지며, 노드 구성은 $272 \rightarrow 544 \rightarrow 272$ 구조이다.
- **학습 방법**: 각 노드의 가중치를 단일 값이 아닌 확률 분포로 표현하며, Bayes by Backprop 알고리즘과 Adam optimizer를 사용하여 학습한다.
- **추론 및 불확실성 측정**: Monte Carlo(MC) 샘플링을 통해 모델을 50회 샘플링하여 예측값의 분포를 얻는다. 이때 가장 많이 나타난 클래스를 예측 클래스(modal class)로 선택하고, 해당 클래스의 비율을 확실성(certainty) 척도로 사용한다.

### 의사결정 프로세스

Decision Maker Agent는 Predictor Agent가 보낸 확실성 값을 확인한다. 본 실험에서는 확실성 임계값을 $80\%$로 설정하여, 이보다 높으면 'Certain', 낮으면 'Uncertain'으로 분류하여 사용자에게 경고를 보내거나 후속 조치를 취한다.

## 📊 Results

### 실험 설정

- **데이터셋**: 유압 시스템의 상태 모니터링을 위한 공개 데이터셋을 사용하였다.
- **작업**: 쿨러(Cooler), 밸브(Valve), 내부 펌프(Internal Pump), 어큐뮬레이터(Accumulator), 안정성(Stability)의 5가지 상태 분류 작업을 수행하였다.
- **지표**: F1-Score와 더불어, 예측이 확실할 때(Certain)와 불확실할 때(Uncertain) 각각 예측이 정확할 확률인 $P(\text{Accurate}|\text{Certain})$ 및 $P(\text{Accurate}|\text{Uncertain})$을 측정하였다.

### 정량적 결과

실험 결과, 모든 작업에서 예측의 확실성이 정확도와 밀접한 관련이 있음이 입증되었다.

| Classification Task | F1-Score | $P(\text{Accurate}|\text{Certain})$ | $P(\text{Accurate}|\text{Uncertain})$ |
| :--- | :---: | :---: | :---: |
| Cooler Condition | $0.99 \pm 0.00$ | $99.70 \pm 0.05$ | $63.57 \pm 5.13$ |
| Valve Condition | $0.84 \pm 0.005$ | $95.46 \pm 0.16$ | $64.69 \pm 0.66$ |
| Internal Pump Leakage | $0.90 \pm 0.002$ | $96.56 \pm 0.23$ | $66.91 \pm 0.45$ |
| Hydraulic Accumulator | $0.76 \pm 0.019$ | $95.84 \pm 0.19$ | $61.26 \pm 1.21$ |
| Stable Flag | $0.92 \pm 0.008$ | $96.84 \pm 0.57$ | $59.92 \pm 2.43$ |

특히 Cooler Condition의 경우, 예측이 확실할 때의 정확도는 $99.70\%$에 달하지만, 불확실할 때는 $63.57\%$로 급격히 떨어진다. 이는 런타임 중에 정답을 알 수 없는 상황에서 불확실성 지표가 예측의 신뢰도를 판단하는 유효한 지표가 될 수 있음을 보여준다.

## 🧠 Insights & Discussion

본 논문은 머신러닝의 불확실성을 정량화하는 것이 단순히 학술적인 분석에 그치지 않고, 실제 제조 현장에서의 **안전성(Safety)**과 **해석 가능성(Interpretability)**을 확보하는 핵심 수단이 될 수 있음을 시사한다. 불확실성 지표를 통해 모델의 예측이 '무작위 추측'인지 '근거 있는 추측'인지를 구분함으로써, 운영자는 불확실성이 높은 예측 결과를 배제하거나 추가 데이터를 수집하는 등의 능동적인 대처가 가능하다.

또한, 제안된 MAS 아키텍처는 하드웨어 상호운용성과 유연성을 제공하여, 센서의 추가/제거와 같은 동적인 환경 변화에 유연하게 대응할 수 있는 구조를 갖추고 있다. 하지만 본 연구는 프로토타입 수준의 구현이며, 실제 클라우드나 엣지 컴퓨팅 환경에서의 분산 처리 성능 및 통신 오버헤드에 대한 상세한 분석은 부족하다는 한계가 있다. 향후 연구에서는 다양한 시나리오(센서 고장, 다중 모델 융합 등)에서의 자율적 대응 메커니즘을 구체화할 필요가 있다.

## 📌 TL;DR

본 연구는 사이버 물리 제조 시스템(CPMS)에서 머신러닝 모델의 불확실성을 정량화하고 이를 관리하기 위한 Multi-Agent System(MAS) 아키텍처를 제안하였다. Bayesian Neural Networks를 통해 예측의 확실성을 측정하고, 이를 기반으로 의사결정을 내리는 구조를 설계하여, 예측의 확실성과 실제 정확도 사이의 강한 상관관계를 실험적으로 입증하였다. 이 연구는 제조 현장에서 데이터 기반 모델의 신뢰성을 높이고, 불확실한 상황에서 시스템이 자율적으로 대응할 수 있는 기반을 마련했다는 점에서 향후 자율 제조 시스템 연구에 중요한 기여를 할 가능성이 높다.
