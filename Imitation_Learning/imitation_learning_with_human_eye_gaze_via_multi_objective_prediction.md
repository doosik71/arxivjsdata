# Imitation Learning with Human Eye Gaze via Multi-Objective Prediction

Ravi Kumar Thakur, MD-Nazmus Samin Sunbeam, Vinicius G. Goecks, Ellen Novoseller, Ritwik Bera, Vernon J. Lawhern, Gregory M. Gremillion, John Valasek, Nicholas R. Waytowich (2023)

## 🧩 Problem to Solve

본 논문은 인간의 시연(demonstration)을 통해 학습하는 Imitation Learning(IL)에서 발생하는 샘플 효율성 및 일반화(generalization) 성능 저하 문제를 해결하고자 한다. 기존의 대부분의 IL 연구는 시연자의 행동 정보, 즉 어떤 액션(action)을 취했는가에만 집중하는 Behavioral Cloning(BC) 방식을 사용하며, 시연자가 어디에 시각적 주의(visual attention)를 기울였는지를 나타내는 시선(eye gaze) 정보를 무시하는 경향이 있다.

시선 정보는 시연자가 환경의 어떤 부분이 작업 수행에 중요하다고 판단했는지에 대한 중요한 문맥적 정보(contextual information)를 제공한다. 따라서 이러한 시선 정보를 학습 과정에 통합한다면, 더 적은 양의 시연 데이터만으로도 에이전트의 성능을 높이고, 학습 데이터에 없는 새로운 시나리오에서도 더 잘 작동하는 강건한 정책(policy)을 학습시킬 수 있을 것이라는 점이 본 연구의 핵심 목표이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 **Gaze Regularized Imitation Learning (GRIL)**이라는 새로운 다목적 예측(Multi-Objective Prediction) 아키텍처를 제안한 것이다. GRIL의 중심 아이디어는 제어 명령(control commands)과 시선 좌표(gaze coordinates)를 동시에 예측하도록 네트워크를 설계하여, 시선 예측 작업이 정책 학습의 정규화(regularization) 역할을 수행하게 하는 것이다.

단순히 시선을 입력값으로 사용하는 것이 아니라, 시선을 예측하는 보조 작업(auxiliary task)을 함께 수행함으로써 모델이 이미지 내에서 작업과 관련된 핵심 특징(relevant features)에 집중하도록 유도하는 Inductive Bias를 제공한다.

## 📎 Related Works

### 기존 접근 방식 및 한계

1. **Behavioral Cloning (BC):** 관찰값에서 액션을 직접 예측하는 단순한 방식이나, 학습 데이터에 없는 상태에 진입했을 때 성능이 급격히 떨어지는 Covariate Shift 문제에 취약하다.
2. **AGIL (Attention-Guided Imitation Learning):** 시선 예측 네트워크를 먼저 학습시켜 시선 맵(heatmap)을 생성하고, 이를 정책 네트워크의 입력으로 사용한다. 하지만 별도의 네트워크를 학습시켜야 하며, 데이터 수집 과정이 복잡하고 계산 비용이 높다.
3. **CGL (Coverage-based Gaze Loss):** 인간의 시선이 집중된 영역의 네트워크 활성화(activation)가 낮을 때 페널티를 주는 보조 손실 함수를 사용한다. 그러나 시선 좌표를 히트맵으로 변환하기 위해 커널 크기 등 하이퍼파라미터 튜닝이 까다롭고 KL-Divergence와 같은 복잡한 손실 함수를 사용해야 한다.

### GRIL의 차별점

GRIL은 시선을 입력으로 사용하거나 네트워크 가중치를 직접 제어하는 대신, 시선 좌표 자체를 직접 예측하는 **Multi-head 구조**를 채택한다. 히트맵 생성 과정이 필요 없으므로 하이퍼파라미터 튜닝이 간편하며, 단순한 MSE(Mean Squared Error) 손실 함수를 사용하여 계산 효율성을 높였다.

## 🛠️ Methodology

### 전체 시스템 구조

GRIL은 이미지 입력을 받아 액션과 시선을 동시에 출력하는 Multi-headed Convolutional Neural Network(CNN) 구조를 가진다.

1. **Feature Extractor:** 사전 학습된 MobileNet을 백본(backbone)으로 사용하여 이미지 특징을 추출하며, 시연 데이터를 통해 미세 조정(fine-tuning)을 수행한다. 이후 추가적인 Convolution 레이어를 통해 특징을 정제한다.
2. **Action Head:** 추출된 특징을 Dense 레이어에 통과시켜 최종 제어 명령(throttle, yaw, pitch, roll)을 예측한다.
3. **Gaze Head:** 동일한 특징을 다른 Dense 레이어에 통과시켜 시연자의 시선 좌표 $(x, y)$를 예측한다.

### 훈련 목표 및 손실 함수

모델은 제어 명령 예측 손실($L_{BC}$)과 시선 예측 손실($L_{GP}$)의 선형 결합으로 구성된 전체 손실 함수를 최소화하는 방향으로 학습된다.

$$L(\theta) = \lambda_1 L_{GP}(\theta) + \lambda_2 L_{BC}(\theta)$$

여기서 $\theta$는 학습 가능한 파라미터이며, $\lambda_1, \lambda_2$는 각 손실 항의 가중치를 조절하는 하이퍼파라미터이다. 각 손실 함수는 정답 값과 예측 값 사이의 MSE로 정의된다.

**Behavioral Cloning Loss:**
$$L_{BC}(\theta) = \frac{1}{M} \sum_{i=1}^{M} \| \pi_{action}(o_i|\theta) - a_i \|^2$$

**Gaze Prediction Loss:**
$$L_{GP}(\theta) = \frac{1}{M} \sum_{i=1}^{M} \| \pi_{gaze}(o_i|\theta) - g_i \|^2$$

($M$: 학습 샘플 수, $o_i$: 관찰 이미지, $a_i$: 정답 액션, $g_i$: 정답 시선 좌표)

## 📊 Results

### 실험 환경 및 설정

- **데이터셋:** Microsoft AirSim 시뮬레이터를 이용해 숲 환경에서 쿼드로터(quadrotor)가 목표 차량을 찾아 이동하는 작업을 수행하며 수집된 데이터.
- **작업:**
    1. **Stationary target:** 고정된 위치의 차량으로 이동.
    2. **Moving target:** 이동하는 차량을 추적 및 추종 (일반화 성능 평가용).
- **측정 지표:**
  - **Task Completion Rate (TCR):** 목표물 5m 이내 진입 성공률.
  - **Collision Rate (CR):** 지면 및 장애물 충돌률.
- **비교 대상:** BC, AGIL, BC-CGL.

### 주요 결과

정량적 결과(Table 1 기준)에 따르면 GRIL이 모든 지표에서 가장 우수한 성능을 보였다.

- **Stationary Task:** GRIL은 TCR 80% $\pm$ 9.9, CR 20% $\pm$ 9.9를 기록하여 BC-CGL(TCR 64%) 및 AGIL(TCR 10%)보다 월등히 높은 성능을 보였다.
- **Moving Target Task (Generalization):** 정지된 타겟으로만 학습한 모델을 이동 타겟에 적용했을 때, GRIL은 TCR 40% $\pm$ 12.3로 가장 높은 일반화 성능을 보였다. 이는 GRIL이 단순한 외형 암기가 아니라 시각적 주의 집중을 통해 타겟의 본질적인 특징을 학습했음을 시사한다.

### 시선 예측 분석

GRIL의 시선 예측 헤드는 인간 시연자의 다음과 같은 특성적 시선 패턴을 성공적으로 모사하였다.

- **Motion leading:** 타겟이 보이지 않을 때 회전 방향의 측면을 먼저 보는 패턴.
- **Target fixation:** 타겟 접근 시 타겟 상단에 시선을 고정하는 패턴.
- **Saccade:** 장애물이 많을 때 주변 장애물들을 빠르게 훑는 패턴.
- **Obstacle fixation:** 충돌 위험이 있을 때 근처 장애물을 주시하는 패턴.

## 🧠 Insights & Discussion

### GRIL의 성능 우위 원인

저자들은 GRIL이 높은 성능을 낸 이유를 **Multi-objective Learning**의 관점에서 설명한다.

1. **특징 불변성(Feature Invariance):** 액션과 시선이라는 두 가지 서로 다른 목표를 동시에 최적화함으로써, 모델은 두 작업 모두에 공통적으로 필요한 핵심 특징(invariant features)에 집중하게 되며, 불필요한 노이즈를 제거하는 효과를 얻는다.
2. **정규화 효과:** 시선 예측 작업이 일종의 정규화 도구로 작용하여 모델의 Rademacher Complexity(무작위 노이즈에 피팅되는 능력)를 낮추고 과적합(overfitting) 위험을 줄인다.
3. **단순성:** CGL과 달리 히트맵 변환 과정이 없어 하이퍼파라미터 튜닝이 쉽고, AGIL과 달리 통합된 네트워크 구조를 가져 학습해야 할 파라미터 수가 적고 효율적이다.

### 한계 및 논의

본 연구는 시뮬레이션 환경에서 수행되었으며, 단일 시연자의 데이터를 사용했다는 점이 한계로 작용할 수 있다. 또한, 시선 데이터 수집을 위해 전용 하드웨어가 필요하므로 실제 데이터 수집 비용이 발생한다. 하지만 제안된 방식이 단순한 좌표 예측만으로도 강력한 정규화 효과를 낸다는 점은 향후 다른 제어 작업이나 강화학습에 적용될 가능성이 높다.

## 📌 TL;DR

본 논문은 인간의 시선 정보를 활용해 Imitation Learning의 성능을 높이는 **GRIL(Gaze Regularized Imitation Learning)**을 제안한다. GRIL은 제어 액션과 시선 좌표를 동시에 예측하는 **Multi-objective 아키텍처**를 통해, 시선 예측 작업을 정책 학습의 정규화 도구로 활용한다. 실험 결과, 복잡한 숲 환경의 쿼드로터 비행 작업에서 기존 Gaze-based IL 기법(AGIL, CGL) 및 일반 BC보다 높은 성공률과 일반화 성능을 보였다. 이 연구는 시각적 주의 집중 정보를 단순한 좌표 예측 형태로 통합하는 것만으로도 샘플 효율성과 강건성을 크게 향상시킬 수 있음을 입증하였다.
