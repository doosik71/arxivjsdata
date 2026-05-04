# Keyframe-Focused Visual Imitation Learning

Chuan Wen, Jieruiui Lin, Jianing Qian, Yang Gao, Dinesh Jayaraman (2021)

## 🧩 Problem to Solve

본 논문은 부분 관측 가능한 환경(Partially Observable Markov Decision Processes, POMDP)에서 Behavioral Cloning(BC)을 적용할 때 발생하는 **'Copycat problem'**을 해결하고자 한다. 일반적으로 에이전트가 현재의 관측치뿐만 아니라 과거의 관측 이력(Observation History)을 함께 참조하는 BC-OH(Behavioral Cloning with Observation History) 방식이 단일 관측치만 사용하는 BC-SO(Behavioral Cloning with Single Observation)보다 성능이 높아야 한다고 기대된다.

그러나 실제로는 BC-OH가 학습 데이터에 대해 더 낮은 손실 값(MSE)을 보임에도 불구하고, 실제 환경에서 실행했을 때 오히려 BC-SO보다 성능이 떨어지거나 매우 비효율적으로 동작하는 역설적인 현상이 보고되어 왔다. 이는 'Copycat problem', 'Inertia problem' 또는 'Causal confusion'이라고 불리며, 모델이 환경의 시각적 정보(Visual observation)를 학습하는 대신, 전문가의 행동 데이터에 존재하는 강한 시간적 상관관계(Temporal correlation)라는 지름길(Shortcut)을 학습하여 단순히 이전 행동을 반복하려는 경향 때문에 발생한다.

따라서 본 논문의 목표는 시각적 모방 학습(Visual Imitation Learning) 설정에서 이러한 Copycat problem을 해결하여, 관측 이력을 효과적으로 활용하면서도 전문가의 핵심 행동 변화 지점을 정확히 모방하는 정책을 학습시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 전문가의 행동 시퀀스에서 행동이 급격히 변하는 **'Changepoint(변화 지점)'**를 식별하고, 이 지점에 해당하는 데이터 샘플에 더 높은 가중치를 부여하여 학습하는 것이다.

전문가의 행동은 시간적으로 매우 유사한 경우가 많기 때문에, 단순히 이전 행동을 따라 하는 'Copycat' 정책만으로도 대부분의 데이터에서 낮은 오차를 얻을 수 있다. 하지만 환경의 변화에 대응하여 행동이 바뀌는 Changepoint에서는 Copycat 정책이 반드시 실패하게 된다. 저자들은 이 지점을 **Keyframe**으로 정의하고, 이를 집중적으로 학습함으로써 모델이 단순한 시간적 상관관계에 의존하지 않고, 행동 변화를 일으키는 실제 시각적 원인(Observation)에 주목하도록 강제한다.

## 📎 Related Works

기존의 모방 학습 연구들은 주로 분포 변화(Distribution shift) 문제를 해결하기 위해 DAGGER와 같은 온라인 상호작용 방식이나 노이즈 주입(Noise injection) 기법을 사용하였다. 특히 Copycat problem을 해결하기 위해 Causal graph learning이나 Deep information bottlenecks 같은 접근법이 제안되었으나, 이러한 방법들은 고차원의 시각적 데이터(Visual imitation) 설정으로 확장하는 데 한계가 있었다.

또한, 데이터 불균형을 해결하기 위한 샘플 재가중치(Sample reweighting) 기법들이 존재하지만, 대부분은 클래스 레이블이나 환경의 보상(Reward) 함수와 같은 명시적인 정보에 의존한다. 반면, 본 논문의 제안 방식은 레이블이나 보상 함수 없이 전문가의 행동 시퀀스 자체에서 무감독(Unsupervised) 방식으로 Changepoint를 발견하여 가중치를 부여한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. Copycat Problem의 분석
저자들은 Copycat problem이 발생하는 조건을 수식으로 분석한다. 보상 최적 정책(Reward-optimal policy)의 학습 오차를 $MSE_D(\theta^{R*})$라고 하고, 최적의 Copycat 정책(이전 행동만으로 현재 행동을 예측하는 정책)이 틀리는 Changepoint의 비율을 $\epsilon_{CP}$라고 할 때, 다음과 같은 조건에서 Copycat problem이 발생한다.

$$MSE_D(\theta^{R*}) > \epsilon_{CP}$$

즉, 정답을 맞히기 위한 최적 정책의 오차보다 Copycat 정책의 오차가 더 작을 때, 학습기는 더 쉬운 길인 Copycat 솔루션을 선택하게 된다.

### 2. Action Prediction Error (APE)
Changepoint를 자동으로 식별하기 위해 저자들은 **Action Prediction Error (APE)**라는 지표를 제안한다.

먼저, 오직 과거의 행동들 $[a_{t-1}, a_{t-2}, \dots]$만을 입력으로 받아 현재 행동 $a_t$를 예측하는 작은 MLP 네트워크인 Copycat 정책 $\psi^*$를 학습시킨다. 학습 목표는 다음과 같다.

$$\psi^* = \arg \min_{\psi} \frac{1}{N} \sum_{t=0}^{N} (\psi(a_{t-1}, a_{t-2}, \dots) - a_t)^2$$

이후, 모든 학습 샘플에 대해 다음과 같이 APE를 계산한다.

$$APE_t = (\psi^*(a_{t-1}, a_{t-2}, \dots) - a_t)^2$$

$APE_t$ 값이 높다는 것은 이전 행동만으로는 현재 행동을 예측할 수 없는 지점, 즉 전문가가 환경의 변화에 반응하여 행동을 바꾼 Changepoint임을 의미한다.

### 3. 재가중치 부여된 BC 목적 함수 (Reweighted BC Objective)
계산된 $APE_t$를 기반으로 각 샘플에 가중치 $w_t = f(APE_t)$를 부여하여 정책 $\pi_\theta$를 학습시킨다. 최종 목적 함수는 다음과 같다.

$$\theta^* = \arg \min_{\theta} \sum_{t=0}^{N} f(APE_t)(\pi_\theta(\tilde{o}_t) - a_t)^2$$

여기서 $f(\cdot)$은 $APE_t$가 클수록 큰 값을 갖는 단조 증가 함수이며, 본 논문에서는 **Softmax** 함수와 **Step** 함수 두 가지를 사용하였다.

- **Softmax**: $\text{weight}_i = \frac{e^{\tau APE_i}}{\sum_j e^{\tau APE_j}}$ (온도 파라미터 $\tau$ 사용)
- **Step**: $APE$ 상위 $THR\%$ 샘플에는 가중치 $W$를 부여하고, 나머지는 $1$을 부여한다.

## 📊 Results

### 1. 실험 설정
- **CARLA**: 실사 도시 주행 시뮬레이터. `%success`, `#collision`, `%progress`, `avg. speed`를 측정한다.
- **MuJoCo-Image**: Hopper, HalfCheetah, Walker2D 환경에서 RGB 이미지를 입력으로 하여 Reward를 측정한다.
- **Baselines**: BC-SO, BC-OH, HistoryDropout, FCA, DAGGER 등과 비교하였다.

### 2. 주요 결과
- **CARLA**: BC-OH는 BC-SO보다 성능이 낮게 나타나 Copycat problem이 명확히 확인되었다. 제안 방법(Ours)은 모든 이력 기반 베이스라인을 압도하며 BC-SO와 대등하거나 더 나은 성능을 보였다. 특히-속도 정보가 제거된 **CARLA-w/o-speed** 설정에서 BC-OH 대비 비약적인 성능 향상을 보여, 관측 이력을 효과적으로 활용하고 있음을 증명하였다.
- **MuJoCo-Image**: 세 가지 로봇 제어 환경 모두에서 제안 방법(특히 Softmax 가중치 적용 시)이 모든 오프라인 베이스라인보다 높은 보상을 획득하였다. 온라인 상호작용을 사용하는 DAGGER(1k queries)와 비교했을 때도 경쟁력 있는 성능을 보였다.

### 3. 분석 및 검증
- **Changepoint 성능**: 검증 데이터셋에서 $APE$가 높은 샘플(Changepoint)에 대해 제안 방법이 BC-OH보다 훨씬 낮은 오차를 보였다. 이는 모델이 핵심 프레임에서 더 정확하게 동작함을 의미한다.
- **분포 변화(Distribution Shift)**: 정책을 실제 환경에서 실행하여 생성한 데이터와 전문가 데이터 사이의 오차(Rollout imitation error)를 측정했을 때, 제안 방법이 BC-OH보다 낮은 수치를 기록하여 분포 변화 문제도 어느 정도 완화됨을 확인하였다.
- **Copycat 경향성**: 정책 $\pi$가 생성한 행동 시퀀스의 $APE$를 측정하는 $\text{avgAPE}(\pi)$ 지표를 통해, 제안 방법이 BC-OH보다 덜 'Copycat'스럽게(즉, 더 역동적으로) 행동함을 확인하였다.

## 🧠 Insights & Discussion

본 논문은 단순한 MSE 최소화가 오프라인 모방 학습에서 최적이 아닐 수 있음을 시사한다. 전문가 데이터에 존재하는 강한 시간적 상관관계는 학습기에게 일종의 '편향'으로 작용하며, 모델은 실제 환경의 인과관계를 배우는 대신 통계적 지름길을 선택한다.

제안 방법의 강점은 매우 단순한 전처리 단계(Copycat MLP 학습 및 가중치 계산)만으로 고차원 시각 데이터에서도 확장 가능한 솔루션을 제공한다는 점이다. 또한, 모든 샘플을 동일하게 학습시키는 것보다, 보상 최적화 관점에서 더 중요한 '핵심 프레임'에 집중하는 것이 실제 환경에서의 성과(Reward)를 높이는 데 효율적임을 보여주었다.

다만, 가중치 함수 $f(\cdot)$의 기울기가 너무 가파르면 일반적인 프레임(Ordinary frames)에 대해 과소적합(Underfitting)이 발생할 수 있다는 한계가 있으며, 전문가 데이터에 노이즈가 섞여 있을 경우 $APE$가 높게 측정되어 잘못된 샘플에 과도한 가중치가 부여될 위험이 있다.

## 📌 TL;DR

본 논문은 시각적 모방 학습에서 모델이 이전 행동을 단순히 반복하는 **Copycat problem**을 해결하기 위해, 행동의 변화 지점인 **Changepoint(Keyframe)**를 식별하고 이에 가중치를 부여하여 학습하는 방법을 제안한다. 단순한 MLP를 통해 행동 예측 오차(APE)를 계산함으로써 핵심 프레임을 찾아내며, 이를 통해 CARLA 주행 및 MuJoCo 로봇 제어 작업에서 기존의 이력 기반 모방 학습 방식보다 월등한 성능 향상을 이루었다. 이 연구는 복잡한 환경에서 관측 이력을 활용하는 모방 학습의 안정성을 높이는 매우 실용적이고 효율적인 방법론을 제시한다.