# Imitation Learning from Video by Leveraging Proprioception

Faraz Torabi, Garrett Warnell and Peter Stone (2019)

## 🧩 Problem to Solve

본 논문은 Imitation Learning(모방 학습)에서 발생하는 현실적인 제약 조건을 해결하고자 한다. 전통적인 모방 학습 알고리즘들은 전문가의 행동 데이터($action$)가 포함되어 있거나, 학습자와 전문가가 완전히 동일한 환경에서 데이터를 수집해야 한다는 이상적인 가정 하에 설계되었다. 그러나 실제 환경에서는 전문가의 내부 제어 신호(actions)를 얻기 어렵고, 인터넷에 공개된 수많은 비디오 데이터처럼 시각적 정보만 존재하는 경우가 많다.

이러한 한계를 극복하기 위해 최근 Imitation from Observation(IfO, 관찰을 통한 모방) 연구가 진행되고 있다. IfO는 전문가의 시각적 관찰 데이터(비디오)만을 이용하여 작업을 학습하는 것을 목표로 한다. 하지만 기존의 많은 IfO 알고리즘들은 오직 시각적 자가 관찰(visual self-observation)에만 의존하여, CNN을 통해 이미지에서 행동으로 직접 매핑하는 방식을 사용한다. 본 논문은 에이전트가 자신의 내부 상태 정보인 Proprioception(고유 수용 감각, 예: 관절 각도 및 토크)을 가지고 있음에도 이를 무시하는 것은 학습 효율성과 성능 면에서 큰 손실이라고 주장한다. 따라서 본 연구의 목표는 시각적 정보와 Proprioceptive state 정보를 동시에 활용하는 IfO 알고리즘을 제안하고 그 효과를 검증하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **정책(Policy) 학습에는 Proprioceptive state를 사용하고, 보상(Reward) 생성에는 시각적 데이터(Visual observations)를 사용**하여 두 정보의 장점을 결합하는 것이다.

기존의 GAIfO(Generative Adversarial Imitation from Observation)가 시각적 입력으로부터 행동을 생성하는 CNN 기반 정책을 사용했다면, 제안된 방법은 MLP(Multi-Layer Perceptron)를 통해 proprioceptive state에서 행동으로 매핑하는 정책을 학습한다. 반면, 학습된 정책이 생성한 행동의 결과물(비디오)을 전문가의 비디오와 비교하여 판별하는 Discriminator(판별자)는 여전히 시각적 데이터를 사용한다. 즉, "제어는 내부 상태로 정밀하게 수행하되, 정답지(보상)는 외부의 시각적 모습으로 확인한다"는 직관을 설계에 반영하였다.

## 📎 Related Works

논문에서는 모방 학습을 크게 두 가지 범주로 나눈다. 첫째는 전문가의 상태-행동 궤적을 사용하는 Behavioral Cloning(BC)과 Inverse Reinforcement Learning(IRL)이다. 둘째는 상태 정보만 사용하는 IfO로, 이는 다시 Model-based와 Model-free 방식으로 나뉜다.

- **Model-based IfO**: BCO(Behavioral Cloning from Observation), RIDM, ILPO 등이 있으며, 이들은 환경의 dynamics model을 학습하여 전문가의 행동을 추론한 뒤 모방한다.
- **Model-free IfO**: TCN(Time Contrastive Networks)과 같이 상태 표현(state representation)을 학습하여 보상 함수를 설계하거나, GAIfO와 같이 GAN 구조를 이용하여 상태 전이 분포를 일치시키는 방식이 있다.

기존 IfO 방식들의 한계는 대부분 시각적 정보에만 의존하여 정책을 수립한다는 점이며, 특히 TCN과 같은 방식은 시간 정렬(time-aligned)된 데이터가 필요하거나 주기적인 작업(cyclical tasks)에서 성능이 떨어진다는 문제점이 있다. 제안된 방법은 이러한 model-free end-to-end 방식에 속하지만, Proprioception을 명시적으로 통합함으로써 학습 속도와 최종 성능을 개선했다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 전체 파이프라인
제안된 알고리즘은 Generator(이하 Policy)와 Discriminator로 구성된 적대적 학습 구조를 가진다. 

1. **Policy ($\pi_\theta$)**: MLP 구조로 설계되었으며, 에이전트의 현재 proprioceptive state $s$를 입력받아 행동 $a$의 분포를 출력한다.
2. **Discriminator ($D_\phi$)**: CNN 구조로 설계되었으며, 전문가의 비디오 $\tau_e$와 에이전트가 생성한 비디오 $\tau_i$를 입력받아 두 데이터의 출처를 판별한다.

### 상세 학습 절차 및 방정식

#### 1. Discriminator 학습
단일 프레임으로는 상태 관찰이 불충분하므로, $\{o_{t-2}, o_{t-1}, o_t, o_{t+1}\}$와 같이 4개의 연속된 프레임을 스택으로 묶어 입력으로 사용한다. 판별자는 전문가 데이터에 대해서는 0에 가까운 값을, 모방자 데이터에 대해서는 1에 가까운 값을 출력하도록 다음 목적 함수를 최대화하는 방향으로 학습된다.

$$\max_{\phi} \left( \mathbb{E}_{\tau_i}[\log(D_\phi(o_{t-2:t+1}))] + \mathbb{E}_{\tau_e}[\log(1 - D_\phi(o_{t-2:t+1}))] \right)$$

#### 2. Policy 학습 (Reward 설계)
모방자의 목표는 판별자를 속이는 것이므로, 판별자가 출력한 값의 음의 로그값을 보상으로 사용한다.

$$\text{Reward} = -\mathbb{E}_{\tau_i}[\log(D_\phi(o_{t-2:t+1}))]$$

이 보상 함수를 기반으로 PPO(Proximal Policy Optimization) 알고리즘을 사용하여 정책 $\pi_\theta$를 업데이트한다. 구체적인 그래디언트 업데이트 식은 다음과 같다.

$$\mathbb{E}_{\tau_i}[\nabla_\theta \log \pi_\theta(a|s)Q(s,a)] - \lambda \nabla_\theta H(\pi_\theta)$$

여기서 $Q(s,a)$는 상태-행동 가치 함수이며, 판별자로부터 얻은 보상의 기대값으로 정의된다.
$$Q(\hat{s}_t, \hat{a}_t) = -\mathbb{E}_{\tau_i}[\log(D_\phi(o_{t-2:t+1})) | s_0 = \hat{s}_t, a_0 = \hat{a}_t]$$

## 📊 Results

### 실험 설정
- **데이터셋**: OpenAI Gym 및 MuJoCo 시뮬레이터의 6개 도메인 (MountainCarContinuous, InvertedPendulum, InvertedDoublePendulum, Hopper, Walker2d, HalfCheetah).
- **전문가 데이터**: PPO를 통해 학습된 전문가 에이전트의 $64 \times 64$, 30-fps 비디오 데이터.
- **비교 대상**: TCN, BCO, GAIfO.
- **측정 지표**: 정규화된 작업 점수(Normalized Task Score), 학습 반복 횟수(Number of Iterations).

### 주요 결과
1. **최종 성능**: 거의 모든 도메인에서 제안된 방법이 다른 baseline들보다 월등히 높은 성능을 보였다. 특히 GAIfO보다 유의미하게 높은 점수를 기록하였으며, 전문가 비디오 궤적의 수가 증가할수록 성능이 향상되는 경향을 보였다.
2. **학습 속도**: InvertedPendulum 도메인에서 GAIfO와 비교했을 때, 제안된 방법이 전문가 수준의 성능에 도달하는 속도가 훨씬 빨랐다. 이는 Proprioception의 활용이 학습 효율성을 크게 높임을 시사한다.
3. **Baseline 분석**: BCO는 행동 복제 특유의 compounding error 문제로 인해 성능이 낮았고, TCN은 특정 데모에 과적합(overfitting)되거나 주기적 작업에 취약하여 낮은 성능을 보였다.

## 🧠 Insights & Discussion

### 강점 및 해석
본 연구는 시각적 정보만을 이용해 정책을 학습하는 것보다, 제어에 직접적으로 연관된 proprioceptive state를 정책의 입력으로 사용하는 것이 훨씬 효율적임을 입증하였다. 시각적 정보는 보상 함수를 정의하는 '기준'으로서 작동하고, 실제 제어는 '정밀한 내부 상태'를 통해 이루어지게 함으로써 IfO의 성능을 극대화하였다.

### 한계 및 분석
- **HalfCheetah 도메인의 낮은 성능**: 전문가의 움직임이 매우 빨라 프레임 간 차이가 크기 때문에, 판별자가 행동 패턴을 추출하기 어려웠던 것으로 분석된다. 이를 해결하기 위해 샘플링 프레임 레이트를 높이거나 더 많은 데모 데이터가 필요하다는 점을 언급한다.
- **Walker2d의 높은 분산**: 에이전트의 다리가 서로 가려지는 occlusion 현상이 발생하여, 시각적 정보만으로는 정확한 상태 파악이 어려웠기 때문으로 추측된다.
- **일반화 문제**: 본 논문은 학습자와 전문가의 신체 구조가 동일하고 시점(viewpoint)이 같다는 가정을 전제로 한다. 따라서 Embodiment mismatch(신체 구조 차이)나 Viewpoint mismatch(시점 차이)를 극복하는 것은 향후 해결해야 할 과제이다.

## 📌 TL;DR

본 논문은 비디오만으로 작업을 배우는 IfO(Imitation from Observation) 프레임워크에서, 에이전트의 **Proprioception(고유 수용 감각)** 정보를 정책 학습에 결합한 알고리즘을 제안한다. 시각적 데이터는 판별자를 통한 보상 생성에만 사용하고, 실제 정책은 proprioceptive state를 기반으로 MLP를 통해 학습함으로써 **학습 속도와 최종 작업 성능을 획기적으로 향상**시켰다. 이 연구는 향후 실제 로봇이 인간의 비디오를 보고 자신의 신체 상태를 활용해 정밀하게 행동을 모방하는 연구에 중요한 기초가 될 수 있다.