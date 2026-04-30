# Correlation Filter Selection for Visual Tracking Using Reinforcement Learning

Yanchun Xie, Jimin Xiao, Kaizhu Huang, Jeyarajan Thiyagalingam, Yao Zhao (2018)

## 🧩 Problem to Solve

본 논문은 시각적 객체 추적(Visual Object Tracking)에서 널리 사용되는 Correlation Filter (CF) 기반 추적기들이 겪는 **모델 드리프트(Model Drift)** 문제를 해결하고자 한다. CF 기반 모델들은 추적 정확도와 속도 사이의 균형이 뛰어나지만, 부분 가려짐(Partial Occlusion), 배경 클러터(Background Clutter), 낮은 해상도와 같은 도전적인 환경에서 부정확한 추적 결과로 인해 모델이 잘못 업데이트되는 경향이 있다.

대부분의 기존 CF 추적기들은 매 프레임마다 단순 이동 평균(Moving Average) 방식을 통해 모델을 업데이트한다. 그러나 타겟이 가려지거나 사라진 상태에서 모델 업데이트가 강행될 경우, 오류가 누적되어 결국 타겟을 완전히 놓치게 되는 irrecoverable model drift가 발생한다. 따라서 본 연구의 목표는 업데이트의 신뢰성을 판단하고, 최적의 CF 모델을 선택적으로 사용할 수 있는 메커니즘을 도입하여 추적의 강건성을 높이는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **여러 개의 CF 모델을 병렬로 유지하고, 딥 강화학습(Deep Reinforcement Learning)을 통해 현재 상황에 가장 적합한 모델을 동적으로 선택**하는 것이다.

주요 기여 사항은 다음과 같다.
1. 단일 모델 사용 시 발생하는 드리프트 문제를 해결하기 위해, 병렬로 업데이트 및 유지되는 여러 CF 모델 중 최적의 모델을 선택하는 새로운 접근 방식을 제안하였다.
2. 모델 선택 과정을 마르코프 결정 과정(Markov Decision Process)으로 정의하고, 이를 최적화하기 위해 Proximal Policy Optimization (PPO) 알고리즘을 도입하였다.
3. 실시간 적용이 가능하도록 경량화된 특징 추출기(Feature Extractor)와 작은 규모의 Decision Network를 설계하였다.
4. OTB100 및 OTB2013 벤치마크를 통해 기존 CF 기반 추적기들보다 우수한 성능(평균 성공률 62.3%, 정밀도 81.2%)을 달성함을 입증하였다.

## 📎 Related Works

### Correlation Filter Based Tracker
CF 기반 추적기들은 푸리에 도메인(Fourier Frequency Domain)에서 릿지 회귀(Ridge Regression) 문제를 효율적으로 해결함으로써 빠른 속도와 높은 정확도를 달성해 왔다. KCF, DSST, SRDCF 등이 대표적이며, 최근에는 VGG와 같은 사전 학습된 심층 신경망의 Convolutional Feature를 결합하여 성능을 높이는 추세이다. 하지만 이러한 모델들도 여전히 온라인 업데이트 과정에서 발생하는 노이즈와 드리프트 문제에서 자유롭지 못하다.

### Deep Reinforcement Learning
강화학습은 최근 시각적 추적 분야에서 특징 선택(Feature Selection)이나 하이퍼파라미터 최적화 등에 적용되기 시작했다. 특히 TRPO나 PPO와 같은 정책 경사(Policy Gradient) 방법론은 고차원 입력 공간에서도 안정적인 학습 성능을 보여준다. 본 논문은 강화학습을 '특징 선택'이 아닌 '모델 업데이트 및 선택'이라는 관점에서 적용했다는 점에서 기존 연구들과 차별화된다.

## 🛠️ Methodology

### 1. Correlation Filter 모델링
표준 CF는 다음과 같은 목적 함수를 통해 학습된다.
$$\arg \min_{f} ||\psi(x) * f - g||^2 + \lambda ||f||^2$$
여기서 $f$는 CF, $*$는 순환 상관(Circular Correlation) 연산, $\psi$는 특징 추출기, $x$는 타겟 중심의 이미지 패치, $g$는 가우시안 형태의 응답 맵(Response Map) 레이블이다.

푸리에 도메인에서 $f$의 표현은 다음과 같이 계산된다.
$$F = \frac{\bar{G} \cdot \bar{X}}{\bar{X} \cdot X + \lambda}$$
여기서 $\bar{G}$와 $\bar{X}$는 각각 $g$와 $x$의 푸리에 변환 및 켤레 복소수(Complex Conjugation)를 의미하며, $\cdot$은 요소별 곱(Element-wise Product)이다. 새로운 프레임 $z$가 들어오면 응답 맵 $P$를 다음과 같이 얻는다.
$$P = F \cdot \bar{Z}$$
응답 맵 $P$에서 최대값을 가진 좌표가 타겟의 위치가 된다.

### 2. 강화학습 기반 모델 선택
본 연구는 추적 과정을 이산 제어 문제로 정의하고, PPO 알고리즘을 사용하여 최적의 CF 모델을 선택한다.

- **상태(State, $s_t$):** 현재 유지되고 있는 모든 CF 모델들에 의해 생성된 응답 맵(Response Maps)의 집합이다.
- **행동(Action, $a_t$):** 후보 CF 모델 풀(Pool) 중에서 하나의 모델을 선택하는 것이다.
- **보상(Reward, $r_t$):** 선택한 모델을 통해 추적한 결과와 Ground-truth 사이의 IOU(Intersection over Union)에 따라 다음과 같이 정의된다.
$$g(s_t, a_t) = \begin{cases} \text{IOU} + 1 & \text{if IOU} > 0.7 \\ -1 & \text{if IOU} < 0.2 \\ -0.1 & \text{otherwise} \end{cases}$$

### 3. PPO 기반 학습 및 Decision Network
학습은 Actor-Critic 구조의 Decision Network를 통해 이루어진다.
- **구조:** Policy Net과 Value Net 두 개의 브랜치로 구성되며, 각각 3개의 Convolutional Layer와 Fully Connected Layer를 가진다. 응답 맵은 $64 \times 64 \times 3$ 크기로 리사이징되어 입력된다.
- **손실 함수:** PPO의 Clipped Surrogate Objective를 사용하여 정책의 급격한 변화를 방지하고 학습 안정성을 높인다.
$$L_t(\theta) = \min(\text{Ratio} * A_t, \text{clip}(\text{Ratio}, 1-\epsilon, 1+\epsilon) A_t)$$
여기서 $\text{Ratio} = \frac{\pi(a_t|s_t;\theta)}{\pi(a_t|s_t;\theta_{old})}$이며, $A_t$는 어드밴티지 추정치(Advantage Estimation)이다.

### 4. CF 모델 풀 및 업데이트 전략
총 $k$개의 CF 모델을 유지한다.
- **Initial Model:** 첫 프레임의 타겟 특징으로 초기화되며, 업데이트 없이 유지된다. (원래 타겟의 기억 유지)
- **Accumulated Model:** 매 프레임마다 항상 업데이트된다. (외형 변화 및 변형에 적응)
- **Dynamic Model ($k-2$개):** Decision Net에 의해 선택되었을 때만 업데이트된다.

## 📊 Results

### 실험 설정
- **데이터셋:** OTB100, OTB2013 벤치마크 사용.
- **비교 대상:** KCF, DSST, SRDCF, DCFnet, HP 등 기존 CF 기반 추적기.
- **평가 지표:** 거리 정밀도(Distance Precision, DP)와 겹침 성공률(Overlap Success rate, OS).

### 정량적 결과
OTB100 데이터셋 기준, 제안 방법은 **OS 68.89%, DP 81.19%**를 기록하여 비교 대상 중 가장 높은 성능을 보였다. 특히 모델 선택 과정이 없는 `dcfnetpy`보다 OS 기준 1.82%p, DP 기준 1.06%p 높은 수치를 기록하였다.

### 정성적 및 속성별 분석
- **가려짐(Occlusion):** 다중 모델 선택 전략 덕분에 가려짐이 발생하는 시퀀스에서 성공률이 3.7%p, 정밀도가 3.1%p 향상되었다.
- **복구 능력:** 타겟을 한 번 놓치더라도 Initial Model이나 이전의 Dynamic Model을 선택함으로써 다시 타겟을 찾을 수 있는 복구 능력이 뛰어남을 확인하였다.

### Ablation Study
- **학습 횟수:** 학습 반복 횟수가 증가함에 따라 정밀도와 성공률이 점진적으로 상승하여 RL 학습이 유효함을 보였다.
- **업데이트 전략:** '항상 업데이트' 또는 '랜덤 업데이트' 전략보다 'Decision Net 기반 선택 업데이트' 전략이 월등히 높은 성능을 보였다.
- **RL 알고리즘 비교:** PPO가 A2C보다 더 높은 보상을 얻고 추적 성능이 좋았으며, DQN은 본 과제에서 제대로 작동하지 않았다.

## 🧠 Insights & Discussion

본 논문은 CF 기반 추적기의 고질적인 문제인 모델 드리프트를 해결하기 위해 '모델의 다변화'와 '지능적 선택'이라는 전략을 성공적으로 결합하였다. 특히 모든 모델을 무작정 업데이트하는 것이 아니라, 강화학습을 통해 현재 응답 맵의 상태를 보고 어떤 모델이 가장 신뢰할 만한지를 판단하게 함으로써 가려짐과 같은 극한 상황에서도 추적을 유지할 수 있게 하였다.

또한, PPO 알고리즘의 도입은 정책 업데이트의 변동성을 제한하여 학습의 안정성을 확보하는 데 기여하였다. 다만, 본 논문에서는 $k=3$일 때의 성능이 $k=4$일 때와 유사하다고 언급하며 복잡도를 낮추기 위해 $k=3$을 사용했는데, 이는 모델의 개수가 늘어난다고 해서 반드시 성능이 선형적으로 증가하지는 않음을 시사한다.

한계점으로는 RL 학습을 위해 대규모 비디오 데이터셋(VID)을 사용하여 오프라인 학습을 진행해야 한다는 점이 있다. 실제 환경에서 도메인이 크게 바뀔 경우, 사전 학습된 Decision Net이 최적의 모델을 선택하지 못할 가능성이 존재한다.

## 📌 TL;DR

본 논문은 CF 기반 시각적 추적기에서 발생하는 모델 드리프트 문제를 해결하기 위해, **여러 개의 CF 모델을 병렬로 유지하고 PPO 강화학습을 통해 최적의 모델을 선택하는 프레임워크**를 제안하였다. 제안된 방법은 특히 가려짐(Occlusion)과 타겟 소실 상황에서 강건한 복구 능력을 보이며, OTB 벤치마크에서 기존 CF 추적기들을 상회하는 성능을 달성하였다. 이 연구는 추적기의 업데이트 전략을 단순한 규칙이 아닌 학습 가능한 정책으로 전환함으로써, 실시간 추적 시스템의 안정성을 크게 향상시킬 가능성을 제시하였다.