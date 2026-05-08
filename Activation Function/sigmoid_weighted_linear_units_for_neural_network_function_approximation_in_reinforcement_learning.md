# Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning

Stefan Elfwing, Eiji Uchibe, Kenji Doya

## 🧩 Problem to Solve

최근 심층 강화 학습(Deep Reinforcement Learning, DRL)이 Atari 게임에서 인간 수준의 성능을 달성하는 등 크게 발전했지만, 신경망 함수 근사에 더 효과적인 활성화 함수가 필요합니다. 또한, 경험 재생(experience replay)이나 분리된 타겟 네트워크(target network)와 같은 복잡한 기법을 사용하는 DQN과 같은 최신 알고리즘에 비해, 적격성 추적(eligibility traces)을 사용하고 타겟 네트워크가 없는 온-폴리시(on-policy) 학습과 소프트맥스(softmax) 액션 선택 같은 전통적인 접근 방식이 경쟁력을 가질 수 있는지에 대한 의문이 제기됩니다.

## ✨ Key Contributions

- **새로운 활성화 함수 제안:** 신경망 함수 근사를 위한 시그모이드 가중 선형 유닛(Sigmoid-Weighted Linear Unit, SiLU) 및 그 미분 함수(dSiLU)를 제안했습니다.
  - SiLU: $a_k(s) = z_k \sigma(z_k)$
  - dSiLU: $a_k(s) = \sigma(z_k)(1 + z_k(1 - \sigma(z_k)))$
- **전통적인 RL 접근 방식의 경쟁력 입증:** 경험 재생 없이 적격성 추적과 소프트맥스 액션 선택을 사용하는 온-폴리시 학습(TD($\lambda$), Sarsa($\lambda$))이 DQN에 필적할 수 있음을 보여주었습니다.
- **최첨단(State-of-the-Art) 결과 달성:**
  - 확률적 SZ-Tetris와 10x10 Tetris에서 얕은(shallow) dSiLU 네트워크 에이전트가 새로운 최첨단 성능을 달성했습니다.
  - Atari 2600 도메인에서 심층(deep) SiLU-dSiLU Sarsa($\lambda$) 에이전트가 12개 게임에서 DQN 및 Double DQN을 능가하는 결과를 보였습니다.
- **가치 추정 분석:** TD($\lambda$) 및 Sarsa($\lambda$)는 Q-러닝 기반 알고리즘의 액션 가치 과대평가 문제를 겪지 않음을 분석적으로 보여주었습니다.
- **액션 선택의 중요성 강조:** 특정 환경(예: 테트리스, Asterix, Asteroids)에서 소프트맥스 액션 선택이 나쁜 결과로 이어질 수 있는 무작위 액션을 피하게 함으로써, $\epsilon$-그리디(epsilon-greedy) 방식보다 우수함을 입증했습니다.

## 📎 Related Works

- **DQN (Mnih et al., 2015):** 심층 신경망, 경험 재생, 분리된 타겟 네트워크를 결합하여 Atari 게임에서 인간 수준 성능을 달성한 DRL의 핵심 알고리즘.
- **DQN 개선 연구:**
  - Double DQN (van Hasselt et al., 2015): 액션 가치 과대평가 감소.
  - Prioritized Experience Replay (Schaul et al., 2016): 더 효율적인 경험 재생.
  - Dueling Network Architectures (Wang et al., 2016): 상태 가치와 각 액션의 이점을 분리하여 추정.
  - Asynchronous methods (A3C) (Mnih et al., 2016): 여러 에이전트의 병렬 비동기 학습.
- **TD-Gammon (Tesauro, 1994):** 20년 전 신경망과 TD($\lambda$) 학습을 사용하여 백개먼에서 마스터 수준 성능을 달성한 인상적인 RL 애플리케이션.
- **EE-RBM (Elfwing et al., 2015, 2016):** 저자들이 이전 연구에서 제안한 Expected Energy-based Restricted Boltzmann Machine으로, SiLU 활성화 함수의 영감이 되었습니다.
- **ReLU (Hahnloser et al., 2000):** SiLU와 비교되는 일반적인 활성화 함수.
- **SZ-Tetris (Szita and Szepesvári, 2010):** 테트리스의 핵심 과제를 보존하면서 빠른 평가를 가능하게 하는 벤치마크.
- **CBMPI (Gabillon et al., 2013; Scherrer et al., 2015):** 10x10 Tetris에서 높은 성능을 보인 알고리즘.
- **Arcade Learning Environment (Bellemare et al., 2013):** Atari 2600 도메인 평가 플랫폼.

## 🛠️ Methodology

- **강화 학습 알고리즘:**
  - **TD($\lambda$)**: 상태-가치 함수 $V_{\pi}$를 추정하는 데 사용됩니다.
  - **Sarsa($\lambda$)**: 액션-가치 함수 $Q_{\pi}$를 추정하는 데 사용됩니다.
  - 두 알고리즘 모두 파라미터 업데이트를 위해 다음 그래디언트-하강 학습 규칙을 따릅니다: $\theta_{t+1} = \theta_t + \alpha \delta_t e_t$.
    - 여기서 $\delta_t$는 TD-오차(TD-error), $e_t$는 적격성 추적 벡터(eligibility trace vector)입니다.
- **활성화 함수:**
  - **SiLU (Sigmoid-weighted Linear Unit):** 입력 $z_k$에 시그모이드 함수 $\sigma(z_k)$를 곱하여 활성화 $a_k(s) = z_k \sigma(z_k)$를 계산합니다. 이는 연속적이고 "부족한" 버전의 ReLU와 유사하며, 비단조적이고 자체 안정화(self-stabilizing) 특성을 가집니다.
  - **dSiLU (Derivative of SiLU):** SiLU의 미분으로 활성화 $a_k(s) = \sigma(z_k)(1 + z_k(1 - \sigma(z_k)))$를 계산합니다. 이는 더 가파르고 "초과하는" 버전의 시그모이드 함수와 유사합니다.
- **액션 선택:**
  - 모든 실험에서 볼츠만 분포를 가진 소프트맥스 액션 선택을 사용했습니다.
  - 탐색과 활용의 균형을 제어하는 온도 $\tau$는 에피소드가 진행됨에 따라 $\tau(i) = \frac{\tau_0}{1 + \tau_k i}$와 같이 쌍곡선으로 감소하는 어닐링(annealing) 기법을 적용했습니다.
- **실험 설정:**
  - **SZ-Tetris:**
    - **얕은 신경망:** 50개의 은닉 유닛을 가진 단일 은닉층 네트워크를 사용하여 SiLU, ReLU, dSiLU, 시그모이드 유닛의 성능을 비교했습니다. 손으로 코딩한 상태 특징(hand-coded state features)을 사용했습니다.
    - **심층 신경망:** 원시 보드 구성(raw board configurations)을 상태로 사용하고, 두 개의 합성곱(convolutional) 층과 하나의 완전 연결(fully-connected) 층을 가진 네트워크를 사용했습니다. SiLU-SiLU, ReLU-ReLU, SiLU-dSiLU 구성을 비교했습니다.
    - TD($\lambda$) 알고리즘을 사용했습니다.
  - **10x10 Tetris:**
    - 250개의 dSiLU 은닉 유닛을 가진 얕은 신경망을 사용하여 TD($\lambda$) 알고리즘으로 학습했습니다. SZ-Tetris와 유사한 손으로 코딩한 상태 특징을 사용했습니다.
  - **Atari 2600 게임 (12개 게임):**
    - 합성곱 층에 SiLU 유닛을, 완전 연결 은닉층에 dSiLU 유닛을 사용하는 심층 합성곱 신경망(SiLU-dSiLU)과 Sarsa($\lambda$) 알고리즘을 사용했습니다.
    - DQN과 유사하게 원시 Atari 프레임을 전처리했지만, 경험 재생이나 분리된 타겟 네트워크는 사용하지 않았습니다.

## 📊 Results

- **SZ-Tetris:**
  - **얕은 네트워크:** dSiLU 에이전트가 평균 263점(이전 SOTA 220점 대비 20% 향상)으로 새로운 최첨단 점수를 달성했습니다. 시그모이드 네트워크도 232점으로 좋은 성능을 보였습니다.
  - **심층 네트워크:** SiLU-dSiLU 네트워크가 평균 229점으로 이전 최첨단(220점)을 능가했습니다.
- **10x10 Tetris:**
  - dSiLU 네트워크 에이전트가 평균 4,900점(이전 SOTA 4,200점 대비 향상)으로 새로운 최첨단 점수를 달성했습니다. 특히, 이 결과는 Bertsekas 특징과 유사한 더 간단한 특징들을 사용하여 얻어졌기에 더욱 인상적입니다.
- **Atari 2600 게임:**
  - SiLU-dSiLU Sarsa($\lambda$) 에이전트는 12개 게임에서 DQN 및 Double DQN을 크게 능가했습니다.
  - 평균 DQN 정규화 최고 점수에서 Double DQN의 127% 대비 332%를 달성했습니다.
  - 중앙값 DQN 정규화 최고 점수에서 Double DQN의 105% 대비 125%를 달성했습니다.
  - 특히 Asterix와 Asteroids 게임에서는 두 번째로 좋은 에이전트 대비 각각 562%와 552%의 성능 향상을 보였습니다. 단, Breakout 게임에서는 학습이 제대로 진행되지 않아 저조한 성능을 보였습니다.

## 🧠 Insights & Discussion

- **SiLU 및 dSiLU의 효율성:** 제안된 SiLU 및 dSiLU 활성화 함수는 다양한 강화 학습 작업에서 효과적임을 입증했으며, 기존 ReLU 및 시그모이드 유닛보다 뛰어난 성능을 보였습니다. 특히 dSiLU는 뛰어난 성능을 보였습니다.
- **전통적인 RL의 경쟁력:** 이 연구는 적격성 추적과 소프트맥스 액션 선택을 사용하는 온-폴리시 학습이 경험 재생이나 타겟 네트워크 없이도 DQN과 같은 최첨단 심층 RL 방법과 충분히 경쟁할 수 있음을 보여주었습니다. 이는 최신 복잡한 아키텍처나 기법만이 유일한 해답이 아닐 수 있음을 시사합니다.
- **가치 추정의 정확성:** TD($\lambda$) 및 Sarsa($\lambda$)는 Q-러닝 기반 방법(최대 연산자 사용)이 겪는 액션 가치 과대평가 문제로부터 자유롭습니다. 이는 더욱 정확한 가치 추정으로 이어지며, 높은 성능의 핵심 요인 중 하나일 수 있습니다.
- **소프트맥스 액션 선택의 중요성:** 테트리스와 같이 나쁜 액션이 심각한 결과를 초래할 수 있는 환경(예: 불필요한 구멍 생성, 게임 종료)에서 소프트맥스 탐색은 매우 중요합니다. 이는 $\epsilon$-그리디 방식과 달리 잠재적으로 해로운 액션들을 균등하게 선택하는 것을 피하여, 훨씬 더 나은 학습 결과를 가져옵니다.
- **향후 연구 방향:** 현재의 전통적인 접근 방식에 타겟 네트워크, 듀얼링 아키텍처(dueling architecture), 비동기 학습(asynchronous learning) 등 DQN의 성공적인 구성 요소를 통합한다면 더욱 개선된 성능을 기대할 수 있다고 제안합니다.

## 📌 TL;DR

**문제:** RL 신경망을 위한 더 나은 활성화 함수를 개발하고, 기존 온-폴리시 학습(적격성 추적, 소프트맥스 선택)이 최신 딥 RL(DQN)과 경쟁할 수 있는지 탐구.

**방법:** 시그모이드 가중 선형 유닛(SiLU)과 그 미분(dSiLU)이라는 새로운 활성화 함수를 제안하고, 경험 재생이나 타겟 네트워크 없이 TD($\lambda$) 및 Sarsa($\lambda$) 기반의 온-폴리시 학습과 소프트맥스 액션 선택을 활용.

**발견:**

- SiLU와 dSiLU는 ReLU 및 시그모이드 활성화 함수보다 뛰어난 성능을 보였습니다.
- 특히 dSiLU는 SZ-Tetris와 10x10 Tetris에서 새로운 최첨단 성능을 달성했습니다.
- Atari 2600 게임에서 SiLU-dSiLU Sarsa($\lambda$) 에이전트가 DQN 및 Double DQN을 평균 DQN 정규화 점수에서 332%로 크게 능가했습니다.
- TD($\lambda$) 및 Sarsa($\lambda$)는 액션 가치 과대평가 문제를 겪지 않으며, 치명적인 액션이 있는 환경에서 소프트맥스 액션 선택이 $\epsilon$-그리디보다 훨씬 우수함을 입증했습니다.
