# Improved Reinforcement Learning with Curriculum

Joseph West, Frederic Maire, Cameron Browne, and Simon Denman (2019)

## 🧩 Problem to Solve

본 논문은 AlphaZero와 같은 강화학습 기반 게임 에이전트가 학습 초기 단계에서 겪는 학습 효율성 저하 문제를 해결하고자 한다. AlphaZero는 스스로 경기를 치르는 self-play를 통해 학습 데이터를 생성하지만, 학습 초기에는 다음과 같은 두 가지 문제가 동시에 발생한다.

첫째, 신경망이 충분히 학습되지 않아 네트워크의 예측값(prediction)이 부정확하다. 둘째, 게임의 초기 단계에서는 트리 탐색(tree search)이 단말 상태(terminal state, 즉 승패가 결정되는 상태)에 도달할 가능성이 낮다. 이 두 조건이 결합되면 트리 탐색이 네트워크의 잘못된 예측을 수정할 만큼의 실제 환경 보상(reward)을 찾지 못하게 되며, 결과적으로 '정보가 없는(uninformed)' 저품질의 학습 데이터가 생성된다.

따라서 본 연구의 목표는 학습 초기 단계에서 이러한 무분별한 데이터 수집을 방지하고, 학습 효율을 높이기 위한 구조화된 학습 커리큘럼을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **End-game-first curriculum**이다. 이는 인간이 복잡한 개념을 배울 때 구조화된 방식으로 배우는 것에서 착안한 것으로, 게임의 끝부분(단말 상태에 가까운 상태)부터 먼저 학습하고 점진적으로 게임의 앞부분으로 학습 범위를 확장하는 방식이다.

단말 상태 근처의 액션 결과부터 학습함으로써 에이전트는 환경 보상이 발생하는 특징과 액션을 먼저 인식하게 되며, 이후 점차적으로 단말 상태에서 멀리 떨어진 상태에서의 행동 방식을 학습하게 된다. 이를 통해 학습 초기 단계에서 발생하는 '정보 없는' 경험을 배제하여 전체적인 학습 속도를 향상시킨다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구들을 소개하며 본 제안 방식과의 차별점을 설명한다.

1. **AlphaZero의 진화**: AlphaGo $\rightarrow$ AlphaGo Zero $\rightarrow$ AlphaZero로 이어지는 흐름을 언급하며, MCTS(Monte Carlo Tree Search)와 신경망을 결합하여 가치 추정치를 얻는 구조를 설명한다.
2. **Curriculum Learning**: 기존의 커리큘럼 학습 방식인 'Reward Shaping'(보상 설계)과 'Problem Complexity Increase'(문제 복잡도 점진적 증가)를 설명한다. 하지만 이러한 방식들은 문제에 대한 사전 지식이나 인간의 수동 설정이 필요하다는 한계가 있다.
3. **Reverse Curriculum**: 에이전트를 목표 상태(goal-state)에 배치하고 점차 멀리 떨어진 상태를 탐색하게 하는 방식이다. 하지만 이는 목표 상태가 알려져 있어야 하며, 단말 상태에 도달했을 때 더 이상의 법적 액션이 불가능한 '게임' 환경에는 적용하기 어렵다는 한계가 있다.

본 논문의 **End-game-first curriculum**은 Reverse Curriculum과 유사하게 목표 근처부터 학습하지만, 특정 목표 상태를 미리 알 필요가 없으며, 탐색 과정에서 발견된 단말 상태와의 거리를 기준으로 학습 데이터를 선택한다는 점에서 차별화된다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

시스템은 게임 환경(Game Environment)과 플레이어(Player)의 두 모듈로 구성된다. 플레이어는 AlphaZero에서 영감을 받은 신경망-MCTS 결합 구조를 사용한다.

### 2. 주요 구성 요소 및 역할

- **Neural Network**: Deep Residual Convolutional Neural Network를 사용하며, 입력 상태 $s$에 대해 정책 벡터 $p_\theta(s)$(최적 액션의 확률 분포)와 상태 가치 추정치 $v_\theta(s)$(스칼라 값)를 출력한다.
- **MCTS (Monte Carlo Tree Search)**: 신경망의 예측을 가이드로 삼아 비대칭 트리를 구축한다. Selection, Expansion, Evaluation, Backpropagation의 4단계로 진행되며, 최종적으로 각 에지에 대한 방문 횟수 비율을 통해 정책 $\pi(s)$를 생성한다.
- **Training Pipeline**: self-play를 통해 생성된 경험 $\{s, \pi(s), z(s)\}$를 경험 버퍼(experience buffer)에 저장하고, 배치 경사 하강법(batched gradient descent)을 통해 신경망을 최적화한다. 여기서 $z(s)$는 게임의 최종 결과(승리 +1, 패배 -1, 무승부 -0.5)이다.

### 3. 손실 함수 (Loss Function)

신경망의 파라미터 $\theta$는 다음의 손실 함수 $l$을 최소화하도록 학습된다.

$$l = (z(s) - v_\theta(s))^2 - \pi(s) \cdot \log(p_\theta(s)) + c \cdot ||\theta||^2$$

- $(z(s) - v_\theta(s))^2$: 실제 결과와 가치 예측값 사이의 평균 제곱 오차(MSE)
- $-\pi(s) \cdot \log(p_\theta(s))$: MCTS 정책과 신경망 정책 사이의 교차 엔트로피(Cross-Entropy)
- $c \cdot ||\theta||^2$: L2 가중치 규제화(Weight Regularization)

### 4. End-game-first Curriculum 적용 방법

커리큘럼 함수 $\zeta(t)$를 도입하여, 학습 에포크 $t$에 따라 각 게임에서 유지할 경험의 비율을 결정한다. 게임의 초반부 $(1 - \zeta(t))\%$에 해당하는 경험은 버퍼에 저장하지 않고 제외한다.

- **효율적 구현**: 단순히 데이터를 버리는 것이 아니라, 제외될 구간에서는 MCTS를 수행하지 않고 **랜덤 액션**을 선택함으로써 계산 시간을 대폭 단축한다. 이후 $\zeta(t)\%$ 구간에 진입하면 다시 전체 MCTS를 사용하여 액션을 결정한다.
- **커리큘럼 설정**: Racing Kings와 Reversi 게임에 대해 서로 다른 $\zeta(t)$ 스케줄을 적용하였다. 예를 들어 Racing Kings의 경우 $t=0$일 때 $0.1(10\%)$에서 시작하여 $t \ge 1000$일 때 $1.0(100\%)$가 되도록 점진적으로 증가시킨다.

## 📊 Results

### 1. 실험 설정

- **대상 게임**: Modified Racing Kings(기물 이동성이 있는 게임), Reversi(보드를 채워가는 게임)
- **비교 대상**: Baseline player(커리큘럼 없음) vs Curriculum player(제안 방식)
- **기준 상대(Reference Opponent)**: Racing Kings는 Stockfish level 2, Reversi는 200 simulations를 사용하는 MCTS 플레이어와 대결
- **측정 지표**: 시간(Time), 학습 단계(Steps), 에포크(Epochs)에 따른 승률(Win ratio)의 이동 평균

### 2. 주요 결과

- **시간 대비 성능**: 모든 케이스에서 커리큘럼 플레이어가 학습 초기 단계의 승률이 Baseline보다 월등히 높았다. 이는 랜덤 무브를 통한 시간 단축 효과와 데이터 품질 향상이 동시에 작용한 결과이다.
- **학습 단계(Steps) 대비 성능**: 시간 외에도 학습 단계(batch 업데이트 횟수) 기준으로 측정했을 때 커리큘럼 플레이어의 성능이 더 좋았다. 이는 단순히 시간이 절약된 것이 아니라, **학습 데이터의 순수 품질(net quality)이 향상**되었음을 시사한다.
- **에포크 대비 성능**: 에포크 기준으로는 두 플레이어의 성능이 비슷하거나 커리큘럼 플레이어가 약간 앞섰다. 이는 경험 데이터를 일부 제외했음에도 불구하고, 제외된 데이터가 유용한 정보를 포함하고 있지 않았음을 의미한다.

## 🧠 Insights & Discussion

### 1. 강점 및 해석

본 연구는 무분별한 self-play 데이터 수집보다, '가치 있는' 데이터를 선별적으로 학습하는 것이 훨씬 효율적임을 입증하였다. 특히 학습 초기에 네트워크가 아무런 정보가 없는 상태에서 생성하는 'uninformed experiences'를 배제함으로써, 신경망이 단말 상태의 보상 신호를 더 빠르게 학습할 수 있도록 돕는다.

### 2. 한계 및 비판적 해석

- **커리큘럼의 임의성**: 논문에서 사용된 $\zeta(t)$ 함수는 반-임의적(semi-arbitrary)으로 설정되었다. 저자들 또한 이 커리큘럼이 최적이 아니라고 명시하며, 커리큘럼이 너무 빠르게 변하면 정보 없는 데이터가 섞이고, 너무 느리면 특정 구간에 과적합(overfitting)될 위험이 있음을 지적한다.
- **성능 수렴**: 결과 그래프를 보면 두 플레이어의 승률이 결국 수렴하는 경향을 보이는데, 이는 커리큘럼이 학습 초기 속도는 높여주지만 최종 도달 성능(asymptotic performance) 자체를 획기적으로 높이는지에 대해서는 추가 분석이 필요함을 시사한다.

### 3. 향후 과제

저자들은 고정된 에포크 기반의 커리큘럼이 아닌, 트리 탐색의 **가시성 지평(visibility horizon)**에 기반하여 커리큘럼을 자동으로 조절하는 방식의 연구 가능성을 제시하였다.

## 📌 TL;DR

본 논문은 AlphaZero 스타일의 강화학습 에이전트가 학습 초기에 생성하는 저품질(uninformed) 데이터를 배제하기 위해, 게임의 끝부분부터 점진적으로 학습 범위를 넓히는 **End-game-first curriculum**을 제안한다. 실험 결과, 제안 방식은 학습 초기 단계에서 시간 및 학습 단계 대비 승률을 유의미하게 향상시켰으며, 이는 데이터의 양보다 질이 학습 속도에 더 중요한 영향을 미친다는 것을 보여준다. 향후 이 커리큘럼을 자동화하는 연구가 진행된다면 범용 게임 AI 학습 효율을 크게 높일 수 있을 것으로 기대된다.
