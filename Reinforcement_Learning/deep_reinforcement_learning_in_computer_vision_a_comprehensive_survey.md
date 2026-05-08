# Deep Reinforcement Learning in Computer Vision: A Comprehensive Survey

Ngan Le, Vidhiwar Singh Rathour, Kashu Yamazaki, Khoa Luu, Marios Savvides (2021)

## 🧩 Problem to Solve

본 논문은 컴퓨터 비전(Computer Vision, CV) 분야에서 딥 강화학습(Deep Reinforcement Learning, DRL)이 어떻게 적용되고 있는지를 체계적으로 분석하고 정리하는 것을 목표로 한다.

전통적인 지도 학습(Supervised Learning)은 대규모의 레이블링된 데이터셋을 필요로 하며, 이는 의료 영상이나 특수 환경의 데이터 확보가 어려운 경우 큰 제약이 된다. 또한, 객체 검출(Object Detection)이나 랜드마크 로컬라이제이션(Landmark Localization)과 같은 작업은 단순히 한 번의 예측으로 끝나는 것이 아니라, 최적의 위치를 찾아가는 순차적 의사결정(Sequential Decision Making) 과정으로 해석될 수 있다. 따라서 본 연구는 DRL이 이러한 순차적 최적화 문제를 해결하는 방식과 CV의 다양한 하위 작업들에 적용된 최신 연구 동향을 종합적으로 리뷰하여 연구자들에게 가이드라인을 제공하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 컴퓨터 비전 분야의 DRL 적용 사례를 체계적인 분류 체계(Taxonomy)로 정리하고, 이론적 배경부터 실제 응용까지의 전체 파이프라인을 분석한 점에 있다. 주요 기여 사항은 다음과 같다.

1. **이론적 기반 정립**: Deep Learning, Reinforcement Learning, 그리고 이 둘을 결합한 DRL의 핵심 이론(MDP, Bellman Equation, Value/Policy Gradient 등)을 상세히 설명하여 독자가 기초 지식을 갖추도록 한다.
2. **CV 응용 분야의 체계적 분류**: DRL의 적용 분야를 (i) 랜드마크 로컬라이제이션, (ii) 객체 검출, (iii) 객체 추적, (iv) 이미지 등록(Registration), (v) 이미지 분할(Segmentation), (vi) 비디오 분석, (vii) 기타 응용 분야의 7가지 카테고리로 구분하여 분석한다.
3. **방법론적 분석**: 각 응용 분야에서 사용된 RL 기법, 네트워크 설계, 성능 지표 및 데이터셋을 분석하고 비교 테이블을 통해 SOTA(State-of-the-art) 접근 방식을 제시한다.
4. **한계 및 미래 방향 제시**: 보상 함수 설계의 어려움, 연속적 상태/행동 공간 처리 문제 등 DRL을 실제 CV 시스템에 적용할 때 발생하는 주요 챌린지들을 논의하고, Inverse RL, Meta-RL 등의 최신 발전 방향을 제안한다.

## 📎 Related Works

논문은 기존의 DRL 서베이 논문들이 주로 일반적인 알고리즘이나 헬스케어와 같은 특정 도메인에 집중되어 있다는 점을 지적하며, 본 논문과의 차별점을 명시한다.

- **기존 연구의 한계**: 이전의 서베이들은 DRL의 중앙 알고리즘 자체를 다루거나, 의료 분야의 일반적인 적용 사례에 치중되어 있어, 컴퓨터 비전의 구체적인 작업(Task)별 구현 방식과 아키텍처에 대한 세부 분석이 부족했다.
- **본 논문의 차별점**: 단순한 알고리즘 소개를 넘어, CV의 핵심 작업들(Landmark, Detection, Tracking 등)에서 DRL이 구체적으로 어떻게 MDP(Markov Decision Process)로 정식화되는지, 어떤 보상 함수가 사용되는지, 그리고 어떤 신경망 구조가 결합되는지를 상세히 다룬다.

## 🛠️ Methodology

본 논문은 서베이 논문이므로 새로운 알고리즘을 제안하기보다, CV에서 DRL이 작동하는 일반적인 메커니즘과 핵심 알고리즘을 설명하는 데 집중한다.

### 1. 이론적 배경 및 DRL 프레임워크

DRL의 핵심은 에이전트(Agent)가 환경(Environment)과 상호작용하며 누적 보상(Cumulative Reward)을 최대화하는 최적 정책 $\pi^*$를 찾는 것이다. 이는 일반적으로 마르코프 결정 과정(Markov Decision Process, MDP)으로 정의된다.

- **MDP 구성 요소**: 상태 집합 $S$, 행동 집합 $A$, 전이 확률 $T(s_{t+1}|s_t, a_t)$, 보상 함수 $R$, 그리고 할인 계수(Discount Factor) $\gamma$.
- **목표**: 다음과 같은 기대 할인 보상을 최대화하는 정책 $\pi$를 찾는 것이다.
$$G(\pi) = E_{T^\pi} \left[ \sum_{t=0}^{\tau-1} \gamma^t r_{t+1} \right]$$

### 2. 주요 DRL 알고리즘 설명

논문은 CV 작업에서 주로 사용되는 세 가지 핵심 알고리즘 체계를 설명한다.

- **Value-based Methods (예: DQN)**: 상태-행동 가치 함수인 $Q$-함수를 학습한다.
  - **DQN**: CNN을 사용하여 고차원 픽셀 입력으로부터 $Q$값을 예측하며, 손실 함수는 다음과 같다.
  $$L_{DQN} = ||y(s_t, a_t) - Q^*(s_t, a_t, \theta_t)||^2$$
  - **Double DQN**: $Q$값의 과대평가(Overestimation) 문제를 해결하기 위해 행동 선택과 가치 평가 네트워크를 분리한다.
  - **Dueling DQN**: 상태 가치 $V(s)$와 이득 함수(Advantage function) $A(s, a)$를 분리하여 학습한다.

- **Policy Gradient Methods**: 정책 $\pi_\theta(a|s)$를 직접 최적화한다.
  - **REINFORCE**: 몬테카를로 추정을 통해 정책 그래디언트를 계산하여 업데이트한다.

- **Actor-Critic Methods (예: A2C, A3C)**: 정책을 결정하는 Actor와 가치를 평가하는 Critic을 동시에 학습시킨다.
  - **Advantage Actor-Critic (A2C)**: 이득 함수 $A^\pi(s_t, a_t) = Q^\pi(s_t, a_t) - V^\pi(s_t)$를 사용하여 분산을 줄이고 학습 효율을 높인다.

### 3. CV 작업으로의 변환 (MDP 정식화)

논문은 CV의 정적 이미지 처리 작업을 어떻게 순차적 의사결정 문제로 바꾸는지 설명한다.

- **State ($s$)**: 현재 랜드마크의 위치, 객체의 바운딩 박스(Bounding Box) ROI, 또는 이전 프레임의 특징 맵.
- **Action ($a$)**: 바운딩 박스의 이동(상, 하, 좌, 우), 크기 조절(확대, 축소), 또는 정지(Stop).
- **Reward ($r$)**: 정답(Ground Truth)과의 거리 감소량, IoU(Intersection over Union)의 증가분, 또는 거리 기반의 이진 보상.

## 📊 Results

본 논문은 개별 연구들의 결과를 취합하여 요약하고 있다. 각 분야별 주요 정량적/정성적 결과는 다음과 같다.

- **Landmark Detection**: DRL 기반의 에이전트가 다중 스케일(Multi-scale) 전략을 사용하여 3D CT 스캔에서 기존 SADNN이나 3D-DL 대비 정확도를 20-30% 향상시켰음을 보고한다.
- **Object Detection**: Tree-RL과 같은 구조적 RL 에이전트가 Faster R-CNN과 비교하여 AP(Average Precision) 면에서 경쟁력 있는 성능을 보이면서도 추론 속도를 개선했음을 보여준다.
- **Object Tracking**: DRL 기반 추적기들이 특히 변형(Deformation)이 심한 객체나 가림(Occlusion) 현상이 발생하는 환경에서 기존의 KCF나 Mean-shift 방식보다 높은 성공률과 정밀도를 기록했음을 분석한다.
- **Image Registration**: DRL을 이용한 정합 방식이 기존의 ITK나 Semantic Registration보다 성공률이 높으며, 특히 Euclidean 거리 오차를 유의미하게 낮추었음을 명시한다.
- **Image Segmentation**: SeedNet과 같은 DRL 기반 인터랙티브 세그멘테이션이 FCN 등 지도 학습 기반 모델보다 더 높은 IoU를 달성했음을 보고한다.

## 🧠 Insights & Discussion

### 강점 및 유용성

DRL은 CV에서 **'반복적 정교화(Iterative Refinement)'** 작업에 매우 강력하다. 한 번에 정확한 위치를 예측하는 대신, 에이전트가 보상을 따라 조금씩 위치를 수정해나가는 방식은 특히 정밀한 로컬라이제이션이 필요한 의료 영상 분석에서 큰 강점을 가진다. 또한, 레이블 데이터가 부족한 상황에서 환경과의 상호작용을 통해 학습할 수 있다는 점이 지도 학습의 한계를 극복하게 한다.

### 한계 및 챌린지

1. **보상 함수 설계의 난해함**: 현실 세계의 복잡한 문제를 스칼라 값인 보상으로 정의하는 것은 매우 어렵다. 너무 단순하면 최적해를 찾지 못하고, 너무 복잡하면 학습이 수렴하지 않는 문제가 발생한다.
2. **차원의 저주**: 고차원 이미지 데이터를 직접 상태 공간으로 사용할 경우 학습 시간이 기하급수적으로 증가한다. 이를 해결하기 위해 Autoencoder 등을 통한 차원 축소가 필수적이다.
3. **연속적 공간 처리**: 대부분의 RL 알고리즘은 이산적(Discrete) 행동 공간을 가정하지만, CV의 좌표나 크기 조절은 연속적(Continuous)이다. 이를 위해 강제로 이산화(Discretization)하는 과정에서 정보 손실이 발생한다.
4. **환경의 비정상성(Non-stationarity)**: 실제 환경은 시뮬레이션과 달리 노이즈가 많고 동적으로 변하므로, 학습된 정책이 실제 환경에서 작동하지 않는 Sim-to-Real 간극 문제가 존재한다.

### 비판적 해석

본 논문은 매우 광범위한 문헌을 조사하였으나, 각 응용 분야에서 DRL이 **'반드시'** 필요한지에 대한 분석은 다소 부족하다. 많은 사례가 단순히 기존의 회귀(Regression) 문제를 RL 형태로 바꾸어 푼 것에 불과하며, RL만이 가질 수 있는 '탐색(Exploration)'의 이점이 어떻게 구체적으로 성능 향상으로 이어졌는지에 대한 심층적인 논의가 더 필요해 보인다.

## 📌 TL;DR

본 논문은 컴퓨터 비전의 다양한 작업(랜드마크 검출, 객체 검출/추적, 이미지 정합/분할, 비디오 분석 등)을 순차적 의사결정 문제로 재정의하고, 이를 해결하기 위한 딥 강화학습(DRL)의 이론과 적용 사례를 총망라한 종합 서베이 보고서이다. DRL은 특히 반복적인 위치 수정이나 정교화 작업에서 지도 학습의 한계를 극복하는 대안으로 제시되며, 향후 Inverse RL이나 Meta-RL을 통한 데이터 효율성 및 보상 설계 문제 해결이 핵심 연구 방향이 될 것임을 시사한다.
