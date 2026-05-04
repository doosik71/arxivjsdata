# Deep Reinforcement Learning for Multi-Agent Interaction

Ibrahim H. Ahmed, Cillian Brewitt, Ignacio Carlucho, Filippos Christianos, Mhairi Dunion, Elliot Fosong, Samuel Garcin, Shangmin Guo, Balint Gyevnar, Trevor McInroe, Georgios Papoudakis, Arrasy Rahman, Lukas Schäfer, Massimiliano Tamborski, Giuseppe Vecchio, Cheng Wang and Stefano V. Albrecht (2022)

## 🧩 Problem to Solve

본 논문은 에든버러 대학교의 Autonomous Agents Research Group에서 수행한 연구 포트폴리오를 소개하는 개요 문서이다. 이들이 해결하고자 하는 핵심 문제는 복잡한 환경에서 다른 에이전트와 상호작용하며 주어진 과업을 수행할 수 있는 자율 에이전트(autonomous agents)를 개발하는 것이다.

구체적으로는 다음과 같은 세부 문제들에 집중한다.

- **Multi-Agent Reinforcement Learning (MARL)의 확장성**: 에이전트 수가 증가함에 따라 공동 결정 공간(joint decision space)이 기하급수적으로 커지는 문제를 해결하고, 효율적인 협력 정책과 통신 방법을 학습시키는 것이다.
- **Ad Hoc Teamwork**: 사전 협의나 공동 학습 기회 없이, 처음 만나는 다양한 파트너와 즉각적으로 협력해야 하는 상황에서 어떻게 빠르게 적응하고 행동할 것인가의 문제이다.
- **RL의 샘플 효율성 및 안정성**: 단일 에이전트 RL에서 환경과의 상호작용을 최소화하면서도 최적의 정책을 학습시키기 위한 샘플 효율성(sample-efficiency)과 학습 안정성 확보 문제이다.
- **자율 주행의 안전성 및 해석 가능성**: 복잡한 도심 환경에서 타 차량의 의도를 정확히 인식하고, 안전하며 인간이 이해할 수 있는(interpretable) 주행 계획을 수립하는 것이다.
- **양자 내성 인증**: 양자 컴퓨팅의 발전으로 인해 기존 공개키 암호화 기반의 인증 프로토콜이 취약해짐에 따라, 이를 대체할 양자 내성(quantum-resistant) 인증 체계를 구축하는 것이다.

## ✨ Key Contributions

본 연구 그룹의 핵심 기여는 딥러닝과 강화학습을 결합하여 다중 에이전트 시스템의 협력, 적응, 그리고 보안을 위한 다양한 알고리즘과 프레임워크를 제안한 것에 있다.

주요 설계 아이디어는 다음과 같다.

- **MARL의 효율화**: 중요도 가중치(importance weighting)를 이용한 오프-폴리시 보정(off-policy correction)을 통해 탐색 효율을 높이고, 에이전트 간의 파라미터 공유를 자동화하여 학습 효율과 다양성을 동시에 확보한다.
- **유연한 팀워크 구현**: 그래프 신경망(GNN)을 사용하여 팀 구성원의 수나 종류가 변하더라도 강건하게 작동하는 정책을 학습시키고, 부분 관측 환경에서도 타 에이전트의 궤적을 통해 상태를 추론하는 모델을 구축한다.
- **학습 구조의 분리**: 탐색(exploration)을 위한 정책과 활용(exploitation)을 위한 정책을 분리하여 내재적 보상(intrinsic reward)으로 인한 학습 불안정성을 제거한다.
- **해석 가능한 모델링**: 자율 주행에서 결정 트리(Decision Tree)나 합리적 역계획(rational inverse planning)을 도입하여 시스템의 판단 근거를 명확히 하고 검증 가능하게 만든다.
- **상호작용 기반 보안**: 수치적 계산 복잡도가 아닌, 다중 에이전트 간의 상호작용 패턴을 마스터 키로 사용하여 정보 이론적 보안(information-theoretic security)을 달성한다.

## 📎 Related Works

본 논문은 다음과 같은 기존 연구의 한계와 차별점을 언급한다.

- **MARL 벤치마크의 부재**: 기존 MARL 연구들은 표준화된 벤치마크 환경이 부족하고 재현성이 낮다는 문제가 있다. 이를 해결하기 위해 저자들은 5개의 환경과 25개의 협력 과업을 정의하고 알고리즘들을 벤치마킹하였다.
- **Ad Hoc Teamwork의 가정**: 기존의 많은 Ad Hoc teamwork 연구들이 환경에 대한 완전 관측(full observability)을 가정하지만, 이는 실제 로보틱스 환경에서는 비현실적이다. 본 연구는 부분 관측 환경에서도 작동하는 Agent Modelling에 집중한다.
- **RL의 일반화(Generalisation)**: 지도 학습과 달리 RL에서의 일반화는 정밀한 정의가 부족하며, 특히 다중 에이전트 환경에서는 에이전트 수의 변화와 파트너의 다양성으로 인해 일반화 문제가 훨씬 더 복잡함을 지적한다.
- **암호학적 취약성**: 기존의 공개키 기반 인증 프로토콜은 양자 컴퓨팅 공격에 취약하므로, 계산 복잡성이 아닌 상호작용의 복잡성에 기반한 새로운 접근 방식이 필요함을 제시한다.

## 🛠️ Methodology

본 보고서에서는 논문에서 소개된 주요 방법론들을 영역별로 설명한다.

### 1. Multi-Agent Reinforcement Learning

- **Shared Experience Actor-Critic (SEAC)**: 서로 다른 에이전트들이 생성한 경험을 통합하여 학습 그래디언트를 생성한다. 이때 Importance Weighting을 사용하여 오프-폴리시 보정을 수행함으로써 샘플 효율성을 높인다.
- **Selective Parameter Sharing (SePS)**: 모든 에이전트가 파라미터를 공유하면 행동의 다양성이 사라지는 문제를 해결하기 위해, Encoder-Decoder 구조를 사용하여 에이전트의 정체성을 임베딩 공간으로 투영하고, 비지도 클러스터링을 통해 파라미터를 공유할 에이전트 그룹을 자동으로 결정한다.

### 2. Ad Hoc Teamwork

- **Graph-based Policy Learning (GPL)**: GNN을 활용하여 동적인 팀 구성과 규모에 강건한 정책을 학습한다. Coordination Graph를 기반으로 공동 행동 가치 모델(joint action value model)을 학습하며, 팀원의 행동을 예측하는 모듈을 추가하여 불확실성을 모델링한다.
- **Local Information Agent Modelling (LIAM)**: 제어 에이전트의 궤적과 모델링 대상 에이전트의 궤적을 연결하는 표현(representation)을 학습하는 Encoder-Decoder 모델이다. 학습은 중앙 집중식으로 수행하지만, 실행 시에는 로컬 궤적만을 이용해 분산 방식으로 작동한다.

### 3. Single-Agent Reinforcement Learning

- **Decoupled RL (DeRL)**: 내재적 보상을 이용한 탐색 정책과 외재적 보상만을 이용한 활용 정책을 분리하여 학습시킨다. 탐색 정책이 수집한 데이터를 활용 정책이 학습에 사용함으로써, 내재적 보상의 변동성으로 인한 학습 불안정성을 제거한다.

### 4. Autonomous Driving

- **IGP2 & GOFI**: Rational Inverse Planning을 통해 타 차량의 목표와 궤적의 사후 확률을 추론하고, 이를 MCTS(Monte Carlo Tree Search) 기반 플래너에 입력하여 최적의 경로를 생성한다. GOFI는 여기서 더 나아가 가려진 물체(occluded objects)의 존재 확률을 함께 모델링한다.
- **GRIT**: 차량 궤적 데이터로부터 결정 트리를 구축하여 목표를 인식한다. 이는 딥러닝 모델보다 해석 가능성이 높으며, 명제 논리로 매핑하여 SMT(Satisfiability Modulo Theories) 솔버를 통해 공식적으로 검증할 수 있다.

### 5. Secure Authentication (AMI)

- **Authentication via Multi-Agent Interaction (AMI)**: 통신 당사자들을 자율 에이전트로 취급하고, 이들의 행동을 규정하는 개인적 에이전트 모델을 마스터 키로 사용한다.
- **PPO 최적화**: 서버의 행동 모델을 PPO 알고리즘으로 학습시켜, 최소한의 상호작용 횟수로 적대적 클라이언트를 식별할 수 있도록 쿼리를 최적화한다.

## 📊 Results

본 논문은 여러 개별 연구의 결과를 요약하여 제시하고 있다.

- **MARL**: SEAC는 기존 방법론 대비 샘플 효율성을 크게 향상시키고 수렴 시 더 높은 리턴을 기록하였다. SePS는 자동화된 파라미터 공유를 통해 기존 MARL 알고리즘의 성능을 유의미하게 개선하였다.
- **Ad Hoc Teamwork**: GPL은 이전에 본 적 없는 팀 구성이나 개방형 팀 설정(open team settings)에서 베이스라인 대비 우수한 일반화 능력을 보였다. LIAM은 심한 부분 관측 환경에서도 강건한 에이전트 모델링 성능을 입증하였다.
- **Single-Agent RL**: DeRL은 내재적 보상의 스케일이나 감쇠율 변화에 대해 훨씬 강건하며, 동일한 성능에 도달하는 데 필요한 상호작용 횟수를 줄였다.
- **Autonomous Driving**: GRIT은 딥러닝 기반 방법론과 유사한 정확도를 달성하면서도, 인간이 해석 가능하고 공식적 검증이 가능하다는 이점을 보였다.
- **Security**: AMI 프로토콜에서 PPO를 통해 서버 모델을 최적화한 결과, 적대자를 거부하는 데 필요한 샘플 수를 최대 $70\%$까지 줄일 수 있었다.

## 🧠 Insights & Discussion

본 연구 그룹은 현재의 성과를 넘어 다음과 같은 비판적 논의와 향후 방향을 제시한다.

- **RL의 일반화 한계**: RL은 훈련 조건의 작은 변화에도 성능이 급격히 떨어지는 문제가 있다. 이를 해결하기 위해 단순히 무작위 샘플링을 하는 것이 아니라, 에이전트의 약점 영역(weak point regions)을 지능적으로 샘플링하는 방법이나, 기대 리턴의 최대화가 아닌 성능 분포의 최적화(Pareto optimality 등)를 목표 함수로 설정하는 방향을 제안한다.
- **인과 강화학습 (Causal RL)**: 모델-프리(model-free) 방식의 한계를 극복하기 위해 환경의 인과 모델을 학습할 필요가 있다. 특히 다중 에이전트 시스템은 타 에이전트의 관측 데이터와 자신의 개입(intervention) 데이터를 모두 얻을 수 있어 인과 구조 학습에 유리한 환경이다.
- **Ad Hoc Teamwork의 현실적 제약**: 실제 환경에서는 통신 채널의 제한이 크므로, 현실적인 제약 조건 하에서의 통신 능력을 갖춘 에이전트 개발이 필수적이다. 또한, 팀원 자체가 학습하거나 변화하는 경우(non-stationary teammates)에 어떻게 대응할 것인가가 중요한 과제이다.
- **자율 주행의 투명성**: 단순한 성능 향상을 넘어, 탑승자가 시스템을 신뢰할 수 있도록 자연어 기반의 직관적인 설명(explanation)을 자동으로 생성하는 시스템이 필요하다.

## 📌 TL;DR

본 논문은 에든버러 대학교 Autonomous Agents Research Group의 광범위한 연구 성과를 정리한 보고서이다. 핵심 기여는 **확장 가능한 MARL 알고리즘(SEAC, SePS)**, **사전 협의 없는 팀워크를 위한 GPL 및 LIAM**, **학습 안정성을 높인 DeRL**, **해석 가능한 자율 주행 시스템(GRIT, IGP2)**, 그리고 **양자 내성 인증 프로토콜(AMI)**의 개발이다. 이 연구들은 자율 에이전트가 실제 복잡한 환경에서 안전하고 효율적으로 상호작용하며 임무를 수행하게 하는 데 핵심적인 역할을 할 것으로 기대된다.
