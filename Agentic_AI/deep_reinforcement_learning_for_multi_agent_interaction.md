# Deep Reinforcement Learning for Multi-Agent Interaction

Ibrahim H. Ahmed, Cillian Brewitt, Ignacio Carlucho, Filippos Christianos, Mhairi Dunion, Elliot Fosong, Samuel Garcin, Shangmin Guo, Balint Gyevnar, Trevor McInroe, Georgios Papoudakis, Arrasy Rahman, Lukas Schäfer, Massimiliano Tamborski, Giuseppe Vecchio, Cheng Wang and Stefano V. Albrecht (2022)

## 🧩 Problem to Solve

본 논문은 에든버러 대학교의 Autonomous Agents Research Group에서 수행한 연구 포트폴리오를 정리한 보고서 형식의 논문이다. 주된 해결 과제는 복잡한 환경에서 다른 에이전트와 상호작용하며 주어진 과업을 완수할 수 있는 자율 에이전트(Autonomous Agents)를 개발하는 것이다.

구체적으로는 다음과 같은 세부 문제들을 해결하고자 한다.

- **Multi-Agent Reinforcement Learning (MARL)의 확장성 문제**: 에이전트 수가 증가함에 따라 공동 결정 공간(joint decision space)이 기하급수적으로 커지는 문제를 해결하고, 효율적인 협력 정책 및 통신 방법을 학습시키는 것이 목표이다.
- **Ad Hoc Teamwork의 적응성**: 사전 협의나 공동 훈련 없이도 동적으로 구성된 팀에서 즉각적으로 협력해야 하는 상황에서, 제한된 관측 정보만으로 타 에이전트의 행동과 목표를 추론하고 적응하는 능력을 확보하고자 한다.
- **단일 에이전트 RL의 샘플 효율성**: 환경과의 상호작용을 최소화하면서도 최적의 정책을 학습할 수 있도록 intrinsic motivation, curriculum learning 등을 통한 효율적인 학습 방법을 연구한다.
- **자율 주행의 해석 가능성과 안전성**: 도시 환경에서 타 차량의 의도를 정확히 인식하고, 이를 바탕으로 안전하고 해석 가능한 주행 계획을 수립하는 문제를 다룬다.
- **양자 내성 인증 및 키 합의**: 기존 공개키 암호화 방식이 양자 컴퓨팅에 취약함을 인지하고, 멀티 에이전트 상호작용의 복잡성을 이용한 새로운 보안 프로토콜을 개발하고자 한다.

## ✨ Key Contributions

본 연구 그룹의 핵심 기여는 MARL, Ad Hoc Teamwork, 자율 주행, 보안이라는 네 가지 주요 축에서 다음과 같은 혁신적인 아이디어와 프레임워크를 제시한 점이다.

1. **MARL 벤치마킹 및 도구 제공**: 9가지 MARL 알고리즘을 5가지 환경에서 평가하여 표준화된 성능 지표를 제시하였으며, 확장된 PyMARL 라이브러리인 `EPyMARL`과 LBF, RWARE와 같은 새로운 환경을 오픈소스로 공개하였다.
2. **효율적인 MARL 학습 기법**: 중요도 가중치(importance weighting)를 이용한 off-policy correction 기반의 `SEAC`와, 에이전트 정체성을 임베딩하여 선택적으로 파라미터를 공유하는 `SePS`를 통해 학습 효율과 행동 다양성을 동시에 확보하였다.
3. **Open Ad Hoc Teamwork 해결책**: 그래프 신경망(GNN)을 활용하여 팀 구성의 변화에 강건한 `GPL`과, 국소적 궤적 정보만으로 타 에이전트를 모델링하는 `LIAM`을 제안하여 부분 관측 환경에서의 협력 가능성을 높였다.
4. **해석 가능한 자율 주행 시스템**: 합리적 역계획(rational inverse planning) 기반의 `IGP2`, 가려진 객체를 추론하는 `GOFI`, 그리고 결정 트리(decision tree)를 통해 검증 가능한 목표 인식을 수행하는 `GRIT`를 개발하였다.
5. **상호작용 기반 보안 프로토콜**: 정보 이론적 보안(information-theoretic security)에 기반하여 양자 공격에 안전한 `AMI` 프로토콜을 제안하였으며, PPO 알고리즘을 통해 인증 효율을 최적화하였다.

## 📎 Related Works

논문에서는 기존 MARL 연구들이 겪고 있는 재현성 부족과 표준 벤치마크 환경의 부재를 지적하며, 이를 해결하기 위해 직접 벤치마크를 수행하였다. Ad Hoc Teamwork 분야에서는 Stone et al.의 초기 정의를 바탕으로, 사전 조율 없이 협력해야 하는 실세계 로보틱스 응용 분야의 한계를 분석하였다.

특히, 기존의 목표 인식(goal recognition) 방식들이 딥러닝 기반으로 구축되어 정확도는 높으나 해석 가능성과 검증 가능성이 떨어진다는 점을 언급하며, 본 연구의 `GRIT` 방식이 이 차별점을 가지고 있음을 명시한다. 또한, 기존의 인증 프로토콜들이 수론적(number-theoretic) 난제에 의존하여 양자 컴퓨팅에 취약하다는 점을 들어 `AMI`의 필요성을 역설한다.

## 🛠️ Methodology

본 논문은 여러 연구를 포함하고 있으므로, 핵심 방법론들을 체계적으로 설명한다.

### 1. Multi-Agent Reinforcement Learning

- **SEAC (Shared Experience Actor-Critic)**: 서로 다른 에이전트가 생성한 경험을 결합하여 학습 그래디언트를 생성한다. 이때 off-policy correction을 위해 importance weighting을 사용한다.
- **SePS (Selective Parameter Sharing)**: 모든 에이전트가 파라미터를 공유하는 대신, encoder-decoder 구조로 에이전트 ID를 임베딩하고 비지도 클러스터링을 통해 공유 그룹을 자동으로 결정한다.
- **Emergent Language Expressivity**: 통신 채널의 표현력(expressivity)을 단순한 상호 정보량(mutual information)이 아닌, 과업 간 일반화 성능의 부분 순서(partial ordering)로 측정하는 방법을 제안한다.

### 2. Ad Hoc Teamwork

- **GPL (Graph-based Policy Learning)**: GNN을 사용하여 동적인 팀 구성에 대응한다. coordination graph를 기반으로 joint action value 모델을 학습하며, 팀원의 행동 예측 모듈을 추가하여 불확실성을 모델링한다.
- **LIAM (Local Information Agent Modelling)**: 에이전트의 국소적 궤적과 모델링 대상 에이전트의 궤적을 연결하는 표현력을 학습하는 encoder-decoder 모델이다. 훈련은 centralized 방식으로 수행하지만, 실행은 decentralized 방식으로 이루어진다.

### 3. Single-Agent RL

- **DeRL (Decoupled RL)**: 탐색(exploration)과 활용(exploitation)을 위한 정책을 분리하여 훈련한다.
  - 탐색 정책: $\text{intrinsic reward} + \text{extrinsic reward}$로 학습.
  - 활용 정책: $\text{extrinsic reward}$로만 학습하며, 탐색 정책이 수집한 데이터를 사용한다.

### 4. Autonomous Driving

- **IGP2**: 합리적 역계획을 통해 타 차량의 목표 확률을 추론하고, 이를 MCTS(Monte Carlo Tree Search) 기반 모션 플래너에 입력하여 최적의 경로를 생성한다.
- **GRIT**: 차량 궤적 데이터로부터 결정 트리를 구축하여 목표를 인식한다. 이는 명제 논리로 변환하여 SMT(satisfiability modulo theories) 솔버를 통해 정형 검증이 가능하다.

### 5. Secure Authentication (AMI)

- **AMI**: 마스터 키로 사용되는 private agent model을 기반으로 상호작용 전사(interaction transcript)를 생성한다.
- **Optimization**: PPO 알고리즘을 사용하여 서버의 행동 모델을 학습시킨다. 목적 함수는 통계적 가설 검정에서 공격자를 가장 빠르게 기각($p\text{-value}$를 낮춤)할 수 있는 쿼리를 생성하는 것이다.

## 📊 Results

본 논문은 개별 연구의 결과를 요약하여 제시한다.

- **MARL 벤치마크**: 9가지 알고리즘을 평가하여 성능 메트릭을 표준화하였으며, `EPyMARL`을 통해 다양한 환경에서의 공정한 비교 기반을 마련하였다.
- **SEAC & SePS**: SEAC는 샘플 효율성을 크게 향상시켰으며, SePS는 파라미터 공유의 효율성과 에이전트 행동의 다양성 사이의 균형을 맞추어 성능을 개선하였다.
- **Ad Hoc Teamwork**: `GPL`은 학습 시 보지 못한 팀 구성에 대해서도 기존 베이스라인 대비 뛰어난 일반화 성능을 보였으며, `LIAM`은 심한 부분 관측 상황에서도 강건한 에이전트 모델링 성능을 입증하였다.
- **자율 주행**: `GRIT`은 딥러닝 기반 방식과 유사한 정확도를 보이면서도, 인간이 해석 가능하고 정형적으로 검증 가능하다는 이점을 증명하였다. `GOFI`는 가려진 객체로 인한 충돌 횟수를 유의미하게 감소시켰다.
- **보안**: `AMI` 프로토콜은 랜덤 공격, 재전송 공격, 키 복구 공격에 대해 높은 탐지 정확도를 보였으며, PPO 최적화를 통해 공격자 기각에 필요한 샘플 수를 최대 $70\%$까지 줄였다.

## 🧠 Insights & Discussion

본 연구 그룹은 현재의 성과에 안주하지 않고 다음과 같은 비판적 해석과 향후 과제를 제시한다.

- **RL의 일반화 문제**: RL은 지도 학습에 비해 작은 환경 변화에도 성능이 급격히 떨어지는 일반화 문제가 심각하다. 단순한 랜덤 샘플링 방식의 훈련보다는 타겟 분포 내의 "취약 지점(weak point)"을 지능적으로 샘플링하는 방식이 필요하다.
- **인과적 RL (Causal RL)**: 모델 프리(model-free) 방식의 한계를 극복하기 위해 환경의 인과 모델(causal model)을 학습해야 한다. 특히 멀티 에이전트 시스템은 관측 데이터와 개입(intervention) 데이터를 모두 얻을 수 있어 인과 구조 학습에 유리한 환경이다.
- **Ad Hoc Teamwork의 현실적 제약**: 대부분의 연구가 완전 관측을 가정하지만, 실제 로봇 응용에서는 부분 관측과 통신 제한이 필수적으로 고려되어야 한다. 또한, 고정된 정책의 팀원이 아닌, 함께 학습하고 적응하는 팀원과의 상호작용을 다루는 'few-shot teamwork'가 중요한 연구 방향이 될 것이다.
- **자율 주행의 투명성**: 단순한 경로 생성을 넘어, 승객이 신뢰할 수 있도록 자연어 기반의 직관적인 설명을 자동으로 생성하는 시스템의 필요성을 강조한다.

## 📌 TL;DR

본 논문은 에든버러 대학교 자율 에이전트 연구 그룹의 광범위한 연구 성과를 정리한 보고서이다. 핵심은 **MARL의 확장성 및 효율성 개선, Ad Hoc Teamwork의 적응력 확보, 자율 주행의 해석 가능성 증대, 그리고 양자 내성 보안 프로토콜 개발**에 있다. 특히 `EPyMARL`과 같은 오픈소스 도구와 `GRIT`, `GPL`과 같은 혁신적 알고리즘을 통해 이론과 실무(자율 주행, 물류 로봇)의 간극을 좁히고자 했으며, 향후 연구 방향으로 RL의 일반화와 인과 추론, 그리고 현실적인 제약 조건 하의 협력을 제시하고 있다.
