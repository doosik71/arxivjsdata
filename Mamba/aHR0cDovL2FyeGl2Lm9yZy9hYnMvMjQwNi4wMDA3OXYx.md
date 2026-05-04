# Decision Mamba: Reinforcement Learning via Hybrid Selective Sequence Modeling

Sili Huang, Jifeng Hu, Zhejian Yang, Liwei Yang, Tao Luo, Hechang Chen, Lichao Sun, Bo Yang (2024)

## 🧩 Problem to Solve

본 논문은 강화학습(Reinforcement Learning, RL)에서 최근 주목받고 있는 In-context RL의 계산 효율성 문제를 해결하고자 한다. In-context RL은 여러 개의 궤적(trajectories)과 같은 태스크 컨텍스트를 제공함으로써 모델이 온라인 환경에서 스스로 성능을 개선(self-improvement)하게 만드는 방식이다. 

기존의 In-context RL 방법론들은 주로 Transformer 기반의 아키텍처를 사용하는데, 이는 두 가지 주요한 문제점을 가진다. 첫째, Transformer의 Self-attention 메커니즘은 시퀀스 길이에 대해 이차 복잡도(quadratic complexity, $O(L^2)$)의 계산 비용이 발생한다. 둘째, self-improvement를 위해 사용하는 Across-episodic contexts(에피소드 간 컨텍스트)가 추가됨에 따라 처리해야 할 시퀀스의 길이가 급격히 증가하여 계산 비용이 기하급수적으로 상승한다. 결과적으로 태스크의 호라이즌(horizon)이 길어질수록 기존 Transformer 기반 에이전트는 막대한 계산 비용으로 인해 실시간 적용에 한계를 보인다.

따라서 본 연구의 목표는 Transformer의 높은 예측 정확도와 Mamba 모델의 효율적인 장기 의존성(long-term dependency) 처리 능력을 결합하여, 긴 호라이즌을 가진 태스크에서도 효율성과 효과성을 동시에 달성하는 In-context RL 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Mamba와 Transformer를 하이브리드 형태로 결합하여, **Mamba는 장기 기억을 통해 상위 수준의 '서브 골(sub-goal)'을 생성하고, Transformer는 이 서브 골과 단기 컨텍스트를 바탕으로 정밀한 '액션(action)'을 예측**하게 하는 것이다.

중심적인 설계 전략은 다음과 같다.
1. **하이브리드 구조(DM-H):** Mamba를 통해 장기 컨텍스트에서 서브 골 $z$를 추출하고, 이를 Transformer의 프롬프트로 입력하여 계산 복잡도를 낮추면서도 예측 품질을 유지한다.
2. **가치 있는 서브 골(Valuable Sub-goals) 도입:** Mamba가 단순히 임의의 벡터를 생성하는 것이 아니라, 실제 보상이 높은 상태(milestone)를 지향하도록 학습시키기 위해, 오프라인 데이터에서 가중 평균 누적 보상을 이용해 '가치 있는 상태'를 추출하여 Transformer와 Mamba의 연결 고리를 정렬(align)한다.

## 📎 Related Works

**1. Mamba 및 Long Sequence Modeling**
S4, S5, H3와 같은 상태 공간 모델(State Space Model, SSM)은 시퀀스 길이에 대해 선형 복잡도를 가지며 장기 의존성을 잘 포착한다. 특히 Mamba는 데이터 의존적 선택 메커니즘(data-dependent selection mechanism)을 도입하여 Transformer에 필적하는 성능과 선형적인 계산 효율성을 동시에 달성하였다.

**2. Decision-Making for Transformers**
Decision Transformer(DT)와 같은 연구들은 RL 문제를 시퀀스 예측 문제로 공식화하여 오프라인 RL에서 성공을 거두었다. 그러나 이러한 모델들은 주로 전문가 정책을 증류(distill)하는 데 집중하며, 온라인 환경에서의 self-improvement 능력은 부족한 경우가 많다.

**3. In-Context RL**
최근 Algorithm Distillation(AD)과 같은 연구들은 여러 과거 궤적을 컨텍스트로 제공하여 모델이 파라미터 업데이트 없이도 정책을 개선하는 In-context RL을 제안하였다. 하지만 앞서 언급했듯이, Across-episodic contexts로 인해 시퀀스가 길어질 때 발생하는 계산 비용 문제가 치명적인 한계로 지적된다.

## 🛠️ Methodology

### 전체 파이프라인 및 시스템 구조
Decision Mamba-Hybrid (DM-H)는 크게 두 개의 모듈로 구성된다.
1. **Mamba Module:** 장기 컨텍스트(Across-episodic contexts)를 입력받아 $c$ 스텝마다 하나의 서브 골 $z$를 생성한다.
2. **Transformer Module:** 생성된 서브 골 $z$와 현재의 짧은 단기 시퀀스를 입력받아 구체적인 액션 $a$를 예측한다.

### 주요 구성 요소 및 상세 설명

**1. Mamba의 서브 골 생성**
Mamba는 다음과 같은 선형 상미분 방정식(Linear ODE) 기반의 시스템을 이산화하여 처리한다.
$$h'_t = Ah_t + Bx_t$$
$$y_t = Ch_t$$
여기서 $h_t$는 숨겨진 상태(hidden state)이며, Mamba는 이를 통해 매우 긴 시퀀스를 효율적으로 요약한다. DM-H에서 Mamba는 전체 궤적 정보를 바탕으로 Transformer가 참조할 벡터 형태의 서브 골 $z$를 출력한다.

**2. 가치 있는 서브 골(Valuable Sub-goals)의 정의**
Mamba가 생성하는 서브 골이 실제로 유의미한 목표가 되도록 하기 위해, 오프라인 데이터에서 다음과 같은 기준으로 '가치 있는 상태' $s^g$를 선정한다.
특정 상태 $s_i$에서 미래 상태 $s_j$까지의 누적 보상을 두 상태 사이의 거리 $(j-i)$로 나눈 가중 평균값을 계산한다.
$$\text{Value}(s_j | s_i) = \frac{\sum_{t=i+1}^{j} r_k}{j-i}$$
이 값을 통해 단순히 보상의 총합이 높은 상태가 아니라, 현재 상태에서 효율적으로 도달 가능하면서도 가치가 높은 '이정표(milestone)' 상태를 서브 골로 지정한다.

**3. 학습 절차 및 손실 함수**
- **오프라인 학습:** 
    - Mamba는 장기 컨텍스트로부터 서브 골 $z$를 예측한다.
    - Transformer는 Mamba가 예측한 $z$ 혹은 위에서 정의한 '가치 있는 서브 골' $f(s^g)$를 입력받아 전문가의 액션 $a$를 예측하도록 학습된다.
    - 손실 함수는 액션의 특성에 따라 Cross-Entropy (이산 액션) 또는 Mean-Squared Error (연속 액션)를 사용하며, 전체 모듈을 end-to-end로 업데이트한다.
- **온라인 테스트:** 
    - 그라디언트 업데이트 없이, 과거 궤적과 현재 상태를 Mamba에 넣어 서브 골을 생성하고, 이를 통해 Transformer가 액션을 생성하는 trial-and-error 방식으로 동작한다.

## 📊 Results

### 실험 설정
- **데이터셋:** Grid World (Darkroom, Key-to-Door 등), Tmaze (장기 기억 회상 능력 측정), D4RL (연속 제어 태스크).
- **비교 대상:** AMAGO, DPT, AD (Transformer 및 Mamba 버전), TD3+BC, BC, DT.
- **평가 지표:** 에피소드당 평균 리턴(Return) 및 온라인 테스트 시간.

### 주요 결과
**1. Grid World 및 Tmaze**
- DM-H는 특히 보상이 희소(sparse)하고 호라이즌이 긴 태스크에서 기존 베이스라인들을 압도하였다. 
- Tmaze 실험에서 DM-H는 컨텍스트 길이 20k까지 처리 가능했으며, 이는 DT나 일반 DM(Decision Mamba)이 처리 가능한 10k보다 두 배 더 긴 기억력을 가짐을 의미한다.

**2. D4RL (연속 제어)**
- DM-H는 대부분의 태스크에서 AD 및 BC보다 우수한 성능을 보였으며, 강력한 오프라인 RL 알고리즘인 TD3+BC와 경쟁 가능한 수준의 성능을 기록하였다. 이는 suboptimal한 데이터에서도 self-improvement가 가능함을 시사한다.

**3. 효율성 (Efficiency)**
- 가장 주목할만한 결과는 계산 속도이다. D4RL과 같은 장기 태스크의 온라인 테스트에서 DM-H는 Transformer 기반 베이스라인보다 **최대 28배 빠른 속도**를 보였다. 이는 Mamba의 선형 복잡도 덕분에 시퀀스가 길어져도 계산 비용이 일정하게 유지되기 때문이다.

## 🧠 Insights & Discussion

**강점 및 해석**
본 논문은 Transformer의 '정밀한 국소적 예측 능력'과 Mamba의 '효율적인 전역적 기억 능력' 사이의 시너지를 성공적으로 입증하였다. 특히 서브 골(sub-goal)이라는 추상화 계층을 도입함으로써, RL 문제를 계층적 구조(Hierarchical structure)로 변환하여 해결한 점이 주효했다. Mamba는 고수준의 전략(어디로 가야 하는가)을 결정하고, Transformer는 저수준의 제어(어떻게 움직여야 하는가)를 담당하는 구조가 된 것이다.

**한계 및 향후 과제**
가장 큰 한계는 하이퍼파라미터 $c$(하나의 서브 골이 제어하는 스텝 수)에 대한 의존성이다. $c$가 너무 크면 효율성은 증가하지만 Transformer에 제공되는 정보가 너무 성겨져 예측 품질이 떨어질 수 있고, 너무 작으면 Mamba의 효율성 이점이 사라진다. 논문에서도 언급되었듯이, 태스크의 특성에 따라 $c$를 동적으로 적응시키는(adaptively) 메커니즘이 향후 연구의 핵심이 될 것으로 보인다.

## 📌 TL;DR

본 논문은 In-context RL의 치명적인 약점인 Transformer의 이차 계산 복잡도 문제를 해결하기 위해, Mamba와 Transformer를 결합한 **Decision Mamba-Hybrid (DM-H)**를 제안한다. Mamba가 장기 기억을 통해 서브 골을 생성하고 Transformer가 이를 바탕으로 액션을 예측하는 하이브리드 구조를 통해, **D4RL 기준 온라인 테스트 속도를 28배 향상시키면서도 성능은 유지하거나 오히려 개선**하였다. 이 연구는 매우 긴 호라이즌을 가진 RL 태스크에서도 실시간으로 작동 가능한 효율적인 In-context 에이전트 설계의 가능성을 제시하였다.