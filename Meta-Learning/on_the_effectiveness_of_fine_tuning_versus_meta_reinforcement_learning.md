# On the Effectiveness of Fine-tuning Versus Meta-reinforcement Learning

Zhao Mandi, Pieter Abbeel, Stephen James (2022)

## 🧩 Problem to Solve

본 논문은 강화학습(Reinforcement Learning, RL) 분야에서 Meta-learning 접근 방식이 실제로 단순한 Multi-task pretraining 후의 Fine-tuning보다 우월한지에 대한 근본적인 의문을 제기한다.

기존의 Meta-RL 알고리즘들은 주로 좁은 범위의 태스크 분포(narrow task distributions)를 가진 단순한 환경에서 검증되었다. 저자들은 기존 연구들이 다루는 '태스크'의 개념이 실제로는 동일한 태스크의 단순한 변형(variation), 예를 들어 마찰 계수의 변화나 보상 함수의 미세한 변경 등에 국한되어 있다는 점을 지적한다. 이러한 'Variation adaptation'은 완전히 새로운 태스크에 적응하는 'Task adaptation'보다 본질적으로 훨씬 쉬운 문제이다.

따라서 본 연구의 목표는 시각 기반의 다양하고 도전적인 벤치마크(Procgen, RLBench, Atari)를 통해, 광범위한 태스크 분포 환경에서 Meta-RL 알고리즘들이 단순한 Multi-task pretraining 및 Fine-tuning 베이스라인과 비교하여 실제로 성능 이점이 있는지 분석하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 Meta-RL의 효용성을 검증하기 위해 "Task adaptation"이라는 보다 엄격한 기준을 제시하고, 이를 대규모 실험으로 증명한 것에 있다.

- **단순 베이스라인의 강력함 증명**: 시각 기반의 다양한 환경에서 Multi-task pretraining 후 Fine-tuning을 수행하는 것이 복잡한 Meta-RL 알고리즘들과 대등하거나 오히려 더 나은 성능을 보임을 입증하였다.
- **대규모 벤치마크 분석**: Procgen(레벨 간 적응), RLBench(태스크 간 적응), Atari(게임 간 적응)라는 세 가지 서로 다른 난이도의 벤치마크를 통해 Meta-RL의 일반화 능력을 체계적으로 평가하였다.
- **평가 패러다임의 전환 제안**: 향후 Meta-RL 연구들이 단순한 변형 적응을 넘어 더 도전적인 태스크 분포에서 평가되어야 하며, Multi-task pretraining + Fine-tuning을 강력한 기본 베이스라인으로 포함시킬 것을 권고한다.

## 📎 Related Works

### 기존 Meta-RL 연구 및 한계

Meta-RL은 새로운 태스크에 빠르게 적응하기 위한 최적의 학습 전략을 찾는 것을 목표로 하며, 크게 Context-based 방법(LSTM이나 별도의 인코더를 통해 문맥을 파악)과 Gradient-based 방법(하이퍼파라미터나 네트워크 파라미터를 최적화)으로 나뉜다. 그러나 기존 연구들은 대부분 상태 정보가 완전히 관찰 가능한(fully observable) 설정이나 정교하게 설계된 보상(shaped rewards) 환경에서 수행되었으며, 실제 세계와 유사한 고차원 시각 입력 및 희소 보상(sparse rewards) 환경에 대한 고려가 부족했다.

### 기존 접근 방식과의 차별점

본 논문은 기존 연구들이 '태스크'라고 정의했던 범위가 실제로는 '변형(variation)'에 불과했다는 점을 명확히 구분한다. 저자들은 훈련 세트와 테스트 세트의 태스크가 시각적으로나 목표 면에서 완전히 분리된(disjoint) 설정을 사용하여, 단순한 스킬의 검색(retrieval)이 아닌 진정한 의미의 적응(adaptation) 능력을 측정하였다.

## 🛠️ Methodology

### 비교 대상 알고리즘

본 연구에서는 세 가지 서로 다른 패러다임의 Meta-RL 알고리즘과 하나의 단순 베이스라인을 비교한다.

1. **Reptile (Gradient-based)**: 일차 미분 기반의 알고리즘으로, 여러 태스크에서 그래디언트 업데이트를 수행한 후, 초기 파라미터와 업데이트된 파라미터 사이의 가중 평균을 구함으로써 빠르게 적응 가능한 초기 지점을 찾는다.
2. **PEARL (Context-based)**: 에이전트의 경험으로부터 문맥 변수(context variables)를 인코딩하여 정책에 조건화한다. 테스트 시에는 새로운 경험을 수집하여 문맥을 업데이트함으로써 적응한다.
3. **$\text{RL}^2$ (Recurrence-based)**: RNN(LSTM)을 사용하여 여러 번의 시도(trials)를 통해 얻은 경험을 은닉 상태(hidden state)에 저장함으로써 RL 업데이트 규칙 자체를 네트워크 가중치에 내재화한다.
4. **Multi-task Pretraining (Baseline)**: 단일 정책으로 모든 훈련 태스크를 동시에 학습하며, 테스트 시에는 대상 태스크에 대해 단순한 Fine-tuning을 수행한다.

### 벤치마크별 구현 상세

각 벤치마크의 특성에 따라 서로 다른 Base RL 알고리즘을 적용하였다.

- **Procgen**: PPO를 기반으로 하며, Coinrun 게임을 사용한다. $\text{RL}^2$의 경우 LSTM 레이어를 추가하여 구현하였다.
- **RLBench**: 희소 보상 문제를 해결하기 위해 전문가 시연(demonstrations)을 사용하는 C2F-ARM을 기반으로 한다.
- **Atari**: RainbowDQN을 기반으로 하며, 서로 다른 액션 공간을 처리하기 위해 최대 크기로 패딩(padding) 후 미사용 차원을 'No-op'으로 처리하였다.

### 주요 학습 절차 (Reptile 예시)

Reptile의 학습은 내부 루프(inner loop)와 외부 루프(outer loop)로 구성된다.

- **내부 루프**: 단일 태스크 $T$에 대해 $k$번의 그래디언트 업데이트를 수행하여 파라미터 $\theta_k$를 얻는다.
- **외부 루프**: 초기 파라미터 $\theta_{old}$와 $\theta_k$ 사이의 소프트 업데이트를 수행한다.
$$\theta \leftarrow \epsilon(\theta_{old} - \theta_k)$$

## 📊 Results

### 1. Procgen (Easy)

- **결과**: Fine-tuning이 샘플 효율성과 최종 성능 모든 면에서 가장 우수하였다.
- **특이사항**: Reptile-PPO는 성능 향상을 보였으나 MT-PPO보다 낮았으며, $\text{RL}^2$는 새로운 레벨에 전혀 적응하지 못했다. 훈련 레벨의 수가 증가할수록 MT-PPO의 Zero-shot 성능과 Fine-tuning 성능이 모두 향상되었다.

### 2. RLBench (Medium)

- **결과**: Multi-task Fine-tuning과 Reptile이 대등하게 우수한 성능을 보였으며, PEARL과 Scratch(처음부터 학습)보다 훨씬 뛰어났다.
- **희소 보상 영향**: 테스트 시 시연 데이터(demo)를 0개에서 2개로 늘렸을 때, Scratch 학습은 성능이 급격히 향상되었으나 MT 및 Reptile은 이미 높은 성능을 유지하여 시연 데이터에 대한 의존도가 훨씬 낮음을 보였다.
- **PEARL의 실패**: 태스크들이 시각적으로 너무 분리되어 있어, Context Encoder가 유의미한 문맥을 생성하지 못하고 단순한 Pretraining 효과에만 의존하는 경향을 보였다.

### 3. Atari (Hard)

- **결과**: 모든 방식에서 전이 학습(transfer)이 매우 어려웠으며, Scratch 학습과 비교했을 때 성능 차이가 크지 않았다.
- **분석**: Atari 게임들은 시각적 요소와 제어 전략의 차이가 너무 커서 공유할 수 있는 지식이 매우 적다는 가설을 뒷받침한다. 다만, Reptile-Rainbow는 일부 게임(Assault 등)에서 Scratch보다 우위를 점했다.

## 🧠 Insights & Discussion

### Meta-RL의 환상과 실제

본 연구는 Meta-RL이 보여준 기존의 성공이 사실은 '태스크'가 아닌 '변형'에 적응한 결과였음을 시사한다. 완전히 새로운 태스크에 적응해야 하는 상황에서는 복잡한 Meta-learning 구조가 단순한 Multi-task pretraining보다 뚜렷한 이점을 제공하지 않는다.

### Context-based 방법의 한계

PEARL과 같은 Context-based 방법은 훈련 태스크 간의 시각적 유사성이 높을 때는 효과적이지만, 태스크들이 서로 완전히 다른 도메인에 있을 때는 문맥을 추론하는 인코더가 제대로 작동하지 않는다. 이는 Meta-RL이 "어떤 태스크인지 식별"하는 능력에 의존하는데, 식별 가능한 공통 구조가 없으면 무용지물이 되기 때문이다.

### Fine-tuning의 강점

Fine-tuning은 사전 학습된 가중치를 기반으로 새로운 태스크에 특화된 파라미터를 찾는다. 특히 RLBench 실험에서 Fine-tuning을 개별 태스크에 독립적으로 수행하는 것이 Multi-task 설정보다 유리했는데, 이는 에이전트가 새로운 태스크에 불필요한 기존 스킬을 '망각'하고 필요한 부분에 집중할 수 있기 때문으로 해석된다.

## 📌 TL;DR

본 논문은 Meta-RL이 단순한 Multi-task pretraining + Fine-tuning보다 실제로 더 나은지를 검증하기 위해 세 가지 시각 기반 벤치마크(Procgen, RLBench, Atari)에서 대규모 실험을 수행하였다. 실험 결과, **단순한 Multi-task pretraining 후 Fine-tuning을 수행하는 것이 Meta-RL 알고리즘들과 대등하거나 더 우수한 성능을 보였으며, 훨씬 계산 비용이 저렴하고 구현이 간단함**을 확인하였다. 이는 기존 Meta-RL의 성과가 좁은 태스크 분포(변형 적응)에 기인했음을 시사하며, 향후 연구에서 더 광범위하고 도전적인 태스크 분포를 사용하고 단순 Fine-tuning을 강력한 베이스라인으로 설정할 것을 제안한다.
