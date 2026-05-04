# Trajectory-based Learning for Ball-in-Maze Games

Sujoy Paul, Jeroen van Baar (2018)

## 🧩 Problem to Solve

본 논문은 로봇을 이용하여 Ball-in-Maze Game(이하 BiMGame)을 해결하는 문제를 다룬다. BiMGame은 정적 마찰(static friction), 기하학적 구조와의 충돌, 그리고 장기적인 계획(long-horizon planning)이 필요하다는 점에서 매우 복잡한 동역학적 특성을 가지고 있다.

기존의 Model-free Deep Reinforcement Learning(RL) 방식은 구조화되지 않은 탐색(unstructured exploration)을 수행하기 때문에 학습에 필요한 샘플 수가 지나치게 많다는 치명적인 단점이 있다. 반면 Model-based RL은 샘플 효율성이 높지만, BiMGame과 같이 고차원 상태 공간과 복잡한 동역학을 가진 환경에서는 정확한 모델을 구축하는 것이 매우 어렵다. 따라서 본 연구의 목표는 시뮬레이터에서 생성한 궤적(trajectory)을 활용하여 학습의 가이드를 제공함으로써, Deep RL의 샘플 효율성(sample-efficiency)을 높이는 방법을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 인간 전문가의 궤적 대신, 시뮬레이터의 물리 엔진을 일종의 모델로 활용하여 생성한 궤적으로 정책(policy)을 사전 학습(pre-training)시키고, 이후 RL을 통해 미세 조정(fine-tuning)하는 것이다.

특히, 사람이 직접 데이터를 생성해야 하는 부담을 없애고 시뮬레이터를 통해 제한된 수의 궤적만으로도 학습 속도를 2~3배가량 향상시킬 수 있음을 입증하였다. 또한, 학습된 가치 함수(value function)를 이용하여 보상을 재설계하는 Reward Shaping 기법을 적용하여 샘플 효율성을 더욱 높이는 방안을 탐구하였다.

## 📎 Related Works

논문에서는 기존의 모방 학습(Imitation Learning, IL) 및 RL 접근 방식들을 언급한다.

1. **A3C (Asynchronous Advantage Actor-Critic):** Sim2Real 전이 학습에 사용되었으나, 시행착오(trial-and-error) 기반 학습으로 인해 샘플 효율성이 낮다.
2. **AggreVateD:** 전문가 정책의 Advantage 추정치를 사용하지만, 모든 방문 상태에서 가치 함수가 매우 정확하게 추정되어야 한다는 전제가 있어 BiMGame에서는 일관된 성능을 보이지 못했다.
3. **DAgger (Dataset Aggregation):** Model-based RL에서 생성된 궤적으로 Model-free RL을 미세 조정하는 방식이다. 그러나 BiMGame과 같이 복잡한 동역학에서는 Model-based 방식의 성능 자체가 낮아 한계가 있다.
4. **Supervised Training from Experts:** 단순한 조작 작업에서는 효과적이지만, BiMGame처럼 복잡한 동역학을 가진 작업에 적용하기에는 어려움이 있다.

본 논문은 이러한 한계를 극복하기 위해 시뮬레이터 기반의 궤적 생성과 Supervised Pre-training, 그리고 RL Fine-tuning을 결합한 파이프라인을 제안하며 기존 방식과 차별화를 둔다.

## 🛠️ Methodology

### 1. 궤적 생성 (Trajectory Generation)
시뮬레이터의 물리 엔진을 활용하여 누적 보상을 최대화하는 궤적을 생성한다. 시간 단계 $t$에서 다음과 같은 최적화 문제를 해결한다.

$$ a^*_{t:t+H} = \arg \max_{a_{t:t+H}} \sum_{t'=t}^{t+H-1} r(s_{t'}, a_{t'}, s_{t'+1}), \quad \text{s.t., } s_{t'+1} = S(s'_t, a'_t) $$

여기서 $S$는 시뮬레이터, $r$은 보상 함수, $H$는 최적화 호라이즌(horizon)이다. 시뮬레이터가 미분 불가능하므로, $K=10$개의 무작위 행동 집합을 샘플링하여 최적의 행동을 선택하는 Random Shooting 전략을 사용한다. 이때 사용된 보상 함수는 공의 중심으로부터의 거리 기반 보상이다.

$$ r(s_t, a_t, s_{t+1}) = d(s_{t+1}) - d(s_t) $$

이를 통해 상태-행동-보상 트리플렛의 집합인 데이터셋 $\mathcal{D} = \{ (s_{ti}, a^*_{ti}, r_{ti}) \}$를 구축한다.

### 2. 지도 사전 학습 (Supervised Pre-training)
데이터셋 $\mathcal{D}$를 사용하여 DNN 정책을 사전 학습시킨다. 네트워크 구조는 A3C를 따르며, 정책 $\pi_\theta(a_t|s_t)$와 가치 추정 $\text{V}_\phi(s_t)$라는 두 개의 헤드를 가진다. 손실 함수 $\mathcal{L}$은 다음과 같이 정의된다.

$$ \mathcal{L} = \sum_{i=1}^n \sum_{t=1}^{n_i} \left[ \sum_{c=1}^C a^*_{tc} \log \pi_\theta(a_{tc} | s_{ti}) + \frac{1}{2} \left( \text{V}_\phi(s_{ti}) - \sum_{t'=t}^{n_i} \gamma r_{ti} \right)^2 \right] + \|\theta \cup \phi\|_2^2 $$

여기서 첫 번째 항은 정책 네트워크를 위한 Cross-entropy loss이며, 두 번째 항은 가치 함수 추정치를 위한 Mean Squared Error(MSE) loss이다. 마지막 항은 $L2$ 규제항이다. 이렇게 학습된 정책을 $\pi_s$라고 한다.

### 3. RL 미세 조정 (RL Fine-tuning)
사전 학습된 $\pi_s$를 초기값으로 하여 A3C 프레임워크에서 Policy Gradient 기반의 RL을 수행한다. 이는 무작위 초기화 대신 $\pi_s$에서 시작함으로써 탐색 공간을 좁히고 수렴 속도를 높이는 효과를 준다.

### 4. 가치 함수 기반 보상 설계 (Value Function as a Reward)
데이터셋 $\mathcal{D}$를 통해 가치 함수 $\hat{\text{V}}_\phi(s_t)$만을 별도로 학습시킨 후, 이를 이용해 보상을 변형(Reward Shaping)한다.

$$ \bar{r}(s_t, a_t, s_{t+1}) = r(s_t, a_t, s_{t+1}) + \gamma \hat{\text{V}}_\phi(s_{t+1}) - \hat{\text{V}}_\phi(s_t) $$

이 방식은 RL 에이전트가 더 풍부한 보상 신호를 받을 수 있게 하여 학습 효율을 높이려 시도한 것이다.

## 📊 Results

### 실험 설정
- **작업(Tasks):**
    - **FULL:** 게이트를 통과할 때마다 $\pm 1$의 보상을 받는 밀집 보상(dense reward) 환경.
    - **STG1 & STG2 (Steps-to-Go):** 최종 목표 지점에 도달했을 때만 $+1$ 보상을 받는 희소 보상(sparse reward) 환경.
- **네트워크 구조:** $\text{Conv-Conv-FC-LSTM}$ 구조를 사용하며, LSTM의 입력으로 이전 층의 특징, 이전 단계의 행동 및 보상을 입력한다.
- **비교 알고리즘:** A3C, Supervised Pre-training + A3C, Value-based Reward + A3C, Supervised + Value-based Reward + A3C, DAgger.

### 주요 결과
- **학습 속도 향상:** 사전 학습을 적용했을 때, 무작위 초기화 기반의 A3C보다 학습 속도가 약 2~3배 빨라지는 결과가 나타났다.
- **DAgger의 한계:** DAgger는 STG1과 같은 쉬운 작업에서는 작동하지만, STG2나 FULL과 같은 어려운 작업에서는 본 논문에서 제안한 프레임워크보다 훨씬 낮은 성능을 보였다.
- **사전 학습의 정확도:** 사전 학습 단계에서의 최대 분류 정확도는 27.5%로 매우 낮았다. 이는 단순한 지도 학습만으로는 BiMGame을 해결하기 어렵다는 것을 의미하지만, 그럼에도 불구하고 RL의 시작점으로 활용했을 때 가속 효과가 있음을 보여준다.
- **보상 설계의 효과:** 거리 기반 보상을 Geodesic distance(측지선 거리)로 변경하여 적용해 보았으나, 학습 속도 향상에 큰 도움이 되지 않았다.

## 🧠 Insights & Discussion

본 연구는 시뮬레이터를 통해 생성한 궤적이 Deep RL의 초기 수렴 속도를 높이는 데 효과적임을 입증하였다. 특히 인간 전문가 없이도 시뮬레이터의 물리 엔진을 활용해 효율적으로 사전 학습 데이터를 구축할 수 있다는 점이 강점이다.

하지만 몇 가지 한계점이 관찰되었다. 첫째, 사전 학습을 통해 속도를 높였음에도 불구하고 여전히 수백만 단위의 샘플이 필요하다는 점이다. 둘째, 보상이 극도로 희소한 STG3, STG4 작업에서는 에이전트가 이전 학습 내용을 잊어버리는(forgetting) 현상이 발생하였으며, 모든 알고리즘이 학습에 실패하였다. 이는 매우 희소한 보상 환경에서는 단순한 사전 학습과 RL의 결합만으로는 부족하며, 작업을 하위 작업(sub-tasks)으로 분할하거나 인간의 사전 지식(human priors)을 통합하는 방식이 필요함을 시사한다.

결론적으로, 본 논문은 Sim2Real 관점에서 시뮬레이터를 데이터 생성기로 활용하는 간단하고 효과적인 파이프라인을 제시하였으나, 극단적인 희소 보상 문제 해결이라는 과제를 남겼다.

## 📌 TL;DR

본 논문은 복잡한 동역학을 가진 Ball-in-Maze Game을 효율적으로 학습시키기 위해, **시뮬레이터 기반의 궤적 생성 $\rightarrow$ 지도 학습 사전 학습 $\rightarrow$ A3C 기반 RL 미세 조정** 파이프라인을 제안하였다. 이 방법은 무작위 초기화 대비 학습 속도를 2-3배 향상시켰으며, 특히 보상이 희소한 환경에서 RL의 탐색 효율을 높이는 데 기여한다. 향후 연구에서는 매우 희소한 보상 문제를 해결하기 위해 하위 작업 분할이나 인간의 사전 지식 도입이 필요할 것으로 보인다.