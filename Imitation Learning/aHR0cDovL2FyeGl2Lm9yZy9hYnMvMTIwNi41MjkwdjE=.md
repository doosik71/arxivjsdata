# Imitation Learning with a Value-Based Prior

Umar Syed, Robert E. Schapire (2007)

## 🧩 Problem to Solve

본 논문은 stochastic environment에서 멘토(mentor)의 행동을 관찰하여 학습하는 Imitation Learning(모방 학습)의 데이터 효율성 문제를 해결하고자 한다. 일반적으로 모방 학습은 지도 학습(supervised learning)의 관점에서 접근하므로, 상태 공간(state space)이 넓을 경우 멘토로부터 방대한 양의 시연(demonstration) 데이터를 얻어야 한다는 한계가 있다.

반면, Reinforcement Learning(강화 학습)은 보상 함수(reward function)를 통해 학습하지만, 실제 환경에서는 사람이 수동으로 보상 함수를 설계해야 하며, 이 과정에서 정밀한 튜닝이 어렵고 설계된 보상이 항상 정확하다는 보장이 없다.

따라서 본 연구의 목표는 **모델링 MDP(modeling MDP)를 통해 멘토의 행동에 대한 사전 지식(prior knowledge)을 인코딩**함으로써, 제한된 시연 데이터만으로도 효율적으로 멘토의 정책을 추정하는 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 멘토의 정책 $\pi$에 대한 사전 확률 $P(\pi)$를 해당 정책의 가치(value)와 연동시키는 **Value-based Prior**를 도입한 것이다.

- **가치 기반 사전 확률 설계**: 멘토의 정책이 모델링 MDP에서 높은 가치를 가질 가능성이 높다는 직관을 바탕으로, 사전 확률을 $P(\pi) = \exp(\alpha V(\pi))$로 정의한다. 여기서 $\alpha$는 사전 지식과 관찰 데이터 사이의 가중치를 조절하는 trade-off 파라미터이다.
- **효율적인 학습 알고리즘**: 비볼록(non-convex) 제약 조건을 가진 최적화 문제를 해결하기 위해 Alternating Maximization 방식을 제안하였으며, 이는 EM 알고리즘과 유사하게 정지점(stationary point)으로 수렴함을 증명하였다.
- **유연한 사전 지식 통합**: 특정 최적 정책 하나만을 강요하는 기존의 방식과 달리, 가치가 높은 여러 정책에 대해 열린 가능성을 부여함으로써 멘토의 정책이 최적이 아니거나 다양한 고가치 정책이 존재할 때 더욱 강건한 학습이 가능하게 한다.

## 📎 Related Works

논문에서는 기존의 사전 지식 활용 방식들과 본 제안 방식의 차별점을 다음과 같이 설명한다.

- **Dirichlet Distribution 기반 방식 (Price et al.)**: 각 상태에서의 정책을 Dirichlet 분포로 모델링하여 최적 행동에 높은 가중치를 둔다. 하지만 이는 특정 정책이 가장 확률이 높다고 가정하므로, 멘토의 정책이 최적 정책과 다를 경우 학습에 편향(skew)이 발생할 수 있다.
- **Modified TD Learning 및 Boltzmann Distribution 방식 (Henderson et al., Fern et al.)**: $Q$-value를 조정하거나 Boltzmann 분포를 사용하여 고가치 행동에 우선순위를 둔다. 본 논문의 방식은 이를 보다 일반적인 Bayesian MAP(Maximum A Posteriori) 추정 프레임워크로 확장한 것이다.
- **Inverse Reinforcement Learning (IRL)**: 시연 데이터로부터 보상 함수 자체를 추출하려 한다. IRL은 보통 보상 함수가 특징(feature)들의 선형 결합으로 표현된다고 가정하지만, 본 논문의 방식은 보상 함수가 이미 주어져 있다고 가정하되 이를 정책 추론의 편향(bias)으로만 사용한다는 점에서 차이가 있다.

## 🛠️ Methodology

### 전체 파이프라인 및 목표 함수
학습자는 유한한 시계(finite-horizon)를 가진 모델링 MDP $(S, A, H, R)$와 멘토의 상태-행동 궤적 데이터셋 $D$를 제공받는다. 학습의 목표는 멘토의 정책 $\pi$를 추정하는 것이며, 이를 위해 다음과 같은 MAP 추정치 $\hat{\pi}$를 찾는다.

$$\hat{\pi} = \arg \max_{\pi} \sum_{s,a,t} K_{sat} \log \pi_{sa}^t + \log P(\pi)$$

여기서 $K_{sat}$는 데이터셋 $D$에서 상태 $s$일 때 행동 $a$가 선택된 횟수이다. 여기에 Value-based Prior인 $\log P(\pi) = \alpha V(\pi)$를 대입하면 최종 목적 함수 $L(\pi)$는 다음과 같다.

$$L(\pi) = \sum_{s,a,t} K_{sat} \log \pi_{sa}^t + \alpha V(\pi)$$

### 알고리즘: Alternating Maximization
$V(\pi)$의 계산 복잡도와 Bellman 방정식의 비선형성(bilinear) 때문에 직접적인 최적화가 어렵다. 이를 해결하기 위해 본 논문은 각 시점 $\tau \in \{0, \dots, H\}$의 정책 $\pi_\tau$를 순차적으로 최적화하는 반복 알고리즘을 제안한다.

1. **초기화**: 정책 $\tilde{\pi}$를 무작위로 설정한다.
2. **반복 최적화**: $\tau=0$부터 $H$까지 순회하며, 다른 시점의 정책 $\pi_{t \neq \tau}$는 고정시킨 채 $\pi_\tau$에 대해서만 $L(\pi)$를 최대화한다.
3. **수렴 조건**: $|L(\tilde{\pi}) - L(\pi)|$가 임계값보다 작아질 때까지 반복한다.

### 세부 최적화 절차 (KKT 조건 활용)
각 $\pi_\tau$의 최적화 단계는 다음과 같은 3단계로 수행된다.

- **Step 1 (Lagrange multiplier 계산)**: $\lambda_{V_{st}}$를 계산한다. 이는 정책 $\pi$ 하에서 상태 $s$의 점유 확률(occupancy probability)에 $\alpha \gamma^t$를 곱한 값과 같다.
- **Step 2 (정책 업데이트)**: 각 상태 $s$에 대해 $\lambda_{\pi_s}$를 찾기 위해 이분법(bisection method)과 같은 root-finding 알고리즘을 사용하여 $\pi_{sa}^\tau$를 결정한다. 이때 $K_{sat}=0$인 행동과 그렇지 않은 행동을 구분하여 KKT 조건을 만족시키는 최적의 확률 분포를 생성한다.
- **Step 3 (가치 함수 업데이트)**: 결정된 $\pi$를 바탕으로 Bellman 방정식을 통해 $V_{ts}$를 역방향으로 재계산한다.

### 전이 확률($\theta$)이 알려지지 않은 경우
전이 확률 $\theta$를 모를 경우, 상태 공간과 행동 공간을 확장하여 $\tilde{S} = S \cup (S \times A)$와 $\tilde{A} = A \cup S$로 재정의함으로써 $\theta$를 정책 $\tilde{\pi}$의 일부로 포함시킨다. 이를 통해 전이 확률 학습과 정책 학습을 동시에 수행할 수 있다.

## 📊 Results

### 실험 설정
- **환경**: $30 \times 30$ 그리드 미로. 시작점에서 목표점까지 도달하는 작업이며, 장애물(음수 보상)과 무작위 이동 확률(30%)이 존재한다.
- **특이사항**: 단일 최적 정책만 존재할 경우 Dirichlet Prior와의 차별점이 적으므로, 모든 행동에 대해 동일한 효과를 내는 "Twin action"을 도입하여 여러 개의 고가치 정책이 존재하도록 설계하였다.
- **비교 대상**: Dirichlet Prior (Price et al.), Hybrid RL/SL (Henderson et al.).
- **지표**: 멘토의 실제 정책과 추정된 정책 간의 RMS Error.

### 주요 결과
- **데이터 효율성**: Value-based Prior는 데이터 양이 적을 때부터 학습 속도를 높이며, 특히 데이터가 증가함에 따라 Dirichlet Prior보다 훨씬 낮은 오차를 기록하였다. 이는 Dirichlet 방식이 특정 하나의 정책만을 강요하는 반면, 제안 방식은 가치가 높은 다양한 정책을 수용하기 때문이다.
- **파라미터 강건성**: $\alpha$ 값을 3자릿수 범위로 변경하며 실험했을 때, 제안된 방법은 $\alpha$ 값의 변화에 매우 강건하게 성능을 유지하였다.
- **멘토의 최적성 여부**: 멘토의 정책이 최적 정책의 가치 대비 80% 수준까지 떨어지더라도, Value-based Prior는 여전히 성능 향상을 제공함을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 이론적 의미
본 논문의 가장 큰 통찰은 **"멘토는 보상을 추구하는 존재"라는 자연스러운 믿음을 수학적으로 모델링**한 점이다. 특히, 제안된 Prior는 상태 간의 **결합(coupling) 특성**을 가진다. 즉, 특정 상태에서의 관찰 데이터가 다른 상태의 정책 추정치에 영향을 줄 수 있는데, 이는 전역적인 가치 함수 $V(\pi)$를 최적화 목표에 포함시켰기 때문에 가능하다.

### 한계 및 비판적 해석
- **수렴성**: 알고리즘이 정지점(stationary point)으로 수렴한다는 것은 증명되었으나, 전역 최적점(global optimum)에 도달한다는 보장은 없다.
- **계산 복잡도**: 매 반복마다 $\lambda$ 값을 계산하고 root-finding을 수행해야 하므로, 상태 공간이 매우 커질 경우 계산 비용이 급격히 증가할 가능성이 크다. 논문에서도 이를 해결하기 위해 향후 Function Approximation 도입의 필요성을 언급하고 있다.
- **가정**: 모델링 MDP의 보상 함수 $R$이 멘토의 실제 의도와 어느 정도 일치한다는 가정이 필요하다. 만약 $R$이 멘토의 가치 체계와 완전히 동떨어져 있다면, 오히려 Prior가 학습을 방해하는 Noise로 작용할 수 있다.

## 📌 TL;DR

본 논문은 모방 학습에서 부족한 시연 데이터를 보완하기 위해, **정책의 가치(Value)를 사전 확률로 사용하는 Value-based Prior**를 제안하였다. 이를 위해 모델링 MDP를 도입하고, Alternating Maximization 기반의 효율적인 추정 알고리즘을 설계하여 정지점으로의 수렴성을 증명하였다. 실험 결과, 제안 방법은 다양한 고가치 정책이 존재하는 환경에서 기존 Dirichlet Prior보다 강건하며, 멘토의 정책이 다소 최적이 아니더라도 효율적으로 학습할 수 있음을 보여주었다. 이 연구는 강화 학습의 보상 체계와 모방 학습의 데이터 기반 접근법을 베이지안 프레임워크 내에서 성공적으로 통합하였다는 점에서 의의가 있다.