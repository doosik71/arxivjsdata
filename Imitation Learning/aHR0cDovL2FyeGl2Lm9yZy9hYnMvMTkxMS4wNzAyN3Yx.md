# On Value Discrepancy of Imitation Learning

Tian Xu, Ziniu Li, Yang Yu (2019)

## 🧩 Problem to Solve

본 논문은 모방 학습(Imitation Learning, IL) 알고리즘들이 가지는 이론적 성질, 특히 **타임 호라이즌(Time Horizon)에 따른 성능 저하와 샘플 복잡도(Sample Complexity)**를 분석하는 것을 목표로 한다.

일반적으로 모방 학습은 전문가의 시연(Expert Demonstrations)을 통해 최적의 정책을 학습하여 강화 학습의 샘플 효율성 문제를 해결하려 한다. 그러나 행동 복제(Behavioral Cloning, BC)와 같은 방식은 전문가가 방문하지 않은 상태에 도달했을 때 오류가 누적되는 **복합 오류(Compounding Errors)** 문제가 발생한다. 반면, GAIL(Generative Adversarial Imitation Learning)과 같은 최신 알고리즘들은 경험적으로 우수한 성능을 보이지만, 이에 대한 엄밀한 이론적 분석은 부족한 상태였다.

따라서 본 논문은 다양한 모방 학습 접근 방식의 성능 차이를 분석할 수 있는 일반적인 이론적 프레임워크를 제안하고, 이를 통해 BC와 GAIL의 가치 차이(Value Discrepancy)가 호라이즌에 따라 어떻게 달라지는지를 정량적으로 증명하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 **불일치 전파 분석(Discrepancy Propagation Analysis)** 프레임워크를 통해 서로 다른 모방 학습 알고리즘의 이론적 상한(Upper Bound)을 도출한 것이다.

1. **일반적 분석 프레임워크 제안**: 정책 불일치 $\rightarrow$ 상태 분포 불일치 $\rightarrow$ 상태-행동 분포 불일치 $\rightarrow$ 가치 불일치로 이어지는 전파 과정을 정립하였다.
2. **BC의 이론적 한계 증명**: 행동 복제(BC)의 가치 불일치가 할인 계수 $\gamma$에 대해 $O(\frac{1}{(1-\gamma)^2})$의 차수를 가짐을 보임으로써, 호라이즌이 길어질수록 성능이 제곱 단위로 빠르게 저하됨을 이론적으로 규명하였다.
3. **GAIL의 이론적 성능 분석**: GAIL의 가치 불일치가 $O(\frac{1}{1-\gamma})$의 차수를 가짐을 증명하였다. 이는 GAIL이 BC보다 복합 오류에 훨씬 강건하며, 호라이즌 의존성이 낮음을 의미한다. 본 논문은 GAIL의 성능을 이론적으로 분석한 첫 번째 연구라고 주장한다.

## 📎 Related Works

논문에서는 모방 학습을 크게 세 가지 범주로 분류하여 기존 연구의 한계를 설명한다.

- **Behavioral Cloning (BC)**: 전문가의 행동을 지도 학습 방식으로 직접 모방한다. 기존 연구(Ross et al., 2011 등)는 BC가 호라이즌 길이에 대해 이차적인 후회(Quadratic Regret)를 가짐을 밝혔으나, 본 논문은 이를 무한 호라이즌(Infinite-horizon) 설정으로 확장하여 분석한다.
- **Apprenticeship Learning (AL)**: 역강화학습(Inverse RL)을 통해 보상 함수를 추론하고 정책을 학습한다. FEM(Abbeel and Ng, 2004)이나 MWAL(Syed and Schapire, 2007) 등이 있으며, 샘플 복잡도에 대한 분석은 존재하지만 실제 정책 가치의 차이에 대한 분석은 미흡했다.
- **GAIL**: 적대적 학습을 통해 상태-행동 점유 측정치(State-action occupancy measure)를 일치시키는 방식이다. 경험적으로는 매우 강력하지만, 비선형 함수 근사기를 사용하기 때문에 이론적 분석이 매우 어려웠다.

본 논문은 이러한 기존 연구들과 달리, 단순한 정책 모방(BC)과 분포 일치(GAIL)가 최종적인 **가치 함수(Value Function)**에 미치는 영향을 직접적으로 비교 분석했다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. 불일치 전파 분석 프레임워크 (Discrepancy Propagation)
논문은 정책의 차이가 최종 가치의 차이로 이어지는 과정을 세 단계의 보조정리(Lemma)로 정의한다.

- **Lemma 4.1 (정책 $\rightarrow$ 상태 분포)**: 정책 $\pi$와 전문가 정책 $\pi_E$ 사이의 총 변동(Total Variation, $D_{TV}$)의 기대값은 상태 분포 $d^\pi$와 $d^{\pi_E}$ 사이의 불일치 상한을 결정한다.
  $$D_{TV}(d^\pi, d^{\pi_E}) \leq \frac{\gamma}{1-\gamma} \mathbb{E}_{s \sim d^{\pi_E}}[D_{TV}(\pi(\cdot|s), \pi_E(\cdot|s))]$$
- **Lemma 4.2 (정책 $\rightarrow$ 상태-행동 분포)**: 위 결과를 확장하여 상태-행동 분포 $\rho^\pi$와 $\rho^{\pi_E}$ 사이의 불일치를 정의한다.
  $$D_{TV}(\rho^\pi, \rho^{\pi_E}) \leq \frac{1}{1-\gamma} \mathbb{E}_{s \sim d^{\pi_E}}[D_{TV}(\pi(\cdot|s), \pi_E(\cdot|s))]$$
- **Lemma 4.3 (상태-행동 분포 $\rightarrow$ 가치 차이)**: 최종적으로 상태-행동 분포의 불일치가 가치 함수의 차이로 연결된다. 보상 함수의 절대값이 $R_{\max}$로 제한될 때 다음과 같다.
  $$|V^\pi - V^{\pi_E}| \leq \frac{2R_{\max}}{1-\gamma} D_{TV}(\rho^\pi, \rho^{\pi_E})$$

### 2. Behavioral Cloning 분석
위 프레임워크를 BC에 적용하면, 정책 불일치가 가치 불일치로 전파되는 과정에서 $\frac{1}{1-\gamma}$ 항이 두 번 곱해지게 된다.

- **결과 (Theorem 5.1)**: BC의 가치 오류 상한은 다음과 같다.
  $$|V^{\pi_{bc}} - V^{\pi_E}| \leq \frac{2R_{\max}}{(1-\gamma)^2} \mathbb{E}_{s \sim d^{\pi_E}}[D_{TV}(\pi_{bc}(\cdot|s), \pi_E(\cdot|s))]$$
  이는 BC가 호라이즌에 대해 **이차적인 의존성(Quadratic Dependency)**을 가짐을 의미한다.

### 3. GAIL 분석
GAIL은 정책 자체를 모방하는 것이 아니라, 전체 상태-행동 분포 $\rho^\pi$를 전문가의 분포 $\rho^{\pi_E}$에 맞추는 Jensen-Shannon (JS) Divergence를 최소화한다.

- **가치 불일치 (Theorem 6.1)**: GAIL의 가치 오류는 JS Divergence의 제곱근에 비례하며, 호라이즌 의존성은 $\frac{1}{1-\gamma}$로 선형적이다.
  $$|V^{\pi_E} - V^{\pi_{GA}}| \leq \frac{2\sqrt{2}R_{\max}}{1-\gamma} \sqrt{D_{JS}(\rho^{\pi_{GA}}, \rho^{\pi_E})}$$
- **샘플 복잡도 분석**: GAIL의 일반화 능력을 분석하기 위해 **Neural Net Distance** $d_D(\mu, \nu)$를 도입한다. 이는 판별자(Discriminator) 신경망 집합 $D$에 대해 두 분포의 기대값 차이의 상한으로 정의된다. 이를 통해 판별자의 복잡도(Rademacher complexity)와 샘플 수 $m$이 가치 불일치에 미치는 영향을 정량화한 Theorem 6.2를 도출하였다.

## 📊 Results

### 실험 설정
- **환경**: Mujoco의 Ant, Hopper, Walker 작업.
- **비교 대상**: BC, DAgger, GAIL, FEM, GTAL.
- **측정 지표**: 실제 보상 함수를 이용한 정책 가치(Policy Value).
- **변수**: 할인 계수 $\gamma$ (호라이즌 의존성 측정), 전문가 궤적 수 $m$ (샘플 복잡도 측정).

### 주요 결과
1. **호라이즌 의존성 (Figure 2)**: 
   - $\gamma$가 0.9에서 0.999로 증가함에 따라 모든 알고리즘의 가치 값은 상승하지만, 전문가와의 격차는 BC에서 가장 급격하게 벌어진다.
   - 특히 y축이 로그 스케일임에도 불구하고 BC의 성능 저하가 두드러지는데, 이는 이론적으로 증명한 **이차적 성능 저하 $O(\frac{1}{(1-\gamma)^2})$**가 실제 환경에서 나타남을 입증한다.
   - DAgger는 BC와 동일한 목적 함수를 가지나, 학습 중 전문가에게 추가 쿼리를 수행함으로써 이 문제를 완화한다.

2. **샘플 복잡도 (Figure 3)**:
   - 전문가 궤적 수 $m$이 매우 적은 상황에서도 GAIL, FEM, GTAL과 같은 적대적 기반 알고리즘들이 BC보다 훨씬 높은 성능을 보인다.
   - 다만, 이러한 알고리즘들은 성능 도달을 위해 환경과의 많은 상호작용(약 $3 \times 10^6$ steps)이 필요하다는 점이 명시되었다.

## 🧠 Insights & Discussion

### 강점 및 해석
본 논문은 단순히 "GAIL이 BC보다 좋다"는 경험적 결과에 그치지 않고, 왜 좋은지를 **분포 불일치의 전파 과정**이라는 수학적 관점에서 풀어냈다. 특히 BC의 목적 함수인 '단순 정책 모방'이 시퀀셜 의사결정 문제에서 왜 위험한지를 $\frac{1}{(1-\gamma)^2}$이라는 수식으로 명확히 제시한 점이 매우 가치 있다.

### 한계 및 논의사항
- **판별자의 최적성 가정**: GAIL 분석 시 판별자가 최적이라고 가정하였으나, 실제 학습에서는 판별자와 생성자가 동시에 학습되므로 이 가정과 실제 사이에 간극이 존재할 수 있다.
- **신경망 거리(Neural Net Distance)의 추상성**: GAIL의 샘플 복잡도 분석에 사용된 $\Lambda_{D, \Pi}$와 같은 상수가 실제 신경망 구조에서 어떤 구체적인 값을 가지는지, 어떻게 제어할 수 있는지는 명확히 제시되지 않았다.
- **상호작용 비용**: 적대적 학습 기반 방법들이 샘플 효율성(전문가 데이터 기준)은 높지만, 환경 상호작용 비용이 매우 크다는 점은 실제 적용 시 고려해야 할 중요한 트레이드-오프이다.

## 📌 TL;DR

본 논문은 모방 학습의 이론적 성능을 분석하기 위한 '불일치 전파 분석' 프레임워크를 제안하였다. 분석 결과, **BC는 호라이즌에 대해 이차적인 오류($O(\frac{1}{(1-\gamma)^2})$)를 가지는 반면, GAIL은 선형적인 오류($O(\frac{1}{1-\gamma})$)를 가짐**을 수학적으로 증명하고 실험적으로 검증하였다. 이는 단순한 정책 모방보다 전체 점유 분포를 일치시키는 방식이 장기적인 시퀀셜 작업에서 훨씬 안정적임을 시사하며, 향후 모방 학습 알고리즘의 일반화 성능 개선을 위한 이론적 토대를 제공한다.