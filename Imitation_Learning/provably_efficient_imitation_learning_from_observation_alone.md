# Provably Efficient Imitation Learning from Observation Alone

Wen Sun, Anirudh Vemula, Byron Boots, J. Andrew Bagnell

## 🧩 Problem to Solve

본 논문은 대규모 MDP(Markov Decision Process) 환경에서 **관측치만을 이용한 모방 학습(Imitation Learning from Observations alone, ILFO)** 문제를 다룹니다. 기존 대부분의 모방 학습 알고리즘은 전문가의 행동 신호에 접근할 수 있다고 가정하지만, ILFO 설정에서는 학습자가 오직 관측치 시퀀스만 제공받고, 행동 또는 보상 신호는 주어지지 않습니다. 이 도전적인 환경에서, 매우 큰 관측 공간($|X_h|$가 매우 큰 고차원 관측치)을 가진 MDP를 위해, **샘플 및 계산 효율적**으로 준최적 정책을 학습하는 알고리즘을 개발하는 것이 목표입니다. 특히, 샘플 복잡도가 관측 공간의 크기에 독립적이고, 우수한 성능 보장(전문가 정책에 근접한 성능), 그리고 계산 효율성(다항 시간 내 오라클 호출 횟수)을 만족하는 알고리즘을 요구합니다.

## ✨ Key Contributions

* **FAIL (Forward Adversarial Imitation Learning)**이라는 새로운 모델 프리(model-free) 알고리즘을 제안합니다. 이는 ILFO 설정에서 최초로 **증명 가능한 효율성(provably efficient)**을 제공하는 알고리즘입니다.
* 학습된 정책이 관련된 모든 매개변수에 대해 다항식적인 샘플 수(horizon $H$, 행동 수 $K$, 함수 근사기의 통계적 복잡도)로 준최적 정책을 학습하며, **관측 공간의 고유한 관측치 수와는 독립적**입니다. 이는 기존의 샘플 효율적인 학습 알고리즘(주로 테이블형 RL 또는 초기화 분포에 접근 가능한 설정)의 영역을 확장합니다.
* ILFO가 순수 강화 학습(Reinforcement Learning, RL)보다 **기하급수적으로 더 샘플 효율적**일 수 있음을 이론적으로 입증합니다.
* FAIL을 립시츠 연속 MDP, 상호작용적 ILFO, 상태 추상화(state abstraction)와 같은 특정 설정으로 확장 가능함을 보여주며, 이 경우 **내재적 벨만 오류(Inherent Bellman Error, IBE)를 제거**할 수 있음을 나타냅니다.
* OpenAI Gym 제어 태스크에서 FAIL의 효능을 입증합니다.

## 📎 Related Works

* **행동 정보가 있는 모방 학습**: DAgger, AggreVaTe, GAIL, Behaviour Cloning과 같은 방법들은 전문가의 행동에 접근해야 하므로 ILFO에는 적용할 수 없습니다.
* **역모델 기반 접근법**: Torabi et al. (2018), Edwards et al. (2018) 등은 관측치로부터 전문가의 행동을 예측하는 역모델을 학습하지만, 이는 공변량 이동(covariate shift) 문제에 취약하며 일반적으로 성능 보장이 어렵습니다.
* **수동 설계 비용 함수**: Liu et al. (2018), Peng et al. (2018) 등은 전문가 궤적과의 편차에 페널티를 주는 수동 설계 비용 함수를 사용하지만, 이는 작업별 지식을 요구하며 실제 비용 함수와 다를 수 있습니다.
* **기존 샘플 효율적 RL**: 대부분의 기존 RL 알고리즘은 작은 테이블형 MDP에 초점을 맞추어 관측치 수에 다항식적으로 의존하므로 대규모 MDP에는 적합하지 않습니다.

## 🛠️ Methodology

FAIL은 순차 학습 문제를 $H$(horizon 길이)개의 독립적인 2인조 최소-최대 게임으로 분해합니다.

1. **전방 학습(Forward Training)**: 시간 단계 $h=1$부터 $H-1$까지 순차적으로 정책 $\pi_h$를 학습합니다.
2. **단일 스텝 정책 학습($\pi_h$)**:
    * 이전 정책 $\pi_1, \dots, \pi_{h-1}$이 고정되었다고 가정하고, 학습자 정책 $\pi$가 생성하는 다음 시간 단계($h+1$)의 관측 분포 $\nu_{h+1}$가 전문가의 관측 분포 $\mu^?_{h+1}$에 최대한 가깝도록 $\pi_h$를 학습합니다.
    * 이 '가까움'의 척도로 **적분 확률 거리(Integral Probability Metric, IPM)**를 사용합니다. IPM은 $d_{\mathcal{F}}(P_1, P_2) = \sup_{f \in \mathcal{F}} (E_{x \sim P_1}[f(x)] - E_{x \sim P_2}[f(x)])$로 정의되며, 여기서 $\mathcal{F}$는 판별자(discriminator) 함수 클래스입니다.
    * 목표는 $\min_{\pi \in \Pi_h} d_{\mathcal{F}_{h+1}}(\pi | \nu_h, \mu^?_{h+1})$를 푸는 것입니다.
3. **경험적 IPM 추정**: 실제 $\mu^?_{h+1}$에 직접 접근할 수 없으므로, 전문가의 관측치 샘플과 학습자의 관측치 샘플을 이용하여 IPM을 경험적으로 추정합니다. 이 과정에서 행동 샘플링에 대한 중요도 샘플링(importance weighting)이 사용됩니다.
    $$
    \hat{d}_{\mathcal{F}_{h+1}}(\pi|\nu_h,\mu^?_{h+1}) = \max_{f \in \mathcal{F}_{h+1}} \left( \frac{1}{N} \sum_{i=1}^N \frac{K\pi(a^i_h|x^i_h)}{1/K} f(x^i_{h+1}) - \frac{1}{N'} \sum_{i=1}^{N'} f(\tilde{x}^i_{h+1}) \right)
    $$
4. **최소-최대 게임(Algorithm 1)**: 위 경험적 IPM을 최소화하기 위한 2인조 최소-최대 게임을 무-후회(no-regret) 온라인 학습(예: FTRL for $\pi$, LP 오라클을 이용한 $f$의 최적 반응)을 통해 해결합니다.
    * $f_n = \arg \max_{f \in \mathcal{F}} u(\pi_n, f)$ (LP 오라클)
    * $\pi_{n+1} = \arg \min_{\pi \in \Pi} \sum_{t=1}^n u(\pi, f_t) + \phi(\pi)$ (정규화된 CS 오라클)
5. **판별자 클래스 $\mathcal{F}$의 중요성**: 샘플 효율성을 위해 판별자 클래스 $\mathcal{F}$는 명시적으로 정규화(예: 유한한 VC 또는 Rademacher 복잡도)되어야 합니다. 너무 강력한 판별자 클래스($\mathcal{F} = \{f : \|f\|_{\infty} \le c\}$)는 일반화되지 않고 샘플 복잡도를 관측 공간 크기에 의존하게 만듭니다(Theorem 3.2). 이 트레이드오프는 내재적 벨만 오류(Inherent Bellman Error, IBE)로 측정됩니다.

## 📊 Results

* **이론적 결과**:
  * **Theorem 3.1**: Algorithm 1은 주어진 데이터셋에서 $O(\epsilon)$ 오차 범위 내에서 경험적 IPM을 최소화하는 정책을 찾습니다. 필요한 샘플 수 $N = \Theta(\frac{K \log(|\Pi_h| |\mathcal{F}_{h+1}| / \delta)}{\epsilon^2})$, 반복 횟수 $T = \Theta(\frac{K^2}{\epsilon^2})$.
  * **Theorem 3.2**: 판별자 클래스 $\mathcal{F}$가 무한한 용량(예: 총 변동 거리)을 가질 경우, 샘플 수가 관측 공간 크기에 다항식적으로 의존하지 않으면 일반화되지 않음을 증명하여 판별자 정규화의 필요성을 강조합니다.
  * **Theorem 3.3 (메인)**: FAIL (Algorithm 2)은 $J(\pi) - J(\pi^?) \le O(H^2 \epsilon'_{\text{be}}) + O(H^2 \epsilon)$의 성능을 달성하며, 필요한 궤적 수는 $\tilde{O}(\frac{HK}{\epsilon^2} \log(\frac{|\Pi||\mathcal{F}|}{\delta}))$입니다. 이는 관측 공간 크기에 독립적인 다항식 샘플 복잡도를 보장합니다.
  * **Proposition 4.1**: 특정 MDP 패밀리에서 ILFO가 순수 RL에 비해 준최적 정책을 찾기 위해 **기하급수적으로 더 적은 궤적**이 필요함을 입증합니다.
  * **특정 설정에서의 확장**: 립시츠 연속 MDP, 상호작용적 ILFO, 상태 추상화 환경에서는 IBE $\epsilon'_{\text{be}}$를 0으로 만들 수 있음을 보여줍니다.
  * **모델 기반 FAIL (Corollary 8.1)**: 모델 클래스가 참 모델을 포함하고(realizability) 판별자 클래스가 참 값 함수를 포함하면, 다항식 샘플 복잡도로 준최적 정책 학습이 가능하지만, 계산 효율성은 낮습니다.
* **실험 결과 (OpenAI Gym)**:
  * **Swimmer, Reacher (밀집 보상)**: FAIL은 수정된 GAIL(행동 입력 없음)과 유사하거나 더 나은 성능을 보여줍니다.
  * **Reacher Sparse, FetchReach (희소 보상)**: FAIL은 수정된 GAIL보다 **상당히 우수한 성능**을 보입니다. FAIL은 각 시간 단계에서 전문가의 상태 분포를 일치시키려 노력하여 희소 보상 환경에서 목표 도달에 효과적입니다.
  * **한계점**: 긴 horizon 태스크에서는 성능 향상이 GAIL보다 적게 나타납니다. 이는 FAIL이 전체 horizon에 걸쳐 시간 의존적 정책 시퀀스를 학습해야 하기 때문이며, 향후 정지 정책(stationary policy) 학습으로의 확장이 필요함을 시사합니다.

## 🧠 Insights & Discussion

* **ILFO의 실현 가능성**: 본 연구는 대규모 MDP에서 관측치만으로도 효율적인 모방 학습이 가능하며, 이는 로봇이 전문가의 시연을 단순히 '관찰'하는 것만으로 학습하는 시나리오에 중요한 의미를 가집니다.
* **판별자 정규화의 중요성**: 샘플 효율적인 학습을 위해서는 판별자 클래스의 용량을 명시적으로 제어하고, 내재적 벨만 오류를 줄이도록 설계해야 합니다. 이는 GAN 문헌에서의 통찰과도 일맥상통합니다.
* **전문가 관측의 가치**: 전문가의 궤적 관측치에 접근하는 것이 순수 RL보다 기하급수적으로 더 샘플 효율적일 수 있음을 이론적으로 보여줌으로써, ILFO의 잠재적 이점을 명확히 합니다.
* **모델 기반 vs. 모델 프리**: 모델 기반 FAIL은 이론적으로 더 강력한 성능 보장을 제공하지만, 계산 복잡성 문제가 있습니다. 모델 프리 설정에서 통계적, 계산적 효율성을 동시에 달성하기 위한 충분 및 필요 조건을 찾는 것이 향후 연구 과제입니다.

## 📌 TL;DR

관측치만으로 모방 학습(ILFO)하는 대규모 MDP 문제에서, FAIL(Forward Adversarial Imitation Learning)은 문제를 $H$개의 독립적인 2인조 최소-최대 게임으로 분해하여 전문가 관측 분포와 학습자 관측 분포 간의 IPM을 시간 단계별로 최소화합니다. 이 알고리즘은 관측 공간 크기와 독립적인 다항식적 샘플 복잡도를 가지며, ILFO가 순수 RL보다 기하급수적으로 효율적일 수 있음을 증명합니다. 실험적으로 희소 보상 환경에서 GAIL보다 뛰어난 성능을 보이며, 대규모 MDP에서 ILFO의 이론적, 실용적 가능성을 제시합니다.
