# Unlabeled Imperfect Demonstrations in Adversarial Imitation Learning

Yunke Wang, Bo Du, Chang Xu (2023)

## 🧩 Problem to Solve

본 논문은 Adversarial Imitation Learning (AIL) 환경에서 전문가의 시연 데이터(expert demonstrations)가 불완전할 때 발생하는 문제를 해결하고자 한다. 일반적으로 GAIL과 같은 AIL 프레임워크는 전문가의 데이터가 항상 최적(optimal)이라는 가정하에, 판별기(discriminator)가 이를 정답(positive)으로 학습하고 정책(policy)이 이를 모방하도록 설계된다.

그러나 실제 환경에서 수집된 데이터는 최적의 데이터와 비최적(non-optimal) 데이터가 섞여 있는 불완전한 상태인 경우가 많다. 만약 이러한 불완전한 데이터를 모두 정답으로 처리하면, 에이전트는 비최적의 행동까지 모방하게 되어 결국 성능이 저하되는 문제가 발생한다. 따라서 본 연구의 목표는 어떤 데이터가 최적인지 라벨이 없는 상태(unlabeled)에서, 최적의 정책을 효율적으로 학습할 수 있는 알고리즘을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 불완전한 전문가 시연 데이터를 '정답'이 아닌 '라벨이 없는(unlabeled)' 데이터로 취급하고, 이를 **Positive-Unlabeled (PU) Learning** 관점에서 접근하는 것이다.

제안된 **UID** 프레임워크의 중심 직관은 에이전트의 현재 정책이 생성하는 궤적(trajectory)과 잘 매칭되는 전문가 데이터만을 동적으로 샘플링하여 학습에 활용하는 것이다. 학습 초기에는 에이전트의 성능이 낮으므로 비최적 데이터와 유사한 궤적을 생성하겠지만, 적대적 학습 과정에서 판별기를 속이기 위해 점차 최적의 데이터와 유사한 궤적을 생성하도록 최적화된다. 결과적으로 에이전트가 쉬운 샘플부터 어려운 샘플 순으로 학습하는 **Self-paced learning** 효과를 얻어 최적의 정책에 도달하게 된다.

## 📎 Related Works

기존의 불완전한 시연 데이터 처리 방법은 크게 두 가지로 나뉜다.

1.  **Confidence-based methods**: 각 시연 데이터에 신뢰도(confidence) 가중치를 부여하는 방식이다. 2IWIL이나 IC-GAIL은 일부 데이터에 대해 사람이 직접 라벨링을 해야 하는 제약이 있으며, WGAIL나 BCND는 모델의 학습 상태에 의존하므로 데이터 오염도가 높을 경우 신뢰도 추정 자체가 붕괴될 위험이 있다.
2.  **Preference-based methods**: 데이터 간의 상대적 순위(ranking)를 이용하는 방식(T-REX, D-REX 등)이다. 하지만 이는 정확한 순위 정보가 필요하다는 전제가 필요하다.

UID는 이러한 기존 방식들과 달리, 사전 라벨이나 순위 정보 없이도 PU learning 프레임워크를 통해 전문가 데이터 내부의 최적 샘플을 동적으로 구분해냄으로써 차별점을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조
UID는 특정 AIL 백본(GAIL, WAIL 등)에 결합할 수 있는 일반적인 프레임워크이다. 전문가 데이터셋 $D_e$를 라벨이 없는 데이터로 간주하고, 에이전트 정책 $\pi_\theta$의 분포 $\rho^{\pi_\theta}$를 이용하여 정답 분포를 추정한다.

### 핵심 방정식 및 손실 함수
전문가 분포 $\rho^{\pi_e}$를 최적 분포 $\rho^{\pi_\epsilon}$와 에이전트와 매칭되는 분포 $\rho^{\pi_{\hat{\theta}}}$의 혼합으로 모델링한다.
$$\rho^{\pi_e}(s, a) = (1-\alpha)\rho^{\pi_\epsilon}(s, a) + \alpha\rho^{\pi_{\hat{\theta}}}(s, a)$$
여기서 $\alpha$는 매칭된 분포의 비율이다. 판별기 $g$의 기대 리스크 $R^{\pi_e}(g)$는 다음과 같이 정의된다.
$$R^{\pi_e}(g) = (1-\alpha)\mathbb{E}_{(s,a)\sim\rho^{\pi_\epsilon}}[\phi(g(s,a))] + \alpha\mathbb{E}_{(s,a)\sim\rho^{\pi_{\hat{\theta}}}}[-\phi(g(s,a))]$$
$\rho^{\pi_\epsilon}$는 알 수 없으므로, 이를 $(\rho^{\pi_e} - \alpha\rho^{\pi_{\hat{\theta}}})$로 대체하고 $\rho^{\pi_{\hat{\theta}}}$를 현재 에이전트의 정책 $\rho^{\pi_\theta}$로 근사하여 다음과 같은 목적 함수 $J(g, \theta)$를 도출한다.
$$\max_\theta \min_g J(g, \theta) = \mathcal{T}\{0, \mathbb{E}_{(s,a)\sim\rho^{\pi_e}}[\phi(g(s,a))] - \alpha\mathbb{E}_{(s,a)\sim\rho^{\pi_\theta}}[\phi(g(s,a))]\} + \alpha\mathbb{E}_{(s,a)\sim\rho^{\pi_\theta}}[-\phi(g(s,a))]$$
여기서 $\phi$는 마진 기반의 손실 함수이며, $\mathcal{T}\{\cdot\}$는 부호를 유지하기 위한 제약 조건이다.

### AIL 프레임워크로의 확장 (Theorem 1)
본 논문은 $\phi$를 $f$-divergence와 연결하여 다양한 AIL 기법에 적용할 수 있음을 증명하였다.

*   **UID-GAIL**: Jensen-Shannon divergence를 사용하여 다음과 같은 목적 함수를 갖는다.
    $$\min_\theta \max_\psi J(\theta, \psi) = \min\{0, \mathbb{E}_{(s,a)\sim\rho^{\pi_e}} \log[D_\psi(s,a)] - \alpha\mathbb{E}_{(s,a)\sim\rho^{\pi_\theta}} \log[D_\psi(s,a)]\} + \alpha\mathbb{E}_{(s,a)\sim\rho^{\pi_\theta}} \log[1-D_\psi(s,a)]$$
*   **UID-WAIL**: Total Variation (TV) 거리와 Lipschitz 제약 조건을 추가하여 Wasserstein distance 기반으로 확장한다.

### 학습 절차
1.  에이전트 정책 $\pi_\theta$와 판별기 $D_\psi$를 초기화한다.
2.  에이전트 궤적과 전문가 데이터 $D_e$를 샘플링한다.
3.  위의 PU 기반 목적 함수를 최대화하여 $D_\psi$를 업데이트한다.
4.  판별기의 출력을 보상으로 사용하여 TRPO와 같은 RL 알고리즘으로 $\pi_\theta$를 업데이트한다.

## 📊 Results

### 실험 설정
*   **데이터셋**: MuJoCo (Ant-v2, HalfCheetah-v2, Walker2d-v2) 및 RoboSuite (Nut Assembly).
*   **불완전 데이터 생성**: 최적 정책 데이터 $D_o$와 비최적 정책(학습 중간 체크포인트 또는 가우시안 노이즈 추가) 데이터 $D_n$을 혼합하여 생성.
*   **지표**: 누적 보상(Cumulative Reward).

### 주요 결과
1.  **최적 데이터 비율 변화**: 최적 데이터의 비율이 50%에서 16.7%까지 낮아질 때, BCND와 같은 기존 방식은 성능이 급격히 하락하여 baseline(BC)보다 낮아지기도 하지만, UID는 GAIL보다 일관되게 높은 성능을 유지하였다.
2.  **AIL 백본과의 호환성**: UID-GAIL과 UID-WAIL 모두 vanilla GAIL/WAIL보다 월등한 성능 향상을 보였으며, 이는 통계적으로 유의미함(p-value < 0.05)이 확인되었다.
3.  **실제 데이터 적용**: RoboSuite의 Nut Assembly 작업에서 실제 인간 운영자의 불완전한 시연 데이터를 사용했을 때, UID가 비교 대상 중 가장 높은 성능을 기록하여 강건함을 증명하였다.
4.  **판별기 분석**: GAIL의 판별기는 최적 데이터와 비최적 데이터를 모두 positive로 인식하지만, UID의 판별기는 학습 과정에서 두 데이터를 구분하는 능력을 갖추게 됨이 확인되었다.

## 🧠 Insights & Discussion

본 논문은 전문가 데이터를 무조건적인 '정답'으로 믿지 않고, 에이전트의 성장 단계에 맞춰 동적으로 정답을 찾아가는 PU learning의 관점을 AIL에 도입하였다.

가장 흥미로운 점은 **Self-paced learning**과의 연결성이다. 이론적 분석에 따르면, 초기에는 에이전트가 비최적 데이터와 유사하므로 판별기가 최적 데이터를 더 강조하게 되고, 에이전트가 점차 최적 데이터를 정복함에 따라 학습 대상이 확장된다. 이는 커리큘럼 학습과 유사한 효과를 내어 학습의 안정성과 최종 성능을 모두 잡을 수 있었다.

한계점으로는 하이퍼파라미터 $\alpha$의 설정이 필요하다는 점이 있으나, 실험 결과 $\alpha$ 값의 변화에 대해 상당히 관대한 내성(tolerance)을 보였으므로 실제 적용 시 큰 걸림돌이 되지 않을 것으로 보인다. 또한, PU-GAIL과 비교했을 때 에이전트 데이터가 아닌 전문가 데이터를 unlabeled로 처리함으로써 불완전한 시연 문제에 더 직접적인 해결책을 제시하였다.

## 📌 TL;DR

이 논문은 불완전한 전문가 시연 데이터가 포함된 환경에서 최적의 정책을 학습하기 위해, 전문가 데이터를 라벨 없는 데이터로 취급하는 **UID (Unlabeled Imperfect Demonstrations)** 프레임워크를 제안한다. PU learning을 통해 최적의 샘플을 동적으로 구분하고 Self-paced learning 방식으로 학습함으로써, 데이터 오염도가 높은 상황에서도 기존 AIL 방식보다 훨씬 강건하고 높은 성능을 달성하였다. 이 연구는 현실 세계의 노이즈 섞인 전문가 데이터로부터 고품질의 정책을 추출해야 하는 로봇 제어 및 강화학습 분야에 중요한 기여를 한다.