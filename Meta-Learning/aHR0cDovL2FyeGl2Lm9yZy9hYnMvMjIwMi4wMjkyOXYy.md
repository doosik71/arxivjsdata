# MODEL-BASED OFFLINE META-REINFORCEMENT LEARNING WITH REGULARIZATION

Sen Lin, Jialin Wan, Tengyu Xu, Yingbin Liang, Junshan Zhang (2022)

## 🧩 Problem to Solve

본 논문은 Offline Reinforcement Learning (Offline RL)에서 발생하는 **Distributional Shift** 문제를 해결하고, 이를 Meta-RL 프레임워크로 확장하여 새로운 작업(Task)에 빠르게 적응하는 것을 목표로 한다. 

기존의 Offline Meta-RL 방법론들은 여러 작업의 데이터를 통해 정보가 풍부한 Meta-policy를 학습하려 하지만, 저자들의 실험적 분석에 따르면 데이터셋의 품질이 좋은 경우 오히려 단일 작업(Single-task) Offline RL 방법론보다 성능이 떨어지는 현상이 발생한다. 이는 Meta-policy를 따라 Out-of-distribution(OOD) 상태-행동 쌍을 '탐색(Exploring)'하는 것과, 기존 데이터셋의 Behavior policy에 밀착하여 '활용(Exploiting)'하는 것 사이의 정교한 균형이 부족하기 때문이다. 

따라서 본 연구의 핵심 목표는 Meta-policy를 통한 탐색과 Behavior policy를 통한 활용 사이의 적절한 균형을 맞추어, 데이터셋의 품질과 관계없이 안전한 성능 향상을 보장하는 Model-based Offline Meta-RL 알고리즘을 설계하는 것이다.

## ✨ Key Contributions

본 논문의 가장 중심적인 아이디어는 **Regularized Policy Optimization**을 도입하여 Meta-policy와 Behavior policy 사이의 트레이드오프를 제어하는 것이다. 이를 위해 저자들은 **MerPO (Model-based offline Meta-RL with regularized Policy Optimization)** 프레임워크를 제안한다.

핵심 기여 사항은 다음과 같다:
1. **Meta-Regularized Model-based Actor-Critic (RAC)**: 내측 루프(Within-task) 최적화 단계에서 Behavior policy와 Meta-policy를 가중 보간(Interpolation)한 정규화 항을 사용하여, 어느 한쪽으로 치우치지 않는 안전한 정책 개선을 가능하게 한다.
2. **Model-based 접근법의 활용**: Supervised Meta-learning을 통해 Meta-model을 학습함으로써 효율적인 Task structure inference를 가능하게 하고, 합성 데이터(Synthetic rollouts)를 생성하여 OOD 상태-행동 공간의 안전한 탐색을 돕는다.
3. **이론적 보장**: 제안된 RAC 방법론이 적절한 조건 하에서 Behavior policy와 Meta-policy 모두보다 성능이 향상됨을 이론적으로 증명하여 'Safe Policy Improvement'를 보장한다.

## 📎 Related Works

본 논문은 다음과 같은 기존 연구들의 한계를 지적하며 차별점을 둔다:

1. **Offline Single-task RL**: COMBO와 같은 최신 방법론들은 Model-based 최적화와 Conservative Policy Evaluation (CQL)을 결합하여 성능을 높였다. 하지만 이러한 방식은 단일 작업에 특화되어 있어, 다양한 환경으로의 일반화(Generalization) 능력이 부족하다.
2. **Offline Meta-RL**: FOCAL과 같은 기존 Offline Meta-RL 연구들은 Behavior regularization을 적용하지만, Meta-policy가 Behavior policy보다 품질이 낮을 때 발생하는 성능 저하 문제를 해결하지 못한다. 즉, 데이터셋의 품질이 매우 높을 때 오히려 Meta-policy의 간섭으로 인해 성능이 떨어지는 현상이 발생한다.

MerPO는 이러한 한계를 극복하기 위해 Meta-policy와 Behavior policy를 동시에 고려하는 새로운 정규화 구조를 제안함으로써, 데이터셋 품질에 관계없이 강건한 성능을 낼 수 있도록 설계되었다.

## 🛠️ Methodology

### 1. 전체 시스템 구조
MerPO는 크게 **Offline Meta-model Learning**과 **Offline Meta-policy Learning**의 두 단계로 구성된다.

### 2. Offline Meta-model Learning
먼저, 주어진 오프라인 데이터셋 $\{D^n\}_{n=1}^N$을 사용하여 Meta-model을 학습한다. 이는 Supervised Meta-learning 방식을 따르며, 다음과 같은 목적 함수를 최소화한다:

$$\min_{\phi_m} \mathcal{L}_{\text{model}}(\phi_m) = \mathbb{E}_{M^n} \left\{ \min_{\theta^n} \left[ \mathbb{E}_{(s,a,s') \sim D^n} [\log \hat{T}_{\theta^n}(s'|s,a)] + \eta \|\theta^n - \phi_m\|_2^2 \right] \right\}$$

여기서 $\phi_m$은 Meta-model의 파라미터이며, $\theta^n$은 각 작업별 다이내믹스 모델의 파라미터이다. 이를 통해 새로운 작업이 주어졌을 때 $\phi_m$으로부터 빠르게 Task-specific model $\hat{T}_{\theta^j}$를 적응(Adaptation)시킬 수 있다.

### 3. Offline Meta-policy Learning (RAC)
본 논문의 핵심인 **RAC (Meta-Regularized model-based Actor-Critic)**는 Bi-level optimization 구조를 가진다.

#### (1) 내측 루프 (Within-task Policy Optimization)
각 작업 $n$에 대해, 다음과 같은 정규화된 정책 개선(Regularized Policy Improvement)을 수행한다:

$$\pi^n \leftarrow \arg \max_{\pi^n} \mathbb{E}_{s \sim \rho^n, a \sim \pi^n(\cdot|s)} [\hat{Q}^{\pi^n}(s,a)] - \lambda \alpha D(\pi^n, \pi_{\beta,n}) - \lambda (1-\alpha) D(\pi^n, \pi^c)$$

- $\hat{Q}^{\pi^n}$: Conservative Policy Evaluation (CQL)을 통해 학습된 가치 함수로, OOD 샘플에 대해 보수적인 값을 갖는다.
- $D(\cdot, \cdot)$: 정책 간의 거리 측정 함수 (실제 구현에서는 KL divergence 사용).
- $\pi_{\beta,n}$: Behavior policy (데이터 수집 정책).
- $\pi^c$: Meta-policy (학습된 공통 정책).
- $\alpha \in [0, 1]$: **핵심 하이퍼파라미터**로, Behavior policy에 밀착할지($\alpha \to 1$) 아니면 Meta-policy를 따라 탐색할지($\alpha \to 0$)를 결정하는 가중치이다.

#### (2) 외측 루프 (Meta-policy Update)
내측 루프에서 학습된 각 작업의 정책 $\{\pi^n\}$들을 바탕으로 Meta-policy $\pi^c$를 업데이트한다:

$$\max_{\pi^c} \mathbb{E}_{M^n} \left\{ \mathbb{E}_{s \sim \rho^n, a \sim \pi^n(\cdot|s)} [\hat{Q}^{\pi^n}(s,a)] + \lambda \alpha \mathbb{E}_{(s,a) \sim D^n} [\log \pi^n(a|s)] - \lambda(1-\alpha) D_{KL}(\pi^n || \pi^c) \right\}$$

### 4. 이론적 보장 (Safe Policy Improvement)
저자들은 Theorem 1을 통해, 적절한 $\alpha$ 값과 $\lambda, \beta$ (보수성 계수)를 선택했을 때, 학습된 정책 $\pi^n$이 Behavior policy $\pi_{\beta,n}$과 Meta-policy $\pi^c$ 모두보다 높은 기대 보상을 얻음을 확률적으로 증명하였다. 특히 $\alpha \in (0, 1/2)$ 범위에서 Meta-policy에 적절히 기우는 것이 성능 향상에 유리함을 보였다.

## 📊 Results

### 1. 실험 설정
- **데이터셋 및 환경**: D4RL 벤치마크의 continuous control task (Walker-2D-Params, Half-Cheetah-Fwd-Back, Ant-Fwd-Back, Point-Robot-Wind)를 사용하였다.
- **비교 대상**: FOCAL, MBML, Batch PEARL, CBCQ 및 단일 작업 baseline인 COMBO.
- **측정 지표**: 평균 리턴(Average Return) 및 샘플 효율성(Sample Efficiency).

### 2. 주요 결과
- **RAC의 강건성**: RAC는 Meta-policy의 품질이 낮을 때(예: Random policy)에도 COMBO보다 우수한 성능을 보였으며, Meta-policy의 품질이 높을 때는 성능이 더욱 가속화됨을 확인하였다 (Figure 4).
- **$\alpha$의 영향**: $\alpha=0.4$일 때 가장 안정적이고 높은 성능을 보였으며, 이는 이론적 예측과 일치한다 (Figure 5).
- **MerPO의 전반적 성능**: MerPO-Adp (작업 및 반복 횟수에 따라 $\alpha$를 적응적으로 조절하는 방식)가 기존의 모든 Offline Meta-RL 방법론보다 뛰어난 성능을 기록하였다 (Figure 6).
- **데이터 품질에 따른 성능**: 데이터셋이 매우 정교한 Expert data인 경우, 기존의 FOCAL은 COMBO보다 성능이 낮았으나, MerPO는 COMBO보다 더 높은 성능을 달성하며 일반화 능력을 입증하였다 (Figure 7(d)).

## 🧠 Insights & Discussion

### 1. 강점 및 통찰
- **탐색과 활용의 균형**: 단순히 Meta-policy를 따르는 것이 아니라, Behavior policy를 '안전장치'로 활용함으로써 Offline RL의 고질적인 문제인 가치 과대평가(Value Overestimation)를 효과적으로 억제하였다.
- **Model-based의 이점**: Meta-model을 통한 Task inference와 합성 데이터 생성은 데이터가 부족한 Offline 환경에서 일반화 성능을 높이는 결정적인 역할을 한다.

### 2. 한계 및 비판적 해석
- **$\alpha$ 설정의 민감도**: 이론적으로 $\alpha$의 범위가 제시되었으나, 실제 환경에서는 $\alpha$ 값에 따라 성능 차이가 발생한다. MerPO-Adp가 이를 해결하려 했으나, $\alpha$를 업데이트하는 추가적인 최적화 과정이 필요하므로 계산 복잡도가 증가한다.
- **가정의 의존성**: 이론적 증명에서 사용된 $\nu^n(\rho, f)$와 같은 조건들이 실제 복잡한 신경망 환경에서 항상 성립하는지에 대해서는 추가적인 검증이 필요하다.

## 📌 TL;DR

본 논문은 Offline Meta-RL에서 Meta-policy를 통한 탐색과 Behavior policy를 통한 활용 사이의 균형을 맞추기 위한 **MerPO** 프레임워크를 제안한다. 핵심은 **RAC**라는 정규화된 최적화 기법을 통해 두 정책 사이의 가중 보간을 수행하는 것이며, 이를 통해 데이터셋의 품질에 관계없이 안전한 성능 향상을 보장한다. 실험 결과, 제안 방법론은 기존의 Model-free 및 Model-based Offline Meta-RL 방법론들을 능가하였으며, 특히 고품질 데이터셋에서도 성능 저하 없이 일반화 능력을 유지함을 보였다. 이는 향후 데이터 효율적인 로봇 제어 및 적응형 시스템 연구에 중요한 기여를 할 것으로 보인다.