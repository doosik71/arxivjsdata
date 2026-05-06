# On Zero-Shot Reinforcement Learning

Scott Jeen (2024)

## 🧩 Problem to Solve

본 논문은 강화학습(Reinforcement Learning, RL) 에이전트가 학습 단계에서 접하지 못한 새로운 작업이나 도메인에 대해 추가적인 연습 없이 즉각적으로 적응하여 해결해야 하는 Zero-Shot Reinforcement Learning의 문제를 다룬다. 현대의 RL 시스템은 시뮬레이션 환경에서 인간을 능가하는 성능을 보여주지만, 실제 세계(Real-world)에 적용하기 위해서는 다음과 같은 치명적인 한계가 존재한다.

첫째, 실제 세계에서는 데이터 생성 비용이 매우 높기 때문에 완벽한 시뮬레이터를 구축하는 것이 불가능에 가깝다. 학습된 시뮬레이터를 사용할 수 있으나, 이는 항상 근사치일 뿐이며 훈련 데이터 분포 밖(Out-of-Distribution, OOD)의 영역에서는 병리적으로 잘못된 예측을 할 가능성이 크다. 둘째, 시뮬레이션과 실제 환경 사이의 미세한 불일치(Misalignment)만으로도 기존 RL 에이전트는 성능이 급격히 저하되거나 환경의 허점을 이용하는 '해킹' 행동을 보일 수 있다.

따라서 저자는 실제 세계의 문제를 해결하기 위해 반드시 극복해야 할 세 가지 제약 조건을 정의한다.

1. **데이터 품질 제약(Data Quality Constraint):** 실제 데이터셋은 규모가 작고 동질적(Homogeneous)이다.
2. **관측 가능성 제약(Observability Constraint):** 상태, 역학(Dynamics), 보상이 부분적으로만 관측(Partially Observed)되는 경우가 많다.
3. **데이터 가용성 제약(Data Availability Constraint):** 사전 데이터에 항상 접근할 수 있다는 보장이 없다.

본 연구의 목표는 이러한 세 가지 제약 조건 하에서도 새로운 작업을 Zero-Shot으로 해결할 수 있는 RL 방법론을 제안하고 검증하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 기존의 Foundation Policy 접근법인 Forward-Backward (FB) Representations와 Universal Successor Features (USF)를 확장하여 실제 세계의 제약 조건을 해결하는 것이다.

1. **보수적 Zero-Shot RL (Conservative Zero-Shot RL):** 데이터 품질이 낮을 때 발생하는 OOD 상태-행동 가치 과대평가(Value Overestimation) 문제를 해결하기 위해, Offline RL의 보수적 정규화(Conservative Regularization) 기법을 Zero-Shot 설정에 도입하였다.
2. **메모리 기반 Zero-Shot RL (Memory-based Zero-Shot RL):** 부분 관측 가능성(Partial Observability)으로 인해 발생하는 상태 및 작업 오인식(Misidentification) 문제를 해결하기 위해, GRU와 같은 메모리 모델을 결합하여 궤적(Trajectory) 정보를 압축해 사용하는 구조를 제안하였다.
3. **PEARL (Probabilistic Emission-Abating Reinforcement Learning):** 사전 데이터가 전혀 없는 상태에서 실시간 시스템 식별(System ID)과 확률적 앙상블 모델, MPPI 플래닝을 결합하여 실제 건물의 탄소 배출을 줄이는 Zero-Shot 제어 시스템을 구현하였다.

## 📎 Related Works

본 논문은 다음과 같은 관련 연구들을 기반으로 하며, 그 한계를 지적한다.

- **Offline RL:** 데이터셋으로부터 정책을 학습하는 분야로, CQL(Conservative Q-Learning)이나 IQL(Implicit Q-Learning) 등이 대표적이다. 하지만 대부분 단일 작업(Single-task)에 집중하며, 새로운 작업에 대한 Zero-Shot 일반화 능력은 부족하다.
- **Successor Representations & Features:** 미래의 상태 방문 확률이나 특징의 합을 학습하여 보상 함수가 바뀌어도 빠르게 적응하는 방법론이다. 최근의 FB Representations와 USF는 이를 일반화하여 다양한 작업을 처리할 수 있는 Foundation Policy의 가능성을 보여주었으나, 이들은 주로 고품질의 탐색적 데이터셋(Exploratory Datasets)이 있다는 가정하에 작동한다.
- **Sim-to-Real & Domain Randomization:** 시뮬레이션과 실제의 간극을 줄이기 위해 환경 파라미터를 무작위화하는 기법이다. 그러나 이는 여전히 어느 정도 정확한 시뮬레이터가 존재해야 한다는 전제가 필요하므로, 시뮬레이터 구축 자체가 불가능한 실제 문제에는 적용하기 어렵다.

## 🛠️ Methodology

본 논문은 세 가지 제약 조건을 해결하기 위해 각각 다른 방법론을 제시한다.

### 1. 데이터 품질 해결: VC-FB 및 MC-FB

데이터가 작고 동질적일 때, 에이전트는 데이터셋에 없는 행동(OOD action)의 가치를 과대평가하는 경향이 있다. 이를 막기 위해 저자는 CQL의 아이디어를 FB 구조에 이식하였다.

- **Value-Conservative FB (VC-FB):** 모든 작업 벡터 $z$에 대해 OOD 행동의 가치 $F(s, a, z)^\top z$를 억제한다. 손실 함수는 다음과 같다.
$$L_{VC-FB} = \alpha \cdot \left( \mathbb{E}_{s \sim D, a \sim \mu, z \sim Z}[F(s, a, z)^\top z] - \mathbb{E}_{(s, a) \sim D, z \sim Z}[F(s, a, z)^\top z] - H(\mu) \right) + L_{FB}$$
여기서 $\mu$는 가치를 최대화하는 행동 분포이며, $\alpha$는 보수적 패널티의 강도를 조절하는 하이퍼파라미터이다.

- **Measure-Conservative FB (MC-FB):** 가치 대신 후속 측정치(Successor Measure) 자체를 억제하여, OOD 행동이 미래의 어떤 상태로도 연결되지 않을 가능성이 높다고 가정한다.

### 2. 관측 가능성 해결: FB-M (FB with Memory)

상태 $s$를 직접 관측할 수 없고 부분적인 관측치 $o$만 주어지는 POMDP 상황에서는 메모리 모델 $f$를 도입하여 과거 궤적 $\tau_L$을 은닉 상태(Hidden State)로 압축한다.

- **구조:** Forward 모델 $F$, Backward 모델 $B$, 정책 $\pi$가 각각 독립적인 메모리 모델(GRU)을 가진다.
- **작동 방식:** 궤적 $\tau_L$을 입력받아 생성된 은닉 상태를 기반으로 후속 측정치를 근사한다. 이를 통해 '상태 오인식'과 '작업 오인식'을 동시에 해결한다.

### 3. 데이터 가용성 해결: PEARL

사전 데이터가 없는 상태에서 건물 에너지 제어를 수행하기 위해 다음과 같은 파이프라인을 구축하였다.

- **System ID (Commissioning):** 초기 180분 동안 Maximum Variance (MV) 탐색을 수행한다. 모델의 예측 분산 $\text{Var}[\cdot]$이 가장 높은 영역을 탐색하여 정보 획득을 최대화한다.
- **Prediction:** 확률적 심층 신경망(Probabilistic DNN)의 앙상블을 사용하여 시스템 역학 $\tilde{P}$를 모델링하며, 이는 평균 $\mu$와 공분산 $\Sigma$를 출력하는 가우시안 분포로 표현된다.
- **Control:** MPPI(Model Predictive Path Integral) 플래닝을 사용하여 예측된 미래 보상을 최대화하는 최적의 행동 시퀀스를 샘플링하고 실행한다.

## 📊 Results

### 1. 보수적 Zero-Shot RL 실험 (ExORL, D4RL)

- **결과:** VC-FB와 MC-FB는 저품질 데이터셋에서 기본 FB보다 훨씬 높은 성능을 보였다. 특히 VC-FB는 일부 벤치마크에서 단일 작업 전용으로 학습된 CQL보다도 높은 성능을 기록하며, 멀티태스크 에이전트가 단일 태스크 에이전트를 능가할 수 있음을 보여주었다.
- **데이터 크기 영향:** 데이터셋의 크기가 작을수록 보수적 정규화의 효과가 극대화되었으며, 데이터가 충분할 때는 기본 FB와 성능 차이가 줄어들었다.

### 2. 메모리 기반 RL 실험 (ExORL POMDP)

- **결과:** 상태에 노이즈가 섞이거나 일부가 누락된(Flickering) 환경에서 FB-M은 메모리가 없는 FB나 단순 스태킹(Stacking) 방식보다 월등한 성능을 보였다. 특히 Quadruped 환경에서는 오라클(Oracle) 성능에 근접하는 결과를 냈다.

### 3. PEARL 실적용 실험 (Energym)

- **결과:** RBC(Rule-Based Controller) 대비 Mixed-Use 건물에서 탄소 배출량을 최대 **31.46%** 감소시켰다.
- **특이사항:** 전력망의 탄소 집약도(Carbon Intensity)가 높은 시간대에 전력 소비를 줄이는 'Load Shifting' 능력을 보여주었으며, 실내 온도 유지(Thermal Comfort) 제약 조건을 충족하면서도 효율적인 제어가 가능함을 입증하였다.

## 🧠 Insights & Discussion

본 논문은 단순한 알고리즘 제안을 넘어 RL의 실용적 배치를 위한 철학적 논의를 제공한다. 특히 **"세상을 얼마나 정확하게 시뮬레이션할 수 있는가"**에 대한 두 가지 가설을 제시한다.

- **Big World Hypothesis:** 세상은 너무 크고 복잡하여 완전한 모델링이 불가능하다는 관점이다. 이 가설이 맞다면, 본 논문에서 제안한 것과 같은 근사 모델 기반의 적응형 제어와 지속적 학습(Continual Learning)이 계속해서 중요할 것이다.
- **Platonic Representation Hypothesis:** 모델의 규모가 커짐에 따라 서로 다른 네트워크들이 현실의 공통된 통계적 모델로 수렴한다는 관점이다. 이 가설이 맞다면, 매우 거대한 Foundation Policy가 결국 현실의 물리 법칙을 내재화하여 완전한 Zero-Shot 전이가 가능해질 것이다.

저자는 현재의 기술 수준에서는 세상이 여전히 '크기(Big World)' 때문에, 시뮬레이션과 실제의 간극을 메우는 보수적 접근과 메모리 기반의 적응 능력이 필수적이라고 주장한다.

## 📌 TL;DR

이 논문은 실제 세계의 세 가지 제약(데이터 품질, 관측 가능성, 데이터 가용성)을 해결하기 위한 **Zero-Shot RL 프레임워크**를 제안한다.

1. **VC-FB/MC-FB**를 통해 저품질 데이터에서의 가치 과대평가 문제를 해결하고,
2. **FB-M**을 통해 부분 관측 환경에서의 적응력을 높였으며,
3. **PEARL**을 통해 사전 데이터 없이도 실시간 학습을 통해 건물 탄소 배출을 30% 절감하는 성과를 거두었다.
결과적으로 본 연구는 시뮬레이션에 의존하지 않고 실제 환경에 즉각 적용 가능한 RL 에이전트 설계의 방향성을 제시한다.
