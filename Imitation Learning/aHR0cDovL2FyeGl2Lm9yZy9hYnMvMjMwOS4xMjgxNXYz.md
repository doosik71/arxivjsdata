# Improving Generalization in Game Agents with Data Augmentation in Imitation Learning

Derek Yadgaroff, Alessandro Sestini, Konrad Tollmar, Ayca Ozcelikkale, Linus Gisslén (2024)

## 🧩 Problem to Solve

본 논문은 게임 플레이 에이전트를 학습시키는 Imitation Learning(IL)에서 발생하는 일반화(Generalization) 문제와 샘플 효율성(Sample Efficiency) 간의 상충 관계를 해결하고자 한다. IL은 전문가의 시연 데이터를 통해 학습하므로 강화학습(RL)에 비해 샘플 효율성이 높지만, 학습 데이터에 포함되지 않은 새로운 시나리오나 환경 변화에 직면했을 때 성능이 급격히 저하되는 분포 변화(Distributional Shift) 문제에 취약하다.

특히 현대의 복잡한 3D 게임 환경에서 충분한 양의 전문가 데이터를 수집하는 것은 시간과 비용 측면에서 매우 비효율적이다. 따라서 적은 양의 데이터만으로도 학습 환경을 넘어 새로운 환경에서 유연하게 동작할 수 있도록 에이전트의 일반화 능력을 향상시키는 것이 본 연구의 핵심 목표이다.

## ✨ Key Contributions

본 연구의 중심 아이디어는 지도 학습(Supervised Learning) 및 강화학습에서 성공적으로 사용된 Data Augmentation(DA) 기법을 특성 기반 상태 공간(Feature-based state space)을 사용하는 IL 에이전트에 적용하는 것이다.

연구의 주요 기여는 다음과 같다.
1. 특성 기반 상태 공간에서 적용 가능한 6가지 데이터 증강 기법의 효과를 종합적으로 분석하였다.
2. 단일 증강뿐만 아니라 여러 증강 기법을 순차적으로 조합했을 때의 성능 변화를 실험적으로 검증하였다.
3. 4가지 서로 다른 테스트 환경을 통해 증강된 모델이 baseline 모델보다 일반화 성능이 뛰어나며, 특정 조합이 일관된 성능 향상을 보인다는 점을 입증하였다.

## 📎 Related Works

기존의 IL 연구는 주로 Behavioral Cloning(BC)이나 Generative Adversarial Imitation Learning(GAIL)을 통해 전문가의 행동을 모방하는 데 집중해 왔다. 그러나 이러한 모델들은 학습 환경에 과적합(Overfitting)되는 경향이 있어, 일반화를 위해 더 많은 시연 데이터나 환경과의 상호작용이 필요하다는 한계가 있다.

강화학습 분야에서는 Domain Randomization이나 Procedural Content Generation을 통해 일반화를 개선하려는 시도가 있었으나, 이는 게임의 코드에 접근 가능하거나 절차적 생성 시스템이 구축되어 있어야 한다는 제약이 있다. 최근 S4RL과 같은 연구가 오프라인 RL의 특성 기반 상태 공간에서 데이터 증강의 효용성을 보였으며, 본 논문은 이러한 직관을 IL 설정으로 확장하여 게임 AI 에이전트의 일반화 문제를 해결하고자 한다.

## 🛠️ Methodology

### 전체 파이프라인 및 알고리즘
본 논문은 가장 단순하고 효과적인 IL 알고리즘인 Behavioral Cloning(BC)을 사용한다. BC는 전문가의 상태-행동 쌍 $(s, a)$ 데이터셋 $D$를 기반으로, 주어진 상태에서 전문가가 취했을 행동을 예측하도록 정책 $\pi_\theta$를 학습시키는 지도 학습 문제로 정의된다. 학습 목표 함수는 다음과 같은 Maximum Entropy objective를 따른다.

$$\arg \max_{\theta} \mathbb{E}_{(s,a) \sim D} [\log \pi_{\theta}(a|s)]$$

이 식은 전문가 데이터셋 $D$에서 샘플링된 $(s, a)$ 쌍에 대해 정책 $\pi_\theta$가 해당 행동 $a$를 선택할 확률을 최대화함으로써 전문가의 행동을 모방하게 한다.

### 데이터 증강 기법 (Data Augmentations)
본 연구에서는 상태 $s$만을 증강하고 행동 $a$는 유지하는 방식을 취한다. 상태 벡터는 연속형(Continuous) 값과 범주형(Categorical) 값으로 나뉘며, 각 값의 특성에 따라 서로 다른 증강 기법을 적용한다.

1. **연속형 값 적용 기법**:
   - **Gaussian Noise**: 상태에 평균 $\mu=0$, 표준편차 $\sigma$인 가우시안 노이즈를 추가한다. $\hat{s}_t = s_t + \epsilon, \epsilon \sim \mathcal{N}(0, \sigma)$.
   - **Uniform Noise**: 일정한 범위 $[-\lambda, \lambda]$ 내의 균등 분포 노이즈를 추가한다. $\hat{s}_t = s_t + \epsilon, \epsilon \sim \mathcal{U}(-\lambda, \lambda)$.
   - **Scaling**: 요소별 곱셈을 통해 상태 값을 스케일링한다. $\hat{s}_t = s_t * \epsilon, \epsilon \sim \mathcal{U}(\alpha, \beta)$.
   - **State-mixup**: 인접한 두 상태를 베타 분포 $\beta(\alpha, \alpha)$ 비율로 선형 보간한다. $\hat{s}_t = \epsilon s_t + (1-\epsilon)s_{t+1}$.
   - **Continuous Dropout**: 연속형 특징 중 무작위로 $n$개를 선택하여 0으로 만든다.

2. **범주형 값 적용 기법**:
   - **Semantic Dropout**: 범주형 특징 중 무작위로 $n$개를 선택하여 0으로 만든다.

증강은 최대 3개까지 순차적으로 적용될 수 있으며(예: $\hat{s}_t = \tau_3(\tau_2(\tau_1(s_t)))$), 데이터의 변형 효과를 최대화하기 위해 '이동 $\rightarrow$ 스케일링 $\rightarrow$ 드롭아웃' 순서로 적용한다.

### 시스템 아키텍처
에이전트의 신경망은 다음과 같은 구조를 가진다.
- **Self-embedding**: 에이전트와 목표 지점 정보를 선형 레이어와 ReLU를 통해 $d=128$ 차원으로 임베딩한다.
- **Entity-embedding**: 주변 엔티티(버튼, 목표 등) 정보를 공유 가중치 선형 레이어로 처리한 후, self-embedding과 결합하여 Transformer Encoder에 입력하고 Average Pooling을 통해 단일 벡터 $x_t$를 생성한다.
- **Local Perception**: $5 \times 5 \times 5$ 크기의 3D semantic map을 3D CNN(3개 레이어, 필터 수 32, 64, 128)에 통과시켜 벡터 $x_M$을 추출한다.
- **Final Output**: $x_t$와 $x_M$을 결합한 후, MLP를 거쳐 최종적으로 9가지 이산 행동에 대한 확률 분포(Softmax)를 출력한다.

## 📊 Results

### 실험 설정
- **환경**: 오픈 월드 도시 시뮬레이션. 에이전트는 건물을 찾아가 버튼을 누르고 문이 닫히기 전 진입해야 한다.
- **데이터셋**: 인간의 시연 데이터 78 에피소드(15,380 샘플).
- **테스트 환경**: 학습 환경에서 건물 위치, 회전, 장애물 수 등을 변경한 4가지 테스트 환경(Test 1~4)을 구축하여 난이도를 'Easy', 'Medium', 'Hard'로 구분하였다.
- **지표**: 제한 시간(750 steps) 내에 목표에 도달했는지 여부인 성공률(Success Rate)을 측정하며, Baseline 대비 상대적 성공률(Relative Success Rate)로 성능을 평가하였다.

### 정량적 결과
- **일반화 성능 향상**: 증강된 모델들은 모든 테스트 환경에서 Baseline보다 높은 성능을 보였다. 평균 상대적 성공률은 환경에 따라 $1.2$배에서 $1.8$배까지 향상되었다.
- **최적의 증강 기법**: 단일 기법 분석 결과, **Scaling**이 가장 일관되게 높은 성능 향상을 보였으며, 그 뒤를 Continuous Dropout과 Gaussian Noise가 이었다. 반면, Semantic Dropout은 오히려 성능을 저하시키는 부정적인 효과를 보였다.
- **조합의 효과**: 최소 2개 이상의 증강 기법을 조합하고 데이터의 80% 이상을 사용한 모델들이 5개 환경(학습 환경 포함) 모두에서 Baseline을 능가하는 일관성을 보였다.

## 🧠 Insights & Discussion

본 연구는 데이터 증강이 IL 에이전트의 샘플 효율성을 유지하면서도 일반화 능력을 유의미하게 향상시킬 수 있음을 보여준다. 특히 단순한 노이즈 추가보다 Scaling이나 특정 특징을 제거하는 Dropout과 같은 기법이 더 효과적이라는 점은, 에이전트가 상태의 절대적인 값보다는 상대적인 관계나 핵심 특징에 집중하게 만들기 때문으로 해석될 수 있다.

하지만 다음과 같은 한계와 논의 사항이 존재한다.
1. **하이퍼파라미터 민감도**: 특히 노이즈 기반 증강의 경우 $\sigma$ 값에 따라 성능 편차가 매우 크게 나타났다. 이는 잘못된 증강이 에이전트를 복구 불가능한 미지의 상태로 밀어 넣을 수 있음을 시사한다.
2. **일관성 vs 최대 성능의 트레이드오프**: 가장 높은 성능을 낸 모델이 반드시 모든 환경에서 일관되게 좋은 성능을 내는 것은 아니었다. 이는 특정 환경에 특화된 증강과 범용적인 증강 사이에 차이가 있음을 의미한다.
3. **과업의 한정성**: 본 실험은 주로 내비게이션과 단순 상호작용 작업에 국한되었다. 더 복잡한 전략적 판단이 필요한 작업에서도 동일한 효과가 나타날지는 추가 연구가 필요하다.

## 📌 TL;DR

본 논문은 특성 기반 상태 공간을 사용하는 Imitation Learning 에이전트의 일반화 문제를 해결하기 위해 6가지 데이터 증강(Data Augmentation) 기법과 그 조합을 체계적으로 분석하였다. 실험 결과, **Scaling**과 같은 특정 증강 기법의 조합이 Baseline 대비 최대 1.8배의 성공률 향상을 가져왔으며, 이는 적은 양의 전문가 데이터만으로도 새로운 게임 환경에 적응할 수 있는 유망한 방법론임을 입증하였다. 이 연구는 향후 게임 AI의 자동 테스트 및 에이전트 제작 공정에서 샘플 효율성을 극대화하는 가이드라인을 제공한다.