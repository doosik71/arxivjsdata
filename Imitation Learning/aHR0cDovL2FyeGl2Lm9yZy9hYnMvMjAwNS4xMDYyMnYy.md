# Triple-GAIL: A Multi-Modal Imitation Learning Framework with Generative Adversarial Nets

Cong Fei, Bin Wang, Yuzheng Zhuang, Zongzhang Zhang, Jianye Hao, Hongbo Zhang, Xuewu Ji, and Wulong Liu (2020)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 Imitation Learning(모방 학습), 특히 Generative Adversarial Imitation Learning(GAIL)이 가진 **단일 모달리티(Single Modality) 가정의 한계**이다.

실제 세계의 전문가 시연(Expert Demonstrations)은 하나의 일관된 방식이 아니라, 상황에 따라 다양한 기술(Skill)이나 습관이 섞여 있는 Multi-modal 특성을 가진다. 예를 들어, 자율주행 상황에서 운전자는 교통 상황에 따라 '차선 유지', '좌측 차선 변경', '우측 차선 변경'이라는 서로 다른 의도(Intention)를 가지고 행동한다.

기존의 GAIL과 같은 알고리즘은 이러한 모드 변이를 구분하지 못하므로, 여러 모달리티가 섞인 데이터를 학습할 때 **Mode Collapse(모드 붕괴)** 문제가 발생하여 평균적인 행동만을 생성하거나 부적절한 행동을 출력하는 경향이 있다. 또한, 기존의 Multi-modal 확장 연구들은 잠재 변수를 무작위로 샘플링하거나 고정된 레이블을 사용하기 때문에, 환경 상황에 맞게 적응적으로 기술을 선택(Adaptive Skill Selection)해야 하는 실제 시나리오를 해결하는 데 한계가 있다.

따라서 본 논문의 목표는 전문가의 시연 데이터로부터 **적절한 기술 선택(Skill Selection)과 그에 따른 행동 모방(Imitation)을 동시에 학습**할 수 있는 프레임워크를 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 GAIL 구조에 보조적인 **Selector(선택기)**를 도입하여, 상태-행동 쌍으로부터 어떤 기술이 사용되었는지를 추론하고, 그 기술 레이블을 바탕으로 다시 행동을 생성하는 상호 보완적 구조를 설계한 것이다.

주요 기여 사항은 다음과 같다.

1. **Triple-GAIL 프레임워크 제안**: Generator(생성기), Selector(선택기), Discriminator(판별기)라는 세 개의 플레이어가 참여하는 새로운 Adversarial Game 프레임워크를 제안하였다.
2. **이론적 수렴 보장**: Generator와 Selector가 각각 자신의 최적점(Optima)으로 수렴한다는 이론적 보장을 제공함으로써, 학습의 안정성과 타당성을 입증하였다.
3. **적응적 기술 선택 가능**: 단순히 레이블을 구분하는 것을 넘어, 현재 환경 상태에 따라 어떤 기술을 사용할지 결정하고 그에 맞는 정책을 수행하는 End-to-End 학습을 구현하였다.

## 📎 Related Works

### 1. Generative Adversarial Imitation Learning (GAIL)

GAIL은 전문가의 상태-행동 분포를 모방하도록 Generator를 학습시키고, Discriminator를 통해 생성된 데이터와 전문가 데이터를 구분함으로써 보상 함수(Reward Function)를 명시적으로 설계하지 않고도 정책을 최적화한다. 하지만 단일 모달리티 가정 하에 설계되어 Multi-modal 데이터셋에서 성능이 저하된다.

### 2. Multi-modal Imitation Learning

- **비지도 학습 방식 (InfoGAIL, Burn-InfoGAIL, VAE-GAIL)**: 잠재 코드(Latent codes)를 통해 모달리티를 구분하려 하지만, 레이블이 없기 때문에 시맨틱(Semantic) 정보나 태스크의 맥락(Context)을 충분히 반영하지 못하는 한계가 있다.
- **지도 학습 방식 (CGAIL, ACGAIL)**: 전문가 데이터의 레이블을 직접 사용하거나 보조 분류기를 도입한다. 그러나 대부분의 경우 레이블을 무작위로 샘플링하여 학습하며, 환경 상황에 맞춰 적응적으로 레이블을 선택하는 능력은 부족하다.

Triple-GAIL은 이러한 기존 방식들과 달리, 전문가 시연 데이터와 실시간으로 생성되는 경험 데이터를 모두 사용하여 기술 선택과 행동 모방을 **결합 최적화(Joint Optimization)** 한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. 시스템 구조 및 구성 요소

Triple-GAIL은 Figure 1과 같이 세 개의 신경망으로 구성된다.

- **Selector ($C_\alpha$):** 현재 상태 $s$와 행동 $a$가 주어졌을 때, 해당 행동이 어떤 기술 레이블 $c$에 해당하는지를 출력하는 조건부 분포 $p_{C_\alpha}(c|s, a)$를 학습한다.
- **Generator ($\pi_\theta$):** 현재 상태 $s$와 선택된 기술 레이블 $c$가 주어졌을 때, 최적의 행동 $a$를 출력하는 조건부 정책 $p_{\pi_\theta}(a|s, c)$를 학습한다.
- **Discriminator ($D_\psi$):** 입력된 $(s, a, c)$ 트리플(Triple)이 전문가의 시연 데이터에서 온 것인지, 아니면 Generator나 Selector에 의해 생성된 가짜 데이터인지 판별한다.

### 2. 학습 목표 및 손실 함수

전체 시스템은 다음과 같은 Min-Max 게임으로 정식화된다.

$$ \min_{\alpha, \theta} \max_{\psi} \mathbb{E}_{\pi_E}[\log(1-D_\psi(s,a,c))] + \omega \mathbb{E}_{\pi_\theta}[\log D_\psi(s,a,c)] + (1-\omega) \mathbb{E}_{C_\alpha}[\log D_\psi(s,a,c)] - \lambda_H H(\pi_\theta) $$

여기서 $\omega$는 생성기와 선택기의 가중치를 조절하는 하이퍼파라미터이며, $H(\pi_\theta)$는 정책의 엔트로피로 조기 수렴을 방지하는 정규화 항이다.

단순히 Adversarial Loss만으로는 Generator와 Selector가 각각 독립적으로 전문가 분포에 수렴한다는 보장이 없기 때문에, 본 논문은 두 가지 **Cross-Entropy 항($R_E, R_G$)**을 추가하여 목적 함수를 확장하였다.

- **$R_E$ (Expert Supervised Loss):** 전문가 데이터 $(s^e, a^e, c^e)$를 사용하여 Selector가 전문가의 레이블을 정확히 맞추도록 강제한다.
  $$ R_E = \mathbb{E}_{\pi_E}[-\log p_{C_\alpha}(c|s, a)] $$
- **$R_G$ (Generator-based Loss):** Generator가 생성한 데이터 $(s^g, a^g, c^g)$를 사용하여 Selector를 학습시킨다. 이는 일종의 데이터 증강(Data Augmentation) 효과를 주어 Selector의 강건함을 높인다.
  $$ R_G = \mathbb{E}_{\pi_\theta}[-\log p_{C_\alpha}(c|s, a)] $$

최종 목적 함수는 식 (5)와 같이 Adversarial Loss와 $R_E, R_G$, 그리고 엔트로피 항의 합으로 구성된다.

### 3. 학습 절차 (Algorithm 1)

1. 전문가 시연 데이터에서 고정된 레이블 $c_j$를 추출하여 환경을 초기화한다.
2. Generator $\pi_\theta(\cdot|c_j)$를 실행하여 궤적(Trajectory)을 샘플링한다.
3. 샘플링된 상태-행동 쌍을 Selector $C_\alpha$에 입력하여 가짜 레이블 $c^c$를 생성한다.
4. 전문가 데이터, Generator 생성 데이터, Selector 생성 데이터를 모두 Discriminator $D_\psi$에 전달한다.
5. **Discriminator 업데이트**: Gradient Ascent를 통해 전문가 데이터를 구분하도록 학습한다.
6. **Selector 업데이트**: Gradient Descent를 통해 $D_\psi$를 속이고 $R_E, R_G$를 최소화하도록 학습한다.
7. **Generator 업데이트**: TRPO(Trust Region Policy Optimization)를 사용하여 $D_\psi$가 제공하는 보상($r = -\log D_\psi$)을 최대화하도록 업데이트한다.

## 📊 Results

### 1. 실험 설정

- **Driving Task**: NGSIM I-80 데이터셋을 사용하며, '차선 변경(좌/우)' 및 '차선 유지'의 3가지 기술을 학습한다.
- **RTS Game**: Mini-RTS 게임에서 서로 다른 전략을 가진 built-in 에이전트(SIMPLE, HIT-N-RUN)를 상대로 승률을 측정한다.
- **비교 대상**: Behavioral Cloning (BC), GAIL, CGAIL (분류기 고정 버전).

### 2. 주요 결과

#### (1) 자율주행 실험 (Driving Task)

- **정량적 지표**: Success Rate, Mean Distance, KL Divergence를 측정하였다.
- **결과**: Triple-GAIL이 모든 지표에서 BC, GAIL, CGAIL보다 우수한 성능을 보였으며, 전문가의 행동에 가장 근접하였다. (Table I)
- **정성적 분석**: 시각화 결과(Figure 2), BC는 충돌로 인해 궤적이 짧고, GAIL은 모드 붕괴로 인해 기술 구분이 모호했으나, Triple-GAIL은 세 가지 주행 기술이 명확하게 구분되어 나타났다.
- **Ablation Study**: $R_E$와 $R_G$를 제거했을 때 Selector의 정확도가 크게 하락함을 확인하여, 지도 학습 신호와 데이터 증강의 중요성을 입증하였다. (Table II)

#### (2) RTS 게임 실험 (RTS Game)

- **결과**: Triple-GAIL이 SIMPLE 및 HIT-N-RUN 에이전트 모두를 상대로 가장 높은 승률을 기록하였다. (Table III)
- **분석**: 적의 전술적 의도를 정확히 파악하고 그에 맞는 대응 정책을 선택하는 능력이 뛰어남을 확인하였다. 특히, 레이블을 직접 제공한 경우(Triple-GAIL+label)와 비교했을 때도 joint optimization의 효과로 인해 높은 성능을 보였다.

## 🧠 Insights & Discussion

### 강점

- **적응적 제어 가능**: 기존 연구들이 단순히 모달리티를 '구분'하는 데 그쳤다면, 본 논문은 환경 상태에 따라 어떤 기술을 '선택'하고 실행할 것인지에 대한 메커니즘을 성공적으로 구축하였다.
- **이론적 뒷받침**: Triple-GAN의 구조를 차용하여 Generator와 Selector의 수렴성을 이론적으로 증명함으로써, 복잡한 세 플레이어 게임의 안정성을 확보하였다.
- **데이터 효율성**: $R_G$를 통한 데이터 증강 기법을 도입하여, 한정된 전문가 데이터 상황에서도 Selector의 성능을 높일 수 있었다.

### 한계 및 논의사항

- **레이블 의존성**: 학습 초기 단계에서 전문가 데이터의 레이블($R_E$)이 매우 중요한 역할을 한다. 만약 전문가 데이터에 정확한 레이블이 없는 완전 비지도 환경이라면 본 프레임워크를 그대로 적용하기 어려울 것이다.
- **계산 복잡도**: 세 개의 네트워크를 동시에 학습시키고 TRPO를 사용하므로, 단일 모델 학습에 비해 계산 비용과 메모리 사용량이 증가할 가능성이 있다.
- **가정 사항**: 본 논문은 $p(s, c)$와 $p(s, a)$를 각각 시연 데이터와 생성 데이터로부터 얻을 수 있다고 가정하고 있으나, 실제 환경에서 이 분포의 괴리가 클 경우 수렴 속도에 영향을 줄 수 있다.

## 📌 TL;DR

본 논문은 Multi-modal 전문가 데이터를 모방할 때 발생하는 **Mode Collapse 문제**와 **적응적 기술 선택의 어려움**을 해결하기 위해 **Triple-GAIL** 프레임워크를 제안한다. Generator, Selector, Discriminator의 세 가지 네트워크를 동시에 학습시켜, **"상황에 맞는 기술 선택 $\to$ 해당 기술의 행동 생성"** 프로세스를 통합적으로 최적화한다. 자율주행 및 RTS 게임 실험을 통해 기존 GAIL 기반 방법론보다 훨씬 정교하고 적응적인 모방 학습이 가능함을 입증하였으며, 이는 복잡한 인간의 행동 패턴을 학습해야 하는 실제 로봇 제어나 자율주행 시스템에 중요하게 적용될 수 있다.
