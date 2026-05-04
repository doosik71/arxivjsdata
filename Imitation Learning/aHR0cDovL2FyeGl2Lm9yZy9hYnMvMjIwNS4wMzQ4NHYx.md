# Diverse Imitation Learning via Self-Organizing Generative Models

Arash Vahabpour, Tianyi Wang, Qiujing Lu, Omead Pooladzandi, Vwani Roychowdhury (2022)

## 🧩 Problem to Solve

본 논문은 전문가의 시연(demonstrations)으로부터 정책을 복제하는 Imitation Learning(IL)에서, 전문가가 여러 가지 서로 다른 행동 양식(mixture of behaviors, 즉 multimodal behavior)을 보일 때 발생하는 문제를 해결하고자 한다. 

일반적인 Behavior Cloning(BC)은 서로 다른 모드의 행동들을 평균내어 학습하는 경향이 있어 부정확한 정책을 생성하며, Generative Adversarial Imitation Learning(GAIL)과 같은 적대적 학습 기반 방법은 Mode Collapse 현상으로 인해 전문가 행동의 일부 모드만을 학습하는 한계가 있다. 또한, 기존의 잠재 변수(latent variable)를 활용한 접근법들은 복잡한 인코더 구조를 설계해야 하거나, 실제 환경에서 개별 모드를 적절히 구분하여 모방하지 못하는 성능 저하 문제가 존재한다.

따라서 본 연구의 목표는 전문가의 행동 모드가 라벨링되지 않은 상황에서도, 다양한 행동 모드를 정확하게 구분하고 복제할 수 있는 강건한 모방 학습 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 인코더가 없는(encoder-free) 생성 모델인 **Self-Organizing Generative Model (SOG)**을 제안하고, 이를 BC 및 GAIL과 결합하는 것이다.

1. **SOG 알고리즘 제안**: 별도의 인코더 네트워크 없이 잠재 변수 $z$의 샘플 공간에서 직접 최적의 $z$를 탐색하여 데이터의 분포를 학습한다. 이는 계산적으로 효율적이며, 데이터의 의미론적 구조를 잠재 공간에 스스로 조직(self-organize)하는 특성을 가진다.
2. **SOG-BC 및 SOG-GAIL 구현**: SOG를 multimodal BC에 적용하고, 이를 다시 GAIL과 결합하여 BC의 정확성과 GAIL의 강건성(robustness)을 동시에 확보한다.
3. **고차원 잠재 공간 확장**: 잠재 변수의 차원이 높아질 때 발생하는 계산 비용 문제를 해결하기 위해 좌표별 탐색(coordinate-wise search) 방식을 도입하였다.
4. **이론적 분석**: SOG 알고리즘이 이산 및 연속 잠재 변수 상황에서 한계 주변 우도(marginal likelihood)의 최대화와 밀접한 관련이 있음을 수학적으로 증명하였다.

## 📎 Related Works

- **Behavior Cloning (BC)**: 상태-행동 쌍을 지도 학습 방식으로 학습하지만, 궤적 내의 장기적 역학을 무시하여 새로운 상태에 진입했을 때 오류가 누적되는 Compounding Error 문제가 발생한다.
- **GAIL**: 보상 함수를 직접 학습하는 IRL의 비용 문제를 해결하기 위해 적대적 학습을 도입하였으며, BC보다 샘플 효율성과 강건성이 뛰어나다. 그러나 앞서 언급한 Mode Collapse 문제에 취약하다.
- **Multimodal IL 접근법**: 
    - **InfoGAIL / Intention-GAN**: 생성된 궤적과 잠재 코드 사이의 상호 정보량(mutual information)을 최대화하여 모드를 구분하려 하지만, 성능이 제한적이다.
    - **VAE-GAIL**: VAE를 통해 전문가 궤적을 연속적인 잠재 변수로 인코딩하여 입력으로 사용한다. 하지만 시퀀스 데이터를 인코딩하기 위한 재귀적 모듈(recurrent module) 설계가 어렵고, 연속 변수 처리에 국한되는 한계가 있다.

본 논문은 인코더를 완전히 제거하고 잠재 변수를 직접 탐색하는 SOG 방식을 통해 위 연구들의 설계 복잡성과 성능 문제를 해결한다.

## 🛠️ Methodology

### 1. SOG (Self-Organizing Generative Model) 일반 구조
SOG는 입력 $x$와 출력 $y$ 사이의 관계를 학습하며, $y$가 잠재 변수 $z$와 조건부 분포 $p(y|z, x; f)$를 통해 생성된다고 가정한다. 여기서 출력의 분포는 다음과 같은 가우시안 형태를 띤다.
$$p(y|z,x;f) = \mathcal{N}(y; f(z,x), \sigma^2 I)$$
여기서 $f$는 신경망 $f_\theta$이며, $z$는 사전 분포 $p(z)$에서 샘플링된다.

**학습 절차 (Algorithm 1):**
1. **잠재 변수 추정**: 현재 파라미터 $\theta$에서 각 데이터 포인트 $(x_i, y_i)$에 대해 손실 함수 $L$을 최소화하는 최적의 잠재 코드 $z_i^*$를 찾는다.
   $$z_i^* = \arg \min_{z_j \sim p(z)} L(f_\theta(z_j, x_i), y_i)$$
2. **파라미터 업데이트**: 찾아낸 $z_i^*$를 고정한 채, $L(f_\theta(z_i^*, x_i), y_i)$를 최소화하도록 $\theta$를 업데이트한다.

### 2. Multimodal Imitation Learning으로의 적용
- **SOG-BC**: 위 알고리즘을 모방 학습에 적용하되, 하나의 궤적(trajectory) 내에서는 동일한 잠재 코드 $z$가 공유되도록 제약 조건을 추가한다. 이를 통해 궤적 전체의 일관된 행동 모드를 학습한다.
- **SOG-GAIL**: SOG의 정확한 모드 복제 능력과 GAIL의 강건성을 결합한다. 목적 함수는 다음과 같이 BC 손실($L_{SOG}$)과 GAIL의 대리 손실($L_{PPO}$)의 가중 합으로 구성된다.
   $$\text{Objective} = L_{PPO} + \lambda_S L_{SOG}$$
   여기서 $L_{PPO}$는 PPO(Proximal Policy Optimization) 규칙을 사용하여 최적화하며, $L_{SOG}$는 SOG-BC의 손실 함수이다.

### 3. 고차원 잠재 공간을 위한 확장 (Algorithm 4)
잠재 공간의 차원 $d$가 커지면 샘플링해야 할 $z$의 수가 기하급수적으로 증가한다. 이를 해결하기 위해 **Coordinate-Wise Search**를 도입한다. 잠재 변수를 블록 단위($\Delta$)로 나누고, 각 블록별로 최적의 값을 순차적으로 찾아나가는 방식으로 계산 복잡도를 지수 시간에서 선형 시간으로 낮추었다.

## 📊 Results

### 1. 실험 설정
- **데이터셋**: 2D 평면상의 원형 궤적(Circles), MuJoCo 로코모션 작업(Ant, HalfCheetah, Humanoid, Walker2d, Hopper), 로봇 팔 제어(FetchReach).
- **비교 대상**: InfoGAIL, VAE-GAIL.
- **지표**: 누적 보상(Reward), 목표 도달률(Hit Rate), 엔트로피(Entropy), 상호 정보량(Mutual Information).

### 2. 주요 결과
- **정량적 성과 (Table 1, 2)**: 거의 모든 로코모션 작업에서 SOG-BC와 SOG-GAIL이 기존 베이스라인(InfoGAIL, VAE-GAIL)을 압도하는 성능을 보였다. 특히 discrete한 모드 구분 작업에서 월등한 성능 차이를 보였다.
- **정성적 분석**: 
    - SOG-BC와 SOG-GAIL은 전문가의 다양한 행동 모드를 정확하게 분리하여 복제하였다.
    - FetchReach 실험에서 잠재 공간의 보간(interpolation)이 의미론적으로 매끄럽게 이루어짐을 확인하였다.
- **강건성 분석**: SOG-BC는 학습 데이터에 없는 상태(unseen states)나 외부 섭동(perturbation)이 발생했을 때 쉽게 무너지는 경향이 있으나, SOG-GAIL은 이러한 상황에서도 안정적으로 동작하였다 (Figure 6, 7).

## 🧠 Insights & Discussion

### 강점 및 분석
- **모드 붕괴 해결**: 기존 GAIL 기반 모델들이 적대적 학습 과정에서 일부 모드만 학습하는 것과 달리, SOG는 BC 단계를 통해 모든 모드를 명시적으로 탐색하고 학습한 뒤 이를 GAIL과 결합함으로써 Mode Collapse 문제를 효과적으로 해결하였다.
- **인코더 제거의 이점**: 복잡한 VAE 인코더를 설계할 필요 없이 샘플링 기반의 탐색만으로도 충분히 정교한 잠재 공간을 구축할 수 있음을 보여주었다.
- **이론적 정당성**: SOG가 단순한 휴리스틱이 아니라, $\sigma \to 0$인 극한 상황에서 Hard-EM 알고리즘 및 주변 우도 최대화(MLE)와 일치한다는 점을 증명하여 수학적 근거를 마련하였다.

### 한계 및 논의
- **샘플 효율성**: 고차원 공간에서 좌표별 탐색을 도입했음에도 불구하고, 여전히 많은 양의 샘플링이 필요할 수 있다.
- **가정**: 본 논문은 $f$가 립시츠 연속(Lipschitz continuous)하다고 가정하여 self-organization 효과를 설명하였으나, 실제 매우 복잡한 신경망에서 이 가정이 어느 정도까지 유지되는지에 대한 추가 논의가 필요할 수 있다.

## 📌 TL;DR

본 논문은 전문가의 다양한 행동 모드를 학습하기 위해 인코더가 없는 생성 모델인 **SOG(Self-Organizing Generative Model)**를 제안한다. SOG는 잠재 변수를 직접 탐색하여 데이터의 모드를 구분하며, 이를 GAIL과 결합한 **SOG-GAIL**은 multimodal 행동 복제의 정확성과 미학습 상태에 대한 강건성을 동시에 달성하였다. 이 연구는 특히 복잡한 로봇 제어 및 로코모션 작업에서 기존의 VAE-GAIL이나 InfoGAIL보다 훨씬 뛰어난 성능을 보이며, 향후 복잡한 다목적 행동 모방 학습의 새로운 방향성을 제시한다.