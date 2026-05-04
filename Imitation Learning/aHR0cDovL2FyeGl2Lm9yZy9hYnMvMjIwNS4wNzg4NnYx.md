# An Empirical Investigation of Representation Learning for Imitation

Xin Chen, Sam Toyer, Cody Wild, Scott Emmons, Ian Fischer, Kuang-Huei Lee, Neel Alex, Steven Wang, Ping Luo, Stuart Russell, Pieter Abbeel, Rohin Shah (2021)

## 🧩 Problem to Solve

본 논문은 모방 학습(Imitation Learning, IL)에서 전문가의 시연(demonstration) 데이터를 수집하는 데 드는 높은 비용 문제를 해결하고자 한다. 일반적으로 에이전트가 배포 환경에서 마주할 수 있는 다양한 상황을 처리하기 위해서는 방대한 양의 시연 데이터셋이 필요하지만, 이를 실제로 수집하는 것은 매우 어렵고 비용이 많이 든다.

컴퓨터 비전, 강화 학습(RL), 그리고 자연어 처리(NLP) 분야에서는 보조적인 표현 학습(Representation Learning, RepL) 목적 함수를 사용하여 적은 양의 태스크 특화 데이터로도 높은 성능을 내는 연구들이 진행되어 왔다. 이에 본 연구는 이러한 표현 학습의 이점이 모방 학습 분야에도 동일하게 적용되는지, 그리고 어떤 설계 선택이 성능에 영향을 미치는지 체계적으로 조사하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 모방 학습을 위한 표현 학습의 효용성을 검증하기 위해 **EIRLI(Empirical Investigation of Representation Learning for Imitation)**라는 모듈형 프레임워크를 제안하고 이를 통해 광범위한 실험적 분석을 수행한 것이다.

중심적인 아이디어는 표현 학습 알고리즘의 설계 결정을 여러 축(target selection, loss type, augmentation, network architecture 등)으로 분해하여 모듈화하고, 이를 통해 다양한 RepL 방법론들이 Behavioral Cloning(BC) 및 Generative Adversarial Imitation Learning(GAIL)의 성능을 실제로 향상시키는지 정량적으로 평가하는 것이다. 결과적으로 저자들은 정교한 RepL 기법보다 잘 튜닝된 이미지 증강(image augmentation) 기법이 더 효과적일 수 있다는 통찰을 제시한다.

## 📎 Related Works

논문은 이미지 분류를 위한 표현 학습(예: SimCLR, MoCo)과 강화 학습을 위한 표현 학습(예: CURL, PI-SAC) 연구들을 언급한다. 기존 연구들에 따르면, 특히 강화 학습 분야에서는 복잡한 표현 학습 알고리즘보다 단순한 이미지 증강만으로도 유사하거나 더 나은 성능을 얻을 수 있다는 결과가 보고된 바 있다.

본 연구는 이러한 기존의 흐름을 모방 학습 설정으로 확장하여, 이미지 기반 환경에서 다양한 RepL 방법론을 체계적으로 비교 분석한다는 점에서 차별점을 가진다. 특히 단순히 성능을 비교하는 것에 그치지 않고, 왜 특정 환경에서 RepL이 효과가 없는지를 분석하기 위해 t-SNE 시각화 및 Saliency map과 같은 분석 도구를 사용한다.

## 🛠️ Methodology

### 1. 모듈형 표현 학습 프레임워크
저자들은 RepL 알고리즘을 다음과 같은 설계 축으로 정의하여 구성한다.

- **Target Selection**: 표현을 계산할 입력인 'context' $x$와 이와 관계를 맺을 'target' $y$를 정의한다. 이미지 분류에서는 주로 동일 이미지의 변형을 사용하지만, 순차적 의사결정 문제에서는 미래 상태 $o_{t+k}$나 보상 $r_{t+k}$를 target으로 설정하여 예측 정보를 학습하도록 유도할 수 있다.
- **Loss Type**: 
    - **Reconstruction**: target $y$를 표현 $z$로부터 복원하는 방식이다. (예: VAE)
    - **Contrast**: context-target 쌍 $(x, y)$의 표현 $z$와 $z'$는 가깝게, 다른 쌍과는 멀게 만든다. 주로 InfoNCE 손실 함수를 사용한다.
    $$L_{\text{InfoNCE}} = \mathbb{E} \left[ \log \frac{e^{f(x_i, y_i)}}{\sum_{j=1}^K e^{f(x_i, y_j)}} \right]$$
    - **Bootstrapping**: 타겟 인코더의 그래디언트 전파를 차단하여 붕괴(collapse)를 방지하며 타겟 표현을 예측한다.
    - **Consistency**: 동일 입력의 서로 다른 변형에 대해 유사한 분포를 출력하도록 강제한다.
    - **Compression**: downstream task에 필요한 최소한의 정보만 추출하도록 제한하며, Conditional Entropy Bottleneck(CEB) 등을 통해 $I(X; Z|Y)$를 최소화한다.
- **Augmentation**: 인코더에 입력되기 전 프레임을 변형하여 일반화 성능을 높인다.
- **Neural Network**: 
    - **Encoder**: 입력을 latent representation $z$로 매핑한다. Momentum encoder를 사용하여 타겟 인코더의 가중치를 천천히 업데이트함으로써 학습 안정성을 높이기도 한다.
    - **Decoder**: 학습 시에만 사용되며, 이미지 복원, 프로젝션 헤드(projection heads), 액션 컨디셔닝(action conditioning) 등의 역할을 수행한다.

### 2. 학습 절차 (Pipeline)
표현 학습을 모방 학습에 통합하는 방식은 두 가지로 나뉜다.
- **Pretraining**: 먼저 RepL 목적 함수로 인코더를 학습시킨 후, 마지막 레이어만 남기고 IL(BC 등)로 전체 네트워크를 미세 조정(fine-tuning)한다.
- **Joint Training**: IL 학습 과정 중에 RepL 목적 함수를 보조 손실(auxiliary loss)로 추가하여 동시에 학습한다.

### 3. 모방 학습 알고리즘
- **Behavioral Cloning (BC)**: 전문가 데이터셋 $D$에 대해 다음과 같은 로그 가능도 최대화 문제를 푼다.
$$L(\theta) = \mathbb{E}_{(x, a) \sim D} [\log \pi_\theta(a|x)]$$
- **GAIL**: 정책 $\pi_\theta$와 판별자 $D_\psi$ 간의 적대적 게임으로 정의하며, 판별자가 전문가의 행동과 정책의 행동을 구분하지 못하도록 학습한다.

## 📊 Results

### 1. 실험 설정
- **데이터셋 및 작업**: DMC (cheetah-run, finger-spin, reacher-easy), Procgen (CoinRun, Fruitbot, Jumper, Miner), MAGICAL (MoveToRegion, MoveToCorner, MatchRegions) 등 총 10개 작업에서 평가하였다.
- **비교 대상**: 단순 BC (증강 없음), BC + 이미지 증강(Baseline), 그리고 다양한 RepL 조합(VAE, SimCLR, TemporalCPC 등).

### 2. 주요 결과
- **RepL의 제한적 효용성**: 대부분의 환경에서 RepL 방법론들이 vanilla BC보다는 성능이 좋았으나, **잘 튜닝된 이미지 증강(BC + Aug)을 사용한 베이스라인보다는 유의미하게 높지 않았다.** 
- **증강의 지배적 영향**: 이미지 증강의 적용 여부가 RepL 알고리즘의 선택보다 성능에 훨씬 더 큰 영향을 미쳤다. 일부 작업(reacher-easy, Fruitbot 등)에서는 증강만으로 리워드가 150% 이상 증가했다.
- **태스크별 반응 차이**: 증강에 대한 반응은 양극단으로 나타났다. 예를 들어 `finger-spin` 작업에서는 오히려 증강을 적용했을 때 성능이 하락했는데, 이는 환경 내 객체가 고정되어 있어 회전 증강이 실제 신호와 혼동을 일으키기 때문으로 분석된다.
- **GAIL 결과**: GAIL 역시 BC와 유사한 경향을 보였으며, 특히 판별자(discriminator)에 대한 증강이 성능 확보에 필수적임을 확인하였다.

## 🧠 Insights & Discussion

### 1. 이미지 분류 vs 모방 학습 데이터의 차이
저자들은 왜 SimCLR와 같은 RepL 기법이 이미지 분류(STL-10)에서는 성공적이었으나 모방 학습에서는 효과가 적었는지 분석한다.
- **시각적 변이의 세밀함**: 이미지 분류는 클래스 간 시각적 차이(예: 하늘색 배경의 비행기 vs 털이 있는 사슴)가 매우 뚜렷하다. 반면, 모방 학습(MAGICAL, Procgen)에서는 액션 결정이 배경이나 전반적인 색상 같은 거시적 특징이 아니라, 매우 세밀하고 국소적인 큐(fine-grained local cues)에 의해 결정된다.
- **표현의 불일치**: RepL 알고리즘들은 주로 시각적으로 가장 두드러진(salient) 특징을 학습하는 경향이 있는데, 이는 보상(reward)이나 가치(value) 예측에는 도움이 될 수 있으나, 정밀한 액션 예측에는 도움이 되지 않을 수 있다.

### 2. 분석적 증거 (t-SNE 및 Saliency Map)
- **t-SNE 분석**: VAE로 학습된 CoinRun의 표현은 액션별로는 잘 뭉치지 않지만, 예상 리워드(returns)별로는 잘 뭉치는 경향을 보였다. 이는 RepL이 거시적인 상태 정보(가치)는 잘 포착하지만 세밀한 제어 정보(액션)는 놓치고 있음을 시사한다.
- **Saliency Map 분석**: SimCLR 인코더가 `finger-spin`에서는 전경(foreground) 객체에 주목하지만, `CoinRun`에서는 배경(background)의 변화에 더 주목하는 것이 확인되었다. 이는 RepL 알고리즘이 태스크의 의미론적 중요도와 상관없이 시각적으로 구분이 쉬운 특징에 편향될 수 있음을 보여준다.

## 📌 TL;DR

본 논문은 이미지 기반 모방 학습에서 표현 학습(RepL)의 효용성을 체계적으로 분석하였으며, **정교한 RepL 알고리즘을 사용하는 것보다 적절한 이미지 증강(Image Augmentation)을 적용하는 것이 성능 향상에 더 효과적**이라는 점을 밝혀냈다. 이는 RepL이 주로 시각적으로 두드러진 거시적 특징을 학습하는 반면, 모방 학습의 액션 예측에는 세밀한 국소적 특징이 필요하기 때문이라는 통찰을 제공한다. 이 연구는 향후 표현 학습 연구가 단순히 시각적 변이를 포착하는 것을 넘어, 의사결정에 필요한 세밀한 불변성(invariances)을 어떻게 획득할 것인지 고민해야 함을 시사한다.