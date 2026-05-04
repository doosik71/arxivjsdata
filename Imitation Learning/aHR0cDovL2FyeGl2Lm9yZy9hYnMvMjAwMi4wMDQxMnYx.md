# Combating False Negatives in Adversarial Imitation Learning

Konrad Zołna, Chitwan Saharia, Léonard Boussioux, David Yu-Tung Hui, Maxime Chevalier-Boisvert, Dzmitry Bahdanau, Yoshua Bengio (2020)

## 🧩 Problem to Solve

본 논문은 Adversarial Imitation Learning(AIL), 특히 GAIL(Generative Adversarial Imitation Learning) 프레임워크에서 발생하는 **False Negatives(FN)** 문제와 그로 인한 성능 저하 문제를 해결하고자 한다.

AIL의 핵심은 Discriminator(판별자)가 전문가의 시연(Expert demonstrations)과 에이전트의 궤적(Agent episodes)을 구분하도록 학습하고, 이를 통해 에이전트에게 보상을 제공하는 것이다. 그러나 에이전트의 성능이 향상되어 전문가와 유사한 성공적인 궤적을 생성하게 되면, Discriminator는 실제로는 성공적인 궤적임에도 불구하고 이를 '에이전트가 생성했다'는 이유만으로 부정적 예시(Negative example)로 분류하여 낮은 보상을 주게 된다.

이러한 불일치하는 학습 신호는 Discriminator의 학습을 방해하며, 결과적으로 에이전트의 전체적인 성능을 불안정하게 만들고 학습 효율을 떨어뜨린다. 본 논문의 목표는 이러한 False Negatives 현상을 진단하고, 이를 완화하여 샘플 효율성과 최종 성능을 높이는 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같다.

1. **False Negatives 문제의 진단**: Oracle Filtering이라는 진단 방법을 통해, 성공적인 에이전트 궤적이 부정적 예시로 레이블링되는 FN 문제가 Adversarial Imitation Learning의 성능을 심각하게 저해한다는 것을 실험적으로 입증하였다.
2. **Fake Conditioning (FC) 제안**: 목표 조건부 작업(Goal-conditioned task)의 특성을 활용하여, 지시어(Instruction)를 무작위로 교체함으로써 FN 발생 가능성을 획기적으로 낮추는 Fake Conditioning 기법을 제안하였다.
3. **Auxiliary Rewards 설계**: FC 기법으로 인해 발생할 수 있는 Discriminator의 부정확성이나 취약점을 보완하기 위해, Blank Conditioning과 Done Detector라는 보조 보상 메커니즘을 설계하여 안정적인 학습을 도모하였다.

## 📎 Related Works

본 논문은 다음과 같은 기존 연구들을 언급하며 차별점을 제시한다.

- **Behavioral Cloning (BC)**: 단순한 지도 학습 방식으로 전문가의 행동을 모방하지만, compounding errors(오차 누적) 문제로 인해 성능 한계가 있다.
- **GAIL (Generative Adversarial Imitation Learning)**: Inverse RL과 GAN의 아이디어를 결합하여 보상 함수를 동시에 학습한다. 하지만 본 논문에서 지적하듯, 에이전트가 숙련될수록 발생하는 FN 문제로 인해 성능이 불안정해지는 한계가 있다.
- **기존 FN 대응 연구**: 이전 연구(Zolna et al. 2019, Bahdanau et al. 2019)에서도 유사한 문제가 언급되었으며 Actor Early Stopping이나 특정 상태 필터링과 같은 휴리스틱한 해결책이 제시되었다. 그러나 본 논문은 이를 정밀하게 측정할 수 있는 진단 도구를 제공하고, 더 일반적이고 체계적인 Fake Conditioning이라는 방법론을 제시한다는 점에서 차별화된다.

## 🛠️ Methodology

### 1. 기본 프레임워크: Conditioned Recurrent Discriminator

본 연구는 부분 관측 가능 환경(Partially-observable environment)을 다루므로, Discriminator에 메모리 능력을 부여하기 위해 Recurrent Neural Network(RNN)를 사용한다. 입력값은 단일 상태-행동 쌍이 아닌 전체 궤적 $\tau = ((o_1, a_1), \dots, (o_T, a_T))$와 지시어 $c$이다.

Discriminator의 기본 손실 함수는 다음과 같다:
$$L_D(\theta) = \mathbb{E}_{(c, \tau) \sim B_{\text{agent}}} [-\log(1 - D_\theta(c, \tau))] + \mathbb{E}_{(c, \tau) \sim B_{\text{expert}}} [-\log(D_\theta(c, \tau))]$$
여기서 $B_{\text{agent}}$와 $B_{\text{expert}}$는 각각 에이전트와 전문가의 궤적 버퍼이다.

### 2. 진단 방법: Oracle Filtering (OF)

FN 문제의 영향을 측정하기 위해, 환경의 실제 보상 신호를 사용하여 에이전트의 궤적 중 성공한 것들을 제외하고 오직 **실패한 궤적($B_{\text{oracle agent}}$)만** 사용하여 Discriminator를 학습시키는 방법이다.
$$L^O_D(\theta) = \mathbb{E}_{(c, \tau) \sim B_{\text{oracle agent}}} [-\log(1 - D_\theta(c, \tau))] + \mathbb{E}_{(c, \tau) \sim B_{\text{expert}}} [-\log(D_\theta(c, \tau))]$$

### 3. 핵심 제안: Fake Conditioning (FC)

환경 보상 없이 FN 문제를 해결하기 위해, 궤적 $\tau$에 대응하는 지시어 $c$를 다른 무작위 지시어 $\tilde{c}$로 교체하여, 해당 궤적이 $\tilde{c}$ 관점에서는 반드시 실패한 궤적이 되도록 만드는 기법이다.

- **Expert FC**: 전문가 궤적을 사용하여, 원래 지시어($c$)일 때는 Positive, 가짜 지시어($\tilde{c}$)일 때는 Negative로 학습한다.
  $$L^E_D(\theta) = \mathbb{E}_{(c, \tau) \sim B_{\text{expert}}, \tilde{c} \sim S \setminus \{c\}} [-\log(1 - D_\theta(\tilde{c}, \tau))] + \mathbb{E}_{(c, \tau) \sim B_{\text{expert}}} [-\log(D_\theta(c, \tau))]$$
- **Agent FC**: 전문가 궤적은 그대로 두고, 에이전트 궤적에 가짜 지시어($\tilde{c}$)를 부여하여 Negative로 학습한다.
  $$L^A_D(\theta) = \mathbb{E}_{(c, \tau) \sim B_{\text{agent}}, \tilde{c} \sim S \setminus \{c\}} [-\log(1 - D_\theta(\tilde{c}, \tau))] + \mathbb{E}_{(c, \tau) \sim B_{\text{expert}}} [-\log(D_\theta(c, \tau))]$$

### 4. 보조 보상 (Auxiliary Rewards)

FC 기법 사용 시 발생할 수 있는 보상 부정확성을 해결하기 위해 두 가지 추가 Discriminator를 운용한다.

- **Blank Conditioning**: 지시어 없이(또는 빈 지시어 $c_\emptyset$ 사용) 에이전트와 전문가의 궤적 자체의 유사성만을 판별한다. 이는 Expert FC 사용 시 에이전트 궤적 데이터가 부족해 발생하는 문제를 보완한다.
- **Done Detector**: 궤적이 완료되었는지 여부를 판별한다. 전문가의 완료된 궤적을 Positive, 중간에 끊긴 불완전한 궤적($B_{\text{sub expert}}$)을 Negative로 학습하여, 에이전트가 단순히 가짜 지시어를 피하는 것이 아니라 실제로 작업을 완수하도록 유도한다.
  $$L^D_D(\theta) = \mathbb{E}_{(c, \tau) \sim B_{\text{sub expert}}} [-\log(1 - D_\theta(c, \tau))] + \mathbb{E}_{(c, \tau) \sim B_{\text{expert}}} [-\log(D_\theta(c, \tau))]$$

## 📊 Results

### 실험 설정

- **환경**: BabyAI (GoToLocal, PickupLoc, PutNextLocal 등 4가지 난이도 수준).
- **비교 대상**: Behavioral Cloning (BC), Baseline GAIL, Oracle Filtering (OF), 제안 방법(Agent FC + Done Detector).
- **평가 지표**: 성공률(Success Rate).

### 주요 결과

1. **Baseline GAIL의 한계**: Baseline GAIL은 전문가 데이터가 충분함에도 불구하고 99% 성공률에 도달하지 못하며 성능이 불안정했다.
2. **FN 문제 확인**: Oracle Filtering을 적용했을 때 성능이 비약적으로 상승했으며, 이는 FN 문제가 GAIL 학습의 주요 저해 요소임을 증명한다.
3. **제안 방법의 우수성**: Agent FC와 Done Detector를 결합한 방법은 BC보다 훨씬 적은 수의 전문가 시연(최대 64배 적은 데이터)만으로도 모든 작업을 해결(성공률 > 99%)하였다.
4. **단일 목표 작업으로의 확장**: 단일 목표 작업(GoToRedBall)에서도 이를 포함하는 더 큰 다중 목표 작업(GoToLocal)으로 구성하여 FC를 적용했을 때, Baseline GAIL보다 월등한 성능을 보였다.

## 🧠 Insights & Discussion

### 강점 및 분석

- 본 논문은 단순한 성능 향상을 넘어, Adversarial Imitation Learning에서 오랫동안 간과되었던 **'성공한 에이전트 궤적의 오분류'**라는 구조적 문제를 날카롭게 지적하였다.
- 특히 Oracle Filtering이라는 진단 도구를 통해 가설을 먼저 검증하고, 그 후에 실제 적용 가능한 Fake Conditioning 솔루션을 제시한 논리 전개가 매우 탄탄하다.

### 한계 및 비판적 해석

- **Done Action 의존성**: 실험 결과에서 `Done` 액션을 통한 에피소드 종료 여부가 성능에 매우 큰 영향을 미치는 것으로 나타났다. 이는 제안 방법이 환경의 특정 구조(종료 시점의 명확성)에 의존하고 있을 가능성을 시사한다.
- **보조 보상의 복잡성**: 최종 성능을 위해 여러 개의 Discriminator head와 보조 보상을 혼합하여 사용하는데, 이는 하이퍼파라미터 튜닝의 복잡성을 증가시킬 수 있다.
- **가정**: 본 방법론은 '지시어'가 존재하는 Goal-conditioned task를 전제로 한다. 지시어가 없는 일반적인 RL 환경에서 FC와 유사한 효과를 낼 수 있는 일반화된 방법론에 대한 논의는 부족하다.

## 📌 TL;DR

이 논문은 에이전트가 숙련될수록 성공적인 궤적이 부정적 예시로 처리되는 **False Negatives(FN)** 문제가 Adversarial Imitation Learning의 성능을 저해함을 밝히고, 이를 해결하기 위해 지시어를 무작위로 교체하는 **Fake Conditioning(FC)**과 **보조 보상(Auxiliary Rewards)** 시스템을 제안하였다. 실험 결과, 제안 방법은 기존 GAIL의 불안정성을 해결하고 BC 대비 샘플 효율성을 획기적으로(최대 수십 배) 개선하였다. 이 연구는 향후 복잡한 지시어 기반의 모방 학습 시스템을 구축하는 데 있어 매우 중요한 가이드라인을 제공한다.
