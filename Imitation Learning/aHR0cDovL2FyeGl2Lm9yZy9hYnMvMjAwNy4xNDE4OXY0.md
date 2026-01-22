# TrajGAIL: Generative Adversarial Imitation Learning을 이용한 도시 차량 궤적 생성

Seongjin Choi, Jiwon Kim, Hwasoo Yeo

## 🧩 Problem to Solve

도시 차량 궤적 데이터는 풍부하게 수집되고 있지만, 데이터 희소성(sparsity) 및 개인 정보 보호(privacy) 문제로 인해 충분히 활용되지 못하는 경우가 많습니다. 기존 연구들은 주로 판별 모델(discriminative modeling)을 통해 다음 위치 예측에 초점을 맞추었으나, 이는 실제와 유사한 전체 궤적을 다양하게 생성하는 데 한계가 있었습니다. 판별 모델은 데이터의 결정 경계만을 학습할 뿐, 현실적인 궤적을 샘플링하는 데 필요한 데이터의 근본적인 분포를 포착하지 못하기 때문입니다. 따라서, 제한된 관측치만으로도 실제 궤적과 유사한 합성 궤적을 생성할 수 있는 생성 모델(generative modeling) 접근 방식이 필요합니다.

## ✨ Key Contributions

* **부분 관측 마르코프 결정 과정(Partially Observable Markov Decision Process, POMDP) 기반 모델링**: 도시 차량 궤적 생성 문제를 POMDP로 공식화하여 차량의 현재 위치뿐만 아니라 이전 위치 정보까지 효과적으로 반영하고 잠재 상태를 추정할 수 있도록 했습니다.
* **TrajGAIL 프레임워크 제안**: 생성적 적대적 모방 학습(Generative Adversarial Imitation Learning, GAIL) 프레임워크를 도시 차량 궤적 생성에 적용한 TrajGAIL을 제안했습니다. 이는 기존 역 강화 학습(IRL)의 높은 계산 비용 문제를 해결하면서, 표준 GAIL 모델의 한계(차량의 과거 위치 고려 부족)를 극복합니다.
* **RNN 임베딩 레이어 활용**: POMDP 프레임워크 내에서 순환 신경망(Recurrent Neural Network, RNN) 임베딩 레이어를 사용하여 관측 시퀀스를 믿음 상태 벡터(belief state vector)로 매핑함으로써, 마르코프 가정 위반 없이 순차적 정보를 효과적으로 포착합니다.
* **다각적인 성능 평가**: 궤적 수준(trajectory-level) 유사도 (BLEU, METEOR 점수)와 데이터셋 수준(dataset-level) 유사도 (경로 분포의 Jensen-Shannon 거리)를 모두 평가하여 모델의 성능과 강건성을 검증했습니다.
* **데이터셋 복잡도에 대한 강건성 입증**: 데이터셋 복잡도(링크 전환 엔트로피)가 증가하더라도 TrajGAIL이 가장 낮은 복잡도 민감도를 보여, 복잡한 실제 차량 이동 패턴 학습에 효과적임을 입증했습니다.

## 📎 Related Works

* **궤적 재구성**: 두 지점 사이의 가장 그럴듯한 경로를 재구성하는 연구 (Chen et al., 2011; Hu et al., 2018; Feng et al., 2015; Rao et al., 2018).
* **다음 위치 예측**: 이전에 방문한 위치를 기반으로 다음 위치를 예측하는 연구 (Monreale et al., 2009; Gambs et al., 2012; Choi et al., 2019b; Jin et al., 2019; Choi et al., 2018, 2019a).
* **모방 학습 (Imitation Learning)**: 전문가의 시연으로부터 의사결정 전략을 학습하는 분야.
  * **행동 복제(Behavior Cloning)**: 지도 학습 문제로 단순하게 접근하는 방식 (MMC, RNN 모델 포함).
  * **역 강화 학습(Inverse Reinforcement Learning, IRL)**: 전문가의 행동을 설명하는 보상 함수를 복구하는 방식 (Ng et al., 2000; Abbeel and Ng, 2004; Ziebart et al., 2008a,b; Wulfmeier et al., 2015).
* **생성적 적대적 네트워크 (Generative Adversarial Networks, GANs)**: 생성자와 판별자의 적대적 학습을 통해 실제와 유사한 데이터 생성 (Goodfellow et al., 2014). 교통 공학 분야에도 응용 (Zhang et al., 2019a; Xu et al., 2020; Li et al., 2020; Liu et al., 2018; Rao et al., 2020).
* **생성적 적대적 모방 학습 (Generative Adversarial Imitation Learning, GAIL)**: IRL의 아이디어와 GAN 프레임워크를 결합하여 효율적으로 모방 학습을 수행 (Ho and Ermon, 2016).

## 🛠️ Methodology

TrajGAIL은 도시 차량 궤적 생성을 위해 POMDP와 GAIL 프레임워크를 결합합니다.

1. **문제 공식화**:
    * 차량 궤적을 링크 ID 시퀀스($LinkSeq = \{l_1, \dots, l_M\}$)로 변환합니다.
    * 차량의 도로 네트워크 내 움직임을 POMDP로 공식화합니다: $(O, S, A, T, R)$.
        * **관측 공간($O$)**: 도로 네트워크의 링크 ID와 가상 토큰($Start, End$).
        * **행동($A$)**: [Straight, Left, Right, Terminate]와 같은 보편적인 움직임 방향을 정의하여 행동 공간을 축소합니다.
        * **믿음 상태($s_t$)**: $s_t = f(o_1, \dots, o_t) = \hat{s}^*_t$. 관측 시퀀스 ($o_1, \dots, o_t$)를 기반으로 잠재적, 비관측 상태($s^*_t$)를 추정하며, RNN 임베딩 레이어를 통해 구현됩니다.
2. **TrajGAIL 프레임워크**: 생성자와 판별자로 구성되며, minimax 게임 방식으로 학습합니다.
    * **생성자(Generator)**:
        * **RNN 임베딩 레이어**: 관측 시퀀스(링크 ID)를 믿음 상태 벡터로 매핑합니다.
        * **정책 생성자(Policy Generator)**: 믿음 상태 벡터($s_t$)를 기반으로 다음 행동의 확률 $\pi(a|s_t)$를 계산하고 샘플링합니다.
        * **가치 추정기(Value Estimator)**: 상태-행동 가치 함수 $Q_{\pi}(s,a)$를 추정하여 예상 누적 보상을 계산하며, 정책 업데이트에 사용됩니다.
        * **목표 함수**: 정책 기울기(Policy Gradient)와 엔트로피 최대화(Entropy Maximization) 목표 $J(\theta) = J_{PG}(\theta) + \lambda H(\pi_{\theta})$를 최대화하여 다양하고 현실적인 궤적을 생성합니다.
    * **판별자(Discriminator)**:
        * **RNN 임베딩 레이어**: 생성자와 유사하게 관측 시퀀스를 믿음 상태로 임베딩합니다.
        * **분류**: 실제 궤적 샘플(0)과 생성된 궤적 샘플(1)을 구분하며, 입력 (상태, 행동) 쌍이 생성자로부터 온 것일 확률 $D_{\omega}(s,a)$를 계산합니다. 이진 교차 엔트로피 손실을 최소화하도록 훈련됩니다.
        * **보상 함수**: 생성자에게 학습 신호를 제공합니다. $R(s,a) = -\log(D_{\omega}(s,a))$로 정의됩니다.
3. **역전파**: 정책 생성자($J_{Policy}$), 가치 추정기($J_{Value}$), 판별자($J_{Discrim}$) 각각의 목표 함수에 따라 관련 파라미터만 업데이트됩니다. 각 모듈은 독립적인 RNN 임베딩 레이어를 가집니다.
4. **훈련 기법**: 판별자와 생성자 간의 학습 균형을 위해 생성자를 판별자보다 더 자주 업데이트하고(6:2 비율), 모드 붕괴를 방지하기 위해 각 훈련 반복에서 충분한 수의 궤적(20,000개)을 생성합니다.

## 📊 Results

* **데이터셋**: AIMSUN 시뮬레이션 데이터 (Single-OD, One-way Multi-OD, Two-way Multi-OD) 및 서울 강남구 실제 택시 DTG 데이터.
* **기준 모델**: Mobility Markov Chain (MMC), Recurrent Neural Network (RNN), MaxEnt Inverse Reinforcement Learning (MaxEnt(SVF), MaxEnt(SAVF)).
* **계산 시간**: 모든 모델이 20,000개 궤적 생성에 2초 미만 소요. RNN과 TrajGAIL은 GPU 사용 시 상당한 속도 향상을 보였습니다.
* **궤적 수준 평가 (BLEU, METEOR 점수)**:
  * **Single-OD 데이터셋**: 대부분의 모델이 0.99 이상의 높은 점수를 기록했습니다.
  * **Multi-OD 및 Gangnam 데이터셋**: 데이터셋 복잡도가 증가함에 따라 MaxEnt 모델들의 점수는 크게 감소한 반면, TrajGAIL과 RNN은 높은 성능을 유지했으며, TrajGAIL이 RNN보다 더 높은 평균 점수와 낮은 표준 편차를 보였습니다 (예: Gangnam 데이터셋에서 TrajGAIL BLEU/METEOR 0.9974). RNN은 때때로 비현실적인 경로를 생성하는 경향이 있었습니다.
* **데이터셋 수준 평가 (경로 분포의 Jensen-Shannon 거리 $d_{JS}$)**:
  * 낮은 $d_{JS}$ 값은 높은 유사도를 의미합니다.
  * **Single-OD 데이터셋**: MaxEnt(SVF)를 제외한 모든 모델이 $d_{JS} < 0.1$로 우수했습니다.
  * **Multi-OD 및 Gangnam 데이터셋**: 모든 모델의 $d_{JS}$가 증가했으나, TrajGAIL은 다른 모델들보다 확연히 낮은 $d_{JS}$를 기록하며 가장 우수한 성능을 보였습니다 (예: Gangnam 데이터셋에서 TrajGAIL $d_{JS}$ 0.4230). TrajGAIL은 다른 모델들에 비해 '알려지지 않은' 경로를 훨씬 적게 생성했습니다.
* **복잡도 민감도**: 링크 전환 엔트로피와 Jensen-Shannon 거리 간의 관계 분석 결과, TrajGAIL이 가장 낮은 복잡도 민감도를 보여 데이터셋 복잡도에 강건함을 입증했습니다.

## 🧠 Insights & Discussion

* **MaxEnt 모델의 성능 차이**: MaxEnt(SVF)가 낮은 성능을 보인 것은 상태 방문 빈도만 고려하고 궤적의 순차적 특성을 간과했기 때문입니다. 상태-행동 방문 빈도를 고려한 MaxEnt(SAVF)의 성능 향상은 순차적 정보의 중요성을 강조합니다.
* **TrajGAIL의 RNN 대비 우위**: TrajGAIL의 생성자는 RNN과 유사하지만, 판별자로부터의 보상 함수와 가치 추정기를 통해 현재의 상태-행동뿐만 아니라 **전체 궤적의 미래 시퀀스에 대한 현실성**까지 고려합니다. 이는 RNN이 다음 위치 예측에만 집중하여 때때로 비현실적이거나 지나치게 긴 궤적을 생성하는 한계를 극복하고, 더 포괄적인 궤적 분포를 학습하도록 돕습니다.
* **이론적 및 방법론적 기여**: TrajGAIL은 POMDP와 RNN 임베딩을 결합하여 표준 GAIL 및 전통적 IRL의 한계인 과거 위치 정보 반영 문제를 해결하고, GAIL 접근 방식이 GAN에 비해 궤적 생성에서 더 현실적인 결과를 도출함을 보여주었습니다. 또한 딥러닝 기반 접근으로 대규모 네트워크에서의 확장성을 제공합니다.
* **한계 및 미래 연구**: 현재 모델은 링크 시퀀스 생성에만 초점을 맞추고 시간적 요소나 교통 상황, OD 정보 등을 직접적으로 고려하지 않습니다. 향후 연구에서는 어텐션 메커니즘, 조건부 GAIL (cGAIL), 또는 InfoGAIL과 같은 고급 기법을 활용하여 이러한 추가 정보를 통합함으로써 모델 성능을 더욱 향상시킬 수 있습니다. 또한, 데이터 기반 딥러닝 모델이 동적 교통 배정(DTA)이나 교통 시뮬레이션 모델의 이론 기반 구성 요소를 보완하거나 대체할 가능성에 대한 탐색이 중요합니다.

## 📌 TL;DR

본 연구는 도시 차량 궤적의 희소성과 사생활 보호 문제를 해결하기 위해 **TrajGAIL**이라는 생성적 적대적 모방 학습(GAIL) 프레임워크를 제안합니다. TrajGAIL은 차량 이동 결정을 **부분 관측 마르코프 결정 과정(POMDP)**으로 모델링하고, RNN 임베딩 레이어를 사용하여 과거 궤적 시퀀스 정보를 효과적으로 통합합니다. 생성자는 판별자가 제공하는 보상 함수를 통해 학습하여 실제와 유사한 궤적을 생성하며, 이때 궤적 전체의 현실성을 고려합니다. 시뮬레이션 및 실제 데이터셋에 대한 평가 결과, TrajGAIL은 기존의 RNN 및 마르코프 기반 모델들보다 궤적 수준(BLEU, METEOR) 및 데이터셋 수준(Jensen-Shannon 거리)에서 모두 우수한 성능을 보였으며, 데이터셋 복잡도에 대한 민감도가 가장 낮아 복잡한 환경에서도 강건한 궤적 생성 능력을 입증했습니다.
