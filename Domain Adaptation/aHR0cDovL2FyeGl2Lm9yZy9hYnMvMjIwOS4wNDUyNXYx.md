# PoliTO-IIT-CINI Submission to the EPIC-KITCHENS-100 Unsupervised Domain Adaptation Challenge for Action Recognition

Mirco Planamente, Gabriele Goletto, Gabriele Trivigno, Giuseppe Averta, Barbara Caputo (2022)

## 🧩 Problem to Solve

본 논문은 1인칭 시점(first-person) 액션 인식 작업에서 발생하는 Unsupervised Domain Adaptation (UDA) 문제를 해결하고자 한다. 1인칭 비디오 데이터는 웨어러블 장치를 통해 수집되므로 시각적, 청각적 정보가 풍부하지만, 동시에 다음과 같은 심각한 문제들을 내포하고 있다.

첫째, 사용자의 머리 움직임으로 인한 Ego-motion은 배경의 급격한 변화를 야기하여 노이즈로 작용하며, 이는 실제 동작과 Ego-motion 간의 혼동을 초래한다. 둘째, 모델의 예측이 촬영 환경(예: 서로 다른 주방)에 강하게 의존하는 'Environmental Bias' 문제가 존재한다. 이러한 환경적 편향은 학습 데이터와 테스트 데이터의 환경이 다를 때 성능을 급격히 저하시키는 주요 원인이 된다.

따라서 본 연구의 목표는 시각(RGB, Optical Flow) 및 청각(Audio) 모달리티를 효과적으로 결합하고, 시간적 변화(Temporal Shift)와 환경적 변화(Environmental Shift)라는 두 가지 도메인 시프트를 동시에 완화하여, 레이블이 없는 타겟 데이터에서도 높은 액션 인식 성능을 달성하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 도메인 일반화(Domain Generalization, DG)와 도메인 적응(Domain Adaptation, DA) 기술을 단계적으로 결합하고, 환경적 편향을 해결하기 위한 다중 소스-다중 타겟 접근 방식을 도입한 것이다. 주요 기여 사항은 다음과 같다.

1. **RNA-Net의 확장**: 오디오-비주얼 특징의 노름(norm)을 정렬하여 도메인 불변 특징을 추출하는 Relative Norm Alignment (RNA) 기법을 Optical Flow 모달리티까지 확장하여 적용하였다.
2. **MSTAA 프레임워크 제안**: 환경적 편향을 해결하기 위해 Multiple Spatio-Temporal Adversarial Alignment (MSTAA)를 제안하였다. 이는 주방별 시간적 정렬(MTAA)과 주방 간 공간적 정렬(MSAA)을 결합한 형태이다.
3. **앙상블 도메인 적응 손실 함수**: 서로 다른 백본 네트워크들이 일관된 예측을 하도록 유도하는 Min-Entropy Consensus (MEC) 손실과, 불확실성이 높은 클래스의 노이즈를 줄이는 Complement Entropy (CENT) 손실을 도입하였다.

## 📎 Related Works

본 논문은 기존의 도메인 적응 연구들이 주로 단일 도메인 시프트에 집중했다는 점을 지적한다. 특히, 1인칭 액션 인식 분야에서 모델이 특정 환경에 과적합되어 환경이 바뀔 때 성능이 하락하는 문제가 지속적으로 보고되어 왔다.

기존의 DANN(Domain-Adversarial Neural Networks)과 같은 표준 UDA 방식은 소스와 타겟 도메인의 분포를 정렬하지만, 본 논문은 여기에 더해 RNA-Net을 통해 모달리티 간의 노름 불균형을 해소함으로써 특정 도메인에 편향되지 않는 특징을 추출하는 DG 접근 방식을 함께 사용한다는 점에서 차별점을 가진다. 또한, 단순한 UDA를 넘어 Multi-Source Multi-Target 설정을 도입하여 환경적 편향 문제를 보다 직접적으로 다루었다.

## 🛠️ Methodology

### 1. Domain Generalization: Relative Norm Alignment (RNA)

다양한 소스 도메인에서 추출된 특징들의 노름 불균형이 특정 도메인으로의 편향을 야기한다는 점에 착안하여, 오디오와 비주얼 특징의 평균 노름 거리를 최소화하는 $L_{RNA}$ 손실 함수를 사용한다.

$$L_{RNA} = \left( \frac{\mathbb{E}[h(X^v)]}{\mathbb{E}[h(X^a)]} - 1 \right)^2$$

여기서 $h(x)$는 특징의 $L_2$-norm을 의미하며, $X^v$와 $X^a$는 각각 비주얼과 오디오 모달리티의 특징 집합이다. 이 손실 함수를 통해 네트워크는 특정 모달리티나 도메인에 의존하지 않는 공통 지식을 학습하게 된다.

### 2. Unsupervised Domain Adaptation (UDA)

타겟 데이터의 레이블이 없는 상황을 해결하기 위해 두 가지 수준의 접근을 취한다.

- **Multi-Level Adversarial Alignment**: 프레임 수준(frame-level)과 비디오 수준(video-level)에서 판별기(Discriminator)를 두어, 소스와 타겟 도메인을 구분하지 못하도록 특징 표현을 학습시킨다.
- **Attentive Entropy**: 분류기의 불확실성을 줄이기 위해 엔트로피 최소화 손실을 사용하되, 도메인 간 차이가 적은 비디오에 더 높은 가중치를 두는 Attentive 방식의 재가중치 기법을 적용한다.

### 3. Multi-Source Multi-Target Domain Adaptation (MSTAA)

환경적 편향을 해결하기 위해 제안된 MSTAA는 두 가지 모듈로 구성된다.

- **MTAA (Multiple Temporal Adversarial Alignment)**: $K$개의 주방(Kitchen) 각각에 대해 도메인 적대적 분기를 두어, 각 주방별로 비디오 및 프레임 수준의 분포를 정렬한다.
- **MSAA (Multiple Spatial Adversarial Alignment)**: $K$차원의 판별기를 추가하여 서로 다른 주방들 간의 분포를 정렬함으로써 환경적 편향을 완화한다.

### 4. Ensemble UDA Losses

다양한 아키텍처(I3D, BN-Inception, ResNet-50+TSM)를 앙상블할 때 발생하는 예측 불일치를 해결하기 위해 다음의 손실 함수를 사용한다.

- **Min Entropy Consensus (MEC)**: 서로 다른 백본 모델들이 타겟 데이터에 대해 유사한 예측을 하도록 강제한다.
  $$L_{MEC} = -\frac{1}{m} \sum_{i=1}^{m} \frac{1}{b} \max_{y \in Y} \sum_{b} \log p_b(y|x_i^t)$$
- **Complement Entropy (CENT)**: 가장 확률이 높은 클래스를 제외한 나머지 '보완 클래스(complement classes)'들의 엔트로피를 최대화하여, 불확실한 예측으로 인한 노이즈를 줄인다.
  $$L_{CENT} = \frac{1}{N} \sum_{i=1}^{N} H(\hat{y}_i^{\bar{c}})$$

## 📊 Results

### 실험 설정

- **데이터셋**: EPIC-KITCHENS-100
- **평가 지표**: Verb, Noun, Action의 Top-1 및 Top-5 정확도
- **백본 아키텍처**: I3D (RGB, Flow), BN-Inception (Audio), ResNet-50 with TSM
- **융합 전략**: Late Fusion 및 SMR (Semantic Mutual Refinement) mid-fusion

### 주요 결과

- **챌린지 순위**: 'Verb' 부문 2위, 'Noun' 및 'Action' 부문 3위를 기록하였다.
- **DG 성능 (Table 3)**: RNA 기법을 적용했을 때, Source Only 대비 Top-1 정확도가 최대 3%, Top-5가 10% 향상되었다. 이는 타겟 데이터 없이도 도메인 불변 특징을 추출하는 RNA의 효과를 입증한다.
- **UDA 및 MSTAA 효과 (Table 2)**: MSTAA를 적용하지 않았을 때 Action Top-1 정확도는 24.83%에 그쳤으나, 제안 기법 적용 후 순위권 성적을 거두었다. 이는 검증 셋과 테스트 셋의 주방 구성이 다르다는 점을 고려할 때, 환경적 편향을 잡는 MSTAA가 필수적이었음을 시사한다.

## 🧠 Insights & Discussion

본 연구는 1인칭 액션 인식에서 단순한 도메인 시프트뿐만 아니라, '환경적 편향'이라는 구체적인 문제에 집중하여 이를 다중 소스-다중 타겟 적응 문제로 정의하고 해결하였다. 특히, DG(RNA)와 UDA(MSTAA), 그리고 앙상블 일관성(MEC, CENT)을 계층적으로 결합한 파이프라인이 실질적인 성능 향상을 이끌어냈음을 보여준다.

한 가지 주목할 점은 검증 셋(Validation set)에서는 MSTAA의 효과가 뚜렷하게 나타나지 않았으나, 테스트 셋(Test set)에서는 큰 성능 향상이 있었다는 것이다. 이는 검증 셋과 테스트 셋의 주방 분포가 서로 다르기 때문이며, 결과적으로 본 모델이 특정 환경에 과적합되지 않고 일반화 능력을 갖추었음을 방증한다. 다만, 다수의 백본과 복잡한 손실 함수를 사용함에 따라 학습 비용과 하이퍼파라미터 튜닝의 복잡성이 증가했을 것으로 판단된다.

## 📌 TL;DR

본 논문은 EPIC-KITCHENS-100 챌린지를 위해 **RNA-Net(도메인 일반화)**, **MSTAA(환경적 편향 제거)**, 그리고 **MEC/CENT(앙상블 일관성)**를 결합한 다중 모달리티 액션 인식 프레임워크를 제안하였다. 이 접근법은 시간적/환경적 도메인 시프트를 효과적으로 완화하여 챌린지 상위권(Verb 2위, Noun/Action 3위) 성적을 거두었으며, 특히 환경적 편향을 해결하기 위한 다중 소스-타겟 정렬의 중요성을 입증하였다.
