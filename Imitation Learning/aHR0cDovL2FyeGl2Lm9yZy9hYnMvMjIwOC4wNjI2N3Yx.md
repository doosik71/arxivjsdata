# Causal Imitation Learning with Unobserved Confounders

Junzhe Zhang, Daniel Kumor, Elias Bareinboim (2020)

## 🧩 Problem to Solve

본 논문은 모방 학습(Imitation Learning)에서 전문가가 행동을 결정할 때 사용하는 공변량(covariates)이 학습자에게는 관찰되지 않는 경우, 즉 Unobserved Confounders(UCs)가 존재하는 상황에서의 학습 문제를 해결하고자 한다.

일반적인 모방 학습의 두 축인 Behavior Cloning(BC)과 Inverse Reinforcement Learning(IRL)은 전문가와 학습자가 동일한 관찰 값(observations)을 공유한다는 가정을 전제로 한다. 그러나 실제 환경에서는 전문가가 학습자가 볼 수 없는 추가적인 정보(예: 드론으로 촬영한 도로 영상에서 운전자가 직접 보는 앞차의 브레이크 등)를 사용하여 행동을 결정하는 경우가 많다. 이러한 상황에서 단순히 관찰된 데이터만을 이용해 전문가의 행동 정책을 복제(cloning)하면, 전문가가 최적의 성능을 냈더라도 학습자의 성능은 현저히 떨어지는 결과가 초래된다.

따라서 본 연구의 목표는 보상 신호(reward signal)가 관측되지 않고 일부 공변량이 누락된 상황에서도, 인과 모델(causal model)의 정성적 지식과 관찰 데이터를 결합하여 전문가의 성능을 동일하게 구현하는 모방 정책을 학습하는 방법을 제시하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 인과 추론(causal inference)의 관점에서 '모방 가능성(imitability)'을 정의하고, 이를 판별하고 구현하는 이론적/실무적 프레임워크를 제공한 것이다.

1. **모방 가능성의 그래프 기준 제시**: 주어진 인과 그래프(causal graph)와 정책 공간(policy space)을 바탕으로, 전문가의 성능을 모방하는 것이 가능한지 여부를 결정하는 필요충분조건인 그래프 기준(graphical criterion)을 제안하였다.
2. **Practical Imitability 개념 도입**: 정성적인 그래프 구조만으로는 모방이 불가능하더라도, 관찰 데이터의 정량적 분포($P(o)$)를 활용하면 모방이 가능할 수 있다는 '실용적 모방 가능성(p-imitable)' 개념을 정의하였다.
3. **IMITATE 알고리즘 및 구현 방법론**: $\pi$-Backdoor admissible set과 Imitation Instrument를 찾아내어 latent reward 상황에서도 정책을 식별하는 알고리즘을 제안하였으며, 고차원 데이터 처리를 위해 GAN(Generative Adversarial Networks) 기반의 파라미터화된 POSCM(Partially Observable SCM) 구현 방법을 제시하였다.

## 📎 Related Works

기존의 모방 학습 연구들은 주로 다음과 같은 접근 방식을 취해왔다.
- **Behavior Cloning**: 전문가의 행동 정책 $\pi_E(a|s)$를 직접 근사하는 방식으로, 상태 $s$가 완전히 관찰되었다는 가정을 전제로 한다.
- **Inverse Reinforcement Learning**: 전문가의 궤적을 통해 잠재적인 보상 함수를 추론하고, 이를 최대화하는 정책을 학습한다.

최근 일부 연구에서 'Causal Confusion'이나 'Causal Transfer' 등을 다루었으나, 본 논문의 저자들은 이러한 기존 연구들이 보상 변수가 잠재적(latent)인 상황에서 발생하는 Unobserved Confounders의 영향을 충분히 다루지 않았다고 지적한다. 본 연구는 보상이 관찰되지 않는 상태에서 인과 그래프의 구조적 제약과 데이터 분포를 동시에 이용하여 정책을 도출한다는 점에서 기존 방식과 차별화된다.

## 🛠️ Methodology

### 1. POSCM (Partially Observable Structural Causal Model)
본 논문은 환경을 $\langle M, O, L \rangle$ 형태의 POSCM으로 정의한다. 여기서 $M$은 구조적 인과 모델(SCM)이며, $O$는 관찰 가능한 변수, $L$은 잠재 변수(latent variables)이다. 정책 $\pi$는 관찰 가능한 공변량 $Pa^*$를 입력으로 받아 행동 $X$를 결정하는 함수이며, 정책의 성능은 잠재 변수인 보상 $Y$의 기대값 $E[Y|do(\pi)]$로 평가된다.

### 2. Imitability 및 $\pi$-Backdoor
전문가의 보상 분포 $P(y)$를 동일하게 구현하는 정책 $\pi$가 존재할 때, 이를 **Imitable**하다고 정의한다. 

- **Theorem 1 (Direct Parents)**: 전문가와 학습자가 동일한 정책 공간을 공유하고 $X$로 들어오는 bi-directed arrow가 없다면, 단순한 Behavior Cloning($\pi(x|pa(\Pi)) = P(x|pa(X))$)으로 모방이 가능하다.
- **Theorem 2 ($\pi$-Backdoor)**: 전문가와 학습자의 정책 공간이 다를 때, $P(y)$가 imitable하기 위한 필요충분조건은 $\pi$-backdoor admissible set $Z$가 존재하는 것이다. 이때 $Z$는 다음 두 조건을 만족해야 한다.
    1. $Z \subseteq Pa(\Pi)$ (학습자가 관찰 가능한 집합)
    2. $(Y \perp \perp X | Z)_{G_X}$ (행동 $X$로 들어오는 화살표를 제거한 그래프에서 $Y$와 $X$가 $Z$에 의해 d-separated 됨)
    이 경우 모방 정책은 $\pi(x|z) = P(x|z)$가 된다.

### 3. Practical Imitability 및 Imitation Instrument
그래프 구조만으로 판단했을 때 imitable하지 않더라도, 실제 데이터 분포 $P(o)$를 분석하면 모방이 가능할 수 있다. 이를 위해 **Imitation Surrogate $S$**와 **Imitation Instrument $\langle S, \Pi' \rangle$** 개념을 도입한다.

- **Imitation Surrogate ($S$)**: $X$에 대한 개입이 $Y$에 미치는 영향이 $S$를 통해 매개되는 변수 집합이다. 즉, $(Y \perp \perp \hat{X} | S)_{G \cup \Pi}$를 만족한다.
- **Imitation Instrument**: $S$가 surrogate이고, $\Pi'$가 $P(s|do(\pi))$를 식별 가능하게 만드는 identifiable subspace일 때, $\langle S, \Pi' \rangle$를 instrument라고 한다.

이를 기반으로 한 **IMITATE 알고리즘**의 흐름은 다음과 같다.
1. $Y$가 관찰되었다고 가정했을 때 식별 가능한 정책 부분 공간 $\Pi'$를 나열한다.
2. $\Pi'$에 대해 $\hat{X}$와 $Y$를 d-separate 하는 최소 surrogate set $S$를 찾는다.
3. $\langle S, \Pi' \rangle$가 instrument인지 확인(IDENTIFY oracle 사용)한다.
4. 식 $P(s|do(\pi)) = P(s)$를 만족하는 정책 $\pi$를 찾아 반환한다.

### 4. GAN 기반 최적화
고차원 데이터의 경우 수식 기반의 식별이 어려우므로, POSCM을 신경망으로 파라미터화한 $\tilde{M}$을 학습한다. 
- **단계 1**: GAN을 이용해 관찰 데이터 분포 $P(o)$와 일치하는 생성 모델 $\tilde{M}$을 학습한다.
- **단계 2**: 학습된 모델 $\tilde{M}$ 내에서 직접 개입(intervention)을 수행하며, $P(s|do(\pi))$가 관찰된 $P(s)$와 일치하도록 정책 $\pi$를 최적화한다.

## 📊 Results

### 실험 설정
- **데이터셋**: HighD(고속도로 주행 데이터), MNIST 숫자 이미지.
- **비교 대상**: Causal Imitation (ci), Naive Behavior Cloning (bc), Expert's Reward (opt).
- **평가 지표**: 유도된 보상 분포 $P(y|do(\pi))$와 실제 전문가 보상 분포 $P(y)$ 사이의 $L_1$ 거리.

### 주요 결과
1. **Highway Driving**: 
    - 앞차의 속도 $Z$는 $\pi$-backdoor admissible set이지만, 주변 차량의 상태 $W$를 포함하면 confounder가 추가되어 $\pi$-backdoor 조건이 깨진다.
    - 실험 결과, $Z$만을 사용한 **ci** 방법은 $L_1 = 0.0018$로 전문가 성능을 거의 완벽히 모방했으나, 모든 공변량($Z, W$)을 사용한 **bc** 방법은 $L_1 = 0.2937$로 매우 낮은 성능을 보였다. 이는 단순히 많은 정보를 사용하는 것이 아니라 '인과적으로 올바른' 변수를 사용하는 것이 중요함을 시사한다.
2. **MNIST Digits**:
    - 고차원 이미지 데이터 $W$가 포함된 front-door 구조에서 실험을 진행하였다.
    - GAN 기반의 **ci** 방법은 $L_1 = 0.0634$를 기록하며 성공적으로 모방한 반면, **bc** 방법은 $L_1 = 0.1900$으로 성능이 낮았다. 이는 제안된 방법론이 고차원 복잡한 분포에서도 유효함을 증명한다.

## 🧠 Insights & Discussion

본 논문은 모방 학습에서 '데이터의 양'보다 '데이터의 인과적 구조'가 더 중요하다는 점을 이론적으로 규명하였다. 특히, 전문가가 사용하는 정보가 누락된 상황에서 단순히 관찰된 조건부 확률 $P(x|z)$를 학습하는 BC 방식이 왜 실패하는지를 인과 그래프를 통해 명확히 설명하였다.

**강점**: 
- latent reward라는 매우 까다로운 설정에서도 모방 가능성을 판별할 수 있는 완전한(complete) 기준을 제시하였다.
- 이론적 분석에 그치지 않고 GAN을 결합하여 고차원 실제 데이터에 적용 가능한 파이프라인을 구축하였다.

**한계 및 논의**:
- 제안된 알고리즘이 작동하려면 시스템의 인과 그래프 $G$에 대한 정성적인 지식이 선행되어야 한다. 현실 세계에서 복잡한 시스템의 인과 그래프를 정확히 그려내는 것은 여전히 어려운 과제이다.
- GAN 기반의 구현은 학습의 불안정성(collapse 등) 문제가 발생할 수 있으며, 이는 실험에서도 일부 데이터가 폐기되는 원인이 되었다.

결론적으로, 본 연구는 인과 추론을 모방 학습에 도입함으로써, 관찰되지 않은 혼란 변수가 존재하는 환경에서도 강건한(robust) 정책을 학습할 수 있는 이론적 토대를 마련하였다.

## 📌 TL;DR

본 논문은 전문가가 사용하는 정보가 학습자에게 보이지 않는 상황(Unobserved Confounders)에서 발생하는 모방 학습의 성능 저하 문제를 해결한다. 인과 그래프의 $\pi$-Backdoor 기준을 통해 모방 가능성을 판별하고, 보상이 관찰되지 않더라도 'Imitation Instrument'를 통해 전문가의 성능을 복구하는 알고리즘을 제안한다. GAN을 이용한 실험을 통해, 단순히 많은 데이터를 학습하는 BC보다 인과적 구조를 고려한 모방이 고차원 환경 및 latent reward 상황에서 압도적으로 우수함을 입증하였다.