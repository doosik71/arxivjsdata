# Learning and Retrieval from Prior Data for Skill-based Imitation Learning

Soroush Nasiriany, Tian Gao, Ajay Mandlekar, Yuke Zhu

## 🧩 Problem to Solve

로봇이 다양한 작업을 수행하는 데 모방 학습(Imitation Learning, IL)이 유망하지만, 기존 방법은 높은 데이터 요구량과 제한된 일반화 능력으로 인해 확장성 문제가 있었습니다. 특히 새로운 작업을 학습할 때, 이전에 수집된 데이터(prior data)를 효과적으로 활용하는 방법이 중요합니다. 기존 스킬 기반 모방 학습은 prior data로부터 스킬을 학습하지만, 두 가지 주요 한계점을 가집니다:

1. **예측 불가능한 스킬 표현:** 학습된 스킬 표현 공간이 후속 정책 학습에 충분히 예측 가능하지 않아, 정책이 부적절한 스킬을 실행하는 경향이 있습니다.
2. **정책 학습에서의 prior data 미활용:** prior data가 주로 스킬 학습에만 사용되고 정책 학습에는 충분히 활용되지 않아, 적은 수의 목표 작업(target task) 데이터로 학습된 정책은 과적합(overfitting) 및 공변량 변화(covariate shift)에 취약합니다.

## ✨ Key Contributions

본 논문은 이러한 한계점을 해결하기 위해 스킬 기반 모방 학습 프레임워크인 SAILOR(Skill-Augmented Imitation Learning with prior Retrieval)를 제안하며, 다음과 같은 주요 기여를 합니다.

- **예측 가능한 스킬 표현 학습:** 스킬 인코더가 시퀀스 내 하위 궤적 간의 시간적 거리를 예측하도록 하는 보조 **시간적 예측 가능성(temporal predictability) 목표**를 도입하여, 더욱 일관되고 예측 가능한 잠재 스킬 표현을 학습합니다.
- **검색 기반 데이터 증강 메커니즘:** 목표 작업에 대한 정책 학습의 효율성을 높이기 위해, prior data에서 목표 작업과 관련된 하위 궤적을 선별적으로 검색하여 정책 감독(supervision) 범위를 확장하는 **검색 기반 데이터 증강(retrieval-based data augmentation) 절차**를 개발했습니다.
- **우수한 성능 입증:** 시뮬레이션(Franka Kitchen, CALVIN) 및 실제 로봇 조작(real-world manipulation) 환경에서 기존 모방 학습 및 오프라인 강화 학습 접근 방식보다 SAILOR가 현저히 뛰어난 성능을 보임을 입증했습니다.
- **심층 분석:** prior data, 제안된 표현 학습 목표, 검색 기반 데이터 증강이 데이터 효율적인 강력한 조작 정책 학습에 미치는 역할을 종합적으로 분석했습니다.

## 📎 Related Works

- **Prior Data를 활용한 학습:** 인간 시연(human demonstrations)을 통한 로봇 조작 학습에 대한 많은 연구가 있지만, 대부분은 작업을 독립적으로 학습하여 높은 데이터 요구량과 낮은 일반화 능력을 보입니다. 본 논문은 작업에 구애받지 않는 '놀이(play)' 데이터, 관련 작업 시연 데이터 등 대규모 오프라인 prior data를 활용하는 접근 방식에 주목합니다. R3M과 같은 사전 학습된 시각적 표현(visual representations)도 사용되지만, 도메인 전환(domain shift) 문제가 있을 수 있습니다.
- **다중 작업 모방 학습 (Multi-Task Imitation Learning):** 작업 조건부(task-conditioned), 언어 조건부(language-conditioned), 스킬 기반(skill-based) 모방 학습이 있으며, 본 논문은 유용한 스킬 표현 공간을 학습하기 위해 대규모 다중 작업 prior dataset을 활용하여 작업별 정책을 학습하는 데 초점을 맞춥니다.
- **스킬 기반 모방 학습 (Skill-based Imitation Learning):** 감각-운동 데이터의 시간적으로 추상적인 표현(스킬)을 학습하여 더 효과적인 모방 학습을 가능하게 하는 방법입니다. 시연을 하위 궤적(sub-trajectories)으로 분할하여 스킬 표현을 학습하며, 최근에는 변이형 오토인코더(Variational Autoencoder, VAE) 기반의 고정 길이 하위 궤적 인코딩이 유망합니다. SAILOR는 이러한 기존 접근 방식([6, 7])에 예측 가능성 목표와 검색 메커니즘을 추가합니다.

## 🛠️ Methodology

SAILOR는 두 가지 주요 단계로 구성됩니다:

### 1. 스킬 학습 (Skill Learning)

Prior data $D_{prior}$를 사용하여 고정 길이 하위 궤적 $\tau = \{o_0, a_0, o_1, \dots, o_{H-1}, a_{H-1}, o_H\}$의 잠재 스킬 표현 공간 $Z \subset \mathbb{R}^d$를 학습합니다.

- **VAE 기반 인코딩:** 하위 궤적 $\tau$를 LSTM 인코더 $q_{\phi}$를 통해 잠재 스킬의 가우시안 분포로 인코딩하고, LSTM 디코더 $p_{\psi}$를 통해 잠재 $z$와 관측 $o_t$를 재구성된 행동 $\hat{a}_t$로 디코딩합니다. 학습된 사전 분포(prior) $p_{\theta}$는 유사한 시작/종료 관측을 가진 하위 궤적이 유사한 잠재 표현을 갖도록 장려합니다.
- **VAE 손실:**
  $$L_{VAE}(\phi, \psi, \theta) = -E_{z \sim q_{\phi}(z|\tau)} \left[ \sum_{t=0}^{H-1} \log p_{\psi}(a_t|z, o_t) \right] + \beta \cdot D_{KL}(q_{\phi}(z|\tau)||p_{\theta}(z|o_0, o_H))$$
- **시간적 예측 가능성 (Temporal Predictability, TP) 목표:** 동일한 궤적에서 $t$ 타임스텝만큼 떨어진 두 하위 궤적 $\tau_1, \tau_2$가 주어졌을 때, 해당 스킬 평균 임베딩을 통해 $t$를 예측하는 모델 $m_{\omega}$를 학습합니다.
- **TP 손실:**
  $$L_{TP}(\omega, \phi) = \left( m_{\omega} \left( \mu(q_{\phi}(z|\tau_1)), \mu(q_{\phi}(z|\tau_2)) \right) - t \right)^2$$
- **전체 스킬 손실:** VAE와 TP 목표의 가중 조합으로 구성됩니다.
  $$L_{Skill}(\phi, \psi, \theta, \omega) = L_{VAE}(\phi, \psi, \theta) + \alpha L_{TP}(\omega, \phi)$$

### 2. 검색 기반 정책 학습 (Retrieval-based Policy Learning)

사전 학습된 스킬 모델을 사용하여 목표 작업 $T$를 위한 정책 $\pi$를 학습합니다. 정책은 다음으로 실행할 스킬 $z$를 출력합니다.

- **정책 아키텍처:** $F$개 관측 기록을 잠재 스킬 $z$로 매핑하는 LSTM 정책을 사용합니다.
- **검색 기반 데이터 증강:**
  1. $D_{prior}$와 $D_{target}$에서 무작위로 샘플링된 하위 궤적의 스킬 임베딩 $Z_{prior}$ 및 $Z_{target}$을 얻습니다.
  2. $Z_{prior}$와 $Z_{target}$ 간의 모든 쌍별 $L_2$ 거리를 계산합니다.
  3. $Z_{prior}$의 각 스킬 임베딩 $z_i$에 대해, $Z_{target}$에서 가장 가까운 스킬 임베딩을 찾습니다.
  4. 거리가 가장 작은 상위 $n$개의 $D_{prior}$ 하위 궤적을 검색하여 $D_{ret}$을 구성합니다.
- **정책 학습:** $D_{ret}$과 $D_{target}$의 통합 데이터셋에 대해 정책을 학습합니다. $D_{ret}$ 데이터에 가중치 $\gamma$를 부여하는 행동 복제(behavioral cloning) 손실을 사용합니다.
- **스킬 모델 미세 조정:** 정책 학습과 동시에, $D_{target}$으로 스킬 모델을 계속 미세 조정합니다.
- **실행:** 정책이 $z$를 예측하면, 폐루프(closed-loop) 스킬 디코더 $p_{\psi}$가 $H$ 타임스텝 동안 $z$를 일련의 행동으로 디코딩합니다.

## 📊 Results

- **시뮬레이션 환경 (Franka Kitchen, CALVIN):**
  - SAILOR는 6가지 최신 기준 모델(BC-RNN, BC-RNN (FT), BC-RNN (R3M), IQL, IQL (UDS), FIST)을 크게 능가하며, 평균 작업 성공률 89.0%를 달성했습니다. 다음으로 우수한 BC-RNN (FT)보다 12.7%p 높은 성능입니다.
  - CALVIN 작업에서 BC-RNN (R3M)은 사전 학습된 시각적 표현의 일반화 능력 한계로 인해 저조한 성능을 보였습니다.
  - 오프라인 강화 학습 기준 모델은 Franka Kitchen에서는 유망했지만, 더 어려운 CALVIN 작업에서는 어려움을 겪었습니다.
  - FIST는 SAILOR와 동일한 스킬 모델을 사용함에도 불구하고 현저히 낮은 성능을 보였는데, 이는 FIST의 준-매개변수적(semi-parametric) 정책의 한계로 분석됩니다.
- **어블레이션 연구 (CALVIN):**
  - 시간적 예측 가능성($L_{TP}$)과 검색 메커니즘 중 어느 하나라도 제거하면 최종 성능이 크게 저하되어, 때로는 단순한 BC-RNN (FT)보다도 나빠졌습니다.
  - prior data 전체를 사용하는 "All Retrieval"은 관련 없는 행동으로 인해 성능이 최적화되지 못했습니다.
  - prior data를 사용하지 않는 "No Prior Data" 버전도 스킬 기반 프레임워크가 제공하는 시간적 추상화 덕분에 BC-RNN보다는 우수했지만, SAILOR보다는 성능이 낮았습니다.
  - prior data 양을 늘릴수록 성능이 향상되었으며, prior data와 target data 간 환경 불일치(environmental mismatch)가 있는 경우에도 효과적으로 작동했습니다.
- **실제 환경 (Real World Kitchen):**
  - Real-Cook 및 Real-Cook-Pan 작업에서 SAILOR는 BC-RNN (FT)보다 성공률에서 각각 73.3% vs. 23.3% 및 76.7% vs. 46.7%로 크게 앞섰습니다.
  - Real-Breakfast 작업에서는 두 방법이 76.7%로 유사한 성능을 보였습니다.
  - 실제 환경에서도 "No Prior" 어블레이션은 BC-RNN (FT)보다 우수했습니다.

## 🧠 Insights & Discussion

- SAILOR는 예측 가능한 스킬 표현과 효과적인 prior data 활용을 통해 적은 수의 목표 작업 시연만으로도 로봇에 새로운 행동을 가르칠 수 있는 데이터 효율적인 방법을 제공합니다.
- 스킬 기반 학습 프레임워크가 제공하는 시간적 추상화는 prior data 없이도 일정 수준의 이점을 제공하지만, prior data를 효과적으로 활용하는 것이 성능 향상에 결정적입니다.
- 검색 메커니즘은 prior data의 다중 모드(multi-modal) 분포에서 목표 작업과 관련된 데이터를 선별하여 정책 학습의 품질을 높이는 데 중요합니다.
- **한계점:**
  - 대규모 다중 작업 prior data 확보의 어려움과 높은 비용. 다양한 작업으로의 확장성이 추가 연구가 필요합니다.
  - BC-RNN보다 더 많은 손실 함수와 네트워크를 사용하므로 계산 비용이 더 높습니다.
  - prior data와 target task 데이터 간의 도메인 전환(domain shift)이 크지 않은 환경에 초점을 맞추었으며, 미시연 객체(unseen objects)에 대한 일반화는 추가 연구가 필요합니다.

## 📌 TL;DR

**SAILOR**는 데이터 비효율적이고 일반화가 취약한 기존 모방 학습의 한계를 극복하기 위한 스킬 기반 프레임워크입니다. 이 방법은 prior data를 활용하여 (1) **시간적 예측 가능성 목표**와 VAE를 통해 **예측 가능한 스킬 표현**을 학습하고, (2) 스킬 유사도 기반 **검색 메커니즘**으로 prior data 중 목표 작업에 **관련된 데이터만 선별**하여 정책 학습을 증강합니다. 시뮬레이션 및 실제 로봇 조작 환경에서 SAILOR는 최신 모방 학습 및 오프라인 강화 학습 방법보다 **현저히 우수한 데이터 효율성과 견고한 성능**을 입증했습니다.
