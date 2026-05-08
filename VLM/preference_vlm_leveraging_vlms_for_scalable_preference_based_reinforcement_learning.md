# Preference VLM: Leveraging VLMs for Scalable Preference-Based Reinforcement Learning

Udita Ghosh, Dripta S. Raychaudhuri, Jiachen Li, Konstantinos Karydis, Amit Roy-Chowdhury (2025)

## 🧩 Problem to Solve

강화학습(Reinforcement Learning, RL)에서 에이전트를 원하는 방향으로 정렬시키기 위한 보상 함수(Reward Function)를 설계하는 것은 매우 어려운 과제이다. 특히 복잡하고 긴 호흡의 작업(long-horizon tasks)에서는 보상 함수를 수동으로 설계하는 데 많은 전문성과 공수가 소모된다. 이를 해결하기 위해 사람이 두 가지 행동 궤적(trajectory) 중 어떤 것이 더 나은지 선택하는 Preference-based RL이 제안되었으나, 이는 여전히 막대한 양의 인간 피드백(human feedback)을 요구한다는 확장성(scalability) 문제가 존재한다.

최근 Vision-Language Models(VLMs)를 이용하여 자연어 설명만으로 제로샷(zero-shot) 보상을 생성하려는 시도가 있었으나, VLM의 출력은 종종 노이즈가 많고 로봇 조작과 같은 세밀한 작업(fine-grained tasks)에 필요한 정밀도가 부족하다는 한계가 있다. 따라서 본 논문의 목표는 VLM의 확장성과 인간 피드백의 정밀함을 결합하여, 인간의 개입을 최소화하면서도 높은 성능의 보상 함수를 학습하는 프레임워크인 **PrefVLM**을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 VLM을 완전한 보상 생성기가 아닌, '거친 수준의 선호도 레이블 생성기(coarse preference label generator)'로 활용하고, VLM이 불확실해하는 일부 샘플에 대해서만 선택적으로 인간의 피드백을 받는 것이다. 이를 위해 다음과 같은 세 가지 핵심 설계를 도입하였다.

1. **VLM 기반의 선호도 생성 및 필터링**: VLM을 통해 초기 선호도 레이블을 생성하고, 학습 과정에서 보상 모델의 예측값과 VLM 레이블 간의 불일치(KL divergence)를 측정하여 신뢰할 수 있는 데이터와 불확실한 데이터를 구분한다.
2. **데이터 효율적인 VLM 적응(Adaptation)**: VLM의 사전 학습 데이터와 실제 환경 간의 도메인 간극(domain gap)을 줄이기 위해, 파라미터 효율적인 미세 조정(PEFT)과 함께 자기지도 학습 기반의 **Inverse Dynamics Loss**를 도입하여 환경의 역학 관계를 학습시킨다.
3. **선택적 인간 피드백(Selective Human Feedback)**: 모든 데이터에 대해 피드백을 받는 대신, VLM과 보상 모델 모두가 판단하기 어려워하는 '불확실한 샘플'에 대해서만 제한된 예산 내에서 인간의 레이블을 획득하여 효율성을 극대화한다.

## 📎 Related Works

- **보상 함수 설계**: 전통적인 수동 설계 방식은 확장이 어렵다. Inverse RL(IRL)은 전문가 시연(expert demonstrations)이 필요하지만, 고품질 데이터를 수집하는 비용이 매우 높다는 한계가 있다.
- **Foundation Models as Rewards**: 최근 MLLM이나 VLM(CLIP 등)을 이용해 자연어 설명으로부터 보상을 직접 유도하는 연구가 진행되었다. 그러나 이러한 방식은 VLM의 출력이 정밀하지 못해 로봇 제어와 같은 세밀한 작업에서는 성능이 떨어진다는 문제가 있다.
- **Preference-based RL**: PEBBLE과 같은 프레임워크는 인간의 상대적 판단을 이용해 보상을 학습하지만, 여전히 많은 양의 인간 피드백이 필요하다.
- **Noisy Label Learning**: 본 논문은 RIME과 같은 견고한 학습 기법에서 영감을 받아, 모델이 초기에 일반적인 패턴을 먼저 학습하고 나중에 노이즈에 오버피팅된다는 특성을 이용하여 샘플을 필터링한다.

## 🛠️ Methodology

### 전체 파이프라인

PrefVLM은 정책($\pi_\phi$), 보상 모델($r_\theta$), VLM의 적응 레이어($G$)를 반복적으로 업데이트하는 구조를 가진다. VLM이 궤적 쌍에 대한 선호도를 생성하면, 이를 필터링하여 정제된 데이터와 인간의 확인이 필요한 데이터를 구분하고, 이를 통해 보상 모델과 VLM을 동시에 학습시킨다.

### VLM as a Preference Feedback Model

VLM(예: CLIP 기반 모델)은 이미지 인코더 $F_I$와 텍스트 인코더 $F_L$로 구성된다. 특정 상태의 이미지 $o_t$와 작업 설명 $l$에 대한 보상은 다음과 같이 코사인 유사도로 계산된다.
$$r^{vlm}_t = \frac{\langle F_L(l), F_I(o_t) \rangle}{\|F_L(l)\| \cdot \|F_I(o_t)\|}$$
두 궤적 세그먼트 $\sigma_0, \sigma_1$이 주어졌을 때, 각 세그먼트의 보상 합(return) $R_i = \sum_{t=0}^T r^{vlm}_t$를 비교하여 더 높은 값을 가진 쪽을 선호하는 것으로 레이블 $\tilde{y}$를 부여한다.

### VLM Adaptation

도메인 간극을 줄이기 위해 VLM의 인코더 위에 학습 가능한 레이어 $G_L$과 $G_I$를 추가하여 적응된 임베딩을 생성한다.

1. **선호도 손실(Preference Loss)**: 소량의 인간 피드백을 사용하여 Bradley-Terry 모델 기반의 교차 엔트로피 손실을 최소화한다.
   $$L_{pref} = -\mathbb{E}_{(\sigma_0, \sigma_1, y) \sim D} [y(0)P_\theta[\sigma_0 \succ \sigma_1] + y(1)P_\theta[\sigma_1 \succ \sigma_0]]$$
2. **Inverse Dynamics Loss**: VLM이 환경의 물리적 변화를 이해하도록, 현재 관측 $o_t$와 다음 관측 $o_{t+1}$로부터 수행된 행동 $a_t$를 예측하는 자기지도 학습을 수행한다.
   $$\min \|f(G_I \circ F_I(o_t), G_I \circ F_I(o_{t+1})) - a_t\|^2$$
   여기서 $f$는 선형 레이어이다.

### Noise Mitigation and Human Feedback

VLM이 생성한 레이블의 노이즈를 제거하기 위해 KL divergence를 이용한 필터링 전략을 사용한다.

- **Clean Samples ($D_{\tau^l}$)**: $\text{KL}(\tilde{y} \| P_\theta)$가 하한 임계값 $\tau_{lower}$보다 낮은 샘플들이다. 보상 모델 학습에 그대로 사용된다.
- **Noisy Samples ($D_{\tau^u}$)**: KL divergence가 상한 임계값 $\tau_{upper}$보다 높은 샘플들이다. 레이블을 반전(flip)시켜 사용함으로써 노이즈의 영향을 상쇄한다.
- **Uncertain Samples ($D_h$)**: 두 임계값 사이에 위치한 샘플들로, 이들 중 일부만 선택하여 **인간의 직접적인 피드백**을 받는다. 이 데이터는 VLM의 미세 조정과 보상 모델 학습 모두에 사용된다.

## 📊 Results

### 실험 설정

- **데이터셋 및 환경**: Meta-World의 5가지 조작 작업(Door Open, Drawer Close, Drawer Open, Window Open, Window Close)을 사용하였다.
- **기준선(Baselines)**: VLM-as-reward, PEBBLE (1000 & 2000 feedback), VLM-pref-reward, PrefVLM w/o selection.
- **평가 지표**: 작업 성공률(Success Rate) 및 환경 상호작용 단계(Environment Steps).

### 주요 결과

1. **피드백 효율성**: PrefVLM은 1,000개의 인간 피드백만으로도 2,000개의 피드백을 사용한 PEBBLE과 비슷하거나 더 높은 성공률을 달성하였다. 즉, 인간의 어노테이션 비용을 약 $2\times$ 감소시켰다.
2. **제로샷 VLM의 한계**: VLM을 단순히 보상 모델로 사용하거나($\text{VLM-as-reward}$), VLM의 선호도만으로 보상 모델을 학습시킨 경우($\text{VLM-pref-reward}$) 유의미한 성공률을 거두지 못했다. 이는 VLM의 원시 출력이 매우 노이즈가 많음을 시사한다.
3. **지식 전이(Knowledge Transfer)**: 한 작업(예: Door Close)에서 적응된 VLM을 유사한 다른 작업(예: Drawer Close)에 적용했을 때, 인간 피드백을 500개만 사용하고도 PEBBLE(2000개) 수준의 성능을 냈다. 이는 피드백 요구량을 최대 $4\times$까지 추가로 줄일 수 있음을 보여준다.

## 🧠 Insights & Discussion

**강점 및 분석**

- **Inverse Dynamics Loss의 중요성**: 분석 결과, Inverse Dynamics Loss 없이 Contrastive Loss만 사용했을 때 초기에는 성능이 오르지만 시간이 지나면 하락하는 경향이 발견되었다. 이는 정책이 진화하며 데이터 분포가 변할 때, 환경의 역학을 학습한 VLM만이 이에 안정적으로 적응할 수 있음을 의미한다.
- **선택적 샘플링의 효과**: 단순히 무작위로 인간 피드백을 받는 것보다, KL divergence를 통해 불확실한 샘플을 골라 피드백을 받는 것이 보상 모델의 수렴 속도와 정확도를 유의미하게 향상시켰다.

**한계 및 비판적 해석**

- **인간 피드백의 초기 의존성**: VLM 단독으로는 성능이 매우 낮기 때문에, 결국 초기 단계에서는 어느 정도의 인간 피드백이 필수적이다. 완전히 인간이 배제된 자동화된 RL 시스템과는 거리가 있다.
- **계산 비용**: VLM의 적응 레이어를 학습시키고 KL divergence를 계속 계산하며 필터링하는 과정이 추가되므로, 단순한 RL 학습보다 연산 오버헤드가 발생할 수 있다.

## 📌 TL;DR

본 논문은 VLM의 풍부한 시각-언어 지식을 활용하되, 그 노이즈를 제어하기 위해 **선택적 인간 피드백**과 **Inverse Dynamics Loss**를 결합한 **PrefVLM** 프레임워크를 제안한다. 이를 통해 인간의 피드백 양을 절반으로 줄이면서도 기존의 Preference-based RL(PEBBLE 등)과 대등하거나 더 우수한 성능을 달성하였으며, 특히 유사 작업 간의 지식 전이를 통해 피드백 요구량을 획기적으로 낮출 수 있음을 입증하였다. 이 연구는 로봇 제어와 같이 데이터 수집 비용이 높은 분야에서 VLM을 효율적인 보상 가이드로 활용하는 실질적인 방법을 제시한다.
