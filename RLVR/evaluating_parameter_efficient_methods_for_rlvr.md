# Evaluating Parameter Efficient Methods for RLVR

Qingyu Yin, Yulun Wu, Zhennan Shen, Sunbowen Li, Zhilin Wang, Yanshu Li, Chak Tou Leong, Jiale Kang, Jinjin Gu (2025)

## 🧩 Problem to Solve

본 논문은 검증 가능한 보상 기반의 강화학습(Reinforcement Learning with Verifiable Rewards, RLVR) 환경에서 어떤 매개변수 효율적 미세 조정(Parameter-Efficient Fine-Tuning, PEFT) 방법론이 가장 최적인지를 탐구한다.

RLVR은 수학적 정답 여부와 같이 명확하게 검증 가능한 피드백을 통해 언어 모델의 추론 능력을 극대화하는 패러다임이다. 현재 많은 연구에서 LoRA(Low-Rank Adaptation)가 기본적으로 사용되고 있으나, RLVR의 독특한 최적화 역학(Optimization Dynamics)을 고려할 때 standard LoRA가 정말 최선의 선택인지에 대한 체계적인 분석이 부족한 상태였다. 특히 강화학습은 지도 학습(SFT)과 달리 보상 신호가 희소(Sparse)하여 업데이트가 특정 부분 집합이나 희소한 매개변수에 집중되는 경향이 있으므로, 이에 최적화된 PEFT 구조를 찾는 것이 매우 중요하다.

## ✨ Key Contributions

본 연구의 핵심 기여는 RLVR 패러다임 하에서 12가지 이상의 PEFT 방법론을 대규모로 벤치마킹하여 다음과 같은 설계 통찰을 제시한 점이다.

1. **구조적 변형(Structural Variants)의 우위 확인**: standard LoRA보다 DoRA, AdaLoRA, MiSS와 같은 구조적 변형 모델들이 일관되게 더 높은 추론 정확도를 보였으며, 특히 DoRA는 전체 매개변수 미세 조정(Full-parameter fine-tuning) 성능을 상회하기도 하였다.
2. **SVD 기반 초기화의 실패 메커니즘 규명**: PiSSA, MiLoRA와 같은 SVD 기반 초기화 전략이 RLVR에서 성능 붕괴(Collapse)를 일으키는 이유가 RLVR의 'off-principal' 업데이트 특성과 SVD의 주성분 중심 업데이트 사이의 불일치 때문임을 밝혀냈다.
3. **표현력의 하한선(Expressivity Floor) 식별**: VeRA나 Rank-1과 같은 극단적인 매개변수 축소 방법은 추론 능력을 획득하는 데 필요한 최소한의 가소성(Plasticity)을 제공하지 못해 성능이 급격히 저하됨을 확인하였다.

## 📎 Related Works

논문은 다음과 같은 관련 연구와 기존 접근 방식을 다룬다.

- **LoRA 및 변형 모델**: weight update를 저차원 행렬의 곱으로 분해하는 LoRA를 비롯하여, 크기와 방향을 분리하는 DoRA, 적응적 랭크를 사용하는 AdaLoRA 등이 언급된다.
- **GRPO (Group Relative Policy Optimization)**: 별도의 Critic 모델 없이 그룹 통계량을 통해 Advantage를 추정하는 RL 알고리즘으로, RLVR의 기본 프레임워크로 사용된다.
- **DAPO 및 Dr. GRPO**: GRPO의 엔트로피 붕괴와 학습 불안정성을 해결하기 위해 제안된 변형 알고리즘들로, 각각 Clip-Higher 전략과 길이 정규화 제거 등의 개선 사항을 포함한다.
- **RLVR의 Spectral 특성**: 기존 연구(Zhu et al., 2025)는 RLVR이 SFT와 달리 주성분(Principal components)이 아닌 off-principal 영역에서 학습된다는 점을 시사하였으며, 본 논문은 이 이론을 PEFT 분석에 적용하여 SVD 기반 방법론의 실패 원인을 설명한다.

## 🛠️ Methodology

### 전체 파이프라인 및 시스템 구조

본 연구는 DeepSeek-R1-Distill-Qwen-1.5B 및 7B 모델을 기반으로, 수학적 추론 데이터셋(DAPO-Math-17k)을 사용하여 RLVR 학습을 수행한다. 학습 과정은 SFT를 통한 Cold-start 이후, DAPO 알고리즘을 통해 강화학습을 진행하는 구조이다.

### 훈련 목표 및 손실 함수

본 논문은 GRPO의 변형인 DAPO를 표준 알고리즘으로 채택한다. GRPO의 기본 목적 함수는 다음과 같다.

$$J_{GRPO}(\theta) = \mathbb{E}_{q \sim D, \{o_i\} \sim \pi_{\theta_{old}}} \left[ \frac{1}{G} \sum_{i=1}^{G} \frac{1}{|o_i|} \sum_{t=1}^{|o_i|} \min \left( \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q, o_{i,<t})} \hat{A}_i, \text{clip} \left( \frac{\pi_\theta(o_{i,t}|q, o_{i,<t})}{\pi_{\theta_{old}}(o_{i,t}|q, o_{i,<t})}, 1 \pm \epsilon \right) \hat{A}_i \right) \right]$$

여기서 $\hat{A}_i$는 그룹 내 보상의 표준화된 Advantage이며, 다음과 같이 계산된다.
$$\hat{A}_i = \frac{R_i - \text{mean}(\{R_j\})}{\text{std}(\{R_j\})}$$

### PEFT 방법론 분류 및 구현

평가 대상이 된 PEFT 방법론은 크게 5가지 그룹으로 나뉜다.

1. **Baselines**: Full-Parameter Fine-Tuning, standard LoRA.
2. **Structural Variants**:
   - **DoRA**: 가중치의 크기(magnitude)와 방향(direction)을 분리하여 업데이트한다.
   - **AdaLoRA**: SVD와 유사한 적응형 랭크 구조를 사용하여 매개변수를 할당한다.
   - **MiSS**: 효율적인 샤드 공유 구조를 사용하여 파라미터를 최적화한다.
3. **Initialization Strategies**:
   - **PiSSA/MiLoRA**: $W_0$의 SVD 주성분을 사용하여 $A, B$ 행렬을 초기화한다.
   - **LoRA+**: $A$와 $B$ 행렬에 서로 다른 학습률($\eta_B \gg \eta_A$)을 적용한다.
4. **Efficiency-Oriented**:
   - **LoRA-FA**: 행렬 $A$를 고정하고 $B$만 학습한다.
   - **VeRA**: 무작위 투영 행렬을 고정하고 스케일링 벡터만 학습한다.
5. **Other PEFTs**: IA$^3$(활성화 벡터 스케일링), LayerNorm Tuning.

### 학습 및 추론 절차

- **대상 모듈**: 모든 선형 레이어($q, k, v, o, \text{gate, up, down proj}$)에 어댑터를 적용한다.
- **하이퍼파라미터**: Rank 32, Alpha 64, Learning Rate $1 \times 10^{-5}$를 기본으로 사용한다.
- **보상 체계**: 정답과 수학적으로 동일하면 $R=1$, 아니면 $R=0$을 부여하는 이진 보상(Binary Reward) 방식을 사용한다.

## 📊 Results

### 실험 설정

- **데이터셋**: MATH-500, AIME 24/25, AMC, HMMT, Minerva 등 수학 추론 벤치마크.
- **평가 지표**: Avg@k(k번 생성 시 평균 정확도), Pass@k(k번 중 한 번이라도 맞을 확률).

### 주요 정량적 결과

- **구조적 변형의 우수성**: DoRA는 평균 정확도 46.6%를 기록하며 standard LoRA(42.5%)와 Full-parameter(44.9%)를 모두 능가하였다. AdaLoRA(44.2%)와 MiSS(43.4%) 역시 LoRA보다 우수하였다.
- **SVD 기반 방법의 붕괴**: PiSSA는 정확도가 0.2%로 사실상 완전히 붕괴되었으며, MiLoRA(18.0%) 역시 매우 저조한 성적을 보였다.
- **표현력 하한선 확인**: VeRA(40.7%)와 IA$^3$(22.3%)는 극단적인 파라미터 축소로 인해 추론 능력이 크게 저하되었다.

### 7B 모델 스케일링 결과

DeepSeek-R1-Distill-Qwen-7B 모델에서도 동일한 경향성이 관찰되었다. DoRA와 LoRA+가 평균 55.0%의 정확도를 기록하며 standard LoRA(54.8%)보다 우수한 성능을 유지하였다.

## 🧠 Insights & Discussion

### SVD 기반 초기화 실패의 기전 분석

본 논문은 PiSSA와 MiLoRA의 실패 원인을 spectral 분석을 통해 설명한다.

- **PiSSA의 실패**: RLVR은 주성분이 아닌 'off-principal' 영역에서 학습되는 특성이 있다. 그러나 PiSSA는 강제로 주성분(Principal components) 상에서 업데이트를 수행하게 하여 RLVR의 본질적인 학습 방향과 정면으로 충돌하므로 학습이 붕괴된다.
- **MiLoRA의 실패**: MiLoRA는 초기에는 off-principal 영역에서 시작하지만, 초기 가중치 크기($\|\Delta W_0\|_F$)가 너무 작아 수치적으로 무의미해진다. 결과적으로 최적화 궤적이 다시 gradient의 최대 분산 방향인 주성분(Principal components) 영역으로 회귀하게 되어 성능이 저하된다.

### 표현력(Expressivity)의 중요성

실험 결과, RLVR은 적당한 파라미터 감소(예: LoRA-FA)는 견딜 수 있지만, 벡터 수준의 업데이트만 수행하는 극단적 압축(VeRA, IA$^3$)에는 매우 취약하다. 이는 복잡한 추론 회로를 재구성하기 위해서는 단순한 스케일링 이상의 모델 가소성(Plasticity)이 필수적임을 시사한다.

### 비판적 해석 및 한계

- **강점**: 다양한 PEFT 방법론을 RLVR이라는 최신 패러다임에 적용하여 체계적인 가이드를 제시하였다. 특히 단순 성능 비교를 넘어 spectral 분석을 통한 이론적 근거를 제시한 점이 훌륭하다.
- **한계**: 현재 연구는 DeepSeek-R1-Distill 모델 제품군에 한정되어 있으며, 학습 기간이 짧은 short-horizon 설정에서 수행되었다. 장기 학습 시의 안정성이나 다른 아키텍처에서의 일반화 여부는 추가 검증이 필요하다.

## 📌 TL;DR

본 논문은 RLVR(검증 가능한 보상 기반 강화학습) 환경에서 standard LoRA가 최적의 선택이 아님을 입증하고, **DoRA와 같은 구조적 변형 모델이 훨씬 효과적임**을 밝혔다. 특히 **SVD 기반 초기화 방법론은 RLVR의 학습 특성과 충돌하여 성능 붕괴**를 일으키며, 추론 능력을 유지하기 위해서는 **최소한의 매개변수 표현력(Expressivity Floor)이 확보**되어야 함을 경고한다. 향후 RLVR 학습 시 단순 LoRA보다는 DoRA와 같은 기하학적 인지 어댑터를 사용할 것을 권장한다.
