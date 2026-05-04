# Visually Robust Adversarial Imitation Learning from Videos with Contrastive Learning

Vittorio Giammarino, James Queeney, and Ioannis Ch. Paschalidis (2024)

## 🧩 Problem to Solve

본 논문은 전문가의 비디오 데이터로부터 행동을 학습하는 **Visual Imitation from Observations (V-IfO)** 문제에서 발생하는 **시각적 불일치(Visual Mismatch)** 문제를 해결하고자 한다.

일반적인 V-IfO 알고리즘들은 전문가와 학습 에이전트가 동일한 환경에서 동작한다는 가정을 전제로 한다. 그러나 실제 환경에서는 조명, 배경, 색상 등이 서로 다른 경우가 많으며, 이러한 시각적 차이는 기존의 end-to-end 알고리즘들이 전문가의 행동을 정확히 모방하는 것을 방해하는 심각한 장벽이 된다.

따라서 본 논문의 목표는 전문가와 에이전트의 도메인 간에 시각적 차이가 존재하더라도, 작업 수행에 필수적인 핵심 정보만을 추출하여 강건하게 모방 학습을 수행할 수 있는 알고리즘을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Contrastive Learning(대조 학습)**과 **Data Augmentation(데이터 증강)**을 결합하여, 도메인에 무관한(Domain-invariant) 잠재 공간(Latent Space) $Z$를 구축하는 것이다.

C-LAIfO(Contrastive Latent Adversarial Imitation from Observations)라고 명명된 이 알고리즘은 관측 데이터에서 작업 완료와 관련된 정보(Goal-completion information)는 유지하고, 시각적 방해 요소(Visual distractors)는 제거함으로써 도메인 불변 특징을 학습한다. 이렇게 구축된 견고한 잠재 공간 내에서만 **Adversarial Imitation Learning (AIL)**을 수행함으로써 시각적 불일치 문제에 강건한 모방 학습을 구현한다.

## 📎 Related Works

### 기존 연구 및 한계
1. **Imitation from Observation (IfO):** 전문가의 액션 정보 없이 비디오만으로 학습하는 방식으로, PatchAIL이나 LAIfO 같은 최신 알고리즘들이 존재한다. 하지만 이들은 전문가와 에이전트가 동일한 환경에 있다는 가정을 가진다.
2. **Domain-adaptive/Cross-domain IL:** 도메인 불일치를 해결하려는 시도들이 있었으나, 많은 경우 보상 함수 학습과 정책 학습을 분리하는 순차적(Sequential) 접근 방식을 취하거나, 비용이 많이 드는 생성 모델(Generative models)을 사용하여 도메인 간 이미지를 변환하는 방식을 사용한다.
3. **End-to-end Mismatch solutions:** DisentanGAIL과 같은 연구는 도메인 불변 특징을 추출하려 했으나, 주로 보상 추론 단계에서만 이를 적용했다.

### C-LAIfO의 차별점
C-LAIfO는 완전한 end-to-end 모델-프리(Model-free) 접근 방식을 취한다. 특히, 단순히 보상 함수 추론뿐만 아니라 전체 AIL 파이프라인(보상 추론 및 RL 단계 모두)을 학습된 도메인 불변 특징 공간 $Z$ 위에서 수행한다는 점에서 기존 연구들과 차별화되며, 이는 성능 향상으로 이어진다.

## 🛠️ Methodology

### 전체 시스템 구조
C-LAIfO는 관측값 $x$를 잠재 공간 $z$로 매핑하는 인코더 $\phi_\delta$를 학습하고, 이 공간 위에서 판별자(Discriminator)와 정책(Policy)을 학습시키는 구조를 가진다.

### 1. 잠재 공간에서의 Adversarial Imitation Learning
에이전트와 전문가의 잠재 상태 전이 $(z, z')$를 저장하는 리플레이 버퍼를 사용한다. 판별자 $D_\chi$는 다음과 같은 목적 함수를 최적화하여 에이전트와 전문가의 상태 방문 분포 차이를 구분한다.

$$\max_{\chi} \mathbb{E}_{(z, z') \sim B_E} [\log(D_\chi(z, z'))] + \mathbb{E}_{(z, z') \sim B} [\log(1 - D_\chi(z, z'))]$$

여기서 학습된 판별자를 통해 보상 함수 $r_\chi(z, z') = -\log(1 - D_\chi(z, z'))$를 유도하고, 이를 RL 단계의 보상으로 사용한다.

### 2. 인코더 및 크리틱 학습 (Critic and Encoder Training)
인코더 $\phi_\delta$는 관측 시퀀스를 잠재 벡터 $z$로 변환한다. 크리틱 $Q_\psi$와 인코더 $\phi_\delta$는 다음의 손실 함수를 통해 공동 학습된다.

$$\min_{\psi, \delta} \mathbb{E}_{(\tilde{z}, a, \tilde{z}') \sim B} [ (Q_\psi(\tilde{z}, a) - sg(y))^2 ] + \mathbb{E}_{\tilde{z} \sim B} [L(\tilde{z})]$$

여기서 $sg(\cdot)$는 stop-gradient를 의미하며, 타겟 값 $y$는 다음과 같이 계산된다.
$$y = r_\chi(z, z') + \gamma \min_{k=1,2} Q_{\bar{\psi}_k}(\tilde{z}', a')$$

이 과정에서 $L(\tilde{z})$는 도메인 불변성을 확보하기 위한 **Contrastive Loss**이다.

### 3. Contrastive Loss 및 데이터 증강
데이터 증강 함수 $\text{aug}(\cdot)$를 통해 하나의 관측 시퀀스에서 두 개의 뷰(Positive pairs) $\tilde{z}_\delta(i), \tilde{z}_\delta(j)$를 생성한다. 이후 **InfoNCE (Information Noise-Contrastive Estimation) loss**를 사용하여 동일한 작업 정보를 가진 샘플끼리는 가깝게, 다른 샘플과는 멀게 배치한다.

$$L(z_\delta) = -\log \frac{\exp(\text{sim}(\tilde{z}_\delta(i), \tilde{z}_\delta(j)) / \eta)}{\sum_{k=1, 2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(\tilde{z}_\delta(i), \tilde{z}_\delta(k)) / \eta)}$$

여기서 $\text{sim}(u, v)$는 코사인 유사도를 의미하며, $\eta$는 온도 파라미터이다. 데이터 증강은 시각적 불일치의 종류(조명, 색상 등)에 맞춰 선택적으로 적용된다.

## 📊 Results

### 실험 설정
- **데이터셋:** 고차원 연속 로봇 제어 태스크 및 Adroit 플랫폼의 Dexterous Manipulation 태스크.
- **비교 대상:** LAIfO, PatchAIL (데이터 증강 추가), DisentanGAIL.
- **평가 지표:** 에피소드당 평균 리턴(Average Return).

### 주요 결과
1. **Ablation Study:**
   - **Contrastive Loss:** InfoNCE loss가 BYOL보다 도메인 불일치가 심한 환경에서 더 높은 성능을 보였다.
   - **Gradient Backpropagation:** 크리틱 $Q$에서 인코더 $\phi$로의 그래디언트 전파가 없으면 모방 학습 자체가 불가능함을 확인하여, 이 단계가 작업 관련 정보를 잠재 공간에 임베딩하는 데 필수적임을 입증했다.
   - **Data Augmentation:** 불일치 유형에 맞는 맞춤형 증강(Mismatch-informed augmentation)이 일반적인 증강이나 증강이 없는 경우보다 압도적으로 성능이 좋았다.

2. **시각적 불일치 상황에서의 성능:**
   - 'Light' 및 'Full' 불일치 설정 모두에서 C-LAIfO가 모든 베이스라인(LAIfO, PatchAIL, DisentanGAIL)보다 월등히 높은 리턴을 기록했다.
   - **PCA 분석:** C-LAIfO만이 시각적 방해 요소를 효과적으로 필터링하고, 도메인에 관계없이 동일한 목표 정보를 가진 데이터들을 하나의 클러스터로 묶는 것을 확인했다.

3. **Adroit Dexterous Manipulation:**
   - 희소 보상(Sparse reward) 환경에서 C-LAIfO로 학습된 보상 $r_\chi$를 환경 보상 $R$과 결합($R_{tot} = R + r_\chi$)했을 때, LAIfO보다 훨씬 효율적으로 학습이 진행됨을 보였다.

## 🧠 Insights & Discussion

### 강점
C-LAIfO는 단순히 이미지 도메인을 변환하는 것이 아니라, 학습 과정에서 **"무엇이 중요한 정보($\bar{X}$)이고 무엇이 방해 요소($\hat{X}$)인가"**를 대조 학습과 크리틱의 피드백을 통해 동시에 학습한다. 특히 잠재 공간 전체를 도메인 불변적으로 구축하여 RL 루프 전체를 그 위에서 수행하게 한 설계가 성능 향상의 핵심이다.

### 한계 및 향후 과제
가장 큰 한계는 **데이터 증강 함수 $\text{aug}(\cdot)$에 대한 의존성**이다. 실험 결과에서 보듯, 불일치 유형에 맞지 않는 증강을 사용하면 성능이 급격히 저하된다. 즉, 사람이 수동으로 설계한 증강 전략에 의존하고 있다는 점이 문제다.

논문은 이에 대한 해결책으로 **생성 모델(Generative models)**을 이용한 자동 데이터 증강이나, 증강에 덜 의존적인 새로운 보조 손실 함수(Auxiliary loss)를 탐색하는 방향을 제시한다. 또한, 시뮬레이션 환경을 넘어 실제 하드웨어에서의 검증이 필요하다.

## 📌 TL;DR

본 논문은 전문가 비디오와 에이전트 환경 사이의 시각적 차이(조명, 배경 등)가 있을 때 발생하는 모방 학습의 어려움을 해결하기 위해 **C-LAIfO**를 제안한다. 이 알고리즘은 **Contrastive Learning**과 **맞춤형 데이터 증강**을 통해 도메인에 무관한 잠재 공간을 구축하고, 그 공간 내에서 **Adversarial Imitation Learning**을 수행한다. 실험을 통해 시각적 불일치 상황에서도 강건한 성능을 입증했으며, 특히 희소 보상 환경의 복잡한 로봇 조작 태스크에서 전문가 비디오의 활용 가치를 높였다.