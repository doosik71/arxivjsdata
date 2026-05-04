# Staged Voxel-Level Deep Reinforcement Learning for 3D Medical Image Segmentation with Noisy Annotations

YuYang Fu, XiuZhen Guo, Ji Shi (2026)

## 🧩 Problem to Solve

본 논문은 3D 의료 영상 분할(Medical Image Segmentation) 과정에서 발생하는 **노이즈 섞인 어노테이션(Noisy Annotations)** 문제를 해결하고자 한다. 의료 영상의 경우 장기의 경계 구조가 복잡하고, 판독자 간의 주관적 차이(inter-observer variability)로 인해 불명확한 해부학적 경계가 발생하며, 이는 결과적으로 학습 데이터에 라벨 노이즈를 유발한다.

이러한 노이즈는 딥러닝 모델이 잘못된 세만틱 연관성을 학습하게 하여 모델의 일반화 성능을 심각하게 저하시킨다. 특히 기존의 노이즈 제거 방식들은 신뢰할 수 없는 샘플을 필터링하여 제거하는 방식을 취하는데, 이는 불완전한 어노테이션 속에 포함된 유용한 세만틱 정보까지 함께 손실시킬 위험이 있다. 따라서 본 연구의 목표는 사람이 사전 지식을 바탕으로 라벨 오류를 수정하는 과정을 모방하여, 데이터 제거 없이 스스로 라벨을 정제하고 강건한 분할 성능을 내는 **SVL-DRL(Staged Voxel-Level Deep Reinforcement Learning)** 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 의료 영상 분할 문제를 **동적인 의사결정 과정**으로 재정의하고, 강화학습(Reinforcement Learning, RL)을 통해 입력 상태를 반복적으로 정제하는 것이다. 주요 기여 사항은 다음과 같다.

1. **단계적 강화학습 프레임워크(Staged RL Framework):** 노이즈의 영향을 완화하기 위해 Warmup, Transition, Full RL의 세 단계로 구성된 학습 전략을 제안하여 학습의 안정성을 확보하였다.
2. **Voxel-level Asynchronous Advantage Actor-Critic (vA3C):** 각 복셀(Voxel)을 하나의 독립적인 에이전트로 간주하는 vA3C 모듈을 도입하였다. 이를 통해 각 에이전트가 자신의 상태 표현을 동적으로 정제함으로써 라벨 노이즈의 영향을 직접적으로 완화한다.
3. **새로운 액션 공간 및 보상 함수 설계:** 강화학습 에이전트가 수행할 액션 공간을 정의하고, Dice 계수와 공간적 연속성(Spatial Continuity) 메트릭을 결합한 복합 보상 함수를 설계하여 해부학적으로 타당한 분할 결과를 유도하였다.

## 📎 Related Works

논문에서는 노이즈 라벨 학습과 강화학습 관련 기존 연구를 네 가지 범주로 나누어 설명한다.

1. **노이즈 전이 행렬(Noise Transition Matrix, NTM) 기반 방법:** 깨끗한 라벨과 노이즈 라벨 간의 전이 확률을 모델링하여 손실 함수를 수정하지만, 복잡한 노이즈 패턴 하에서는 NTM을 정확히 추정하기 어렵다는 한계가 있다.
2. **샘플 선택 기반 방법 (예: Co-Teaching):** 두 개의 네트워크가 서로 손실 값이 작은 샘플을 선택하여 노이즈를 필터링한다. 하지만 유용한 정보까지 버려질 수 있다.
3. **라벨 수정 및 디노이징 전략 (예: Confident Learning, JCAS):** 픽셀 간의 관계나 분포 추정을 통해 직접적으로 라벨을 수정한다.
4. **협력 학습 및 지식 증류(Knowledge Distillation) 프레임워크:** 듀얼 모델 협력을 통해 라벨을 정제한다.

본 논문은 이러한 기존 방식들이 주로 '필터링'이나 '정적 수정'에 의존하는 것과 달리, **강화학습의 탐색(Exploration) 능력과 해부학적 사전 지식**을 결합하여 입력 상태를 동적으로 수정함으로써 노이즈에 대응한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 전체 파이프라인 및 아키텍처

SVL-DRL은 **Swin UNETR**를 베이스라인 세그멘테이션 모델로 사용하며, 공유 인코더와 세 개의 독립적인 디코더(Segmentation, Value, Strategy)로 구성된다.

### 훈련 단계 (Staged Training)

학습의 발산(Divergence)을 막기 위해 세 단계를 거친다.

1. **Warmup Stage:** 세그멘테이션 네트워크만 활성화하여 기본적인 분할 능력을 학습시킨다.
    $$L_{\text{Warmup stage}} = L_{\text{Dice Loss}} = 1 - \text{Dice}(P, G)$$
2. **Transition Stage:** 가치 네트워크(Value Network)를 활성화하여 상태 평가 능력을 학습시킨다.
    $$L_{\text{Transition stage}} = (1 - \lambda)L_{\text{Dice Loss}} + \lambda L_{\text{value}}$$
    여기서 $L_{\text{value}}$는 다단계 할인 수익(Multi-step discount return) $R^{(t)}$와 예측 가치 $V(s^{(t)})$의 차이의 제곱으로 계산된다.
3. **Full RL Stage:** 정책 네트워크(Policy Network)를 포함한 모든 모듈을 활성화하여 동적 정제를 수행한다.
    $$L_{\text{Full RL stage}} = (1 - \alpha - \beta)L_{\text{Dice Loss}} + \alpha L_{\text{value}} + \beta L_{\text{policy}}$$

### vA3C 모듈 및 액션 공간

각 복셀 $i$를 독립적인 에이전트로 취급하며, 에이전트는 다음과 같은 세 가지 액션 $\delta$ 중 하나를 선택한다.

- **Action 0:** 아무것도 하지 않음 (Keep original).
- **Action 1:** 조직/병변 강화 (Enhance). $I_{\text{new}} = \min(\max(I_{\text{orig}} \times (1.0 + 0.3\epsilon), 0.0), 1.0)$
- **Action 2:** 조직/병변 약화 (Weaken). $I_{\text{new}} = \min(\max(I_{\text{orig}} \times (1.0 - 0.3\epsilon), 0.0), 1.0)$
($\epsilon \sim U(0, 1)$는 랜덤 스케일링 계수이다.)

### 보상 함수 (Reward Function)

보상 함수 $r^{(t)}$는 세그멘테이션의 개선 정도와 해부학적 제약 조건을 결합하여 정의된다.
$$r^{(t)} = \Delta \text{Dice}(f^{(t)}, f^{(t-1)}, G) + C(f^{(t)})$$

- $\Delta \text{Dice}$: 이전 단계 대비 현재 단계의 Dice 계수 증가량이다.
- $C(f)$: 해부학적 타당성을 평가하는 제약 조건으로, 연결 성분(Connected Components)의 개수 $N_{cc}$와 픽셀 값의 공간적 변화량(Gradient magnitude)의 합으로 정의된다.
  $$C(f) = \max(N_{cc}(f) - 1, 0) + \sum_{i,j} |\nabla f_{i,j}|$$

## 📊 Results

### 실험 설정

- **데이터셋:** LA(Left Atrium), Pancreas-CT, BraTS 2021.
- **노이즈 유형:** SFDA-Noise (Source-Free Domain Adaptation 기반), MT-Noise (Morphological Transformation 기반). 노이즈 비율은 50%로 설정되었다.
- **지표:** Dice, IoU, HD95, ASD.
- **비교 대상:** Loss Correction, Co-Teaching, MTCL, ADELE, JCAS, RSF-Assisted, CLCS 등 SOTA 방법론.

### 주요 결과

1. **정량적 성능:** SVL-DRL은 모든 데이터셋과 노이즈 설정에서 SOTA 성능을 달성하였다. Dice와 IoU 점수에서 평균 3% 이상의 향상을 보였다.
2. **강건성 증명:** Pancreas-CT 데이터셋의 MT-Noise 설정에서는 노이즈 섞인 데이터로 학습했음에도 불구하고, 깨끗한 라벨로 학습한 베이스라인 모델의 성능(Dice 79.19%)을 상회하는 결과(Dice 81.52%)를 기록하였다. 이는 본 모델의 강력한 디노이징 및 일반화 능력을 보여준다.
3. **노이즈 비율 영향:** 노이즈 비율이 증가함에 따라 성능이 하락하지만, SVL-DRL은 베이스라인 모델보다 훨씬 완만한 성능 하락 곡선을 보이며 높은 강건성을 입증하였다.
4. **절제 연구(Ablation Study):** Warmup Stage를 제거했을 때 성능이 베이스라인보다 낮아지는 현상이 관찰되었다. 이는 RL이 학습되기 전 올바른 세만틱 가이드가 필수적임을 시사한다.

## 🧠 Insights & Discussion

본 논문의 가장 큰 강점은 **'데이터의 선택적 제거'가 아닌 '동적 정제'**라는 관점의 전환에 있다. 기존의 샘플 선택 기반 방법들은 노이즈가 섞인 라벨을 신뢰할 수 없다고 판단하여 제거하는데, 이 과정에서 희소하지만 유용한 경계 정보까지 소실될 수 있다. 반면 SVL-DRL은 모든 데이터를 유지하면서 보상 기반의 피드백 루프를 통해 라벨의 질을 점진적으로 개선한다.

또한, 복셀 단위로 에이전트를 배치하고 해부학적 제약(연결성, 매끄러움)을 보상 함수에 직접 반영함으로써, 딥러닝 모델이 단순히 수치적인 손실 함수를 줄이는 것을 넘어 임상적으로 타당한(clinically plausible) 형태의 분할 결과를 생성하도록 유도하였다.

다만, 각 복셀을 에이전트로 다루는 방식은 이론적으로는 정교하나, 실제 구현 시 계산 복잡도와 학습 시간에 영향을 줄 수 있으며, 본문에서는 이를 vA3C의 비동기 업데이트 방식으로 해결하려 하였다.

## 📌 TL;DR

본 연구는 노이즈 섞인 어노테이션 환경에서 3D 의료 영상 분할을 수행하기 위해, 각 복셀을 에이전트로 사용하는 **단계적 복셀 레벨 심층 강화학습(SVL-DRL)** 프레임워크를 제안하였다. 이 모델은 데이터를 버리는 대신 보상 함수를 통해 입력 상태를 동적으로 정제하며, 특히 해부학적 연속성을 고려한 보상 설계를 통해 SFDA 및 형태학적 노이즈 환경에서 기존 SOTA 모델 대비 우수한 성능(Dice/IoU 평균 3% 이상 향상)을 보였다. 이는 향후 고품질의 라벨을 구하기 어려운 임상 환경에서 매우 유용한 접근 방식이 될 것으로 기대된다.
