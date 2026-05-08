# Mamba-CL: Optimizing Selective State Space Model in Null Space for Continual Learning

De Cheng, Yue Lu, Lingfeng He, Shizhou Zhang, Xi Yang, Nannan Wang, Yanning Zhang, Xinbo Gao (2025)

## 🧩 Problem to Solve

본 논문은 인공지능 모델이 이전의 지식을 잊지 않고 순차적인 작업을 학습할 수 있게 하는 **Continual Learning (CL)**, 그 중에서도 특히 **Class-Incremental Learning (CIL)** 환경에서의 **Catastrophic Forgetting (치명적 망각)** 문제를 해결하고자 한다.

최근 Computer Vision 분야에서 **Mamba** 모델로 대표되는 **State Space Models (SSMs)**가 뛰어난 성능을 보이고 있으나, 이를 CL에 직접 적용하는 것에는 어려움이 있다. 기존의 정규화 기반 방식이나 하위 공간 투영(Subspace Projection) 방식은 주로 CNN이나 Vision Transformer (ViT)의 선형 레이어에 최적화되어 있다. Mamba와 같은 SSM은 고차원의 재귀적 상태 공간 구조(higher-order recurrent state-space structure)와 비선형 이산화(non-linear discretization) 과정을 거치기 때문에, 단순히 기존의 그래디언트 직교 투영(Gradient Orthogonal Projection) 방식을 적용하는 것만으로는 이전 작업의 지식을 보존하기 어렵다.

따라서 본 연구의 목표는 Mamba 모델의 핵심 SSM 블록을 파인튜닝할 때, 이전 작업의 특징 하위 공간(feature subspace)에 직교하는 방향으로 파라미터를 업데이트함으로써 치명적 망각을 방지하는 **Mamba-CL** 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 SSM의 출력값이 새로운 작업을 학습한 후에도 이전 작업에 대해 일관되게 유지되도록 하는 **출력 일관성(Output Consistency)** 조건을 이론적으로 도출하고, 이를 **Null-space Projection (영공간 투영)**을 통해 구현하는 것이다.

주요 기여 사항은 다음과 같다:

1. **Mamba 모델을 위한 직교 투영의 최초 도입**: Mamba 블록의 출력이 작업 간에 일관되게 유지되도록 보장하여 망각을 억제한다.
2. **이론적 일관성 조건 도출**: SSM 블록 내의 4가지 핵심 시불변 파라미터($A, W_B, W_C, W_\delta$)에 대해 출력이 변하지 않기 위한 4가지 충분 조건(Sufficient Conditions)을 수학적으로 유도하였다.
3. **효율적인 구현**: 유도된 조건을 실무적으로 적용하기 위해 Null-space 기반의 근사 솔루션을 도입하여 그래디언트 직교 투영을 효율적으로 수행한다.
4. **성능 검증**: 4가지 CIL 벤치마크 데이터셋에서 기존 최신(SOTA) 방법론들보다 우수한 성능을 입증하였으며, 특히 긴 시퀀스의 CL 시나리오에서도 강건함을 보였다.

## 📎 Related Works

### State Space Models (SSMs)

SSM은 NLP에서 긴 의존성을 모델링하기 위해 제안되었으며, 특히 Mamba는 데이터 의존적인 SSM 블록을 통해 효율적인 시퀀스 모델링을 가능케 했다. 최근에는 VisionMamba, VMamba 등 시각 지능 분야로 확장되어 ViT보다 메모리 효율적이고 뛰어난 성능을 보여주고 있다.

### Continual Learning (CL) 기술

- **Rehearsal-based**: 일부 데이터를 저장하여 재학습시키지만, 저장 공간 오버헤드와 개인정보 보호 문제가 있다.
- **Network Expansion**: 작업마다 네트워크를 확장하지만, 작업 수가 늘어날수록 추론 비용이 증가한다.
- **Regularization-based**: 중요 파라미터의 변화를 제한한다. 특히 **Orthogonal Subspace Projection** 방식은 파라미터를 이전 특징 공간에 직교하는 방향으로 업데이트하여 이론적으로 특징 표류(feature drift)를 제거할 수 있다. 그러나 기존 연구는 주로 CNN이나 ViT의 선형 레이어에 국한되어 있었다.

### Pre-trained Model-based CL

최근 ViT 기반의 사전 학습 모델에 Prompt Tuning을 적용한 CL 방법들이 성과를 거두고 있다. 일부 연구에서는 Visual Prompt를 이전 특징 하위 공간에 직교하게 업데이트하는 방식을 사용했으나, Mamba의 복잡한 구조(재귀 구조 및 비선형 이산화) 때문에 이러한 방식을 그대로 적용할 수 없다는 한계가 있다.

## 🛠️ Methodology

### 전체 시스템 구조

Mamba-CL은 사전 학습된 De-focus Mamba를 백본으로 사용한다. 전체 모델 중 **SSM 블록 내부의 파라미터**와 **SSM 이후의 선형 투영 레이어**만을 파인튜닝하고, 나머지 부분은 동결(Frozen)한다. 각 작업의 분류기는 독립적으로 학습되며, 추론 시에는 모든 분류기의 결과를 결합하여 최종 예측을 수행한다.

### SSM의 순전파 과정 및 일관성 분석

SSM의 출력 $Y$는 다음과 같이 정의된다:
$$Y = \text{SSM}(X | A, W_B, W_C, W_\delta) = X * K$$
여기서 $K$는 다음과 같은 커널이다:
$$K = [CB, CAB, \dots, CA^{k-1}B]$$
이때 $\bar{A} = \exp(\delta A)$와 $\bar{B} = (\delta A)^{-1}(\exp(\delta A) - I)\delta B$라는 이산화 과정을 거친다.

본 논문은 새로운 작업을 학습한 후에도 $\text{SSM}(X_t | \theta_t) = \text{SSM}(X_t | \theta_{t+1})$가 성립하기 위한 조건을 분석하였다. 이를 위해 커널 $K$의 구성 요소인 $\bar{A}, \bar{B}, C$가 각각 유지되어야 한다는 충분 조건을 설정하고, 이를 학습 가능한 파라미터 $\Delta A, \Delta W_B, \Delta W_C, \Delta W_\delta$에 대한 제약 조건으로 변환하였다.

### 4가지 충분 일관성 조건

분석 결과, 이전 작업 $t$의 출력 일관성을 유지하기 위해서는 다음의 4가지 조건이 충족되어야 한다:

1. $\delta_t \Delta A = 0$: 파라미터 $A$의 업데이트가 이산화 단계 파라미터 $\delta$와 직교해야 한다.
2. $X_t \Delta W_\delta = 0$: $\delta$를 결정하는 가중치 $W_\delta$의 업데이트가 입력 특징 $X_t$와 직교해야 한다.
3. $\delta_t X_t \Delta W_B = 0$: $W_B$의 업데이트가 $\delta$와 입력 특징 $X_t$의 곱에 직교해야 한다.
4. $X_t \Delta W_C = 0$: $W_C$의 업데이트가 입력 특징 $X_t$와 직교해야 한다.

### Null-space Projection을 이용한 최적화

위의 조건을 구현하기 위해, 본 논문은 각 특징 공간의 공분산 행렬(Covariance Matrix)을 구하고, **특이값 분해(SVD)**를 통해 영공간(Null Space) 기저를 추출한다.

- 공분산 행렬 $Q_1 = X_t^\top X_t, Q_2 = \delta_t^\top \delta_t, Q_3 = (\delta_t X_t)^\top (\delta_t X_t)$를 계산한다.
- SVD를 통해 특이값이 0(또는 0에 가까운 값)인 우특이 벡터들을 추출하여 투영 행렬 $H_1, H_2, H_3$를 구성한다.
- 그래디언트 $G$를 다음과 같이 투영하여 파라미터를 업데이트한다:
$$\Delta W_\delta = H_1 G_\delta, \quad \Delta W_C = H_1 G_C, \quad \Delta A = H_2 G_A, \quad \Delta W_B = H_3 G_B$$

### Stability-Plasticity Trade-off

모델의 안정성(Stability)과 가소성(Plasticity) 사이의 균형을 맞추기 위해 밸런스 팩터 $\eta \in [0, 1]$를 도입한다.
$$\tilde{H} = \eta H + (1-\eta)I$$
$\eta$가 1에 가까울수록 직교 제약이 엄격해져 망각이 줄어들지만(안정성 증가), 너무 높으면 새로운 지식을 배우는 능력(가소성)이 저하된다.

## 📊 Results

### 실험 설정

- **데이터셋**: ImageNet-R (10/20-split), CIFAR-100 (10-split), DomainNet (10-split).
- **백본**: De-focus Mamba-Large (ImageNet-21k 사전 학습).
- **지표**: 최종 평균 정확도(Final Average Accuracy) 및 최종 평균 망각률(Final Average Forgetting).

### 주요 결과

1. **SOTA 대비 성능**: Mamba-CL은 4가지 벤치마크에서 기존 ViT 기반 CL 방법론들보다 평균 2.5%~3.8% 높은 정확도를 보였다.
2. **망각 방지 효과**: Baseline인 Mamba-Seq(직교 투영 미적용)와 비교했을 때, 정확도는 18%~45% 향상되었고 망각률은 24%~61% 감소하였다.
3. **긴 시퀀스 CL 성능**: 작업 수가 50개 또는 100개로 늘어난 극한의 상황에서도 Mamba-CL은 다른 방법론들보다 우수한 정확도를 유지하며 강건함을 입증했다.
4. **일반화 성능**: MambaVision, Vim, VMamba 등 다양한 Mamba 변형 모델에 적용했을 때도 일관되게 성능 향상이 나타났으며, 동일 조건의 ViT 기반 모델보다 평균 3.76% 높은 정확도를 기록했다.

### Ablation Study

- **구성 요소 분석**: 4가지 투영 중 $\delta$에 대한 투영($H_{1, \delta}$)이 망각 방지에 가장 중요한 역할을 하는 것으로 나타났다.
- **$\eta$의 영향**: $\eta$가 0.90~0.95일 때 정확도가 정점에 도달하며, 이는 안정성과 가소성의 최적 균형점임을 시사한다.
- **학습 방식**: 사전 학습 모델뿐만 아니라, 처음부터 학습(Training from scratch)하는 시나리오에서도 Mamba-Seq 대비 21.76%의 정확도 향상을 보였다.

## 🧠 Insights & Discussion

### 강점

본 연구는 SSM의 수학적 구조를 깊이 있게 분석하여, 단순히 선형 레이어에 적용하던 직교 투영 방식을 Mamba의 특수 구조(이산화 및 재귀)에 맞게 이론적으로 확장했다는 점이 매우 뛰어나다. 또한, SVD 기반의 Null-space Projection을 통해 추가적인 연산 비용을 최소화하면서도 이론적 보장(Consistency)을 제공했다.

### 한계 및 논의사항

- **망각률과 정확도의 관계**: Mamba-CL이 가장 낮은 망각률을 기록한 것은 아니다. 이는 저자들이 언급했듯이 가소성을 확보하기 위해 의도적으로 $\eta$를 조정하여 약간의 망각을 허용한 결과이며, 이것이 최종 정확도 향상으로 이어졌음을 알 수 있다.
- **메모리 비용**: 투영 행렬 $H$를 저장하기 위한 추가 메모리가 필요하다. 하지만 이는 작업 수나 클래스 수에 관계없이 일정하므로, 네트워크를 확장하는 방식보다는 실용적이다.
- **추론 시간**: ViT 기반의 Prompt-tuning 방식보다는 추론 시간이 다소 길지만, 정확도 측면에서의 이득이 이를 상쇄한다.

## 📌 TL;DR

Mamba-CL은 Mamba 모델의 핵심 SSM 파라미터($A, W_B, W_C, W_\delta$)를 이전 작업의 특징 공간에 직교하도록 업데이트하는 **Null-space Projection** 프레임워크이다. 4가지 이론적 일관성 조건을 유도하여 치명적 망각을 수학적으로 억제하였으며, 이를 통해 다양한 시각 지능 벤치마크에서 SOTA 성능을 달성하였다. 이 연구는 Mamba 모델이 Continual Learning 환경에서 매우 강력한 잠재력을 가지고 있음을 증명하였으며, 향후 SSM 기반의 적응형 AI 모델 연구에 중요한 기초가 될 것으로 보인다.
