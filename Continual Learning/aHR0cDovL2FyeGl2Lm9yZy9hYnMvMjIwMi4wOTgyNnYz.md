# CONTINUAL LEARNING BEYOND A SINGLE MODEL

Thang Doan, Seyed Iman Mirzadeh, Mehrdad Farajtabar (2023)

## 🧩 Problem to Solve

본 논문은 지속 학습(Continual Learning, CL)에서 발생하는 핵심 문제인 **치명적 망각(Catastrophic Forgetting, CF)**을 해결하고자 한다. 치명적 망각이란 새로운 작업(task)을 학습할 때 이전 작업에서 학습한 지식이 손실되어 성능이 급격히 저하되는 현상을 의미한다.

기존의 많은 지속 학습 연구들은 단일 모델(single model) 환경을 가정하고 이를 완화하려 노력해 왔다. 하지만 단일 모델은 모든 작업의 지식을 하나의 파라미터 공간에 저장해야 하므로, 학습해야 할 데이터 스트림이 증가할수록 모델에 가해지는 부담이 커지며 성능 향상에 한계가 있다. 따라서 본 연구의 목표는 단일 모델이라는 제약에서 벗어나 **앙상블 모델(ensemble models)**을 활용함으로써 지속 학습 성능을 높이는 동시에, 앙상블 모델의 고질적인 문제인 높은 계산 비용(computational cost)을 효율적으로 해결하는 알고리즘을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 아이디어는 앙상블 모델이 제공하는 **솔루션의 다양성(diversity in solutions)**이 지속 학습의 성능을 높이는 핵심 동력이라는 점이다. 주요 기여 사항은 다음과 같다.

1.  **앙상블의 유효성 증명**: 단순한 앙상블 기법(Vanilla Ensemble)만으로도 단일 모델보다 지속 학습 성능을 유의미하게 높일 수 있음을 보여주었다.
2.  **계산 비용 문제 제기 및 분석**: 모델 수가 증가함에 따라 계산 비용이 선형적으로 증가하는 문제를 지적하고, 이를 해결하기 위해 다양한 앙상블 기법(Vanilla, Batch, Subspace Ensemble)의 비용-성능 트레이드오프를 분석하였다.
3.  **Subspace-Connectivity 알고리즘 제안**: Neural Network Subspace와 Mode Connectivity 개념을 결합하여, 단일 모델과 유사한 계산 비용을 유지하면서도 앙상블의 성능 이점을 누릴 수 있는 효율적인 알고리즘을 제안하였다.

## 📎 Related Works

논문에서는 지속 학습의 기존 접근 방식을 세 가지 그룹으로 분류하여 설명한다.

-   **정규화 기반 방법(Regularization-based methods)**: EWC와 같이 중요한 파라미터의 변화를 제한하는 방식이다. 하지만 데이터에 대한 여러 번의 패스가 필요하며, 작업 수가 많아지면 특징 표류(feature drift) 문제로 인해 성능이 저하되는 한계가 있다.
-   **메모리 기반 방법(Memory-based methods)**: ER이나 A-GEM처럼 과거 데이터의 일부를 저장하여 재학습(replay)하거나 그래디언트를 수정하는 방식이다.
-   **파라미터 격리 방법(Parameter isolation methods)**: 작업마다 네트워크 모듈을 확장하거나 별도의 서브 네트워크를 할당하는 방식이다. 그러나 작업 수에 따라 메모리와 계산 비용이 계속 증가하며, 예측 시 작업 ID(task identifier)가 필수적이라는 단점이 있다.

본 논문은 특히 **Mode Connectivity**(서로 다른 최적점 사이에 손실 값이 낮은 경로가 존재한다는 성질)와 **Neural Network Subspaces**(가중치의 볼록 조합으로 형성되는 저손실 영역) 연구에 기반하고 있으며, 이를 지속 학습 환경으로 확장하여 기존의 단일 모델 기반 접근법과 차별화를 꾀하였다.

## 🛠️ Methodology

### 1. 비교 대상 앙상블 방법론
본 논문은 제안 방법의 성능을 검증하기 위해 세 가지 앙상블 방식을 비교 분석한다.

-   **Vanilla Ensemble (VE)**: $n$개의 모델을 독립적으로 학습시킨 후, 예측 시 각 모델 결과의 평균(Averaging)을 취한다. 성능은 가장 좋으나 계산 비용이 모델 수에 비례해 선형적으로 증가한다.
-   **Batch Ensemble (BE)**: 공통 가중치 $\omega$를 공유하되, 각 멤버가 고유한 $\{r_i, s_i\}$ 튜플을 가져 가중치를 $\omega_i = \omega \circ (r_i s_i^T)$ 형태로 팩토라이제이션(factorization)한다. 계산 효율적이지만 성능은 상대적으로 낮다.
-   **Subspace Ensemble (SE)**: $n$개의 가중치 $\{\omega_i\}$의 볼록 조합(convex combination) $\bar{\omega} = \sum_{i=1}^n \alpha_i \omega_i$ ($\sum \alpha_i = 1, \alpha \in \Delta_n$)를 학습한다. 추론 시에는 가중치들의 중점(midpoint)을 사용하며, 계산 비용이 단일 모델과 매우 유사하다.

### 2. 제안 방법: Subspace-Connectivity (S-C)
단순 SE 방식은 작업이 진행됨에 따라 최적의 서브스페이스가 표류(drift)하는 문제가 발생하여 망각이 일어난다. 이를 해결하기 위해 본 논문은 **Mode Connectivity**를 이용해 서브스페이스 간의 연결성을 강제하는 2단계 학습 절차를 제안한다.

**단계 1: 새로운 작업에 대한 서브스페이스 솔루션 학습**
새로운 작업 $\tau$에 대해 다음과 같은 목적 함수를 최적화하여 $\hat{W}_\tau = \{\hat{\omega}_{\tau,i}\}_{i=1}^n$를 찾는다.
$$\{\hat{\omega}_{\tau,i}\}_{i=1}^n = \text{argmin}_W \mathbb{E}_{\alpha \sim U[\Delta_n]} [L_\tau(W^T \alpha)]$$
이 단계 이후, 각 클래스당 $m_B$개의 샘플을 버퍼 $B$에 저장한다.

**단계 2: 이전 서브스페이스와의 연결 (Connectivity)**
이전 작업의 서브스페이스 중점 $\omega^*_{\tau-1, mid}$와 현재 작업의 중점 $\hat{\omega}_{\tau, mid}$ 사이에 저손실 경로(low-loss path)를 생성하여 연결성을 강제한다. 목적 함수는 다음과 같다.
$$\{\omega^*_{\tau,i}\}_{i=1}^n = \text{argmin}_W \mathbb{E}_{\alpha \sim U(\Delta_{n+1})} \left[ \sum_{j=1}^{\tau-1} L_j(W^T \alpha_n + \alpha_{n+1} \omega^*_{\tau-1, mid}) + L_\tau(W^T \alpha_n + \alpha_{n+1} \hat{\omega}_{\tau, mid}) \right]$$
여기서 $\alpha = (\alpha_n, \alpha_{n+1}) \in \mathbb{R}^{n+1}$이며, 이 식은 이전 지식의 보존(stability)과 새로운 지식의 습득(plasticity) 사이의 균형을 맞추는 정규화 역할을 한다.

## 📊 Results

### 1. 실험 설정
-   **데이터셋**: Permuted MNIST, Rotated MNIST, Split CIFAR-100, Split miniImageNet (각 20개 작업으로 구성).
-   **지표**: 최종 정확도(Final Accuracy $A_T$) 및 망각 측정치(Forgetting Measure $F_T$).
-   **비교 대상**: Naive SGD, EWC, A-GEM, ER, Stable SGD, MC-SGD, Ensemble MC-SGD 및 제안 방법(S-C).

### 2. 주요 결과
-   **앙상블의 효과**: 앙상블 모델을 사용했을 때 단일 모델보다 최종 정확도가 일관되게 높았으며, 특히 모델 수가 증가할수록 성능이 향상되는 경향을 보였다.
-   **파라미터 수 vs. 앙상블 구조**: 단일 모델의 파라미터 수를 늘린 'Scaled MC-SGD'보다, 파라미터 수는 같지만 앙상블 구조를 가진 'Ensemble MC-SGD'의 성능이 더 우수했다. 이는 성능 향상이 단순히 모델 용량(capacity)의 증가가 아니라 솔루션의 **다양성**에서 기인함을 시사한다.
-   **계산 비용 효율성**:
    -   Vanilla Ensemble은 성능은 최상이나 계산 비용(FLOPS)이 모델 수에 따라 선형적으로 증가한다.
    -   제안된 **Subspace-Connectivity**는 Ensemble MC-SGD에 근접하는 성능을 보이면서도, 계산 비용은 단일 모델(MC-SGD)과 거의 동일한 수준($\approx 1 \times$ FLOPS)을 유지하였다.
-   **정량적 성과**: Rotated MNIST와 Split CIFAR-100 등의 벤치마크에서 S-C는 기존의 단일 모델 기반 SOTA 방법론들(MC-SGD 등)보다 낮은 망각률과 높은 정확도를 기록하였다.

## 🧠 Insights & Discussion

### 1. 강점 및 해석
본 연구는 지속 학습에서 '단일 모델'이라는 고정관념을 깨고 앙상블의 다양성이 치명적 망각을 억제하는 강력한 도구가 될 수 있음을 입증하였다. 특히 Subspace-Connectivity 방법은 단순한 서브스페이스 학습에 Mode Connectivity라는 기하학적 제약을 추가함으로써, 계산 효율성을 유지하면서도 지식의 표류를 효과적으로 막았다는 점이 학술적으로 가치가 있다.

### 2. 한계 및 비판적 논의
-   **버퍼 의존성**: Connectivity를 유지하기 위해 과거 작업의 샘플을 저장하는 작은 버퍼($B$)에 의존한다. 비록 매우 작은 크기이지만, 완전히 메모리가 없는(memory-free) 환경에서의 동작 여부는 명시되지 않았다.
-   **안정성-가소성 트레이드오프**: 논문에서도 언급되었듯이, 연결성을 강제하여 안정성(stability)을 높이는 것은 새로운 작업을 배우는 가소성(plasticity)을 일부 희생시키는 결과를 초래할 수 있다.
-   **데이터셋 크기의 영향**: Convolutional 모델 실험에서 S-C의 성능이 Ensemble MC-SGD보다 낮게 나타난 원인을 데이터셋 크기가 작기 때문으로 추측하고 있으나, 이에 대한 더 정밀한 분석이 필요해 보인다.

## 📌 TL;DR

본 논문은 지속 학습의 치명적 망각 문제를 해결하기 위해 **앙상블 모델의 다양성**을 활용하는 방안을 제시한다. 단순 앙상블은 계산 비용이 너무 높다는 단점이 있으나, 저자들은 **Neural Network Subspace**와 **Mode Connectivity**를 결합한 **Subspace-Connectivity** 알고리즘을 통해 단일 모델 수준의 계산 비용으로 앙상블의 성능 이점을 구현하였다. 이 연구는 향후 효율적인 앙상블 기반 지속 학습 시스템 설계 및 모델 간 상호작용 연구에 중요한 기초 자료가 될 가능성이 크다.