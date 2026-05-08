# DIGRAF: Diffeomorphic Graph-Adaptive Activation Function

Krishna Sri Ipsit Mantri, Xinzhi (Aurora) Wang, Carola-Bibiane Schönlieb, Bruno Ribeiro, Beatrice Bevilacqua, Moshe Eliasof (2024)

## 🧩 Problem to Solve

본 논문은 그래프 신경망(Graph Neural Networks, GNNs)에서 사용되는 활성화 함수(Activation Function)의 한계를 해결하고자 한다. 대부분의 GNN은 ReLU와 같은 표준 활성화 함수를 기본적으로 사용하지만, 활성화 함수의 선택은 네트워크의 성능에 상당한 영향을 미친다.

특히 그래프 데이터는 노드의 차수(degree) 차이나 그래프 크기의 변화와 같은 고유한 구조적 특성을 가지고 있어, 이에 적응할 수 있는 **Graph-adaptive**한 활성화 함수가 필요하다. 기존의 그래프 적응형 활성화 함수(예: GReLU)가 제안되었으나, 이들은 조각마다 선형(piecewise linear)인 고정된 구조(blueprint)를 가지고 있어 표현력에 한계가 있으며, 미분 불가능한 지점이 존재한다는 단점이 있다.

따라서 본 연구의 목표는 미분 가능성, 유계성(boundedness), 계산 효율성을 갖추면서도, 입력 그래프의 구조에 따라 유연하게 변화할 수 있는 적응형 활성화 함수인 **DIGRAF**를 설계하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Diffeomorphism(미분동형사상)**, 특히 **Continuous Piecewise-Affine Based (CPAB)** 변환을 활성화 함수 설계에 도입한 것이다.

1. **유연한 설계 기반(Blueprint)**: 고정된 수식 대신 CPAB 프레임워크를 사용하여, 매우 넓은 범위의 함수를 근사할 수 있는 유연한 활성화 함수 구조를 제안한다.
2. **그래프 적응성(Graph-Adaptivity)**: 추가적인 GNN ($\text{GNN}_{\text{ACT}}$)을 통해 입력 그래프의 구조와 특징에 따라 활성화 함수의 파라미터 $\theta$를 동적으로 생성함으로써, 각 그래프에 최적화된 활성화 함수를 end-to-end 방식으로 학습한다.
3. **이론적 보장**: 제안된 DIGRAF가 미분 가능성, 유계성, 노드 순열 등가성(permutation equivariance), 그리고 Lipschitz 연속성이라는 신경망 활성화 함수의 바람직한 성질들을 모두 만족함을 이론적으로 증명한다.

## 📎 Related Works

### 관련 연구 및 한계

- **Diffeomorphisms in Neural Networks**: 미분동형사상은 전단사(bijective)이며 미분 가능하고 역함수 또한 미분 가능한 매핑이다. 기존의 CPAB 접근법은 1D 공간에서 효율적인 계산이 가능함을 보였으나, 주로 정렬(alignment)이나 회귀 작업에 사용되었으며 활성화 함수로의 적용은 시도되지 않았다.
- **General-Purpose Activation Functions**: ReLU, Tanh, Swish 등 다양한 함수가 연구되었으나, 이들은 입력 데이터의 구조에 따라 형태가 변하는 '입력 적응성'이 결여되어 있다.
- **Graph Activation Functions**: Max/Median 필터나 GReLU와 같은 그래프 전용 활성화 함수가 제안되었다. 하지만 GReLU의 경우 piecewise linear 구조로 인해 표현력이 제한적이며, 미분 불가능한 지점이 발생하여 최적화에 불리하다.

### 차별점

DIGRAF는 기존의 고정된 blueprint(예: piecewise linear)를 벗어나 CPAB를 통해 훨씬 더 복잡하고 유연한 비선형 함수를 학습할 수 있으며, $\text{GNN}_{\text{ACT}}$를 통해 그래프 구조에 직접적으로 적응한다는 점에서 기존 연구와 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조

DIGRAF의 전체 파이프라인은 그림 1과 같이 구성된다.

1. **GNN 레이어**: 입력 노드 특징 $H^{(l-1)}$과 인접 행렬 $A$를 통해 중간 특징 $\bar{H}^{(l)}$를 생성한다.
2. **$\text{GNN}_{\text{ACT}}$**: $\bar{H}^{(l)}$과 $A$를 입력으로 받아, 해당 그래프에 최적화된 활성화 함수 파라미터 $\theta^{(l)}$를 결정한다. 이때 $\text{GNN}_{\text{ACT}}$ 이후에 Graph-wise pooling(max 또는 mean pooling)을 적용하여 단일 벡터 $\theta^{(l)}$를 추출한다.
3. **DIGRAF 활성화**: 결정된 $\theta^{(l)}$를 사용하여 CPAB 변환 $T^{(l)}$을 정의하고, 이를 $\bar{H}^{(l)}$에 요소별(element-wise)로 적용하여 최종 특징 $H^{(l)}$를 얻는다.

### 주요 메커니즘 및 방정식

#### 1. CPAB Diffeomorphism

활성화 함수는 1차원 함수이므로, 1D CPAB를 사용한다. 먼저 폐구간 $\Omega = [a, b]$를 $N_P$개의 구간으로 나누는 테셀레이션(tessellation) $P$를 정의한다.

파라미터 $\theta \in \mathbb{R}^{N_P-1}$에 의해 정의되는 **Continuous Piecewise-Affine (CPA) velocity field** $v_\theta$를 구축한다. 이 속도장(velocity field)은 다음과 같은 적분 방정식을 통해 궤적 $\phi_\theta(x, t)$를 생성한다.

$$\phi_\theta(x, t) = x + \int_0^t v_\theta(\phi_\theta(x, \tau)) d\tau$$

최종적인 CPAB 미분동형사상 $f_\theta(x)$는 시간 $t=1$일 때의 위치로 정의된다.
$$f_\theta(x) \triangleq \phi_\theta(x, t=1)$$

#### 2. DIGRAF 정의

입력값이 도메인 $\Omega$ 내에 있을 때는 CPAB 변환 $T^{(l)}$을 적용하고, 도메인 밖의 값은 항등 함수(identity function)로 처리한다.

$$\text{DIGRAF}(\bar{h}^{(l)}_{u,c}, \theta^{(l)}) =
\begin{cases}
T^{(l)}(\bar{h}^{(l)}_{u,c}; \theta^{(l)}), & \text{if } \bar{h}^{(l)}_{u,c} \in \Omega \\
\bar{h}^{(l)}_{u,c}, & \text{otherwise}
\end{cases}$$

#### 3. 학습 절차 및 손실 함수
$\text{GNN}_{\text{ACT}}$를 통해 $\theta^{(l)}$를 학습하며, 속도장의 부드러움(smoothness)을 유지하기 위해 Gaussian smoothness prior 기반의 정규화 항 $R$을 추가한다.

$$L_{\text{TOTAL}} = L_{\text{TASK}} + \lambda \sum_{l=1}^L \theta^{(l)\top} \Sigma_{\text{CPA}}^{-1} \theta^{(l)}$$

여기서 $L_{\text{TASK}}$는 다운스트림 태스크의 손실 함수(예: Cross-entropy, MAE)이며, $\lambda$는 정규화 강도를 조절하는 하이퍼파라미터이다. 또한 $\theta^{(l)}$의 값은 $\tanh$ 함수를 통해 $[-1, 1]$ 범위로 제한하여 훈련 안정성을 높인다.

## 📊 Results

### 실험 설정
- **데이터셋 및 작업**:
    - 노드 분류: BlogCatalog, Flickr, CiteSeer, Cora, PubMed.
    - 그래프 분류 및 회귀: ZINC-12K (회귀), OGB (MoleSol, MolTox21, MolBace, MolHIV), TU Datasets.
- **비교 대상 (Baselines)**:
    - 표준 활성화 함수: Identity, Sigmoid, ReLU, LeakyReLU, Tanh, GeLU, ELU.
    - 학습 가능 활성화 함수: PReLU, Maxout, Swish.
    - 그래프 적응형 활성화 함수: Max, Median, GReLU.
- **측정 지표**: Accuracy (%), MAE ($\downarrow$), ROC-AUC ($\uparrow$), RMSE ($\downarrow$).

### 주요 결과
1. **노드 분류**: DIGRAF는 모든 데이터셋에서 표준 및 학습 가능 활성화 함수보다 우수한 성능을 보였다. 특히 GReLU와 같은 기존 그래프 적응형 함수보다 높은 정확도를 기록했다.
2. **그래프 회귀 및 분류**:
    - **ZINC-12K**: MAE $0.1302$를 기록하며 최우수 baseline인 Maxout($0.1587$) 대비 약 $18\%$ 성능 향상을 이루었다.
    - **OGB (MolHIV)**: ROC-AUC $80.28\%$를 달성하여 ReLU($75.58\%$) 대비 절대 수치로 $4.7\%$ 향상되었다.
3. **그래프 적응성의 효과**: $\text{GNN}_{\text{ACT}}$를 제거한 $\text{DIGRAF (W/O ADAP.)}$보다 $\text{DIGRAF}$의 성능이 일관되게 높게 나타났으며, 이는 그래프 구조에 맞게 활성화 함수를 동적으로 변경하는 것이 매우 중요함을 시사한다.
4. **수렴 속도**: Convergence Analysis 결과, DIGRAF는 다른 baseline들과 비슷하거나 더 빠른 훈련 수렴 속도를 보이면서도 더 나은 일반화 성능을 달성했다.

## 🧠 Insights & Discussion

### 강점 및 해석
- **Blueprint의 유연성**: CPAB 기반의 설계는 ELU나 Tanh 같은 기존 함수들을 매우 정교하게 근사할 수 있으며, piecewise linear 구조인 GReLU보다 훨씬 복잡한 비선형성을 학습할 수 있다. 이는 실험 결과에서 DIGRAF가 일관되게 우위를 점하는 핵심 이유이다.
- **이론적 견고함**: 미분동형사상의 성질을 이용함으로써 활성화 함수가 가져야 할 필수 조건(미분 가능성, 유계성 등)을 자연스럽게 만족하며, 이는 최적화 과정의 안정성으로 이어진다.
- **효율성**: 1D CPAB는 닫힌 형태의 해(closed-form solution)가 존재하여 계산 복잡도가 선형적이며, 실제 런타임 측면에서도 GReLU보다 빠른 추론 속도를 보였다.

### 한계 및 논의
- **함수 공간의 제한**: 본 연구의 DIGRAF는 '미분동형사상(Diffeomorphisms)' 클래스에 속하는 함수만을 학습한다. 비록 이 클래스가 매우 방대하지만, 모든 가능한 활성화 함수를 포함하는 것은 아니므로 최적의 함수가 이 클래스 외부에 존재할 가능성이 있다.
- **계산 비용**: 표준 ReLU에 비해서는 추가적인 $\text{GNN}_{\text{ACT}}$ 연산으로 인해 훈련 및 추론 시간이 증가한다. (ZINC 데이터셋 기준 ReLU 대비 추론 시간이 약 3.5배 증가)

## 📌 TL;DR

본 논문은 GNN을 위해 **CPAB(Continuous Piecewise-Affine Based) 미분동형사상**을 활용한 그래프 적응형 활성화 함수 **DIGRAF**를 제안한다. DIGRAF는 보조 GNN을 통해 각 그래프의 구조에 맞는 활성화 함수 파라미터를 동적으로 생성하며, 이를 통해 기존의 고정된 형태나 단순한 조각 선형 함수보다 훨씬 유연하고 강력한 비선형 표현력을 갖는다. 다양한 벤치마크 데이터셋에서 기존의 표준 및 그래프 전용 활성화 함수들을 압도하는 성능을 입증하였으며, 특히 약물 발견과 같은 분자 특성 예측 작업에서 높은 잠재력을 보여준다.
