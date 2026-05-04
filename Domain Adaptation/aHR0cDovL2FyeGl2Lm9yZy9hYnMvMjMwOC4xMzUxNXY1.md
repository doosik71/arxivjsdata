# Robust Core-Periphery Constrained Transformer for Domain Adaptation

Xiaowei Yu, Zeyu Zhang, Dajiang Zhu, Tianming Liu (2023/2025)

## 🧩 Problem to Solve

본 논문은 비지도 도메인 적응(Unsupervised Domain Adaptation, UDA)에서 소스 도메인(Source Domain)과 타겟 도메인(Target Domain) 사이의 큰 도메인 간극(Domain Gap)이 존재할 때 발생하는 성능 저하 문제를 해결하고자 한다.

UDA의 핵심 목표는 레이블이 있는 소스 도메인과 레이블이 없는 타겟 도메인 간의 공통 잠재 공간(Common Latent Space)에서 판별 가능하고 도메인 불변적인(Domain-invariant) 특징을 학습하는 것이다. 최근 Vision Transformer(ViT) 기반의 방법들이 우수한 성과를 보이고 있으나, 여전히 도메인 간극이 클 때 이를 효과적으로 극복하고 전이 가능한 표현(Transferable Representation)을 학습하는 것은 어려운 과제로 남아 있다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 인간의 기능적 뇌 네트워크(Human Functional Brain Networks)에서 보편적으로 발견되는 **Core-Periphery(핵심-주변) 원리**를 인공신경망의 Self-Attention 메커니즘에 도입하는 것이다.

- **Core-Periphery 구조의 도입**: 뇌 네트워크에서 밀접하게 연결된 '핵심(Core)' 노드들과 이들에 희소하게 연결된 '주변(Periphery)' 노드들이 존재하는 구조에서 영감을 얻어, 이미지 패치 간의 정보 교환 강도를 조절하는 메커니즘을 설계하였다.
- **RCCT(Robust Core-Periphery Constrained Transformer) 제안**: 패치별 Coreness(핵심도)를 학습하여 도메인 불변 특징(Core)과 도메인 특화 특징(Periphery)을 구분하고, 이를 통해 Self-Attention의 연결 강도를 동적으로 조절하는 프레임워크를 제안한다.
- **강건한 그래프 생성**: 잠재 공간에서 의도적인 섭동(Perturbation)을 추가하는 Embedding Fusion 기법을 통해 노이즈에 강건한 Core-Periphery 그래프를 생성함으로써 모델의 안정성을 높였다.

## 📎 Related Works

### 관련 연구 및 한계

1. **UDA 방법론**: 기존의 UDA는 주로 CNN 기반의 구조에서 적대적 학습(Adversarial Learning)이나 분포 차이 측정(Discrepancy techniques)을 통해 도메인 불변 특징을 추출해 왔다.
2. **Transformer 기반 UDA**: 최근 CDTrans, TVT, SSRT와 같은 연구들이 ViT를 UDA에 적용하기 시작했다. 이들은 Cross-attention을 사용하거나, 클래스 토큰과 패치 토큰을 구분하여 판별하는 방식을 취한다.

### 기존 접근 방식과의 차별점

기존의 Transformer 기반 UDA 모델들이 주로 아키텍처의 구성 요소나 학습 전략(Self-refinement 등)에 집중한 반면, RCCT는 생물학적 뇌 네트워크의 조직 원리인 Core-Periphery 구조를 Self-Attention에 직접적으로 주입하여 정보 흐름을 제어한다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 1. 전체 파이프라인

RCCT는 ViT-base를 백본으로 하며, 다음과 같은 구성 요소로 이루어진다:

- **Vision Transformer Backbone**: 이미지 패치를 임베딩하고 특징을 추출한다.
- **Patch Discriminator ($D^l$)**: 각 패치의 도메인 소속을 판별하여 Coreness를 측정한다.
- **CP Graph Module**: 측정된 Coreness를 바탕으로 패치 간 연결 강도를 정의하는 인접 행렬 $M_{cp}$를 생성한다.
- **CP-guided Transformer Layers**: 생성된 $M_{cp}$를 사용하여 Self-Attention의 가중치를 재조정한다.
- **Domain Discriminator & Self-clustering**: 클래스 토큰 레벨에서 도메인 정렬 및 타겟 도메인의 클래스 응집도를 높인다.

### 2. 주요 구성 요소 및 상세 설명

#### 2.1 Coreness 측정 및 CP 그래프 생성

패치 레벨의 도메인 판별자 $D^l$을 통해 각 패치 $f_{ip}$가 소스 도메인인지 타겟 도메인인지 판별한다. 이때, 판별자가 어느 도메인인지 쉽게 구분하지 못하는(즉, 엔트로피가 높은) 패치일수록 도메인 불변적(Domain-invariant)이라고 판단하여 높은 Coreness를 부여한다.

Coreness $C(f_{ip})$는 다음과 같이 엔트로피 함수 $H(\cdot)$를 통해 계산된다:
$$C(f_{ip}) = H(D^l(f_{ip})) \in [0, 1]$$

이후, 패치 간의 관계를 정의하는 Core-Periphery 그래프의 인접 행렬 $M_{cp}$를 생성한다.
$$M_{cp} = \frac{1}{BH} \sum_{h=1}^{H} \sum_{b=1}^{B} [C(f_{ip})]^T C(f_{ip})$$
여기서 $B$는 배치 크기, $H$는 헤드 수이다. 이후 $M_{cp}(i, j)$ 값이 $0.5$보다 크면 제곱근($\sqrt{\cdot}$)을, 작으면 제곱($\text{square}(\cdot)$) 연산을 적용하여 핵심-주변 구조의 대비를 더욱 명확하게 만든다.

#### 2.2 Coreness-aware Self-Attention (CSA)

마지막 Transformer 레이어에서 클래스 토큰($q$)이 핵심 패치로부터 더 많은 정보를 얻도록 다음과 같이 CSA를 정의한다:
$$\text{CSA}(q, K, V) = \text{softmax}\left(\frac{qK^T}{\sqrt{d}}\right) \odot [1; C(K_{patch})]V$$
여기서 $C(K_{patch})$는 패치들의 Coreness 벡터이며, 이를 통해 도메인 불변 패치의 영향력은 높이고 도메인 특화 패치의 영향력은 억제한다.

#### 2.3 CP Guided Transformer Layer

앞선 레이어($L-1$개)에서는 생성된 $M_{cp}$를 Self-Attention의 마스크로 사용하여 정보 흐름을 제어한다:
$$\text{Attention}(Q, K, V, M_{cp}) = \text{softmax}\left(\frac{QK^T \odot M_{cp}}{\sqrt{d_k}}\right)V$$
이를 통해 핵심 패치 간의 통신은 강화하고, 주변 패치 간의 불필요한 정보 흐름은 약화시킨다.

#### 2.4 Robustness를 위한 Embedding Fusion

데이터의 노이즈에 대응하기 위해 잠재 공간에서 섭동을 추가한다. 동일 도메인의 다른 샘플 $b_{x_j}$를 이용하여 입력 토큰 시퀀스를 다음과 같이 변형한다:
$$\tilde{b}_{x_i} = b_{x_i} + \mu [b_{x_j} - b_{x_i}], \quad i \neq j$$
여기서 $\mu$는 섭동 강도이며, 이를 통해 더 강건한 CP 그래프를 학습할 수 있다.

### 3. 최종 목적 함수 (Overall Objective Function)

모델은 다음 네 가지 손실 함수의 합을 최소화하는 방향으로 학습된다:
$$\mathcal{L} = L_{clc}(x_s, y_s) + \alpha L_{dis}(x_s, x_t) + \beta L_{pat}(x_s, x_t) - \gamma I(p_t; x_t)$$

- $L_{clc}$: 소스 도메인의 분류 손실 (Cross-Entropy).
- $L_{dis}$: 클래스 토큰 기반의 글로벌 도메인 적대적 손실.
- $L_{pat}$: 패치 레벨의 도메인 적대적 손실 (Coreness 측정에 사용).
- $I(p_t; x_t)$: 타겟 도메인의 상호 정보량 최대화를 통한 Self-clustering 손실.

## 📊 Results

### 실험 설정

- **데이터셋**: Office-31, Office-Home, VisDA-2017, DomainNet.
- **백본**: ImageNet으로 사전 학습된 ViT-B/16.
- **최적화**: SGD (Momentum 0.9), Cosine decay 스케줄러 사용.

### 정량적 결과

RCCT-B는 모든 벤치마크 데이터셋에서 기존 SOTA 방법론들을 능가하는 성능을 보였다.

| Dataset | RCCT-B Accuracy | 비고 |
| :--- | :---: | :--- |
| **Office-Home** | 88.3% | ViT-B 대비 13.8% 향상 |
| **Office-31** | 95.3% | ViT-B 대비 3.9% 향상 |
| **VisDA-2017** | 90.7% | ViT-B 대비 23.6% 향상 |
| **DomainNet** | 46.0% | ViT-B 대비 7.9% 향상 |

### 분석 및 결과의 중요성

- **CCT vs RCCT**: 섭동(Perturbation)을 제거한 모델(CCT)보다 RCCT의 성능이 일관되게 높게 나타났으며, 이는 강건한 CP 그래프 생성이 UDA 성능 향상에 필수적임을 시사한다.
- **CP 그래프 시각화**: RCCT가 학습한 CP 그래프는 CCT에 비해 더 밀도 있고 가중치가 명확한 패턴을 보였으며, 이는 도메인 불변 핵심 패치를 더 정확하게 캡처했음을 의미한다.
- **섭동 강도 ($\mu$) 영향**: 실험 결과 $\mu$ 값이 $0.0$에서 $0.5$ 사이의 넓은 범위에서 성능 향상이 관찰되었으며, 일부 데이터셋에서는 $0.5$ 이상의 높은 섭동에서도 강건한 성능을 유지하였다.

## 🧠 Insights & Discussion

### 강점

본 논문은 단순한 아키텍처 변경이 아니라, 뇌 과학의 조직 원리라는 외부의 지식을 딥러닝의 Self-Attention 메커니즘에 성공적으로 이식하였다. 특히, 패치 레벨의 판별자를 통해 '핵심도(Coreness)'라는 개념을 정의하고 이를 통해 정보의 흐름을 제어함으로써, 도메인 간극이 큰 상황에서도 전이 가능한 특징을 효율적으로 추출할 수 있음을 증명하였다.

### 한계 및 논의사항

- **계산 복잡도**: CP 그래프를 생성하고 이를 모든 Transformer 레이어의 Attention에 적용하는 과정에서 추가적인 연산 비용이 발생한다. 다만, 본문에서는 이로 인한 오버헤드에 대한 상세한 분석은 제시되지 않았다.
- **하이퍼파라미터 민감도**: $\alpha, \beta, \gamma$ 및 $\mu$와 같은 다수의 하이퍼파라미터가 성능에 영향을 미치며, 데이터셋마다 서로 다른 최적 값을 설정해야 하는 번거로움이 있다.
- **가정**: 패치 판별자의 엔트로피가 높을수록 도메인 불변적이라는 가정이 실제 모든 도메인 쌍에 대해 보편적으로 성립하는지에 대한 추가적인 이론적 검토가 필요할 수 있다.

## 📌 TL;DR

본 연구는 뇌 네트워크의 **Core-Periphery(핵심-주변) 구조**를 Vision Transformer에 도입하여, 도메인 불변 특징(Core)과 도메인 특화 특징(Periphery)을 구분하고 정보 흐름을 제어하는 **RCCT** 모델을 제안하였다. 특히 잠재 공간의 섭동을 통해 강건한 CP 그래프를 학습함으로써, Office-Home, VisDA-2017 등 주요 UDA 벤치마크에서 SOTA 성능을 달성하였다. 이 연구는 생물학적 영감을 받은 신경망 설계가 복잡한 도메인 적응 문제 해결에 매우 유망한 방향임을 보여준다.
