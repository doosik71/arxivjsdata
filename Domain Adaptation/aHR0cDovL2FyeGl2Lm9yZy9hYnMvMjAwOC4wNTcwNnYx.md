# Network Architecture Search for Domain Adaptation

Yichen Li, Xingchao Peng (2020)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 문제는 비지도 도메인 적응(Unsupervised Domain Adaptation, UDA)에서 사용하는 신경망 구조의 최적화 문제이다. 기존의 UDA 방법론들은 주로 이미지 분류 작업(Image Classification)을 위해 설계된 ResNet, VGG와 같은 수작업 기반의 네트워크(Hand-crafted networks)를 그대로 사용한다. 그러나 이러한 범용 아키텍처가 도메인 간의 전이(Transfer) 작업에 최적인 구조라고 보장할 수 없으며, 이는 결국 UDA 성능의 하락으로 이어진다.

또한, 기존의 신경망 구조 탐색(Neural Architecture Search, NAS) 알고리즘들은 훈련 데이터와 테스트 데이터가 동일한 분포에서 샘플링되었다는 i.i.d. 가정을 전제로 한다. 따라서 서로 다른 데이터 분포를 가진 소스(Source) 도메인과 타겟(Target) 도메인 간의 간극을 고려하여 최적의 구조를 찾는 NAS 연구는 부족한 실정이다.

결과적으로 본 논문의 목표는 타겟 도메인의 검증 손실($L_{t}^{val}$)을 최소화하는 최적의 네트워크 구조를 자동으로 탐색하는 NASDA(Neural Architecture Search for Domain Adaptation) 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 도메인 적응을 위한 아키텍처 탐색을 '소스 도메인에서의 성능 유지'와 '도메인 간의 분포 차이 감소'라는 두 가지 목표를 동시에 최적화하는 이중 목적 함수(Dual-objective) 문제로 정의한 것이다.

이를 위해 저자들은 다음과 같은 두 단계의 학습 전략을 제안한다.
1. **미분 가능한 아키텍처 탐색(Differentiable NAS)**: MK-MMD(Multi-kernel Maximum Mean Discrepancy)를 활용하여 소스 도메인의 성능을 높이면서 도메인 간 거리(Domain Discrepancy)를 줄이는 최적의 구조 $\alpha^*$를 탐색한다.
2. **적대적 학습을 통한 특징 생성기 강화**: 탐색된 구조를 기반으로 구축된 특징 생성기(Feature Generator, $G$)와 여러 개의 분류기(Classifiers, $C$) 사이의 적대적 학습을 통해, 타겟 도메인에서도 강건한 특징을 추출할 수 있도록 모델을 정교화한다.

## 📎 Related Works

### Neural Architecture Search (NAS)
NAS는 강화 학습이나 진화 알고리즘을 통해 네트워크 구조를 자동 설계하는 기술이다. 최근에는 DARTS와 같이 검색 공간을 연속적으로 완화(Continuous relaxation)하여 경사 하강법으로 빠르게 최적의 구조를 찾는 미분 가능한 NAS 방식이 주목받고 있다. 그러나 기존 NAS는 단일 도메인 내에서의 성능 최적화에만 집중하며, 도메인 전이 상황을 고려하지 않는다.

### Unsupervised Domain Adaptation (UDA)
UDA는 라벨이 있는 소스 도메인에서 학습한 지식을 라벨이 없는 타겟 도메인으로 전이하는 것을 목표로 한다. 기존 방식은 크게 세 가지로 나뉜다.
- **Discrepancy-based**: 두 도메인의 특징 분포 차이를 직접 줄이는 방식 (예: DAN).
- **Adversarial-based**: 도메인 판별기를 속여 도메인 불변 특징을 학습하는 방식 (예: DANN).
- **Reconstruction-based**: 데이터를 재구성하며 공통 특징을 찾는 방식.

기존 UDA 연구들은 고정된 백본 네트워크를 사용하므로, 도메인 전이에 특화된 최적의 구조를 가질 수 없다는 한계가 있다.

## 🛠️ Methodology

### 1. 전체 파이프라인
NASDA는 크게 두 단계(Phase)로 구성된다.
- **Phase I (Searching)**: MK-MMD를 가이드로 사용하여 도메인 적응에 최적화된 셀(Cell) 구조 $\alpha$를 찾는다.
- **Phase II (Consolidating)**: 찾아낸 구조를 고정한 후, 다수의 분류기를 이용한 적대적 학습을 통해 특징 생성기 $G$를 강화한다.

### 2. Phase I: Neural Architecture Searching
본 단계에서는 DARTS를 기본 프레임워크로 사용한다. 셀 내의 각 엣지(Edge)에 가능한 연산들의 가중치 합을 배치하여 검색 공간을 연속적으로 표현한다.

**최적화 목표**
타겟 도메인의 라벨이 없으므로 $L_{t}^{val}$을 직접 계산할 수 없다. 따라서 다음과 같은 대리 목적 함수를 최소화하는 $\alpha$를 찾는다.
$$\Phi_{\alpha,w} = \text{argmin}_{\alpha} (L_{s}^{val}(w^*(\alpha), \alpha) + \lambda \hat{d}_{k}^{2}(\Phi(x_s), \Phi(x_t)))$$
$$\text{s.t. } w^*(\alpha) = \text{argmin}_{w} L_{s}^{train}(w, \alpha)$$
여기서 $L_{s}^{val}$은 소스 도메인의 검증 손실이며, $\hat{d}_{k}^{2}$는 두 도메인 간의 MK-MMD 거리이다.

**MK-MMD (Multi-kernel Maximum Mean Discrepancy)**
MK-MMD는 재생 커널 힐베르트 공간(RKHS)에서 두 분포의 평균 임베딩 간의 거리를 측정한다. 본 논문에서는 가우시안 커널의 선형 결합을 사용하여 다음과 같이 정의한다.
$$\hat{d}_{k}^{2}(P, Q) = \left\| \mathbb{E}_{P}[\Phi_{\alpha}(x_s)] - \mathbb{E}_{Q}[\Phi_{\alpha}(x_t)] \right\|_{H_k}^{2}$$
이 값은 미분 가능하므로 DARTS의 아키텍처 파라미터 $\alpha$를 업데이트하는 데 직접 사용될 수 있다.

**근사적 아키텍처 탐색 (Approximate Search)**
이중 최적화(Bilevel optimization)의 계산 복잡도를 줄이기 위해, $w^*$를 완전히 최적화하는 대신 단 한 번의 경사 하강법 단계로 근사한다. 또한, Hessian-vector product 계산의 비용을 줄이기 위해 중앙 차분 근사(Central difference approximation)를 사용하여 계산 복잡도를 2차에서 1차로 낮춘다.

### 3. Phase II: Adversarial Training for Domain Adaptation
Phase I에서 얻은 $\alpha^*$로 특징 생성기 $G$를 구축하고, $N$개의 독립적인 분류기 $C = \{C^{(1)}, \dots, C^{(N)}\}$를 준비한다.

**학습 절차 (3단계)**
1. **기본 학습**: $G$와 $C$를 소스 데이터로 학습시켜 기본적인 분류 능력을 갖추게 한다.
   $$\min_{G,C} L_{s}(x_s, y_s)$$
2. **분류기 다양화**: $G$를 고정하고, 분류기 $C$들이 서로 다른 출력을 내도록 유도한다. 이때 소스 도메인의 성능을 잃지 않도록 $L_{s}$를 정규화 항으로 사용한다.
   $$\min_{C} L_{s}(x_s, y_s) - L_{adv}(x_t)$$
   여기서 $L_{adv}(x_t)$는 분류기들 간의 출력 차이(L1-norm)의 합이다.
3. **특징 생성기 강화**: $C$를 고정하고, 분류기들 간의 출력 차이가 최소화되도록 $G$를 학습시킨다. 즉, 어떤 분류기가 들어와도 일관된 특징을 추출하도록 유도하는 것이다.
   $$\min_{G} L_{adv}(x_t)$$

## 📊 Results

### 실험 설정
- **데이터셋**: Digits (MNIST, USPS, SVHN), STL $\to$ CIFAR10, SYN SIGNS $\to$ GTSRB.
- **비교 대상**: DANN, MCD, G2A 등 기존 UDA 모델 및 NASNet, DARTS 등 기존 NAS 모델.
- **지표**: Accuracy (%)

### 주요 결과
- **UDA 성능**: Digits 데이터셋에서 평균 98.4%의 정확도를 기록하며 베이스라인들을 압도했다. 특히 STL $\to$ CIFAR10 작업에서 76.8%의 정확도를 달성하여, 기존 UDA 및 NAS 기반 모델들보다 월등한 성능을 보였다.
- **NAS 효율성**: 탐색 비용(Search Cost) 측면에서 NASDA는 0.3 GPU day만을 소모하여, NASNet(1,800 days)이나 DARTS(1.5 days)보다 매우 효율적임을 입증했다.
- **분류기 수의 영향**: Phase II에서 분류기 수 $N$이 증가할수록 타겟 도메인의 정확도가 유의미하게 상승하는 경향을 보였다. (단, 계산 복잡도는 $O(N^2)$으로 증가함)

## 🧠 Insights & Discussion

### 강점
본 연구는 단순히 고정된 모델을 사용하는 대신, 도메인 적응이라는 특수한 목적에 맞는 아키텍처를 자동으로 탐색함으로써 UDA의 성능 한계를 돌파했다. 특히 MK-MMD를 NAS 과정에 도입하여 구조 탐색 단계부터 도메인 정렬(Domain Alignment)을 고려했다는 점이 매우 독창적이다.

### 한계 및 논의사항
- **계산 복잡도**: Phase II의 적대적 손실 함수 계산 비용이 분류기 수 $N$의 제곱에 비례하므로, $N$을 무한정 늘릴 수 없는 트레이드-오프 관계가 존재한다.
- **해석 가능성**: 탐색된 아키텍처가 왜 특정 도메인 적응 작업에 더 유리한지에 대한 구조적 분석이나 이론적 설명이 부족하다.
- **가정**: 본 논문은 소스 도메인의 검증 세트($L_{s}^{val}$)가 존재한다고 가정하며, 이를 통해 타겟 도메인의 성능을 간접적으로 최적화한다.

## 📌 TL;DR

**요약**: 본 논문은 도메인 적응을 위해 최적화된 신경망 구조를 자동으로 찾는 **NASDA** 프레임워크를 제안한다. MK-MMD를 이용한 미분 가능한 아키텍처 탐색(Phase I)과 다수 분류기 기반의 적대적 학습(Phase II)을 통해 소스-타겟 도메인 간의 간극을 최소화하는 최적의 모델을 구축한다.

**의의**: 기존의 수작업 기반 백본 네트워크의 한계를 극복하고, 매우 적은 계산 비용(0.3 GPU day)으로 SOTA 성능의 UDA 모델을 생성할 수 있음을 보였다. 이는 향후 합성 데이터(Synthetic data)를 실제 환경(Real-world)에 적용하는 다양한 전이 학습 연구에 중요한 기반이 될 것으로 보인다.