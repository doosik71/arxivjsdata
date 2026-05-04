# Knowledge Distillation via Token-level Relationship Graph

Shuoxi Zhang, Hanpeng Liu, Kun He (2023)

## 🧩 Problem to Solve

본 논문은 지식 증류(Knowledge Distillation, KD) 과정에서 발생하는 정보 손실과 모델 간의 용량 차이(Capacity Gap) 문제를 해결하고자 한다. 기존의 지식 증류 방식은 주로 개별 인스턴스의 정보(Logit 또는 Feature)나 인스턴스 간의 관계(Instance-level relationship)를 전이하는 데 집중해 왔다. 그러나 이러한 접근 방식은 이미지 내부의 세부적인 시맨틱 정보를 담고 있는 토큰 수준(Token-level)의 관계를 간과한다는 한계가 있다.

특히, 데이터셋이 불균형한 롱테일(Long-tail) 환경에서는 클래스 간 불균형으로 인해 인스턴스 수준의 관계 기반 증류 방식의 효율성이 저하되는 문제가 발생한다. 따라서 본 연구의 목표는 토큰 수준의 관계 그래프(Token-level Relationship Graph, TRG)를 통해 더욱 세밀한(fine-grained) 시맨틱 정보를 전이함으로써, 일반적인 상황뿐만 아니라 불균형 데이터셋에서도 학생 모델(Student Model)의 성능을 극대화하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 인스턴스 단위가 아닌, 이미지 내부의 패치 토큰(Patch Token) 간의 관계를 그래프 구조로 모델링하여 이를 증류하는 것이다. 

1. **Token Relationship Graph (TRG) 제안**: 이미지 내의 토큰들을 노드로, 토큰 간의 유사도를 엣지로 하는 그래프를 구축하여 토큰 수준의 관계 지식을 전이한다. 이를 통해 서로 다른 이미지에 있더라도 유사한 패턴(예: 개와 고양이의 털 패턴)을 가진 토큰 간의 관계를 학습할 수 있다.
2. **Contextual Loss 도입**: 이미지 내부의 토큰 간 문맥적 유사성을 보존하기 위한 Contextual Loss를 추가하여, 개별 인스턴스의 시맨틱 구조를 유지하도록 한다.
3. **Dynamic Temperature Strategy**: 글로벌 관계 손실을 최적화하기 위해 학습 초기에는 거친(coarse) 구조를 학습하고, 시간이 지남에 따라 세밀한(fine-grained) 구조를 학습하도록 온도를 동적으로 조절하는 전략을 제안한다.
4. **범용적 아키텍처 적용**: CNN뿐만 아니라 토큰화 기반의 Vision Transformer (ViT) 아키텍처에도 쉽게 적용 가능한 프레임워크를 구축하였다.

## 📎 Related Works

기존의 지식 증류 연구는 크게 세 가지 방향으로 발전해 왔다. 첫째는 Soft Logit을 이용한 Vanilla KD와 중간 특징 맵을 맞추는 Feature-based KD이다. 둘째는 인스턴스 간의 상대적 거리나 유사도를 전이하는 Relation-based KD(예: RKD, CRD)이다. 셋째는 그래프 구조를 활용한 Graph-based KD(예: IRG, HKD)이다.

기존의 Graph-based KD 방식들은 주로 인스턴스 관계 그래프(Instance Relationship Graph, IRG)를 사용하여 이미지 전체를 하나의 노드로 처리하였다. 하지만 저자들은 이러한 방식이 이미지 내부의 세부적인 시맨틱 정보를 소실시키며, 특히 데이터 불균형 상황에서 롱테일 효과로 인해 성능이 급격히 저하된다는 점을 지적한다. 본 논문은 분석 단위를 '인스턴스'에서 '토큰'으로 낮춤으로써 기존 연구들이 포착하지 못한 세밀한 구조적 지식을 전달한다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 전체 파이프라인
전체 시스템은 교사 모델과 학생 모델의 특징 맵을 토큰화하고, 이를 기반으로 토큰 관계 그래프를 구축하여 전이하는 구조이다. 전체 손실 함수는 다음과 같이 정의된다.
$$\mathcal{L}_{total} = \mathcal{L}_{logit} + \alpha \mathcal{L}_{inner} + \beta \mathcal{L}_{local} + \gamma \mathcal{L}_{global}$$

### 주요 구성 요소 및 방법론

**1. Token Patching 및 Sampling**
- **ViT**: 이미지 수준에서 이미 패치 토큰화가 수행된다.
- **CNN**: 마지막 전 단계(penultimate) 레이어의 특징 맵을 고정된 크기의 패치로 나누어 토큰화한다.
- **Random Sampling**: 모든 토큰을 그래프에 포함할 경우 연산량이 급증하므로, 배치의 각 이미지에서 무작위로 토큰을 샘플링하여 관리 가능한 크기의 그래프를 구성한다.

**2. k-NN Graph Construction**
샘플링된 토큰들을 노드로 설정하고, 유클리드 거리가 가장 가까운 $k$개의 이웃과 연결하는 k-NN 그래프를 구축한다. 인접 행렬 $A$의 원소는 다음과 같이 가우시안 커널로 계산된다.
$$A(T_i, T_j) = \begin{cases} 0, & T_i \notin k\text{-NN}(T_j) \\ e^{-\frac{1}{2\sigma}\|T_i - T_j\|^2}, & \text{otherwise} \end{cases}$$

**3. 손실 함수 (Loss Functions)**
- **Local Preserving Loss ($\mathcal{L}_{local}$)**: 교사 모델의 토큰 주변 구조(Local structure)를 모방하도록 하며, 두 모델의 인접 행렬에 대한 Softmax 확률 분포 간의 KL Divergence를 최소화한다.
$$\mathcal{L}_{local} = \sum_{i \in T} \text{KL} \left( \text{softmax}_{j \in T}(A^S_{ij}) \| \text{softmax}_{j \in T}(A^T_{ij}) \right)$$
- **Global Relationship Loss ($\mathcal{L}_{global}$)**: 전체적인 토폴로지를 전이하기 위해 InfoNCE 손실을 사용한다. 선형 투영(Linear Projection)을 통해 차원을 맞춘 후, 동일한 위치의 토큰(Positive pair)은 가깝게, 다른 토큰(Negative pair)은 멀게 학습한다.
$$\mathcal{L}_{global} = -\sum_{i \in T} \log \frac{\exp(\text{SIM}(T^S_i, T^T_i)/\tau_g)}{\sum_{j \in T} \exp(\text{SIM}(T^S_i, T^T_j)/\tau_g)}$$
- **Contextual Loss ($\mathcal{L}_{inner}$)**: 단일 이미지 내의 토큰 간 유사도 행렬(CS)을 계산하고, 교사와 학생 모델 간의 MSE(Mean Squared Error)를 통해 내부 문맥을 보존한다.
$$\text{CS} = \text{Softmax} \left( \frac{F \cdot F'}{\sqrt{D}} \right), \quad \mathcal{L}_{inner} = \text{MSE}(\text{CS}^T, \text{CS}^S)$$

**4. Dynamic Temperature Adjustment**
$\mathcal{L}_{global}$에 사용되는 온도 파라미터 $\tau_g$를 학습 과정에 따라 동적으로 변경한다. 초기에는 높은 온도를 사용하여 거친(coarse) 전역 구조를 먼저 학습하고, 점차 온도를 낮추어 세밀한(fine-grained) 하드 네거티브 샘플을 구분하도록 유도한다.

## 📊 Results

### 실험 설정
- **데이터셋**: CIFAR-100, ImageNet 및 이들의 롱테일 버전인 CIFAR-100-LT, ImageNet-LT를 사용하였다.
- **모델**: ResNet, VGG, WideResNet, MobileNet, ShuffleNet(CNN 계열) 및 DeiT(ViT 계열)를 사용하여 교사-학생 쌍을 구성하였다.
- **지표**: Top-1 Accuracy 및 Top-5 Error rate를 측정하였다.

### 주요 결과
- **일반 데이터셋**: CIFAR-100와 ImageNet 모두에서 TRG가 Vanilla KD, RKD, IRG, HKD 등 기존 SOTA 방법론들을 일관되게 상회하는 성능을 보였다.
- **불균형 데이터셋 (Long-tailed)**: CIFAR-100-LT와 ImageNet-LT에서 매우 강력한 성능을 보였으며, 특히 일부 시나리오에서는 **학생 모델이 교사 모델의 성능을 능가**하는 결과가 나타났다. 이는 토큰 수준의 관계 정보가 롱테일 효과로 인한 정보 부족을 효과적으로 보완함을 시사한다.
- **ViT 적용**: CNN-based teacher를 사용하여 ViT-based student를 학습시켰을 때 가장 좋은 성능을 보였는데, 이는 CNN의 지역적 귀납 편향(Local inductive bias)이 효과적으로 전달되었기 때문으로 분석된다.
- **정성적 분석**: t-SNE 시각화 결과, TRG를 적용한 모델이 다른 KD 방법론들에 비해 클래스 간 경계가 훨씬 더 뚜렷하고 분리도가 높게 나타났다.

## 🧠 Insights & Discussion

**강점 및 기여**
본 논문은 KD의 분석 단위를 인스턴스에서 토큰으로 세분화함으로써, 단순한 특징 일치를 넘어 구조적인 시맨틱 관계를 전이하는 새로운 패러다임을 제시하였다. 특히 롱테일 데이터셋에서 교사 모델보다 더 높은 성능을 낸 점은, 적절한 관계 정보의 전이가 모델의 일반화 성능을 극적으로 향상시킬 수 있음을 입증한다.

**한계 및 비판적 해석**
1. **연산 비용**: 토큰 수준에서 그래프를 구축하고 k-NN을 계산하는 과정은 인스턴스 수준보다 연산 복잡도가 높다. 비록 랜덤 샘플링으로 이를 완화했으나, 샘플링 비율에 따른 정보 손실 가능성에 대한 심층적인 분석이 부족하다.
2. **하이퍼파라미터 민감도**: $\alpha, \beta, \gamma$ 등 여러 손실 함수의 가중치와 $\tau_g$의 동적 스케줄링 등 튜닝해야 할 파라미터가 많아, 다른 데이터셋이나 태스크에 적용 시 최적화 과정이 까다로울 수 있다.
3. **배치 사이즈 의존성**: 실험 결과 배치 사이즈가 커질수록(최대 512까지) 성능이 향상되는데, 이는 그래프의 대표성을 확보하기 위해 큰 배치가 필수적임을 의미한다. 이는 메모리 자원이 제한된 환경에서 제약 사항이 될 수 있다.

## 📌 TL;DR

본 논문은 기존의 인스턴스 단위 지식 증류의 한계를 극복하기 위해, 이미지 내부의 패치 토큰 간 관계를 그래프로 모델링하여 전이하는 **Token Relationship Graph (TRG)** 방법을 제안한다. 로컬/글로벌 그래프 손실과 내부 문맥 손실을 결합하고 동적 온도 조절 전략을 적용하여, CNN과 ViT 모두에서 SOTA 성능을 달성하였다. 특히 **데이터 불균형(Long-tail) 상황에서 매우 강력한 성능 향상**을 보여, 향후 불균형 데이터 학습 및 모델 압축 분야에 중요한 기여를 할 것으로 기대된다.