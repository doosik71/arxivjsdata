# GAN for Vision, KG for Relation: a Two-stage Deep Network for Zero-shot Action Recognition

Bin Sun, Dehui Kong, Shaofan Wang, Jinghua Li, Baocai Yin, Xiaonan Luo (2021)

## 🧩 Problem to Solve

본 논문은 학습 단계에서 본 적 없는 클래스의 행동을 인식하는 Zero-shot Action Recognition (ZSL) 문제를 해결하고자 한다. 행동 인식 분야에서 대규모 비디오 데이터에 일일이 레이블을 지정하는 것은 시간과 비용이 많이 소요되며, 전문가의 주관적 판단에 의해 부정확할 가능성이 높다. 따라서 기존의 학습된 데이터를 활용하여 미학습 클래스(unseen classes)를 인식할 수 있는 일반화 능력을 확보하는 것이 매우 중요하다.

기존 ZSL 방법론들은 크게 두 가지 한계를 가진다. 첫째, 행동 클래스 간의 내포적 관계(connotative relation)와 외연적 관계(extensional relation)를 간과하여 일반화 성능이 떨어진다. 둘째, 학습된 분류기가 학습 데이터가 존재하는 seen class로 예측하려는 경향(bias)이 강해, 결과적으로 unseen class에 대한 분류 성능이 저하되는 문제가 발생한다. 본 연구의 목표는 이러한 샘플 수준의 불균형과 분류기 수준의 일반화 문제를 동시에 해결하는 2단계 딥러닝 네트워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 샘플 수준의 분석(GAN)과 분류기 수준의 분석(GCN)을 결합한 **FGGA (Joint Feature Generation network and Graph Attention network)** 구조를 제안하는 것이다.

1. **샘플 수준의 보완**: Conditional Wasserstein GAN (WGAN-GP)을 사용하여 unseen class의 가상 시각 특징(visual feature)을 생성함으로써, seen class와 unseen class 간의 학습 데이터 불균형 문제를 해결한다.
2. **분류기 수준의 보완**: 행동 클래스와 관련 객체(objects) 간의 관계를 정의한 Knowledge Graph (KG)를 구축하고, 여기에 Attention mechanism을 결합한 Graph Convolutional Network (GCN)를 적용하여 분류기의 일반화 능력을 동적으로 향상시킨다.
3. **브릿지 역할의 Word Vector**: 특징 생성 단계와 분류기 일반화 단계 모두에서 단어 벡터(word vectors)를 매개체로 활용하여 seen class에서 unseen class로 지식을 전이한다.

## 📎 Related Works

기존의 ZSL 접근 방식은 크게 두 가지로 나뉜다.

1. **Attribute-based methods**: 사람이 직접 정의한 속성(attribute)을 사용하여 클래스를 구분한다. 직관적이지만, 모든 행동을 설명할 수 있는 속성 세트를 정의하는 것이 어렵고 도메인 지식이 부족할 경우 주관성이 개입되며, 대규모 시나리오로 확장하기 어렵다는 한계가 있다.
2. **Word embedding-based methods**: 클래스 이름의 의미적 표현(예: word vectors)을 사용하여 시맨틱 공간에서 관계를 모델링한다. 속성 기반 방법의 한계를 극복했지만, 시각적 정보와 텍스트 정보 간의 시맨틱 갭으로 인해 클래스 간 구분 능력이 부족하며, 비디오의 다른 부가 정보를 충분히 활용하지 못하는 단점이 있다.

최근에는 GAN을 통해 unseen class의 특징을 합성하거나, Knowledge Graph를 통해 구조적 지식을 전달하는 방법들이 제시되었다. 그러나 GAN 기반 방법은 샘플 관점에, KG 기반 방법은 클래스 관점에 치우쳐 있으며, 기존 KG 기반 방법의 인접 행렬(adjacency matrix)은 고정되어 있어 노드 간의 동적인 관계 변화를 반영하지 못한다는 한계가 있다.

## 🛠️ Methodology

FGGA는 크게 **특징 생성 단계(Sampling stage)**와 **분류기 학습 단계(Classification stage)**의 두 단계로 구성된다.

### 1. 특징 생성 네트워크 (Feature Generation Network)

학습 데이터가 없는 unseen class의 특징을 생성하여 분류기의 편향을 줄이기 위해 Conditional WGAN-GP를 사용한다.

- **Generator ($G$)**: 랜덤 노이즈 $z$와 클래스의 단어 벡터 $c(y)$를 입력받아 시각적 특징 $\tilde{x}$를 생성한다.
- **Discriminator ($D$)**: 생성된 특징과 실제 특징을 구분한다.
- **손실 함수**: 안정적인 학습을 위해 WGAN-GP의 손실 함수를 사용한다.
$$L_{WGAN} = \mathbb{E}[D(x, c(y))] - \mathbb{E}[D(\tilde{x}, c(y))] - \lambda \mathbb{E}[(||\nabla_{\hat{x}} D(\hat{x}, c(y))||^2 - 1)^2]$$
여기서 $\hat{x}$는 실제 샘플 $x$와 합성 샘플 $\tilde{x}$ 사이의 보간된 값이며, 마지막 항은 gradient penalty이다.
- **Cycle-consistency Loss**: 생성된 특징이 원래의 단어 벡터를 다시 복원할 수 있도록 Decoder를 추가하여 $L_{CYC} = ||\hat{c}(y) - c(y)||^2$를 최소화한다.
- **최종 목적 함수**: $\min_{G} \max_{D} (L_{WGAN} + \beta L_{CYC})$

### 2. 그래프 어텐션 네트워크 (Graph Attention Network)

구조화된 지식을 활용하여 unseen class의 분류기를 학습시킨다.

- **Knowledge Graph (KG) 구축**: ConceptNet을 활용하여 행동 클래스와 관련 객체들을 노드로 하는 그래프를 구성한다.
- **GCN 구조**: 입력 특징 행렬 $Z^{(0)}$를 받아 여러 층의 합성곱 연산을 통해 최종 클래스 분류기 $w$를 도출한다.
$$Z^{(l)} = D^{-1/2} \hat{A} D^{-1/2} (Z^{(l-1)})^T \Phi^{(l-1)}$$
- **Attention Mechanism**: 고정된 인접 행렬 $A$ 대신, 노드 간의 유사도를 기반으로 하는 어텐션 계수 $B$를 계산하여 인접 행렬을 동적으로 업데이트한다.
$$[B]_{ij} = \frac{w_i^T w_j}{||w_i||_2 ||w_j||_2} \text{ (if neighbors, else 0)}$$
이후 Softmax를 통해 정규화된 $[A]_{ij}$를 사용하여 클래스 간, 클래스-객체 간의 관계를 동적으로 반영한다.
- **분류기 학습**: 합성된 unseen class 특징과 실제 seen class 특징을 모두 사용하여 Cross-entropy loss($L_{CE}$)로 학습한다.

## 📊 Results

### 실험 설정

- **데이터셋**: UCF101, HMDB51
- **특징 추출**: Inflated 3D (I3D) 네트워크를 사용하여 8192차원의 특징 벡터를 추출한다.
- **단어 임베딩**: YFCC100M으로 학습된 500차원의 Skip-gram word vector를 사용한다.
- **평가 지표**: ZSL (unseen class만 테스트) 및 GZSL (seen/unseen 모두 테스트)에서의 평균 정확도(Accuracy)와 표준편차.

### 주요 결과

- **ZSL 성능**: FGGA는 HMDB51에서 $31.2\%$, UCF101에서 $28.3\%$의 정확도를 기록하며 기존 SOTA 방법들(CEWGAN, CLSWGAN 등)보다 우수한 성능을 보였다.
- **GZSL 성능**: HMDB51 $36.4\%$, UCF101 $37.6\%$를 기록하며 CEWGAN-OD를 상회하는 결과를 얻었다.
- **분석**:
  - **Attention의 효과**: TS-GCN과 비교했을 때, 학습 epoch가 진행됨에 따라 FGGA의 성능이 지속적으로 상승하여 어텐션 메커니즘의 유효성을 입증했다.
  - **클래스별 정확도**: GZSL에서 CEWGAN-OD보다 unseen class의 정확도는 약간 낮으나, seen class의 정확도는 유의미하게 높았다. 이는 CEWGAN-OD가 사용하는 OD(Out-of-distribution) 검출기 없이도 경쟁력 있는 성능을 낸다는 것을 의미한다.
  - **파라미터 분석**: GCN 층수는 3~4층일 때 최적의 성능을 보였으며, 특징 차원은 4096차원일 때 가장 높은 정확도를 기록했다.

## 🧠 Insights & Discussion

본 논문은 ZSL의 고질적인 문제인 '데이터 불균형'과 '분류기 일반화'라는 두 가지 관점을 결합하여 시너지 효과를 냈다. 특히, 기존의 KG 기반 방법론들이 가진 정적인 관계 모델링의 한계를 Attention mechanism을 통해 동적으로 해결한 점이 돋보인다.

다만, GZSL 실험에서 unseen class의 정확도가 CEWGAN-OD보다 소폭 낮은 이유는, FGGA가 별도의 OD detector를 사용하지 않고 단일 분류기만으로 판단하기 때문이다. 이는 향후 연구에서 OD-detector와 FGGA의 구조를 결합한다면 더욱 향상된 성능을 얻을 수 있을 가능성을 시사한다. 또한, GCN 층수가 너무 깊어질 경우 성능 향상이 없거나 오히려 저하되는 현상이 관찰되었는데, 이는 학습 샘플 수가 충분하지 않아 발생하는 overfitting 문제로 해석된다.

## 📌 TL;DR

본 연구는 ZSL 행동 인식 성능 향상을 위해 **GAN을 이용한 특징 합성(샘플 레벨)**과 **Attention-GCN을 이용한 지식 전이(분류기 레벨)**를 결합한 **FGGA** 프레임워크를 제안하였다. Word vector를 매개체로 하여 unseen class의 시각적 특징과 분류기를 성공적으로 생성 및 일반화하였으며, UCF101 및 HMDB51 데이터셋에서 SOTA 수준의 성능을 달성하였다. 이 연구는 데이터가 부족한 희귀 행동 인식이나 대규모 클래스 확장 시스템에 적용될 가능성이 높다.
