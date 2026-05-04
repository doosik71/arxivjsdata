# MST-Distill: Mixture of Specialized Teachers for Cross-Modal Knowledge Distillation

Hui Li, Pengfei Yang, Juanyang Chen, Le Dong, Yanxin Chen, and Quan Wang (2025)

## 🧩 Problem to Solve

본 논문은 서로 다른 모달리티 간의 지식을 전수하는 Cross-Modal Knowledge Distillation (CMKD) 과정에서 발생하는 효율성 저하 문제를 해결하고자 한다. 전통적인 지식 증류(Knowledge Distillation) 방식은 단일 모달리티 내에서는 성공적이었으나, 서로 다른 데이터 형식과 통계적 특성을 가진 Cross-Modal 환경에서는 데이터 및 통계적 이질성(Heterogeneity)으로 인해 상호 보완적인 지식을 충분히 활용하지 못하는 한계가 있다.

저자들은 기존 연구들이 간과하고 있던 두 가지 핵심적인 문제점을 실증적으로 제시한다. 첫째는 **Distillation Path Selection** 문제로, 특정 태스크나 샘플에 따라 어떤 교사 모델(Multimodal 또는 Cross-modal)이 더 효과적인지 결정하기 어려운 비대칭성과 불확실성이 존재한다는 점이다. 둘째는 **Knowledge Drift** 문제로, 서로 다른 도메인에서 학습된 모델 간의 귀납적 편향(Inductive Bias) 차이로 인해, 동일한 입력에 대해서도 교사 모델과 학생 모델의 주의 집중 영역(Attention region)이 일치하지 않아 지식 전수가 제대로 이루어지지 않는 현상이다. 따라서 본 논문의 목표는 이러한 경로 선택의 불확실성과 지식 표류 문제를 해결하여, 적응적이고 강건한 Cross-Modal 지식 증류 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 단일한 정적 교사 모델에 의존하는 대신, 특성화된 여러 교사 모델의 집합(Mixture of Specialized Teachers)을 구성하고, 각 입력 인스턴스에 최적화된 경로를 동적으로 선택하는 것이다.

이를 위해 저자들은 인스턴스 수준의 라우팅 네트워크(Instance-level routing network)를 도입하여 학생 모델이 학습 과정에서 가장 도움이 되는 교사를 적응적으로 선택하게 하였다. 또한, 교사와 학생 간의 행동 정렬(Behavioral Alignment)을 위해 **MaskNet**이라는 플러그인 모듈을 제안하였다. MaskNet은 교사 모델의 중간 표현(Representation)을 재구성하여 모달리티 특유의 불일치를 억제함으로써 Knowledge Drift를 완화하고 지식 전수의 효과를 극대화한다.

## 📎 Related Works

기존의 Multimodal Learning 연구들은 서로 다른 모달리티의 정보를 통합하여 표현 학습을 개선해 왔으나, 학습 과정에서 모달리티 간의 충돌(Modality conflict)이나 특정 모달리티에 과도하게 의존하는 단일 모달리티 편향(Unimodal bias) 문제에 직면해 왔다.

Cross-Modal Knowledge Distillation (CMKD) 분야에서는 최근 Contrastive Learning, Modality Decoupling, Shared Semantic Representation 등의 기법이 도입되었다. 특히 $C^2KD$나 MGDFR 같은 최신 기법들은 동적 샘플 선택이나 특성 공간의 조정을 통해 성능을 높이려 하였다. 그러나 이러한 기존 접근 방식들은 여전히 특정 시나리오나 고정된 교사 설정에 국한되어 있어, 다양한 모달리티 조합과 태스크 요구사항에 유연하게 대응하지 못한다는 한계가 있다. MST-Distill은 다양한 교사 모델의 앙상블과 학습 가능한 정렬 메커니즘을 통해 이러한 제약을 극복하고 일반화된 프레임워크를 제공한다는 점에서 차별화된다.

## 🛠️ Methodology

MST-Distill 프레임워크는 크게 세 가지 단계인 **Collaborative Initialization (S1)**, **Specialized Teacher Adaptation (S2)**, 그리고 **Dynamic Knowledge Distillation (S3)** 순으로 진행된다.

### 1. Collaborative Initialization (S1)
먼저 타겟 학생 모달리티를 미리 정하지 않고, 모든 모달리티별 모델($f_{m_i}$)들을 동등한 구성원으로 취급하여 공동 학습시킨다. 학습 목표는 정답 라벨을 이용한 태스크 손실 $\ell_{task}$와 모든 모달리티 쌍 간의 예측 일관성을 유지하기 위한 정렬 손실 $\ell_{align}$의 합으로 정의된다.

$$ \ell_{task} = \sum_{i=0}^{M} CE(f_{m_i}(x_i; \theta_{m_i}), y) $$
$$ \ell_{align} = \sum_{0 \le i < j \le M} [KL(P_{m_i} \| P_{m_j}) + KL(P_{m_j} \| P_{m_i})] $$

여기서 $P_{m_i}$는 온도 $\tau$가 적용된 소프트맥스 출력 분포이다. 특이한 점은 교사 모델의 출력에 Gradient Detachment를 적용하지 않아 모든 멤버 간에 상호 그래디언트 전파가 가능하게 하여 초기 정렬을 강화한다는 것이다.

### 2. MaskNet-Driven Specialized Teacher Adaptation (S2)
Knowledge Drift를 해결하기 위해, 교사 모델의 중간 레이어에 학습 가능한 **MaskNet** 모듈을 삽입한다. MaskNet은 다음과 같은 과정을 통해 소프트 마스크를 생성하고 특징 맵을 재구성한다.

1. **Projector**: 중간 특징 $z_l$을 잠재 공간으로 투영한다.
2. **MHSA & Linear**: Multi-Head Self-Attention과 선형 레이어를 거쳐 특징의 중요도를 계산한다.
3. **Sigmoid**: $0$과 $1$ 사이의 소프트 마스크를 생성한다.
4. **Hadamard Product**: 입력 특징과 마스크를 요소별 곱셈하여 재구성된 특징 $z^*_l$을 얻는다.

$$ z^*_l = \text{MaskNet}(z_l; \theta_{MN}) = \sigma(\text{Linear}(\text{MHSA}(\text{Projector}(z_l)))) \otimes z_l $$

이 단계에서는 교사 모델의 기본 파라미터는 동결시키고 MaskNet의 파라미터만 학습시킨다. 학습 목표는 특성화된 교사의 출력 분포와 타겟 학생 모델의 출력 분포 사이의 KL 발산(KL Divergence)을 최소화하여 두 모델의 행동을 정렬하는 것이다.

### 3. Dynamic Knowledge Distillation (S3)
마지막으로, 인스턴스별로 최적의 교사를 선택하여 지식을 전수한다. 학생 모델의 로짓($z_{out}$)을 입력으로 받는 **GateNet**(MLP 구조)이 각 특성화된 교사들에 대한 신뢰도 점수 $C$를 생성한다.

$$ C = \text{softmax}(\text{GateNet}(z_{out}; \theta_{GN})) $$

이 점수를 바탕으로 상위 $K$개의 교사($T^{top-k}$)를 선택하고, 선택된 교사들과 학생 간의 KL 발산을 통해 지식 증류 손실 $\ell_{dist}$를 계산한다. 또한, 특정 교사에게만 의존하는 현상을 방지하기 위해 라우팅 분포가 균등 분포 $U$에 가깝게 유지되도록 하는 Load Balancing 손실 $L_{LB}$를 추가한다.

$$ L_{S3} = \frac{1}{B} \sum_{b=1}^{B} (\ell_{ce}^{(b)} + \mu_1 \cdot \ell_{dist}^{(b)}) + \mu_2 \cdot KL(U \| \bar{C}) $$

## 📊 Results

### 실험 설정
- **데이터셋**: AV-MNIST(숫자 인식), RAVDESS(감정 인식), VGGSound-50k(장면 분류), CrisisMMD-V2(재난 분류)의 분류 작업과 NYU-Depth-V2의 시맨틱 세그멘테이션 작업을 수행하였다.
- **비교 대상**: response-based KD, MLLD, FitNets, OFA, RKD, CRD, DML, 그리고 최신 CMKD 방법론인 MGDFR, $C^2KD$와 비교하였다.
- **평가 지표**: 분류 작업에서는 Accuracy를, 세그멘테이션 작업에서는 OA, AA, mIoU를 사용하였다.

### 주요 결과
1. **분류 성능**: Table 1에 따르면, MST-Distill은 모든 데이터셋에서 최상위 또는 차상위 성능을 기록하였다. 특히 모달리티 불균형이 심한 AV-MNIST와 VGGSound-50k에서 기존 방법론 대비 뚜렷한 성능 향상을 보였다.
2. **세그멘테이션 성능**: NYU-Depth-V2 데이터셋에서 RGB와 Depth 모달리티 모두에 대해 대부분의 지표에서 1위를 차지하였으며, 특히 mIoU에서 가장 높은 성능을 보여 정밀한 구조적 지식 전수 능력을 입증하였다.
3. **Ablation Study**: S1(초기화), S2(특성화), S3(동적 증류) 세 단계가 모두 포함되었을 때 최적의 성능이 나타났으며, 특히 Multimodal 교사와 Cross-modal 교사를 함께 사용할 때(CM+MM) 가장 안정적이고 높은 성능 향상이 관찰되었다.
4. **라우팅 분석**: 학습이 진행됨에 따라 특정 교사 모델의 선택 확률이 동적으로 변화하는 것을 확인하였으며, 이는 모델이 학습 단계별로 필요한 지식 소스를 스스로 찾아냄을 의미한다.

## 🧠 Insights & Discussion

본 논문은 Cross-Modal 지식 증류에서 단순히 강력한 교사를 사용하는 것보다, **교사의 다양성(Diversity)**과 **인스턴스별 적응적 선택(Adaptive Selection)**이 더 중요하다는 점을 시사한다. 특히 Multimodal 교사는 보완적인 정보를 제공하지만 모든 샘플에 최적은 아니며, Cross-modal 교사는 특정 방향으로의 전수 효율이 높은 비대칭성을 가진다는 점을 밝혀낸 것이 주요한 통찰이다.

**강점**으로는 MaskNet을 통해 모델 간의 행동 불일치(Knowledge Drift)를 효과적으로 억제하고, GateNet을 통해 데이터 기반의 최적 경로를 찾았다는 점을 들 수 있다. 또한, 분류뿐만 아니라 픽셀 단위의 정밀함이 요구되는 세그멘테이션 작업에서도 일반화 가능성을 입증하였다.

**한계 및 논의사항**으로는, 다수의 교사 모델을 동시에 처리해야 하므로 학습 단계에서의 메모리 사용량과 학습 시간이 기존 단일 교사 방식보다 크게 증가한다는 점이 언급되었다. 하지만 추론 단계에서는 학생 모델만 사용하므로 배포 효율성은 유지된다. 또한, 모달리티 간 정렬이 매우 느슨한(Loosely aligned) 데이터셋에 대해서는 여전히 개선의 여지가 남아 있으며, 향후 3개 이상의 다중 모달리티 확장 연구가 필요할 것으로 보인다.

## 📌 TL;DR

본 논문은 Cross-Modal 지식 증류의 고질적 문제인 **경로 선택의 불확실성**과 **지식 표류(Knowledge Drift)**를 해결하기 위해 **MST-Distill** 프레임워크를 제안한다. 이 프레임워크는 $\text{Collaborative Initialization} \rightarrow \text{MaskNet 기반 교사 특성화} \rightarrow \text{동적 라우팅 기반 증류}$의 3단계 파이프라인을 통해, 각 샘플에 최적화된 교사 조합을 선택하고 교사와 학생의 행동을 정렬한다. 실험 결과, 시각, 오디오, 텍스트 등 다양한 모달리티 조합과 분류 및 세그멘테이션 태스크에서 SOTA 성능을 달성하였으며, 이는 효율적인 엣지 디바이스용 모델 구축을 위한 강력한 기반 기술이 될 가능성이 높다.