# Graph Relation Distillation for Efficient Biomedical Instance Segmentation

Xiaoyu Liu, Yueyi Zhang, Zhiwei Xiong, Wei Huang, Bo Hu, Xiaoyan Sun and Feng Wu (2024)

## 🧩 Problem to Solve

본 논문은 생의학 이미지 분석에서 매우 중요하고 도전적인 과제인 **Biomedical Instance Segmentation (생의학 인스턴스 분할)**의 효율성 문제를 해결하고자 한다.

최근 딥러닝 기반의 인스턴스 분할 방법론, 특히 픽셀 임베딩(pixel embedding) 기반 방식은 복잡한 장면에서도 뛰어난 성능을 보이지만, 픽셀 수준의 조밀한 추정을 위해 매우 무거운 모델 아키텍처를 필요로 한다. 이러한 높은 계산 비용과 메모리 요구량은 실제 의료 및 생물학적 응용 시나리오에서 실용성을 떨어뜨리는 주요 원인이 된다.

이를 해결하기 위해 지식 증류(Knowledge Distillation, KD)가 대안으로 제시되었으나, 기존의 KD 방법들은 다음과 같은 두 가지 한계를 가진다:

1. **인스턴스 간 관계 간과**: 인스턴스를 구분하는 데 핵심적인 인스턴스 수준의 특징(instance-level features)과 인스턴스 간의 관계(instance relations)를 효과적으로 추출하지 못하며, 픽셀 수준의 경계 구조 정보를 충분히 전달하지 않는다.
2. **단일 이미지 중심의 학습**: 기존 KD는 주로 개별 입력 이미지 내의 지식 추출에 집중하여, 서로 다른 이미지 간의 전역적 관계(inter-image relations)가 제공하는 구조적 정보를 활용하지 못한다.

따라서 본 논문의 목표는 인스턴스 수준의 특징, 인스턴스 간 관계, 그리고 픽셀 수준의 경계 정보를 모두 고려하는 효율적인 지식 증류 프레임워크를 제안하여, 모델 크기와 추론 시간을 획기적으로 줄이면서도 성능을 유지하는 가벼운 학생(student) 네트워크를 학습시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **그래프 관계 증류(Graph Relation Distillation)**를 통해 교사(teacher) 네트워크의 복잡한 구조적 지식을 학생 네트워크로 전이하는 것이다. 주요 기여 사항은 다음과 같다:

- **Instance Graph Distillation (IGD)**: 인스턴스의 중심 임베딩을 노드로, 인스턴스 간의 유사도를 엣지로 하는 그래프를 구성하여 인스턴스 수준의 특징과 관계 지식을 증류한다.
- **Affinity Graph Distillation (AGD)**: 픽셀 임베딩 간의 관계를 어피니티 그래프(affinity graph)로 변환하여, 인스턴스 경계에 대한 구조적 지식을 전이함으로써 학생 네트워크의 경계 모호성 문제를 해결한다.
- **Inter-image Relation Extension**: IGD와 AGD를 단일 이미지 내부(intra-image)를 넘어 이미지 간(inter-image) 수준으로 확장한다. 이를 위해 **Memory Bank** 메커니즘을 도입하여 제한된 GPU 메모리 환경에서도 과거의 특징 맵을 저장하고 샘플링함으로써 전역적인 관계 정보를 학습할 수 있게 한다.

## 📎 Related Works

### 생의학 인스턴스 분할 (Biomedical Instance Segmentation)

기존 방법론은 크게 두 가지로 나뉜다:

- **Proposal-based**: Mask R-CNN과 같이 바운딩 박스를 먼저 예측하고 마스크를 생성하는 방식이다. 그러나 인스턴스가 밀집되어 있거나 크기가 매우 클 경우 바운딩 박스 예측이 어려워 성능이 저하된다.
- **Proposal-free**: 픽셀 임베딩 기반 방법론이 대표적이며, 각 픽셀을 고차원 특징 공간으로 투영한 후 클러스터링을 통해 인스턴스를 분리한다. 복잡하고 겹쳐진 객체 분리에 유리하지만 계산 비용이 매우 높다는 단점이 있다.

### 지식 증류 (Knowledge Distillation)

일반적으로 교사 모델의 출력 확률(logits)이나 중간 특징 맵(feature maps), 어텐션 맵(attention maps)을 전이하는 방식이 사용된다. 하지만 이러한 일반적인 KD 방법들은 생의학 이미지의 특성인 인스턴스의 다양한 크기, 형태, 분포 및 경계의 모호성을 해결하기 위한 특화된 구조적 지식을 전달하지 못하는 한계가 있다.

## 🛠️ Methodology

### 전체 파이프라인

본 방법론은 무거운 교사 네트워크 $T$와 가벼운 학생 네트워크 $S$를 사용한다. 두 네트워크 모두 입력 이미지에 대해 임베딩 맵 $E \in \mathbb{R}^{D \times H \times W}$를 예측하며, 각 픽셀 $p$의 임베딩 벡터는 $e_p \in \mathbb{R}^D$로 표현된다.

### 1. Instance Graph Distillation (IGD)

인스턴스 수준의 특징과 관계를 전이하기 위해 인스턴스 그래프를 구축한다.

- **노드 정의**: 라벨 마스크를 이용하여 각 인스턴스 $i$에 속하는 픽셀들의 평균 임베딩을 중심 특징 $v_i$로 정의한다.
  $$v_i = \frac{1}{|S_i|} \sum_{p \in S_i} e_p$$
- **엣지 정의**: 두 노드 간의 코사인 유사도로 엣지 $\epsilon_{ij}$를 정의한다.
  $$\epsilon_{ij} = \text{Cos}(v_i, v_j) = \frac{v_i^\top \cdot v_j}{\|v_i\|\|v_j\|}$$
- **손실 함수**: 학생과 교사의 노드 특징 및 엣지 유사도가 일치하도록 MSE(Mean Squared Error) 손실을 적용한다.
  $$L_{\text{Intra}}^{\text{Node}} = \frac{1}{|I|} \sum_{i \in I} \|(v_i)^S - (v_i)^T\|^2$$
  $$L_{\text{Intra}}^{\text{Edge}} = \frac{1}{|I|^2} \sum_{i \in I} \sum_{j \in I} \|(\epsilon_{ij})^S - (\epsilon_{ij})^T\|^2$$

### 2. Affinity Graph Distillation (AGD)

픽셀 수준의 경계 정보를 전이하기 위해 어피니티 그래프를 사용한다.

- **인트라-이미지 증류**: 인접한 픽셀 간의 코사인 유사도를 계산하여 어피니티 맵 $A$를 생성하고, 교사의 어피니티 맵 $A^T$와 학생의 어피니티 맵 $A^S$ 사이의 일관성을 강제한다.
  $$L_{\text{Intra}}^{\text{AGD}} = \|A^S - A^T\|^2$$
  또한, 정답(Ground-truth) 어피니티 $\hat{A}$를 이용한 감독 학습 손실 $L_{\text{aff}}$를 추가한다.
- **인터-이미지 증류**: 서로 다른 이미지 $m$과 $l$ 사이의 픽셀 임베딩 내적을 통해 이미지 간 어피니티 맵 $A_{ml} = E_m^\top E_l$을 구축하고 이를 증류한다.

### 3. Inter-image Relation & Memory Bank

GPU 메모리 한계를 극복하기 위해 **Memory Bank**를 도입한다. 과거 이터레이션에서 생성된 특징 맵들을 큐(Queue) 형태로 저장하고, 현재 이미지와 함께 랜덤하게 샘플링하여 이미지 간 인스턴스 그래프 및 어피니티를 계산함으로써 전역적인 구조 정보를 학습한다.

### 4. 전체 최적화 목표

최종 손실 함수는 다음과 같이 정의되며, 각 항의 가중치 $\lambda$는 실험적으로 설정되었다.
$$L_{\text{total}} = L_{\text{aff}} + \lambda_1 L_{\text{Intra}}^{\text{Node}} + \lambda_2 L_{\text{Intra}}^{\text{Edge}} + \lambda_3 L_{\text{Intra}}^{\text{AGD}} + \lambda_4 L_{\text{Inter}}^{\text{Edge}} + \lambda_5 L_{\text{Inter}}^{\text{AGD}}$$

## 📊 Results

### 실험 설정

- **데이터셋**: 2D (CVPPP, BBBC039V1, C.elegans), 3D (AC3/4, CREMI)
- **지표**: SBD, $|DiC|$ (CVPPP), AJI, Dice, F1, PQ (2D 세포), VOI, ARAND (3D 뉴런)
- **비교 대상**: AT, SPKD, ReKD 및 저자의 이전 연구인 BISKD

### 주요 결과

1. **효율성**: 학생 네트워크(ResUNet-tiny, MALA-tiny)는 교사 네트워크 대비 **파라미터 수는 1% 미만**, **추론 시간은 10% 미만**으로 줄였음에도 불구하고 매우 경쟁력 있는 성능을 보였다.
2. **정량적 성능**:
   - 2D 데이터셋에서 기존 KD 방법들(AT, SPKD, ReKD)보다 일관되게 높은 성능을 기록했다. 특히 MobileNet을 학생 모델로 사용했을 때, 성능 격차를 크게 줄여 방법론의 범용성을 입증했다.
   - 3D 데이터셋(AC3/4, CREMI)에서도 VOI 지표 기준, 교사와 학생 간의 성능 격차를 각각 93.3%와 72.9%까지 감소시켰다.
3. **시각적 분석**: PCA를 이용한 임베딩 시각화 결과, 제안 방법론을 적용한 학생 모델이 인접한 인스턴스 간의 색상 차이(특징 차이)가 뚜렷하고 경계 영역의 구조가 더 정확하게 형성됨을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 분석

- **구조적 지식의 중요성**: 단순히 특징 맵의 값을 복제하는 것이 아니라, 인스턴스 간의 '관계'를 그래프 형태로 정의하여 증류함으로써, 학생 모델이 인스턴스를 구분하는 능력을 효과적으로 학습하게 했다.
- **전역적 맥락 활용**: Memory Bank를 통한 Inter-image 증류는 단일 이미지 내에서 학습할 때보다 더 다양한 인스턴스 특징과 경계 구조를 경험하게 하여, 특히 복잡한 형태의 객체 분할 성능을 향상시켰다.
- **아키텍처 독립성**: 교사와 학생의 구조가 완전히 다른 경우(예: ResUNet $\rightarrow$ MobileNetV2)에도 효과적으로 작동함을 보여, 다양한 경량 모델에 적용 가능하다는 가능성을 제시했다.

### 한계 및 논의사항

- **하이퍼파라미터 민감도**: $\lambda_3, \lambda_4, \lambda_5$와 같은 가중치 값에 따라 성능 변화가 나타나므로, 최적의 가중치를 찾는 과정이 필요하다.
- **메모리 뱅크 비용**: 큐 크기 $K$와 샘플링 수 $L$이 증가할수록 GPU 메모리 점유율이 높아지므로, 가용 자원과 성능 사이의 트레이드오프를 고려해야 한다.

## 📌 TL;DR

본 논문은 생의학 인스턴스 분할 모델을 경량화하기 위해 **인스턴스 그래프 증류(IGD)**와 **어피니티 그래프 증류(AGD)**를 제안하였다. 특히 이를 이미지 내부뿐만 아니라 **메모리 뱅크**를 이용한 이미지 간(inter-image) 수준으로 확장하여 전역적인 구조 지식을 전이했다. 그 결과, **파라미터 1% 미만, 추론 시간 10% 미만**의 극도로 가벼운 모델로도 교사 모델에 근접한 성능을 달성하였으며, 이는 실시간 생의학 이미지 분석 시스템 구축에 크게 기여할 수 있을 것으로 기대된다.
