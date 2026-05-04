# Graph Relation Distillation for Efficient Biomedical Instance Segmentation

Xiaoyu Liu, Yueyi Zhang, Zhiwei Xiong, Wei Huang, Bo Hu, Xiaoyan Sun and Feng Wu (2024)

## 🧩 Problem to Solve

본 논문은 생의학 이미지 분석에서 매우 중요한 과제인 **Biomedical Instance Segmentation**의 효율성 문제를 해결하고자 한다. 최근 딥러닝 기반의 인스턴스 인식 임베딩(Instance-aware embeddings) 방식이 비약적인 발전을 이루었으나, 픽셀 수준의 정밀한 추정을 위해 매우 복잡하고 무거운 모델이 필요하다는 단점이 있다. 이러한 높은 계산 자원 요구량은 실제 실시간 환경이나 하드웨어 제한이 있는 시나리오에서 적용하기 어렵게 만든다.

기존의 Knowledge Distillation(KD) 방식은 무거운 Teacher 네트워크의 지식을 경량화된 Student 네트워크로 전송하여 이 문제를 해결하려 했으나, 두 가지 핵심적인 한계가 존재한다. 첫째, 인스턴스를 구분 짓는 핵심 요소인 인스턴스 수준의 특징(Instance-level features)과 인스턴스 간의 관계(Instance relations)를 충분히 추출하지 못한다. 둘째, 개별 이미지 내의 정보에만 집중하여 서로 다른 이미지들 간의 전역적 관계(Inter-image relations) 정보를 간과한다. 따라서 본 논문의 목표는 이러한 인스턴스 간 관계와 전역적 구조 정보를 효과적으로 전송하여, 매우 적은 파라미터와 계산 비용으로도 높은 성능을 내는 경량화된 생의학 인스턴스 분할 모델을 구축하는 것이다.

## ✨ Key Contributions

본 연구의 중심 아이디어는 인스턴스 간의 관계를 그래프 형태로 모델링하여 전송하는 **Graph Relation Distillation (GRD)** 방법론이다. 구체적인 설계 핵심은 다음과 같다.

1. **Instance Graph Distillation (IGD)**: 인스턴스 중심 임베딩을 노드로, 이들 간의 유사도를 엣지로 하는 그래프를 구축하여 인스턴스 수준의 특징과 관계 지식을 전송한다.
2. **Affinity Graph Distillation (AGD)**: 픽셀 임베딩 간의 유사도를 기반으로 Affinity Graph를 구축하여, 인스턴스 경계(Boundary)에 대한 구조적 지식을 전송함으로써 경계의 모호성을 해결한다.
3. **Intra-image 및 Inter-image 수준의 확장**: 위 두 가지 기법을 단일 이미지 내(Intra-image)뿐만 아니라 여러 이미지 간(Inter-image) 수준으로 확장하여 전역적 구조 정보를 학습시킨다.
4. **Memory Bank 메커니즘**: GPU 메모리 한계를 극복하면서 다수의 이미지 간 관계를 계산하기 위해, 과거의 예측 피처 맵을 저장하는 Memory Bank를 도입하여 장거리(Long-range) 관계를 캡처한다.

## 📎 Related Works

생의학 인스턴스 분할 방법은 크게 두 가지로 나뉜다. **Proposal-based** 방법(예: Mask R-CNN)은 바운딩 박스를 먼저 예측하고 마스크를 생성하지만, 인스턴스가 밀집되어 있거나 크기가 너무 커서 수용 영역(Receptive field)을 벗어나는 경우 성능이 저하된다. 반면, **Proposal-free** 방법, 특히 Pixel Embedding 기반 방식은 각 픽셀을 고차원 특징 공간으로 매핑하여 유사한 픽셀끼리 클러스터링함으로써 복잡하고 겹쳐진 객체 분할에 뛰어난 성능을 보인다. 하지만 이 방식은 연산 및 메모리 요구량이 매우 높다.

Knowledge Distillation 연구들은 주로 이미지 분류나 시맨틱 분할(Semantic Segmentation)에 집중되어 왔으며, 인스턴스 분할 특유의 '인스턴스 구분' 및 '경계 구조' 지식을 전송하는 연구는 부족했다. 특히 기존의 Feature Distillation(예: Attention Transfer)이나 Logit Distillation은 인스턴스 간의 상대적 관계나 전역적 구조 정보를 활용하지 못한다는 한계가 있다. 본 논문은 이러한 관계 기반 지식 전송을 생의학 인스턴스 분할 영역에 처음으로 도입하여 차별점을 둔다.

## 🛠️ Methodology

### 전체 파이프라인

본 방법론은 사전 학습된 무거운 Teacher 네트워크($T$)와 경량화된 Student 네트워크($S$)를 사용한다. 두 네트워크 모두 입력 이미지에 대해 임베딩 맵 $E \in \mathbb{R}^{D \times H \times W}$를 예측하며, 이후 후처리 알고리즘을 통해 인스턴스를 클러스터링한다.

### 1. Instance Graph Distillation (IGD)

인스턴스 수준의 특징과 관계를 전송하기 위한 기법이다.

- **Intra-Image IGD**:
  - 노드 생성: 레이블 마스크를 이용하여 각 인스턴스 $i$에 속한 픽셀들의 평균 임베딩을 계산하여 중심 특징 $v_i$를 추출한다.
        $$v_i = \frac{1}{|S_i|} \sum_{p \in S_i} e_p$$
  - 엣지 정의: 두 노드 $v_i, v_j$ 사이의 코사인 유사도(Cosine similarity)를 계산하여 $\epsilon_{ij}$를 정의한다.
        $$\epsilon_{ij} = \text{Cos}(v_i, v_j) = \frac{v_i^\top \cdot v_j}{\|v_i\|\|v_j\|}$$
  - 손실 함수: Student가 Teacher의 노드 특징과 엣지 관계를 모두 모방하도록 MSE 손실을 적용한다.
        $$L_{\text{Intra}}^{\text{IGD}} = \lambda_1 L_{\text{Intra}}^{\text{Node}} + \lambda_2 L_{\text{Intra}}^{\text{Edge}}$$

- **Inter-Image IGD**:
  - Memory Bank에 저장된 과거 피처 맵들과 현재 이미지 간의 관계를 계산한다. 서로 다른 이미지에서 추출된 인스턴스 노드들 간의 엣지 $\epsilon_{ml}^{ij}$를 구축하고, Teacher와 Student 간의 그래프 일관성을 강제한다.
        $$L_{\text{Inter}}^{\text{Edge}} = \frac{1}{L|I_m||I_l|} \sum_{l=1}^L \sum_{i \in I_m} \sum_{j \in I_l} ((\epsilon_{ml}^{ij})^S - (\epsilon_{ml}^{ij})^T)^2$$

### 2. Affinity Graph Distillation (AGD)

픽셀 수준의 경계 지식을 전송하기 위한 기법이다.

- **Intra-Image AGD**:
  - 픽셀 $p$와 인접 픽셀 $p+n$ 사이의 코사인 유사도를 계산하여 Affinity Map $A$를 생성한다.
  - Student의 Affinity Map $A^S$가 Teacher의 $A^T$ 및 Ground-truth 레이블 기반의 $\hat{A}$와 일치하도록 학습시킨다.
        $$L_{\text{Intra}}^{\text{AGD}} = \|A^S - A^T\|^2, \quad L_{\text{aff}} = \|A^S - \hat{A}\|^2$$

- **Inter-Image AGD**:
  - 서로 다른 두 이미지 $m$과 $l$의 모든 픽셀 쌍 간의 유사도를 계산한 행렬 $A_{ml} = E_m^\top E_l$을 구축하고, 이를 Teacher-Student 간에 전송한다.
        $$L_{\text{Inter}}^{\text{AGD}} = \frac{1}{L} \sum_{l=1}^L \|(A_{ml})^S - (A_{ml})^T\|^2$$

### 3. Overall Optimization

최종 목적 함수는 다음과 같이 정의되며, 하이퍼파라미터 $\lambda$를 통해 각 항의 비중을 조절한다.
$$L_{\text{total}} = L_{\text{aff}} + \lambda_1 L_{\text{Intra}}^{\text{Node}} + \lambda_2 L_{\text{Intra}}^{\text{Edge}} + \lambda_3 L_{\text{Intra}}^{\text{AGD}} + \lambda_4 L_{\text{Inter}}^{\text{Edge}} + \lambda_5 L_{\text{Inter}}^{\text{AGD}}$$

## 📊 Results

### 실험 설정

- **데이터셋**: 2D 데이터셋(CVPPP, BBBC039V1, C.elegans) 및 3D 데이터셋(AC3/4, CREMI)을 사용하였다.
- **평가 지표**: SBD, $|DiC|$ (CVPPP), AJI, Dice, F1, PQ (BBBC039V1, C.elegans), VOI, ARAND (3D EM 데이터셋)를 사용하였다.
- **모델 구성**: Teacher로 ResUNet, NestedUNet, MALA를 사용하였고, Student로는 이들의 경량화 버전(tiny) 및 MobileNetV2를 사용하였다.

### 주요 결과

1. **정량적 성능**: 제안 방법은 기존의 KD 방식(AT, SPKD, ReKD, BISKD)보다 일관되게 우수한 성능을 보였다. 특히 MobileNet과 같이 구조가 완전히 다른 Student 모델에서도 Teacher의 성능에 근접하는 결과를 얻었다.
2. **효율성**: Student 네트워크는 Teacher 대비 **파라미터 수 1% 미만, 추론 시간 10% 미만**으로 획기적으로 줄였음에도 불구하고 매우 경쟁력 있는 성능을 유지하였다.
3. **시각적 분석**: PCA를 통한 임베딩 시각화 결과, 제안 방법으로 학습된 Student 모델은 인접한 인스턴스 간의 색상 차이(특징 차이)가 뚜렷하고 경계 영역이 훨씬 명확하게 나타났다. 이는 IGD와 AGD가 인스턴스 구분 능력을 효과적으로 전송했음을 증명한다.
4. **3D 데이터셋**: VOI 지표에서 Student-Teacher 간의 성능 격차를 AC3/4 데이터셋에서는 93.3%, CREMI 데이터셋에서는 72.9%까지 줄이는 성과를 거두었다.

## 🧠 Insights & Discussion

### 강점 및 해석

본 논문의 가장 큰 강점은 단순히 특징 맵의 값을 복제하는 것이 아니라, **'관계(Relation)'라는 구조적 지식**을 전송했다는 점이다. 특히 Inter-image distillation을 위해 도입한 Memory Bank는 제한된 GPU 자원 내에서 전역적 컨텍스트를 학습할 수 있게 하여, Student 모델이 더 일반화된 인스턴스 특징을 학습하도록 도왔다. 또한, AGD를 통해 픽셀 간의 Affinity를 직접 학습시킴으로써, 경량 모델에서 흔히 발생하는 과분할(Over-segmentation)과 과병합(Over-merge) 문제를 효과적으로 완화하였다.

### 한계 및 논의사항

- **하이퍼파라미터 민감도**: 실험 결과 $\lambda_3, \lambda_4, \lambda_5$와 같은 가중치 파라미터가 성능에 상당한 영향을 미치는 것으로 나타났다. 이는 새로운 데이터셋에 적용할 때 최적의 가중치를 찾기 위한 추가적인 튜닝 과정이 필요함을 시사한다.
- **Memory Bank 비용**: Queue 크기($K$)와 샘플링 수($L$)가 증가할수록 성능은 향상되지만 GPU 메모리 점유율이 높아진다. 따라서 메모리와 성능 사이의 Trade-off를 최적화하는 지점을 찾는 것이 중요하다.
- **후처리 의존성**: 본 방법론은 임베딩 맵을 생성하는 단계에 집중하고 있으며, 최종 결과는 Mutex, Waterz, LMC와 같은 외부 후처리 알고리즘에 의존한다. 후처리 알고리즘 자체의 효율성 개선이 병행된다면 전체 파이프라인의 속도를 더욱 높일 수 있을 것이다.

## 📌 TL;DR

본 논문은 무거운 Teacher 모델의 지식을 경량 Student 모델로 전송하여 효율적인 생의학 인스턴스 분할을 가능하게 하는 **Graph Relation Distillation (GRD)** 프레임워크를 제안한다. 인스턴스 간 관계를 다루는 **IGD**와 경계 구조를 다루는 **AGD**를 Intra- 및 Inter-image 수준에서 모두 적용하였으며, **Memory Bank**를 통해 전역적 관계 정보를 캡처하였다. 결과적으로 **파라미터 1% 미만, 추론 시간 10% 미만의 초경량 모델**로도 Teacher 모델에 근접한 고성능 분할 성능을 달성하였으며, 이는 실시간 생의학 영상 분석 시스템 구축에 크게 기여할 것으로 기대된다.
