# Relational Prior Knowledge Graphs for Detection and Instance Segmentation

Osman Ulger, Yu Wang, Ysbrand Galama, Sezer Karaoglu, Theo Gevers, Martin R. Oswald (2023)

## 🧩 Problem to Solve

본 논문은 객체 검출(Object Detection) 및 인스턴스 분할(Instance Segmentation) 작업에서 객체 간의 관계(Relationship) 정보를 효과적으로 활용하는 문제를 다룬다. 인간은 주변 세계의 객체들 사이의 관계를 이해함으로써 환경에 대한 정신적 표상을 구축하고 상황을 추론하는 뛰어난 능력을 갖추고 있으나, 기존의 많은 컴퓨터 비전 모델들은 특징 공간(Feature Space)에서의 공간적 문맥 정보는 활용할지언정, 객체 간의 명시적인 관계 정보를 충분히 활용하지 못한다는 한계가 있다.

따라서 본 연구의 목표는 객체 제안(Object Proposal)들 사이의 관계를 모델링하여 제안된 특징들을 강화함으로써, 확률적으로 낮은 클래스 예측을 억제하고 중복 예측을 방지하여 최종적인 검출 및 분할 성능을 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Relational Prior-based Feature Enhancement Model (RP-FEM)이라는 새로운 프레임워크를 제안한 것이다. 이 모델의 중심적인 설계 아이디어는 다음과 같다.

1. **Relational Prior Knowledge Graph (RPKG)의 도입**: Visual Genome (VG) 데이터셋의 통계 정보를 바탕으로 객체 간의 공생 관계(Co-occurrence), 상대적 방향(Relative Orientation), 상대적 거리(Relative Distance)를 포함하는 사전 지식 그래프를 구축하여 이를 외부 지식으로 활용한다.
2. **관계 기반 특징 강화**: 초기 제안된 영역(Proposals)들로 구성된 Scene Graph를 구축하고, RP-FEM의 Relation Head가 RPKG와 Scene Graph 간의 어텐션(Attention) 메커니즘을 통해 각 엣지(Edge)의 가중치를 예측하게 한다.
3. **Graph Transformer를 통한 문맥 업데이트**: 예측된 관계 가중치를 바탕으로 Graph Transformer를 적용하여 각 노드(객체 제안)의 특징을 전역적인 문맥 정보로 업데이트한다.
4. **분류 독립적 접근**: 제안된 방법은 제안 영역의 초기 분류 결과에 의존하지 않고 특징 공간에서 직접 관계를 추론하므로, 오분류로 인해 잘못된 사전 지식이 주입되는 위험을 최소화한다.

## 📎 Related Works

기존의 인스턴스 분할 연구들은 주로 Mask R-CNN과 같은 CNN 기반 모델이나 Mask2Former와 같은 Transformer 기반 모델을 중심으로 발전해 왔으며, 일부 연구들은 객체의 형태(Shape)나 윤곽선(Contour)과 같은 사전 지식을 활용하여 마스크를 정교화하는 방식을 취했다. 하지만 객체와 객체 사이의 관계 정보를 직접적으로 활용하는 연구는 여전히 미흡한 상태이다.

또한, 관계 추론을 통해 특징을 강화하려는 기존 시도들이 있었으나, 다음과 같은 한계가 존재한다.

- 일부 모델은 추론 단계에서 Ground Truth 바운딩 박스에 의존하므로 엔드-투-엔드(End-to-End) 학습이 불가능하다.
- 다른 모델들은 제안 영역의 초기 분류(Initial Classification) 결과에 기반하여 관계를 추론하는데, 이는 초기 분류가 틀렸을 경우 잘못된 문맥 정보가 전파되는 문제를 야기한다.

RP-FEM은 이러한 한계를 극복하기 위해 초기 분류 없이 제안 영역의 특징(Proposal Features)과 사전 지식 그래프의 클래스 임베딩 간의 유사도를 어텐션으로 계산하여 관계를 도출한다.

## 🛠️ Methodology

본 모델은 Mask R-CNN 프레임워크를 기반으로 하며, 그 위에 RPKG와 Graph Transformer를 통합하여 특징을 강화한다.

### 1. Relational Prior Knowledge Graph (RPKG) 구축

먼저 Visual Genome (VG) 데이터셋을 사용하여 세 가지 유형의 RPKG를 구축한다. 노드는 Faster R-CNN의 마지막 레이어에서 추출한 클래스별 특징 표현 $d \in \mathbb{R}^{C \times F}$로 구성되며, 엣지는 다음 세 가지 관계를 담고 있다.

- **Co-occurrence**: 두 클래스가 한 이미지에 동시에 나타나는 평균 빈도.
- **Relative Orientation**: 객체 A가 B의 중심, 왼쪽, 오른쪽, 위, 아래에 위치하는 빈도를 5차원 벡터로 표현.
- **Relative Distance**: 두 객체 간의 평균 거리 및 표준 편차를 이미지 크기 및 바운딩 박스 크기에 상대적으로 계산.

결과적으로 $R = \langle D, K \rangle$ 형태의 그래프가 생성되며, 여기서 $K \in \mathbb{R}^{C \times C \times \mathcal{R}}$이다.

### 2. Relation Head: 사전 지식에서 유용한 지식으로의 변환

입력 이미지로부터 얻은 $N$개의 제안 특징 $P \in \mathbb{R}^{N \times F}$를 노드로 하는 Scene Graph $S = \langle P, E \rangle$를 구축한다. Relation Head는 어텐션 메커니즘을 통해 $P$의 이웃 구조와 $R$의 이웃 구조 간의 유사도를 계산하여 엣지 가중치 $E$를 예측한다.

어텐션 계수 $\alpha_{(ij),(uv)}$는 Scene Graph의 노드 쌍 $[p_i, p_j]$와 RPKG의 노드 쌍 $[d_u, d_v]$ 사이에서 계산된다.
$$\alpha_{(ij),(uv)} = \frac{\exp(\text{att}(\hat{p}_{ij}, \hat{d}_{uv}))}{\sum_{u=0}^{C} \sum_{v=0}^{C} \exp(\text{att}(\hat{p}_{ij}, \hat{d}_{uv}))} \quad (1)$$
여기서 $\hat{p}_{ij}$와 $\hat{d}_{uv}$는 각각 선형 변환된 지역적 이웃 표현이다. 최종 엣지 값 $E_{ij}$는 다음과 같이 결정된다.
$$e_{(ij),(kl)} = \alpha_{(ij),(kl)} W_v^R R_{kl} \quad (2)$$
$$E_{ij} = W_E \sum_{k=0}^{C} \sum_{l=0}^{C} e_{(ij),(kl)} \quad (3)$$

### 3. Context Update (Graph Transformer)

예측된 엣지 행렬 $E$를 사용하여 노드 특징을 업데이트한다. Multi-layered Graph Transformer를 통해 각 노드는 주변 노드들로부터 메시지 $m$을 수신하며, 이는 다음과 같은 절차를 거친다.

- **메시지 생성**: 엣지 특징 $f_{ij}^{(l)}$와 노드 특징 $n_i^{(l)}$를 결합하여 어텐션 $\alpha_{ij}^{(l)}$를 계산하고, 이를 통해 가중 합산된 메시지 $m_i^{(l)}$를 생성한다.
  $$m_i^{(l)} = \sum_{j \in I} \alpha_{ij}^{(l)} f_{ij}^{(l)} \quad (7)$$
- **특징 업데이트**: 업데이트된 노드 특징 $\hat{z}_i^{(l)}$는 레이어 정규화(LayerNorm)와 잔차 연결(Residual Connection)을 통해 최종 $z_i^{(l)}$가 된다.
  $$z_i^{(l)} = \text{LN}(\hat{z}_i^{(l)} + f(\hat{z}_i^{(l)})) \quad (9)$$

### 4. Mask Prediction 및 통합

최종 업데이트된 특징 $Z^{(L)}$는 원래의 제안 특징 $P$와 결합(Concatenate)되어 바운딩 박스 예측 및 마스크 예측 헤드로 전달된다.
$$O_{\text{box}} = [P_{\text{box}} \oplus Z], \quad O_{\text{mask}} = [P_{\text{mask}}^{\text{fg}} \oplus Z] \quad (14, 15)$$
학습은 Mask R-CNN의 기존 손실 함수 $\mathcal{L} = \mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{box}} + \mathcal{L}_{\text{mask}}$를 사용하여 엔드-투-엔드로 수행된다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: 사전 지식은 Visual Genome(VG)에서 추출하고, 학습 및 평가는 COCO 데이터셋(80개 클래스)에서 수행하였다.
- **베이스라인**: ResNet-50-FPN 백본을 사용하는 Mask R-CNN.
- **평가 지표**: Average Precision (AP).

### 2. Ablation Study 결과

- **Relation Heads**: 어텐션 헤드 수를 1, 2, 4로 테스트한 결과, 2개의 헤드를 사용할 때 전반적인 성능이 가장 좋았으며, 4개일 때는 작은/중간 크기의 객체 분할 성능이 향상되었다.
- **Context Updates**: 단일 Graph Transformer 레이어($L=1$)에서 가장 좋은 성능을 보였으며, 고차원 이웃 문맥보다 단일 레이어를 풍부하게 모델링하는 것이 더 중요함을 확인하였다.
- **Relationship Types**: Relative Distance(상대적 거리)가 모든 지표에서 가장 강력한 성능을 보였으며, Co-occurrence(공생 관계)는 $\text{AP}_{50}$에서 효과적이었다.

### 3. 정량적 결과

- **성능 향상**: 448개의 Proposal을 사용했을 때, RP-FEM은 Object Detection과 Instance Segmentation 모두에서 Mask R-CNN 베이스라인을 상회하였다.
- **비교 모델**: 전역 문맥을 모델링하는 GCNet과 비교했을 때, RP-FEM은 훨씬 적은 수의 Proposal(448개 vs 2000개)을 사용하고도 더 높은 AP(35.2 vs 33.8)를 달성하였다.
- **클래스별 분석**: 데이터셋 내 샘플 수가 적은 롱테일(Long-tail) 클래스(예: toaster)에서 성능 향상 폭이 특히 컸다.

## 🧠 Insights & Discussion

### 강점 및 기여

본 연구는 단순한 시각적 특징뿐만 아니라, 외부 데이터셋에서 얻은 객체 간의 통계적 관계 정보를 신경망에 주입함으로써 모델의 추론 능력을 강화하였다. 특히 정성적 분석 결과, 다음과 같은 이점이 확인되었다.

- **문맥 기반 억제**: 시각적으로 유사하지만 문맥상 나타날 가능성이 낮은 객체(예: 철도를 키보드로 오인한 경우)를 효과적으로 제거한다.
- **중복 예측 감소**: 공생 관계 정보를 통해 한 장면에 존재할 가능성이 높은 객체 수를 더 정확히 예측하여 중복 검출을 억제한다.

### 한계 및 비판적 해석

- **메모리 비용**: Proposal의 수나 RPKG의 클래스 수가 증가함에 따라 Scene Graph의 모든 엣지를 계산하는 비용이 급격히 증가하는 메모리 문제가 존재한다.
- **환각(Hallucination) 현상**: 사전 지식에 너무 의존할 경우, 시각적 증거가 부족함에도 불구하고 강한 공생 관계(예: 사람과 휴대폰) 때문에 객체가 있다고 잘못 예측하는 환각 현상이 발생할 수 있다.
- **데이터셋 의존성**: VG와 COCO 간의 클래스 매핑을 수동으로 수행한 부분이 있으며, 이는 데이터셋 간의 세밀한 의미 차이를 완전히 반영하지 못했을 가능성이 있다.

## 📌 TL;DR

본 논문은 외부 지식 그래프(RPKG)를 활용하여 객체 제안 특징을 강화하는 **RP-FEM** 모델을 제안한다. 이 모델은 **Relation Head**와 **Graph Transformer**를 통해 객체 간의 관계를 모델링하며, 이를 통해 **중복 예측을 줄이고 문맥에 맞지 않는 오검출을 억제**함으로써 Mask R-CNN의 성능을 향상시킨다. 특히 데이터가 부족한 소수 클래스에서 성능 향상이 뚜렷하여, 향후 롱테일 분포 문제 해결 및 정교한 씬 이해 연구에 기여할 가능성이 크다.
