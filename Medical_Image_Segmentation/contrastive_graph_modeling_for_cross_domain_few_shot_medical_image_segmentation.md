# Contrastive Graph Modeling for Cross-Domain Few-Shot Medical Image Segmentation

Yuntian Bo, Tao Zhou, Zechao Li, Haofeng Zhang, and Ling Shao (2026)

## 🧩 Problem to Solve

본 논문은 **Cross-Domain Few-Shot Medical Image Segmentation (CD-FSMIS)** 문제를 해결하고자 한다. CD-FSMIS는 매우 적은 수의 어노테이션 데이터만으로 새로운 카테고리를 분할해야 할 뿐만 아니라, 학습 시 사용되지 않은 새로운 도메인(예: CT $\rightarrow$ MRI)에서도 모델이 일반화되어 작동해야 하는 도전적인 과제이다.

기존의 CD-FSMIS 방법론들은 주로 도메인 간의 차이를 줄이기 위해 도메인 특이적 정보(domain-specific information)를 필터링하여 제거하는 방식에 집중하였다. 그러나 저자들은 이러한 단순한 정보 제거 방식이 다음과 같은 두 가지 심각한 문제를 야기한다고 주장한다:

1. **특징 붕괴(Feature Collapse):** 유용한 정보까지 함께 제거되어 교차 도메인 성능의 상한선이 제한된다.
2. **소스 도메인 성능 저하:** 일반화 성능을 높이는 대신, 원래 학습 데이터였던 소스 도메인에서의 분할 정확도가 심각하게 떨어진다.

따라서 본 논문의 목표는 도메인에 관계없이 유지되는 의료 영상의 **구조적 일관성(Structural Consistency)**을 활용하여, 소스 도메인의 성능을 보존하면서도 타겟 도메인으로의 전이 성능을 극대화하는 프레임워크를 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 의료 영상이 모달리티(Modality)에 따라 외형은 다르지만, 해부학적 특성으로 인해 픽셀 간의 상대적 관계인 '구조(Structure)'는 일관되게 유지된다는 점에 착안하여 이를 **Graph** 형태로 모델링하는 것이다.

주요 기여 사항은 다음과 같다:

- **C-Graph 프레임워크:** 의료 영상의 구조적 일관성을 활용하여 도메인 간 일반화 성능을 높이는 새로운 프레임워크를 제안한다.
- **Structural Prior Graph (SPG) Layer:** 이미지 특징을 그래프로 표현하고, 서포트 셋(Support set)의 구조적 지식을 쿼리 그래프(Query graph)에 전이하며 전역적인 구조 모델링을 수행한다.
- **Subgraph Matching Decoding (SMD) 메커니즘:** 기존의 단순한 프로토타입 매칭(Prototypical matching) 방식에서 벗어나, 그래프 노드 간의 연결성(Connectivity)을 고려한 디코딩 방식을 제안한다.
- **Confusion-minimizing Node Contrast (CNC) Loss:** 노드 간의 모호성을 줄이고 서브그래프의 이질성을 완화하기 위해, 대조 학습(Contrastive Learning)을 통해 노드의 판별력을 높이는 손실 함수를 설계하였다.

## 📎 Related Works

### Few-Shot Medical Image Segmentation (FSMIS)

기존 FSMIS 연구들은 주로 프로토타입 네트워크(Prototypical Network)를 기반으로 하며, 적은 샘플로부터 클래스 대표 벡터(Prototype)를 추출하고 이를 쿼리 이미지와 매칭하는 방식을 사용한다. 최근에는 프로토타입 정제(Refinement)나 트랜스포머 기반의 정렬 방법들이 제안되었으나, 이들은 대부분 동일 도메인 내에서의 일반화만을 가정한다.

### Cross-Domain FSMIS (CD-FSMIS)

도메인 시프트를 해결하기 위해 RobustEMD는 텍스처 차이를 억제하는 매칭 메커니즘을, FAMNet은 주파수 영역에서 도메인 특이적 성분을 필터링하는 방식을 제안하였다. 하지만 본 논문은 이러한 '필터링' 방식이 소스 도메인의 성능을 희생시킨다는 점을 지적하며, 정보를 제거하는 대신 도메인 불변적인(Domain-agnostic) 구조 정보를 추출하는 방향으로 차별화를 꾀한다.

### Graph Neural Networks (GNNs)

GNN은 이미지 패치나 영역 간의 관계를 모델링하는 데 효과적이다. 기존 FSS 연구에서도 그래프를 사용해 서포트-쿼리 간의 상호작용을 강화하려 했으나, 본 논문은 단순한 매칭 보조 도구가 아니라 의료 영상의 핵심인 '해부학적 구조' 그 자체를 그래프로 모델링하여 Prior로 활용한다는 점에서 차별점이 있다.

## 🛠️ Methodology

### 1. 이미지 특징의 그래프화 (Image Features as a Graph)

입력 이미지 특징 맵 $F \in \mathbb{R}^{C \times H \times W}$의 각 픽셀 위치 $(h, w)$를 그래프의 노드 $\nu_i$로 정의한다. 이때 학습 가능한 위치 인코딩(Positional Encoding) $E$를 더해 공간 정보를 보존한다.
$$\nu_i = [F + E]_{:,h,w} \in \mathbb{R}^C$$
노드 간의 엣지 $e_{i,j}$는 코사인 유사도를 통해 결정하며, 이는 두 픽셀 간의 세만틱 관계를 반영한다.
$$e_{i,j} = \cos(\nu_i, \nu_j) = \frac{\nu_i^\top \nu_j}{\|\nu_i\|_2 \cdot \|\nu_j\|_2}$$

### 2. Structural Prior Graph (SPG) Layer

SPG 레이어는 다음의 세 단계로 구성되며, 로컬에서 글로벌 구조로 단계적으로 모델링을 수행한다.

- **Support Subgraph Linking (SSL):** 서포트 셋의 특정 클래스 노드들 간의 전역적 의존성을 모델링하여 서브그래프의 응집력을 높인다. 트랜스포머의 셀프 어텐션(Self-attention)을 통해 인접 행렬 $A^t$를 계산하고 노드 특징을 업데이트한다.
- **Interactive Subgraph Injection (ISI):** 서포트 서브그래프에서 학습된 구조적 지식을 크로스 어텐션(Cross-attention)을 통해 쿼리 그래프에 주입한다. 이를 통해 쿼리 이미지 내에서 타겟 카테고리에 해당하는 영역을 강조한다.
- **Graph Structure Modeling (GSM):** 동적 GCN(Dynamic GCN)을 사용하여 각 노드와 세만틱 유사도가 높은 $k$-최근접 이웃(k-nearest neighbors) 간의 엣지를 생성하고, MRConv(Max-relative graph convolution)를 통해 전역적인 해부학적 구조를 모델링한다.

### 3. Subgraph Matching Decoding (SMD)

기존의 프로토타입 방식(평균값 사용)이 클래스 내 변동성을 무시하고 픽셀을 고립된 개체로 취급하는 문제를 해결하기 위해, SMD는 노드 간의 연결성을 직접 활용한다.
먼저 노드들을 공통 임베딩 공간으로 투영한 후, 채널 가중치를 self-update 하여 중요 채널을 강조한다. 이후 서포트 노드와 쿼리 노드 간의 연결성 맵 $\Phi$를 생성하고, 이를 디코더 $D$에 입력하여 최종 마스크를 예측한다.
$$\Phi = \sigma(\psi(e^l_{V_q} e^o_{V_s(c)}))$$
$$\hat{y}_q(c) = \sigma(D(\Phi, V^l_q))$$

### 4. Confusion-minimizing Node Contrast (CNC) Loss

노드의 모호성을 제거하기 위해 설계된 대조 학습 손실 함수이다.

- **모호한 노드 식별:** 예측 결과의 엔트로피 $H$를 계산하여 임계값 $\delta$보다 높은, 즉 분류가 불확실한 '혼동 노드'들을 추출한다.
- **대조 학습:** 서브그래프 내의 혼동 노드 $p$는 세만틱 중심(Semantic center) $q$ 쪽으로 끌어당기고(Pull), 서브그래프 외의 혼동 노드 $n$은 멀리 밀어낸다(Push).
- **손실 함수:** 코사인 유사도 기반의 비용 행렬 $J$를 구성하여 다음과 같이 정의한다.
$$L_{cnc} = -\frac{1}{|p|} \langle 1, \log(\text{softmax}(J/\tau)) \rangle$$

전체 학습 목적 함수는 세그멘테이션 손실 $L_{seg}$와 CNC 손실의 가중 합으로 구성된다: $L_{total} = L_{seg} + \alpha L_{cnc}$.

## 📊 Results

### 실험 설정

- **데이터셋:** Abdominal CT, Abdominal MRI (교차 모달리티), Cardiac b-SSFP, Cardiac LGE (교차 시퀀스).
- **평가 지표:** Dice Sørensen coefficient (DSC %).
- **설정:** 1-way 1-shot 환경, ResNet-50 인코더 사용.

### 주요 결과

- **교차 도메인 성능:** Abdominal CT $\rightarrow$ MRI 시나리오에서 FAMNet 대비 DSC를 크게 향상시켰으며, 전반적으로 SOTA 성능을 달성하였다. 특히 간(Liver)보다 상대적으로 크기가 작은 신장(LK, RK) 및 심장 근육(LV-MYO) 등 소형 구조물 분할에서 매우 강력한 성능 향상을 보였다.
- **소스 도메인 성능 보존:** Table III에서 확인할 수 있듯, 기존 CD-FSMIS 방법론들이 소스 도메인에서 성능이 급격히 떨어지는 것과 달리, C-Graph는 소스 도메인에서도 매우 높은 정확도를 유지하였다. 이는 도메인 정보를 억제하지 않고 구조적 일관성을 활용한 전략의 승리라고 볼 수 있다.
- **범용적 일반화 (Cross-Context):** 학습하지 않은 전혀 다른 해부학적 컨텍스트(예: 복부 CT $\rightarrow$ 흉부 X-ray, 피부 경진 현미경)에서도 타 방법론 대비 우수한 성능을 보여, 제안 모델이 특정 도메인에 오버피팅되지 않고 범용적인 그래프 구조 추정 능력을 갖췄음을 입증하였다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 논문은 '도메인 특이적 정보의 제거'가 일반화에는 도움이 될 수 있으나, 모델의 표현력을 제한하여 소스 도메인 성능을 망치고 결국 교차 도메인 성능의 상한선을 낮춘다는 통찰을 제시하였다. 대신 의료 영상의 본질적인 특성인 해부학적 구조를 그래프로 모델링함으로써, 외형적 변화에 강건하면서도 세밀한 구조적 특징을 유지할 수 있었다.

### 한계 및 향후 과제

- **대형 객체 분할 성능:** 소형 객체에서는 탁월하지만, 간(Liver)과 같은 대형 객체에서는 성능 향상 폭이 상대적으로 적다. 이는 현재 고정된 이웃 크기 $k$가 대형 객체의 광범위한 수용 영역(Receptive field)을 커버하기에 부족하기 때문으로 분석되며, 향후 적응형 $k$ 설정 전략이 필요하다.
- **극심한 컨텍스트 시프트:** 모달리티 차이를 넘어 해부학적 구조 자체가 완전히 바뀌는 상황(Cross-context)에서는 일부 실패 사례가 발견되었다.

## 📌 TL;DR

본 논문은 의료 영상의 도메인 간 **구조적 일관성**을 활용한 **C-Graph** 프레임워크를 제안한다. 이미지 특징을 그래프로 변환하여 모델링하는 **SPG 레이어**, 연결성 기반의 **SMD 디코더**, 그리고 노드 모호성을 해결하는 **CNC 손실 함수**를 통해, 소스 도메인의 성능 저하 없이 타겟 도메인으로의 뛰어난 일반화 성능을 달성하였다. 이는 의료 영상 분석에서 단순한 특징 필터링보다 해부학적 구조 사전 지식(Structural Prior)을 활용하는 것이 훨씬 효과적임을 시사한다.
