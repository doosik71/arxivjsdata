# A Unified Query-based Paradigm for Camouflaged Instance Segmentation

Bo Dong, Jialun Pei, Rongrongrong Gao, Tian-Zhu Xiang, Shuo Wang, Huan Xiong (2023)

## 🧩 Problem to Solve

본 논문은 **Camouflaged Instance Segmentation (CIS)** 문제를 해결하고자 한다. 위장된 객체(camouflaged objects)는 배경과 색상 및 패턴이 매우 유사하여, 일반적인 객체 검출보다 정확한 위치 파악(localization)과 인스턴스 분할(instance segmentation)이 훨씬 어렵다. 

기존의 인스턴스 분할 방식은 크게 세 가지(detect-then-segment, label-then-cluster, direct instance segmentation)로 나뉘는데, 전자의 두 방식은 바운딩 박스나 픽셀 임베딩 클러스터링과 같은 간접적이고 단계적인 과정에 의존하며, 특히 위장 시나리오에서는 배경과 객체의 구분이 모호하여 성능이 저하되는 한계가 있다. 최근 Transformer 기반의 OSFormer가 제안되었으나, 여전히 NMS(Non-Maximum Suppression)와 같은 수작업 기반의 후처리 과정이 필요하며 성능 개선의 여지가 남아있다.

따라서 본 논문의 목표는 후처리가 필요 없는 **Direct Set Prediction** 방식을 통해 위장된 인스턴스를 효율적으로 분할하고, 특히 객체의 영역(region) 정보와 경계(boundary) 정보를 통합적으로 활용하여 CIS 성능을 극대화하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Composed Query Learning** 패러다임을 도입하여 영역과 경계 정보를 동시에 학습하는 통합 쿼리 기반 프레임워크인 **UQFormer**를 설계한 것이다.

1. **Composed Query Learning**: Mask Query와 Boundary Query를 각각 설계하고, 이 둘을 상호작용시켜 통합된 공유 표현(shared composed query representation)을 학습한다. 이를 통해 위장 객체의 전역적 영역 특성과 국소적 경계 단서를 효율적으로 통합한다.
2. **Multi-task Learning Framework**: 인스턴스 분할(Instance Segmentation)과 인스턴스 경계 검출(Instance Boundary Detection)을 동시에 수행하는 멀티태스크 학습 구조를 제안한다. 두 작업이 공유 쿼리를 통해 서로 보완하며 더 강건한 인스턴스 레벨 표현을 학습하도록 유도한다.
3. **Direct Set Prediction**: DETR 스타일의 쿼리 기반 예측 방식을 채택하여, NMS와 같은 복잡한 후처리 과정 없이 직접적으로 인스턴스 집합을 예측한다.

## 📎 Related Works

### 인스턴스 분할 (Instance Segmentation)
기존의 Mask R-CNN과 같은 2단계 모델은 추론 속도가 느리며, YOLACT나 CondInst 같은 1단계 모델은 픽셀 임베딩 클러스터링에 의존한다. 최근에는 SOLO, Mask2Former와 같이 직접적으로 마스크를 예측하는 방식이 주목받고 있다. CIS 분야에서는 OSFormer가 최초의 Transformer 기반 모델로 제안되었으나, 여전히 NMS 기반 후처리가 필요하다는 한계가 있다.

### 쿼리 기반 모델 (Query-Based Models)
DETR 이후 Object Query를 통해 객체의 전역적 의미를 학습하는 방식이 다양한 비전 태스크(MaskFormer, SeqFormer 등)에 적용되었다. 기존 모델들은 주로 쿼리를 무작위로 초기화하지만, 본 논문은 위장 객체의 특성을 고려하여 영역과 경계라는 두 가지 관점의 쿼리를 설계하고 이를 통합하는 방식을 제안함으로써 차별점을 둔다.

## 🛠️ Methodology

### 1. Feature Encoder
입력 이미지 $I \in \mathbb{R}^{H \times W \times 3}$로부터 CNN 백본(ResNet-50/101)을 통해 멀티스케일 특징 $\mathcal{F}_{ms}^{enc} = \{f_1^{enc}, \dots, f_i^{enc}\}$를 추출한다. 이후 Deformable Attention이 적용된 Transformer Encoder를 통해 강화된 특징 $\mathcal{F}_{ms}^{p} = \{f_2^{p}, \dots, f_i^{p}\}$를 얻는다. 

이때, 쿼리 학습을 위해 멀티스케일 마스크 특징 $\mathcal{F}_{ms}^{m}$과 경계 특징 $\mathcal{F}_{ms}^{b}$를 다음과 같이 정의한다:
$$\begin{cases} \mathcal{F}_{ms}^{m} = (\mathcal{F}_{ms}^{p}), \\ f_i^{b} = f_{lm}^{b}(f_{close}^{b}(f_i^{p}) + f_i^{p}), i \in \{2, 3, 4\} \end{cases}$$
여기서 $f_{close}^{b}$는 Closing 연산(Erosion 후 Dilation)으로, 위장 객체 내부의 노이즈를 제거하여 경계 쿼리 학습의 오동작을 방지한다.

### 2. Multi-scale Unified Query Learning
#### 쿼리 초기화 (Query Initialization)
- **Mask Queries ($\mathcal{Q}_m^0$):** PointRend 전략을 참고하여, 전역 특징에서 신뢰도가 높은 Salient Points를 선택하여 초기화한다.
  $$\mathcal{Q}_m^0 = \text{Topk}(\sum f_{int}^{m}(f_i^m)), i \in \{2, 3, 4\}$$
- **Boundary Queries ($\mathcal{Q}_b^0$):** DETR과 유사하게 무작위로 초기화한다.

#### 쿼리 상호작용 학습 (Query Interactive Learning)
Multi-scale Unified Learning Transformer Decoder는 다음과 같은 단계로 쿼리를 업데이트한다:
1. **Mask Cross-Attention**: 마스크 쿼리가 $\mathcal{F}_{ms}^{m}$와 상호작용하여 $\mathcal{Q}_m'$를 생성한다.
   $$\mathcal{Q}_i^{m \prime} = \text{CA}(\mathcal{Q}_i^{m} + \mathcal{P}_{mq}^i, f_i^{m \prime})$$
2. **Boundary Cross-Attention**: 경계 쿼리가 $\mathcal{F}_{ms}^{b}$와 상호작용하여 $\mathcal{Q}_b'$를 생성한다.
   $$\mathcal{Q}_i^{b \prime} = \text{CA}(\mathcal{Q}_i^{b} + \mathcal{P}_{bq}^i, f_i^{b \prime})$$
3. **Composed Query 생성**: 두 쿼리를 결합한 후 MHSA(Multi-Head Self-Attention)와 FFN을 통해 통합 쿼리 $\mathcal{Q}_{mb}$를 생성한다.
   $$\begin{cases} \mathcal{Q}_i^{mb \prime} = \mathcal{Q}_i^{m \prime} + \mathcal{Q}_i^{b \prime}, \\ \mathcal{Q}_i^{mb} = \text{FFN}(\text{MHSA}(\mathcal{Q}_i^{mb \prime})) \end{cases}$$
이 $\mathcal{Q}_{mb}$는 다음 스테이지의 마스크 및 경계 쿼리로 다시 사용되어 반복적으로 최적화된다.

### 3. Multi-task Learning 및 예측
학습된 통합 쿼리 $\mathcal{Q}_{mb}$와 고해상도(HR) 특징을 결합하여 세 가지를 예측한다:
- **위치 신뢰도 점수 ($\mathcal{S}_i$):** $\mathcal{S}_i = \text{MLP}_{\times 1}(\mathcal{Q}_i^{mb})$
- **인스턴스 경계 ($\mathcal{B}_i$):** $\mathcal{B}_i = \text{MLP}_{\times 3}(\mathcal{Q}_i^{mb}) \otimes f_{hr}^{b}$
- **인스턴스 마스크 ($\mathcal{M}_i$):** $\mathcal{M}_i = \text{MLP}_{\times 3}(\mathcal{Q}_i^{mb}) \otimes f_{hr}^{m}$
여기서 $\otimes$는 요소별 곱셈(element-wise multiplication)을 의미한다.

### 4. 손실 함수 (Loss Function)
DETR의 Bipartite Matching을 사용하여 예측값과 Ground Truth를 매칭한다.
- 마스크와 경계 감독에는 $\mathcal{L}_{BCE}$와 $\mathcal{L}_{Dice}$를 사용한다: $\mathcal{L}_{n} = \gamma(\mathcal{L}_{BCE} + \mathcal{L}_{Dice}), n \in \{\text{mask, boundary}\}$
- 위치 점수 감독에는 $\mathcal{L}_{BCE}$만 사용한다.
- 최종 손실 함수: 
$$\mathcal{L}_{total} = \gamma_{score} \mathcal{L}_{score}^{BCE} + \alpha \mathcal{L}_{mask} + \beta \mathcal{L}_{boundary}$$

## 📊 Results

### 실험 설정
- **데이터셋**: COD10K, NC4K
- **지표**: $\text{AP}, \text{AP}_{50}, \text{AP}_{75}$
- **백본**: ResNet-50, ResNet-101, Swin-tiny

### 정량적 결과
Table 1에 따르면, UQFormer는 14개의 SOTA 방법론보다 우수한 성능을 보였다.
- **COD10K (ResNet-50)**: $\text{AP}$ 기준 OSFormer 대비 9.2% 향상, Mask2Former 대비 10.2% 향상.
- **NC4K (ResNet-50)**: $\text{AP}$ 기준 OSFormer 대비 11.5% 향상.
- 특히 $\text{AP}_{75}$에서 큰 폭의 상승이 관찰되었는데, 이는 위장 객체의 정밀한 경계 추출 능력이 향상되었음을 시사한다.

### 효율성 분석
Table 7에서 UQFormer는 OSFormer에 비해 파라미터 수를 19.5% 줄이고 FLOPs를 31.9% 감소시켰음에도 불구하고 $\text{mAP}$를 4.4% 높였다. 이는 OSFormer가 사용한 방대한 양의 쿼리(2200개)에 비해, UQFormer는 효율적인 쿼리 설계(20개)를 통해 훨씬 가벼운 구조로 더 높은 성능을 냈음을 의미한다.

## 🧠 Insights & Discussion

### 강점 및 분석
- **상호보완적 쿼리 학습**: 단순한 마스크 예측이 아니라, 경계(Boundary) 정보를 쿼리 단계에서 통합함으로써 배경과 객체의 미세한 차이를 더 잘 포착할 수 있게 되었다.
- **멀티태스크 학습의 효과**: 경계 검출 태스크가 마스크 분할 태스크의 가이드 역할을 하여, 겹쳐 있거나 인접한 객체들을 개별 인스턴스로 분리하는 능력이 향상되었다. (Ablation study에서 boundary learning 제거 시 $\text{AP}$가 감소함을 통해 입증)
- **효율적인 초기화**: 무작위 초기화보다 Salient Points를 이용한 초기화가 학습 수렴 속도를 높이고 성능을 개선했다.

### 한계 및 논의
- **데이터셋 의존성**: CAMO++ 데이터셋이 공개되지 않아 COD10K로 학습하고 NC4K로 평가하는 방식을 취했는데, 더 다양한 데이터셋에서의 일반화 성능 검증이 필요하다.
- **백본 영향**: Swin-tiny 백본 사용 시 성능이 비약적으로 상승하는 것으로 보아, 모델 구조뿐만 아니라 강력한 Feature Extractor의 영향이 크다는 점을 알 수 있다.

## 📌 TL;DR

본 논문은 위장된 객체 분할(CIS)을 위해 **영역과 경계 정보를 통합 학습하는 쿼리 기반 프레임워크 UQFormer**를 제안한다. Mask Query와 Boundary Query를 결합하여 Composed Query를 생성하고, 이를 통해 인스턴스 분할과 경계 검출을 동시에 수행함으로써 후처리 없이도 SOTA 성능을 달성하였다. 특히 기존 모델 대비 연산량과 파라미터를 크게 줄이면서도 정밀도는 높여, 향후 CIS 연구의 효율적인 베이스라인이 될 가능성이 높다.