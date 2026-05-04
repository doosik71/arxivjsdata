# Associate Everything Detected: Facilitating Tracking-by-Detection to the Unknown

Zimeng Fang, Chao Liang, Xue Zhou, Shuyuan Zhu, and Xi Li (2024)

## 🧩 Problem to Solve

본 논문은 다중 객체 추적(Multi-Object Tracking, MOT) 분야에서 기존의 폐쇄 어휘 MOT(Closed-Vocabulary MOT, CV-MOT)와 개방 어휘 MOT(Open-Vocabulary MOT, OV-MOT) 사이의 성능 격차를 해소하고자 한다. 

CV-MOT는 훈련 데이터에 정의된 특정 카테고리만을 추적하도록 설계되어 있어, 자율 주행이나 증강 현실(AR)과 같이 예상치 못한 새로운 클래스를 처리해야 하는 실제 응용 환경에서는 확장성이 떨어진다는 문제가 있다. 반면, OV-MOT는 Multimodal Large Language Model(MLLM) 등을 활용해 미지의 카테고리를 추적할 수 있으나, 특정 카테고리에 대해 정교하게 튜닝된 CV-MOT 모델보다 성능이 낮게 나타나는 경향이 있다.

결과적으로 본 논문의 목표는 어떠한 오프더쉘(off-the-shelf) 검출기(detector)와도 결합 가능하며, 알려진 클래스와 미지의 클래스 모두를 동시에 효율적으로 추적할 수 있는 통합 프레임워크인 AED(Associate Everything Detected)를 구축하는 것이다.

## ✨ Key Contributions

AED의 핵심 아이디어는 추적 문제를 '검출된 모든 것을 연관 짓는' 단순하고 강력한 구조로 재정의하는 것이다. 구체적인 기여 사항은 다음과 같다.

1. **통합 추적 프레임워크 제안**: CV-MOT와 OV-MOT의 간극을 메우기 위해, 검출 단계에서 어떤 객체가 검출되든 이를 연관(association)시킬 수 있는 통합 Tracking-by-Detection 프레임워크를 제안한다.
2. **Sim-decoder 설계**: 사전 지식(예: 모션 큐)에 의존하지 않고, 오직 강력한 특징 학습을 통해 객체 쿼리와 트랙 쿼리 간의 유사도를 디코딩하는 $\text{sim-decoder}$를 도입한다.
3. **Association-Centric Learning 메커니즘**: 학습 단계에서 공간적(Spatial), 시간적(Temporal), 그리고 클립 간(Cross-clip) 세 가지 관점의 대조 학습을 수행하여, 추론 단계의 연관 작업과 학습 목표를 일치시킨다. 이를 통해 미지의 카테고리에 대해서도 일반화 가능한 강건한 특징을 학습한다.

## 📎 Related Works

본 논문은 MOT 패러다임을 세 가지 방향으로 분석하며 기존 연구의 한계를 지적한다.

1. **Tracking-by-Detection MOT**: SORT, ByteTrack 등 많은 방법론이 IoU 거리나 Kalman Filter와 같은 모션 사전 지식(prior knowledge)에 크게 의존한다. 그러나 OV-MOT 환경에서는 다양한 카테고리마다 모션 패턴이 매우 상이하므로, 이러한 고정된 사전 지식에 의존하는 방식은 성능 저하를 초래한다.
2. **Tracking-by-Query MOT**: MOTR, MOTRv2와 같이 Transformer 쿼리를 사용하여 객체를 추적하는 방식은 엔드투엔드(end-to-end) 학습이 가능하다는 장점이 있으나, 쿼리 자체가 특정 카테고리에 종속적이기 때문에 학습 시 보지 못한 미지의 카테고리를 추적하는 데 어려움이 있다.
3. **Open-Vocabulary MOT**: OVTrack과 같이 CLIP의 지식을 활용하거나 GLEE와 같이 대규모 데이터를 활용하는 방법들이 제안되었다. 하지만 이러한 방법들은 실제 연관(association) 과정에서 모델이 필요로 하는 특성을 간과하여 최적의 성능을 내지 못하는 한계가 있다.

## 🛠️ Methodology

### 전체 파이프라인
AED는 추론 시 임의의 검출기(Co-DETR, YOLOX 등)로부터 bounding box를 입력받아 이를 $\text{object queries}$로 인코딩한다. 이전 프레임에서 유지되고 있는 $\text{track queries}$와 현재의 $\text{object queries}$를 $\text{sim-decoder}$에 입력하여 유사도를 계산하고, 이를 기반으로 Hungarian 알고리즘을 통해 매칭을 수행한다.

### Sim-decoder 및 Multi-head Weight Attention
$\text{sim-decoder}$는 $\text{object queries}$와 $\text{track queries}$ 사이의 유사도 $S$와 가중치 응답 $W$를 디코딩한다. 이를 위해 본 논문은 $\text{multi-head weight attention}$을 제안하며, 이는 기존 Attention에서 Value 성분을 제거하고 유사도 산출에 집중한 구조이다.

각 헤드 $i$에 대해 유사도 $S_i$와 가중치 $W_i$는 다음과 같이 계산된다.
$$S_i = \text{norm}(\text{linear}_i(Q)) \cdot \text{norm}(\text{linear}_i(K))^T$$
$$W_i = \text{linear}_i(Q) \cdot \text{linear}_i(K)^T$$

여기서 $S_i$는 코사인 유사도(cosine similarity)를 사용하며, 이는 Softmax를 사용했을 때 발생하는 '상대적 최댓값' 문제(새로 등장한 객체도 강제로 높은 유사도를 갖게 되는 현상)를 방지하기 위함이다. 최종 출력 $S$와 $W$는 모든 헤드의 평균으로 계산된다.
$$S = \max(0, \text{mean}(S_1, S_2, \dots, S_h))$$
$$W = \text{mean}(W_1, W_2, \dots, W_h)$$

### Association-Centric Learning
학습 시 $\text{sim-decoder}$를 통해 도출된 $S$와 $W$를 활용하여 세 가지 차원의 대조 학습을 수행한다.

1. **Spatial Contrastive Learning**: 단일 프레임 내의 객체들을 구분하여 공간적 변별력을 높인다.
2. **Temporal Contrastive Learning**: 현재 프레임의 검출 결과와 과거의 트랙 쿼리를 매칭하여 시간적 ID 일관성을 확보한다.
3. **Cross-Clip Contrastive Learning**: 비디오 클립 내의 모든 객체 쿼리를 버퍼에 저장하고 이를 대조하여 장기적인 ID 일관성을 학습한다.

학습을 위한 손실 함수로는 동일 ID 간의 응답을 높이고 서로 다른 ID 간의 응답을 낮추는 $\text{embedding loss}$ $L_{embed}$와 $\text{focal loss}$ 기반의 $L_{aux}$를 사용한다.

$$L_{embed}(W) = \sum_{r} \log [1 + \sum_{w^+_r} \sum_{w^-_r} \exp(w^-_r - w^+_r)]$$

최종 손실 함수 $L_{total}$은 다음과 같이 구성된다.
$$L_{total} = \lambda_{spatial} L_{spatial} + \lambda_{temporal} L_{temporal} + \lambda_{cross-clip} L_{cross-clip} + \lambda_{l1} L_{l1} + \lambda_{giou} L_{giou}$$
여기서 $L_{l1}$과 $L_{giou}$는 박스 정밀도를 높이기 위한 box refinement loss이다.

## 📊 Results

### 실험 설정
- **데이터셋**: TAO (OV-MOT 평가), SportsMOT, DanceTrack (CV-MOT 평가).
- **검출기**: RegionCLIP, Co-DETR, YOLOX.
- **지표**: OV-MOT는 TETA를, CV-MOT는 HOTA, MOTA, IDF1를 사용한다.

### 주요 결과
1. **OV-MOT 성능 (TAO)**: Co-DETR 검출기를 사용할 때, AED는 Base 카테고리뿐만 아니라 Novel(미지) 카테고리에 대해서도 OVTrack보다 높은 TETA 성능을 보였다. 특히 연관 성능을 직접적으로 나타내는 $\text{AssocA}$ 지표에서 비약적인 향상을 보였다.
2. **CV-MOT 성능 (SportsMOT, DanceTrack)**: 
    - **SportsMOT**: 사전 지식을 사용하지 않았음에도 불구하고, 최신 Tracking-by-Detection 방법론인 Deep-EIoU보다 높은 HOTA($77.0\%$)와 IDF1($80.0\%$)을 기록하며 SOTA 성능을 달성했다.
    - **DanceTrack**: 복잡한 모션과 심한 가려짐(occlusion)이 발생하는 환경에서도 기존 CV-MOT 방법론들을 능가하는 결과를 보였다.
3. **성능 상한 분석 (Performance Ceiling)**: 검출 박스를 Ground Truth(GT)로 대체하여 실험한 결과, AED는 ByteTrack나 OVTrack보다 더 높은 TETA와 AssocA를 기록하며 모델 자체의 연관 능력이 매우 뛰어남을 입증하였다.

## 🧠 Insights & Discussion

### 강점
AED는 모션 큐와 같은 사전 지식에 의존하지 않고 오직 외관 특징의 강건한 학습에 집중함으로써, 매우 역동적인 움직임이나 다양한 카테고리가 등장하는 환경에서도 유연하게 대처할 수 있다. 또한, 특정 검출기에 종속되지 않는 plug-and-play 구조를 가져 범용성이 매우 높다.

### 한계 및 비판적 해석
논문에서 명시했듯이, AED는 모션 큐를 완전히 배제했기 때문에 객체가 매우 밀집된(extremely crowded) 상황에서는 추적 성능이 저하되는 한계가 있다. 이는 외관 특징만으로는 구분이 어려운 상황에서 모션 정보가 보완책이 될 수 있음을 시사한다. 또한, 현재의 쿼리는 검출기에서 제공하는 정보에만 의존하고 있어, 텍스트 쿼리나 다른 모달리티 정보를 통합한다면 미지 객체에 대한 일반화 성능을 더욱 높일 수 있을 것으로 보인다.

## 📌 TL;DR

본 논문은 CV-MOT와 OV-MOT를 통합하여 '검출된 모든 객체를 추적'하는 AED 프레임워크를 제안한다. 사전 지식 없이 유사도 디코딩 기반의 $\text{sim-decoder}$와 추론 과정을 모사한 $\text{association-centric learning}$을 통해, 알려진 객체와 미지의 객체 모두에서 뛰어난 추적 성능을 달성하였다. 이 연구는 향후 특정 클래스에 국한되지 않고 모든 사물을 추적해야 하는 범용 MOT 시스템 구축에 중요한 기여를 할 것으로 판단된다.