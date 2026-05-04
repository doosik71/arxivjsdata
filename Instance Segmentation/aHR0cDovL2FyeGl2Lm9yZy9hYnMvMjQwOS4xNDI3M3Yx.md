# Lidar Panoptic Segmentation in an Open World

Anirudh S Chakravarthy, Meghana Reddy Ganesina, Peiyun Hu, Laura Leal-Taixé, Shu Kong, Deva Ramanan, Aljosa Osep (2024)

## 🧩 Problem to Solve

본 논문은 Lidar Panoptic Segmentation (LPS)의 현실적인 제약 사항인 '폐쇄된 세계(Closed-world)' 가정을 해결하고자 한다. 기존의 LPS 방법론들은 테스트 환경에서도 학습 시 정의된 $K$개의 시맨틱 클래스 어휘집(vocabulary)만이 존재한다고 가정한다. 그러나 실제 자율주행 환경에서는 로봇이 학습 데이터셋에 포함되지 않은 새로운 클래스의 객체(예: 쓰러진 나무 줄기, 전복된 트럭 등)를 마주하게 되며, 이러한 '알 수 없는(Unknown)' 영역을 정확히 인식하고 분할하는 것이 안전한 주행을 위해 필수적이다.

따라서 본 연구는 Lidar Panoptic Segmentation in an Open World (LiPSOW)라는 새로운 문제 설정을 제안한다. LiPSOW의 목표는 미리 정의된 $K$개의 알려진(Known) 클래스를 정확히 분할하는 동시에, 어휘집에 없는 새로운 Thing 클래스와 Stuff 클래스가 등장했을 때 이를 'Unknown'으로 인식하고 개별 인스턴스로 분할해내는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 알려진 클래스와 알 수 없는 클래스에 대해 '통합된 처리 방식(Unified treatment)'을 적용하는 것이다. 저자들은 클래스별 인스턴스 분할 방식보다는 클래스 불가지론적(Class-agnostic)인 Bottom-up 그룹화 방식이 알 수 없는 클래스에 대해 더 강건한 성능을 보인다는 점에 주목하였다.

이를 위해 본 논문은 다음과 같은 설계 아이디어를 제시한다. 첫째, Outlier Exposure 기법을 통해 알려진 $K$개 클래스와 그 외의 'Other' 클래스를 구분하는 시맨틱 분할 네트워크를 학습시킨다. 둘째, 'Thing'과 'Unknown'으로 분류된 포인트들에 대해 계층적 분할 트리(Hierarchical Segmentation Tree)를 구축하여 가능한 모든 세그먼트 후보를 생성한다. 셋째, 학습된 Objectness scoring function을 통해 이 트리에서 최적의 컷(Cut) 위치를 찾아내어, 클래스 정보와 무관하게 객체 인스턴스를 추출하는 방식을 제안한다.

## 📎 Related Works

기존의 LPS 연구들은 주로 데이터 기반의 엔드투엔드 학습에 집중하며, 포인트 세트의 표현 학습이나 인코더-디코더 구조의 최적화에 주력해 왔다. 하지만 이러한 방법들은 학습된 클래스 내에서만 그룹화를 수행하므로, 학습 시 보지 못한 새로운 클래스에 대해서는 일반화 능력이 현저히 떨어진다는 한계가 있다.

또한, 기존의 Open-set Lidar Segmentation 연구들은 주로 'Stuff' 클래스가 모두 레이블링 되어 있다는 비현실적인 가정을 세우고, 오직 'Thing' 클래스만이 알 수 없는 상태로 등장한다고 가정하였다. 그러나 실제 환경에서는 교량, 터널과 같은 새로운 'Stuff' 영역 역시 등장할 수 있다. 본 논문은 Thing과 Stuff 모두가 Unknown이 될 수 있다는 점에서 기존 연구들과 차별화된다.

## 🛠️ Methodology

본 논문에서 제안하는 방법론인 OWL (Open-World Lidar Panoptic Segmentor)은 크게 두 단계의 파이프라인으로 구성된다.

### 1. Semantic Segmentation Network
첫 번째 단계에서는 포인트 클라우드 $P \in \mathbb{R}^{N \times 3}$를 입력받아 $K+1$개의 클래스로 분류하는 네트워크를 학습시킨다. 여기서 $K$는 알려진 클래스이며, $+1$은 'Other' 클래스로, 학습 데이터에서 희귀한 클래스들을 하나로 묶어 Unknown의 대표값으로 사용한다.
- **구조**: KPConv 기반의 인코더-디코더 아키텍처를 사용하며, 최종 출력은 $S \in \mathbb{R}^{N \times (K+1)}$ 형태의 시맨틱 맵이다.
- **학습**: Cross-entropy loss를 사용하여 학습하며, 이를 통해 네트워크가 알려진 클래스와 알 수 없는 클래스(Other)를 구분할 수 있게 한다.

### 2. Class-Agnostic Instance Segmentation
두 번째 단계에서는 시맨틱 분류 결과 중 'Thing' 또는 'Other'로 분류된 포인트들을 대상으로 인스턴스 분할을 수행한다.

**계층적 분할 트리(Hierarchical Segmentation Tree) 구축**: 
DBSCAN 기반의 유클리드 거리 클러스터링을 거리 임계값 $\epsilon$을 점진적으로 줄여가며 반복 적용한다. 이를 통해 포인트 세그먼트들이 부모-자식 관계를 갖는 트리 구조 $T$가 형성되며, 트리의 아래로 내려갈수록 더 세밀하게 분할된 세그먼트들이 위치하게 된다.

**Objectness Scoring Function**:
각 세그먼트 $p$가 실제 객체일 확률을 예측하는 함수 $f(p) \rightarrow [0, 1]$를 학습시킨다.
- **방식**: 포인트 세그먼트의 특징을 풀링하여 PointNet 스타일의 분류기나 회귀기를 통해 점수를 매긴다.
- **손실 함수**: 세그먼트와 실제 정답(GT) 인스턴스 간의 Intersection-over-Union (IoU) 값을 직접 회귀하거나, 이진 교차 엔트로피(Binary Cross-Entropy) 손실을 사용하여 객체 여부를 판단한다. 논문에서는 회귀 방식이 더 부드러운 점수 분포를 생성하여 최적의 컷을 찾는 데 유리하다고 언급한다.

**Unique Point-to-Instance Assignment (Tree-Cut)**:
트리의 각 노드에 부여된 Objectness 점수를 바탕으로, 전체 세그먼트의 품질을 최대화하는 최적의 '컷' 위치를 결정한다. 알고리즘은 트리를 순회하며 자식 노드의 점수가 부모보다 낮아지는 지점에서 컷을 수행함으로써, 포인트들이 서로 중복되지 않고 유일한 인스턴스 ID를 갖도록 보장한다.

## 📊 Results

### 실험 설정
- **데이터셋**: SemanticKITTI를 학습 및 검증셋으로, KITTI360을 테스트셋으로 사용하여 교차 데이터셋 평가(Cross-dataset evaluation)를 수행한다.
- **지표**: 알려진 클래스에 대해서는 PQ (Panoptic Quality)와 mIoU를 사용하고, 알 수 없는 클래스에 대해서는 Recall과 UQ (Unknown Quality)를 측정한다. 특히 UQ는 False Positive에 대한 페널티를 제외하고 Recall 중심으로 측정한다.

### 주요 결과
- **Closed-World 성능**: SemanticKITTI 검증셋에서 OWL은 최신 방법론들과 경쟁 가능한 수준의 PQ를 달성하였다. 특히 Semantic Oracle(정답 시맨틱 맵 사용) 실험에서 PQ 98.3%를 기록하여, 본 방법론의 인스턴스 그룹화 능력이 매우 뛰어남을 입증하였다.
- **Open-World 성능**: KITTI360 테스트셋에서 OWL은 알려진 클래스뿐만 아니라 알 수 없는 클래스에 대해서도 압도적인 성능을 보였다.
    - **Unknown Recall**: 4D-PLS가 약 6.0%의 알 수 없는 객체를 회수할 때, OWL은 45.1%를 회수하였다.
    - **UQ**: OWL은 36.3%의 UQ를 기록하여 베이스라인들을 크게 상회하였다.
- **Vocabulary 분석**: 희귀 클래스들을 'Other' 클래스로 통합하여 학습시킨 Vocabulary 1이 단순히 매우 희귀한 클래스만 제외한 Vocabulary 2보다 교차 데이터셋 일반화 성능이 더 높게 나타났다.

## 🧠 Insights & Discussion

본 논문은 Lidar Panoptic Segmentation에서 인스턴스 그룹화 자체보다는 **시맨틱 분류(Point Classification)의 정확도가 전체 성능의 병목 지점**임을 시사한다. Semantic Oracle 실험 결과, 정답 시맨틱 맵만 주어진다면 단순한 계층적 클러스터링만으로도 거의 완벽한 인스턴스 분할이 가능함을 확인했기 때문이다.

또한, 교차 데이터셋 평가에서 성능 저하가 발생하는 원인을 분석한 결과, 동일 도시 내에서도 지역에 따라 'Building'이나 'Vegetation' 같은 Stuff 클래스의 외형적 특성이 달라지며, 이것이 'Wall'이나 'Gate' 같은 새로운 클래스와 혼동을 일으키기 때문임을 밝혔다. 이는 향후 연구에서 Outlier Synthesis 등을 통해 해결해야 할 과제이다.

비판적으로 해석하자면, 본 모델은 실시간성(Real-time)을 확보하지 못했다. 포인트 클라우드의 크기가 매우 크기 때문에 계층적 트리 구축과 추론 과정에서 상당한 연산 시간이 소요된다는 점이 실제 시스템 적용 시의 한계점으로 작용할 수 있다.

## 📌 TL;DR

본 논문은 자율주행 환경에서 학습되지 않은 새로운 객체를 인식하고 분할해야 하는 **Open-World Lidar Panoptic Segmentation (LiPSOW)** 문제를 정의하고, 이를 해결하기 위한 **OWL** 프레임워크를 제안한다. OWL은 $(K+1)$-way 시맨틱 분할과 클래스 불가지론적인 계층적 세그먼트 트리 컷(Tree-cut) 방식을 결합하여, 알려진 객체와 알 수 없는 객체 모두를 효율적으로 분할한다. 실험 결과, 특히 새로운 클래스의 객체를 찾아내는 Recall 성능에서 기존 방법론 대비 비약적인 향상을 보였으며, 이는 향후 자율주행 시스템의 안전성과 지속적 학습(Continual Learning)을 위한 중요한 기초 연구가 될 것으로 기대된다.