# Is clustering enough for LiDAR instance segmentation? A state-of-the-art training-free baseline

Corentin Sautier, Gilles Puy, Alexandre Boulch, Renaud Marlet, Vincent Lepetit (2025)

## 🧩 Problem to Solve

본 논문은 자율주행의 핵심 기술인 LiDAR 기반 Panoptic Segmentation에서 발생하는 데이터 라벨링 비용 문제를 해결하고자 한다. Panoptic Segmentation은 모든 포인트에 대해 시맨틱 클래스를 할당하는 Semantic Segmentation과, 'thing' 클래스(차량, 보행자 등)의 개별 인스턴스를 구분하는 Instance Segmentation을 결합한 작업이다.

현재의 State-of-the-art (SOTA) 방법론들은 주로 end-to-end 딥러닝 아키텍처를 사용하며, 특히 인스턴스 예측을 위해 방대한 양의 수동 인스턴스 라벨링 데이터에 의존한다. 그러나 대규모 포인트 클라우드 데이터셋에 대해 인스턴스 단위의 정밀한 라벨을 생성하는 것은 시간과 비용 측면에서 매우 큰 병목 현상을 야기한다. 따라서 본 연구의 목표는 인스턴스 라벨에 대한 학습 없이, 오직 Semantic 라벨만을 활용하여 경쟁력 있는 수준의 Panoptic Segmentation을 달성하는 training-free baseline을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 현대의 고성능 Semantic Segmentation 네트워크가 생성하는 예측 결과가 매우 정교하므로, 이를 기반으로 적절히 설계된 단순한 클러스터링 알고리즘만으로도 SOTA 수준의 인스턴스 분리가 가능하다는 것이다.

이를 위해 저자들은 **ALPINE (A Light Panoptic INstance Extractor)**이라는 학습이 필요 없는 클러스터링 파이프라인을 제안한다. ALPINE의 핵심은 3D 공간이 아닌 Bird's Eye View (BEV) 투영 공간에서 클래스별 특성을 고려한 k-nearest neighbors (kNN) 그래프를 구축하고, 객체의 물리적 크기 정보를 활용한 거리 임계값 및 박스 분할(Box Splitting) 메커니즘을 통해 인스턴스를 정밀하게 추출하는 것이다.

## 📎 Related Works

기존의 LiDAR Panoptic Segmentation 접근 방식은 크게 세 가지로 분류된다.

1.  **Detection-based methods**: 객체 검출기(Detector)와 시맨틱 분할기(Segmentor)를 결합하여 박스를 먼저 치고 내부를 분할하는 방식이다.
2.  **Query-based methods**: MaskFormer와 같이 학습 가능한 쿼리를 통해 포인트들을 인스턴스에 할당하는 end-to-end Transformer 구조를 사용한다.
3.  **Clustering-based methods**: 시맨틱 예측 후 'thing' 클래스 포인트들을 클러스터링하는 전통적인 방식이다. Euclidean Cluster나 SLIC 기반 방법들이 있으나, 최근의 딥러닝 기반 방법들에 비해 성능이 낮다고 평가받아 외면받는 경향이 있었다.

본 논문은 이러한 전통적인 클러스터링 기반 방식이 현대의 강력한 Semantic Segmentation 백본과 결합될 때, 복잡한 학습 기반 인스턴스 헤드보다 더 효율적이고 강력할 수 있음을 입증하며 기존의 end-to-end 패러다임에 의문을 제기한다.

## 🛠️ Methodology

ALPINE은 별도의 학습 과정 없이, 입력으로 주어진 Semantic Segmentation 결과물만을 사용하여 인스턴스를 추출한다. 전체 파이프라인은 다음과 같은 단계로 구성된다.

### 1. 클래스별 독립 처리 및 BEV 투영
먼저, 전체 포인트 클라우드 $\mathcal{P}$에서 특정 시맨틱 클래스 $c$에 해당하는 포인트 집합 $\mathcal{P}^c$만을 추출하여 독립적으로 처리한다. 이후, 객체들이 수직으로 쌓여 있는 경우가 드물다는 점을 이용하여 3D 포인트를 (x, y) 평면으로 직교 투영한 Bird's Eye View (BEV) 표현으로 변환한다.

### 2. k-nearest neighbors (kNN) 그래프 구축
BEV 공간에서 각 포인트에 대해 $k$개의 가장 가까운 이웃(기본값 $k=32$)을 탐색하여 유향 그래프 $\mathcal{G}$를 생성한다. 이때, 두 포인트 사이의 거리가 클래스별 임계값 $t^c$보다 크면 엣지를 제거한다. 최종적으로는 누락된 반대 방향 엣지를 추가하여 무향 그래프로 만든다.

### 3. 연결 요소(Connected Components) 추출
구축된 그래프에서 연결 요소(Connected Components)를 찾아 각각의 컴포넌트에 고유한 인스턴스 ID를 부여한다. 이 과정은 단순한 그래프 탐색으로 이루어지므로 매우 빠르다.

### 4. 거리 임계값 $t^c$의 결정
학습을 하지 않기 때문에 $t^c$를 설정하는 것이 매우 중요하다. 저자들은 데이터셋의 정답 라벨을 보는 대신, 인터넷에서 검색 가능한 각 클래스별 객체의 평균 물리적 크기(예: 미국/유럽 차량의 평균 크기)를 활용한다. $t^c$는 해당 클래스 객체의 기준 바운딩 박스(Reference Bounding Box)의 가장 짧은 변의 길이로 설정된다.

### 5. 박스 분할 (Box Splitting) 메커니즘
단순 클러스터링으로 인해 두 개 이상의 인스턴스가 하나로 묶이는 과소 분할(Under-segmentation) 문제를 해결하기 위해 박스 분할 전략을 사용한다.
- 만약 생성된 클러스터의 바운딩 박스가 기준 박스 $B$에 30%의 마진을 더한 크기보다 크다면, 해당 클러스터는 여러 인스턴스가 합쳐진 것으로 간주한다.
- 이 경우, 임계값 $t$에 대해 이진 탐색(Binary Search)을 수행하여, 모든 서브 클러스터가 기준 박스 $B$ 안에 들어올 때까지 재귀적으로 클러스터를 분할한다.

## 📊 Results

### 실험 설정
- **데이터셋**: SemanticKITTI, nuScenes, SemanticPOSS.
- **평가 지표**: Panoptic Quality (PQ), mIoU, Recognition Quality (RQ), Segmentation Quality (SQ).
- **백본**: MinkUNet, WaffleIron (WI), PTv3 등 다양한 SOTA 시맨틱 분할 모델을 사용하였다.

### 주요 결과
- **전반적 성능**: ALPINE은 인스턴스 라벨 학습 없이도 많은 SOTA 지도학습(Supervised) 방법론들을 능가하였다. 특히 SemanticKITTI의 공식 리더보드에서 UniSeg의 시맨틱 예측과 ALPINE을 결합하여 1위를 달성하였다.
- **인스턴스 헤드 교체 실험**: 기존 Panoptic 방법론들의 인스턴스 예측 헤드를 제거하고 ALPINE으로 교체했을 때, 거의 모든 경우에서 PQ가 상승하였다. 이는 현재의 학습 기반 인스턴스 헤드들이 단순한 클러스터링보다 월등한 성능을 내지 못하고 있음을 시사한다.
- **속도 및 효율성**: GPU 없이 단일 스레드 CPU만으로 실시간 처리가 가능하며, 파라미터 튜닝이 거의 필요 없다.
- **오라클 분석**: 정답 시맨틱 라벨을 사용한 'Semantic Oracle'과 정답 인스턴스 경계를 사용한 'Instance Oracle'을 비교한 결과, 성능 향상 폭이 시맨틱 오라클에서 훨씬 크게 나타났다. 이는 현재 Panoptic Segmentation의 성능 병목이 인스턴스 추출보다는 시맨틱 분할의 정확도에 있음을 의미한다.

## 🧠 Insights & Discussion

본 논문은 "인스턴스 라벨링에 기반한 학습이 정말 필요한가?"라는 근본적인 질문을 던진다. 실험 결과, 고성능의 시맨틱 분할 결과만 있다면 단순한 기하학적 클러스터링만으로도 충분한 수준의 인스턴스 분리가 가능하다는 것이 입증되었다.

**강점**: 
- 학습이 필요 없으므로 라벨링 비용이 전혀 들지 않는다.
- 완전히 설명 가능한(Explainable) 알고리즘이며, 구현이 매우 간단하다.
- CPU 기반의 실시간 추론이 가능하여 실제 자율주행 시스템 임베딩에 유리하다.

**한계 및 비판적 해석**:
- **임계값 의존성**: 클래스당 하나의 고정 임계값을 사용하므로, 두 객체가 임계값보다 더 가깝게 붙어 있는 경우 분리에 실패할 수 있다. 
- **포인트 밀도 영향**: LiDAR의 각도 해상도가 낮아 포인트가 희소해질수록(특히 원거리 객체) 클러스터링 성능이 저하되는 경향이 있다. 이는 하드웨어 성능 향상(더 조밀한 포인트 클라우드)으로 해결될 수 있는 문제이기도 하다.

결론적으로, ALPINE은 향후 제안되는 모든 Panoptic Segmentation 모델이 반드시 넘어야 할 '강력한 베이스라인'으로서의 역할을 수행한다.

## 📌 TL;DR

본 논문은 LiDAR Panoptic Segmentation에서 비용이 많이 드는 인스턴스 학습 대신, 고성능 시맨틱 분할 결과와 BEV 기반의 최적화된 클러스터링(ALPINE)만으로 SOTA 성능을 달성할 수 있음을 보여준다. 특히 인스턴스 추출 단계는 이미 포화 상태(Saturated)이며, 전체 성능 향상을 위해서는 인스턴스 헤드 학습보다 시맨틱 분할의 정확도를 높이는 것이 더 중요함을 시사한다. 이 연구는 향후 인스턴스 라벨링 없는 효율적인 3D 씬 이해 연구에 중요한 기준점이 될 것이다.