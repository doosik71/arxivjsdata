# Vocabulary-Free 3D Instance Segmentation with Vision-Language Assistant

Guofeng Mei, Luigi Riz, Yiming Wang, Fabio Poiesi (2025)

## 🧩 Problem to Solve

본 논문은 3D 인스턴스 분할(3D Instance Segmentation, 3DIS) 분야에서 기존의 '폐쇄형 어휘(Closed-vocabulary)' 및 '개방형 어휘(Open-vocabulary)' 방식이 가진 한계를 극복하고자 한다. 

기존의 개방형 어휘 3DIS(OV3DIS) 모델들은 훈련 단계에서 보지 못한 클래스를 인식할 수 있다는 유연성을 제공하지만, 여전히 추론 시점에 사용자가 제공하는 특정 개념의 집합, 즉 '어휘 사전 정보(Vocabulary prior)'가 필요하다. 이는 모델이 "이 장면에는 어떤 물체들이 있는가?"라는 개방형 질문에 스스로 답할 수 없음을 의미한다.

이러한 제약은 특히 장면의 시맨틱 정보가 동적으로 변하거나, 사용자가 정의하지 않은 희귀한 물체가 포함된 환경에서 동작해야 하는 보조 로봇 애플리케이션 등에 큰 제약이 된다. 따라서 본 논문의 목표는 어떠한 사전 정의된 어휘나 사용자 쿼리 없이도 3D 장면 내의 모든 객체 인스턴스를 스스로 발견하고 레이블링하는 **Vocabulary-Free 3D Instance Segmentation (VoF3DIS)** 태스크를 정의하고 이를 해결하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 대규모 시각-언어 어시스턴트(Vision-Language Assistant, VLA)와 2D 개방형 어휘 분할 모델을 결합하여, 3D 장면 내의 시맨틱 카테고리를 자율적으로 발견하고 이를 3D 포인트 클라우드로 투영하는 것이다.

주요 기여 사항은 다음과 같다:
1. **VoF3DIS 태스크 정의**: 추론 시점에 사전 어휘 정보 없이 인스턴스를 분할하고 레이블링하는 새로운 문제 설정을 제안하였다.
2. **PoVo 프레임워크 제안**: 훈련이 필요 없는(Training-free) 제로샷 방식으로, VLA를 통해 장면 어휘를 생성하고 이를 3D 인스턴스 마스크로 연결하는 시스템을 구축하였다.
3. **Spectral Clustering 기반 마스크 형성**: 기하학적 특성으로 분할된 Superpoint들을 2D 마스크의 일관성(Mask coherence)과 시맨틱 일관성(Semantic coherence)을 모두 고려하여 병합하는 새로운 전략을 제안하였다.

## 📎 Related Works

**1. Vocabulary-free 모델**
이미지 인식 분야에서는 CaSED와 같이 웹 규모의 데이터베이스에서 캡션을 검색하거나 VLA를 쿼리하여 후보 어휘를 생성하는 방식이 연구되었다. 그러나 이러한 접근은 주로 2D 이미지 분류나 세그멘테이션에 국한되었으며, 3D 데이터의 희소성과 포인트 레벨의 시맨틱 표현이 필요한 3D 인스턴스 분할로 확장된 사례는 본 논문이 처음이다.

**2. Open-vocabulary 3D Scene Understanding**
최근 연구들은 CLIP과 같은 Vision-Language Model(VLM)을 3D로 확장하여 OV3DIS를 구현해 왔다. OpenMask3D, Open3DIS 등이 대표적이며, 이들은 주로 클래스 불가지론적(Class-agnostic) 3D 세그멘테이션 모델을 사용하거나 Superpoint를 병합하는 방식을 취한다. 하지만 이 모든 방법은 추론 시점에 사용자가 쿼리를 제공하거나 미리 정의된 클래스 리스트가 있어야 한다는 점에서 본 논문이 제안하는 VoF3DIS와 차별화된다.

## 🛠️ Methodology

PoVo의 전체 파이프라인은 크게 세 단계로 구성된다: 장면 어휘 생성, 3D 인스턴스 마스크 형성, 그리고 텍스트 정렬 포인트 표현 생성이다.

### 1. 장면 어휘 생성 (Scene Vocabulary Generation)
먼저 다각도에서 촬영된 이미지 $V = \{I_n\}_{n=1}^N$를 활용한다.
- **VLA 쿼리**: 각 이미지 $I_n$에 대해 LLaVA와 같은 시각-언어 어시스턴트에게 "장면 내의 객체 이름을 나열하라"고 요청하여 초기 리스트 $C^-_n$을 얻는다.
- **그라운딩(Grounding)**: VLA의 환각(Hallucination) 현상을 방지하기 위해, Grounded-SAM을 사용하여 $C^-_n$의 카테고리들이 실제 이미지 내에 존재하는지 확인하고 2D 인스턴스 마스크 $M^{2D}_n$을 생성한다. 
- **최종 어휘 집합**: 확인된 카테고리들의 합집합을 통해 해당 장면의 최종 어휘 사전 $C$를 구축한다.

### 2. 3D 인스턴스 마스크 형성 (3D Instance Mask Formation)
기하학적 특성만을 이용해 생성된 Superpoint들을 시맨틱 정보를 활용해 병합한다.
- **Superpoint 생성**: Graph cut 알고리즘을 사용하여 포인트 클라우드를 기하학적으로 균일한 영역인 Superpoint $Q$로 분할한다.
- **Superpoint 병합 (Spectral Clustering)**: 세 가지 점수(Score)를 기반으로 하는 인접 행렬(Affinity Matrix) $A$를 구성한다. $A = A^M \odot A^S \odot A^C$

  1. **마스크 일관성 점수 ($a^M_{ij}$)**: 두 Superpoint $Q_i, Q_j$가 동일한 2D 인스턴스 마스크 $M^{2D}_t$와 겹치는 정도를 IoU로 측정한다.
     $$a^M_{ij} = \sum_{t} g(O_{i,t}, \tau_{iou}) \cdot g(O_{j,t}, \tau_{iou})$$
     여기서 $O_{i,t}$는 Superpoint $Q_i$와 2D 마스크 $M^{2D}_t$의 IoU이며, $g(x, \tau)$는 $x > \tau$일 때만 값을 가지는 함수이다.
  2. **시맨틱 일관성 점수 ($a^S_{ij}$)**: 각 Superpoint에 할당된 가장 빈번한 2D 레이블을 텍스트 인코더로 임베딩하여 코사인 유사도를 계산한다.
     $$a^S_{ij} = \frac{f_{Q_i}^\top f_{Q_j}}{\|f_{Q_i}\| \|f_{Q_j}\|} \odot \mathbb{1}(f_{Q_i}^\top f_{Q_j} > \tau_{sim})$$
  3. **공간 연결성 점수 ($a^C_{ij}$)**: 두 Superpoint 간의 최소 거리가 임계값 $\tau_c$보다 작으면 1, 아니면 0으로 설정한다.

이후 Laplacian 행렬 $L = D^{-1/2}(D-A)D^{-1/2}$의 고유벡터(Eigenvector)를 계산하고, K-means 클러스터링을 통해 Superpoint들을 최종 3D 인스턴스 마스크 $M^{3D}$로 병합한다.

### 3. 텍스트 정렬 포인트 표현 (Text-aligned Point Representation)
최종적으로 각 3D 인스턴스 마스크에 시맨틱 레이블을 할당하기 위해 포인트 레벨 특징을 추출한다.
- **시각적 특징**: CLIP 시각 인코더를 사용하여 다각도 이미지에서 추출한 객체 크롭(Crop) 이미지의 특징 $f^v_l$을 얻는다.
- **텍스트적 특징**: 병합된 Superpoint 수준의 텍스트 임베딩 $f_{Q_i}$를 추가하여 시각-텍스트 모달리티 갭을 줄인다.
- **최종 할당**: 생성된 특징과 장면 어휘 $C$ 내의 텍스트 임베딩 간의 유사도를 측정하여 최적의 레이블을 부여한다.

## 📊 Results

### 실험 설정
- **데이터셋**: ScanNet200, Replica, S3DIS.
- **평가 지표**: Average Precision (AP), $AP_{50}$, $AP_{25}$. 
- **VoF3DIS 특수 지표**: VLA가 생성한 레이블이 Ground Truth와 정확히 일치하지 않을 수 있으므로, **BERTScore**를 사용하여 시맨틱 유사도가 0.8 이상인 경우 정답으로 처리하였다.
- **비교 대상**: OpenScene, OpenMask3D, OVIR-3D, SAM3D, SAI3D, OVSAM3D, Open3DIS.

### 주요 결과
- **ScanNet200**: PoVo는 OV3DIS 설정과 VoF3DIS 설정 모두에서 기존 방법론들을 압도하였다. 특히 VoF3DIS 설정에서 Open3DIS$\dagger$보다 높은 성능을 보였으며, 희귀 클래스($AP_{tail}$)에서도 강건한 성능을 나타냈다.
- **Replica**: OV3DIS 설정에서 Open3DIS 대비 AP가 2.7%p, OVIR-3D 대비 9.7%p 향상되었으며, VoF3DIS 설정에서도 가장 높은 성능을 기록하였다.
- **S3DIS**: 제로샷 설정임에도 불구하고 Novel 클래스에 대해 $AP_{50}$ 기준 Open3DIS보다 우수한 성능을 보였다.

### 절제 연구 (Ablation Study)
- **특징 융합**: 시각적 특징(VisEmb)과 텍스트 특징(TxtEmb)을 모두 사용하고 Superpoint 기반 풀링(SPool)을 적용했을 때 성능이 가장 높았다.
- **시맨틱 병합**: Superpoint 병합 시 텍스트 유사도($TxtSim$)를 고려하는 것이 2D 마스크의 노이즈를 억제하여 성능을 향상시킴을 확인하였다.

## 🧠 Insights & Discussion

**강점**
PoVo는 3D 데이터에 대한 추가적인 훈련 없이 2D 기초 모델(Foundation Models)과 VLA의 지식을 효과적으로 전이하였다. 특히, 단순히 2D 마스크를 3D로 투영하는 것에 그치지 않고, Spectral Clustering 과정에서 시맨틱 일관성을 통합함으로써 훨씬 정교한 3D 인스턴스 마스크를 형성할 수 있었다.

**한계 및 논의**
1. **VLA 의존성**: 시스템의 성능이 LLaVA와 같은 VLA의 객체 인식 능력에 크게 의존한다. 비록 Grounded-SAM으로 환각을 필터링하지만, VLA가 아예 인식하지 못한 객체는 발견할 수 없다.
2. **계산 비용**: 다각도 이미지에 대해 VLA와 SAM을 반복적으로 실행해야 하므로, 실시간 처리에는 한계가 있을 수 있다.
3. **기하학적 이해**: 본 모델은 주로 시각-언어 정보에 의존하며, 3D 포인트 클라우드 자체의 고유한 기하학적 구조를 깊게 활용하는 부분은 상대적으로 부족하다.

## 📌 TL;DR

본 논문은 사전 어휘 정보 없이 3D 장면의 객체를 스스로 발견하고 분할하는 **Vocabulary-Free 3D Instance Segmentation (VoF3DIS)**라는 새로운 태스크와 이를 해결하는 **PoVo** 프레임워크를 제안하였다. PoVo는 VLA(LLaVA)로 장면 어휘를 생성하고, Grounded-SAM과 Spectral Clustering을 통해 Superpoint를 정교하게 병합함으로써, 훈련 없이도 기존의 개방형 어휘 모델들을 뛰어넘는 성능을 달성하였다. 이 연구는 향후 어떠한 사전 정보 없이도 주변 환경을 이해해야 하는 자율 주행 로봇의 환경 인식 분야에 중요한 기여를 할 것으로 기대된다.