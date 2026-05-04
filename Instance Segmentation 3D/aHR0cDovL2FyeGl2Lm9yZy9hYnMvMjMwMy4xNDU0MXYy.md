# UnScene3D: Unsupervised 3D Instance Segmentation for Indoor Scenes

David Rozenberszki, Or Litany, Angela Dai (2024)

## 🧩 Problem to Solve

본 논문은 실내 장면의 3D Instance Segmentation을 수행함에 있어 발생하는 막대한 수동 어노테이션 비용 문제를 해결하고자 한다. 기존의 3D 인스턴스 분할 방법론들은 정밀하고 조밀한 3D 마스크 레이블에 의존하는 지도 학습(Supervised Learning) 방식이었으며, 이는 데이터 구축 비용이 매우 높다는 한계가 있다.

따라서 본 연구의 목표는 사전 정의된 클래스 범주에 구애받지 않는 Class-agnostic 3D Instance Segmentation을 위해, 어떠한 인간의 개입이나 수동 레이블링 없이도 작동하는 완전 비지도 학습(Fully-unsupervised) 프레임워크인 UnScene3D를 제안하는 것이다. 특히 복잡하고 물체가 밀집된 실내 환경에서도 효과적으로 객체를 분리해내는 것을 지향한다.

## ✨ Key Contributions

UnScene3D의 핵심 아이디어는 자기지도 학습(Self-supervised) 기반의 멀티모달 특징량과 기하학적 구조를 결합하여 초기 가설을 세우고, 이를 반복적인 자기 학습(Self-training) 루프를 통해 정교화하는 것이다.

주요 기여 사항은 다음과 같다:
1. **멀티모달 특징 기반의 Pseudo Mask 생성**: 2D 색상 정보와 3D 기하 구조 정보를 결합한 자기지도 학습 특징량을 활용하여, 수동 레이블 없이도 초기 인스턴스 후보군인 Pseudo Mask를 생성한다.
2. **기하학적 프리미티브(Geometric Primitives) 도입**: 고해상도 3D 데이터의 계산 효율성을 높이고 노이즈를 억제하기 위해, 씬을 기하학적으로 과분할(Oversegmentation)하여 단순화된 프리미티브 단위로 연산을 수행한다.
3. **반복적 자기 학습(Iterative Self-training)**: 생성된 희소한 Pseudo Mask를 초기 감독 신호로 사용하여 3D Transformer 모델을 학습시키고, 모델의 확신도가 높은 예측값을 다시 학습 데이터에 추가하는 루프를 통해 마스크의 밀도와 정확도를 높인다.

## 📎 Related Works

본 논문은 3D 인스턴스 분할을 위한 기존 접근 방식들을 다음과 같이 분석하며 차별점을 제시한다.
- **자기지도 3D 사전 학습(Self-supervised 3D Pretraining)**: 다양한 뷰나 로컬 증강을 통해 강력한 특징 추출기를 만들지만, 객체 인스턴스에 대한 개념 자체를 구축하지는 않는다.
- **약지도 학습(Weakly-supervised 3D Segmentation)**: 3D 박스 어노테이션이나 템플릿 매칭을 사용하지만, 여전히 일부 수동 데이터나 대규모 2D 데이터셋에 의존한다.
- **클러스터링 기반 분할(Clustering-based Segmentation)**: HDBSCAN나 Felzenszwalb 알고리즘 같은 전통적 방법은 기하학적 특성만 사용하므로, 복잡하고 밀집된 실내 장면에서 정밀한 인스턴스 분리가 어렵다.
- **비지도 2D 인스턴스 분할(Unsupervised 2D Instance Segmentation)**: FreeSolo나 CutLER와 같은 최신 2D 방법론이 존재하지만, 이를 3D로 투영할 경우 뷰 간 불일치(View inconsistency)와 폐쇄(Occlusion) 문제로 인해 성능이 크게 저하된다.

## 🛠️ Methodology

UnScene3D의 전체 파이프라인은 '초기 Pseudo Mask 생성' 단계와 '반복적 자기 학습' 단계로 구성된다.

### 1. 초기 Pseudo Mask 생성 (Initial Pseudo Mask Generation)
먼저 씬의 복잡도를 낮추기 위해 **Geometric Primitives** 과정을 거친다. Mesh의 정점(Vertex)들을 법선 벡터(Normal)와 색상이 유사한 그룹으로 묶어 그래프를 단순화(Coarsening)한다.

이후, 다음과 같은 절차로 Pseudo Mask를 추출한다:
- **특징량 집계(Feature Aggregation)**: 자기지도 학습으로 사전 학습된 2D 특징량(RGB 이미지에서 추출하여 3D로 투영)과 3D 특징량을 각각 추출한다.
- **Normalized Cut (NCut) 적용**: 두 모달리티의 코사인 유사도를 기반으로 인접 행렬(Adjacency Matrix) $W$를 구성한다. 이후 다음의 일반화된 고유값 문제(Generalized Eigenvalue Problem)를 푼다:
$$(D - W)v = \lambda Dv$$
여기서 $D$는 차수 행렬(Degree Matrix)이며, 두 번째로 작은 고유값 $\lambda$에 대응하는 고유벡터 $v$를 통해 배경과 전경을 분리한다.
- **연결성 제약**: NCut 결과가 공간적으로 분리된 영역을 포함할 수 있으므로, 물리적 연결성을 확인하여 가장 활성화 값이 높은 연결 성분만을 유지함으로써 마스크의 연속성을 보장한다.

### 2. 반복적 자기 학습 (Self-Training)
초기 생성된 Pseudo Mask는 매우 희소(Sparse)하므로, 이를 정교화하기 위해 **3D Transformer** 기반의 백본(Mask3D 아키텍처)을 사용한다.

- **학습 루프**: 
    1. 초기 Pseudo Mask $M_0$를 사용하여 모델을 학습시킨다.
    2. 학습된 모델로부터 예측된 마스크 중 확신도가 높은 상위 $K$개의 예측값을 선택한다.
    3. 기존 Pseudo Mask 집합에 새로운 예측값을 추가하여 $M_t$를 업데이트하고 다시 학습한다.
- **손실 함수 및 최적화**: 헝가리안 할당(Hungarian assignment) 방식을 사용하여 Dice Loss와 Binary Cross-Entropy Loss의 가중 합을 사용한다. 특히, 이전 주기와 겹침 정도가 낮은 샘플의 손실을 제외하는 **DropLoss**를 적용하여 노이즈가 전파되는 것을 방지한다.

## 📊 Results

### 실험 설정
- **데이터셋**: ScanNet(학습 및 평가), S3DIS(전이 학습 평가), ARKitScenes(정성 평가).
- **지표**: Average Precision (AP) @ IoU 25%, 50% 및 평균 AP (AP).

### 주요 결과
- **ScanNet 성능**: UnScene3D는 AP 기준 15.9를 기록하며, 기존의 비지도 학습 베이스라인인 Felzenszwalb(5.0)나 HDBSCAN(1.6) 대비 약 3배 이상의 성능 향상을 보였다.
- **S3DIS 전이 학습**: ScanNet에서 사전 학습된 특징을 사용하여 S3DIS 데이터셋에서도 AP 21.4를 달성하며 타 방법론들을 압도했다.
- **멀티모달 효과**: 3D 전용 특징량만 사용했을 때(AP 13.3)보다 2D와 3D를 모두 활용했을 때(AP 15.9) 더 높은 성능을 보였으며, 이는 색상과 기하 정보가 상호 보완적임을 시사한다.
- **자기 학습의 효과**: 반복 횟수가 증가함에 따라 AP가 지속적으로 상승하며, 약 4회 반복 시 성능이 포화(Saturate)되는 양상을 보인다.

## 🧠 Insights & Discussion

### 강점 및 해석
본 논문은 단순히 클러스터링에 의존하지 않고, **'희소하지만 깨끗한'** 초기 마스크를 생성한 뒤 이를 **'점진적으로 확장'**하는 전략을 취했다. 실험 결과, FreeMask와 같이 초기부터 밀도 높지만 노이즈가 많은 마스크를 사용하는 것보다, 적은 수의 정확한 마스크로 시작하여 자기 학습을 통해 밀도를 높이는 것이 최종 성능에 훨씬 유리함을 입증했다.

### 한계 및 비판적 논의
1. **표현 방식의 의존성**: 현재 기하학적 단순화를 위해 Mesh 표현 방식에 의존하고 있다. Point Cloud나 Voxel 기반의 다른 표현 방식에서도 동일한 수준의 그래프 단순화 효과를 얻을 수 있을지에 대한 검증이 더 필요하다.
2. **소형 객체 누락**: 그래프 단순화(Coarsening) 과정에서 펜이나 휴대폰과 같은 매우 작은 객체들이 프리미티브 단계에서 사라져 Pseudo Mask 생성 단계에서 누락될 가능성이 있다.
3. **노이즈 강화 위험**: 자기 학습 루프 특성상, 초기에 잘못 예측된 마스크가 학습 데이터에 포함될 경우 이후 단계에서 해당 오류가 강화(Reinforce)될 위험이 존재한다.

## 📌 TL;DR

UnScene3D는 수동 어노테이션 없이 실내 3D 장면의 인스턴스를 분할하는 완전 비지도 학습 프레임워크이다. **멀티모달 특징량 $\rightarrow$ 기하학적 단순화 $\rightarrow$ NCut 기반 Pseudo Mask 생성 $\rightarrow$ 반복적 자기 학습**으로 이어지는 파이프라인을 통해 기존 클러스터링 기반 방법론 대비 성능을 3배 이상 끌어올렸다. 이 연구는 고비용의 3D 데이터 레이블링 문제를 해결함으로써, 향후 로보틱스나 자율 주행을 위한 3D 환경 이해 연구에 중요한 기반이 될 것으로 기대된다.