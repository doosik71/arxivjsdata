# OpenMask3D: Open-Vocabulary 3D Instance Segmentation

Ayça Takmaz, Elisabetta Fedele, Robert W. Sumner, Marc Pollefeys, Federico Tombari, Francis Engelmann (2023)

## 🧩 Problem to Solve

기존의 3D Instance Segmentation 방식은 훈련 데이터셋에 정의된 제한된 클래스 집합만을 인식할 수 있는 **Closed-vocabulary** 패러다임에 의존한다. 이는 실제 환경에서 훈련 단계에 포함되지 않은 새로운 객체(Novel objects)를 인식하거나, 사용자의 자유로운 텍스트 쿼리(Free-form queries)에 따라 객체를 분할해야 하는 요구사항을 충족시키지 못한다는 치명적인 한계가 있다.

최근 Open-vocabulary 3D scene understanding 연구들이 등장하여 각 포인트(point)에 대해 쿼리가 가능한 특징(feature)을 학습함으로써 이 문제를 해결하려 했으나, 이러한 방식들은 주로 **Semantic segmentation** 수준에 머물러 있으며, 동일한 클래스에 속하는 여러 개의 개별 객체 인스턴스를 분리해내는 **Instance segmentation** 능력은 갖추지 못했다.

따라서 본 논문의 목표는 훈련 시 정의되지 않은 클래스에 대해서도 제로샷(Zero-shot)으로 대응할 수 있는 **Open-vocabulary 3D Instance Segmentation** 작업을 정의하고, 이를 수행할 수 있는 OpenMask3D 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 직관은 포인트 기반(Point-based) 특징 추출 방식에서 벗어나 **인스턴스 마스크 기반(Instance-mask oriented)**의 특징 계산 방식을 도입하는 것이다.

1. **Open-vocabulary 3D Instance Segmentation 작업 정의**: 텍스트 쿼리와 유사한 3D 객체 인스턴스를 식별하는 새로운 태스크를 정의하였다.
2. **Zero-shot 프레임워크 제안**: 추가적인 파인튜닝 없이 클래스 불가지론적(Class-agnostic) 3D 마스크 제안과 CLIP 기반의 이미지 특징 집계(Aggregation)를 결합하여 제로샷 인스턴스 분할을 가능하게 하였다.
3. **인스턴스 중심의 특징 표현**: 포인트별 특징을 단순히 평균 내는 것이 아니라, 각 인스턴스가 가장 잘 보이는 뷰(View)를 선택하고 SAM(Segment Anything Model)을 통해 정밀한 2D 마스크를 생성하여 CLIP 특징을 추출함으로써 인스턴스 구분 능력을 극대화하였다.

## 📎 Related Works

### Closed-vocabulary 3D Semantic and Instance Segmentation

기존의 Mask3D와 같은 최신 기법들은 Transformer 아키텍처를 통해 높은 성능을 보이지만, 훈련 데이터의 라벨 수(예: ScanNet200의 200개 클래스)에 묶여 있다. 이는 수십만 개의 명사가 존재하는 실제 언어의 다양성을 수용하기에는 턱없이 부족하다.

### Foundation Models & Open-vocabulary 2D Segmentation

CLIP, ALIGN과 같은 대규모 시각-언어 모델(VLM)은 텍스트-이미지 임베딩 공간을 통해 제로샷 전이 학습을 가능하게 했다. 이를 2D 세그멘테이션에 적용한 OpenSeg, OV-Seg 등이 있으며, 본 연구는 이러한 2D의 성공을 3D 인스턴스 수준으로 확장하고자 한다.

### Open-vocabulary 3D Scene Understanding

OpenScene, LERF, DFF 등의 연구는 2D 특징을 3D로 리프팅(Lifting)하거나 NeRF를 활용해 포인트별 특징 필드를 생성한다. 그러나 이들은 포인트 단위의 세만틱 정보를 제공할 뿐, 개별 객체 인스턴스를 분리하는 능력은 부족하다는 한계가 있다.

## 🛠️ Methodology

OpenMask3D는 전체적으로 두 단계의 파이프라인으로 구성된다: **Class-agnostic mask proposal head**와 **Mask-feature computation module**이다.

### 1. Class-agnostic Mask Proposals

먼저, 클래스 정보와 상관없이 3D 공간에서 객체 후보군인 binary instance masks $\{m_{1}^{3D}, \dots, m_{M}^{3D}\}$를 생성한다.

- **구조**: MinkowskiUNet 기반의 Sparse convolutional backbone과 Transformer decoder로 구성된 pre-trained Mask3D 모델을 사용한다.
- **특이사항**: 모델의 가중치는 고정(Frozen)하며, 기존 모델이 출력하는 클래스 라벨과 신뢰도 점수는 완전히 버리고 오직 **Binary mask**만을 활용한다.
- **후처리**: 공간적으로 연속되지 않은 마스크를 처리하기 위해 DBSCAN 클러스터링을 수행하여 마스크를 더 작은 단위로 쪼갠다.

### 2. Mask-feature Computation Module

생성된 각 마스크 $m_i^{3D}$에 대해 Open-vocabulary 쿼리가 가능한 특징 벡터를 계산한다.

#### (1) Frame Selection (뷰 선택)

인스턴스가 가장 잘 보이는 $k_{view}$개의 프레임을 선택한다. 가시성 점수 $s_{ij}$는 다음과 같이 계산된다:
$$s_{ij} = \frac{vis(i, j)}{\max_{j'} (vis(i, j'))}$$
여기서 $vis(i, j)$는 $i$번째 마스크의 포인트 중 $j$번째 프레임에서 가려지지 않고 보이는 포인트의 수이다. 가시성 판단은 카메라의 내/외적 파라미터를 이용해 2D로 투영한 후, 측정된 깊이 값 $d$와 계산된 거리 $w$를 비교하여 $w - d > k_{threshold}$인 경우 occluded(가려짐)로 판단한다.

#### (2) 2D Mask Computation & Multi-scale Crops

선택된 프레임에서 정밀한 2D 마스크 $m_{2D}^*$를 얻기 위해 SAM을 활용한다.

- **SAM 기반 정제**: 3D 마스크의 투영 포인트 중 $k_{sample}$개를 무작위로 샘플링하여 SAM의 입력으로 넣는다. 이 과정을 $k_{rounds}$번 반복하여 가장 높은 신뢰도(confidence score)를 가진 2D 마스크를 최종 선택한다.
- **Multi-scale Cropping**: 생성된 2D 마스크를 중심으로 3단계의 크롭 영역 $b_1, b_2, b_3$를 생성한다. $b_1$은 타이트한 바운딩 박스이며, $b_2, b_3$는 $k_{exp}=0.1$ 상수를 이용하여 점진적으로 확장된 영역이다. 이는 주변 문맥(Context) 정보를 함께 포착하기 위함이다.

#### (3) CLIP Feature Extraction & Aggregation

- **추출**: 생성된 모든 크롭 이미지들을 CLIP visual encoder(ViT-L/14)에 통과시켜 이미지 임베딩을 얻는다.
- **집계**: 각 크롭 및 선택된 뷰 전체에 대해 평균 풀링(Average-pooling)을 수행하여 최종적인 **per-mask feature representation**을 생성한다.

## 📊 Results

### 실험 설정

- **데이터셋**: ScanNet200 (Validation set), Replica.
- **지표**: Average Precision (AP), $\text{AP}_{50}$, $\text{AP}_{25}$.
- **비교 대상**: Mask3D (Fully supervised), OpenScene (Open-vocabulary, point-based). OpenScene의 경우 공정한 비교를 위해 OpenMask3D의 마스크 내 포인트 특징을 평균 내어 사용하도록 수정하였다.

### 정량적 결과

- **ScanNet200**: Fully supervised 모델인 Mask3D가 전체적인 AP는 가장 높지만, 클래스 빈도가 낮은 **Tail categories**에서는 OpenMask3D가 다른 Open-vocabulary 모델들보다 압도적으로 높은 성능을 보였다. (Table 1 참조)
- **Generalization**: ScanNet200으로 학습된 마스크 예측기를 사용하여 Replica 데이터셋(OOD)에서도 평가한 결과, OpenMask3D가 다른 Open-vocabulary 기법들보다 우수한 성능을 보이며 일반화 능력을 입증하였다. (Table 2 참조)
- **Oracle Mask 실험**: Ground Truth 마스크를 제공했을 때, OpenMask3D는 특히 Tail categories에서 Fully supervised 모델인 Mask3D보다도 높은 AP를 기록하였다. 이는 특징 추출 방식 자체가 매우 강력함을 시사한다.

### 정성적 결과

- **다양한 쿼리 대응**: 단순 클래스 명칭뿐만 아니라 색상, 질감, 기하학적 구조, 용도(Affordance), 상태(State) 등의 자유로운 텍스트 쿼리에 대해 정확하게 인스턴스를 분할해내는 능력을 보여주었다.
- **경계 명확성**: 포인트 기반의 OpenScene은 히트맵 형태의 모호한 결과를 내놓는 반면, OpenMask3D는 인스턴스 단위의 특징을 사용하므로 매우 뚜렷한 인스턴스 경계를 생성한다.

## 🧠 Insights & Discussion

### 강점 및 기여

본 연구는 3D Open-vocabulary task에서 **'포인트 중심'에서 '인스턴스 중심'으로** 패러다임을 전환함으로써, 단순 세만틱 맵을 넘어 실제 객체 단위의 상호작용이 가능한 표현력을 확보하였다. 특히 데이터가 부족한 Long-tail 클래스에서 제로샷 성능이 뛰어나다는 점은 실제 로봇 공학이나 AR 환경에서 매우 유용할 것으로 보인다.

### 한계 및 비판적 해석

1. **마스크 품질 의존성**: Oracle mask 실험 결과에서 나타나듯, 최종 성능은 3D class-agnostic mask의 품질에 크게 의존한다. 즉, 현재의 성능 병목은 특징 추출보다는 마스크 제안 단계에 있다.
2. **뷰 의존적 특성**: 특징이 이미지 기반이므로 카메라 시야(Frustum) 내에 들어온 정보만 활용 가능하다. 이는 장면 전체에 대한 글로벌한 공간 관계나 보이지 않는 부분에 대한 이해가 부족함을 의미한다.
3. **평가 체계의 부재**: Closed-vocabulary 데이터셋으로 Open-vocabulary 모델을 평가하는 현재의 방식은 모델의 잠재력을 완전히 측정하기 어렵다.

## 📌 TL;DR

OpenMask3D는 3D 장면에서 임의의 텍스트 쿼리에 대응하는 객체 인스턴스를 분할하는 **최초의 제로샷 Open-vocabulary 3D Instance Segmentation** 모델이다. 클래스 불가지론적 3D 마스크 제안 $\rightarrow$ 최적 뷰 선택 $\rightarrow$ SAM 기반 2D 마스크 정제 $\rightarrow$ CLIP 특징 집계로 이어지는 파이프라인을 통해, 특히 데이터가 부족한 희귀 객체(Long-tail) 인식에서 탁월한 성능을 보인다. 이 연구는 향후 로봇의 객체 조작이나 지능형 3D 검색 시스템의 핵심 기반 기술이 될 가능성이 높다.
