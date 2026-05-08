# Part2Object: Hierarchical Unsupervised 3D Instance Segmentation

Cheng Shi, Yulin Zhang, Bin Yang, Jiajin Tang, Yuexin Ma, and Sibei Yang (2024)

## 🧩 Problem to Solve

본 논문은 어떠한 인간의 주석(annotation) 없이 3D 포인트 클라우드에서 객체를 분할하는 **Unsupervised 3D Instance Segmentation** 문제를 해결하고자 한다.

기존의 비지도 학습 기반 3D 인스턴스 분할 방법들은 주로 전통적인 클러스터링이나 그래프 컷(graph-cut) 방식에 의존하는데, 이는 **분할 정밀도(segmentation granularity)** 설정에서 심각한 트레이드오프 문제를 야기한다. 즉, 클러스터링 기준을 느슨하게 잡으면 여러 객체가 하나로 묶이는 **under-segmentation**이 발생하고, 반대로 너무 엄격하게 잡으면 하나의 객체가 여러 조각으로 쪼개지는 **over-segmentation**이 발생한다.

또한, 3D 기하학적 특징만으로는 특정 포인트 집합이 의미 있는 '객체'인지 판단하기 어렵다는 점과, 2D 특징을 3D로 투영할 때 발생하는 일대다(many-to-one) 매핑의 취약성(노이즈 및 불일치 문제)이 주요 해결 과제로 제시된다. 본 연구의 목표는 이러한 정밀도 문제를 해결하고, 2D-3D 간의 정보를 효율적으로 융합하여 정교한 비지도 3D 인스턴스 분할을 달성하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **"Gather & Aim"** 전략으로 요약할 수 있다.

1. **Gather (계층적 클러스터링):** 단일 레벨의 클러스터링 대신, 포인트에서 시작해 객체 부분(part), 그리고 최종 객체(object)로 나아가는 **Hierarchical Clustering**을 제안한다. 이를 통해 서로 다른 크기와 형태를 가진 객체들이 각기 다른 계층 레벨에서 포착될 수 있도록 하여, over-segmentation과 under-segmentation의 트레이드오프를 완화한다.
2. **Aim (3D Objectness Priors):** 2D RGB 프레임의 시계열적 일관성(temporal consistency)을 활용하여 **3D Objectness Priors**를 추출하고, 이를 클러스터링 과정의 가이드로 사용한다. 특히 픽셀 단위의 투영이 아닌 객체 단위의 바운딩 박스를 활용함으로써 투영 과정의 취약성을 극복한다.
3. **Hi-Mask3D:** 계층적 구조를 학습할 수 있는 딥러닝 모델인 **Hi-Mask3D**를 제안한다. 이는 객체 부분(part)과 객체(object) 간의 명시적 상호작용을 통해 인스턴스 분할 성능을 높이며, Part2Object에서 생성된 의사 라벨(pseudo-labels)을 통해 학습된다.

## 📎 Related Works

- **Unsupervised 3D Instance Segmentation:** 초기 연구들은 좌표, 색상, 법선 벡터(normal vector) 등 원시 기하 정보 기반의 클러스터링을 사용했다. 최근에는 DINO와 같은 자기지도 학습(self-supervised) 모델의 특징을 활용해 의사 라벨을 생성하고 학습하는 방식이 등장했으나, 복잡한 실내 환경에서 일관된 분할 정밀도를 유지하는 데 한계가 있다.
- **Transfer 2D Foundation Models into 3D:** 2D의 강력한 파운데이션 모델(SAM, DINO 등)을 3D로 전이하려는 시도가 많다. 하지만 2D 마스크를 단순히 3D로 투영하는 방식은 객체가 겹쳐 있거나 cluttered한 실내 장면에서 성능이 저하되는 문제가 있다.
- **Supervised Point Cloud Segmentation:** 지도 학습 기반 모델들은 높은 성능을 보이지만 방대한 양의 주석 데이터가 필요하며, 주로 객체 수준의 특성에만 집중하여 객체를 구성하는 '부분(part)'에 대한 이해가 부족하다는 한계가 있다.

## 🛠️ Methodology

### 1. Part2Object: 계층적 클러스터링

Part2Object는 포인트 $\rightarrow$ super-point $\rightarrow$ object part $\rightarrow$ object 순으로 점진적으로 병합하는 구조를 가진다.

**포인트 특징 및 초기화:**

- 각 포인트 $p_i$는 위치, 법선 벡터, 색상 및 DINO를 통해 추출된 2D 시맨틱 특징 $f_i$를 포함한다.
- VCCS 알고리즘을 사용하여 초기 super-point 집합 $\{c_0^i\}$를 생성한다.
- 클러스터 특징 $f_0^i$는 단순 평균이 아닌, 유사도 기반의 가중 합으로 계산하여 노이즈의 영향을 줄인다.
$$f_0^i = \frac{\sum_{p_j \in c_0^i} \text{sim}(f_j, \bar{f}_0^i) f_j}{\sum_{p_j \in c_0^j} \text{sim}(f_j, \bar{f}_0^i)}, \quad \text{where } \bar{f}_0^i = \frac{1}{|c_0^i|} \sum_{p_j \in c_0^i} f_j$$

**계층적 병합 및 중단 기준:**

- 두 클러스터 $c_t^i, c_t^j$가 다음 조건들을 만족할 때 병합하여 $c_{t+1}^k$를 형성한다.
  1. 특징 유사도 $\text{sim}(f_t^i, f_t^j)$의 순위가 상위 $K$ 이내일 것.
  2. 두 클러스터 간의 최소 유클리드 거리 $\text{dist}(c_t^i, c_t^j) \le T$ 일 것.
- **Stopping Criteria:** 서로 다른 3D 객체 바운딩 박스 $b^{3D}_k$에 걸쳐 있는 클러스터들은 병합을 거부하여, 서로 다른 객체가 하나로 묶이는 것을 방지한다.

### 2. 3D Objectness Priors 추출

2D 프레임들로부터 3D 객체 후보군($B^{3D}$)을 추출하는 **Grouping-first-then-projection** 파이프라인을 사용한다.

1. **2D Co-segmentation:** 각 프레임에서 MaskCut을 통해 2D 마스크를 추출하고, 인접 프레임 간 마스크 특징의 코사인 유사도를 측정하여 동일 객체 여부를 판단한다.
2. **3D Projection:** 동일 객체로 묶인 2D 마스크들을 3D 공간으로 투영하여 병합하고, 최종적으로 3D 바운딩 박스를 생성하여 이를 클러스터링의 가이드로 활용한다.

### 3. Hi-Mask3D 아키텍처

Mask3D를 확장하여 객체 부분과 객체를 동시에 예측하는 계층적 인지 디텍터를 구현했다.

- **Hierarchical Attention:** 기존 Mask3D가 쿼리와 특징 맵 사이의 직접적인 cross-attention을 수행했다면, Hi-Mask3D는 다음과 같은 단계를 거친다.
  1. **Part Query 업데이트:** 특징 맵 $F_r$과 part query $Q_p$ 간의 attention 수행.
  2. **Object Query 업데이트:** 업데이트된 $Q_p^r$와 object query $Q_o$ 간의 attention 수행.
$$Q_p^r = \text{softmax}(Q_p^{r-1} {F_r}^T / \sqrt{C}) F_r$$
$$Q_o^r = \text{softmax}(Q_o^{r-1} {Q_p^r}^T / \sqrt{C}) Q_p^r$$
- **학습 목표:** Part2Object에서 생성된 $\hat{P}_{object}$와 $\hat{P}_{part}$를 의사 라벨로 사용하며, DICE loss와 Binary Cross Entropy(BCE) loss를 조합하여 학습한다. 또한, 예측 결과를 다시 라벨로 사용하는 iterative self-training을 적용한다.

## 📊 Results

### 실험 설정

- **데이터셋:** ScanNet, ScanNet200, S3DIS, Replica.
- **지표:** mAP@25, mAP@50, mAP (class-agnostic).

### 주요 결과

1. **Training-free 설정 (ScanNet):** 학습 없이 클러스터링만으로 평가했을 때, SOTA인 Unscene3D 대비 **mAP@50에서 16.8%p 상승**한 26.8%를 기록했다.
2. **Data-efficient 설정:** 매우 적은 양의 데이터(1%~20%)로 파인튜닝했을 때, 모든 구간에서 기존 방법론들을 압도했다. 특히 0% 데이터(의사 라벨만 사용)에서도 mAP@50 32.6%라는 높은 성능을 보였다.
3. **Cross-dataset Generalization:** ScanNet의 의사 라벨로 학습한 Hi-Mask3D가 ScanNet으로 완전 지도 학습을 한 Mask3D보다 ScanNet200, S3DIS, Replica 등 타 데이터셋에서 더 높은 제로샷(Zero-shot) 성능을 보였다. 이는 비지도 학습을 통해 더 일반화된(generalizable) 객체 표현을 학습했음을 시사한다.

### Ablation Study

- **Objectness Guidance (OG):** OG가 없을 때 성능이 크게 하락하여, 2D 기반의 객체 가이드가 클러스터링의 적절한 중단 지점을 찾는 데 필수적임을 확인했다.
- **Cluster Feature (FU):** 단순 평균보다 가중 합 방식의 특징 추출이 노이즈 제거에 효과적임을 입증했다.
- **Architecture:** Mask3D보다 Hi-Mask3D가 더 높은 성능을 보였으며, 이는 객체 부분(part) 정보가 전체 객체 이해에 도움을 준다는 것을 의미한다.

## 🧠 Insights & Discussion

**강점 및 의의:**
본 논문은 3D 인스턴스 분할의 고질적 문제인 '분할 정밀도' 문제를 계층적 구조라는 직관적인 방법으로 해결했다. 특히, 2D-3D 투영 시 발생하는 픽셀 단위의 취약성을 '객체 단위 그룹화 후 투영'이라는 전략으로 극복한 점이 돋보인다. 또한, 지도 학습 기반 모델보다 비지도 학습 기반 모델이 크로스 데이터셋 일반화 성능이 더 좋다는 결과는, 특정 클래스 라벨에 종속되지 않은 범용적인 객체 기하학적 특성을 학습했기 때문으로 해석된다.

**한계 및 논의:**

- **2D 모델 의존성:** DINO 및 MaskCut과 같은 2D 파운데이션 모델의 성능과 편향(bias)에 직접적으로 의존하고 있다. 2D 모델이 객체를 잘못 인식할 경우 3D 결과에도 영향을 미칠 수 있다.
- **계층적 깊이 설정:** 하이퍼파라미터 $K$(병합 비율)에 따른 성능 변화가 관찰되는데, 최적의 $K$를 장면의 복잡도에 따라 동적으로 결정하는 메커니즘에 대한 논의가 부족하다.
- **계산 복잡도:** 계층적 클러스터링과 반복적인 self-training 과정이 포함되어 있어, 실시간 처리 가능 여부에 대한 분석이 필요하다.

## 📌 TL;DR

본 논문은 비지도 3D 인스턴스 분할에서 발생하는 over/under-segmentation 문제를 해결하기 위해 **계층적 클러스터링(Part2Object)**과 **계층 인지 모델(Hi-Mask3D)**을 제안한다. 2D RGB 프레임에서 추출한 객체 바운딩 박스를 가이드로 사용하여 3D 포인트들을 '부분 $\rightarrow$ 객체' 순으로 정교하게 묶어내며, 이를 통해 학습 데이터가 거의 없는 환경이나 타 데이터셋 전이 학습에서 SOTA 성능을 달성했다. 이 연구는 3D 장면 이해에서 '부분-전체'의 계층적 관계가 매우 중요한 사전 지식임을 입증하며, 향후 데이터 효율적인 3D 인식 연구에 중요한 기여를 할 것으로 보인다.
