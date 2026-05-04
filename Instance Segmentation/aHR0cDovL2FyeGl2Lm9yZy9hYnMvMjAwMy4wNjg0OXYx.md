# Deep Affinity Net: Instance Segmentation via Affinity

Xingqian Xu, Mang Tik Chiu, Thomas S. Huang, and Honghui Shi (2020)

## 🧩 Problem to Solve

본 논문은 Instance Segmentation 문제를 해결하기 위해 기존의 두 가지 주류 패러다임인 Region-based 접근 방식과 Keypoint-based 접근 방식의 한계를 극복하고자 한다. Region-based 방식은 객체의 Bounding Box를 먼저 탐지한 후 세그멘테이션을 수행하므로 다단계 실행 과정이 필요하고 하이퍼파라미터 민감도가 높다는 단점이 있다. Keypoint-based 방식은 개별 인스턴스를 키포인트 세트로 표현하고 주변 픽셀을 클러스터링하는 방식을 취한다.

이에 대해 저자들은 픽셀 간의 관계성을 나타내는 Affinity를 기반으로 한 새로운 패러다임을 제안한다. 특히 기존의 Affinity-based 방식들이 직면했던 두 가지 주요 문제, 즉 고정된 그리드 간격(fixed grid spacing)을 넘어서는 Affinity 예측의 어려움과 NP-hard 문제인 Multicut 그래프 분할(Graph Partitioning) 알고리즘의 높은 계산 복잡도 문제를 해결하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Affinity-based Instance Segmentation을 실용적인 수준으로 끌어올리기 위한 네트워크 구조와 알고리즘의 최적화에 있다.

첫째, 임의의 그리드 간격을 가진 픽셀 간의 Affinity를 추론하고 활용할 수 있는 새로운 Grouping Module을 제안한다. 이를 통해 폐색(occlusion) 등으로 인해 분리된 인스턴스 조각들을 효과적으로 다시 그룹화할 수 있다.

둘째, 기존의 탐욕적 멀티컷 알고리즘인 GAEC(Greedy Additive Edge Contraction)를 확장한 Cascade-GAEC 알고리즘을 제안한다. 이 알고리즘은 저해상도 맵에서 대략적인 세그먼트를 먼저 찾고 고해상도로 갈수록 이를 정교화하는 계층적 구조를 가져, 고해상도 이미지에서도 연산 시간을 획기적으로 단축시킨다.

## 📎 Related Works

### Instance Segmentation의 기존 접근 방식
- **Region-based (Proposal-based):** Mask R-CNN, PANet, UPSNet 등이 대표적이며, 탐지 네트워크를 통해 Bounding Box를 먼저 생성한다. 효과적이지만 다단계 처리 과정으로 인해 효율성이 떨어진다.
- **Keypoint-based (Proposal-free):** PFN, Deeperlab 등이 있으며, 인스턴스의 중심점이나 모서리 등의 키포인트를 예측한 후 픽셀을 클러스터링한다.
- **Affinity-based:** Instance Cut, GMIS, SSAP 등이 있으며, 픽셀 간의 Affinity를 예측하고 그래프 분할 알고리즘을 적용한다. SSAP는 최근 Affinity Pyramid 구조를 통해 성능을 높였으나 연산 속도 면에서 한계가 있었다.

### Graph Partitioning 알고리즘
최적의 Multicut 솔루션을 찾는 것은 정수 계획법(Integer Programming) 문제로 정의되며 매우 복잡하다. 본 논문은 정확도를 일부 희생하는 대신 효율성을 극대화한 GAEC 알고리즘을 채택하고 이를 개선하여 사용한다. GAEC는 가장 높은 Affinity 값을 가진 두 파티션을 탐욕적으로 병합하며, 설정된 임계값보다 낮은 Affinity만 남을 때까지 반복한다.

## 🛠️ Methodology

### 전체 시스템 구조 (DaffNet)
DaffNet은 FPN(Feature Pyramid Network) 백본(ResNet-101)을 기반으로 하며, 각 피라미드 레벨에서 세 가지 모듈을 통해 결과물을 생성한다.
1. **Semantic Module:** 각 픽셀의 클래스를 예측하는 Semantic Map을 생성한다.
2. **Affinity Module:** 인접한 4개 방향 이웃 픽셀 간의 관계를 나타내는 Affinity Map을 생성한다.
3. **Grouping Module:** 임의의 거리로 떨어진 픽셀들을 묶기 위한 Grouping Embedding을 생성한다.

### Grouping Module 및 Loss 함수
Grouping Module은 $k$차원의 임베딩 맵을 생성한다. 이때 "Push-Pull" 원리를 사용하여 같은 인스턴스에 속한 픽셀 임베딩은 가깝게(Pull), 서로 다른 인스턴스의 임베딩은 멀게(Push) 학습시킨다.

두 임베딩 $x_a$와 $x_b$ 사이의 유사도는 다음과 같은 가우시안 함수 $\Phi_{a,b}$로 정의한다.
$$\Phi_{a,b} = \Phi(x_a, x_b) = \exp(-\alpha(x_a - x_b)^T(x_a - x_b)) \in (0, 1]$$

이를 기반으로 한 Grouping Loss는 다음과 같다.
$$\text{L}_{gd} = \frac{1}{\binom{N_{ins}}{2}} \sum_{S \neq S'} \text{BCE}(0, \Phi(x_S, x_{S'})) \quad (\text{Push Loss})$$
$$\text{L}_{gs} = \frac{1}{N_{ins}} \sum_{i=1}^{N_{ins}} \frac{1}{N_S} \sum_{i \in S} \text{BCE}(1, \Phi(x_S, x_i)) \quad (\text{Pull Loss})$$
여기서 $x_S$는 인스턴스 $S$에 속한 모든 픽셀 임베딩의 평균값이다.

### 전체 학습 목표 (Total Loss)
최종 손실 함수는 Semantic Loss($\text{L}_{sem}$), 경계 픽셀의 Affinity Loss($\text{L}_{ab}$), 비경계 픽셀의 Affinity Loss($\text{L}_{as}$), 그리고 Grouping Loss($\text{L}_{gd}, \text{L}_{gs}$)의 가중 합으로 구성된다.
$$\text{L}_{total} = \sum_{level} (\lambda_{sem}\text{L}_{sem} + \lambda_{ab}\text{L}_{ab} + \lambda_{as}\text{L}_{as} + \lambda_{gd}\text{L}_{gd} + \lambda_{gs}\text{L}_{gs})$$

### Cascade-GAEC 알고리즘
Cascade-GAEC는 저해상도에서 고해상도 방향으로 세그멘테이션을 정교화하는 프로세스를 가진다.
1. **Upsampling:** 저해상도 단계의 분할 결과를 고해상도로 확장하고, 경계 부분의 픽셀을 미지정(unlabeled) 상태로 되돌려 다음 단계에서 다시 계산하게 한다.
2. **PA-GAEC (Position-Aware GAEC):** 단순 Affinity뿐만 아니라 픽셀 간의 물리적 거리를 고려하여, 너무 멀리 떨어진 세그먼트들이 잘못 병합되는 것을 방지한다.
3. **GAS (Greedy Association):** 가장 하위 레벨(최고해상도)에서 우선순위 큐(Priority Queue) 연산을 생략하고 가장 가까운 이웃의 라벨을 빠르게 할당함으로써 연산 속도를 추가로 높인다.

## 📊 Results

### 실험 설정
- **데이터셋:** Cityscapes (도시 거리 씬 이미지)
- **지표:** Average Precision (AP), IoU 임계값 0.5~0.95 범위에서 계산
- **비교 대상:** Mask R-CNN(Region-based), SSAP(Affinity-based) 등

### 정량적 결과
DaffNet은 Cityscapes Validation 셋에서 **32.4% AP**, Test 셋에서 **27.5% AP**를 기록하였다. 이는 기존의 모든 Affinity-based 모델보다 높은 수치이며, 대표적인 Region-based 모델인 Mask R-CNN(Val 31.5% AP)보다도 우수한 성능을 보였다. 특히 버스, 트럭, 기차와 같은 대형 객체에 대해 매우 강점을 보였다.

### 효율성 분석
연산 속도 면에서 Cascade-GAEC는 매우 효율적이다. Cityscapes 이미지 한 장당 처리 시간이 **0.244초**로, 이전의 최신 Affinity-based 연구인 SSAP(약 1.74초 추정)보다 약 **7배 더 빠르다**. 실질적인 시간 복잡도는 이미지 크기에 대해 거의 선형(Linear) 관계를 보인다.

## 🧠 Insights & Discussion

### 강점 및 해석
본 연구는 Affinity-based 방식이 단순한 보조 수단이 아니라, 적절한 알고리즘(Cascade-GAEC)과 보완 모듈(Grouping Module)이 결합된다면 Region이나 Keypoint 없이도 매우 효율적으로 Instance Segmentation을 수행할 수 있음을 증명하였다. 특히 계층적 구조를 통해 대형 객체는 저해상도에서 빠르게 처리하고, 소형 객체는 고해상도에서 정밀하게 처리하는 전략이 유효했음을 알 수 있다.

### 한계 및 향후 과제
그럼에도 불구하고, 극단적으로 크거나 작은 인스턴스(Extreme-size instances)를 처리하는 문제는 여전히 어려운 과제로 남아 있다. 또한, 현재의 Affinity 예측은 주로 국소적인 관계에 치중되어 있으므로, 이미지 전체의 맥락을 더 지능적으로 이해할 수 있는 고도화된 Affinity 정의가 필요하다.

## 📌 TL;DR

본 논문은 임의의 거리 픽셀 간 관계를 학습하는 **Grouping Module**과 연산 속도를 획기적으로 개선한 **Cascade-GAEC** 알고리즘을 제안하여, 효율적인 Affinity-based Instance Segmentation 모델인 **DaffNet**을 구현하였다. 결과적으로 Mask R-CNN보다 높은 성능과 기존 Affinity-based 모델보다 7배 빠른 속도를 달성하였으며, 이는 향후 실시간성 기반의 인스턴스 분할 연구에 중요한 가능성을 제시한다.