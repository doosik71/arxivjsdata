# Multi-Task Learning on Networks

Andrea Ponti (2021)

## 🧩 Problem to Solve

본 연구는 Multi-Task Learning (MTL) 환경에서 발생하는 여러 상충하는 목적 함수(conflicting objectives)를 동시에 최적화하는 문제를 다룬다. 일반적으로 MTL은 여러 작업의 가중 합(weighted sum)을 최소화하는 방식으로 접근하지만, 이는 각 작업 간의 트레이드-오프(trade-off)를 정밀하게 모델링하지 못하는 한계가 있다.

특히, 실제 응용 분야(예: 수도망 센서 배치, 추천 시스템)에서 목적 함수들은 서로 경쟁 관계에 있는 경우가 많으며, 이를 해결하기 위해서는 단순한 선형 결합이 아닌 파레토 분석(Pareto analysis)을 통한 파레토 최적해(Pareto optimal solutions)의 집합을 찾는 Multi-Objective Optimization (MOO) 접근 방식이 필요하다. 하지만 기존의 Multi-Objective Evolutionary Algorithms (MOEAs)는 함수 평가 횟수가 많아 샘플 효율성(sample efficiency)이 낮고, Bayesian Optimization (BO)은 각 작업을 독립적인 목적 함수로 처리하여 작업 간의 내재적 구조를 활용하지 못한다는 문제가 있다. 따라서 본 논문의 목표는 검색 공간(search space)을 확률 분포로 매핑하여 탐색 효율과 파레토 집합의 품질을 높이는 새로운 알고리즘인 MOEA/WST를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 검색 공간(Input Space)의 해를 직접 다루는 대신, 이를 확률 분포(특히 히스토그램)로 표현된 중간 단계인 '정보 공간(Information Space)'으로 매핑하는 것이다. 이 정보 공간에서 Wasserstein (WST) 거리를 사용하여 개체 간의 유사성과 거리를 측정함으로써, 단순한 유클리드 거리나 해밍 거리가 줄 수 있는 오해(misleading)를 방지하고 더 효과적인 탐색을 수행한다.

주요 기여 사항은 다음과 같다:

1. **정보 공간 매핑**: 검색 공간의 해를 확률 분포(히스토그램)로 변환하는 매핑 체계를 정의하여, 확률적 거리 측정 도구를 최적화 과정에 도입하였다.
2. **MOEA/WST 알고리즘 제안**: 부모 개체를 선택할 때 Wasserstein 거리를 기반으로 가장 서로 다른(diverse) 쌍을 선택함으로써 탐색(exploration)과 다양성(diversification)을 극대화하는 선택 연산자를 도입하였다.
3. **실제 도메인 적용 및 검증**: 수도망(Water Distribution Networks)의 센서 배치 문제와 추천 시스템(Recommender Systems)의 다중 목적 최적화 문제에 제안 방법론을 적용하여 그 유효성을 입증하였다.

## 📎 Related Works

논문은 다음과 같은 기존 접근 방식과 그 한계를 설명한다:

- **Gradient-based Methods**: 효과적이지만 손실 함수가 블랙박스 형태이거나 다봉성(multimodal)을 띨 경우 그래디언트 계산이 어려워 적용이 제한적이다.
- **MOEAs (e.g., NSGA-II, MOEA/D)**: 미분 정보가 필요 없고 구현이 간단하며 파레토 분석에 적합하다. 그러나 목적 함수 평가 비용이 높은 문제에서 샘플 효율성이 매우 낮다는 치명적인 단점이 있다.
- **Bayesian Optimization (BO) & ParEGO**: 가우시안 프로세스(GP)와 같은 대리 모델(surrogate model)을 사용하여 샘플 효율성을 높인다. 하지만 대부분의 MO-BO 접근법은 각 목적 함수마다 별도의 GP를 유지하므로, 작업 간의 상관관계나 구조적 특징을 충분히 활용하지 못한다.

본 연구는 이러한 한계를 극복하기 위해 대리 모델을 목적 함수 자체에 적용하는 대신, 입력 공간과 목적 공간 사이의 '정보 공간'에 적용하여 유전 연산자의 효율을 높이는 차별화된 전략을 취한다.

## 🛠️ Methodology

### 전체 파이프라인

MOEA/WST의 전체 구조는 다음과 같은 흐름으로 진행된다:

1. **초기화**: 검색 공간에서 무작위로 초기 집단을 샘플링한다.
2. **목적 함수 평가**: 각 개체에 대해 다중 목적 함수를 평가하여 파레토 집합(Pareto set)을 구성한다.
3. **정보 공간 매핑**: 각 해 $s$를 히스토그램 형태의 확률 분포 $h(s)$로 변환한다.
4. **WST 기반 선택**: Wasserstein 거리를 사용하여 다양성이 높은 부모 쌍을 선택한다.
5. **유전 연산 및 생존**: 문제 특화 교차(crossover) 및 변이(mutation)를 수행하고, NSGA-II의 비지배 정렬(non-dominated sorting)과 혼잡 거리(crowding distance)를 통해 다음 세대의 생존자를 결정한다.

### 핵심 구성 요소 및 방정식

#### 1. Wasserstein Distance (WST)

두 확률 분포 $f$와 $g$ 사이의 거리를 측정하기 위해 $p$-Wasserstein 거리를 사용한다. 이는 한 분포를 다른 분포로 변형하는 데 드는 최소 비용(Earth Mover's Distance)으로 해석된다.
$$W_p(f,g) = \left( \inf_{\gamma \in \Gamma(f,g)} \int_{X \times X} d(x,y)^p d\gamma(x,y) \right)^{1/p}$$
여기서 $\Gamma(f,g)$는 주변 분포가 $f$와 $g$인 모든 결합 분포의 집합이며, $d(x,y)$는 지면 거리(ground distance)이다.

#### 2. WST 기반 선택 연산자

부모 개체 쌍 $(F_1, M_1)$과 $(F_2, M_2)$가 샘플링되었을 때, 두 개체의 히스토그램 표현 $h(F)$와 $h(M)$ 사이의 Wasserstein 거리가 더 큰 쌍을 선택한다:
$$i = \operatorname{argmax}_{i \in \{1,2\}} d_{WST}(h(F_i), h(M_i))$$
이 방식은 서로 매우 다른 특성을 가진 개체들을 교배하게 함으로써 탐색 범위를 넓히고 조기 수렴을 방지한다.

#### 3. 문제 특화 교차 연산자 (Problem-specific Crossover)

센서 배치 문제와 같이 제약 조건(예: 센서 개수 $p$개 이하)이 있는 경우, 단순한 1-point crossover 대신 부모의 활성화된 인덱스 집합에서 샘플링하여 자식 또한 자동으로 제약 조건을 만족하도록 설계된 'feasible-by-design' 교차 연산자를 사용한다.

## 📊 Results

### 실험 설정

- **데이터셋 및 작업**:
  - **수도망(WDN)**: Hanoi, Neptun 등 벤치마크 및 실제 네트워크에서 센서 최적 배치 수행. 목적 함수는 평균 탐지 시간($f_1$)과 탐지 시간의 표준편차($f_2$)이다.
  - **추천 시스템(RS)**: MovieLens 데이터셋을 사용하여 정확도(Accuracy), 커버리지(Coverage), 참신성(Novelty)의 3가지 목적 함수를 최적화.
- **비교 대상**: NSGA-II, qParEGO.
- **측정 지표**: Hypervolume (HV), Coverage (C-metric).

### 주요 결과

- **수도망 문제**:
  - **Hypervolume**: MOEA/WST는 NSGA-II보다 전반적으로 높은 HV를 기록하였으며, 특히 네트워크 규모가 큰 Neptun WDN에서 NSGA-II 대비 압도적인 성능 향상을 보였다.
  - **샘플 효율성**: qParEGO는 매우 적은 함수 평가 횟수로 빠르게 HV를 높였으나, GP 모델 업데이트 및 행렬 연산으로 인해 실제 계산 시간(wall-clock time)은 MOEA/WST보다 훨씬 오래 걸렸다.
- **추천 시스템 문제**:
  - MOEA/WST는 NSGA-II보다 파레토 프런트로의 수렴 속도가 빨랐으며, 특히 커버리지(Coverage) 지표에서 더 우수한 성능을 보였다.
  - Wasserstein 그래프 기반의 클러스터링을 통해 사용자를 그룹화했을 때 알고리즘의 성능이 더욱 안정적으로 나타났다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 연구는 입력 공간의 기하학적 구조가 목적 공간의 구조와 일치하지 않는 'Misleading Landscape' 문제를 정보 공간으로의 매핑을 통해 해결하였다. 특히 Wasserstein 거리는 분포 간의 겹침(overlap)이 없는 경우에도 의미 있는 거리 값을 제공하므로, KL-divergence나 JS-divergence보다 최적화 과정에서 훨씬 안정적인 가이드 역할을 수행한다.

### 한계 및 비판적 해석

- **계산 복잡도**: WST 계산은 기본적으로 선형 계획법(Linear Programming)을 필요로 하므로, NSGA-II와 같은 단순 알고리즘보다 연산 시간이 증가한다. 다만, 본 논문에서는 이를 통해 얻는 샘플 효율성과 해의 품질 향상이 계산 시간의 증가분보다 더 가치 있음을 보여주었다.
- **가정**: 본 연구는 모든 이벤트(오염원 유입 등)가 동일한 확률로 발생한다는 가정을 사용하였다. 실제 환경에서는 특정 지점의 발생 확률이 더 높을 수 있으며, 이를 가중치로 반영한 확장 연구가 필요할 것으로 보인다.

## 📌 TL;DR

본 논문은 다중 목적 최적화 문제에서 탐색 효율을 높이기 위해 **입력 공간을 확률 분포(히스토그램) 형태의 정보 공간으로 매핑**하고, **Wasserstein 거리**를 이용해 다양성이 높은 부모를 선택하는 **MOEA/WST** 알고리즘을 제안하였다. 실험 결과, 수도망 센서 배치 및 추천 시스템 문제에서 기존 NSGA-II보다 우수한 파레토 해를 더 빠르게 찾아냈으며, 이는 복잡한 네트워크 구조를 가진 실제 시스템의 최적 설계 및 개인화 추천 서비스의 품질 향상에 기여할 가능성이 크다.
