# SSFL: Discovering Sparse Unified Subnetworks at Initialization for Efficient Federated Learning

Riyasat Ohib, Bishal Thapaliya, Gintare Karolina Dziugaite, Jingyu Liu, Vince D. Calhoun, Sergey Plis (2025)

## 🧩 Problem to Solve

본 논문은 연합 학습(Federated Learning, FL) 환경에서 발생하는 심각한 통신 병목 현상을 해결하고자 한다. 특히, 수많은 클라이언트 기기가 참여하는 cross-device 시나리오에서는 모델의 크기가 커짐에 따라 통신 비용이 기하급수적으로 증가하며, 이는 시스템의 실용성을 저해하는 핵심 요인이 된다.

기존의 희소 학습(Sparse Training) 기반 FL 방법론들은 다음과 같은 한계를 지닌다. 첫째, 동적 희소성(Dynamic Sparsity) 전략은 훈련 과정에서 마스크(mask)를 지속적으로 업데이트해야 하므로 반복적인 조정 과정과 복잡한 하이퍼파라미터 튜닝이 필요하며, 클라이언트마다 서로 다른 부분 공간(subspace)에서 업데이트가 이루어져 글로벌 표현 학습의 일관성을 해칠 수 있다. 둘째, 초기화 단계에서의 가지치기(Pruning-at-Initialization, PaI) 방식은 대개 공개된 프록시 데이터셋(public proxy datasets)을 사용하여 중요 파라미터를 식별하는데, 이는 FL의 핵심 가치인 데이터 프라이버시 보호 원칙에 위배된다.

따라서 본 연구의 목표는 보조 데이터셋 없이, 프라이버시를 유지하면서, 단 한 번의 통신만으로 비-IID(non-IID) 데이터 환경에서도 효율적으로 작동하는 성능 좋은 정적 희소 서브 네트워크(static sparse subnetwork)를 초기화 단계에서 찾아내는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **단 한 번의 분포된 saliency 점수 집계(distributed saliency aggregation)**를 통해 모든 클라이언트가 공유하는 단일 글로벌 희소 마스크를 생성하는 것이다.

주요 기여 사항은 다음과 같다.
1. **SSFL(Salient Sparse Federated Learning) 프레임워크 제안**: 훈련 시작 전, 로컬 클라이언트의 데이터만을 활용해 글로벌 공유 마스크를 생성하는 single-shot 방식의 희소 FL 프레임워크를 도입하였다.
2. **통신 효율적 마스크 생성 메커니즘**: 각 클라이언트가 계산한 로컬 saliency 점수를 데이터 양에 따라 가중 평균하여 집계함으로써, 데이터 불균형(heterogeneity)을 고려한 대표성 있는 글로벌 중요도 벡터를 생성한다.
3. **정적 서브 공간의 안정성 확보**: 훈련 내내 고정된 마스크를 사용함으로써, 모든 클라이언트가 동일한 파라미터 부분 공간에서 최적화를 진행하게 하여 동적 마스크 방식에서 나타나는 불안정성을 제거하였다.
4. **실제 배포 효율성 검증**: AWS 기반의 실제 지리적 분산 환경에서 실험을 진행하여, dense 모델 대비 통신 시간을 최대 2.3배 이상 단축시킴을 입증하였다.

## 📎 Related Works

본 논문은 모델 가지치기(Model Pruning)와 통신 효율적 FL의 접점에서 관련 연구를 분석한다.

1. **동적 희소성(Dynamic Sparsity) in FL**: SparsyFed, DisPFL, FedDST 등은 훈련 중 마스크를 진화시키는 방식을 사용한다. 하지만 이들은 반복적인 서버-클라이언트 조정이 필요하며, 클라이언트 간 마스크가 불일치하여 서버 집계 시 유효한 글로벌 모델이 다시 밀집(dense)해지는 경향이 있다.
2. **구조적 희소성(Structured Sparsity)**: HeteroFL, AdaptCL 등은 필터나 레이어 단위의 구조적 가지치기를 통해 계산 비용을 줄이려 한다. 그러나 본 논문은 통신 병목 해결을 위해 파라미터 감소 효과가 더 큰 비구조적(unstructured) 희소성에 집중한다.
3. **초기화 단계의 가지치기(PaI)**: SNIP, GraSP 등이 중앙 집중식 학습에서 제안되었으나, FL에 적용된 FedTiny 등은 프라이버시 문제를 야기하는 공개 데이터셋에 의존해 왔다. SSFL은 이러한 프록시 데이터 없이 로컬 데이터만으로 이 문제를 해결하려 한다.

## 🛠️ Methodology

SSFL은 크게 두 단계로 구성된다: (i) 분포된 그래디언트 saliency 집계를 통한 일회성 마스크 발견 단계, (ii) 발견된 마스크에 제한된 희소 훈련 단계이다.

### 1. 파라미터 Saliency 정의
초기화 단계에서 파라미터 $w_j$의 중요도를 판단하기 위해, 해당 파라미터를 제거했을 때 손실 함수 $L$의 변화량을 1차 테일러 전개(first-order Taylor expansion)로 근사한 saliency 점수 $s_j$를 다음과 같이 정의한다.

$$s_j = \left| \frac{\partial L(w_0; D)}{\partial w_j} \cdot w_j \right|$$

여기서 $w_0$는 랜덤하게 초기화된 모델의 가중치이며, $D$는 데이터셋이다. 이 점수가 클수록 해당 파라미터가 초기 손실에 미치는 영향이 크므로 유지될 가능성이 높다.

### 2. SSFL 프로세스

**단계 1: 로컬 Saliency 추정 (Local Saliency Estimation)**
모든 클라이언트는 공통 모델 $w_0$를 공유한다. 각 클라이언트 $k$는 자신의 프라이빗 데이터에서 단 하나의 미니배치 $B_k$를 샘플링하여 로컬 saliency 벡터 $s^k$를 계산한다.

**단계 2: Saliency 점수 집계 (Aggregation)**
각 클라이언트는 계산된 $s^k$와 로컬 데이터 크기 $n_k$를 서버로 전송한다. 서버는 이를 데이터 비중에 따라 가중 평균하여 글로벌 saliency 벡터 $s$를 생성한다.

$$s = \sum_{k=1}^K p_k s^k, \quad \text{where } p_k = \frac{n_k}{\sum_{i=1}^K n_i}$$

**단계 3: 글로벌 마스크 선택 (Global Mask Selection)**
목표 희소도 $\sigma \in (0,1)$가 주어졌을 때, 활성화할 파라미터 수 $k = \lfloor (1-\sigma) \cdot d \rfloor$를 결정한다. 이후 $s$에서 가장 큰 값을 가진 상위 $k$개 인덱스를 선택하여 이진 마스크 $m \in \{0, 1\}^d$를 생성하고 이를 모든 클라이언트에 배포한다.

**단계 4: 희소 연합 학습 (Sparse Federated Training)**
이후의 모든 훈련 과정은 마스크 $m$에 의해 선택된 서브 네트워크 내에서만 이루어진다. 순전파와 역전파 모두 마스크가 적용된 가중치만을 사용하며, 그래디언트 업데이트 식은 다음과 같다.

$$w_{t+1}^{k,m} \leftarrow w_{t}^{k,m} - \eta \left( \nabla_w L(w_t^{k,m} \odot m; B_{k,t}) \odot m \right)$$

### 3. OOD 적응 (Out-of-Distribution Adaptation)
데이터 분포의 급격한 변화나 새로운 클래스를 가진 클라이언트가 추가될 경우, 본 논문은 **One-Shot OOD Adaptation**을 제안한다. 분포 변화가 감지되면 기존 마스크를 일시적으로 해제하고, 현재 모델 상태에서 다시 한번 saliency 집계를 수행하여 마스크를 갱신한다.

## 📊 Results

### 1. 정량적 성능 비교
ResNet-18 구조에서 50% 희소도를 적용하여 CIFAR-10, CIFAR-100, Tiny-ImageNet 데이터셋으로 실험을 진행하였다.
- **Dirichlet Partition (non-IID)**: CIFAR-10에서 SSFL은 88.29%의 정확도를 기록하며, 최강의 희소 베이스라인인 DisPFL뿐만 아니라 dense 베이스라인인 FedAvg-FT(88.02%)보다 높은 성능을 보였다. CIFAR-100에서는 DisPFL 대비 상대 오차(relative error)를 20% 이상 줄였다.
- **Pathological Partition**: 매우 극단적인 non-IID 설정에서도 SSFL은 CIFAR-10에서 94.61%의 정확도를 달성하며 강건함을 입증하였다.
- **Tiny-ImageNet**: 더 복잡한 데이터셋에서도 19.4%의 정확도로 다른 베이스라인들을 능가하였다.

### 2. 모델 규모 및 희소도 확장성
ResNet-50 모델을 사용하여 희소도를 50%에서 95%까지 확장했을 때, SSFL과 DisPFL의 격차는 더욱 벌어졌다.
- **95% 희소도**: DisPFL의 정확도가 12.04%로 급락한 반면, SSFL은 47.53%를 유지하였다. 이는 모델 규모가 커지고 희소도가 높아질수록 안정적인 글로벌 공유 서브 공간을 확보하는 것이 결정적임을 시사한다.

### 3. 실세계 배포 효율성
AWS 5개 지역에 분산 배치된 환경에서 통신 시간을 측정한 결과, 90% 희소도에서 dense 모델 대비 평균 집계 지연 시간을 1.91초에서 0.82초로 단축하여 약 **2.3배의 속도 향상**을 달성하였다.

### 4. 마스크 구조 분석
- **Global vs Local Mask**: 동일한 랜덤 마스크라도 클라이언트별로 다른 마스크를 사용하는 것보다, 하나의 글로벌 마스크를 공유하는 것이 성능이 훨씬 좋았다. 이는 서브 공간의 정렬(structural alignment)이 유효한 집계를 위해 필수적임을 의미한다.
- **마스크 수렴도**: Oracle 마스크(전체 데이터를 사용해 만든 마스크)와 비교했을 때, 약 80~100개의 미니배치(클라이언트 수 $K \approx 100$)만 집계해도 마스크 에러가 빠르게 감소하며 수렴함을 확인하였다.

## 🧠 Insights & Discussion

**강점 및 설계의 타당성**
본 논문은 복잡한 동적 마스크 업데이트 없이 초기화 단계에서의 단 한 번의 집계만으로도 고성능의 희소 네트워크를 찾을 수 있음을 보였다. 특히, 정적 마스크 방식은 컴파일 타임 최적화가 가능하고 런타임 오버헤드가 없기 때문에, NVIDIA Ampere 아키텍처의 Sparse Tensor Cores와 같은 최신 하드웨어 가속기(Hardware Accelerators)를 활용하기에 매우 유리한 구조이다.

**비판적 해석 및 한계**
1. **정적 마스크의 경직성**: OOD Adaptation 메커니즘을 제안했으나, 이는 트리거 기반의 일회성 업데이트이다. 개념 드리프트(concept drift)가 매우 빈번하고 연속적으로 발생하는 환경에서는 정적 마스크의 안정성과 가소성(plasticity) 사이의 트레이드오프를 해결하기 위한 더 세밀한 전략이 필요할 수 있다.
2. **Saliency 기준의 단순성**: 현재는 1차 그래디언트 기반의 saliency를 사용하고 있다. 더 깊은 모델이나 더 거대한 파운데이션 모델(Foundation Models)로 확장할 경우, 단순한 saliency 기준 외에 앙상블 기법이나 다른 중요도 지표를 도입하는 것이 성능 향상으로 이어질 가능성이 있다.

## 📌 TL;DR

본 논문은 연합 학습의 통신 병목을 해결하기 위해, 훈련 전 단 한 번의 통신으로 글로벌 공유 희소 마스크를 찾는 **SSFL**을 제안한다. 각 클라이언트의 로컬 saliency 점수를 데이터 가중 평균하여 정적 서브 네트워크를 구축함으로써, 프라이버시를 보호함과 동시에 통신 비용을 획기적으로 줄였다. 실험 결과, SSFL은 non-IID 환경에서 기존 동적 희소 FL 방법론보다 높은 정확도와 안정성을 보였으며, 특히 모델 규모가 커질수록 그 이점이 극대화된다. 또한 정적 구조 덕분에 하드웨어 가속기 적용이 용이하며 실물 네트워크 환경에서 통신 속도를 2.3배 이상 향상시켰다.