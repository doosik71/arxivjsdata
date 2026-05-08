# Active Class Selection for Few-Shot Class-Incremental Learning

Christopher McClurg, Ali Ayub, Harsh Tyagi, Sarah M. Rajtmajer, Alan R. Wagner (2023)

## 🧩 Problem to Solve

실제 환경에 배치된 로봇은 사용자 및 주변 환경과의 제한된 상호작용을 통해 지속적으로 새로운 객체를 학습해야 한다. 기존의 Few-Shot Class Incremental Learning (FSCIL) 연구들은 매우 적은 수의 샘플로 새로운 클래스를 학습하면서 이전 지식을 유지하는 성과를 거두었으나, 현실과는 동떨어진 제약적인 설정에서 테스트되었다. 구체적으로, FSCIL은 각 증분(increment) 단계에서 로봇이 이미 완전히 레이블링된 이미지 데이터셋을 제공받는다고 가정하며, 한 번 학습한 클래스에 대해 추가 데이터를 받지 않는다고 전제한다.

그러나 실제 환경에서 로봇은 수많은 레이블되지 않은 객체들과 마주하며, 그중 어떤 객체에 대해 학습해야 할지 스스로 결정해야 하는 상황에 놓인다. 따라서 본 논문은 로봇이 환경 내의 수많은 미지의 객체 중 가장 정보 가치가 높은(informative) 소수의 객체만을 선택적으로 사용자에게 레이블링 요청을 하여 효율적으로 학습할 수 있도록 하는 프레임워크를 개발하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Active Class Selection (ACS)**과 **Few-Shot Class Incremental Learning (FSCIL)**을 결합하여, 로봇이 스스로 학습할 클래스를 선택하고 이를 증분적으로 학습하는 **FIASco (Few-shot Incremental Active class SeleCtiOn)** 모델을 제안한 것이다.

FIASco의 중심적인 직관은 모델의 내부 상태인 **클러스터 통계(Cluster Statistics)**를 활용하여 어떤 클래스가 현재 모델에게 가장 부족하거나 불확실한지를 판단하고, 이를 기반으로 학습 우선순위를 정하는 것이다. 또한, 이를 자율 주행 에이전트에 통합하기 위해 ACS 결과가 내비게이션의 목적지를 결정하는 잠재 필드(Potential Field) 방식과 결합하여, '인식 $\rightarrow$ 선택 $\rightarrow$ 이동 $\rightarrow$ 학습'으로 이어지는 완전한 파이프라인을 구축하였다.

## 📎 Related Works

### 관련 연구 및 한계

1. **Class-Incremental Learning (CIL):** 새로운 클래스를 추가 학습할 때 이전 클래스의 정보를 잊어버리는 Catastrophic Forgetting 문제가 발생한다. 기존의 해결책인 데이터 저장 방식은 메모리 제한이 있는 로봇 시스템에 부적합하며, 정규화 손실 함수나 생성 모델을 사용하는 방식들이 제안되었다.
2. **Few-Shot Class Incremental Learning (FSCIL):** CIL의 설정을 확장하여 클래스당 아주 적은 수의 예시만으로 학습하는 것을 목표로 한다. TOPIC, FACT 등 최신 기법들이 제안되었으나, 여전히 데이터 제공 방식이 정적이고 고정적이라는 한계가 있다.
3. **Active Class Selection (ACS):** 모델이 학습 효율을 높이기 위해 특정 클래스의 데이터를 요청하는 기법이다. 하지만 대부분 배치 학습(batch learning) 설정으로 설계되어 이전 모든 데이터를 다시 사용해야 하며, 실제 로봇이 아닌 정적 데이터셋에서만 검증되었다.

### 차별점

FIASco는 기존 ACS의 배치 학습 제약을 극복하고 FSCIL의 증분 학습 능력을 결합하였다. 특히, 단순히 데이터 포인트(sample)를 선택하는 Active Learning과 달리, 학습해야 할 **클래스(class)** 자체를 선택하여 로봇의 이동 경로와 학습 전략에 직접적으로 반영했다는 점에서 기존 연구와 차별화된다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

FIASco의 전체 프로세스는 다음과 같은 흐름으로 구성된다.

- **특징 추출:** 사전 학습된 CNN(ResNet-34)을 고정된 Feature Extractor로 사용하여 이미지에서 특징 벡터를 추출한다.
- **증분 학습 (CBCL-PR):** 추출된 특징 벡터를 기반으로 클러스터링을 수행하여 클래스를 표현한다. 이때 CBCL-PR(Centroid Based Concept Learning with Pseudo-Rehearsal) 기법을 사용하여 새로운 클래스를 센터로 정의하고, 가상 예시(Pseudo-exemplars)를 생성하여 이전 지식을 보존하며 Linear SVM으로 분류기를 학습시킨다.
- **클래스 선택 (ACS):** 클러스터의 통계 정보를 통해 학습 우선순위를 결정한다.
- **내비게이션:** 선택된 우선순위에 따라 잠재 필드를 형성하여 로봇이 가장 정보 가치가 높은 객체로 이동하게 한다.

### 2. Active Class Selection (ACS) 메커니즘

모델은 클러스터 공간의 통계 정보를 사용하여 다음 학습 대상을 결정한다. 사용되는 주요 지표는 다음과 같다.

- **Class Weight:** 클래스당 포함된 전체 학습 예시의 수.
- **Cluster Weight:** 개별 클러스터 내에 포함된 학습 예시의 수.
- **Cluster Variance:** 클러스터 내 데이터의 분산. 이는 Welford의 방법을 통해 이전 데이터를 저장하지 않고 재귀적으로 계산된다.
  $$(n-1)s^2_n - (n-2)s^2_{n-1} = (x_n - \bar{x}_n)(x_n - \bar{x}_{n-1})$$

이 지표들을 바탕으로 다음과 같은 우선순위 전략을 세운다.

1. **Low Class Weight:** 샘플 수가 적은 클래스를 우선시하여 전체 정확도를 높인다.
2. **Low Cluster Weight:** 덜 발달된 클러스터(아웃라이어 등)를 가진 클래스를 우선시하여 클래스 공간을 확장한다.
3. **Low Cluster Variance:** 노이즈가 적은 클래스를 우선시하여 가치 있는 정보를 효율적으로 습득한다.
4. **High Cluster Variance:** 불확실성이 높은 클래스를 우선시하여 더 뚜렷한 클러스터를 형성하고자 한다.

### 3. 내비게이션 및 추론 절차

로봇은 주변 객체를 인식하고 ACS 우선순위에 따라 인력(attractive force) 또는 척력(repulsive force) $f_i$를 할당하여 잠재 필드를 생성한다. 로봇의 현재 위치 $(x_0, y_0)$에서의 전체 힘 $(F_x, F_y)$는 다음과 같이 계산된다.
$$(F_x, F_y) = \sum_{i=1}^{n} f_i (x_i - x_0, y_i - y_0)$$
다만, 실제 로봇 실험에서는 센서 노이즈와 드리프트 오류로 인해 잠재 필드 방식 대신, 한 번 주변을 관측한 후 **A* 경로 계획(A* path planning)** 알고리즘을 사용하여 선택된 목표 객체로 이동하는 방식을 채택하였다.

## 📊 Results

### 1. Minecraft 시뮬레이션 실험

- **설정:** CIFAR-100 및 Grocery Store 데이터셋 사용. ResNet-34 특징 추출기 사용.
- **비교 대상:** Uniform ACS(무작위 선택), Redistricting ACS(변동성이 큰 클래스 선택)를 사용하는 Batch Learner(SVM)와 비교.
- **결과:**
  - CIFAR-100: 'Low class weight' 전략을 사용한 FIASco가 가장 높은 성능(44.2%)을 보였으며, 이는 Uniform Batch learning 대비 3.7% 향상된 결과이다.
  - Grocery Store: 'Low class weight' 전략의 FIASco가 63.4%의 정확도를 기록하여, Uniform Batch learning 대비 5.3% 향상되었다.

### 2. Pepper 로봇 실물 실험

- **설정:** Grocery Store 데이터셋의 일부(41개 클래스)를 실제 객체로 구성한 실내 환경. YOLO를 이용한 객체 탐지와 A* 내비게이션 사용.
- **결과:** 'High cluster variance' 전략을 사용한 FIASco가 60.7%의 정확도를 기록하였다. 이는 Uniform Batch learning(60.3%)보다 소폭 향상된 결과이다.

## 🧠 Insights & Discussion

### 강점 및 성과

본 연구는 FSCIL과 ACS를 결합하여 로봇이 현실적인 제약 하에서 스스로 학습 데이터를 수집하고 모델을 업데이트하는 프레임워크를 성공적으로 제안하였다. 특히 클러스터 통계라는 내부 지표를 통해 외부의 명시적인 가이드 없이도 '학습 동기'를 스스로 생성하여 효율적인 학습이 가능함을 입증하였다.

### 한계 및 해석

실물 로봇 실험에서의 성능 향상 폭이 시뮬레이션보다 적게 나타났는데, 이는 내비게이션 방식의 차이에서 기인한다. 시뮬레이션에서는 매 순간 잠재 필드를 업데이트하며 반응적으로 이동했지만, 실물 로봇은 센서 노이즈 문제로 인해 정적인 A* 경로 계획을 사용했기 때문에 ACS의 동적인 이점을 완전히 활용하지 못했다. 또한, 실험 환경이 식료품점과 같이 매우 구조화된 환경이었다는 점이 결과에 영향을 주었을 가능성이 있다.

### 비판적 논의

논문에서는 $\text{SOTA}$ FSCIL 모델인 CBCL-PR을 기반으로 하였으나, 실제 로봇 환경에서 $\text{SOTA}$ 성능을 유지하는 것보다 더 중요한 것은 **'데이터 수집의 효율성'**이다. FIASco는 모델 자체의 아키텍처 개선보다는 학습 프로세스의 '루프(Loop)'를 최적화하는 데 집중하였으며, 이는 로봇 공학 관점에서 매우 실용적인 접근이다.

## 📌 TL;DR

본 논문은 로봇이 환경 내의 미지의 객체 중 가장 정보 가치가 높은 클래스를 스스로 선택해 학습하는 **FIASco** 프레임워크를 제안한다. 클러스터 통계(가중치, 분산)를 활용해 학습 우선순위를 정하고, 이를 내비게이션과 결합하여 효율적인 증분 학습을 수행한다. 시뮬레이션과 실물 로봇 실험을 통해 무작위 학습보다 높은 정확도를 달성하였으며, 이는 향후 로봇의 자율적인 지속 학습(Lifelong Learning) 구현에 중요한 기여를 할 것으로 보인다.
