# Towards Federated Long-Tailed Learning

Zihan Chen, Songshang Liu, Hualiang Wang, Howard H. Yang, Tony Q.S. Quek, Zuozhu Liu (2022)

## 🧩 Problem to Solve

본 논문은 데이터 프라이버시 보호를 위한 **Federated Learning (FL)** 환경에서 데이터의 클래스 불균형이 극심한 **Long-Tailed (LT)** 분포가 나타날 때 발생하는 학습의 어려움을 해결하고자 한다.

일반적인 머신러닝 태스크에서 데이터 프라이버시 문제와 클래스 불균형 문제는 매우 빈번하게 발생한다. 기존 연구들은 프라이버시 보호를 위한 FL 연구와 중앙 집중식 환경에서의 Long-Tailed Learning 연구를 개별적으로 진행해 왔다. 그러나 실제 응용 분야(예: 의료 데이터, 사용자 행동 데이터)에서는 이 두 가지 문제가 동시에 발생하는 경우가 많으며, 이를 동시에 해결할 수 있는 효과적인 방법론은 아직 충분히 개발되지 않았다.

따라서 본 논문의 목표는 FL 프레임워크 내에서 Long-Tailed 데이터 분포가 로컬 및 글로벌 수준에서 어떻게 나타나는지를 체계적으로 정의하고, 이에 따른 도전 과제를 분석하며, 향후 연구 방향을 제시하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Federated Long-Tailed (F-LT) 학습 문제를 세 가지 구체적인 시나리오로 정형화하고, 이에 대한 벤치마크 분석을 통해 통찰을 제공한 점에 있다.

중심적인 아이디어는 단순히 '불균형한 데이터'로 치부하는 것이 아니라, **로컬 데이터 분포($p^k$)**와 **글로벌 데이터 분포($p^G$)**의 관계에 따라 문제의 성격이 완전히 달라진다는 점을 규명한 것이다. 이를 통해 각 시나리오별로 서로 다른 학습 목표(단일 글로벌 모델 학습 vs 다수의 개인화된 로컬 모델 학습)가 필요함을 역설한다.

## 📎 Related Works

### 중앙 집중식 Long-Tailed Learning (Centralized LT Learning)

중앙 집중식 환경에서는 다음과 같은 방법들이 제안되었다.

- **Re-balancing**: ROS, RUS, Simple calibration, Dynamic curriculum learning 등을 통해 샘플 수를 조절한다.
- **Re-weighting**: Focal Loss, LDAM Loss와 같이 손실 함수에 가중치를 부여하여 tail 클래스의 영향력을 높인다.
- **Representation & Classifier Decoupling**: 표현 학습(Representation learning)과 분류기 학습(Classifier learning)을 분리하여 분류기의 결정 경계를 재조정하는 방식이 제안되었다.

### Federated Learning (FL)

기존 FL 연구들은 주로 데이터의 Heterogeneity(비동질성), 즉 non-IID 분포나 데이터 크기의 불균형 문제를 다루어 왔다. FedProx와 같은 최적화 알고리즘이나 FedPer와 같은 Personalization FL (PFL) 기법들이 제안되었으나, 극단적인 Long-Tailed 분포가 FL 시스템에 미치는 영향에 대한 심층적인 분석은 부족한 상태이다.

## 🛠️ Methodology

본 논문은 새로운 알고리즘을 제안하기보다 F-LT 문제를 정의하고 분석하는 프레임워크를 제시한다.

### 1. 데이터 분포의 정형화

$N$명의 클라이언트와 $M$개의 클래스가 있을 때, 클라이언트 $k$의 로컬 데이터셋 $D^k$에서 클래스 $i$의 샘플 수를 $n_k^{(i)}$, 전체 샘플 수를 $n_k$라고 정의한다.

- **로컬 데이터 분포 ($p^k$):**
  $$p^k = \left[ \frac{n_k^{(1)}}{n_k}, \dots, \frac{n_k^{(j)}}{n_k}, \dots, \frac{n_k^{(M)}}{n_k} \right]$$
- **글로벌 데이터 분포 ($p^G$):**
  $$p^G = \left[ \frac{\sum_{k=1}^N n_k^{(1)}}{|D|}, \dots, \frac{\sum_{k=1}^N n_k^{(M)}}{|D|} \right]$$
  여기서 $|D|$는 전체 시스템의 총 샘플 수이다.

### 2. Imbalance Factor (IF)

데이터의 Long-Tailed 정도를 측정하기 위해 Imbalance Factor를 사용한다.

- **로컬 불균형 지수 ($IF_L^{(k)}$):** $$\text{IF}_L^{(k)} = \frac{\max_j \{n_k^{(j)}\}}{\min_s \{n_k^{(s)}\}}$$
- **글로벌 불균형 지수 ($\text{IF}_G$):** $$\text{IF}_G = \frac{\max_j \{\sum_{i=1}^N n_i^{(j)}\}}{\min_s \{\sum_{i=1}^N n_i^{(s)}\}}$$

### 3. F-LT의 세 가지 유형 (Taxonomy)

논문은 로컬과 글로벌 분포의 관계에 따라 F-LT를 세 가지 타입으로 분류한다.

- **Type 1 (Homogeneous LT):** 로컬과 글로벌 분포가 모두 동일한 Long-Tailed 분포를 가진다. 모든 클라이언트가 동일한 head/tail 클래스를 공유하며, 목표는 성능 좋은 **단일 글로벌 모델**을 학습하는 것이다.
- **Type 2 (Heterogeneous Global LT):** 글로벌 분포는 Long-Tailed이지만, 로컬 분포는 클라이언트마다 다양하다(Long-tailed, imbalanced, balanced가 섞여 있음). 목표는 **다수의 좋은 로컬 모델**을 학습하는 것이다.
- **Type 3 (Heterogeneous Local LT):** 일부 또는 모든 로컬 클라이언트는 Long-Tailed 분포를 가지나, 이를 모두 합친 글로벌 분포는 균형 잡힌(Balanced) 상태이다. 즉, 클라이언트마다 head 클래스가 서로 다르다. 목표는 **다수의 좋은 로컬 모델**을 학습하는 것이다.

## 📊 Results

### 실험 설정

- **데이터셋**: CIFAR-10-LT (합성 데이터), iNaturalist, Google Landmarks (실제 데이터).
- **데이터 분할**: IID 샘플링(Type 1), Dirichlet 분포 기반 샘플링(Type 2), 다양한 head/tail 패턴 샘플링(Type 3)을 사용하였다.
- **비교 알고리즘**: FedAvg, FedProx, CReFF (LT 특화), FedPer (개인화 특화).
- **평가 지표**: Test Accuracy.

### 주요 결과

1. **불균형도와 성능의 관계**: $\text{IF}_G$ 또는 $\text{IF}_L$ 수치가 커질수록(즉, Long-Tailed 특성이 강해질수록) 모든 알고리즘의 테스트 정확도가 하락하는 경향을 보였다.
2. **알고리즘별 특성**:
   - **FedProx**: 일반적인 non-IID 설정에서는 FedAvg보다 우수하지만, 글로벌 Long-Tailed 데이터 설정에서는 성능이 저하되는 경우가 있었다.
   - **CReFF**: Long-Tailed 특화 방법론답게 많은 설정에서 가장 높은 성능을 보였으나, 데이터의 Heterogeneity가 매우 심해지면 FedProx보다 성능이 낮아지기도 하였다.
   - **FedPer (PFL)**: 특히 Type 2와 Type 3처럼 로컬 분포가 다양한 경우, 별도의 LT 학습 기법 없이도 개인화(Personalization) 접근 방식이 매우 효과적임을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 논문은 단순히 성능을 높이는 알고리즘을 제시하는 대신, FL 환경에서 Long-Tailed 문제가 가질 수 있는 수학적 구조와 시나리오를 명확히 정의하였다. 특히, 글로벌 분포가 균형 잡혔더라도 로컬 분포가 Long-Tailed일 수 있다는 Type 3 시나리오를 제시함으로써, 기존의 중앙 집중식 LT 학습 관점만으로는 FL 문제를 해결할 수 없음을 시사하였다.

### 한계 및 논의사항

- **데이터 손실 문제**: CIFAR-10-LT와 같은 합성 데이터셋을 만들 때 Exponential/Pareto 샘플링을 사용하면 많은 양의 데이터를 버리게 되는데, 이는 성능 저하가 단순히 불균형 때문이 아니라 전체 데이터 양의 감소 때문일 가능성이 있음을 언급하였다.
- **개인화의 중요성**: 실험 결과, PFL(Personalized FL) 기법이 LT 문제 해결의 실마리가 될 수 있음을 보여주었으나, 구체적으로 CLT의 Decoupling 기법이나 Re-weighting 기법을 PFL 구조에 어떻게 통합할지에 대한 상세 설계는 향후 과제로 남겨두었다.

## 📌 TL;DR

본 논문은 프라이버시 보호를 위한 **Federated Learning(FL)**과 극심한 클래스 불균형인 **Long-Tailed(LT)** 분포가 결합된 새로운 문제 정의(**Federated Long-Tailed Learning**)를 제시한다.

핵심 기여는 로컬/글로벌 분포의 관계에 따라 F-LT 문제를 세 가지 타입으로 분류하고, 벤치마크를 통해 **단순한 글로벌 모델 학습보다는 개인화된 모델 학습(PFL)이 LT 문제 해결에 더 유망함**을 입증한 것이다. 이 연구는 향후 PFL과 CLT(Centralized LT) 기법의 통합, 계층적 FL 구조 설계 등의 연구 방향을 제시하여 실제 환경의 불균형한 프라이빗 데이터 학습을 위한 기초 틀을 마련하였다.
