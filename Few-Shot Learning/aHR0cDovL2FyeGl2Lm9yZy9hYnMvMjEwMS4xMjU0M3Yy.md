# Few-Shot Learning for Road Object Detection

Anay Majee, Kshitij Agrawal, Anbumani Subramanian (2021)

## 🧩 Problem to Solve

본 논문은 실제 도로 환경의 이미지 데이터셋에서 발생하는 클래스 불균형(class-imbalance) 상황을 해결하기 위한 Few-Shot Object Detection (FSOD) 문제를 다룬다. 일반적인 딥러닝 기반의 객체 탐지 모델은 높은 정확도를 달성하기 위해 수백만 개의 레이블링된 샘플을 필요로 하지만, 이는 막대한 데이터 수집 및 레이블링 비용을 초래한다.

특히 자율 주행과 같은 도로 환경에서는 객체의 종류가 매우 다양하며, 일부 객체는 매우 드물게 나타나는 롱테일 분포(long-tail distribution) 특성을 가진다. 따라서 소수의 샘플만으로도 새로운 객체 클래스를 학습할 수 있는 Few-Shot Learning의 적용이 필수적이다. 본 연구의 목표는 실세계의 도로 이미지 데이터셋을 활용하여 Metric-learning 기반 방법과 Meta-learning 기반 방법의 성능을 비교 분석하고, 실제 배포 환경과 유사한 Open-set 설정에서의 효용성을 검증하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같다.

1. **실세계 도로 환경으로의 FSOD 확장**: 정제된 벤치마크 데이터셋(PASCAL-VOC, MS-COCO)이 아닌, 실제 주행 환경의 복잡성과 클래스 불균형이 반영된 India Driving Dataset (IDD)에 Few-Shot Learning을 적용하였다.
2. **데이터셋 확장 및 오픈셋 설정 구축**: IDD의 `vehicle-fallback` 클래스를 세분화하여 street-cart, tractor, water-tanker, excavator, crane 등 5가지의 새로운 희귀 객체 클래스를 정의함으로써, 실제 환경과 유사한 Open-set 학습 환경을 구축하였다.
3. **Metric-learning vs Meta-learning 비교 분석**: 두 가지 서로 다른 FSOD 접근 방식인 Feature Similarity 기반 방법(Metric-learning)과 Auxiliary Network 기반 방법(Meta-learning)의 성능을 정량적으로 비교하여, 도로 객체 탐지 작업에서 어떤 방식이 더 효율적인지 분석하였다.

## 📎 Related Works

### Few-Shot Learning (FSL)

FSL은 매우 적은 수(예: 10~50개)의 샘플만을 사용하여 새로운 클래스에 적응하는 기술이다. 주로 $N$-way, $K$-shot 방식으로 정의되며, episodic training 전략을 통해 일반화된 특징 추출기(feature extractor)를 학습한다. 그러나 새로운 클래스를 학습할 때 기존에 학습한 베이스 클래스를 잊어버리는 치명적 망각(catastrophic forgetting) 문제가 주요 한계로 지적된다.

### Few-Shot Object Detection (FSOD)

FSOD는 이미지 분류(Classification)의 FSL 기법을 객체 탐지 네트워크에 확장한 것이다. 기존 연구들은 전이 학습(transfer learning)이나 거리 기반 메트릭 학습(distance-metric learners)을 사용하였다. 최근에는 Cosine similarity와 같은 비선형 유사도 연산자를 도입하거나, 클래스별 어텐션 벡터(attention vectors)를 사용하여 베이스 클래스와 노벨 클래스를 구분하는 방법들이 제안되었다.

## 🛠️ Methodology

### 문제 정의

Few-shot learner $h(I, \theta)$는 베이스 클래스 $C_{base}$와 노벨 클래스 $C_{novel}$로부터 입력을 받는다. 학습 데이터는 다음과 같이 구성된다.

- $D_{base} = \{(x_{base}^i, y_{base}^i)\}_{i=1}^b$: 충분한 양의 샘플을 가진 베이스 클래스 데이터.
- $D_{novel} = \{(x_{novel}^i, y_{novel}^i)\}_{i=1}^n$: 클래스당 $K$개의 샘플만 가진 노벨 클래스 데이터.

학습의 목표는 $D_{base}$에서 일반화된 특징 표현을 학습하고, 이를 $D_{novel}$로 전이하여 베이스 클래스의 성능 저하를 최소화하면서 노벨 클래스의 탐지 성능을 높이는 것이다.

### 전체 파이프라인 및 아키텍처

본 논문은 2단계 훈련 메커니즘(Two-stage training mechanism)을 채택하며, 크게 두 가지 아키텍처를 비교한다.

#### 1. Feature Similarity 기반 (TFA/FsDet)

Two-stage Fine-tuning Architecture (TFA)는 Faster-RCNN의 최종 분류 레이어에 Cosine similarity 연산을 추가한 형태이다.

- **Base training stage**: 풍부한 베이스 데이터셋을 사용하여 네트워크를 학습시킨다.
- **Fine-tuning stage**: $D_{base} \cup D_{novel}$ 데이터를 사용하여 미세 조정을 수행한다. 이때 Cosine similarity를 통해 입력 특징과 각 카테고리의 학습 가능한 가중치 벡터 사이의 유사도를 계산함으로써 클래스 내 분산을 줄이고 치명적 망각을 방지한다.

#### 2. Auxiliary Network 기반 (Meta-learning)

표준 객체 탐지기 $D$와 별도의 보조 네트워크 $A$를 함께 사용한다. 보조 네트워크는 서포트 셋(support set)으로부터 클래스별 특징 벡터를 생성한다.

- **Meta-Reweight / Meta-RCNN**: 서포트 셋의 특징 $F_{sup}$와 쿼리 셋의 특징 $F_{qry}$ 사이의 채널별 곱셈(channel-wise multiplication)을 통해 클래스 특화 특징 $F_{cls}$를 생성하여 중요한 저수준 특징을 강조한다.
- **Add-Info**: $F_{qry}$와 $F_{sup}$의 차이($F_{qry} - F_{sup}$)를 기존 특징 셋에 연결(concatenate)하여 클래스 간의 유사성과 차이점을 인코딩한다.
- 이 방법들은 episodic training 전략을 사용하며, 학습 시 추가적인 손실 함수 $L_{meta}$를 도입한다.

### 실험 설정 (Datasets & Tasks)

- **IDD-10 (Same Domain)**: 15개 클래스 중 10개만 선택하여 7개의 베이스 클래스와 3개의 노벨 클래스로 나누어 실험한다.
- **IDD-OS (Open-Set)**: 10개의 베이스 클래스에 더해, `vehicle-fallback` 클래스에서 확장된 4개의 새로운 노벨 클래스(crane 제외)를 추가하여 실제 배포 환경을 시뮬레이션한다.

## 📊 Results

### 객체 탐지 베이스라인 (Roofline)

Few-shot 모델의 성능 상한선을 확인하기 위해 충분한 데이터를 사용하여 학습시킨 모델들을 비교한 결과, Faster-RCNN (ResNet-101 + FPN)이 $\text{mAP}$ 27.7, $\text{mAP}_{50}$ 45.4로 가장 높은 성능을 보여 이를 Roofline으로 설정하였다.

### Few-Shot 탐지 성능 ($\text{mAP}_{50}$ 기준)

실험 결과, Metric-learning 기반의 TFA(FsDet)가 Meta-learning 기반 방법들보다 우수한 성능을 보였다.

- **IDD-10 (Same Domain)**: TFA의 노벨 클래스 성능($\text{mAP}_{novel}$)은 22.1로, Meta-learning 기반 방법들보다 약 11.2 $\text{mAP}$ 포인트 높게 나타났다.
- **IDD-OS (Open-Set)**: TFA는 노벨 클래스에서 37.0 $\text{mAP}$를 기록하여 Meta-learning 방법들보다 약 1.0 $\text{mAP}$ 포인트 앞섰다.

### 주요 결과 분석

- **베이스 클래스 유지력**: TFA 아키텍처는 노벨 클래스를 추가한 후에도 베이스 클래스의 성능 저하가 Meta-learning 방법들에 비해 현저히 적었다.
- **클래스 간 혼동**: Confusion Matrix 분석 결과, Truck vs Car, Bicycle vs Motorcycle, Water-tanker vs Car 간의 혼동이 최대 40%까지 발생하였다. 이는 도로 객체들이 서로 유사한 저수준 특징을 공유하기 때문으로 분석된다.

## 🧠 Insights & Discussion

본 연구는 실세계 도로 이미지라는 특수한 환경에서 Metric-learning이 Meta-learning보다 더 효과적임을 입증하였다. Meta-learning 기반 방법의 성능이 상대적으로 낮게 나타난 이유는 새로운 객체 카테고리 간의 클래스 내 거리(inter-class distance)가 매우 가깝기 때문인 것으로 추정된다.

특히, 도로 위 객체들은 외형적 특징이 유사하여 Few-shot 알고리즘이 이를 정밀하게 구분하는 데 어려움을 겪는다. 이는 MetaDet 등 기존 연구에서 지적된 "클래스 간 혼동(class confusion)" 문제가 실제 도로 환경에서도 핵심적인 챌린지임을 시사한다. 또한, 단순한 Cosine similarity 기반의 TFA 구조가 복잡한 보조 네트워크 구조보다 실제 데이터셋에서 더 강건한(robust) 성능을 보였다는 점은 주목할 만하다.

## 📌 TL;DR

본 논문은 도로 환경의 클래스 불균형 문제를 해결하기 위해 IDD 데이터셋을 활용하여 Few-Shot Object Detection을 연구하였다. 실험 결과, Cosine similarity 기반의 Metric-learning 방법(FsDet)이 Meta-learning 방법보다 노벨 클래스 탐지 성능과 베이스 클래스 유지력 모든 면에서 우수함을 확인하였다. 이 연구는 실세계 주행 데이터셋을 확장하여 FSOD 연구를 위한 새로운 기반을 마련하였으며, 도로 객체 간의 높은 유사성으로 인한 클래스 혼동 문제가 여전한 과제임을 제시하였다.
