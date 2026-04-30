# Mitigating Sampling Bias and Improving Robustness in Active Learning

Ranganath Krishnan, Alok Sinha, Nilesh Ahuja, Mahesh Subedar, Omesh Tickoo, Ravi Iyer (2021)

## 🧩 Problem to Solve

본 논문은 능동 학습(Active Learning, AL) 과정에서 발생하는 **샘플링 편향(Sampling Bias)** 문제를 해결하고, 결과적으로 모델의 **강건성(Robustness)**과 **교정(Calibration)** 성능을 향상시키는 것을 목표로 한다.

딥러닝 모델 학습을 위해서는 방대한 양의 레이블링된 데이터가 필요하지만, 의료 진단이나 시맨틱 세그멘테이션과 같이 전문 지식이 필요하거나 비용이 많이 드는 작업에서는 데이터 확보가 매우 어렵다. 이를 해결하기 위해 모델이 학습에 가장 유용한 샘플을 직접 선택하는 능동 학습이 사용되지만, 휴리스틱 기반의 샘플 선택 과정은 학습 데이터가 실제 데이터 분포를 충분히 대표하지 못하게 만드는 샘플링 편향을 야기한다. 이러한 편향은 모델이 실제 환경에 배포되었을 때 공정성, 강건성, 신뢰성 측면에서 부정적인 영향을 미치며, 특히 데이터 시프트(Dataset Shift)나 분포 외 데이터(Out-of-Distribution, OOD) 상황에서 성능 저하를 일으킨다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 신경망이 학습한 **특징 표현(Feature Representation)**을 활용하여, 정보량이 많으면서도 특징 공간 상에서 다양하게 분포된 샘플을 선택함으로써 샘플링 편향을 완화하는 것이다.

이를 위해 저자들은 다음 두 가지 핵심 방법론을 제안한다:
1. **Supervised Contrastive Active Learning (SCAL)**: 지도 학습 설정에서의 대조 학습(Contrastive Learning) 손실 함수를 도입하여 특징 공간 상에서 동일 클래스는 가깝게, 서로 다른 클래스는 멀게 배치함으로써 더 정교한 특징 표현을 학습하고, 이를 기반으로 편향되지 않은 샘플을 쿼리한다.
2. **Deep Feature Modeling (DFM)**: 클래스 조건부 PCA(Principal Component Analysis)를 활용하여 특징 재구성 오차(Feature Reconstruction Error)가 큰 샘플을 선택함으로써 데이터의 다양성을 확보한다.

## 📎 Related Works

능동 학습의 기존 접근 방식은 크게 불확실성 기반(Uncertainty-based), 다양성 기반(Diversity-based), 그리고 위원회 기반(Query-by-committee) 방식으로 나뉜다. 예를 들어, CoreSet은 k-center-greedy 방식을 통해 다양성을 확보하려 하지만, 탐색 과정의 계산 비용이 매우 높다는 한계가 있다.

또한, 대조 학습(Contrastive Learning)은 주로 자기지도 학습(Self-supervised Learning) 분야에서 성공적으로 사용되었으며, 최근 지도 학습 설정으로 확장된 연구(Khosla et al., 2020)들이 등장하였다. 하지만 능동 학습 프레임워크 내에서 샘플링 편향을 해결하기 위해 대조 학습을 도입하고, 특히 모델의 강건성과 분포 외 데이터에 대한 성능을 분석한 연구는 본 논문이 처음이라고 명시하고 있다.

## 🛠️ Methodology

### 1. 전체 파이프라인 및 문제 정의
다중 클래스 분류 문제에서 초기에는 무작위로 샘플링된 작은 크기의 레이블 데이터셋 $D_L^1$로 모델을 학습시킨다. 이후 쿼리 전략(Query Strategy) $Q$를 통해 나머지 미레이블 데이터셋 $D_U$에서 가장 정보량이 많은 $M$개의 샘플을 선택하여 레이블을 추가하고 모델을 재학습하는 사이클을 반복한다.

### 2. Supervised Contrastive Active Learning (SCAL)
SCAL은 지도 대조 학습 손실 함수를 사용하여 모델이 더 유용한 특징 표현을 학습하도록 유도한다.

**손실 함수:**
학습 시 다음과 같은 Supervised Contrastive Loss를 사용한다.
$$L_{con} = \sum_{i \in I} \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(z_i \cdot z_p / T)}{\sum_{a \in A(i)} \exp(z_i \cdot z_a / T)}$$
여기서 $z_i$는 신경망에서 추출된 특징 벡터, $T$는 온도 파라미터, $P(i)$는 동일 클래스에 속하는 긍정 샘플의 집합, $A(i)$는 자기 자신을 제외한 배치 내 모든 샘플의 집합이다.

**쿼리 전략 (Scoring Function):**
학습된 특징 공간에서 각 클래스별 클러스터를 고려하여, 해당 클래스의 기존 학습 데이터들과의 유사도가 가장 낮은(즉, 가장 먼) 샘플을 선택한다.
$$S_{score}(x) := \max_{z(x' | y' = c_k)} \frac{z(x' | y' = c_k) \cdot z(x | y = c_k)}{\|z(x' | y' = c_k)\|}$$
각 클래스 $c_k$에 대해 위 점수가 가장 낮은 샘플들을 균등하게 선택하여 클래스 간 균형을 맞추고 샘플링 편향을 줄인다.

### 3. Deep Feature Modeling (DFM)
DFM은 클래스 조건부 확률 분포를 학습하여 OOD 샘플을 탐지하는 기법을 능동 학습의 샘플 선택에 응용한다.

**방법론:**
먼저 클래스별로 PCA 변환 $\{T_k\}_{k=1}^K$를 적용하여 특징 공간의 차원을 축소한다. 쿼리 단계에서는 원래의 특징 벡터 $z$와, 이를 축소했다가 다시 복원한 벡터 사이의 거리인 **특징 재구성 오차(Feature Reconstruction Error, FRE)**를 점수로 사용한다.
$$S_{FRE}(z | y = c_k) = \|z - (T_k^\dagger \circ T_k)(z)\|^2$$
여기서 $T_k^\dagger$는 Moore-Penrose 유사 역행렬이다. 직관적으로 PCA 하위 공간에서 가장 멀리 떨어진 샘플이 가장 정보량이 많다고 판단하여 선택한다.

## 📊 Results

### 실험 설정
- **데이터셋**: CIFAR-10, Fashion-MNIST, SVHN
- **모델**: ResNet-18
- **비교 대상**: CoreSet, Learning Loss, BALD, Random, Entropy
- **지표**: 테스트 정확도, 샘플링 편향(Sampling Bias), 예상 교정 오차(ECE), 쿼리 시간, AUROC (OOD 탐지 성능)

### 주요 결과
1. **정확도 및 샘플링 편향**: SCAL은 초기 학습 사이클에서 다른 방법들보다 빠르게 정확도를 높였으며, 특히 Fashion-MNIST와 SVHN에서 최적의 최종 정확도를 달성하였다. 제안 방법들(SCAL, DFM)은 기존 방식들보다 샘플링 편향(클래스 불균형)을 유의미하게 낮추었으며, 특히 불균형 데이터셋인 SVHN에서 효과가 뚜렷하였다.
2. **쿼리 시간**: SCAL의 쿼리 계산 속도는 BALD보다 26배, CoreSet보다 11배 빨랐다. 이는 실용적인 관점에서 매우 큰 이점이다.
3. **모델 교정(Calibration)**: BALD가 가장 낮은 ECE를 기록하며 최고의 교정 성능을 보였으나, SCAL 역시 매우 낮은 쿼리 시간으로 BALD에 근접하는 우수한 교정 성능을 보였다.
4. **강건성(Robustness)**:
   - **OOD 탐지**: CIFAR-10으로 학습하고 SVHN으로 평가했을 때, SCAL의 AUROC가 다른 방법들보다 10% 이상 높게 나타났다.
   - **데이터 시프트**: Gaussian Blur가 적용된 CIFAR-10 데이터셋에 대해 SCAL이 가장 높은 정확도와 가장 낮은 ECE를 기록하여, 분포 변화에 가장 강건함을 입증하였다.

## 🧠 Insights & Discussion

본 연구는 능동 학습에서 단순히 정확도를 높이는 것을 넘어, **특징 공간의 구조적 최적화**가 모델의 강건성에 결정적인 영향을 미친다는 점을 시사한다. SCAL이 OOD 탐지와 데이터 시프트 상황에서 압도적인 성능을 보인 이유는 지도 대조 학습을 통해 클래스 내 응집도와 클래스 간 분리도를 극대화한 특징 표현을 학습했기 때문이다.

다만, 본 논문에서 제시한 쿼리 전략은 모델의 **예측 클래스(Predicted Class)**를 기반으로 유사도를 계산한다. 만약 모델의 초기 예측이 크게 틀린다면, 잘못된 클러스터를 기준으로 샘플을 선택하게 되어 오히려 편향이 심화될 가능성이 있다는 점은 잠재적인 한계점으로 해석될 수 있다. 하지만 실험 결과는 이러한 위험보다 대조 학습을 통한 다양성 확보의 이득이 훨씬 크다는 것을 보여준다.

## 📌 TL;DR

본 논문은 능동 학습의 고질적인 문제인 샘플링 편향을 해결하기 위해 **지도 대조 학습(Supervised Contrastive Learning)**과 **특징 재구성 오차(FRE)** 기반의 쿼리 전략을 제안하였다. 특히 **SCAL** 방법론은 기존의 BALD나 CoreSet보다 훨씬 빠른 속도로 샘플을 선택하면서도, 데이터 시프트 및 OOD 상황에서 매우 강력한 강건성을 보여주었다. 이 연구는 향후 안전성이 중요한 실환경(Safety-critical applications)에 배포될 딥러닝 모델의 능동 학습 프레임워크를 설계하는 데 중요한 기초 자료가 될 것으로 보인다.