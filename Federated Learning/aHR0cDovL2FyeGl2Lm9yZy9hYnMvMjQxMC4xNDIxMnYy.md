# Comparative Evaluation of Clustered Federated Learning Methods

Michael Ben Ali, Omar El-Rifai, Imen Megdiche, André Peninou, Olivier Teste (2024)

## 🧩 Problem to Solve

Federated Learning (FL)은 데이터 프라이버시를 유지하면서 분산된 환경에서 모델을 학습시킬 수 있는 유망한 방법론이다. 그러나 실제 환경에서는 클라이언트 간의 데이터 분포가 서로 다른 Non-IID(non-independent and identically distributed) 상황이 빈번하게 발생하며, 이는 모델 가중치의 드리프트(drift)를 유발하여 전역 모델의 수렴을 방해하고 성능을 저하시킨다.

이러한 문제를 해결하기 위해 클라이언트를 유사한 데이터 분포를 가진 그룹으로 묶어 개별 모델을 학습시키는 Clustered Federated Learning (CFL)이 제안되었다. 하지만 기존의 CFL 관련 연구들은 데이터의 이질성(heterogeneity) 시나리오를 체계적으로 정의하지 않았거나, 일부 특정 케이스에 대해서만 실험을 진행하여 어떤 상황에서 CFL이 효과적인지에 대한 종합적인 분석이 부족한 실정이다. 따라서 본 논문의 목표는 제안된 데이터 이질성 분류 체계(Taxonomy)를 바탕으로 최신 CFL 알고리즘들의 성능을 엄밀하게 비교 분석하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Federated Learning에서 발생할 수 있는 데이터 이질성을 5가지 유형으로 체계화하고, 이에 따라 Server-side 및 Client-side CFL 알고리즘의 성능을 정량적으로 평가했다는 점이다. 특히 기존 문헌에서 간과되었던 Quantity Skew 시나리오를 포함하여, 데이터 분포의 특성에 따라 CFL의 클러스터링 품질과 모델 정확도가 어떻게 변하는지를 분석함으로써 CFL 적용의 가이드라인을 제시하였다.

## 📎 Related Works

기존의 FL 연구에서는 FedAvg와 같은 단순 집계 방식이 Non-IID 환경에서 성능이 떨어진다는 점이 지적되어 왔다. 이를 해결하기 위해 등장한 CFL은 단일 전역 모델 대신 여러 개의 개인화된 모델(personalized models)을 학습시키는 방향으로 발전하였다.

저자들은 기존 CFL 관련 논문들을 분석한 결과, 대부분의 연구가 MNIST, CIFAR-10과 같은 이미지 분류 데이터셋에 치중되어 있으며, 데이터 이질성의 유형을 명확히 정의하지 않은 채 임의의 시나리오를 사용한다는 점을 발견하였다. 특히 5가지 이질성 유형 중 Quantity Skew(데이터 양의 차이)를 다룬 연구가 거의 없다는 점을 지적하며 본 연구의 차별성을 강조한다.

## 🛠️ Methodology

### 1. 데이터 이질성 분류 체계 (Taxonomy of Data Heterogeneities)
논문은 두 클라이언트 $i$와 $j$의 데이터 분포 $P_i(x, y)$와 $P_j(x, y)$를 비교하여 다음과 같이 5가지 이질성을 정의한다.

1. **Concept shift on features**: 레이블은 같으나 특징 분포가 다른 경우, $P_i(x|y) \neq P_j(x|y)$. (예: 이미지 회전)
2. **Concept shift on labels**: 특징은 같으나 레이블이 다른 경우, $P_i(y|x) \neq P_j(y|x)$. (예: 레이블 스와핑)
3. **Feature distribution skew**: 레이블 조건부 분포는 같으나 특징 분포 자체가 다른 경우, $P_i(x) \neq P_j(x)$. (예: 필기구의 굵기 차이)
4. **Label distribution skew**: 특징 조건부 분포는 같으나 레이블 분포가 다른 경우, $P_i(y) \neq P_j(y)$. (예: 클래스 불균형)
5. **Quantity skew**: 클라이언트 간 데이터의 총량 차이가 심한 경우, $|D_i| \ll |D_j|$.

### 2. CFL 일반 공식
전체 $N$개의 클라이언트를 $K$개의 상호 배타적인 클러스터 $C_1, \dots, C_K$로 분할하며, 각 클러스터 $C_k$의 목적 함수 $F_k(\omega)$는 다음과 같이 정의된다.

$$ \min_{\omega \in \mathbb{R}^d} F_k(\omega) := \sum_{i \in C_k} \frac{n_i}{\sum_{j \in C_k} n_j} f_i(\omega) $$

여기서 $f_i(\omega) = \mathbb{E}_{(x,y) \sim D_i} [L_i(x,y, \omega)]$는 클라이언트 $i$의 로컬 손실 함수 기대값이다.

### 3. CFL 구현 방식
본 논문은 두 가지 대표적인 CFL 접근 방식을 비교한다.

- **Server-side Clustering (Gradient-based)**: 서버가 모델 가중치 $\omega_i$를 직접 이용하여 클러스터를 구성한다. k-means 알고리즘을 사용하여 가중치 간의 거리를 최소화하는 방향으로 클러스터를 할당한다.
  $$ \min_{\mu_k, k \in \{1, \dots, K\}} \sum_{k=1}^K \sum_{i=1}^N \mathbb{1}_{i \in C_k} \text{dist}(\omega_i, \mu_k) $$
- **Client-side Clustering (IFCA)**: 클라이언트가 서버로부터 제공받은 여러 후보 모델 $\mu_k$ 중 자신의 로컬 데이터 손실(loss)을 최소화하는 모델을 선택하여 클러스터에 합류한다.
  $$ \omega_i = \arg \min_{\mu_k, k \in \{1, \dots, K\}} \mathbb{E}_{(x,y) \sim D_i} [L_i(x,y, \mu_k)] $$

## 📊 Results

### 실험 설정
- **데이터셋**: MNIST, Fashion-MNIST, KMNIST
- **모델**: ReLU 활성화 함수를 가진 단일 은닉층(크기 200)의 fully connected 신경망
- **평가 지표**: Accuracy, ARI (Adjusted Rand Index), AMI (Adjusted Mutual Information), Homogeneity, Completeness, V-measure
- **비교 대상**: Standard FL (FedAvg), Oracle (동질적 데이터셋에 대해 중앙 집중식 학습)

### 시나리오별 결과 분석

1. **Concept shift on features**: Server-side CFL이 거의 Oracle에 근접하는 성능을 보였으며, 클러스터링 품질 또한 완벽했다. 반면 Client-side는 초기 할당의 영향으로 인해 상대적으로 성능이 낮았다.
2. **Concept shift on labels**: 두 방법 모두 Standard FL보다 성능이 크게 향상되었으며, 특히 Server-side가 매우 정확한 클러스터링을 통해 Oracle 성능에 도달하였다.
3. **Feature distribution skew**: 가장 까다로운 시나리오였다. Client-side는 대부분의 클라이언트를 하나의 클러스터로 묶어버리는 경향이 있어 성능이 저조했다. Server-side는 FL보다 낫지만, 특징 변환이 가중치에 완전히 반영되지 않아 클러스터링 품질이 완벽하지 않았다.
4. **Label distribution skew**: 두 방식 모두 클러스터링은 정확하게 수행했으나, Accuracy 향상 폭은 작았다. 이는 Standard FL 자체의 성능이 이미 충분히 높았기 때문이다.
5. **Quantity skew**: CFL이 오히려 성능을 저하시키는 경향을 보였다. 데이터 양이 매우 적은 클라이언트로 구성된 클러스터는 모델 성능이 급격히 떨어지기 때문이다.

## 🧠 Insights & Discussion

본 연구를 통해 데이터의 이질성 유형에 따라 CFL의 효용성이 극명하게 갈린다는 것을 확인하였다. 특징(Feature)이나 레이블(Label)의 정의가 완전히 다른 'Concept Shift' 상황에서는 CFL이 매우 강력한 성능 향상을 제공한다. 그러나 특징의 미묘한 차이(Feature distribution skew)가 있는 경우에는 가중치 기반의 Server-side 방식이 더 안정적이다.

특히 **Quantity Skew** 상황에서의 분석은 매우 중요하다. 단순하게 데이터를 분포 기반으로 클러스터링할 경우, 데이터 수가 너무 적은 그룹이 형성되어 학습이 제대로 이루어지지 않는 문제가 발생한다. 이는 CFL을 적용하기 전, 데이터의 양적 분포를 먼저 고려해야 함을 시사한다.

또한, 클러스터의 개수 $K$에 대한 분석 결과, $K$를 실제 이질성 클래스 수보다 약간 과대평가하여 설정하는 것이 과소평가하는 것보다 안전하며, 어느 시점 이후로는 성능이 포화(saturate)된다는 점을 발견하였다.

## 📌 TL;DR

이 논문은 FL의 Non-IID 문제를 해결하기 위한 Clustered Federated Learning(CFL)을 5가지 데이터 이질성 시나리오별로 정밀 분석하였다. 분석 결과, Concept Shift 상황에서는 Server-side CFL이 매우 효과적이지만, 데이터 양의 차이가 심한 Quantity Skew 상황에서는 CFL 적용이 오히려 독이 될 수 있음을 밝혔다. 결과적으로 CFL의 성공적인 적용을 위해서는 학습 환경의 데이터 이질성 유형에 대한 사전 이해가 필수적임을 입증하였다.