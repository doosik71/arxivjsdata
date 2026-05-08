# Self-Supervised Learning for Large-Scale Unsupervised Image Clustering

Evgenii Zheltonozhskii, Chaim Baskin, Alex M. Bronstein, and Avi Mendelson (2020)

## 🧩 Problem to Solve

본 논문은 대규모 이미지 데이터셋에서 레이블 없이 데이터를 분류하는 **완전 비지도 학습(Fully Unsupervised Learning)**의 어려움을 해결하고자 한다. 최근 컴퓨터 비전 분야에서는 **자기지도 학습(Self-Supervised Learning, SSL)**을 통해 매우 강력한 표현 학습(Representation Learning)이 가능해졌으나, 이러한 방법들의 성능 평가는 주로 소량의 레이블을 사용하는 선형 평가(Linear Evaluation)나 미세 조정(Fine-tuning)에 의존하고 있다.

따라서 저자들은 SSL로 학습된 표현이 실제 레이블이 전혀 없는 상황에서 얼마나 효과적으로 클러스터링(Clustering)될 수 있는지 평가하는 체계적인 벤치마크를 구축하고, 이를 통해 SSL이 비지도 이미지 분류의 강력한 기초 모델이 될 수 있음을 입증하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 복잡한 비지도 학습 알고리즘을 새로 설계하는 대신, **기존의 최신 SSL 모델들로부터 추출한 특징(Feature)에 단순한 차원 축소와 클러스터링 알고리즘을 적용**하여 성능을 측정하는 것이다.

주요 기여 사항은 다음과 같다:

- SSL 기반 표현을 활용한 단순하고 효과적인 비지도 분류 파이프라인을 제안하였다.
- ImageNet과 같은 대규모 데이터셋에서 SSL 모델들이 비지도 설정에서도 경쟁력 있는 결과(Top-1 정확도 약 39%~46%)를 낼 수 있음을 보여주었다.
- SSL 모델의 성능을 측정하는 새로운 표준 벤치마크로 '비지도 클러스터링 평가'를 제안하여, 향후 SSL 연구가 단순한 표현 학습을 넘어 실제 비지도 분류 성능을 추적할 수 있도록 하였다.

## 📎 Related Works

논문에서는 SSL의 주요 접근 방식을 세 가지로 분류하여 설명한다:

1. **대조 학습(Contrastive Losses):** 동일 이미지의 서로 다른 뷰(View)는 가깝게, 서로 다른 이미지의 표현은 멀게 학습하는 방식이다 (예: MoCo, SimCLR).
2. **프리텍스트 태스크(Pretext Tasks):** 이미지 회전 예측, 색상 복원, 직소 퍼즐 맞추기 등 레이블이 필요 없는 가공의 문제를 풀게 하여 특징을 학습하는 방식이다.
3. **생성 모델(Generative Models):** Autoencoder나 GAN과 같은 모델의 잠재 벡터(Latent Vector)를 표현으로 사용하는 방식이다.

기존의 비지도 학습 연구들은 주로 특정 클러스터링 목적 함수를 학습 과정에 직접 포함시켰으나, 본 논문은 SSL로 미리 학습된 고정된 특징 추출기(Feature Extractor)를 사용하여 클러스터링을 수행함으로써 표현 학습과 클러스터링 단계를 분리하여 평가한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. 전체 파이프라인

제안하는 시스템의 구조는 다음과 같은 순차적 흐름을 가진다:
$$\text{SSL Pre-trained Model} \rightarrow \text{Feature Extraction} \rightarrow \text{Dimensionality Reduction (PCA)} \rightarrow \text{K-means Clustering}$$

### 2. 세부 구성 요소 및 절차

- **특징 추출(Feature Extraction):** 미리 학습된 SSL 모델을 사용하여 훈련 및 검증 세트의 특징 벡터를 추출한다. 이때 데이터 증강(Augmentation)은 적용하지 않는다.
- **차원 축소(Dimensionality Reduction):** 고차원 공간에서의 거리 측정 효율성 저하(차원의 저주)를 방지하기 위해 **Incremental PCA**를 적용한다. PCA의 배치 크기는 $\max(4096, 2 \cdot n_{fd})$로 설정하며, 여기서 $n_{fd}$는 추출된 특징의 차원이다.
- **클러스터링(Clustering):** 차원이 축소된 특징을 바탕으로 **Mini-batch K-means** 알고리즘을 적용한다. 기본적으로 클러스터 수 $k$는 ImageNet의 클래스 수인 1000으로 설정하지만, 성능 향상을 위해 $k$를 더 크게 설정하는 과클러스터링(Overclustering) 실험도 수행한다.

### 3. 평가 지표 및 방정식

클러스터링 결과의 정확도를 측정하기 위해 다음과 같은 지표를 사용한다.

- **Accuracy:** 예측된 클러스터와 실제 클래스 간의 최적 매핑을 찾기 위해 선형 할당(Linear Assignment)을 사용한다. 과클러스터링의 경우, 하나의 클래스에 하나의 클러스터를 할당하고 나머지는 정확도를 최대화하는 방향으로 탐욕적(Greedy)으로 할당한다.
- **Normalized Mutual Information (NMI):** 두 파티션 $U, V$ 사이의 상호 정보량(MI)을 정규화한 값이다.
  $$H(U) = -\sum_{i=1}^R P(u_i) \log P(u_i)$$
  $$MI(U, V) = \sum_{i=1}^R \sum_{j=1}^C P(u_i, v_j) \log \frac{P(u_i, v_j)}{P(u_i)P(v_j)}$$
  $$NMI(U, V) = \frac{MI(U, V)}{\text{avg}(H(U), H(V))}$$
- **Adjusted Mutual Information (AMI):** 무작위 우연으로 발생할 수 있는 MI 값을 보정하여 계산한다.
- **Adjusted Rand Index (ARI):** 샘플 쌍(Pair)들이 두 파티션에서 동일하게 그룹화되었는지를 측정하는 Rand Index를 우연 확률에 대해 보정한 지표이다.

## 📊 Results

### 1. 실험 설정

- **데이터셋:** ImageNet, ObjectNet
- **비교 모델:** MoCo v2, InfoMin, SwAV, SimCLRv2, BigBiGAN (SSL 모델) 및 ResNet-152, EfficientNet-L2 (지도 학습 모델)
- **측정 지표:** ACC, ARI, AMI, NMI

### 2. 주요 결과

- **ImageNet 성능:** SSL 모델 중 SimCLRv2 (ResNet-152)가 가장 우수한 성과를 보였으며, 비지도 설정에서 **Top-1 정확도 39%**를 기록하였다. 특히 과클러스터링을 적용했을 때 정확도가 **46%**까지 상승하였다.
- **선형 평가와의 상관관계:** 대부분의 모델에서 선형 평가(Linear Evaluation) 정확도와 K-means 정확도 사이에 강한 상관관계가 관찰되었다. 다만, SimCLRv2(ResNet-152 3x)와 SwAV는 예외적인 양상을 보였다.
- **ObjectNet 일반화 성능:** ImageNet으로 학습된 모델들을 ObjectNet에 적용했을 때, 지도 학습 모델을 포함하여 대부분의 모델이 무작위 수준의 낮은 성능을 보였다. 특이하게도 **BigBiGAN**만이 무작위보다 유의미하게 높은 성능을 기록하였다.
- **어블레이션 연구(Ablation Study):**
  - PCA 차원이 1024를 넘어가면 오히려 성능이 저하되는 경향이 확인되었다.
  - 클러스터 수 $k$를 늘리는 과클러스터링이 정확도와 ARI를 모두 향상시켰다.

## 🧠 Insights & Discussion

본 논문은 SSL 모델들이 비지도 분류에서 강력한 베이스라인이 될 수 있음을 입증하였다. 그러나 분석 과정에서 몇 가지 흥미로운 논의 사항이 제기되었다.

첫째, **차원의 역설**이다. 더 큰 모델이나 더 높은 차원의 임베딩을 가진 모델(예: ResNet-152 3x)이 반드시 더 좋은 클러스터링 성능으로 이어지지 않았다. 이는 제안된 클러스터링 방법의 한계일 수도 있고, 혹은 고차원 임베딩 자체의 특성일 수 있다.

둘째, **실제 환경 일반화(ObjectNet)의 어려움**이다. ImageNet에서 높은 성능을 보인 모델들이 현실적인 조건의 ObjectNet에서 무너지는 현상은, 현재의 SSL 표현이 데이터셋의 편향(Bias)에 취약할 수 있음을 시사한다. 특히 생성 모델 기반의 BigBiGAN이 여기서 더 강건한 모습을 보인 점은 향후 연구의 중요한 실마리가 될 수 있다.

셋째, **평가 지표의 선택**이다. 저자들은 비지도 학습의 진행 상황을 추적하기 위해서는 단순 정확도보다는 AMI가 더 적절한 지표가 될 수 있다고 제안한다.

## 📌 TL;DR

본 연구는 최신 **자기지도 학습(SSL) 모델 $\rightarrow$ PCA 차원 축소 $\rightarrow$ K-means**로 이어지는 단순한 파이프라인을 통해 대규모 이미지 데이터셋의 비지도 분류 가능성을 탐색하였다. ImageNet에서 최대 46%의 정확도를 달성하며 SSL 표현의 강력함을 입증하였으며, 동시에 실제 환경(ObjectNet)으로의 일반화 부족이라는 한계를 지적하였다. 이 연구는 SSL 평가 방식에 '완전 비지도 클러스터링'이라는 새로운 기준을 제시함으로써, 향후 SSL 연구가 실제 분류 성능 향상으로 이어지도록 유도하는 역할을 할 것으로 기대된다.
