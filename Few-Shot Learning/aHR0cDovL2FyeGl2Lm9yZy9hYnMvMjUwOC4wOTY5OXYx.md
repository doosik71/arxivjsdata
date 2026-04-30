# Slot Attention-based Feature Filtering for Few-Shot Learning

Javier Rodenas, Eduardo Aguilar, Petia Radeva (2025)

## 🧩 Problem to Solve

본 논문은 Few-Shot Learning (FSL) 환경에서 발생하는 **무관한 특징(irrelevant features)에 의한 성능 저하 문제**를 해결하고자 한다. 

FSL은 매우 적은 수의 학습 데이터만으로 새로운 클래스를 분류해야 하므로, 모델이 제한된 데이터로부터 일반화 가능한 핵심 특징을 추출하는 것이 매우 어렵다. 특히 이미지 분류 작업에서 배경 요소와 같은 비관련 특징(non-relevant features)은 쿼리 이미지와 서포트 이미지 간의 유사도 측정 과정에서 혼란을 야기하며, 이는 결과적으로 오분류와 오버피팅(overfitting)으로 이어진다.

따라서 본 연구의 목표는 Slot Attention 메커니즘을 활용하여 클래스와 무관한 약한 특징들을 식별하고 필터링함으로써, 가장 변별력 있는 특징(discriminative features)에 집중하게 하여 FSL의 분류 성능을 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Slot Attention을 이용한 특징 필터링(SAFF)** 구조를 제안하는 것이다. 주요 기여 사항은 다음과 같다.

1. **Slot Attention 기반의 특징 필터링:** Slot Attention을 FSL 파이프라인에 통합하여 서포트 및 쿼리 이미지에서 관련 없는 특징을 효과적으로 배제하고 변별력 있는 특징만을 추출한다.
2. **Class-Aware Slot 기반의 패치 임베딩 필터링:** 무작위 초기화 대신 Class Token을 시드(seed)로 사용하여 Slot을 초기화함으로써, 클래스 표현과 가장 잘 정렬된 Slot만을 선택해 필터링 마스크를 생성하는 결합 어텐션(combined attention) 메커니즘을 도입하였다.
3. **성능 검증 및 시각화:** CIFAR-FS, FC100, miniImageNet, tieredImageNet 등 4가지 벤치마크 데이터셋에서 기존 최신 방법론(SOTA)보다 우수한 성능을 입증하였으며, 시각화 분석을 통해 불필요한 특징이 성공적으로 제거됨을 보였다.

## 📎 Related Works

### 1. Few-Shot Learning (FSL)
FSL 접근 방식은 크게 세 가지로 나뉜다.
- **Optimization-based:** MAML, Reptile과 같이 새로운 태스크에 빠르게 적응할 수 있도록 모델 파라미터를 최적화하는 방식이다.
- **Metric-based:** ProtoNet, Matching Networks와 같이 특징 공간에서 동일 클래스 샘플들이 가깝게 군집화되도록 유사도 측정 방식을 설계하는 방식이다.
- **Transfer learning-based:** 대규모 데이터셋에서 사전 학습된 모델을 특징 추출기로 사용하고 소량의 데이터로 파인튜닝하는 방식이다.

### 2. 특징 필터링 및 어텐션 메커니즘
데이터 부족으로 인한 오버피팅을 막기 위해 Hilbert-Schmidt Independence Criterion 기반의 특징 선택이나, 전역/지역 특징에 집중하는 어텐션 메커니즘이 연구되었다. 특히 **Slot Attention**은 복잡한 장면을 객체 중심의 표현으로 분해하고 반복적인 정제(iterative refinement)가 가능하다는 점에서 FSL의 특징 정제에 유리하다.

### 3. 기존 방식과의 차별점
기존의 단순한 어텐션 방식과 달리, SAFF는 Class Token을 기준으로 Slot의 유사도를 계산하여 **클래스 관련성이 높은 Slot만 선택적으로 결합**하는 필터링 과정을 거친다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 1. 전체 시스템 구조
SAFF의 전체 파이프라인은 다음과 같은 단계로 구성된다.
1. **특징 추출:** 사전 학습된 Vision Transformer (ViT)를 통해 이미지를 패치 임베딩($\mathbb{R}^{P \times D}$)과 글로벌 정보를 담은 Class Token으로 분리한다.
2. **Slot Attention 정제:** Class Token으로 초기화된 Slot들이 반복적인 어텐션 과정을 통해 패치 임베딩에서 서로 다른 패턴을 학습하고 정제한다.
3. **특징 필터링:** 정제된 Slot들과 Class Token 간의 유사도를 계산하여, 기준치 이상의 유사도를 가진 Slot들만 사용하여 어텐션 마스크를 생성한다.
4. **가중치 적용 및 결합:** 생성된 마스크를 패치 임베딩에 곱해 무관한 영역의 가중치를 낮추고, 다시 Class Token을 더해 최종 특징 표현을 생성한다.
5. **분류:** 정제된 쿼리와 서포트 특징 간의 유사도 행렬을 생성하고, 이를 MLP에 통과시켜 최종 클래스 점수를 산출한다.

### 2. 주요 메커니즘 및 방정식

#### 2.1 Slot 초기화 및 정제
일반적인 Slot Attention은 무작위 초기화를 사용하지만, SAFF는 **Class Token을 시드로 사용**하여 Slot이 처음부터 클래스 관련 특징에 집중하도록 강제한다.

#### 2.2 Class-Aware 특징 필터링
정제된 Slot $\mathbf{S}^R$과 Class Token $\mathbf{C}$에 대해 $L_2$ 정규화를 수행한다.
$$\hat{\mathbf{S}}^R = \frac{\mathbf{S}^R}{\|\mathbf{S}^R\|_2}, \quad \hat{\mathbf{C}} = \frac{\mathbf{C}}{\|\mathbf{C}\|_2}$$

이후 코사인 유사도 $d(\cdot, \cdot)$를 계산하고, 이를 $[0, 1]$ 범위로 Min-Max 정규화한다.
$$\text{similarity}_{\text{norm}} = \frac{\text{similarity} - \text{similarity}_{\min}}{\text{similarity}_{\max} - \text{similarity}_{\min}}$$

유사도가 $0.5$보다 큰 Slot들에 대해서만 바이너리 마스크 $\mathbf{M}$을 생성하고, 선택된 Slot들의 평균을 통해 결합 어텐션 $\mathbf{A}_{\text{combined}}$를 구한다.
$$\mathbf{A}_{\text{combined}} = \frac{\sum_{i=1}^{N} \mathbf{A}_{i}^{\text{masked}}}{N_M}$$

최종적으로 패치 임베딩에 이 가중치를 곱하여 무관한 정보를 제거한다.
$$\text{embeddings}_{\text{weighted}} = \text{embeddings} \cdot \mathbf{A}_{\text{combined}}$$

#### 2.3 최종 특징 표현 및 분류
필터링된 임베딩에 Class Token의 영향력을 조절하는 파라미터 $\lambda$를 더해 최종 특징 $\mathbf{F}$를 생성한다. ($\lambda=2$ 사용)
$$\mathbf{F} = \text{embeddings}_{\text{weighted}} + \lambda \cdot \text{classtoken}$$

쿼리와 서포트 간의 유사도 행렬 $\mathbf{S}_{ij}$를 구한 뒤, 이를 평탄화(Flatten)하여 MLP를 통해 최종 점수를 도출한다.
$$\text{scores}_{ij} = \text{MLP}(\text{Flatten}(\mathbf{S}_{ij}))$$

## 📊 Results

### 1. 실험 설정
- **데이터셋:** CIFAR-FS, FC100, miniImageNet, tieredImageNet.
- **백본:** ViT-B/16 (사전 학습됨).
- **하이퍼파라미터:** 5개의 Slot, 5회 반복 정제, 임베딩 차원 384.
- **평가 지표:** 5-way 1-shot 및 5-way 5-shot 정확도 (Median, Mean, Std. Dev. 보고).

### 2. 주요 결과
SAFF는 모든 데이터셋에서 기존 SOTA 방법론인 CPEA를 능가하는 성능을 보였다.

| Dataset | Shot | SAFF (Median) | CPEA (Median) |
| :--- | :---: | :---: | :---: |
| CIFAR-FS | 1 | **78.48%** | 78.32% |
| CIFAR-FS | 5 | **90.30%** | 89.50% |
| FC100 | 1 | **47.17%** | 46.70% |
| FC100 | 5 | **66.00%** | 65.70% |
| miniImageNet | 1 | **71.51%** | 71.34% |
| miniImageNet | 5 | **87.19%** | 86.73% |
| tieredImageNet | 1 | **74.73%** | 74.58% |
| tieredImageNet | 5 | **88.97%** | 88.71% |

- **통계적 유의성:** McNemar 테스트 결과, 4개 데이터셋 중 3개(tieredImageNet, miniImageNet, FC100)에서 $p \le 0.001$로 통계적으로 유의미한 향상이 확인되었다.

### 3. 분석 및 절제 실험 (Ablation Study)
- **Slot 수 및 반복 횟수:** Slot 수는 5개일 때 최적의 성능을 보였으며, 10개로 늘리면 오히려 성능이 감소하였다. 반복 횟수는 3회에서 10회 사이에서 비교적 안정적인 성능을 유지하였다.
- **필터링 전략:** 단순 바이너리 마스크(0 또는 1)보다 **가중치 기반 마스크(Weighted Mask)**를 사용했을 때 더 높은 정확도를 보였다. 이는 문맥 정보를 보존하면서 중요한 패치에 더 많은 가중치를 주는 것이 FSL의 데이터 부족 상황에서 더 효과적임을 의미한다.
- **어텐션 메커니즘 비교:** Dot-Product, Cross Attention보다 Slot Attention을 사용했을 때 가장 높은 정확도를 기록하였다.

## 🧠 Insights & Discussion

### 1. 강점
본 논문은 Slot Attention의 '객체 중심 표현 학습' 능력을 FSL의 '특징 필터링' 문제에 성공적으로 적용하였다. 특히 단순한 어텐션 맵 생성에 그치지 않고, **Class Token과의 유사도를 통해 Slot 자체를 필터링**함으로써 클래스 관련성이 높은 특징만을 정교하게 추출해낸 점이 우수하다. 또한, 바이너리 필터링의 과도한 정보 손실을 Weighted Mask로 해결하여 성능을 최적화하였다.

### 2. 한계 및 미해결 질문
- **하이퍼파라미터 의존성:** Slot의 수와 반복 횟수를 사전에 정의해야 하며, 이 설정에 따라 성능이 민감하게 반응한다.
- **복잡한 환경의 한계:** 클래스 본체와 노이즈를 구분하기 매우 어려운 복잡한 환경에서는 Slot Attention의 캡처 능력이 저하될 가능성이 있다.
- **계산 비용:** 반복적인 정제 과정이 추가됨에 따라 단순한 메트릭 기반 방식보다 추론 시간이 증가할 수 있다.

### 3. 비판적 해석
논문에서 제시한 성능 향상 폭이 CPEA 대비 수치적으로는 크지 않은 경우가 많으나(약 0.1%~0.8%), 4개의 서로 다른 데이터셋에서 일관되게 향상되었다는 점과 통계적 유의성을 검증했다는 점에서 의미가 있다. 다만, Slot의 최적 개수를 찾는 과정이 실험적(empirical)으로 결정되었으므로, 이를 자동화하거나 동적으로 조절하는 메커니즘이 추가된다면 더 견고한 모델이 될 것이다.

## 📌 TL;DR

본 논문은 Few-Shot Learning에서 배경 노이즈 등으로 인한 성능 저하를 막기 위해 **Slot Attention 기반의 특징 필터링(SAFF)** 기법을 제안한다. Class Token을 시드로 하여 Slot을 정제하고, 클래스 유사도가 높은 Slot들만 결합하여 가중치 마스크를 생성함으로써 변별력 있는 특징만을 추출한다. 실험 결과 4가지 주요 벤치마크에서 SOTA 성능을 달성하였으며, 특히 가중치 기반 필터링이 유효함을 입증하였다. 향후 클래스별 전용 Slot을 할당하는 연구를 통해 하이퍼파라미터 의존성을 줄일 가능성이 크다.