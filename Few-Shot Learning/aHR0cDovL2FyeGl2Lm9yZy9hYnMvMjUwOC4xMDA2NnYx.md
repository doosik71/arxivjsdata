# Stochastic-based Patch Filtering for Few-Shot Learning

Javier Rodenas, Eduardo Aguilar, Petia Radeva (2025)

## 🧩 Problem to Solve

본 논문은 음식 이미지 분류를 위한 Few-Shot Learning (FSL)에서 발생하는 성능 저하 문제를 해결하고자 한다. 음식 이미지는 조리 방식, 가니쉬(garnishes), 조명 조건, 카메라 각도 등에 따라 동일한 클래스 내에서도 시각적 변동성(intra-class variability)이 매우 크며, 서로 다른 클래스 간에도 시각적 유사성이 높은 inter-class overlap 문제가 빈번하게 발생한다.

이러한 복잡성은 FSL 모델이 쿼리(query) 이미지와 서포트(support) 이미지를 비교할 때, 클래스를 식별하는 핵심 요소보다는 배경이나 무관한 요소에 집중하게 만들어 오분류를 유발한다. 특히 데이터가 극소수인 FSL 환경에서는 모델이 일반적인 특징을 학습하기보다 제한된 샘플을 단순히 암기하는 overfitting에 취약하며, 무관한 특징(noisy features)을 효과적으로 제거하지 못하는 문제가 발생한다. 따라서 본 논문의 목표는 음식 이미지의 복잡한 배경 속에서 클래스 고유의 특징을 가진 패치만을 효과적으로 선택하여 분류 성능을 높이는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 **Stochastic-based Patch Filtering (SPFF)**으로, 모든 패치를 사용하거나 단순히 유사도가 높은 상위 $k$개의 패치를 선택하는 결정론적 방식 대신, 확률 기반의 샘플링을 통해 유의미한 패치를 필터링하는 것이다.

중심적인 설계 직관은 다음과 같다. 클래스 표현(class-aware embedding)과 유사도가 높은 패치일수록 선택될 확률을 높게 부여하되, 어느 정도의 확률적 자유도(stochastic freedom)를 둠으로써 모델이 훈련 과정에서 더 다양하고 변별력 있는 패치들을 탐색하게 하는 것이다. 이를 통해 배경 소음을 제거함과 동시에, 특정 부분에만 과도하게 집중되어 발생하는 overfitting을 방지하고 일반화 능력을 향상시킬 수 있다.

## 📎 Related Works

FSL 연구는 크게 metric-based, optimization-based, transfer learning-based 방법론으로 나뉜다. Optimization-based 방법인 MAML 등은 빠른 적응을 위한 최적 파라미터를 찾는 데 집중하며, Metric-based 방법은 쿼리와 서포트 이미지 간의 거리나 유사도를 통해 클래스를 구분한다.

최근에는 Vision Transformer (ViT)를 활용하여 이미지를 패치 단위로 나누어 처리하는 방식이 주목받고 있다. 특히 CPEA나 CPES와 같은 연구들은 패치 레벨의 임베딩과 클래스 표현을 결합하여 배경 추론 문제를 해결하려 시도하였다. 하지만 기존의 많은 방식들이 결정론적인 가중치 부여나 단순한 필터링에 의존했다는 한계가 있다.

본 논문의 SPFF는 추가적인 세부 재료(ingredients) 정보나 외부의 시맨틱 표현 없이, 오직 ViT에서 추출된 클래스 토큰과 패치 임베딩 간의 확률적 관계만을 이용하여 관련 특징을 추출한다는 점에서 기존 접근 방식과 차별화된다.

## 🛠️ Methodology

### 전체 파이프라인

SPFF의 전체 구조는 ViT-S/16 백본을 통한 특징 추출, 확률 기반의 패치 필터링, 그리고 최종 유사도 기반 분류의 단계로 구성된다.

### 상세 방법론 및 주요 방정식

**1. 특징 추출 및 정규화**
입력 이미지는 ViT-S/16을 통해 $P=196$개의 패치 임베딩 $\text{P} \in \mathbb{R}^{P \times D}$와 전역 정보를 담은 클래스 토큰 $\text{C} \in \mathbb{R}^{1 \times D}$로 변환된다. 여기서 $D=384$이다. 먼저 L2 정규화를 수행한다.
$$\hat{P} = \frac{P}{\|P\|_2}, \quad \hat{C} = \frac{C}{\|C\|_2}$$

**2. 확률 기반 패치 필터링 (Stochastic Patch Filtering)**
각 패치 임베딩 $\hat{P}_i$와 클래스 토큰 $\hat{C}$ 사이의 코사인 유사도 $S_i$를 계산한다.
$$S_i = \hat{P}_i \cdot \hat{C}, \quad \forall i \in \{1, 2, \dots, P\}$$
계산된 유사도 점수를 Softmax 함수에 통과시켜 각 패치가 선택될 확률 분포 $p$를 생성한다.
$$p_i = \frac{\exp(S_i)}{\sum_{j=1}^{P} \exp(S_j)}$$
이후, 결정론적인 top-k 선택 대신 다항 분포(Multinomial distribution)를 사용하여 $k$개의 패치 인덱스를 확률적으로 샘플링한다.
$$\{i_1, i_2, \dots, i_k\} \sim \text{Multinomial}(k, p)$$
최종적으로 선택된 패치들의 집합을 $P_{\text{selected}}$라고 한다.

**3. 클래스 인식 강화 (Class-aware Addition)**
필터링된 패치들이 클래스 고유의 특성을 더 강하게 갖도록 클래스 토큰을 선형적으로 더해주는 융합 전략을 사용한다.
$$\hat{P} = P_{\text{selected}} + \lambda \cdot C$$
이때 $\lambda=2$로 설정하여 클래스 관련 표현을 강화한다.

**4. 분류 절차**
쿼리 이미지와 서포트 이미지의 필터링된 패치들 사이의 코사인 유사도를 계산하여 밀집 유사도 행렬 $T_{ij}$를 생성한다.
$$T_{ij} = d(\hat{P}_{\text{support}}, \hat{P}_{\text{query}})$$
이 행렬을 Flatten 하여 MLP(Multi-Layer Perceptron)에 통과시켜 최종 점수 $\text{scores}_{ij}$를 얻고, 동일 클래스의 모든 서포트 샘플에 대해 점수를 합산하여 최종 클래스 확률을 결정한다.
$$s_i^n = \sum_{m=1}^{M} \text{scores}_{im}^n, \quad p_i^n = \frac{\exp(s_i^n)}{\sum_{m=1}^{N} \exp(s_i^m)}$$

## 📊 Results

### 실험 설정

- **데이터셋**: Food-101, VireoFood-172, UECFood-256의 세 가지 공개 데이터셋을 사용하였다.
- **설정**: 5-way 1-shot 및 5-way 5-shot 시나리오에서 평가되었으며, 선택하는 패치 수 $k$는 전체의 50%인 98개로 설정하였다.
- **비교 대상**: MVFSL-LC, MVFSL-TC, Fusion Learning, RER 등 최신 SOTA 방법론들과 비교하였다.

### 주요 결과

SPFF는 모든 데이터셋과 설정에서 기존 방법론들을 상회하는 성능을 보였다.

- **Food-101**: 5-shot 설정에서 83.32%의 정확도를 기록하여 RER 대비 2.85% 향상되었다.
- **VireoFood-172**: 5-shot에서 94.64%의 정확도로 RER를 능가하였다.
- **UECFood-256**: 특히 이 데이터셋에서 괄목할 만한 성장을 보여, 5-shot 기준 88.71%의 정확도를 달성하며 LR 대비 약 18.97%의 성능 향상을 이루었다.

### 분석 결과

- **확률적 선택의 효과**: 결정론적(Deterministic) 선택보다 확률적(Stochastic) 선택이 더 높은 정확도를 보였다. 이는 무작위성이 가미된 샘플링이 더 다양하고 변별력 있는 패치를 캡처하여 일반화 능력을 높였기 때문이다.
- **패치 수 $k$의 영향**: $k=98$일 때 성능이 정점에 도달하며, 너무 적은 패치는 정보 부족을, 너무 많은 패치($k=196$)는 노이즈 포함으로 인해 성능이 하락함을 확인하였다.
- **유사도 지표**: 코사인 유사도가 Manhattan이나 Euclidean 거리보다 가장 높은 성능을 보였다.

## 🧠 Insights & Discussion

본 논문은 단순한 필터링을 넘어 **'제어된 무작위성(controlled randomness)'**이 FSL의 고질적인 문제인 overfitting을 해결하는 데 기여할 수 있음을 입증하였다. 시각화 분석 결과, 결정론적 방식은 특정 영역에 지나치게 집중되거나 때로는 배경에 뭉치는 경향이 있는 반면, 확률적 방식은 클래스를 대표하는 더 넓고 다양한 영역을 포착하는 특성을 보였다.

다만, 몇 가지 한계점이 존재한다. 첫째, 최적의 패치 수 $k$에 대한 설정이 정적이며, 이미지의 내용에 따라 적절한 패치 수가 다를 수 있음에도 이를 자동으로 결정하는 기전이 부족하다. 둘째, 음식 이미지 외의 다른 도메인에서도 동일한 효과가 나타날지는 추가적인 검증이 필요하다.

비판적으로 해석하자면, 클래스 토큰을 더해주는 $\lambda=2$라는 상수가 이전 연구(CPEA)의 값을 그대로 차용한 점은 본 모델의 고유한 최적화 결과라기보다 기존 기법의 전이로 볼 수 있다.

## 📌 TL;DR

본 논문은 음식 이미지의 높은 변동성과 배경 노이즈 문제를 해결하기 위해, ViT 패치 임베딩을 클래스 유사도 기반의 확률 분포로 샘플링하는 **SPFF(Stochastic-based Patch Filtering)**를 제안한다. 이 방법은 결정론적 선택보다 일반화 성능이 뛰어나며, Food-101, VireoFood-172, UECFood-256 데이터셋에서 SOTA 성능을 달성하였다. 확률적 패치 선택이라는 단순하지만 효과적인 전략을 통해 데이터 부족 환경에서의 overfitting을 완화하였으며, 이는 향후 다양한 도메인의 Few-Shot 이미지 분류 연구에 중요한 영감을 줄 수 있다.
