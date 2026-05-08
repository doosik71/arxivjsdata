# Probabilistic AND-OR Attribute Grouping for Zero-Shot Learning

Yuval Atzmon, Gal Chechik (2018)

## 🧩 Problem to Solve

본 논문은 Zero-Shot Learning (ZSL)에서 시각적 클래스를 인식하기 위해 사용되는 속성(Attribute) 공간의 구조적 정보를 어떻게 효과적으로 모델링할 것인가에 대한 문제를 다룬다. ZSL은 학습 단계에서 본 적 없는 클래스를 인식해야 하므로, 텍스트 설명이나 속성 집합과 같은 시맨틱 정보에 의존한다.

기존의 많은 ZSL 접근 방식은 속성들을 유클리드 공간이나 심플렉스(Simplex)와 같은 '평면적(Flat)'인 공간에 임베딩하여 처리한다. 그러나 실제 속성들 사이에는 복잡한 논리적 관계가 존재한다. 예를 들어, 새의 깃털 색상은 '빨간색 또는 파란색'일 수 있지만 두 색상이 동시에 나타나는 경우는 드물며, 부리의 모양과 깃털의 색상은 서로 독립적인 관계를 가질 수 있다. 이러한 속성 간의 논리적 상호작용을 무시하고 평면적으로 처리할 경우, 시맨틱 구조의 중요한 정보를 손실하게 되어 분류 성능이 저하된다. 또한, 일반적인 확률 모델을 통해 이러한 복잡한 구조를 데이터로부터 직접 학습시키기에는 ZSL 데이터셋의 크기가 너무 작다는 한계가 존재한다.

따라서 본 논문의 목표는 속성들 간의 자연스러운 'Soft-AND' 및 'Soft-OR' 관계를 캡처할 수 있는 확률 모델을 제안하고, 이를 딥러닝 기반의 속성 검출 모델과 결합하여 end-to-end로 학습시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 속성 공간에 논리적 구조를 도입한 **LAGO (Learning Attribute Grouping for 0-shot learning)** 모델의 제안이다.

핵심 아이디어는 속성들을 여러 개의 그룹으로 묶고, 그룹 내부에서는 **Soft-OR (가중 합)** 관계를, 그룹 간에는 **Soft-AND (확률의 곱)** 관계를 적용하는 것이다. 이는 사람이 객체를 인식할 때 "부리 모양이 뾰족하거나 길고(OR), 깃털 색이 초록색이며(AND), 다리가 노란색이다(AND)"와 같은 논리적 구조로 판단하는 직관을 모델링한 것이다.

특히, 이러한 그룹 구조를 고정된 것으로 보지 않고 데이터로부터 함께 학습할 수 있는 'Soft Grouping' 메커니즘을 도입하였다. 이를 통해 도메인 지식이 있을 경우 이를 사전 지식으로 활용할 수 있으며, 없을 경우 데이터로부터 최적의 그룹핑을 찾아낼 수 있다. 또한, LAGO는 기존의 대표적인 ZSL 방법론인 DAP와 ESZSL이 LAGO의 특수한 극단적 사례(Special Cases)임을 수학적으로 증명함으로써 두 방법론을 하나의 통일된 확률 프레임워크로 통합하였다.

## 📎 Related Works

기존의 속성 기반 ZSL 연구들은 주로 다음과 같은 접근 방식을 취했다.

1. **Direct Attribute Prediction (DAP):** 각 속성에 대한 이진 분류기를 학습시키고, 클래스-속성 매핑의 하드 임계값(Hard-threshold)을 사용하여 클래스를 예측하는 베이지안 접근 방식을 사용한다. 하지만 이는 속성 간의 상호작용을 고려하지 않는 단순한 곱셈 구조이다.
2. **Embedding-based ZSL (예: ESZSL, DEVISE):** 시각적 특징과 시맨틱 속성 간의 호환성 함수(Compatibility function)를 학습하여 매칭시키는 방식이다. 대부분의 경우 속성들을 평면적인 벡터 공간에 임베딩하여 처리하며, 이로 인해 속성 간의 고차원적 논리 구조를 놓치는 경향이 있다.
3. **Structured Models:** 베이지안 네트워크나 AND-OR 문법(Grammar)을 사용하여 속성 구조를 모델링하려는 시도가 있었으나, 이러한 복잡한 구조를 학습시키기 위해서는 매우 방대한 양의 데이터가 필요하다는 단점이 있다.

LAGO는 이러한 기존 방식들의 한계를 극복하기 위해, 복잡한 그래프 모델보다는 단순하지만 강력한 AND-OR 구조를 채택하고 이를 Soft-relaxation하여 딥러닝 네트워크 내에서 효율적으로 학습 가능하게 설계함으로써 차별점을 갖는다.

## 🛠️ Methodology

LAGO의 전체 파이프라인은 $X \rightarrow A \rightarrow G \rightarrow Z$의 세 단계 매핑 과정으로 구성된다.

### 1. 속성 예측 (Attribute Prediction: $X \rightarrow A$)

이미지 $x$를 입력으로 하여 각 이진 속성 $a_m$이 존재할 확률 $p(a_m|x)$를 예측한다. 이는 ResNet-101로 추출한 이미지 특징 위에 Fully-connected layer와 Sigmoid 활성화 함수를 얹은 형태로 구현된다.

### 2. 그룹 내 Soft-OR 스코어링 (Within-Group Model: $A \rightarrow G$)

속성들을 $K$개의 그룹 $G_k$로 나눈다. 각 그룹은 해당 클래스 $z$에 대해 가중치 적용된 OR 관계를 계산한다.

- **Soft-OR 정의:** 그룹 내의 속성들이 상호 배타적이라고 가정할 때, 그룹 $g_{k,z}$가 참일 확률은 다음과 같이 근사된다.
$$p(g_{k,z}=T|x) \approx p(g_{k,z}=T) \sum_{m \in G'_k} \frac{p(a_m=T|z)}{p(a_m=T)} p(a_m=T|x)$$
여기서 $G'_k$는 보완 속성(complementary attribute) $\tilde{a}_k$를 포함한 집합이다. 이 식은 속성 $a_m$이 검출될 확률과 해당 속성이 클래스 $z$에 속할 확률의 비율을 가중치로 사용하여 합산하는 구조이다.

### 3. 그룹 간 Soft-AND 결합 (Conjunction of Groups: $G \rightarrow Z$)

최종 클래스 $z$에 대한 확률은 각 그룹 스코어들의 Soft-AND(곱셈)를 통해 계산된다.
$$p(z|x) \approx p(z) \prod_{k=1}^{K} \frac{p(g_{k,z}=T|x)}{p(g_{k,z}=T)}$$
결과적으로 전체 모델의 식은 다음과 같이 요약된다.
$$p(z|x) \approx p(z) \prod_{k=1}^{K} \left[ \sum_{m \in G'_k} \frac{p(a_m=T|z)}{p(a_m=T)} p(a_m=T|x) \right]$$

### 4. Soft Grouping 및 학습 절차

속성을 그룹에 할당하는 것을 확률적으로 처리하기 위해 그룹 멤버십 변수 $\Gamma_{m,k} = p(m \in G'_k)$를 도입한다. $\Gamma$는 행렬 $V$에 Softmax를 적용하여 생성되며, 이를 통해 특정 속성이 여러 그룹에 걸쳐 있을 수 있는 Soft-relaxation을 구현한다.

**학습 목적 함수 (Loss Function):**
LAGO는 다음과 같은 정규화된 교차 엔트로피 손실 함수를 사용하여 end-to-end로 학습된다.
$$\mathcal{L} = \text{CXE}(p(z|x), z) + \alpha \text{BXE}(p(a|x), a) + \beta ||W||^2_{Fro} + \lambda ||WS||^2_{Fro} + \psi ||\Gamma(V) - \Gamma(V_{SEM})||^2_{Fro}$$

- $\text{CXE}$: 클래스 예측을 위한 categorical cross-entropy.
- $\text{BXE}$: 속성 예측을 위한 binary cross-entropy (실험 결과 $\alpha=0$일 때 성능이 더 좋았다고 명시됨).
- $\|W\|^2_{Fro}, \|WS\|^2_{Fro}$: 가중치 $W$에 대한 정규화 항.
- $\|\Gamma(V) - \Gamma(V_{SEM})\|^2_{Fro}$: 사전 정의된 시맨틱 그룹 구조($\Gamma_{SEM}$)가 있을 경우, 학습되는 구조가 이와 유사해지도록 강제하는 정규화 항.

## 📊 Results

### 실험 설정

- **데이터셋:** CUB (새 종 인식), AWA2 (동물 인식), SUN (장면 인식).
- **비교 대상:** DAP, ESZSL, ALE, SYNC, SJE, DEVISE, Zhang2018.
- **평가 지표:** 클래스 균형 정확도(Class-balanced accuracy).

### 주요 결과

1. **정량적 성과:** LAGO는 CUB(57.8%)와 AWA2(64.8%) 데이터셋에서 기존의 모든 베이스라인을 유의미한 차이로 앞서며 SOTA 성능을 달성하였다. SUN 데이터셋(57.5%)에서는 소폭 뒤처졌으나 경쟁력 있는 성능을 보였다.
2. **변체 분석 (Variants):**
    - 시맨틱 그룹 정보를 사전 지식으로 활용한 `LAGO-Semantic-Soft`가 가장 높은 성능을 보였다.
    - 사전 지식이 전혀 없는 `LAGO-K-Soft` 역시 데이터로부터 그룹 구조를 학습하여, 단순 singleton 그룹을 사용하는 `LAGO-Singletons`보다 우수한 성능을 기록하였다.
3. **DAP와의 비교:** `LAGO-Singletons` (모든 속성이 개별 그룹인 경우)는 DAP의 soft-relaxation 버전으로 볼 수 있으며, 단순히 이 변경만으로도 DAP보다 평균 40% 더 높은 성능을 보였다.

### 정성적 분석 및 인사이트

- **Soft-OR의 효과:** 특정 속성이 검출되지 않더라도 유사한 다른 속성이 검출되면 동일 그룹 내 OR 관계를 통해 정답 클래스를 맞출 수 있어 False Negative를 줄인다.
- **Soft-AND의 효과:** 여러 그룹의 조건이 모두 충족되어야 하므로, 일부 속성만 일치하여 발생하는 False Positive(오탐지)를 효과적으로 억제한다.
- **학습된 구조의 특징:** 데이터로부터 학습된 $\Gamma$는 매우 희소(sparse)한 특성을 보이며, 특히 서로 반대되는 성질을 가진 속성(anti-correlated attributes, 예: 빨간 발 vs 파란 발)들을 하나의 그룹으로 묶는 경향이 있음이 확인되었다.

## 🧠 Insights & Discussion

본 논문은 ZSL에서 속성 간의 논리적 구조를 모델링하는 것이 성능 향상에 결정적임을 보여주었다. 특히 LAGO의 강점은 다음과 같다.
첫째, 복잡한 확률 모델을 단순화하여 딥러닝 프레임워크 내에서 end-to-end로 학습 가능하게 만든 점이다.
둘째, 상충되는 속성들을 하나의 OR 그룹으로 묶는 모델의 자생적 학습 능력은 인간의 인지 구조와 유사한 특성을 보인다는 점이 흥미롭다.

다만, 몇 가지 한계점과 논의 사항이 존재한다.

- **그룹 수 $K$의 의존성:** $K$가 너무 작으면 모델이 지나치게 허용적(permissive)이 되어 성능이 급격히 떨어진다. 이는 SUN 데이터셋에서 $K$가 적어 성능이 낮게 나온 이유이기도 하다.
- **가정의 단순함:** 그룹 내 속성들이 상호 배타적이라는 가정이나, 보완 속성 $\tilde{a}_k$를 상수로 처리하는 등의 근사는 실제 복잡한 시맨틱 관계를 완전히 반영하지 못할 수 있다.
- **논리 표현의 확장성:** 현재의 AND-OR 구조는 매우 기초적인 수준이다. 향후 연구에서 "A가 있지만 B는 없다"와 같은 더 풍부한 논리 표현( richer logical expressions)을 도입한다면 더욱 정밀한 분류가 가능할 것이다.

## 📌 TL;DR

LAGO는 ZSL의 평면적인 속성 공간을 극복하기 위해 **속성 그룹 간의 Soft-AND 및 그룹 내의 Soft-OR 관계**를 모델링한 확률적 프레임워크이다. 이 모델은 그룹 구조를 데이터로부터 직접 학습하거나 사전 지식을 활용할 수 있으며, CUB와 AWA2 벤치마크에서 SOTA 성능을 달성하였다. 또한, 기존의 DAP와 ESZSL을 하나의 통합된 수식으로 설명해 냈다는 점에서 학술적 가치가 높으며, 향후 더 복잡한 시맨틱 논리 구조를 ZSL에 접목하는 연구의 기초가 될 것으로 보인다.
