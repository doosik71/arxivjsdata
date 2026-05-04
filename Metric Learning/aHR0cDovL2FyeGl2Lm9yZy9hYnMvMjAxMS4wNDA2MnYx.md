# MLAS: Metric Learning on Attributed Sequences

Zhongfang Zhuang, Xiangnan Kong, Elke Rundensteiner, Jihane Zouaoui, Aditya Arora (2020)

## 🧩 Problem to Solve

본 논문은 **Attributed Sequences**(속성付き 시퀀스)에 대한 거리 측정 학습(Distance Metric Learning) 문제를 해결하고자 한다.

일반적인 데이터 형태는 단순한 속성 벡터(Attribute vector)이거나 구조적 정보만 가진 시퀀스(Sequence) 데이터인 경우가 많다. 하지만 실제 세계의 애플리케이션(예: 웹 로그 분석, 봇 탐지 시스템)에서는 사용자의 세션 컨텍스트와 같은 **정적 속성(Attributes)**과 사용자의 행동 순서와 같은 **범주형 아이템 시퀀스(Sequence of categorical items)**가 동시에 존재하는 'Attributed Sequence' 형태의 데이터가 빈번하게 발생한다.

이러한 데이터의 핵심적인 난제는 속성과 시퀀스가 서로 독립적이지 않고 상호 의존적(Dependencies)이라는 점이다. 예를 들어, 모바일 기기라는 속성(Attribute)은 '내 주변 맛집'이라는 검색어 시퀀스(Sequence)에 영향을 줄 수 있다. 따라서 기존의 Mahalanobis 거리 기반 학습이나 단일 데이터 타입에 집중한 딥러닝 방식으로는 이러한 복합적인 구조와 상호 의존성을 충분히 반영할 수 없다.

본 논문의 목표는 속성과 시퀀스의 정보뿐만 아니라 그들 사이의 의존성까지 효과적으로 학습하여, 유사한 Attributed Sequence 간의 거리는 최소화하고 서로 다른 시퀀스 간의 거리는 일정 마진(Margin) 이상으로 멀어지게 하는 거리 측정 함수 $\Theta$를 학습하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 속성 정보를 처리하는 **AttNet**, 시퀀스 정보를 처리하는 **SeqNet**, 그리고 이 두 정보를 통합하여 상호 의존성을 포착하는 **FusionNet**으로 구성된 딥러닝 프레임워크 **MLAS**를 제안하는 것이다.

중심적인 설계 직관은 단순히 두 정보의 표현형(Representation)을 결합하는 것을 넘어, 네트워크 구조 자체를 세 가지 변형(Balanced, AttNet-centric, SeqNet-centric)으로 설계하여 데이터셋의 특성에 따라 최적의 의존성 모델링 방식을 선택할 수 있도록 한 점이다. 또한, Contrastive Loss를 통해 명시적인 클래스 라벨 없이도 유사/비유사 쌍(Pair)의 피드백만으로 거리 공간을 학습하게 하였다.

## 📎 Related Works

기존의 거리 측정 학습 연구들은 주로 다음과 같은 한계를 지닌다.

- **전통적 접근 방식:** Mahalanobis 거리 측정법을 사용하여 데이터 속성에 대한 선형 변환을 학습하는 데 집중하였다. 이는 비선형적인 복잡한 관계를 포착하는 데 한계가 있다.
- **최근의 딥러닝 접근 방식:** 신경망을 이용해 데이터를 새로운 공간으로 투영한 뒤 Euclidean 거리를 계산하는 비선형 방식을 도입하였다. 하지만 대부분의 연구가 단일 데이터 타입(속성만으로 구성되거나 시퀀스만으로 구성된 데이터)에 국한되어 있다.
- **시퀀스 기반 학습:** LSTM 등을 이용해 시퀀스 간의 유사도를 측정하는 연구가 진행되었으나, 시퀀스와 함께 제공되는 정적 속성과의 상호작용을 고려한 연구는 부족한 실정이다.

MLAS는 이러한 한계를 극복하기 위해 속성과 시퀀스를 동시에 입력으로 받고, 그들 사이의 의존성을 학습하는 통합 구조를 제안함으로써 기존의 단일 타입 기반 접근 방식과 차별화된다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

MLAS는 크게 세 가지 서브 네트워크로 구성된다: **AttNet** $\rightarrow$ **SeqNet** $\rightarrow$ **FusionNet** $\rightarrow$ **MetricNet**.

### 2. 주요 구성 요소 및 역할

- **AttNet (Attribute Network):**
  정적 속성 벡터 $x_k \in \mathbb{R}^u$를 입력으로 받는 Fully Connected Neural Network이다. $M$개의 층으로 구성되며, 각 층의 출력 $V^{(m)}$은 다음과 같이 계산된다.
  $$V^{(1)} = \delta(W^{(1)}_A x_k + b^{(1)}_A)$$
  $$V^{(M)} = \delta(W^{(M)}_A V^{(M-1)} + b^{(M)}_A)$$
  여기서 $\delta$는 $\tanh$ 활성화 함수이다.

- **SeqNet (Sequence Network):**
  범주형 아이템 시퀀스 $S_k \in \mathbb{R}^{T \times r}$를 입력으로 받는 LSTM 네트워크이다. 아이템 간의 시간적 의존성을 학습하며, 최종 타임스텝 $T$에서의 은닉 상태 $h^{(T_k)}$를 출력으로 사용한다.

- **FusionNet ($\Theta$):**
  AttNet과 SeqNet의 출력을 결합하여 최종 특징 표현형을 생성한다. 본 논문은 세 가지 설계 방식을 제안한다.
  1. **Balanced Design:** AttNet의 출력 $V^{(M)}$과 SeqNet의 출력 $h^{(T_k)}$를 연결(Concatenation)한 후, 추가적인 FC 레이어를 통해 의존성을 학습한다.
     $$y = V^{(M)} \oplus h^{(T_k)}, \quad z = \delta(W_z y + b_z)$$
  2. **AttNet-centric Design:** SeqNet의 출력을 AttNet의 첫 번째 층 입력으로 함께 넣어, 속성 학습 단계에서 시퀀스 정보가 반영되게 한다.
     $$V^{(1)} = \delta(W^{(1)}_A (x_k \oplus h^{(T_k)}) + b^{(1)}_A)$$
  3. **SeqNet-centric Design:** AttNet의 출력 $V^{(M)}$을 SeqNet의 첫 번째 타임스텝 은닉 상태 $h^{(1)}$에 더해줌으로써 시퀀스 학습이 속성 정보에 조건화(Conditioned)되게 한다.
     $$h^{(1)} = o^{(1)} \otimes \tanh(c^{(1)}) + V^{(M)}$$

- **MetricNet:**
  FusionNet을 통해 생성된 두 벡터 $\Theta(p_i)$와 $\Theta(p_j)$ 사이의 Euclidean 거리 $D_\Theta$를 계산하고, 이를 기반으로 Contrastive Loss를 적용한다.

### 3. 손실 함수 및 학습 절차

학습의 목표는 유사한 쌍은 가깝게, 비유사한 쌍은 일정 마진 $g$ 이상으로 멀어지게 하는 것이다. 사용된 **Contrastive Loss** 함수는 다음과 같다.
$$L(p_i, p_j, \ell_{ij}) = \frac{1}{2}(1-\ell_{ij})(D_\Theta)^2 + \frac{1}{2}\ell_{ij}\{\max(0, g - D_\Theta)\}^2$$
여기서 $\ell_{ij}=0$이면 유사(similar), $\ell_{ij}=1$이면 비유사(dissimilar)를 의미한다.

**학습 과정:**

1. 네트워크 파라미터를 초기화한다.
2. **Pre-training:** 입력 데이터를 재구성(Reconstruct)하는 방식으로 가중치를 초기화하여 학습 효율을 높인다.
3. **Metric Learning:** 유사/비유사 피드백 세트를 이용해 Forward propagation을 수행하고, Contrastive Loss를 계산한 뒤 Back-propagation을 통해 파라미터를 업데이트한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋:** Amadeus의 내부 애플리케이션 로그(AMS-A, AMS-B)와 Wikispeedia 데이터셋(Wiki-A, Wiki-B) 총 4종을 사용하였다.
- **비교 대상 (Baselines):**
  - **ATT:** 속성 정보만 사용.
  - **SEQ:** 시퀀스 정보만 사용.
  - **ASF:** 속성과 시퀀스를 각각 별도의 모델로 학습시킨 후 나중에 특징 벡터를 결합(Concatenate)하는 방식.
  - **MLAS (B/A/S):** 본 논문에서 제안한 세 가지 FusionNet 설계 방식.
- **평가 지표:** 학습된 특징 표현형을 사용하여 **HDBSCAN** 클러스터링을 수행하고, 그 결과의 품질을 **NMI (Normalized Mutual Information)** 점수로 측정하였다.

### 2. 주요 결과

- **피드백의 효과:** 모든 방법론에서 피드백을 사용했을 때 NMI 점수가 상승하였으며, 특히 MLAS 모델들이 가장 큰 폭의 성능 향상을 보였다.
- **정량적 성능 향상:** 가장 강력한 베이스라인인 ASF와 비교했을 때, MLAS-A는 AMS 데이터셋에서 최대 25.4% 향상되었고, MLAS-S는 Wiki 데이터셋에서 최대 26.3% 향상된 성능을 기록하였다.
- **설계 방식별 특성:** 데이터셋에 따라 최적의 구조가 달랐다. AMS-A/B 데이터셋에서는 **MLAS-A (AttNet-centric)**가, Wiki-A/B 데이터셋에서는 **MLAS-S (SeqNet-centric)**가 가장 높은 성능을 보였다.
- **강건성 분석:** 출력 차원(Output Dimension)의 변화나 Pre-training 파라미터 $\omega_A$의 변화에도 불구하고 MLAS는 일관되게 베이스라인보다 우수한 성능을 유지하였다.

## 🧠 Insights & Discussion

본 논문은 Attributed Sequence라는 복합 데이터 타입에서 거리 측정 학습을 수행하기 위해 속성-시퀀스 간의 **상호 의존성(Dependency)**을 모델링하는 것이 필수적임을 입증하였다.

특히, 단순히 두 벡터를 결합하는 ASF 방식보다, 네트워크 구조 내부에서 정보를 융합하는 MLAS 방식이 월등한 성능을 보였다는 점은 시사하는 바가 크다. 이는 속성 정보가 시퀀스의 맥락을 결정하거나, 반대로 시퀀스의 패턴이 속성의 의미를 구체화하는 비선형적 관계가 존재함을 의미한다.

다만, 실험 결과에서 데이터셋마다 최적의 아키텍처(MLAS-A vs MLAS-S)가 다르게 나타난 점은 한계이자 향후 연구 과제이다. 이는 데이터셋에 따라 속성의 영향력이 더 큰지, 아니면 시퀀스의 구조적 영향력이 더 큰지가 다르기 때문으로 추측된다. 논문에서는 이를 이론적으로 분석하지 못하고 실험적 결과로만 제시하였다.

## 📌 TL;DR

이 논문은 속성과 시퀀스가 결합된 **Attributed Sequence** 데이터의 거리 측정 학습을 위한 딥러닝 프레임워크 **MLAS**를 제안한다. AttNet(속성), SeqNet(시퀀스), FusionNet(의존성 융합) 구조를 통해 단순한 결합 이상의 상호작용을 학습하며, Contrastive Loss를 통해 유사도 공간을 최적화한다. 실세계 데이터셋 실험 결과, 기존의 단일 타입 학습 및 단순 결합 방식(ASF)보다 훨씬 높은 클러스터링 성능(NMI)을 달성하였으며, 이는 향후 부정 결제 탐지나 사용자 행동 분석과 같은 복합 시퀀스 데이터 분석 분야에 중요하게 적용될 수 있다.
