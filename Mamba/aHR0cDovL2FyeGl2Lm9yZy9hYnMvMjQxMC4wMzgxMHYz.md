# Exploring the Limitations of Mamba in COPY and CoT Reasoning

Ruifeng Ren, Zhicong Li, Yong Liu (2025)

## 🧩 Problem to Solve

본 논문은 최근 Transformer의 대안으로 주목받고 있는 Mamba 아키텍처의 표현 능력(Expressive Ability)에 대한 근본적인 한계를 분석한다. Transformer는 추론 시 시퀀스 길이에 따라 계산 비용이 선형적으로 증가하는 반면, Mamba는 상태 공간 모델(State Space Model, SSM)을 기반으로 하여 추론 시 일정한(constant) 계산 비용을 유지한다는 강력한 장점이 있다.

그러나 저자들은 Mamba가 모든 작업에서 Transformer와 대등한 성능을 내면서 동시에 계산 비용을 절감할 수 있는지, 즉 "공짜 점심(Free Lunch)"이 항상 존재하는지에 대해 의문을 제기한다. 특히, 문맥 내에서 특정 정보를 정확히 추출하여 복제하는 **COPY 연산**과, 복잡한 추론 단계를 거쳐 답을 도출하는 **Chain-of-Thought (CoT) 추론** 능력을 중심으로 Mamba의 이론적, 실험적 한계를 규명하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Mamba의 추론 효율성(상수 크기의 상태 유지)과 특정 작업 수행 능력 사이의 트레이드-오프를 이론적으로 증명하고 실험적으로 검증한 것이다.

1.  **COPY 연산의 이론적 한계 분석**: Mamba를 선형 어텐션(Linear Attention)의 특수 형태로 재정의하고, 상수 크기의 Mamba 모델이 COPY 연산을 수행하기 위해서는 어텐션 스코어가 시퀀스 길이에 따라 지수적으로 증가해야 함을 증명하였다. 반면, Transformer는 로그 수준의 증가만으로도 이를 수행할 수 있다.
2.  **CoT 추론 및 DP 문제 분석**: CoT 추론 과정을 동적 계획법(Dynamic Programming, DP) 문제로 정식화하였다. 일반적인 DP 문제를 해결하기 위해 Mamba는 Transformer와 마찬가지로 모델 크기가 시퀀스 길이에 따라 선형적으로 증가해야 하며, 결과적으로 전체 추론 비용이 $O(T^2)$가 됨을 보였다.
3.  **m-locality DP의 효율성 제시**: 모든 DP 문제가 아닌, 국소성(locality)을 가진 $m$-locality DP 문제의 경우 Mamba가 상수 크기 $O(m)$만으로도 효율적으로 해결할 수 있음을 이론적으로 제시하였다.
4.  **실험적 검증**: COPY 작업과 LIS(Longest Increasing Subsequence), 산술 연산 등의 CoT 작업을 통해 Mamba가 Transformer보다 학습 속도가 느리고, 특히 시퀀스 길이가 길어질수록 성능 저하가 뚜렷함을 입증하였다.

## 📎 Related Works

본 논문은 다음과 같은 관련 연구들을 바탕으로 차별점을 둔다.

-   **SSM 및 Mamba**: Mamba는 하드웨어 최적화 알고리즘을 통해 선형 시간 복잡도를 달성하며 많은 시퀀스 모델링 작업에서 Transformer에 근접한 성능을 보인다.
-   **Mamba vs Transformer 비교**: 기존 연구(Jelassi et al., 2024; Park et al., 2024)들은 Mamba가 문맥 복제 및 정보 검색 작업에서 Transformer보다 열세임을 경험적으로 발견하였다. 본 논문은 이러한 현상을 단순한 관찰을 넘어 **수학적 증명(Theorem)**을 통해 이론적으로 뒷받침한다.
-   **CoT 및 DP 모델링**: CoT가 모델의 추론 능력을 향상시킨다는 점은 널리 알려져 있으며, 최근 연구들은 이를 DP 문제로 모델링하여 분석하고 있다. 본 논문은 이 프레임워크를 Mamba에 적용하여, Mamba가 CoT를 사용할 때 실제로 계산 비용 절감이 발생하는지 분석하였다.

## 🛠️ Methodology

### 1. Mamba의 재정의 (Linear Attention 관점)
저자들은 Mamba의 SSM 모듈을 다음과 같이 선형 어텐션과 유사한 형태로 재구성하여 분석한다.
$$y_i = \sum_{j=1}^{i} \alpha_j (c_i^T b_j) v_j$$
여기서 $v_j = \Delta_j \odot x_j$는 historical records(value), $b_j$는 key, $c_i$는 query에 해당하며, $\alpha_j = \prod_{k=j+1}^{i} a_k$는 누적 망각 계수(cumulative forgetting coefficient)이다. 이는 표준 Transformer의 어텐션과 달리, 과거 정보에 $\alpha_j \in [0, 1]$라는 가중치를 곱해 거리가 멀수록 정보를 잊어버리는 구조이다.

### 2. COPY 연산 분석
COPY 연산을 특정 위치 $pos(i)$의 값 $v_{pos(i)}$를 정확히 복원하는 작업으로 정의한다.
-   **상수 크기 Mamba (Theorem 1)**: $v_{pos(i)}$를 복원하기 위해서는 어텐션 스코어 $c_i^T b_{pos(i)}$가 거리 $L$에 대해 지수적으로 증가하는 하한선($c(1/a_{min})^{L-1} + d$)을 만족해야 한다. 이는 실질적으로 달성하기 매우 어렵다.
-   **상수 크기 Transformer (Theorem 2)**: Transformer는 동일 작업 수행 시 어텐션 스코어가 $\log L$ 수준으로만 증가해도 충분하다.
-   **선형 확장 Mamba (Theorem 3)**: Mamba의 모델 크기가 시퀀스 길이 $N$에 비례하여 $O(N)$으로 커지면, 모든 과거 정보를 저장할 수 있어 COPY 연산을 완벽히 수행할 수 있다. 하지만 이 경우 Mamba의 추론 비용 이점(상수 비용)은 사라진다.

### 3. CoT 및 DP 문제 분석
CoT 추론을 **입력 시퀀스 $\rightarrow$ 상태 공간 $\rightarrow$ 전이 함수 $\rightarrow$ 집계 함수**로 구성된 DP 문제로 정식화한다.
-   **일반 DP 해결 (Theorem 4)**: 임의의 DP 문제를 해결하려면 Mamba 레이어의 크기가 정답 길이 $T$에 비례하는 $O(T)$가 되어야 한다. 각 단계의 비용이 $O(T)$이므로 전체 비용은 $O(T^2)$가 되어 Transformer와 동일한 복잡도를 가진다.
-   **m-locality DP 해결 (Theorem 5)**: 현재 상태가 이전 $m$개의 결과에만 의존하는 $m$-locality DP의 경우, Mamba는 $O(m)$ 크기의 상수 모델만으로 해결 가능하며 전체 비용은 $O(mT)$로 절감된다.

## 📊 Results

### 1. COPY 작업 실험
-   **학습 속도 및 수렴**: Transformer(TF)는 Mamba보다 훨씬 빠르게 COPY 작업을 학습한다.
-   **모델 크기의 영향**: Mamba의 경우 레이어 수보다 Hidden size를 줄였을 때 성능 저하가 더 심하며, 특정 설정(Mamba-LD-69M)에서는 학습이 불안정하여 실패하였다.
-   **시퀀스 길이 확장**: 입력 길이 $N_{max}$가 증가함에 따라 Mamba는 학습에 필요한 데이터 양이 급격히 늘어나며, $N_{max}=40$에서는 학습에 실패하는 반면, Transformer는 안정적으로 학습을 완료하였다.

### 2. CoT 작업 실험 (LIS 및 산술 연산)
-   **Direct vs CoT**: 짧은 시퀀스의 Direct 작업에서는 Mamba가 우세한 경우도 있었으나, 시퀀스가 길어지는 CoT 설정에서는 Transformer가 일관되게 더 높은 정확도를 보였다.
-   **특이 사항**: Mamba는 CoT를 적용했을 때 오히려 Direct 설정보다 성능이 떨어지는 경향을 보였다. 이는 Mamba의 고정된 추론 용량이 긴 추론 체인을 처리하는 데 한계가 있음을 시사한다.
-   **모델 크기 대비 성능**: 동일한 파라미터 규모에서 Mamba는 Transformer보다 CoT 기반 DP 문제 해결 능력이 낮았다.

## 🧠 Insights & Discussion

본 논문은 Mamba의 효율성이 특정 작업에서는 **'표현 능력의 제한'**이라는 비용을 지불하고 얻어진 것임을 시사한다.

-   **강점**: $m$-locality와 같은 국소적 의존성을 가진 작업이나, 특정 패턴의 시퀀스 모델링에서는 Mamba가 Transformer보다 훨씬 효율적일 수 있다.
-   **한계**: 임의의 위치에서 정보를 정확히 추출해야 하는 Retrieval 작업이나, 전체 문맥을 참조하여 단계적으로 추론해야 하는 일반적인 DP/CoT 작업에서는 상수 크기의 상태(state)만으로는 부족하다. 이를 해결하기 위해 모델 크기를 키우면 결국 Transformer와 동일한 계산 비용을 지불하게 된다.
-   **비판적 해석**: Mamba가 Transformer의 완전한 대체제가 되기 위해서는, 단순히 선형 복잡도를 달성하는 것을 넘어, 필요한 경우에만 선택적으로 상태 크기를 확장하거나, Transformer의 어텐션 메커니즘이 가진 '전역적 참조 능력'을 효율적으로 모사할 수 있는 하이브리드 구조에 대한 연구가 필요하다.

## 📌 TL;DR

본 논문은 Mamba가 추론 비용을 상수로 유지하는 이점이 있지만, 이로 인해 **정확한 정보 복제(COPY)와 복잡한 추론(CoT/DP) 능력에서 한계**가 있음을 이론적으로 증명하고 실험적으로 확인하였다. 특히 일반적인 DP 문제를 해결하려면 Mamba 역시 Transformer와 동일한 $O(T^2)$의 비용을 소모해야 하며, 오직 국소적 특성($m$-locality)이 있을 때만 효율성이 유지된다. 이 연구는 향후 Mamba 기반 모델의 설계 시, 작업의 성격에 따라 상태 크기를 최적화하거나 하이브리드 아키텍처를 고려해야 함을 시사한다.