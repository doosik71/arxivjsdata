# Ground Metric Learning

Marco Cuturi, David Avis (2011)

## 🧩 Problem to Solve

본 논문은 정규화된 히스토그램(normalized histograms) 간의 유사도를 측정하기 위한 **Ground Metric Learning (GML)** 문제를 다룬다. 컴퓨터 비전, 자연어 처리 등 다양한 분야에서 객체는 특징(feature)들의 빈도수로 구성된 히스토그램으로 표현된다. 이러한 히스토그램을 비교하기 위해 주로 Transportation Distance(예: Earth Mover's Distance, EMD)가 사용된다.

Transportation Distance의 핵심 파라미터는 특징 공간 상의 두 점 사이의 거리를 정의하는 **Ground Metric**이다. 기존의 접근 방식은 전문가의 사전 지식(prior knowledge)에 의존하여 이 ground metric을 고정적으로 설정했다. 그러나 이러한 방식은 두 가지 한계가 있다. 첫째, 사전 지식이 없는 데이터셋에는 적용이 불가능하다. 둘째, 사전 지식이 있더라도 모든 학습 문제에 보편적으로 적용 가능한 '유니버설' ground metric은 존재하지 않으며, 문제의 특성에 맞게 적응적으로 선택되어야 한다. 따라서 본 논문의 목표는 레이블이 지정된 히스토그램 훈련 세트를 이용하여 ground metric을 직접 학습하는 알고리즘을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 ground metric 학습 문제를 **두 볼록 함수(convex functions)의 차이(Difference of Convex, DC)를 최소화하는 최적화 문제**로 정식화한 것이다. 

구체적으로, 유사한 히스토그램 사이의 transportation distance는 작게, 서로 다른 히스토그램 사이의 거리는 크게 만드는 목적 함수를 설계하고, 이를 거리 행렬(distance matrices)의 볼록 집합 위에서 최적화한다. 또한, 비볼록(non-convex) 최적화 문제의 한계를 극복하기 위해 'Typical Table'과 같은 대표성 있는 전송 계획(transportation plan)을 이용하여 효율적인 초기값($M_0$)을 설정하는 선형 근사 방법을 제안하였다.

## 📎 Related Works

논문에서는 히스토그램 비교 방법을 크게 두 가지로 분류하여 설명한다.

1.  **Bin-to-bin Distances**: 각 빈(bin)의 값을 독립적으로 비교하는 방식이다. Jensen-divergence, $\chi^2$, Hellinger distance 등이 이에 해당한다. 이러한 방식은 계산이 빠르지만, 특징 간의 유사성(예: '나달'과 '페더러'라는 단어가 서로 유사하다는 점)을 반영할 수 없다는 한계가 있다.
2.  **Cross-bin Distances**: 모든 빈의 쌍을 고려하는 방식이며, 대표적으로 Mahalanobis 거리 등이 있다. Mahalanobis Metric Learning(예: LMNN, ITML)은 선형 변환 행렬을 학습하여 거리를 최적화한다. 

본 논문이 제안하는 GML은 Mahalanobis 학습과 달리, transportation distance라는 메타-거리의 기반이 되는 ground metric 자체를 학습한다는 점에서 수학적 대상과 개념적 접근 방식이 완전히 다르다.

## 🛠️ Methodology

### 1. Transportation Distance의 정의
두 히스토그램 $r, c \in \Sigma^{d-1}$에 대해, ground metric $M$이 주어졌을 때의 transportation distance $d_M(r, c)$는 다음과 같이 정의된다.

$$d_M(r, c) = G_{rc}(M) = \min_{X \in U(r, c)} \langle M, X \rangle$$

여기서 $U(r, c)$는 행 합이 $r$이고 열 합이 $c$인 전송 계획(transportation plans)의 집합인 폴리토프(polytope)이며, $\langle M, X \rangle$는 프로베니우스 내적으로 $\sum_{i,j} M_{ij} X_{ij}$를 의미한다.

### 2. 학습 목적 함수 (Criterion)
훈련 세트의 히스토그램 쌍 $(r_i, r_j)$와 이들의 유사도를 나타내는 가중치 $\omega_{ij}$가 주어졌을 때, 다음과 같은 목적 함수 $C_k(M)$를 최소화한다.

$$C_k(M) = \sum_{i=1}^n (S_{ik}^+(M) + S_{ik}^-(M))$$

여기서 $S_{ik}^+$는 유사한 이웃($\omega_{ij} > 0$)들의 거리 합이고, $S_{ik}^-$는 서로 다른 이웃($\omega_{ij} < 0$)들의 거리 합이다. 즉, 유사한 것들은 가깝게, 다른 것들은 멀게 배치하도록 유도한다. 이때 $k$는 고려할 이웃의 수를 의미하며, $k=\infty$인 경우 모든 쌍을 고려한다.

### 3. DC (Difference of Convex) 최적화
함수 $G_{ij}(M)$은 $M$에 대해 선형 계획법의 최솟값이므로 오목 함수(concave function)이다. 따라서 $C_k(M)$는 다음과 같이 두 볼록 함수의 차이로 나타낼 수 있다.

$$C_k(M) = S_k^-(M) - (-S_k^+(M))$$

본 논문은 **Projected Subgradient Descent**와 오목 부분의 **국소 선형화(local linearization)**를 결합한 알고리즘을 사용하여 국소 최적해(local minima)를 찾는다. 구체적으로, 외곽 루프에서는 오목 함수를 1차 테일러 전개로 근사하여 볼록 함수로 만든 뒤, 내부 루프에서 이를 최소화하는 과정을 반복한다.

### 4. 초기값 설정 및 선형 근사
비볼록 최적화의 특성상 초기값 $M_0$가 매우 중요하다. 이를 위해 $G_{ij}(M)$을 내적 $\langle M, \Xi_{ij} \rangle$로 근사하여 선형 문제로 푼 뒤 초기값을 설정한다. $\Xi_{ij}$로 사용되는 대표 행렬(Representative Tables)은 다음과 같다.
- **Independence Table**: 최대 엔트로피 테이블로 계산이 매우 간단하다.
- **Typical Table**: 전송 폴리토프의 중심에 가까운 테이블로, Barvinok(2010)의 이론에 근거하여 independence table보다 더 정확한 근사를 제공한다.

## 📊 Results

### 실험 설정
- **데이터셋**: Caltech-256 이미지 데이터셋에서 무작위로 선택된 클래스들.
- **특징 추출**: GIST descriptors (차원 $d=128$)를 사용하여 정규화된 히스토그램으로 표현.
- **작업**: 이진 분류(Binary Classification).
- **비교 대상**: $l_1, l_2$, Hellinger distance, Mahalanobis 학습 기법(LMNN, ITML).
- **평가 지표**: $\kappa$-NN 분류 에러 및 테스트 세트에서의 Recall Accuracy (이웃 중 정답 클래스 비율).

### 주요 결과
1.  **성능 우위**: GML-EMD는 모든 비교 대상보다 우수한 성능을 보였다. 특히 $\kappa$-NN 분류 에러에서 가장 낮은 오차율을 기록하였다.
2.  **초기값의 영향**: Typical Table을 이용한 초기화가 Independence Table보다 성능이 좋았으며, 이는 알고리즘의 수렴 지점에 결정적인 영향을 미친다.
3.  **최적화의 효과**: 단순히 좋은 초기값 $M_0$를 사용하는 것보다, 제안된 subgradient descent 최적화를 거친 후의 성능 향상이 훨씬 컸다.
4.  **기존 거리의 한계**: $l_2$(Euclidean) 거리는 히스토그램 비교에 매우 부적절함이 확인되었으며, Mahalanobis 학습 또한 Hellinger 표현 공간에서 수행했을 때만 성능이 개선되었다.

## 🧠 Insights & Discussion

본 논문의 강점은 ground metric을 데이터로부터 직접 학습함으로써, 특징 간의 관계에 대한 사전 지식 없이도 데이터셋의 특성에 최적화된 거리 척도를 찾을 수 있다는 점이다. 특히 GIST와 같이 특징 간의 기하학적 관계를 사전에 정의하기 어려운 경우에 매우 유용하다.

다만, 계산 복잡도 측면에서 한계가 명확하다. 알고리즘의 핵심 루프 내에서 $O(n^2)$번의 최적 전송(Optimal Transport) 문제를 풀어야 하므로, 훈련 세트 $n$이 커질수록 계산 시간이 급격히 증가한다. 또한, 제안된 방법은 국소 최적해를 찾는 알고리즘이므로 전역 최적해(global optimum)를 보장하지 않는다.

비판적으로 해석하자면, 본 연구는 학습된 ground metric의 '구조'보다는 '값'에 집중했다. 최근 연구들에서 ground metric에 특정 구조(sparsity 등)를 부여하여 계산 속도를 높이는 방법들이 제안되었는데, 이를 GML 프레임워크에 통합한다면 실용성이 더욱 높아질 것이다.

## 📌 TL;DR

본 논문은 히스토그램 비교를 위한 Transportation Distance의 유일한 파라미터인 **Ground Metric을 데이터로부터 학습하는 GML(Ground Metric Learning)** 프레임워크를 제안한다. 학습 문제는 DC(Difference of Convex) 최적화 문제로 정식화되었으며, Typical Table을 이용한 효율적인 초기화 전략을 함께 제시한다. GIST 특징 기반의 이미지 분류 실험을 통해, 제안 방법이 기존의 bin-to-bin 거리나 Mahalanobis 학습 방식보다 훨씬 뛰어난 성능을 보임을 입증하였다. 이 연구는 사전 지식이 없는 복잡한 특징 공간에서의 히스토그램 비교 및 분류 성능을 획기적으로 높일 수 있는 가능성을 제시한다.