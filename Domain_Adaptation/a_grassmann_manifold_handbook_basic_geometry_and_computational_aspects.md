# A Grassmann Manifold Handbook: Basic Geometry and Computational Aspects

Thomas Bendokat, Ralf Zimmermann, and P.-A. Absil

## 🧩 Problem to Solve

이 논문은 고정된 차원의 선형 부분 공간(linear subspaces)으로 구성된 Grassmann 다형체(Grassmann manifold, Gr($n,p$))의 기본 기하학 및 계산적 측면에 대한 필수적인 사실과 공식을 수집하여 제공하는 것을 목표로 합니다. 특히, 기계 학습, 컴퓨터 비전, 저랭크 행렬 최적화 등 다양한 응용 분야에서 행렬 기반 알고리즘을 사용하여 이러한 문제를 해결하는 데 적합한 형태로 제시합니다. 또한, 부분 공간을 직교 투영자(orthogonal projectors)로 표현하는 방식과 직교군(orthogonal group)의 몫공간(quotient space)으로 간주하여 (직교) 기저의 동치류로 식별하는 두 가지 주요 연구 흐름을 연결하여 쉽게 전환할 수 있도록 합니다.

## ✨ Key Contributions

- **수정된 리만 로그맵 알고리즘 제안:** Grassmannian에서 리만 로그맵(Riemannian logarithm map)을 계산하기 위한 수정된 알고리즘(Algorithm 5.3)을 제시합니다. 이 알고리즘은 수치적으로 이점을 가지며, Cut Locus의 점들을 탄젠트 공간으로 (비고유하게) 매핑할 수 있게 하여 임의의 두 점 사이의 (복수 개의) 최단 곡선과 해당 탄젠트 벡터에 대한 명시적인 공식을 제공합니다.
- **Cut Locus 및 Conjugate Locus에 대한 상세한 설명:** 부분 공간 사이의 주각(principal angles) 관점에서 Cut Locus 및 Conjugate Locus에 대한 더 근본적이고 완전한 설명을 제공합니다(Theorem 7.2).
- **평행 이동(Parallel Transport) 공식 유도:** 직교 투영자 관점에서 측지선을 따른 평행 이동 공식을 유도합니다(Proposition 3.5).
- **지수 맵(Exponential Map)의 도함수 공식 유도:** 지수 맵의 도함수에 대한 공식을 제시합니다.
- **한 점에서 소멸하는 야코비 필드(Jacobi fields) 공식 유도:** 한 점에서 소멸하는 야코비 필드에 대한 공식을 유도합니다(Proposition 7.1).
- **포괄적인 핸드북:** Lie 군, ONB(Orthogonal Normal Basis), 투영자 관점 등 다양한 접근 방식을 통합하여 Grassmann 다형체의 리만 기하학적 측면을 매트릭스 공식으로 정리하여 계산 알고리즘에 즉시 활용할 수 있도록 합니다. 특히 $O(np^2)$의 계산 복잡도를 가진 공식에 중점을 둡니다.

## 📎 Related Works

- **Grassmann 다형체의 응용:** 데이터 분석 및 신호 처리 [24, 51, 52], 부분 공간 추정 및 추적 [16, 9, 65], 구조화된 행렬 최적화 문제 [21, 2, 3], 동적 저랭크 분해 [28, 37], 투영 기반 매개변수 모델 축소 [8, 47, 67, 48, 68], 컴퓨터 비전 [45].
- **Grassmann 다형체의 수학적 연구:** [42, 60, 61, 62, 54, 44, 38].
- **Grassmann 다형체의 표현 방식:**
  - **ONB 관점:** [21]
  - **투영자 관점:** [44, 32, 10, 33]
  - **Lie 군 관점:** [24, 56]
  - **플뤼커 임베딩(Plücker embeddings) 및 슈베르트 다양체(Schubert varieties):** 대수 기하학적 관점 [25].
  - **복소 Grassmannian:** [44, 11]
- **리만 기하학적 개념의 선행 연구:**
  - **측정(Metric) 및 지수 맵:** Stiefel 관점 [21, 2], 투영자 관점 [10].
  - **리만 접속(Riemannian connection)의 수평 리프트(horizontal lift):** [2].
  - **평행 이동:** ONB 관점 [21].
  - **Cut Locus:** [60, 54] (정의), [55] (일반 이론).
  - **리만 로그맵:** Stiefel 관점 [2, 섹션 3.8], 투영자 관점 [10, Theorem 3.3], CS 분해를 이용한 방법 [24].
  - **야코비 필드 및 Conjugate Locus:** [61] (불완전), [54, 11] (복소수 케이스).

## 🛠️ Methodology

논문은 Grassmann 다형체에 대한 세 가지 주요 접근 방식(Lie 군, ONB, 투영자 관점)을 통합하고, 다음 리만 기하학적 개념에 대한 행렬 기반 공식을 제공합니다:

- **다형체 구조 및 탄젠트 공간:**
  - Grassmann 다형체 Gr($n,p$)를 $p$차원 부분 공간에 대한 직교 투영자 집합으로 정의하고, $O(n)/(O(p) \times O(n-p))$ 또는 $St(n,p)/O(p)$와 같은 몫공간 구조를 통해 매끄러운 다양체임을 입증합니다.
  - 탄젠트 공간 $T_P \text{Gr}(n,p)$를 수직 및 수평 부분으로 분해하여 행렬 형태로 표현합니다. (Prop. 2.1)
- **리만 측정(Riemannian Metric) 및 기울기(Gradient):**
  - 몫공간 구조에서 유도된 리만 측정은 유클리드 측정의 절반인 $g^{\text{Gr}}_P(\Delta_1, \Delta_2) = \frac{1}{2} \text{tr}(\Delta_1 \Delta_2)$임을 보여줍니다.
  - 함수의 기울기는 유클리드 기울기를 탄젠트 공간으로 투영하여 계산합니다.
- **리만 접속(Riemannian Connection):**
  - 내장된 부분다형체(embedded submanifold)의 경우, 주변 다형체의 Levi-Civita 접속을 탄젠트 공간에 투영하여 리만 접속을 얻습니다. (Prop. 3.1)
- **지수 맵(Exponential Map):**
  - **투영자 관점:** $\text{Exp}_{\text{Gr}, P}(\Delta) = \exp_m([\Delta, P]) P \exp_m(-[\Delta, P])$로 계산됩니다. (Prop. 3.2)
  - **ONB 관점:** $\Delta^{\text{hor}}_U$의 SVD를 활용하여 $\text{Exp}_{\text{Gr}, P}(t\Delta) = [UV\cos(t\Sigma)V^T + \hat{Q}\sin(t\Sigma)V^T + UV_{\perp}V^T_{\perp}]$로 계산됩니다. (Prop. 3.3)
- **지수 맵의 도함수:**
  - SVD 요소의 미분을 사용하여 (특이값이 서로 다르고 0이 아닌 경우) 지수 맵의 도함수를 계산하는 공식을 제공합니다. (Prop. 3.4)
  - QR 분해의 미분을 활용한 대안적인 방법도 제시합니다.
- **평행 이동:**
  - **투영자 관점:** $P_{\Delta}(\text{Exp}_{\text{Gr}, P}(t\Gamma)) = \exp_m(t[\Gamma, P]) \Delta \exp_m(-t[\Gamma, P])$로 계산됩니다. (Prop. 3.5)
  - **ONB 관점:** $\Gamma^{\text{hor}}_U$의 SVD를 사용하여 $O(np^2)$ 복잡도로 계산할 수 있는 공식을 제시합니다.
- **대칭 공간 구조 및 단면 곡률(Sectional Curvature):**
  - Grassmannian이 대칭 공간임을 직접적으로 구성하여 보이고, 이를 통해 단면 곡률 $K_P(\Delta_1, \Delta_2) = 4 \frac{\text{tr}(\Delta_1^2 \Delta_2^2) - \text{tr}((\Delta_1 \Delta_2)^2)}{\text{tr}(\Delta_1^2) \text{tr}(\Delta_2^2) - (\text{tr}(\Delta_1 \Delta_2))^2}$을 유도합니다. (Prop. 4.1)
- **Cut Locus 및 리만 로그맵:**
  - **Cut Locus 정의:** $P$의 Cut Locus는 $P$에서 시작하는 측지선이 더 이상 길이를 최소화하지 않는 점들로 정의됩니다. (식 5.1)
  - **Algorithm 5.3 (확장된 Grassmann 로그맵):** $P=UU^T$와 $F=YY^T$가 주어졌을 때, $Y^TU$의 SVD를 통해 $Y$를 $U$에 Procrustes 정렬하고, $(I_n - UU^T)Y^*$의 SVD와 $\arcsin$ 함수를 사용하여 $\Delta^{\text{hor}}_U$를 계산합니다. 이 알고리즘은 Cut Locus의 점들도 처리할 수 있습니다. (Thm. 5.4, 5.5)
- **야코비 필드 및 Conjugate Locus:**
  - **야코비 필드:** 측지선을 따라가는 벡터 필드이며, 지수 맵의 도함수를 사용하여 계산됩니다. (Prop. 7.1)
  - **Conjugate Locus:** $P$의 Conjugate Locus는 $P$와 $F$ 사이의 (반드시 최소화되지 않는) 측지선을 따라 $P$와 $F$에서 소멸하는 0이 아닌 야코비 필드가 존재하는 $F$로 구성됩니다. (Thm. 7.2)

## 📊 Results

- **리만 로그맵의 수치적 성능:** Algorithm 5.3(새로운 로그 알고리즘)은 Cut Locus에 가까워질수록 기존 알고리즘에 비해 월등히 뛰어난 수치적 정확도를 보였습니다. (Figure 5.2) 기존 알고리즘은 Cut Locus 근처에서 오류가 급격히 증가했습니다.
- **Cut Locus의 특성화:** 점 $P=UU^T$의 Cut Locus는 $P$와 $F=YY^T$ 사이의 주각 중 적어도 하나가 $\pi/2$인 모든 $F \in \text{Gr}(n,p)$로 구성됩니다. 이는 $\text{rank}(U^T Y) < p$와 동치입니다.
- **측지선 길이 및 고유성:**
  - $\Delta^{\text{hor}}_U$의 가장 큰 특이값 $\sigma_1 < \pi/2$인 경우, 측지선은 고유하며 최소화됩니다.
  - $\sigma_1 = \pi/2$인 경우, 측지선은 최소화되지만 고유하지 않습니다.
  - $\sigma_1 > \pi/2$인 경우, 측지선은 최소화되지 않습니다.
- **Injectivity Radius:** Gr($n,p$)의 모든 점 $P$에서의 Injectivity Radius는 $\pi/2$입니다.
- **Conjugate Locus의 특성화:** $p \leq n/2$일 때, $P$의 Conjugate Locus는 $P$와 $F$ 사이의 주각 중 적어도 두 개가 같거나, $p < n/2$인 경우 적어도 하나가 0인 모든 $F \in \text{Gr}(n,p)$로 구성됩니다.
  - Cut Locus와 Conjugate Locus는 서로의 부분집합이 아니지만, 최소화 측지선을 따라 Conjugate인 점들은 Cut Locus에도 속합니다.
- **계산 복잡도:** Grassmannian의 주요 연산(리만 측정, 기울기, 지수 맵, 평행 이동, 리만 로그맵)의 계산 복잡도가 $O(np^2)$임을 분석하여 제시합니다. (Table C.1)

## 🧠 Insights & Discussion

- **다양한 관점의 통합:** 이 논문은 Grassmann 다형체에 대한 Lie 군, ONB, 투영자 관점을 성공적으로 통합하여 각 관점 간의 전환을 용이하게 하고, 이론과 계산 모두에 있어 견고한 기반을 제공합니다.
- **알고리즘적 효율성:** 제안된 행렬 기반 공식들은 알고리즘 구현에 최적화되어 있으며, 특히 $n \gg p$인 시나리오에서 $O(np^2)$의 계산 복잡도를 달성하여 대규모 응용 분야에서 실용적인 효율성을 제공합니다.
- **리만 로그맵의 혁신:** Cut Locus에 대한 개선된 이해를 바탕으로 한 새로운 리만 로그맵 알고리즘은 수치적 안정성을 크게 향상시키고, Cut Locus 내의 점들까지도 처리하여 복수 개의 최단 측지선을 명시적으로 계산할 수 있게 합니다. 이는 리만 평균(Riemannian barycenter) 계산과 같은 최적화 문제에서 "거의 기울기(almost gradients)"를 계산하는 데 매우 중요합니다.
- **Cut Locus 및 Conjugate Locus에 대한 심층 분석:** 이전 문헌에서 부족했던 Cut Locus 및 Conjugate Locus에 대한 더 완전하고 명시적인 기하학적 설명을 제공하여, 이러한 개념이 포함된 곡선 피팅, 측지 회귀(geodesic regression) 등의 문제를 더 정확하게 분석하고 해결할 수 있도록 합니다.
- **실용적인 가치:** 이 핸드북은 Grassmann 다형체에서 리만 연산을 수행하는 데 필요한 핵심 도구와 빌딩 블록을 제공하며, 이론적 분석과 데이터 처리 문제 해결에 즉시 적용될 수 있습니다.

## 📌 TL;DR

- **문제:** Grassmann 다형체의 기하학을 행렬 기반 알고리즘에 적합하도록 통합하고, 다양한 표현 방식(투영자, ONB, Lie 군) 간의 간극을 연결하며, 특히 Cut Locus와 같은 도전적인 기하학적 영역에서 계산 효율성과 정확도를 개선하는 것이 필요.
- **방법:** Grassmann 다형체의 몫공간 구조와 대칭 공간 특성을 활용하여 리만 기하학의 핵심 개념들(측정, 측지선, 로그맵, 평행 이동, 곡률)을 행렬 기반 공식으로 체계화. 특히, SVD의 모호성을 활용하여 Cut Locus를 포함한 모든 지점 간의 리만 로그맵을 계산하는 수정된 알고리즘(Algorithm 5.3)을 제안하고, 주각을 통해 Cut Locus와 Conjugate Locus를 상세히 설명.
- **주요 발견:** 새로운 리만 로그맵 알고리즘은 Cut Locus 근처에서 뛰어난 수치적 안정성을 제공하며, Cut Locus의 점들까지도 매핑하여 가능한 모든 최단 측지선을 명시적으로 계산할 수 있게 함. 투영자 관점에서의 평행 이동 및 지수 맵 도함수 등 중요한 공식들을 유도하고, 주각을 통한 Cut Locus 및 Conjugate Locus의 더 완전한 설명을 제시. 또한, 주요 연산의 계산 복잡도를 $O(np^2)$로 분석하여 대규모 문제에 대한 실용적 효율성을 입증.
