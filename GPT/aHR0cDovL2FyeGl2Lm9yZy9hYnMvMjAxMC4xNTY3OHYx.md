# On the Failure of the Smart Approach of the GPT Cryptosystem

Hervé Talé Kalachi (2020)

## 🧩 Problem to Solve

본 논문은 rank metric 기반의 암호 체계인 GPT(Gabidulin-Paramonov-Tretjakov) 암호 시스템의 보안 취약점, 그중에서도 특히 Rashwann et al. (2010)이 제안한 이른바 '스마트 접근 방식(smart approach)'의 실패를 증명하고 이를 공격하는 새로운 알고리즘을 제안하는 것을 목표로 한다.

코드 기반 암호 체계인 McEliece 시스템은 강력한 보안성을 제공하지만, 공개 키(public key)의 크기가 지나치게 크다는 단점이 있다. 이를 해결하기 위해 Hamming metric 대신 rank metric을 사용하는 GPT 암호 시스템이 제안되었다. 그러나 GPT 시스템의 기반이 되는 Gabidulin 코드는 구조적 특성이 매우 강하여, Overbeck(2008)이 Frobenius 연산자를 이용해 공개 키로부터 비밀 키를 복구하는 치명적인 구조적 공격(structural attack)을 성공시켰다. 이에 대한 방어책으로 Rashwann et al. (2010)은 왜곡 행렬(distortion matrix) $X$의 일부를 $q$-Vandermonde 행렬로 구성하는 '스마트 접근 방식'을 제안하여 Overbeck의 공격을 무력화하려 했다. 본 논문은 이 방어 기제가 여전히 취약하며, 특정 조건 하에서 공개 코드를 적절히 puncture(일부 열을 제거)함으로써 다시 Overbeck의 공격이 가능함을 보이고자 한다.

## ✨ Key Contributions

본 논문의 핵심적인 아이디어는 Rashwann et al. (2010)의 스마트 접근 방식이 결과적으로 표준 GPT 암호 시스템의 공개 코드에 일정한 '중복성(redundancies)'을 삽입한 것과 동일하다는 관점을 제시한 것이다.

저자는 공격자가 공개 코드에서 이 중복된 열들을 식별하여 제거할 수 있다면, 변형된 공개 키를 다시 표준 GPT 형태의 공개 키로 되돌릴 수 있음을 수학적으로 증명하였다. 즉, Frobenius 연산자를 통해 코드의 차원을 분석함으로써 중복 세트(redundancy set)를 찾아내고, 이를 제거한 후 Overbeck의 공격을 적용하여 다항 시간 내에 대체 비밀 키를 생성하는 파이프라인을 구축하였다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구와 그 한계를 설명한다.

1. **McEliece (1978):** Goppa 코드를 이용한 최초의 공개 키 암호 체계이나, 공개 키의 크기가 매우 크다는 치명적인 단점이 있다.
2. **GPT Cryptosystem (Gabidulin et al., 1991):** rank metric을 도입하여 키 크기를 줄였으나, Gabidulin 코드의 강한 구조적 특성으로 인해 Gibson(1995, 1996)과 Overbeck(2008)의 공격에 노출되었다.
3. **Overbeck's Attack (2008):** Frobenius 연산자 $\Lambda^i$를 적용했을 때 Gabidulin 코드의 차원이 선형적으로 증가하는 특성을 이용한다. 특히 공개 키 $G^{pub} = S(X|G)P$에서 $\Lambda^{n-k-1}(A)^\perp$의 codimension이 1이 되는 점을 이용해 비밀 키를 복구한다.
4. **The Smart Approach (Rashwann et al., 2010):** Overbeck의 공격을 막기 위해 왜곡 행렬 $X$를 $q$-Vandermonde 행렬 $X_1$과 랜덤 행렬 $X_2$의 결합으로 구성하여, $\dim \Lambda^{n-k-1}(C^{pub})^\perp \neq 1$이 되도록 설계하였다.
5. **Horlemann-Trautmann et al. (2017):** 스마트 접근 방식에 대한 또 다른 공격을 제안하였으며, $\Lambda^{t-a}(C^{pub})$ 내의 rank-one 요소들을 수집하여 $P$ 행렬을 복구하는 방식이다.

본 논문의 접근 방식은 Horlemann-Trautmann 등의 방식과 달리, 코드의 dual space 구조를 활용하여 중복성을 제거한 뒤 Overbeck의 공격을 재적용한다는 점에서 차별화된다.

## 🛠️ Methodology

### 1. 시스템 구조 및 정의

본 논문에서 다루는 스마트 접근 방식의 공개 키 $G^{pub}$는 다음과 같이 구성된다.
$$G^{pub} = S(X_1 | X_2 | G)P$$
여기서 $S \in GL_k(\mathbb{F}_{q^m})$는 가역 행렬, $P \in GL_{n+\ell}(\mathbb{F}_q)$는 열 교환 행렬, $G$는 Gabidulin 코드의 생성 행렬이다. 핵심은 $X = (X_1 | X_2)$에서 $X_1$이 $q$-Vandermonde 행렬이라는 점이다.

### 2. 핵심 공격 메커니즘: 중복성 제거

저자는 $G^{pub}$가 사실상 $w = a-s$개의 중복성을 가진 표준 GPT 시스템의 공개 키와 같음을 보였다. 여기서 $s$는 $|(X_1 | G)|$의 rank 값이다.

**중복 세트 식별 방법:**
공개 코드 $C^{pub}$에서 특정 열들을 제거한(puncturing) 코드를 $C^{pub}_J$라고 할 때, $J$가 중복 세트라면 Frobenius 연산자 $\Lambda^f$를 적용한 결과의 차원이 일정하게 유지된다.
$$\dim \Lambda^f(C^{pub}_J) = n + s + \ell - a$$
이 성질을 이용하여 공격자는 무작위로 열을 하나씩 제거하며 위 차원 조건이 만족되는지 확인함으로써 중복 세트 $I$를 찾아낼 수 있다.

### 3. 전체 공격 절차 (Algorithm 1)

1. **$s$ 값 추정:** $\text{rank}(\Lambda^{n+s-k}(G^{pub})) = \text{rank}(\Lambda^{n+s+1-k}(G^{pub}))$가 성립하지 않는 최소의 $s$를 찾는다.
2. **중복 세트 $J$ 탐색:**
    - $w = a - s$개의 열을 제거해야 한다.
    - 랜덤하게 열 $j$를 선택하여 $\dim (\Lambda^{n+s-k}(C^{pub}_j)) = y$ (단, $y = n+s+\ell-a$)인지 확인한다.
    - 조건을 만족하면 해당 열을 제거하고 $J$에 추가하며, 이를 $w$번 반복한다.
3. **Overbeck 공격 적용:** 중복성이 제거된 $C^{pub}$의 생성 행렬 $G^{pub}_{new}$에 대하여, $f = n+s-k-1$ 값을 사용하여 Overbeck의 알고리즘을 적용해 대체 비밀 키를 복구한다.

## 📊 Results

### 1. 실험 설정

- **도구:** Magma V2.21-6 소프트웨어 사용.
- **대상:** Rashwann et al. (2010)이 제안한 파라미터 설정(예: $am > 60$)을 적용한 스마트 접근 방식의 GPT 시스템.

### 2. 정량적 결과 및 복잡도

- **계산 복잡도:** 전체 공격의 시간 복잡도는 $O(k^3(n+a+1-k)^3)$의 $\mathbb{F}_{q^m}$ 연산 횟수를 가진다. 이는 다항 시간 내에 수행 가능하다.
- **실행 시간:** 실제 구현 결과, 모든 테스트 케이스에서 비밀 키를 복구하는 데 **5초 미만**의 시간이 소요되었다.

### 3. 비교 분석

Horlemann-Trautmann et al. (2017)의 공격과 비교했을 때, $m$과 $n$이 비슷할 때는 기존 공격의 비용이 낮을 수 있으나, $m$이 $n$에 비해 매우 큰 경우에는 본 논문에서 제안한 공격의 효율성이 더 높다는 것을 확인하였다.

## 🧠 Insights & Discussion

본 논문은 암호학적 방어 기제가 단순한 파라미터 변경이나 구조적 추가(중복성 삽입)만으로는 충분하지 않음을 보여준다. 특히 Frobenius 연산자라는 Gabidulin 코드의 강력한 대수적 특성이 변형된 시스템에서도 여전히 유효하게 작용한다는 점이 인상적이다.

**강점:**

- 기존의 복잡한 rank-one 요소 수집 방식보다 직관적인 '중복성 제거 $\rightarrow$ 기존 공격 재적용' 전략을 취함으로써 공격의 단순성과 효율성을 높였다.
- 이론적 증명과 실제 구현(Magma)을 통해 공격의 유효성을 완벽히 입증하였다.

**한계 및 논의사항:**

- 본 공격은 $X_2$가 랜덤 행렬이라는 가정 하에 높은 확률로 성공한다. 만약 $X_2$ 또한 특수한 구조를 가진다면 공격의 성공 확률이나 복잡도가 달라질 가능성이 있다.
- 논문에서는 $s$ 값을 찾는 과정이 효율적임을 보였으나, $s$의 범위가 매우 넓어질 경우 탐색 시간이 증가할 수 있다.

## 📌 TL;DR

본 논문은 GPT 암호 시스템의 Overbeck 공격을 방어하기 위해 제안된 '스마트 접근 방식(smart approach)'이 사실상 표준 GPT에 중복 열을 추가한 것에 불과함을 밝혀냈다. 저자는 Frobenius 연산자를 이용해 이 중복 열들을 식별하고 제거함으로써, 다시 Overbeck의 공격을 적용해 비밀 키를 복구하는 새로운 알고리즘을 제안하였다. 실험 결과 5초 이내에 키 복구가 가능함을 보여주었으며, 이는 스마트 접근 방식이 보안상 안전하지 않음을 시사한다. 이 연구는 향후 rank metric 기반 암호 설계 시 단순한 구조적 은닉보다는 근본적인 대수적 특성을 고려한 설계가 필요함을 시사한다.
