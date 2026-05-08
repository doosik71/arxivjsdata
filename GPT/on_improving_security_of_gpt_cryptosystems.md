# On improving security of GPT cryptosystems

Ernst M. Gabidulin, Haitham Rashwan, Bahram Honary (2010)

## 🧩 Problem to Solve

본 논문은 Rank error-correcting codes(랭크 오류 정정 코드)를 기반으로 한 공공키 암호 시스템인 GPT(Gabidulin-Paramonov-Tretjakov) 암호 시스템의 보안성을 향상시키는 문제를 다룬다.

GPT 암호 시스템은 Hamming metric 기반의 Goppa 코드(McEliece 시스템)에 비해 공공키의 크기를 줄일 수 있다는 장점이 있어 실용적인 가치가 높다. 그러나 Rank 코드의 정형화된 구조로 인해 Gibson의 공격과 더 최근의 Overbeck의 공격과 같은 구조적 공격에 취약하다는 치명적인 문제가 존재한다. 특히 Overbeck의 공격은 기존의 여러 변형된 GPT 시스템마저 무력화시키며 강력한 공격 성능을 보였다.

따라서 본 논문의 목표는 Overbeck의 공격을 효과적으로 방어하면서도, 정당한 수신자가 복호화를 수행하는 데 지장이 없는 새로운 Column Scrambler의 설계 방안을 제시하고 이를 다양한 GPT 변형 모델에 적용하여 보안성을 검증하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Column Scrambler matrix $P$를 기저 필드(Base field $\mathbb{F}_q$)가 아닌 확장 필드(Extension field $\mathbb{F}_{q^n}$) 상에서 정의**하는 것이다.

Overbeck의 공격은 Column Scrambler $P$가 기저 필드 $\mathbb{F}_q$ 상에서 정의되었을 때, Frobenius automorphism $\sigma$에 대해 $\sigma(P) = P$가 성립한다는 성질을 이용한다. 공격자는 이를 통해 공공키로부터 확장된 공공키(Extended public key)를 구성하고, 최종적으로 Rank 코드의 검사 행렬(Check matrix $H$)의 첫 번째 행을 찾아내어 시스템을 붕괴시킨다.

저자들은 $P$를 확장 필드 $\mathbb{F}_{q^n}$ 상에서 적절하게 선택하면 $\sigma(P) \neq P$가 되어 Overbeck의 공격 및 Gibson의 공격을 원천적으로 차단할 수 있음을 제안한다. 또한, 확장 필드 상의 $P$를 사용하더라도 복호화 과정에서 발생하는 오류의 랭크가 정정 가능 범위 내에 있도록 하는 $P^{-1}$의 특수 구조를 설계하여 시스템의 기능성을 유지하였다.

## 📎 Related Works

본 논문은 다음과 같은 관련 연구들을 언급하고 한계를 지적한다.

1. **McEliece 암호 시스템**: Hamming metric의 Goppa 코드를 사용한 최초의 코드 기반 공공키 시스템이다. 보안성은 강력하지만 공공키의 크기가 너무 커서 실용적인 구현에 한계가 있다.
2. **GPT 암호 시스템**: Rank 코드를 사용하여 키 크기를 줄인 시스템이다. 하지만 Rank 코드의 강한 구조적 특성 때문에 Gibson의 공격에 노출되었다.
3. **GPT의 초기 변형 모델들**: rectangular row scramble matrix 사용, MRD(Maximum Rank Distance) 코드의 수정, Reducible codes 도입 등을 통해 Gibson의 공격을 방어하려 했다. 그러나 이러한 시도들은 Overbeck의 더 강력한 공격에 의해 다시 무너졌다.
4. **일반 랭크 디코딩 알고리즘**: Ourivski–Johansson 및 Levy-dit-Vehel 등이 제안한 일반적인 랭크 오류 정정 알고리즘들이 존재한다. 하지만 본 논문에서 분석한 바에 따르면, 이들의 시간 복잡도는 실용적인 파라미터 설정 하에서 브루트-포스 공격보다 훨씬 더 많은 연산을 요구하므로 현실적인 위협이 되지 않는다.

## 🛠️ Methodology

### 1. 기본 정의 및 MRD 코드

Rank metric에서 벡터 $x$의 Rank norm $\text{Rk}(x|\mathbb{F}_q)$는 $\mathbb{F}_q$ 상에서 선형 독립인 좌표의 최대 개수로 정의된다. MRD 코드의 생성 행렬 $G_k$는 다음과 같이 정의된다.

$$G_k = \begin{bmatrix} g_1 & g_2 & \cdots & g_n \\ g_1^{[1]} & g_2^{[1]} & \cdots & g_n^{[1]} \\ \vdots & \vdots & \ddots & \vdots \\ g_1^{[k-1]} & g_2^{[k-1]} & \cdots & g_n^{[k-1]} \end{bmatrix}$$

여기서 $g_i \in \mathbb{F}_{q^n}$이며, $g^{[i]} := g^{q^i \pmod n}$는 $i$번째 Frobenius power를 의미한다.

### 2. GPT 암호 시스템의 구조

공공키 $G_{pub}$는 다음과 같은 형태로 구성된다.

$$G_{pub} = S G_k P \quad \text{또는} \quad G_{pub} = S [X \mid G_k] P$$

- $S$: Row scrambling matrix (행의 구조를 섞어 구조를 숨김)
- $G_k$: MRD 코드의 생성 행렬
- $P$: Column scrambler (열을 섞어 구조를 파괴함)
- $X$: Distortion source matrix (추가적인 노이즈를 통해 보안성 강화)

**암호화 과정**: 평문 $m$에 대해 암호문 $c$는 다음과 같다.
$$c = m G_{pub} + e$$
여기서 $e$는 랭크가 $t_2$ 이하인 인위적인 오류 벡터이다.

**복호화 과정**: 수신자는 $P^{-1}$를 곱하여 중간 암호문을 얻는다.
$$c' = c P^{-1} = m S [X \mid G_k] + e P^{-1}$$
이후 $m S$를 디코딩 알고리즘으로 추출하고, $S^{-1}$를 곱해 평문 $m$을 복구한다.

### 3. 보안 향상을 위한 $P$의 설계

Overbeck의 공격을 막기 위해 $P$를 $\mathbb{F}_{q^n}$ 상에서 선택한다. 이때 핵심은 복호화 시 $e P^{-1}$의 랭크가 MRD 코드의 정정 능력 $t = \lfloor \frac{n-k}{2} \rfloor$를 넘지 않아야 한다는 점이다.

저자들은 $P^{-1}$를 다음과 같은 구조로 설계할 것을 제안한다.
$$P^{-1} = [Q_1 \mid Q_2]$$

- $Q_1$: 크기가 $n \times (t-t_1)$인 행렬이며, 원소들이 확장 필드 $\mathbb{F}_{q^n}$에 속한다.
- $Q_2$: 크기가 $n \times (n-t+t_1)$인 행렬이며, 원소들이 기저 필드 $\mathbb{F}_q$에 속한다.

이때 $e$의 랭크가 $t_1$이라고 가정하면, $e P^{-1} = [e Q_1 \mid e Q_2]$가 된다. $e Q_1$의 랭크는 최대 $t-t_1$이고, $e Q_2$의 랭크는 최대 $t_1$이므로, 전체 랭크 $\text{Rk}(e P^{-1} | \mathbb{F}_q) \le (t-t_1) + t_1 = t$가 되어 안전하게 복호화가 가능하다.

## 📊 Results

### 1. 이론적 분석 및 복잡도

논문은 일반적인 랭크 디코딩 알고리즘의 복잡도를 분석하여, 실용적인 파라미터($n=28, k=14, q=2, t=7$)에서 공격에 필요한 연산량이 $2^{113} \sim 2^{302}$ 수준임을 보였다. 이는 사실상 계산적으로 불가능한(infeasible) 수준이다.

### 2. 파라미터 설정에 따른 보안성 평가

$(28, 14)$ Rank 코드를 기준으로, 인위적 오류의 랭크 $t_1$과 $P$의 필드 선택에 따른 보안 상태를 분석하였다.

- $t_1 = 0 \sim 2$: $P$가 확장 필드에 있더라도 브루트-포스 공격에 취약하여 **불안전(Insecure)**.
- $t_1 = 7$ 및 $P \in \mathbb{F}_q$: Gibson-Overbeck 공격에 취약하여 **불안전(Insecure)**.
- $t_1 = 3 \sim 6$ 및 $P \in \mathbb{F}_{q^n}$: 알려진 모든 공격(구조적 공격 및 브루트-포스)에 대해 내성을 가져 **안전(Secure)**.

결과적으로 저자들은 $t_1 = 3$ 또는 $4$의 값을 사용할 것을 권장한다.

## 🧠 Insights & Discussion

본 논문의 강점은 기존 GPT 시스템의 치명적인 약점이었던 $\sigma(P)=P$ 성질을 확장 필드 도입이라는 단순하면서도 강력한 방법으로 해결했다는 점이다. 특히, 단순히 필드를 확장하는 것에 그치지 않고 $P^{-1}$의 구조를 $[Q_1 \mid Q_2]$ 형태로 설계하여, 확장 필드 사용 시 발생할 수 있는 '복호화 불가능 문제(오류 랭크 증가)'를 수학적으로 해결하고 증명한 점이 돋보인다.

다만, 본 논문은 특정 파라미터 예시를 통해 보안성을 논의하고 있으나, 더 광범위한 파라미터 범위에 대한 정량적인 보안 강도 분석이 부족하다는 한계가 있다. 또한, $P$를 확장 필드에서 선택함에 따라 발생하는 연산 복잡도의 증가량에 대한 구체적인 분석이 명시되지 않았다.

결론적으로, 이 연구는 코드 기반 암호 시스템에서 스크램블러 행렬의 필드 선택이 시스템의 전체 보안 구조를 결정짓는 핵심 요소임을 시사한다.

## 📌 TL;DR

본 논문은 GPT 암호 시스템이 Overbeck의 구조적 공격에 취약한 이유가 Column Scrambler $P$가 기저 필드 $\mathbb{F}_q$ 상에서 정의되어 $\sigma(P)=P$가 성립하기 때문임을 밝히고, 이를 해결하기 위해 **$P$를 확장 필드 $\mathbb{F}_{q^n}$ 상에서 정의하는 방법**을 제안한다. 특히 $P^{-1}$의 일부를 기저 필드 상의 행렬로 구성하는 설계를 통해 복호화 가능성을 보장하면서도 보안성을 획기적으로 높였다. 이 연구는 향후 Rank 코드 기반의 보안 통신 시스템 설계 시 키 생성 단계에서 필드 선택의 중요성에 대한 가이드라인을 제공한다.
