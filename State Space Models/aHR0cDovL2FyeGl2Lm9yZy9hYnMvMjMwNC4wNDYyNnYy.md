# On the state-space model of unawareness

Alex A.T. Rathke (2023)

## 🧩 Problem to Solve

본 논문은 인식론적 상태를 모델링하는 표준 상태-공간 모델(standard state-space model)에서 발생하는 '무의식(unawareness)'의 논리적 모순을 해결하고자 한다.

전통적인 지식 모델에서는 '필연성(necessitation)'이라는 성질, 즉 전체 상태 공간 $\Omega$에 대해 지식 연산자를 적용하면 다시 전체 집합이 된다는 $K\Omega = \Omega$라는 가정을 기본적으로 채택한다. 그러나 Dekel, Lipman, Rustichini(DLR)의 연구에 따르면, 이러한 필연성 가정을 유지할 경우 '무의식'을 가진 에이전트가 실제로는 모든 것을 인지하고 있어야 한다는 모순된 결과가 도출된다. 즉, 표준 모델 내에서 비자명한 무의식(non-trivial unawareness)이 존재한다면 모델 자체가 불일치(inconsistent)하게 된다는 점이 문제의 핵심이다.

논문의 목표는 필연성 성질과 무의식 모델 사이의 충돌을 분석하고, 이를 해결하여 모델의 일관성을 유지할 수 있는 일반화된 지식 연산자(generalized knowledge operator)를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 비자명한 무의식이 존재할 때 표준적인 '필연성(necessitation)' 성질이 반드시 위배됨을 증명하고, 이를 수용하는 새로운 지식 연산자 $K'$를 정의하여 DLR의 모순을 해결한 것이다.

저자는 에이전트가 특정 사건 $E$에 대해 무의식 상태에 있다면, 그 에이전트는 자신이 무엇을 모르는지 정확히 알 수 없으므로 '부정적 내성(negative introspection)'이 작동하지 않으며, 결과적으로 필연성 성질을 그대로 적용할 수 없다는 직관을 제시한다. 이를 바탕으로 지식 집합에서 무의식 집합을 제외하는 방식의 일반화된 연산자를 도입하여, 모델의 일관성을 해치지 않으면서도 무의식의 특성을 보존하는 'R-필연성(R necessitation)' 개념을 정립하였다.

## 📎 Related Works

본 논문은 지식과 무의식을 모델링하는 기존의 두 가지 주요 접근 방식을 검토한다.

1. **양상 논리(Modal Logic) 및 집합론적 접근**: Aumann(1976)의 공통 지식(common knowledge) 개념과 Bacharach(1985)의 상태-공간 모델이 기초가 된다. 특히 Geanakoplos(1989)는 분할되지 않은(non-partitional) 가능성 대응(possibility correspondence)을 통해 무의식을 모델링할 수 있는 이론적 구조를 제공하였다.
2. **DLR의 모순 및 그 이후의 연구**: Dekel, Lipman, Rustichini(1998)는 표준 모델의 성질들을 결합하면 무의식 상태의 에이전트가 결국 모든 것을 인지하게 된다는 모순을 증명하였다. 이를 해결하기 위해 Heifetz et al.(2006)이나 Li(2009) 등은 여러 개의 상태 공간(multiple state-spaces)을 도입하거나 '약한 필연성(weak necessitation)' 또는 '주관적 필연성(subjective necessitation)'과 같은 제한적인 성질을 제안하였다.

본 논문은 여러 개의 상태 공간을 도입하는 복잡한 방식 대신, 단일 표준 상태-공간 모델 내에서 지식 연산자를 수정함으로써 동일한 문제를 해결하려 한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. 기본 정의

- **상태 공간 및 가능성 대응**: 유한한 상태 집합 $\Omega$와 각 상태 $s \in \Omega$를 가능한 상태들의 부분집합으로 매핑하는 $\text{P}: \Omega \to 2^\Omega$가 존재한다.
- **표준 지식 연산자**: 사건 $E \subseteq \Omega$에 대해, 에이전트가 $E$를 안다는 것은 $\text{P}(s) \subseteq E$인 상태들의 집합으로 정의된다.
  $$K(E) := \{s \in \Omega \mid \text{P}(s) \subseteq E\}$$
- **무의식 연산자**: 사건 $E$에 대해 에이전트가 완전히 무의식 상태에 있는 집합 $U(E)$는 지식 연산자의 보수 $\neg K$를 무한히 반복 적용한 교집합으로 정의된다.
  $$U(E) := \bigcap_{i=1}^{\infty} (\neg K)^i(E)$$

### 2. 필연성 위배 증명 (Theorem 1)

저자는 비자명한 무의식($U E \neq \emptyset$)이 존재하면, 표준 필연성 $K\Omega = \Omega$가 성립할 수 없음을 증명한다.

- 부정적 내성(negative introspection, $\neg KE \subseteq K\neg KE$)은 무의식이 존재할 때 붕괴한다.
- 증명 과정에서 부정적 내성이 성립하면 $\Omega \subseteq K\Omega$가 되어 필연성이 도출되지만, 무의식이 존재하면 이 전제가 깨지므로 $K\Omega \neq \Omega$가 된다.

### 3. 일반화된 지식 연산자 $K'$ (Definition 1)

필연성과 무의식의 충돌을 해결하기 위해, 모든 사건 $E \subseteq \Omega$에 대해 다음과 같이 일반화된 지식 연산자를 정의한다.
$$K' E := KE \setminus U E$$
이는 에이전트가 $E$를 안다고 하는 상태 집합에서 $E$에 대해 무의식인 상태들을 명시적으로 제외하는 것이다.

### 4. R-필연성 (R necessitation)

새로운 연산자 $K'$는 전체 상태 공간 $\Omega$에 대해 다음과 같은 새로운 성질을 만족한다.
$$K' \Omega = \Omega \setminus U \Omega$$
여기서 $U \Omega$는 에이전트가 비자명한 무의식을 가지는 상태들의 집합이다. 만약 무의식이 없다면($U E = \emptyset$), 이는 표준 필연성 $K\Omega = \Omega$와 동일해진다.

## 📊 Results

### 1. DLR 모순의 해결

DLR의 논리는 다음과 같은 연쇄 과정에서 모순을 발견했다:
$$\emptyset \neq U E \subseteq U(U E) \subseteq \neg K \neg K(U E) = \neg K \Omega = \emptyset \text{ (필연성 적용 시)}$$
결과적으로 $\emptyset \neq \emptyset$라는 모순이 발생한다. 하지만 본 논문이 제안한 $K'$와 R-필연성을 적용하면 결과가 다음과 같이 변경된다:
$$\emptyset \neq U E \subseteq U(U E) \subseteq \neg K' \neg K'(U E) = \neg K' \Omega = U \Omega$$
결론적으로 $U E \subseteq U \Omega$라는 타당한 결과가 도출되며, 모델의 일관성이 유지된다.

### 2. 표준 성질의 보존

일반화된 연산자 $K'$는 다음과 같은 인식론적 성질들을 여전히 만족함을 보였다:

- **KU 내성 (KU introspection)**: $K(U E) = \emptyset$
- **AU 내성 (AU introspection)**: $U E \subseteq U(U E)$
- **대칭성 (Symmetry)**: $U E = U(\neg E)$ (전체 상태 공간 $\Omega$에 대해서도 $U \Omega = U(\neg \Omega)$가 성립함)

## 🧠 Insights & Discussion

### 강점 및 해석

본 논문은 무의식 모델링의 고질적인 문제였던 '필연성' 가정을 정면으로 다루었다. 특히 필연성이 사실은 '분할 가능성(partitional property)'에서 기인한 것이며, 무의식을 가진 에이전트에게 이를 강제하는 것은 에이전트가 모든 것을 알게 만드는 것과 같다는 점을 지적한 것이 매우 날카롭다. 복잡한 다중 상태 공간 모델을 도입하지 않고도 단일 모델 내에서 연산자 수정만으로 모순을 해결했다는 점에서 이론적 효율성이 높다.

### 한계 및 논의 사항

- **무한 상태 공간**: 저자는 $\Omega$가 무한할 경우 $U \Omega$ 또한 무한할 수 있으며, 이를 통해 학습하는 에이전트가 항상 새로운 것을 배울 수 있는 구조를 제안한다. 하지만 이에 대한 구체적인 수학적 전개나 사례 분석은 부족하며 아이디어 수준에서 언급되었다.
- **실제 적용**: 본 연구는 매우 추상적인 집합론적 모델링에 집중하고 있다. 이 논리가 실제 게임 이론이나 경제학적 의사결정 모델에서 어떤 정량적 차이를 만드는지에 대한 실험적 데이터는 제시되지 않았다.

## 📌 TL;DR

이 논문은 표준 상태-공간 모델에서 '필연성($K\Omega = \Omega$)' 가정이 무의식 모델과 충돌하여 발생하는 논리적 모순(DLR 결과)을 해결한다. 저자는 비자명한 무의식이 존재할 때 필연성이 위배됨을 증명하고, 무의식 집합을 제외한 새로운 지식 연산자 $K' E = KE \setminus UE$를 제안하여 모델의 일관성을 회복하였다. 이 연구는 향후 무의식을 포함한 인식론적 모델링에서 필연성 가정을 재검토해야 함을 시사하며, 무한 상태 공간으로의 확장 가능성을 열어두었다.
