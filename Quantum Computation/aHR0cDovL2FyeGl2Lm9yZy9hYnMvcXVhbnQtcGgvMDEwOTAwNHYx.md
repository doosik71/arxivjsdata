# Quantum Computing and Shor's Factoring Algorithm

I.V. Volovich (2001)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 커다란 정수 $N$을 두 개의 소수 $p$와 $q$의 곱으로 분해하는 소인수분해(Factoring) 문제이다. 정수 $N$이 소수들의 곱으로 유일하게 분해된다는 사실은 알려져 있으나, 이를 효율적으로 수행할 수 있는 고전적 알고리즘은 아직 발견되지 않았다.

소인수분해 문제의 중요성은 현대 암호학의 기반이 되는 많은 시스템이 소인수분해의 계산적 어려움에 의존하고 있다는 점에 있다. 본 논문의 목표는 소인수분해 문제를 다항 시간(Polynomial time) 내에 해결할 수 있는 Shor의 양자 알고리즘(Shor's algorithm)을 상세히 설명하고, 그 수학적 배경과 계산 복잡도를 분석하는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 아이디어는 소인수분해 문제를 정수론적 성질을 이용하여 '차수 찾기(Order finding)' 문제로 환원(Reduction)시키는 것이다. 즉, 임의의 정수 $m$에 대하여 $m^r \equiv 1 \pmod N$을 만족하는 가장 작은 양의 정수 $r$(차수)을 찾는 문제로 변환한다.

이 과정에서 고전 컴퓨터로는 지수 시간이 소요되는 차수 찾기 문제를 양자 컴퓨터의 양자 푸리에 변환(Quantum Fourier Transform, QFT)을 통해 다항 시간 내에 해결함으로써, 결과적으로 소인수분해의 효율성을 극대화하는 설계를 제시한다.

## 📎 Related Works

논문에서는 고전적인 소인수분해 알고리즘으로 Number Field Sieve를 언급하며, 이는 점근적으로 $\exp(cn^{1/3}(\log n)^{2/3})$의 연산이 필요하여 입력 비트 수 $n$에 대해 지수적 시간 복잡도를 가짐을 지적한다.

또한 알고리즘의 일반적 정의를 위해 튜링 머신(Turing Machine)과 회로(Circuit) 모델을 소개하며, D. Deutsch에 의해 제안된 양자 회로 및 양자 튜링 머신이 고전적 모델의 한계를 극복하는 수학적 모델임을 명시한다. 기존의 고전적 접근 방식이 $O(\sqrt{N})$ 혹은 그 이상의 시간이 걸리는 반면, Shor의 알고리즘은 $O(n^2 \log n \log \log n)$의 다항 시간 복잡도를 가진다는 점에서 결정적인 차별점을 보인다.

## 🛠️ Methodology

### 전체 파이프라인
소인수분해를 위한 전체 프로세스는 다음과 같은 단계로 구성된다.
1. 무작위 정수 $m$을 선택하고, 양자 알고리즘을 통해 $m$의 차수 $r$을 찾는다.
2. $r$이 짝수이고 $m^{r/2} \not\equiv -1 \pmod N$인 경우, 유클리드 알고리즘을 통해 $\text{g.c.d.}(m^{r/2}-1, N)$과 $\text{g.c.d.}(m^{r/2}+1, N)$을 계산하여 $N$의 인수를 찾는다.

### 차수 찾기(Order Finding)의 양자 절차
차수를 찾기 위한 양자 알고리즘은 다음의 5단계로 수행된다.

1. **양자 상태 준비(Preparation of quantum state):** 첫 번째 레지스터를 모든 가능한 상태의 균일 중첩 상태로 만든다.
   $$|\psi_1\rangle = \frac{1}{\sqrt{q}} \sum_{a=0}^{q-1} |a\rangle \otimes |0\rangle$$
2. **모듈로 지수 연산(Modular exponentiation):** 두 번째 레지스터에 $m^a \pmod N$을 계산하여 저장한다.
   $$|\psi_2\rangle = \frac{1}{\sqrt{q}} \sum_{a=0}^{q-1} |a\rangle \otimes |m^a \pmod N\rangle$$
3. **양자 푸리에 변환(Quantum Fourier Transform):** 첫 번째 레지스터에 QFT를 적용한다. QFT의 일반식은 다음과 같다.
   $$F_q |a\rangle = \frac{1}{\sqrt{q}} \sum_{b=0}^{q-1} e^{2\pi iab/q} |b\rangle$$
4. **측정(Measurement):** 두 레지스터를 측정하여 $|c\rangle$와 $|m^k \pmod N\rangle$ 값을 얻는다.
5. **고전적 계산(Classical computation):** 측정된 $c/q$ 값을 연분수 전개(Continued fraction expansion)를 통해 근사하여 차수 $r$을 도출한다.

### 주요 방정식 및 복잡도
- **차수 찾기의 확률:** 정수 $d$가 $-\frac{r}{2} \leq rc - dq \leq \frac{r}{2}$를 만족할 때, 해당 상태가 측정될 확률 $P(c, m^k \pmod N)$은 최소 $\frac{1}{3r^2}$ 이상이다.
- **계산 복잡도:** 전체 알고리즘의 양자 게이트 연산 횟수는 다음과 같이 다항 시간으로 수렴한다.
  $$O((\log N)^2 (\log \log N)(\log \log \log N))$$

## 📊 Results

본 논문은 정량적인 실험 데이터보다는 수학적 증명과 복잡도 분석에 집중한다.

- **비교 대상:** 고전적 알고리즘(Number Field Sieve) vs 양자 알고리즘(Shor's Algorithm).
- **측정 지표:** 입력 비트 수 $n = \log N$에 따른 연산 횟수(Time Complexity).
- **결과:** 고전 알고리즘은 $n^{1/3}$에 대한 지수적 시간이 필요하지만, Shor의 알고리즘은 $n$에 대한 다항 시간 내에 연산이 완료됨을 보였다.
- **성공 확률:** 정수 $N$이 충분히 크고 두 개 이상의 소수의 곱일 때, $O(\log \log N)$번의 반복 수행을 통해 일정한 확률 $\gamma > 0$로 인수를 찾을 수 있음을 증명하였다.

## 🧠 Insights & Discussion

본 논문은 Shor의 알고리즘이 현대 암호 체계(특히 RSA)에 가하는 잠재적 위협을 시사한다. 소인수분해의 효율적 해결은 기존의 공개키 암호화 방식을 무력화할 수 있기 때문이다.

저자는 소인수분해 문제가 NP-완전(NP-complete) 문제인지 여부는 아직 불분명하며, 아마도 아지수적(Sub-exponential) 시간 복잡도를 가질 것으로 추측한다. 또한, 양자 튜링 머신을 넘어선 새로운 계산 패러다임으로 카오스 역학 증폭기(Chaotic dynamics amplifier)나 비선형 하트리-포크 역학(Nonlinear Hartree-Fock dynamics)을 결합한 접근 방식을 통해 NP-완전 문제를 해결할 가능성을 제안하며 논의를 확장한다.

## 📌 TL;DR

본 논문은 정수 소인수분해 문제를 '차수 찾기' 문제로 환원하고, 양자 푸리에 변환(QFT)을 이용하여 이를 다항 시간 내에 해결하는 Shor의 알고리즘을 학술적으로 분석한다. 이 연구는 고전 컴퓨터로는 불가능했던 대규모 정수의 소인수분해를 효율적으로 수행할 수 있음을 수학적으로 입증하였으며, 이는 향후 양자 컴퓨팅 기반의 암호 해독 및 새로운 보안 체계 연구에 결정적인 역할을 할 것으로 평가된다.