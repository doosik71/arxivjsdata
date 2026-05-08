# Adiabatic Quantum Computing

Tameem Albash, Daniel A. Lidar (2018)

## 🧩 Problem to Solve

본 논문은 단열 양자 계산(Adiabatic Quantum Computing, 이하 AQC) 분야의 주요 이론적 발전을 종합적으로 분석하고 리뷰하는 것을 목표로 한다. AQC의 핵심 문제는 계산하고자 하는 문제의 해를 최종 Hamiltonian($H_1$)의 기저 상태(ground state)에 인코딩하고, 초기 Hamiltonian($H_0$)의 기저 상태에서 시작하여 시스템을 서서히 진화시켜 최종 기저 상태에 도달하게 하는 것이다.

이 과정에서 가장 큰 문제는 **에너지 갭(Energy Gap, $\Delta$)**이다. 기저 상태와 첫 번째 들뜬 상태 사이의 최소 에너지 차이인 $\Delta$가 시스템 크기에 따라 지수적으로 작아질 경우, 단열 정리를 만족하기 위해 필요한 계산 시간($t_f$)이 지수적으로 증가하여 계산 효율성이 급격히 떨어진다. 따라서 본 논문은 AQC의 보편성(Universality), 알고리즘적 성취와 한계, 그리고 특히 Stoquastic AQC(StoqAQC)의 특성과 성능 저하 요인을 분석하여 이를 극복할 방법을 제시하고자 한다.

## ✨ Key Contributions

본 보고서의 중심적인 직관은 AQC가 단순히 최적화 문제를 푸는 도구를 넘어, 표준 양자 회로 모델(Circuit Model)과 동등한 계산 능력을 가진 **보편적 양자 계산 모델**이라는 점을 이론적으로 규명하고 정리한 것이다. 주요 기여 사항은 다음과 같다.

1. **단열 정리의 체계적 정리**: 다양한 조건 하에서의 단열 정리 변형들을 검토하여, Hamiltonian의 매끄러움(smoothness)과 에너지 갭에 따른 실행 시간($t_f$)의 하한 및 상한을 명확히 하였다.
2. **AQC의 보편성 증명 리뷰**: '역사 상태(History State)' 구성 등을 통해 AQC가 양자 회로 모델을 효율적으로 시뮬레이션할 수 있으며, 그 반대도 가능함을 설명하여 두 모델의 다항 시간 동등성을 확인하였다.
3. **StoqAQC의 심층 분석**: 실제 구현이 용이한 Stoquastic Hamiltonian을 사용하는 StoqAQC의 계산 복잡도 클래스($BStoqP$)를 정의하고, 이것이 보편적 AQC보다 능력이 제한적일 수 있음을 이론적/수치적 근거로 제시하였다.
4. **성능 저하 극복 방안 제시**: 에너지 갭이 작아지는 '느려짐(slowdown)' 현상을 피하기 위해 최적 스케줄링(Brachistochrone), 촉매 Hamiltonian 추가, 비-스토카스틱(non-stoquastic) 항의 도입, 비단열 진화(diabatic evolution) 등의 전략을 분석하였다.

## 📎 Related Works

논문은 AQC의 배경이 되는 여러 관련 연구와 모델을 언급하며 기존 접근 방식과의 차이점을 설명한다.

- **표준 회로 모델(Circuit Model)**: 유니터리 게이트의 연속적인 적용으로 계산을 수행한다. AQC는 게이트 기반이 아니라 시스템의 Hamiltonian을 시간에 따라 변화시켜 기저 상태를 추적하는 방식이라는 점에서 근본적으로 다르다.
- **양자 어닐링(Quantum Annealing, QA)**: 주로 최적화 문제를 해결하기 위해 스토카스틱 Hamiltonian을 사용한다. 본 논문은 폐쇄계(closed system)에서의 AQC에 집중하며, 개방계(open system)에서의 진화인 QA와는 구분하여 서술한다.
- **Hamiltonian 복잡도 이론**: $k$-local Hamiltonian 문제와 QMA(Quantum Merlin-Arthur) 복잡도 클래스를 통해 AQC의 계산 능력을 분석한다. 특히 2-local Hamiltonian이 QMA-complete 하다는 점은 AQC의 보편성을 뒷받침하는 근거가 된다.

## 🛠️ Methodology

### 1. AQC의 기본 파이프라인 및 원리

AQC는 다음의 절차를 따른다.

- **초기화**: 기저 상태를 쉽게 준비할 수 있는 Hamiltonian $H_0$의 기저 상태 $| \psi(0) \rangle$에서 시작한다.
- **시간 진화**: 시간 $t \in [0, t_f]$ 동안 Hamiltonian을 $H(s) = (1-s)H_0 + sH_1$ 로 변화시킨다. 여기서 $s = t/t_f$는 스케줄(schedule)이다.
- **최종 상태**: 단열 정리에 의해 $t_f$가 충분히 크다면, 시스템은 최종 Hamiltonian $H_1$의 기저 상태에 도안하게 되며, 이 상태가 문제의 해를 포함한다.

### 2. 핵심 방정식 및 조건

- **실행 시간(Run time)**: 단열 조건을 만족하기 위한 최소 시간 $t_f$는 일반적으로 다음과 같은 관계를 가진다.
    $$t_f \gg \frac{\max_s \| \partial_s H(s) \|}{\Delta^2}$$
    여기서 $\Delta$는 진화 과정 중 발생하는 최소 에너지 갭이다. 즉, 갭이 작을수록 실행 시간은 제곱에 반비례하여 급격히 증가한다.
- **비용(Cost) 정의**: 에너지 스케일 변화로 인해 실행 시간이 임의로 작아지는 것을 방지하기 위해 무차원량인 $\text{cost} = t_f \max_s \| H(s) \|$를 사용한다.

### 3. 보편성 구현: 역사 상태(History State) 구성

양자 회로 $\mathcal{U} = U_L \dots U_1$을 AQC로 구현하기 위해 다음과 같은 '역사 상태' $| \eta \rangle$를 정의한다.
$$| \eta \rangle = \frac{1}{\sqrt{L+1}} \sum_{\ell=0}^{L} | \gamma(\ell) \rangle$$
여기서 $| \gamma(\ell) \rangle = | \alpha(\ell) \rangle \otimes | 1^\ell 0^{L-\ell} \rangle_c$ 이며, $| \alpha(\ell) \rangle$은 $\ell$번째 게이트 적용 후의 상태, $| 1^\ell 0^{L-\ell} \rangle_c$는 현재 시간을 나타내는 '파인만 클록(Feynman clock)' 레지스터이다. $H_{\text{final}}$을 이 역사 상태가 기저 상태가 되도록 설계함으로써, AQC를 통해 임의의 양자 회로를 시뮬레이션할 수 있다.

### 4. StoqAQC (Stoquastic AQC)

- **정의**: 계산 기저(computational basis)에서 모든 비대각 성분이 0 이하($\langle x | H | x' \rangle \le 0$)인 Hamiltonian을 사용하는 방식이다.
- **특징**: 이러한 Hamiltonian의 기저 상태는 모든 진폭이 0 이상의 실수인 상태로 표현될 수 있어, 고전적인 몬테카를로 시뮬레이션(QMC)이 가능하며 '사인 문제(sign problem)'가 발생하지 않는다.

## 📊 Results

### 1. 명시적 알고리즘의 속도 향상 (Speedup)

- **Adiabatic Grover**: 단순 선형 스케줄을 사용하면 고전 알고리즘과 동일한 $O(N)$ 시간이 걸리지만, 갭이 최소가 되는 지점에서 속도를 늦추는 **적응적 스케줄링(adaptive scheduling)**을 적용하면 $O(\sqrt{N})$의 이차적 속도 향상(quadratic speedup)을 얻을 수 있다.
- **Deutsch-Jozsa & Bernstein-Vazirani**: 적절한 Hamiltonian 구성을 통해 회로 모델과 동일한 $O(1)$ 시간 내에 해결 가능함을 보였다.

### 2. StoqAQC의 한계와 성능 저하

- **느려짐(Slowdown) 사례**: 3-XORSAT, 스핀 유리(Spin Glass) 모델 등에서 $\Delta$가 시스템 크기에 따라 지수적으로 감소하는 것이 확인되었으며, 이는 StoqAQC가 특정 NP-완전 문제에서 효율적이지 않을 수 있음을 시사한다.
- **SA 대비 우위**: 'Spike' 형태의 에너지 장벽이 있는 문제에서는 고전적인 시뮬레이션 어닐링(Simulated Annealing, SA)은 장벽을 넘지 못해 지수적 시간이 걸리지만, StoqAQC는 양자 터널링을 통해 다항 시간 내에 해결 가능한 경우가 존재한다.

### 3. 보편성 결과

- AQC는 다항 시간 오버헤드 내에서 양자 회로 모델을 시뮬레이션할 수 있으며, 반대로 회로 모델 역시 AQC를 시뮬레이션할 수 있다. 따라서 **AQC $\equiv$ Circuit Model** (Polynomial equivalence) 임이 증명되었다.

## 🧠 Insights & Discussion

### 1. 강점 및 가능성

AQC는 물리적 시스템의 자연스러운 진화를 이용하므로, 게이트 기반 모델보다 하드웨어 구현 시 오류 제어에 유리할 수 있는 잠재력이 있다. 또한, 비-스토카스틱 항을 추가함으로써 1차 양자 상전이(지수적 갭 감소)를 2차 상전이(다항적 갭 감소)로 변환하여 계산 속도를 획기적으로 높일 수 있다는 점이 인상적이다.

### 2. 한계 및 비판적 해석

- **StoqAQC의 정체성**: 많은 실험적 장치가 StoqAQC 기반이지만, 이론적으로 StoqAQC는 보편적 양자 계산 능력을 갖추지 못했을 가능성이 크다. 즉, 단순한 구현 가능성이 계산 능력의 보장으로 이어지지는 않는다.
- **갭 분석의 난해함**: 알고리즘의 성능을 결정하는 $\Delta$를 분석하는 것이 매우 어려우며, 많은 결과가 수치적 시뮬레이션에 의존하고 있어 점근적(asymptotic) 성능에 대한 확신이 부족하다.
- **터널링과 얽힘의 역할**: 터널링이 속도 향상의 핵심이라고 흔히 말하지만, 본 논문은 터널링이 반드시 속도 향상을 보장하는 것은 아니며, 얽힘(entanglement)의 양과 계산 효율성 사이의 직접적인 상관관계 또한 명확히 밝혀지지 않았음을 지적한다.

## 📌 TL;DR

본 논문은 **단열 양자 계산(AQC)의 이론적 토대부터 보편성, 그리고 실제적인 성능 한계와 극복 방안을 총망라한 리뷰 보고서**이다. AQC가 표준 양자 회로 모델과 계산적으로 동등함을 확인하였으며, 특히 구현이 쉬운 StoqAQC가 가지는 이론적 한계(지수적 갭 감소로 인한 느려짐)를 분석하였다. 결론적으로, 단순한 단열 진화를 넘어 **최적 스케줄링, 비-스토카스틱 항 도입, 비단열 진화** 등의 전략을 통해 양자 속도 향상을 달성할 수 있음을 시사하며, 이는 향후 양자 최적화 하드웨어 설계의 핵심 지침이 될 것으로 보인다.
