# Approaches to Artificial General Intelligence: An Analysis

Soumil Rathi (2021)

## 🧩 Problem to Solve

본 논문은 인간 수준의 지능을 가진 인공 일반 지능(Artificial General Intelligence, AGI)을 구현하기 위해 제안된 다양한 방법론들을 분석하고, 그중 어떤 접근 방식이 가장 실현 가능성이 높은지를 평가하는 것을 목표로 한다.

저자는 먼저 지능을 '어떤 시나리오에서도 목표를 달성하기 위해 최선의 행동을 선택하고, 최종 목표 달성을 위한 하위 목표(sub-goals)를 생성하는 능력'으로 정의한다. 이에 따라 AGI는 어떤 환경에서도 목표 달성을 위해 최적의 행동을 결정하고 장기적 목표를 위한 하위 목표를 설계할 수 있는 인공 모델로 정의된다.

이 문제의 중요성은 AGI가 단순한 특정 작업 수행을 넘어 인간이 수행하는 대부분의 일상적이고 번거로운 작업들을 대체함으로써 인류에게 큰 도움을 줄 수 있다는 점에 있다. 또한, 기존의 인공 협소 지능(Artificial Narrow Intelligence, ANI)이 가진 일반화(generalization) 능력의 부재를 해결하는 것이 AGI 구현의 핵심 과제이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 AGI를 달성하기 위한 세 가지 주요 접근 방식인 인간 뇌 에뮬레이션(Human Brain Emulation), 알고리즘 확률(Algorithmic Probability), 그리고 통합 인지 아키텍처(Integrated Cognitive Architecture)를 체계적으로 비교 분석한 점이다.

저자는 각 방법론의 기술적 요구 사항, 계산 능력의 한계, 그리고 AGI의 요구 조건 충족 여부를 검토하였다. 최종적으로, 하드웨어 및 스캐닝 기술의 제약이 큰 뇌 에뮬레이션이나 계산 불가능성(uncomputability) 문제가 있는 알고리즘 확률 방식보다, 인간의 인지 프로세스를 개별적으로 복제하여 통합하는 '통합 인지 아키텍처'가 현재 기술 수준에서 가장 유망한 경로임을 제시하였다.

## 📎 Related Works

논문은 AGI와 대비되는 개념으로 ANI(Artificial Narrow Intelligence)의 사례를 언급한다. AlphaGo나 AlphaZero와 같은 프로그램들은 특정 영역에서 탁월한 성능을 보이지만, 학습한 기술을 유사한 다른 분야로 전이시키지 못하는 일반화의 한계를 가지고 있다.

또한, AGI 구현을 위한 기존의 이론적 토대로 다음의 연구들을 인용한다.

- **Ray Kurzweil**: 인간 뇌 에뮬레이션을 통해 2020~2030년대에 AGI가 가능할 것이라고 주장하였다.
- **Solomonoff's Induction**: 데이터로부터 가장 가능성 높은 환경을 예측하는 확률적 척도를 제안하였다.
- **Marcus Hutter의 AIXI**: Solomonoff의 유도를 기반으로 어떤 미지의 시나리오에서도 최적의 행동을 선택하는 이론적 모델을 제시하였다.
- **Ben Goertzel의 CogPrime**: 인지 시너지(Cognitive Synergy)를 통해 일반 지능을 구현하려는 통합 인지 아키텍처를 제안하였다.

## 🛠️ Methodology

본 논문은 새로운 알고리즘을 제안하는 것이 아니라, 기존의 AGI 접근법들을 분석하는 리뷰 성격의 논문이다. 분석 대상이 된 세 가지 방법론의 상세 내용은 다음과 같다.

### 1. Human Brain Emulation (HBE)

인간의 뇌를 계산적으로 시뮬레이션하는 방법이다.

- **하드웨어 요구사항**: 뇌의 시냅스 트랜잭션과 계산량을 고려할 때 약 $10^{16}$에서 $10^{19}$ calculations per second의 성능이 필요하다.
- **스캐닝 기술**: 나노봇(nanobots)을 이용해 뇌의 해마(hippocampus)나 소뇌(cerebellum) 같은 영역을 비침습적으로 실시간 스캐닝하여 데이터를 수집해야 한다.
- **구현 절차**: 뇌 영역의 알고리즘을 이해하고 이를 합성 신경망 등으로 복제하여 계산 매체에 구현한다.

### 2. Algorithmic Probability & AIXI

Solomonoff의 유도(induction)와 베이즈 정리(Bayes theorem)를 사용하여 환경을 예측하고 보상을 최대화하는 방식이다.

- **환경 예측**: 보편 튜링 머신(Universal Turing Machine, UTM)을 통해 관측값 $x$가 나올 확률 $P(x)$를 다음과 같이 계산한다.
  $$P(x) = 2^{-l(x)}$$
  여기서 $l(x)$는 환경 $x$의 이진 길이(binary length)이다.
- **AIXI 모델**: 모든 가능한 환경의 확률 합 $M(x)$를 기반으로 보상을 최대화하는 행동 $a_k$를 선택한다.
  $$M(x) = \sum_{p:UTM(p)=x} 2^{-l(p)}$$
- **한계**: 무한한 수의 환경을 검토해야 하므로 계산 불가능(uncomputable)하며, 튜링 머신의 정지 문제(halting problem)가 발생한다. 이를 해결하기 위해 계산 시간과 길이를 제한한 $AIXI_{tl}$이나 타입 람다 계산법(typed lambda calculi)을 사용한 UCAI 같은 근사 모델이 제안되었으나, 여전히 계산 비용이 지나치게 높다.

### 3. Integrative Cognitive Architecture (ICA)

인간 뇌의 핵심 인지 프로세스를 식별하고 이를 개별적으로 복제하여 통합하는 방법이다.

- **아키텍처 유형**:
  - **Symbolic**: 지식을 기호 형태로 표현. 추론은 강하나 학습과 창의성이 부족하다.
  - **Subsymbolic/Emergent**: 신경망과 유사하게 퍼셉트론의 가중치 조절을 통해 학습. 추상적 추론과 언어 처리가 어렵다.
  - **Hybrid**: 위 두 방식을 결합하여 추론(Symbolic)과 학습(Subsymbolic) 능력을 모두 확보한다.
- **CogPrime 시스템**: 하이브리드 아키텍처를 기반으로 하며, 다음과 같은 중앙 프로세스를 수행한다.
  $$\text{Context} + \text{Procedure} \rightarrow \text{Goal} \langle \text{prob} \rangle$$
  즉, 특정 문맥(Context) 하에서 목표(Goal)를 달성하기 위해 확률 $p$를 가진 절차(Procedure)를 수행하는 방식이다.
- **주요 구성 요소**:
  - **PLN (Probabilistic Logical Networks)**: 선언적 기억(Declarative memory)을 바탕으로 문맥을 파악하고 확률적 추론을 수행한다.
  - **MOSES**: 절차적 기억(Procedural memory)을 통해 과거 경험을 바탕으로 최적의 행동 절차를 선택한다.
  - **Cognitive Synergy**: 각 인지 프로세스가 서로 협력하여 계산 효율을 높이고 조합 폭발(Combinatorial Explosion) 문제를 방지하는 메커니즘이다.

## 📊 Results

저자는 각 방법론의 실현 가능성을 다음과 같이 평가하였다.

- **Human Brain Emulation**: 하드웨어 성능(Fugaku 슈퍼컴퓨터 등)은 어느 정도 도달했으나, 뇌를 정밀하게 스캐닝할 수 있는 나노봇 기술은 2030년대 이후에나 가능할 것으로 예측되어 단기적 실현 가능성이 낮다.
- **Algorithmic Probability (AIXI)**: 이론적으로는 완벽한 벤치마크가 될 수 있으나, 계산 복잡도가 지수적으로 증가($2^l$)하여 실제 구현이 사실상 불가능하다.
- **Integrative Cognitive Architecture (CogPrime)**: AIXI의 이론적 구조를 계승하면서도 PLN과 같은 효율적인 추론 방식을 도입하여 계산 불가능성 문제를 해결하였다. 필요한 기술들이 이미 상당 부분 존재하며 계산 요구량도 뇌 에뮬레이션보다 낮다.

결과적으로 저자는 **CogPrime으로 대표되는 통합 인지 아키텍처가 AGI에 도달하는 가장 빠르고 효율적인 경로**라고 결론짓는다.

## 🧠 Insights & Discussion

본 논문은 AGI를 구현하기 위한 세 가지 서로 다른 패러다임을 '계산 가능성'과 '기술적 가용성'이라는 관점에서 날카롭게 분석하였다. 특히 AIXI라는 이론적 이상향과 CogPrime이라는 실천적 구현체 사이의 관계를 설명하며, 왜 하이브리드 인지 아키텍처가 현실적인 대안이 되는지를 논리적으로 풀어냈다.

다만, 다음과 같은 한계와 논의점이 존재한다.

1. **정의의 모호성**: '인간 수준'의 지능에 대한 정의가 주관적일 수 있으며, 인간의 부정적인 특성(편향, 거짓 기억 등)을 배제한 AGI를 설계하는 구체적인 방법론은 명시되지 않았다.
2. **시너지 구현의 난이도**: 저자는 Cognitive Synergy가 중요하다고 언급했지만, 서로 다른 성격의 Symbolic/Subsymbolic 모듈을 어떻게 효율적으로 결합하고 최적화할 것인가에 대한 상세한 메커니즘 설명은 부족하다.
3. **추측에 기반한 타임라인**: 뇌 에뮬레이션의 가능 시점을 2030~2040년대로 설정한 것은 Ray Kurzweil의 주장에 의존한 것으로, 실제 기술 발전 속도와는 차이가 있을 수 있다.

## 📌 TL;DR

본 논문은 AGI 달성을 위한 세 가지 경로(뇌 에뮬레이션, 알고리즘 확률, 통합 인지 아키텍처)를 분석하였다. 뇌 에뮬레이션은 스캐닝 기술의 부재로, AIXI는 계산 불가능성으로 인해 한계가 명확하다. 반면, 인간의 인지 기능을 모듈화하여 통합한 **CogPrime과 같은 통합 인지 아키텍처**가 현재 기술적/계산적 관점에서 가장 유망한 AGI 구현 방법임을 제시한다. 이는 향후 AGI 연구가 단순한 모델 확장이 아닌, 인지 프로세스의 시너지와 효율적 구조 설계 방향으로 나아가야 함을 시사한다.
