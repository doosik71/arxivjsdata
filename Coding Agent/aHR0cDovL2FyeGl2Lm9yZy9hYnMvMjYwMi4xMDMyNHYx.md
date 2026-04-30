# Discovering Differences in Strategic Behavior between Humans and LLMs

Caroline Wang, Daniel Kasenberg, Kim Stachenfeld, and Pablo Samuel Castro (2026)

## 🧩 Problem to Solve

본 연구는 대규모 언어 모델(Large Language Models, LLMs)이 사회적 및 전략적 시나리오에 빈번하게 배치됨에 따라, LLM의 행동이 인간의 행동과 어디서, 그리고 왜 갈라지는지를 이해하는 것을 목표로 한다. 특히 최근 연구자들은 LLM을 인간 행동의 대리인(proxy)으로 사용하여 사회적 상호작용을 시뮬레이션하는 '디지털 트윈' 방식으로 활용하고 있으나, LLM이 본질적으로 비인간적 존재임에도 불구하고 인간과 유사하게 행동한다는 가정하에 분석을 진행하는 위험이 있다.

따라서 본 논문은 반복적 행렬 게임(Iterated Matrix Games)이라는 통제된 샌드박스 환경을 통해 인간과 LLM의 전략적 행동 차이를 구조적으로 분석하고자 한다. 이를 통해 LLM을 인간의 대리인으로 사용할 때의 한계를 명확히 하고, 행동 정렬(behavioral alignment)을 개선하며, LLM의 전략적 능력을 모니터링할 수 있는 기초를 마련하는 것이 연구의 핵심 목표이다.

## ✨ Key Contributions

본 논문의 가장 중심적인 기여는 **자동화된 심볼릭 모델 발견(Automated Symbolic Model Discovery)** 기법을 행동 게임 이론(Behavioral Game Theory, BGT)에 최초로 적용하여, 인간과 LLM의 행동을 설명하는 해석 가능한(interpretable) 모델을 직접 도출했다는 점이다.

핵심 아이디어는 블랙박스 형태의 신경망 모델 대신, 데이터로부터 직접 최적의 파이썬 프로그램(Python programs)을 진화시켜 행동 모델을 생성하는 `AlphaEvolve` 도구를 사용하는 것이다. 이를 통해 단순히 승률과 같은 통계적 수치를 비교하는 것을 넘어, 상대방을 모델링하는 방식의 차원(dimensionality)이나 가치 학습(value learning)의 메커니즘과 같은 **구조적 요인(structural factors)**을 분석하여 인간과 frontier LLM 사이의 전략적 사고 깊이 차이를 규명하였다.

## 📎 Related Works

기존의 행동 게임 이론(BGT) 연구들은 주로 단순한 행동 통계 분석이나, 인간의 인지적 편향을 가정하여 수동으로 설계된 매개변수화된 수학 모델을 통해 인간의 행동을 설명하려 했다. 그러나 이러한 접근 방식은 LLM과 같은 비인간 에이전트가 가지는 고유한 행동적 사전 지식(behavioral priors)을 포착하지 못한다는 한계가 있다.

또한, 기존의 LLM 전략 분석 연구들은 LLM의 전략적 능력을 단독으로 평가하거나, 특정 게임의 통계적 결과에 집중하는 경향이 있었다. 반면 본 연구는 LLM이 생성한 해석 가능한 프로그램 공간을 탐색함으로써, 데이터 기반의 오픈엔디드(open-ended) 방식으로 인간과 LLM의 행동 차이를 구조적으로 분석한다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 1. 실험 환경: Iterated Rock-Paper-Scissors (IRPS)
연구팀은 가위바위보 게임을 300라운드 동안 반복하는 IRPS를 테스트베드로 설정하였다. 승리는 3점, 무승부는 0점, 패배는 -1점을 부여한다. 상대역으로는 단순한 규칙을 따르는 '비적응형(nonadaptive) 봇'과 상대의 패턴을 추적하는 '적응형(adaptive) 봇' 등 총 15종의 봇을 사용하였다.

### 2. 데이터셋 구성
- **Human Dataset**: Brockbank와 Vul(2024)이 수집한 411명의 인간 참가자 데이터(총 129,087회 선택)를 사용하였다.
- **LLM Datasets**: 인간과 동일한 조건(프롬프트, 보상 체계 등)을 부여하여 Gemini 2.5 Pro, Gemini 2.5 Flash, GPT 5.1, 그리고 오픈소스 모델인 GPT OSS 120B의 데이터를 수집하였다.

### 3. AlphaEvolve를 이용한 행동 모델 발견
`AlphaEvolve`는 LLM을 사용하여 수학적 목적 함수를 최대화하는 파이썬 프로그램을 생성하고 진화시키는 도구이다.

**가. 문제 정의 (Behavior Modeling)**
행동 모델은 함수 $f(a^t, a_{opp}^t, r^t, h^t, \theta) \to (\hat{p}^{t+1}, h^{t+1})$로 정의된다. 여기서 $a^t$는 자신의 선택, $a_{opp}^t$는 상대의 선택, $r^t$는 보상, $h^t$는 내부 상태, $\theta$는 학습 가능한 파라미터이며, 출력값 $\hat{p}^{t+1}$은 다음 수에 대한 확률 분포이다.

**나. 학습 및 최적화 절차**
모델의 파라미터 $\theta$는 다음과 같은 음의 로그 가능도(negative log likelihood)를 최소화하는 방향으로 SGD를 통해 학습된다.
$$\theta^* = \arg\min_{\theta} \sum_{i=1}^{N} \sum_{t=1}^{T-1} -\log \Pr(a_{i,t+1} | a_{i,t}, a_{opp,i,t}, r_{i,t}, h_{i,t}, \theta)$$

**다. 다목적 최적화 (Multi-objective Optimization)**
단순히 예측 정확도만 높이면 오버피팅이 발생하고 모델이 복잡해져 해석력이 떨어진다. 따라서 본 연구는 다음 두 가지를 동시에 최적화하는 Pareto frontier를 탐색한다.
1. **교차 검증된 가능도 (Cross-validated Likelihood)**: 예측 성능 측정.
2. **Halstead Effort**: 프로그램의 연산자와 피연산자 수를 기반으로 한 코드 복잡도 측정.

최종적으로는 성능이 최상위권이면서 가장 단순한 **SBB(Simplest-But-Best)** 프로그램을 선택한다. SBB 선택 규칙은 다음과 같다.
$$\text{SBB}(\epsilon) \in \arg\max_{f \in PF(\hat{\Phi})} \{s(f) \mid \ell(f) > \max_{f' \in \hat{\Phi}} \ell(f') - \epsilon\}$$
여기서 $\ell(f)$는 평가 점수, $s(f)$는 단순성(negative Halstead effort)을 의미한다.

## 📊 Results

### 1. 승률 분석 (Win Rates)
- **정량적 결과**: Gemini 2.5 Pro/Flash와 GPT 5.1 같은 frontier 모델들은 인간보다 훨씬 높은 승률을 기록하였으며, 특히 복잡한 봇을 상대로 더 빠르게 적응하여 최적 승률에 도달하였다.
- **특이사항**: GPT OSS 120B는 시간이 지남에 따라 오히려 승률이 감소하는 경향을 보였는데, 이는 긴 컨텍스트에서의 정보 합성 능력이 부족하기 때문으로 분석된다.

### 2. 모델 적합도 비교 (Quality-of-fit)
AlphaEvolve로 발견된 프로그램들은 기존의 BGT 모델인 CS-EWA(Contextual Sophisticated EWA)보다 유의미하게 높은 예측 성능을 보였으며, 블랙박스 모델인 GRU(RNN)와 비슷하거나 더 나은 성능을 기록하였다. 이는 AlphaEvolve가 해석 가능성을 유지하면서도 높은 예측력을 가진 구조를 성공적으로 찾아냈음을 시사한다.

### 3. 구조적 차이 분석 (SBB 프로그램 분석)
발견된 SBB 프로그램들을 분석한 결과, 모든 에이전트가 **가치 기반 학습(Value-based learning)**과 **상대 모델링(Opponent Modeling)**이라는 두 가지 핵심 요소를 공통적으로 사용하고 있었다. 그러나 상대 모델링의 구현 수준에서 결정적인 차이가 나타났다.

- **인간 및 GPT OSS 120B**: 1차원(1D) 모델을 사용하여 단순히 상대방이 어떤 수를 얼마나 자주 냈는지(frequency)만을 추적한다.
- **Gemini 2.5 Pro/Flash**: 3x3 행렬 모델을 사용하여 상대방의 이전 수에 따른 다음 수의 전이 확률(transition)을 추적한다.
- **GPT 5.1**: 3x3x3 텐서 모델을 사용하여 더 깊은 수준의 전이 패턴을 추적한다.

## 🧠 Insights & Discussion

본 연구는 frontier LLM들이 인간보다 더 정교한 **마음 이론(Theory of Mind, ToM)** 능력을 갖추고 있음을 시사한다. 구체적으로, 인간은 상대의 단순 빈도만을 보는 반면, 최신 LLM들은 상대의 전략적 패턴(전이 확률)을 인식하고 이를 이용해 최적의 대응책을 계산하는 더 깊은 수준의 전략적 사고를 수행한다.

**강점 및 한계**:
- **강점**: 블랙박스 모델의 내부를 추측하는 대신, 행동 데이터를 통해 역으로 해석 가능한 심볼릭 모델을 도출함으로써 객관적인 구조적 비교를 가능하게 했다.
- **한계**: 본 연구는 IRPS라는 특정 게임에 국한되었으며, 전문가 수준의 인간 플레이어까지는 포함하지 않았다. 따라서 모든 인간이 LLM보다 전략적 깊이가 얕다고 단정할 수는 없다.

**비판적 해석**:
결과적으로 LLM은 인간의 행동을 흉내 내는 '디지털 트윈'으로서의 역할에는 부적합할 수 있다. LLM은 인간보다 더 '합리적'이거나 '전략적'으로 행동하는 경향이 있으므로, 인간의 인지적 편향이나 한계를 시뮬레이션해야 하는 사회과학 연구에서 LLM을 그대로 사용할 경우 결과가 왜곡될 가능성이 크다.

## 📌 TL;DR

본 논문은 `AlphaEvolve`라는 프로그램 진화 도구를 통해 인간과 LLM의 가위바위보 전략을 모델링한 파이썬 프로그램을 자동으로 생성하여 비교하였다. 분석 결과, frontier LLM들은 인간보다 높은 승률을 기록했으며, 그 이유는 상대방의 행동 패턴을 추적하는 **상대 모델링의 차원(Dimension)**이 인간(1D)보다 훨씬 높기(3x3, 3x3x3) 때문임이 밝혀졌다. 이는 LLM이 단순한 텍스트 생성기를 넘어 정교한 전략적 추론 능력을 갖추었음을 보여주며, 동시에 LLM을 단순한 인간 대리인으로 사용하는 것에 대한 주의가 필요함을 시사한다.