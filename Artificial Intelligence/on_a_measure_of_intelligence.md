# On a measure of intelligence

Yuri Gurevich (2024)

## 🧩 Problem to Solve

본 논문(또는 대담 형식의 보고서)은 지능(Intelligence)을 어떻게 정의하고 측정할 것인가라는 근본적인 문제에 대해 다룬다. 특히 François Chollet이 제안한 지능의 측정 방식과 그 기반이 되는 이론적 프레임워크를 분석하고 비판하는 것을 목표로 한다.

인공지능(AI) 분야에서 '일반 지능(General Intelligence)'에 도달하기 위해서는 단순히 특정 작업의 성능을 높이는 것이 아니라, 지능에 대한 작동 가능한 정의와 정량적인 측정 방법이 필수적이다. 기존의 AI 벤치마크들이 주로 패턴 인식과 통계적 학습 능력에 의존했다는 점은 실제 지능의 핵심인 '새로운 상황에 대한 적응력'을 측정하지 못한다는 한계가 있다. 따라서 본 글은 지능을 '기술 습득 효율성(Skill-acquisition efficiency)'으로 정의하려는 시도와 이를 구체화한 ARC(Abstraction and Reasoning Corpus) 벤치마크의 타당성을 검토한다.

## ✨ Key Contributions

본 보고서의 핵심 기여는 François Chollet의 지능 측정 이론에 대한 수학적, 철학적 비판적 분석을 제공하는 것이다. 주요 내용은 다음과 같다.

1. **지능의 재정의**: 지능을 '무엇을 알고 있는가(Skill)'가 아니라 '새로운 기술을 얼마나 효율적으로 습득하는가(Process)'의 관점에서 바라보는 Chollet의 시각을 분석한다.
2. **AIT 기반 공식화의 한계 지적**: 알고리즘 정보 이론(Algorithmic Information Theory, AIT)과 콜모고로프 복잡도(Kolmogorov Complexity)를 이용해 일반화 난이도(Generalization Difficulty)를 정량화하려는 시도가 가진 수학적 불안정성을 증명한다.
3. **계산 불가능성 논의**: 지능 측정의 기초로 사용된 $H(s)$ 함수가 계산 불가능(Uncomputable)하며, 근사조차 불가능하다는 점을 들어 해당 공식화가 실질적인 도구라기보다 상징적인 표현에 가깝다는 점을 지적한다.
4. **다학제적 관점 제시**: 신경과학(Jeff Hawkins)과 컴퓨터 과학(Leslie Valiant)의 관점을 인용하여, 지능이라는 개념이 가진 모호성과 '교육 가능성(Educability)'이라는 대안적 개념을 제시한다.

## 📎 Related Works

논문에서는 지능과 관련된 여러 학문적 배경을 언급한다.

- **Cybernetics (Norbert Wiener)**: 동물과 기계의 제어 및 통신으로 정의되며, 현대 AI의 통계적/패턴 인식적 접근의 모태가 되었다.
- **g-factor (Charles Spearman)**: 서로 다른 인지 작업 간의 양의 상관관계를 통해 일반 지능이라는 단일 능력이 존재한다는 가설이다. 현대의 IQ 테스트가 이를 측정하려 한다.
- **Core Knowledge (Spelke & Kinzler)**: 인간이 태어날 때부터 혹은 생후 직후 빠르게 습득하도록 하드와이어드된 기본 지식(객체, 에이전트, 기하학, 수 체계)을 의미한다.
- **Educability (Leslie Valiant)**: 지능이라는 광범위하고 모호한 개념 대신, 모델이 얼마나 효율적으로 학습하고 확장할 수 있는지를 다루는 '교육 가능성'의 개념을 제안한다.

## 🛠️ Methodology

본 보고서는 Chollet의 방법론을 상세히 소개하고 이를 수학적으로 비판한다.

### 1. 지능의 정의: Skill-acquisition efficiency

Chollet은 지능을 특정 작업의 수행 능력이 아니라, 예상치 못한 새로운 환경에 적응하거나 새로운 기술을 습득하는 효율성으로 정의한다. 이는 지능을 결과물(Skill)이 아닌 프로세스(Process)로 보는 관점이다.

### 2. 알고리즘 정보 이론(AIT)을 통한 공식화

Chollet은 콜모고로프 복잡도(Kolmogorov Complexity) $H(s)$를 사용하여 지능을 정량화하려 한다. 여기서 $H(s)$는 문자열 $s$를 생성하는 가장 짧은 프로그램의 길이이다.

그는 특정 작업 $T$에 대한 **일반화 난이도(Generalization Difficulty, GD)**를 다음과 같이 정의한다.

$$\text{GD}_{\theta, T, C} = \frac{H(\text{Sol}_{\theta, T} | \text{TrainSol}_{\text{opt}, T, C})}{H(\text{Sol}_{\theta, T})}$$

- $\text{Sol}_{\theta, T}$: 평가 시 임계값 $\theta$를 만족하는 $T$의 최단 해결책.
- $\text{TrainSol}_{\text{opt}, T, C}$: 커리큘럼 $C$가 주어졌을 때 $T$의 최단 최적 훈련 해결책.
- 이 값은 $0 < \text{GD} \le 1$ 범위의 값을 가지며, 지능적인 시스템일수록 적은 훈련 데이터(경험)만으로도 해결책을 찾아내어 이 효율성이 높아진다.

### 3. ARC-AGI 벤치마크

위 이론을 실체화한 것이 ARC(Abstraction and Reasoning Corpus)이다. ARC는 통계적 패턴 인식이 아닌, 추상화와 추론, 그리고 퓨샷 학습(Few-shot learning) 능력을 측정하도록 설계되었다. 이는 인간의 Core Knowledge를 전제로 하여 AI가 얼마나 적은 예시로 새로운 규칙을 추론해내는지를 평가한다.

## 📊 Results

본 텍스트는 실험 데이터를 직접 제시하는 논문이 아니므로, Chollet의 주장에 대한 Gurevich의 비판적 분석 결과가 주를 이룬다.

- **수학적 불안정성**: 콜모고로프 복잡도 $H(s)$는 보편 튜링 기계(Universal Turing Machine) 간에 상수 $c$만큼의 차이가 발생한다. 하지만 $\text{GD}$ 식에서는 이 상수가 분자와 분모에 영향을 주어, 언어 $L$의 선택에 따라 $\text{GD}$ 값이 $0$ 근처에서 $1$ 근처로 극단적으로 변할 수 있음을 지적한다.
- **계산 불가능성**: $H(s)$ 함수는 계산 불가능(Uncomputable)하며, 계산 가능한 함수 $f(s)$로 근사하는 것 또한 불가능하다는 점이 논의된다. 따라서 $\text{GD}$ 식은 실제 계산 가능한 지표가 아니라 개념적인 수식에 불과하다.
- **ARC의 한계**: ARC는 야심찬 프로젝트이나, 현재로서는 인간의 Core Knowledge를 일부 전제하고 있으며, 이를 완전히 명시화하여 AI와 비교하는 것은 여전히 매우 도전적인 과제이다.

## 🧠 Insights & Discussion

### 강점 및 가치

Chollet의 접근 방식은 AI 평가의 패러다임을 '정적 데이터셋의 정확도'에서 '새로운 작업에 대한 적응 효율성'으로 전환했다는 점에서 매우 중요하다. 특히 ARC-AGI는 LLM과 같은 거대 모델들이 가진 '암기 기반의 성능'과 '실제 추론 능력'을 구분 짓는 중요한 척도가 된다.

### 비판적 해석

Gurevich는 Chollet이 사용한 AIT 프레임워크가 수학적으로는 엄밀하지 않으며, 실제 구현 불가능한 이론에 지나치게 의존하고 있다고 비판한다. 즉, AIT는 영감을 주는 동기(Motivation)로서는 훌륭하지만, 이를 통해 정량적인 '측정치'를 도출하려는 시도는 이론적 허점이 많다는 것이다.

### 추가 논의 (지능과 감정)

보고서는 신경과학적 관점을 통해 지능(신피질, Neocortex)과 감정(구뇌, Old brain)의 물리적 분리를 언급한다. 이는 감정이 없는 인공지능이 충분히 가능하며, 지능의 측정에서 감정적 지능(Emotional Intelligence)을 배제하는 것이 타당할 수 있음을 시사한다.

## 📌 TL;DR

본 보고서는 지능을 **'기술 습득 효율성(Skill-acquisition efficiency)'**으로 정의하고 이를 ARC-AGI 벤치마크와 AIT(알고리즘 정보 이론)로 측정하려는 François Chollet의 시도를 분석하고 비판한다. 지능을 프로세스로 정의한 직관은 뛰어나지만, 이를 정량화하기 위해 도입한 콜모고로프 복잡도 기반의 수식은 **언어 의존성(Additive constant 문제)**과 **계산 불가능성**이라는 치명적인 수학적 한계를 가진다. 결론적으로, 지능에 대한 명확한 정의와 측정은 여전히 미해결 과제이며, ARC와 같은 벤치마크는 그 방향성을 제시하는 중요한 시도로 평가된다.
