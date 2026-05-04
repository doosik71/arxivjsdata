# SuperARC: Can Increasing Complexity Explain Intelligence? A Test for Artificial Super Intelligence Based On the Principles of Causal Recursive Compression and Algorithmic Probability

Alberto Hernández-Espinosa, Luan Ozelim, Felipe S. Abrahão, and Hector Zenil (2025)

## 🧩 Problem to Solve

본 논문은 현대의 대규모 언어 모델(LLM)이 주장하는 인공 일반 지능(AGI) 및 인공 초지능(ASI)의 수준을 객관적으로 정량화할 수 있는 평가 체계의 부재라는 문제를 해결하고자 한다. 기존의 AI 벤치마크들은 다음과 같은 한계를 지닌다. 첫째, 학습 데이터에 테스트 문제가 포함되어 성능이 과대평가되는 Benchmark Contamination 문제에서 자유롭지 못하다. 둘째, 인간의 언어 능력이나 특정 작업 수행 능력에 치중된 Human-centric 지표를 사용하여, 지능의 본질적인 특성보다는 통계적 패턴 매칭 능력을 측정하는 경향이 있다.

따라서 본 연구의 목표는 Kolmogorov-Chaitin Complexity, 정보 이론(Information Theory), 그리고 Algorithmic Probability에 기반하여, 벤치마크 오염을 방지하고 모델의 진정한 추상화 및 예측 능력을 측정할 수 있는 개방형 테스트 프레임워크인 SuperARC를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **'지능은 곧 압축 능력과 동일하며, 이는 곧 예측 능력과 직결된다'**는 직관에 기반한다. 구체적인 기여 사항은 다음과 같다.

1. **지능의 정량적 정의**: 지능을 "주어진 데이터를 최대한 손실 없이 설명할 수 있는 계산 가능한 모델을 생성하는 능력"으로 정의하고, 이를 알고리즘 복잡도(Algorithmic Complexity)의 관점에서 수식화하였다.
2. **SuperARC 프레임워크 제안**: 통계적 압축(GZIP 등)을 넘어선 재귀적 압축(Recursive Compression)과 프로그램 합성(Program Synthesis) 능력을 테스트하여, 모델이 데이터의 본질적인 규칙을 발견했는지 아니면 단순히 패턴을 암기했는지를 판별한다.
3. **Neurosymbolic 접근법과의 비교**: LLM과 알고리즘 확률론에 기반한 하이브리드 Neurosymbolic 접근법(BDM/CTM)을 비교하여, 현재의 LLM이 가진 근본적인 한계(단순 암기 및 통계적 모사)를 정량적으로 입증하였다.
4. **압축-예측 등가성 증명**: Martingales를 이용하여 모델의 압축 능력이 곧 미래 상태를 예측하는 능력과 직접적으로 비례함을 수학적으로 증명하였다.

## 📎 Related Works

논문은 지능 측정의 역사를 Spearman의 $g$-factor와 같은 심리학적 관점에서 시작하여, 최근의 ARC(Abstraction and Reasoning Corpus) 챌린지까지 검토한다. ARC 챌린지는 암기보다는 추상화 능력을 측정하려 했으나, 고정된 데이터셋으로 인해 결과가 유출되거나 이를 '해킹'하는 방식의 최적화가 가능했다는 한계가 있다.

또한, Solomonoff의 보편적 유도(Universal Induction)와 Hutter의 AIXI 모델과 같은 이론적 프레임워크를 언급하며, 지능을 알고리즘 복잡도의 관점에서 바라보는 이론적 토대를 제시한다. 기존 LLM들이 수행하는 예측 작업이 사실상 대규모 데이터셋에 대한 통계적 압축의 일종라는 점을 지적하며, 본 연구는 여기서 한 단계 나아가 '최소 설명 길이(Minimum Description Length)'를 찾는 능력을 통해 진정한 지능을 구분하고자 한다.

## 🛠️ Methodology

### 1. 전체 시스템 구조 및 이론적 배경

SuperARC는 **Algorithmic Information Theory (AIT)**를 기반으로 한다. 핵심은 어떤 데이터 문자열 $\sigma$를 생성할 수 있는 가장 짧은 이진 프로그램의 길이인 Kolmogorov 복잡도 $K(\sigma)$를 측정하는 것이다.

### 2. 주요 구성 요소: BDM과 CTM

본 논문은 계산 불가능한 $K(\sigma)$를 근사하기 위해 다음과 같은 하이브리드 방법을 사용한다.

* **CTM (Coding Theorem Method)**: 작은 튜링 머신들의 출력 빈도 분포를 이용하여 짧은 문자열의 알고리즘 확률 $P(s)$를 추정하고, 이를 통해 복잡도를 계산한다.
* **BDM (Block Decomposition Method)**: 데이터를 작은 블록으로 나누어 각 블록의 CTM 값을 합산하고, 전역적인 Shannon Entropy를 결합하여 긴 문자열의 복잡도를 근사한다.
    $$ \text{BDM}(x) = \sum_{i=1}^{n} \text{CTM}(x_i) + \log m_i $$
    여기서 $m_i$는 블록 $x_i$가 나타나는 빈도에 대한 보정 계수이다.

### 3. SuperARC 테스트 절차

모델(LLM)에게 특정 수열(Binary 또는 Integer sequence)을 주고, 이를 생성할 수 있는 가장 짧은 코드나 공식(프로그램)을 작성하게 한다.

* **목표 함수**: $\text{minimize}_{A, A^{-1}} M(A \circ A^{-1})$ subject to $A \circ A^{-1} : \{\tau \to \partial \to \tau\}$. 즉, 데이터를 가장 짧게 압축($\partial$)하고 다시 완벽하게 복원할 수 있는 알고리즘 $A$를 찾는 것이다.
* **평가 지표 ($\phi$)**: 단순히 정답 여부만 보는 것이 아니라, 생성된 코드의 유형을 다음과 같이 분류하여 가중치를 부여한다.
    1. **Prints**: 수열을 그대로 출력하는 코드 (압축률 0%, 지능 낮음)
    2. **Ordinal**: 인덱스 매핑 방식의 코드 (단순 구조 인식)
    3. **Non-Both**: 실제 규칙을 발견하여 생성한 공식/코드 (높은 추상화 및 지능)
    $$\phi = \delta_1 \rho_1 + \frac{\delta_2 \rho_2}{10} + \frac{\delta_3 \rho_3}{100}$$
    ($\rho$는 각 유형의 비율, $\delta$는 정규화된 BDM 복잡도 기반의 가중치)

## 📊 Results

### 1. 실험 설정

* **대상 모델**: GPT-4o, o1, Claude 3.5 Sonnet, Gemini, DeepSeek, Grok-3 등 최신 Frontier 모델 및 BDM/CTM 기반 ASI 모델.
* **데이터셋**: 저/중/고 복잡도로 구분된 정수 수열 및 이진 수열(Binary sequences). 특히 학습 데이터 오염을 막기 위해 이진 수열을 중점적으로 사용하였다.
* **작업**: 다음 숫자 예측(Next-digit prediction), 자유 형식 공식 생성(Free-form generation), 특정 언어(Python, C++ 등) 기반 코드 생성.

### 2. 주요 결과

* **예측 성능**: 수열의 복잡도가 증가함에 따라 모든 LLM의 예측 정확도가 급격히 하락하였다. 특히 이진 수열의 경우, 모델들의 정확도가 무작위 추측 수준인 50%에 근접하였다.
* **코드 생성의 허구적 정확성**: LLM이 정답을 맞춘 경우의 대부분은 단순 `print()` 문을 통해 수열을 그대로 출력한 'Print' 케이스였다. 이는 모델이 수열의 규칙을 이해(압축)한 것이 아니라, 단순히 데이터를 복제했음을 의미한다.
* **모델 버전 간 비교**: 최신 모델(o1, Gemini 1.5 Advanced 등)이 이전 버전보다 반드시 더 나은 성능을 보이지 않았으며, 일부 경우 오히려 창의적인 솔루션 생성 능력이 저하되는 경향이 관찰되었다.
* **BPDM/CTM의 압도적 성능**: Neurosymbolic 접근법은 모든 복잡도 단계에서 LLM을 압도하며, 최단 프로그램을 찾아내어 완벽한 예측과 복원을 수행하였다.

## 🧠 Insights & Discussion

### 1. 강점 및 발견

본 논문은 LLM이 보여주는 '추론 능력'이 실제로는 학습 데이터셋에 존재하는 유사 패턴의 통계적 검색 및 재구성(Hash-like retrieval)에 의존하고 있음을 정량적으로 입증하였다. 특히 정수 수열에서는 성적이 좋지만 이진 수열에서 성적이 낮은 점은, LLM이 규칙을 학습한 것이 아니라 훈련 데이터의 분포를 기억하고 있음을 시사한다.

### 2. 한계 및 비판적 해석

* **계산 복잡도**: CTM 기반의 접근법은 이론적으로 최적이지만, 탐색 공간이 넓어질수록 계산 비용이 기하급수적으로 증가한다. (물론 논문에서는 LLM 훈련 비용보다는 적다고 주장한다.)
* **범용성**: 본 테스트는 수열 생성이라는 특정 작업에 국한되어 있다. 실제 ASI가 갖추어야 할 다른 능력(사회적 상호작용, 도구 사용 등)을 모두 포괄하지는 못한다.

### 3. 결론적 논의

저자들은 LLM이 언어적 유창함(Fluency)을 통해 지능이 있는 것처럼 '보이게' 설계되었을 뿐, 진정한 의미의 추상화와 계획(Planning) 능력을 갖추지 못했다고 주장한다. 진정한 AGI/ASI로 가기 위해서는 단순한 통계적 패턴 매칭을 넘어, BDM/CTM과 같은 상징적 계산(Symbolic Computation)과 인과적 추론이 결합된 Neurosymbolic 시스템으로의 전환이 필수적이다.

## 📌 TL;DR

본 논문은 **"지능 = 압축 능력 = 예측 능력"**이라는 가설 하에, LLM의 진정한 지능을 측정하기 위한 **SuperARC** 벤치마크를 제안하였다. 실험 결과, 최신 LLM들은 복잡한 수열의 규칙을 찾아내어 짧은 코드로 압축하는 능력이 거의 없었으며, 정답을 맞춘 경우에도 대부분 단순 복제(`print`) 방식에 의존하였다. 이는 LLM이 진정한 추론을 하는 것이 아니라 통계적 암기에 의존하고 있음을 보여주며, 향후 ASI 달성을 위해서는 통계적 학습과 상징적 복잡도 이론이 결합된 **Neurosymbolic AI** 접근법이 필요함을 시사한다.
