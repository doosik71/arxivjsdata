# CreativeBench: Benchmarking and Enhancing Machine Creativity via Self-Evolving Challenges

Zi-Han Wang, Lam Nguyen, Zhengyang Zhao, Mengyue Yang, Chengwei Qin, Yujiu Yang, Linyi Yang (2026)

## 🧩 Problem to Solve

본 연구는 코드 생성 분야에서 기계의 창의성(Machine Creativity)을 정량적으로 평가할 수 있는 엄격한 벤치마크가 부족하다는 문제를 해결하고자 한다. 기존의 코드 생성 모델 평가는 주로 기능적 정확성(Functional Correctness)을 측정하는 $\text{Pass@k}$ 지표에 의존해 왔으며, 이는 모델이 얼마나 창의적인 해결책을 제시했는지를 평가하는 데 한계가 있다.

특히 기존의 창의성 평가 시도들은 다음과 같은 세 가지 주요 문제점을 가지고 있다. 첫째, 객관적인 기준이 부족하여 창의성과 환각(Hallucination)을 명확히 구분하지 못한다. 둘째, 태스크의 복잡도가 낮아 모델이 단순 암기(Rote Memorization)를 통해 답을 내놓는 경우가 많다. 셋째, 진화 시스템(Evolutionary Systems)에서 사용할 수 있는 자동화된 정량적 지표가 부족하다. 따라서 본 논문의 목표는 인지 과학적 프레임워크에 기반하여 기계의 창의성을 객관적으로 측정하는 벤치마크인 CreativeBench를 구축하고, 이를 통해 모델의 스케일링 및 추론 능력이 창의성에 미치는 영향을 분석하며, 나아가 이를 향상시킬 수 있는 제어 전략을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Boden의 인지 창의성 프레임워크를 코드 생성에 적용하여, 창의성을 '조합적 창의성(Combinatorial Creativity)'과 '탐색적 창의성(Exploratory Creativity)'이라는 두 가지 차원으로 정의하고 이를 개별적으로 평가하는 것이다.

1. **CreativeBench 구축**: 역공학(Reverse Engineering)과 셀프 플레이(Self-Play) 기반의 자동화 파이프라인을 통해 높은 난이도의 조합적/탐색적 코드 생성 데이터셋을 구축하였다.
2. **통합 창의성 지표(Unified Creativity Score) 제안**: 창의성을 품질(Quality)과 신규성(Novelty)의 곱으로 정의하여, 단순히 정답을 맞히는 것뿐만 아니라 기존의 전형적인 해결책과 얼마나 다른지를 정량적으로 측정하였다.
3. **EvoRePE 제안**: 진화적 탐색 경로에서 추출한 '창의성 벡터(Creativity Vector)'를 추론 시점에 주입하여 모델의 창의적 성능을 일관되게 향상시키는 Plug-and-play 방식의 표현 제어(Representation Engineering) 전략을 제안하였다.
4. **심층 분석**: 모델 스케일링이 조합적 창의성에는 긍정적이나 탐색적 창의성에는 한계가 있다는 점, 그리고 추론(Reasoning) 능력이 제약 조건 기반의 탐색적 창의성에 주로 기여한다는 인사이트를 도출하였다.

## 📎 Related Works

본 논문은 다음과 같은 관련 연구들을 다루고 있다.

- **기계 창의성 평가**: 기존 연구들은 주로 발산적/수렴적 사고를 측정하거나 주관적 평가에 의존하였다. 하지만 이러한 방식은 앞서 언급한 대로 환각과의 구분 문제 및 데이터 누수(Data Leakage)로 인한 암기 문제에서 자유롭지 못하다.
- **진화 알고리즘(Evolutionary Algorithms)**: Fun-Search나 AlphaEvolve와 같이 LLM을 변이 연산자로 사용하여 최적의 프로그램을 찾는 시스템들이 제안되었다. 본 연구는 이러한 진화 시스템의 결과물이 실제로 얼마나 '창의적인지'를 측정하는 평가 체계에 집중한다.
- **표현 공학(Representation Engineering)**: 모델의 내부 활성화 값(Internal Activations)을 조작하여 정직성이나 편향성을 제어하는 연구들이 존재한다. 본 논문은 이를 확장하여 '창의성'이라는 고차원적 특성을 제어 벡터로 추출하여 적용하였다.

## 🛠️ Methodology

### 1. CreativeBench 데이터셋 구축

CreativeBench는 두 가지 하위 집합으로 구성된다.

- **CreativeBench-Combo (조합적 창의성)**: 서로 다른 도메인의 코드 컴포넌트를 융합하여 고난도 문제를 생성한다.
    - **Solution Fusion**: 서로 다른 도메인(예: 데이터 처리 + 그래프 알고리즘)의 코드를 결합하여 실행 가능한 통합 솔루션을 먼저 만든다.
    - **Test Function Generation**: 검증된 솔루션으로부터 테스트 케이스를 역으로 생성한다.
    - **Problem Synthesis**: 솔루션과 테스트 케이스를 바탕으로, 모델이 해당 코드를 작성하도록 유도하는 문제 설명을 생성하는 역공학 과정을 거친다.

- **CreativeBench-Explore (탐색적 창의성)**: 제약 조건 생성자와 솔버(Solver) 간의 상호작용을 통해 난이도를 점진적으로 높인다.
    - **Dynamic Constraint Stacking**: 레벨 0(제약 없음)부터 시작하여, 솔버가 문제를 해결하면 생성자가 해당 솔루션의 특정 알고리즘 선택을 무효화하는 '부정적 제약(Negative Constraint)'을 추가한다.
    - **Refinement**: 새로운 제약 조건이 추가된 문제가 실제로 해결 가능한지 확인하기 위해 참조 가이드 기반의 정제 과정을 거치며, 해결 가능함이 확인된 경우에만 데이터셋에 포함시킨다.

### 2. 평가 지표: Unified Creativity Score

창의성 점수는 다음과 같이 품질과 신규성의 곱으로 정의된다.

$$\text{Creativity} = \mathbb{E}[\text{Quality} \times \text{Novelty}]$$

- **Quality (품질)**: 샌드박스 내 실행 결과의 정확성을 측정하는 $\text{Pass@1}$ 지표를 사용한다.
- **Novelty (신규성)**: 생성된 솔루션 $u$와 베이스라인 솔루션 $v$ 사이의 거리를 측정한다. semantic 수준의 거리(CodeXEmbed 임베딩)와 lexical 수준의 거리(character 4-gram Jaccard 거리)를 결합하여 사용한다.
  $$N(u, v) = (1 - \cos(e_u, e_v)) + \left( 1 - \frac{|G_4(u) \cap G_4(v)|}{|G_4(u) \cup G_4(v)|} \right)$$
  여기서 $e$는 임베딩 벡터, $G_4$는 문자 4-gram 집합을 의미한다.

### 3. EvoRePE (Evolutionary Representation Engineering)

진화 알고리즘을 통한 최적화 경로를 잠재 공간의 스티어링 벡터로 내부화하는 방법이다.

- **벡터 추출**: 표준 프롬프트($x_{\text{base}}$)와 진화적으로 최적화된 프롬프트($x_{\text{evo}}$) 쌍에 대해, 특정 레이어 $\ell$에서의 활성화 값 차이 $\Delta h_\ell^{(i)} = h^\ell(x_{\text{evo}}^{(i)}) - h^\ell(x_{\text{base}}^{(i)})$를 계산한다. 이 차이 벡터들의 집합에 대해 PCA(주성분 분석)를 수행하여 주성분인 창의성 벡터 $v^\ell$를 추출한다.
- **추론 시 적용**: 추론 과정에서 잔차 스트림(Residual Stream)에 추출된 벡터를 더해 모델의 출력을 창의적인 방향으로 유도한다.
  $$\tilde{h}^\ell = h^\ell + \alpha v^\ell$$
  여기서 $\alpha$는 제어 강도를 조절하는 하이퍼파라미터이다.

## 📊 Results

### 1. 모델 성능 분석
- **전반적 난이도**: 최신 모델인 Gemini-3-Pro조차 두 데이터셋 모두에서 $\text{Pass@1}$이 $60\%$ 미만으로 나타나, 본 벤치마크가 단순 암기가 아닌 실제 창의적 문제 해결 능력을 요구함을 보여주었다.
- **스케일링의 영향**: 모델 크기가 커질수록 조합적 창의성($\text{CreativeBench-Combo}$)은 유의미하게 향상되었으나, 탐색적 창의성($\text{CreativeBench-Explore}$)의 향상은 정체되거나 오히려 약간 감소하는 경향을 보였다.
- **추론 능력의 영향**: Reasoning 모드를 활성화했을 때, 조합적 창의성에는 거의 영향이 없었으나 탐색적 창의성 성능은 크게 향상되었다. 이는 제약 조건 하의 탐색 과정이 구조적인 '사고 체인(Chain-of-Thought)'의 도움을 많이 받는다는 것을 의미한다.

### 2. EvoRePE의 효과
- **성능 향상**: Qwen2.5-7B-Instruct 모델에 EvoRePE를 적용했을 때, Vanilla 프롬프트뿐만 아니라 AlphaEvolve, GEPA와 같은 진화 알고리즘과 결합했을 때도 창의성 점수가 추가로 향상되었다.
- **효율성**: 진화 알고리즘은 수많은 추론 호출이 필요하여 비용이 높지만, EvoRePE는 한 번 추출된 벡터를 더하는 단순 연산($O(1)$)만으로 유사한 효과를 낼 수 있음을 입증하였다.

## 🧠 Insights & Discussion

### 1. Convergence-by-Scaling (스케일링에 의한 수렴)
연구진은 모델 크기가 커질수록 정답률(Quality)은 올라가지만, 신규성(Novelty)은 낮아지거나 정체되는 현상을 발견하고 이를 'Convergence-by-Scaling'이라 명명하였다. 이는 거대 모델이 학습 데이터의 고빈도 패턴을 더 잘 학습하여, 정답에 가까운 '전형적인' 솔루션을 생성하는 경향이 강해지기 때문으로 해석된다. 결과적으로 스케일링은 지식의 재조합(Combination)에는 유리하지만, 기존 패턴을 벗어나는 '0에서 1로의 도약'과 같은 탐색적 혁신에는 한계가 있다.

### 2. 추론 능력과 창의성의 관계
Reasoning 능력이 탐색적 창의성에만 기여한다는 점은 흥미로운 지점이다. 도메인 간의 융합(Combination)은 방대한 지식의 검색과 효율적인 구성 능력이 핵심인 반면, 제약 조건 하의 해결책 찾기(Exploration)는 가용 가능한 경로를 체계적으로 탐색하고 검증하는 논리적 추론 과정이 필수적이기 때문으로 분석된다.

### 3. 한계 및 논의
- **언어적 제한**: 현재 파이썬(Python) 언어에 국한되어 구축되었으나, 파이프라인이 자동화되어 있어 다른 언어로 확장이 가능하다.
- **생성자 편향**: LLM을 이용해 데이터를 생성했으므로 생성 모델의 편향이 벤치마크에 반영되었을 가능성이 있다.

## 📌 TL;DR

본 논문은 기계의 창의성을 정량적으로 평가하기 위해 Boden의 인지 프레임워크를 기반으로 한 **CreativeBench**를 제안한다. 이 벤치마크는 조합적/탐색적 창의성을 구분하여 측정하며, **$\text{Creativity} = \text{Quality} \times \text{Novelty}$**라는 통합 지표를 통해 환각과 창의성을 엄격히 구분한다. 분석 결과, 모델 스케일링은 조합적 창의성을 높이지만 탐색적 창의성에는 한계가 있으며(Convergence-by-Scaling), 추론 능력은 제약 조건 기반의 탐색에 주로 기여함을 밝혔다. 또한, 진화적 경로를 벡터화하여 주입하는 **EvoRePE**를 통해 추론 비용 없이 모델의 창의성을 높이는 방법을 제시하였다. 이 연구는 향후 AI가 단순한 지식 재조합을 넘어 진정한 의미의 혁신적 솔루션을 생성하게 하는 평가 및 제어 기반을 마련하였다는 점에서 중요하다.