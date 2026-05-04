# Evolving Prompts In-Context: An Open-ended, Self-replicating Perspective

Jianyu Wang, Zhiqiang Hu, Lidong Bing (2025)

## 🧩 Problem to Solve

본 논문은 대규모 언어 모델(LLM)의 프롬프팅에 관한 기존의 통념, 즉 "잘 설계된 지시문(instructions)과 정교한 예시(demonstrations)가 In-context Learning(ICL)의 성능을 극대화한다"는 관점에 도전한다.

기존의 프롬프트 엔지니어링은 인간의 언어적 직관에 의존하여 세밀하게 구조화된 프롬프트를 작성하는 데 집중해 왔으나, LLM은 매우 미묘한 문구 변화나 구조적 차이에도 예측 불가능하게 반응하는 특성이 있다. 저자들은 인간이 보기에 일관성이 없고 무의미해 보이는 'gibberish'(횡설수설) 형태의 프롬프트가 오히려 모델의 성능을 향상시킬 수 있다는 가설을 세운다.

따라서 본 연구의 목표는 자연어 프롬프트에서 일부 토큰을 제거(pruning)하여 LLM이 선호하는 최적의 '부분 컨텍스트(Partial Context)'를 찾는 것이며, 이를 위해 자동화된 진화 탐색 프레임워크인 **PROMPTQUINE**을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **프롬프트 압축(Prompt Compression)을 가이드된 프롬프트 탐색(Guided Prompt Search)으로 재정의**하는 것이다.

1. **Partial Context Hypothesis 제안**: 정교한 자연어 예시를 오히려 무작위로 가지치기(pruning)하여 문법적·의미적으로 파괴된 'gibberish' 상태로 만드는 것이, 역설적으로 다양한 태스크에서 성능을 향상시킬 수 있음을 증명하였다.
2. **PROMPTQUINE 프레임워크 개발**: 인간의 직관이나 기존의 토큰 중요도 측정 방식으로는 이러한 최적의 가지치기 전략을 찾을 수 없으므로, 유전 알고리즘(Genetic Algorithm) 기반의 자기 발견형 프롬프트 최적화 프레임워크를 제안하였다.
3. **범용적 효과 입증**: 분류(Classification), 다지선다형 질의응답(MCQ), 텍스트 생성(Generation), 수학적 추론(Math Reasoning) 등 광범위한 태스크와 다양한 모델(Llama-3, GPT-2 등)에서 기존의 최첨단(SOTA) 자동 프롬프트 최적화 기법과 대등하거나 이를 능가하는 성능을 보였다.

## 📎 Related Works

본 논문은 기존의 프롬프트 최적화 접근 방식을 다음과 같이 분류하고 그 한계를 지적한다.

- **Soft Prompt Tuning**: 연속적인 임베딩 공간에서 최적의 토큰을 찾지만, 이는 이산적인 텍스트 형태의 프롬프트를 필요로 하는 실제 사용 환경과는 거리가 있다.
- **Hard Prompt Optimization**: 토큰 레벨의 탐색이나 LLM을 프롬프트 엔지니어로 사용하는 방식(예: EvoPrompt, Promptbreeder)이 존재한다. 그러나 이들은 대개 정교한 자연어 지시문을 생성하는 데 집중하며, 본 논문처럼 기존 컨텍스트 내의 토큰을 가지치기하여 '비자연어 공간'을 탐색하는 접근과는 차별화된다.
- **Prompt Compression**: 기존의 압축 기법(예: LLMLingua)은 주로 추론 속도 향상(Efficiency)을 목표로 한다. 반면, 본 논문은 압축을 통해 오히려 작업 성능(Performance)을 높이는 것을 목표로 하며, 단순한 효율성 도구가 아닌 최적화 도구로 프롬프트 가지치기를 활용한다.

## 🛠️ Methodology

### 1. 문제 정의 (Formalization)

입력 프롬프트 $x = (x_1, x_2, \dots, x_n)$가 주어졌을 때, 목표는 $x$의 부분 수열(subsequence)인 가지치기 된 프롬프트 $z = (z_1, z_2, \dots, z_m)$ ($m \le n$)를 찾아 태스크 성능 $f(z; x, D)$를 최대화하는 것이다.

### 2. TAPruning (Baseline)

본격적인 프레임워크 이전에 제안된 단순 힐클라이밍(Hill-climbing) 방식이다.

- 토큰을 하나씩 제거하며 성능을 측정하고, 성능이 향상되거나 설정된 임계값 $\delta$ 이내로 유지되면 해당 변경 사항을 수용한다.
- 이는 국소 최적점(Local Optima)에 빠지기 쉽지만, 가지치기 가설을 빠르게 검증하는 베이스라인 역할을 한다.

### 3. PROMPTQUINE 프레임워크

PROMPTQUINE은 유전 알고리즘(GA)을 사용하여 최적의 토큰 마스크를 탐색한다.

- **유전체(Genotype)와 표현형(Phenotype)**: 각 프롬프트의 토큰 유지 여부를 나타내는 이진 마스크(Binary Mask)가 유전체이며, 실제 가지치기 된 텍스트가 표현형이 된다.
- **변이(Mutation)**: 비트 플립(bit-flip) 연산을 통해 토큰을 제거($1 \to 0$)한다.
- **선택 및 생존(Selection & Survival)**:
  - **Tournament Selection**: 소수의 후보군 중 최적의 개체를 선택하여 복제한다.
  - **Regularized Evolution**: 새로운 자손들이 기존 부모 세대와 직접 경쟁하지 않고, 오직 새로운 자손들끼리만 경쟁하여 생존 여부를 결정하게 함으로써 조기 수렴(Premature Convergence) 문제를 완화한다.
- **적합도 함수(Fitness Function)**: 분류 태스크의 경우, 정답 레이블의 확률과 그 외 가장 높은 확률 사이의 간격(Gap)을 이용한 piecewise reward function을 사용한다.

$$R(z, x, c) =
\begin{cases}
\lambda_2 \cdot \text{Gap}_z(c) & \text{if Correct} \\
\lambda_1 \cdot \text{Gap}_z(c) & \text{if not Correct}
\end{cases}$$
여기서 $\text{Gap}_z(c) := P_z(c) - \max_{c' \neq c} P_z(c')$ 이다.

### 4. 추론 및 최종 선택 절차
- **Calibration-then-selection**: 적합도 함수는 근사치(proxy)이므로 과적합 위험이 있다. 따라서 상위 $k\%$의 엘리트 프롬프트들을 먼저 선별한 후, 실제 검증 데이터셋(held-out set)에서 정확도를 다시 측정하여 최종 최적 프롬프트를 결정한다.

## 📊 Results

### 1. 실험 설정
- **데이터셋**: SST-2, Subj, AG's News, Yelp-5, SNLI, Yahoo (분류), Yelp Style Transfer (생성), AdvBench (Jailbreaking), PIQA (MCQ), GSM8K, MAWPS (추론).
- **비교 대상**: LLMLingua, LLMLingua2, RLPrompt, EvoPrompt, Promptbreeder, Original ICL.
- **측정 지표**: Accuracy, Joint Score (스타일 전이), ASR (탈옥 성공률).

### 2. 주요 결과
- **분류 태스크**: 1-shot ICL 기반의 PROMPTQUINE이 대부분의 SOTA 기법을 능가하거나 대등한 성능을 보였다. 특히 4-shot으로 확장했을 때 성능 향상이 더욱 뚜렷하게 나타났다.
- **텍스트 생성 및 탈옥**:
    - 스타일 전이(Style Transfer)에서 기존 SOTA인 RLPrompt와 Promptbreeder보다 높은 Joint Score를 기록하였다.
    - 탈옥(Jailbreaking) 실험에서는 가지치기 된 프롬프트가 기존 ICL 대비 공격 성공률(ASR)을 거의 두 배 가까이 높여, 모델의 정렬(Alignment)을 우회하는 데 효과적임을 보였다.
- **추론 태스크 (CoT)**: 복잡한 수학 추론에서도 가지치기가 효과적이었으며, 일부 경우 1-shot pruned prompt가 8-shot original prompt와 유사한 성능을 내면서도 컨텍스트 길이를 획기적으로 줄였다.
- **효율성**: RL-기반 방법론(RLPrompt 등)이 수 시간에서 수일이 걸리는 반면, PROMPTQUINE은 단 몇 분 만에 최적화를 완료하였다.

## 🧠 Insights & Discussion

### 1. 레이블 단어의 중요성
분석 결과, 가지치기 된 프롬프트에서도 태스크 관련 레이블 단어(label words)들이 높은 확률로 유지되었다. 이는 레이블 단어가 LLM의 추론 과정에서 '앵커(anchor)' 역할을 하며, 언어적 구조가 파괴되더라도 레이블 정보만 유지되면 성능이 보존되거나 향상될 수 있음을 시사한다.

### 2. 정렬(Alignment)의 취약성
본 연구는 LLM의 정렬이 '표면적 정렬(superficial alignment)'에 그치고 있음을 보여준다. 인간이 이해하는 자연어 지시문으로는 안전 가드레일이 작동하지만, 최적화된 '비자연어(unnatural language)' 프롬프트는 이러한 가드레일을 쉽게 무력화시킨다. 이는 내부 정렬(inner alignment)의 필요성을 제기한다.

### 3. 한계점
- **템플릿 민감도**: 프롬프트의 기본 템플릿(구분자, 띄어쓰기 등)이 바뀌면 성능 변동이 크게 나타나는 불안정성이 여전히 존재한다.
- **탐색 공간의 제한**: 현재는 기존 토큰의 제거(pruning)만 고려하고 있으며, 새로운 토큰의 삽입이나 교체까지 확장한다면 더 큰 성능 향상이 가능할 것으로 보인다.

## 📌 TL;DR

본 논문은 LLM 프롬프트에서 정교한 자연어 구조보다 **최적화된 부분 집합(subsequence)의 토큰 조합**이 더 효과적일 수 있다는 점을 발견하고, 이를 자동으로 탐색하는 유전 알고리즘 기반의 **PROMPTQUINE**을 제안한다. 이 방법은 분류, 생성, 추론 등 다양한 태스크에서 기존 SOTA 최적화 기법보다 빠르고 강력한 성능을 보였으며, 특히 모델의 안전 정렬을 우회할 수 있는 '비자연어 프롬프트'의 가능성을 제시함으로써 향후 LLM의 기전 연구 및 보안 강화에 중요한 시사점을 제공한다.
