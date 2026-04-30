# Implicit Reasoning in Large Language Models: A Comprehensive Survey

Jindong Li, Yali Fu, Li Fan, Jiahong Liu, Yao Shu, Chengwei Qin, Menglin Yang, Irwin King, Rex Ying (2025)

## 🧩 Problem to Solve

본 논문은 대규모 언어 모델(Large Language Models, LLMs)의 추론 과정에서 발생하는 **Explicit Reasoning(명시적 추론)**의 효율성 저하 문제를 해결하기 위한 **Implicit Reasoning(암시적 추론)**의 체계적인 분석과 정리를 목표로 한다. 

최근의 추론 모델들은 Chain-of-Thought(CoT)와 같이 중간 추론 단계를 자연어 텍스트로 출력하는 명시적 방식을 통해 복잡한 문제 해결 능력을 높여왔다. 그러나 이러한 방식은 불필요하거나 중복된 텍스트를 생성함으로써 추론 시간이 길어지고, 계산 비용이 증가하며, 추론 지연 시간(latency)이 늘어나는 치명적인 단점이 있다. 

따라서 본 연구는 중간 단계의 텍스트 출력 없이 모델 내부의 잠재 공간(latent space)에서 추론을 수행하는 Implicit Reasoning의 메커니즘을 분석하고, 이를 체계적으로 분류한 택소노미(Taxonomy)를 제공함으로써 더 효율적이고 빠른 추론 시스템 구축을 위한 이론적 기반을 마련하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 LLM의 암시적 추론을 '표현 형태'가 아닌 **'실행 패러다임(Execution Paradigms)'**이라는 기능적 관점에서 정의하고 분류했다는 점이다.

1. **실행 중심의 택소노미 제안**: 내부 계산이 '어떻게' 그리고 '어디서' 일어나는지에 따라 암시적 추론 방법을 다음의 세 가지 패러다임으로 분류하였다.
    - **Latent Optimization (잠재 최적화)**: 내부 표현을 직접 조작하여 추론 능력을 향상시키는 방식.
    - **Signal-Guided Control (신호 가이드 제어)**: 특수 토큰 등의 제어 신호를 통해 내부 계산 과정을 조절하는 방식.
    - **Layer-Recurrent Execution (층 재귀 실행)**: 아키텍처 수준에서 반복적인 계산을 통해 상태를 점진적으로 정제하는 방식.
2. **암시적 추론의 증거 합성**: 구조적 패턴(Structural), 행동 특성(Behavioral), 표현 기반 분석(Representation-based) 관점에서 LLM 내부에 실제로 암시적 추론이 존재한다는 경험적/메커니즘적 증거들을 통합하여 제시하였다.
3. **평가 체계의 구조화**: 암시적 추론의 효과와 신뢰성을 측정하기 위해 사용되는 다양한 지표(Metrics)와 벤치마크 데이터셋을 체계적으로 정리하여 제시하였다.

## 📎 Related Works

기존의 LLM 추론 관련 서베이들은 주로 CoT 프롬프팅이나 심볼릭 추론과 같은 **Explicit Reasoning(명시적 추론)** 패러다임에 집중해 왔다. 일부 연구에서 잠재 표현(latent representations)에 대해 다루긴 했으나, 이는 주로 표현 방식의 변경이나 메커니즘적 해석에 치우쳐 있었다.

본 논문은 기존 연구들과 달리 **'계산 전략(Computational Strategies)'**에 초점을 맞춘다. 즉, 단순히 "어떤 형태로 저장되는가"가 아니라 "모델이 내부적으로 어떻게 계산을 수행하여 결론에 도달하는가"라는 실행 관점에서 접근함으로써, 파편화되어 있던 암시적 추론 연구들을 하나의 통합된 프레임워크로 묶어냈다는 점에서 차별점을 가진다.

## 🛠️ Methodology

논문은 LLM의 추론 과정을 다음과 같이 정형화한다. 입력 $x$가 주어졌을 때, 모델 $\pi_\theta$는 먼저 내부 추론 흔적인 $z_{1:M}$을 생성하고, 이를 바탕으로 최종 답변 $a$를 출력한다.

$$z_{1:M} \sim \pi_\theta(\cdot|x), \quad a \sim \pi_\theta(\cdot|x, z_{1:M})$$

여기서 $z_{1:M}$이 텍스트로 출력되면 명시적 추론이 되고, 내부 상태(hidden state)로만 존재하면 암시적 추론이 된다. 본 논문이 제안하는 세 가지 기술적 패러다임의 상세 내용은 다음과 같다.

### 1. Latent Optimization (잠재 최적화)
내부 표현을 직접 최적화하여 텍스트 출력 없이 추론을 수행하는 방식이며, 최적화 단위에 따라 세분화된다.
- **Token-Level**: 개별 토큰 수준에서 조작한다. 예를 들어, Sparse Autoencoders(SAE)를 통해 추출된 개념 토큰을 삽입하거나(CoCoMix), 학습 가능한 잠재 토큰을 추가하여 계산 과정을 제어한다.
- **Trajectory-Level**: 추론 경로 전체를 하나의 단위로 최적화한다. 명시적인 CoT 경로를 연속적인 잠재 공간의 경로로 압축(CCoT)하거나, 점진적으로 명시적 단계를 잠재 단계로 내면화(Coconut)하는 방식이다.
- **Internal-State-Level**: 모델의 은닉 상태 자체를 조절한다. 교사 모델의 은닉 상태를 학생 모델이 모방하게 하는 지식 증류(ICoT-KD)나, 내부 메모리 모듈을 도입하여 상태를 저장하고 회수하는 방식 등이 포함된다.

### 2. Signal-Guided Control (신호 가이드 제어)
특수 설계된 제어 토큰을 삽입하여 모델의 내부 계산량을 조절하는 방식이다.
- **Single-Type Signal**: 하나의 제어 메커니즘을 사용한다. `thinking tokens`나 `pause tokens`를 삽입하여 모델이 답변을 내놓기 전 더 많은 계산 시간을 갖도록 유도한다. (예: Quiet-STaR)
- **Multi-Type Signal**: 여러 종류의 신호를 사용하여 세밀하게 제어한다. 예를 들어 `<memory>`와 `<reason>` 토큰을 구분하여 기억 회상과 추론 과정을 분리해 제어한다.

### 3. Layer-Recurrent Execution (층 재귀 실행)
트랜스포머 아키텍처에 재귀(Recurrence) 구조를 도입하여, 동일한 가중치를 가진 층을 여러 번 통과하며 상태를 정제하는 방식이다.
- **구조적 특징**: 모델의 깊이를 물리적으로 늘리는 대신, 가중치 공유(Weight Sharing)를 통해 루프(Loop)를 형성한다.
- **작동 원리**: 토큰별로 필요한 계산량을 다르게 설정하는 Adaptive Token Routing을 사용하거나, CoT의 단계별 추론을 루프의 반복 횟수와 정렬(Alignment)시켜 내부적으로 깊은 추론을 수행하게 한다.

## 📊 Results

### 1. 암시적 추론의 존재 증거
논문은 LLM이 텍스트 출력 없이도 추론한다는 것을 다음과 같은 증거로 뒷받침한다.
- **구조적 증거**: 중간 층의 활성화 값만으로도 최종 정답을 상당히 정확하게 예측할 수 있다는 점(Jump to Conclusions)과 층별로 하위 작업이 순차적으로 수행된다는 점이 확인되었다.
- **행동적 증거**: 과적합을 넘어선 추가 학습 단계에서 갑자기 추론 능력이 발현되는 'Grokking' 현상과, 일부 추론 단계를 건너뛰어도 정답을 맞히는 'Step-skipping' 행동이 관찰되었다.
- **표현 기반 증거**: Probing 기법을 통해 은닉 상태 내에 추론 트리(Reasoning Tree) 구조가 인코딩되어 있음을 확인하였다.

### 2. 평가 지표 및 벤치마크
암시적 추론의 성능을 측정하기 위해 다음과 같은 지표와 데이터셋이 사용된다.
- **핵심 지표**:
    - **정확도**: Accuracy, Pass@k, Exact Match (EM).
    - **효율성**: Decoding Latency(지연 시간), Output Length(출력 길이), $\text{ACU} = \frac{\text{Accuracy}}{\#\text{Params} \times \#\text{Tokens}}$ (계산 단위당 정확도).
    - **내부 분석**: Perplexity (PPL), Probing Accuracy.
- **주요 벤치마크**:
    - **상식/일반지식**: CommonsenseQA, HellaSwag, ARC-challenge.
    - **수학/프로그래밍**: GSM8K, MATH, HumanEval, LiveCodeBench.
    - **복잡한 다단계 추론**: HotpotQA, StrategyQA, MMLU, GPQA.
    - **멀티모달 추론**: MathVista, ScienceQA.

## 🧠 Insights & Discussion

### 강점 및 가능성
암시적 추론은 인간이 겉으로 말하지 않고 속으로 생각하는 과정과 유사하며, 텍스트 생성이라는 병목 현상을 제거함으로써 추론 속도를 획기적으로 높일 수 있다. 또한, 잠재 공간에서의 계산은 자연어라는 제약에서 벗어나 더 다양하고 풍부한 추론 경로를 탐색할 수 있는 가능성을 제공한다.

### 한계 및 비판적 해석
1. **불투명성(Opacity)**: 가장 큰 문제는 내부에서 무슨 일이 일어나는지 알 수 없다는 점이다. 모델이 실제로 논리적 추론을 수행하는 것인지, 아니면 단순한 패턴 매칭이나 지름길 학습(Shortcut Learning)을 하는 것인지 구분하기 어렵다.
2. **신뢰성 및 제어 부족**: 명시적 추론에서는 중간 단계의 오류를 발견하고 수정할 수 있지만, 암시적 추론은 '조용히 실패(Silent Failure)'한다. 이는 고위험 응용 분야에서 치명적일 수 있다.
3. **성능 격차**: 현재까지는 대다수의 암시적 추론 방법이 명시적인 CoT 방식보다 최종 정확도가 낮다. 이는 암시적 추론이 아직 견고한 일반화 능력을 갖추지 못했음을 시사한다.
4. **데이터 의존성**: 많은 암시적 추론 모델들이 역설적으로 '명시적인 CoT 데이터'를 이용해 학습된다. 이는 순수하게 잠재 공간에서만 작동하는 추론 체계를 구축하는 데 한계가 있음을 보여준다.

## 📌 TL;DR

본 논문은 LLM의 추론 효율성을 극대화하기 위해 중간 텍스트 출력 없이 내부적으로 계산하는 **Implicit Reasoning(암시적 추론)**을 체계적으로 분석한 서베이 논문이다. 연구진은 이를 **Latent Optimization, Signal-Guided Control, Layer-Recurrent Execution**이라는 세 가지 실행 패러다임으로 분류하고, 이에 대한 메커니즘적 증거와 평가 체계를 정리하였다. 이 연구는 향후 LLM이 '생각하는 시간'을 효율적으로 관리하고, 추론 비용을 낮추면서도 정확도를 유지하는 고성능 추론 시스템을 설계하는 데 중요한 가이드라인을 제공할 것으로 기대된다.