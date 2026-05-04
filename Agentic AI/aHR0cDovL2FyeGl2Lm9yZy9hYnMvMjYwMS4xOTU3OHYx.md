# YUNQUE DEEPRESEARCH TECHNICAL REPORT

Yuxuan Cai, Xinyi Lai, Peng Yuan, Weiting Liu, Huajian Li, Mingda Li, Xinghua Wang, Shengxie Zheng, Yanchao Hao, Yuyang Yin, Zheng Wei (2026)

## 🧩 Problem to Solve

본 논문은 자율 에이전트가 복잡하고 개방형인 심층 연구(Deep Research) 과제를 수행할 때 직면하는 세 가지 핵심적인 한계를 해결하고자 한다.

첫째, **장기 과제에서의 인지 과부하(Cognitive Overload in Long-Horizon Tasks)** 문제이다. 기존의 ReAct 기반 에이전트들은 수백 단계의 상호작용 과정에서 발생하는 원시 실행 로그(raw execution logs)를 그대로 누적한다. 이는 컨텍스트 노이즈를 증가시켜 원래의 사용자 의도를 희석시키고 추론 성능을 저하시킨다.

둘째, **시스템의 취약성과 연쇄적 오류(Systemic Fragility and Cascading Failures)** 문제이다. 견고한 오류 탐지 및 복구 메커니즘이 부족하여, 작은 실수가 발생했을 때 시스템이 최적이지 않은 재귀적 루프에 빠지며 전체 작업이 실패하는 연쇄적 오류가 발생한다.

셋째, **모듈식 확장성의 부족(Lack of Modular Extensibility)** 문제이다. 경직된 단일 아키텍처는 다양한 도구나 도메인 특화 서브 에이전트를 유연하게 통합하는 것을 방해하며, 이는 복잡해지는 연구 환경에 대한 적응력을 떨어뜨린다.

결과적으로 본 논문의 목표는 이러한 한계들을 극복하여, 장기적인 추론 일관성을 유지하고 오류에 강인하며 확장 가능한 계층적 모듈 구조의 **Yunque DeepResearch** 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

Yunque DeepResearch의 핵심 아이디어는 **전략적 계획과 세부 실행의 분리**, 그리고 **의미 단위의 동적 메모리 관리**에 있다.

1. **계층적 오케스트레이션(Hierarchical Orchestration):** 중앙의 Main Agent가 전략적 핵심 역할을 수행하며, 단순 작업은 기본 도구(Basic Tools)로, 복잡한 작업은 전문 서브 에이전트(Specialized Sub-agents)로 동적으로 라우팅하는 구조를 설계하였다.
2. **서브 목표 기반의 동적 컨텍스트 관리(Dynamic Context Management):** 전체 경로를 단순히 나열하는 대신, '서브 목표(Sub-goal)'를 기본 단위로 하여 완료된 작업은 구조화된 요약본으로 압축하고, 현재 진행 중인 작업만 세부 로그를 유지함으로써 정보 밀도를 극대화하였다.
3. **능동적 감독 및 자가 교정(Proactive Supervisor Module):** 에이전트의 궤적을 실시간 모니터링하여 이상 징후를 감지하고, 실패 시 컨텍스트를 가지치기(Pruning)하고 다시 생성하는 복구 프로토콜을 도입하여 시스템의 견고함을 높였다.
4. **Atomic Capability Pool 구축:** GUI 상호작용을 위한 Browser-Use GUI Agent와 데이터 처리를 위한 Data Analysis Agent를 모듈화하여, 일반적인 LLM이 가지지 못한 정밀한 실행 능력을 부여하였다.

## 📎 Related Works

논문은 기존의 Deep Research 에이전트를 크게 두 가지 패러다임으로 분류하여 설명한다.

- **단일 에이전트(Single-Agent) 패러다임:** OpenAI와 Gemini의 Deep Research 시스템 등이 이에 해당하며, 단일 제어 아키텍처를 통해 도구를 오케스트레이션한다. 하지만 이러한 방식은 컨텍스트 윈도우가 확장됨에 따라 인지적 포화 상태에 이르며, 장기 추론의 일관성을 유지하는 데 '신뢰성 병목 현상(reliability bottlenecks)'이 발생한다는 한계가 있다.
- **다중 에이전트(Multi-Agent) 패러다임:** 작업을 세분화하여 플래너, 검색기, 비판자 등 서로 다른 역할을 가진 에이전트들이 협업하는 방식이다. 이는 단일 에이전트보다 오류 전파를 효과적으로 완화할 수 있다는 장점이 있다.

또한, **작업 메모리 관리(Working Memory Management)** 측면에서 ReSum, MemAgent, AgentFold 등의 연구가 언급된다. 기존 방식들은 주로 컨텍스트 제한에 도달했을 때 내용을 압축하는 '콘텐츠 압축'에 집중했으나, Yunque DeepResearch는 단순 압축을 넘어 서브 목표 중심의 '구조적 합성(structured synthesis)'을 통해 정보 밀도를 높였다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

Yunque DeepResearch는 크게 네 가지 모듈로 구성된다:

- **Main Agent:** 의도 인식, 동적 계획 수립 및 전체 오케스트레이션을 담당하는 중앙 집행부이다.
- **Context Manager:** 장기 과제 수행을 위해 즉각적인 정밀도와 장기적 전략 컨텍스트 사이의 균형을 맞추는 이중 레벨 메모리 구조를 관리한다.
- **Atomic Capability Pool:** 전문 서브 에이전트(GUI, 데이터 분석)와 기본 도구들을 보유한 실행 집합이다.
- **Supervisor:** 실행 궤적을 감시하고 오류를 교정하여 연쇄적 실패를 방지하는 안전장치이다.

### 2. 메모리 및 컨텍스트 관리 (Memory)

본 프레임워크는 메모리 단위를 다음과 같은 4-튜플로 정의한다:
$$m_i = (R_i, g_i, T_i, s_i)$$
여기서 $R_i$는 해당 서브 목표에 기여한 라운드 인덱스 리스트, $g_i$는 서브 목표의 의미적 설명, $T_i$는 도구 사용 로그, $s_i$는 실행 중 추출된 핵심 정보의 증분 요약(incremental summary)이다.

**동적 폴딩 및 추가 메커니즘(Dynamic Folding and Adding):**
메모리 모델 $F_{mem}$은 현재 라운드의 응답 $a_t$, 관찰 $o_t$, 그리고 최신 메모리 유닛 $m_n$을 입력받아 이진 지표 $\delta_{fold}$와 업데이트된 유닛 $m_{out}$을 생성한다:
$$(m_{out}, \delta_{fold}) = F_{mem}(a_t, o_t, m_n)$$

- $\delta_{fold} = 1$인 경우: 현재 라운드가 기존 서브 목표 $g_n$과 일치한다고 판단하여 기존 메모리 유닛을 업데이트한다.
- $\delta_{fold} = 0$인 경우: 새로운 서브 목표가 시작된 것으로 판단하여 새로운 메모리 유닛을 리스트 $M$에 추가한다.

**적응형 컨텍스트 구성(Adaptive Context Construction):**
컨텍스트 $C_t$는 현재 메모리 유닛의 라운드 수 $|R_n|$에 따라 다르게 구성된다:
$$C_t = \begin{cases} C_{t-1} \oplus [r_t, o_t] & \text{if } |R_n| > 1 \\ (Q, M_{1:n-1}) \oplus [r_t, o_t] & \text{if } |R_n| = 1 \end{cases}$$
즉, 서브 목표를 수행 중일 때는 세부적인 ReAct 트레이스를 유지하고, 새로운 서브 목표가 시작되는 시점($|R_n|=1$)에는 과거의 모든 기록을 구조화된 요약본 $M_{1:n-1}$으로 대체하여 컨텍스트를 리셋한다. 이를 통해 복잡도를 $O(t)$에서 $O(n)$(서브 목표 개수)으로 낮춘다.

### 3. Atomic Capability Pool

- **Browser-Use GUI Agent:** 웹 브라우저 상호작용을 POMDP(Partially Observable Markov Decision Process)로 모델링한다. 관측값 $o_t$는 텍스트 컨텍스트 $c_t$, 구조화된 브라우저 상태 $b_t$, 그리고 현재 페이지의 스크린샷 $x_t$로 구성된다. 스크린샷은 멀티모달 입력으로만 사용되고 텍스트 히스토리에는 저장하지 않아 컨텍스트 폭발을 방지한다.
- **Data Analysis Agent:** (a) 데이터 프로파일링(메타데이터, 스키마, 프리뷰 추출) $\rightarrow$ (b) 다단계 추론 및 자가 개선(Python 코드 생성 $\rightarrow$ 샌드박스 실행 $\rightarrow$ 피드백 기반 수정)의 파이프라인으로 동작한다.

### 4. Supervisor 및 자가 교정 메커니즘

Supervisor는 에이전트가 구문 오류나 의미적 정체(Semantic Stagnation, 예: 무한 루프)에 빠졌는지 감시한다. 이상 징후 감지 시, 시스템을 'Acting Mode'에서 'Reflective Mode'로 강제 전환하며 다음 3단계 복구 프로토콜을 수행한다:

1. **Anomaly Diagnosis:** 실패의 근본 원인을 분석한다.
2. **Trajectory Pruning:** 컨텍스트 윈도우에서 최근의 잘못된 상호작용 트레이스를 명시적으로 제거하여 메모리 오염을 막는다.
3. **Re-generation:** 수정된 계획이나 결론을 다시 생성하여 루프를 끊고 정상 경로로 복귀시킨다.

## 📊 Results

### 실험 설정

- **벤치마크:** GAIA, BrowseComp, BrowseComp-ZH, Humanity’s Last Exam (HLE).
- **평가 지표:** Pass@1 (단일 시도 성공률).
- **기본 모델:** Gemini-3-pro를 기본 백본으로 사용하였으며, 비교군으로 GPT-5 High, OpenAI-o3, Claude-4.5-Sonnet 및 다양한 오픈소스/폐쇄형 에이전트 프레임워크를 설정하였다.

### 주요 결과

Yunque DeepResearch는 BrowseComp(62.5), BrowseComp-ZH(75.9), HLE(51.7)에서 SOTA(State-of-the-art) 성능을 달성하였으며, GAIA(78.6)에서는 2위를 기록하였다. 특히 단순 ReAct 방식의 Gemini 3 Pro와 비교했을 때 BrowseComp에서 +10.0, GAIA에서 +4.8의 성능 향상을 보여, 프레임워크 설계가 모델의 잠재력을 효과적으로 끌어올렸음을 입증하였다.

### 분석 및 절제 연구(Ablation Study)

- **메모리 모듈의 영향:** 메모리 모듈 제거 시 BrowseComp에서 -10.4의 큰 하락이 발생하였다. 이는 장기 정보 탐색 과제에서 서브 목표 기반의 메모리 관리가 노이즈 제거에 결정적임을 시사한다.
- **Supervisor의 영향:** 제거 시 GAIA(-8.7), BrowseComp-ZH(-10.5) 등 전반적인 성능이 크게 하락하였다. 이는 오류 누적을 막고 경로를 정화하는 Supervisor의 역할이 복잡한 과제 수행의 필수 조건임을 보여준다.
- **전문 에이전트의 영향:** GAIA 벤치마크에서 Browser-Use GUI Agent와 Data Analysis Agent를 제거했을 때 각각 -6.8, -2.9의 성능 저하가 나타났다. 이는 범용 어시스턴트의 성능이 결국 도메인 특화된 정밀한 실행 능력의 조합에서 나온다는 것을 확인시켜 준다.

## 🧠 Insights & Discussion

본 논문의 강점은 단순히 LLM의 컨텍스트 윈도우 크기에 의존하지 않고, **'의미적 단위(Sub-goal)'**를 통해 메모리를 구조화하여 정보 밀도를 능동적으로 관리했다는 점이다. 이는 LLM이 흔히 겪는 'Lost-in-the-middle' 현상을 방지하고, 장기 과제에서도 전략적 일관성을 유지하게 한다.

또한, Supervisor 모듈의 **'컨텍스트 가지치기(Trajectory Pruning)'** 개념은 매우 실용적인 접근이다. 많은 에이전트들이 단순히 '다시 생각하라(Reflection)'고 지시하지만, 이미 오염된 컨텍스트 내에서는 동일한 오류를 반복할 가능성이 높다. 잘못된 경로를 물리적으로 제거함으로써 에이전트가 편향되지 않은 상태에서 새로운 해결책을 찾도록 유도한 점이 성능 향상의 주요 요인으로 분석된다.

다만, 한계점으로는 전문 서브 에이전트에 대한 절제 연구가 주로 GAIA에 집중되어 있어, 다른 도메인에서의 일반화 성능 검증이 더 필요하다는 점과, 전체적인 토큰 소비량 및 추론 지연 시간(Latency)에 대한 정량적 분석이 부족하다는 점이 언급되었다.

## 📌 TL;DR

Yunque DeepResearch는 장기 연구 과제 수행 시 발생하는 인지 과부하, 시스템 취약성, 확장성 부족 문제를 해결하기 위해 제안된 **계층적·모듈형 에이전트 프레임워크**이다. 서브 목표 기반의 동적 메모리 압축, 전문 서브 에이전트 풀, 그리고 능동적 오류 교정을 수행하는 Supervisor 모듈을 통해 GAIA, BrowseComp, HLE 등 주요 벤치마크에서 SOTA급 성능을 달성하였다. 이 연구는 향후 복잡한 실세계 워크플로우를 자동화하는 자율 에이전트 설계에 있어, 단순한 모델 스케일업보다 **구조적인 컨텍스트 관리와 모듈형 실행 체계**가 더 중요하다는 점을 시사한다.
