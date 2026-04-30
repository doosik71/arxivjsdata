# CODEEVOLVE: an open-source evolutionary framework for algorithmic discovery and optimization

Henrique Assumpção, Diego Ferreira, Leandro Campos, Fabricio Murai (2026)

## 🧩 Problem to Solve

본 논문은 자동화된 알고리즘 발견(Automated Algorithmic Discovery) 및 최적화 문제를 해결하고자 한다. 최근 LLM을 활용하여 새로운 수학적 해법이나 효율적인 알고리즘을 찾는 연구가 활발히 진행되고 있으나, Google DeepMind의 AlphaEvolve와 같은 기존의 고성능 시스템들은 대부분 폐쇄 소스(closed-source)이며, 거대 모델과 막대한 계산 자원에 의존한다는 한계가 있다. 이러한 폐쇄성은 연구의 재현성을 저해하고 접근성을 제한한다. 따라서 본 연구의 목표는 투명하고 재현 가능한 오픈 소스 프레임워크인 CODEEVOLVE를 구축하여, 상대적으로 작은 open-weight 모델들을 효율적으로 오케스트레이션함으로써 폐쇄형 모델에 필적하거나 이를 능가하는 알고리즘 발견 능력을 확보하는 것이다.

## ✨ Key Contributions

CODEEVOLVE의 핵심 아이디어는 LLM을 진화 알고리즘(Evolutionary Algorithm)의 변이 및 교차 연산자로 활용하여, 프로그램 코드와 이를 생성하는 프롬프트를 동시에 최적화하는 것이다. 특히, 탐색(Exploration)과 활용(Exploitation)의 균형을 맞추기 위해 다음의 세 가지 설계 전략을 도입하였다.

1.  **Islands-based Genetic Algorithm**: 여러 개의 독립적인 인구 집단(islands)을 유지하며 병렬적으로 탐색을 수행하고, 주기적으로 최상위 개체를 교환하는 migration 기법을 통해 다양성을 유지한다.
2.  **Modular LLM Operators**: 단순한 프롬프팅을 넘어, 조상 경로를 활용한 깊이 기반 최적화(Depth exploitation), 프롬프트 자체를 진화시키는 메타 프롬프팅(Meta-prompting), 그리고 여러 성공 사례를 참조하는 영감 기반 교차(Inspiration-based crossover) 연산자를 통해 효율적인 코드 수정을 수행한다.
3.  **Quality-Diversity Archive**: MAP-Elites 아카이브를 도입하여 단순한 성능 최적화뿐만 아니라, 다양한 특성을 가진 고성능 솔루션들을 보존함으로써 조기 수렴(premature convergence) 문제를 방지한다.

## 📎 Related Works

기존의 자동 프로그램 합성 및 알고리즘 발견 연구는 크게 유전 프로그래밍(Genetic Programming, GP)과 LLM 기반의 진화 루프(Evolution through Large Models)로 나뉜다. FunSearch와 Evolution of Heuristics(EoH)는 LLM을 진화 루프에 통합하여 인간의 기준을 넘어서는 해법을 제시하였다. 특히 AlphaEvolve는 전체 코드베이스를 진화시켜 SOTA 성능을 달성하였으나, 앞서 언급한 바와 같이 폐쇄적인 구조로 인해 세부 구현이 공개되지 않았다. 

CODEEVOLVE는 이러한 기존 접근 방식과 달리, 단순한 휴리스틱 설계를 넘어 엔드-투-엔드 프로그램 발견을 목표로 하며, 특히 프로그램의 진화와 프롬프트의 진화를 동시에 수행하는 메타 최적화(Meta-Optimization) 구조를 가진다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조
CODEEVOLVE는 Islands-based Genetic Algorithm을 기반으로 하며, 각 섬(island)은 독립적인 프롬프트 집단 $P^i_t$와 솔루션 집단 $S^i_t$를 유지한다. 전체 파이프라인은 LLM 앙상블을 통한 코드 생성, 샌드박스 내 실행 및 평가, 그리고 인구 관리 루프로 구성된다.

### 주요 구성 요소 및 작동 원리

#### 1. LLM Ensemble
가중치 기반의 LLM 앙상블을 사용하여 솔루션을 생성한다. 모델은 `SEARCH/REPLACE` 형태의 diff-based 포맷을 사용하여 기존 코드를 정밀하게 수정한다. 연구에서는 Gemini-2.5(폐쇄형)와 Qwen3-Coder-30B(오픈 가중치) 두 가지 설정을 평가하였다.

#### 2. Evolutionary Operators
탐색 확률 $p_{explr}$에 따라 다음 두 가지 경로 중 하나를 선택한다.

-   **Depth Exploitation (활용)**: 고성능 솔루션 $S$를 랭크 기반으로 선택하여 정밀하게 개선한다. 이때 솔루션 $S$, 부모 프롬프트 $P(S)$, 그리고 $k$개의 조상 솔루션 집합 $A_k(S)$를 컨텍스트로 제공하여 점진적인 개선을 유도한다. 부모 선택 확률 $\Pr(S)$는 다음과 같다.
    $$\Pr(S) := \frac{rk(S)^{-1}}{\sum_{S' \in S^i_t} rk(S')^{-1}}$$
    여기서 $rk(S)$는 적합도 $f_{sol}$ 기준 내림차순 정렬 시의 순위이다.
-   **Meta-prompting Exploration (탐색)**: 다양성을 확보하기 위해 `MetaPromptingLLM`이 기존 프롬프트 $P$와 솔루션 $S$를 분석하여 더 강화된 프롬프트 $P'$를 생성한다. 이후 이 $P'$를 사용하여 새로운 솔루션 $S'$를 생성하며, 이때는 조상 경로를 배제하여 완전히 새로운 전략을 탐색하게 한다.
-   **Inspiration-based Crossover**: LLM이 단순한 코드 결합(splicing) 대신, 여러 '영감(inspiration)' 솔루션들의 성공적인 패턴과 로직을 시맨틱하게 통합하도록 유도하여 문법적 오류 없는 교차를 수행한다.

#### 3. Exploration Scheduling 및 Population Management
-   **Plateau Scheduler**: 최상위 적합도의 개선이 정체될 경우 $p_{explr}$를 일시적으로 높여 탐색을 촉진하고, 개선이 재개되면 다시 낮추는 적응형 스케줄러를 사용한다.
-   **Elitist Migration**: 각 섬의 최상위 개체들을 주기적으로 인접 섬으로 복사하여 성공적인 솔루션을 전파한다.
-   **MAP-Elites Archive**: 솔루션의 특성(feature)에 따라 공간을 분할하고 각 셀마다 최적의 솔루션을 저장하여, 구조적으로 다양한 고성능 해법들을 유지한다.

#### 4. 적합도 함수 (Fitness Function)
솔루션의 적합도 $f_{sol}$은 실행 결과의 성능 지표를 기반으로 하며, 프롬프트의 적합도 $f_{prompt}$는 해당 프롬프트로 생성된 솔루션 중 최대 적합도로 정의된다.
$$f_{prompt}(P) = \max_{S:P(S)=P} \{f_{sol}(S)\}$$

## 📊 Results

### 실험 설정
-   **벤치마크**: AlphaEvolve 벤치마크 세트(원형/육각형 패킹, 거리 최소화, 자기상관 부등식) 및 EoH 벤치마크(Online Bin Packing, TSP, Flow Shop Scheduling)를 사용하였다.
-   **비교 대상**: AlphaEvolve(보고된 수치), ThetaEvolve, OpenEvolve, ShinkaEvolve.
-   **지표**: 각 문제별 최적화 목표값(예: 원의 반지름 합 최대화, 거리 비율 최소화 등).

### 주요 결과
-   **SOTA 달성**: CODEEVOLVE는 9개의 AlphaEvolve 벤치마크 인스턴스 중 5개에서 AlphaEvolve와 동등하거나 이를 능가하는 성능을 보였으며, 특히 `MinimizeMaxMinDist`와 `CirclePackingSquare(n=32)`에서 새로운 최적 기록을 달성하였다.
-   **Open-weight 모델의 효율성**: Qwen3-Coder-30B를 사용한 설정이 Gemini-2.5 기반 설정보다 샘플 효율성은 다소 낮았으나, 최종 성능은 비슷하거나 더 높았으며, 비용은 약 10% 수준으로 매우 저렴했다.
-   **구성 요소 분석(Ablation)**: 'Full method'가 'Naive evolution'이나 'No evolution'보다 월등한 성능과 샘플 효율성을 보였다. 특히 Depth-based refinement와 Inspiration-based crossover 간의 긍정적인 시너지가 확인되었다. 또한 MAP-Elites 아카이브와 Cycle 토폴로지의 migration이 SOTA 달성에 필수적임이 밝혀졌다.

## 🧠 Insights & Discussion

본 연구는 알고리즘 발견 과정에서 모델의 절대적인 크기(scale)보다, 모델을 어떻게 오케스트레이션하느냐(orchestration)가 더 중요하다는 점을 시사한다. 특히 잘 설계된 진화 프레임워크 내에서는 상대적으로 작은 오픈 소스 모델이 비용 효율적으로 폐쇄형 거대 모델의 성능을 재현하거나 추월할 수 있음을 입증하였다.

**강점 및 한계:**
-   **강점**: 투명한 오픈 소스 프레임워크를 제공하여 재현성을 높였으며, 프롬프트와 코드를 동시에 진화시키는 메타 최적화 구조를 통해 탐색 능력을 극대화하였다.
-   **한계**: 여전히 대규모 진화 탐색에는 상당한 추론 비용이 발생하며, migration 토폴로지나 조상 깊이($k$)와 같은 하이퍼파라미터 튜닝에 민감한 부분이 존재한다. 또한, 메타 프롬프팅 단계에는 고성능 모델을 쓰고 실행 단계에는 작은 모델을 쓰는 '이종 오케스트레이션(heterogeneous orchestration)'의 가능성이 남아있다.

## 📌 TL;DR

CODEEVOLVE는 Islands-based GA와 LLM 앙상블을 결합한 오픈 소스 알고리즘 발견 프레임워크이다. 메타 프롬프팅, 깊이 기반 최적화, MAP-Elites 아카이브 등의 모듈형 연산자를 통해 탐색과 활용의 균형을 맞췄으며, 이를 통해 Qwen3와 같은 오픈 가중치 모델만으로도 AlphaEvolve와 같은 폐쇄형 SOTA 시스템에 필적하는 성능을 매우 낮은 비용으로 달성하였다. 이 연구는 향후 자동화된 과학적 발견 및 알고리즘 최적화 연구를 위한 민주적이고 효율적인 기반 도구가 될 가능성이 높다.