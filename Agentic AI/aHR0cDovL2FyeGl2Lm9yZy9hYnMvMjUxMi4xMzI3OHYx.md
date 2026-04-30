# AUTOTOOL: DYNAMIC TOOL SELECTION AND INTEGRATION FOR AGENTIC REASONING

Jiaru Zou, Ling Yang, Yunzhe Qi, Sirui Chen, Mengting Ai, Ke Shen, Jingrui He, Mengdi Wang (2025)

## 🧩 Problem to Solve

본 논문은 LLM 에이전트가 외부 도구를 사용하여 복잡한 추론을 수행할 때, **고정된 도구 세트(Fixed Tool Inventory)**에 의존하는 기존 방식의 한계를 해결하고자 한다. 기존의 에이전트 강화학습 접근 방식은 특정 작업에 대해 미리 정의된 정적인 도구들만을 사용하도록 학습된다. 이러한 방식은 다음과 같은 두 가지 주요 문제를 야기한다.

첫째, 에이전트가 복잡하고 도메인이 다양한 도구 세트 중에서 현재 상황에 가장 적합한 도구를 스스로 선택하는 능력이 부족하다. 둘째, 학습 단계에서 보지 못한 새로운 도구(Unseen Tools)가 추론 시점에 도입될 경우, 모델이 이를 활용하지 못하고 기존에 학습된 도구 세트에 과적합(Overfitting)되어 일반화 능력이 떨어진다. 

결과적으로 본 논문의 목표는 LLM 에이전트가 추론 과정에서 **동적으로 도구를 선택(Dynamic Tool Selection)**할 수 있는 능력을 갖추게 하여, 진화하는 도구 환경(Evolving Tool Environment)에서도 강력한 일반화 성능을 유지하도록 하는 프레임워크인 AutoTool을 제안하는 것이다.

## ✨ Key Contributions

AutoTool의 핵심 아이디어는 도구 선택 과정을 단순한 분류 문제가 아닌, **임베딩 공간에서의 정렬(Representation Alignment)과 랭킹 최적화 문제**로 재정의하는 것이다. 주요 기여 사항은 다음과 같다.

1.  **대규모 동적 도구 사용 데이터셋 구축**: 1,000개 이상의 도구와 100개 이상의 작업(수학, 과학, 코드 생성, 멀티모달 추론 등)을 포괄하는 200k 규모의 데이터셋을 구축하였다. 특히, 도구를 선택한 이유에 대한 명시적인 근거(Rationale)를 포함하여 모델이 논리적으로 도구를 선택하도록 유도하였다.
2.  **Embedding-Anchored Tool Selection**: 도구의 이름을 직접 생성하는 대신, 도구의 설명과 기능을 포함한 임베딩 벡터를 생성하고, 모델이 예측한 앵커 토큰의 임베딩과 가장 가까운 도구를 선택하는 방식을 도입하였다. 이를 통해 학습 시 보지 못한 새로운 도구가 추가되더라도 임베딩 유사도를 통해 유연하게 대응할 수 있다.
3.  **이단계 최적화 파이프라인(Dual-phase Optimization)**: 
    *   **Phase I (Trajectory Stabilization)**: SFT와 RL(GRPO 등)을 통해 안정적인 긴 사고 사슬(CoT) 생성 및 도구 통합 능력을 확보한다.
    *   **Phase II (Tool-Selection Refinement)**: Plackett-Luce (PL) Ranking 최적화를 통해 도구 선택의 정확도와 일관성을 정교하게 다듬는다.

## 📎 Related Works

기존의 도구 사용 연구는 크게 세 가지 방향으로 나뉜다.
1.  **고정 도구 통합(Fixed Tool Integration)**: 특정 도메인에 특화된 도구 사용법을 SFT나 RL로 학습시키는 방식이다. 하지만 이는 폐쇄된 환경에서는 효과적이나, 새로운 도구가 도입되는 개방형 환경에서는 일반화 능력이 현저히 떨어진다.
2.  **리트리벌 기반 선택(Retrieval-based Selection)**: 외부 리트리버를 사용하여 쿼리에 맞는 도구를 찾는 방식이다. 그러나 리트리버가 LLM과 독립적으로 작동하므로, LLM의 내부 표현(Internal Representation)이 도구 선택 과정에 직접적으로 최적화되지 않는다는 한계가 있다.
3.  **도구 생성(Tool Generation)**: 필요한 도구가 없을 때 코드를 생성하여 도구를 직접 만드는 방식이다. 이는 유연하지만, 실행 및 검증 과정에서 많은 오버헤드가 발생하며 디버깅이 어렵다는 단점이 있다.

AutoTool은 이러한 한계를 극복하기 위해 LLM의 내부 임베딩과 도구 임베딩을 직접 정렬시키고, PL 랭킹을 통해 선택 전략을 최적화함으로써 효율성과 일반화 능력을 동시에 확보하였다.

## 🛠️ Methodology

### 1. 데이터 큐레이션 파이프라인
AutoTool은 다음의 3단계 과정을 통해 200k의 데이터셋을 구축한다.
*   **Toolset & Task Collection**: 코드, 검색, 이미지 도구 등 1,000개 이상의 도구와 이에 대응하는 100개 이상의 작업을 수집한다.
*   **Tool-Selection Rationale Generation**: DeepSeek-R1과 같은 전문가 모델을 사용하여, 특정 도구를 선택한 이유에 대한 명시적인 근거(Rationale)를 생성한다.
*   **Trajectory Augmentation**: 생성된 근거를 원래의 추론 궤적에 삽입하고, LLM-as-a-judge를 통해 무효한 근거를 필터링한 뒤 전체 궤적을 매끄럽게 다듬는다.

### 2. 도구 인식 궤적 생성 (Tool-Aware Trajectory Generation)
에이전트의 전체 궤적 $\tau$는 내부 추론($\tau_{reason}$), 도구 선택($\tau_{select}$), 도구 통합($\tau_{integrate}$) 단계의 반복으로 구성된다.

**임베딩 기반 도구 선택:**
각 도구 $t_k$에 대해 기능 설명 $\mu(t_k)$를 포함한 임베딩 $e_{t_k}$를 생성한다.
$$e_{t_k} = \text{Emb}_{\pi_\theta}[t_k, \mu(t_k)]$$
모델은 선택 근거를 생성한 후 앵커 토큰의 임베딩 $e'_i$를 예측하며, 최종 도구 $t_k$는 다음과 같은 소프트맥스 정규화된 거리 분포를 통해 선택된다.
$$\pi_\theta(t_k | x, \tau_{<i}, s_i, T) = \frac{\exp(-\gamma \|e'_i - e_{t_k}\|_F^2)}{\sum_{t_j \in T} \exp(-\gamma \|e'_i - e_{t_j}\|_F^2)}$$
여기서 $\| \cdot \|_F$는 Frobenius norm이며, $\gamma$는 분포의 기울기를 조절하는 하이퍼파라미터이다.

### 3. 이단계 정책 학습 파이프라인
**Phase I: Trajectory Stabilization**
기본 참조 모델 $\pi_{\theta_{ref}}$에서 시작하여 SFT와 RL(Policy Optimization)을 적용한다. 이 단계의 목표는 모델이 내부 추론 $\rightarrow$ 도구 선택 $\rightarrow$ 도구 통합으로 이어지는 안정적인 CoT 패턴을 학습하게 하는 것이다.

**Phase II: Tool-Selection Refinement**
내부 추론과 통합 단계는 마스킹하고, **도구 선택 단계**만을 집중적으로 최적화한다. KL-정규화된 RL 목적 함수를 사용하여 최적화한다.
$$\max_{\pi_\theta} \mathbb{E}_{\tau \sim \pi_\theta} [R_{tool}(\tau)] - \beta \cdot D_{KL}(\pi_\theta(\cdot | x, T) \| \pi_{old}(\cdot | x, T))$$
여기서 $R_{tool}(\tau)$는 도구 선택의 근거 품질(PRM 사용)과 최종 정답의 정확도($\text{Acc}$)를 합산하여 계산한 보상이다.

**PL Ranking 최적화:**
본 논문은 도구 선택 문제를 랭킹 문제로 변환하기 위해 Plackett-Luce (PL) 모델을 도입한다. 궤적들의 집합 $\mathcal{T}$에 대해 보상이 높은 궤적이 상위에 배치될 확률을 모델링한다.
$$P_{\pi_\theta}(\sigma | \mathcal{T}) = \prod_{j=1}^N \frac{\exp(R_{tool}(\tau_{\sigma(j)}))}{\sum_{l=j}^N \exp(R_{tool}(\tau_{\sigma(l)}))}$$
이론적으로 최적 정책 $\pi^*$를 학습하는 것은 PL 랭킹 분포를 일치시키는 것과 동일함을 증명(Proposition 3.1)하고, 이를 통해 계산 가능한 Cross-Entropy (CE) 손실 함수로 최적화를 수행한다.
$$L_{CE} = -\mathbb{E}_{x \sim \mathcal{D}} \sum_{\tau \in \mathcal{T}} \pi^*(\tau | x, T) \log \pi_\theta(\tau | x, T)$$

## 📊 Results

### 실험 설정
*   **모델**: Qwen3-8B, Qwen2.5-VL-7B
*   **벤치마크**: 
    *   수학/과학: AIME24, AIME25, GPQA-Diamond
    *   검색 기반 QA: HotpotQA, 2Wiki, Bamboogle
    *   멀티모달: MMSearch, V-Chart, V-Math, V-Code
*   **비교 대상**: GPT-4o, Qwen2.5-VL-72B, ReTool, Search-R1 등

### 주요 결과
1.  **도메인 간 균형 잡힌 성능**: 표 1에 따르면, 특정 도메인에 특화된 모델(예: ReTool은 수학에 강함, Search-R1은 검색에 강함)과 달리 AutoTool은 모든 도메인에서 고르게 높은 성능을 보였다. 특히 AutoTool (Qwen3-8B)는 7개 벤치마크 중 6개에서 최고 성능을 달성하였다.
2.  **표준 학습 패러다임 대비 우위**: SFT 및 GRPO만 적용했을 때보다 Phase II의 PL 랭킹 최적화를 추가한 AutoTool의 성능이 일관되게 높았다. (예: Qwen3-8B의 AIME24 점수가 GRPO 대비 6.7% 향상)
3.  **Oracle Tool Assignment와의 비교**: 정답 도구를 미리 알려준 'Oracle' 설정과 비교했을 때, AutoTool은 매우 근소한 차이의 성능을 보였다. 이는 AutoTool의 동적 도구 선택 능력이 거의 정답 수준에 도달했음을 의미한다.
4.  **일반화 능력**: 학습 시 보지 못한 886개의 도구가 포함된 환경에서도 성공적으로 도구를 선택하여 작업을 수행함으로써 강력한 일반화 성능을 입증하였다.

## 🧠 Insights & Discussion

본 연구는 LLM 에이전트의 도구 사용 능력을 향상시키기 위해 **'무엇을 사용할 것인가'에 대한 논리적 근거(Rationale)**와 **임베딩 기반의 유연한 선택 메커니즘**을 결합한 것이 주효했다.

**강점:**
*   도구 이름을 직접 생성하지 않고 임베딩 공간에서의 거리로 선택함으로써, 새로운 도구가 추가되어도 모델을 다시 학습시킬 필요 없이 임베딩 벡터만 추가하면 되는 확장성을 확보하였다.
*   PL 랭킹을 통해 보상 기반의 궤적 선호도를 모델의 정책에 직접 반영함으로써, 단순한 정답 맞추기를 넘어 효율적인 도구 선택 전략을 학습시켰다.

**한계 및 논의사항:**
*   임베딩 기반 선택 방식의 성능은 초기 도구 임베딩 생성 모델의 품질에 의존할 가능성이 크다.
*   현재는 텍스트 기반의 도구 설명($\mu(t_k)$)을 사용하지만, 도구의 실제 동작 특성을 더 잘 반영할 수 있는 동적 임베딩 방식에 대한 연구가 추가적으로 필요할 것으로 보인다.

## 📌 TL;DR

AutoTool은 LLM 에이전트가 고정된 도구 세트를 넘어, **진화하는 환경에서 동적으로 도구를 선택하고 통합**할 수 있게 하는 프레임워크이다. 명시적인 선택 근거가 포함된 200k 데이터셋과 **임베딩 기반 선택 방식**, 그리고 **PL 랭킹 최적화**를 통해, 소형 모델(7B~8B)임에도 불구하고 수학, 검색, 멀티모달 등 다양한 도메인에서 GPT-4o와 같은 거대 모델이나 도메인 특화 에이전트보다 균형 잡힌 고성능과 강력한 일반화 능력을 보여주었다. 이 연구는 향후 확장 가능한 AI 에이전트를 구축하는 데 있어 핵심적인 방법론을 제시한다.