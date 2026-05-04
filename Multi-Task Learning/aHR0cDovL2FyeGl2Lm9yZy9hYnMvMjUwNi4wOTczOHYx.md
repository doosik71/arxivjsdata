# Towards Multi-modal Graph Large Language Model

Xin Wang, Zeyang Zhang, Linxin Xiao, Haibo Chen, Chendi Ge and Wenwu Zhu (2025)

## 🧩 Problem to Solve

현대의 많은 실세계 애플리케이션에서는 텍스트, 이미지, 오디오, 비디오 등 다양한 모달리티(modality)의 특징과 관계가 결합된 Multi-modal Graph 데이터가 널리 사용되고 있다. 그러나 기존의 multi-modal graph learning 방법론들은 대개 특정 그래프 데이터와 특정 작업(task)을 위해 처음부터(from scratch) 학습되는 경향이 있다.

이러한 특수화(specialization)는 모델의 범용성을 제한하며, 서로 다른 종류의 multi-modal graph 데이터나 새로운 작업에 직면했을 때 효과적으로 일반화(generalization)되지 못하는 문제를 야기한다. 결과적으로 새로운 시나리오가 등장할 때마다 모델을 재설계하고 재학습시켜야 하는 비효율성이 발생한다. 본 논문의 목표는 다양한 multi-modal graph 데이터와 작업 전반에 걸쳐 통합 및 일반화가 가능한 Multi-modal Graph Large Language Model (MG-LLM)의 가능성을 탐색하고, 이를 실현하기 위한 통합 프레임워크와 연구 로드맵을 제시하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 거대 언어 모델(LLM)이 자연어 처리의 다양한 작업을 통합한 것처럼, MG-LLM이 multi-modal graph의 복잡한 데이터와 작업을 통합하는 패러다임이 될 수 있다는 점이다. 주요 기여 사항은 다음과 같다.

1.  **통합 프레임워크 제안**: multi-modal graph 데이터, 작업, 모델을 바라보는 통합된 관점을 제시하며, 특히 multi-modal graph가 가진 고유한 특성인 multi-granularity(다양한 입도)와 multi-scale(다양한 규모) 특성을 규명하였다.
2.  **MG-LLM의 5가지 핵심 특성 정의**: 성공적인 MG-LLM이 갖추어야 할 다섯 가지 필수 역량(통합 공간, 다양한 작업 처리 능력, In-context Learning, 자연어 상호작용, 복합 추론)을 정의하였다.
3.  **연구 로드맵 제시**: 위 특성들을 달성하기 위해 해결해야 할 핵심 도전 과제와 미래 연구 방향을 상세히 논의하였다.
4.  **데이터셋 벤치마크 정리**: MG-LLM의 학습과 평가에 활용될 수 있는 기존의 다양한 multi-modal graph 데이터셋을 체계적으로 정리하여 제공하였다.

## 📎 Related Works

논문에서는 기존의 접근 방식을 크게 두 가지 방향으로 구분하여 설명하며, 제안하는 MG-LLM과의 차별점을 강조한다.

1.  **Multi-modal Graph Neural Networks (MGNNs)**: 
    *   **특징**: 모달리티별 인코더를 통해 특징을 추출한 뒤, GNN의 메시지 패싱 메커니즘을 통해 구조적 정보를 학습한다.
    *   **한계**: 입력과 출력 공간이 고정되어 있어 범용적인 파운데이션 모델로 확장하기 어렵고, 후기 융합(late fusion) 방식의 특성상 세밀한 모달리티 간 상호작용을 포착하는 데 한계가 있다.
2.  **Graph Large Language Models (GraphLLMs)**:
    *   **특징**: 그래프를 텍스트로 기술(textualization)하여 LLM에 입력하거나, 파라미터화된 프로젝터를 통해 그래프 데이터를 LLM의 토큰 공간으로 매핑한다.
    *   **한계**: 그래프를 텍스트로 변환할 때 정보 손실이 발생하며, 복잡한 그래프 구조를 텍스트로 표현할 경우 컨텍스트 길이가 지나치게 길어져 모델의 이해도가 떨어진다. 또한, 현재의 GraphLLM들은 multi-modal graph 문제를 직접적으로 다루는 경우가 드물다.

본 연구는 단순히 기존 모델을 결합하는 것을 넘어, multi-modal graph의 원시 데이터(native data)를 직접 처리할 수 있는 통합 모델의 비전을 제시한다는 점에서 기존 연구들과 차별화된다.

## 🛠️ Methodology

### 1. Multi-modal Graph 데이터의 통합 정식화
본 논문은 multi-modal graph를 다음과 같은 4-튜플(quadruple)로 정의한다.
$$G = (V, E, F, M)$$
여기서 $V$는 정점의 집합, $E$는 간선의 집합, $M$은 사용된 모든 모달리티의 집합이며, $F = \{F_m\}_{m=1}^M$은 각 모달리티 특성 공간을 공유 표현 공간 $X$로 매핑하는 함수들의 집합이다.

데이터의 구성 방식에 따라 세 가지 분해 가능한(decomposable) 유형으로 분류한다.
*   **Feature-level**: 정점이나 간선의 특징이 여러 모달리티의 결합으로 이루어진 경우. (예: 상품 노드가 '제목(텍스트)'과 '이미지'를 동시에 가짐)
*   **Node-level**: 정점 자체가 서로 다른 모달리티에서 온 경우. (예: 어떤 노드는 이미지이고, 어떤 노드는 텍스트 설명임)
*   **Graph-level**: 그래프 전체가 서로 다른 모달리티로 구성된 경우.

### 2. 생성적 모델링을 통한 작업의 통합
모든 multi-modal graph 작업을 **'입력 그래프 $G_{in}$이 주어졌을 때 출력 그래프 $G_{out}$을 생성하는 문제'**로 정의한다.
$$\text{Objective: } P(G_{out} | G_{in}; \Theta)$$
이 관점을 통해 다음과 같은 다양한 작업을 하나의 생성적 패러다임으로 통합할 수 있다.
*   **Node Classification (NC)**: 타겟 노드 주변의 서브그래프를 입력받아 클래스 레이블을 담은 그래프를 생성.
*   **Link Prediction (LP)**: 두 노드와 인접 영역을 입력받아 연결 여부나 속성을 담은 그래프를 생성.
*   **Graph Question Answering (GQA)**: 그래프와 쿼리 노드를 입력받아 정답 그래프(텍스트/이미지 노드 등)를 생성.
*   **Graph Reasoning (GR)**: 복잡한 다단계 추론 경로가 포함된 결과 그래프를 생성.

### 3. 통합 모델 뷰 (Unified Model View)
모든 multi-modal graph 모델은 입력 변환, 핵심 모델링, 출력 변환의 세 단계로 추상화할 수 있다.
$$G_{out} = T_{out}(\phi_{\theta}(T_{in}(G_{in})))$$
*   $T_{in}$: 입력 그래프를 모델이 처리 가능한 형태로 변환하는 함수 (예: 이미지 $\rightarrow$ 텍스트).
*   $\phi_{\theta}$: 핵심 MG-LLM 모델.
*   $T_{out}$: 모델의 출력을 최종 원하는 형태(레이블, 이미지 등)로 변환하는 함수.

논문은 궁극적으로 $T$가 항등 함수(identity mapping)가 되어 정보 손실 없이 데이터를 직접 처리하는 **Native MG-LLM**을 지향하지만, 현실적인 한계로 인해 기능별 모듈을 통합한 **Modular MG-LLM** 구조가 실용적인 대안이 될 것이라고 주장한다.

## 📊 Results

본 논문은 새로운 모델의 성능을 측정하는 실험 논문이 아니라, 분야의 방향성을 제시하는 **Position Paper/Roadmap** 성격의 논문이다. 따라서 특정 알고리즘의 수치적 결과 대신, MG-LLM 구현을 위해 필요한 **데이터셋의 체계적 분석 결과**를 제시한다.

저자들은 NC, LP, GC, GQA, GR, TG, IG의 7가지 작업별로 활용 가능한 데이터셋을 분류하였다.
*   **Node Classification**: ELE Fashion (이커머스), Pan-Cancer Atlas (바이오메디컬) 등.
*   **Link Prediction**: Books LP, TIVA-KG (멀티미디어 지식 그래프) 등.
*   **Graph Reasoning**: MARS & MarKG (시각-텍스트 유사 추론) 등.
*   **Text/Image Generation**: Richpedia (텍스트 생성), ART500K (예술 작품 이미지 생성) 등.

이러한 정리는 MG-LLM을 학습시키기 위해 어떤 데이터가 필요하며, 현재 데이터의 양이 LLM의 사전 학습 데이터에 비해 턱없이 부족하다는 점을 정량적으로 시사한다.

## 🧠 Insights & Discussion

### 강점 및 통찰
본 논문은 단순한 모델 제안을 넘어 multi-modal graph가 가진 **Multi-granularity(입도)** 문제를 깊이 있게 다루었다. 픽셀/단어 같은 미세 입도부터 이미지/문서 같은 조립 입도까지 하나의 그래프 내에 공존한다는 점을 지적하며, 이를 해결하기 위한 **통합 보카블러리(Unified Vocabulary)와 토크나이저**의 필요성을 역설한 점이 매우 날카로운 분석이다.

### 한계 및 논의사항
1.  **데이터 부족 문제**: LLM의 성공 요인은 방대한 데이터였으나, multi-modal graph 데이터는 이에 비해 매우 적다. 이를 해결하기 위해 인터넷 전체를 그래프화하는 등의 극단적인 데이터 수집 전략이 필요할 수 있다.
2.  **Semantic Gap**: 자연어의 모호함과 그래프 구조의 엄격함 사이의 간극을 어떻게 메울 것인가에 대한 구체적인 아키텍처 설계안은 아직 부족하며, 이는 향후 연구의 핵심 과제가 될 것이다.
3.  **환각(Hallucination)**: 기존 MLLM에서 발생하는 환각 문제가 그래프 구조와 결합했을 때 어떻게 증폭되거나 완화될 수 있을지에 대한 심층적인 논의가 필요하다.

## 📌 TL;DR

본 논문은 특정 작업에 국한된 기존 multi-modal graph 학습의 한계를 극복하기 위해, 범용적으로 일반화 가능한 **Multi-modal Graph Large Language Model (MG-LLM)**의 비전과 프레임워크를 제시한다. 모든 그래프 작업을 생성적 관점($P(G_{out}|G_{in})$)에서 통합하고, Native/Modular 모델 구조를 통해 데이터 손실을 최소화하며, 5가지 핵심 역량(통합 공간, 범용 작업 처리, ICL, 자연어 상호작용, 복합 추론)을 갖춘 모델을 지향한다. 이 연구는 향후 multi-modal graph 분야가 단순한 분류/예측을 넘어 거대 모델 기반의 일반 인공지능(AGI) 방향으로 나아가기 위한 이론적 토대와 연구 지도를 제공한다는 점에서 매우 중요한 역할을 할 것으로 보인다.