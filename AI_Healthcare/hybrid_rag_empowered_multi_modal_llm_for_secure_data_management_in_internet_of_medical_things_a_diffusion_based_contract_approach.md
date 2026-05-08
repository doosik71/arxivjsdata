# Hybrid RAG-empowered Multi-modal LLM for Secure Data Management in Internet of Medical Things: A Diffusion-based Contract Approach

Cheng Su, Jinbo Wen, Jiawen Kang*, Yonghua Wang, Yuanjia Su, Hudan Pan, Zishao Zhong, and M. Shamim Hossain (2024)

## 🧩 Problem to Solve

본 논문은 의료 사물 인터넷(Internet of Medical Things, IoMT) 환경에서 멀티모달 거대 언어 모델(Multi-modal Large Language Models, MLLMs)을 활용한 의료 데이터 관리의 핵심적인 문제들을 해결하고자 한다. 구체적으로 다음과 같은 네 가지 주요 과제에 집중한다.

첫째, 의료 데이터는 기본적으로 멀티모달 특성을 가지며 분산 저장되어 있어, 기존의 단일 모달 기반 RAG(Retrieval-Augmented Generation) 방식으로는 복합적인 의료 데이터를 효율적으로 검색하여 MLLM의 태스크를 지원하는 데 한계가 있다. 둘째, 의료 데이터의 민감성으로 인해 MLLM 처리 과정에서의 보안 및 개인정보 보호 위험이 매우 크다. 셋째, 데이터셋의 편향으로 인해 사전 학습된 의료 MLLM이 부정확한 추론을 내놓을 수 있으며, 이를 해결하기 위해 최신성(Freshness)이 보장된 고품질 데이터의 지속적인 주입이 필수적이다. 넷째, 정보 비대칭성(Information Asymmetry)으로 인해 데이터 보유자가 최신 데이터를 공유할 유인이 부족한 문제가 존재한다.

따라서 본 논문의 목표는 보안성이 보장된 데이터 전송 체계, 멀티모달 데이터를 효과적으로 검색하는 Hybrid RAG, 그리고 데이터 보유자의 참여를 유도하는 최신성 기반의 인센티브 메커니즘을 결합한 통합 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 보안, 품질, 그리고 동기부여라는 세 가지 관점을 통합하여 MLLM의 성능을 최적화하는 것이다.

1. **Cross-chain 기반 보안 프레임워크**: 메인 체인과 서브 체인, 릴레이 체인으로 구성된 계층적 구조를 통해 중앙 기관 없이도 의료 데이터를 안전하게 전송하고 관리할 수 있는 체계를 제안한다.
2. **Hybrid Multi-modal RAG**: 단일 모달 검색 결과들을 통합하고, Multi-modal Information Similarity (MIS) 지표를 통해 검색 결과를 재정렬(Re-ranking)함으로써 MLLM에 입력되는 컨텍스트의 질을 높인다.
3. **AoI 기반의 계약 이론(Contract Theory) 모델**: 데이터의 최신성을 측정하는 Age of Information (AoI)을 도입하여 데이터 품질을 정의하고, 정보 비대칭 상황에서도 데이터 보유자가 최신 데이터를 공유하도록 유도하는 인센티브 계약 모델을 설계한다.
4. **GDM 기반의 DRL 알고리즘**: 계약 설계의 고차원적인 복잡성을 해결하기 위해 생성 확산 모델(Generative Diffusion Model, GDM) 기반의 심층 강화학습(DRL)을 적용하여 최적의 계약 조건을 도출한다.

## 📎 Related Works

논문은 크게 세 가지 관련 연구 분야를 검토한다.

첫째, **RAG-empowered LLMs** 분야에서는 외부 지식 베이스를 통해 LLM의 환각 현상을 줄이고 정확도를 높이는 연구들이 진행되어 왔다. 그러나 기존 연구들은 주로 텍스트 기반의 단일 모달 RAG에 치중되어 있으며, 복잡한 의료 멀티모달 데이터를 다루는 능력은 부족한 상태이다.

둘째, **LLMs for Data Management** 분야에서는 LLM을 에이전트로 활용하여 데이터 저장 및 분석을 자동화하려는 시도가 있었다. 하지만 이 역시 텍스트 중심의 처리가 주를 이루며, 이미지와 정형 데이터가 혼합된 IoMT 환경의 특성을 충분히 반영하지 못하고 있다.

셋째, **Contract Theory for Data Sharing** 분야에서는 정보 비대칭 상황에서 참여자의 유인을 일치시키기 위한 경제학적 접근이 사용되었다. 기존의 계약 모델들은 동적인 네트워크 환경이나 데이터의 최신성(Freshness) 문제를 정밀하게 다루지 못하는 한계가 있으며, 특히 확산 모델(Diffusion Model)을 계약 최적화에 적용한 사례는 미비하다.

## 🛠️ Methodology

### 1. Cross-Chain Interaction

보안 전송을 위해 메인 체인(Main Chain), 릴레이 체인(Relay Chain), 서브 체인(Subchain) 구조를 사용한다.

- **서브 체인**: 지역 병원에서 IoMT 기기를 통해 실시간 데이터를 수집하고 관리한다.
- **릴레이 체인**: 서브 체인과 메인 체인 사이의 요청을 검증하고 중계한다.
- **메인 체인**: 전체적인 데이터 수집 태스크 관리 및 MLLM 학습을 수행하며, 기여도에 따른 보상을 분배한다.

### 2. Hybrid RAG Pipeline

MLLM의 추론 능력을 높이기 위해 다음과 같은 5단계 절차를 거친다.

1. **데이터 저장**: 멀티모달 데이터를 임베딩 모델을 통해 벡터화하여 로컬 DB에 저장한다.
2. **데이터 검색**: 쿼리를 벡터화하여 코사인 유사도 기반으로 Top-K 결과를 추출한다.
3. **재정렬(Re-rank)**: Multi-modal Information Similarity (MIS) 지표를 사용하여 결과를 필터링한다. MIS는 다음과 같이 정의된다.
    $$MIS = \sum_{i=0}^{n} w_i f_i(x_1, x_2)$$
    여기서 $f_i(\cdot)$는 각 모달리티별 유사도 측정 함수이며, $w_i$는 가중치이다.
4. **입력 최적화**: 제로샷 프롬프팅(Zero-shot prompting)과 'probability'와 같은 제어 키워드를 사용하여 프롬프트를 구성한다.
5. **콘텐츠 생성**: 선형 투영 어댑터(Linear Projection Adapter)를 통해 통일된 임베딩으로 변환 후 MLLM이 최종 답변을 생성한다.

### 3. Incentive Mechanism based on Contract Theory

데이터의 최신성을 보장하기 위해 AoI를 도입한다.

- **데이터 품질 지표**: $\text{AoI}$가 낮을수록 데이터가 신선함을 의미하며, 품질 지표 $G(A_m)$은 다음과 같이 정의된다.
    $$G(A_m) = \frac{A_{\max}}{A_m}$$
- **유틸리티 함수**: 데이터 보유자의 효용 $U_k$는 보상 $R_k$에서 업데이트 비용 $f_k \delta_k$를 뺀 값이다. 서비스 제공자의 기대 효용 $U_s$는 다음과 같다.
    $$U_s(f, R) = \sum_{k=1}^{K} Q_k (\beta S_k - R_k)$$
    여기서 $S_k = \alpha \log(G(A_k) + 1)$는 서비스 만족도 함수이다.
- **계약 제약 조건**: 개별 합리성(Individual Rationality, IR) 조건 $\left(R_k - f_k \delta_k \geq 0\right)$과 인센티브 호환성(Incentive Compatibility, IC) 조건을 모두 만족해야 한다.

### 4. GDM-based Optimal Contract Design

계약 최적화 문제를 마르코프 결정 과정(MDP)으로 정의하고 GDM을 통해 해결한다.

- **상태 공간($s$)**: $\left\{M, K, A_{\max}, Q, \mathcal{K}\right\}$로 구성된다.
- **동작 공간($a_t$)**: 각 타입별 업데이트 빈도와 보상으로 구성된 계약 세트 $\Psi_t = \{(f_k^t, R_k^t), k \in K\}$이다.
- **GDM 프로세스**: 가우시안 노이즈가 섞인 초기 샘플에서 시작하여, 역확산 과정(Reverse Diffusion Process)을 통해 노이즈를 제거하며 최적의 계약 조건을 생성한다. 이때 Double Q-learning 기반의 비평가(Critic) 네트워크가 보상을 평가하여 정책을 업데이트한다.

## 📊 Results

### 1. Hybrid RAG 성능 평가 (Case Study)

LLaVA-Med 모델을 기반으로 prototype을 구현하여 실험하였다. 평가 지표로는 RAI(Responsible AI), SS(Semantic Similarity), 그리고 이를 통합한 Relative LLM Score ($\zeta$)를 사용하였다.

- **정량적 결과**: Hybrid RAG를 적용한 LLaVA-Med는 $\zeta$ 점수에서 0.96을 기록하며, GPT-4o(0.49)나 기본 LLaVA-Med(0.51)보다 월등히 높은 성능을 보였다.
- **정성적 결과**: 특히 의사가 오판하기 쉬운 까다로운 X-ray 케이스에서도 Hybrid RAG는 유사한 질병 사례를 검색하여 정확한 진단을 내리는 강건함을 보였다.

### 2. GDM 기반 계약 설계 성능

- **보상 비교**: 정보 비대칭 상황에서 GDM 기반 스킴은 Greedy나 Random 정책보다 훨씬 높은 보상을 획득하였다.
- **GDM vs DRL-PPO**: GDM은 PPO 대비 최종 테스트 보상이 유의미하게 높았으며, 수치적으로 MLLM 서비스 제공자의 유틸리티를 280.85까지 끌어올려 PPO의 233.2보다 우수한 성능을 보였다. 이는 GDM의 미세한 정책 조정 능력이 지역 최적점(Local Optima) 탈출에 효과적임을 시사한다.

### 3. 블록체인 보안성 분석

PBFT(Practical Byzantine Fault Tolerance) 합의 알고리즘을 적용하여 보안 확률 $P_{\text{safety}}$를 분석하였다. 릴레이 체인의 크기가 커질수록 악의적인 노드가 존재하더라도 보안 확률이 상승하는 경향을 확인하였다.

## 🧠 Insights & Discussion

본 논문은 의료 AI 모델의 성능 향상을 위해 단순히 모델 구조를 개선하는 것이 아니라, **데이터의 수집 $\rightarrow$ 전송 $\rightarrow$ 검색 $\rightarrow$ 추론**으로 이어지는 전체 파이프라인을 시스템적으로 설계했다는 점에서 강점이 있다. 특히, RAG가 모델의 일반화 능력을 직접적으로 높이지 못한다는 한계를 정확히 짚어내고, 이를 해결하기 위해 '최신 데이터 주입'을 위한 경제학적 인센티브 모델(Contract Theory)을 결합한 점이 매우 독창적이다.

다만, 몇 가지 한계점과 논의 사항이 존재한다. 첫째, GDM 기반의 강화학습은 계산 복잡도가 높으며, 실제 실시간 IoMT 환경에서 계약을 얼마나 빠르게 업데이트할 수 있는지에 대한 지연 시간(Latency) 분석이 부족하다. 둘째, Cross-chain 구조는 보안성을 높이지만 네트워크 오버헤드를 증가시킬 수 있는데, 이에 대한 정밀한 트레이드-오프 분석이 필요하다. 마지막으로, 제안된 MIS 지표의 가중치 $w_i$를 결정하는 구체적인 기준이 명시되지 않아, 실제 적용 시 도메인 전문가의 개입이 많이 필요할 것으로 보인다.

## 📌 TL;DR

이 논문은 IoMT 환경에서 의료 MLLM의 성능과 보안을 높이기 위해 **Cross-chain 보안 전송 + Hybrid RAG(MIS 기반 재정렬) + GDM 기반 최신성 인센티브 계약**을 통합한 프레임워크를 제안한다. 실험 결과, Hybrid RAG는 진단 정확도를 획기적으로 높였으며, GDM 기반 계약 모델은 기존 DRL-PPO보다 효율적으로 데이터 공유 유인을 설계하여 서비스 제공자의 효용을 극대화하였다. 이 연구는 향후 고품질 의료 데이터의 지속적인 확보와 안전한 활용이 필수적인 의료 AI 서비스 구축에 중요한 가이드라인을 제공할 가능성이 크다.
