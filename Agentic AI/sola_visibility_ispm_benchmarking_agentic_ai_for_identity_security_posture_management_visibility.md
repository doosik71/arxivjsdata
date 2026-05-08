# SOLA-VISIBILITY-ISPM: BENCHMARKING AGENTIC AI FOR IDENTITY SECURITY POSTURE MANAGEMENT VISIBILITY

Gal Engelberg, Konstantin Koutsyi, Leon Goldberg, Reuven Elezra, Idan Pinto, Tal Moalem, Shmuel Cohen, Yoni Weintrob (2026)

## 🧩 Problem to Solve

현대 기업은 AWS와 같은 클라우드 환경과 Okta, Google Workspace와 같은 SaaS 환경이 혼합된 멀티 클라우드 및 하이브리드 환경에서 운영되고 있다. 이러한 환경에서 정체성(Identity)은 실질적인 보안 경계의 역할을 하며, Identity Security Posture Management (ISPM)는 정체성 인벤토리 파악, 설정 위생(Configuration Hygiene) 관리, 권한 최적화 등을 통해 보안 사고를 예방하는 핵심적인 규율이 되었다.

그러나 ISPM 관련 가시성 질문(예: 정체성 인벤토리 확인, 설정 오류 탐지)에 답하기 위해서는 매우 복잡한 정체성 데이터를 해석해야 하므로, 최근에는 자율적 추론과 도구 사용이 가능한 Agentic AI 시스템에 대한 관심이 높아지고 있다. 그럼에도 불구하고, 실제 기업 수준의 정체성 데이터셋을 사용하여 이러한 Agentic AI 시스템의 ISPM 수행 능력을 표준화된 방식으로 평가할 수 있는 벤치마크가 부재한 상황이다. 본 논문의 목표는 실제 운영 환경의 데이터를 기반으로 Agentic AI의 ISPM 가시성 작업 성능을 측정할 수 있는 최초의 벤치마크인 'Sola Visibility ISPM Benchmark'를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 실제 프로덕션 수준의 정체성 환경(AWS, Okta, Google Workspace)을 활용하여 Agentic AI의 ISPM 능력을 평가하는 체계적인 프레임워크를 구축한 것이다.

중심적인 설계 아이디어는 단순한 질의응답 성능 측정을 넘어, AI 에이전트가 자연어 쿼리를 실행 가능한 데이터 탐색 단계로 변환하고, 이를 바탕으로 증거 기반의 답변을 생성하는 전체 파이프라인을 평가하는 것이다. 특히, 기존의 합성 데이터 기반 벤치마크에서 벗어나 실제 기업의 IAM(Identity and Access Management), IdP(Identity Provider), SaaS 데이터가 얽혀 있는 복잡한 환경에서 에이전트의 데이터 그라운딩(Data Grounding)과 SQL 추론 능력을 검증하고자 하였다.

## 📎 Related Works

논문은 AI 평가 체계를 크게 두 가지 흐름으로 구분하여 설명한다. 첫째는 Spider 시리즈와 같은 Text-to-SQL 벤치마크로, 자연어 인터페이스를 통한 구조화된 데이터베이스 추론 능력을 평가한다. 특히 Spider 2.0은 실제 기업 데이터 환경에서의 다단계 워크플로우와 오류 수정을 강조하며, 단순 결과값의 정확성보다 에이전트 워크플로우의 강건성을 평가하는 방향으로 진화하였다.

둘째는 ExCyTIn-Bench, CyberSOCEval, CTIBench 등 사이버 보안 특화 벤치마크이다. 이들은 주로 SOC(Security Operations Center) 조사, 위협 인텔리전스(CTI) 추론, 취약점 탐지 등에 집중하고 있다.

본 연구가 기존 연구와 차별화되는 점은 다음과 같다. 기존의 보안 벤치마크들이 주로 위협 조사나 공격적 역량 평가에 치중한 반면, 본 논문은 현대 보안의 제어 평면(Control Plane)인 '정체성 보안'에 집중한다. 또한, RBAC 규칙 적용 능력을 평가하는 OrgAccess와 같은 기존 연구가 합성(Synthetic) 조직 구조를 사용하는 것과 달리, 본 벤치마크는 실제 프로덕션 IAM 및 SaaS 데이터에 기반한 운영 가시성(Operational Visibility)을 평가한다는 점에서 실무적 가치가 높다.

## 🛠️ Methodology

### Sola AI Agent 아키텍처

Sola AI Agent는 자연어 ISPM 쿼리를 받아 실행 가능한 데이터 탐색 단계로 변환하고 증거 기반의 답변을 생성하는 도구 사용형 에이전트이다. 전체 프로세스는 Schema-grounded 실행 모델을 따르며, 다음의 두 가지 실행 모드를 지원한다.

1. **Fast-path Exploration**: 검색된 예시 쿼리를 대상 스키마에 직접 적응시켜 단일 패스로 실행하는 방식이다. 예시와의 유사성과 스키마 신뢰도가 높을 때 사용하며, 낮은 지연 시간으로 빠른 결과를 내는 데 최적화되어 있다.
2. **Full-path Exploration**: 질문을 중간 탐색 단계로 분해하고, 각 단계의 성공 기준을 설정하여 반복적으로 추론하는 방식이다. 이는 Tree-of-Thought 스타일의 추론에 영감을 받았으며, 실행 단계마다 결과를 검증하고 단계별 저널(Step Journal)을 기록하여 추론의 투명성을 높인다.

최종적으로 두 모드 모두 쿼리 결과를 통합하여 자연어 답변, 실행된 SQL, 그리고 이를 뒷받침하는 증거를 포함한 응답을 생성한다.

### Sola ISPM Visibility Benchmark 구성

벤치마크는 AWS(클라우드 서비스), Okta(외부 IdP), Google Workspace(협업 플랫폼)라는 세 가지 계층의 실제 데이터 소스를 통합하여 구축되었다.

- **질문 생성 방법**: Scout Suite(AWS), ScubaGoggles(GWS), Okta 보안 베스트 프랙티스 등 권위 있는 보안 기준을 바탕으로, 실제 데이터에서 정답을 도출할 수 있는 '데이터 제한적(Data-bounded)' 질문으로 재구성하였다. 총 77개의 질문이 생성되었으며, 사이버 보안 전문가의 검증을 거쳤다.
- **평가 프레임워크**: 다음과 같은 4단계 프로세스를 통해 에이전트를 평가한다.
    1. **에이전트 실행**: 모든 질문을 Sola 에이전트로 실행하여 추론 흔적(Reasoning Trace)을 생성한다.
    2. **증거 번들(Evidence Bundle) 수집**: 입력 질문, 최종 답변, 생성된 SQL, 검색된 예시 쿼리, 단계별 저널 및 도구 출력값을 수집한다.
    3. **전문가 평가**: 5명의 전문가가 답변의 의미적 정확성을 $\{0, 0.5, 1\}$ 척도로 평가하여 Accuracy와 Success Rate(점수 1점만 성공으로 인정)를 계산한다.
    4. **LLM-as-Judge 평가**: Anthropic Claude Sonnet 4.5와 OpenAI GPT-4.1를 판사로 사용하여 Answer-Quality, Reasoning-Quality, Retrieval and Context-Use, SQL-Quality의 4개 영역을 평가하며, 두 판사 중 낮은 점수를 최종 점수로 채택하는 보수적 접근 방식을 취한다.

## 📊 Results

실험 결과, Sola 에이전트는 77개의 질문에 대해 전반적으로 강력한 성능을 보였다. 전문가 기준 Accuracy는 $0.84$, Success Rate는 $0.77$를 기록하였다.

### 도메인별 성능 분석

- **AWS Hygiene**: 가장 높은 성능을 보였으며, 전문가 Accuracy $0.95$, Success Rate $0.90$를 기록하였다.
- **Inventory 및 GWS Hygiene**: 각각 Accuracy $0.75$ 수준으로 양호한 성능을 보였다.
- **Okta Hygiene**: 상대적으로 가장 낮은 성능을 보였으며, 특히 Success Rate가 $0.50$으로 떨어져 부분적으로 정답인 경우가 많았음을 시사한다.

### 추론 전략별 분석 (Full-path vs Fast-path)

- **Full-path**: 전반적으로 더 일관된 정확도를 보였다. 특히 복잡한 Okta 및 GWS 위생 작업에서 강점을 보였다. 추론 일관성과 답변 정렬도가 매우 높게 나타났다.
- **Fast-path**: AWS 및 Inventory 작업에서는 강력했으나, 도메인 간 변동성이 컸다. 특히 Fast-path에서는 예시 적응 능력(Example Adaptation)과 정답률 사이의 상관관계가 매우 높게($\rho \approx 0.8$) 나타났다. 이는 명시적 추론 단계가 없는 경우, 검색된 SQL 템플릿을 얼마나 잘 수정하느냐가 성능의 핵심임을 의미한다.

### 평가 도구 간 일치도

전문가 평가와 LLM-as-Judge 간의 평균 절대 일치도(MAA)를 분석한 결과, AWS Hygiene에서는 매우 높은 일치도를 보였으나, Google Workspace Hygiene에서는 일치도가 낮게 나타났다. 이는 GWS 도메인의 복잡성으로 인해 자동화된 평가 도구가 전문가의 판단을 완벽히 대체하기 어려움을 보여준다.

## 🧠 Insights & Discussion

본 연구를 통해 Agentic AI가 실제 기업의 정체성 보안 가시성 작업에서 상당히 유용하게 활용될 수 있음이 입증되었다. 특히 주목할 점은 에이전트의 성능이 단순히 모델의 추론 깊이나 단일 샷(Single-shot) 정확도에 의해 결정되는 것이 아니라, **복잡한 엔터프라이즈 스키마 위에서 다단계 워크플로우를 실행하고, 검색된 쿼리 패턴을 대상 환경에 맞게 적응(Adaptation)시키는 능력**에 달려 있다는 것이다.

실제로 Full-path 추론을 사용하더라도 예시 적응 능력이 정답률에 유의미한 영향($\rho \approx 0.5$)을 미쳤다는 점은, 정교한 추론 체인만으로는 부족하며 강력한 구조적 그라운딩(Structural Grounding)이 필수적임을 시사한다.

한계점으로는 본 벤치마크가 ISPM의 기초 단계인 '가시성 및 위생'에 집중되어 있다는 점이다. 실제 ISPM의 전체 생애주기를 관리하기 위해서는 향후 교차 시스템 상관관계 분석(Cross-system Correlation), 행동 분석, 리스크 스코어링, 그리고 거버넌스 정렬과 같은 더 고도화된 분석 능력을 평가할 수 있는 확장된 벤치마크가 필요하다.

## 📌 TL;DR

본 논문은 실제 AWS, Okta, Google Workspace 환경의 데이터를 사용하여 Agentic AI의 정체성 보안 자세 관리(ISPM) 능력을 평가하는 최초의 벤치마크인 'Sola Visibility ISPM Benchmark'를 제안하였다. Sola AI 에이전트는 전문가 평가 기준 0.84의 정확도를 기록하였으며, 분석 결과 AI 에이전트의 성공 여부는 단순한 추론 능력보다 실제 데이터 스키마에 맞게 SQL 템플릿을 적응시키고 실행하는 '데이터 그라운딩' 능력에 크게 의존함을 밝혔다. 이 연구는 향후 AI 기반의 자율적 정체성 보안 운영 체계를 구축하기 위한 표준 평가 기반을 마련하였다는 점에서 중요한 의의를 가진다.
