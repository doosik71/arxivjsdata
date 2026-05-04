# Multi-agent Searching System for Medical Information

Mariya Evtimova-Gardair (2019)

## 🧩 Problem to Solve

본 연구는 인터넷 환경에서 사용자가 자신의 건강 문제에 적합한 의료 정보를 효율적이고 안전하게 검색할 수 있도록 하는 Multi-agent 시스템을 구축하는 것을 목표로 한다.

현재 시장에 출시된 검색 시스템들은 사용자의 구체적인 의료적 필요를 항상 충족시키지 못하는 한계가 있다. 특히 의료 정보 검색 결과는 특정 질병의 진단에 결정적인 영향을 미칠 수 있으므로, 검색 과정에서 데이터 프라이버시를 보호하고 보안을 강화하는 것이 매우 중요하다. 따라서 본 논문은 데이터 분석 전 보안을 확보하고, 개인화된 결과를 제공할 수 있는 지능형 검색 시스템의 필요성에서 출발한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 분산 인공지능(Distributed Artificial Intelligence, DAI)의 일환인 Multi-agent 시스템을 적용하여 복잡한 의료 정보 검색 작업을 분산 처리하는 것이다. 특히, 고정된 위치에서 작동하는 Static Agent 대신, 네트워크 상의 여러 위치를 직접 이동하며 데이터를 수집하는 Mobile Agent를 도입하여 시스템 구조를 단순화하고 리소스 관리 효율성을 높인 점이 주요 기여 사항이다.

## 📎 Related Works

논문에서는 Multi-agent 시스템이 빅데이터 처리 및 의사결정 분산에 널린 응용 분야를 가지고 있음을 언급한다. 특히 에이전트의 자율성(autonomy), 반응성(reactivity), 능동성(proactivity), 사회적 능력(social capabilities)이 정보 검색 시스템의 효율성을 높이는 핵심 요소임을 설명한다. 기존의 검색 엔진들과 달리, 본 연구는 사용자의 프로필을 기반으로 한 개인화(Personalization)와 보안 인증 메커니즘을 결합하여 차별화를 꾀한다.

## 🛠️ Methodology

### 전체 시스템 구조 및 파이프라인
제안된 시스템은 크게 보안 로그인을 통한 사용자 인증, 사용자 쿼리 정의, 그리고 Mobile coordinating agent를 이용한 인터넷 정보 검색의 세 단계로 구성된다. 전체 시스템은 다음과 같은 6가지 주요 구성 요소(에이전트 및 모듈)로 이루어져 있다.

1.  **Log-in Authority Module**: IP 주소와 사용자 식별 번호를 기반으로 보안을 제어한다. 실세계의 사용자 식별을 방지하고, 데이터 추적 가능성을 차단하기 위해 검색마다 다른 키를 사용하여 레코드를 인덱싱하는 프라이버시 보존 분석을 수행한다.
2.  **Interface Agent**: 사용자로부터 입력을 받고 최종 결과를 출력하는 인터페이스 역할을 수행한다. 사용자의 피드백을 명시적(Explicit) 또는 암시적(Implicit)으로 수집하여 시스템에 반영한다.
3.  **User Profile Agent**: 사용자의 일반 정보, 의료 정보, 건강 상태 등을 포함한 프로필을 관리하여 검색 결과의 개인화를 지원한다.
4.  **Query Modification Agent**: 사용자 쿼리를 분석하고 최적화한다. 철자 검사(Spellcheck), 동의어 관리(Synonym manager), 불용어 제거(Stop words filtering) 과정을 거치며, 다국어 사전(Multilingual vocabulary)을 통해 쿼리를 정교화한다.
5.  **Personalization Agent**: 수집된 결과들 사이의 충돌을 제거하고, 유사한 결과를 그룹화하며, 사용자 프로필에 따라 랭킹 및 정렬을 수행하여 최적의 개인화된 결과를 생성한다.
6.  **Mobile Agent**: 본 시스템의 핵심으로, 네트워크상의 여러 웹사이트를 직접 이동하며 정보를 수집한다.

### Mobile Agent의 작동 원리 및 학습 절차
Mobile Agent는 Static Agent와 달리 좌표자(Coordinator)와 웹 에이전트의 역할을 동시에 수행한다. 

- **작동 흐름**: $\text{Query Modification Agent} \rightarrow \text{Mobile Agent} \rightarrow \text{Websites} \rightarrow \text{Personalization Agent}$ 순으로 데이터가 흐른다.
- **상세 절차**: Mobile Agent는 `GetAvailableLocations` 동작을 통해 이용 가능한 위치를 확인하고, `move()` 메서드를 사용하여 대상 웹사이트로 직접 이동한다. 각 사이트에 도착하면 해당 페이지에서 데이터를 수집하여 데이터베이스에 저장하며, 모든 방문지가 완료될 때까지 이 과정을 반복한다.

### 구현 기술
- **JADE (Java Agent DEvelopment Framework)**: 에이전트의 실행 및 관리를 위해 사용되었으며, AMS(Agent Management System)와 DF(Directory Facilitator)를 통해 에이전트 간 서비스 등록 및 탐색이 이루어진다.
- **HtmlUnit**: 웹 페이지의 검색 폼을 채우고 버튼을 클릭하며, 결과 HTML 페이지를 분석하여 데이터를 추출하는 라이브러리로 활용되었다.

## 📊 Results

### 정량적 성능 비교 (Static vs. Mobile Agent)
Intel Core i5-5200 CPU, 4GB RAM 환경에서 응답 시간을 측정한 결과는 다음과 같다.

| 측정 항목 | Static Agent | Mobile Agent |
| :--- | :---: | :---: |
| 응답 시간 (Response Time) | $80,524\text{ms}$ | $75,123\text{ms}$ |

Mobile Agent를 사용했을 때 응답 시간이 단축되었다. 이는 Static Agent 구조에서는 수많은 에이전트가 동시에 실행되며 발생하는 프로세스 오버헤드와 통신 비용이 크기 때문이며, Mobile Agent는 에이전트 수를 줄여 리소스 관리를 최적화했기 때문으로 분석된다.

### 검색 성능 평가
225개의 다양한 증상 카테고리(복부, 심혈관, 소화기, 신경계 등)에 대해 테스트를 진행한 결과, 다음과 같은 성능 지표를 얻었다.
- **Precision (정밀도)**: $96\%$
- **Recall (재현율)**: $91\%$
- **F-measure**: $93\%$

## 🧠 Insights & Discussion

본 연구는 Mobile Agent를 통해 분산 환경에서의 리소스 소모를 줄이고 검색 효율성을 높일 수 있음을 입증하였다. 특히 의료 정보라는 민감한 데이터를 다루는 만큼, 단순한 검색을 넘어 사용자 프로필 기반의 개인화와 보안 모듈을 통합한 아키텍처를 제시한 점이 강점이다.

다만, 실험 결과에서 Mobile Agent의 우위가 하드웨어 구성이나 인터넷 연결 속도에 따라 달라질 수 있음을 명시하고 있다. 또한, 데이터 수집 방식이 HtmlUnit을 통한 웹 스크래핑 기반이므로, 대상 웹사이트의 구조가 변경될 경우 시스템의 유지보수 비용이 증가할 수 있다는 한계가 존재한다. 또한, 구체적인 보안 알고리즘의 수학적 증명이나 암호화 방식에 대한 상세 설명이 부족하여 실제 보안 수준을 객관적으로 검증하기에는 어려움이 있다.

## 📌 TL;DR

본 논문은 JADE 프레임워크와 Mobile Agent를 활용하여 보안성과 개인화 기능이 강화된 의료 정보 검색 시스템을 제안하였다. Mobile Agent는 여러 웹사이트를 직접 이동하며 데이터를 수집함으로써 Static Agent 대비 리소스 효율성을 높였으며, 결과적으로 $96\%$의 정밀도와 $93\%$의 F-measure를 달성하였다. 이 연구는 향후 대규모 의료 빅데이터 환경에서 효율적인 정보 추출 및 분산 처리 시스템을 설계하는 데 기초 자료로 활용될 수 있다.