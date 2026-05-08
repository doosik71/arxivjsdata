# Privacy and Copyright Protection in Generative AI: A Lifecycle Perspective

Dawen Zhang et al. (2024)

## 🧩 Problem to Solve

본 논문은 Generative AI(이하 GenAI) 모델이 학습 과정에서 방대한 양의 데이터를 필요로 함에 따라 발생하는 데이터 프라이버시 침해와 저작권 위반 문제를 해결하고자 한다. GenAI 모델, 특히 Foundation Model은 텍스트, 오디오, 이미지 등 다양한 형태의 콘텐츠를 생성할 수 있는 능력을 갖추었으나, 이 과정에서 학습 데이터셋에 포함된 개인정보나 저작권 보호 대상 데이터를 그대로 복제하여 출력하는 등의 리스크를 내포하고 있다.

기존의 Differential Privacy, Machine Unlearning, Data Poisoning과 같은 접근 방식들은 이러한 복잡한 문제들에 대해 단편적인 해결책만을 제공한다는 한계가 있다. 따라서 본 논문의 목표는 데이터의 전체 생애주기(Data Lifecycle) 관점에서 프라이버시 및 저작권 보호 문제를 다각도로 분석하고, 기술적 혁신과 윤리적 통찰이 결합된 통합적인 프레임워크의 필요성을 제시하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 설계 아이디어는 프라이버시와 저작권 보호를 특정 단계의 기술적 수정으로 보는 것이 아니라, 데이터의 생성부터 배포까지 이어지는 **전체 생애주기(Lifecycle Perspective)** 관점에서 접근해야 한다는 점이다.

주요 기여 사항은 다음과 같다:

- GenAI의 데이터 생애주기 각 단계에서 발생하는 프라이버시 및 저작권 관련 핵심 챌린지들을 정의하고 이를 매핑하였다.
- EU의 GDPR(General Data Protection Regulation)과 미국의 저작권법(US Copyright Law)이라는 법적 근거를 바탕으로 AI 엔지니어링 차원에서 해결해야 할 요구사항을 구체화하였다.
- 단편적인 솔루션을 넘어, 생애주기 전반을 아우르는 통합적 접근 방식(Consent management, AIBOM 등)의 연구 방향을 제시하였다.

## 📎 Related Works

논문에서는 프라이버시 및 저작권 보호를 위해 제안된 기존의 기술적 접근 방식들을 언급한다:

- **Differential Privacy**: 학습 데이터의 개별 레코드가 모델 출력에 과도하게 영향을 주지 않도록 노이즈를 추가하는 방식이다.
- **Machine Unlearning**: 특정 데이터를 모델의 가중치에서 제거하여 해당 데이터의 영향을 삭제하는 방식이다.
- **Data Poisoning**: 데이터에 특정 패턴을 삽입하여 무단 학습을 방지하거나 모델의 성능을 제어하는 방식이다.

**기존 접근 방식의 한계**:
이러한 방법들은 데이터 처리 과정의 특정 세그먼트나 단일 시점의 문제 해결에만 집중하며, 데이터 수집부터 모델 배포 및 추론, 그리고 그 이후의 다운스트림 배포에 이르기까지의 광범위한 생애주기 전반에서 발생하는 상호 연결된 문제들을 통합적으로 해결하지 못한다.

## 🛠️ Methodology

본 논문은 특정 알고리즘을 제안하기보다, GenAI 시스템의 데이터 생애주기를 정의하고 각 단계별 챌린지를 분석하는 프레임워크 중심의 방법론을 제시한다.

### 1. 데이터 생애주기 단계 정의

본 논문은 데이터셋 개발 과정을 다음의 8단계로 정의한다:
$\text{Problem Formulation} \rightarrow \text{Data Collection} \rightarrow \text{Data Cleaning} \rightarrow \text{Data Annotation} \rightarrow \text{Model Training} \rightarrow \text{Model Evaluation} \rightarrow \text{Model Deployment \& Inference} \rightarrow \text{Downstream Distribution}$

### 2. 생애주기별 핵심 챌린지 매핑

각 단계에서 발생하는 주요 문제는 다음과 같다:

- **Consent and License Acquisition**: 방대한 데이터 규모로 인해 개별 데이터 주체로부터 동의를 얻는 것이 현실적으로 어려우며, 이는 문제 정의 및 데이터 수집 단계와 밀접하다.
- **Transparency and Data Access**: 자신의 데이터가 학습에 사용되었는지 알기 어렵고, 데이터셋 공개 시 프라이버시 노출과 투명성 사이의 역설적 상황이 발생한다. (수집 및 배포/추론 단계)
- **Consent Modification and Withdrawal**: GDPR의 '잊혀질 권리(Right to be Forgotten)'에 따라 데이터 삭제 요청이 올 수 있으나, 이미 가중치에 내재된 데이터를 제거하는 것은 기술적으로 매우 어렵다. (수집 및 학습 단계)
- **Provenance and Attribution**: 생성물이 원본과 유사할 때 출처를 명시(Attribution)해야 하는 법적 의무가 있으나, 모델의 복잡성으로 인해 추적 가능성(Traceability)이 낮다. (수집, 어노테이션, 학습, 배포 단계)
- **Side Effects**: 프라이버시 보호 조치(데이터 삭제 등)가 모델의 성능 저하나 데이터 다양성 감소, 공정성 문제로 이어질 수 있다. (수집, 학습, 평가 단계)
- **Continuous Monitoring**: 단순 필터링은 우회 공격(예: Grandma Exploit)에 취약하므로 지속적인 모니터링 체계가 필요하다. (배포 및 추론 단계)
- **Downstream Distribution**: 생성된 결과물이 다시 다른 모델의 학습 데이터로 사용되는 재귀적 사이클로 인해 법적 책임 소재가 불분명해진다. (데이터 배포 단계)

### 3. 제안하는 라이프사이클 접근법 (Research Directions)

- **Lifecycle-wide Consent**: HTTP 요청이나 HTML DOM 요소에 암호화 태그를 사용하는 Consent Tagging 등을 확장하여 생애주기 전반의 동의 관리 체계를 구축해야 한다.
- **Reliable Guardrails**: 단순한 입출력 필터를 넘어, Differential Privacy 학습과 같은 다층적 방어 체계를 아키텍처 수준에서 통합해야 한다.
- **AI Bill of Materials (AIBOM)**: 소프트웨어의 SBOM처럼, AI 모델이 사용한 데이터의 라이선스와 암호화된 동의 정보를 명시한 명세서를 도입하여 법적 준수성을 확보해야 한다.

## 📊 Results

본 논문은 정량적인 실험 결과나 특정 벤치마크 성능을 제시하는 기술 논문이 아니라, 프레임워크와 방향성을 제시하는 **Position Paper(관점 제시 논문)**이다. 따라서 구체적인 수치적 결과나 비교 실험 데이터는 포함되어 있지 않으며, 대신 법적 프레임워크(GDPR, US Copyright Law)와 기술적 챌린지 간의 논리적 매핑 결과물을 제시하였다.

## 🧠 Insights & Discussion

**강점**:
본 논문은 프라이버시와 저작권 문제를 단순한 '데이터 삭제'나 '필터링'의 문제가 아니라, 소프트웨어 공학적 관점에서 데이터의 생애주기 전체로 확장하여 해석하였다는 점에서 매우 높은 통찰력을 제공한다. 특히 AIBOM과 같은 개념을 도입하여 AI 엔지니어링의 표준화 가능성을 제시한 점이 돋보인다.

**한계 및 논의사항**:

- **실행 가능성의 모호함**: 생애주기 전반의 통합적 접근이 필요하다고 주장하지만, 실제로 이를 구현하기 위한 구체적인 아키텍처 설계도나 프로토콜에 대한 상세 설명은 부족하다.
- **상충 관계(Trade-off) 해결책 미비**: 데이터 삭제(Unlearning)와 모델 성능(Utility) 사이의 상충 관계에 대해 'Human-in-the-loop' 전략을 언급하였으나, 이를 자동화하거나 최적화할 수 있는 구체적인 방법론은 제시되지 않았다.
- **현실적 제약**: AIBOM의 도입은 모델 제공자의 자발적인 협조와 산업 표준 합의가 전제되어야 하므로, 기술적 해결책보다 정책적 합의가 더 큰 난관이 될 가능성이 크다.

## 📌 TL;DR

본 논문은 GenAI의 프라이버시 및 저작권 문제를 해결하기 위해 단편적인 기술 적용에서 벗어나 **데이터 생애주기(Data Lifecycle) 전체를 아우르는 통합적 프레임워크**를 제안한다. 데이터 수집부터 배포 후의 다운스트림 유통까지 각 단계의 법적·기술적 챌린지를 매핑하였으며, 특히 **AIBOM(AI Bill of Materials)**과 **다층적 가드레일** 도입을 통해 AI 엔지니어링의 책임성을 높여야 한다고 주장한다. 이 연구는 향후 AI 모델의 법적 준수성을 보장하는 표준 아키텍처 설계 및 AI 거버넌스 연구에 중요한 이론적 토대를 제공할 것으로 보인다.
