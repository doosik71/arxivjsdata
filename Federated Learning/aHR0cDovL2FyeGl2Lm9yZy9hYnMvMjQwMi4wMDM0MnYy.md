# Survey of Privacy Threats and Countermeasures in Federated Learning

Masahiro HAYASHITANI, Junki MORI, and Isamu TERANISHI (2024)

## 🧩 Problem to Solve

본 논문은 Federated Learning (FL) 환경에서 발생하는 프라이버시 위협과 이를 방어하기 위한 대응책을 체계적으로 분석하는 것을 목표로 한다. FL은 중앙 집중식 데이터 수집 없이 협업 모델 학습을 가능하게 하여 프라이버시를 보호하는 패러다임으로 등장하였으나, 학습 과정에서 교환되는 모델 파라미터나 그래디언트(gradient)를 통해 민감한 정보가 유출될 수 있는 취약점이 존재한다.

특히 기존의 관련 연구들은 Horizontal Federated Learning (HFL)과 Vertical Federated Learning (VFL)을 개별적으로 다루었을 뿐, 이 모든 패러다임을 통합하여 분석한 연구는 부족했다. 따라서 본 논문은 HFL, VFL, 그리고 Federated Transfer Learning (FTL)이라는 세 가지 주요 FL 패러다임을 모두 아우르는 통합적인 프라이버시 위협 분류 체계(taxonomy)를 제시하고, 각 위협에 대응하는 방어 메커니즘을 논의함으로써 보안성 높은 FL 시스템 설계의 가이드를 제공하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 FL의 세 가지 주요 변형(HFL, VFL, FTL) 전체를 포괄하는 통합적인 프라이버시 위협 분석 프레임워크를 구축한 것이다. 구체적인 설계 아이디어는 다음과 같다.

첫째, 대상 데이터의 성격에 따라 프라이버시 공격을 Feature Inference, Property Inference, Membership Inference, Label Inference, ID Leakage, Relation Leaks의 6가지 유형으로 범주화하였다.
둘째, 공격자의 역할(Server, Client, Third Party), 공격에 사용되는 정보(Gradients, Model Parameters), 공격 스타일(Malicious, Honest-but-curious, Honest)을 기준으로 상세한 위협 모델(Threat Model)을 정의하였다.
셋째, 각 공격 유형에 대해 일반적인 방어 방법(General Defense)과 특정 공격에 특화된 방어 방법(Specialized Defense)을 구분하여 매핑함으로써, 실무적인 관점에서 효율적인 방어 전략을 선택할 수 있도록 하였다.

## 📎 Related Works

논문에서는 기존의 프라이버시 보존 연구들이 주로 HFL과 VFL이라는 두 가지 축으로 나뉘어 진행되었음을 지적한다. HFL 관련 서베이 연구들은 주로 그래디언트 누출을 통한 데이터 복원 공격에 집중했으며, VFL 관련 연구들은 서로 다른 피처를 가진 클라이언트 간의 정렬 과정에서 발생하는 위험을 주로 다루었다.

그러나 이러한 기존 접근 방식은 다음과 같은 한계가 있다.
1. **통합적 관점의 부재**: HFL, VFL, FTL의 구조적 차이로 인해 발생하는 서로 다른 위협들을 하나의 체계 내에서 비교 분석하지 못했다.
2. **FTL 연구의 부족**: FTL은 샘플과 피처가 모두 다른 복잡한 설정임에도 불구하고, 이에 특화된 프라이버시 위협 및 방어 기제에 대한 체계적인 분석이 거의 이루어지지 않았다.

본 논문은 이러한 공백을 메우기 위해 세 가지 패러다임을 통합한 분류 체계를 제안하며 차별성을 갖는다.

## 🛠️ Methodology

본 논문은 분석 대상인 FL의 구조를 먼저 정의하고, 이를 바탕으로 위협 모델과 공격 유형, 방어 기법을 체계화하는 방법론을 취한다.

### 1. Federated Learning의 분류
데이터 구조에 따라 FL을 다음과 같이 세 가지로 정의한다.
- **Horizontal Federated Learning (HFL)**: 피처 공간과 레이블 공간은 동일하지만, 샘플 공간이 서로 다른 경우이다. 주로 중앙 서버가 모델 파라미터를 취합하는 $\text{FedAvg}$ 방식이 사용된다.
- **Vertical Federated Learning (VFL)**: 샘플 공간은 동일하지만, 피처 공간이 서로 다른 경우이다. 레이블을 가진 Active client와 피처만 가진 Passive client가 중간 출력값을 교환하며 학습한다.
- **Federated Transfer Learning (FTL)**: 샘플 공간과 피처 공간이 모두 서로 다른 경우이다. 소스 클라이언트의 지식을 타겟 클라이언트로 전이하여 레이블이 없는 타겟 클라이언트의 예측 성능을 높이는 것이 목표이다.

### 2. 위협 모델 (Threat Model)
공격 상황을 분석하기 위해 네 가지 관점을 정의한다.
- **공격자 역할**: Server(서버), Client(클라이언트), Third Party(외부 제3자).
- **공격 정보**: $\text{Gradients}$(그래디언트) 또는 $\text{Model Parameters}$(모델 파라미터).
- **공격 스타일**: $\text{Malicious}$(능동적 간섭), $\text{Honest-but-curious}$(프로토콜 준수하에 추론 시도), $\text{Honest}$(추론 시도 없음).
- **FL 유형**: HFL, VFL, FTL.

### 3. 프라이버시 공격의 분류
타겟 데이터의 성격에 따라 6가지로 분류한다.
- **Feature Inference**: 입력 피처를 복구하는 공격 ($\text{Reconstruction attack}$ 또는 $\text{Attribute inference}$).
- **Property Inference**: 클래스 분포나 민감한 속성과 같은 데이터셋의 전반적 특성을 추론하는 공격.
- **Membership Inference**: 특정 데이터 포인트가 학습 세트에 포함되었는지 여부를 확인하는 공격.
- **Label Inference**: VFL/FTL에서 레이블이 없는 클라이언트가 레이블 정보를 추론하는 공격.
- **ID Leakage**: 샘플 ID 정렬 과정에서 겹치지 않는 ID나 교집합 내의 멤버십 정보가 노출되는 위험.
- **Relation Leaks**: 그래프 데이터 등에서 샘플 간의 연결 관계를 추론하는 공격.

### 4. 방어 메커니즘
방어 방법은 크게 두 가지 범주로 나눈다.

#### 일반 방어 방법 (General Defense)
- **통신 채널 방어**: $\text{SMPC}$(Secure Multi-Party Computation)를 통해 암호화된 상태로 연산을 수행한다. 여기에는 $\text{Homomorphic Encryption (HE)}$, $\text{Garbled Circuit (GC)}$, $\text{Secret Sharing (SS)}$ 등이 포함된다. 또한 $\text{Blockchain}$을 이용해 신뢰할 수 없는 서버를 배제한 P2P FL을 구현할 수 있다.
- **차분 프라이버시 (Differential Privacy, DP)**: 데이터에 노이즈를 추가하여 개별 레코드의 영향력을 제한한다.
  - **$\text{CDP}$ (Central DP)**: 신뢰할 수 있는 서버가 집계 결과에 노이즈를 추가한다.
  - **$\text{LDP}$ (Local DP)**: 각 클라이언트가 로컬 업데이트에 노이즈를 추가한다.
  - **$\text{DDP}$ (Distributed DP)**: 암호화 프로토콜을 결합하여 $\text{LDP}$보다 높은 유틸리티와 $\text{CDP}$ 수준의 프라이버시를 달성한다.
  - **$\text{PDP}$ (Participant-level DP)**: 개별 데이터가 아닌 클라이언트 수준의 특성 누출을 방지한다.

$\text{DP}$의 수학적 정의는 다음과 같다. $\epsilon > 0, 0 \le \delta < 1$일 때, 인접한 데이터셋 $D, D'$에 대해 메커니즘 $M$이 다음을 만족하면 $(\epsilon, \delta)\text{-DP}$라고 한다.
$$\Pr[M(D) \in S] \le \exp(\epsilon) \cdot \Pr[M(D') \in S] + \delta$$

#### 특화 방어 방법 (Specialized Defense)
- **Feature Inference**: $\text{Batch size}$를 크게 설정하여 최적화 복잡도를 높이거나, $\text{Dropout}$을 통해 가시적인 그래디언트 수를 줄인다.
- **Property Inference**: $\text{Dropout}$ 적용 및 그래디언트 업데이트 공유 횟수를 제한한다.
- **Membership Inference**: 정규화($\text{Regularization}$), $\text{Dropout}$, $\text{Distillation}$ 등을 적용한다.
- **Label Inference**: 그래디언트 섭동($\text{Perturbation}$)이나 합성 그래디언트 생성, 중간 임베딩과 레이블 간의 상관관계 최소화 최적화를 수행한다.
- **ID Leakage**: $\text{PSI}$(Private Set Intersection) 기술을 사용하거나 $\text{Dummy ID}$를 삽입하여 실제 매칭 여부를 은폐한다.

## 📊 Results

본 논문은 서베이 논문이므로 별도의 실험적 수치보다는, 기존 문헌들을 분석하여 도출한 **위협-패러다임 매핑 결과(Table I)**가 핵심 결과물이다.

- **HFL의 취약점**: Feature, Property, Membership Inference에 가장 많이 노출되어 있으며, 특히 서버가 그래디언트를 통해 데이터를 복구하는 공격이 활발히 연구되었다.
- **VFL의 취약점**: Label Inference와 ID Leakage라는 고유한 위험을 가지고 있다. 이는 VFL의 구조적 특성(중간 출력값 교환, ID 정렬 필요성)에서 기인한다.
- **FTL의 취약점**: 본 논문의 분석 결과, FTL은 구조적으로 HFL(나중 단계)과 VFL(ID-FTL의 경우)의 취약점을 모두 계승한다. 하지만 FTL 전용 프라이버시 위협에 대한 체계적인 연구는 매우 부족한 상태임이 확인되었다.

## 🧠 Insights & Discussion

본 논문을 통해 도출된 주요 인사이트는 다음과 같다.

첫째, **FL은 원천 데이터가 전송되지 않음에도 불구하고, 모델 업데이트 값 자체가 데이터의 대리자(Proxy) 역할을 하여 심각한 정보 유출을 초래할 수 있다**는 점이다. 특히 VFL에서 발생하는 ID Leakage는 정직한 클라이언트 간에도 발생할 수 있는 구조적 결함으로, 실무 적용에 큰 걸림돌이 된다.

둘째, **방어 기법의 트레이드-오프(Trade-off) 문제**이다. $\text{SMPC}$나 $\text{HE}$ 같은 암호화 기반 방어는 보안성은 매우 높지만 계산 비용이 막대하며, $\text{DP}$는 계산 효율적이지만 노이즈 추가로 인한 모델 성능 저하(Utility loss)를 피하기 어렵다. 따라서 상황에 맞는 특화 방어 방법(Specialized Defense)의 개발이 실용적 관점에서 매우 중요하다.

셋째, **FTL에 대한 연구 공백**이다. FTL은 실제 산업 현장에서 가장 유용한 시나리오(데이터/피처가 모두 다른 기업 간 협업)임에도 불구하고, 프라이버시 분석은 HFL/VFL의 부수적인 결과물로만 다뤄지고 있다. FTL 전용 위협 모델과 벤치마크 프레임워크 구축이 시급하다.

## 📌 TL;DR

본 논문은 HFL, VFL, FTL이라는 세 가지 Federated Learning 패러다임을 통합하여, 프라이버시 위협을 6가지 유형으로 분류하고 이에 대응하는 일반 및 특화 방어 기법을 체계적으로 정리한 서베이 보고서이다. 특히 VFL의 고유 위협(레이블 추론, ID 누출)과 FTL의 연구 부족 상태를 지적하며, 향후 유틸리티를 보존하면서도 여러 위협을 동시에 해결할 수 있는 경량화된 범용 방어 기법 연구의 필요성을 강조한다.