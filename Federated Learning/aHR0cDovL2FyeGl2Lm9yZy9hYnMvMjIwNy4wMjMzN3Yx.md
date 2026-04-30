# Federated and Transfer Learning: A Survey on Adversaries and Defense Mechanisms

Ehsan Hallaji, Roozbeh Razavi-Far and Mehrdad Saif (2022)

## 🧩 Problem to Solve

본 논문은 Federated Learning (FL)과 Transfer Learning (TL)이 결합된 Federated Transfer Learning (FTL) 환경에서의 보안 취약점과 이를 방어하기 위한 메커니즘을 체계적으로 분석하는 것을 목표로 한다. 

전통적인 FL은 참여자들이 유사한 속성(attribute)의 데이터를 보유해야 한다는 제약이 있으나, 실제 산업 현장(예: 의료, 금융)에서는 데이터의 속성이 서로 다른 경우가 많다. TL은 이를 해결하기 위해 한 도메인에서 학습된 지식을 다른 도메인에 적용하는 방식을 제공하며, FTL은 이 두 개념을 결합하여 데이터의 속성과 샘플의 중첩이 적은 환경에서도 협력 학습을 가능하게 한다. 하지만 이러한 구조적 복잡성은 시스템의 프라이버시와 성능을 저해하는 새로운 공격 벡터를 생성한다. 따라서 본 연구는 FTL 시스템을 위협하는 잠재적 취약점을 발굴하고, 현재 제안된 방어 기법들을 종합적으로 검토하여 안전한 FTL 프레임워크 구축을 위한 기초 자료를 제공하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 FL과 TL의 교차 지점에서 발생하는 보안 위협과 방어 전략에 대한 포괄적인 택소노미(Taxonomy)를 제시한 것이다. 특히 다음과 같은 직관적인 분석을 제공한다.

1. **통합적 위협 모델 분석**: FL과 TL 각각에서 발생하는 공격들을 분류하고, 두 영역의 공통 분모가 FTL 시스템에 어떤 구체적인 위협이 되는지를 명시하였다.
2. **다층적 방어 메커니즘 체계화**: 프라이버시 보호(Privacy Preserving)와 모델 강건성(Model Robustness)이라는 두 가지 큰 축을 중심으로, 암호학적 방법론부터 알고리즘적 최적화까지의 방어 기법을 정리하였다.
3. **FTL 특화 취약점 식별**: 전이 학습 과정에서 사용되는 Pre-trained 모델(Teacher model)을 통한 Backdoor 공격이나, 타겟 도메인의 특성을 모방하는 Adversarial attack 등 FTL 특유의 위험 요소를 분석하였다.

## 📎 Related Works

논문은 FL을 세 가지 범주로 구분하여 설명하며, 각 접근 방식의 한계와 FTL의 필요성을 역설한다.

- **Horizontal Federated Learning**: 참여자들이 동일한 속성의 데이터를 가지되 서로 다른 사용자의 데이터를 보유한 경우이다.
- **Vertical Federated Learning**: 샘플 공간은 상당 부분 중첩되나, 각 참여자가 보유한 속성이 서로 다른 경우이다.
- **Federated Transfer Learning (FTL)**: 샘플과 속성 공간의 중첩이 모두 최소화된 상황에서 지식을 전이하는 방식이다.

기존의 FL 연구들은 주로 데이터의 동질성을 가정하거나 단순한 프라이버시 보호에 집중하였으나, 실제 산업 데이터의 이질성(Heterogeneity) 문제를 해결하기 위해 TL의 도입이 필수적임을 강조한다. 또한, 기존 FTL 프레임워크들이 주로 특정 응용 분야의 맞춤형 구현에 치중되어 있어, 일반적인 보안 위협과 방어 체계에 대한 종합적인 분석이 부족했다는 점을 차별점으로 제시한다.

## 🛠️ Methodology

본 논문은 서베이 논문으로서 새로운 알고리즘을 제안하기보다, 기존 문헌들을 분석하여 FTL의 보안 구조를 체계화하는 방법론을 취한다.

### 1. 공격 모델 (Threat Models)
공격자를 다음과 같이 분류하여 분석한다.
- **Outsider Adversaries**: 통신 라인의 도청자나 서비스화된 모델의 사용자.
- **Insider Adversaries**: 서버나 네트워크 엣지에서 공격을 주도하는 Byzantine 또는 Sybil 공격자.
- **Semi-Honest Adversaries**: 프로토콜은 준수하지만, 수신된 파라미터를 통해 다른 사용자의 숨겨진 상태를 알아내려는 공격자.

### 2. 성능 및 프라이버시 공격 메커니즘
- **성능 공격 (Attacks on Performance)**: 
    - **Data Poisoning**: 레이블 플리핑(Label flipping)을 통한 DoS 공격이나, 특정 트리거에만 오작동하게 만드는 Backdoor 공격이 포함된다.
    - **Model Poisoning**: Gradient Manipulation을 통해 전역 모델의 정확도를 낮추거나, 손실 함수에 편향된 항을 추가하는 Training Objective Manipulation을 수행한다.
- **프라이버시 공격 (Attacks on Privacy)**: 
    - **Model Inversion**: 모델의 출력을 통해 입력 데이터의 민감한 특징을 역추적한다.
    - **Membership Inference**: 특정 샘플이 학습 데이터셋에 포함되었는지 여부를 판별한다.
    - **GAN Reconstruction**: GAN을 이용하여 학습 데이터와 통계적으로 유사한 가짜 샘플을 생성하여 정보를 유출한다.

### 3. 방어 메커니즘 (Defense Mechanisms)
- **프라이버시 보호**: 
    - **Homomorphic Encryption (HE)**: 데이터를 암호화한 상태에서 연산을 수행하여 평문 노출을 방지한다. (FHE, PHE, SHE로 구분)
    - **Secure Multiparty Computation (SMC)**: 여러 당사자가 입력값을 공유하지 않고 함수 값을 계산하는 암호학적 방법이다.
    - **Differential Privacy (DP)**: 업데이트 값에 무작위 노이즈를 추가하여 개별 데이터의 존재 여부를 숨긴다.
- **모델 강건성 강화**: 
    - **Anomaly Detection**: 업데이트 값의 통계적 특성이나 거리 측정(Distance measure)을 통해 악성 업데이트를 탐지하고 제거한다.
    - **Robust Aggregation**: 단순 평균(FedAvg) 대신 중앙값 회귀(Median regression) 등을 사용하여 이상치에 강건한 집계 방식을 적용한다.
    - **Adversarial Training**: $\min \max$ 최적화 문제를 통해 적대적 샘플을 생성하고 이를 다시 학습에 이용하여 모델의 내성을 키운다.

## 📊 Results

본 논문은 정량적인 실험 결과보다는 기존 연구들의 분석 결과를 종합한 정성적 결과를 제시한다.

1. **공통 위협 식별**: FL과 TL에서 공통적으로 발생하는 위협(Backdoor, Membership Inference, Feature Inference, Adversarial Samples)이 FTL 시스템의 가장 치명적인 약점이 됨을 확인하였다.
2. **방어 기법의 Trade-off**: 
    - **암호화 기반 방어(HE, SMC)**: 높은 보안성을 제공하지만, 계산 오버헤드와 통신 비용이 급격히 증가하여 대규모 시스템 적용이 어렵다.
    - **노이즈 기반 방어(DP)**: 통신 비용은 낮으나, 주입된 노이즈로 인해 모델의 정확도(Utility)가 하락하는 문제가 발생한다.
3. **FTL 특화 공격 경로**: TL 과정에서 사용되는 Pre-trained Teacher 모델이 오염되었을 경우, 이를 통해 전이된 Student 모델까지 Backdoor가 전파될 수 있음을 분석하였다.

## 🧠 Insights & Discussion

### 강점 및 해석
본 논문은 파편화되어 있던 FL과 TL의 보안 논의를 FTL이라는 하나의 틀로 통합하여 분석했다는 점에서 학술적 가치가 높다. 특히, 프라이버시 보호와 모델 강건성이라는 두 목표가 서로 상충(Trade-off) 관계에 있음을 명확히 지적하였다. 예를 들어, 프라이버시를 위해 데이터를 암호화하거나 노이즈를 추가하면, 서버가 업데이트 값을 검사할 수 없게 되어 Poisoning 공격에 더욱 취약해지는 역설적인 상황이 발생한다.

### 한계 및 미해결 질문
- **Non-i.i.d. 데이터 문제**: 많은 방어 기법(특히 Adversarial Training)이 데이터가 독립 동일 분포(i.i.d.)라는 가정하에 설계되었으나, 실제 FL/FTL 환경은 매우 이질적인 Non-i.i.d. 특성을 가진다. 이 환경에서 방어 기법들이 실제로 얼마나 효과적인지에 대한 실증적 분석이 부족하다.
- **분산 FTL의 부재**: 현재 대부분의 FTL 프로토콜은 중앙 서버 기반이다. 서버 자체가 신뢰할 수 없는 환경을 위한 Decentralized FTL 보안 모델에 대한 연구가 여전히 미흡하다.

### 비판적 논의
논문은 다양한 방어 기법을 나열하고 있으나, 각 기법을 어떤 상황에서 우선적으로 적용해야 하는지에 대한 결정 가이드라인(Decision Matrix)은 제공하지 않는다. 실제 시스템 설계자에게는 단순한 나열보다 비용-효과 분석에 기반한 최적의 방어 조합(Combination of defenses) 제안이 더 유용할 것이다.

## 📌 TL;DR

본 논문은 Federated Learning과 Transfer Learning이 결합된 FTL 환경에서 발생할 수 있는 보안 위협과 방어 기법을 종합적으로 분석한 서베이 논문이다. 데이터 중첩이 적은 환경에서의 협력 학습이라는 FTL의 특성이 가져오는 새로운 취약점을 식별하였으며, 암호학적 기법과 알고리즘적 강건성 향상 방안을 체계적으로 분류하였다. 이 연구는 향후 프라이버시 보호와 모델 성능, 그리고 공격 내성을 동시에 달성할 수 있는 통합적 FTL 보안 프레임워크 설계의 이정표 역할을 할 것으로 기대된다.