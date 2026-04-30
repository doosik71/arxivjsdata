# Split Learning for collaborative deep learning in healthcare

Maarten G. Poirot, Praneeth Vepakomma, Ken Chang, Jayashree Kalpathy-Cramer, Rajiv Gupta, Ramesh Raskar (2019)

## 🧩 Problem to Solve

의료 분야에서 딥러닝 모델의 성능을 높이기 위해서는 방대한 양의 라벨링된 데이터가 필요하지만, 실제 환경에서는 여러 제약으로 인해 데이터 확보가 매우 어렵다. 첫째, 희귀 질환의 경우 개별 의료 기관이 보유한 샘플 크기가 너무 작아 모델의 일반화 성능을 확보하기 어렵다. 둘째, HIPAA와 같은 법적 규제 및 윤리적 문제로 인해 환자의 개인 정보가 포함된 raw data를 외부로 전송하거나 중앙 서버에 수집하는 것이 엄격히 제한된다. 셋째, 의료 기관들이 자신들의 데이터를 가치 있는 자산으로 여겨 공유를 꺼리는 경향이 있으며, 대규모 데이터를 중앙 집중식으로 관리하기 위한 저장 공간과 대역폭 비용 또한 큰 부담이 된다.

따라서 본 논문의 목표는 데이터의 프라이버시를 보호하면서도 여러 의료 기관이 협력하여 모델을 학습시킬 수 있는 분산 학습 방법론인 Split Learning을 의료 분야에 처음으로 적용하고, 그 효용성을 검증하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 신경망을 여러 개의 '링크(link)'로 분할하여, 일부는 중앙 서버에 두고 일부는 각 로컬 클라이언트(병원)에 배치하는 Split Learning 구조를 활용하는 것이다. 특히, 데이터의 프라이버시와 라벨의 보안을 동시에 달성하기 위해 'U-shaped' 구성을 제안하였다. 이 방식은 raw data와 label 모두를 로컬에 유지하면서, 중간 단계의 추상화된 표현(obfuscated intermediate representation)만을 주고받음으로써 데이터 유출 위험을 최소화하고, 계산 부하를 중앙 서버로 집중시켜 클라이언트의 컴퓨팅 자원 제약을 해결한다.

## 📎 Related Works

기존의 분산 학습 방법으로는 Model Averaging, Large Scale Synchronous Gradient Descent (LS-SGD), Federated Learning, Cyclical Weight Transfer 등이 있다. 그러나 이들은 다음과 같은 한계를 가진다.

1.  **동기화 문제**: Model Averaging과 LS-SGD는 모든 클라이언트가 업데이트를 완료할 때까지 기다려야 하는 synchronous training 방식이다. 이는 각 기관의 네트워크 속도나 하드웨어 성능이 다를 경우 심각한 병목 현상을 야기한다.
2.  **성능 저하**: Cyclical Weight Transfer와 같은 일부 방식은 설계 특성상 중앙 집중식 학습 대비 최적의 성능을 달성하지 못하는 경우가 있다.
3.  **프라이버시 및 부하**: 각 방법론마다 유출되는 정보의 양이 다르며, Federated Learning의 경우 클라이언트가 모델 전체를 학습시켜야 하므로 로컬 컴퓨팅 자원 소모가 크다.

반면, Split Learning은 비동기적 학습이 가능하며, 모델의 대부분을 서버에 배치함으로써 클라이언트의 계산 부담을 크게 줄일 수 있다는 차별점을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조: U-shaped Split Learning
본 논문에서 구현한 U-shaped 구성은 전체 신경망을 세 가지 링크로 분할한다.

1.  **Front (Local)**: 로컬 클라이언트가 보유한다. raw data를 입력받아 처리한 후, 의미를 알 수 없게 암호화된 중간 표현(intermediate representation)을 생성하여 중앙 서버로 전송한다.
2.  **Center (Central)**: 중앙 서버가 보유한다. Front로부터 받은 중간 표현을 입력받아 신경망의 대부분의 계산을 수행하고, 다시 또 다른 중간 표현을 생성하여 Back 링크로 전송한다.
3.  **Back (Local)**: 다시 로컬 클라이언트가 보유한다. Center로부터 받은 표현을 디코딩하여 최종 출력값을 생성한다. 이 단계에서 로컬에 저장된 실제 라벨과 비교하여 손실(loss)을 계산하고 기울기(gradient)를 산출한다.

### 학습 절차 및 흐름
학습은 순차적으로 진행된다. 하나의 클라이언트가 한 에포크(epoch) 동안 네트워크를 학습시키면, 해당 클라이언트의 로컬 상태(Front와 Back의 가중치)가 다음 클라이언트로 복사되어 업데이트된다. 이러한 방식은 모든 클라이언트의 응답을 기다릴 필요가 없으므로 동기화 문제를 해결한다.

### 실험 설정
본 연구에서는 두 가지 의료 데이터셋을 사용하였다.
*   **Diabetic Retinopathy (DR)**: 9,000장의 안저 사진을 이용한 이진 분류 문제이다. $\text{ResNet-34}$ 아키텍처를 사용하였으며, $\text{Binary Cross Entropy Loss}$를 통해 학습하였다.
*   **CheXpert**: 156,535장의 흉부 X-ray 영상을 이용한 14개 항목의 다중 라벨 분류 문제이다. $\text{DenseNet-121}$ 아키텍처를 사용하였으며, $\text{Sigmoid Binary Cross Entropy Loss}$를 적용하였다.

두 실험 모두 $\text{Adam Optimizer}$ ($\beta_1=0.9, \beta_2=0.999, \text{learning rate}=10^{-4}$)를 사용하였다.

## 📊 Results

실험은 Split Learning 구성과 non-collaborative(협력하지 않는 단일 기관 학습) 구성을 비교 분석하였다. 참여 클라이언트의 수를 1명에서 50명까지 변화시키며 성능을 측정하였다.

1.  **Diabetic Retinopathy (DR) 결과**:
    *   Split Learning의 경우 클라이언트 수에 관계없이 높은 정확도를 일정하게 유지하였다.
    *   반면, non-collaborative 구성에서는 데이터가 여러 클라이언트로 분산될수록(즉, 개별 클라이언트의 샘플 사이즈가 작아질수록) 정확도가 급격히 하락하였다. (예: 1명일 때 $0.869 \rightarrow 50명일 때 $0.588)

2.  **CheXpert 결과**:
    *   성능 지표로 AUROC를 사용하였으며, Split Learning이 non-collaborative 설정보다 월등히 높은 성능을 보였다.
    *   특히 클라이언트 수가 2명을 넘어가는 시점부터 두 그룹 간의 성능 차이가 통계적으로 유의미하게 나타났다 ($p < 0.001$).

결론적으로, Split Learning은 데이터를 중앙으로 모으지 않고도 중앙 집중식 학습과 유사한 성능을 낼 수 있음을 입증하였다.

## 🧠 Insights & Discussion

본 논문은 의료 데이터의 고질적인 문제인 '데이터 사일로(Data Silo)' 현상을 Split Learning으로 해결할 수 있음을 보여주었다. 특히 U-shaped 구조를 통해 raw data뿐만 아니라 label까지 로컬에 유지함으로써 프라이버시 보호 수준을 높였다는 점이 강점이다. 또한, 계산 부하를 서버로 이전함으로써 저사양 장비를 가진 의료 기관도 협력 학습에 참여할 수 있는 가능성을 제시하였다.

다만, 본 연구에서는 non-collaborative 설정과의 비교에 집중하였으며, Federated Learning이나 LS-SGD와 같은 다른 분산 학습 방법론과의 직접적인 성능 비교는 의료 데이터셋 상에서 수행되지 않았다. 또한, 실제 의료 현장에 배포했을 때의 네트워크 지연 시간(latency)이나 통신 오버헤드에 대한 상세 분석이 부족하다는 점이 한계로 남는다.

## 📌 TL;DR

이 논문은 의료 데이터의 프라이버시 규제를 준수하면서 다기관 협력 학습을 가능하게 하는 **U-shaped Split Learning** 방법론을 제안하였다. 안저 사진 및 흉부 X-ray 데이터셋을 통해 실험한 결과, raw data를 공유하지 않고도 단일 기관 학습보다 월등히 높은 성능을 달성하였으며, 이는 향후 의료 AI 모델의 데이터 부족 문제를 해결하고 실질적인 다기관 협업 연구를 가능하게 하는 중요한 발판이 될 것으로 기대된다.