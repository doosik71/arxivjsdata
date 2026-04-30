# Towards More Efficient Data Valuation in Healthcare Federated Learning using Ensembling

Sourav Kumar, A. Lakshminarayanan, Ken Chang, Feri Guretno, Ivan Ho Mien, Jayashree Kalpathy-Cramer, Pavitra Krishnaswamy, and Praveer Singh (2022)

## 🧩 Problem to Solve

본 논문은 Federated Learning(FL) 환경에서 각 참여 기관이 기여한 데이터의 가치를 공정하게 평가하는 문제에 집중한다. 일반적으로 FL에 참여하는 여러 기관은 데이터의 양, 질, 그리고 다양성 측면에서 서로 다른 기여도를 가지게 된다. 이를 공정하게 정량화하기 위해 협동 게임 이론(Cooperative Game Theory)의 Shapley Value(SV)가 최적의 방법으로 제시되었으나, SV의 계산 복잡도는 참여자 수 $n$에 대해 지수 함수적($2^n$)으로 증가한다.

특히 헬스케어 분야의 FL은 수백 개의 장치가 참여하는 cross-device 설정보다는 수십 개의 신뢰할 수 있는 기관이 참여하는 cross-silo 설정이 일반적이다. 비록 참여 기관의 수가 수백 개에 달하지 않더라도, 30개 정도의 기관만 참여해도 $2^{30}$번의 모델 학습이 필요하므로 기존의 방식으로는 계산 비용이 천문학적으로 높다. 또한, 기존의 Monte Carlo 기반 근사(approximation) 기법들은 샘플링 과정에서 발생하는 무작위성으로 인해 데이터의 가치를 불공정하게 평가할 위험이 있다. 따라서 본 논문의 목표는 헬스케어 FL 환경(참여자 수 30명 이하)에서 정확도를 유지하면서도 계산 효율성을 획기적으로 높인 데이터 가치 평가 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 **SaFE (Shapley Value for Federated Learning using Ensembling)**라는 효율적인 SV 계산 기법을 제안한 것이다. SaFE의 중심 아이디어는 'SV 계산 과정의 근사'가 아닌 '모델의 근사(Model Approximation)'를 사용하는 것이다. 

기존 방식이 $2^n$개의 전체 FL 모델을 직접 학습시켜 각 조합의 성능을 측정했다면, SaFE는 각 기관의 데이터를 대표하는 가벼운 Logistic Regression(LR) 모델들을 먼저 생성하고, 이를 중앙 서버에서 Ensembling함으로써 $2^n$개의 조합에 대한 성능을 매우 빠르게 추론한다. 이를 통해 계산 복잡도는 유지하면서도 실제 학습에 소요되는 시간을 획기적으로 단축하여, 실질적으로 Exact SV에 근접한 값을 계산할 수 있게 한다.

## 📎 Related Works

논문은 Federated Learning을 cross-device와 cross-silo 두 가지 설정으로 구분하며, 헬스케어 분야는 후자에 해당함을 명시한다. 기존의 데이터 가치 평가 연구들은 주로 중앙 집중식(Centralized) 환경에서의 SV 계산에 집중되어 왔으며, 특히 Data Shapley와 같은 기법들이 제안되었다. 

그러나 중앙 집중식 환경에서도 데이터 포인트가 많을 경우 Exact SV 계산이 불가능하여 Monte Carlo(MC) 또는 Truncated Monte Carlo(TMC)와 같은 근사 기법을 사용한다. FL 환경에서의 SV 연구는 극히 제한적이며, 특히 cross-device 설정에 초점을 맞춘 기존 연구들은 모든 참여자의 기여도를 동시에 고려하지 못한다는 한계가 있다. 또한, MC 기반 근사법은 소수 클래스나 희귀 인종 데이터를 보유한 기관의 가치를 샘플링 여부에 따라 다르게 평가할 수 있어, 공정성 측면에서 취약점이 존재한다.

## 🛠️ Methodology

SaFE의 전체 파이프라인은 다음과 같은 3단계 과정으로 구성된다.

**Step 1: Traditional FL**
먼저 FedAvg와 같은 표준적인 모델 집계 기법을 사용하여 글로벌 FL 모델 $G$를 학습시킨다. 학습이 완료되면 모든 참여 기관은 동일한 글로벌 모델을 보유하게 된다.

**Step 2: Fine-Tuning 및 로컬 모델 생성**
각 기관은 보유한 로컬 데이터 $D_i$를 사용하여 글로벌 모델 $G$를 기반으로 로컬 모델을 생성한다. 이때 계산 효율성을 위해 복잡한 딥러닝 모델 대신 Logistic Regression(LR) 모델을 사용한다. 구체적으로는 글로벌 모델 $G$를 Feature Extractor로 사용하여 데이터의 특징 벡터(feature vector)를 추출하고, 이 벡터를 입력으로 하는 LR 모델 $L_i$를 학습시킨다. 학습된 $L_i$는 중앙 서버로 전송된다.

**Step 3: Ensembling을 통한 SV 계산**
중앙 서버는 수집된 모든 LR 모델 집합 $L = \{L_1, ..., L_n\}$을 사용하여 모든 가능한 조합($2^n$개)에 대해 Ensembling을 수행한다. 특정 조합 $S$에 대한 예측값은 해당 조합에 속한 LR 모델들의 Softmax 예측값을 결합하여 산출한다. 이후 다음과 같은 SV 방정식을 통해 각 플레이어 $i$의 가치 $\phi_i(v)$를 계산한다.

$$\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|!(n-|S|-1)!}{n!}(v(S \cup \{i\}) - v(S))$$

여기서 $v$는 Ensembled 모델의 성능 지표(예: Accuracy 또는 AUROC)를 나타내는 유틸리티 함수이다. 

**계산 효율성 분석**
전통적인 방식으로는 20개 기관 참여 시 $2^{20}$개의 FL 모델을 학습시켜야 하며, 모델 하나당 15분이 소요된다고 가정할 때 약 30년이 걸린다. 반면 SaFE는 1개의 FL 모델과 $n$개의 LR 모델만 학습시키면 되며, 나머지 $2^n$번의 연산은 단순한 Ensembling(추론) 단계이므로 모델당 약 15ms밖에 걸리지 않는다. 따라서 20개 기관 기준 전체 계산 시간을 약 4.5시간으로 단축시킬 수 있다.

## 📊 Results

**실험 설정**
- **데이터셋:** MNIST, CIFAR10 (Toy 데이터셋) / ROP(망막병증), Brain-MRI (의료 데이터셋)
- **설정:** IID 및 Non-IID 환경 모두에서 실험 수행
- **비교 대상:** Exact SV, Truncated Monte Carlo(TMC)
- **지표:** Model Accuracy, AUROC, Cosine Similarity (SV 벡터 간 유사도)

**주요 결과**
1. **LR Ensembling vs Global FL:** Table 1에 따르면, 로컬 LR 모델들을 Ensembling하여 만든 모델의 성능이 전통적인 FL 모델의 성능과 매우 유사함을 확인하였다. (예: Brain-MRI의 경우 양쪽 모두 AUROC 0.94 기록)
2. **SV 정확도:** SaFE로 계산한 SV와 Exact SV 간의 Cosine Similarity를 측정한 결과, IID 설정에서는 0.99 이상의 매우 높은 유사도를 보였으며, Non-IID 설정에서도 상당히 높은 유사도를 유지하였다 (Table 2).
3. **TMC와의 비교:** Figure 2의 결과에서 SaFE가 TMC 기반의 근사치보다 Exact SV에 훨씬 더 가까운 값을 생성함을 정량적으로 확인하였다. 이는 모델 근사 방식이 샘플링 기반 근사보다 더 안정적인 가치 평가가 가능함을 시사한다.

## 🧠 Insights & Discussion

**강점 및 의의**
본 논문은 헬스케어 FL의 특성(적은 수의 고가치 참여 기관)을 정확히 파악하여, 계산 불가능한 영역을 계산 가능한 영역으로 끌어들였다. 특히 단순한 모델 근사(LR Ensembling)만으로도 Exact SV에 근접한 결과를 낼 수 있음을 보임으로써, 데이터 가치 평가의 공정성과 효율성이라는 두 마리 토끼를 잡았다. 또한, 로컬 모델 $L_i$만을 전송하므로 데이터 프라이버시를 유지하면서도 중앙 서버에서 병렬 계산이 가능하다는 점이 강력한 이점이다.

**한계 및 비판적 해석**
SaFE는 여전히 $2^n$번의 연산을 수행하므로, 참여 기관의 수가 30~40명을 넘어서는 시점에서는 CPU 병렬화를 하더라도 다시 계산 병목 현상이 발생할 것이다. 논문에서는 1000개의 클라우드 VM을 사용하는 가정을 언급했으나, 이는 실제 적용 시 상당한 비용 부담이 될 수 있다. 또한, LR 모델이 딥러닝 모델의 복잡한 비선형 관계를 충분히 대변할 수 있는지에 대한 이론적 보장(Theoretical Guarantee)이 부족하며, 이는 향후 연구 과제로 남겨져 있다.

## 📌 TL;DR

본 논문은 헬스케어 Federated Learning에서 참여 기관의 데이터 기여도를 공정하게 평가하기 위한 효율적인 SV 계산 기법인 **SaFE**를 제안한다. SaFE는 $2^n$개의 무거운 FL 모델을 학습시키는 대신, 로컬 LR 모델들을 생성하고 이를 중앙 서버에서 Ensembling하는 '모델 근사' 방식을 통해 계산 시간을 획기적으로 단축하면서도 Exact SV에 매우 근접한 가치 평가 결과를 제공한다. 이 연구는 데이터가 매우 귀하고 불균형한 의료 AI 협업 환경에서 참여자에게 공정한 보상을 제공하고 저품질 데이터를 식별하는 데 중요한 기반이 될 수 있다.