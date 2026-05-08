# ATR-Bench: A Federated Learning Benchmark for Adaptation, Trust, and Reasoning

Tajamul Ashraf, Mohammed Mohsen Peerzada, Moloud Abdar, Yutong Xie, Yuyin Zhou, Xiaofeng Liu, Iqra Altaf Gillani, Janibul Bashir (2025)

## 🧩 Problem to Solve

연합 학습(Federated Learning, FL)은 데이터 프라이버시를 유지하면서 분산된 참여자들 간의 협동 모델 학습을 가능하게 하는 유망한 패러다임으로 성장해 왔다. 그러나 FL의 실제 적용 과정에서 마주하는 다양한 도전 과제들, 특히 데이터의 비균질성(Non-IID), 보안 취약성, 모델의 추론 능력 부족 등을 해결하기 위한 수많은 기법들이 제안되었음에도 불구하고, 이를 체계적으로 평가할 수 있는 표준화된 벤치마크가 부족한 실정이다. 기존의 벤치마크들은 일반화(Generalization), 강건성(Robustness), 또는 공정성(Fairness)과 같은 단일 차원에만 집중하는 경향이 있어, 여러 핵심 차원을 통합적으로 분석하고 공정하게 비교하는 것이 어렵다.

본 논문의 목표는 **Adaptation(적응성), Trust(신뢰성), Reasoning(추론 능력)**이라는 세 가지 핵심 차원을 통해 연합 학습을 분석하는 통합 프레임워크인 **ATR-Bench**를 제안하는 것이다. 이를 통해 FL 방법론들에 대한 체계적이고 총체적인 평가 기반을 마련하고, 현재 연구의 한계와 향후 나아가야 할 방향을 제시하고자 한다.

## ✨ Key Contributions

본 논문의 중심적인 기여는 연합 학습의 평가 기준을 단일 지표에서 다차원적인 체계로 확장했다는 점에 있다. 구체적인 기여 사항은 다음과 같다.

1. **ATR-Bench 프레임워크 제안**: Adaptation, Trustworthiness, Reasoning의 세 가지 차원으로 FL의 도전 과제를 분류하고, 각 테마에 대한 작업 설정(Task Formulation), 평가 기준, 그리고 현재 문헌에서의 연구 공백을 공식화하였다.
2. **광범위한 실증적 평가**: 데이터 불균형(Label/Domain Skew) 상황에서의 적응성, 적대적 환경에서의 강건성 및 공정성을 평가하기 위해 다양한 데이터셋과 최신 방법론들을 벤치마킹하였다.
3. **추론 능력(Reasoning)에 대한 로드맵 제시**: FL 분야에서 추론 능력 평가를 위한 성숙한 모델과 지표가 부족함을 인식하고, 문헌 조사를 바탕으로 추론 능력을 향상시키기 위한 개념적 프레임워크와 향후 연구 방향을 제시하였다.
4. **오픈소스 생태계 기여**: 분석에 사용된 전체 코드베이스와 최신 FL 연구 동향을 지속적으로 추적하는 큐레이션 저장소를 공개하여 연구의 재현성과 확장성을 높였다.

## 📎 Related Works

기존의 FL 관련 연구들은 주로 특정 문제 해결에 집중해 왔다.

- **데이터 비균질성 해결**: Non-IID 데이터 문제를 해결하기 위해 로컬 정규화(Local Regularization)나 개인화 레이어(Personalized Layers)를 도입한 연구들이 많았으나, 이는 주로 In-distribution 성능 향상에 치중되어 있었다.
- **도메인 적응 및 일반화**: Federated Domain Adaptation(FDA)과 Generalization(FDG) 연구들이 존재하지만, 이들 역시 특정 도메인 시프트 상황에 국한되어 분석되는 경우가 많았다.
- **보안 및 강건성**: Byzantine 공격이나 Backdoor 공격에 대응하는 강건한 집계(Robust Aggregation) 방법론들이 제안되었으나, 데이터의 다양성과 보안성 사이의 트레이드-오프에 대한 통합적인 분석은 부족했다.
- **공정성**: 기여도에 따른 보상(Collaboration Fairness)이나 성능의 균등한 배분(Performance Fairness)에 관한 연구들이 진행되었으나, 이를 적응성이나 강건성과 함께 통합적으로 다룬 사례는 드물다.

ATR-Bench는 이러한 파편화된 접근 방식을 통합하여, 하나의 프레임워크 내에서 적응성, 신뢰성, 추론 능력을 동시에 고려함으로써 실제 배포 환경에 더 가까운 총체적 평가를 수행한다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조 및 공식화

본 논문은 수평적 연합 학습(Horizontal Federated Learning, HFL) 설정을 기반으로 한다. $M$ 명의 클라이언트가 각각 프라이빗 데이터셋 $\mathcal{D}_i$를 보유하고 있으며, 목표는 모든 클라이언트의 국소적 경험적 위험(Local Empirical Risk)의 가중 합을 최소화하는 글로벌 파라미터 $w^*$를 찾는 것이다.

$$w^* = \text{argmin}_w \sum_{i=1}^M \alpha_i L_i(w; \mathcal{D}_i)$$

여기서 $L_i(w; \mathcal{D}_i)$는 클라이언트 $i$의 평균 손실 함수이며, $\alpha_i$는 가중치(일반적으로 데이터 크기에 비례)이다. 학습 절차는 **Broadcast $\rightarrow$ Local Update $\rightarrow$ Aggregation**의 세 단계로 구성되는 통신 라운드의 반복으로 이루어진다.

### 3가지 핵심 분석 차원

#### 1. Adaptation (적응성)

데이터의 Non-IID 특성으로 인한 분포 변화(Distribution Shift)를 해결하는 능력을 평가한다.

- **Cross-Client Shift**: 클라이언트 간 데이터 분포 차이로 인해 로컬 최적점이 서로 달라지는 문제. 이를 위해 클라이언트 정규화(Client Regularization), 데이터 증강(Client Augmentation), 서버 측 적응적 최적화(Server Operation) 기법들을 분석한다.
- **Out-of-Client Shift**: 학습에 참여하지 않은 새로운 도메인의 데이터에 대한 일반화 성능 문제. 이를 해결하기 위한 Federated Domain Adaptation(FDA)과 Federated Domain Generalization(FDG) 접근 방식을 다룬다.

#### 2. Trust (신뢰성)

강건성(Robustness)과 공정성(Fairness)을 통해 시스템의 신뢰 경계를 정의한다.

- **Byzantine Tolerance**: 악의적인 클라이언트가 전송하는 오염된 업데이트를 걸러내기 위해 거리 기반(Distance-based), 통계 기반(Statistical-based), 프록시 데이터 기반(Proxy-based) 필터링 기법을 적용한다.
- **Backdoor Defense**: 특정 트리거에 반응하도록 설계된 백도어 공격을 방어하기 위한 모델 정화(Sanitization) 및 인증된 방어(Certified Defenses)를 평가한다.
- **Fairness**: Shapley Value 등을 이용해 클라이언트의 실제 기여도를 측정하는 협력 공정성과, 모든 참여자가 유사한 성능을 얻도록 하는 성능 공정성을 다룬다.

#### 3. Reasoning (추론 능력)

단순한 예측을 넘어 구조화된 추론을 가능하게 하는 능력을 탐구한다. 현재의 기술적 한계로 인해 실험보다는 개념적 프레임워크를 제시한다.

- **해결 방안**: 해석 가능성이 포함된 증류(Distillation with Interpretability), 설명 가이드 기반 집계(Explanation-Guided Aggregation), 심볼릭-뉴럴 하이브리드 모델(Symbolic-Neural Hybrids) 등을 제안한다.
- **LLM 통합**: Chain-of-Thought(CoT) 증류나 연합 프롬프트 튜닝(Federated Prompt Tuning)을 통해 분산 환경에서의 추론 능력을 확보하는 방안을 제시한다.

## 📊 Results

### 실험 설정

- **데이터셋**: Label Skew 평가를 위해 CIFAR-10/100, MNIST, Fashion-MNIST를 사용하였으며, Domain Skew 및 Out-Client Shift 평가를 위해 Digits, Office Caltech, PACS, Office-31 데이터셋을 사용하였다.
- **데이터 불균형 구현**: Dirichlet 분포 $\text{Dir}(\beta)$를 사용하여 $\beta=0.5$ 설정을 통해 강한 데이터 비균질성을 시뮬레이션하였다.
- **측정 지표**:
  - $A_U$ (Cross-Client Accuracy): 클라이언트 간 평균 정확도.
  - $A_O$ (Out-of-Distribution Accuracy): 미학습 도메인에 대한 정확도.
  - $I$ (Accuracy Degradation): Byzantine 공격 전후의 성능 하락 폭.
  - $R$ (Backdoor Success Rate): 백도어 공격의 성공률.
  - $V$ (Accuracy Consistency): 도메인 간 정확도의 표준편차(낮을수록 공정함).

### 주요 결과

1. **Adaptation 성능**: Label Skew 상황에서 `FedProto`, `SCAFFOLD` 등이 기본 `FedAvg`보다 경쟁력 있는 성능을 보였으며, Out-of-Client Shift 상황에서는 FDA 기법인 `KD3A`가 Office Caltech 데이터셋에서 67.16%의 높은 정확도를 기록하며 강한 일반화 능력을 입증하였다.
2. **Byzantine 강건성**: Byzantine 공격 시나리오에서 `DnC` 방법론이 다양한 공격 유형(Pair Flipping, Random Noise 등)에 대해 가장 일관되게 강한 회복력을 보였다. 반면 프록시 데이터 기반 방법들은 외부 데이터 의존성이 높아 실용성에 한계가 있었다.
3. **백도어 방어**: `RLR` 및 `CRFL`과 같은 전문 방어 기법들이 백도어 공격에 대해 효과적인 억제력을 보였다.
4. **공정성 분석**: 대부분의 최적화 기법들이 협력 공정성(기여도 기반 보상)을 충분히 고려하지 않고 있으며, 도메인 스큐가 심할수록 성능 불균형($V$)이 심화되는 경향이 확인되었다.

## 🧠 Insights & Discussion

본 논문은 실험 결과를 통해 연합 학습 연구의 몇 가지 중요한 통찰을 제공한다.

첫째, **추론 벤치마크의 공백(Reasoning Benchmark Void)**이다. 현재 FL 연구는 주로 분류 정확도와 같은 성능 지표에 매몰되어 있으며, 모델이 '왜' 그런 결과를 도출했는지에 대한 추론 과정이나 해석 가능성에 대한 평가 체계가 전무하다. 이는 의료나 자율주행 같은 고위험 도메인 적용에 큰 걸림돌이 된다.

둘째, **재현성 딜레마(Reproducibility Dilemma)**이다. 많은 FL 논문들이 실험 설정이나 코드를 투명하게 공개하지 않아, 동일한 조건에서의 공정한 비교가 어렵다는 점이 지적되었다. ATR-Bench는 이를 해결하기 위해 통합된 분류 체계와 표준 프로토콜을 제공한다.

셋째, **강건성과 일반화의 트레이드-오프**이다. 데이터의 다양성을 수용하여 일반화 성능을 높이려는 시도는, 때때로 특이한 데이터를 가진 정상 클라이언트를 공격자로 오인하여 배제하게 만드는 강건성 메커니즘과 충돌할 수 있다. 따라서 정당한 다양성과 악의적인 공격을 구분할 수 있는 정교한 필터링 메커니즘이 필요하다.

## 📌 TL;DR

본 논문은 연합 학습의 평가 관점을 **적응성(Adaptation), 신뢰성(Trust), 추론 능력(Reasoning)**의 세 가지 차원으로 통합한 **ATR-Bench**를 제안한다. 수많은 기존 방법론들을 이 프레임워크 아래에서 체계적으로 벤치마킹하여, 특히 추론 능력 평가의 부재와 재현성 문제를 지적하였다. 이 연구는 파편화되어 있던 FL 평가 지표를 통합함으로써, 향후 더 강건하고 공정하며 지능적인 연합 학습 시스템을 개발하기 위한 표준적인 가이드라인을 제공한다는 점에서 중요한 가치를 지닌다.
