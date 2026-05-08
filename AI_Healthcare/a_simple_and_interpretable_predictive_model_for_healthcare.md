# A Simple and Interpretable Predictive Model for Healthcare

Subhadip Maji, Raghav Bali, Sree Harsha Ankem, Kishore V Ayyadevara (2020)

## 🧩 Problem to Solve

현재 질병 예측 분야에서는 Deep Learning 기반 모델들이 최첨단 솔루션으로 자리 잡고 있다. 특히 RNN과 다양한 Attention mechanism을 결합하여 예측 성능을 높이고 해석 가능성(Interpretability)을 제공하려는 시도가 많다. 그러나 이러한 Deep Learning 모델들은 수백만 개의 학습 가능한 파라미터를 가지고 있어, 학습과 배포 단계에서 막대한 양의 컴퓨팅 자원과 데이터가 필요하다는 치명적인 단점이 있다. 이는 실제 의료 현장에서 모델의 도입을 어렵게 만드는 경제적, 기술적 진입 장벽이 된다.

본 논문은 이러한 복잡성 문제를 해결하기 위해 EHR(Electronic Health Records) 데이터에 적용 가능한, 단순하면서도 해석 가능한 non-deep learning 기반의 예측 모델을 개발하는 것을 목표로 한다. 특히 기존 연구들이 간과했던 '첫 발생(First Occurrence)' 예측 작업에 집중하여, 정교한 Deep Learning 모델들에 대항할 수 있는 강력한 Baseline을 제시하고 실제 의료 환경에서의 실용성을 입증하고자 한다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 복잡한 시퀀스 모델링 대신 최적화된 트리 기반 앙상블 모델인 XGBoost를 사용하여 성능과 효율성을 동시에 잡는 것이다. 주요 기여 사항은 다음과 같다.

1. **강력한 Baseline 제시**: 파라미터 수가 매우 많은 Deep Learning 모델들과 대등하거나 이를 능가하는 성능을 가진 트리 기반 모델의 가능성을 입증하였다.
2. **새로운 데이터 준비 파이프라인**: 시퀀셜한 EHR 데이터를 non-deep learning 모델이 처리할 수 있도록 효과적으로 변환하는 데이터 준비 방법을 제안하였다.
3. **모델 불가지론적 해석 방법 적용**: Deep Learning의 Attention mechanism과 유사한 수준의 해석력을 제공하기 위해 SHAP(SHapley Additive exPlanations)를 도입하여 개별 환자 수준의 예측 근거를 제시하였다.
4. **실용적 대안 제시**: 낮은 컴퓨팅 비용으로도 높은 성능을 낼 수 있는 효율적인 배포 대안을 제시하였다.

## 📎 Related Works

기존의 의료 예측 연구들은 주로 RNN(특히 LSTM)과 Attention mechanism을 결합한 형태를 띤다. 대표적으로 RETAIN은 Reverse Time Attention mechanism을 사용하여 임상적으로 해석 가능한 예측 모델을 제안하였으며, Dipole은 Bidirectional RNN을 도입하여 환자의 과거와 미래 의료 경험을 모두 캡처함으로써 RETAIN의 단점을 보완하였다.

하지만 저자들은 기존 연구들이 다음과 같은 한계가 있다고 지적한다. 첫째, 비교 대상으로 Logistic Regression과 같은 매우 단순한 모델만을 사용하여 본인들의 성능을 과시하는 경향이 있다. 둘째, 질병의 '첫 발생'을 예측하는 제약 조건을 고려하지 않는 경우가 많다. 실제 의료 현장에서는 질병의 재발보다 첫 발생을 조기에 예측하여 예방 조치를 취하는 것이 훨씬 더 중요함에도 불구하고, 기존 모델들은 반복 발생 사례가 포함된 데이터셋에서 성능을 높이는 경향이 있다.

## 🛠️ Methodology

### 데이터 준비 및 특성 공학 (Data Preparation)

본 연구는 각 환자의 24개월 historical timeline 데이터를 사용하며, 예측의 실용성을 위해 학습 기간과 첫 발생일 사이에 3개월의 간격(delta)을 둔다.

환자의 진료 기록을 $H=\{t_1, t_2, ..., t_N\}$라고 할 때, 각 타임스텝 $t_i$는 다음과 같이 구성된다.
$$t_i = \{ICD^{(1..N)}, CPT^{(1..N)}, demographics\}$$
여기서 $ICD$는 진단 코드, $CPT$는 처치 코드, $RX$는 처방 코드를 의미한다.

대상 질병 $D$의 첫 발생을 예측하기 위해, 저자들은 시퀀셜한 데이터를 정적 벡터로 변환하는 'dissolve' 과정을 거친다. 환자 $p$의 특성 벡터는 다음과 같이 정의된다.
$$p = \{ICD_{1..N}, CPT_{1..N}, RX_{1..N}, age, gender\}$$

특성 값의 생성 방식에 대해 두 가지 실험을 진행하였다.

1. **Count-based**: 각 코드의 출현 횟수를 값으로 설정한다.
    $$p_i = \{X_i, y_i\} : \{(n_{11}, n_{12}, ..., a_1, g_1, ...), y_i\}$$
    ($n_{ij}$는 환자 $i$의 $ICD_j$ 카운트, $a_i$는 나이, $g_i$는 성별, $y_i$는 응답 변수)
2. **Binary-based**: 각 코드의 존재 여부를 이진값(0 또는 1)으로 설정한다.

### 모델 아키텍처 및 학습 절차

모델로는 트리 기반 부스팅 알고리즘인 **XGBoost**를 선택하였다. 이는 낮은 편향(low bias), 이상치에 대한 강건함, 빠른 학습 및 추론 속도라는 장점이 있기 때문이다.

하이퍼파라미터 튜닝을 위해 일반적인 Grid Search 대신 **Greedy Search** 방식을 사용하였다. 그 과정은 다음과 같다.

1. 기본 설정의 $xgb_{def}$ 모델을 베이스로 설정한다.
2. 하이퍼파라미터 집합 $H = \{learning\_rate, n\_estimators, ..., reg\_lambda\}$를 중요도 순으로 정렬한다.
3. 다른 파라미터를 고정시킨 채, Validation set에서 최적의 성능을 내는 $h_i \in H$ 값을 하나씩 찾아 업데이트한다.

### 추론 및 해석 절차 (Interpretability)

- **Global Interpretability**: XGBoost 자체의 feature importance 기능을 사용하여 전체 데이터셋에서 어떤 변수가 중요한지 분석한다.
- **Instance-level Interpretability**: 개별 환자에 대해 어떤 특성이 예측 결과에 기여했는지 분석하기 위해 모델 불가지론적 방법인 **SHAP**를 사용한다. SHAP value가 양수이면 예측 확률을 높이는 방향으로, 음수이면 낮추는 방향으로 작용함을 통해 개별 환자의 진단 근거를 시각화(Force plot)한다.

## 📊 Results

### 실험 설정

- **데이터셋**: 당뇨병(Diabetes), 심부전(Heart Failure), 신부전(Kidney Failure) 3종의 EHR 데이터셋을 사용하였다.
- **지표**: 클래스 불균형이 심하므로 Accuracy 대신 **ROC-AUC**를 주 지표로 사용하였으며, 상위 30% 확률 예측군 내의 정답 비율인 **Recall@30**을 추가 지표로 활용하였다.
- **비교 대상**: Logistic Regression, $xgb_{base}$ (기본 설정), $xgb_{opt}$ (최적화), RETAIN, Dipole.

### 주요 결과

최적화된 XGBoost 모델($xgb_{opt}$)은 모든 대상 질병에서 Deep Learning 모델인 RETAIN과 Dipole보다 우수한 성능을 보였다.

| 모델 | 당뇨병 (ROC-AUC) | 심부전 (ROC-AUC) | 신부전 (ROC-AUC) |
| :--- | :---: | :---: | :---: |
| RETAIN | $0.7831 \pm 0.0062$ | $0.8385 \pm 0.0020$ | $0.8202 \pm 0.0010$ |
| Dipole | $0.7901 \pm 0.0085$ | $0.8291 \pm 0.0090$ | $0.8211 \pm 0.0080$ |
| $xgb_{opt}$ | $\mathbf{0.8281 \pm 0.0005}$ | $\mathbf{0.8531 \pm 0.0005}$ | $\mathbf{0.8497 \pm 0.0007}$ |

또한, ICD 코드를 3자리로 절삭(ICD3)하지 않고 전체 코드(ICD-Full)를 사용했을 때 성능이 더욱 향상됨을 확인하였다. (예: 심부전 예측 시 $xgb_{opt}$의 ROC-AUC가 0.8626까지 상승)

## 🧠 Insights & Discussion

### 트리 기반 모델의 우위 원인

본 논문은 단순한 모델이 복잡한 모델을 이긴 이유를 두 가지 관점에서 분석한다.

1. **데이터의 특성**: 의료 전문가와의 협의 결과, 당뇨나 심부전 같은 질병은 시퀀셜한 시간 순서보다는 '어떤 진단을 받았는가'라는 상태 정보가 더 중요하다는 점이 밝혀졌다. 또한 EHR의 시간 정보는 실제 발병일이 아니라 '병원 방문일' 기준이므로 노이즈가 많아, 정교한 시퀀스 모델링이 오히려 독이 될 수 있다.
2. **알고리즘적 관점**: Deep Learning은 비정형 데이터에는 강하지만, 정형(Tabular) 데이터에서는 트리 기반 앙상블 모델이 여전히 압도적이다. 트리 모델은 고차원이면서 희소한(Sparse) 데이터에서 하이퍼플레인 경계를 찾는 **Manifold Learning**에 매우 효율적이기 때문이다.

### 한계점 및 보완책

XGBoost의 가장 큰 한계는 Deep Learning과 달리 '방문 단위(visit-level)'의 중요도를 직접 제공하지 못한다는 점이다. 저자들은 이를 해결하기 위해 **'환자 수준의 중요 특성 추출 $\rightarrow$ 해당 특성이 나타난 방문 시점 역추적'**이라는 워크아운드(Workaround)를 제시하여 임상의에게 보조 정보를 제공할 수 있다고 주장한다.

## 📌 TL;DR

본 연구는 복잡한 Deep Learning 모델 대신 최적화된 **XGBoost**를 사용하여 EHR 데이터 기반의 질병 첫 발생 예측을 수행하였으며, 결과적으로 **RETAIN, Dipole과 같은 최신 DL 모델보다 높은 ROC-AUC 성능과 빠른 추론 속도**를 달성하였다. 특히 **SHAP**를 통해 개별 환자 수준의 해석 가능성을 확보함으로써, 의료 현장에서 비용 효율적이고 신뢰할 수 있는 예측 도구로서의 가능성을 보여주었다. 이는 정형 의료 데이터 분석 시 무조건적인 Deep Learning 도입보다는 데이터 특성에 맞는 모델 선택이 중요함을 시사한다.
