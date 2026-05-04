# RADAR: Mechanistic Pathways for Detecting Data Contamination in LLM Evaluation

Ashish Kattamuri, Harshwardhan Fartale, Arpita Vats, Rahul Raja, Ishita Prasad (2025)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 LLM(Large Language Model) 평가에서 발생하는 **데이터 오염(Data Contamination)** 문제이다. 데이터 오염이란 모델이 평가 데이터셋의 일부를 학습 과정에서 미리 접하여, 실제 추론(Reasoning) 능력이 없음에도 불구하고 단순히 학습된 내용을 암기하여 출력(Recall)함으로써 성능 지표가 인위적으로 높아지는 현상을 의미한다.

이 문제의 중요성은 기존의 평가 지표가 모델의 진정한 지적 능력과 단순 암기 능력을 구분하지 못한다는 점에 있다. 특히, 최신 모델들이 벤치마크 데이터셋의 상당 부분을 학습 데이터에 포함하고 있을 가능성이 높아짐에 따라, 평가 결과의 신뢰성이 심각하게 훼손되고 있다. 따라서 본 논문의 목표는 모델의 외부 출력값이 아닌 내부의 계산 역학(Internal Computation Dynamics)을 분석하여, 모델이 응답을 생성할 때 **'단순 회상(Recall)'**을 했는지 아니면 **'실제 추론(Reasoning)'**을 했는지를 판별하는 프레임워크인 RADAR를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Mechanistic Interpretability(기계적 해석 가능성)**를 활용하여 회상과 추론의 내부 신호(Internal Signatures)가 서로 다르다는 점을 이용하는 것이다.

- **회상(Recall)**: 특정 정보에 집중된 Attention 패턴을 보이며, 신뢰도(Confidence)가 매우 빠르게 수렴하는 경향이 있다.
- **추론(Reasoning)**: 더 넓은 범위의 네트워크 자원을 사용하며, Attention이 분산되어 있고 신뢰도가 점진적으로 안정화되는 양상을 보인다.

RADAR는 이러한 직관을 바탕으로 표면적 특성(Surface features)과 기계적 특성(Mechanistic features)을 모두 추출하여, 학습 데이터에 대한 접근 권한 없이도 모델의 내부 상태만으로 데이터 오염 여부를 탐지할 수 있는 도구를 제공한다.

## 📎 Related Works

기존의 데이터 오염 탐지 방법들은 주로 다음과 같은 접근 방식을 취했다.
1. **데이터셋 비교**: 평가 데이터와 학습 코퍼스를 직접 비교하는 방식이다.
2. **N-gram 중첩 확인**: 출력된 텍스트와 학습 데이터 간의 n-gram 겹침 정도를 측정한다.
3. **축자적 출력(Verbatim outputs) 탐지**: 모델이 학습 데이터를 그대로 출력하는지 확인한다.

그러나 이러한 기존 방식들은 다음과 같은 한계가 있다.
- **데이터 접근성**: 폐쇄형 모델의 경우 학습 데이터에 접근할 수 없어 적용이 불가능하다.
- **우회 가능성**: 데이터가 약간 변형(Paraphrased)된 경우 탐지하지 못한다.
- **본질적 구분 불가**: 모델이 문제를 풀었을 때, 그것이 내부적인 추론의 결과인지 아니면 암기된 패턴의 결과인지 구분할 수 있는 메커니즘이 없다.

RADAR는 외부 텍스트 비교가 아닌 내부 활성화 표현(Activation Representation)을 분석함으로써 이러한 한계를 극복한다.

## 🛠️ Methodology

### 1. 전체 시스템 구조
RADAR의 파이프라인은 크게 세 가지 구성 요소로 이루어져 있다.
1. **Mechanistic Analyzer**: 대상 LLM(예: DialoGPT-medium)에서 Attention 가중치와 Hidden states를 추출한다.
2. **Feature Extraction**: 추출된 내부 상태를 바탕으로 37개의 특성(Surface 17개, Mechanistic 20개)을 계산한다.
3. **Classifier**: 앙상블 분류기를 통해 최종적으로 해당 응답이 'Recall'인지 'Reasoning'인지 판별한다.

### 2. 특성 공학 (Feature Engineering)
분석을 위해 사용되는 37개의 특성은 다음과 같이 구분된다.

#### A. 표면적 특성 (Surface Features - 17개)
모델의 레이어별 출력 궤적을 분석하여 예측 역학을 캡처한다.
- **신뢰도 기반 특성**: 평균 신뢰도 $\bar{c}$, 표준편차 $\sigma_c$, 최대/최소 신뢰도 및 범위($\Delta c$), 신뢰도가 최대치에 도달하는 레이어($l^*$)와 수렴 속도($v_{conv}$) 등이 포함된다.
- **궤적 역학 특성**: 신뢰도 변화의 진동 횟수(Oscillation count), 초기 및 후기 레이어의 신뢰도 평균 등이 포함된다.
- **정보 이론 특성**: 레이어별 평균 엔트로피 $\bar{H}$, 엔트로피 변화량 $\Delta H$, 정보 획득량(Information Gain) 등을 측정한다.

#### B. 기계적 특성 (Mechanistic Features - 20개)
Attention 가중치와 Hidden states의 심층적인 특성을 분석한다.
- **Attention 특성화**: 엔트로피가 특정 임계값 $\tau=1.5$보다 낮은 Specialized Heads의 개수($N_{spec}$)와 특성화 점수를 계산한다.
- **회로 역학(Circuit Dynamics)**: 활성화 분산 성장과 노름(Norm) 성장 궤적의 곱으로 정의되는 회로 복잡도 $C_{circuit} = \sigma^2_{var} \cdot \gamma_{norm}$를 측정한다.
- **중재 및 민감도**: 구성 요소 제거 시의 강건성(Ablation robustness)과 개입 민감도(Intervention sensitivity)를 분석한다.
- **작업 기억(Working Memory)**: Hidden state의 분산과 랭크 진화(Rank evolution)를 통해 정보 축적 과정을 측정한다.
- **인과적 효과**: 직접적 로짓 기여도(Direct logit attribution)와 인과적 중재 점수(Causal mediation score) 등을 통해 출력에 미치는 영향을 분석한다.

### 3. 분류 시스템 및 학습 절차
추출된 특성 벡터 $x$는 $\text{StandardScaler}$를 통해 정규화된다.
$$x'_i = \frac{x_i - \mu_i}{\sigma_i}$$

분류기는 Random Forest, Gradient Boosting, SVM, Logistic Regression의 4가지 모델로 구성된 앙상블 시스템을 사용한다. 각 모델 $j$가 예측한 하드 라벨 $\hat{y}_j$와 확률 $p_j$를 다음과 같이 통합한다.
$$\hat{y} = 1 \left[ \frac{1}{M} \sum_{j=1}^{M} \hat{y}_j > \frac{1}{2} \right], \quad \bar{p} = \frac{1}{M} \sum_{j=1}^{M} p_j$$
여기서 $\hat{y}=1$은 Recall, $\hat{y}=0$은 Reasoning을 의미한다. 최종 신뢰도 점수(conf)는 예측 라벨에 따라 $\bar{p}$ 또는 $1-\bar{p}$로 결정된다.

## 📊 Results

### 1. 실험 설정
- **대상 모델**: `microsoft/DialoGPT-medium`
- **데이터셋**: 
    - 학습 세트: 30개 예시 (Recall 15, Reasoning 15)
    - 테스트 세트: 100개 예시 (Clear Recall 20, Clear Reasoning 20, Challenging 30, Complex Reasoning 30)
- **평가 지표**: Accuracy (정확도)

### 2. 정량적 결과
RADAR는 테스트 세트에서 **전체 정확도 93.0%**를 달성하였다. 세부 항목별 결과는 다음과 같다.
- **Recall Task**: 97.7% 정확도 (Clear Recall의 경우 100% 탐지)
- **Reasoning Task**: 89.3% 정확도 (Complex Reasoning의 경우 100% 탐지)
- **Challenging Cases**: 76.7% 정확도 (가장 낮은 성능을 보였으나 여전히 유효한 수준)

### 3. 특성 분석 결과
- **Recall**: Specialized Heads의 수가 많고, 신뢰도 수렴 속도가 빠르며, Recall Detection Score(RDS)의 평균이 0.933으로 높게 나타났다.
- **Reasoning**: 회로 복잡도(Circuit Complexity)가 높고, 활성화 흐름의 분산이 크며, RDS 평균이 0.375로 낮게 나타났다.

## 🧠 Insights & Discussion

### 1. 강점 및 분석
본 연구는 모델의 내부 메커니즘을 분석함으로써, 기존의 텍스트 기반 탐지법이 가진 한계(학습 데이터 필요성, 변형된 텍스트 탐지 불가)를 완전히 해결하였다. 특히, 추론 과제임에도 불구하고 모델이 내부적으로는 Recall 신호(빠른 수렴, 집중된 Attention)를 보인다면, 이는 명백한 데이터 오염의 증거가 된다는 점을 입증하였다.

### 2. 한계 및 비판적 해석
- **대리 지표(Proxy Measures)의 사용**: 논문의 Appendix F에서 명시되었듯, 인과적 효과(Causal effects)나 활성화 패칭(Activation patching) 등을 실제 실험이 아닌 Attention 엔트로피 기반의 대리 지표로 계산하였다. 이는 실제 인과 관계를 완벽하게 반영하지 못했을 가능성이 있다.
- **데이터셋 규모**: 학습 데이터가 30개로 매우 적다. 앙상블 분류기가 적은 데이터로도 높은 성능을 냈다는 점은 고무적이나, 더 다양하고 방대한 데이터셋에서의 일반화 성능 검증이 필요하다.
- **모델 확장성**: DialoGPT-medium이라는 상대적으로 작은 모델에서 검증되었으므로, 수천억 개의 파라미터를 가진 거대 모델에서도 동일한 메커니즘 신호가 유지될지는 미지수이다.

## 📌 TL;DR

RADAR는 LLM의 내부 활성화 패턴(Attention, Hidden states)을 분석하여 모델이 응답을 생성할 때 **단순 암기(Recall)를 했는지 실제 추론(Reasoning)을 했는지**를 구분하는 프레임워크이다. 37개의 표면적/기계적 특성을 추출하여 앙상블 분류기에 적용한 결과, 학습 데이터 없이도 **93%의 높은 정확도**로 데이터 오염을 탐지할 수 있음을 보여주었다. 이는 향후 LLM 평가의 신뢰성을 높이고, 벤치마크 성능 부풀리기를 방지하는 중요한 도구가 될 가능성이 크다.