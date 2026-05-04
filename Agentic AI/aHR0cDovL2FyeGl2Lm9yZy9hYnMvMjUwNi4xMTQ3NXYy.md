# AutoGen-Driven Multi-Agent Framework for Iterative Crime Data Analysis and Prediction

Syeda Kisaa Fatima, Tehreem Zubair, Noman Ahmed, and Asifullah Khan (2025)

## 🧩 Problem to Solve

본 논문은 복잡하게 진화하는 범죄 데이터를 효과적으로 분석하고 미래의 범죄 경향을 예측하기 위한 자율적인 AI 시스템 구축을 목표로 한다. 기존의 범죄 분석 방식은 단순한 회귀 모델이나 정적인 시공간 분석에 의존하여 유연성이 떨어지며, 새로운 맥락이나 범죄 유형이 등장했을 때 빠르게 적응하지 못하는 한계가 있다. 또한, 범죄 데이터는 매우 민감한 정보를 포함하고 있어 데이터 프라이버시 보호가 필수적이며, 분석 과정에서 인간 전문가의 지속적인 감독 없이도 시스템 스스로 성능을 개선할 수 있는 자율적인 학습 메커니즘이 필요하다. 따라서 본 연구의 목표는 오프라인 환경에서 동작하며, 다중 에이전트 간의 협력적 대화를 통해 분석 능력을 반복적으로 향상시키는 LUCID-MA 프레임워크를 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 AutoGen 스타일의 Multi-Agent System(MAS)을 활용하여, 서로 다른 역할을 가진 AI 에이전트들이 피드백 루프를 통해 상호작용하며 '창발적 지능(Emergent Intelligence)'을 구현하는 것이다. 구체적으로는 분석, 피드백, 예측이라는 세 가지 전문화된 역할을 분담하고, 이들이 생성한 결과물을 서로 비판하고 보완하는 반복적인 대화 과정을 통해 모델의 가중치를 직접 수정하는 Fine-tuning 없이도 분석의 정교함과 예측의 정확도를 높이는 시뮬레이션 학습 구조를 제안한다. 또한, LLaMA-2-13B-Chat-GPTQ 모델을 사용하여 전체 시스템을 오프라인으로 구축함으로써 데이터 보안을 강화하였다.

## 📎 Related Works

기존의 범죄 연구 방법론은 초기에는 회귀 기반 모델이나 단순한 시공간 분석에 의존하였으며, 최근에는 Convolutional Neural Networks(CNN), Recurrent Neural Networks(RNN), Graph Neural Networks(GNN)와 같은 딥러닝 기법과 하이퍼파라미터 최적화를 위한 AutoML, 그리고 보고서 분석을 위한 Natural Language Processing(NLP) 기술이 도입되었다. 하지만 이러한 접근 방식들은 여전히 보편적으로 설명 가능한 변수를 찾는 데 어려움이 있으며, 특히 사회과학 분야에서 반복적인 피드백을 통해 스스로 학습하는 적응형 AI 시스템의 적용은 미비한 상태였다. 본 연구는 이러한 공백을 메우기 위해 AutoGen 프레임워크를 도입하여, 정적인 데이터셋 내에서도 에이전트 간의 협업을 통해 동적인 지식 확장이 가능함을 보여줌으로써 기존의 단일 모델 기반 예측 방식과 차별점을 둔다.

## 🛠️ Methodology

### 1. 데이터 전처리 파이프라인

시카고 범죄 데이터셋(Chicago Crime dataset)을 사용하여 다음과 같은 전처리를 수행하였다.

- **속성 제거**: 분석과 무관한 ID, Case Number, Location 등의 중복 속성을 제거하여 계산 효율을 높였다.
- **결측치 처리**: 범주형 변수는 "unknown regions"로 레이블링하고, 지리적 좌표는 평균값으로 대체하였다.
- **시간적 구조화**: 날짜 데이터를 연, 월, 일, 시, 요일 단위로 분해하여 다각도 분석이 가능하게 하였다.
- **공간 정규화 및 합성**: 위도와 경도를 Min-Max Scaling을 통해 $[0, 1]$ 범위로 정규화하였으며, DBSCAN 클러스터링을 통해 범죄 밀집 구역을 도출하였다. 또한, Nearest Neighbors 모델($k=10$)을 사용하여 인접 범죄 간의 관계를 측정하는 'relation' 특성을 생성하였다.

### 2. LUCID-MA 시스템 구조 및 에이전트 역할

시스템은 LLaMA-2-13B-Chat-GPTQ 모델을 기반으로 하며, 다음의 에이전트들이 협력한다.

- **CrimeAnalysisAssistant**: 데이터셋을 로드하여 통계적 요약, 핫스팟 식별 및 시각화 자료(Heatmaps, Bar charts)를 생성한다.
- **FeedbackAgent**: 분석 에이전트의 결과물을 검토하여 레이블 명명, 범례 개선, 누락된 분석 관점(예: 성별 기반 분포) 등을 제안하는 건설적 피드백을 제공한다.
- **CrimePredictorAgent**: 앞선 분석과 피드백을 바탕으로 미래의 범죄 핫스팟과 고위험 기간을 예측하고 예방 조치를 제안한다.
- **LearningOptimizerAgent (Ablation Study용)**: 에이전트 간의 협업 효율성을 모니터링하고 점수를 기반으로 시스템의 일관성과 다양성을 조정하는 감독 역할을 수행한다.

### 3. 스코어링 함수 및 학습 절차

에이전트의 성능 향상을 정량적으로 측정하기 위해 다음과 같은 스코어링 함수를 정의하였다.
$$ \text{Score} = \text{Base Score} + \text{Keyword Bonuses} + \text{Repetition Penalty} + \text{Exponential Learning Boost} $$

- **Base Score**: 분석 에이전트는 $0.02$, 그 외 에이전트는 $0.01$ 부여.
- **Keyword Bonuses**: "crime", "hotspot", "predict", "suggest" 등의 핵심 단어 사용 시 $+0.05$ 부여.
- **Repetition Penalty**: 이전 에포크와 응답이 중복될 경우 $-0.05$ 감점.
- **Exponential Learning Boost**: 에포크가 진행됨에 따라 학습 효과를 반영하기 위해 $0.5 \times (1 - e^{-0.05 \times \text{epoch}})$ 값을 더한다.

이 시스템은 총 100회의 커뮤니케이션 에포크(Epoch)를 거치며, 각 에포크마다 [분석 $\rightarrow$ 피드백 $\rightarrow$ 예측]의 사이클이 반복된다.

## 📊 Results

### 1. 실험 환경

- **Hardware**: NVIDIA DGX Station (4x Tesla V100 32GB), CUDA 12.0.
- **Software**: Python 3.8.19, PyTorch (Half-precision 계산 적용).
- **Execution**: 완전 오프라인 모드.

### 2. 에이전트 성능 진화

100 에포크 동안 에이전트들의 응답 품질이 현저히 향상되었다.

- **CrimeAnalysisAssistant**: 초기에는 단순 요약에 그쳤으나, 후기에는 성별/시간대별 정밀 분석 및 정규화된 히트맵을 생성하는 수준으로 발전하였다.
- **FeedbackAgent**: 단순한 미적 수정 제안에서 데이터 기반의 구조적 피드백(강점, 약점, 누락 요소 명시)을 제공하는 형태로 진화하였다.
- **CrimePredictorAgent**: 막연한 예측에서 "급여일 이후 중앙 구역의 차량 절도 증가 예상"과 같이 구체적인 시간/장소 기반의 예측 및 대응책 제안이 가능해졌다.

### 3. 정량적 결과

에이전트별 초기 점수와 최종 점수의 변화는 다음과 같다.

| Agent | Initial Score | Final Score | Highlighted Improvement |
| :--- | :---: | :---: | :--- |
| CrimeAnalysisAssistant | 0.07 | 0.94 | 분석 깊이 및 시각적 다양성 강화 |
| FeedbackAgent | 0.05 | 0.89 | 데이터 기반 피드백으로 전환 |
| CrimePredictorAgent | 0.04 | 0.85 | 패턴 기반 예측 및 개입 전략 포함 |

### 4. Ablation Study (OptimizerAgent의 영향)

`LearningOptimizerAgent`를 추가한 4-에이전트 프레임워크를 적용했을 때, 3-에이전트 베이스라인 대비 모든 에이전트의 최종 점수가 상승하였으며, 특히 에포크 간 응답 중복률(Redundancy)이 $14.2\%$에서 $6.8\%$로 크게 감소하여 시스템의 안정성과 다양성이 향상됨을 확인하였다.

## 🧠 Insights & Discussion

### 1. 강점 및 창발적 행동 (Emergent Behaviors)

본 연구는 모델의 파라미터를 수정하지 않고도 다중 에이전트의 상호작용만으로 다음과 같은 창발적 지능을 구현하였다.

- **전문화 및 협업**: 복잡한 작업을 분담하여 개별 에이전트가 수행할 수 없는 수준의 다차원적 통찰을 도출하는 '전문성 창발'이 관찰되었다.
- **자기 수정 루프**: `FeedbackAgent`가 동료 검토(Peer Review)와 유사한 역할을 수행하며 시스템의 견고함을 높였다.
- **집단 지성(Wisdom of the Crowd)**: 단일 LLM의 내재적 편향을 여러 에이전트의 서로 다른 관점이 상쇄함으로써 더 객관적인 분석 결과를 도출하였다.

### 2. 한계 및 비판적 해석

- **프롬프트 민감도**: 시스템이 프롬프트 엔지니어링에 크게 의존하고 있어, 지시문의 품질에 따라 결과의 편차가 발생할 수 있다.
- **검증 메커니즘의 부재**: `CrimePredictorAgent`가 생성한 예측 결과가 실제 정답(Ground Truth)과 얼마나 일치하는지 검증하는 피드백 루프가 부족하다. 즉, "그럴듯한" 예측을 내놓는 것과 "정확한" 예측을 내놓는 것 사이의 간극을 메울 정량적 검증 모듈이 필요하다.
- **중간 단계의 중복성**: 반복 페널티를 적용했음에도 불구하고 중간 단계에서 응답이 정체되는 현상이 발견되었으며, 이는 반복적 추론 기계에서 나타나는 전형적인 문제로 분석된다.

## 📌 TL;DR

본 논문은 AutoGen 기반의 다중 에이전트 프레임워크인 **LUCID-MA**를 제안하여, 오프라인 환경에서 범죄 데이터를 자율적으로 분석하고 예측하는 시스템을 구축하였다. 분석, 피드백, 예측 에이전트가 100회의 반복적인 대화를 통해 상호 보완하며 성능을 높이는 '시뮬레이션 학습'을 구현하였으며, 이를 통해 데이터 프라이버시를 유지하면서도 전문가 수준의 통찰력을 얻을 수 있음을 입증하였다. 이 연구는 사회과학 분야의 복잡한 문제 해결에 있어 LLM 기반의 분산형 에이전트 시스템이 효과적인 대안이 될 수 있음을 시사하며, 향후 Vision Transformer(ViT)를 결합한 멀티모달 분석으로 확장될 가능성이 높다.
