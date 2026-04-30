# Foundation Models Meet Visualizations: Challenges and Opportunities

Weikai Yang, Mengchen Liu, Zheng Wang, and Shixia Liu (2022/2024)

## 🧩 Problem to Solve

본 논문은 최근 인공지능 분야의 핵심으로 부상한 Foundation Models(기반 모델)와 데이터 시각화(Visualization) 기술의 교차점에서 발생하는 문제와 기회를 다룬다.

Foundation Models는 거대한 파라미터 규모와 복잡한 구조로 인해 내부 동작 원리를 이해하기 어렵고(Lack of transparency), 학습 데이터의 편향성이나 모델의 강건성(Robustness)을 평가하는 것이 매우 까다롭다. 기존의 시각화 도구로는 이러한 초거대 모델의 특성을 충분히 분석하기 어렵다는 문제가 있다.

반대로, 시각화 분야에서는 고품질의 시각적 표현을 생성하고 상호작용을 설계하는 데 많은 전문가의 지식과 수작업이 필요하다. 데이터에서 유의미한 특징을 추출하고 이를 최적의 시각적 형태로 매핑하는 과정의 자동화 및 지능화에 대한 수요가 높다.

따라서 본 논문의 목표는 시각화와 Foundation Models의 상호작용을 **VIS4FM**과 **FM4VIS**라는 두 가지 패러다임으로 정의하고, 각각의 현재 상태를 분석하며 향후 연구 방향을 제시하는 가이드라인을 제공하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Foundation Models와 시각화의 관계를 체계적인 프레임워크로 구조화한 것에 있다.

1.  **VIS4FM (Visualizations for Foundation Models)**: 시각화 기술을 활용하여 Foundation Models의 데이터 큐레이션, 학습 진단, 적응 제어, 모델 평가를 수행하는 방법론을 제시한다. 즉, 시각화를 통해 모델의 투명성과 설명 가능성을 높이는 방향이다.
2.  **FM4VIS (Foundation Models for Visualizations)**: Foundation Models의 강력한 표현 학습 및 생성 능력을 활용하여 특징 추출, 시각화 생성, 시각화 이해, 능동적 참여(Active Engagement)를 개선하는 방법론을 제시한다. 즉, 모델을 통해 시각화 파이프라인 전체를 지능화하는 방향이다.
3.  **연구 로드맵 제시**: 현재까지의 연구 사례를 분류한 테이블을 제공하고, 데이터 스케일링 문제, 실시간 진단, 다중 모달리티 정렬 등 향후 해결해야 할 구체적인 도전 과제와 기회를 정의한다.

## 📎 Related Works

논문에서는 기존의 시각적 분석(Visual Analytics) 및 머신러닝을 위한 시각화(VIS4ML), 시각화를 위한 머신러닝(ML4VIS) 연구들을 언급한다.

기존 접근 방식은 주로 전통적인 딥러닝 모델이나 소규모 머신러닝 모델을 대상으로 하였다. 하지만 Foundation Models는 다음과 같은 차별점을 가진다.
- **규모의 차이**: 수십억 개 이상의 파라미터를 가지므로 기존의 개별 파라미터 분석 방식으로는 한계가 있다.
- **학습 방식의 차이**: 자기지도학습(Self-supervision)을 통해 방대한 일반 지식을 습득하므로, 특정 태스크에 맞게 조정하는 Adaptation(적응) 과정이 매우 중요하다.
- **범용성**: 단일 목적 모델이 아니라 다양한 다운스트림 태스크에 적용 가능하므로, 평가 지표와 방법론이 훨씬 복잡하다.

## 🛠️ Methodology

본 논문은 특정 알고리즘을 제안하는 것이 아니라, 두 가지 핵심 파이프라인에 대한 분석 체계를 제안한다.

### 1. VIS4FM 파이프라인
Foundation Models의 학습 생애주기에 따라 시각화가 개입하는 지점을 다음과 같이 정의한다.

- **Data Curation (데이터 큐레이션)**: 데이터 생성, 통합, 선택, 수정 단계에서 시각화를 통해 데이터의 커버리지를 확인하고 노이즈를 제거한다.
- **Training Diagnosis (학습 진단)**: 모델 설명(Model Explanation), 성능 진단(Performance Diagnosis), 효율성 진단(Efficiency Diagnosis)을 통해 학습 과정의 병목 현상과 내부 작동 기제를 분석한다.
- **Adaptation Steering (적응 제어)**: 파인튜닝(Fine-tuning), 프롬프트 엔지니어링(Prompt Engineering), 인간 피드백 기반 정렬(Alignment via Human Feedback) 과정에서 시각적 피드백을 통해 모델을 원하는 방향으로 유도한다.
- **Model Evaluation (모델 평가)**: 정량적 지표의 시각적 제시와 정성적 분석(인간의 판단 개입)을 통해 모델의 성능과 행동을 평가한다.

### 2. FM4VIS 파이프라인
전형적인 시각화 파이프라인의 각 단계에 Foundation Models를 통합하는 구조이다.

- **Data Transformation $\rightarrow$ Feature Extraction & Pattern Recognition**: 비정형 데이터(텍스트, 이미지)에서 의미론적 특징 벡터를 추출하여 시각화의 입력값으로 사용한다.
- **Visual Mapping $\rightarrow$ Visualization Generation**: 자연어 프롬프트를 통해 시각화 콘텐츠(차트 종류, 인코딩) 및 스타일(색상, 레이아웃)을 자동으로 생성한다.
- **View Transformation $\rightarrow$ Visualization Understanding**: 시각화 결과물에서 데이터를 역으로 추출하거나, 시각화 내용을 자연어로 요약하여 사용자에게 전달한다.
- **Visual Perception $\rightarrow$ Active Engagement**: 사용자의 의도를 예측하여 상호작용을 최적화하거나, 자연어 기반의 직접적인 인터랙션을 지원한다.

## 📊 Results

본 논문은 실험 논문이 아니므로 정량적 수치 대신, 기존 문헌 분석을 통한 정성적 결과(Taxonomy)를 제시한다.

- **VIS4FM 분석 결과**: 프롬프트 엔지니어링과 데이터 수정 분야에서는 많은 연구가 진행되었으나, 데이터 통합(Integration) 및 데이터 선택(Selection) 단계에서의 시각화 연구는 상대적으로 부족함을 확인하였다.
- **FM4VIS 분석 결과**: 특징 추출(Feature Extraction) 및 패턴 인식 분야는 활발히 연구되고 있으나, 상호작용 생성(Interaction Generation) 및 예측적 상호작용 강화(Predictive Interaction Enhancement) 분야는 거의 연구되지 않은 미개척 영역임을 밝혀냈다.
- **주요 사례**:
    - `LinguisticLens`: LLM이 생성한 데이터셋의 반복성을 시각적으로 식별.
    - `AttentionViz`: 트랜스포머 모델의 셀프 어텐션 패턴을 다수 샘플에 대해 동시에 분석.
    - `ADVISor`: 표 데이터와 자연어 질문을 바탕으로 최적의 시각화 및 주석을 자동 생성.

## 🧠 Insights & Discussion

### 강점 및 기회
- **상호 보완적 관계**: Foundation Models는 시각화의 진입 장벽을 낮추고(자동 생성), 시각화는 Foundation Models의 '블랙박스' 문제를 해결하는(투명성 확보) 상호 보완적 관계임을 논리적으로 입증하였다.
- **실시간 진단의 필요성**: 기존의 오프라인 분석에서 벗어나, 학습 과정 중에 실시간으로 성능과 효율성을 모니터링하는 In-situ Visualization의 필요성을 강조한 점이 통찰력 있다.

### 한계 및 도전 과제
- **스케일링 문제**: 데이터와 모델의 규모가 너무 커서 메모리에 모두 올릴 수 없으므로, Out-of-memory 샘플링 기술과 실시간 인터랙션의 조화가 필수적이다.
- **데이터 무결성(Data Integrity)**: FM4VIS에서 스타일 전이(Style Transfer) 등을 적용할 때, 시각적 미감뿐만 아니라 원본 데이터의 수치적 정확성이 훼손되지 않도록 보장하는 메커니즘이 필요하다.
- **윤리 및 공정성**: 모델의 설명 가능성(Explainability)을 시각화할 때, 문화적 차이나 윤리적 편향성이 개입될 수 있음을 지적하며 이에 대한 책임 있는 배포를 강조한다.

## 📌 TL;DR

본 논문은 초거대 기반 모델(Foundation Models)과 데이터 시각화의 교차점을 **VIS4FM**(모델 분석을 위한 시각화)과 **FM4VIS**(시각화 지능화를 위한 모델)로 정의하고 체계적인 분석 프레임워크를 제안한다. 특히 모델의 블랙박스 문제를 해결하기 위한 시각적 진단 도구의 필요성과, 자연어 기반의 자동 시각화 생성 및 지능형 인터랙션으로의 확장 가능성을 제시함으로써 향후 AI-시각화 융합 연구의 핵심 로드맵을 제공한다.