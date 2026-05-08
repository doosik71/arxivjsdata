# Trustworthy AI: A Computational Perspective

Haochen Liu, Yiqi Wang, Wenqi Fan, Xiaorui Liu, Yaxin Li, Shaili Jain, Yunhao Liu, Anil K. Jain, Jiliang Tang (2021)

## 🧩 Problem to Solve

인공지능(AI) 기술은 비약적인 발전을 이루었으나, 안전 필수적인 시나리오에서의 신뢰할 수 없는 의사결정, 특정 집단에 대한 차별 및 불공정성, 사용자 개인정보 유출 등 의도치 않은 위해를 가할 수 있는 취약성을 드러내고 있다. 이러한 신뢰성 문제는 AI 기술이 더 널리 채택되고 경제적 가치를 창출하는 데 있어 거대한 장애물이 되고 있다.

본 논문의 목표는 Trustworthy AI(신뢰할 수 있는 AI)를 달성하기 위한 최신 기술들을 **Computational Perspective(계산적 관점)**에서 종합적으로 분석하고 평가하는 것이다. 기존의 연구들이 주로 법적 규제나 고수준의 윤리적 가이드라인에 집중했던 것과 달리, 본 논문은 실제 구현 가능한 기술적 솔루션과 알고리즘을 중심으로 신뢰성을 확보하는 방법을 제시하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Trustworthy AI를 구성하는 가장 중요한 6가지 차원을 정의하고, 각 차원을 달성하기 위한 계산적 방법론을 체계적인 분류 체계(Taxonomy)에 따라 정리한 것이다.

1. **6대 핵심 차원 정의**: Safety & Robustness, Non-discrimination & Fairness, Explainability, Privacy, Accountability & Auditability, Environmental Well-being을 Trustworthy AI의 필수 요소로 설정하였다.
2. **계산적 솔루션 제공**: 각 차원별로 구체적인 알고리즘, 훈련 목표, 손실 함수 및 추론 절차를 검토하여 기술적인 구현 경로를 제시하였다.
3. **차원 간 상호작용 분석**: 각 차원이 서로를 촉진하는 '일치(Accordance)' 관계와 서로 충돌하는 '갈등(Conflict)' 관계에 있음을 분석하여, 실제 시스템 설계 시 고려해야 할 Trade-off 관계를 명시하였다.

## 📎 Related Works

논문은 유럽 연합(EU)의 AI 윤리 가이드라인과 같은 정부 지침이나, 기술적 세부 사항이 배제된 고수준의 리뷰 논문들을 언급한다. 이러한 기존 접근 방식은 "무엇을 해야 하는가"에 대한 원칙은 제시하지만, "어떻게 기술적으로 구현하는가"에 대한 답은 부족하다는 한계가 있다.

본 연구는 이러한 간극을 메우기 위해 단순한 윤리적 논의를 넘어, 실제 딥러닝 모델의 아키텍처, 최적화 과정, 데이터 처리 기법 등 계산적 방법론을 통해 신뢰성을 확보하는 구체적인 방안을 다룬다는 점에서 기존 연구와 차별화된다.

## 🛠️ Methodology

본 논문은 Trustworthy AI를 달성하기 위한 6가지 차원의 방법론을 다음과 같이 설명한다.

### 1. Safety & Robustness (안전성 및 강건성)

AI 시스템이 입력값의 작은 섭동(Perturbation)에도 안정적인 성능을 유지하는 능력을 의미한다.

- **Threat Models**: 공격 시점에 따라 Poisoning Attack(훈련 데이터 오염)과 Evasion Attack(테스트 시 입력 변조)으로 나뉘며, 공격자의 지식 수준에 따라 White-box와 Black-box 공격으로 구분된다.
- **핵심 알고리즘**:
  - **PGD (Projected Gradient Descent)**: 손실 함수를 최대화하는 섭동을 찾아 모델의 취약점을 찾는 대표적인 공격 방법이다.
    $$\text{maximize } L(\theta, x') \quad \text{subject to } ||x' - x||_p \le \epsilon \text{ and } x' \in [0, 1]^m$$
  - **Adversarial Training**: 훈련 과정에서 생성된 Adversarial Examples를 함께 학습시켜 모델의 강건성을 높이는 방법으로, 다음과 같은 Min-Max 최적화 문제로 정식화된다.
    $$\min_{\theta} \mathbb{E}_{(x,y) \sim D} [\max_{||x'-x|| \le \epsilon} L(\theta, x', y)]$$

### 2. Non-discrimination & Fairness (비차별성 및 공정성)

특정 집단이나 개인에 대해 편향된 결정을 내리지 않는 것을 목표로 한다.

- **공정성 정의**:
  - **Group Fairness**: 서로 다른 집단이 통계적으로 유사한 결과를 얻어야 함 (예: Demographic Parity).
  - **Individual Fairness**: 유사한 개인은 유사하게 처리되어야 함.
    $$|f_M(i) - f_M(j)| < \epsilon$$
- **편향 완화 방법론**:
  - **Pre-processing**: 학습 데이터에서 샘플링이나 재가중치(Reweighting)를 통해 편향을 제거한다.
  - **In-processing**: 손실 함수에 공정성 제약 조건(Regularization)을 추가하거나 Adversarial Learning을 통해 편향된 특징을 학습하지 못하게 한다.
  - **Post-processing**: 모델의 출력 결과에 대해 임계값(Threshold)을 조정하여 최종 결과의 공정성을 맞춘다.

### 3. Explainability (설명 가능성)

AI의 결정 메커니즘을 인간이 이해할 수 있도록 제시하는 능력이다.

- **분류**: 모델 자체가 투명한 **Interpretable AI**(예: Decision Tree)와, 블랙박스 모델에 사후 설명을 추가하는 **Explainable AI**(예: LIME, SHAP)로 구분된다.
- **주요 기법**:
  - **Gradient-based**: Grad-CAM과 같이 입력값에 대한 출력의 기울기를 계산하여 중요한 영역을 시각화한다.
  - **Perturbation-based**: 입력을 조금씩 변형시키며 출력의 변화를 관찰하여 특징의 중요도를 산출한다.
  - **Counterfactual Explanations**: "만약 $\text{X}$였다면 결과가 $\text{Y}$가 되었을 것"이라는 가상 상황을 제시한다.
    $$\arg \min_{\hat{x}} \max_\lambda \lambda \cdot (f(\hat{x}) - \hat{y})^2 + d(\hat{x}, x)$$

### 4. Privacy (프라이버시)

학습 데이터나 모델로부터 민감한 정보가 유출되지 않도록 보호하는 것이다.

- **공격 모델**: Membership Inference Attack(데이터 포함 여부 확인), Model Inversion Attack(입력 데이터 복구) 등이 있다.
- **보호 기법**:
  - **Federated Learning**: 데이터를 중앙 서버로 전송하지 않고 로컬에서 학습한 모델 업데이트 값만 공유한다.
  - **Differential Privacy (DP)**: 데이터에 무작위 노이즈를 추가하여 개별 데이터의 영향을 마스킹한다.
    $$\text{Pr}[A(D) \in S] \le e^\epsilon \text{Pr}[A(D') \in S] + \delta$$

### 5. Accountability & Auditability (책임성 및 감사 가능성)

AI 실패 시 책임 소재를 명확히 하고, 제3자가 시스템을 객관적으로 평가할 수 있는 체계이다.

- **역할 정의**: 시스템 설계자, 의사 결정자, 배포자, 감사자, 최종 사용자로 역할을 나누어 책임을 배분한다.
- **감사 방법**: 내부 감사(Internal Audit)와 독립적인 제3자가 수행하는 외부 감사(External Audit)로 구분한다.

### 6. Environmental Well-being (환경적 안녕)

AI 모델의 거대한 파라미터 수로 인한 막대한 에너지 소비와 탄소 배출 문제를 해결하는 것이다.

- **에너지 절감 기법**:
  - **Model Compression**: Pruning(가지치기), Quantization(양자화), Knowledge Distillation(지식 증류)을 통해 모델 크기를 줄인다.
  - **Hardware Acceleration**: NPU와 같은 AI 전용 가속기를 사용하여 연산 효율을 높인다.

## 📊 Results

본 논문은 서베이 논문이므로 개별 실험 결과보다는 각 방법론이 적용된 실제 시스템의 사례와 정성적 분석 결과를 제시한다.

- **강건성 사례**: 자율주행 자동차의 도로 표지판 인식 시스템에서 작은 스티커 부착만으로 정지 표지판을 인식하지 못하게 만드는 Adversarial Attack의 위험성을 경고하였다.
- **공정성 사례**: 안면 인식 시스템이 피부색에 따라 정확도 차이가 발생하거나, 채용 시스템이 성별에 따라 STEM 직군 추천 빈도가 다른 편향 사례를 분석하였다.
- **설명 가능성 사례**: 의료 진단 시스템에서 AI가 왜 특정 질병으로 진단했는지에 대한 근거를 제시함으로써 의사와 환자의 신뢰를 얻을 수 있음을 보여주었다.
- **프라이버시 사례**: 의료 데이터 분석 시 Federated Learning을 도입하여 환자의 민감 정보를 외부로 유출하지 않고도 협력 학습이 가능함을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 논문은 Trustworthy AI라는 추상적인 개념을 6가지 구체적인 계산적 차원으로 정량화하여, 엔지니어가 실제로 구현해야 할 체크리스트를 제공했다는 점에서 매우 강력하다. 특히, 단순히 개별 기술을 나열하는 것에 그치지 않고 차원 간의 상호작용을 분석한 점이 돋보인다.

### 차원 간의 상호작용 (Critical Analysis)

- **일치 관계 (Accordance)**:
  - $\text{Robustness} \leftrightarrow \text{Explainability}$: 강건하게 학습된 모델은 기울기(Gradient)가 더 깨끗하여 설명 가능성이 높아지는 경향이 있다.
  - $\text{Fairness} \leftrightarrow \text{Environmental Well-being}$: 모델의 에너지 효율을 높여 학습 비용을 낮추는 것은, 고가의 컴퓨팅 자원을 가진 소수 연구자만이 AI를 개발하는 불평등을 해소하여 공정성에 기여한다.
- **갈등 관계 (Conflict)**:
  - $\text{Robustness} \leftrightarrow \text{Privacy}$: Adversarial Training을 통해 강건성을 높이면 모델이 훈련 데이터에 과적합(Overfitting)되는 경향이 있어, Membership Inference Attack에 더 취약해질 수 있다.
  - $\text{Robustness} \leftrightarrow \text{Fairness}$: 강건성을 높이기 위한 훈련이 특정 집단의 성능을 희생시키며 그룹 간 성능 격차를 벌릴 수 있다.

### 한계 및 미해결 과제

- **평가 지표의 부재**: 설명 가능성(XAI)의 경우, 무엇이 '좋은 설명'인지에 대한 객관적인 정량적 지표가 부족하여 여전히 인간의 주관적 평가에 의존하고 있다.
- **실시간성 문제**: 인증된 방어(Certified Defense)나 복잡한 암호화 기법(HE, MPC)은 계산 비용이 너무 커서 실제 실시간 시스템에 적용하기에는 한계가 있다.

## 📌 TL;DR

본 논문은 Trustworthy AI를 달성하기 위한 **Safety, Fairness, Explainability, Privacy, Accountability, Environmental Well-being**이라는 6가지 핵심 차원을 정의하고, 이를 구현하기 위한 계산적 방법론을 집대성한 종합 서베이 보고서이다. 특히 각 차원이 서로 어떻게 충돌하고 보완하는지에 대한 분석을 통해, 단순한 성능 최적화를 넘어 **'신뢰할 수 있는 시스템'**을 설계하기 위한 기술적 프레임워크를 제공한다. 향후 AI가 안전 필수 분야(의료, 자율주행 등)에 완전히 통합되기 위해 반드시 해결해야 할 기술적 로드맵을 제시하고 있어, 향후 연구 및 실제 시스템 구축에 매우 중요한 지침서가 될 것으로 보인다.
