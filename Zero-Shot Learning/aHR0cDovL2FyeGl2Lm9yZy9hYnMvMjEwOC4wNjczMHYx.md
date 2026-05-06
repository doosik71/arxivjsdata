# Towards Visual Explainable Active Learning for Zero-Shot Classification

Shichao Jia, Zeyu Li, Nuo Chen, and Jiawan Zhang (2021)

## 🧩 Problem to Solve

본 논문은 Zero-Shot Classification (ZSC) 모델 구축의 핵심 단계인 **Class-Attribute Matrix (클래스-속성 행렬)** 설계 과정에서 발생하는 비효율성과 어려움을 해결하고자 한다.

ZSC는 학습 단계에서 본 적 없는 클래스(unseen classes)를 인식하기 위해, 전문가가 각 클래스가 어떤 속성(attribute)을 가지고 있는지 정의한 클래스-속성 행렬을 필요로 한다. 그러나 이 행렬을 설계하는 과정은 다음과 같은 문제점을 가진다.

- **수동 설계의 고충:** 도메인 전문가가 직접 속성을 선택하고 라벨링하는 과정은 매우 지루하며, 명확한 가이드라인 없이 시행착오(trial-and-error) 방식으로 진행된다.
- **자동화의 한계:** 속성을 자동으로 추출하는 방식은 결과물의 해석 가능성(interpretability)이 떨어져, 인간이 왜 그런 결과가 나왔는지 이해하기 어렵다.
- **속성 선택의 어려움:** 모든 속성이 유용한 것은 아니며, 일부 속성은 오히려 오분류를 유발하여 성능을 저하시킬 수 있다.

결과적으로 본 논문의 목표는 인간과 AI의 협업을 통해 해석 가능하면서도 효율적으로 클래스-속성 행렬을 구축할 수 있는 **Visual Explainable Active Learning** 접근법과 이를 구현한 시스템인 **Semantic Navigator**를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 인간과 AI가 **Ask $\rightarrow$ Explain $\rightarrow$ Recommend $\rightarrow$ Respond**라는 상호작용 루프를 통해 공동으로 모델을 개선하는 **Human-AI Teaming** 구조를 설계한 것이다.

특히, 딥러닝 모델의 저수준 특징 공간(Feature Space)과 인간의 고수준 의미 공간(Semantic Space) 사이의 **Semantic Gap**을 메우기 위해, 두 공간을 하나의 공유 공간인 **Mutual Mental Space**로 매핑하여 시각화함으로써 인간이 모델의 상태를 직관적으로 이해하고 제어할 수 있도록 한 점이 주요 기여이다.

## 📎 Related Works

논문은 다음과 같은 세 가지 관련 연구 분야를 다룬다.

1. **Visual Attributes 설계:** 기존 연구들은 수동으로 속성을 정의하거나 인터넷 데이터를 통해 자동으로 태깅하는 방식을 사용했다. 하지만 자동 방식은 분류 작업에 불필요한 의미가 포함될 수 있고, 시각적 특징 공간에서 분리 가능성이 떨어지는 한계가 있다.
2. **Interactive Classification:** Active Learning은 기계 중심적으로 라벨을 요청하는 방식이며, Visual-Interactive Labeling은 인간 중심적인 패턴 인식에 의존한다. 본 논문은 이 두 가지를 결합한 Mixed-initiative 방식을 통해 제로샷 분류의 속성 설계를 지원한다.
3. **Guidance in Visual Analytics:** 데이터 분석가의 지식 격차를 줄이기 위한 가이드 시스템들이 존재하지만, 제로샷 학습을 위한 클래스-속성 행렬 구축을 가이드하는 연구는 이전까지 탐색되지 않았다.

## 🛠️ Methodology

### 1. 전체 시스템 구조 및 Mutual Mental Space

시스템은 딥러닝 모델에서 추출된 특징 공간과 사용자가 정의한 속성 공간을 연결하는 **Mutual Mental Space**를 기반으로 한다.

- **특징 투영:** 사전 학습된 DNN에서 추출된 고차원 특징($D$-dimensional)을 PCA를 통해 $d$-차원($d < D$)의 Mutual Mental Space로 투영한다.
- **Visual Exemplars:** 투영된 공간에서 각 클래스의 데이터들이 형성하는 클러스터의 중심점을 **Visual Exemplar**($v_i$)라고 정의하며, 이는 해당 클래스의 이상적인 시각적 대표값 역할을 한다.

### 2. Zero-Shot Classifier (EXEM 모델)

본 시스템은 계산 효율성을 위해 EXEM 모델을 채택하였다.

- **학습:** 의미 공간의 클래스 프로토타입($p_i$)에서 Mutual Mental Space의 시각적 엑셈플러($v_i$)로 가는 변환 함수 $\psi(\cdot)$를 Support Vector Regressors (SVR)를 통해 학습한다.
- **추론:** 새로운 데이터 $x^u$가 들어오면, 이를 PCA 투영 행렬 $M_{PCA}$를 통해 투영한 후, 예측된 엑셈플러 $\psi(p^u_i)$와의 거리가 가장 가까운 클래스로 분류한다.
$$ y = \arg \min_i \text{dist}(M_{PCA} x^u, \psi(p^u_i)) $$

### 3. Interaction Loop: 네 가지 핵심 액션

**Semantic Navigator**는 다음의 루프를 통해 클래스-속성 행렬을 정교화한다.

- **Ask (질문):** 모델의 혼동 행렬(Confusion Matrix)을 분석하여 가장 많이 오분류되는 클래스 쌍 $(A, B)$를 찾는다. 이후 "기존 속성 외에 $A$와 $B$를 구분하는 가장 눈에 띄는 속성은 무엇인가?"라는 **Contrastive Question**을 던져 사용자가 새로운 속성을 고안하도록 유도한다.
- **Explain (설명):** **Semantic Map**이라는 시각화 도구를 통해 모델의 상태를 설명한다. t-SNE를 사용하여 Mutual Mental Space를 2D 평면에 투영하고, 클래스 클러스터를 외곽선(contour)으로, 프로토타입을 점으로 표시한다.
  - 프로토타입이 엑셈플러(클러스터 중심)와 얼마나 일치하는지를 통해 속성 설계의 적절성을 판단한다.
  - 논문은 프로토타입과 컨투어의 상대적 위치에 따라 5가지 시각적 패턴(Pattern 1~5)을 제시하여, 분석가가 속성을 추가해야 할지 혹은 잘못 정의된 속성을 수정해야 할지 진단할 수 있게 한다.
- **Recommend (추천):** 새로운 속성이 도입되었을 때, 모든 클래스를 일일이 라벨링하는 부담을 줄이기 위해 **Semi-supervised SVM ($S^3VM$)**을 사용하여 나머지 클래스의 속성 값(Positive/Negative)을 추천한다. 이때 연산 효율을 위해 전체 데이터가 아닌 클래스 엑셈플러들만을 사용하여 $QN-S^3VM$ 알고리즘으로 최적의 초평면(hyperplane)을 찾는다.
- **Respond (응답):** 사용자는 Lasso 도구를 통해 추천된 라벨을 수정하고, 최종적으로 속성 이름을 지정하여 제출한다. 이후 모델은 업데이트된 행렬로 재학습되며 시각화 결과가 갱신된다.

## 📊 Results

### 1. 실험 설정

- **데이터셋:** AWA2 (포유류 50종) 및 SUB-CUB200 (조류 데이터셋의 부분 집합)을 사용하였다.
- **비교 대상:**
  - **With Guidance:** Semantic Navigator를 사용한 그룹.
  - **Without Guidance:** 가이드 없이 행렬 뷰와 정확도 차트만 제공된 그룹.
  - **Random Baseline:** 벤치마크 행렬에서 무작위로 15개 속성을 선택한 경우.
- **측정 지표:** 테스트 정확도(Testing Accuracy) 및 속성 하나를 정의하는 데 걸리는 평균 시간.

### 2. 주요 결과

- **정량적 결과:**
  - **정확도:** Semantic Navigator를 사용했을 때, AWA2에서는 평균 4.96%, SUB-CUB200에서는 4.13%의 테스트 정확도 향상이 있었다. 특히 속성 수가 5개를 넘어갈 때 가이드의 효과가 뚜렷하게 나타났다.
  - **효율성:** 속성 하나를 지정하는 데 걸리는 시간이 가이드가 없을 때보다 AWA2와 SUB-CUB200 모두에서 약 14초 정도 단축되었다.
- **정성적 결과:**
  - 가이드가 없는 사용자는 속성을 추가해도 정확도가 변하지 않거나 오히려 떨어지는 상황에서 좌절감을 느꼈으며, 뇌가 정지되는 느낌(brain is stuck)을 받았다고 보고하였다.
  - 가이드 사용자는 "질문이 많은 도움이 되었고", "노란 점(프로토타입)을 검은 점(엑셈플러)에 도달하게 만드는 과정이 직관적이었다"고 평가하였다.

## 🧠 Insights & Discussion

본 논문의 강점은 단순히 모델 성능을 높이는 것이 아니라, **인간의 인지 과정(비교를 통한 속성 발견)과 기계의 상태(특징 공간 내의 거리)를 시각적으로 일치시킨 점**에 있다. Mutual Mental Space라는 공유 공간을 구축함으로써 분석가는 모델의 오분류 원인을 '사례 기반 추론(case-based reasoning)'을 통해 진단할 수 있게 되었다.

**한계 및 논의사항:**

- **이진 속성의 한계:** 본 연구는 속성을 0 또는 1의 이진 값으로만 처리하였다. 하지만 실제 인간의 인식은 "A가 B보다 더 털이 많다"와 같은 상대적 속성(relative attributes)을 사용하는 경우가 많으므로, 이를 확장한다면 더 자연스러운 상호작용이 가능할 것이다.
- **개념 학습의 검증:** 모델의 정확도가 올라갔다고 해서 모델이 실제로 인간이 의도한 '속성'을 정확히 학습했는지는 알 수 없다. 단순히 다른 상관관계가 있는 특징을 학습했을 가능성이 있으며, 이를 검증하기 위해 Concept-level explanation 연구와의 결합이 필요하다.

## 📌 TL;DR

본 논문은 Zero-Shot Classification의 핵심인 클래스-속성 행렬을 효율적으로 구축하기 위해, 인간과 AI가 협업하는 **Visual Explainable Active Learning** 프레임워크와 **Semantic Navigator** 시스템을 제안하였다. 이 시스템은 **질문-설명-추천-응답**의 루프를 통해 사용자가 직관적으로 속성을 설계하도록 돕는다. 실험 결과, 가이드가 없는 방식보다 모델의 테스트 정확도가 향상되었으며, 전문가의 속성 정의 시간과 인지적 부담을 유의미하게 줄였음을 입증하였다. 이 연구는 향후 의료 진단 등 도메인 지식 주입이 중요한 다른 AI 분야의 Human-AI Teaming 연구에 중요한 기초가 될 수 있다.
