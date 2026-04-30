# AAPMT: AGI Assessment Through Prompt and Metric Transformer

Benhao Huang (2024)

## 🧩 Problem to Solve

본 논문은 텍스트-이미지 생성 모델(Text-to-Image models)의 발전으로 인해 생성된 AI 이미지(AI-generated images, AGIs)의 품질 평가 필요성이 증대됨에 따라, 인간의 지각(Human perception)과 밀접하게 일치하는 정밀한 평가 지표를 개발하는 것을 목표로 한다.

현재 BLIP나 DBCNN과 같은 기술적 진보가 있었으나, AGIQA-3K와 같은 최근 연구 결과에 따르면 기존의 평가 방법들과 최신 표준(SOTA) 사이에는 여전히 상당한 간극이 존재한다. 특히 AI 생성 이미지의 지각적 품질(Perceptual quality), 진위성(Authenticity), 그리고 텍스트-이미지 일치도(Text-image correspondence/alignment)라는 세 가지 핵심 요소를 정확하게 측정할 수 있는 정교한 메트릭의 부재가 문제로 지적된다.

## ✨ Key Contributions

본 논문의 핵심 기여는 단일 모델을 통해 다중 평가 지표를 효율적으로 산출하기 위한 두 가지 접근 방식을 제안한 것이다.

첫째, 특정 묘사 문구(Prompt)를 설계하여 이미지 품질과 진위성을 평가하는 프롬프트 기반 설계(Prompt design) 방법을 제안한다. 이는 고품질 이미지의 특징을 텍스트로 정의하고, 모델이 이미지와 해당 텍스트 간의 일치도를 측정하게 함으로써 품질 점수를 도출하는 직관적인 아이디어이다.

둘째, 여러 품질 지표 간의 복잡한 상호 관계를 학습하기 위한 **Metric Transformer** 구조를 제안한다. 서로 다른 평가 지표들이 세만틱(Semantic) 측면에서 유사성을 공유한다는 가설을 바탕으로, Self-attention 메커니즘을 도입하여 단일 모델이 여러 지표를 동시에 평가할 수 있도록 설계하여 연산 효율성과 정확도를 동시에 높였다.

## 📎 Related Works

논문은 Text-Image Matching 작업에서 우수한 성능을 보이는 **ImageReward**와 **BLIP**을 기반으로 연구를 수행하였다. 또한, 평가를 위한 데이터셋으로 **AGIQA-3K**와 **AIGCIQA2023**을 사용하였다.

기존 방식들은 각 평가 지표(품질, 진위성, 일치도)를 위해 개별적인 모델을 학습시켜야 했으므로 매우 비효율적이고 메모리 소모가 컸다. 본 논문은 이러한 개별 모델 방식의 한계를 극복하고, 지표 간의 파라미터 공간(Parameter space)이 일부 겹친다는 점에 착안하여 이를 통합하는 구조로 차별화를 두었다.

## 🛠️ Methodology

### 1. 전체 파이프라인 및 기본 구조
본 모델은 **ImageReward**의 사전 학습된 모델을 백본(Backbone)으로 사용하며, AGIQA-3K 및 AIGCIQA2023 데이터셋을 사용하여 미세 조정(Fine-tuning)을 진행하였다. 데이터셋은 8:2 비율로 학습 및 테스트 세트로 분할하였으며, 동일한 프롬프트에서 생성된 이미지를 함께 그룹화하는 '콘텐츠 격리(Content Isolation)' 원칙을 적용하여 테스트 신뢰도를 높였다.

### 2. 지표별 평가 방법론
- **텍스트-이미지 일치도(Correspondence):** 이미지와 텍스트를 입력받아 일치 정도를 점수로 출력하는 ImageReward의 기본 구조를 그대로 활용하여 학습하였다.
- **이미지 품질(Quality):** "extremely high quality image, with vivid details"와 같은 고품질 묘사 프롬프트를 설계하였다. 모델이 이미지와 이 프롬프트 사이의 일치도를 측정하게 함으로써, 일치도가 높을수록 이미지 품질이 높다고 판단하는 방식을 취했다.
- **이미지 진위성(Authenticity):** "very authentic image"라는 프롬프트를 사용하여 동일한 방식으로 진위성을 측정하였다.

### 3. Metric Transformer
단일 모델로 다중 지표를 평가하기 위해 기존 ImageReward의 MLP(Multi-Layer Perceptron) 층을 **Metric Transformer**로 대체하였다. 

**학습 절차 및 구조:**
1. 텍스트 특징(Text features)을 3-head Transformer Encoder에 통과시켜 세 가지 지표(품질, 진위성, 일치도)에 대한 기본 개념을 학습한다.
2. 인코딩된 특징 $\text{EF}$를 Metric Transformer에 입력하여 각 지표의 점수를 계산한다.
3. 각 지표 $i$에 대해 Query($Q$), Key($K$), Value($V$) 행렬을 다음과 같이 생성한다.
   $$Q_i = W_{Q_i}^T \cdot \text{EF}$$
   $$K_i = W_{K_i}^T \cdot \text{EF}$$
   $$V_i = W_{V_i}^T \cdot \text{EF}$$
4. 최종 메트릭 점수 $S_i$는 다음과 같은 어텐션 메커니즘을 통해 산출된다.
   $$S_i = \sum_{j=1}^{3} \text{SoftMax}\left(\frac{Q_i K_j^T}{\sqrt{d_k}} V_j\right)$$
   여기서 $d_k$는 $K_i$의 차원이다.

## 📊 Results

### 1. 실험 설정
- **데이터셋:** AGIQA-3K, AIGCIQA2023
- **평가 지표:** PLCC(Pearson Linear Correlation Coefficient), SRCC(Spearman Rank Correlation Coefficient)
- **비교 대상:** 원본 ImageReward(학습 전), 각 지표별로 개별 학습된 ImageReward 모델, Metric Transformer

### 2. 주요 결과
- **프롬프트 설계의 효과:** 단순한 프롬프트 변경만으로도 이미지 품질 및 진위성 평가에서 매우 높은 PLCC/SRCC 값을 얻었으며, 특히 "vivid details"라는 표현이 "high resolution"보다 모델의 품질 판단에 더 큰 영향을 미침을 확인하였다.
- **Metric Transformer의 효율성:** Metric Transformer는 단일 모델임에도 불구하고, 세 개의 개별 모델을 사용한 ImageReward의 결과와 대등하거나 오히려 일부 지표(진위성 등)에서는 더 높은 성능을 보였다. 
  - 특히 AIGCIQA2023-Authenticity 작업에서 Metric Transformer는 PLCC 0.9112, SRCC 0.9075를 기록하여 매우 강력한 성능을 입증하였다.

## 🧠 Insights & Discussion

### 1. 파라미터 공간의 공유 (Parameter Space Overlap)
저자는 한 작업(예: 품질 평가)으로 학습된 모델을 다른 작업(예: 일치도 평가)으로 재학습시켰을 때, 완전히 처음부터 학습한 모델보다 더 좋은 성능을 보이는 현상을 발견하였다. 이를 벤 다이어그램으로 설명하며, 서로 다른 평가 지표들이 최적 파라미터 공간을 일부 공유($A \cap B$)하고 있음을 시사한다. 이는 이미지 품질이 높으면 텍스트-이미지 일치도 또한 높을 가능성이 크다는 세만틱 유사성에 기인한다.

### 2. 한계점 및 향후 과제
- **교차 검증 부재:** 시간 제약으로 인해 랜덤 시드를 고정한 단순 테스트만 수행하였으며, 교차 검증(Cross-validation)을 수행하지 못한 점이 한계로 언급되었다.
- **동적 손실 함수:** 부록에서 각 지표의 중요도 $\alpha_i$를 동적으로 업데이트하는 $L = \sum \alpha_i \cdot \text{Loss}_i$ 형태의 손실 함수 아이디어를 제시하였으나, 실제 구현 및 검증까지는 이르지 못했다.
- **품질 지표의 세분화:** 이미지 품질을 해상도, 선명도, 노이즈 레벨, 색상 정확도 등으로 세분화하여 분석하는 시도를 하였으나, 이는 이론적 근거가 부족한 예비 실험 단계이다.

## 📌 TL;DR

본 논문은 AI 생성 이미지의 품질, 진위성, 일치도를 평가하기 위해 프롬프트 설계 기법과 **Metric Transformer**라는 새로운 구조를 제안하였다. 특히 서로 다른 평가 지표들이 유사한 특성을 공유한다는 점을 이용해, 단일 모델로 다중 지표를 동시에 산출하는 구조를 설계함으로써 연산 효율성을 극대화하면서도 인간의 지각과 유사한 높은 평가 정확도를 달성하였다. 이 연구는 향후 효율적인 AGI 평가 프레임워크 구축 및 다중 지표 통합 학습 연구에 중요한 기초 자료가 될 것으로 보인다.