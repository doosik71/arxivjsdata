# Are Compressed Language Models Less Subgroup Robust?

Leonidas Gee, Andrea Zugarini, Novi Quadrianto (2024)

## 🧩 Problem to Solve

본 연구는 거대 언어 모델(Large Language Models, LLMs)의 추론 비용을 줄이기 위해 널리 사용되는 모델 압축(Model Compression) 기술이 **서브그룹 강건성(Subgroup Robustness)**에 미치는 영향을 분석하는 것을 목표로 한다.

데이터셋 내에는 레이블과 속성의 조합으로 정의되는 여러 서브그룹이 존재하며, 데이터 불균형으로 인해 일반적인 경험적 위험 최소화(Empirical Risk Minimization, ERM) 방식의 학습은 다수 그룹(Majority Group)의 성능은 높이지만 소수 그룹(Minority Group)의 성능은 저하시키는 경향이 있다. 모델 압축이 전체적인 성능(Average Accuracy)을 유지하면서도, 이러한 소수 그룹에 대한 성능인 최악 그룹 정확도(Worst-group Accuracy, WGA)를 어떻게 변화시키는지에 대해서는 지금까지 충분히 연구되지 않았다. 따라서 본 논문은 다양한 압축 방법론이 언어 모델의 서브그룹 강건성에 미치는 영향을 체계적으로 조사하고자 한다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 모델 압축이 서브그룹 강건성에 미치는 영향이 단순히 모델의 크기에 의해서만 결정되는 것이 아니라, **사용된 압축 방법론과 데이터셋의 특성에 따라 달라진다**는 점을 밝혀낸 것이다. 

특히, 모델 압축이 항상 소수 그룹의 성능을 희생시키는 것이 아니며, 특정 상황(예: 모델이 데이터셋에 쉽게 오버피팅되는 경우)에서는 압축이 일종의 정규화(Regularization) 역할을 하여 오히려 서브그룹 강건성을 향상시킬 수 있다는 직관을 제시한다.

## 📎 Related Works

기존의 모델 압축 연구는 주로 지식 증류(Knowledge Distillation), 가지치기(Pruning), 양자화(Quantization) 등의 기법을 통해 모델 크기와 지연 시간을 줄이는 데 집중해 왔다. 서브그룹 강건성과 압축의 관계를 다룬 기존 연구들은 주로 컴퓨터 비전 분야(예: CIFAR-10, ImageNet, CelebA)에 국한되어 있었다. 

이전의 이미지 데이터셋 기반 연구들에 따르면, 모델 압축 시 전체 성능을 유지하기 위해 소수 클래스나 저빈도 속성을 가진 그룹의 성능이 희생되는 '자기 잠식(Cannibalize)' 현상이 관찰되었다. 하지만 본 논문은 이러한 분석을 NLP 설정으로 확장하였으며, 이전 연구들보다 훨씬 더 광범위한 압축 방법론을 적용하여 분석했다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 전체 파이프라인 및 분석 대상
연구진은 BERT 모델을 기반으로 총 18가지의 압축 설정과 방법론을 적용하여 3개의 텍스트 데이터셋에서 성능을 측정하였다.

### 주요 압축 방법론
1.  **Knowledge Distillation (KD):** $\text{BERT}_{\text{Base}}$를 교사 모델로 하여 $\text{BERT}_{\text{Medium}}$, $\text{BERT}_{\text{Small}}$, $\text{BERT}_{\text{Mini}}$, $\text{BERT}_{\text{Tiny}}$, $\text{DistilBERT}$, $\text{TinyBERT}_6$, $\text{TinyBERT}_4$ 등 7가지 모델을 분석하였다.
2.  **Pruning:** L1-norm을 기준으로 Transformer head를 정렬하여 제거하는 구조적 가지치기(Structured Pruning)를 수행하였으며, 희소성(Sparsity) 수준을 20%, 40%, 60%, 80%로 나누어 $\text{BERT}_{\text{PR20}}$부터 $\text{BERT}_{\text{PR80}}$까지 분석하였다.
3.  **Quantization:** PyTorch에서 지원하는 동적 양자화(Dynamic Quantization, $\text{BERT}_{\text{DQ}}$), 정적 양자화(Static Quantization, $\text{BERT}_{\text{SQ}}$), 양자화 인식 훈련(Quantization-aware Training, $\text{BERT}_{\text{QAT}}$)을 통해 FP32 가중치를 INT8로 매핑하였다.
4.  **Vocabulary Transfer (VT):** 어휘 사전의 크기를 100%, 75%, 50%, 25%로 조정하여 토큰화 효율성을 높이는 방식을 분석하였다.

### 데이터셋 및 서브그룹 정의
- **MultiNLI:** 가설-전제 관계 예측 작업. 속성은 가설 내에 부정어(nobody, no, never, nothing)의 포함 여부로 정의된다.
- **CivilComments:** 댓글의 독성(Toxicity) 예측 작업. 속성은 인구통계학적 정체성(성별, 종교, 인종 등)의 포함 여부로 정의된다.
- **SCOTUS:** 미국 연방 대법원 판결문의 주제 분류 작업. 속성은 판결의 방향성(자유주의 vs 보수주의)으로 정의된다.

### 학습 및 측정 절차
- 모든 압축 모델은 ERM 방식으로 학습되었으며, 결과의 신뢰성을 위해 5번의 무작위 초기화 후 평균값을 사용하였다.
- 평가지표로는 전체 평균 정확도(Average Accuracy)와 최악 그룹 정확도(Worst-group Accuracy, WGA)를 사용하였다.

## 📊 Results

### 모델 크기와 서브그룹 강건성의 관계
실험 결과, 데이터셋에 따라 상반된 경향이 나타났다.
- **MultiNLI 및 SCOTUS:** 모델 크기가 감소함에 따라 평균 정확도와 WGA가 모두 감소하는 경향을 보였다. 특히 SCOTUS에서는 레이어 수가 6개 미만인 모델($\text{BERT}_{\text{Mini}}$, $\text{BERT}_{\text{Tiny}}$, $\text{TinyBERT}_4$)의 WGA가 0으로 수렴하는 현상이 발생하였다.
- **CivilComments:** 흥미롭게도 모델 크기가 줄어들었음에도 불구하고 WGA가 오히려 향상되는 결과가 나타났다. $\text{BERT}_{\text{Tiny}}$와 같은 극도로 압축된 모델조차 $\text{BERT}_{\text{Base}}$보다 높은 WGA를 기록하였다. 이는 $\text{BERT}_{\text{Base}}$가 해당 데이터셋에 과적합(Overfitting)되어 있으며, 모델 크기 축소가 일종의 정규화 효과를 주어 소수 그룹에 대한 일반화 성능을 높였기 때문으로 해석된다.

### 압축 방법론의 영향
동일한 파라미터 수를 가진 모델이라도 압축 방법(가중치 초기화 상태)에 따라 WGA가 다르게 나타났다.
- **$\text{DistilBERT}$ vs $\text{TinyBERT}_6$:** 두 모델은 파라미터 수가 비슷하지만, MultiNLI와 CivilComments에서는 $\text{TinyBERT}_6$가 더 높은 WGA를 보였고, SCOTUS에서는 $\text{DistilBERT}$가 더 우세하였다.
- **양자화 방식:** 추가적인 파인튜닝 단계가 없는 사후 양자화(Post-training Quantization, $\text{BERT}_{\text{DQ}}$, $\text{BERT}_{\text{SQ}}$) 방식이 양자화 인식 훈련($\text{BERT}_{\text{QAT}}$) 방식보다 서브그룹 강건성이 현저히 낮았다.

### 작업 복잡도 및 분포 분석
MultiNLI를 이진 분류 작업으로 단순화하여 복잡도를 낮추었을 때, 전체적인 성능은 향상되었으나 모델 크기 감소에 따라 WGA가 하락하는 전반적인 추세는 변하지 않았다. 이는 서브그룹 강건성이 단순히 맞춰야 할 서브그룹의 개수(작업 복잡도)보다는 모델의 용량과 데이터셋의 특성에 더 의존함을 시사한다.

## 🧠 Insights & Discussion

본 연구는 모델 압축이 반드시 소수 그룹의 성능을 희생시켜 전체 성능을 유지한다는 기존의 통념을 반박한다. 분석 결과, 모델 압축은 데이터셋의 특성에 따라 오히려 소수 그룹의 성능을 개선하는 정규화 도구로 작용할 수 있음을 보여주었다.

**강점 및 시사점:**
- 다양한 압축 기법(KD, Pruning, Quantization, VT)을 통합적으로 분석하여 방법론별 특성을 규명하였다.
- 단순한 크기 감소가 아닌, 압축 방식에 따른 '추정 오차(Estimation Error)'가 강건성에 영향을 준다는 점을 지적하였다.

**한계 및 향후 과제:**
- 분석이 영어 데이터셋에 국한되어 있어, 다국어 환경에서의 강건성 분석이 추가로 필요하다.
- 각 압축 방법을 독립적으로 적용하였으며, 여러 기법을 조합(예: KD 후 Quantization)했을 때의 상호작용에 대해서는 다루지 않았다.

## 📌 TL;DR

이 논문은 모델 압축이 언어 모델의 서브그룹 강건성(특히 소수 그룹의 성능)에 미치는 영향을 분석하였다. 실험 결과, 압축은 데이터셋에 따라 WGA를 떨어뜨리기도 하지만, 오버피팅이 심한 데이터셋에서는 오히려 정규화 효과를 통해 강건성을 높이기도 한다는 것을 발견하였다. 또한, 동일한 크기의 모델이라도 압축 방법론에 따라 강건성이 달라지므로, 효율적인 모델 구축 시 단순한 크기 축소보다 적절한 압축 기법의 선택이 중요함을 시사한다.