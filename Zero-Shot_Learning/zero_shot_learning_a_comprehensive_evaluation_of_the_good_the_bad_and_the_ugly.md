# Zero-Shot Learning - A Comprehensive Evaluation of the Good, the Bad and the Ugly

Yongqin Xian, Christoph H. Lampert, Bernt Schiele, and Zeynep Akata (2017)

## 🧩 Problem to Solve

본 논문은 Zero-Shot Learning (ZSL) 분야에서 급증하는 제안 방법론들에 비해 이를 객관적으로 평가할 수 있는 표준화된 벤치마크와 평가 프로토콜이 부재하다는 문제를 지적한다. 특히, 기존 연구들에서 나타난 다음과 같은 문제점들에 집중한다.

첫째, 데이터셋의 분할(split) 방식이 통일되지 않아 서로 다른 논문의 결과물을 직접적으로 비교하는 것이 어렵다. 둘째, 일부 연구에서는 테스트 클래스가 포함된 데이터셋으로 사전 학습(pre-training)된 모델을 사용하여 특징을 추출함으로써 ZSL의 기본 가정을 위반하는 '결함이 있는(flawed)' 평가를 수행하고 있다. 셋째, 테스트 시점에 오직 보지 못한 클래스($Y_{ts}$)만 존재한다고 가정하는 기존 ZSL 설정은 비현실적이며, 학습 클래스($Y_{tr}$)와 테스트 클래스가 모두 존재할 수 있는 Generalized Zero-Shot Learning (GZSL) 설정에서의 분석이 부족하다.

따라서 본 논문의 목표는 통일된 평가 프로토콜과 데이터 분할 방식을 정의하고, 새로운 데이터셋인 AWA2를 제안하며, 최신 ZSL 방법론들을 ZSL 및 GZSL 설정에서 심층적으로 비교 분석하여 해당 분야의 현주소를 진단하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 ZSL 연구의 신뢰성을 높이기 위한 '표준화'와 '현실화'에 있으며, 주요 내용은 다음과 같다.

1. **통일된 평가 벤치마크 정의**: 데이터셋 분할과 평가 프로토콜을 통일하여 방법론 간의 공정한 비교가 가능하도록 하였다. 특히 ResNet-101 사전 학습 모델의 학습 데이터와 ZSL의 테스트 클래스가 겹치지 않도록 하는 Proposed Splits (PS)를 제안하여 데이터 누수(leakage) 문제를 해결하였다.
2. **Animals with Attributes 2 (AWA2) 데이터셋 제안**: 기존 AWA1 데이터셋은 원본 이미지에 대한 공개 라이선스가 없어 새로운 DNN 특징 추출기를 적용하기 어려웠다. 이를 해결하기 위해 동일한 클래스와 속성을 가지면서 공개 라이선스를 확보한 AWA2 데이터셋을 구축하고 이미지와 ResNet 특징을 공개하였다.
3. **광범위한 방법론 분석**: Linear/Nonlinear Compatibility Learning, Attribute Classifiers, Hybrid Models 등 다양한 SOTA 방법론들을 ZSL과 GZSL 두 가지 설정에서 체계적으로 비교 평가하였다.
4. **GZSL의 중요성 강조**: 실제 환경에 더 가까운 GZSL 설정을 도입하고, 학습 클래스와 테스트 클래스의 성능을 동시에 고려하는 Harmonic Mean 지표를 통해 보다 현실적인 성능 평가 기준을 제시하였다.

## 📎 Related Works

논문에서는 ZSL 접근 방식을 크게 다음과 같이 분류하여 설명한다.

- **Two-stage Approach**: 먼저 이미지의 속성(attribute)을 예측하고, 이후 예측된 속성과 클래스 간의 유사도를 통해 라벨을 추론하는 방식이다. DAP, IAP 등이 이에 해당하며, 중간 단계의 task와 최종 task 간의 도메인 시프트(domain shift) 문제가 발생한다는 한계가 있다.
- **Compatibility Learning**: 이미지 특징 공간에서 시맨틱 공간(semantic space)으로의 매핑을 직접 학습하는 방식이다. ALE, DeViSE, SJE 등이 있으며, 최근의 주류를 이루고 있다.
- **Nonlinear & Hybrid Models**: 비선형 매핑을 도입한 LATEM, CMT나, 이미지와 시맨틱 임베딩을 공통 공간으로 투영하는 SSE, CONSE, SYNC 등이 있다.
- **Generative Models**: 각 클래스를 확률 분포로 표현하여 가상 인스턴스를 생성하는 GFZSL 등이 제안되었다.
- **Transductive ZSL**: 학습 단계에서 라벨이 없는 테스트 이미지들을 활용하여 성능을 높이는 방식이다.

기존 연구들은 주로 ZSL 설정(테스트 시 unseen 클래스만 존재)에 치중했으나, 본 논문은 이를 확장하여 GZSL 설정에서의 분석을 수행함으로써 차별점을 둔다.

## 🛠️ Methodology

### 1. ZSL 및 GZSL의 정의

ZSL의 목표는 학습 시 보지 못한 클래스 $Y_{ts}$에 대해 예측하는 것이다.

- **ZSL**: 테스트 시 검색 공간이 $Y_{ts}$로 제한된다.
- **GZSL**: 테스트 시 검색 공간이 $Y_{tr} \cup Y_{ts}$로 확장되어, 모델이 학습 클래스와 테스트 클래스를 모두 구분해야 한다.

### 2. 평가 대상 방법론의 분류 및 핵심 원리

본 논문은 다양한 방법론을 평가하며, 특히 **Linear Compatibility Learning**의 기본 구조를 다음과 같이 정의한다.

$$F(x,y;W) = \theta(x)^T W \phi(y)$$

여기서 $\theta(x)$는 이미지 임베딩, $\phi(y)$는 클래스 임베딩, $W$는 학습해야 할 매핑 행렬이다. $F(\cdot)$는 이미지와 클래스 간의 호환성 점수(compatibility score)를 나타내며, 가장 높은 점수를 가진 클래스를 최종 예측값으로 선택한다.

- **Ranking Loss (DeViSE, ALE, SJE)**: 정답 클래스의 점수가 오답 클래스의 점수보다 일정 마진 이상 높도록 학습한다.
- **Square Loss & Regularization (ESZSL)**: Frobenius norm 등을 이용한 명시적 정규화를 통해 과적합을 방지한다.
- **Auto-encoder (SAE)**: 이미지 특징을 시맨틱 공간으로 투영했다가 다시 복원하는 제약 조건을 추가하여 학습한다.

### 3. 평가 프로토콜 (Proposed Evaluation Protocol)

- **특징 추출**: ResNet-101 (ImageNet-1K 사전 학습)을 사용하여 2048차원 특징을 추출한다.
- **Proposed Splits (PS)**: ResNet-101이 학습한 ImageNet-1K 클래스가 테스트 클래스에 포함되지 않도록 분할을 재설계하여, 사전 학습 모델을 통한 정보 누수를 차단한다.
- **평가 지표**:
  - **Per-class Average Top-1 Accuracy**: 클래스별 이미지 수의 불균형을 해소하기 위해 클래스별 정확도를 먼저 계산한 후 평균을 낸다.
  - **Harmonic Mean ($H$)**: GZSL에서 학습 클래스 정확도($acc_{Y_{tr}}$)와 테스트 클래스 정확도($acc_{Y_{ts}}$)의 조화 평균을 사용한다.
    $$H = \frac{2 \times acc_{Y_{tr}} \times acc_{Y_{ts}}}{acc_{Y_{tr}} + acc_{Y_{ts}}}$$

## 📊 Results

### 1. ZSL 실험 결과

- **분할 방식의 영향**: Standard Split (SS)보다 Proposed Split (PS)에서 성능이 유의미하게 하락했다. 이는 기존의 높은 성능이 상당 부분 사전 학습 데이터와 테스트 데이터의 중복(overlap)에서 기인했음을 시사한다.
- **최우수 모델**: 전반적으로 Generative model인 GFZSL과 Compatibility learning 방식인 ALE가 가장 견고한 성능을 보였다.
- **AWA2의 유효성**: AWA2 데이터셋에서의 결과가 AWA1과 매우 유사하게 나타났으며, 교차 데이터셋 평가(Cross-dataset evaluation)를 통해 AWA2가 AWA1의 적절한 대체제임을 확인하였다.

### 2. GZSL 실험 결과

- **성능 저하**: ZSL 설정보다 GZSL 설정에서 정확도가 현저히 낮게 나타났다. 이는 학습 클래스가 '방해 요소(distractor)'로 작용하기 때문이다.
- **모델별 경향**:
  - Compatibility Learning (ALE, DeViSE, SJE)은 테스트 클래스($Y_{ts}$) 예측에 강점이 있다.
  - Attribute Classifiers (DAP, CONSE)는 학습 클래스($Y_{tr}$) 예측에 강점이 있다.
  - 조화 평균($H$) 기준으로는 ALE가 여러 데이터셋에서 가장 우수한 성능을 보였다.

### 3. ImageNet 대규모 실험

- 대규모 설정에서는 SYNC가 가장 우수한 성능을 보였는데, 이는 Word2Vec 임베딩의 활용 능력이 크기 때문으로 분석된다. 반면, 속성(attribute)에 의존하는 GFZSL은 속성 정보가 부족한 ImageNet에서 성능이 크게 하락하였다.

## 🧠 Insights & Discussion

본 논문은 ZSL 분야의 "Good, Bad, Ugly"를 명확히 구분하여 분석하였다.

- **The Good**: GFZSL과 ALE와 같은 모델들이 보여준 강력한 일반화 성능과 Generative approach의 잠재력이다.
- **The Bad**: 일관되지 않은 벤치마크와 평가 프로토콜로 인해 방법론 간의 객관적 비교가 불가능했던 상황이다.
- **The Ugly**: 테스트 클래스가 사전 학습 데이터에 포함되어 성능이 부풀려진 데이터 누수 문제이다.

**비판적 해석 및 논의**:
ZSL 연구들이 그동안 "상수"처럼 여겼던 사전 학습 특징 추출기(Pre-trained Feature Extractor)가 사실은 학습 과정의 일부임을 지적한 점이 매우 중요하다. 이는 단순히 알고리즘의 개선뿐만 아니라, 데이터 분할과 같은 실험 설계의 엄밀함이 딥러닝 기반 ZSL 연구에서 필수적임을 보여준다. 또한, GZSL에서 나타난 학습/테스트 클래스 간의 성능 불균형은 단순한 정확도 향상보다 두 클래스 군의 성능 균형을 맞추는 정교한 정규화 기법이 필요함을 시사한다.

## 📌 TL;DR

본 논문은 ZSL 분야의 파편화된 평가 방식을 바로잡기 위해 **통일된 벤치마크**와 **데이터 누수가 없는 새로운 분할 방식(Proposed Splits)**을 제안하고, 공개 라이선스를 가진 **AWA2 데이터셋**을 구축하였다. 실험을 통해 GFZSL과 ALE가 ZSL에서 강력함을 입증하였으며, 현실적인 **GZSL 설정**에서는 조화 평균 지표를 통해 모델의 진정한 일반화 능력을 평가해야 함을 강조하였다. 이 연구는 향후 ZSL 연구들이 더 엄격하고 현실적인 기준 위에서 진행될 수 있도록 가이드라인을 제시한 중요한 이정표가 될 가능성이 높다.
