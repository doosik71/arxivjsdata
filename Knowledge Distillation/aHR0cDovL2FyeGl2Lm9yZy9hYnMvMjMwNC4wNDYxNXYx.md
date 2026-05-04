# A Survey on Recent Teacher-student Learning Studies

Gao Minghong(2023)

## 🧩 Problem to Solve

본 논문은 복잡하고 거대한 딥러닝 모델인 Teacher 네트워크의 지식을 작고 효율적인 Student 네트워크로 전이하여, 모델의 크기와 계산 비용을 줄이면서도 성능(정확도)을 최대한 유지하고자 하는 Knowledge Distillation (KD)의 한계점과 이를 해결하기 위한 최신 연구들을 분석한다.

전통적인 KD 방식이 직면한 주요 문제점은 다음과 같다.
- **Capacity Gap**: Teacher와 Student 사이의 모델 크기 차이가 너무 클 경우, Student가 Teacher의 복잡한 지식을 충분히 흡수하지 못해 오히려 성능이 저하되는 현상이 발생한다.
- **Knowledge Imbalance**: Teacher의 예측 확률 분포가 클래스별로 불균형하여, 특정 "tail" 클래스의 지식 전이가 제대로 이루어지지 않는 문제가 존재한다.
- **Over-confidence**: 매우 강력한 Teacher 모델이 너무 확신에 찬(confident) 예측을 내놓을 경우, Student가 배워야 할 세밀한 클래스 간 관계(dark knowledge)가 억제되는 결과가 나타난다.
- **Inefficient Training**: 고정된 Temperature 하이퍼파라미터나 단순한 Logit 모방 방식은 학습 과정에서의 효율성과 유연성을 떨어뜨린다.

본 논문의 목표는 이러한 문제들을 해결하기 위해 제안된 다양한 KD 변형 기법들(Teaching Assistant, Curriculum, Mask, Decoupling Distillation 등)의 핵심 아이디어와 방법론을 정리하고 그 효과를 분석하는 것이다.

## ✨ Key Contributions

본 논문은 최신 KD 연구들을 분석하여 다음과 같은 핵심적인 설계 아이디어들을 제시한다.

- **단계적 지식 전이**: Teacher와 Student 사이에 중간 크기의 모델(Teaching Assistant)을 배치하여 Capacity Gap을 완화한다.
- **학습 난이도 조절**: 인간의 교육 과정과 유사하게 학습 난이도를 점진적으로 높이는 Curriculum learning 개념을 KD에 도입하여 학습 효율을 높인다.
- **손실 함수의 분리(Decoupling)**: 전체 KD 손실을 타겟 클래스(Target Class)와 비타겟 클래스(Non-target Class) 지식으로 분리하여, 각각의 중요도에 따라 가중치를 독립적으로 조절한다.
- **가중치 및 온도 최적화**: 데이터의 불균형을 해결하기 위한 Inverse Probability Weighting 도입 및 클래스별 정답 여부에 따라 온도를 다르게 적용하는 Asymmetric Temperature Scaling을 제안한다.
- **생성적 특징 복원**: 단순한 특징 모방에서 벗어나, 마스킹된 특징으로부터 전체 특징을 복원하게 함으로써 Student의 표현력을 강화한다.

## 📎 Related Works

본 논문은 Hinton 등이 제안한 전통적인 Knowledge Distillation [7]을 기본 배경으로 하며, 이를 확장한 다양한 최신 접근 방식들을 다룬다.

- **기존 접근 방식**: 주로 Teacher와 Student의 Softmax 출력값(Logits) 사이의 KL Divergence를 최소화하는 방식에 의존하였다.
- **한계점**: 기존 방식은 Teacher가 무조건적으로 "더 정확할수록 더 좋은 스승"이라는 가정을 전제로 하지만, 실제로는 Teacher와 Student의 능력 차이가 너무 클 때 성능이 하락하는 현상이 관찰되었다.
- **차별점**: 본 논문에서 소개하는 최신 기법들은 단순히 결과값을 맞추는 것을 넘어, 지식 전이의 '과정'(Curriculum), '구조'(TA), '세부 구성 요소'(Decoupling), '관계'(DIST) 등에 집중하여 전이 효율을 극대화한다는 점에서 기존 방식과 차별화된다.

## 🛠️ Methodology

본 논문은 여러 최신 KD 방법론을 소개하고 있으며, 각 방법론의 핵심 메커니즘은 다음과 같다.

### 1. Teacher Assistant (TA)
Teacher($T$)와 Student($S$)의 크기 차이가 너무 클 때 발생하는 성능 저하를 막기 위해, 중간 크기의 모델인 $TA$를 도입한다. 학습 절차는 $T \rightarrow TA \rightarrow S$ 순으로 진행되며, $T$가 $TA$를 가르치고, 다시 $TA$가 $S$를 가르치게 함으로써 지식의 전이 간격을 좁힌다.

### 2. Inverse Probability Weighted Distillation (IPWD)
Teacher의 예측 분포가 불균형하여 발생하는 "transfer gap"을 해결한다.
- **핵심 아이디어**: 클래스 인식 예측과 문맥 인식 예측을 비교하여 Propensity score를 추정하고, 이를 역수로 취한 가중치를 distillation loss에 적용한다.
- **효과**: 상대적으로 가중치가 낮게 부여되었던 "tail" 클래스의 샘플들에 더 큰 가중치를 부여함으로써 편향되지 않은(de-biased) 지식 전이를 가능하게 한다.

### 3. Decoupled Knowledge Distillation (DKD)
전통적인 KD Loss를 다음과 같이 두 부분으로 분리한다.
- **Target Class Knowledge Distillation (TCKD)**: 정답 클래스에 관련된 지식으로, 학습 샘플의 난이도와 관련이 있다.
- **Non-Target Class Knowledge Distillation (NCKD)**: 정답 외 클래스들 간의 관계에 대한 지식이다.
- **목표**: Teacher가 너무 확신에 찬 예측을 할 때 NCKD가 억제되는 문제를 해결하기 위해, 두 손실을 독립적으로 다루어 가중치를 최적화한다.

### 4. Asymmetric Temperature Scaling (ATS)
Temperature scaling 시 정답 클래스와 오답 클래스에 서로 다른 온도를 적용한다.
- **원리**: 정답 클래스의 Logit은 낮추고 오답 클래스 간의 다양성을 높여, Student가 클래스 간의 변별력(Category discrimination)을 더 잘 학습하도록 유도한다.

### 5. DIST (Distillation from a Stronger Teacher)
KL Divergence를 통한 정확한 확률값 일치 대신, 상관관계 기반의 손실 함수를 사용한다.
- **방법**: Pearson correlation coefficient를 사용하여 클래스 간(inter-class) 및 클래스 내(intra-class) 관계를 캡처한다.
- **특징**: 절대적인 확률값이 아닌 상대적인 순위(relative rank)를 맞추는 "loose matching" 방식을 통해 매우 강력한 Teacher로부터의 전이 효율을 높인다.

### 6. Course Temperature for Knowledge Distillation (CTKD)
온도 $\tau$를 고정하지 않고 학습 가능하며 동적인 값으로 설정한다.
- **절차**: Adversarial approach를 사용하여 Student의 학습 난이도를 점진적으로 높이는 Easy-to-Hard curriculum을 구성하며, Cosine schedule을 통해 $\tau$를 조절한다.

### 7. Masked Generative Distillation (MGD)
단순한 Feature imitation이 아니라, 생성적 복원 방식을 취한다.
- **방법**: Teacher의 Feature map에 랜덤 마스크를 씌워 일부를 가리고, Student가 가려지지 않은 부분만을 이용해 Teacher의 전체 Feature를 복원하도록 학습시킨다.
- **효과**: Student가 단순 모방을 넘어 더 강력한 표현(representation) 능력을 갖추게 한다.

### 8. Simple Knowledge Distillation (SimKD)
복잡한 하이퍼파라미터 튜닝 없이 Teacher의 Classifier를 Student에게 직접 재사용하는 방식이다.
- **방법**: Student의 Encoder가 Teacher의 Classifier와 잘 맞도록 Feature alignment만을 통해 학습시킨다.

## 📊 Results

본 논문은 개별 방법론들의 실험 결과를 요약하여 제시하고 있다.

- **데이터셋 및 작업**: CIFAR-100, ImageNet-2012, MS-COCO (Object Detection), Cityscapes (Semantic Segmentation) 등이 사용되었다.
- **주요 결과**:
    - **TA**: Teacher의 깊이가 너무 깊을 때보다 TA를 거쳤을 때 Student의 성능이 유의미하게 향상되었다.
    - **DKD**: CIFAR-100, ImageNet에서 기존 Logit 기반 방식보다 우수한 성능을 보였으며, Object Detection에서도 효과적임이 입증되었다.
    - **MGD**: ImageNet Top-1 정확도를 ResNet-18 기준 69.90%에서 71.69%로 향상시켰으며, 특히 MS-COCO에서 Bounding-box mAP와 Mask mAP가 크게 개선되었다.
    - **DIST**: 강력한 Teacher 모델을 사용할 때 vanilla KD보다 일관되게 높은 성능을 보였으며, 특히 Semantic Segmentation 작업에서 우수함을 확인하였다.
    - **CTKD**: 동적 온도를 적용했을 때 다양한 Student 네트워크(VGG, ResNet 등)에서 성능 향상이 나타났다.

## 🧠 Insights & Discussion

본 논문을 통해 도출할 수 있는 비판적 해석과 통찰은 다음과 같다.

- **The "Stronger is Better" Paradox**: 일반적으로 더 정확한 모델이 더 좋은 Teacher가 될 것이라 생각하지만, 실제로는 Capacity Gap으로 인해 성능이 하락하는 지점이 존재한다. 이는 지식 전이가 단순한 정답 모방이 아니라, Student가 수용 가능한 수준의 "적절한 정보 밀도"를 전달하는 과정임을 시사한다.
- **Logit vs Feature**: 최근 연구들은 Logit 수준의 전이를 넘어, Feature의 생성적 복원(MGD)이나 구조적 관계(DIST)를 학습하는 방향으로 진화하고 있다. 이는 단순한 확률 분포의 일치보다 모델 내부의 표현 공간(Representation space)을 일치시키는 것이 더 근본적인 해결책임을 보여준다.
- **한계점 및 미해결 과제**:
    - 많은 기법들이 특정 데이터셋(CIFAR, ImageNet)에서 성능 향상을 보였으나, 실제 다양한 도메인의 데이터셋에서도 일반화될 수 있는지에 대한 추가 검증이 필요하다.
    - MGD와 같은 Feature 기반 방식은 여전히 Teacher와 Student의 아키텍처가 유사할 때 더 효과적인 경향이 있으며, 완전히 서로 다른 아키텍처 간의 전이 효율을 높이는 문제는 여전히 과제로 남아 있다.

## 📌 TL;DR

본 논문은 모델 압축을 위한 Knowledge Distillation의 최신 동향을 분석한 서베이 논문이다. 단순히 Teacher의 출력을 모방하는 기존 방식의 한계(Capacity Gap, Knowledge Imbalance 등)를 극복하기 위해 **중간 모델 도입(TA), 손실 함수 분리(DKD), 동적 온도 조절(CTKD), 생성적 특징 복원(MGD), 관계 기반 매칭(DIST)** 등의 혁신적인 방법론들을 체계적으로 정리하였다. 이 연구는 향후 더 효율적인 모델 압축 기술을 설계하고, 특히 매우 거대한 모델(LLM 등)의 지식을 작은 모델로 전이하는 연구에 중요한 기초 자료가 될 가능성이 높다.