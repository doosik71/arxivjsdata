# Any-Shot Object Detection

Shafin Rahman, Salman Khan, Nick Barnes, Fahad Shahbaz Khan (2020)

## 🧩 Problem to Solve

본 논문은 기존의 객체 탐지(Object Detection) 연구가 가진 한계인 제로샷 탐지(Zero-Shot Detection, ZSD)와 퓨샷 탐지(Few-Shot Detection, FSD)의 이분법적 접근 방식을 해결하고자 한다. 기존 연구들은 학습 데이터가 전혀 없는 상황(Zero-shot)이나, 극소수의 샘플만 존재하는 상황(Few-shot) 중 하나만을 가정하여 모델을 설계하였다. 하지만 실제 현실 세계의 시나리오에서는 완전히 새로운 클래스(Unseen)와 소수의 샘플만 존재하는 클래스(Few-shot), 그리고 이미 학습된 클래스(Seen)가 동시에 공존하는 경우가 훨씬 더 빈번하게 발생한다.

따라서 본 연구의 목표는 이 세 가지 설정이 동시에 발생하는 'Any-shot Detection(ASD)'이라는 새로운 설정을 정의하고, 이를 통합적으로 해결할 수 있는 프레임워크를 제안하는 것이다. 특히 ASD 환경에서는 다음과 같은 세 가지 주요 문제가 발생한다. 첫째, Seen, Few-shot, Unseen 클래스 간의 극심한 데이터 불균형(Class Imbalance) 문제이다. 둘째, 소수의 샘플로 모델을 미세 조정(Fine-tuning)하는 과정에서 기존에 학습한 지식을 잊어버리는 치명적 망각(Catastrophic Forgetting) 문제이다. 셋째, 완전히 새로운 클래스를 배경(Background)과 구분하여 정확히 탐지해야 하는 어려움이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 클래스의 시맨틱 정보(Semantic Semantics)를 객체 탐지를 위한 프로토타입(Prototype)으로 활용하는 것이다. 이를 통해 시각적 특징 공간을 시맨틱 공간으로 매핑함으로써, 네트워크 파라미터를 직접적으로 대폭 수정하지 않고도 새로운 클래스를 확장할 수 있는 유연성을 확보한다.

주요 기여 사항은 다음과 같다.

1. ZSD, FSD, ASD 및 이들의 일반화된 설정(Generalized settings)을 모두 수용할 수 있는 통합 프레임워크를 제안하였다.
2. 시각적 정보와 잘 정렬된 시맨틱 클래스 프로토타입을 학습하여, 새로운 클래스를 추가할 때 기존 지식의 망각을 최소화하는 메커니즘을 설계하였다.
3. 데이터 불균형 문제를 해결하기 위해, 어려운 사례(Difficult cases)에는 높은 패널티를 부여하되 과적합(Overfitting)은 방지하여 Unseen 클래스에 대한 일반화 성능을 유지하는 새로운 재균형 손실 함수(Rebalancing Loss Function)를 제안하였다.

## 📎 Related Works

기존의 N-shot 인식(Recognition) 연구는 주로 ZSR(Zero-shot Recognition)과 FSR(Few-shot Recognition)로 나뉘어 발전해 왔다. ZSR은 속성(Attribute)이나 워드 벡터(Word vector)를 이용해 Seen과 Unseen 클래스를 연결하며, FSR은 메타 학습(Meta-learning)이나 거리 학습(Metric learning)을 통해 소수 샘플로부터 새로운 클래스를 학습한다. 최근에는 이 두 가지를 통합하려는 시도가 있었으나, 이는 단순한 분류(Classification) 작업에 국한되었으며 탐지(Detection) 작업으로 확장되지는 않았다.

객체 탐지 분야에서 ZSD 연구들은 주로 고정된 텍스트 설명이나 프로포절(Proposal) 기반 방법론을 사용하지만, 추론 단계에서 새로운 시각적 샘플이 제공되는 FSD 상황을 처리할 수 없다. 반면 FSD 연구들은 소수의 인스턴스를 통해 모델을 적응시키지만, 테스트 시점에 Seen 클래스를 동시에 탐지해야 하는 일반화된 설정(Generalized FSD)이나 데이터가 전혀 없는 ZSD 상황을 동시에 처리하지 못하는 한계가 있다. 본 논문은 이러한 간극을 메워 ZSD와 FSD를 단일 프레임워크 내에서 ASD로 통합하였다.

## 🛠️ Methodology

### 전체 시스템 구조 및 파이프라인

본 모델은 RetinaNet 아키텍처를 기반으로 수정되었으며, 크게 시각적 특징을 추출하는 Backbone 네트워크와 이를 시맨틱 공간으로 매핑하는 분류/회귀 서브넷으로 구성된다. 전체 프로세스는 '기본 학습(Base Training)'과 '미세 조정(Fine-tuning)'의 두 단계로 진행된다.

### 시각적-시맨틱 정렬 (Visual-Semantic Alignment)

모델은 시각적 특징 학습과 분류 단계를 분리하여 망각 문제를 해결한다. 입력 이미지 $X$가 네트워크 $f(\cdot)$를 통과하여 시각적 특징 벡터 $f(X) \in \mathbb{R}^n$를 생성하면, 동시에 클래스 워드 벡터 $W_s$는 가벼운 서브넷 $g(\cdot)$를 통해 변환된 워드 벡터 $g(W_s)$가 된다. 이 두 스트림은 학습 가능한 투영 층(Projection layer) $U \in \mathbb{R}^{n \times d}$를 통해 연결되며, 최종 예측 점수 $p_s$는 다음과 같이 계산된다.

$$p_s = \sigma(f(X)^T U g(W_s))$$

여기서 $\sigma$는 시그모이드 활성화 함수이다. $U$는 시각적 도메인과 시맨틱 도메인을 잇는 가교 역할을 하며, 시각적 특징과 해당 클래스의 시맨틱 프로토타입 간의 정렬 호환성 점수를 극대화하도록 학습된다.

### 학습 절차

1. **Base Training**: Seen 클래스의 이미지와 시맨틱 정보만을 사용하여 시각적 특징을 시맨틱 공간으로 매핑하는 법을 학습한다.
2. **Fine-tuning**: 추론 단계에서 Few-shot 클래스의 샘플과 Novel 클래스의 시맨틱 정보가 제공되면, 기본 네트워크의 파라미터는 유지한 채 시맨틱 공간을 적응시킨다. 이때 $W_s$를 전체 클래스 $W$로 대체하여 $p = \sigma(f(X)^T U g(W))$를 계산한다.

### 재균형 손실 함수 (Rebalancing Loss)

데이터 불균형을 해결하기 위해 본 논문은 미세 조정 단계에서 특별한 손실 함수를 도입한다. 기본 Cross-Entropy(CE) 손실에 정렬 품질에 따른 패널티 함수 $h(p, p^*)$를 추가한다.

$$L(p) = -\log p + \beta h(p, p^*)$$
$$h(p, p^*) = \log(1 + p^* - p)$$

여기서 $p^*$는 패널티 수준을 결정하는 임계값으로, 고정된 값이나 동적으로 결정되는 값($p^* = \max_{i \in C} p_i$)을 사용할 수 있다. 이를 Focal Loss의 구조와 결합하여 최종적인 손실 함수는 다음과 같이 정의된다.

$$L(p) = \max[0, -\alpha_t (1-p_t)^\gamma \log p_t]$$

이때 $p_t$는 다음과 같이 정의되어, 정답 레이블 $y=1$인 경우에만 패널티가 적용된다.
$$p_t = \begin{cases} \frac{p}{(1+p^*-p)^\beta}, & \text{if } y=1 \\ 1-p, & \text{otherwise} \end{cases}$$

이 손실 함수는 예측 점수가 $p^*$보다 낮은 'Extreme case'에서는 높은 패널티를 부여하여 Few-shot 클래스에 대한 학습을 강제하고, 예측이 정확한 'Expected case'에서는 패널티를 낮추어 과적합을 방지함으로써 Unseen 클래스에 대한 일반화 성능을 유지한다. 최종 손실은 Seen 클래스에 대한 Focal Loss $L(s)$와 Novel 클래스에 대한 재균형 손실 $L(n)$의 가중 합으로 계산된다.

$$L = \lambda L(s) + (1-\lambda) L(n)$$

## 📊 Results

### 실험 설정

- **데이터셋**: MS-COCO 2014(65/15 split) 및 PASCAL VOC 2007/12(15/5 split)를 사용하였다.
- **ASD 설정**: MS-COCO의 15개 Novel 클래스를 8개의 Few-shot 클래스와 7개의 Unseen 클래스로 나누어 평가하였다.
- **지표**: FSD와 ASD에서는 mean Average Precision(mAP)을, Generalized ASD(GASD)에서는 Seen과 Novel 클래스 mAP의 조화 평균(Harmonic Mean, HM)을 사용하였다. ZSD는 Recall@100으로 측정하였다.
- **비교 대상**: 고정 시맨틱을 사용하는 Baseline-I, 소수의 Seen 샘플을 함께 사용하는 Baseline-II 및 기존 SOTA 모델들(LSTD, Kang et al. 등)과 비교하였다.

### 주요 결과

1. **ASD 성능**: 제안 방법은 모든 Shot 설정(1, 5, 10-shot)에서 Baseline-II를 크게 상회하는 성능을 보였다. 특히 Unseen 클래스의 mAP 향상 폭이 가장 컸으며, 이는 제안한 손실 함수가 Few-shot 과적합을 막아 Unseen 클래스 탐지 능력을 보존했음을 시사한다.
2. **망각 방지**: GASD 결과에서 Seen 클래스의 성능이 일정하게 유지됨을 확인하였다. 이는 시맨틱 프로토타입을 앵커로 사용함으로써 새로운 클래스를 학습할 때 기존 지식을 잊어버리지 않았음을 증명한다.
3. **FSD 및 ZSD 확장성**: $U=0$인 FSD 설정과 $Q=0$인 ZSD 설정에서도 각각 기존 SOTA 방법론보다 우수한 성능을 기록하였다. 특히 PASCAL VOC 데이터셋에서 3-shot, 10-shot 탐지 성능이 기존 LSTD나 Kang et al.의 방법보다 높게 나타났다.

## 🧠 Insights & Discussion

본 논문의 가장 큰 강점은 ZSD와 FSD를 별개의 문제가 아닌, 데이터 가용성의 연속선상에 있는 하나의 통합된 문제(ASD)로 정의하고 해결했다는 점이다. 특히 모델의 가중치를 직접 수정하여 새로운 클래스를 학습시키는 대신, 시각적 특징을 고정된(혹은 가볍게 변환된) 시맨틱 프로토타입에 정렬시키는 방식은 딥러닝 모델의 고질적인 문제인 치명적 망각을 매우 효율적으로 해결하는 전략이다.

또한, 제안된 재균형 손실 함수는 단순한 가중치 조절을 넘어, 예측 값의 품질($p^*$)에 따라 패널티를 동적으로 조절함으로써 '학습 강제'와 '일반화 유지'라는 상충하는 목표를 동시에 달성하였다.

다만, 본 연구는 시맨틱 정보를 위해 외부에서 사전 학습된 word2vec 벡터에 의존하고 있다. 만약 시맨틱 임베딩 자체가 클래스의 특성을 충분히 반영하지 못하거나, 도메인 특화 용어가 많은 데이터셋의 경우 성능이 저하될 가능성이 있다. 또한, 추론 시 시맨틱 정렬을 위한 연산 비용이 추가로 발생하며, 이에 대한 효율성 분석이 명시적으로 제시되지 않은 점이 아쉽다.

## 📌 TL;DR

본 논문은 Seen, Few-shot, Unseen 클래스가 동시에 존재하는 **Any-Shot Object Detection(ASD)**이라는 새로운 과제를 정의하고, 이를 해결하기 위한 통합 프레임워크를 제안한다. **시맨틱 프로토타입**을 이용한 시각적-시맨틱 정렬을 통해 모델의 망각 문제를 해결하고, **재균형 손실 함수(Rebalancing Loss)**를 도입하여 클래스 불균형 및 과적합 문제를 해결하였다. 실험 결과, 제안 방법은 ZSD와 FSD를 포함한 모든 설정에서 기존 방법론보다 뛰어난 성능을 보였으며, 이는 실제 환경에서 데이터 가용성이 가변적인 객체 탐지 시스템을 구축하는 데 중요한 기반이 될 것으로 보인다.
