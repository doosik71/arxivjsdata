# Active Imitation Learning with Noisy Guidance

Kianté Brantley, Amr Sharaf, Hal Daumé III (2020)

## 🧩 Problem to Solve

본 논문은 구조적 예측(Structured Prediction) 문제에서 모방 학습(Imitation Learning)을 사용할 때 발생하는 과도한 전문가 쿼리 비용 문제를 해결하고자 한다. 

일반적인 모방 학습 알고리즘은 학습 과정에서 모든 상태(state)에 대해 전문가(expert)가 제공하는 최적의 행동(optimal action)에 접근할 수 있다고 가정한다. 그러나 실제 환경에서 전문가에게 매 단계마다 정답을 묻는 것은 비용과 시간이 매우 많이 소요되며, 때로는 불가능에 가깝다. 따라서 전문가의 개입을 최소화하면서도 전문가의 성능에 근접한 정책을 학습시키는 것이 본 연구의 핵심 목표이다.

## ✨ Key Contributions

본 논문은 전문가의 비싼 가이드 대신, 저렴하지만 품질이 낮은 **Noisy Heuristic**을 결합한 능동적 모방 학습 알고리즘인 **LEAQI(Learning to Query for Imitation)**를 제안한다.

핵심 아이디어는 전문가와 휴리스틱 간의 의견 일치 여부를 예측하는 **Difference Classifier**를 학습시키는 것이다. 모델은 다음과 같은 단계적 필터링을 통해 쿼리를 수행한다:
1. 현재 정책이 예측에 확신이 없는 경우에만 레이블을 요청한다(Active Learning).
2. 레이블이 필요하다고 판단되면, 우선 저렴한 휴리스틱의 의견을 확인한다.
3. Difference Classifier가 "전문가가 휴리스틱과 의견이 다를 것"이라고 예측하는 경우에만 최종적으로 전문가에게 쿼리를 보낸다.

또한, Difference Classifier가 "일치한다"고 잘못 예측하여 전문가에게 묻지 않는 경우(Type II error) 발생하는 편향(bias) 문제를 해결하기 위해 **Apple Tasting** 프레임워크를 도입하여 학습 데이터의 오염을 방지한다.

## 📎 Related Works

본 연구는 세 가지 주요 연구 분야를 기반으로 한다.
- **Learning to Search**: 구조적 예측 문제를 일련의 작은 분류 문제로 변환하여 해결하는 접근법이다. 특히 **DAgger** 알고리즘은 전문가의 정책을 모방하여 상태 분포의 불일치(distribution shift) 문제를 해결하지만, 모든 상태에서 전문가의 쿼리를 요구한다는 한계가 있다.
- **Active Learning**: 불확실성 기반의 샘플링(Uncertainty Sampling)을 통해 필요한 데이터만 선택적으로 레이블링하는 기법이다. 본 논문은 특히 약한 레이블러(weak labeler)와 강한 레이블러(strong labeler)를 동시에 사용하는 Zhang and Chaudhuri(2015)의 연구에서 영감을 받았다.
- **One-sided Feedback**: 예측이 "긍정"일 때만 실제 정답을 확인할 수 있는 환경에서의 학습 문제이다. **Apple Tasting**은 이러한 부분적 모니터링 게임(partial-monitoring games)에서 발생하는 편향을 줄이기 위해 제안된 프레임워크이다.

## 🛠️ Methodology

### 전체 파이프라인 및 시스템 구조
LEAQI는 기본적으로 DAgger의 구조를 따르되, 쿼리 단계에서 능동적 선택과 휴리스틱 가이드를 추가한 구조이다. 

1. **경로 생성**: 현재 학습된 정책 $\pi_i$를 사용하여 궤적(trajectory)을 생성한다.
2. **확신도 측정**: 각 상태 $s$에서 정책 $\pi_i$의 확신도(certainty)를 계산하여 쿼리 여부를 결정한다.
3. **차이 분류기 적용**: 쿼리가 결정되면 Difference Classifier $h_i$가 전문가와 휴리스틱 $\pi^h$의 일치 여부를 예측한다.
4. **최종 쿼리**: $h_i$가 '불일치'를 예측하거나 Apple Tasting 알고리즘에 의해 강제로 쿼리가 발생한 경우에만 전문가 $\pi^*$에게 정답을 묻는다. 그렇지 않으면 휴리스틱의 정답을 그대로 사용한다.
5. **업데이트**: 수집된 데이터로 정책 $\pi$를 업데이트하고, 전문가 쿼리가 발생한 데이터를 통해 Difference Classifier $h$를 업데이트한다.

### 주요 구성 요소 및 수식

**1. 정책 확신도(Policy Certainty) 측정**
정책 $\pi$가 각 행동에 대해 부여하는 점수 중, 가장 높은 점수와 두 번째로 높은 점수의 차이(margin)를 확신도로 정의한다.
$$\text{certainty}(\pi, s) = \max_{a} \pi(s, a) - \max_{a' \neq a} \pi(s, a')$$

**2. 샘플링 확률(Sampling Probability)**
확신도 $z$를 바탕으로 레이블을 요청할 확률 $\rho$를 다음과 같이 결정한다. 여기서 $b$는 샘플링의 공격성을 조절하는 하이퍼파라미터이다.
$$\rho = \frac{b}{b + z}$$

**3. Difference Classifier 및 학습 목표**
Difference Classifier $h$는 상태 $s$에서 전문가 $\pi^*$와 휴리스틱 $\pi^h$가 동일한 행동을 취할지 여부($d \in \{0, 1\}$)를 예측한다.
$$d = \mathbb{1}[\pi^*(s) = \pi^h(s)]$$

**4. Apple Tasting (STAP)**
Difference Classifier가 "일치한다(agree)"고 예측하면 전문가에게 묻지 않으므로, "불일치(disagree)"인데 "일치"라고 예측한 오류(Type II error)를 발견할 수 없다. 이를 해결하기 위해 STAP 알고리즘은 예측 결과와 상관없이 일정 확률로 전문가에게 쿼리를 보내어 "나쁜 사과(오류)"를 찾아낸다. 
이 확률은 과거의 실수 횟수 $m$과 샘플링 횟수 $t$를 이용하여 $\sqrt{(m+1)/t}$ 로 결정된다.

### 학습 절차 및 아키텍처
- **특징 추출**: BERT(English BERT, SciBERT, M-BERT)를 사용하여 단어 임베딩을 생성하며, 상태 $s_t$는 현재 단어 임베딩과 이전 행동의 one-hot 벡터를 결합하여 표현한다.
- **모델 구조**: BERT 특징 위에 단순한 선형 층(linear layer)을 쌓아 정책 $\pi$와 차이 분류기 $h$를 구현하며, 데이터 부족 문제를 해결하기 위해 BERT의 가중치는 고정하고 최종 층만 학습시킨다.

## 📊 Results

### 실험 설정
- **데이터셋 및 작업**: 
    1. English NER (CoNLL'03): 휴리스틱으로 가제티어(gazetteer) 기반 문자열 매칭 사용.
    2. English Scientific Keyphrase Extraction (SemEval 2017): 휴리스틱으로 비지도 학습 기반 모델 사용.
    3. Greek POS Tagging (UD): 휴리스틱으로 Wiktionary 기반 사전 사용.
- **평가 지표**: F-score (NER, Keyphrase), Accuracy (POS).
- **기준선(Baselines)**: Passive DAgger, ActiveDAgger(불확실성 기반 쿼리만 수행), DAgger+Feat(휴리스틱 결과를 특징으로 사용).

### 주요 결과
- **쿼리 효율성**: 모든 작업에서 LEAQI가 가장 적은 수의 전문가 쿼리를 사용하면서도, Passive DAgger나 ActiveDAgger와 비슷하거나 더 높은 성능을 달성하였다.
- **능동 학습의 효과 (Q1)**: ActiveDAgger가 Passive DAgger보다 훨씬 적은 쿼리로 빠르게 수렴함을 확인하여, 본 설정에서도 능동 학습이 유효함을 입증하였다.
- **Difference Classifier의 효과 (Q2)**: Apple Tasting이 없는 Difference Classifier(LEAQI+NoAT)는 초기 학습 속도는 빠르나 최종 성능이 매우 낮게 정체(plateau)되는 현상이 나타났다. 이는 Type II error로 인한 데이터 편향 때문이다.
- **Apple Tasting의 효과 (Q3)**: LEAQI는 Apple Tasting을 통해 편향을 제거함으로써, 쿼리 수를 획기적으로 줄이면서도 높은 최종 성능을 유지하였다.
- **휴리스틱 품질에 대한 강건성 (Q4)**: 무작위 레이블을 제공하는 휴리스틱(LEAQI+NoisyHeur)을 사용했을 때도, Difference Classifier가 이를 빠르게 학습하여 무시함으로써 ActiveDAgger 수준의 성능을 유지하였다.

## 🧠 Insights & Discussion

본 논문은 저렴한 휴리스틱과 정교한 차이 분류기를 결합함으로써 전문가의 개입을 최소화하는 효율적인 모방 학습 프레임워크를 제시하였다. 특히, 단순히 휴리스틱의 결과를 입력 특징(feature)으로 사용하는 것보다, 이를 정책 수준에서 선택적으로 활용하는 방식이 더 효과적임을 보였다.

**강점 및 시사점:**
- **편향 제어**: One-sided feedback 환경에서 발생할 수 있는 데이터 편향 문제를 Apple Tasting이라는 정교한 샘플링 전략으로 해결하였다.
- **범용성**: 서로 다른 성격의 휴리스틱(사전 기반, 비지도 모델 기반, 규칙 기반)과 서로 다른 언어(영어, 그리스어) 작업에서 일관된 성능 향상을 보였다.

**한계 및 비판적 해석:**
- **분류기 학습 비용**: 차이 분류기 $h$ 자체를 학습시키는 것이 때로는 구조적 예측 모델 $\pi$를 학습시키는 것만큼 어려울 수 있다. 특히 이진 분류와 같이 클래스가 매우 적은 경우에는 차이 분류기의 효용성이 떨어질 수 있다.
- **실제 비용의 단순화**: 본 논문은 쿼리의 개수를 비용으로 정의하였으나, 실제 전문가의 비용은 문맥을 읽는 고정 비용과 개별 단어를 레이블링하는 가변 비용이 복합적으로 작용하므로, 단순 쿼리 수 감소가 실제 비용 감소와 완벽히 일치하지 않을 수 있다.

## 📌 TL;DR

본 연구는 전문가 쿼리 비용이 높은 모방 학습 문제를 해결하기 위해, **저렴한 휴리스틱**과 **전문가-휴리스틱 간의 일치 여부를 예측하는 Difference Classifier**를 도입한 **LEAQI** 알고리즘을 제안한다. 특히 **Apple Tasting** 기법을 통해 휴리스틱 사용으로 인한 데이터 편향 문제를 해결하여, 전문가 쿼리 수를 획기적으로 줄이면서도 높은 성능을 유지하였다. 이 방법론은 규칙 기반 시스템이나 비지도 모델이 존재하는 다양한 구조적 예측 작업에 실제적으로 적용될 가능성이 매우 높다.