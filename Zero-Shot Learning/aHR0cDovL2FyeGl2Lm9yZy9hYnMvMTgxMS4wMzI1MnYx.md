# Model Selection for Generalized Zero-shot Learning

Hongguang Zhang and Piotr Koniusz (2018)

## 🧩 Problem to Solve

본 논문은 Generalized Zero-shot Learning (GZSL)에서 발생하는 데이터 불균형 문제를 해결하고자 한다. 일반적인 Zero-shot Learning (ZSL)은 테스트 단계에서 오직 학습 시 보지 못한 'unseen classes'만 분류하면 되지만, GZSL은 테스트 데이터셋에 'seen classes'와 'unseen classes'가 모두 포함되어 있다.

이때 발생하는 핵심 문제는 학습 데이터가 seen classes에만 존재하기 때문에, 분류기가 테스트 샘플을 접했을 때 이를 unseen class보다는 seen class로 편향되게 예측하는 경향이 있다는 점이다. 즉, 데이터 분포의 강한 불균형으로 인해 unseen class에 속하는 샘플이 seen class로 오분류되는 문제가 발생하며, 이는 전체적인 정확도를 떨어뜨리는 주요 원인이 된다. 본 연구의 목표는 이러한 불균형의 부정적인 영향을 줄여 seen class와 unseen class 모두에 대해 높은 분류 성능을 달성하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 GZSL 문제를 하나의 거대한 분류 작업으로 처리하지 않고, **두 개의 서로 다른 분류 작업(disjoint classification tasks)으로 분리하여 처리하는 '모델 선택(Model Selection)' 메커니즘**을 도입하는 것이다.

구체적으로, Generative Adversarial Network (GAN)를 통해 생성된 unseen class의 보조 데이터(auxiliary datapoints)와 실제 seen class의 데이터를 활용하여, 입력 데이터가 seen class에 속하는지 혹은 unseen class에 속하는지를 먼저 판단하는 '선택기(selector)'를 학습시킨다. 이후 선택기의 결과에 따라 seen class 전용 모델 또는 unseen class 전용 모델 중 하나를 선택하여 최종 클래스를 예측함으로써, seen/unseen 간의 간섭을 줄이고 편향 문제를 완화한다.

## 📎 Related Works

ZSL은 기본적으로 seen class에서 학습된 지식을 속성(attribute) 벡터 등을 통해 unseen class로 전이하는 Transfer Learning의 일종이다. 기존 연구들은 주로 다음과 같은 접근 방식을 취했다.

- **선형 매핑 기반 방식:** ALE, ESZSL, SJE 등은 특징(feature) 벡터와 속성 벡터 간의 선형 관계를 학습하여 분류를 수행한다.
- **비선형 및 커널 방식:** ZSKL은 비선형 커널 방법을 통해 투영 행렬의 일관성 제약을 적용한다.
- **생성 모델 기반 방식:** 최근에는 f-CLSWGAN과 같이 조건부 Wasserstein GAN (WGAN)을 사용하여 unseen class의 보조 데이터를 생성하고, 이를 통해 일반적인 Softmax 분류기를 학습시키는 방식이 SOTA(State-of-the-art) 성능을 보였다.

본 논문은 f-CLSWGAN과 같이 GAN을 통해 보조 데이터를 생성하는 아이디어를 계승하지만, 기존 방식이 생성 데이터와 실제 데이터를 단순히 합쳐서 하나의 분류기를 학습시켰던 것과 달리, 이를 분리하여 모델 선택 과정에 활용한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. 전체 시스템 구조 및 구성 요소

본 모델은 크게 네 가지 분류기($M_{sel}, M_s, M_u, M_t$)를 사용한다.

- $M_{sel}$: 입력 데이터 $x$가 seen class에 속하는지($1$) 아니면 unseen class에 속하는지($-1$)를 판단하는 선형 SVM 분류기이다.
- $M_s$: seen classes ($C_s$)만을 분류하기 위한 모델이다.
- $M_u$: GAN으로 생성된 보조 데이터를 통해 unseen classes ($C_u$)만을 분류하기 위한 모델이다.
- $M_t$: seen과 unseen 모든 클래스 ($C_s \cup C_u$)를 동시에 분류하기 위한 모델이다.

각 분류기의 출력 함수는 다음과 같이 정의된다.
$$g_s(x) = W_s^T x + b_s$$
$$g_u(x) = W_u^T x + b_u$$
$$g_t(x) = W_t^T x + b_t$$

### 2. 모델 선택 메커니즘 (세 가지 변형)

본 논문은 선택기 $M_{sel}$의 출력 $s(x) = w_{sel}^T x + b_{sel}$을 활용하여 최종 예측을 수행하는 세 가지 방식을 제안한다.

#### (1) ModelSel-2Way (Hard Switching)

가장 단순한 형태로, $M_{sel}$의 출력값의 부호에 따라 모델을 완전히 선택한다.
$$f(x, s(x)) = \begin{cases} g_s(x), & \text{if } s \ge 0 \\ g_u(x), & \text{otherwise} \end{cases}$$

#### (2) ModelSel-2Way-SA (Soft Assignment)

Hard switching에서 발생하는 결정 경계 근처의 양자화 오류(quantization errors)를 줄이기 위해 시그모이드(Sigmoid) 함수를 사용하여 가중 합으로 계산한다.
먼저 $x$가 seen class일 확률 $p_s(x)$를 다음과 같이 구한다.
$$p_s(x) = \frac{1}{1 + e^{-\sigma s(x)}}$$
최종 예측값은 두 모델의 가중 합으로 계산된다.
$$f(x) = p_s(x) \cdot g_s(x) + (1 - p_s(x)) \cdot g_u(x)$$
여기서 $\sigma$는 시그모이드 함수의 기울기를 조절하는 파라미터이다.

#### (3) ModelSel-3Way (Masking & Correction)

$M_s$와 $M_u$가 각 도메인에서는 강하지만 경계 지역에서 취약하다는 점에 착안하여, 전체 클래스를 학습한 $M_t$를 보조적으로 사용한다. $M_t$의 출력을 마스크처럼 사용하여 잘못된 예측을 교정한다.
$$f(x, s(x)) = \max \begin{pmatrix} c \cdot g_t(x) + g_s(x) - o_s & \text{if } s \ge 0 \\ c \cdot g_t(x) + g_u(x) - o_u & \text{if } s < 0 \\ g_t(x) & \end{pmatrix}$$
여기서 $c$는 $M_t$의 중요도를, $o_s$와 $o_u$는 각 모델의 오프셋을 조정하는 하이퍼파라미터이다. 결과적으로 경계 지역에서는 $g_t(x)$가 선택될 가능성이 높아진다.

## 📊 Results

### 1. 실험 설정

- **데이터셋:** APY, AWA1, AWA2, FLO 등 4개의 공개 데이터셋을 사용하였다.
- **평가 지표:** seen class의 정확도($Acc^S$)와 unseen class의 정확도($Acc^U$)의 조화 평균(Harmonic Mean, $H$)을 최종 점수로 사용한다.
$$H = \frac{2 \cdot Acc^S \cdot Acc^U}{Acc^S + Acc^U}$$
- **학습 절차:** Adam 옵티마이저를 사용하였으며, learning rate는 $1e-4$, 배치 크기는 $60$, 총 $50$ epoch 동안 학습하였다.

### 2. 결과 분석

실험 결과, 제안된 ModelSel 방식들이 기존의 f-CLSWGAN 및 다른 베이스라인 모델들보다 우수한 성능을 보였다.

- **정량적 결과:** ModelSel-3Way는 AWA1, AWA2, FLO 데이터셋에서 f-CLSWGAN 대비 각각 $2.8\%$, $3.6\%$, $0.8\%$ 높은 정확도를 기록하였다.
- **특이 사항:** 특히 APY 데이터셋에서는 ModelSel-2Way-SA가 ZSKL의 $20.5\%$에서 $42.3\%$로 비약적인 성능 향상을 보였다.
- **파라미터 영향:** $\sigma$ 값의 변화에 따른 실험을 통해, 시그모이드를 통한 soft assignment가 단순 hard switching보다 성능 향상에 기여함을 확인하였다.

## 🧠 Insights & Discussion

본 논문은 GZSL의 고질적인 문제인 'seen class로의 편향'을 해결하기 위해 모델을 분리하고 선택하는 전략이 유효함을 증명하였다. 특히, 단일 분류기가 모든 클래스를 처리하게 하는 대신, 각 영역의 전문가 모델($M_s, M_u$)을 두고 이를 조율하는 선택기를 둠으로써 데이터 불균형의 영향을 효과적으로 차단하였다.

또한, ModelSel-3Way의 결과는 특정 도메인에 특화된 모델과 전체 도메인을 아우르는 모델을 적절히 결합했을 때, 결정 경계 근처의 모호함을 해결할 수 있음을 시사한다. 다만, 본 논문에서는 GAN으로 생성된 보조 데이터의 품질이 전체 성능에 미치는 영향이나, 생성된 데이터의 분포가 실제 unseen 데이터의 분포와 얼마나 유사한지에 대한 심층적인 분석은 명시적으로 다루지 않았다.

## 📌 TL;DR

본 연구는 Generalized Zero-Shot Learning에서 발생하는 seen class 편향 문제를 해결하기 위해, seen/unseen 전용 분류기를 각각 두고 이를 선택하는 **Model Selection 메커니즘**을 제안하였다. 특히 GAN 생성 데이터를 활용해 선택기와 전용 모델들을 학습시킨 ModelSel-3Way 방식은 여러 벤치마크 데이터셋에서 SOTA 성능을 달성하였다. 이 연구는 GZSL 문제를 개별 클래스 분류 작업으로 분해하여 접근하는 방식이 실제 적용 및 향후 연구에서 효율적인 대안이 될 수 있음을 보여준다.
