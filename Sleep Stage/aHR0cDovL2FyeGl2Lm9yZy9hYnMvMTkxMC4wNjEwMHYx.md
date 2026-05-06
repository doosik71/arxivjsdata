# SLEEPER: interpretable Sleep staging via Prototypes from Expert Rules

Irfan Al-Hussaini, Cao Xiao, M. Brandon Westover, Jimeng Sun (2019)

## 🧩 Problem to Solve

수면 단계 측정(Sleep staging)은 불면증, 기면증, 수면 무호흡증과 같은 수면 장애를 진단하는 데 있어 필수적인 작업이다. 일반적으로 신경과 전문의는 다원수면검사(Polysomnogram, PSG)의 다변량 시계열 데이터를 시각적으로 분석하여 깨어 있음(Wake), REM 수면, 그리고 비-REM 수면(N1, N2, N3) 단계로 구분한다. 하지만 이 과정은 매우 노동 집약적이며, 전문가가 한 환자의 하룻밤 기록을 주석 처리하는 데에만 수 시간이 소요된다.

최근 딥러닝 모델들이 수면 단계 자동 측정에서 최첨단 성능을 보여주고 있으나, 이러한 모델들은 내부 작동 원리를 알 수 없는 '블랙박스(Black-box)' 형태라는 한계가 있다. 임상 현장의 의료진은 데이터의 노이즈나 예상치 못한 편향을 피하기 위해 각 분류 결과의 근거를 이해해야 하므로, 해석 가능성(Interpretability)은 실무 도입의 핵심적인 제약 사항이 된다. 반면, 기존의 AASM(American Academy of Sleep Medicine) 수면 측정 매뉴얼은 해석 가능하지만, 기준이 모호하고 계산적으로 정밀하게 구현하기 어렵다는 단점이 있다. 따라서 본 논문의 목표는 딥러닝 모델의 정확도와 수면 측정 매뉴얼의 해석 가능성을 동시에 갖춘 수면 단계 측정 모델을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Prototype Learning** 프레임워크를 사용하여 딥러닝의 고차원 특징 표현(Embedding)과 전문가가 정의한 규칙(Expert Rules)을 결합하는 것이다.

단순히 딥러닝 모델의 결과를 설명하려 하는 사후 분석(Post-hoc analysis) 방식이 아니라, 전문가의 규칙을 기반으로 한 **프로토타입(Prototype)**을 생성하고, 입력 데이터와 이 프로토타입 간의 유사도를 계산하여 최종 결정을 내리는 구조를 설계하였다. 이를 통해 최종 분류기는 얕은 의사결정 나무(Shallow Decision Tree)와 같이 매우 단순하고 해석 가능한 모델을 사용하면서도, 내부적으로는 CNN이 추출한 풍부한 특징 정보를 활용함으로써 높은 정확도를 유지할 수 있게 한다.

## 📎 Related Works

기존의 수면 단계 자동 측정 연구들은 주로 Convolutional Neural Networks (CNN), Recurrent Neural Networks (RNN), Deep Belief Nets 등을 활용하여 높은 분류 성능을 달성하는 데 집중하였다. 하지만 이러한 접근 방식은 앞서 언급한 바와 같이 투명성이 부족하여 임상 적용에 한계가 있다.

또한, 사례 기반 추론(Case-based reasoning)에서 영감을 받은 Prototype Learning 연구들이 존재하지만, 기존의 딥러닝 기반 프로토타입 모델들은 최종 모델이 여전히 복잡한 신경망인 경우가 많아 완전한 해석 가능성을 제공하지 못하며, 수면 측정 매뉴얼과 같은 도메인 지식(Domain Knowledge)을 직접적으로 통합하지 못한다는 한계가 있다. SLEEPER는 전문가 규칙을 통해 프로토타입을 정의함으로써 이 두 가지 문제를 동시에 해결하고자 한다.

## 🛠️ Methodology

### 1. 전체 파이프라인

SLEEPER의 전체 구조는 **신호 임베딩(Signal Embedding) $\rightarrow$ 전문가 규칙 임베딩(Expert Rule Embedding) $\rightarrow$ 프로토타입 학습 및 유사도 매칭(Prototype Learning & Relevance Matching) $\rightarrow$ 해석 가능한 분류기(Interpretable Classifier)**의 순서로 구성된다.

### 2. 주요 구성 요소 및 상세 설명

#### (1) Signal Embedding Module

먼저 raw PSG 데이터를 입력받아 수면 단계를 예측하는 CNN 모델을 학습시킨다. 이 네트워크는 3개의 Convolutional Layer(ReLU 활성화 함수 및 Max Pooling 포함)로 구성되며, 커널 크기를 201로 설정하여 1초 단위의 세그먼트 특징을 추출한다. 학습이 완료된 후, 마지막 Fully-connected layer를 제거하여 각 epoch $X_n$에 대한 고차원 잠재 표현(Latent representation) $h(X_n) \in \mathbb{R}^{2,496}$을 얻는다. CNN 학습 시 사용되는 손실 함수는 다음과 같은 Cross Entropy Loss이다.
$$L(y_i, s_i) = -\sum_{j=1}^{5} y_i[j] \log(s_i[j])$$
여기서 $y_i$는 실제 라벨, $s_i$는 예측 확률이다.

#### (2) Expert Rule Module

AASM 매뉴얼과 전문가의 제안을 바탕으로 240개의 규칙 $R'$을 정의한다. 규칙들은 수면 방추사(Sleep spindles), 서파 수면(Slow wave sleep, SWS), 주파수 대역(Delta, Theta, Alpha, Beta), 진폭(Amplitude), 첨도(Kurtosis) 등의 특징을 기반으로 하며, 각 채널별로 백분위수(Percentile) 기준에 따라 이진 벡터(Binary vector)로 인코딩된다. 이후 ANOVA 테스트를 통해 변별력이 높은 96개의 핵심 규칙 $R = \{r_1, \dots, r_{96}\}$을 선정하여 이진 규칙 할당 행렬 $R(X) \in \mathbb{R}^{N \times 96}$을 생성한다.

#### (3) Prototype Learning Module

각 규칙 $r_j$에 대응하는 프로토타입 $p_j \in \mathbb{R}^{2,496}$를 생성한다. 프로토타입은 해당 규칙을 만족하는 모든 epoch들의 잠재 임베딩의 합으로 정의된다.
$$P = h'(X)^T R(X)$$
여기서 $h'(X)$는 열 정규화(Column normalization)된 임베딩 행렬이다. 이렇게 생성된 프로토타입 $P$는 특정 전문가 규칙이 잠재 공간(Latent space)에서 어떻게 표현되는지를 나타내는 대표 벡터가 된다.

#### (4) Relevance Matching 및 최종 분류

새로운 데이터 $X_{test}$가 들어오면 CNN을 통해 임베딩 $h(X_{test})$를 추출하고, 이를 각 프로토타입 $p_j$와 코사인 유사도(Cosine Similarity)를 통해 비교한다.
$$c_{i,j} = \frac{h(X_i) \cdot p_j}{\|h(X_i)\| \|p_j\|}$$
이 유사도 점수 $C(h(X)|P) \in \mathbb{R}^{N \times 96}$가 최종 분류기의 입력 특징으로 사용된다. 분류기로는 얕은 의사결정 나무(Decision Tree)나 로지스틱 회귀(Logistic Regression)를 사용하여 최종 수면 단계를 예측한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: MGH(2,000명 대상) 및 ISRUC(100명 대상) 데이터셋을 사용하였다.
- **비교 대상**:
  - Black-box CNN (성능 상한선)
  - Rules + Interpretable Classifier (프로토타입 없이 규칙만 사용한 경우)
  - Mimic Learning (RCNN의 Soft label을 학습한 GBT)
  - 기타 기존 수면 측정 알고리즘 (MODWT, LSTAR 등)
- **측정 지표**: Accuracy, ROC-AUC, Cohen's $\kappa$ 및 단계별 민감도(Sensitivity).

### 2. 정량적 결과

실험 결과, SLEEPER(특히 Decision Tree 사용 시)는 블랙박스 CNN 모델에 근접하는 성능을 보였다.

- **ROC-AUC**: MGH와 ISRUC 데이터셋 모두에서 약 84~86%를 기록하였다.
- **Cohen's $\kappa$**: ISRUC 데이터셋 기준 0.72를 기록하여 전문가 간의 일치도(0.78)에 근접하는 '상당한 일치(Substantial agreement)' 수준을 보였다.
- **규칙 기반 모델과의 비교**: 단순히 전문가 규칙만 사용하여 분류했을 때(Rule & DT)보다 SLEEPER-DT의 성능이 월등히 높았다. 이는 전문가 규칙을 딥러닝의 임베딩 공간으로 투영하여 유사도를 측정하는 방식이 훨씬 강력함을 시사한다.

### 3. 정성적 분석 및 해석

깊이 $D=5$인 의사결정 나무를 분석한 결과, 실제 AASM 매뉴얼의 기준과 일치하는 경로가 발견되었다. 예를 들어, 중앙 채널(Central Channels)에서 3초 이상의 서파(Slow Waves)가 발생했을 때 N3 단계로 분류하는 경로가 생성되었으며, 후두부(Occipital Region)의 Alpha 활동을 통해 Wake 단계를 구분하는 특성이 나타났다. 이는 모델이 임상적으로 유의미한 근거를 바탕으로 판단하고 있음을 보여준다.

## 🧠 Insights & Discussion

### 1. 강점

본 연구는 딥러닝의 강력한 특징 추출 능력과 전문가의 도메인 지식을 '프로토타입'이라는 매개체를 통해 성공적으로 결합하였다. 특히, 최종 모델로 얕은 의사결정 나무를 채택함으로써, 의료진이 "어떤 규칙의 프로토타입과 유사하여 이 단계로 분류되었는가"를 즉각적으로 확인할 수 있게 하여 임상적 신뢰도를 높였다.

### 2. 한계 및 비판적 해석

- **N1 단계 분류의 어려움**: 실험 결과, 모델과 인간 전문가 모두 N1 단계 분류에서 낮은 민감도를 보였다. 이는 N1과 N2의 특징이 매우 유사하기 때문으로 분석된다.
- **규칙 선정의 임의성**: 240개의 규칙 중 ANOVA를 통해 96개를 선정하였으나, 이 과정에서 수면 방추사(Sleep spindles)나 첨도(Kurtosis) 관련 규칙들이 제외되었다. 논문에서는 방추사가 다른 단계에서도 나타나 변별력이 낮기 때문이라고 설명하지만, 이는 프로토타입 방식이 특정 복잡한 패턴을 포착하는 데 한계가 있을 수 있음을 시사한다.
- **데이터셋 규모**: ISRUC 데이터셋의 경우 표본 수가 100명으로 적어 일반화 성능에 의문이 있을 수 있으나, MGH라는 대규모 데이터셋에서도 유사한 결과가 나온 점이 이를 보완한다.

## 📌 TL;DR

SLEEPER는 딥러닝(CNN)의 정확도와 전문가 규칙의 해석 가능성을 결합한 수면 단계 측정 프레임워크이다. 전문가 규칙을 CNN의 잠재 공간 상의 **프로토타입(Prototype)**으로 변환하고, 입력 데이터와의 유사도를 측정하여 얕은 의사결정 나무로 분류함으로써 **"정확하면서도 이유를 설명할 수 있는"** 모델을 구현하였다. 이 연구는 블랙박스 모델의 도입을 꺼리는 의료 현장에서 딥러닝 기술을 실무에 적용할 수 있는 중요한 방법론적 가이드를 제시한다.
