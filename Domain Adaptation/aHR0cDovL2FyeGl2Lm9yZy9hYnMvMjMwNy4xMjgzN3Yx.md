# EPIC-KITCHENS-100 Unsupervised Domain Adaptation Challenge: Mixed Sequences Prediction

Amirshayan Nasirimajd, Simone Alberto Peirone, Chiara Plizzari, Barbara Caputo (2023)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 문제는 1인칭 시점 행동 인식(Egocentric Action Recognition, EAR)에서 발생하는 도메인 시프트(Domain Shift) 현상이다. 도메인 시프트란 학습 데이터(Source domain)와 테스트 데이터(Target domain) 사이의 시각적 외형이나 맥락의 차이로 인해 데이터 분포가 달라지는 것을 의미하며, 이는 모델의 성능을 크게 저하시키는 원인이 된다.

특히 본 연구는 Unsupervised Domain Adaptation (UDA) 설정에 집중한다. UDA 설정에서는 타겟 도메인의 레이블이 없는 상태에서 타겟 데이터만을 이용하여 모델을 적응시켜야 하므로, 소스 도메인에서 학습한 지식을 타겟 도메인으로 효과적으로 전이시키는 것이 핵심 목표이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **행동이 수행되는 순서(Temporal Sequence)는 소스 도메인과 타겟 도메인 간에 유사하다**는 직관에서 출발한다. 예를 들어, 냉장고에서 우유를 꺼내는 동작은 환경이나 사람에 관계없이 '냉장고 열기 $\rightarrow$ 우유 집기 $\rightarrow$ 우유 놓기'라는 일관된 순서를 가진다.

이를 위해 저자들은 소스 도메인의 행동 시퀀스 중 일부를 동일한 레이블을 가진 타겟 도메인의 샘플로 교체하여 **혼합 시퀀스(Mixed Sequence)**를 생성하고, 이를 통해 모델이 도메인에 구애받지 않는 공통적인 행동 패턴을 학습하도록 유도하였다. 또한, 언어 모델(Language Model)과 공생 행렬(Co-occurrence Matrix)을 도입하여 예측 결과의 정밀도를 높였다.

## 📎 Related Works

본 연구는 기존의 여러 방법론을 결합하고 확장하였다.

- **Temporal Context Reasoning**: MTCN [5]의 연구를 참고하여 행동 간의 시간적 관계를 모델링하는 Transformer 구조와 Masked Language Model (MLM) 접근 방식을 채택하였다.
- **Domain Adaptation**: Gradient Reversal Layer (GRL) [4]를 활용한 도메인 적대적 학습(Domain Adversarial Learning)을 통해 도메인 불변 특징(Domain-agnostic features)을 추출하고자 하였다.
- **Action Refinement**: 기존 연구 [1]에서 제안된 Verb-Noun 공생 행렬을 사용하여 발생 가능성이 낮은 행동 조합을 필터링하는 기법을 도입하였다.

기존 방식들이 주로 개별 프레임이나 단일 클립의 특징 정렬에 집중했다면, 본 논문은 **시퀀스 수준의 구조적 유사성**을 활용하여 UDA 문제를 해결하려 했다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. 혼합 시퀀스 생성 (Mixed Sequence Generation)

타겟 도메인의 레이블이 없으므로, 먼저 신뢰도 임계값 $\lambda$를 기준으로 pseudo-labeling을 수행한다. 특정 샘플의 예측 신뢰도는 동사($p_v$)와 명사($p_n$) 예측 확률의 평균인 $(p_v + p_n)/2$로 정의하며, 이 값이 $\lambda$ 이상인 샘플만 사용한다.

소스 도메인의 행동 시퀀스 $S_i = \{s_{-w/2}, \dots, s_0, \dots, s_{w/2}\}$가 주어졌을 때, 중앙 행동 $s_0$를 제외한 나머지 행동 중 일부를 동일한 pseudo-label을 가진 타겟 도메인의 샘플 $t_i$로 무작위 교체한다. 이렇게 생성된 $\tilde{S}_i = \{s_{-w/2}, \dots, t_i, \dots, s_{w/2}\}$를 통해 모델을 학습시켜, 타겟 도메인의 정보가 포함된 상태에서 중앙 행동의 레이블을 예측하게 한다.

### 2. 시퀀스 예측기 (Sequence Predictor)

전체적인 구조는 Transformer 모델을 기반으로 한다.

- **입력 및 임베딩**: RGB, Optical Flow, Audio 세 가지 모달리티의 특징을 결합하여 저차원 $D$로 투영한 후, 학습 가능한 위치 임베딩(Positional Embedding)을 추가한다. 또한, 중앙 행동의 동사와 명사를 예측하기 위한 두 개의 특수 토큰을 추가하여 $X_e$를 구성한다.
- **Transformer Encoding**: Transformer 인코더 $f(\cdot)$를 통해 시퀀스 내의 주변 행동들이 서로 정보를 교환하게 하며, 출력은 $Z = f(X_e)$가 된다.
- **분류 및 손실 함수**: 동사와 명사를 예측하는 두 개의 분류 헤드를 사용한다.
  - **Mixed Sequence Prediction Loss ($L_{MS}$)**: 중앙 행동뿐만 아니라 시퀀스 내 모든 행동의 레이블을 예측하는 Cross Entropy 손실을 사용한다.
  - **Domain Classifier Loss ($L_{DC}$)**: GRL을 포함한 도메인 분류기를 통해, 네트워크가 소스와 타겟을 구분하지 못하는 도메인 불변 특징을 학습하도록 강제한다.
  - 최종 손실 함수는 $L = L_{MS} + L_{DC}$ 형태로 구성된다.

### 3. 언어 모델 (Language Model)

MLM(Masked Language Model) 방식을 사용하여 소스 데이터로부터 행동 간의 고수준 의존성을 학습한다. 추론 시에는 시퀀스 예측기가 예측한 상위 $k$개의 결과로 가능한 모든 시퀀스를 생성하고, 언어 모델이 판단하기에 가장 확률이 높은 시퀀스를 선택한다. 최종 예측값 $y_{final}$은 다음과 같이 선형 결합으로 계산된다.
$$y_{final} = (1-\beta)y_{MS} + \beta y_{LM}$$
여기서 $\beta$는 언어 모델의 반영 비중을 조절하는 계수이다.

### 4. 공생 행렬 (Co-Occurrence Matrix)

소스 도메인에서 동사와 명사 레이블이 함께 나타난 횟수를 기록한 행렬 $M_{CO}$를 생성한다. 테스트 시, 소스 데이터에서 한 번도 관찰되지 않은 동사-명사 조합의 예측 확률은 $0.01$을 곱해 대폭 낮춤으로써 비현실적인 예측을 제거한다.

## 📊 Results

### 실험 설정

- **데이터셋**: EPIC-Kitchens-100 UDA split.
- **특징 추출**: TBN (Temporal Binding Network) 특징 사용 (RGB, Flow, Audio).
- **하이퍼파라미터**: $\lambda = 0.75$, $\beta = 0.25$, 시퀀스 윈도우 크기 $w = 5$.
- **백본 모델**: TBN, TSM, SlowFast 등을 사용하고 최종적으로 앙상블 기법을 적용하였다.

### 주요 결과

- **Ablation Study**: 표 1에 따르면, 단일 샘플 분류(Baseline)에서 시작하여 시퀀스 예측 $\rightarrow$ 혼합 시퀀스 적용($L_{MS}$) $\rightarrow$ 도메인 분류기($L_{DC}$) $\rightarrow$ 언어 모델(LM) $\rightarrow$ 공생 행렬($M_{CO}$)을 순차적으로 추가할수록 Action 정확도가 $20.93\%$에서 $25.46\%$까지 지속적으로 향상됨을 확인하였다.
- **시퀀스 길이 및 교체 횟수**: 윈도우 크기 $w=5$일 때 가장 좋은 성능을 보였으며, 시퀀스 내에서 최소 1개 이상의 샘플을 타겟 샘플로 교체했을 때 성능 향상이 뚜렷했다.
- **최종 성능**: 다양한 백본을 앙상블하고 LM과 $M_{CO}$를 적용한 결과, 공식 리더보드에서 Verb 부문 2위, Noun 및 Action 부문 4위를 기록하였다.

## 🧠 Insights & Discussion

본 논문은 도메인 시프트라는 어려운 문제 앞에서, 데이터의 외형적 특징보다는 **행동의 순서라는 구조적 불변성(Structural Invariance)**에 집중하여 유의미한 성능 향상을 거두었다. 특히 타겟 도메인의 pseudo-label을 활용해 소스 시퀀스에 직접 주입하는 '혼합 시퀀스' 방식은 모델이 두 도메인의 공통적인 컨텍스트를 학습하게 만드는 효과적인 전략으로 판단된다.

다만, 본 연구는 행동 시퀀스가 도메인 간에 거의 동일하게 유지된다는 강한 가정에 의존하고 있다. 실제 환경에서는 문화적 차이나 주방 구조에 따라 행동 순서가 달라질 수 있으며, 이러한 경우에는 본 방법론의 효과가 감소할 가능성이 있다. 또한, pseudo-label의 정확도가 초기 성능에 큰 영향을 미치므로, 낮은 신뢰도의 레이블이 혼입될 경우 학습에 노이즈로 작용할 수 있다는 한계가 있다.

## 📌 TL;DR

본 논문은 1인칭 시점 행동 인식의 UDA 문제를 해결하기 위해, 소스와 타겟 도메인의 샘플을 섞은 **혼합 시퀀스(Mixed Sequence)**를 생성하여 Transformer로 학습시키는 방법을 제안하였다. 여기에 언어 모델과 동사-명사 공생 행렬을 통한 후처리를 더해 도메인 시프트를 극복하였으며, EPIC-Kitchens-100 챌린지 리더보드 상위권에 진입하는 성과를 거두었다. 이 연구는 시계열적 맥락 정보가 도메인 적응에 강력한 도구가 될 수 있음을 시사한다.
