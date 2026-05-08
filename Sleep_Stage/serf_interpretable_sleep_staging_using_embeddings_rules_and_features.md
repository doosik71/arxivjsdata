# SERF: Interpretable Sleep Staging using Embeddings, Rules, and Features

Irfan Al-Hussaini and Cassie S. Mitchell (2022)

## 🧩 Problem to Solve

본 논문은 수면 단계 분류(Sleep Staging) 과정에서 발생하는 **딥러닝 모델의 불투명성(Black-box nature)**과 **수동 분석의 고비용 문제**를 동시에 해결하고자 한다.

현재 수면의 질을 평가하고 수면 장애를 진단하는 골드 표준은 전문가가 수면다원검사(Polysomnogram, PSG) 신호를 직접 분석하여 수면 단계를 판정하는 것이다. 그러나 이 과정은 전문가의 많은 시간과 노동력을 요구하며 비용이 매우 높다. 이를 자동화하기 위해 다양한 딥러닝 모델이 제안되었고 높은 정확도를 보였으나, 의료 분야의 특성상 모델이 왜 그러한 판단을 내렸는지에 대한 '해석 가능성(Interpretability)'이 부족하여 실제 임상 현장에 도입하는 데 큰 장애물이 되고 있다.

따라서 본 연구의 목표는 딥러닝의 높은 분류 성능을 유지하면서도, 임상 전문가가 이해할 수 있는 의미 있는 특징(Clinical features)을 통해 판단 근거를 제공하는 해석 가능한 수면 단계 분류 프레임워크인 **SERF**를 개발하는 것이다.

## ✨ Key Contributions

SERF의 핵심 아이디어는 **블랙박스 모델의 잠재 임베딩(Latent Embedding) 공간과 임상 전문가가 정의한 해석 가능한 특징(Expert-defined Features) 공간을 선형 맵(Linear Map)으로 연결**하는 것이다.

단순히 딥러닝 모델의 결과를 사후에 해석하는 것이 아니라, 모델이 학습한 고차원 임베딩을 임상적으로 의미 있는 특징 공간으로 투영(Projection)시킨다. 이렇게 변환된 '대표 특징(Representative Features)'을 의사결정 나무(Decision Tree)와 같은 단순하고 투명한 분류기에 입력함으로써, 최종 결과에 대해 "어떤 임상적 특징이 결정적인 역할을 했는지"를 명확하게 제시할 수 있게 한다.

## 📎 Related Works

수면 단계 분류를 위해 CNN, RNN, Attention mechanism, Graph Convolutional Networks 등 다양한 딥러닝 구조가 사용되어 왔다. 이러한 모델들은 높은 정확도를 보이지만, 내부 작동 원리를 알 수 없는 블랙박스 형태라는 한계가 있다.

기존의 해석 가능한 모델인 SLEEPER의 경우, 전문가 규칙 기반의 프로토타입(Prototype)을 사용했다. 하지만 SERF는 다음과 같은 차별점을 갖는다.

1. **데이터 표현의 효율성**: 이진 규칙(Binary rules) 대신 원시 특징 값(Raw feature values)을 사용하여 대표 특징의 차원을 낮추고 행렬 크기를 줄였다.
2. **학습 방식의 간소화**: 코사인 유사도를 이용한 복잡한 프로토타입 학습 대신, Ridge Regression을 통한 선형 맵을 학습하여 모델 크기를 줄이고 추론 속도를 높였다.
3. **직관적인 기준 제시**: 단순한 유사도 지수가 아니라, 의사결정 나무의 노드를 통해 구체적인 특징 값의 임계치(Cutoff value)를 제공함으로써 임상적 해석력을 높였다.

## 🛠️ Methodology

SERF의 전체 파이프라인은 크게 네 단계로 구성된다.

### 1. Latent Embedding (잠재 임베딩 추출)

다채널 PSG 신호를 입력받아 복잡한 패턴을 캡처하기 위해 CNN-LSTM 하이브리드 구조를 사용한다.

- **CNN**: 3개의 합성곱 계층으로 구성되며, 첫 번째 계층은 1초 세그먼트(커널 크기 201)를 추출하고 이후 계층은 커널 크기 11을 사용하여 특징을 추출한다.
- **Bi-LSTM**: CNN의 출력물을 입력받아 에포크(Epoch) 간의 시간적 관계를 캡처한다. 256개의 은닉 상태를 가진 양방향 LSTM을 통해 각 에포크당 512차원의 잠재 임베딩 $f(\mathbf{x}_i) \in \mathbb{R}^{512}$를 생성한다.
- **학습**: 소프트맥스(Softmax) 함수와 교차 엔트로피 손실 함수(Cross-entropy loss)를 사용하여 수면 단계를 예측하도록 학습시킨다.
  $$\mathcal{L}(\mathbf{y}_i, \hat{\mathbf{y}}_i) = -\sum_{j=1}^{5} y_i[j] \log(\hat{y}_i[j])$$

### 2. Expert Defined Features (전문가 정의 특징 추출)

AASM(American Academy of Sleep Medicine) 매뉴얼과 전문가의 제안을 바탕으로 임상적으로 의미 있는 특징들을 추출한다.

- **주요 특징**: Sleep spindles(N2 특징), Slow-wave sleep(N3 특징), 주파수 대역별(Delta, Theta, Alpha, Beta) 전력 스펙트럼 밀도(PSD), 진폭(Amplitude), 통계적 분포(Mean, Variance, Kurtosis, Skew) 등.
- **특징 선택**: ANOVA 테스트를 통해 변별력이 높은 상위 90%의 특징만을 선택하여 특징 행렬 $\mathbf{s}(\mathbf{X}) \in \mathbb{R}^{N \times M}$을 구성한다.

### 3. Linear Map (선형 맵 학습)

딥러닝의 임베딩 공간과 전문가 특징 공간을 연결하기 위해 Ridge Regression을 사용하여 선형 변환 행렬 $\mathbf{T}$를 학습한다.
$$\min_{\mathbf{T}} ||f(\mathbf{X}) - \mathbf{s}(\mathbf{X})\mathbf{T}||_2^2 + ||\mathbf{T}||_2^2$$
이 과정은 블랙박스 임베딩 $f(\mathbf{X})$가 전문가 특징 $\mathbf{s}(\mathbf{X})$의 선형 조합으로 표현될 수 있도록 맵핑하는 과정이다.

### 4. Representative Features & Classification (대표 특징 및 분류)

학습된 선형 맵 $\mathbf{T}$를 이용하여 새로운 입력 데이터의 임베딩을 대표 특징 공간으로 투영한다.
$$\mathbf{S} = f(\mathbf{x}_j)\mathbf{T}^T$$
이렇게 생성된 대표 특징 행렬 $\mathbf{S}$를 얕은 의사결정 나무(Shallow Decision Tree)나 Gradient Boosted Trees(GBT)와 같은 해석 가능한 분류기에 입력하여 최종 수면 단계를 예측한다.

## 📊 Results

### 실험 설정

- **데이터셋**: PhysioNet EDFx (197명, 4채널), ISRUC (100명, 9채널)
- **비교 대상**: CNN-LSTM (블랙박스), U-Time (SOTA 블랙박스), SLEEPER (해석 가능 모델), 단순 전문가 특징 기반 모델
- **평가 지표**: Accuracy, ROC-AUC, Cohen's $\kappa$, Macro F1

### 정량적 결과

ISRUC 데이터셋 기준으로 SERF-XG(XGBoost 사용) 모델은 $\kappa = 0.766$, AUC-ROC $= 0.870$을 기록하였다. 이는 최신 블랙박스 모델인 U-Time과 비교했을 때 성능 차이가 2% 이내로 매우 근접한 수치이며, 기존의 해석 가능 모델인 SLEEPER보다는 확연히 높은 성능을 보였다.

### 정성적 결과 및 해석

- **채널 수의 영향**: 9개 채널을 가진 ISRUC에서 4개 채널인 EDFx보다 성능이 높게 나타났다. 이는 AASM 가이드라인이 다채널 정보를 활용하므로, 입력 채널이 많을수록 해석 가능한 특징 추출이 더 정확해짐을 시사한다.
- **특징 중요도**: SHAP 값 분석 결과, N3 단계를 구분하는 데 'Slow Wave' 특징이 결정적인 역할을 했으며, REM 단계를 구분하는 데는 'Spindle'의 부재가 중요하게 작용했음이 확인되었다. 이는 실제 AASM 가이드라인과 일치한다.

## 🧠 Insights & Discussion

본 논문은 딥러닝의 고성능과 임상적 해석 가능성 사이의 트레이드-오프(Trade-off)를 성공적으로 완화하였다. 특히 의사결정 나무를 통해 "Slow Waves $\ge 0.01$일 때 N3일 확률이 높다"와 같이 구체적인 수치적 기준을 제시할 수 있다는 점은 의료진이 모델의 판단을 신뢰하고 검증하는 데 매우 유용하다.

다만, 본 연구는 선형 맵(Linear Map)을 통해 두 공간을 연결하는데, 임베딩 공간과 특징 공간의 관계가 비선형적일 경우 정보 손실이 발생할 가능성이 있다. 또한, 사용된 데이터셋의 규모가 상대적으로 작아 더 다양한 환자 군에 대한 일반화 성능 검증이 추가로 필요할 것으로 보인다.

## 📌 TL;DR

SERF는 딥러닝(CNN-LSTM)의 강력한 특징 추출 능력과 AASM 기반의 임상 전문가 지식을 결합한 해석 가능한 수면 단계 분류 프레임워크이다. 딥러닝 임베딩을 전문가 특징 공간으로 투영하는 선형 맵을 도입하여, 블랙박스 모델에 근접한 정확도를 유지하면서도 의사결정 나무를 통해 임상적으로 납득 가능한 판단 근거를 제공한다. 이 연구는 AI 모델의 투명성이 필수적인 의료 진단 보조 시스템 설계에 중요한 방법론을 제시한다.
