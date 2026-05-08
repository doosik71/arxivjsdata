# ELECTRA is a Zero-Shot Learner, Too

Shiwen Ni and Hung-Yu Kao (2022)

## 🧩 Problem to Solve

최근 자연어 처리(NLP) 분야에서는 '사전 학습 후 미세 조정(pre-train, fine-tune)' 방식보다 '사전 학습, 프롬프트, 예측(pre-train, prompt, and predict)' 패러다임이 퓨샷(few-shot) 및 제로샷(zero-shot) 학습에서 훨씬 뛰어난 성과를 거두고 있다. 특히 GPT-3와 같은 거대 언어 모델의 성공 이후 BERT나 RoBERTa와 같은 Masked Language Model(MLM) 기반의 프롬프트 학습 방법들이 널리 사용되어 왔다.

그러나 효율적인 사전 학습 판별 모델인 ELECTRA는 이러한 프롬프트 기반 학습 연구에서 상대적으로 간과되어 왔다. 본 논문은 ELECTRA의 핵심 메커니즘인 Replaced Token Detection(RTD)을 프롬프트 학습에 접목하여, ELECTRA를 제로샷 학습자로 활용함으로써 다양한 NLP 태스크에서 성능을 극대화하는 것을 목표로 한다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 ELECTRA의 사전 학습 목표인 RTD(치환된 토큰 탐지)를 다운스트림 태스크의 예측 메커니즘으로 직접 연결하는 것이다. 기존 MLM 기반 모델들이 마스킹된 토큰에 들어갈 단어를 예측하는 방식이었다면, 본 논문에서는 특정 라벨 단어가 '원래 데이터에 있던 것(original)'인지 아니면 '생성기에 의해 치환된 것(replaced)'인지를 판별하는 이진 분류 문제로 태스크를 재정의한다. 이를 통해 ELECTRA가 사전 학습 단계에서 습득한 판별 능력을 제로샷 시나리오에서 그대로 활용할 수 있도록 설계하였다.

## 📎 Related Works

기존의 사전 학습 언어 모델들은 BERT의 MLM 방식이나 GPT의 자기회귀(autoregressive) 방식 등을 통해 문맥적 표현을 학습해 왔다. RoBERTa는 BERT의 학습 절차를 최적화하였고, ALBERT는 파라미터 공유를 통해 효율성을 높였다. 이후 GPT-3의 등장으로 프롬프트를 통해 모델의 가중치 업데이트 없이 작업을 수행하는 제로샷/퓨샷 학습이 주목받기 시작했다.

기존의 프롬프트 학습 연구들은 주로 MLM 사전 학습 태스크를 기반으로 하며, 마스킹된 부분에 적절한 단어가 올 확률을 계산하는 방식을 취한다. 반면, 본 논문은 RTD라는 판별적(discriminative) 사전 학습 태스크를 기반으로 한 프롬프트 학습을 제안하며, 이는 모든 입력 토큰을 학습에 활용한다는 점에서 MLM보다 효율적이고 강력한 잠재력을 가지고 있다고 주장한다.

## 🛠️ Methodology

### 1. Replaced Token Detection (RTD) 사전 학습

ELECTRA는 생성기(Generator, $G$)와 판별기(Discriminator, $D$)라는 두 개의 네트워크를 사용한다.

- **생성기($G$):** MLM 방식으로 학습하며, 마스킹된 토큰 $[MASK]$ 위치에 올 가장 확률 높은 토큰 $x_t$를 생성한다.
  $$p_G(x_t|x) = \frac{\exp(e(x_t)^T h_G(x)_t)}{\sum_{x'} \exp(e(x')^T h_G(x)_t)}$$
- **판별기($D$):** 입력 시퀀스의 각 토큰이 원래의 토큰인지, 아니면 생성기에 의해 치환된 토큰인지를 판별하는 이진 분류를 수행한다.
  $$D(x,t) = \text{sigmoid}(w^T h_D(x)_t)$$

학습 과정에서 생성기가 토큰을 치환하여 오염된 문장 $x^{\text{corrupt}}$를 만들면, 판별기는 각 위치 $t$에 대해 다음과 같은 손실 함수를 통해 최적화된다.
$$\mathcal{L}_{\text{Disc}}(x, \theta_D) = \mathbb{E} \left( \sum_{t=1}^{n} -\mathbb{1}(x^{\text{corrupt}}_t = x_t) \log D(x^{\text{corrupt}}, t) - \mathbb{1}(x^{\text{corrupt}}_t \neq x_t) \log(1 - D(x^{\text{corrupt}}, t)) \right)$$
결과적으로 판별기는 모든 토큰에 대해 "원래 토큰인가(Original)" 또는 "치환된 토큰인가(Replaced)"를 판단하는 능력을 갖게 된다.

### 2. RTD 기반 프롬프트 및 예측 (Prompt and Predict)

본 논문은 다운스트림 태스크를 RTD의 이진 분류 문제로 변환한다.

#### 분류 작업 (Classification Task)

각 클래스 $y^{(m)}$에 대응하는 라벨 단어 $LW^{(m)}$와 프롬프트 템플릿 $t^{(m)}$를 구성한다. 입력 형태는 다음과 같다.
$$x_{\text{input}} = [CLS] t^{(m)} [SEP] x_n [EOS]$$

모델이 라벨 단어 $LW^{(m)}$를 '치환됨(Replaced)'이라고 예측할 확률을 $P(LW^{(m)} = \text{Rep} | [t^{(m)}, x_n])$라고 할 때, 해당 샘플이 클래스 $y^{(m)}$일 확률은 라벨 단어가 '원래의 것(Original)'일 확률과 같다.
$$P(y^{(m)}|x_n) = \frac{(1 - P(LW^{(m)} = \text{Rep} | [t^{(m)}, x_n]))}{\sum_{m=1}^{M} (1 - P(LW^{(m)} = \text{Rep} | [t^{(m)}, x_n]))}$$
최종 예측값 $\hat{y}$는 이 확률이 가장 높은 클래스로 결정된다.

#### 회귀 작업 (Regression Task)

회귀 작업에서는 단일 라벨 단어 $LW$를 사용하며, 모델이 예측한 $P(LW = \text{Rep})$ 확률값 $[0, 1]$을 실제 타겟 값의 범위 $[V_1, V_2]$로 매핑한다.
$$\hat{y} = |V_2 - V_1| \times P(LW = \text{Rep} | [t, x_n]) + V_1$$

## 📊 Results

### 실험 설정

- **데이터셋:** SST-2, MNLI, QNLI, RTE, MRPC, QQP, STS-B를 포함한 GLUE 벤치마크와 SNLI, MR, CR, MPQA, Subj, TREC, CoLA 등 총 15개의 NLP 데이터셋을 사용하였다.
- **비교 모델:** MLM-BERT, NSP-BERT, MLM-RoBERTa 모델의 Base-sized(110M) 및 Large-sized(335M) 버전을 비교 대상으로 설정하였다.
- **평가 방식:** 어떤 훈련 데이터나 검증 데이터도 사용하지 않은 완전한 제로샷(Zero-shot) 설정에서 테스트 세트만으로 평가하였다.

### 주요 결과

- **제로샷 성능:** RTD-ELECTRA는 대부분의 태스크에서 다른 프롬프트 기반 모델들을 압도하였다. Large 모델 기준으로 15개 태스크 중 12개에서 최고 성능을 기록하였다.
- **정량적 수치:** $\text{RTD-ELECTRA}_{\text{large}}$는 $\text{MLM-RoBERTa}_{\text{large}}$, $\text{NSP-BERT}_{\text{large}}$, $\text{MLM-BERT}_{\text{large}}$ 대비 평균적으로 각각 약 8.22%, 12.97%, 13.63%의 성능 향상을 보였다.
- **특이 사항:** 특히 SST-2 태스크에서는 훈련 데이터 없이 90.1%라는 매우 높은 정확도를 달성하였다.
- **모델 크기의 영향:** 모델의 크기가 커질수록 프롬프트 기반 학습의 성능 향상 폭이 뚜렷하게 나타났다. 이는 모델 크기가 사전 학습 단계에서 습득한 지식의 양과 직결되기 때문으로 분석된다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 논문은 ELECTRA의 RTD 사전 학습 방식이 제로샷 학습에 매우 유리함을 입증하였다. 특히 이진 분류 형태의 RTD 태스크가 감성 분석(Sentiment Analysis)과 같은 이진 분류 다운스트림 태스크와 구조적으로 유사하여, 별도의 미세 조정 없이도 높은 성능을 낼 수 있었던 것으로 보인다. 또한, 제로샷 RTD-프롬프트 예측 성능이 소수의 데이터를 사용한 퓨샷 미세 조정(Fine-tuning) 성능보다 높게 나타난 점은, 프롬프트 학습이 모델 내부의 사전 지식을 인출(extraction)하는 데 더 효과적임을 시사한다.

### 한계 및 논의

본 연구에서는 수동으로 설계된 템플릿(Manual Templates)을 사용하였다. 프롬프트의 구성에 따라 성능이 일부 변동되는 모습이 관찰되었으며, 이는 최적의 프롬프트를 찾는 과정이 여전히 중요함을 의미한다. 향후 자동화된 프롬프트 생성 방식이나 연속적(continuous) 프롬프트 임베딩을 ELECTRA에 적용한다면 더 높은 성능을 기대할 수 있을 것이다.

## 📌 TL;DR

본 논문은 ELECTRA의 **Replaced Token Detection(RTD)** 사전 학습 메커니즘을 활용한 새로운 제로샷 프롬프트 학습 방법을 제안하였다. 실험 결과, RTD-ELECTRA는 기존 MLM 기반의 BERT, RoBERTa보다 제로샷 및 퓨샷 시나리오에서 월등한 성능을 보였으며, 특히 SST-2 태스크에서 90.1%의 정확도를 기록하였다. 이는 판별적 사전 학습 방식이 언어 모델의 제로샷 능력을 극대화하는 데 매우 효율적임을 입증하며, 향후 사전 학습 태스크 설계의 중요성을 시사한다.
