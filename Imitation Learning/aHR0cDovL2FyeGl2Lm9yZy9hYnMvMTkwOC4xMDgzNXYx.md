# An Empirical Comparison on Imitation Learning and Reinforcement Learning for Paraphrase Generation

Wanyu Du, Yangfeng Ji (2019)

## 🧩 Problem to Solve

본 논문은 주어진 문장에 대해 의미가 동일한 다른 문장을 생성하는 Paraphrase Generation 작업에서 발생하는 **Exposure Bias** 문제를 해결하고자 한다.

Encoder-Decoder 프레임워크를 사용하는 일반적인 Supervised Learning 방식에서는 학습 시 현재 토큰을 예측할 때 정답(Ground Truth) 데이터를 조건으로 사용하지만, 실제 추론(Decoding) 시에는 이전 단계에서 모델이 예측한 값을 조건으로 사용한다. 이러한 학습과 추론 간의 괴리는 생성 과정에서 오류가 누적되고 전파되는 Exposure Bias를 유발하며, 이는 최종 생성 품질을 저하시키는 주요 원인이 된다.

기존 연구들은 이를 완화하기 위해 Reinforcement Learning(RL)이나 Imitation Learning(IL)을 도입해 왔으나, 두 방법론에 대한 직접적인 비교 분석이 부족하여 각각의 이점과 한계를 명확히 파악하기 어려웠다. 따라서 본 연구의 목표는 RL과 IL이 Paraphrase Generation 성능 향상에 어떻게 기여하는지 실증적으로 비교 분석하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 RL과 IL을 하나의 통합된 프레임워크 내에서 비교할 수 있도록 정식화하고, 이를 통해 Paraphrase Generation 작업에서 어떤 학습 전략이 더 효과적인지 규명한 것이다.

가장 중심적인 아이디어는 학습 과정에서 입력값($\tilde{y}_{t-1}$)과 출력값($\tilde{y}_t$)을 결정하는 스케줄링 비율 $\alpha, \beta$를 도입하여, $\alpha$와 $\beta$의 설정값에 따라 REINFORCE(RL), DAGGER(IL), MLE(Supervised Learning)를 모두 특수 사례로 포함하는 통합 목적 함수를 제안한 것이다. 이를 통해 다양한 변형 알고리즘을 설계하고 성능을 정밀하게 비교하였다.

## 📎 Related Works

Paraphrase Generation은 초기에 병렬 말뭉치에서 패턴을 추출하는 방식에서 시작하여, 통계적 기계 번역(SMT)을 거쳐 최근에는 신경망 기반의 Encoder-Decoder 구조로 발전하였다. 특히 Stacked Residual LSTM이나 Pointer-Generator Network와 같은 모델들이 성능 향상을 이끌었다.

기존의 RL 기반 접근 방식인 RbM(Reinforced by Matching) 등은 Pointer-Generator를 기반으로 강화학습을 적용해 Exposure Bias를 줄이려 하였다. 또한, 구조적 예측(Structured Prediction) 분야에서는 오래전부터 Imitation Learning이 사용되어 왔으며, NLP 분야에서는 Scheduled Sampling이라는 이름으로 IL의 일종이 제안되었다. 하지만 이러한 기법들이 RL과 비교하여 실제로 어느 정도의 효용성이 있는지에 대한 정량적 비교는 미비한 상태였다.

## 🛠️ Methodology

### 전체 시스템 구조

본 연구는 기본 모델(Base Model)로 **Pointer-Generator Network**를 사용한다. 이 모델은 일반적인 생성 능력과 더불어 입력 문장에서 단어를 직접 복사해오는 Pointer 메커니즘을 결합하여 OOV(Out-of-Vocabulary) 문제에 강건한 특성을 가진다.

### 통합 학습 프레임워크

저자들은 RL과 IL을 통합하여 최적화할 수 있는 다음과 같은 목적 함수 $L(\theta)$를 제안한다.

$$L(\theta) = \left\{ \sum_{t=1}^{T} \log \pi_{\theta}(\tilde{y}_t | h_t) \right\} \cdot r(\tilde{y}, y)$$

여기서 $\pi_{\theta}$는 현재 상태 $h_t$에서 단어를 선택하는 정책 함수(Policy Function)이며, $r(\tilde{y}, y)$는 생성된 문장 $\tilde{y}$와 정답 문장 $y$ 사이의 유사도를 측정하는 보상 함수(Reward Function)이다. 본 논문에서는 보상 함수로 **ROUGE-2** 점수를 사용한다.

### 학습 절차 및 알고리즘 변형

학습은 크게 MLE를 통한 Pre-training 단계와 제안하는 알고리즘들을 통한 Fine-tuning 단계로 나뉜다. 핵심은 입력 $\tilde{y}_{t-1}$과 출력 $\tilde{y}_t$를 결정하는 $\alpha, \beta$의 제어이다.

- **$\alpha$ (Input Schedule Rate):** $\alpha$의 확률로 정답($y_{t-1}$)을 사용하고, $1-\alpha$의 확률로 모델이 예측한 값($\hat{y}_{t-1}$)을 입력으로 사용한다.
- **$\beta$ (Output Schedule Rate):** $\beta$의 확률로 정답($y_t$)을 선택하고, $1-\beta$의 확률로 모델이 예측한 값($\hat{y}_t$)을 선택하여 손실 함수에 반영한다.

이 설정을 통해 다음과 같은 알고리즘들을 정의한다:

1. **MLE:** $\alpha=1, \beta=1$인 경우이다. 항상 정답을 입력하고 정답에 대해 최적화한다.
2. **REINFORCE:** $\alpha=0, \beta=0$이며 $\text{Decode}(\cdot)$가 Random Sampling인 경우이다. 전체 궤적을 샘플링한 후 보상을 곱해 업데이트한다.
3. **DAGGER (Imitation Learning):** $0 < \alpha < 1, \beta=1$이며 $\text{Decode}(\cdot)$가 $\text{argmax}$ (Greedy)인 경우이다. 입력은 정답과 예측값 사이를 오가지만, 출력은 항상 정답(Expert Action)을 따라가도록 학습한다.

또한, RL의 변형으로 입력을 항상 정답으로 고정하는 **REINFORCE-GTI** ($\alpha=1, \beta=0$), 출력을 일부 정답으로 섞는 **REINFORCE-SO** ($\alpha=1, 0 < \beta < 1$), 입력과 출력 모두를 섞는 **REINFORCE-SIO** ($0 < \alpha < 1, 0 < \beta < 1$)를 제안하여 비교 실험을 진행하였다.

## 📊 Results

### 실험 설정

- **데이터셋:** Quora Question Pair Dataset, Twitter URL Paraphrasing Dataset.
- **평가 지표:** ROUGE-1, ROUGE-2, BLEU 및 이들의 평균 점수(Avg-Score).
- **비교 대상:** Seq2Seq, RbM, Res-LSTM, Dis-LSTM.

### 주요 결과

1. **IL의 우수성:** 두 데이터셋 모두에서 DAGGER(IL) 기반 모델이 REINFORCE(RL) 및 그 변형 모델들보다 일관되게 높은 성능을 보였다.
2. **SOTA 달성:** 특히 Quora 데이터셋에서 DAGGER* (고정된 $\alpha$를 사용하는 설정) 모델은 기존 SOTA 방법론들을 큰 차이로 앞지르며 평균 점수 기준 약 13%의 성능 향상을 보였다.
3. **데이터셋별 특성:** Twitter 데이터셋에서는 Quora에 비해 성능 향상 폭이 적었는데, 이는 Twitter 데이터셋의 경우 하나의 소스 문장에 대해 여러 개의 정답 페러프레이즈가 존재하는 'One-to-Many' 특성이 강해 학습이 더 어렵기 때문으로 분석된다.
4. **$\alpha$ 설정의 중요성:** DAGGER에서 $\alpha$의 감쇠(decay) 속도가 너무 빠르면 최적 정책을 배우기 전에 학습이 멈추고, 너무 느리면 지역 최적점(sub-optimal)에 빠지는 경향이 확인되었다. Quora에서는 $\alpha=0.5$, Twitter에서는 $\alpha=0.2$일 때 최적의 성능을 보였다.

## 🧠 Insights & Discussion

본 논문은 Paraphrase Generation 작업에서 **Imitation Learning(특히 DAGGER)이 Reinforcement Learning보다 훨씬 효율적이고 안정적**임을 입증하였다.

그 원인에 대한 분석은 다음과 같다. REINFORCE와 같은 RL 알고리즘은 보상 함수가 희소(sparse)하거나 분산이 클 경우 학습이 불안정해지는 경향이 있다. 반면, DAGGER는 항상 정답(Expert Action)을 출력으로 사용하므로 학습 신호가 명확하며, 입력값에 $\alpha$를 통해 점진적으로 모델의 예측값을 섞어줌으로써 Exposure Bias를 효과적으로 완화한다.

다만, 최적의 $\alpha$ 값을 찾는 과정이 매우 까다롭다는 점(Tricky)이 한계로 지적된다. 이는 태스크의 특성에 따라 적절한 $\alpha$ 값이 다르기 때문에 발생하는 문제로, 향후 이를 자동화하거나 최적화하는 방법론에 대한 연구가 필요함을 시사한다.

## 📌 TL;DR

본 연구는 Paraphrase Generation에서 Exposure Bias를 해결하기 위해 RL과 IL을 통합 프레임워크에서 비교 분석하였다. 실험 결과, **DAGGER(Imitation Learning)가 REINFORCE(Reinforcement Learning)보다 성능과 학습 안정성 면에서 월등히 우수함**을 확인하였으며, 특히 적절한 스케줄링 비율 $\alpha$를 적용한 DAGGER* 모델이 Quora 데이터셋에서 SOTA 성능을 기록하였다. 이 결과는 텍스트 생성 작업에서 RL의 대안으로 IL 및 Scheduled Sampling의 중요성을 강조하며, 향후 유사한 생성 태스크의 학습 전략 수립에 중요한 지침을 제공한다.
