# IMPROVING END-TO-END SPEECH RECOGNITION WITH POLICY LEARNING

Yingbo Zhou, Caiming Xiong, Richard Socher (2017)

## 🧩 Problem to Solve

종단간(End-to-End) 음성 인식 모델에서 일반적으로 사용되는 학습 목적 함수인 Maximum Likelihood Estimation(MLE), 특히 Connectionist Temporal Classification(CTC) 손실 함수는 실제 성능 평가 지표인 Word Error Rate(WER) 또는 Character Error Rate(CER)와 간극이 존재한다.

MLE는 정답 전사(Transcription)의 확률을 최대화하는 것에 집중하지만, 이는 모든 종류의 오답을 동일하게 처리하는 경향이 있다. 반면, WER과 같은 평가 지표는 정답과 예측값 사이의 Edit Distance를 고려하여, 정답에 더 가까운 오답에 더 적은 페널티를 부여한다. 이러한 목적 함수와 평가 지표 사이의 불일치(Mismatch)로 인해, 모델이 학습 과정에서 최적화하는 방향이 실제 성능 향상으로 직결되지 않는 문제가 발생한다.

본 논문의 목표는 미분 불가능한 성질을 가진 WER과 같은 성능 지표를 직접 최적화함으로써, 이 간극을 줄이고 음성 인식 성능을 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Maximum Likelihood(MLE) 학습과 Policy Gradient 기반의 강화학습을 결합한 공동 학습(Joint Training)** 체계를 도입하는 것이다.

특히, 강화학습의 분산(Variance) 문제를 해결하기 위해 **Self-Critical Sequence Training(SCST)** 기법을 적용하였다. SCST는 모델의 Greedy 디코딩 결과를 Baseline으로 사용하여 보상을 계산함으로써 학습의 안정성을 높인다. 또한, 학습 초기 단계의 불안정성을 극복하기 위해 MLE 손실 함수를 함께 사용하는 다중 목적 함수(Multi-objective) 설계를 제안하여, 모델이 빠르게 수렴하면서도 최종적으로는 성능 지표를 최적화하도록 유도하였다.

## 📎 Related Works

기존의 End-to-End 음성 인식 모델들은 주로 CTC를 이용한 MLE 방식으로 학습되었다. 또한 Graves와 Jaitly는 WER을 최적화하기 위해 Expected Transcription Loss를 제안하였으나, 이는 어휘집(Vocabulary) 크기와 시퀀스 길이에 비례하여 계산 비용이 매우 높다는 한계가 있다.

반면, 기계 번역이나 이미지 캡셔닝 분야에서는 Policy Gradient를 통해 미분 불가능한 지표를 최적화하는 연구가 진행되어 왔다. 본 논문은 이러한 강화학습 접근법을 음성 인식에 도입하되, 계산 효율성이 높은 SCST를 채택하여 기존의 고비용 최적화 방식과 차별화를 두었다.

## 🛠️ Methodology

### 1. 전체 아키텍처

모델 구조는 Deep Speech 2(DS2)와 유사하며, 크게 특징 추출을 위한 Convolutional layers와 시퀀스 모델링을 위한 Recurrent layers로 구성된다.

- **Front-end (CNN):** 시간과 주파수 축의 변화를 모두 포착하기 위해 2D Convolution을 사용하며, 계산 효율성을 위해 **Depth-wise Separable Convolution**을 적용하였다.
  - 입력 $x \in \mathbb{R}^{F \times T \times D}$에 대해, 먼저 채널별로 컨볼루션을 수행하여 $s(i, j, d)$를 얻고, 이후 $1 \times 1$ 컨볼루션을 통해 출력 채널 $N$으로 매핑하는 $o(i, j, n)$을 생성한다.
  - 구체적인 수식은 다음과 같다.
    $$s(i, j, d) = \sum_{f=0}^{F-1} \sum_{t=0}^{T-1} x(f, t, d) c(i-f, j-t, d)$$
    $$o(i, j, n) = \sum_{k=0}^{D-1} s(i, j, k) w(k, n)$$
  - 총 6개의 컨볼루션 레이어를 사용하며, 첫 번째 레이어 이후에는 5개의 Residual Convolution Block을 배치하여 학습을 용이하게 하였다.

- **Back-end (RNN & FC):** 4개의 Bidirectional GRU(각 방향당 1024 units) 레이어를 통해 시퀀스를 모델링하고, 마지막으로 2개의 Fully Connected(FC) 레이어를 통해 각 캐릭터에 대한 예측값을 출력한다.

### 2. 학습 목적 함수 (Model Objective)

#### (1) Maximum Likelihood Training (CTC)

정렬(Alignment) 정보가 없는 데이터에서 $P(y|x)$를 최대화하기 위해 CTC를 사용한다. 이는 모든 가능한 정렬 경로를 주변화(Marginalize)하여 계산한다.

#### (2) Policy Learning (SCST)

모델을 에이전트로, 학습 샘플을 환경으로 간주한다. 모델 파라미터 $\theta$는 정책 $P_\theta(y|x)$를 정의하며, 생성된 전사 결과 $y^s$에 대해 reward $r(y^s)$를 부여한다. 이때 보상 함수는 $g(\cdot, y) = 1 - \max(1, \text{WER}(\cdot, y))$로 정의하여 WER이 낮을수록 높은 보상을 얻게 한다.

분산을 줄이기 위해 SCST를 적용한 Policy Gradient의 근사식은 다음과 같다.
$$\nabla_\theta L_p(\theta) \approx -(r(y^s) - r(\hat{y})) \nabla_\theta \log P_\theta(y^s | x)$$
여기서 $\hat{y}$는 모델의 Greedy 디코딩 결과이며, 이를 Baseline으로 사용하여 현재 샘플링된 결과 $y^s$가 Greedy 결과보다 더 나은지 판단한다.

#### (3) Multi-objective Policy Learning

강화학습의 초기 불안정성을 해결하기 위해 MLE와 SCST를 결합한 최종 손실 함수를 제안한다.
$$L(\theta) = -\log P_\theta(y|x) + \lambda L_{scst}(\theta)$$
$$L_{scst}(\theta) = -\{g(y^s, y) - g(\hat{y}, y)\} \log P_\theta(y^s | x)$$
여기서 $\lambda$는 SCST의 기여도를 조절하는 계수로, 학습 초기에는 $\lambda=0.1$로 설정하고 모델이 수렴한 후에는 $\lambda=1$로 높여 성능 지표 최적화에 집중하게 한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋:** Wall Street Journal(WSJ) 및 LibriSpeech.
- **입력 데이터:** 20ms 윈도우, 10ms 스텝 사이즈의 Spectrogram (Zero mean, Unit variance 정규화 적용).
- **디코딩:** 4-gram Language Model과 Beam width 100인 Beam Search 사용.

### 2. 주요 결과

- **WSJ 데이터셋:** Baseline(MLE 전용) 대비 Policy Gradient(Eq. 5)는 성능이 향상되었으며, SCST(Eq. 7)를 적용한 경우 **상대적으로 13.8%의 성능 향상**을 보였다. 최종 WER은 5.53%를 달성하였다.
- **LibriSpeech 데이터셋:** Baseline 대비 **약 4%의 상대적 성능 향상**이 관찰되었으며, test-clean에서 5.42%, test-other에서 14.70%의 WER을 기록하였다.

### 3. 타 모델과의 비교

- WSJ 데이터셋에서 본 모델은 기존의 여러 End-to-End 방식(Deep Speech 2, Graves & Jaitly 등)과 경쟁력 있는 성능을 보였다.
- 특히 LibriSpeech로 학습한 모델을 WSJ에 테스트했을 때, WSJ 데이터로만 학습한 모델보다 훨씬 좋은 성능을 보였는데, 이는 End-to-End 모델이 데이터 양의 증가에 매우 민감하게 반응함을 시사한다.

## 🧠 Insights & Discussion

본 연구는 MLE 학습이 가진 본질적인 한계, 즉 "모든 오류를 동일하게 취급하는 문제"를 강화학습의 Policy Gradient를 통해 효과적으로 해결하였다. 특히 SCST를 도입하여 계산 효율성을 확보하면서도 학습의 분산을 줄인 점이 주효했다.

또한, MLE와 Policy Learning을 결합한 Multi-objective 학습 방식은 매우 영리한 전략이다. 강화학습 단독으로는 초기에 유의미한 보상을 얻기 어려워 학습이 매우 느리거나 불안정할 수 있는데, 정답 라벨을 직접 활용하는 MLE가 가이드 역할을 수행함으로써 안정적인 초기 학습을 가능하게 했다.

비판적으로 해석하자면, 본 논문은 데이터 증강(Tempo, Pitch, Volume 변화 등)과 대규모 데이터셋(LibriSpeech)의 영향을 함께 보여주었다. 결과적으로 성능 향상이 오직 Policy Learning 덕분인지, 아니면 대량의 데이터와 정교한 전처리의 시너지 효과인지에 대한 정밀한 분리 분석은 다소 부족하다. 하지만 제안한 알고리즘이 동일 조건의 Baseline 대비 일관된 향상을 보였다는 점에서 방법론적 유효성은 입증되었다고 볼 수 있다.

## 📌 TL;DR

이 논문은 End-to-End 음성 인식 모델에서 학습 목적 함수(MLE)와 평가 지표(WER) 간의 불일치 문제를 해결하기 위해, **Maximum Likelihood와 SCST(Self-Critical Sequence Training) 기반의 Policy Learning을 결합한 공동 학습 방법**을 제안하였다. 이를 통해 계산 효율성을 유지하면서도 WER을 직접 최적화할 수 있었으며, WSJ와 LibriSpeech 데이터셋에서 각각 최대 13.8% 및 4%의 상대적 성능 향상을 달성하였다. 이 연구는 미분 불가능한 성능 지표를 최적화해야 하는 다양한 시퀀스 생성 Task에 응용될 가능성이 높다.
