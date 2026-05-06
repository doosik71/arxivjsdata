# AN IMPROVED HYBRID CTC-ATTENTION MODEL FOR SPEECH RECOGNITION

Zhe Yuan, Zhuoran Lyu, Jiwei Li and Xi Zhou (2018)

## 🧩 Problem to Solve

본 논문은 자동 음성 인식(Automatic Speech Recognition, ASR) 시스템에서 발생하는 정렬(alignment) 문제와 단어 외부 단어(Out-of-Vocabulary, OOV) 문제를 해결하고자 한다. 전통적인 ASR 방식인 HMM(Hidden Markov Models)과 GMM(Gaussian Mixture Models)은 수작업으로 제작된 발음 사전과 정밀한 정렬 정보가 필요하여 구축 비용이 높다는 단점이 있다.

최근의 End-to-End ASR 모델 중 CTC(Connectionist Temporal Classification) 기반 모델과 Attention 기반의 seq2seq 모델이 제안되었으나, 각각의 한계가 존재한다. Attention 메커니즘은 소음이 많은 실제 환경에서 정렬 추정이 쉽게 무너지는 경향이 있으며, CTC는 단독 사용 시 언어 모델링 능력이 부족하다. 따라서 본 연구의 목표는 CTC의 강점인 강건한 정렬 능력과 Attention의 강력한 디코딩 능력을 결합한 Hybrid CTC-Attention 모델의 구조를 개선하여 LibriSpeech 데이터셋에서 최적의 Word Error Rate(WER)를 달성하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 Hybrid CTC-Attention 구조를 최적화하기 위한 세 가지 전략적 설계에 있다.

첫째, CTC 디코더 전용 BiLSTM 레이어를 추가한 새로운 구조를 제안한다. 공유 인코더와 CTC 디코더 사이에 전용 레이어를 배치함으로써, 전체 손실 함수에서 CTC의 비중이 낮아질 때 발생할 수 있는 CTC 브랜치의 성능 저하를 보완한다.

둘째, 인코더의 깊이(depth)가 인식 성능에 미치는 영향을 실험적으로 분석하여 최적의 레이어 구성을 도출한다.

셋째, Subword 기반 디코딩의 성능을 높이기 위해 Attention Smoothing 메커니즘을 도입한다. 이는 Attention 점수 분포가 너무 급격하게(sharp) 변하는 것을 방지하여 디코더가 더 많은 문맥 정보(context information)를 활용할 수 있게 한다.

## 📎 Related Works

논문에서는 기존의 ASR 접근 방식을 크게 전통적 방식과 End-to-End 방식으로 구분하여 설명한다. HMM/GMM 기반의 전통적 방식은 높은 정확도를 보이지만 도메인 지식이 많이 필요하며 유연성이 떨어진다는 한계가 있다.

End-to-End 방식 중 Deep Speech 2와 같은 CTC 기반 모델과 LAS(Listen, Attend and Spell)와 같은 Attention 기반 모델이 등장하였다. 하지만 앞서 언급한 바와 같이 Attention 모델은 노이즈에 취약한 정렬 문제(misalignment problem)를 가지고 있다. 이를 해결하기 위해 Watanabe 등이 제안한 Hybrid CTC-Attention 구조는 공유 인코더를 사용하고 CTC와 Attention 디코더를 동시에 학습시키는 Multi-task Learning 방식을 취한다.

또한, OOV 문제를 해결하기 위해 기존의 Phoneme이나 Word 단위 대신 BPE(Byte-Pair Encoding) 알고리즘을 이용한 Subword 단위를 채택하여, 음향 정보와 문자 정보 사이의 관계를 보다 효과적으로 학습하도록 한다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

본 모델은 CNN 레이어와 BiLSTM 레이어로 구성된 공유 인코더(Shared Encoder)를 기반으로 하며, 이후 CTC 브랜치와 Attention 브랜치로 나뉘는 구조를 가진다.

- **공유 인코더**: 4층의 CNN 레이어(Max-pooling을 통해 입력 특징을 1/4로 다운샘플링)와 7층의 BiLSTM 레이어로 구성된다.
- **CTC 브랜치**: 공유 인코더의 출력값에 1층의 BiLSTM 레이어를 추가로 배치하고, 최종적으로 Fully Connected(FC) 레이어를 통해 CTC 디코더로 연결된다.
- **Attention 브랜치**: 2층의 LSTM 레이어로 구성된 디코더를 사용하여 시퀀스를 생성한다.

### 2. 학습 목표 및 손실 함수

모델은 CTC 손실과 Attention 손실을 동시에 최소화하는 방향으로 학습된다. 전체 손실 함수 $L$은 다음과 같이 정의된다.

$$L = \alpha L_{ctc} + (1-\alpha) L_{att}$$

여기서 $\alpha$는 두 손실 함수의 가중치를 조절하는 하이퍼파라미터이다. $L_{ctc}$는 입력 시퀀스 $x$에 대해 출력 라벨 $y$가 나타날 확률의 음의 로그 우도(negative log-likelihood)로 계산되며, $L_{att}$는 Attention 기반 디코더가 예측한 시퀀스의 확률을 바탕으로 계산된다.

### 3. Attention Smoothing

Subword 단위의 디코딩에서는 더 넓은 문맥 정보가 필요하지만, 일반적인 Softmax 기반 Attention은 분포가 매우 좁게 형성되는 경향이 있다. 이를 해결하기 위해 본 논문은 Sigmoid 함수를 이용한 Smoothing 메커니즘을 적용한다.

먼저 Location-based attention 에너지를 $e_{s,t}$라고 할 때, 기존의 Softmax 방식 대신 다음과 같이 $\epsilon_{s,t}$를 계산한다.

$$\epsilon_{s,t} = \frac{\text{sigmoid}(e_{s,t})}{\sum_{i=1}^{T} \text{sigmoid}(e_{s,i})}$$

이 방식은 Attention score 분포를 완만하게 만들어, 디코더가 특정 프레임에만 과하게 집중하지 않고 주변 문맥을 더 충분히 참조할 수 있도록 돕는다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: LibriSpeech (train-clean-100, 360, train-other-500 사용)
- **특징 추출**: 80차원 Mel-filterbank features (KALDI 툴킷 사용)
- **단위**: BPE 알고리즘을 통한 5,000개의 Subword units
- **데이터 증강**: 3-fold speed perturbation (0.9x, 1.0x, 1.1x)
- **언어 모델(LM)**: 14,500권의 공개 도서로 학습된 2층 LSTM RNN-LM 사용

### 2. 주요 결과

본 모델은 LibriSpeech의 `test-clean` 서브셋에서 LM 없이 **4.43%**, RNN-LM 적용 시 **3.34%**의 WER을 달성하였다. 이는 당시 End-to-End ASR 시스템 중 가장 우수한 성능이다.

**구조적 개선에 따른 성능 변화(Table 1 분석):**

- 인코더의 BiLSTM 레이어 수를 5층에서 7층으로 늘릴수록 WER이 감소한다.
- 동일한 인코더 깊이에서 CTC 브랜치에 전용 BiLSTM 레이어를 추가했을 때 성능이 일관되게 향상된다. (예: 7층 인코더 기준, no LM 시 4.64% $\rightarrow$ 4.43%)

**가중치 $\alpha$의 영향(Fig 4 분석):**

- $\alpha$ 값을 낮출수록(즉, Attention 손실의 비중을 높일수록) WER이 감소하며, 최적의 값은 $\alpha=0.1$로 나타났다. 이는 CTC가 초기 정렬과 수렴 속도를 돕는 보조적인 역할을 수행하고, 최종 디코딩 효율은 Attention 디코더가 주도한다는 점을 시사한다.

## 🧠 Insights & Discussion

본 연구는 Hybrid CTC-Attention 모델에서 각 구성 요소의 역할 분담을 최적화하는 것이 성능 향상의 핵심임을 보여준다. 특히 $\alpha$ 값을 낮추어 Attention 디코더에 더 큰 비중을 두었을 때 성능이 좋아지는데, 이때 발생할 수 있는 CTC 브랜치의 학습 부족 문제를 'CTC 전용 BiLSTM 레이어'라는 구조적 장치로 해결한 점이 매우 영리한 접근이다.

또한, Subword 단위를 사용할 때 발생하는 Attention의 과도한 집중 현상을 Smoothing 기법으로 완화함으로써, 음성 인식의 고질적인 문제인 문맥 파악 능력을 개선하였다.

다만, 본 논문은 주로 `test-clean` 데이터셋에서의 성과에 집중하고 있으며, 매우 노이즈가 심한 환경에서의 강건성에 대한 심층적인 분석은 부족한 편이다. 또한, 제안된 구조가 다른 언어(예: 한국어, 중국어와 같은 교착어 및 성조 언어)에서도 동일하게 작동할지는 추가적인 검증이 필요하다.

## 📌 TL;DR

본 논문은 Hybrid CTC-Attention ASR 모델에 **(1) CTC 전용 BiLSTM 레이어 추가**, **(2) 인코더 깊이 최적화**, **(3) Attention Smoothing 기법**을 도입하여 LibriSpeech 데이터셋에서 SOTA 수준의 WER(3.34% with LM)을 달성하였다. 특히 CTC를 보조 수단으로 활용하면서도 그 성능을 유지하기 위한 구조적 개선과 Subword 디코딩을 위한 문맥 정보 확장 전략이 핵심이며, 이는 향후 고성능 End-to-End 음성 인식 시스템 설계에 중요한 참고 자료가 될 것으로 보인다.
