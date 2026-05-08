# BERT Meets CTC: New Formulation of End-to-End Speech Recognition with Pre-trained Masked Language Model

Yosuke Higuchi, Brian Yan, Siddhant Arora, Tetsuji Ogawa, Tetsunori Kobayashi, Shinji Watanabe (2023)

## 🧩 Problem to Solve

본 논문은 종단간 음성 인식(End-to-End Automatic Speech Recognition, E2E-ASR) 시스템에서 발생하는 입력-출력 간의 거대한 격차 문제를 해결하고자 한다. 입력인 음성 신호는 미세한 패턴을 가진 연속적인 신호인 반면, 출력인 텍스트는 장거리 의존성을 가진 이산적인 언어 심볼로 구성되어 있다. 이러한 특성 차이로 인해 모델이 음성으로부터 적절한 텍스트를 생성하는 데 필요한 의미적/형태통사적 정보를 추출하는 것이 어렵다.

특히, 기존의 Connectionist Temporal Classification (CTC) 방식은 출력 토큰 간의 조건부 독립 가정(conditional independence assumption)을 전제로 한다. 즉, 특정 시점의 토큰 예측이 이전이나 이후의 토큰에 의존하지 않는다고 가정하는데, 이는 모델이 타겟 시퀀스의 다중 모드 분포(multimodal distribution)를 제대로 캡처하지 못하게 하여 성능을 제한하는 주요 원인이 된다. 본 연구의 목표는 사전 학습된 Masked Language Model (MLM)인 BERT를 CTC에 결합하여 이 조건부 독립 가정을 완화하고, 풍부한 언어적 지식을 E2E-ASR에 직접 통합하는 새로운 정식화(formulation)를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 BERT를 단순한 리스코어링(rescoring) 도구가 아닌, CTC의 예측 과정에 직접 관여하는 컨텍스트 임베딩 추출기로 사용하는 것이다.

주요 기여점은 다음과 같다.

1. **BERT-CTC 제안**: 사전 학습된 BERT를 CTC 기반 E2E-ASR에 효율적으로 적응시킨 BERT-CTC를 제안하였다. 이는 별도의 파인튜닝 없이 BERT를 frozen 상태로 사용하여 언어적 정보를 주입한다.
2. **조건부 독립 가정 완화**: BERT의 양방향 컨텍스트 임베딩을 통해 CTC의 출력 조건부 확률을 계산함으로써, 기존 CTC의 한계였던 토큰 간 독립 가정을 완화하였다.
3. **반복적 정제 추론**: 추론 단계에서 Mask-predict 알고리즘과 CTC 디코딩을 결합하여, 출력 시퀀스를 반복적으로 정제하고 길이를 유연하게 조정하는 메커니즘을 도입하였다.
4. **범용성 및 확장성 입증**: 다양한 언어와 발화 스타일의 데이터셋에서 성능 향상을 증명하였으며, 이를 종단간 구어 이해(End-to-End Spoken Language Understanding, SLU) 태스크로 확장 적용할 수 있음을 보였다.

## 📎 Related Works

기존의 E2E-ASR 연구들은 사전 학습된 언어 모델(LM)을 다음과 같은 방식으로 통합해 왔다.

- **간접적 통합**: N-best 가설에 대한 리스코어링(rescoring)이나 지식 증류(knowledge distillation)를 통해 LM의 지식을 전달하는 방식이다.
- **직접적 통합**: LM을 ASR 모델과 통합하여 End-to-End로 파인튜닝하는 방식이다.
- **비자기회귀(Non-autoregressive) 접근**: CMLM(Conditional Masked Language Model)이나 Mask-CTC와 같이 마스킹된 토큰을 예측하는 방식을 사용한다. 하지만 이는 대개 모델 내부의 MLM 목적 함수를 학습하는 방식이며, BERT와 같은 거대 외부 사전 학습 모델의 임베딩을 CTC 프레임워크 내에 이론적으로 통합하려는 시도는 부족했다.

본 논문의 BERT-CTC는 외부의 frozen BERT를 사용하여 컨텍스트 임베딩을 제공한다는 점에서 Cold Fusion의 변형으로 볼 수 있으나, 이를 CTC의 확률론적 정식화 내에서 체계적으로 통합했다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 전체 구조 및 파이프라인

BERT-CTC는 Audio Encoder, frozen BERT, 그리고 Self-Attention 모듈로 구성된다. 음성 입력 $O$는 Audio Encoder를 통해 음향 특징 벡터 $H^{ae}$로 변환된다. 동시에, 마스킹된 텍스트 시퀀스 $\tilde{W}$는 BERT를 통해 컨텍스트 임베딩 $H^{bert}$로 변환된다. 이 두 벡터는 Self-Attention 모듈에 입력되어 최종적으로 각 프레임에서의 토큰 확률 분포를 생성한다.

### 확률론적 정식화 및 주요 방정식

BERT-CTC는 부분적으로 마스킹된 시퀀스 $\tilde{W}$를 도입하여 조건부 확률 $p(W|O)$를 다음과 같이 정의한다.

$$p^{bc}(W|O) = \sum_{\tilde{W} \in \mathcal{A}(W)} p(W|\tilde{W}, O) p(\tilde{W}|O)$$

여기서 $\mathcal{A}(W)$는 $W$에서 가능한 모든 마스킹 패턴의 집합이다. $p(W|\tilde{W}, O)$는 다음과 같이 CTC 정렬(alignment) $A$를 통해 분해된다.

$$p(W|\tilde{W}, O) \approx \sum_{A \in \mathcal{B}_{ctc}^{-1}(W)} \prod_{t=1}^{T} p(a_t | \text{BERT}(\tilde{W}), O)$$

최종적인 토큰 방출 확률 $p(a_t | \text{BERT}(\tilde{W}), O)$는 다음과 같이 계산된다.

$$p(a_t | \text{BERT}(\tilde{W}), O) = \text{Softmax}(\text{SelfAttn}_t(H^{ae}, H^{bert}))$$
$$H^{bert} = \text{BERT}(\tilde{W})$$

여기서 $\text{SelfAttn}$은 $H^{ae}$와 $H^{bert}$를 입력으로 받는 트랜스포머 기반의 셀프 어텐션 층이다.

### 학습 절차 및 손실 함수

BERT-CTC의 학습 목표는 위 확률식의 음의 로그 가능도(negative log-likelihood)를 최소화하는 것이다. 계산의 복잡성을 줄이기 위해 젠슨의 부등식(Jensen's inequality)을 사용하여 다음과 같은 상한선(upper bound) 손실 함수 $L_{bc}$를 정의한다.

$$L_{bc}(O, W) \approx -\mathbb{E}_{\tilde{W}} \left[ \log \sum_{A} \prod_{t} p(a_t | \text{BERT}(\tilde{W}), O) \right]$$

학습 시 $\tilde{W}$는 정답 시퀀스 $W$에서 무작위로 선택된 $M \sim \text{Uniform}(1, N)$개의 토큰을 $[MASK]$로 대체하여 샘플링한다.

또한, BERT의 어휘 사전(vocabulary) 크기가 매우 커서 ASR 학습이 어려우므로, 작은 어휘 사전 $V'$를 사용하는 **계층적 손실(Hierarchical Loss)**을 추가로 적용한다. 최종 손실 함수는 다음과 같다.

$$(1-\lambda_{ctc})L_{bc}(O, W) + \lambda_{ctc}L_{ctc}(O, W')$$

### 추론 절차 (Mask-Predict Algorithm)

BERT-CTC는 비자기회귀 모델이므로 출력 길이를 미리 정해야 한다.

1. **초기화**: Audio Encoder의 예측 결과 $\hat{W}'$를 통해 타겟 길이 $\hat{N}$을 결정하고, 모든 위치를 $[MASK]$로 채운 초기 시퀀스 $\bar{W}^{(1)}$을 생성한다.
2. **반복적 정제**: $K$번의 반복 동안 다음 과정을 수행한다.
    - **토큰 예측**: $\bar{W}^{(k)}$와 $H^{ae}$를 이용하여 CTC 디코딩을 수행하고 가설 시퀀스 $\hat{W}^{(k)}$를 얻는다.
    - **토큰 마스킹**: $\hat{W}^{(k)}$에서 확률 점수가 가장 낮은 $m(k)$개의 토큰을 다시 $[MASK]$로 치환하여 $\bar{W}^{(k+1)}$을 생성한다.
3. **최종 출력**: $K$번째 반복 후의 $\hat{W}^{(K)}$를 최종 결과로 반환한다.

## 📊 Results

### 실험 설정

- **데이터셋**: LibriSpeech-100h, TED-LIUM2 (영어), AISHELL-1 (중국어), SLURP (SLU 태스크).
- **비교 대상**: 기본 CTC (Intermediate CTC 적용), RNN-T (Auxiliary CTC 적용).
- **지표**: Word Error Rate (WER), Character Error Rate (CER), Accuracy (SLU).

### 주요 결과

1. **ASR 성능**: BERT-CTC는 모든 데이터셋에서 CTC와 RNN-T보다 유의미하게 낮은 WER/CER을 기록하였다. 특히 LibriSpeech의 'other' 셋과 TED-LIUM2와 같은 일반 발화 데이터에서 성능 향상이 두드러졌다.
2. **SLU 성능**: SLURP 데이터셋의 의도 분류(intent classification) 작업에서 기존 ESPnet-SLU 베이스라인보다 높은 정확도를 달성하였다.
3. **추론 속도**: RNN-T와 같은 자기회귀 모델보다 빠른 추론 속도(RTF)를 보였으며, GPU 병렬 연산을 통해 효율성을 높였다.

| Model | LibriSpeech-100h (Test WER $\downarrow$) | TED-LIUM2 (Test WER $\downarrow$) | AISHELL-1 (Test CER $\downarrow$) |
| :--- | :---: | :---: | :---: |
| CTC | 21.4 / 22.0 | 9.3 / 5.6 | 5.5 |
| RNN-T | 21.5 / 22.2 | 10.2 / 9.6 | 5.5 |
| **BERT-CTC** | **16.3 / 16.6** | **8.1 / 7.6** | **3.9 / 3.9** |

## 🧠 Insights & Discussion

### 강점 및 분석

- **양방향 컨텍스트의 이점**: RNN-T는 인과적(causal) 의존성만 고려하지만, BERT-CTC는 BERT를 통해 전체 시퀀스의 양방향 컨텍스트를 고려한다. 이는 특히 음성적으로 유사한 단어(homophones)를 문맥에 맞게 교정하는 데 매우 효과적임이 Error Analysis를 통해 확인되었다.
- **계층적 손실의 중요성**: BERT의 거대 어휘 사전과 ASR 도메인 간의 불일치(domain mismatch) 문제가 존재하지만, 작은 사전 기반의 계층적 CTC 손실을 통해 이를 성공적으로 완화하였다.
- **독립성 가정의 검증**: BERT에 오디오 정보를 추가로 주입하는 어댑터 구조를 실험했으나 성능 향상이 없었다. 이는 BERT가 이미 충분히 강력한 언어적 표현을 가지고 있어 추가적인 오디오 적응 없이도 충분함을 시사한다.

### 한계 및 비판적 해석

- **어휘 사전 제약**: BERT의 고정된 어휘 사전을 사용해야 하므로, ASR 도메인에 특화된 작은 사전($V_{asr}$)을 사용한 모델보다 기본 성능(K=1일 때)이 낮게 시작하는 경향이 있다. 이는 CharacterBERT와 같은 대안적인 MLM 사용 필요성을 제기한다.
- **연산 비용**: 추론 시 BERT를 $K$번 반복적으로 통과시켜야 하므로, 단순 CTC보다는 연산량이 훨씬 많다.
- **비스트리밍 특성**: 전체 시퀀스를 한꺼번에 처리하는 구조이므로 실시간 스트리밍 서비스에 바로 적용하기 어렵다. 이를 해결하기 위해서는 Causal Masking이나 Two-pass 알고리즘 도입이 필요하다.

## 📌 TL;DR

본 논문은 CTC의 치명적 약점인 '출력 토큰 간 조건부 독립 가정'을 해결하기 위해, 사전 학습된 **BERT를 컨텍스트 임베딩 추출기로 통합한 BERT-CTC**를 제안하였다. BERT의 양방향 언어 지식을 활용해 음성 인식 결과를 반복적으로 정제함으로써, 기존 CTC 및 RNN-T 대비 월등한 인식 성능을 달성하였으며, 이를 SLU 태스크까지 확장 적용하였다. 이 연구는 거대 언어 모델의 지식을 E2E-ASR의 디코딩 과정에 이론적으로 결합하는 효과적인 방법을 제시했다는 점에서 향후 연구에 중요한 역할을 할 것으로 보인다.
