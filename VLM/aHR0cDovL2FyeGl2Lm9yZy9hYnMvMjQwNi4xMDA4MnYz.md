# Whisper-Flamingo: Integrating Visual Features into Whisper for Audio-Visual Speech Recognition and Translation

Andrew Rouditchenko, Yuan Gong, Samuel Thomas, Leonid Karlinsky, Hilde Kuehne, Rogerio Feris, James Glass (2024)

## 🧩 Problem to Solve

본 논문은 소음이 심한 환경에서 음성 인식 성능을 높이기 위해 입술 움직임 비디오를 함께 사용하는 Audio-Visual Speech Recognition (AVSR)의 데이터 부족 문제를 해결하고자 한다. 일반적으로 비디오 데이터는 오디오 데이터보다 수집하기 훨씬 어렵기 때문에, 기존 AVSR 모델들의 학습 데이터는 수천 시간 수준에 머물러 있다. 반면, Whisper와 같은 오디오 전용 모델은 수십만 시간의 대규모 데이터를 통해 학습되어 매우 강력한 speech-to-text 디코더 성능을 갖추고 있다.

따라서 본 연구의 목표는 대규모 데이터로 사전 학습된 Whisper의 강력한 디코더 능력을 유지하면서, 비디오 특징(visual features)을 효과적으로 통합하여 소음에 강인한 AVSR 및 음성 번역(Speech Translation) 모델을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 시각-언어 모델인 Flamingo에서 영감을 얻어, **Gated Cross-Attention** 메커니즘을 통해 Whisper의 디코더에 시각적 특징을 주입하는 것이다.

핵심적인 설계 직관은 다음과 같다.

1. **기존 지식의 보존**: 이미 대규모 데이터로 학습된 Whisper의 가중치를 동결(freeze)함으로써 오디오 인식 능력을 유지한다.
2. **점진적 적응**: Gated Cross-Attention 레이어의 파라미터를 0으로 초기화하여, 초기에는 identity function으로 작동하게 함으로써 모델이 점진적으로 시각 정보에 적응하도록 유도한다.
3. **다목적 통합**: 단일 파라미터 셋으로 영어 인식과 여러 언어로의 번역(En-X Translation)을 동시에 수행하는 범용적인 구조를 제안한다.

## 📎 Related Works

기존의 AVSR 융합 방식은 크게 두 가지로 나뉜다.

- **Early Fusion**: 오디오와 비디오 특징을 초기 단계에서 더하거나 연결(concatenation)하여 Transformer 인코더의 입력으로 사용하는 방식이다. AV-HuBERT 등의 SSL 모델과 많은 지도 학습 모델이 이 방식을 채택하고 있다.
- **Late Fusion**: 각 모달리티를 별도의 인코더로 처리한 후, 마지막 단계에서 MLP 등을 통해 융합하여 디코더로 전달하는 방식이다.

본 논문은 기존의 Early/Late Fusion 방식들이 오디오와 비디오의 특징 추출 속도(frame rate)를 동일하게 맞춰야 하는 제약이 있으며, 특히 오디오 전용 모델을 AVSR로 적응시킬 때 텍스트 디코더를 처음부터 다시 학습시켜야 하는 경우가 많아 최적의 성능을 내지 못한다는 한계를 지적한다. Whisper-Flamingo는 사전 학습된 강력한 디코더를 그대로 활용하며, 서로 다른 샘플링 속도를 가진 모달리티를 Cross-Attention으로 유연하게 통합함으로써 이들 방식과 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조

Whisper-Flamingo는 사전 학습된 **AV-HuBERT Large** 인코더와 **Whisper** 모델을 결합한 구조이다. 비디오 입력은 AV-HuBERT를 통해 시각적 특징으로 변환되고, 오디오 입력은 Whisper 인코더를 통해 처리된다. 이후 Whisper의 디코더 블록 내에 삽입된 Gated Cross-Attention 레이어가 비디오 특징을 참조하여 최종 텍스트를 생성한다.

### Gated Cross-Attention 및 방정식

Whisper의 각 디코더 블록(Self-Attention $\rightarrow$ Cross-Attention $\rightarrow$ MLP)의 시작 부분에 Gated Cross-Attention 레이어가 추가된다. 이 레이어의 작동 방식은 다음과 같은 방정식으로 설명된다.

$$x' = x + \tanh(\alpha_{xattn}) \times \text{Attn}(\text{LN}(x), v)$$
$$y = x' + \tanh(\alpha_{mlp}) \times \text{FFW}(\text{LN}(x'))$$

여기서 $x$는 디코더 블록의 입력, $v$는 AV-HuBERT에서 추출된 시각적 특징, $\text{Attn}$은 Multi-head Cross-Attention, $\text{LN}$은 Layer-norm, $\text{FFW}$는 MLP(Feed-Forward Network)를 의미한다.

가장 중요한 점은 학습 가능한 파라미터인 $\alpha_{xattn}$과 $\alpha_{mlp}$를 **0으로 초기화**한다는 것이다. $\tanh(0) = 0$이므로, 학습 초기 단계에서 이 레이어들은 아무런 영향을 주지 않는 identity function으로 작동하며, 학습이 진행됨에 따라 모델이 시각적 특징을 얼마나 반영할지를 스스로 조절하게 된다.

### 학습 절차 (Training Pipeline)

학습은 총 2단계로 진행된다.

1. **Whisper Fine-tuning**: 먼저 오디오 전용 Whisper 모델을 대상 도메인(예: TED talk)에 맞게 미세 조정한다. 이때 소음 강인성을 높이기 위해 오디오에 노이즈를 섞어 학습하며, 표준 Cross-Entropy 손실 함수를 사용한다.
2. **Audio-Visual Adaptation**: 미세 조정된 Whisper의 모든 파라미터를 동결한다. 이후 Gated Cross-Attention 레이어와 시각적 특징 상단의 Linear 레이어만을 추가하여, 오디오-비디오 입력을 통해 이 새로운 레이어들만 학습시킨다.

## 📊 Results

### 실험 설정

- **데이터셋**: 영어 AVSR을 위해 LRS3 및 LRS2를 사용하였으며, 다국어 번역을 위해 MuAViC 데이터셋을 사용하였다.
- **지표**: 음성 인식 성능은 Word Error Rate (WER)로, 번역 성능은 BLEU score로 측정하였다.
- **기준선 (Baseline)**: Zero-shot Whisper, Fine-tuned Whisper, 그리고 기존 SOTA AVSR 모델들(AV-HuBERT, Llama-AVSR 등)과 비교하였다.

### 주요 결과

1. **영어 음성 인식 (LRS3, LRS2)**:
   - LRS3에서 **ASR WER 0.68%**, **AVSR WER 0.76%**라는 SOTA 성능을 달성하였다.
   - LRS2에서도 **AVSR WER 1.4%**로 SOTA를 기록하였다.
   - 특히 0-SNR babble noise 환경에서 오디오 전용 Whisper(WER 11.7%)보다 월등히 낮은 **WER 5.6%**를 기록하며 강력한 소음 강인성을 입증하였다.

2. **영어-다국어 번역 (MuAViC)**:
   - 오디오-비디오 입력을 사용했을 때, 소음 환경에서 평균 BLEU score **20.5**를 기록하여 오디오 전용 모델(18.6)보다 우수한 성능을 보였다.
   - 기존 Bilingual AV-HuBERT가 각 언어 쌍마다 별도의 모델을 학습시켜야 했던 것과 달리, Whisper-Flamingo는 **단일 파라미터 셋**으로 6개 언어 번역과 영어 인식을 동시에 수행하였다.

3. **융합 방식 비교 (Ablation Study)**:
   - Whisper-Medium 모델을 이용해 실험한 결과, Gated Cross-Attention이 Early Fusion이나 Late Fusion보다 소음 환경(Noisy WER)에서 훨씬 뛰어난 성능을 보였다 (Gated: 7.0% vs Early: 10.0% vs Late: 16.5%).

## 🧠 Insights & Discussion

### 강점 및 분석

본 연구는 거대 오디오 모델의 사전 학습 지식을 유지하면서 새로운 모달리티를 효율적으로 통합하는 방법을 제시하였다. 특히 $\tanh$ 게이팅 메커니즘을 통해 기존 모델의 붕괴 없이 시각 정보를 서서히 통합한 점이 주효했다. 또한, Llama-AVSR(8B+ 파라미터)보다 훨씬 작은 규모(2.5B 파라미터)임에도 불구하고 대등하거나 더 나은 성능을 낸 것은 효율성 측면에서 큰 이점이다.

### 한계 및 비판적 해석

1. **학습 단계의 의존성**: 실험 결과, Whisper를 먼저 미세 조정(FT)하지 않고 바로 AV 학습을 진행하면 성능이 매우 낮게 나타났다. 이는 모델이 '소음 처리', '도메인 적응', '모달리티 통합'이라는 세 가지 어려운 과제를 동시에 수행하기 어렵기 때문으로 분석된다. 즉, 단계별 학습이 필수적이라는 제약이 있다.
2. **시각 인코더의 고정**: 본 연구에서는 AV-HuBERT의 가중치를 동결하여 사용하였다. LRS2 데이터셋에서 LRS3보다 성능 향상 폭이 적었던 이유가 LRS3로 학습된 AV-HuBERT를 그대로 사용했기 때문이라고 저자들은 분석하고 있으며, 이는 향후 시각 인코더의 공동 최적화 가능성을 시사한다.

## 📌 TL;DR

본 논문은 대규모 오디오 모델인 Whisper에 AV-HuBERT의 시각 특징을 **Gated Cross-Attention**으로 통합한 **Whisper-Flamingo**를 제안한다. 이 모델은 LRS3와 LRS2 데이터셋에서 SOTA AVSR 성능을 달성하였으며, 특히 극심한 소음 환경에서 탁월한 강인성을 보인다. 또한 단일 모델로 영어 인식과 다국어 번역을 동시에 수행할 수 있음을 증명하여, 향후 거대 음성-언어 모델의 다중 모달 확장 연구에 중요한 방법론적 기여를 하였다.
