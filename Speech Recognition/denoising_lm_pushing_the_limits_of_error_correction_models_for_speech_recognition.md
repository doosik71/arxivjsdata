# Denoising LM: Pushing the Limits of Error Correction Models for Speech Recognition

Zijin Gu, Tatiana Likhomanenko, He Bai, Erik McDermott, Ronan Collobert, Navdeep Jaitly (2024)

## 🧩 Problem to Solve

본 논문은 자동 음성 인식(Automatic Speech Recognition, ASR) 시스템의 성능을 향상시키기 위해 기존의 언어 모델(Language Model, LM)이 가진 근본적인 한계를 해결하고자 한다. 전통적인 LM은 방대한 텍스트 코퍼스로 학습되어 문법적 완성도는 높으나, ASR 시스템이 실제로 어떤 종류의 오류를 범하는지에 대한 정보가 없는 'ASR-agnostic' 특성을 가진다.

이러한 문제를 해결하기 위해 ASR의 출력을 정제하는 Error Correction(EC) 모델들이 제안되었으나, '노이즈 섞인 ASR 출력'과 '정답 텍스트'가 쌍을 이룬 대규모 지도 학습 데이터셋의 부족으로 인해 기존의 Neural LM 기반 rescoring 성능을 뛰어넘지 못했다. 따라서 본 연구의 목표는 대규모 합성 데이터를 통해 ASR 오류를 효과적으로 수정할 수 있는 Denoising LM(DLM)을 구축하고, 이를 통해 ASR의 성능을 SOTA(State-of-the-Art) 수준으로 끌어올리는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Text-to-Speech(TTS) 시스템을 활용하여 대규모의 합성 ASR 오류 데이터를 생성**함으로써 데이터 부족 문제를 해결하는 것이다. 구체적인 핵심 기여 사항은 다음과 같다.

1. **합성 데이터 파이프라인 구축**: TTS로 오디오를 생성하고 이를 다시 ASR로 인식시켜 '노이즈 섞인 가설-정답 텍스트' 쌍을 대량으로 생성하여 DLM을 학습시켰다.
2. **모델 및 데이터의 확장(Scaling)**: 모델 파라미터 수, 학습 텍스트 코퍼스의 크기, 그리고 TTS에서 생성하는 화자(Speaker)의 수를 대폭 늘려 성능의 확장성을 증명하였다.
3. **다양한 노이즈 증강 전략**: Multi-speaker TTS 사용, 주파수 마스킹(Frequency Masking), 랜덤 문자 치환(Random Substitution) 및 실제 ASR 데이터를 혼합하여 DLM이 실제 환경의 오류 분포를 학습하게 하였다.
4. **DSR-decoding 기법 제안**: DLM의 출력과 ASR의 acoustic score를 결합하여 최적의 전사 결과를 도출하는 새로운 디코딩 방식을 제안하였다.

## 📎 Related Works

기존의 ASR-LM 통합 방식은 크게 두 가지 방향으로 진행되었다. 하나는 ASR의 acoustic score와 LM score를 단순 결합하는 방식이며, 다른 하나는 LM의 특징을 ASR 모델의 레이어에 통합하는 방식이다.

최근에는 ASR의 1차 출력을 후처리하는 Error Correction 모델들이 등장하였으며, Transformer 기반의 seq2seq 모델이나 BART와 같은 사전 학습 모델을 파인튜닝하는 방식이 사용되었다. 또한, ChatGPT나 LLaMA와 같은 대규모 언어 모델(LLM)을 이용한 교정 시도도 있었다. 그러나 이러한 접근법들은 대부분 실제 ASR 오류 데이터의 부족으로 인해, 전통적인 Neural LM 기반의 rescoring 성능을 넘어서지 못했다는 한계가 있다. 본 논문은 TTS를 통한 데이터 합성이라는 전략을 통해 이 한계를 극복하고 EC 모델이 Neural LM을 능가할 수 있음을 보여준다.

## 🛠️ Methodology

### 전체 시스템 구조

본 논문에서 제안하는 Denoising Speech Recognition(DSR)은 ASR 모델과 DLM의 캐스케이드(Cascade) 구조로 정의된다. ASR 모델이 오디오 $x$로부터 노이즈가 섞인 시퀀스 $\tilde{y}$를 생성하면, DLM이 이를 다시 깨끗한 텍스트 $y$로 변환한다. 전체 확률 모델은 다음과 같이 표현된다.

$$ \log p_{DSR}(y|x) = \log \left( \sum_{\tilde{y}} p_{DLM}(y|\tilde{y}) p_{ASR}(\tilde{y}|x) \right) $$

### 학습 절차 및 데이터 생성

데이터 부족 문제를 해결하기 위해 다음과 같은 파이프라인으로 학습 데이터를 생성한다.

1. **TTS 생성**: 텍스트 코퍼스 $y$를 Multi-speaker TTS 시스템에 입력하여 합성 오디오 $\tilde{x}$를 생성한다.
2. **ASR 인식**: 생성된 오디오 $\tilde{x}$를 ASR 모델에 통과시켜 노이즈가 섞인 가설 $\hat{y}$를 얻는다.
3. **DLM 학습**: 생성된 $(\hat{y}, y)$ 쌍을 사용하여 Transformer 기반의 seq2seq 모델인 DLM을 Cross-entropy loss로 학습시킨다.

### DSR-decoding

단순한 Greedy-decoding(ASR $\rightarrow$ DLM 순차 적용)을 넘어, Acoustic 정보를 다시 활용하는 **DSR-decoding**을 제안한다. 절차는 다음과 같다.

1. ASR 모델로부터 Greedy 가설 $\hat{y}$를 생성한다.
2. DLM이 $\hat{y}$를 입력받아 Beam Search를 통해 $n$-best 후보군을 생성한다.
3. 각 후보 $y$에 대해 ASR 모델의 acoustic score를 계산한다.
4. 최종 결과는 DLM score와 ASR score의 가중 합으로 결정한다.

$$ y^* = \arg \max_{y \in n\text{-best}[p_{DLM}(\cdot|\hat{y})]} \lambda \log p_{DLM}(y|\hat{y}) + \log p_{ASR}(y|x) $$
여기서 $\lambda$는 두 모델의 상대적 강도를 조절하는 하이퍼파라미터이다.

## 📊 Results

### 실험 설정

- **데이터셋**: LibriSpeech (test-clean, test-other)
- **ASR 모델**: Transformer-CTC (255M params) 및 다양한 아키텍처(Quartznet, Conformer, Whisper)
- **DLM 모델**: Transformer seq2seq (최대 484M params)
- **지표**: Word Error Rate (WER)

### 주요 결과

1. **SOTA 달성**: `baseline ASR (LS+TTS)` 모델에 `DSR-decoding`을 적용했을 때, LibriSpeech에서 **test-clean 1.5%, test-other 3.3%**라는 뛰어난 WER을 기록하였다. 이는 외부 오디오 데이터를 사용하지 않은 설정에서 최상위 성능이며, 대규모 외부 데이터를 사용한 self-supervised 모델들과 대등한 수준이다.
2. **Neural LM 대비 우위**: 동일한 규모의 Neural LM rescoring보다 DLM 기반의 DSR-decoding이 훨씬 낮은 WER을 보였다. 특히 DLM은 Greedy-decoding만으로도 기존 Neural LM의 무거운 Beam search 결과를 맞먹는 성능을 냈다.
3. **범용성(Universality)**: 특정 ASR로 학습된 DLM이 Quartznet, Conformer, Whisper 등 다른 구조의 ASR 모델 출력값도 효과적으로 교정할 수 있음을 확인하였다. 또한 TED-LIUM과 같은 도메인 외 데이터셋에서도 성능 향상을 보였다.
4. **확장성(Scalability)**: 모델 크기, 텍스트 코퍼스 크기, 그리고 합성 오디오 생성 시 사용된 화자(Speaker)의 수가 증가할수록 WER이 지속적으로 감소하는 경향을 보였다.

## 🧠 Insights & Discussion

### 분석 및 강점

본 연구는 ASR 오류 교정 모델이 성공하기 위해 단순한 데이터 양의 증가보다 **'오류 분포의 다양성'**이 중요함을 시사한다. 고품질의 TTS(예: Tacotron)보다 다소 품질이 낮더라도 다양한 스타일과 화자를 생성할 수 있는 TTS가 DLM 학습에 더 유리했다. 이는 모델이 단순히 정답을 맞히는 것이 아니라, ASR이 범할 수 있는 다양한 형태의 '실수'를 학습해야 하기 때문이다.

### 한계 및 비판적 해석

- **CTC 모델 중심**: 실험이 주로 CTC 기반 ASR에 집중되어 있어, Encoder-Decoder 구조의 ASR 모델에서도 동일한 효과가 나타날지에 대한 추가 검증이 필요하다.
- **합성 데이터 의존성**: TTS-ASR 파이프라인을 통해 데이터를 생성하므로, TTS 시스템 자체가 가진 편향(Bias)이 ASR 시스템으로 전이될 가능성이 있다.
- **추론 효율성**: DSR-decoding은 성능은 좋으나, 여전히 DLM의 Beam search 과정이 포함되어 있어 실시간 스트리밍 시스템에 적용하기에는 최적화가 더 필요할 것으로 보인다.

## 📌 TL;DR

본 논문은 TTS를 이용해 대규모 합성 ASR 오류 데이터를 생성하고, 이를 학습한 **Denoising LM(DLM)**을 통해 ASR 성능을 극대화하는 방법을 제안한다. 제안된 **DSR-decoding** 방식은 기존의 Neural LM 기반 rescoring을 뛰어넘는 SOTA 성능(LibriSpeech test-clean 1.5%)을 달성하였으며, 모델 크기와 데이터 규모에 따른 확장성 및 다양한 ASR 모델에 대한 범용성을 입증하였다. 이 연구는 향후 ASR 시스템에서 전통적인 LM을 대체하여 오류 교정 중심의 새로운 패러다임을 제시할 가능성이 크다.
