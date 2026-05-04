# MM-KWS: Multi-modal Prompts for Multilingual User-defined Keyword Spotting

Zhiqi Ai, Zhiyong Chen, Shugong Xu (2024)

## 🧩 Problem to Solve

본 논문은 사용자가 직접 정의한 키워드를 인식하는 User-defined Keyword Spotting (UDKWS) 문제를 해결하고자 한다. 기존의 전통적인 Keyword Spotting (KWS) 시스템은 "Ok Google"이나 "Hey Siri"와 같이 미리 정의된 키워드를 인식하기 위해 방대한 데이터셋과 모델 학습이 필요하며, 새로운 키워드를 추가할 때마다 높은 비용이 발생한다.

최근의 UDKWS 연구는 소수의 예시만을 사용하여 새로운 키워드를 검출하려 시도하고 있으며, 크게 두 가지 접근 방식이 존재한다. 첫 번째는 사용자가 음성 템플릿을 등록하는 Query-by-Audio (QbyA) 방식이나, 이는 녹음 환경의 일관성에 크게 의존하며 등록 과정이 번거롭다는 단점이 있다. 두 번째는 텍스트를 이용하는 Query-by-Text (QbyT) 방식인데, 이는 사용자 친화적이지만 화자의 억양이나 발음 오류가 있을 때 성능이 저하되는 한계가 있다. 따라서 본 논문의 목표는 텍스트와 음성이라는 두 가지 모달리티(Multi-modal)를 모두 활용하여, 발음 오류에 강건하면서도 효율적인 다국어 UDKWS 시스템을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 텍스트와 음성 템플릿을 모두 활용하는 Multi-modal Prompt를 도입하여, 각 모달리티의 상호 보완적인 특성을 이용해 키워드 검출 성능을 높이는 것이다.

주요 기여 사항은 다음과 같다:

1. **MM-KWS 프레임워크 제안**: 텍스트와 음성 템플릿을 모두 enrollment(등록) 단계에서 활용하는 새로운 multi-modal prompt 기반 UDKWS 방법을 제안한다.
2. **다국어 확장성 확보**: 다국어 사전 학습 모델(Multilingual pre-trained models)을 통합하여 영어와 중국어(Mandarin) 모두에서 높은 성능을 입증하였다.
3. **Hard Case Mining 기법 도입**: 발음이나 의미가 유사하여 혼동하기 쉬운 'Confusable words'를 생성하고 이를 학습에 활용하는 데이터 증강 기법을 통해 모델의 변별력을 강화하였다.
4. **WenetPhrase 데이터셋 구축**: 중국어 KWS 평가를 위해 Confusable words를 포함한 새로운 데이터셋인 WenetPhrase를 제안하여 기존 데이터셋의 공백을 메웠다.

## 📎 Related Works

기존의 UDKWS 접근 방식과 그 한계점은 다음과 같다:

- **LVCSR 기반 방식**: 대규모 어휘 연속 음성 인식 시스템을 통해 오디오를 래티스(lattice)로 변환하여 검색한다. 그러나 사전 정의된 어휘집(vocabulary)에 의존하므로, 어휘집에 없는 단어(Out-of-Vocabulary, OOV)에 대해서는 성능이 급격히 떨어진다.
- **QbyA (Query-by-Audio)**: 사전 학습된 acoustic model로 임베딩을 추출하고 Dynamic Time Warping (DTW) 등을 통해 유사도를 측정한다. 하지만 화자의 녹음 일관성에 따라 성능 편차가 크고 등록 과정이 복잡하다.
- **QbyT (Query-by-Text)**: 텍스트와 음성 간의 대응 관계를 학습하여 매칭한다. 등록 과정은 편리하지만, 화자의 억양이나 오발음(Mispronunciation)이 발생할 경우 매칭 실패 확률이 높다는 치명적인 약점이 있다.

MM-KWS는 이러한 QbyA와 QbyT의 단점을 극복하기 위해 두 모달리티의 임베딩을 동시에 추출하고 이를 결합하여 판별함으로써 강건성을 높였다.

## 🛠️ Methodology

### 1. 전체 아키텍처

MM-KWS는 크게 세 가지 모듈로 구성된다: **Feature Extractor**, **Pattern Extractor**, 그리고 **Pattern Discriminator**이다.

### 2. 상세 구성 요소

#### (1) Feature Extractor

- **Query Branch**: 입력된 쿼리 음성(Query speech)을 $\text{Conformer}$ 아키텍처 기반의 Audio Encoder를 통해 음성 임베딩 $E_{q}^{a} \in \mathbb{R}^{T_{q}^{a} \times d}$로 변환한다.
- **Support Branch**: 등록된 템플릿에서 세 가지 종류의 임베딩을 추출한다.
  - **Phoneme Embedding ($E_{s}^{p}$)**: Multilingual G2P 모델을 사용하여 텍스트를 음소 단위로 변환한다.
  - **Text Embedding ($E_{s}^{t}$)**: Multilingual DistilBERT를 사용하여 텍스트 임베딩을 추출한다.
  - **Speech Embedding ($E_{s}^{a}$)**: 고성능 다국어 음성 모델인 $\text{XLR-S}$를 사용하여 음성 템플릿 임베딩을 추출한다.
- 모든 임베딩은 경량 Mapper를 통해 동일한 차원 $d=128$로 표준화된다.

#### (2) Pattern Extractor

Self-attention 메커니즘을 통해 쿼리와 서포트 간의 교차 모달 매칭을 수행한다.

- **QTAM (Query-by-Text Attention Module)**: 쿼리 음성($E_{q}^{a}$), 서포트 음소($E_{s}^{p}$), 서포트 텍스트($E_{s}^{t}$)를 입력으로 받는다. 각 입력의 출처를 구분하기 위해 learnable coding vector $e_{\text{type}}$와 sinusoidal position encoding $e_{\text{pos}}$를 더해준다:
  $$E = E + e_{\text{pos}} + e_{\text{type}}$$
  이후 이를 시간축으로 연결(concatenate)하여 $E_{c}^{ta}$를 구성하고 self-attention을 통해 joint feature $E_{j}^{ta}$를 계산한다:
  $$E_{c}^{ta} = (E_{q}^{a}; E_{s}^{p}; E_{s}^{t}) \in \mathbb{R}^{(T_{q}^{a}+T_{s}^{p}+T_{s}^{t}) \times d}$$
  $$E_{j}^{ta} = \text{Attention}(E_{c}^{ta}, E_{c}^{ta}, E_{c}^{ta})$$
- **QAAM (Query-by-Audio Attention Module)**: 쿼리 음성($E_{q}^{a}$)과 서포트 음성($E_{s}^{a}$)만을 입력으로 하여 유사도를 측정하며, 과정은 QTAM과 동일하게 self-attention을 통해 $E_{j}^{aa}$를 생성한다.

#### (3) Pattern Discriminator

- QTAM과 QAAM에서 도출된 utterance-level posterior probabilities $P_{utt}^{t}$와 $P_{utt}^{a}$를 결합하여 최종 결정 $P_{utt}$를 내린다. 이때 텍스트 기반 출력이 더 안정적이므로 이를 중심으로 하되, 음성 기반 출력을 보조적으로 활용한다:
  $$P_{utt} = \sigma(W_{u} \cdot (P_{utt}^{t} + P_{utt}^{a}) + b_{u})$$
- 또한, QTAM의 joint embedding에서 음소 단위($P_{phon}$) 및 단어 단위($P_{text}$)의 매칭 확률을 별도로 계산한다.

### 3. 학습 절차 및 손실 함수

최종 손실 함수 $L_{total}$은 세 가지 Binary Cross-Entropy (BCE) 손실의 합으로 정의된다:
$$L_{total} = L_{utt} + L_{phon} + L_{text}$$

- $L_{utt}$: 쿼리 음성이 타겟 키워드인지 여부를 판별하는 주 손실 함수이다.
- $L_{phon}, L_{text}$: 쿼리 음성 내에 타겟 음소 또는 단어가 존재하는지를 판별하는 보조 손실 함수이다.

### 4. Hard Case Mining을 통한 데이터 증강

혼동하기 쉬운 단어(Confusable words)에 대한 변별력을 높이기 위해 2단계 증강을 수행한다.

- **Stage 1 (단어 생성)**: G2P 모델을 이용해 음성적으로 유사한 단어(Edit distance 기준)와 DistilBERT를 이용해 의미적으로 유사한 단어(Cosine similarity 기준)를 찾아내어 Negative instance로 생성한다. 또한 LLM을 활용해 실제 발생 가능한 시나리오 기반의 부정 예시를 생성한다.
- **Stage 2 (음성 합성)**: 생성된 텍스트에 대응하는 오디오가 없으므로, 다국어 $\text{ZS-TTS}$ (Zero-Shot TTS) 모델을 사용하여 음성 데이터를 합성한다. 이를 통해 영어 150만 건, 중국어 240만 건의 데이터를 추가 학습시켰다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: LibriPhrase (English), WenetPhrase (Mandarin, 신규), Speech Commands (SPC, Zero-shot 평가용).
- **평가 지표**: Equal Error Rate (EER $\downarrow$), Area Under the ROC Curve (AUC $\uparrow$).
- **비교 대상**: Whisper-Tiny/Small/Large, AdaKWS, PhonMatchNet, CED 등.

### 2. 주요 결과

- **LibriPhrase**: MM-KWS는 특히 어려운 데이터셋인 LH(Hard) subset에서 매우 뛰어난 성능을 보였다. 특히 Hard case mining을 적용한 $\text{MM-KWS}^*$는 AUC $96.25\%$, EER $9.30\%$를 기록하며, 더 많은 파라미터를 가진 AdaKWS보다 우수한 성능을 보였다.
- **WenetPhrase**: 중국어 데이터셋에서도 강점을 보였다. WE(Easy)에서는 AUC $99.79\%$, EER $1.95\%$를 달성했으며, WH(Hard)에서는 $\text{MM-KWS}^*$가 AUC $85.84\%$를 기록했다. 특히 ASR 기반 시스템(Whisper, FunASR) 대비 추론 지연 시간(Latency)이 $6\text{ms}$로 매우 짧아 효율적이다.
- **Zero-shot (SPC)**: 등록된 음성 데이터가 전혀 없는 상태(0 supports)에서도 텍스트 입력만으로 $\text{Acc(open)} 88.4\%$라는 높은 성능을 보여, QbyA 기반의 Baseline(66.0%)을 압도하였다.

### 3. Ablation Study 및 시각화

- **Ablation**: Confusable keywords generation, Support speech branch, Auxiliary loss 세 가지 요소 모두가 성능 향상에 기여함을 확인하였다. 특히 Hard case mining을 제거했을 때 LH subset의 EER이 $9.30\% \to 12.45\%$로 상승하여 그 중요성이 입증되었다.
- **시각화**: Attention map 분석 결과, 타겟 키워드가 포함된 경우 쿼리 음성과 서포트 임베딩 간에 명확한 단조성(monotonicity)이 나타나며, 부정 예시에서는 이러한 패턴이 사라짐을 확인하였다.

## 🧠 Insights & Discussion

본 논문의 강점은 단순히 텍스트와 음성을 합친 것이 아니라, **음소(Phoneme) 수준의 정보**를 통합하고 **Hard case mining**을 통해 모델이 가장 취약한 지점(유사 발음 단어)을 집중적으로 학습시킨 점에 있다. 특히 다국어 사전 학습 모델을 Freeze 상태로 사용함으로써 추가적인 계산 비용 없이 강력한 feature extraction 능력을 확보한 점이 효율적이다.

다만, 본 연구는 고성능의 사전 학습 모델(XLR-S, BERT 등)과 대규모 합성 데이터(TTS)에 의존하고 있다. 실제 환경에서 TTS로 생성된 데이터가 실제 화자의 다양한 변이(Variation)를 모두 커버할 수 있는지는 추가적인 검증이 필요하다. 또한, 모델의 경량화에 대한 언급은 있으나 구체적인 온디바이스(on-device) 배포 성능 수치는 제시되지 않았다.

## 📌 TL;DR

MM-KWS는 텍스트, 음소, 음성이라는 세 가지 모달리티의 임베딩을 모두 활용하여 사용자가 정의한 키워드를 검출하는 다국어 KWS 시스템이다. 특히 발음/의미가 유사한 단어들을 생성하여 학습시키는 Hard case mining 기법을 통해 변별력을 극대화하였으며, 영어와 중국어 데이터셋에서 기존 SOTA 모델들을 능가하는 성능과 매우 낮은 지연 시간을 달성하였다. 이 연구는 향후 온디바이스 환경에서의 유연한 맞춤형 키워드 인식 서비스 구현에 중요한 기여를 할 것으로 보인다.
