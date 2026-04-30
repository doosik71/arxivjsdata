# AudioLM: a Language Modeling Approach to Audio Generation

Zalán Borsos, Raphaël Marinier, Damien Vincent, Eugene Kharitonov, Olivier Pietquin, Matt Sharifi, Dominik Roblek, Olivier Teboul, David Grangier, Marco Tagliasacchi, Neil Zeghidour (2023)

## 🧩 Problem to Solve

오디오 신호(음성, 음악, 환경음 등)는 매우 국소적인 음향적 특성부터 구문, 의미, 화자의 정체성, 음악의 화성과 리듬 같은 고차원적인 장기 구조(long-term structure)까지 다양한 척도의 추상화 수준을 가지고 있다. 기존의 오디오 생성 모델들은 WaveNet과 같은 자기회귀 모델이나 GAN, Diffusion 기반 모델을 통해 매우 높은 신호 품질(fidelity)을 달성했지만, 강한 조건부 입력(예: 텍스트 전사, MIDI 시퀀스)이 없을 경우 생성된 오디오가 구조적으로 일관되지 않은 '옹알이'와 같은 상태가 되는 문제가 있었다.

반면, 언어 모델(Language Model)은 텍스트나 이미지 생성에서 보여주었듯 복잡한 장기 의존성을 모델링하는 데 탁월한 능력을 갖추고 있다. 그러나 오디오를 단순한 이산 토큰으로 변환하여 언어 모델링을 적용할 경우, 재구성 품질(reconstruction quality)과 장기적 일관성(long-term consistency) 사이의 트레이드-오프가 발생한다. 즉, 고품질 복원을 위해 비트레이트를 높이면 시퀀스 길이가 길어져 모델이 장기 구조를 학습하기 어려워지고, 반대로 압축률을 높이면 구조적 일관성은 좋아지나 음질이 크게 저하된다. 본 논문의 목표는 이 두 가지 상충하는 목표를 동시에 달성하여, 텍스트 전사 없이도 고품질이며 장기적으로 일관된 오디오 생성을 가능하게 하는 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 오디오를 서로 다른 특성을 가진 두 종류의 토큰, 즉 **Semantic tokens**와 **Acoustic tokens**로 분리하여 계층적으로 모델링하는 하이브리드 토큰화(hybrid tokenization) 체계이다.

1.  **Semantic tokens**: w2v-BERT와 같은 자기지도학습(self-supervised learning) 모델의 중간 레이어 표현을 $k$-means 클러스터링하여 생성한다. 이는 음성에서는 음소와 의미적 내용을, 음악에서는 멜로디와 리듬 같은 고차원적 구조를 캡처한다.
2.  **Acoustic tokens**: SoundStream과 같은 신경망 오디오 코덱(neural audio codec)을 통해 생성한다. 이는 오디오 파형의 세부적인 음향적 특성을 캡처하여 고품질의 신호 복원을 가능하게 한다.

AudioLM은 먼저 Semantic tokens를 생성하고, 이를 조건으로 하여 Acoustic tokens를 단계적으로 예측하는 계층적 언어 모델링 방식을 채택함으로써, 의미적 일관성과 음향적 품질을 동시에 확보하였다.

## 📎 Related Works

기존의 오디오 합성 연구는 크게 고충실도 신호 합성과 자기지도 표현 학습의 두 갈래로 나뉜다. WaveNet이나 HiFi-GAN 등은 신호 수준의 품질을 높이는 데 집중했으나, 텍스트와 같은 강한 가이드 없이는 구조적 일관성을 유지하지 못했다. 한편, HuBERT나 w2v-BERT 같은 모델은 오디오에서 고차원적인 특징을 추출하는 데 성공했으나, 이러한 표현들은 손실이 커서 다시 오디오 파형으로 복원(invert)했을 때 품질이 매우 낮다.

최근에는 텍스트 없이 오디오 토큰만을 이용해 언어 모델링을 수행하는 "textless NLP" 접근 방식(예: GSLM)이 등장했다. 하지만 GSLM 등은 단일 화자의 깨끗한 음성으로 제한되거나 음질이 낮다는 한계가 있었다. AudioLM은 이러한 기존 연구들의 한계를 극복하기 위해, 구조를 담당하는 semantic 토큰과 품질을 담당하는 acoustic 토큰을 결합하는 전략을 사용하며, 이는 단순한 단일 토큰 시퀀스 모델링보다 훨씬 효율적이고 정교한 생성 능력을 제공한다.

## 🛠️ Methodology

### 1. 전체 파이프라인 및 구성 요소
AudioLM의 전체 시스템은 다음 세 가지 구성 요소로 이루어진다.
- **Tokenizer**: 입력 오디오 $x$를 이산 토큰 시퀀스 $h$로 매핑한다.
- **Transformer LM**: 디코더 전용(decoder-only) 트랜스포머 모델로, 이전 토큰들이 주어졌을 때 다음 토큰의 확률 $\prod_{t=1}^{T'} p(h_t | h_{<t})$를 최대화하도록 학습된다.
- **Detokenizer**: 예측된 토큰 시퀀스를 다시 오디오 파형 $\hat{x}$로 복원한다.

### 2. 하이브리드 토큰화 (Hybrid Tokenization)
- **Semantic tokens ($z$)**: w2v-BERT의 중간 레이어에서 추출한 임베딩에 $k$-means 클러스터링을 적용하여 얻는다. 샘플링 레이트는 $25\text{Hz}$이며, 저비트레이트(예: $250\text{bps}$)로 고차원적 의미를 담는다.
- **Acoustic tokens ($y$)**: SoundStream 코덱의 Residual Vector Quantizer(RVQ)를 통해 얻는다. 샘플링 레이트는 $50\text{Hz}$이며, 여러 층의 양자화기를 통해 고비트레이트의 세부 음향 정보를 담는다.

### 3. 계층적 모델링 단계 (Three-stage Modeling)
AudioLM은 세 단계의 트랜스포머 모델을 순차적으로 적용한다.

**단계 1: Semantic Modeling**
가장 먼저 의미적 토큰의 시퀀스를 자기회귀적으로 예측한다.
$$p(z_t | z_{<t})$$
이 단계는 전체적인 문맥, 구문, 화자의 말투(prosody) 및 음악의 멜로디와 같은 장기 구조를 결정한다.

**단계 2: Coarse Acoustic Modeling**
생성된 semantic tokens $z$를 조건으로, SoundStream의 상위 $Q'$개 레이어에서 생성된 coarse acoustic tokens를 예측한다.
$$p(y_{q,t} | z, y_{\le Q', <t}, y_{<q,t}) \quad \text{for } q \le Q'$$
여기서 $y_{q,t}$는 $t$번째 시점의 $q$번째 양자화기 토큰이다. 이 단계는 화자의 정체성과 녹음 환경 같은 거친 음향 특성을 결정한다.

**단계 3: Fine Acoustic Modeling**
앞서 예측된 coarse tokens $y_{\le Q'}$를 조건으로, 나머지 세부 레이어($Q'+1$부터 $Q$까지)의 fine acoustic tokens를 예측한다.
$$p(y_{q,t} | y_{\le Q'}, y_{>Q', <t}, y_{<q,t}) \quad \text{for } q > Q'$$
이 단계는 압축으로 인한 손실을 메우고 최종적인 오디오 품질을 극대화한다.

### 4. 추론 절차 (Inference)
- **무조건 생성**: Semantic 토큰부터 시작하여 3단계까지 순차적으로 샘플링한다.
- **Acoustic 생성**: 실제 오디오에서 추출한 ground-truth semantic 토큰을 입력으로 하여 acoustic 토큰들만 생성한다.
- **연속 생성(Continuation)**: 짧은 프롬프트(예: 3초)의 semantic 및 coarse acoustic 토큰을 입력으로 주어, 그 뒤에 이어질 semantic 토큰을 먼저 생성하고, 이를 바탕으로 acoustic 토큰들을 순차적으로 생성한다.

## 📊 Results

### 1. 실험 설정
- **데이터셋**: 음성의 경우 Libri-Light(60k 시간)를 사용하였으며, 음악의 경우 내부 피아노 데이터셋(40k 시간)을 사용하였다.
- **모델 구조**: 각 단계마다 12레이어, 16헤드, 임베딩 차원 1024의 디코더 전용 트랜스포머($0.3\text{B}$ 파라미터)를 사용하였다.
- **평가 지표**: ViSQOL(재구성 품질), ABX error(음소 판별력), sWUGGY/sBLIMP(언어적 지식), WER/CER(음성 인식 오류율) 등을 사용하였다.

### 2. 주요 결과
- **토큰 특성 분석**: Table I에서 semantic 토큰은 음소 판별력(ABX)이 매우 높지만 복원 품질(ViSQOL)이 낮고, acoustic 토큰은 복원 품질은 높지만 판별력이 낮음을 확인하였다. 이는 두 토큰을 모두 사용해야 함을 정당화한다.
- **언어적 지식**: sWUGGY와 sBLIMP 벤치마크에서 AudioLM은 텍스트 감독 없이 학습된 모델 중 가장 높은 성능을 보였으며, 일부 텍스트 기반 베이스라인보다도 뛰어난 성적을 거두었다(Table IV).
- **화자 정체성 유지**: 3초의 짧은 프롬프트만으로도 학습 데이터에 없던 새로운 화자의 목소리를 생성했을 때, 화자 분류기(Speaker Classifier)의 정확도가 $92\%$ 이상으로 나타나 화자의 정체성이 매우 잘 유지됨을 보였다.
- **주관적 평가**: 인간 평가자가 AudioLM이 생성한 음성과 실제 음성을 구분하는 정확도가 $51.2\%$로, 사실상 무작위 추측(50%)과 차이가 없었다.
- **피아노 음악 생성**: 음악 도메인에서도 동일한 계층적 구조를 적용했을 때, 단순 acoustic 모델링보다 멜로디와 리듬의 일관성이 훨씬 뛰어난 결과가 나타났다.

## 🧠 Insights & Discussion

### 1. 강점
AudioLM의 가장 큰 강점은 오디오의 **'의미'와 '음향'을 완전히 분리(disentanglement)**하여 모델링했다는 점이다. 이를 통해 텍스트라는 명시적인 가이드 없이도 언어 모델의 강력한 문맥 파악 능력을 오디오 생성에 그대로 이식하였다. 특히 zero-shot으로 처음 듣는 화자의 목소리와 톤을 유지하며 말을 이어가는 능력은 매우 인상적이다.

### 2. 한계 및 비판적 해석
- **계산 복잡도**: 3단계의 모델을 순차적으로 실행해야 하므로 추론 시간이 길어질 수 있다.
- **데이터 편향**: 거대 데이터셋으로 학습된 언어 모델의 특성상, 학습 데이터에 포함된 사회적 편향이나 특정 억양/방언의 누락 문제가 발생할 수 있다.
- **남용 가능성**: 매우 사실적인 음성 합성이 가능하므로, 생체 인식 보안 시스템을 무력화하거나 타인을 사칭하는 딥페이크(Deepfake) 위험이 존재한다.

논문에서는 이러한 위험을 인지하고, AudioLM이 생성한 음성을 $98.6\%$의 정확도로 탐지할 수 있는 분류기를 함께 제안함으로써 책임감 있는 AI 개발 태도를 보였다.

## 📌 TL;DR

AudioLM은 오디오를 **Semantic tokens(구조/의미 담당)**와 **Acoustic tokens(품질/세부음향 담당)**로 나누어 계층적으로 생성하는 프레임워크이다. 텍스트 전사 없이 오직 raw audio만으로 학습되었음에도 불구하고, 인간이 실제 음성과 구분하기 어려울 정도의 고품질 음성을 생성하며 화자의 정체성과 장기적인 문맥을 완벽하게 유지한다. 이 연구는 음성뿐만 아니라 피아노 음악 생성에서도 성공적인 결과를 보였으며, 향후 TTS(Text-to-Speech)나 음성-음성 번역 등 다양한 조건부 생성 작업으로 확장될 가능성이 매우 높다.