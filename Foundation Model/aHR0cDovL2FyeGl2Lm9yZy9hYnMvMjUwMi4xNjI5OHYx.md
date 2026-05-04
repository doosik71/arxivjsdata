# voc2vec: A Foundation Model for Non-Verbal Vocalization

Alkis Koudounas, Moreno La Quatra, Sabato Marco Siniscalchi, Elena Baralis (2025)

## 🧩 Problem to Solve

본 논문은 인간의 음성 데이터 중 언어적 의미를 담고 있지 않은 **비언어적 발성(Non-verbal vocalization)**, 즉 'vocal bursts'(웃음, 한숨, 비명, 신음 등)를 효과적으로 인식하고 처리하기 위한 모델의 부재 문제를 해결하고자 한다.

기존의 Speech Foundation Model(예: Wav2Vec 2.0, HuBERT, WavLM)은 대규모 언어 데이터로 학습되어 음성 운율(Speech prosody) 처리에는 능숙하지만, 비언어적 소리에 담긴 미묘한 정서적 특성을 포착하는 데는 한계가 있다. 반면, 일반적인 Audio Foundation Model(예: AudioSet 기반 모델)은 비언어적 데이터를 다룰 수는 있으나, 인간 발성 특유의 세밀한 뉘앙스를 구분하는 성능이 부족하다.

따라서 본 연구의 목표는 오픈소스 비언어적 오디오 데이터셋을 활용하여, 비언어적 발성 인식 작업에 특화된 범용 표현 학습 모델인 **voc2vec**을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 비언어적 인간 발성 데이터만을 전문적으로 다루는 **최초의 범용 표현 모델(Universal representation model)**을 구축했다는 점이다.

중심 설계 아이디어는 다음과 같다:

1. **특화된 데이터 큐레이션**: 총 10개의 다양한 오픈소스 비언어적 오디오 데이터셋을 수집하여 약 125시간 분량의 `VOC125` 데이터셋을 구축하고 이를 사전 학습(Pre-training)에 사용하였다.
2. **전이 학습 전략의 탐색**: 모델을 처음부터 학습시키는 방법뿐만 아니라, LibriSpeech(언어 중심) 및 AudioSet(일반 오디오 중심)으로 사전 학습된 모델에서 시작하여 비언어적 데이터로 추가 학습시키는 전략을 비교 분석하였다.
3. **범용성 입증**: 6개의 서로 다른 비언어적 발성 벤치마크 데이터셋에서 기존의 강력한 베이스라인 모델들(OpenSmile, emotion2vec 등)보다 우수한 성능을 보임을 입증하였다.

## 📎 Related Works

기존의 정서 컴퓨팅(Affective computing) 연구들은 주로 단어와 함께 나타나는 음성 운율(Speech prosody)에 집중해 왔으나, 최근 연구들은 비언어적 발성이 정서를 더 직접적이고 강력하게 전달한다는 점을 시사하고 있다.

하지만 다음과 같은 한계점이 존재했다:

- **데이터 부족**: 다양한 비언어적 소리를 캡처한 대규모 데이터셋이 부족하여, 기존 모델들은 주로 '웃음'과 같은 특정 범주에만 치중되어 있었다.
- **모델의 최적화 부족**: 기존의 SSL(Self-Supervised Learning) 모델들은 언어 데이터에 최적화되어 있어, 아기의 울음소리나 한숨의 의도와 같은 미묘한 음향적/맥락적 단서를 포착하는 데 어려움이 있었다.

voc2vec은 이러한 한계를 극복하기 위해 언어 기반 모델이 아닌, 비언어적 발성 전용 데이터셋을 통한 도메인 특화 사전 학습을 수행함으로써 차별점을 갖는다.

## 🛠️ Methodology

### 전체 아키텍처

voc2vec은 **wav2vec 2.0** 프레임워크를 기반으로 설계되었으며, 크게 두 가지 주요 구성 요소로 이루어져 있다.

1. **CNN Encoder**: raw audio 입력에서 저차원 잠재 표현(Latent representations)을 추출한다.
2. **Transformer Network**: 추출된 표현들 사이의 시간적 맥락 관계를 캡처하여 국소적(local) 특징과 전역적(global) 특징을 동시에 처리한다.

입력 raw audio를 $X=\{x_1, x_2, \dots, x_T\}$라고 할 때, CNN Encoder를 통해 다음과 같은 잠재 표현 시퀀스 $Z$가 생성된다:
$$Z = \text{Encoder}(X) = \{z_1, z_2, \dots, z_{T'}\}, \quad T' < T$$
여기서 $T'$는 인코딩된 시퀀스의 프레임 수이며, 각 $z_t \in \mathbb{R}^d$는 해당 프레임의 오디오 특징을 담은 $d$차원 벡터이다. 이후 Transformer는 이 $Z$를 입력으로 받아 문맥화된 벡터(Contextualized vectors) $C=\{c_1, c_2, \dots, c_{T'}\}$를 출력한다.

### 사전 학습 (Pre-training)

모델은 10개의 오픈소스 데이터셋(총 125시간)으로 구성된 `VOC125`를 사용하여 SSL 방식으로 학습되었다. 사용된 데이터셋은 AudioSet(vocalization split), FreeSound(babies), NNIME, NonSpeech7K, ReCANVo, VocalSound 등으로, 아기 울음, 웃음, 한숨, 기침, 비명 등 광범위한 비언어적 소리를 포함한다.

사전 학습 전략은 세 가지로 나뉜다:

- **Cold-start**: `VOC125` 데이터로 처음부터 학습.
- **voc2vec-ls**: LibriSpeech로 사전 학습된 모델에서 시작하여 `VOC125`로 추가 학습.
- **voc2vec-as**: AudioSet으로 사전 학습된 모델에서 시작하여 `VOC125`로 추가 학습.

### 파인튜닝 (Fine-tuning)

사전 학습된 Transformer 위에 랜덤하게 초기화된 출력 레이어(Output layer)를 추가하여 하위 작업(Downstream tasks)에 적응시킨다. 10-fold 교차 검증(Cross-validation)을 통해 성능을 평가하였으며, 배치 크기는 16, 최대 50 에포크(Epoch) 동안 학습을 진행하였다.

## 📊 Results

### 실험 설정

- **데이터셋**: ASVP-ESD, CNVVE, Donate a Cry, VIVAE 등 6개의 분류 작업 데이터셋을 사용하였다.
- **비교 대상**: wav2vec 2.0, HuBERT, WavLM, data2vec (각각 LibriSpeech 또는 AudioSet 사전 학습 버전), 그리고 특성 기반 모델인 OpenSmile과 emotion2vec.
- **평가 지표**: 불균형 데이터셋을 고려하여 Unweighted Average Recall (UAR), Accuracy, F1 Macro Score를 사용하였다.

### 주요 결과

1. **초기화 전략 분석**: `voc2vec-ls`(LibriSpeech 기반 $\rightarrow$ VOC125 학습)가 가장 높은 성능을 기록하였다. 이는 일반적인 오디오 데이터(AudioSet)보다 언어 데이터(LibriSpeech)로 학습된 모델이 비언어적 발성 학습을 위한 더 강력한 기초를 제공함을 의미한다.
2. **기존 모델 대비 성능**: `voc2vec-ls`는 기존 SSL 모델들보다 일관되게 우수한 성능을 보였다.
   - 차순위 모델 대비 평균적으로 **UAR 5%, Accuracy 2%, F1 Macro 4%**의 향상을 보였다.
   - 특히 OpenSmile과 emotion2vec 베이스라인에 비해 UAR 성능이 2배 이상 높게 나타났다.
3. **정성적 분석**: VIVAE 데이터셋에 대한 t-SNE 시각화 결과, `voc2vec-ls`가 wav2vec 2.0이나 HuBERT보다 클래스 내 응집도(Intra-cluster cohesion)가 높고 클래스 간 분리도(Inter-cluster separation)가 뚜렷한 클러스터링 양상을 보였다.

## 🧠 Insights & Discussion

본 연구는 비언어적 발성이라는 특정 도메인에 특화된 사전 학습이 모델의 표현 능력을 얼마나 향상시킬 수 있는지를 명확히 보여주었다. 특히, 완전히 새로운 도메인임에도 불구하고 기존의 언어 모델(LibriSpeech 기반)에서 시작하는 것이 성능상 유리했다는 점은, 인간의 발성 체계(vocal apparatus)에서 기인하는 공통적인 특징이 언어와 비언어 발성 사이에 존재함을 시사한다.

**강점**:

- 오픈소스 데이터만을 활용하여 범용적인 비언어 발성 모델을 구축함으로써 재현성과 접근성을 높였다.
- 다양한 하위 작업(아기 울음, 감정 인식 등)에서 일관된 성능 향상을 입증하였다.

**한계 및 논의**:

- 사전 학습 데이터의 총량(125시간)이 일반적인 Speech Foundation Model의 학습 규모에 비해 매우 작다. 따라서 더 대규모의 비언어적 데이터가 확보된다면 성능이 더욱 향상될 가능성이 크다.
- 본 논문에서는 분류(Classification) 작업에 집중하였으나, 향후 비언어적 발성의 생성(Generation)이나 더 복잡한 시퀀스 분석으로 확장될 필요가 있다.

## 📌 TL;DR

본 논문은 인간의 비언어적 발성(웃음, 울음, 한숨 등) 처리에 특화된 파운데이션 모델인 **voc2vec**을 제안한다. 125시간 분량의 오픈소스 비언어 오디오 데이터셋(`VOC125`)으로 사전 학습되었으며, 특히 LibriSpeech 사전 학습 모델을 기반으로 추가 학습했을 때 최적의 성능을 낸다. 실험 결과, 기존의 언어/오디오 파운데이션 모델 및 전문 베이스라인들을 압도하는 성능을 보였으며, 이는 향후 아동 발달 모니터링이나 정신 건강 진단과 같은 비언어적 음성 분석 분야에 크게 기여할 것으로 기대된다.
