# SpeechPrompt: Prompting Speech Language Models for Speech Processing Tasks

Kai-Wei Chang, Haibin Wu, Yu-Kai Wang, Yuan-Kuei Wu, Hua Shen, Wei-Cheng Tseng, Iu-thing Kang, Shang-Wen Li, Hung-yi Lee (2024)

## 🧩 Problem to Solve

본 논문은 음성 처리 분야에서 전통적으로 사용되어 온 "사전 학습 후 미세 조정(pre-train, fine-tune)" 패러다임이 가진 한계를 해결하고자 한다. 기존 방식은 각 다운스트림 작업(downstream task)마다 전문가가 직접 작업별 모델 아키텍처를 설계하고 손실 함수를 정의해야 하므로, 막대한 인적 자원과 시간이 소모된다. 또한, 작업의 수가 증가함에 따라 각 작업 전용 모델의 파라미터를 개별적으로 저장하고 학습시켜야 하므로 계산 및 저장 비용이 기하급수적으로 증가하는 문제가 발생한다.

따라서 본 연구의 목표는 자연어 처리(NLP) 분야에서 성공적으로 적용된 Prompting 기법을 텍스트가 없는(textless) 음성 언어 모델(Speech LM)에 도입하여, 단일한 프레임워크 내에서 다양한 음성 처리 작업을 효율적으로 수행하는 방법을 제시하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 모든 음성 처리 작업을 **'음성-단위 생성(speech-to-unit generation)'** 작업으로 재정의하는 것이다. 이를 통해 모델 아키텍처의 변경 없이 입력 프롬프트만을 조정하여 다음과 같은 통합된 시스템을 구축하였다.

1.  **통합 프롬프트 프레임워크**: 음성 분류, 시퀀스 생성, 음성 생성 작업을 단일한 생성 프로세스로 통합하였다.
2.  **Textless Speech LM 활용**: 텍스트 데이터 없이 SSL(Self-Supervised Learning) 표현형을 양자화한 이산 단위(discrete units)만을 사용하여 모델을 구동함으로써, 텍스트 자원이 부족한 언어에서도 적용 가능하게 하였다.
3.  **학습 가능한 Verbalizer(Learnable Verbalizer) 제안**: 이산 단위와 다운스트림 레이블 간의 의미적 간극을 메우기 위해, 단순한 매핑이 아닌 학습 가능한 선형 변환 층을 도입하여 성능과 설명 가능성을 높였다.

## 📎 Related Works

### 1. 자기지도 학습(SSL) 및 이산화
최근 HuBERT와 같은 SSL 모델을 통해 음성 표현을 학습하고, 이를 K-means 클러스터링 등으로 양자화하여 이산 단위(discrete units)로 변환하는 연구가 활발하다. 이산 단위는 원본 파형(waveform)보다 저장 공간을 획기적으로 줄이며, 화자 정보(speaker-specific information)를 최소화하여 프라이버시 보호에 유리하다는 장점이 있다.

### 2. Textless Speech Language Models
GSLM이나 Unit mBART와 같은 모델들은 이산 단위를 '가상 텍스트(pseudo-text)'로 간주하여 다음 토큰 예측(next-token prediction)이나 디노이징(denoising) 작업을 통해 사전 학습된다.

### 3. 기존 Prompting 및 Reprogramming과의 차이점
- **WavPrompt**: 텍스트 LM(GPT-2)과 오디오 인코더를 결합하여 사용하지만, 본 논문은 텍스트가 전혀 없는 Textless Speech LM을 사용하여 범위가 더 넓다.
- **Whisper Prompting**: 텍스트-음성 쌍으로 학습된 모델을 사용하지만, 본 논문은 텍스트 감독 없이 학습된 모델을 사용하며 음성 생성 작업까지 확장하였다.
- **Model Reprogramming**: 입력 데이터를 변환하여 기존 모델을 재사용하는 기법으로, 본 논문의 Verbalizer와 유사한 역할을 수행하지만 본 연구는 프롬프트 튜닝과 결합하여 더 체계적인 프레임워크를 제안한다.

## 🛠️ Methodology

### 1. 전체 파이프라인
입력 음성 파형은 SSL 모델과 양자화기(Quantizer)를 통해 이산 단위 시퀀스 $u_x$로 변환된다. 이후 Unit LM이 작업별 프롬프트 $p$를 입력받아 타겟 단위 시퀀스 $u_y$를 생성하며, 마지막으로 Verbalizer나 Speech Decoder를 통해 최종 결과물(레이블 또는 음성)을 도출한다.

### 2. Unit Language Models (Backbone)
본 연구에서는 두 가지 Transformer 기반 구조를 사용한다.
- **Decoder-only LM (GSLM)**: GPT와 유사하며, 소스 음성 $u_x$와 $\langle \text{sep} \rangle$ 토큰 뒤에 타겟 $u_y$를 생성한다.
- **Encoder-Decoder LM (Unit mBART)**: BART와 유사하며, 인코더가 $u_x$를 처리하고 디코더가 이를 참조하여 $u_y$를 생성한다.

두 모델 모두 자기회귀(autoregressive) 방식으로 작동하며, $t$ 시점의 단위 $u_j$ 생성 확률은 다음과 같다.
$$P(u_j | C_t) = \frac{e^{z_{tj}}}{\sum_{k=1}^{|V|} e^{z_{tk}}}$$
여기서 $C_t$는 입력 음성, 프롬프트, 그리고 이전까지 생성된 단위들의 집합이며, $z_{tj}$는 $j$-번째 단위의 로짓(logit)이다.

### 3. Prompt Tuning
모델의 파라미터는 고정(freeze)하고, 오직 프롬프트 벡터만을 학습시킨다.
- **Input Prompt Tuning**: 입력 임베딩 시퀀스의 앞에 학습 가능한 연속 벡터 $p_I$를 추가한다.
  $$h^{(1)} \leftarrow \text{Concat}(p_I, h^{(1)})$$
- **Deep Prompt Tuning (Prefix-Tuning)**: 각 Transformer 레이어의 Attention 모듈의 Key($K$)와 Value($V$) 앞에 학습 가능한 벡터 $p_K, p_V$를 추가하여 모델의 추론 과정을 정밀하게 제어한다.
  $$K \leftarrow \text{Concat}(p_K, h)W_K, \quad V \leftarrow \text{Concat}(p_V, h)W_V$$

### 4. Learnable Verbalizer 및 입력 변환
이산 단위는 텍스트와 달리 명확한 의미가 없으므로, 학습 가능한 선형 변환 행렬 $W \in \mathbb{R}^{|Y| \times |V|}$를 도입하여 로짓 $z_t$를 레이블 공간 $\hat{z}_t$로 매핑한다.
$$\hat{z}_t = W \cdot z_t$$
또한, 예측된 레이블을 다시 모델의 입력으로 사용하기 위해, 원본 단어장 임베딩 $e$를 가중합 하여 클래스 전용 임베딩 $\hat{e}(y)$를 생성하는 입력 변환 메커니즘을 제안한다.
$$\hat{e}(y) = \sum_{i=1}^{|V|} \left( \text{softmax} \left( \frac{W_{y:}}{\tau} \right) \right)_i \cdot e(u_i)$$
여기서 $W_{y:}$는 행렬 $W$의 $y$번째 행이며, $\tau$는 온도(temperature) 파라미터이다.

## 📊 Results

### 1. 실험 설정
- **데이터셋**: Google Speech Commands(음성 명령), LibriSpeech(ASR, PR), CoVoST2(음성 번역), LJSpeech(음성 계속 생성) 등 다양한 언어와 작업 포함.
- **비교 대상**: "Pre-train, Fine-tune" (FT) 패러다임 (SSL 모델 + 작업별 전문가 모델).
- **평가 지표**: Accuracy(분류), WER/CER/PER(시퀀스 생성), BLEU/MOS/SIM(음성 생성).

### 2. 주요 결과
- **음성 분류(Speech Classification)**: 프롬프팅(PT) 방식이 FT 방식과 대등하거나 오히려 능가하는 성능을 보였다. 특히 mHuBERT + Unit mBART 조합에서 10개 데이터셋 중 8개에서 FT보다 우수한 성능을 기록하였다.
- **시퀀스 생성(Sequence Generation)**: Decoder-only 모델(GSLM)은 FT 대비 성능이 낮았으나, Encoder-Decoder 모델(Unit mBART)은 대부분의 시나리오에서 FT를 능가하거나 대등한 성능을 보였다. 특히 Learnable Verbalizer를 사용했을 때 성능 향상이 뚜렷했다.
- **음성 생성(Speech Generation)**: FT 방식으로는 구현이 어려운 음성 번역(ST)과 음성 계속 생성(SC) 작업에서 PT 방식이 유의미한 결과를 냈으며, 특히 Unit mBART는 자연스러운 음성을 생성하였다.
- **Few-shot Learning (10-shot)**: 데이터가 극도로 적은 상황에서 PT 방식이 FT 방식을 압도하였다. 이는 LM이 사전 학습 단계에서 습득한 풍부한 사전 지식을 프롬프트가 효과적으로 인출해내기 때문으로 분석된다.

## 🧠 Insights & Discussion

### 1. 모델 아키텍처의 중요성
본 연구는 음성 처리에서 Decoder-only 모델보다 **Encoder-Decoder 모델이 더 유리함**을 시사한다. 이는 음성 신호라는 연속적인 모달리티를 먼저 압축된 표현으로 인코딩한 후, 이를 바탕으로 타겟(텍스트, 클래스, 음성)을 생성하는 구조가 음성-텍스트 간 변환 작업에 더 적합하기 때문으로 추측된다.

### 2. Verbalizer의 분석 및 설명 가능성
Learnable Verbalizer의 가중치 $W$를 시각화한 결과, 특정 캐릭터(예: 'B')나 음소(phoneme)와 강하게 연결된 이산 단위들이 실제로 해당 음소와 높은 상관관계를 가진다는 것이 확인되었다. 이는 Verbalizer가 단순히 수치적인 매핑을 넘어 음성의 음향-음성학적 특성을 학습하고 있음을 보여준다.

### 3. 한계점 및 향후 방향
- **성능 격차**: 시퀀스 생성 작업의 경우, 여전히 최첨단(SOTA) SSL 기반 전문가 모델들보다는 성능이 낮다. 이는 더 강력한 Speech LM의 개발이 필요함을 의미한다.
- **정보 손실**: 양자화 과정에서 운율(prosody), 화자의 감정 등의 정보가 일부 손실되어 액센트 분류(AcC) 등의 작업에서 성능 저하가 나타났다. 향후 화자 및 감정 정보를 보존하는 플러그인 모듈의 도입이 필요하다.
- **Zero-shot의 가능성**: 현재는 프롬프트를 최적화하는 학습 과정이 필요하지만, 지시어 튜닝(instruction-tuning)을 통해 진정한 Zero-shot 추론을 달성하는 것이 다음 목표가 될 것이다.

## 📌 TL;DR

본 논문은 다양한 음성 처리 작업을 **'음성-단위 생성'**이라는 단일한 프레임워크로 통합하고, 이를 **Textless Speech LM**과 **Prompt Tuning**으로 해결하는 방법을 제안한다. 특히 학습 가능한 Verbalizer를 통해 성능을 높였으며, 실험 결과 **계산 효율성**과 **Few-shot 성능** 면에서 기존의 미세 조정 방식보다 압도적인 우위를 점했다. 이 연구는 향후 범용 음성 모델(Universal Speech Model) 구축을 위한 효율적인 경로를 제시하였다.