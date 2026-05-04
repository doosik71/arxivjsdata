# SLICK: EXPLOITING SUBSEQUENCES FOR LENGTH-CONSTRAINED KEYWORD SPOTTING

Kumari Nishu, Minsik Cho, Devang Naik (2024)

## 🧩 Problem to Solve

본 논문은 자원이 제한된 엣지 디바이스(edge device) 환경에서 사용자 정의 키워드 스포팅(User-defined Keyword Spotting, KWS)을 효율적으로 수행하는 문제를 해결하고자 한다. 사용자 정의 KWS는 미리 정해진 단어가 아니라 사용자가 입력한 텍스트 기반의 키워드를 인식해야 하므로, 오디오와 텍스트라는 서로 다른 모달리티(modality) 간의 정렬이 필수적이다.

기존 연구들은 가변적인 텍스트 길이를 처리하기 위해 RNN을 사용하거나 단순 평균(average)을 내는 등의 aggregation 방식을 사용하였으나, 이 과정에서 정보 손실이 발생하여 정확도가 저하되는 문제가 있었다. 또한, 단순한 음소(phoneme) 단위의 검출은 문맥(context)이 부족하여 발음이 유사한 키워드를 구분하는 데 한계가 있었다. 따라서 본 논문의 목표는 정보 손실을 최소화하는 길이 제한(length-constrained) 접근 방식과 문맥 정보를 강화한 서브시퀀스(subsequence) 매칭 기법을 통해, 경량 모델에서도 높은 인식 성능을 달성하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 키워드의 최대 길이에 실질적인 제약을 두어 복잡한 aggregation 과정을 제거하고, 세밀한 단위의 오디오-텍스트 관계를 학습하는 서브시퀀스 매칭 체계를 도입하는 것이다.

1. **길이 제한 접근 방식(Length-constrained approach):** 실제 키워드 길이 분포를 분석하여 대부분의 키워드가 일정 길이 이하임을 확인하고, 최대 길이를 고정함으로써 가변 길이에 따른 정보 손실 없이 효율적인 처리가 가능하게 하였다.
2. **서브시퀀스 레벨 매칭(Subsequence-level matching):** 텍스트의 부분 시퀀스와 오디오를 매칭하는 학습 방식을 도입하여, 발음이 유사한 키워드 간의 미세한 차이를 문맥적으로 구분할 수 있는 능력을 강화하였다.
3. **멀티태스크 학습 프레임워크(Multi-task learning framework):** 발화 단위 매칭(utterance-level), 서브시퀀스 단위 매칭(subsequence-level), 그리고 음소 인식(phoneme recognition)의 세 가지 작업을 동시에 학습시켜 모델의 강건성을 높였다.

## 📎 Related Works

기존의 사용자 정의 KWS 연구는 크게 두 가지 방향으로 진행되었다. 첫 번째는 사용자가 오디오 샘플을 직접 등록하는 Query-by-Example (QbE) 방식이다. 하지만 이는 등록 과정이 번거롭고 등록된 오디오와 테스트 샘플 간의 일관성에 성능이 크게 의존한다는 단점이 있다. 두 번째는 텍스트 기반의 등록 방식으로, 오디오와 텍스트 인코더를 독립적으로 학습시켜 잠재 공간(latent space)에서 정렬하는 방식이다. 그러나 이러한 방법들은 모델 파라미터 수가 많아 엣지 디바이스에 탑재하기에는 부적합한 경우가 많았다.

저리소스 환경을 타겟으로 한 최근 연구들은 monotonic matching loss나 보조적인 음소 검출 손실(phoneme-level detection loss)을 사용하였다. 하지만 고정된 패턴 매칭은 정렬의 유연성을 떨어뜨리며, 개별 음소 단위의 검출은 문맥 정보가 결여되어 유사한 발음의 키워드를 정확히 구별하지 못하는 한계가 존재한다.

## 🛠️ Methodology

### 전체 시스템 구조

SLiCK는 크게 **Encoder**와 **Matcher** 두 가지 모듈로 구성된다. 학습 단계에서는 세 가지 태스크를 동시에 수행하는 멀티태스크 학습을 진행하며, 추론 단계에서는 추론에 불필요한 모듈을 제거하고 발화 단위 매칭 관련 레이어만 남겨 모델을 경량화한다.

### 1. Encoder

- **Audio Encoding:** Tiny Conformer를 사용하여 입력 오디오를 $n \times D$ 차원의 임베딩으로 변환한다. 여기서 $n$은 가변적인 시간 차원이며, $D$는 임베딩 차원이다. 이 출력은 Matcher의 Key($K$)와 Value($V$)로 사용된다. 또한, 오디오 인코더 상단에 선형 레이어를 추가하여 CTC(Connectionist Temporal Classification) 손실 함수 $\mathcal{L}_{CTC}$를 통한 음소 인식 작업을 수행한다.
- **Length-constrained Keyword Spotting:** 가변 길이 텍스트 처리 문제를 해결하기 위해 최대 키워드 길이 $T$를 25개 음소로 제한하였다. 분석 결과, 대부분의 키워드가 25개 음소 이내에 분포함을 확인하였다.
- **Text Encoding:** 텍스트를 G2P(grapheme-to-phoneme) 시스템을 통해 음소 시퀀스로 변환한 후, 최대 길이 $T$에 맞춰 패딩(padding)을 수행한다. 이후 P2V(phoneme-to-vector) 데이터베이스를 통해 $T \times D$ 차원의 텍스트 임베딩을 생성하며, 이는 Matcher의 Query($Q$)로 사용된다.

### 2. Matcher

- **Utterance-level Matching:** Transformer의 Cross-Attention 메커니즘을 사용하여 오디오와 텍스트의 관계를 계산한다.
$$C = \text{CrossAttention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{D}}\right)V$$
여기서 $Q \in \mathbb{R}^{T \times D}$, $K \in \mathbb{R}^{n \times D}$, $V \in \mathbb{R}^{n \times D}$이다. 길이 제한 덕분에 출력 $C$의 차원은 항상 $T \times D$로 일정하며, 별도의 aggregation 없이 $C$를 벡터화하여 선형 레이어를 통해 매칭 여부를 예측하고 $\mathcal{L}_{utt}$ 손실을 계산한다.

- **Subsequence-level Matching:** 세밀한 오디오-텍스트 관계 학습을 위해 텍스트의 부분 시퀀스 $C_{1:t}$ (첫 번째부터 $t$번째 벡터까지)를 사용하여 매칭을 수행한다.
  - 학습 시, 앵커 텍스트와 실제 발화 텍스트를 비교하여 모든 부분 시퀀스($t \in [T]$)에 대해 match/mismatch 레이블을 생성한다.
  - 각 길이 $t$에 대해 별도의 완전 연결 레이어 $\text{FC}_t$를 두어 예측을 수행하며, 이에 따른 손실을 $\mathcal{L}_{ss}$라고 정의한다.

### 3. Training Criterion

전체 학습 목표는 다음과 같은 가중치 합산 손실 함수를 최소화하는 것이다.
$$\mathcal{L} = \alpha_1 \mathcal{L}_{utt} + \alpha_2 \mathcal{L}_{ss} + \alpha_3 \mathcal{L}_{CTC}$$
실험적으로 최적의 가중치 조합은 $\alpha_1=2, \alpha_2=1, \alpha_3=5$임을 확인하였다.

## 📊 Results

### 실험 설정

- **데이터셋:** Libriphrase 데이터셋을 사용하였으며, 난이도에 따라 LibriPhrase-Hard(LH)와 LibriPhrase-Easy(LE)로 나누어 평가하였다.
- **평가 지표:** AUC(Area Under the ROC Curve)와 EER(Equal-Error-Rate)을 사용하였다.
- **비교 대상:** 저리소스 환경을 위해 설계된 CMCD와 PhoneMatchNet을 베이스라인으로 설정하였다.

### 정량적 결과

제안 방법인 SLiCK는 모델 크기가 596K 파라미터로 베이스라인들보다 작음에도 불구하고 성능은 대폭 향상되었다.

- **LibriPhrase-Hard (LH):** AUC가 $88.52 \to 94.9$로 상승하였고, EER은 $18.82 \to 11.1$로 크게 감소하였다.
- **LibriPhrase-Easy (LE):** AUC $99.82$, EER $1.78$을 기록하며 베이스라인을 상회하였다.

### Ablation Study

각 태스크의 기여도를 분석한 결과는 다음과 같다.

1. $\mathcal{L}_{utt}$만 사용 시: LH AUC 88.8
2. $\mathcal{L}_{utt} + \mathcal{L}_{CTC}$ 사용 시: LH AUC 92.9 (음소 인식 태스크가 성능 향상에 기여)
3. $\mathcal{L}_{utt} + \mathcal{L}_{CTC} + \mathcal{L}_{ss}$ 사용 시: LH AUC 94.9 (서브시퀀스 매칭이 가장 결정적인 성능 향상을 이끌어냄)

## 🧠 Insights & Discussion

본 논문의 가장 큰 강점은 실무적인 제약 조건(최대 길이 $T=25$)을 역이용하여 계산 복잡도를 낮추면서도 정보 손실을 막았다는 점이다. 특히 서브시퀀스 레벨 매칭은 'service'와 'surface'처럼 매우 유사한 발음을 가진 키워드를 구분할 때, 앞부분의 일치 여부를 확인하고 뒷부분에서 불일치를 발견하는 세밀한 판단을 가능하게 함을 시각화 결과(Fig. 3)를 통해 증명하였다.

다만, 최대 길이 $T$를 고정했기 때문에 $T$보다 긴 매우 긴 문장 형태의 키워드에 대해서는 대응할 수 없다는 가정이 전제되어 있다. 하지만 논문에서 제시한 데이터 분석에 따르면 일반적인 KWS 상황에서 25개 음소는 충분한 길이로 판단된다. 또한, 본 연구는 Libriphrase 데이터셋에 집중되어 있어, 극단적인 소음 환경이나 다양한 언어 확장성에 대한 검증은 명시되지 않았다.

## 📌 TL;DR

본 논문은 엣지 디바이스용 사용자 정의 KWS를 위해 **최대 길이 제한(Length-Constraint)**과 **서브시퀀스 매칭(Subsequence Matching)** 기법을 제안한 SLiCK를 소개한다. 멀티태스크 학습(발화 매칭, 서브시퀀스 매칭, 음소 인식)을 통해 모델 크기를 줄이면서도 유사 발음 키워드 구분 능력을 획기적으로 높였으며, 특히 LH 데이터셋에서 AUC 94.9, EER 11.1라는 우수한 성적을 거두었다. 이 연구는 자원 제한적인 환경에서 고성능의 개인화된 음성 인식 시스템을 구축하는 데 중요한 기여를 할 것으로 보인다.
