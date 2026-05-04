# End-to-End User-Defined Keyword Spotting using Shifted Delta Coefficients

Kesavaraj V, Anuprabha M, Anil Kumar Vuppala (2024)

## 🧩 Problem to Solve

본 논문은 사용자가 임의로 정의한 키워드를 인식하는 User-Defined Keyword Spotting (UDKWS) 분야의 문제를 다룬다. 일반적인 키워드 스포팅(KWS)은 미리 정해진 폐쇄 어휘(closed vocabulary)만을 인식하지만, UDKWS는 학습 과정에서 접하지 못한 임의의 키워드를 인식해야 하는 개방 어휘(open vocabulary) 문제로 인해 복잡도가 매우 높다.

기존의 UDKWS 접근 방식들은 주로 MFCC(Mel Frequency Cepstral Coefficients)와 같은 단기 스펙트럼 특징(short-term spectral features)에 의존해 왔다. 그러나 이러한 특징들은 음성 신호의 시간적 역동성(temporal dynamics)을 포착하는 능력이 제한적이어서, 특히 발음이 매우 유사한 오디오-텍스트 쌍을 정확하게 구별하는 데 어려움이 있다. 따라서 본 연구의 목표는 장기 시간 정보(long-term temporal information)를 반영할 수 있는 특징 추출 방법을 도입하여, 발음의 가변성을 효과적으로 포착하고 UDKWS의 인식 성능을 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 언어 식별(Language Identification) 작업에서 효과가 입증된 Shifted Delta Coefficients (SDC)를 UDKWS 작업에 도입하는 것이다. SDC는 여러 프레임에 걸쳐 델타 특징을 쌓음으로써 장기적인 시간 문맥을 캡처하며, 이를 통해 연결되는 음소 간의 전이와 같은 발음의 가변성을 모델링할 수 있다. 

주요 기여 사항은 다음과 같다.
- SDC 특징을 MFCC, Mel-spectrogram, PLP, RASTA-PLP 등 기존의 단기 스펙트럼 특징들과 동일한 실험 환경에서 비교 분석하였다.
- SDC의 파라미터 구성을 변경하며 UDKWS 작업에 가장 적합한 시간적 문맥(temporal context) 범위를 탐색하였다.
- 키워드의 단어 길이에 따른 시스템 성능 변화를 분석하였다.
- 최신 UDKWS 기법들과의 비교를 통해 제안 방법론의 우수성을 입증하였다.

## 📎 Related Works

기존의 UDKWS 연구는 크게 세 가지 방향으로 진행되었다. 
첫째, 대규모 어휘 연속 음성 인식(LVCSR) 시스템을 사용하여 음성을 디코딩한 후 격자(lattice)에서 키워드를 검색하는 방식이다. 둘째, 키워드와 비키워드 구간을 모델링하는 별도의 HMM(Hidden Markov Model)을 사용하는 방식이다. 이 두 방식은 계산 비용이 매우 높다는 한계가 있다. 

최근에는 엔드투엔드(End-to-End) 시스템이 주목받고 있으며, 크게 두 가지 접근법이 있다. 하나는 등록된 음성 예제와 입력 쿼리를 매칭하는 Query-by-Example (QbyE) 방식인데, 이는 등록 당시의 음성과 평가 당시의 음성 간 유사성에 지나치게 의존하여 사용자별 음성 특성이나 배경 소음에 취약하다는 단점이 있다. 다른 하나는 텍스트 등록 기반 방법으로, 오디오와 텍스트를 공통 잠재 공간(latent space)으로 투영하여 일치 여부를 판단한다. 하지만 이러한 최근 연구들은 주로 딥러닝 모델 구조나 학습 전략에 집중했을 뿐, 성능의 핵심인 특징 공학(feature engineering) 측면에서의 탐구는 부족했다는 점이 기존 연구의 한계로 지적된다.

## 🛠️ Methodology

### 전체 시스템 구조
본 논문에서 제안하는 아키텍처는 오디오 인코더, 텍스트 인코더, 패턴 추출기, 패턴 판별기의 네 가지 서브모듈로 구성된다.

1.  **Audio Encoder**: 입력 특징을 받아 2D 합성곱 층(Conv2D) 2개와 양방향 GRU(Bi-GRU) 2개를 거쳐 128차원의 오디오 임베딩 $E_a \in \mathbb{R}^{m \times D}$를 생성한다.
2.  **Text Encoder**: 사전 학습된 Tacotron 2 모델의 LSTM 블록에서 중간 표현(512차원)을 추출한다. 이를 Bi-GRU와 Dense 층에 통과시켜 128차원의 텍스트 임베딩 $E_t \in \mathbb{R}^{n \times D}$를 생성한다. 이는 텍스트가 오디오 투영 정보를 가질 수 있도록 하기 위함이다.
3.  **Pattern Extractor**: Cross-Attention 메커니즘을 사용하여 오디오와 텍스트 임베딩 간의 시간적 상관관계를 포착한다. 텍스트 임베딩 $E_t$가 Query 역할을 하고, 오디오 임베딩 $E_a$가 Key와 Value 역할을 수행하여 문맥 벡터(context vector)를 생성한다.
4.  **Pattern Discriminator**: 생성된 문맥 벡터를 Bi-GRU 층에 입력하고, 마지막 프레임의 출력을 Sigmoid 활성화 함수가 적용된 Dense 층으로 보내 오디오와 텍스트가 동일한 키워드인지 여부를 이진 분류한다.

### Shifted Delta Coefficients (SDC)
본 연구의 핵심인 SDC는 Mel-spectrogram을 기반으로 계산된다. SDC는 $N-d-p-k$라는 네 가지 파라미터로 정의되며, $N$은 프레임당 계수 수, $d$는 현재 프레임으로부터의 시프트(지연) 양, $p$는 연속된 델타 블록 간의 시프트, $k$는 연결될 델타의 개수를 의미한다.

특정 반복 회차 $i$에서 $t$번째 프레임의 델타 특징 $\delta_c(t, i)$는 다음과 같이 계산된다.
$$\delta_c(t, i) = c(t + ip + d) - c(t + ip - d), \quad \text{where } 0 \le i \le k-1$$

이렇게 계산된 $k$개의 델타 값들을 수직으로 쌓아 $k \times N$ 차원의 SDC 벡터를 구성한다.
$$SDC(t) = \begin{pmatrix} \delta_c(t, 0) \\ \delta_c(t, 1) \\ \vdots \\ \delta_c(t, k-1) \end{pmatrix}$$

최종적으로 이 SDC 벡터를 정적(static) Mel-spectrogram 특징과 결합하여 오디오 인코더의 입력으로 사용한다.

### 학습 절차
- **손실 함수**: 이진 교차 엔트로피(Binary Cross-Entropy) 손실 함수를 사용한다.
- **최적화**: Adam 옵티마이저를 사용하며, 학습률은 $10^{-4}$, 배치 크기는 128로 설정하였다.
- **전처리**: 25ms 윈도우 길이와 10ms 오버랩을 사용하며, 고주파 에너지를 높이기 위해 0.97의 pre-emphasis 계수를 적용하고 Hamming 윈도우를 사용하였다.

## 📊 Results

### 실험 설정
- **데이터셋**: LibriPhrase (Easy $\text{LP}_E$ 및 Hard $\text{LP}_H$), Google Speech Commands V1 (G), Qualcomm Keyword Speech (Q) 데이터셋을 사용하였다. 특히 $\text{LP}_H$는 "madame"와 "modem"처럼 발음이 유사한 쌍을 포함하여 난이도가 높다.
- **평가 지표**: Equal Error Rate (EER)와 Area Under the Curve (AUC)를 주요 지표로 사용하였다.

### 주요 결과
1.  **특징별 비교**: SDC가 모든 데이터셋에서 가장 우수한 성능을 보였다. 특히 $\text{LP}_H$ 데이터셋에서 MFCC 대비 AUC는 8.69%, EER은 8.32% 개선되었다. 이는 SDC가 시간적 역동성을 포착하여 유사한 발음을 더 잘 구분함을 시사한다.
2.  **SDC 설정 최적화**: 
    - 시프트 값 $d$가 증가할수록 성능이 하락하여, $d=1$일 때 최적의 성능을 보였다.
    - 델타 스택 수 $k$의 경우, $k$가 5에서 8로 증가함에 따라 성능이 향상되었으나, 8을 초과하면 성능이 포화되거나 하락하였다. 결과적으로 $40-1-3-8$ 설정이 가장 최적이었다.
3.  **키워드 길이에 따른 분석**: 단어 길이가 길어질수록 EER이 증가하는 경향을 보였으나, 모든 길이에서 SDC가 Mel-spectrogram보다 높은 F1-score를 기록하였다.
4.  **SOTA 비교**: 제안 방법은 Google 데이터셋(G)을 제외한 모든 데이터셋에서 최신 기법들을 압도하였다. 특히 $\text{LP}_H$에서 CMCD 모델 대비 EER을 11.42%나 낮추는 성과를 거두었다.

## 🧠 Insights & Discussion

본 논문은 UDKWS 성능 향상을 위해 모델 구조의 복잡성을 높이는 대신, 입력 특징 단계에서 시간적 문맥을 강화하는 전략이 매우 효과적임을 입증하였다. 특히 발음이 유사한 단어들을 구분하는 문제는 UDKWS의 고질적인 난제로, SDC를 통한 장기 시간 정보의 반영이 이 문제를 해결하는 핵심 열쇠가 되었음을 확인하였다.

다만, 키워드의 길이가 길어질수록 인식 성능이 저하되는 현상은 여전히 해결해야 할 과제로 남아 있다. 이는 오디오 시퀀스가 길어짐에 따라 텍스트와의 정렬(alignment) 난이도가 상승하고, 특징 정보가 희석되기 때문으로 해석된다. 또한, 본 연구는 단일 특징 추출 방식에 집중하였으므로, 향후 서로 다른 특성을 가진 특징들을 결합하는 하이브리드 방식의 탐구가 필요하다.

## 📌 TL;DR

본 논문은 발음이 유사한 키워드를 구분하기 위해 장기 시간 정보를 포착하는 **Shifted Delta Coefficients (SDC)** 특징을 도입한 엔드투엔드 UDKWS 시스템을 제안한다. 실험 결과, SDC는 기존 MFCC나 Mel-spectrogram보다 우수한 성능을 보였으며, 특히 난이도가 높은 LibriPhrase-Hard 데이터셋에서 괄목할만한 성능 향상을 이루었다. 이 연구는 UDKWS 분야에서 특징 공학(feature engineering)의 중요성을 재조명하였으며, 향후 맞춤형 음성 인식 시스템의 강건성을 높이는 데 기여할 것으로 기대된다.