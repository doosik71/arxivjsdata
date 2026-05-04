# DONUT: CTC-based Query-by-Example Keyword Spotting

Loren Lugosch, Samuel Myer, Vikrant Singh Tomar (2018)

## 🧩 Problem to Solve

본 논문은 사용자가 직접 정의한 커스텀 웨이크워드(Custom Wakeword)를 효율적이고 정확하게 인식하는 문제를 해결하고자 한다. 

현대적인 음성 제어 장치들은 "Hey Siri"나 "OK Google"과 같이 미리 설정된 웨이크워드를 사용하지만, 사용자 경험을 향상시키기 위해서는 사용자가 원하는 단어로 웨이크워드를 설정할 수 있는 기능이 필요하다. 그러나 이를 구현하기에는 다음과 같은 기술적 난제가 존재한다:
- **재학습의 비용:** 신경망 기반의 키워드 스포팅(Keyword Spotting, KWS)은 새로운 단어를 인식하기 위해 모델을 재학습시켜야 하며, 이는 많은 시간과 대량의 학습 데이터가 소요된다.
- **Query-by-string의 한계:** CTC(Connectionist Temporal Classification) 기반 방식은 텍스트 형태로 단어를 입력받는 'Query-by-string' 방식을 사용하는데, 이는 텍스트 인터페이스가 추가로 필요하며 무엇보다 실제 사용자의 개별적인 발음 특성을 반영하지 못한다는 단점이 있다.
- **Query-by-example의 해석력 부족:** 소수의 음성 샘플을 사용하는 'Query-by-example(QbE)' 방식(예: DTW, Siamese Network)은 사용자 적응력이 뛰어나지만, 내부 동작이 블랙박스 형태여서 왜 오인식이나 미인식이 발생하는지 분석하고 최적화하기 어렵다.

따라서 본 논문의 목표는 QbE 방식의 사용자 편의성과 적응력, 그리고 CTC 방식의 일반화 성능과 해석 가능성을 동시에 결합한 알고리즘인 DONUT을 제안하는 것이다.

## ✨ Key Contributions

DONUT의 핵심 아이디어는 **사용자가 제공한 소수의 음성 예제로부터 가능한 레이블 시퀀스 가설(Label Sequence Hypotheses)의 집합을 생성하고, 이를 통해 웨이크워드를 검출**하는 것이다.

단일한 정답 레이블을 추정하는 대신, Beam Search를 통해 상위 $N$개의 유력한 가설들을 유지함으로써 레이블 추정 과정에서 발생할 수 있는 오류를 최소화한다. 추론 시에는 이 모든 가설에 대해 점수를 매기고 이를 합산하여 최종 결정함으로써, 모델의 해석 가능성을 유지하면서도 사용자 개별 발음에 최적화된 검출 성능을 확보한다.

## 📎 Related Works

논문에서는 기존의 커스텀 웨이크워드 인식 접근 방식을 다음과 같이 분류하고 한계를 지적한다:
- **CTC-based KWS:** 키워드를 음소(Phoneme) 레이블 시퀀스로 표현하여 재학습 없이 인식할 수 있게 한다. 하지만 입력이 텍스트 기반(Query-by-string)이므로 사용자의 실제 발음과 괴리가 생길 수 있다.
- **DTW(Dynamic Time Warping) 기반 QbE:** MFCC나 Posteriorgram 같은 특징 벡터의 정렬 점수를 사용한다. 사용자의 음성 샘플을 직접 사용하므로 적응력이 좋으나, DTW 행렬만으로는 구체적인 오인식 원인을 파악하기 어려워 해석력이 떨어진다.
- **RNN/Siamese Network 기반 QbE:** 고정 길이의 임베딩 벡터(Hidden state 등) 간의 코사인 거리를 측정한다. 이 역시 벡터 공간에서의 비교이므로 인간이 이해할 수 있는 방식으로 모델을 디버깅하거나 최적화하기 어렵다.

DONUT은 이러한 기존 방식들과 달리, CTC의 출력물인 '읽을 수 있는 문자열(레이블 시퀀스)'을 기반으로 QbE를 구현함으로써 해석력과 사용자 적응력을 동시에 잡고자 한다.

## 🛠️ Methodology

DONUT은 크게 **Label Model**과 **Wakeword Model**의 두 가지 구성 요소로 이루어진다.

### 1. 전체 시스템 구조
- **Label Model ($\phi$):** 대규모 음성 코퍼스(LibriSpeech)로 사전 학습된 신경망이다. 입력 오디오 $\mathbf{x} = \{x_t \in \mathbb{R}^d | t=1, \dots, T\}$를 받아 각 타임스텝별 레이블 및 CTC blank 심볼의 확률 분포인 Posteriorgram $\pi = f_\phi(\mathbf{x})$를 출력한다. 본 연구에서는 3개 층의 GRU 네트워크를 사용하였다.
- **Wakeword Model:** 사용자가 선택한 웨이크워드에 대한 레이블 시퀀스 집합과 각 시퀀스에 대한 신뢰도(Confidence)의 쌍으로 구성된다.

### 2. 학습 절차 (Enrollment)
사용자가 3개의 웨이크워드 예제 $\{x_{train,1}, x_{train,2}, x_{train,3}\}$를 녹음하면 다음 과정을 거친다:
1. 각 예제에 대해 Label Model $\phi$를 통해 Posteriorgram $\pi_{train,i}$를 계산한다.
2. **Beam Search** (폭 $B$)를 수행하여 가능성이 높은 상위 $B$개의 레이블 시퀀스를 추출한다.
3. 그 중 상위 $N$개의 가설 $\hat{y}_{train,i,j}$를 선택하고, 해당 가설의 로그 확률을 기반으로 신뢰도 $w_{train,i,j}$를 계산하여 저장한다. 신뢰도는 다음과 같이 정의된다:
   $$w_{train,i,j} = -\frac{1}{\log p_\phi(\hat{y}_{train,i,j} | x_{train,i})}$$
4. 최종적으로 $\text{wake\_model} = \bigcup (\hat{y}_{train,i,j}, w_{train,i,j})$ 형태로 저장된다.

### 3. 추론 절차 (Inference)
새로운 테스트 오디오 $x_{test}$가 입력되면 다음 과정을 수행한다:
1. VAD(Voice Activity Detector)를 통해 음성 구간만 추출하여 Label Model $\phi$에 입력하고 $\pi_{test}$를 얻는다.
2. Wakeword Model에 저장된 모든 가설 $(\hat{y}, w)$에 대하여, **CTC Forward 알고리즘**을 사용하여 해당 레이블 시퀀스가 나타날 로그 확률 $\log p_\phi(\hat{y} | x_{test})$를 계산한다.
3. 계산된 로그 확률에 신뢰도 $w$를 곱하여 전체 점수를 합산한다:
   $$\text{score} = \sum_{(\hat{y}, w) \in \text{wake\_model}} \log p_\phi(\hat{y} | x_{test}) \cdot w$$
4. 이 최종 점수가 미리 정해진 임계값(Threshold)보다 높으면 웨이크워드가 검출된 것으로 판단한다.

### 4. 계산 복잡도
- **Label Model:** $O(nT)$ (여기서 $n$은 파라미터 수, $T$는 프레임 수).
- **Wakeword Model:** $O(NUT)$ (여기서 $N$은 가설 수, $U$는 시퀀스 평균 길이).
- 일반적으로 $O(nT) \gg O(NUT)$이므로, 시스템 전체의 부하는 Label Model이 지배하며, Wakeword Model의 연산량은 매우 적어 임베디드 시스템에 적합하다.

## 📊 Results

### 1. 실험 설정
- **데이터셋:** LibriSpeech (Label Model 학습용), LibriSpeech-Fewshot 및 English-Fewshot (웨이크워드 검출 테스트용).
- **지표:** Equal Error Rate (EER, 낮을수록 좋음), Area Under ROC Curve (AUC, 높을수록 좋음).
- **비교 대상:** DTW (FBANK 기반), DTW (Posteriorgram 기반), Query-by-string CTC.

### 2. 주요 결과
- **QbE 방법론 간 비교 (Table 1):** DONUT은 모든 케이스(동일 화자-혼동 단어, 동일 화자-비혼동 단어, 타 화자-비혼동 단어)에서 DTW 기반 방식보다 압도적으로 낮은 EER과 높은 AUC를 기록하였다. 특히 동일 화자의 혼동 단어 케이스에서 EER $7.8\%$를 기록하여 DTW($20.8\% \sim 24.2\%$) 대비 성능이 월등히 높았다.
- **Query-by-string과의 비교 (Table 2):** 정답 레이블 시퀀스를 텍스트로 입력했을 때와 비교하여, 비혼동 단어에서는 비슷한 성능을 보였으며, 오히려 음성적으로 혼동하기 쉬운(Confusing) 사례에서는 DONUT(EER $21.0\%$)이 텍스트 기반 방식(EER $26.9\%$)보다 우수한 성능을 보였다. 이는 사용자의 실제 발음 특성이 반영되었기 때문으로 분석된다.
- **하이퍼파라미터 영향 (Table 3, 4):**
    - 유지하는 가설 수 $N$이 증가할수록 EER이 감소하는 경향을 보였으나, 일정 수준(약 $N=20 \sim 100$) 이후에는 수렴하는 양상을 보였다.
    - Label Model의 음소 인식 오류율(Phoneme Error Rate)이 낮을수록 웨이크워드 검출 성능(EER)이 직접적으로 향상됨이 확인되었다.

## 🧠 Insights & Discussion

### 1. 강점 및 해석 가능성
DONUT의 가장 큰 강점은 **해석 가능성(Interpretability)**이다. 웨이크워드 모델이 단순한 벡터가 아니라 '음소 시퀀스'의 집합으로 구성되어 있기 때문에, 개발자는 모델이 사용자의 음성을 어떻게 해석했는지 직접 확인할 수 있다. 예를 들어, 사용자가 "of dress"라고 말했을 때 모델이 'V' 음소 대신 'N' 음소로 인식했다면, 이를 통해 Label Model의 데이터를 보강하거나 사용자의 특이한 억양임을 인지할 수 있다.

### 2. 한계 및 가정
- **가설 수의 트레이드오프:** $N$을 늘리면 성능은 향상되지만 연산량과 메모리 사용량이 선형적으로 증가한다. 다만, 논문에서 언급했듯 $N$과 $U$가 매우 작은 값들이므로 실질적인 영향은 적다.
- **가중치 합산 방식:** 저자들은 가설들의 점수를 단순 가중치 합산으로 처리하였다. LogSumExp와 같은 확률론적 합산 방식보다 가중치 합산이 더 좋은 성능을 보였는데, 이는 LogSumExp가 $\max()$ 함수처럼 동작하여 단일 가설에 치중되는 경향이 있기 때문인 것으로 추측된다.

### 3. 비판적 해석
본 연구는 매우 효율적인 프레임워크를 제시하였으나, 실험 데이터셋에서 '타 화자가 동일한 웨이크워드를 말하는 경우(Imposter)'에 대한 테스트가 부족하다는 점이 아쉽다. 실제 환경에서는 타인이 사용자의 웨이크워드를 통해 기기를 활성화하는 보안 문제가 중요하므로, 이에 대한 추가 검증이 필요할 것으로 보인다.

## 📌 TL;DR

DONUT은 **사전 학습된 CTC 기반 Label Model을 활용하여, 소수의 음성 샘플로부터 복수의 레이블 가설을 생성하고 이를 합산하여 웨이크워드를 검출**하는 알고리즘이다. 이 방식은 기존의 DTW 기반 QbE보다 정확도가 높고, 텍스트 기반 방식보다 사용자 발음 적응력이 뛰어나며, 무엇보다 **인간이 이해할 수 있는 레이블 시퀀스를 사용함으로써 모델의 디버깅과 최적화가 가능하다**는 강력한 장점이 있다. 연산 효율성이 매우 높아 개인정보 보호를 위해 클라우드 전송 없이 온디바이스(On-device) 환경에서 구현하기에 최적화된 구조이다.