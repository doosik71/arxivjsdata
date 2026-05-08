# CTC-aligned Audio-Text Embedding for Streaming Open-vocabulary Keyword Spotting

Sichen Jin, Youngmoon Jung, Seungjin Lee, Jaeyoung Roh, Changwoo Han, Hoonyoung Cho (2024)

## 🧩 Problem to Solve

본 논문은 사용자가 텍스트 형태로 키워드를 등록하면 이를 실시간으로 감지하는 **Streaming Open-vocabulary Keyword Spotting (KWS)** 문제를 해결하고자 한다.

전통적인 KWS는 고정된 어휘집(Fixed vocabulary)을 사용하므로 새로운 키워드를 추가하려면 모델을 다시 학습시켜야 하는 한계가 있다. 이를 극복하기 위해 등장한 Open-vocabulary KWS 중 Query-by-Example (QbyE) 방식은 예시 음성 데이터를 필요로 하여 사용자에게 불편함을 주며, 환경 변화나 화자 변경에 취약하다. 텍스트 기반 등록 방식은 사용자 인터페이스 측면에서 매우 직관적이지만, 기존의 텍스트-음성 매핑 방식들은 주로 Attention 메커니즘이나 동적 계획법(Dynamic Programming)을 사용하여 전체 문맥(Global context)을 계산해야 하므로, 항상 켜져 있어야 하는 보이스 어시스턴트의 특성상 필수적인 '스트리밍(Streaming)' 방식의 구현이 어렵다는 문제가 있다.

따라서 본 연구의 목표는 낮은 연산 비용과 작은 모델 크기를 유지하면서도, 스트리밍 환경에서 텍스트로 등록된 임의의 키워드를 정확하게 검출할 수 있는 시스템을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **CTC(Connectionist Temporal Classification)**를 통한 즉각적인 프레임 단위 정보 추출과 **Embedding**을 통한 전체 키워드의 전역 정보 비교를 결합하는 것이다.

가장 큰 기여점은 음성과 키워드 텍스트를 실시간으로 동적으로 정렬(Dynamic alignment)하여 공동의 Audio-Text Embedding 공간을 구축하고, 이를 통해 KWS를 수행하는 구조를 제안했다는 점이다. 특히, 추론 시 시간 복잡도를 키워드 길이 $U$에 비례하는 $O(U)$ 수준으로 낮추어 매우 효율적인 스트리밍 처리를 가능하게 하였다.

## 📎 Related Works

### CTC (Connectionist Temporal Classification)

CTC는 음성 프레임과 전사 텍스트 간의 정렬 레이블이 없는 상태에서 자동으로 정렬을 학습하는 기법이다. 입력 시퀀스와 출력 시퀀스의 길이 차이를 해결하기 위해 Blank 토큰을 도입하며, 가능한 모든 경로의 확률을 합산하여 타겟 레이블의 확률을 계산한다.

### Multi-view Loss

Multi-view learning은 서로 다른 뷰(예: 텍스트와 음성)에서 얻은 특징을 공통된 잠재 공간으로 매핑하는 방법이다. 본 논문에서는 Asymmetric Proxy Loss (AsyP)를 기반으로 한 multi-view loss를 활용하여, 텍스트 임베딩(TE)을 프록시로 설정하고 음성 임베딩(AE)이 이에 가깝게 정렬되도록 학습시킨다.

기존의 Open-vocabulary KWS 접근 방식들이 전체 시퀀스를 분석하는 Non-streaming 방식에 치중했던 반면, 본 연구는 CTC의 프레임 단위 정렬 능력과 Embedding의 전역적 비교 능력을 결합하여 스트리밍 제약을 해결하며 차별점을 둔다.

## 🛠️ Methodology

### 전체 시스템 구조

시스템은 크게 **Text Encoder**, **Acoustic Encoder**, 그리고 **CTC Aligner** 세 가지 구성 요소로 이루어져 있다.

1. **Text Encoder**: 키워드 텍스트를 토큰화한 후, 두 개의 Bi-directional LSTM 레이어를 통과시켜 토큰 수준의 Text Embedding ($\text{h}^{\text{tokenTE}}_{y_u}$)을 생성한다.
2. **Acoustic Encoder**: 스트리밍 모드로 동작하며 입력 오디오 $\text{X}$를 잠재 벡터로 인코딩한다. 이후 두 개의 프로젝션 블록을 통해 각 프레임 $t$에 대한 토큰 분포 $\text{P}(\text{y}|\text{x}_t)$와 프레임 수준의 Acoustic Embedding ($\text{h}^{\text{frameAE}}_t$)을 생성한다. 모델 크기를 최소화하기 위해 MobileNet 블록을 사용하였다.
3. **CTC Aligner**: 타겟 키워드의 토큰과 Blank 토큰이 교차 배치된 디코딩 그래프를 구성한다. 이 그래프의 각 상태는 누적 CTC 점수, 전이 타이밍 정보, 그리고 누적된 AE를 저장한다.

### CTC Aligner 및 추론 절차

매 시간 단계 $t$마다 CTC 프로젝션 블록의 출력 분포를 바탕으로 디코딩 그래프의 전이 확률을 업데이트한다. Viterbi 알고리즘을 사용하여 현재 프레임에서 끝나는 최적의 정렬 경로를 찾는다.

- **상태 업데이트**: 상태 $l$의 점수 $\text{z}^{\text{CTC}}_{l,t}$는 이전 상태의 최대 점수와 현재 프레임의 토큰 확률의 로그 합으로 계산된다:
  $$\text{z}^{\text{CTC}}_{l,t} = \text{z}^{\text{CTC}}_{I_{l,t},t-1} + \log \text{P}(\text{st}_l | \text{x}_t)$$
- **AE 누적**: Non-blank 상태인 경우 현재 프레임의 $\text{h}^{\text{frameAE}}_t$를 해당 토큰의 누적 AE에 더하며, Blank 상태인 경우 이전 Non-blank 토큰의 AE에 더한다.

### 훈련 전략 및 손실 함수

모델은 CTC 정렬과 AE 학습을 동시에 수행하기 위해 End-to-end로 학습된다. 훈련 시에는 전체 시퀀스가 제공되며, 가장 높은 CTC 점수를 가진 최적 경로를 선택하여 토큰 수준의 AE를 다음과 같이 풀링(Pooling)하여 계산한다:
$$\text{h}^{\text{tokenAE}}_{y_u} = \frac{\sum_{f=t_{y_u}}^{t_{y_{u+1}}-1} \text{h}^{\text{frameAE}}_f}{t_{y_{u+1}} - t_{y_u}}$$

최종 목적 함수 $\text{L}$은 CTC 손실과 Multi-view 손실의 합으로 정의된다:
$$\text{L} = \text{L}_{\text{CTC}} + \text{L}_{\text{multi-view}}(\text{h}^{\text{tokenAE}}_{y_{1:U}}, \text{h}^{\text{tokenTE}}_{y_{1:U}})$$
여기서 $\text{L}_{\text{multi-view}}$는 앵커-포지티브(AP) 쌍은 가깝게, 앵커-네거티브(AN) 쌍은 멀게 밀어내는 구조를 가진다.

### 최종 점수 계산 (Inference)

추론 시에는 CTC 점수($\text{z}^{\text{CTC}}$)와 AE-TE 간의 코사인 유사도 기반 임베딩 점수($\text{z}^{\text{embed}}$)를 선형적으로 결합하여 최종 결정한다:
$$\text{z} = \text{z}^{\text{CTC}} + \lambda \text{z}^{\text{embed}}$$
$$\text{z}^{\text{embed}} = \frac{1}{U} \sum_{u=1}^{U} \cos(\text{h}^{\text{tokenAE}}_{y_u}, \text{h}^{\text{tokenTE}}_{y_u})$$

## 📊 Results

### 실험 설정

- **데이터셋**: LibriPhrase 데이터셋을 사용하였으며, 강건성을 위해 RIR(Room Impulse Responses) 합성 및 MUSAN 노이즈를 추가하였다.
- **비교 대상**: Cross-attention 기반 방식 [7]과 Dynamic Sequence Partitioning (DSP) [9] 기반의 Non-streaming 방식과 비교하였다.
- **평가 지표**: Equal Error Rate (EER)와 Area Under the Curve (AUC)를 사용하였으며, 난이도에 따라 Easy ($\text{LP}_E$)와 Hard ($\text{LP}_H$) 세트로 나누어 측정하였다.

### 주요 결과

- **정량적 결과**: 제안 방법은 단 155K의 파라미터만으로 Non-streaming 방식들과 경쟁 가능한 성능을 보였다. 특히 $\text{LP}_E$ 세트에서는 기존 방식들을 상회하는 결과를 얻었다.
- **임베딩 레벨별 성능**: 임베딩 학습 단위를 Character $\rightarrow$ Word $\rightarrow$ Phrase 수준으로 높일수록 성능이 크게 향상되었다. Phrase 레벨에서 $\text{LP}_E$ 기준 EER 6.06%, AUC 98.32%로 가장 우수한 성능을 기록하였다.
- **효율성**: 시간 복잡도 $O(U)$와 매우 작은 모델 크기(155K params) 덕분에 실시간 스트리밍 환경에 매우 적합함을 입증하였다.

## 🧠 Insights & Discussion

본 논문은 CTC를 통한 국소적 정렬과 Embedding을 통한 전역적 특징 비교를 결합함으로써, 스트리밍 KWS의 고질적인 문제인 '전역 문맥 파악'과 '실시간성' 사이의 트레이드오프를 효과적으로 해결하였다.

특히 실험을 통해 임베딩의 추상화 수준(Character $\rightarrow$ Word $\rightarrow$ Phrase)이 높아질수록 성능이 향상됨을 확인하였는데, 이는 잠재 공간(Latent space)에서 더 넓은 범위의 전역적 문맥을 고려할수록 클래스 간 변별력이 높아진다는 점을 시사한다.

다만, $\text{LP}_H$ (Hard negative) 세트에서는 $\text{LP}_E$에 비해 성능 향상 폭이 적거나 일부 하락하는 경향이 관찰되었다. 이는 텍스트가 매우 유사한 하드 네거티브 사례의 경우, 단순한 임베딩 유사도만으로는 구분하기 어려울 수 있음을 의미하며, 향후 더 정교한 변별력 강화 전략이 필요할 것으로 보인다.

## 📌 TL;DR

본 연구는 CTC 기반의 동적 정렬과 Audio-Text Embedding을 결합하여, 매우 작은 모델 크기(155K 파라미터)와 낮은 연산 복잡도($O(U)$)로 동작하는 **스트리밍 오픈 보캐블러리 키워드 검출기(CTCAT)**를 제안하였다. 텍스트 기반 등록만으로 실시간 음성 검출이 가능하며, 특히 Phrase 레벨의 임베딩을 사용할 때 Non-streaming 방식에 근접하거나 이를 능가하는 성능을 보였다. 이 연구는 자원이 제한된 온디바이스(On-device) 환경의 보이스 어시스턴트 시스템에 즉시 적용 가능한 실용적인 구조를 제시했다는 점에서 가치가 크다.
