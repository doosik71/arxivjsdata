# Time Series Language Model for Descriptive Caption Generation

Mohamed Trabelsi, Aidan Boyd, Jin Cao, and Huseyin Uzunalioglu (2025)

## 🧩 Problem to Solve

본 논문은 시계열 데이터(Time Series Data)에서 관찰되는 패턴을 자연어 설명으로 변환하는 **시계열 캡셔닝(Time Series Captioning)** 문제를 해결하고자 한다. 시계열 데이터의 복잡한 수치 정보를 이해하기 쉬운 서술형 문장으로 자동 생성하는 것은 데이터의 해석 가능성(Interpretability)을 높이고, 데이터 과학 전문 지식이 없는 사용자에게도 인사이트를 제공하며, 대규모 데이터셋에서 트렌드 분석 및 이상 탐지를 효율화한다는 점에서 매우 중요하다.

최근 LLM(Large Language Models)이 자연어 처리 및 컴퓨터 비전 분야에서 비약적인 발전을 이루었으나, 시계열 캡셔닝 분야에 적용하는 데에는 **학습 데이터의 희소성(Data Scarcity)**이라는 결정적인 한계가 존재한다. 따라서 본 연구의 목표는 데이터 부족 문제를 극복하면서 시계열의 미세한 시간적 패턴을 정확하게 캡처하여 정교한 텍스트 설명을 생성하는 새로운 시계열 언어 모델인 **TSLM(Time Series Language Model)**을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 시계열 데이터를 **텍스트 기반 표현(Textual Representation)**과 **임베딩 기반 표현(Embedding Representation)**으로 동시에 정의하여 결합한 **Joint Representation**을 사용하는 것이다. 이를 통해 거시적인 흐름(Coarse-grained)과 미세한 변동(Fine-grained) 정보를 모두 포착한다.

또한, 데이터 부족 문제를 해결하기 위해 다음의 두 가지 전략을 제안한다.

1. **In-context Prompting**을 이용한 합성 데이터(Synthetic Data) 생성: 오픈소스 LLM을 활용하여 대량의 시계열-캡션 쌍을 생성함으로써 학습 데이터를 증강한다.
2. **Cross-modal Dense Retrieval Scoring** 기반의 노이즈 제거(Denoising): LLM의 환각(Hallucination) 현상으로 인해 생성된 잘못된 합성 데이터 쌍을 필터링하여 학습 데이터의 품질을 보장한다.

## 📎 Related Works

기존의 시계열 캡셔닝 연구는 크게 세 가지 방향으로 진행되었다.

- **수치 기반 접근 방식:** MLP, CNN, RNN/LSTM 등을 사용하여 수치 데이터를 인코딩하고 디코더를 통해 텍스트를 생성하는 방식이다. 하지만 복잡한 시맨틱 표현 능력이 부족하다.
- **이미지 기반 접근 방식:** 시계열 데이터를 차트나 그래프 이미지로 변환한 뒤, Image Captioning 모델(예: DenseNet, SwinTransformer 기반 모델)을 적용하는 방식이다. 그러나 이미지 변환 과정에서 수치 데이터의 정밀한 변동 정보가 손실될 위험이 있다.
- **텍스트 기반 접근 방식:** 수치 데이터를 단순히 텍스트 토큰 시퀀스로 변환하여 LLM에 입력하는 방식이다. 이 방식은 LLM의 강력한 생성 능력을 활용하지만, 수치 데이터의 특성상 위치 정보가 결여되기 쉽고 데이터 희소성 문제에 취약하다.

TSLM은 이러한 한계를 극복하기 위해 텍스트와 임베딩 모달리티를 동시에 사용하는 **Multi-modal** 구조를 채택하고, 체계적인 데이터 증강 및 정제 파이프라인을 도입하여 차별성을 갖는다.

## 🛠️ Methodology

### 1. 시계열 표현 (Time Series Representations)

TSLM은 시계열 $\mathcal{T}$를 두 가지 형태로 표현하여 결합한다.

**가. Textual Representations (거시적 정보)**
시계열의 각 수치를 문자열 토큰으로 변환하되, 위치 정보를 명시적으로 주입하기 위해 **Phase Tagging**을 적용한다. 전체 시퀀스를 시작($\langle start \rangle$), 중간($\langle middle \rangle$), 끝($\langle end \rangle$)의 세 단계로 나누어 다음과 같이 구성한다:
$$\langle time\_series \rangle = \langle start \rangle \mathcal{T}_{1:\frac{L}{3}} \langle /start \rangle + \langle middle \rangle \mathcal{T}_{\frac{L}{3}+1:\frac{2L}{3}} \langle /middle \rangle + \langle end \rangle \mathcal{T}_{\frac{2L}{3}+1:L} \langle /end \rangle$$

**나. Embedding Representations (미세적 정보)**
수치 데이터의 미세한 변화를 포착하기 위해 **1D CNN 기반의 Encoder**를 사용한다. 이 인코더는 레이블이 없는 상태에서 학습 가능한 **Autoencoder** 구조로 사전 학습되어, 효율적인 저차원 임베딩 $\langle time\_series\_embedding \rangle \in \mathbb{R}^{f \times d}$를 생성한다.

**다. Joint Representation**
위의 두 표현을 결합하여 최종 입력 $\text{JR}(\mathcal{T})$를 구성한다:
$$\text{JR}(\mathcal{T}) = [\text{CLS}] \text{ Textual Rep} \oplus \text{ Embedding Rep}$$

### 2. Multi-Modal Encoder Architecture

TSLM의 인코더는 LLM 기반의 구조를 가지며, 서로 다른 모달리티 간의 정렬을 위해 **Reprogramming Layer**를 도입한다.

- **Reprogramming Layer:** 시계열 임베딩을 텍스트 표현 공간으로 매핑한다. 텍스트 프로토타입(Text Prototypes) $\mathcal{A}_p$를 정의하고, 시계열 임베딩과 프로토타입 간의 **Cross-Attention**을 통해 정렬된 임베딩 $Z$를 생성한다.
- **최종 임베딩:** 텍스트 임베딩 $\mathcal{A}_t$와 정렬된 시계열 임베딩 $Z$를 연결하여 $\mathcal{A}_{ts} = \mathcal{A}_t \oplus Z$를 형성하고, 이를 Transformer 블록에 통과시켜 최종 self-attention 임베딩 $\mathcal{U}$를 얻는다.

### 3. 학습 파이프라인 및 데이터 정제

학습은 다음의 4단계 과정으로 이루어진다.

1. **합성 데이터 생성:** LLaMA2-13B-Chat에 Few-shot 예시를 제공하는 In-context Prompting을 통해 대량의 시계열-캡션 쌍을 생성한다.
2. **1D CNN Autoencoder 학습:** 시계열 데이터만 사용하여 비지도 학습 방식으로 임베딩 추출기를 학습시킨다.
3. **Cross-modal Denoising:** 생성된 데이터의 환각 문제를 해결하기 위해, 깨끗한 원본 데이터를 이용해 **Cross-modal Dense Retrieval** 모델을 학습시킨다. 시계열 벡터 $u$와 캡션 벡터 $a$의 내적(dot product)으로 유사도 $\text{sim}(\mathcal{T}, c) = u^\top a$를 계산하고, 임계값 $\theta$보다 낮은 유사도를 가진 쌍은 노이즈로 간주하여 제거한다.
4. **TSLM 학습:** 정제된 합성 데이터와 원본 데이터를 사용하여 Next Token Prediction 태스크로 모델을 학습시킨다.

### 4. 추론 및 묘사적 캡션 생성 (Descriptive Caption Generation)

추론 단계에서 TSLM은 하나의 시계열에 대해 여러 개의 캡션을 생성한다. 이후, 이 여러 개의 캡션들을 LLaMA2-13B-Chat에 입력하여 하나의 통합된 **묘사적 캡션(Descriptive Caption)**으로 요약하도록 함으로써, TSLM이 시계열 데이터와 LLM 사이의 가교(Bridge) 역할을 수행하게 한다.

## 📊 Results

### 실험 설정

- **데이터셋:** STOCK(주식 가격 데이터, 1,900개 시계열) 및 SYNTH(인위적 패턴 데이터, 560개 시계열)를 사용하였다.
- **평가 지표:** ROUGE-1, ROUGE-2, ROUGE-L, BERTScore 및 본 논문에서 제안한 TSLMScore(Cross-modal 유사도 기반)를 사용하였다.
- **비교 대상:** TRUCE(수치 기반), LLaVA(이미지 기반), LLaMA2-7B/13B/70B(텍스트 전용), T5, BART 등 다양한 모달리티의 모델들과 비교하였다.

### 주요 결과

1. **성능 우위:** TSLM은 모든 지표에서 기존 SOTA 모델들을 유의미하게 상회하였다. 특히 STOCK 데이터셋에서 ROUGE-L과 BERTScore가 크게 향상되었다.
2. **모달리티 결합의 효과:** Ablation Study 결과, 텍스트 전용(TSLM Text)이나 임베딩 전용(TSLM TimeSeries) 모델보다 두 정보를 모두 사용한 Joint Representation 모델의 성능이 가장 높았다.
3. **Denoising의 중요성:** 노이즈 제거 과정을 거치지 않은 모델(TSLM w/o denoising)에 비해 ROUGE-L 기준 약 6.71%의 성능 향상이 확인되어, 합성 데이터의 정제 과정이 필수적임이 입증되었다.
4. **효율성:** TSLM(약 1B 파라미터)은 LLaMA-70B와 같은 거대 모델보다 훨씬 작은 크기임에도 불구하고, 시계열 특화 모달리티를 활용함으로써 더 우수한 성능과 낮은 추론 비용을 달성하였다.

## 🧠 Insights & Discussion

**강점 및 통찰:**

- **모달리티 정렬:** Reprogramming Layer를 통해 수치 임베딩을 텍스트 공간으로 투영함으로써, LLM이 시계열의 특성을 텍스트와 동일한 맥락에서 이해할 수 있게 하였다.
- **LLM의 도구화:** LLM이 직접 수치 데이터를 처리하게 하는 대신, TSLM이 이를 정교한 텍스트로 번역해 제공함으로써 LLM의 고질적인 문제인 수치 계산 능력 부족 및 환각 현상을 효과적으로 억제하였다.
- **데이터 증강 전략:** 단순한 데이터 생성이 아니라, '생성 $\rightarrow$ 정제 $\rightarrow$ 학습'으로 이어지는 파이프라인을 통해 데이터 희소성 문제를 체계적으로 해결하였다.

**한계 및 향후 과제:**

- **정적 태깅:** 현재는 시작/중간/끝의 3단계 정적 태깅만을 사용하고 있으나, 더 세밀한 세그멘테이션이나 x축 좌표 정보를 추가하는 방식에 대한 연구가 필요하다.
- **단변량 제한:** 본 연구는 univariate 시계열에 국한되어 있으며, 향후 multivariate 시계열 캡셔닝으로의 확장이 필요하다.
- **정성적 평가:** 노이즈 제거 과정의 평가가 일부 샘플에 대한 정성적 분석에 의존하고 있어, 더 객관적인 정량적 평가 지표가 요구된다.

## 📌 TL;DR

본 논문은 시계열 데이터의 거시적 텍스트 표현과 미세한 CNN 임베딩을 결합한 **TSLM**이라는 멀티모달 언어 모델을 제안한다. 데이터 부족 문제를 해결하기 위해 **LLM 기반 합성 데이터 생성**과 **Cross-modal Dense Retrieval 기반의 노이즈 제거** 기법을 도입하였다. 실험 결과, TSLM은 기존의 수치/이미지/텍스트 기반 모델들보다 뛰어난 성능을 보였으며, 특히 시계열 데이터를 LLM이 이해할 수 있는 텍스트로 변환하는 '가교' 역할을 수행함으로써 고품질의 묘사적 캡션을 생성할 수 있음을 입증하였다. 이 연구는 향후 도메인 범용 시계열 분석 모델 및 다변량 시계열 캡셔닝 연구에 중요한 기초가 될 것으로 보인다.
