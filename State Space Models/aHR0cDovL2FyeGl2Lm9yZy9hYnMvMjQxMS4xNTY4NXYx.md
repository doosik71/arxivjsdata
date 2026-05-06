# State-Space Large Audio Language Models

Saurabhchand Bhati, Yuan Gong, Leonid Karlinsky, Hilde Kuehne, Rogerio Feris, James Glass (2024)

## 🧩 Problem to Solve

본 논문은 Large Audio Language Models(LALM)가 직면한 계산 효율성 문제를 해결하고자 한다. 기존의 LALM은 오디오 인지 모델과 Large Language Model(LLM)을 결합하여 오디오 추론, 의미 추론 및 의도 파악에서 뛰어난 성능을 보여주었으나, 그 핵심 아키텍처인 Transformer의 특성상 입력 시퀀스 길이에 따라 계산 복잡도가 이차적으로 증가하는(quadratic complexity) 문제가 있다.

이러한 특성은 특히 긴 오디오 및 음성 신호를 처리할 때 심각한 병목 현상을 일으키며, 메모리와 시간이 제한된 환경이나 저사양 디바이스로의 배포를 어렵게 만든다. 따라서 본 연구의 목표는 Transformer를 대체하여 선형 복잡도(linear complexity)를 가지는 State-Space Models(SSMs)를 LALM의 구성 요소에 도입함으로써, 계산 효율성을 높이면서도 기존 Transformer 기반 모델에 필적하는 성능을 유지하는 State-space LALM을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 LALM의 파이프라인에서 Transformer 기반의 구성 요소를 체계적으로 SSM으로 교체하는 것이다. 구체적인 기여 사항은 다음과 같다.

1. **오디오 인지 모듈의 교체**: Transformer 기반의 AST(Audio Spectrogram Transformer) 대신 SSM 기반의 DASS(Distilled Audio State-Space model)를 도입하여 오디오 특징 추출 단계의 효율성을 높였다.
2. **언어 모델의 교체**: 기존의 Transformer 기반 LLM을 State-space LLM으로 교체함으로써, 모델 전체의 파라미터 수를 대폭 줄이면서도 효율적인 추론이 가능한 최초의 State-space 기반 LALM을 제안하였다.
3. **효율성 및 성능 검증**: 제안된 모델이 훨씬 적은 수의 파라미터를 사용함에도 불구하고, 다양한 닫힌 문제(close-ended tasks)에서 기존 Transformer 기반 LALM들과 경쟁 가능한 수준의 성능을 보임을 입증하였다.

## 📎 Related Works

기존의 LALM 연구들은 주로 Transformer 기반의 강력한 인코더와 LLM을 결합하는 방향으로 진행되었다.

- **Pengi, AudioGPT, SALMONN**: 각각 HTSAT, Whisper, BEATs 등의 인코더와 GPT2, LLaMA 등의 LLM을 사용하여 오디오 이해 능력을 확장하였다.
- **LTU**: AST를 오디오 인코더로, LLaMA를 언어 모델로 사용하여 닫힌 문제뿐만 아니라 자유 형식의 개방형 질의응답(open-ended QA) 능력을 보여주었다.
- **GAMA**: LTU를 기반으로 Audio Q-Former를 통해 AST의 다양한 레이어 정보를 결합하여 복잡한 추론 능력을 개선하였으나, 학습 가능한 파라미터 수가 약 300M으로 매우 많다는 한계가 있다.

본 연구는 LTU의 구조를 기반으로 하되, 모델의 백본을 Transformer에서 SSM으로 교체함으로써 GAMA나 LTU 대비 훨씬 적은 파라미터(약 40M~60M의 학습 가능 파라미터)로 유사한 성능을 달성하여 계산 비용을 획기적으로 낮추었다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. State-Space Models (SSMs)

SSM은 연속적인 1차원 신호 $x(t)$를 은닉 상태 $h(t)$를 통해 출력 $y(t)$로 매핑하는 선형 상미분 방정식(Linear ODE)에 기반한다.

$$h'(t) = Ah(t) + Bx(t)$$
$$y(t) = Ch(t)$$

여기서 $A$는 진화 파라미터(evolution parameter), $B$와 $C$는 투영 파라미터(projection parameters)이다. 이를 실제 컴퓨터에서 처리하기 위해 Zero-order hold 방식을 사용하여 이산화(discretization)하면 다음과 같은 형태가 된다.

$$\bar{A} = \exp(\Delta A)$$
$$\bar{B} = (\Delta B)^{-1}(\exp(\Delta A) - I)\Delta B$$

이산화된 상태 방정식은 다음과 같이 재귀적인 형태로 표현될 수 있다.
$$h_t = \bar{A}h_{t-1} + \bar{B}x_t$$
$$y_t = C h_t$$

SSM은 학습 시에는 합성곱(Convolution) 뷰를 사용하여 병렬 학습이 가능하고, 추론 시에는 순환 신경망(RNN) 뷰를 사용하여 빠른 추론 속도와 무제한적인 컨텍스트 처리가 가능하다.

### 2. DASS: Distilled Audio State-Space Model

오디오 특징 추출기로 사용된 DASS는 AST(Transformer 기반)를 교사 모델로 하여 학습된 SSM 기반의 오디오 분류기이다.

- **구조**: 4개의 그룹으로 구성되며, 각 그룹은 SSM 블록과 다운샘플링 레이어로 이루어져 있다. 레이어를 거치며 시퀀스 길이는 줄어들고 특징 차원은 증가한다.
- **특징 추출 과정**: $1024 \times 128$ 크기의 스펙트로그램을 입력받아 $32 \times 4 \times 768$ 크기의 특징 맵을 생성한다. 이후 $3 \times 3$ 커널과 stride 2를 가진 2D Convolution 레이어를 통해 공간 차원을 더 줄이고, 선형 레이어를 통해 LLM의 입력 차원(LLaMA의 경우 4096, State-space LLM의 경우 2560)에 맞게 매핑한다.

### 3. Large Language Model (LLM) 및 LoRA

본 연구에서는 두 가지 종류의 LLM을 실험하였다.

- **Transformer-based**: Vicuna로 명령어 튜닝된 LLaMA-7B를 사용하였다.
- **State-space-based**: Pile 데이터셋으로 학습된 State-space LLM-2.8B를 사용하였다.

전체 가중치를 미세 조정하는 대신 **Low-rank Adaptation (LoRA)**를 적용하였다.

- **LLaMA**: 모든 Self-attention 레이어의 Key 및 Query 투영 레이어에 LoRA 어댑터($rank=8, \alpha=16$)를 추가하여 4.2M의 파라미터를 학습시켰다.
- **State-space LLM**: SSM 블록의 입력 투영 레이어에 LoRA 어댑터($rank=8, \alpha=16$)를 추가하여 6.5M의 파라미터를 학습시켰다.

### 4. Training Objective

훈련 목표는 입력 오디오 $A$와 이전 토큰들 $x_{1:t-1}$이 주어졌을 때 다음 토큰 $x_t$를 예측하는 확률 $P(x_t | x_{1:t-1}, A)$를 최대화하는 것이다. 손실 함수로는 텍스트 토큰들에 대해 **Cross-entropy loss**를 사용하였다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: OpenAQA 데이터셋 (Audio, Question, Answer 튜플)
- **평가 작업 및 지표**:
  - **오디오 분류 (8개 벤치마크)**: Accuracy, F1-score, mAP (LALM의 텍스트 출력을 텍스트 인코더로 인코딩하여 정답 레이블과 코사인 유사도 측정)
  - **오디오 캡셔닝 (AudioCaps, Clotho)**: SPICE 지표 사용
- **비교 대상**: SALMONN, Pengi, AudioGPT, LTU 및 제안 모델(Hybrid-LALM, ssLALM)

### 2. 주요 결과

- **정량적 성능**: 제안된 모델들은 SALMONN, Pengi, AudioGPT보다 우수한 성능을 보였으며, 기반이 된 LTU와 대등한 성능을 기록하였다. 특히 오디오 캡셔닝 작업에서는 최상위 지도 학습(supervised) 시스템과 유사한 성능을 보였다.
- **효율성**: `ssLALM (3B)` 모델은 `LLaMA-7B` 기반 모델보다 파라미터 수가 훨씬 적음에도 불구하고 경쟁력 있는 성능을 유지하였다.
- **학습 속도**: State-space LLM을 사용할 경우 배치 사이즈를 4에서 16으로 늘릴 수 있어, 학습 시간이 3일에서 2일 미만으로 단축되었다.
- **개방형 질의응답**: Table II의 샘플을 통해 제안 모델들이 오디오의 분위기 파악이나 세부 음향 특징(예: Liquid vs Gurgling)의 차이를 합리적으로 설명할 수 있음을 확인하였다.

## 🧠 Insights & Discussion

### 1. 강점 및 성과

본 연구는 SSM이 Transformer의 효율적인 대안이 될 수 있음을 LALM 영역에서 처음으로 증명하였다. 특히 모델의 크기를 획기적으로 줄이면서도(7B $\rightarrow$ 2.8B) 성능 하락이 거의 없었다는 점은 자원 제한적인 환경에서 LALM을 운용할 수 있는 가능성을 제시한다. 또한, DASS 인코더가 AST보다 오디오 길이 확장성(duration scalability)이 뛰어나다는 점이 전체 시스템의 견고함에 기여한 것으로 보인다.

### 2. 한계 및 분석

- **멀티 레이블 분류 저조**: 모든 LALM이 AudioSet의 멀티 레이블 분류 작업에서 낮은 mAP를 기록하였다. 이는 모델이 오디오 내에서 가장 지배적인 클래스(prominent class)만을 예측하고, 부수적인 클래스의 확률을 과소평가하기 때문으로 분석된다.
- **학습 속도 향상의 원인**: 학습 시간이 단축된 것이 SSM의 계산 효율성 때문인지, 단순히 모델 크기가 작아서인지에 대해 명확한 결론을 내리지 못했다.

### 3. 비판적 해석

본 논문은 SSM의 도입 효과를 정량적으로 잘 보여주었으나, 실험에 사용된 LLM의 크기가 상대적으로 작아(2.8B) 매우 거대한 LLM(예: 70B 이상)에서도 동일한 효율성과 성능 유지 비율이 나타날지는 미지수이다. 또한, 명령어 튜닝(instruction tuning)이 되지 않은 State-space LLM을 사용했음에도 성능이 좋았다는 점은, 오디오-언어 정렬(alignment) 과정에서 LoRA가 매우 효과적으로 작용했음을 시사한다.

## 📌 TL;DR

본 논문은 계산 복잡도가 높은 Transformer를 대체하여 선형 복잡도를 가진 **State-Space Models(SSMs)**를 오디오 인코더와 언어 모델 모두에 적용한 **최초의 State-space LALM**을 제안한다. 실험 결과, 제안된 모델은 파라미터 수를 크게 줄였음에도 불구하고 기존 Transformer 기반 LALM(LTU 등)과 대등한 성능을 보였으며, 추론 및 학습 효율성을 크게 개선하였다. 이 연구는 향후 저전력/실시간 오디오 추론 시스템 구축 및 초대형 SSM 기반 오디오-언어 모델 연구에 중요한 토대가 될 것으로 보인다.
