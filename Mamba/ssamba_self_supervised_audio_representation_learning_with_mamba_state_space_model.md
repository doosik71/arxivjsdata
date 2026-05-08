# SSAMBA: SELF-SUPERVISED AUDIO REPRESENTATION LEARNING WITH MAMBA STATE SPACE MODEL

Siavash Shams, Sukru Samet Dindar, Xilin Jiang, Nima Mesgarani (2025)

## 🧩 Problem to Solve

효과적인 오디오 표현 학습(Audio Representation Learning)을 위해서는 단기 및 장기 의존성을 모두 포착하는 것이 필수적이다. 기존의 Convolutional Neural Network(CNN)는 전역적 의존성 포착에 한계가 있었으며, 이를 해결하기 위해 등장한 Transformer 기반 모델(예: AST, SSAST)은 뛰어난 성능을 보여주었으나 치명적인 효율성 문제를 가지고 있다. Transformer의 Self-attention 메커니즘은 입력 시퀀스 길이에 대해 메모리 사용량과 계산 복잡도가 이차 함수적으로 증가하는 Quadratic Complexity 문제를 야기하며, 이는 특히 긴 오디오 데이터를 처리할 때 추론 시간과 자원 소모를 급격히 증가시킨다. 따라서 본 논문은 Transformer 수준의 성능을 유지하면서도 계산 복잡도를 획기적으로 낮춘 효율적인 오디오 표현 학습 모델을 개발하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Transformer의 Attention 메커니즘을 대체하여 선형 복잡도를 가진 State Space Model(SSM)의 최신 구조인 Mamba를 오디오 도메인에 적용하는 것이다. 주요 기여 사항은 다음과 같다.

- **최초의 SSM 기반 자가 지도 학습 오디오 모델**: Attention-free 구조이면서 자가 지도 학습(Self-supervised Learning)이 가능한 최초의 SSM 기반 오디오 표현 학습 모델인 SSAMBA를 제안하였다.
- **Bidirectional Mamba 구조 도입**: 오디오 데이터의 특성을 효과적으로 포착하기 위해 양방향(Bidirectional) Mamba 인코더를 설계하여 전방향과 후방향의 문맥 정보를 모두 학습할 수 있도록 하였다.
- **복합 학습 목표 설정**: 판별적(Discriminative) 목표와 생성적(Generative) 목표를 동시에 최적화하는 자가 지도 학습 프레임워크를 구축하여, 라벨이 없는 대규모 데이터셋으로부터 강건한 표현을 학습하도록 설계하였다.
- **압도적인 효율성 증명**: Transformer 기반의 SSAST와 비교하여 유사하거나 더 높은 성능을 달성함과 동시에, 추론 속도와 메모리 효율성 면에서 비약적인 향상을 이루었다.

## 📎 Related Works

기존의 오디오 분류 작업에서는 Audio Spectrogram Transformer(AST)가 SOTA 성능을 달성하였으나, 대량의 라벨링된 데이터가 필요하다는 단점이 있었다. 이를 해결하기 위해 Masked Spectrogram Patch Modeling(MSPM)을 도입한 SSAST가 제안되어 라벨 없는 데이터로 사전 학습하는 경로를 열었으나, 앞서 언급한 Transformer의 Quadratic Complexity 문제는 여전히 해결되지 않은 과제로 남아 있었다.

최근 Mamba와 같은 SSM은 시퀀스 모델링 능력을 유지하면서도 선형 복잡도를 제공하여 텍스트, 비전, 생물학적 데이터 등 다양한 분야에서 Transformer의 대안으로 주목받고 있다. 기존의 Mamba 관련 오디오 연구들은 주로 음성 향상(Speech Enhancement)이나 음원 분리(Speech Separation)와 같은 특정 작업에 집중되었으나, 본 논문은 이를 확장하여 다양한 다운스트림 작업에 범용적으로 적용 가능한 '일반적인 오디오 표현 학습'으로 넓혔다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 1. Mamba의 수학적 기초

Mamba는 연속적인 상태 공간 모델(State Space Model)을 기반으로 한다. 기본적으로 입력 $x(t)$를 은닉 상태 $h(t)$를 통해 출력 $y(t)$로 매핑하며, 다음과 같은 미분 방정식으로 정의된다.
$$h'(t) = Ah(t) + Bx(t)$$
$$y(t) = Ch(t)$$
디지털 시스템 구현을 위해 Zero-Order Hold(ZOH) 방법을 사용하여 이산화(Discretization)하며, 이 과정에서 timescale 파라미터 $\Delta$가 도입되어 $A^d$와 $B^d$가 결정된다. Mamba의 핵심은 $\Delta, A, B, C$ 파라미터를 입력 $x_t$에 따라 동적으로 변화시키는 Selective Scan 알고리즘을 통해 입력 내용에 따라 선택적으로 정보를 처리하는 능력을 갖춘 것이다.

### 2. SSAMBA 아키텍처

전체 시스템은 오디오 신호를 입력받아 고차원 표현을 추출하는 파이프라인으로 구성된다.

- **입력 표현 및 패칭**: 오디오 파형을 STFT를 통해 128차원 Log Mel Filterbank 특징으로 변환하여 스펙트로그램을 생성한다. 이를 $16 \times 16$ 크기의 패치(Patch)로 나누고, Linear Projection을 통해 $D$차원의 임베딩으로 변환한다. 이후 학습 가능한 Positional Encoding을 더해 위치 정보를 부여한다.
- **Bidirectional Mamba Encoder**: 단방향 SSM의 한계를 극복하기 위해 전방향(Forward)과 후방향(Backward) SSM을 병렬로 배치한다.
  - 각 방향의 SSM은 $\text{SiLU} \to \text{Conv1D} \to \text{Linear}$ 과정을 거쳐 $A, B, C, \Delta$ 파라미터를 생성하고 SSM 연산을 수행한다.
  - 양방향의 결과물 $y_{\text{forward}}$와 $y_{\text{backward}}$는 게이팅 메커니즘인 $\text{SiLU}(z)$와 원소별 곱셈($\odot$)을 통해 조절된 후, 합산되어 최종 출력 $H_i$가 된다.

### 3. 자가 지도 학습 프레임워크 (Pretraining)

라벨 없는 데이터셋(AudioSet-2M, LibriSpeech)을 활용하여 Masked Spectrogram Patch Modeling(MSPM)을 수행한다. 일부 패치를 마스킹하고 이를 예측하는 두 가지 손실 함수를 동시에 사용한다.

- **판별적 목표(Discriminative Objective)**: 마스킹된 패치가 무엇인지 식별하는 작업으로, InfoNCE loss를 사용하여 실제 임베딩과 예측값 사이의 대조 학습을 수행한다.
$$L_d = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\exp(\langle c_i, x_i \rangle)}{\sum_{j=1}^{N} \exp(\langle c_i, x_j \rangle)}$$
- **생성적 목표(Generative Objective)**: 마스킹된 패치의 원래 내용을 재구성하는 작업으로, 예측값 $\hat{x}_i$와 실제값 $x_i$ 사이의 Mean Squared Error(MSE) loss를 사용한다.
$$L_g = \frac{1}{N} \sum_{i=1}^{N} \|\hat{x}_i - x_i\|^2$$
- **최종 손실 함수**: 두 손실을 가중치 $\lambda$로 결합하여 최적화한다.
$$L = L_d + \lambda L_g$$

## 📊 Results

### 1. 실험 설정

- **데이터셋**: 사전 학습에는 AudioSet-2M과 LibriSpeech를 사용하였으며, 다운스트림 평가에는 AudioSet-20K(AS), ESC-50(ESC), Speech Commands V1/V2(KS1, KS2), VoxCeleb 1(SID), IEMOCAP(ER), Urban8K(DASL) 등 총 7개의 다양한 데이터셋을 활용하였다.
- **비교 대상**: 동일한 사전 학습 방식을 공유하는 Transformer 기반의 SSAST 모델(Tiny, Small, Base 사이즈)과 비교하였다.

### 2. 성능 분석

정량적 결과(Table 2)에 따르면, SSAMBA는 대부분의 작업에서 SSAST와 비슷하거나 더 높은 정확도를 보였다. 특히 모델 크기가 커질수록(Base 모델) 성능 향상 폭이 뚜렷했으며, AudioSet-20K 및 환경음 분류(ESC) 작업에서 강점을 보였다. 또한, 사전 학습을 거치지 않은 경우 DASL과 같은 복잡한 작업에서 수렴하지 못하는 모습을 보여, 제안된 자가 지도 학습 프레임워크의 중요성이 입증되었다.

### 3. 효율성 분석

가장 주목할 만한 결과는 계산 효율성이다. 입력 패치 수가 22k인 경우, **SSAMBA Tiny 모델은 SSAST Tiny 대비 추론 속도가 약 92.7% 빠르며, GPU 메모리 사용량은 약 95.4% 절감**되었다. 이는 입력 시퀀스 길이에 따라 복잡도가 선형적으로 증가하는 Mamba의 특성이 오디오 데이터 처리에서 매우 효과적임을 보여준다.

## 🧠 Insights & Discussion

본 연구는 오디오 표현 학습에서 Transformer의 지배적인 구조를 SSM으로 대체할 수 있음을 성공적으로 입증하였다. 특히 bidirectional 구조를 통해 시퀀스의 양방향 문맥을 모두 포착함으로써, Attention 메커니즘 없이도 경쟁력 있는 성능을 낼 수 있음을 보여주었다.

**비판적 해석 및 한계점**:

- **음성 작업의 최적화**: 실험 결과, 화자 식별(SID)이나 감정 인식(ER)과 같은 음성 특화 작업에서는 성능 향상 폭이 상대적으로 작았다. 저자들은 이것이 패치 기반 마스킹(Patch-based masking) 때문이라고 분석하며, 음성의 시간적 역동성을 더 잘 포착할 수 있는 프레임 기반 마스킹(Frame-based masking)을 적용한다면 성능을 더 높일 수 있을 것이라고 제언한다.
- **범용성 vs 특수성**: 일반적인 오디오 이벤트 분류에서는 매우 강력하지만, 매우 정밀한 음성 분석 작업에서는 여전히 Wav2Vec 2.0나 HuBERT 같은 음성 전용 모델보다 낮은 성능을 보였다(Table 3 참조). 이는 범용 오디오 모델로서의 가치는 높으나, 특정 도메인 최적화에는 추가적인 전략이 필요함을 의미한다.

## 📌 TL;DR

SSAMBA는 Transformer의 Quadratic Complexity 문제를 해결하기 위해 **Bidirectional Mamba(SSM)**를 도입한 최초의 자가 지도 학습 오디오 표현 학습 모델이다. 판별적/생성적 목표를 결합한 사전 학습을 통해 **SSAST보다 우수하거나 대등한 성능을 달성**하면서도, **추론 속도는 92.7% 향상시키고 메모리 사용량은 95.4% 감소**시키는 압도적인 효율성을 달성하였다. 이 연구는 실시간 오디오 처리나 자원 제한적인 엣지 디바이스 환경에서 고성능 오디오 AI를 구현하는 데 중요한 기반이 될 것으로 기대된다.
