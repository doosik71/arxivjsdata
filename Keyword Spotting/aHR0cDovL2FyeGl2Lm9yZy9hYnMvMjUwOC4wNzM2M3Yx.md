# Keyword Mamba: Spoken Keyword Spotting with State Space Models

Hanyu Ding, Wenlong Dong and Qirong Mao (2025)

## 🧩 Problem to Solve

본 논문은 음성 신호에서 특정 키워드를 감지하는 Keyword Spotting (KWS) 작업의 효율성과 성능을 동시에 개선하고자 한다. KWS는 음성 비서, 스마트 홈 기기, 지능형 콕핏 시스템 등 인간-컴퓨터 상호작용(HCI) 시스템의 핵심 진입점으로 활용되므로 매우 중요하다.

기존의 딥러닝 기반 KWS 모델들은 다음과 같은 한계를 가지고 있다.

- **Convolutional Neural Networks (CNNs):** 수용 영역(receptive field)이 고정되어 있어 장기적인 시간적 의존성(long-range temporal dependencies)을 포착하는 데 한계가 있다.
- **Recurrent Neural Networks (RNNs):** 순차적인 계산 특성으로 인해 병렬 처리가 어려워 학습 및 추론 속도가 느리다.
- **Transformers:** 입력 컨텍스트 윈도우 크기에 따라 계산 복잡도가 이차적으로 증가($O(L^2)$)하여, 긴 시퀀스를 처리할 때 효율성과 확장성이 떨어진다.

따라서 본 연구의 목표는 이러한 연산 효율성 문제와 장기 패턴 포착 능력을 동시에 해결하기 위해, 선형 시간 복잡도를 가지면서도 강력한 시퀀스 모델링 능력을 갖춘 Neural State Space Model (SSM)인 Mamba를 KWS에 처음으로 도입하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Mamba 아키텍처를 KWS 작업의 특성에 맞게 최적화하여 적용하는 것이다. 주요 기여 사항은 다음과 같다.

1. **KWS를 위한 최초의 SSM 적용:** State Space Model, 특히 Mamba를 KWS 작업에 적용한 첫 번째 사례이며, 이를 통해 기존 Transformer 기반 모델보다 적은 파라미터로 더 높은 정확도를 달성하였다.
2. **시간 도메인 중심의 설계:** KWS가 주파수 정보보다 시간적 패턴에 더 민감하다는 점에 착안하여, Mamba를 시간 축(temporal domain)을 따라 적용함으로써 효율적인 장기 의존성 모델링을 구현하였다.
3. **BiMamba 및 하이브리드 구조 제안:** 음성 데이터의 특성상 과거와 미래의 문맥을 모두 고려해야 하므로 양방향 Mamba(BiMamba)를 도입하였으며, Mamba의 선형성을 보완하기 위해 Transformer의 Feed-Forward Network(FFN)를 결합한 KWM-T 구조를 제안하였다.

## 📎 Related Works

### State Space Models (SSMs)

SSM은 시계열 데이터를 모델링하기 위한 통계적 프레임워크로, 잠재 상태(latent state)의 진화를 통해 입출력 관계를 정의한다.

- **S4 (Structured State Space Sequence Model):** HiPPO 메모리 메커니즘과 FFT 기반의 효율적인 커널 계산을 통해 계산 복잡도를 $O(L \log L)$로 낮추었으나, 성능 면에서 Transformer에 미치지 못하는 한계가 있었다.
- **Mamba:** S4를 기반으로 입력 기반 파라미터화(input-aware parameterization)와 선택 메커니즘(selection mechanism)을 도입하여, Transformer의 $O(L^2)$ 복잡도를 $O(L)$로 줄이면서도 성능을 비약적으로 향상시켰다.

### Vision 및 Speech Mamba

최근 Mamba는 시각 영역(Vision Mamba, VMamba)과 음성 영역(Speech Enhancement, ASR)으로 확장되고 있다. 특히 Vision Mamba(Vim)는 양방향 SSM을 사용하여 비인과적(non-causal) 이미지 시퀀스를 처리한다. 하지만 KWS 작업에 Mamba를 어떻게 효율적으로 설계하고 적용할지에 대한 연구는 이전까지 이루어지지 않았다.

## 🛠️ Methodology

### 전체 파이프라인

Keyword Mamba의 전체 프로세스는 다음과 같다.

1. **입력 처리:** 오디오 파형을 MFCC(Mel-frequency cepstral coefficients) 스펙트로그램 $M \in \mathbb{R}^{F \times T}$로 변환한다. 여기서 $F$는 주파수, $T$는 시간 차원이다.
2. **패칭(Patching):** 시간 도메인 모델링을 위해 패치 크기를 $[F, 1]$로 설정하여 $T$개의 패치 시퀀스로 만든다. 이후 선형 투영(Linear Projection)을 통해 고차원 $d$로 매핑한다.
3. **토큰 구성:** 학습 가능한 Class Token $\text{cls} \in \mathbb{R}^{1 \times d}$를 시퀀스의 중간에 삽입하고, 학습 가능한 Positional Embedding $\text{pos}$를 더해 Mamba Encoder의 입력 $\mathbf{x}_0$를 생성한다.
4. **분류:** Mamba Encoder를 거친 후 최종 Class Token을 정규화(Normalization)하고 MLP Head에 통과시켜 최종 클래스를 예측한다.

### Mamba 및 BiMamba 구조

Mamba의 기본 동역학은 다음과 같은 연속 시간 상태 방정식으로 정의된다.
$$h'(t) = Ah(t) + Bx(t)$$
$$y(t) = Ch(t)$$
실제 구현을 위해 이를 이산화(discretization)하여 다음 형태로 변환한다.
$$\bar{A} = \exp(\Delta A), \quad \bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I)\Delta B$$
이산화된 상태 방정식은 다음과 같다.
$$h'(t) = \bar{A}h(t-1) + \bar{B}x(t)$$
$$y(t) = Ch(t)$$

KWS에서는 전체 발화 내용을 한 번에 처리하므로 양방향 문맥 파악이 필수적이다. 따라서 본 논문은 **BiMamba**를 사용한다. 이는 입력 데이터를 정방향(Forward) SSM과 역방향(Backward) SSM에 각각 통과시킨 후 그 결과를 결합하는 구조이다.

### Task-Aware Encoder Designs

본 논문은 두 가지 인코더 변형을 제안한다.

- **KWM (Mamba Layer):** Transformer 레이어를 완전히 대체하여 BiMamba 레이어를 쌓은 구조이다. 경량화가 가능하며 실시간 처리에 유리하다.
- **KWM-T (Transformer Layer with Mamba):** Transformer 레이어 내의 Multi-Head Self-Attention (MHSA) 모듈만을 BiMamba로 대체하고, Feed-Forward Network (FFN)와 Norm 층은 유지한 구조이다. 이는 SSM의 선형적인 특성을 FFN의 비선형 변환으로 보완하여 고차원적인 의미 표현 능력을 높이기 위함이다.

## 📊 Results

### 실험 설정

- **데이터셋:** Google Speech Commands V1 및 V2. (12-label, 30-label, 35-label 작업 수행)
- **평가 지표:** 정확도 (Accuracy, ACC).
- **비교 대상:** Att-RNN, MHAtt-RNN, BC-ResNet, MatchboxNet, KWT (Keyword Transformer) 등.

### 주요 결과

1. **성능 우위:** KWM 및 KWM-T 모두 기존 SOTA 모델들을 능가하였다. 특히 KWM-T-192 모델은 V2-12 데이터셋에서 98.91%의 최고 정확도를 기록하였다.
2. **파라미터 효율성:** Transformer 기반의 KWT-3가 5M 이상의 파라미터를 사용하는 반면, Keyword Mamba(KWM-192)는 3.4M의 파라미터만으로 더 높은 정확도를 달성하여 뛰어난 파라미터 효율성을 보였다.
3. **모델 크기 및 깊이 영향:** 모델 차원을 64에서 192까지, 깊이를 6에서 12까지 변화시킨 실험에서, KWM은 크기가 작아져도 성능 하락폭이 적은 강건함(robustness)을 보였다. 이는 Mamba가 적은 파라미터로도 효율적으로 장기 의존성을 모델링할 수 있음을 시사한다.
4. **KWM vs KWM-T:** 하이브리드 구조인 KWM-T가 순수 Mamba 구조인 KWM보다 전반적으로 더 높은 성능을 보였다. 이는 FFN의 비선형성이 복잡한 음성 특징 표현에 도움을 준다는 것을 의미한다.

### 절제 연구 (Ablation Studies)

- **패치 형태:** 주파수 방향이나 혼합 패치보다 시간 도메인 전용 패치($[40, 1]$)를 사용했을 때 성능이 가장 높았다. 이는 KWS에서 시간적 모델링이 결정적임을 입증한다.
- **Class Token 위치:** 토큰을 시퀀스의 중간(Mid)에 배치했을 때 가장 좋은 성능을 보였다. 이는 양방향 Mamba가 앞뒤 문맥을 균형 있게 수집할 수 있기 때문으로 분석된다.
- **방향성:** 단방향 Mamba(Mamba-Fo-Fo)의 경우 정확도가 78.29%로 급격히 하락하였다. 반면 양방향 구조(BiMamba-Bi-Bi)는 98.01%를 기록하여, 음성 작업에서 양방향 정보 흐름이 필수적임을 보여주었다.

## 🧠 Insights & Discussion

### 강점 및 성과

본 연구는 계산 복잡도가 높은 Transformer의 Self-Attention을 Mamba의 Selective SSM으로 성공적으로 대체하였다. 이를 통해 $O(L)$의 선형 복잡도를 유지하면서도 KWS 성능을 향상시켰으며, 특히 하드웨어 자원이 제한된 임베디드 환경에서 매우 유용한 대안이 될 수 있음을 증명하였다.

### 한계 및 해석

본 논문에서는 주로 Google Speech Commands라는 정제된 데이터셋을 사용하였다. 실제 환경의 극심한 소음이나 다양한 언어 환경에서의 일반화 성능에 대한 분석은 명시적으로 제시되지 않았다. 또한, Class Token의 위치가 성능에 영향을 주긴 하지만 그 영향력이 다른 음성/언어 작업에 비해 상대적으로 작다는 점은 흥미로운 지점이며, 이에 대한 심층적인 원인 분석이 추가로 필요할 것으로 보인다.

### 비판적 해석

KWM-T가 KWM보다 성능이 좋다는 결과는 Mamba 단독으로는 복잡한 비선형 특징을 충분히 추출하기 어렵다는 점을 시사한다. 이는 Mamba가 효율적인 '전달자' 역할은 수행하지만, 고차원적 '특징 추출기'로서의 능력은 여전히 Transformer의 FFN 같은 구조에 의존하고 있음을 보여준다.

## 📌 TL;DR

본 논문은 KWS 작업에 최초로 **Mamba (Selective State Space Model)**를 도입하여, 기존 Transformer의 이차 복잡도 문제를 해결하고 연산 효율성을 극대화한 **Keyword Mamba (KWM)** 아키텍처를 제안한다. 특히 시간 도메인 중심의 양방향 모델링(BiMamba)과 Transformer의 FFN을 결합한 하이브리드 구조(KWM-T)를 통해, **더 적은 파라미터로 기존 SOTA 모델보다 높은 정확도**를 달성하였다. 이 연구는 향후 저전력/실시간 음성 인식 시스템 설계에 있어 Mamba 기반 모델이 강력한 대안이 될 수 있음을 시사한다.
