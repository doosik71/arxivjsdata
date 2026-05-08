# A GENERATIVE MODEL FOR RAW AUDIO USING TRANSFORMER ARCHITECTURES

Prateek Verma and Chris Chafe (2020)

## 🧩 Problem to Solve

본 논문은 오디오 신호의 파형(waveform) 수준에서 직접적인 오디오 합성을 수행하는 생성 모델의 성능 향상을 목표로 한다. 일반적으로 오디오 합성은 주파수 도메인에서의 합성이나 스펙트로그램(spectrogram)을 통한 접근 방식이 사용되지만, 이는 다시 시간 도메인으로 변환하는 과정에서 정보 손실이나 왜곡이 발생할 수 있는 문제가 있다.

따라서 저자들은 raw audio, 즉 가공되지 않은 파형 수준에서 직접 샘플을 생성하는 방식을 취한다. 특히 기존의 표준적인 접근 방식이었던 WaveNet의 한계를 극복하고, Transformer 아키텍처의 Attention 메커니즘을 통해 더 효율적으로 장기 의존성(long-term dependencies)을 학습하여 다음 샘플을 더 정확하게 예측하는 것을 연구의 주된 목표로 설정한다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 Transformer 아키텍처를 raw audio synthesis에 적용하여 기존의 WaveNet보다 우수한 성능을 보임을 입증한 것이다.

중심적인 아이디어는 고정된 수용장(receptive field)을 가진 Dilated Convolution 대신, 데이터의 중요도에 따라 동적으로 가중치를 할당하는 Attention 메커니즘을 사용하는 것이다. 이를 통해 모델은 미래의 샘플을 예측할 때 과거의 어떤 샘플이 중요한지를 스스로 학습할 수 있다. 또한, 단순한 Transformer 구조를 넘어 합성곱 신경망(CNN)을 통해 학습된 잠재 표현(latent representation)을 조건부(conditioning)로 입력하는 구조를 제안하여 예측 정확도를 추가적으로 향상시켰다.

## 📎 Related Works

오디오 합성 분야에서는 전통적으로 FM synthesis나 Karplus Strong 알고리즘과 같이 파라미터 기반의 합성 방식이 사용되었다. 이후 딥러닝 기반의 WaveNet이 등장하며 Causal Dilated Convolution을 통해 autoregressive하게 샘플을 생성하는 방식이 표준이 되었으며, 이는 TTS(Text-to-Speech), 소음 제거, 악기 변환 등 다양한 분야에 적용되었다.

하지만 WaveNet은 고정된 위상(topology)의 연결 구조를 가지므로, 매우 긴 시퀀스의 복잡한 의존성을 학습하는 데 한계가 있다. 반면 Transformer는 Attention 메커니즘을 통해 시퀀스 내의 모든 위치를 참조할 수 있어 NLP 등 다양한 분야에서 혁신을 일으켰다. 다만, Transformer의 Attention 블록은 계산 복잡도와 메모리 사용량이 시퀀스 길이의 제곱($O(T^2)$)에 비례하여 증가한다는 치명적인 단점이 있으며, 이는 raw audio와 같이 샘플 수가 매우 많은 데이터셋에서 큰 제약 사항이 된다.

## 🛠️ Methodology

### 전체 시스템 구조

본 모델은 확률적(probabilistic), 자기회귀적(auto-regressive), 그리고 인과적(causal)인 생성 모델이다. 즉, $t$ 시점의 샘플 $x_t$를 예측하기 위해 오직 이전 시점까지의 샘플들만을 참조한다. 전체적인 확률 모델은 다음과 같이 정의된다.

$$p(x) = \prod_{t=1}^{T} p(x_t | x_1, x_2, \dots, x_{t-1})$$

### 주요 구성 요소 및 학습 절차

**1. Transformer Architecture**

- **Causal Multi-Head Attention**: 쿼리($Q$), 키($K$), 값($V$) 벡터를 학습하여 각 샘플 간의 관계를 계산한다. 인과성을 유지하기 위해 미래의 샘플을 참조하지 못하도록 triangular mask를 적용한다. 수식으로는 다음과 같이 표현된다.
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
- **Feed-Forward Network (FFN)**: Attention 층 이후에 적용되며, 다음과 같은 구조를 가진다.
$$\text{FF}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$
- **Positional Encoding**: Transformer는 순서 정보가 없으므로, sinusoidal 함수를 이용해 각 샘플의 위치 정보를 임베딩에 추가한다.
- **Layer Normalization & Residual Connections**: 학습의 안정성과 수렴 속도를 높이기 위해 각 서브 블록 이후에 적용한다.

**2. Conditioned Generative Transformer**
Transformer의 연산 비용 문제로 인해 context window를 무한히 늘릴 수 없으므로, 저자들은 외부 조건부 정보를 제공하는 구조를 제안한다.

- 과거 250ms(4000 샘플)의 데이터를 6층의 CNN 아키텍처에 통과시켜 128차원의 잠재 공간(latent space)을 학습한다.
- 이 잠재 표현을 Transformer의 출력 로짓(logits)과 결합하는 Late Fusion 방식을 사용하여 최종 예측을 수행한다.

**3. 학습 세부 사항**

- **데이터셋**: YouTube에서 수집한 약 56시간 분량의 피아노 녹음 데이터.
- **해상도**: 16kHz 샘플링 레이트 및 8-bit 해상도(0-255 단계).
- **손실 함수**: 256개의 클래스에 대한 Cross-Entropy Loss를 사용하여 다음 비트 수준을 예측한다.
- **최적화**: Adam Optimizer를 사용하며, 손실 값이 정체될 때마다 학습률(learning rate)을 단계적으로 낮추는 방식을 적용하였다.

## 📊 Results

### 실험 설정

- **측정 지표**: Top-5 Accuracy를 사용한다. 이는 모델이 예측한 상위 5개 확률 값 중에 실제 정답 비트 레벨이 포함되어 있는지를 측정하는 방식이다.
- **비교 대상**: Vanilla WaveNet(10층), Stacked WaveNet(30층), 그리고 제안하는 Transformer의 다양한 변형 모델들.
- **데이터**: 50개의 테스트 트랙 중 무작위로 추출한 30분 분량의 데이터를 사용하였다.

### 정량적 결과

실험 결과, Transformer 기반 모델이 WaveNet 기반 모델보다 일관되게 높은 정확도를 보였다.

| Neural Model Architecture | Accuracy (Top-5) |
| :--- | :---: |
| Vanilla WaveNet (N=10, F=128) | 74% |
| Stacked WaveNet (N=30, F=128) | 76% |
| 3-Layer Transformer (H=4, E=128) | 80% |
| Conditioned 3-Layer Transformer (H=4, E=128) | 82% |
| Large 6-Layer Transformer (H=8, E=128) | 84% |
| Large 8-Layer Transformer (H=8, E=128) | 85% |

- 단순한 3층 Transformer만으로도 30층의 Stacked WaveNet보다 4%p 높은 성능을 기록하였다.
- 조건부(Conditioned) Transformer를 사용할 경우 성능이 추가적으로 2%p 향상되었다.
- 모델의 깊이와 Attention Head의 수를 늘린 Large Transformer(8-layer)에서는 최대 85%의 정확도를 달성하였다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 연구는 Transformer의 Attention 메커니즘이 WaveNet의 고정된 Dilated Convolution 구조보다 raw audio의 복잡한 패턴을 학습하는 데 훨씬 유리함을 보여주었다. 특히 Attention이 시퀀스 내에서 어떤 부분이 중요한지를 동적으로 학습할 수 있기 때문에, 더 적은 층수로도 더 강력한 표현력을 가질 수 있음을 입증하였다. 또한 CNN을 통한 잠재 표현의 결합이 문맥 파악 능력을 높여 성능 향상에 기여한다는 점을 확인하였다.

### 한계 및 비판적 해석

- **생성 품질의 문제**: 다음 샘플 예측(Next-step prediction)이라는 정량적 지표에서는 우수한 성과를 거두었으나, 조건 없이(unconditioned) 오디오를 생성했을 때는 의미 있는 음악적 소리가 만들어지지 않았다. 이는 단순히 다음 샘플 하나를 맞추는 확률적 정확도가 전체적인 오디오의 청각적 품질이나 음악적 구조를 보장하지 않음을 시사한다.
- **연산 효율성**: 저자들은 100ms라는 매우 짧은 context window를 사용하였다. 이는 Transformer의 $O(T^2)$ 복잡도 때문에 발생하는 제약이며, 실제 음악적 구조를 생성하기 위해서는 훨씬 더 긴 context가 필요함에도 불구하고 이를 해결하지 못한 점은 아쉬운 부분이다.
- **데이터셋의 특성**: 피아노 데이터셋만을 사용하였으므로, 음색의 다양성이 매우 높은 다른 악기나 환경에서도 동일한 성능 향상이 나타날지는 추가적인 검증이 필요하다.

## 📌 TL;DR

본 논문은 raw audio 합성을 위해 Transformer 아키텍처를 도입하여, 기존 WaveNet보다 우수한 다음 샘플 예측 성능(Top-5 Accuracy 기준 최대 85%)을 달성하였다. 특히 Attention 메커니즘과 CNN 기반의 조건부 잠재 표현을 결합하여 성능을 최적화하였다. 비록 조건 없는 생성 단계에서는 음악적 완성도가 낮았으나, 이 연구는 향후 TTS, 소음 제거, 악기 변환 등 WaveNet 기반의 기존 오디오 생성 파이프라인을 Transformer로 대체하여 성능을 높일 수 있는 가능성을 제시하였다.
