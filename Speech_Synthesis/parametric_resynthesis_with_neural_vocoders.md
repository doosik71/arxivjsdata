# PARAMETRIC RESYNTHESIS WITH NEURAL VOCODERS

Soumi Maiti, Michael I. Mandel (2019)

## 🧩 Problem to Solve

본 논문은 기존의 잡음 억제(Noise Suppression) 시스템이 출력 음성의 품질을 저하시키는 문제를 해결하고자 한다. 전통적인 음성 향상(Speech Enhancement) 방법들은 잡음이 섞인 신호를 수정하여 깨끗한 음성과 유사하게 만드는 방식을 취한다. 그러나 이러한 방식은 음성 성분을 과도하게 억제(Over-suppression)하거나 잡음을 충분히 제거하지 못하는 저억제(Under-suppression) 문제로 인해 신호 왜곡이 발생하며, 결과적으로 음성 품질이 손상되는 한계가 있다.

따라서 본 연구의 목표는 텍스트-음성 합성(TTS) 분야에서 검증된 **Neural Vocoder**의 고품질 음성 생성 능력을 잡음 억제에 활용하는 것이다. 즉, 잡음 섞인 신호를 필터링하는 대신, 깨끗한 음성의 특징을 예측하고 이를 통해 음성을 완전히 다시 생성하는 **Parametric Resynthesis (PR)** 방식을 제안하여 왜곡 없는 고품질의 음성을 복원하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 음성 향상 과정에 파라메트릭 합성(Parametric Synthesis) 개념을 도입하는 것이다. 단순히 잡음을 제거하는 마스킹 방식에서 벗어나, 다음과 같은 파이프라인을 구축하였다.

1. **특징 예측**: 잡음이 섞인 음성에서 깨끗한 음성의 acoustic representation(log mel-spectrogram)을 예측한다.
2. **신경망 기반 재합성**: 예측된 특징을 조건(Condition)으로 하여 Neural Vocoder(WaveNet, WaveGlow)를 통해 깨끗한 음성 파형을 생성한다.

이러한 **Enhancement-by-Synthesis** 접근법은 잡음의 특성에 구애받지 않고 깨끗한 음성 모델만을 학습하여 생성함으로써, 기존의 신호 처리 기반 방식보다 더 자연스럽고 왜곡이 적은 음성을 얻을 수 있다는 직관에 기반한다.

## 📎 Related Works

논문에서는 기존의 음성 향상 및 합성 연구를 다음과 같이 구분하여 설명한다.

- **Concatenative Resynthesis**: 깨끗한 음성 데이터베이스에서 작은 조각들을 이어 붙여 합성하는 방식이다. 품질은 좋으나 화자 의존적이며 방대한 양의 깨끗한 음성 사전이 필요하다는 한계가 있다.
- **Traditional Parametric Synthesis**: 텍스트로부터 음성 파라미터를 예측하고 Vocoder로 생성하는 방식이다. 최근 WaveNet과 같은 Neural Vocoder의 등장으로 품질이 비약적으로 향상되었다.
- **End-to-End Time-domain Models**: SEGAN, Wave-U-Net 등 시간 영역에서 직접 작동하는 모델들이 존재한다. 그러나 이러한 모델들은 잡음과 음성을 모두 모델링해야 하는 반면, 본 제안 모델은 깨끗한 음성만을 모델링하므로 구조가 더 단순하고 잡음 독립적이라는 차별점이 있다.
- **Mask-based Methods**: Chimera++와 같이 마스크를 추정하여 잡음을 제거하는 방식이 있으나, 본 연구의 PR-neural 방식이 주관적/객관적 품질 면에서 더 우수함을 보인다.

## 🛠️ Methodology

### 전체 시스템 구조

본 시스템은 크게 **Prediction Model**과 **Neural Vocoder**라는 두 단계의 파이프라인으로 구성된다.

### 1. Prediction Model (특징 예측 모델)

잡음이 섞인 mel-spectrogram $Y(\omega, t)$를 입력받아 깨끗한 mel-spectrogram $X(\omega, t)$를 예측한다.

- **아키텍처**: 다층 Bidirectional LSTM (3-layer, 400 units each)을 사용한다.
- **손실 함수**: 예측된 $\hat{X}(\omega, t)$와 실제 정답 $X(\omega, t)$ 사이의 평균 제곱 오차(MSE)를 최소화하는 것을 목표로 한다.
$$L = \sum_{\omega, t} \| X(\omega, t) - \hat{X}(\omega, t) \|^2$$

### 2. Neural Vocoders (신경망 보코더)

예측된 mel-spectrogram을 조건으로 하여 음성 파형을 생성하며, 본 논문에서는 서로 다른 구조를 가진 WaveNet과 WaveGlow를 비교 분석한다.

#### WaveNet (Autoregressive Model)

- **구조**: Dilated Causal Convolutional layers를 사용하여 수용 영역(Receptive Field)을 넓히며, 이전 샘플들에 의존하여 현재 샘플을 예측하는 자기회귀(Autoregressive) 방식이다.
- **출력 방식**: 고품질 합성을 위해 $K$-component logistic mixture를 사용하여 확률 밀도 함수를 모델링한다.
- **확률 밀도 함수**:
$$P(x_t | \Theta, X) = \sum_{i=1}^{K} \pi_i \left[ \sigma \left( \frac{\tilde{x}_{ti} + 0.5}{s_i} \right) - \sigma \left( \frac{\tilde{x}_{ti} - 0.5}{s_i} \right) \right]$$
여기서 $\tilde{x}_{ti} = x_t - \mu_i$이며, $\pi, \mu, s$는 모델이 예측하는 파라미터이다.

#### WaveGlow (Flow-based Model)

- **구조**: Glow 개념을 기반으로 하며, 가우시안 분포와 음성 샘플 사이의 가역 변환(Invertible Transformation)을 학습하는 Normalizing Flow 기반 모델이다.
- **특징**: WaveNet과 달리 병렬 생성이 가능하여 추론 속도가 매우 빠르다.
- **손실 함수**: 깨끗한 음성 샘플 $x$의 로그 가능도(log-likelihood)를 최대화하도록 학습한다.
$$\ln P(x|X) = \ln P(z) - \sum_{j=0}^{J} \ln s_j(x, X) - \sum_{k=0}^{K} \ln |W_k|$$

### 3. Joint Training (결합 학습)

예측 모델이 생성한 $\hat{X}$와 실제 보코더가 학습한 $X$ 사이의 괴리를 줄이기 위해 두 모델을 함께 학습시키는 전략을 시도하였다. 보코더의 가능도(Likelihood)와 mel-spectrogram의 MSE 손실을 결합하여 최적화한다.

## 📊 Results

### 실험 설정

- **데이터셋**: LJSpeech(깨끗한 음성) + CHiME-3(환경 잡음)을 혼합하여 사용하였다. SNR은 평균 1dB ($-9\text{dB} \sim 9\text{dB}$) 범위이다.
- **비교 대상**: PR-World (기존 파라메트릭 방식), Oracle Wiener Mask (이상적인 마스크), Chimera++ (최신 소스 분리 모델).
- **평가 지표**:
  - 객관적 지표: SIG(신호 왜곡), BAK(배경 잡음 침투), OVL(전체 품질), PESQ(음질), STOI(명료도).
  - 주관적 지표: MUSHRA 테스트를 통한 품질, 잡음 억제, 전체 품질 및 명료도 점수(0-100).

### 주요 결과

1. **객관적 품질**: PR-neural 모델들이 Chimera++보다 모든 지표에서 우수한 성능을 보였다. Oracle Wiener Mask보다는 약간 낮았으나, 이는 재합성된 음성이 원본과 시간적으로 완벽하게 정렬되지 않아 발생하는 지표상의 손실로 분석된다.
2. **주관적 품질 (MUSHRA)**: **PR-WaveNet**이 모든 품질 점수에서 가장 높은 평가를 받았으며, Oracle Wiener Mask보다도 유의미하게 높은 주관적 품질을 기록하였다.
3. **명료도**: PR-WaveNet이 가장 높은 주관적 명료도를 보였으며, PR-neural 모델들 전반이 Chimera++보다 우수하였다.
4. **속도**: WaveNet은 샘플을 순차적으로 생성하므로 매우 느리다 ($1\text{s}$ 음성 생성에 $\sim 232\text{s}$ 소요). 반면 WaveGlow는 병렬 생성이 가능하여 실시간으로 생성이 가능하다 ($\sim 1\text{s}$ 소요).

## 🧠 Insights & Discussion

### Joint Training의 한계

연구진은 결합 학습(Joint Training)을 통해 성능 향상을 꾀했으나, 실제로는 오히려 성능이 하락하는 결과가 관찰되었다. 분석 결과, 결합 학습 과정에서 예측 모델이 mel-spectrogram을 실제보다 너무 크게(louder) 예측하는 경향이 발생하여, 결과적으로 웅얼거리는 소리(mumbled speech)나 음성 탈락(drop-outs) 현상이 나타났다.

다만, 보코더를 고정한 채 예측 모델만을 보코더 손실 함수를 포함해 최적화했을 때는 SIG와 OVL, 명료도 지표에서 약간의 성능 향상이 있었다.

### 비판적 해석

- **강점**: 단순히 잡음을 깎아내는 마스킹 방식의 한계를 극복하고, 생성 모델을 통해 "깨끗한 음성"을 복원함으로써 주관적 음질을 획기적으로 높였다.
- **한계**: WaveNet의 극도로 느린 추론 속도는 실제 서비스 적용에 큰 걸림돌이 된다. 또한, 현재 모델은 단일 화자 데이터셋(LJSpeech)을 기반으로 하므로 화자 의존성(Speaker-dependence) 문제가 해결되지 않았다.
- **논의사항**: 객관적 지표(PESQ, STOI)와 주관적 평가 간의 괴리가 크다는 점은, 현재의 객관적 지표들이 Neural Vocoder가 생성하는 '자연스러운' 음성을 정확히 측정하지 못하고 있음을 시사한다.

## 📌 TL;DR

본 논문은 잡음 억제를 위해 잡음 섞인 음성에서 깨끗한 mel-spectrogram을 예측하고, 이를 **WaveNet** 또는 **WaveGlow**와 같은 Neural Vocoder로 재합성하는 **Parametric Resynthesis (PR)** 시스템을 제안하였다. 실험 결과, PR-neural 방식은 기존의 마스크 기반 방식(Chimera++)이나 전통적 보코더(WORLD)보다 월등한 음질과 명료도를 보였으며, 특히 **PR-WaveNet**은 매우 느린 속도에도 불구하고 최고의 주관적 음질을 달성하였다. 이 연구는 음성 향상 분야에 생성 모델을 도입하여 '필터링'이 아닌 '재합성' 관점의 접근이 고품질 음성 복원에 효과적임을 입증하였다.
