# Quasi-Periodic WaveNet: An Autoregressive Raw Waveform Generative Model with Pitch-dependent Dilated Convolution Neural Network

Yi-Chiao Wu, Tomoki Hayashi, Patrick Lumban Tobing, Kazuhiro Kobayashi, and Tomoki Toda (2020)

## 🧩 Problem to Solve

본 논문은 vanilla WaveNet (WN) 기반의 오디오 생성 모델이 가진 **제한적인 피치 제어 능력(Limited Pitch Controllability)** 문제를 해결하고자 한다.

WaveNet은 확률적 자기회귀(Autoregressive, AR) 생성 모델로서 고충실도(High-fidelity) 오디오 파형 생성이 가능하지만, 순수하게 데이터 기반(Pure-data-driven)으로 학습되며 오디오 신호에 대한 사전 지식(Prior knowledge)이 부족하다는 한계가 있다. 이로 인해 학습 데이터에서 관찰되지 않은 범위의 기본 주파수($F_0$) 특징이 보조 입력으로 주어질 경우, 오디오 신호의 주기적 성분을 정밀하게 생성하지 못하는 문제가 발생한다.

또한, WaveNet의 고정된 아키텍처는 모든 샘플에 대해 동일한 길이의 Receptive Field를 가정하므로, 준주기적(Quasi-periodic) 특성을 가진 음성 신호를 모델링할 때 불필요하게 많은 샘플을 포함하게 되어 네트워크 규모가 커지고 연산 비용이 증가하는 비효율성을 초래한다. 따라서 본 연구의 목표는 피치에 적응적인 구조를 도입하여 피치 제어 능력을 향상시키고, 연산 효율성을 높인 Quasi-Periodic WaveNet (QPNet)을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **피치-의존적 확장 컨볼루션 신경망(Pitch-dependent Dilated Convolution Neural Network, PDCNN)**과 **계층적 네트워크 구조(Cascaded Network Structure)**를 도입하는 것이다.

1. **PDCNN**: 보조 입력으로 들어오는 $F_0$ 특징에 따라 네트워크의 확장 계수(Dilation size)를 동적으로 변경하여, 각 샘플이 해당 피치에 최적화된 전용 Receptive Field를 갖도록 설계하였다.
2. **계층적 구조**: 고정된 DCNN을 사용하는 Macroblock과 PDCNN을 사용하는 Adaptive Macroblock을 직렬로 연결(Cascade)하여, 음성 신호의 단기 상관관계(Short-term correlation)와 장기 주기적 상관관계(Long-term correlation)를 동시에 모델링하도록 하였다.

## 📎 Related Works

### 1. 기존 보코더 (STRAIGHT, WORLD)

전통적인 소스-필터(Source-filter) 모델 기반의 보코더들은 피치와 음색을 유연하게 조작할 수 있는 능력을 갖추고 있다. 특히 STRAIGHT와 WORLD는 피치 동기화(Pitch-synchronized) 메커니즘을 통해 $F_0$에 독립적인 안정적인 스펙트럼을 추출한다. 그러나 이러한 방식은 위상 정보와 시간적 세부 사항을 손실시켜 음질 저하를 유발한다는 치명적인 한계가 있다.

### 2. 신경망 기반 보코더 (Neural Vocoder)

WaveNet과 같은 AR 모델들은 고해상도 오디오의 장기 의존성을 모델링하여 매우 높은 음질을 구현하였다. 하지만 앞서 언급한 바와 같이, 고정된 구조로 인해 학습 데이터 외의 피치를 생성하는 능력이 부족하며, 거대한 네트워크 크기로 인해 실시간 생성에 어려움이 있다.

### 3. 차별점

본 제안 방법은 기존 신경망 보코더의 고음질 특성을 유지하면서, 전통적인 보코더의 강점인 피치 제어 능력을 확보하기 위해 PDCNN이라는 적응형 구조를 도입하였다. 이는 단순히 데이터를 학습하는 것을 넘어, 오디오의 주기성이라는 물리적 사전 지식을 네트워크 구조 자체에 반영했다는 점에서 기존 접근 방식과 차별화된다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

QPNet은 두 개의 매크로블록(Macroblock)이 직렬로 연결된 구조를 가진다.

- **Macroblock 0 (Fixed)**: vanilla WaveNet과 유사하게 고정된 DCNN을 사용하여 현재 샘플과 인접한 과거 샘플 간의 단기적인 순차적 관계를 모델링한다.
- **Macroblock 1 (Adaptive)**: PDCNN을 사용하여 현재 주기와 과거 주기의 관련 세그먼트 간의 장기적인 주기적 상관관계를 모델링한다.

### 2. Pitch-dependent Dilated Convolution (PDCNN)

PDCNN의 핵심은 확장 계수 $d$를 고정하지 않고, 입력되는 $F_0$ 값에 따라 동적으로 결정하는 것이다.

#### 2.1 확장 계수 결정 방정식

PDCNN의 확장 계수 $d'$는 다음과 같이 계산된다.
$$d' = E_t \times d$$
여기서 $d$는 일반적인 DCNN의 확장 계수이며, $E_t$는 **피치-의존적 확장 계수(Pitch-dependent dilated factor)**이다. $E_t$는 다음 식에 의해 결정된다.
$$E_t = \frac{F_s}{F_{0,t} \times a}$$

- $F_s$: 샘플링 레이트 (Sampling rate)
- $F_{0,t}$: 시간 $t$에서의 기본 주파수 (Fundamental frequency)
- $a$: **Dense factor** (하이퍼파라미터). 한 주기의 신호를 예측할 때 고려할 샘플의 수를 결정하며, 이 값이 클수록 샘플링 그리드가 조밀해진다.

이 구조를 통해 $F_0$가 낮아지면 $E_t$가 커져 Receptive Field가 효율적으로 확장되고, $F_0$가 높아지면 $E_t$가 작아져 좁은 영역을 집중적으로 보게 된다.

### 3. 학습 및 추론 절차

- **입력**: WORLD 보코더로 추출한 mcep(Mel-cepstral coefficients), ap(aperiodicity), $F_0$ 특징을 보조 입력으로 사용한다.
- **출력**: $\mu$-law 알고리즘으로 8비트 양자화된 오디오 샘플의 범주형 분포(Categorical distribution)를 예측한다.
- **손실 함수**: 예측된 분포와 실제 샘플 간의 교차 엔트로피(Cross-entropy) 손실을 최소화하도록 학습한다.
- **추론**: 자기회귀 방식으로 이전 샘플들을 입력으로 하여 다음 샘플을 순차적으로 생성한다.

## 📊 Results

### 1. 사인파 생성 실험 (Sinusoid Generation)

피치 제어 능력을 정량적으로 평가하기 위해 단순 사인파 생성 실험을 수행하였다.

- **측정 지표**: SNR(신호 대 잡음비), $\log F_0$ RMSE(피치 정확도).
- **결과**:
  - **피치 제어력**: QPNet은 학습 데이터 범위 밖(Outside $F_0$ range)의 주파수에 대해서도 매우 낮은 $\log F_0$ RMSE를 기록하여, vanilla WaveNet(WNf)보다 월등한 피치 제어 능력을 보였다.
  - **Dense Factor ($a$) 분석**: $a=2^3$일 때 SNR과 RMSE의 균형이 가장 좋았다. $a$가 너무 작으면($2^0, 2^1$) 불안정하고, 너무 크면($2^6$) PDCNN이 일반 DCNN으로 퇴화하여 성능이 저하되었다.

### 2. 음성 생성 실험 (Speech Generation)

- **데이터셋**: CMU-ARCTIC 및 VCC2018 코퍼스.
- **측정 지표**: MCD(Mel-cepstral distortion, 스펙트럼 재구성 능력), $\log F_0$ RMSE, U/V decision error(유성음/무성음 판별 오류).
- **정량적 결과**:
  - QPNet은 동일 규모의 WNc(Compact WN)보다 MCD와 피치 정확도 면에서 훨씬 뛰어난 성능을 보였다.
  - 특히 $F_0$를 $1/2$배 또는 $3/2$배로 변조한 경우에도 WNf보다 높은 피치 정확도를 유지하였다.
- **주관적 평가**:
  - **MOS (Mean Opinion Score)**: 음성 자연스러움 평가에서 QPNet은 WNc보다 월등히 높았으며, 특히 피치가 낮아진 경우($1/2 F_0$) WNf보다 더 자연스러운 음성을 생성하였다.
  - **ABX Test**: 피치 정확도 선호도 테스트 결과, 대부분의 케이스에서 QPNet이 WNf보다 참조 신호(WORLD 생성 음성)의 피치 궤적과 더 일치함을 확인하였다.

## 🧠 Insights & Discussion

### 1. 강점 및 분석

- **효율적인 Receptive Field 확장**: QPNet은 고정된 거대 네트워크를 사용하는 대신, 피치 정보에 따라 Receptive Field를 동적으로 조절함으로써 파라미터 수를 절반으로 줄이면서도(WNf 대비) 유사하거나 더 나은 성능을 달성하였다.
- **물리적 사전 지식의 효과**: PDCNN을 통해 오디오의 주기성을 네트워크 구조에 직접 반영한 것이 unseen $F_0$ 범위에서도 정밀한 피치 제어를 가능하게 한 핵심 요인이다.

### 2. 한계 및 비판적 해석

- **네트워크 균형 문제**: Full-size QPNet(WNf 구조에 Adaptive block 4개를 추가한 모델)의 피치 정확도가 Compact-size QPNet보다 낮게 나타났다. 이는 Fixed block의 비중이 너무 높으면 Adaptive block의 영향력이 희석되어 피치 제어 능력이 오히려 떨어진다는 것을 시사한다. 즉, 단기-장기 모델링 블록 간의 적절한 비율(Balance)이 필수적이다.
- **메모리 효율성**: 파라미터 수와 생성 시간은 줄었으나, 학습 단계에서의 메모리 사용량은 여전히 WNf와 유사하게 높다. 이는 매우 긴 Effective Receptive Field를 처리해야 하기 때문이며, 향후 메모리 효율화가 필요하다.
- **고주파수에서의 퇴화**: 피치가 매우 높아질 경우 $E_t$ 값이 1 이하로 떨어지며 PDCNN이 일반 DCNN으로 퇴화하는 현상이 발생하며, 이로 인해 고음역대 음성 생성 품질이 다소 저하되는 경향이 있다.

## 📌 TL;DR

본 논문은 WaveNet의 고질적인 문제인 **피치 제어 능력 부족**을 해결하기 위해, $F_0$ 값에 따라 확장 계수가 변하는 **PDCNN**과 **계층적 AR 구조**를 제안한 **QPNet**을 선보였다. 실험 결과, QPNet은 기존 WaveNet보다 훨씬 적은 파라미터로도 학습 데이터 외의 피치를 정확하게 생성할 수 있었으며, 특히 저음역대에서의 음성 생성 품질과 피치 정확도가 크게 향상되었다. 이 연구는 딥러닝 기반 보코더에 신호처리적 사전 지식을 결합하여 효율성과 제어력을 동시에 잡을 수 있음을 보여주었으며, 향후 음성 변환(VC) 및 텍스트-음성 변환(TTS) 시스템의 피치 정밀도 향상에 기여할 것으로 기대된다.
