# END-TO-END INTEGRATION OF SPEECH RECOGNITION, DEREVERBERATION, BEAMFORMING, AND SELF-SUPERVISED LEARNING REPRESENTATION

Yoshiki Masuyama, Xuankai Chang, Samuele Cornell, Shinji Watanabe, Nobutaka Ono (2023)

## 🧩 Problem to Solve

본 논문은 소음이 심하고 잔향(reverberation)이 존재하는 환경에서 자동 음성 인식(Automatic Speech Recognition, ASR)의 성능을 향상시키는 문제를 해결하고자 한다. 일반적으로 딥러닝 기반의 ASR은 깨끗한 음성 데이터에서는 높은 성능을 보이지만, 실제 환경에서 발생하는 배경 소음과 잔향은 음성 신호를 왜곡시켜 인식률을 크게 떨어뜨리는 주요 원인이 된다.

특히 기존의 단일 채널 음성 향상(Speech Enhancement, SE) 방식은 다중 마이크로폰이 제공하는 공간적 정보(spatial diversity)를 활용하지 못한다는 한계가 있다. 따라서 본 연구의 목표는 다중 채널 음성 향상 기술과 자기지도학습 표현(Self-Supervised Learning Representation, SSLR) 및 E2E ASR을 하나의 신경망으로 통합하여, 소음 및 잔향에 강건한 인식 시스템인 MultiIRIS를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 다중 채널 신호의 공간적 이점을 활용하는 빔포밍(Beamforming) 기술과, 대규모 unlabeled 데이터로 사전 학습된 강력한 음성 표현 모델인 WavLM을 E2E ASR 프레임워크 내에 통합하는 것이다.

중심적인 설계 직관은 다음과 같다. 첫째, WPD(Weighted Power Minimization Distortionless Response) 빔포머를 통해 잔향 제거(Dereverberation)와 소음 제거(Denoising)를 동시에 수행하여 깨끗한 단일 채널 신호를 얻는다. 둘째, 이렇게 정제된 신호를 WavLM에 입력하여 강건한 특징 벡터를 추출한다. 셋째, 이 모든 과정을 단일 ASR 기준(Criterion)으로 공동 최적화(Joint Optimization)함으로써, 음성 향상 모듈이 단순히 신호를 깨끗하게 만드는 것을 넘어 ASR 성능을 극대화하는 방향으로 학습되도록 유도한다.

## 📎 Related Works

기존의 다중 채널 음성 향상 연구에서는 잔향 제거를 위한 WPE(Weighted Prediction Error)와 소음 제거를 위한 MVDR(Minimum Variance Distortionless Response) 또는 MPDR(Minimum Power Distortionless Response) 빔포머가 널리 사용되었다. 최근에는 이 둘을 통합하여 동시에 처리하는 WPD 빔포머가 제안되었으나, 초기 WPD는 반복적인 최적화 과정이 필요하여 신경망과의 통합이 어려웠다. 이를 해결하기 위해 NN을 통해 닫힌 형태(closed-form)로 필터를 추정하는 방식이 제안되었으며, 본 논문은 이 방식을 채택한다.

또한, 단일 채널 환경에서 SE와 SSLR, ASR을 통합한 IRIS라는 시스템이 제안된 바 있다. 그러나 IRIS는 Fully-neural SE 방식을 사용하므로 다중 채널의 공간 정보를 충분히 활용하지 못하며, 실제 데이터에서의 도메인 불일치(domain mismatch) 문제에 취약할 수 있다. MultiIRIS는 이러한 한계를 극복하기 위해 검증된 빔포밍 기술을 NN과 결합하여 모듈성을 유지하면서도 강건성을 높였다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

MultiIRIS는 다음과 같은 파이프라인으로 구성된다.
$$\hat{S} = \text{SE}(X) \rightarrow Z = \text{SSLR}(\hat{S}) \rightarrow \hat{C} = \text{ASR}(Z)$$
여기서 $X$는 다중 채널 입력 신호, $\hat{S}$는 향상된 음성, $Z$는 추출된 특징, $\hat{C}$는 최종 전사(transcription) 결과이다.

### 2. WPD 빔포머를 통한 음성 향상 (SE)

다중 채널 입력 $x(t, f)$를 타겟 음원 $s(t, f)$, 후기 잔향 $r(t, f)$, 소음 $n(t, f)$의 합으로 모델링한다. WPD 빔포머의 목표는 타겟 음원의 왜곡을 최소화하면서 전체 출력을 최소화하는 필터 $w(f)$를 찾는 것이다.

필터 $w(f)$는 다음과 같이 계산된다.
$$w(f) = \frac{R^{-1}(f)H(f)}{\text{Trace}[R^{-1}(f)H(f)]}u$$
여기서 $u$는 참조 마이크로폰을 나타내는 one-hot 벡터이며, $R(f)$는 시간-주파수(T-F) 마스크 $M^m(t, f)$를 사용하여 계산된 가중치 다중 탭 공간 공분산 행렬이다. $H(f)$는 타겟 음원의 공간 공분산 행렬이다. T-F 마스크는 BLSTM 기반의 NN이 추정하며, $\text{CI-SDR}$ 손실 함수를 통해 학습된다.

### 3. WavLM을 이용한 특징 추출 (SSLR)

본 시스템은 WavLM을 사용하여 향상된 신호 $\hat{S}$로부터 특징 $Z$를 추출한다. WavLM의 여러 Transformer 레이어 출력들의 가중 합(weighted sum)을 사용하여 최종 특징을 생성한다.
$$Z = \sum_{l=0}^{L} \alpha_l Z_l$$
여기서 $\alpha_l$은 학습 가능한 가중치이며, $\sum \alpha_l = 1$을 만족해야 한다.

### 4. Joint CTC/Attention 기반 ASR

ASR 모델은 Conformer Encoder와 Transformer Decoder로 구성된 Joint CTC/Attention 구조를 사용한다. 학습 목표 함수는 CTC 손실과 Attention 손실의 가중 합으로 정의된다.
$$\mathcal{L} = \beta \log p_{\text{ctc}}(C|Z) + (1-\beta) \log p_{\text{att}}(C|Z)$$

### 5. 학습 절차

전체 시스템을 처음부터 공동 학습시키면 불안정하므로 다음과 같은 단계적 학습을 수행한다.

1. **개별 사전 학습**: SE 모델은 시뮬레이션 데이터로 $\text{CI-SDR}$ 손실을 통해 학습하고, ASR 모델은 단일 채널 데이터로 학습한다.
2. **공동 미세 조정(Joint Fine-tuning)**: 사전 학습된 SE와 ASR 모델을 결합하여 ASR 성능을 극대화하도록 미세 조정한다. 이때, 이미 대규모 데이터로 학습된 WavLM의 파라미터는 고정(freeze)하여 계산 비용을 줄이고 과적합을 방지한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: $\text{CHiME-4}$ (2채널, 6채널) 및 $\text{REVERB}$ (8채널) 데이터셋을 사용하였다.
- **평가 지표**: $\text{WER}$ (Word Error Rate)을 주 지표로 사용하였으며, 음성 향상 성능 평가를 위해 $\text{SDR}$, $\text{STOI}$, $\text{PESQ}$를 측정하였다.

### 2. 주요 결과

- **SSLR의 효과**: $\text{CHiME-4}$ 데이터셋에서 Fbank, HuBERT, WavLM을 비교한 결과, WavLM이 가장 낮은 WER을 기록하였다. 이는 소음 및 중첩 음성으로 사전 학습된 WavLM이 강건한 표현력을 가짐을 보여준다.
- **빔포머 비교**: MPDR, MVDR과 비교했을 때, 잔향 제거와 소음 제거를 동시에 수행하는 WPD 빔포머가 가장 우수한 ASR 성능을 보였다. 특히 공동 학습을 적용한 WPD 기반 시스템이 가장 뛰어난 결과를 냈다.
- **SOTA 달성**: $\text{CHiME-4}$ 6채널 트랙에서 $\text{WER } 1.77\%$를 달성하여 문헌상 최고 성능을 기록하였다.
- **일반화 성능**: $\text{REVERB}$ 데이터셋에서도 기존 베이스라인 및 SOTA 시스템보다 월등한 성능 향상을 보였으며, $\text{CHiME-4}$에서 학습된 모델을 적용했을 때도 어느 정도의 성능이 유지됨을 확인하였다.

## 🧠 Insights & Discussion

본 연구의 가장 큰 강점은 **모듈성(Modularity)**과 **통합 최적화**의 조화이다. Fully-neural 방식이 아닌 마스크 기반 빔포밍을 사용함으로써, 다양한 마이크로폰 배열 구조에 유연하게 대응할 수 있고 시스템의 해석 가능성을 높였다. 또한, SSLR 모델(WavLM)을 중간에 배치함으로써 대규모 unlabeled 데이터의 지식을 활용하면서도, 앞단의 SE 모듈이 ASR의 최종 목적에 맞게 최적화되도록 설계하였다.

한계점으로는 WavLM 사전 학습에 사용된 외부 데이터가 $\text{CHiME-4}$ 챌린지의 엄격한 규칙(외부 데이터 사용 금지)에는 어긋난다는 점이 있다. 하지만 이는 외부 자원을 활용할 수 있는 실제 환경에서의 잠재력을 보여준다는 점에서 의미가 있다. 또한, $\text{REVERB}$ 데이터셋에서 $\text{CHiME-4}$ 학습 모델의 성능이 일부 저하된 것은 두 데이터셋 간의 잔향 시간(reverberation time) 차이에서 기인한 것으로 분석된다.

## 📌 TL;DR

본 논문은 다중 채널 음성 향상(WPD Beamformer), 자기지도학습 표현(WavLM), 그리고 E2E ASR을 하나로 통합한 **MultiIRIS** 시스템을 제안한다. 각 모듈을 사전 학습시킨 후 ASR 기준으로 공동 미세 조정함으로써 소음과 잔향이 심한 환경에서도 매우 낮은 WER을 달성하였으며, 특히 $\text{CHiME-4}$ 6채널 트랙에서 SOTA 성능을 기록하였다. 이 연구는 다중 채널 공간 정보와 대규모 사전 학습 표현의 결합이 강건한 음성 인식 시스템 구축에 필수적임을 시사한다.
