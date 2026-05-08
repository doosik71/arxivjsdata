# MULTI-ENCODER MULTI-RESOLUTION FRAMEWORK FOR END-TO-END SPEECH RECOGNITION

Ruizhi Li, Xiaofei Wang, Sri Harish Mallidi, Takaaki Hori, Shinji Watanabe, Hynek Hermansky (2018)

## 🧩 Problem to Solve

본 논문은 종단간 자동 음성 인식(End-to-End Automatic Speech Recognition, ASR) 시스템에서 인코더의 구조적 한계로 인해 발생하는 정보 손실 문제를 해결하고자 한다. 기존의 ASR 시스템에서는 연산 복잡도를 줄이고 강건성을 높이기 위해 RNN의 temporal subsampling이나 CNN의 max-pooling 기법을 사용하는데, 이러한 기법들은 필연적으로 시간적 해상도(temporal resolution)의 손실을 야기한다.

또한, 기존의 Joint CTC/Attention 모델은 단일 인코더를 사용함으로써 다양한 수준의 음향 정보를 동시에 포착하는 데 한계가 있다. 따라서 본 연구의 목표는 서로 다른 아키텍처와 시간적 해상도를 가진 복수의 인코더를 병렬로 배치하여, 상호 보완적인 음향 정보를 추출하고 이를 효과적으로 융합하는 Multi-Encoder Multi-Resolution (MEMR) 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 서로 다른 특성을 가진 두 개의 이질적(heterogeneous) 인코더를 병렬로 구성하여, 하나의 인코더가 놓칠 수 있는 정보를 다른 인코더가 보완하도록 설계한 것이다.

주요 기여 사항은 다음과 같다.

1. **Multi-Resolution 인코더 구성**: 시간적 해상도가 다른 두 가지 인코더(RNN 기반 및 CNN-RNN 기반)를 통해 다각적인 음향 특징을 추출한다.
2. **Hierarchical Attention Network (HAN) 도입**: 각 인코더에서 추출된 서로 다른 수준의 정보를 동적으로 가중합 하여 최적의 컨텍스트 벡터를 생성하는 계층적 어텐션 메커니즘을 적용하였다.
3. **Per-encoder CTC 적용**: 각 인코더의 구조와 해상도가 다르므로, 개별 인코더에 전용 CTC 네트워크를 배치하여 각 스트림의 정렬(alignment) 과정을 독립적으로 가이드하였다.

## 📎 Related Works

기존의 ASR 시스템은 DNN을 통해 음소 상태를 예측하고 언어 모델과 결합하는 하이브리드(Hybrid) 방식이 주류였으나, 이는 발음 사전(pronunciation dictionary)과 같은 수작업 기반의 언어적 가정이 필요하다는 단점이 있었다. 이를 해결하기 위해 CTC와 Attention 기반의 Encoder-Decoder 모델이 등장하였다.

- **CTC**: 정렬 단계 없이 음성 벡터를 문자열로 매핑할 수 있으나, 라벨 시퀀스의 조건부 독립성 가정으로 인해 유연성이 떨어진다.
- **Attention-based 모델**: 조건부 독립성 가정 없이 유연한 모델링이 가능하지만, 음성 인식의 핵심인 단조성(monotonic property)을 유지하는 데 어려움이 있다.
- **Joint CTC/Attention**: 두 방식의 장점을 결합하여 Multi-Task Learning (MTL) 방식으로 학습하며, 현재 종단간 ASR의 표준적인 접근법으로 자리 잡았다.

본 논문은 이러한 Joint 프레임워크를 확장하여, 단일 인코더의 한계를 극복하기 위해 다중 스트림(multi-stream) 패러다임을 도입함으로써 기존 연구들과 차별점을 둔다.

## 🛠️ Methodology

### 전체 시스템 구조

MEMR 프레임워크는 두 개의 병렬 인코더, 각 인코더에 대응하는 CTC 네트워크, 계층적 어텐션 융합 층, 그리고 최종 디코더로 구성된다.

### 1. Multi-Encoder 구성

두 인코더는 동일한 입력 특징을 받지만 서로 다른 해상도로 처리한다.

- **Encoder 1 (RNN-based)**: subsampling 없이 BLSTM 레이어만으로 구성되어 원본 시간 해상도를 유지한다.
  $$h^1_t = \text{Encoder}_1(X), \text{BLSTM}$$
- **Encoder 2 (CNN-RNN-based)**: VGG 네트워크의 초기 레이어와 BLSTM을 결합한 VGGBLSTM 구조이다. CNN의 풀링 레이어를 통해 시간 해상도를 4배로 감소시켜 국부적인 상관관계를 포착한다.
  $$h^2_t = \text{Encoder}_2(X), \text{VGGBLSTM}$$

### 2. Hierarchical Attention (계층적 어텐션)

두 인코더에서 나온 서로 다른 해상도의 특징을 융합하기 위해 두 단계의 어텐션을 사용한다.

- **Frame-level Attention**: 각 인코더 내에서 시간축에 대한 어텐션을 수행하여 각 인코더별 컨텍스트 벡터 $r^1_l$과 $r^2_l$을 생성한다.
  $$r^1_l = \sum_{t=1}^{T} a^1_{lt} h^1_t, \quad r^2_l = \sum_{t=1}^{T/4} a^2_{lt} h^2_t$$
- **Stream-level Attention**: 이전 디코더 상태 $q_{l-1}$과 각 인코더의 컨텍스트 벡터를 기반으로, 어떤 인코더의 정보가 더 중요한지 결정하는 가중치 $\beta_{li}$를 계산하여 최종 융합 컨텍스트 벡터 $r_l$을 생성한다.
  $$r_l = \beta_{l1} r^1_l + \beta_{l2} r^2_l, \quad \beta_{li} = \text{ContentAttention}(q_{l-1}, r^i_l)$$

### 3. Per-encoder CTC 및 학습 목표

각 인코더가 서로 다른 구조를 가지므로, 공통의 CTC를 사용하는 대신 인코더별로 독립적인 CTC를 적용한다. 학습 시의 전체 목적 함수 $L_{MTL}$은 다음과 같이 정의된다.
$$\mathcal{L}_{MTL} = \lambda \log p_{ctc}(C|X) + (1-\lambda) \log p^\dagger_{att}(C|X)$$
여기서 $\log p_{ctc}(C|X)$는 각 인코더 CTC 손실의 평균으로 계산된다.
$$\log p_{ctc}(C|X) = \frac{1}{2} (\log p_{ctc1}(C|X) + \log p_{ctc2}(C|X))$$

## 📊 Results

### 실험 설정

- **데이터셋**: WSJ1 (81시간) 및 CHiME-4 (18시간, 소음 환경)
- **입력 특징**: 80차원 mel-scale filterbank coefficients + 3차원 pitch features
- **기준선 및 지표**: 단일 인코더(BLSTM, VGGBLSTM) 기반의 Joint CTC/Attention 모델과 비교하였으며, 성능 지표로는 Word Error Rate (WER)를 사용하였다.

### 주요 결과

- **성능 향상**: MEMR 모델은 단일 인코더 모델 대비 CHiME-4에서 9.6%, WSJ1에서 21.7%의 상대적 WER 감소를 달성하였다.
- **SOTA 달성**: WSJ eval92 테스트 셋에서 3.6% WER을 기록하며, 당시 종단간(end-to-end) 시스템 중 최고 성능을 기록하였다.
- **다중 해상도의 효과**: 두 인코더의 subsampling 계수 차이가 클수록(더 이질적일수록) 성능이 향상됨을 확인하였다. 예를 들어, $(s_1, s_2) = (1, 4)$일 때 $(4, 4)$보다 훨씬 낮은 WER을 보였다.
- **HAN의 유효성**: 인코더 2의 레이어 수를 의도적으로 줄여 약하게 만들었을 때, HAN이 이를 감지하고 인코더 1에 더 높은 가중치를 부여하는 것을 확인하여 동적 융합 메커니즘이 정상적으로 작동함을 입증하였다.

## 🧠 Insights & Discussion

본 논문은 음성 인식에서 시간적 해상도의 보존과 연산 효율성 사이의 트레이드-오프를 '병렬적 이질적 구조'를 통해 해결하였다. 특히, 단순히 여러 인코더를 사용하는 것에 그치지 않고, 각 인코더의 특성에 맞는 독립적인 CTC를 부여하여 정렬 문제를 개별적으로 해결한 점이 성능 향상의 주요 요인으로 분석된다.

또한, 계층적 어텐션을 통해 모델이 상황에 따라 어떤 인코더의 정보에 더 의존할지 스스로 결정하게 함으로써, 정적인 융합 방식보다 유연한 대응이 가능함을 보여주었다. 다만, 인코더가 늘어남에 따라 모델의 파라미터 수와 연산량이 증가한다는 점은 향후 효율성 측면에서 고려해야 할 과제로 보인다.

## 📌 TL;DR

본 연구는 서로 다른 시간 해상도를 가진 두 개의 인코더(BLSTM 및 VGGBLSTM)와 각 인코더 전용 CTC, 그리고 이를 융합하는 계층적 어텐션 네트워크(HAN)를 결합한 MEMR 프레임워크를 제안하였다. 실험 결과 WSJ1 데이터셋에서 3.6%의 WER이라는 획기적인 성능 향상을 거두었으며, 이는 다중 해상도 특징 추출과 동적 정보 융합이 종단간 음성 인식 성능을 크게 높일 수 있음을 시사한다.
