# INTERMPL: MOMENTUM PSEUDO-LABELING WITH INTERMEDIATE CTC LOSS

Yosuke Higuchi, Tetsuji Ogawa, Tetsunori Kobayashi, Shinji Watanabe (2023)

## 🧩 Problem to Solve

본 연구는 End-to-End 자동 음성 인식(Automatic Speech Recognition, ASR) 모델 학습에 필요한 방대한 양의 레이블링 데이터 의존성 문제를 해결하고자 한다. 이를 위해 레이블이 없는 데이터를 활용하는 준지도 학습(Semi-supervised Learning) 기법 중 하나인 Pseudo-labeling(PL)에 주목한다.

특히, 기존의 Momentum Pseudo-labeling(MPL)은 CTC(Connectionist Temporal Classification) 기반 모델을 사용하여 실시간으로 pseudo-label을 생성하고 학습하는 구조를 가진다. CTC는 추론 속도가 빠르고, Autoregressive 모델(예: Attention-based Encoder-Decoder, Transducer)에서 발생하는 Label collapse(단어 생략 또는 반복) 현상에 강건하다는 장점이 있다.

그러나 CTC는 출력 토큰들이 서로 조건부 독립(Conditional Independence)이라고 가정하는 한계가 있으며, 이로 인해 문맥 정보를 충분히 활용하지 못해 Autoregressive 모델보다 성능이 낮게 나타난다. 결과적으로 이러한 CTC의 근본적인 한계는 이를 기반으로 하는 MPL의 성능 향상 폭을 제한하는 요소가 된다. 따라서 본 논문의 목표는 CTC의 장점은 유지하면서 조건부 독립 가정을 완화하여 MPL의 성능을 극대화하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 CTC 기반 모델의 성능을 높이기 위해 제안된 **Intermediate CTC Loss(중간 층 CTC 손실)** 개념을 MPL 프레임워크에 통합하는 것이다.

구체적으로, Self-conditional CTC(SC-CTC)와 Hierarchical conditional CTC(HC-CTC)를 도입하여 인코더의 중간 레이어에 보조적인 CTC 손실을 적용함으로써 모델이 더 풍부한 문맥 정보를 학습하도록 유도한다. 또한, MPL의 반복 학습 과정에서 중간 레이어의 pseudo-label을 어떻게 생성하고 이를 어떻게 지도 신호로 사용할 것인지에 대한 두 가지 전략(InterMPL 및 InterMPL-Last)을 제안하여 준지도 학습의 효율성을 높였다.

## 📎 Related Works

**1. CTC 및 Intermediate CTC**
CTC는 입력 시퀀스와 출력 시퀀스 간의 모든 가능한 정렬(Alignment)을 고려하여 학습하는 방식이다. 하지만 앞서 언급한 조건부 독립 가정으로 인해 성능 한계가 존재한다. 이를 해결하기 위해 중간 레이어에 보조 손실을 추가하는 Intermediate CTC 연구들이 진행되었으며, 특히 SC-CTC는 중간 예측값을 다음 레이어의 조건으로 입력하여 문맥화를 유도하고, HC-CTC는 출력 단위의 입도를 계층적으로 다르게 설정하여 점진적인 생성을 학습하게 한다.

**2. Momentum Pseudo-labeling (MPL)**
MPL은 Online 모델과 Offline 모델의 쌍을 유지하는 Mean Teacher 프레임워크를 기반으로 한다. Offline 모델은 Online 모델의 가중치에 대한 지수 이동 평균(Exponential Moving Average, EMA)을 유지하며, 실시간으로 pseudo-label을 생성하여 Online 모델을 지도한다. 이 과정이 반복되면서 pseudo-label의 품질이 점진적으로 향상되는 구조이다.

## 🛠️ Methodology

### 전체 시스템 구조 및 파이프라인

본 연구에서 제안하는 **InterMPL**은 다음과 같은 단계로 진행된다. 먼저 SC-CTC 또는 HC-CTC를 사용하여 중간 손실이 적용된 Seed 모델을 학습시킨다. 이후 이 Seed 모델을 초기값으로 하여 Online/Offline 모델 쌍을 구성하고 준지도 학습을 수행한다.

### 주요 구성 요소 및 학습 절차

**1. CTC 및 Intermediate Loss 정의**
기본적인 CTC 손실 함수는 다음과 같이 정의된다.
$$\mathcal{L}_{ctc}(W|H) = -\log \sum_{A \in \mathcal{B}^{-1}(W)} \prod_{t} p(a_t|H)$$
여기서 $W$는 정답 시퀀스, $H$는 인코더의 출력, $\mathcal{B}^{-1}(W)$는 $W$로 축약될 수 있는 모든 가능한 정렬 경로 $A$의 집합을 의미한다.

Intermediate CTC loss는 인코더의 특정 중간 레이어 집합 $\mathcal{K}$에 대해 CTC 손실을 합산하여 계산한다.
$$\mathcal{L}_{ic}(W|O) = \sum_{k \in \mathcal{K}} \mathcal{L}_{ctc}(W|H^{(k)})$$

**2. Pseudo-label 생성 및 학습 전략**
Offline 모델(파라미터 $\phi$)이 생성한 pseudo-label $\hat{W}$를 사용하여 Online 모델(파라미터 $\xi$)을 학습시킨다. 본 논문은 두 가지 세부 전략을 제시한다.

* **InterMPL (Fig 1b):**
    Offline 모델의 각 중간 레이어 $k \in \mathcal{K}$에서 각각 독립적인 pseudo-label $\hat{W}^{(k)}_m$을 생성한다. Online 모델은 자신의 대응하는 레이어 $k$에서 생성된 $\hat{W}^{(k)}_m$을 타겟으로 학습한다.
    $$\mathcal{L}_{ic}(\{\hat{W}^{(k)}_m\}_{k \in \mathcal{K}}|O_m, \xi) = \sum_{k \in \mathcal{K}} \mathcal{L}_{ctc}(\hat{W}^{(k)}_m|H^{(k)}, \xi)$$
    이 방식은 특히 각 레이어마다 출력 단위의 입도가 다른 HC-CTC에 적합하다.

* **InterMPL-Last (Fig 1c):**
    Offline 모델의 가장 마지막 레이어에서 생성된 최적의 pseudo-label $\hat{W}^{(K)}_m$ 하나만을 생성한다. Online 모델의 모든 중간 레이어 $\mathcal{K}$는 이 하나의 고품질 레이블을 동일한 타겟으로 하여 학습한다.
    $$\mathcal{L}_{ic}(\hat{W}^{(K)}_m|O_m, \xi) = \sum_{k \in \mathcal{K}} \mathcal{L}_{ctc}(\hat{W}^{(K)}_m|H^{(k)}, \xi)$$
    이 방식은 모든 레이어가 동일한 타겟을 공유하는 SC-CTC에 더 적합한 구조이다.

**3. 모델 업데이트**
Online 모델은 Gradient Descent를 통해 업데이트되며, Offline 모델은 다음과 같은 EMA 방식을 통해 온라인 모델의 가중치를 추적한다.
$$\phi \leftarrow \alpha\phi + (1-\alpha)\xi$$

## 📊 Results

### 실험 설정

* **데이터셋:** LibriSpeech(LS) 및 TED-LIUM3(TED3).
* **준지도 설정:**
  * In-domain: LS-100(레이블 있음) $\rightarrow$ LS-360 또는 LS-860(레이블 없음).
  * Out-of-domain: LS-100(레이블 있음) $\rightarrow$ TED3(레이블 없음).
* **모델:** 18개 레이어의 Conformer 인코더.
* **지표:** Word Error Rate(WER) 및 WER Recovery Rate(WRR). WRR은 Seed 모델과 Oracle 모델 사이의 성능 간극을 준지도 학습이 얼마나 메웠는지를 나타낸다.

### 주요 결과

**1. In-domain 성능 (LibriSpeech)**
InterMPL 및 InterMPL-Last 모두 기존 MPL 대비 유의미한 성능 향상을 보였다. 특히 HC-CTC 기반의 InterMPL은 다양한 입도의 pseudo-label을 활용함으로써 효과적인 성능 향상을 이루었으며, InterMPL-Last는 SC-CTC 기반 학습 시 가장 높은 WRR을 기록하였다.

**2. Out-of-domain 성능 (TED-LIUM3)**
도메인이 다른 환경에서도 InterMPL 계열의 방법론들이 MPL보다 안정적인 성능을 보였다. 특히 InterMPL-Last가 가장 낮은 WER을 기록했는데, 이는 도메인 불일치 상황에서는 중간 레이어의 개별 예측치보다 최종 레이어의 정제된 레이블을 사용하는 것이 더 유리함을 시사한다. 반면, HC-CTC는 큰 어휘 사전 크기로 인해 일반화 능력이 떨어져 Out-of-domain 설정에서는 상대적으로 이점이 적었다.

**3. Ablation Study**
SC-CTC나 HC-CTC로 초기화된 Seed 모델을 사용하더라도, 준지도 학습 과정에서 Intermediate loss를 제거하면 성능이 급격히 저하되는 것을 확인하였다. 이는 단순히 좋은 모델로 시작하는 것보다, 학습 과정 내내 중간 레이어에 지도 신호를 주는 것이 결정적임을 의미한다.

## 🧠 Insights & Discussion

본 연구는 CTC 기반 ASR 모델의 고질적인 문제인 조건부 독립 가정을 중간 레이어의 보조 손실(Intermediate Loss)을 통해 효과적으로 완화할 수 있음을 보여주었다. 특히 이를 MPL의 Online/Offline 구조에 통합함으로써, 데이터가 부족한 상황에서도 모델이 더 견고한 표현력을 학습할 수 있게 하였다.

**강점 및 해석:**

* InterMPL-Last의 우수성은 준지도 학습에서 '정확한 타겟'의 중요성을 다시 한번 확인시켜 준다. 특히 도메인이 다른 데이터를 다룰 때는 중간 단계의 불확실한 예측치보다 최종 결과물을 타겟으로 삼는 것이 안정적이다.
* HC-CTC의 성능 향상은 서로 다른 해상도의 정보를 단계적으로 학습하는 것이 ASR의 성능을 높이는 데 기여함을 입증한다.

**한계 및 논의사항:**

* HC-CTC의 일반화 성능 부족 문제는 데이터셋의 텍스트 분포에 따라 어휘 사전 구성이 성능에 큰 영향을 미칠 수 있음을 보여준다.
* 본 연구는 CTC 기반의 모델에 집중하였으나, 향후 외부 언어 모델(LM)을 InterMPL의 중간 예측 단계에 결합(Shallow Fusion)한다면 더 비약적인 성능 향상이 가능할 것으로 보인다.

## 📌 TL;DR

본 논문은 CTC 기반 준지도 학습 방법인 MPL의 성능을 높이기 위해 **중간 레이어 CTC 손실(Intermediate CTC Loss)**을 도입한 **InterMPL**을 제안한다. Seed 모델 학습부터 준지도 학습 단계까지 중간 레이어에 보조적인 지도 신호를 제공함으로써 CTC의 조건부 독립 가정 한계를 극복하였다. 실험 결과, 특히 InterMPL-Last 방식이 In-domain 및 Out-of-domain 설정 모두에서 기존 MPL보다 뛰어난 성능을 보였으며, 이는 향후 데이터 효율적인 E2E ASR 연구에 중요한 방법론적 기반을 제공한다.
