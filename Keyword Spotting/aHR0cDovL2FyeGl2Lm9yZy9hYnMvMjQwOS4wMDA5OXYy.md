# Query-by-Example Keyword Spotting Using Spectral-Temporal Graph Attentive Pooling and Multi-Task Learning

Zhenyu Wang, Shuyu Kong, Li Wan, Biqiao Zhang, Yiteng Huang, Mumin Jin, Ming Sun, Xin Lei, Zhaojun Yang (2024)

## 🧩 Problem to Solve

본 논문은 사용자가 직접 정의한 키워드를 인식할 수 있는 사용자 맞춤형 키워드 스포팅(Customized Keyword Spotting, KWS) 문제를 해결하고자 한다. 기존의 KWS 시스템은 사전에 정의된 키워드 세트에 의존하므로, 지능형 기기와의 상호작용을 개인화하기 위해서는 사용자가 원하는 임의의 키워드를 인식할 수 있는 능력이 필수적이다.

이러한 맞춤형 KWS를 구현하기 위해 Query-by-Example (QbyE) 방식이 사용된다. QbyE는 사용자가 제공한 소수의 키워드 오디오 샘플을 통해 임베딩을 생성하고, 테스트 샘플과의 유사도를 측정하여 키워드 존재 여부를 판단한다. 하지만 이 과정에서 다음과 같은 문제들이 발생한다.
1. **데이터 분포 불일치**: 사용자가 정의한 키워드가 학습 데이터의 분포와 일치하지 않아 인식 성능이 저하될 수 있다.
2. **하드웨어 제약**: 항상 켜져 있는(always-on) 시스템 특성상, 메모리 사용량을 최소화하고 지연 시간(latency)을 줄여야 하므로 계산 복잡도가 높은 모델을 적용하기 어렵다.
3. **화자 가변성**: 동일한 단어라도 화자에 따라 피치, 톤, 발음이 달라지므로, 화자의 특성에 영향을 받지 않으면서 단어의 언어적 정보만 추출하는 Speaker-invariant 임베딩 학습이 필요하다.

## ✨ Key Contributions

본 논문의 핵심 기여는 화자 독립적(speaker-invariant)이면서 언어적 정보가 풍부한(linguistic-informative) 임베딩을 학습하기 위해 **Spectral-Temporal Graph Attentive Pooling (GAP)**과 **Multi-Task Learning (MTL)** 프레임워크를 제안한 것이다.

핵심 설계 아이디어는 다음과 같다.
- **Spectral-Temporal Graph Attentive Pooling (GAP)**: 그래프 신경망(GNN)을 활용하여 오디오 데이터의 주파수-시간(spectral-temporal) 영역 내의 복잡한 관계를 파악하고, 중요한 특징을 동적으로 추출하여 임베딩의 변별력을 높인다.
- **Multi-Task Learning (MTL)**: 단어 분류, 음소(phoneme) 인식, 화자 식별이라는 세 가지 작업을 동시에 학습시킨다. 특히 화자 식별 작업에는 Gradient Reversal Layer (GRL)를 적용하여 화자 정보를 제거함으로써, 화자에게 무관한 강건한 단어 임베딩을 얻도록 설계하였다.
- **효율적 모델의 잠재력 극대화**: 제안한 프레임워크를 통해 LiCoNet과 같은 가벼운 모델이 Conformer와 같은 무거운 모델에 근접하는 성능을 낼 수 있음을 입증하였다.

## 📎 Related Works

기존의 KWS 연구는 주로 다음과 같은 방향으로 진행되었다.
- **Pre-defined KWS**: Encoder-Decoder 구조나 End-to-End 모델을 통해 특정 키워드를 직접 검출한다. 하지만 이는 사전 정의된 키워드에 대해서만 작동하며, 학습을 위해 많은 양의 타겟 데이터가 필요하다는 한계가 있다.
- **ASR 및 DTW 기반 QbyE**: 자동 음성 인식(ASR) 시스템의 음성학적 사후 확률(phonetic posteriors)을 생성하고, Dynamic Time Warping (DTW)를 통해 쿼리와의 유사도를 측정한다.
- **임베딩 기반 QbyE**: LSTM 기반 인코더나 Multi-head Attention, SoftTriplet loss 등을 사용하여 음향 임베딩을 학습하는 방식이 제안되었다. 특히 Attention 기반 모델은 성능은 우수하지만, 계산 비용과 메모리 부담이 커서 실시간 하드웨어 배포에 부적합하다는 단점이 있다. 이를 해결하기 위해 MLPMixer 등이 시도되었으나, 행렬 전치(matrix transpose) 연산으로 인한 오버헤드가 여전히 존재한다.

## 🛠️ Methodology

### 전체 파이프라인
시스템은 학습 단계에서 **Encoder-Decoder** 구조를 취한다. 인코더가 입력 오디오에서 특징을 추출하고, Pooling 레이어를 통해 차원을 축소하여 고정 길이의 임베딩을 생성하면, 디코더가 이를 분류한다. 테스트 단계에서는 사용자가 등록한 소수 샘플의 임베딩과 테스트 입력의 임베딩 간 코사인 거리를 비교하여 키워드를 검출한다.

### Feature Encoder
본 논문에서는 세 가지 인코더 아키텍처를 조사하였다.
1. **LiCoNet**: MobileNetV2 기반의 하드웨어 효율적 구조로, 1D Convolution 레이어들로 구성된 bottleneck 구조를 사용하여 추론 효율성을 극대화하였다.
2. **Conformer**: Convolution과 Self-attention 메커니즘을 결합하여 지역적 특징과 전역적 상호작용을 동시에 캡처하는 고성능 시퀀스 모델이다.
3. **ECAPA-TDNN**: 화자 확인(Speaker Verification) 작업에서 널리 쓰이는 구조로, SE-Res2Block과 Attentive Statistical Pooling을 포함한다.

### Feature Aggregator (Pooling)
시퀀스 데이터를 고정 길이 임베딩으로 변환하기 위해 두 가지 전략을 비교하였다.
- **Attentive Statistic Pooling (ASP)**: 시간 축을 따라 각 요소의 중요도에 가중치를 두어 통계량을 추출한다.
- **Spectral-Temporal Graph Attentive Pooling (GAP)**: 그래프 어텐션 네트워크(GAT)와 그래프 풀링을 사용하여 주파수-시간 데이터의 복잡한 관계를 동적으로 학습하며 특징을 추출한다.

### Loss Function 및 Multi-Task Learning
단어의 변별력을 높이기 위해 다음과 같은 손실 함수들을 조합한 Multi-Task Learning을 제안한다.

**1. Word Discrimination (단어 변별력)**
- **Additive Angular Margin (AAM) Loss**: 클래스 간의 각도 마진을 강제하여 임베딩 공간에서 클래스 간 분리도를 높인다.
$$L_{aam} = -\log \frac{e^{s\cos(\theta_{y_i}+m)}}{e^{s\cos(\theta_{y_i}+m)} + \sum_{j=1, j \neq y_i}^C e^{s\cos(\theta_j)}}$$
- **SoftTriplet Loss**: Triplet loss와 Softmax loss를 결합하여 클래스 내 분산은 줄이고 클래스 간 거리는 넓힌다.
$$L_{st}(x_i) = -\log \frac{\exp(\lambda(S'_{i,y_i} - \delta))}{\exp(\lambda(S'_{i,y_i} - \delta)) + \sum_j \exp(\lambda S'_{i,j})}$$

**2. Speaker Variability (화자 가변성 제거)**
화자 독립적인 임베딩을 얻기 위해 **Gradient Reversal Layer (GRL)**를 포함한 역화자 손실(Reverse Speaker Loss)을 설계하였다. 순전파 시에는 동일하게 동작하지만, 역전파 시에는 기울기에 $-1$을 곱하여 화자 분류 성능을 최소화(최대 손실화)함으로써 화자 정보를 제거한다.

**3. Phoneme Context (음소 문맥)**
단어의 기본 단위인 음소 정보를 학습하기 위해 음소 분류기를 추가하고 AAM loss를 적용하였다.

**4. Hybrid Loss (최종 손실 함수)**
최종 학습 목표는 다음과 같이 세 가지 손실의 조합으로 정의된다.
$$L(x,y) = L(x,y_w) - \eta L_{aam}(x,y_s) + \mu L_{aam}(x,y_p)$$
여기서 $L(x,y_w)$는 단어 손실(AAM 또는 SoftTriplet), $L_{aam}(x,y_s)$는 화자 손실, $L_{aam}(x,y_p)$는 음소 손실이며, $\eta$와 $\mu$는 가중치 계수이다.

## 📊 Results

### 실험 설정
- **데이터셋**: 학습에는 LibriSpeech를 사용하였고, 평가는 629명의 화자가 포함된 내부 데이터셋을 사용하였다. 평가 데이터의 키워드는 학습 세트에 포함되지 않은 단어들로 구성하여 실제 QbyE 환경을 모사하였다.
- **측정 지표**: 0.3 FAs/Hr (시간당 오인식 횟수) 기준에서의 FRR (False Reject Rate, 오거부율)을 측정하였다.

### 주요 결과
- **손실 함수 영향**: 단일 손실(CE)보다 AAM이 우수하며, 하이브리드 손실(단어+화자+음소)을 적용했을 때 성능이 가장 크게 향상되었다. 특히 Conformer 모델의 경우 하이브리드 손실 적용 시 FRR이 40.9% 감소하였다.
- **풀링 전략 비교**: ASP보다 GAP를 사용했을 때 모든 인코더에서 성능이 크게 향상되었다. Conformer의 경우 FRR이 51.1% 감소하는 효과를 보였다.
- **최적 조합**: **Hybrid Loss + SoftTriplet + GAP** 조합이 모든 모델에서 최상의 성능을 기록하였다.

### 모델 효율성 및 성능 비교
- **성능**: 최적 설정에서 Conformer가 $1.63\%$의 FRR로 가장 우수하였으나, **LiCoNet이 $1.98\%$의 FRR로 매우 근접한 성능**을 보였다.
- **효율성**: Table 2에 따르면, LiCoNet은 Conformer 대비 파라미터 수는 약 $1/2$ 수준이며, 연산량(FLOPs)은 약 $13\text{x}$ 더 효율적이다 ($46.5\text{M}$ vs $642.2\text{M}$).

## 🧠 Insights & Discussion

본 연구의 가장 중요한 통찰은 **정교한 풀링 전략(GAP)과 다중 작업 학습(MTL) 프레임워크가 모델 아키텍처의 복잡성을 상당 부분 보완할 수 있다**는 점이다.

- **LiCoNet의 재발견**: 단순한 선형 연산 기반의 LiCoNet이 복잡한 어텐션 기반의 Conformer와 대등한 성능을 낸다는 점은, 모델의 크기를 키우는 것보다 데이터의 특성을 잘 반영하는 풀링과 손실 함수를 설계하는 것이 효율성 측면에서 훨씬 유리함을 시사한다.
- **화자 독립성 확보**: GRL을 이용한 역화자 손실 학습이 실제 QbyE 성능 향상에 기여했음을 통해, 화자 정보와 언어 정보의 분리가 임베딩의 일반화 성능에 핵심적임을 확인하였다.
- **한계 및 가정**: 본 연구는 내부 데이터셋을 사용하여 평가하였으므로, 다른 도메인의 데이터셋에서도 동일한 수준의 효율성-성능 트레이드오프가 나타날지는 추가 검증이 필요하다. 또한, 3개의 등록 샘플을 사용한 가정이 실제 사용자 환경과 얼마나 일치하는지에 대한 논의가 부족하다.

## 📌 TL;DR

본 논문은 사용자 맞춤형 키워드 스포팅(QbyE KWS)을 위해 **Spectral-Temporal Graph Attentive Pooling (GAP)**과 **Multi-Task Learning (단어, 음소 학습 및 화자 정보 제거)**을 결합한 프레임워크를 제안하였다. 실험 결과, 제안된 프레임워크는 모델의 변별력을 극대화하여, 매우 가벼운 **LiCoNet 모델이 연산 효율성을 13배 높이면서도 고성능 모델인 Conformer와 대등한 인식 성능**을 낼 수 있게 하였다. 이는 저전력/저사양 하드웨어 기반의 실시간 맞춤형 KWS 시스템 구현에 매우 중요한 기여를 할 것으로 기대된다.