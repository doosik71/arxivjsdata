# Fake-Mamba: Real-Time Speech Deepfake Detection Using Bidirectional Mamba as Self-Attention’s Alternative

Xi Xuan, Zimo Zhu, Wenxin Zhang, Yi-Cheng Lin, Tomi Kinnunen (2025)

## 🧩 Problem to Solve

본 논문은 텍스트 음성 합성(TTS) 및 음성 변환(VC) 기술의 발전으로 인해 고도화된 음성 딥페이크(Speech Deepfake)가 생성됨에 따라 발생하는 보안 위협을 해결하고자 한다. 특히 음성 생체 인식 시스템의 스푸핑(spoofing), 금융 사기, 정치적 갈등 유발 등의 위험을 방지하기 위한 실시간 음성 딥페이크 탐지(Speech Deepfake Detection, SDD) 기술의 필요성이 증대되고 있다.

기존의 SOTA(State-of-the-art) 모델인 Conformer 기반 접근 방식은 다음과 같은 두 가지 주요 한계를 가진다. 첫째, Multi-Head Self-Attention(MHSA)의 시간 복잡도가 시퀀스 길이 $t$에 대해 $O(t^2)$로 증가하여, 메모리가 제한된 엣지 디바이스(edge devices)에서 사용하기 어렵다. 둘째, MHSA는 시간 차원(temporal dimension)의 토큰 간 내적에 의존하므로, 시간과 채널 차원(temporal and channel dimensions) 간의 상호 의존성을 간과할 수 있으며, 이는 오디오 압축이나 코덱으로 인한 변형이 발생한 환경에서 강건성(robustness)과 일반화 성능을 저하시킨다.

따라서 본 연구의 목표는 MHSA를 대체하여 계산 효율성을 높이면서도, 시간-채널 차원의 특징을 효과적으로 포착하여 실시간 탐지가 가능한 고성능 SDD 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Transformer나 Conformer의 핵심 구성 요소인 MHSA를 **Bidirectional Mamba(양방향 Mamba)**로 대체하는 것이다. Mamba는 State Space Model(SSM)을 기반으로 하여 선형 시간 복잡도를 가지면서도 전역 수용장(global receptive field)을 확보할 수 있는 모델이다.

구체적인 기여 사항은 다음과 같다.

1. **Fake-Mamba 프레임워크 제안**: XLSR(Cross-lingual Speech Representation) 프론트엔드와 양방향 Mamba 백본을 결합하여 로컬 및 글로벌 아티팩트(artifact)를 동시에 포착하는 구조를 설계하였다.
2. **세 가지 BiMamba 인코더 변형 제안**: TransBiMamba, ConBiMamba, 그리고 성능이 가장 우수한 **PN-BiMamba**를 제안하였다. 특히 PN-BiMamba는 Pre-LayerNorm 안정화와 양방향 특징 융합을 통해 미세한 합성 흔적을 정밀하게 탐지한다.
3. **실시간성 및 성능 입증**: ASVspoof 2021 및 In-the-Wild 벤치마크에서 기존 SOTA 모델(XLSR-Conformer 등) 대비 낮은 EER(Equal Error Rate)을 달성하였으며, 추론 속도(RTF) 측면에서도 우월함을 보였다.

## 📎 Related Works

### Mamba for Audio and Speech Processing

최근 Mamba는 언어 모델링, 컴퓨터 비전, 시계열 모델링 등 다양한 분야에서 Transformer 수준의 성능을 보이면서도 효율적인 추론이 가능함을 입증하였다. 오디오 및 음성 처리 분야에서도 음성 인식, 음성 향상, 화자 분리 및 검증 등의 작업에 Mamba가 적용되기 시작하였다. 하지만 실시간 SDD를 위한 효율적인 Mamba 아키텍처 설계에 대한 연구는 여전히 부족한 상태이다.

### Mamba for Speech Deepfake Detection

기존에 제안된 RawBMamba나 XLSR-Mamba와 같은 모델들이 양방향 Mamba를 SDD에 도입하였으나, 이들은 주로 단순한 Mamba 블록의 적층(stacked pure Mamba blocks) 구조를 사용하였다. 이러한 설계는 양방향 아키텍처 내에서 깊은 차원 간 상호작용(cross-dimensional interaction)을 제한하며, 결과적으로 딥페이크 탐지에 필수적인 시간-채널 차원의 아티팩트 단서를 포착하는 데 한계가 있다. Fake-Mamba는 이를 해결하기 위해 특징 변환 및 상호작용 메커니즘을 개선한 세 가지 변형 구조를 제안하여 차별점을 둔다.

## 🛠️ Methodology

### 전체 파이프라인

Fake-Mamba의 전체 처리 과정은 다음과 같은 4단계 파이프라인으로 구성된다.

1. **Frame-Level Feature Extraction**: 입력 오디오에서 XLSR 모델을 통해 프레임 수준의 특징을 추출한다.
2. **Backbone**: 제안된 BiMamba 변형 블록(특히 PN-BiMamba)을 통해 시공간적 특징을 학습한다.
3. **Utterance-level Pooling**: Linear Attention Pooling을 사용하여 발화 수준의 임베딩을 생성한다.
4. **Classification Head**: MLP(Multi-Layer Perceptron)를 통해 해당 음성이 진짜(real)인지 가짜(fake)인지 분류한다.

### 주요 구성 요소 상세 설명

#### 1. XLSR 프론트엔드

128개 언어, 약 43.6만 시간의 무라벨 음성 데이터로 사전 학습된 XLSR을 사용한다. XLSR은 실제 인간 음성의 핵심 특성을 잘 포착하며, SDD 작업에서 다른 파운데이션 모델보다 우수한 성능을 보인다고 보고되었다. 추출된 특징 $S_f \in \mathbb{R}^{T \times C}$는 선형 투영 층을 통해 차원이 $D$로 축소된 $S'_f \in \mathbb{R}^{T \times D}$가 되어 백본으로 전달된다.

#### 2. PN-BiMamba (Proposed Backbone)

PN-BiMamba는 특징 변환과 상호작용을 극대화한 구조로, $i$번째 블록의 입력 $h_{i-1}$에 대해 다음과 같은 절차로 출력 $h_i$를 계산한다.

- **정규화 및 투영**: 우선 LayerNorm을 적용한 후, 입력을 두 개의 변수 $x, z \in \mathbb{R}^{T \times E}$로 투영한다 (여기서 $E$는 확장된 상태 차원).
  $$x = \text{Linear}_x(\text{LayerNorm}(h_{i-1})), \quad z = \text{Linear}_z(\text{LayerNorm}(h_{i-1}))$$
- **순방향 경로**: $x$는 1D Convolution과 SiLU 활성화 함수를 거쳐 SSM 모듈을 통과하고, gating 변수인 $\text{SiLU}(z)$와 Hadamard product($\otimes$) 연산을 수행한다.
  $$x' = \text{SiLU}(\text{Conv1d}(x)), \quad y = \text{SSM}(x') \otimes \text{SiLU}(z)$$
  최종적으로 선형 층을 통해 순방향 특징 $h_{\text{forward}}$를 얻는다.
- **역방향 경로**: 입력을 뒤집은(Flip) 후 Mamba 블록을 통과시키고 다시 뒤집어 역방향 특징 $h_{\text{backward}}$를 얻는다.
  $$h_{\text{backward}} = \text{Flip}(\text{Mamba}(\text{LayerNorm}(\text{Flip}(h_{i-1}))))$$
- **특징 융합 및 후처리**: 양방향 특징을 합산하고, 잔차 연결(Residual Connection)과 LayerNorm을 적용한 뒤 FFN(Feed-Forward Network)을 거쳐 최종 출력 $h_i$를 산출한다.

#### 3. 풀링 및 분류

인코더의 출력은 Linear Attention Pooling을 통해 발화 수준의 임베딩 $S_u \in \mathbb{R}^D$로 압축되며, MLP 헤드가 최종적으로 Real/Fake 로짓을 예측한다.

## 📊 Results

### 실험 설정

- **데이터셋**: ASVspoof 2019 LA(학습), ASVspoof 2021 LA/DF 및 In-the-Wild(평가) 데이터셋을 사용하였다.
- **지표**: EER(Equal Error Rate)과 min-tDCF를 사용하여 성능을 측정하였으며, 추론 속도는 RTF(Real-Time Factor)로 평가하였다.
- **비교 대상**: RawBMamba, XLSR-Conformer, XLSR-DuaBiMamba 등과 비교하였다.

### 주요 결과

1. **정량적 성능**: PN-BiMamba 기반의 Fake-Mamba(L) 모델은 세 가지 벤치마크에서 각각 **0.97% (21LA), 1.74% (21DF), 5.85% (ITW)**의 EER을 기록하였다. 특히 ITW 데이터셋에서는 기존 SOTA인 XLSR-DuaBiMamba보다 12.82%의 상대적 성능 향상을 보였다.
2. **구조적 비교**: TransBiMamba, ConBiMamba, PN-BiMamba 중 PN-BiMamba가 모든 지표에서 가장 우수한 성능을 보였으며, 이는 병렬 SSM 경로와 Pre-LayerNorm 배치가 시간-채널 정보 융합에 효과적임을 시사한다.
3. **추론 속도**: RTF 측정 결과, Fake-Mamba(L)는 모든 음성 길이에 대해 XLSR-Conformer보다 낮은 RTF를 기록하여, 실시간 처리 능력이 훨씬 뛰어남을 입증하였다.
4. **시각화 분석**: t-SNE를 통해 발화 수준 임베딩 $S_u$를 시각화한 결과, XLSR-Conformer는 Real과 Fake 클래스가 많이 겹치는 반면, Fake-Mamba(L)는 두 클래스가 명확하게 분리되는 양상을 보였다.

## 🧠 Insights & Discussion

### 강점 및 분석

- **효율성과 성능의 트레이드오프 해결**: Mamba의 선형 복잡도를 활용하여 추론 속도를 획기적으로 높이면서도, 정교한 BiMamba 변형 구조를 통해 Transformer/Conformer 수준 혹은 그 이상의 탐지 성능을 확보하였다.
- **강건한 일반화 능력**: 특히 제어되지 않은 환경에서 수집된 In-the-Wild 데이터셋에서 큰 성능 향상을 보인 점은, Mamba의 전역 수용장과 제안된 PN-BiMamba의 특징 융합 방식이 실제 환경의 다양한 아티팩트를 포착하는 데 유리함을 의미한다.
- **구성 요소의 중요성**: Ablation study를 통해 Pre-LN, FFN, 양방향 구조, 풀링 층 중 어느 하나라도 제거했을 때 성능이 크게 저하됨을 확인하였다. 특히 Pre-LN 제거 시 성능이 급격히 떨어지는 것은 학습 안정화에 핵심적인 역할을 함을 보여준다.

### 한계 및 논의

- 본 연구는 주로 공개 벤치마크 데이터셋에 의존하고 있다. 실제 서비스 환경에서 발생할 수 있는 더 다양한 형태의 노이즈나 최신 생성 모델에 대한 일반화 성능에 대한 추가 검증이 필요할 수 있다.
- 논문에서는 향후 연구로 Source Tracing(생성원 추적) 작업으로의 확장 가능성을 언급하고 있다.

## 📌 TL;DR

본 논문은 연산 비용이 높은 MHSA를 대체하여 **양방향 Mamba(Bidirectional Mamba)**를 도입한 실시간 음성 딥페이크 탐지 모델 **Fake-Mamba**를 제안한다. 특히 **PN-BiMamba** 구조를 통해 시간-채널 차원의 특징을 효과적으로 융합함으로써, ASVspoof 2021 및 In-the-Wild 데이터셋에서 SOTA 성능을 달성하였다. 또한, 선형 시간 복잡도를 통해 추론 속도를 크게 개선하여 콜센터, 화상 회의, 스트리밍 서비스 등 실시간 보안 적용 가능성이 매우 높은 연구이다.
