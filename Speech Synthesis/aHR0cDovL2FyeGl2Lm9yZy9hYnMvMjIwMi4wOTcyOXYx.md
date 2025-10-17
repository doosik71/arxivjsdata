# It’s Raw! Audio Generation with State-Space Models

Karan Goel, Albert Gu, Chris Donahue, and Christopher R ́e

## 🧩 Problem to Solve

원시 오디오 파형은 초당 수만 개의 타임스텝과 다중 시간 규모에 걸친 장거리 의존성을 가지는 고차원 데이터이기 때문에, 이를 효과적으로 모델링하는 아키텍처를 개발하는 것은 매우 어려운 문제입니다. 기존의 RNN이나 CNN 기반 접근 방식은 오디오의 요구사항에 맞춰 조정되었지만, 계산 효율성 측면에서 바람직하지 않은 절충점을 가지거나 파형을 효과적으로 모델링하는 데 어려움을 겪습니다. 특히 다음과 같은 특성을 갖는 아키텍처가 필요합니다:

1. **전역적으로 일관된 생성 (Globally coherent generation)**: 무한한 컨텍스트와 장거리 의존성을 모델링해야 합니다.
2. **계산 효율성 (Computational efficiency)**: 병렬 훈련, 그리고 빠른 자기회귀(AR) 및 비자기회귀(non-AR) 추론이 가능해야 합니다.
3. **샘플 효율성 (Sample efficiency)**: 고속 파형 데이터에 적합한 귀납적 편향(inductive biases)을 가진 모델이 필요합니다.

## ✨ Key Contributions

- **S4 파라미터화 안정화**: 자기회귀 생성 시 S4 모델의 수치적 불안정성 문제를 해결하기 위해, Hurwitz 행렬과의 연관성을 통해 파라미터화($\Lambda - pp^*$ 대신 $\Lambda + pq^*$)를 개선하여 안정성을 이론적으로 보장했습니다.
- **SaShiMi 아키텍처 도입**: S4를 기반으로 한 새로운 다중 스케일(multi-scale) 아키텍처인 SaShiMi를 제안했습니다. 이 아키텍처는 무조건부 자기회귀 음악 및 음성 파형 모델링에서 최첨단 효율성과 성능을 달성합니다.
- **다용도성 입증**: SaShiMi가 기존의 다른 심층 생성 모델(예: DiffWave)에 쉽게 통합되어 성능을 향상시킬 수 있음을 보여주었습니다.
- **우수한 성능 및 효율성**: WaveNet 대비 무조건부 음성 생성 작업에서 2배 더 나은 MOS(Mean Opinion Scores)를 기록하는 등, AR 설정에서 인간이 더 음악적이고 일관적이라고 평가하는 피아노 및 음성 파형을 생성합니다. 또한, 더 적은 파라미터로도 밀도 추정 및 훈련/추론 속도에서 WaveNet을 능가합니다.

## 📎 Related Works

이 연구는 조건 정보를 사용하지 않는 원시 오디오 파형 생성에 초점을 맞추며, 주로 다음 범주의 이전 연구들과 비교하거나 이들을 개선합니다.

- **자기회귀(AR) 접근 방식**: 오디오 샘플을 이전 샘플에 기반하여 하나씩 생성하는 방식으로, WaveNet [39], SampleRNN [28], Transformers [5] 등이 있습니다. WaveNet은 TTS, 무조건부 생성, non-AR 생성 등 다양한 오디오 생성 시스템의 핵심 구성 요소로 사용되어 왔지만, 수용 필드(receptive field)의 한계로 장기 구조 모델링에 제약이 있습니다.
- **비자기회귀(non-AR) 접근 방식**: 전체 파형을 한 번에 생성하여 효율성을 높이는 방식으로, WaveGAN [10], DiffWave [23] 등이 있습니다. DiffWave는 현재 SC09 데이터셋에서 무조건부 파형 생성의 최신 기술(state-of-the-art)로 알려져 있으며, WaveNet을 백본으로 사용합니다.
- **상태 공간 모델 (SSM)**: S4 [13]와 같은 최근 개발된 심층 상태 공간 모델은 CNN과 RNN의 특성을 모두 가지며 장거리 의존성 모델링에 강점을 보입니다.

## 🛠️ Methodology

SaShiMi는 S4 레이어를 핵심 구성 요소로 사용하며, 다음 두 가지 주요 개선 사항을 포함합니다.

1. **S4 재귀 안정화 (Stabilizing S4 for Recurrence)**:

   - 기존 S4 모델의 파라미터화인 $\Lambda + pq^*$ 대신 $\Lambda - pp^*$를 사용합니다. 이는 $p$와 $q$ 파라미터를 연결하고 부호를 반전하는 것과 유사합니다.
   - $\Lambda - pp^*$ 형태는 여전히 S4의 주요 속성(DPLR 행렬, HiPPO 행렬 초기화 가능)을 만족하면서도, 상태 행렬 $A$의 스펙트럼(spectrum)을 제어하기 쉽게 만듭니다.
   - **Hurwitz 행렬**: $A$ 행렬이 모든 고유값의 실수부가 음수인 Hurwitz 행렬일 때 SSM($$h'(t) = Ah(t) + Bx(t)$$)이 점근적으로 안정적(asymptotically stable)임을 활용합니다. 이 안정성은 이산 시간 SSM의 RNN 모드($$h_k = Ah_{k-1} + Bx_k$$)에서 $A$의 거듭제곱이 반복될 때 중요합니다.
   - **안정성 보장**: **Proposition 4.3**은 $A = Λ - pp^*$ 형태의 행렬은 $\Lambda$의 모든 엔트리가 음의 실수부를 가질 경우 Hurwitz 행렬임을 증명합니다. 이는 $\Lambda + \Lambda^*$와 $-pp^\*$가 모두 음의 준정부(Negative Semidefinite, NSD) 행렬이기 때문입니다. 이 재파라미터화를 통해 $A$의 스펙트럼을 $\Lambda$의 대각선 부분만 제어함으로써 훨씬 쉽게 안정성을 확보할 수 있습니다.

2. **SaShiMi 아키텍처 (SaShiMi Architecture)**:
   - **S4 블록 (S4 Block)**: S4 레이어와 표준 포인트와이즈 선형 함수, 비선형성(GELU), 잔여 연결(residual connection)로 구성됩니다. S4 레이어 후에 피드포워드 네트워크(Transformer) 또는 역병목 레이어(CNN) 스타일의 추가 포인트와이즈 선형 레이어를 포함합니다.
   - **다중 스케일 아키텍처 (Multi-scale Architecture)**: 여러 해상도에서 원시 입력 신호의 정보를 통합합니다. 최상위 계층은 원본 샘플링 속도로 오디오 파형을 처리하고, 하위 계층은 다운샘플링된 신호 버전을 처리합니다. 하위 계층의 출력은 업샘플링되어 상위 계층의 입력과 결합되어 강력한 조건 신호를 제공합니다. 이는 SampleRNN 및 PixelCNN++와 같은 다중 스케일 AR 모델에서 영감을 받았습니다.
     - **풀링 (Pooling)**: 간단한 재형성(reshape) 및 선형 연산을 통해 구현됩니다. (Down-pool: $(T, H) \xrightarrow{\text{reshape}} (T/p, p \cdot H) \xrightarrow{\text{linear}} (T/p, q \cdot H)$, Up-pool: $(T, H) \xrightarrow{\text{linear}} (T, p \cdot H/q) \xrightarrow{\text{reshape}} (T \cdot p, H/q)$). $p$는 풀링 인자, $q$는 확장 인자입니다. AR 설정에서는 인과성(causality)을 위해 업풀링 레이어를 시간 스텝만큼 이동해야 합니다.
   - **양방향 S4 (Bidirectional S4)**: non-AR 작업을 위해 S4의 간단한 양방향 변형을 제공합니다. 입력 시퀀스를 S4 레이어에 통과시키고, 역방향으로도 S4 레이어에 통과시켜 그 출력을 연결하고 위치별 선형 레이어를 통해 처리합니다:
     $$y = \text{Linear}(\text{Concat}(\text{S4}(x), \text{rev}(\text{S4}(\text{rev}(x)))))$$

## 📊 Results

- **무조건부 음악 생성 (Unbounded Music Generation)**:
  - **Beethoven 데이터셋**: WaveNet(4092)의 4K 대비 128K 길이의 컨텍스트를 학습할 수 있으며, 기존 AR 모델보다 NLL(Negative Log-Likelihood)에서 0.09 BPB 더 우수한 성능($$0.946$$ 대 $$1.032$$)을 달성했습니다. 더 긴 컨텍스트를 효과적으로 활용하는 능력을 입증했습니다.
  - **YouTubeMix 데이터셋**: SampleRNN 및 WaveNet보다 NLL($$1.294$$)이 훨씬 우수합니다. MOS(Mean Opinion Scores) 평가에서 오디오 충실도(fidelity)는 유사하지만, 음악성(musicality)에서 약 0.40점 향상($$3.11$$ 대 $$2.71$$)을 보여, 더 긴 샘플을 일관성 있게 생성하는 능력을 입증했습니다.
  - **효율성**: 벽시계 시간(wall clock time)에서 baseline보다 더 안정적이고 효율적으로 훈련됩니다. 동일한 품질 수준에서 WaveNet 대비 3배 이상 적은 파라미터를 사용하면서도 훈련 및 추론 속도에서 우위를 보였습니다.
- **모델 개선 효과 (Model Ablations)**:
  - **S4 안정화**: $\Lambda - pp^*$ 파라미터화는 $\Lambda + pq^*$ 대비 안정적인 생성을 가능하게 하면서도 유사한 NLL 성능($$1.4193$$ 대 $$1.4207$$)을 유지합니다. 훈련된 $A$ 행렬의 스펙트럼 반경(spectral radii) 분석을 통해 안정성이 개선되었음을 시각적으로 확인했습니다.
  - **다중 스케일 아키텍처**: 풀링 레이어를 추가한 SaShiMi 아키텍처는 Isotropic S4 모델 대비 계산 및 모델링 성능에서 상당한 개선을 가져옵니다.
- **무조건부 음성 생성 (Unconditional Speech Generation, SC09)**:
  - **자기회귀 (AR) 설정**: 기존 AR 파형 모델(SampleRNN, WaveNet)보다 모든 평가 지표에서 월등한 성능을 보였으며, 품질 및 이해도(intelligibility) MOS에서 2배 높은 점수($$3.29$$ 및 $$3.53$$)를 달성했습니다. WaveGAN보다 4배 적은 파라미터로 더 높은 MOS를 달성했습니다.
  - **비자기회귀 (non-AR) 설정**: 최신 Diffusion 모델인 DiffWave의 백본 아키텍처를 WaveNet에서 SaShiMi로 교체했을 때, 모델 튜닝 없이도 모든 정량적 및 정성적 지표에서 최신 기술(state-of-the-art) 성능을 갱신했습니다 (FID $$1.42$$, IS $$69.17$$).
  - **효율성 및 견고성**: SaShiMi는 WaveNet 백본보다 훨씬 샘플 효율적이며, 절반의 훈련 스텝으로도 최상의 DiffWave 모델과 동등한 성능을 달성합니다. 또한, WaveNet 백본이 소규모 모델에서 학습에 실패하는 등 아키텍처 파라미터에 매우 민감한 반면, SaShiMi는 훨씬 견고함을 보여주었습니다.
  - **양방향 S4의 효과**: non-AR DiffWave 모델에서 양방향 S4는 단방향 S4보다 훨씬 우수한 성능을 보였습니다.

## 🧠 Insights & Discussion

SaShiMi는 원시 오디오 파형 모델링을 위한 유망한 새로운 아키텍처입니다. 인간 평가에서 SaShiMi가 생성한 오디오는 이전 아키텍처의 파형보다 각각 더 음악적이고 이해하기 쉽다는 평가를 받았으며, 이는 SaShiMi가 더 높은 수준의 전역적 일관성(global coherence)을 가진 오디오를 생성한다는 것을 시사합니다. S4의 이중 컨볼루션 및 재귀 형태를 활용함으로써, SaShiMi는 훈련과 추론 모두에서 이전 아키텍처보다 계산적으로 효율적입니다. 또한, SaShiMi는 지속적으로 더 샘플 효율적으로 훈련되며, 더 적은 훈련 스텝으로 더 나은 정량적 성능을 달성합니다. 마지막으로, WaveNet을 대체하는 드롭인(drop-in) 방식으로 사용될 때, SaShiMi는 기존의 최첨단 무조건부 생성 모델의 성능을 향상시켰으며, 이는 SaShiMi가 오디오 생성 전반을 개선하는 파급 효과를 가져올 잠재력이 있음을 나타냅니다.

## 📌 TL;DR

고차원 원시 오디오 파형의 효율적이고 일관된 생성을 목표로, 이 논문은 S4 상태 공간 모델을 기반으로 한 SaShiMi 아키텍처를 제안합니다. SaShiMi는 자기회귀 생성 시 S4의 불안정성을 Hurwitz 행렬 개념을 통해 $\Lambda - pp^*$ 파라미터화로 개선하고, 다중 스케일(multi-scale) 구조를 채택하여 장거리 컨텍스트를 효과적으로 모델링합니다. 실험 결과, SaShiMi는 WaveNet 대비 음악 및 음성 생성에서 인간 평가(MOS) 및 정량적 지표(NLL) 모두에서 최첨단 성능을 달성하며, 훈련/추론 효율성과 샘플 효율성에서도 우위를 보였습니다. 특히, DiffWave와 같은 기존 non-AR 모델의 백본으로 사용될 때도 성능을 크게 향상시켜 SaShiMi의 다용도성을 입증했습니다.
