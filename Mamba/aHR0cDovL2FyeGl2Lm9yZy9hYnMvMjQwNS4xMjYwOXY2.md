# Mamba in Speech: Towards an Alternative to Self-Attention

Xiangyu Zhang, Qiquan Zhang, Hexin Liu, Tianyi Xiao, Xinyuan Qian, Beena Ahmed, Eliathamby Ambikairajah, Haizhou Li, Julien Epps (2025)

## 🧩 Problem to Solve

본 논문은 음성 처리(Speech Processing) 분야에서 Transformer의 핵심 모듈인 Multi-Head Self-Attention(MHSA)이 가지는 계산 복잡도 문제를 해결하고자 한다. MHSA는 컨텍스트 윈도우의 크기가 증가함에 따라 계산량이 제곱(Quadratic)으로 증가하는 특성이 있으며, 특히 프레임 레벨의 음향 특징 시퀀스를 처리해야 하는 음성 작업에서는 매우 큰 계산 부담을 야기한다.

최근 NLP 및 CV 분야에서는 이를 대체하기 위해 선형 시간 복잡도를 가진 Selective State Space Model(이하 Mamba)이 제안되었으나, 음성 처리 분야에서의 적용 가능성과 효율성에 대한 연구는 부족한 상태이다. 특히, 단순히 저수준의 신호 처리(예: 음성 향상)뿐만 아니라 고수준의 시맨틱 정보가 필요한 작업(예: 음성 인식)에서 Mamba가 MHSA의 실질적인 대안이 될 수 있는지를 규명하는 것이 본 연구의 목표이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Mamba를 음성 처리 작업에 최적화하기 위해 **양방향 Mamba(Bidirectional Mamba, BiMamba)** 구조를 제안하고, 이를 다양한 추상화 수준의 음성 작업에 적용하여 그 효과를 검증한 것이다.

주요 설계 아이디어는 다음과 같다:
1. **양방향성 도입**: 단방향으로 작동하는 기존 Mamba의 한계를 극복하기 위해, 입력 시퀀스를 정방향과 역방향으로 동시에 처리하는 BiMamba 구조를 탐색하였다.
2. **ExtBiMamba 제안**: 내부적으로 방향을 나누는 InnBiMamba보다, 정방향과 역방향의 입력/출력 투영(Projection) 층을 완전히 분리한 **External Bidirectional Mamba(ExtBiMamba)**가 음성 신호의 보완적인 특징 공간을 학습하는 데 더 효율적임을 제시하였다.
3. **비선형성(Nonlinearity)의 중요성 규명**: Mamba는 본질적으로 선형 연산 위주로 구성되어 있어 고수준 시맨틱 정보를 학습하는 능력이 부족하다. 이를 해결하기 위해 Mamba를 독립적으로 사용하기보다, Feed-Forward Network(FFN)나 Convolution 모듈이 포함된 Transformer 및 Conformer 구조 내에서 MHSA를 대체하는 방식이 최적임을 밝혀냈다.

## 📎 Related Works

기존의 State Space Model(SSM) 연구들은 S4와 같은 구조를 통해 긴 시퀀스를 효율적으로 모델링 하였으나, 입력 내용에 따라 파라미터가 변하지 않는 LTI(Linear Time-Invariant) 시스템의 한계로 인해 컨텍스트 정보를 충분히 캡처하지 못했다. 이를 개선한 Mamba는 선택적 메커니즘(Selective Mechanism)을 통해 입력에 따라 파라미터를 동적으로 조정함으로써 NLP와 CV에서 뛰어난 성능을 보였다.

음성 처리 분야에서도 Mamba를 적용하려는 시도가 있었으나, 주로 음성 분리(Speech Separation)나 향상(Enhancement)과 같은 저수준 작업에 집중되었다. 또한, 기존 연구들은 Mamba를 사용하면서도 여전히 MHSA와 결합하거나 Dual-path 전략을 사용하여 Mamba의 최대 장점인 낮은 시간 복잡도의 이점을 완전히 누리지 못하는 경우가 많았다. 본 논문은 이러한 한계를 넘어, Mamba를 MHSA의 완전한 대체제로 사용하며 그 가능성을 분석했다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. State Space Models and Mamba
SSM은 연속적인 선형 시계 불변(LTI) 시스템에서 영감을 얻었으며, 입력 시퀀스 $x(t)$를 상태 변수 $h(t)$를 통해 출력 $y(t)$로 매핑한다.

$$h'(t) = Ah(t) + Bx(t)$$
$$y(t) = Ch(t) + Dx(t)$$

여기서 $A, B, C, D$는 SSM 파라미터이다. 이를 딥러닝 아키텍처에 적용하기 위해 Zero-Order Hold(ZOH) 방식을 통해 이산화(Discretization) 과정을 거치며, 이산화된 파라미터 $\bar{A}, \bar{B}$는 다음과 같이 계산된다.

$$\bar{A} = \exp(\Delta A)$$
$$\bar{B} = (\Delta A)^{-1}(\exp(\Delta A) - I)\Delta B \approx \Delta B$$

Mamba는 여기서 $\Delta, B, C$를 입력 $X$의 함수로 만들어 입력에 따라 적응적으로 파라미터를 조정하는 **Selective SSM**을 구현하였다.

### 2. Bidirectional Mamba (BiMamba)
음성 신호는 전체 시퀀스를 한 번에 처리하는 비인과적(Non-causal) 특성이 중요하므로, 본 논문은 두 가지 양방향 구조를 제안한다.

- **InnBiMamba (Inner BiMamba)**: 동일한 입력/출력 투영 층을 공유하며, 한 쪽 SSM에는 정방향 데이터를, 다른 쪽 SSM에는 역방향 데이터를 입력한 뒤 결과를 결합한다.
- **ExtBiMamba (External BiMamba)**: 정방향과 역방향 경로 각각에 독립적인 입력/출력 투영 층을 배치한다. 이를 통해 각 방향에 최적화된 특징 변환을 학습할 수 있으며, 보다 보완적인 특징 공간을 형성한다.

### 3. Task-aware Model Designs
정보의 추상화 수준에 따라 세 가지 적용 전략을 탐색하였다.
1. **Independent Mamba/BiMamba**: Mamba 층을 단순히 쌓아 올려 독립적인 모델로 구성한다.
2. **TransMamba/TransBiMamba**: Transformer 구조에서 MHSA 모듈만 Mamba/BiMamba로 교체한다. FFN과 Layer Norm이 비선형성을 제공한다.
3. **ConMamba/ConBiMamba**: Conformer 구조에서 MHSA를 Mamba/BiMamba로 교체한다. Conformer의 Convolution 모듈이 추가적인 비선형성을 제공하여 더 고수준의 정보를 학습하게 한다.

## 📊 Results

### 1. 음성 향상 (Speech Enhancement)
- **데이터셋 및 지표**: LibriSpeech-clean-100 및 다양한 소음 데이터를 사용하였으며, PESQ, ESTOI, COVL 등의 지표로 평가하였다.
- **결과**: ExtBiMamba가 InnBiMamba보다 성능, 학습 속도, 추론 속도 및 계산 복잡도(MACs) 면에서 모두 우수하였다. 또한, 유사한 파라미터 규모에서 Mamba 기반 모델이 Transformer보다 높은 성능을 보였으며, 특히 입력 시퀀스 길이가 길어질수록 RTF(Real-Time Factor)와 MACs 측면에서 Transformer 대비 압도적인 효율성을 증명하였다.

### 2. 음성 인식 (Speech Recognition)
- **데이터셋 및 지표**: LibriSpeech, AN4, SEAME, ASRU 데이터셋을 사용하였으며, Word Error Rate(WER)와 Mixed WER(MER)로 평가하였다.
- **결과**:
    - **독립 모델의 한계**: 독립적인 Mamba/BiMamba 모델은 Transformer/Conformer에 비해 성능이 현저히 낮았으며, 레이어 수를 늘려 파라미터 수를 맞춰도 성능 개선이 미미했다.
    - **MHSA 대체 효과**: Conformer의 MHSA를 ExtBiMamba로 교체한 **ConExtBiMamba**는 여러 데이터셋에서 SOTA인 Branchformer보다 더 높은 성능을 기록하였다.
    - **강건성**: 매우 작은 데이터셋인 AN4에서 ConExtBiMamba는 Conformer보다 낮은 WER과 훨씬 작은 분산을 보여, 과적합에 강하고 학습 안정성이 높음을 확인하였다.

## 🧠 Insights & Discussion

### 1. 비선형성 가설 (Nonlinearity Hypothesis)
본 논문의 가장 중요한 통찰은 **"Mamba는 본질적으로 비선형성이 부족하다"**는 점이다.
- 음성 향상과 같은 저수준 작업은 spectral 정보 위주이므로 독립적인 Mamba만으로도 충분한 성능이 나온다.
- 그러나 음성 인식과 같은 고수준 시맨틱 작업은 복잡한 결정 경계를 학습해야 하므로 더 강력한 비선형성이 필요하다.
- 이를 증명하기 위해 합성 데이터셋을 이용한 시각화 실험을 수행하였으며, BiMamba 단독으로는 복잡한 데이터의 결정 경계를 찾지 못하지만, FFN을 추가하면 성공적으로 학습함을 보였다.

### 2. 아키텍처적 결론
결과적으로 Mamba를 음성 인식에 적용하려면, Mamba 자체의 성능에 의존하기보다 **Conformer와 같이 비선형성을 보완해 줄 수 있는 구조(Convolution, FFN 등) 내부에 통합**하는 것이 가장 적절한 전략이다. 또한, Transformer보다 Conformer에서 Mamba의 효과가 더 컸던 이유는 Conformer가 더 많은 비선형 활성화 함수를 포함하고 있기 때문으로 분석된다.

## 📌 TL;DR

본 논문은 계산 복잡도가 높은 MHSA의 대안으로 Mamba를 음성 처리 분야에 도입하고, 특히 정/역방향 투영 층이 분리된 **ExtBiMamba** 구조가 가장 효율적임을 입증하였다. 실험 결과, 저수준 작업(음성 향상)에서는 Mamba 단독으로도 우수하지만, 고수준 작업(음성 인식)에서는 **비선형성 부족**으로 인해 독립 모델의 성능이 낮았다. 그러나 Mamba를 Conformer의 MHSA 대체제로 사용했을 때(ConExtBiMamba)는 SOTA 수준의 성능과 매우 낮은 계산 복잡도를 동시에 달성하였다. 이는 향후 실시간 음성 인식 시스템 및 긴 컨텍스트를 처리해야 하는 음성 AI 모델 설계에 중요한 지침이 될 것이다.