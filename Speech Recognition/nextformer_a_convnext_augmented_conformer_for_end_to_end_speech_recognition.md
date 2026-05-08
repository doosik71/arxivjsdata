# NEXTFORMER: A CONVNEXT AUGMENTED CONFORMER FOR END-TO-END SPEECH RECOGNITION

Yongjun Jiang, Jian Yu, Wenwen Yang, Bihong Zhang, Yanfeng Wang (Tencent PCG)

## 🧩 Problem to Solve

본 논문은 종단간 음성 인식(End-to-End Speech Recognition, E2E ASR) 시스템에서 널리 사용되는 Conformer 모델의 한계를 해결하고자 한다. Conformer는 CNN과 Self-Attention을 결합하여 전역적 문맥(Global Context)과 지역적 상관관계(Local Correlation)를 동시에 포착할 수 있어 매우 우수한 성능을 보여왔다. 그러나 Conformer는 주로 시간적 모델링(Temporal Modeling)에 집중하고 있으며, 음성 특징(Speech Feature)이 갖는 시간-주파수(Time-Frequency) 특성 활용에는 상대적으로 소홀하다는 문제점이 있다.

특히, Conformer 인코더의 서브샘플링(Subsampling) 모듈은 단순히 두 개의 $\text{Conv2d}$ 층만을 사용하여 시간-주파수 특징을 시간적 특징으로 변환하는데, 이는 복잡한 시간-주파수 도메인의 정보를 충분히 추출하기에는 모델 용량(Capacity)이 부족하다는 점이 지적된다. 따라서 본 연구의 목표는 ConvNeXt 구조를 도입하여 음성 특징의 시간-주파수 특성을 보다 정밀하게 추출하고, 모델의 효율성과 정확도를 동시에 향상시킨 Nextformer 구조를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 컴퓨터 비전 분야에서 성능이 검증된 ConvNeXt 구조를 Conformer의 전처리에 도입하여 시간-주파수 정보 추출 능력을 극대화하는 것이다. 구체적인 기여 사항은 다음과 같다.

1. **CNTF (ConvNeXt Time-Frequency) 모듈 제안**: 기존 Conformer의 단순한 서브샘플링 모듈을 ConvNeXt 블록의 스택으로 대체하여, 음성 스펙트로그램의 시간-주파수 특성을 보다 효과적으로 활용한다.
2. **추가 다운샘플링(Additional Downsampling) 모듈 도입**: ConvNeXt 도입으로 인해 증가한 계산 비용을 상쇄하고 모델의 정확도를 높이기 위해, Conformer 레이어 중간 지점에 경량화된 다운샘플링 모듈을 삽입한다.
3. **SOTA 성능 달성**: AISHELL-1 및 WenetSpeech 데이터셋에서 Non-streaming 및 Streaming 모드 모두에서 기존 Conformer 대비 낮은 CER(Character Error Rate)을 기록하며 최첨단 성능(SOTA)을 달성하였다.

## 📎 Related Works

음성의 시간-주파수 특성을 처리하기 위한 기존 연구로 VGG-net 기반의 4층 CNN 및 MaxPool 구조나 MobileNetV2 스타일의 네트워크를 스펙트로그램에 적용한 사례가 있었다. 하지만 이러한 접근 방식은 단순히 층을 깊게 쌓는 수준에 그쳐 기존 서브샘플링 모듈과 큰 차이가 없거나, 과도한 스트라이드(Stride) 사용으로 인해 스펙트로그램 정보가 심하게 뭉개지는(Blurring) 한계가 있었다.

또한, 점진적 다운샘플링(Progressive Downsampling)을 적용한 Squeezeformer나 Temporal U-Net 구조를 사용하여 80ms까지 다운샘플링 후 다시 40ms로 복구하는 방식의 연구가 진행된 바 있다. Nextformer는 이러한 복잡한 구조 대신, ConvNeXt를 통한 고도화된 특징 추출과 단순하지만 효율적인 추가 다운샘플링 모듈을 조합하여 차별점을 둔다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

Nextformer는 CTC/AED(Connectionist Temporal Classification / Attention-based Encoder-Decoder) 통합 시스템을 기반으로 한다. 인코더는 **CNTF 모듈 $\rightarrow$ Conformer 블록 ($N/2$개) $\rightarrow$ 추가 다운샘플링 모듈 $\rightarrow$ Conformer 블록 ($N/2$개)** 순으로 구성된다. 학습을 위한 목적 함수(Objective Function)는 다음과 같이 CTC 손실과 AED 손실의 가중 합으로 정의된다.

$$\mathcal{L} = \alpha \mathcal{L}_{ctc}(H, Y) + (1-\alpha) \mathcal{L}_{att}(H, Y)$$

여기서 $\alpha$는 두 손실 함수의 균형을 맞추기 위한 하이퍼파라미터이다.

### 2. ConvNeXt Time-Frequency (CNTF) 모듈

CNTF 모듈은 ConvNeXt 블록을 적층하여 구성되며, 시간-주파수 특징을 정밀하게 처리한다.

**ConvNeXt Block의 구조:**
하나의 블록은 다음과 같은 순서로 구성된다.
$$\text{Depthwise Conv (7x7)} \rightarrow \text{Layer Norm (LN)} \rightarrow \text{Pointwise Conv (1x1)} \rightarrow \text{GeLU} \rightarrow \text{Pointwise Conv (1x1)} \rightarrow \text{LayerScale}$$
여기서 Depthwise Convolution은 음성 특징의 2차원 특성(시간, 주파수)을 반영하여 $7 \times 7$ 커널 크기를 사용하며, 이후의 Pointwise Convolution 층들은 Transformer의 FeedForward 네트워크와 유사한 역할을 수행한다.

**CNTF의 단계적 구성:**
CNTF 모듈은 총 3단계(Stage)로 구성되며, 각 단계의 채널 수는 $c, 2c, 3c$로 점진적으로 증가한다.

- **1단계**: $2 \times 2$ 스트라이드를 가진 $\text{Conv2d}$와 $\text{LN}$을 통해 시간과 주파수 축을 모두 다운샘플링한다.
- **2단계**: 1단계와 유사하게 다운샘플링을 수행하며 $\text{Conv2d}$와 $\text{LN}$의 순서만 변경한다.
- **3단계**: 주파수 축으로만 다운샘플링을 수행한다. (시간 축 다운샘플링은 결과 저하를 초래하므로 배제하고, 이후 추가 다운샘플링 모듈로 위임한다.)
최종적으로 $10\text{ms}$의 프레임 레이트가 $40\text{ms}$로 변환되며, Linear 레이어를 통해 시간-채널 출력으로 변환되어 Conformer 블록으로 전달된다.

### 3. 추가 다운샘플링 모듈 (Additional Downsampling Module)

계산 효율성을 높이기 위해 Conformer 레이어 중간에 삽입되는 이 모듈은 FSMN-memory 스타일의 다운샘플링 층, Swish 활성화 함수, $\text{LN}$ 층으로 구성된다. FSMN-like 다운샘플링 층의 수식은 다음과 같다.

$$p_t' = \sum_{i=0}^1 w_i p_{t-i}, \quad t\%2=1$$

여기서 $w$는 학습 가능한 가중치이며, 이 방식은 Average Pooling보다 성능이 약간 더 좋으면서 $\text{Conv1d}$보다 파라미터 수가 훨씬 적어 매우 경량화된 구조이다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: AISHELL-1 (150시간 학습), WenetSpeech (10,005시간 학습)
- **특징 추출**: 80-dimension log-Mel filterbank, 25ms window, 10ms shift.
- **비교 대상**: Conformer-S (Small), Conformer-L (Large)
- **평가 지표**: CER (Character Error Rate)

### 2. 정량적 결과

**AISHELL-1 결과:**

- Non-streaming 모드에서 Nextformer-S는 Conformer-S 대비 상대적으로 $7.3\%$ (WNARS 디코딩 기준)의 CER 감소를 보였다.
- Streaming 모드에서는 $6.3\%$의 상대적 CER 감소를 기록하였다.
- 최종적으로 AISHELL-1에서 **CER 4.06%**라는 SOTA 결과를 달성하였다.

**WenetSpeech 결과:**

- Non-streaming 모드에서 $5.0\% \sim 6.5\%$의 상대적 CER 감소를 보였다.
- Streaming 모드에서는 $7.5\% \sim 14.6\%$의 상대적 CER 감소를 보이며 더 큰 성능 향상을 기록하였다.
- Test_Net에서 **7.56%**, Test_Meeting에서 **11.29%**의 CER을 기록하며 SOTA를 달성하였다.

### 3. 구성 요소 분석 (Component Study)

- **CNTF 및 추가 다운샘플링 효과**: 두 모듈을 각각 제거했을 때 CER이 상승($4.06\% \rightarrow 4.18\% / 4.14\%$)하여, 두 구성 요소 모두 성능 향상에 필수적임을 확인하였다.
- **CNTF vs 8-layer CNN**: 단순 8층 CNN으로 대체했을 때보다 CNTF가 훨씬 적은 FLOPs(약 1/4 수준)로 더 우수한 성능($4.06\% \text{ vs } 4.19\%$)을 보였다.
- **다운샘플링 위치**: 추가 다운샘플링 모듈을 Conformer 레이어의 중간 지점에 배치하는 것이 가장 좋은 결과를 냄을 확인하였다.

## 🧠 Insights & Discussion

본 논문은 음성 인식 모델에서 간과되었던 '시간-주파수 특성'의 중요성을 강조하며, 이를 최신 CV 아키텍처인 ConvNeXt를 통해 성공적으로 해결하였다. 특히 단순한 CNN 적층보다 ConvNeXt의 Depthwise Convolution과 Inverted Bottleneck 구조가 음성 특징 추출에 훨씬 효율적이라는 점을 입증하였다.

흥미로운 점은 Streaming 모드에서 Non-streaming 모드보다 더 큰 성능 향상이 나타났다는 것이다. 저자들은 이를 Streaming 모드에서 사용 가능한 좌측 시간-주파수 문맥(Context)이 제한적이기 때문에, CNTF 모듈을 통한 정밀한 초기 특징 추출이 더 결정적인 역할을 한 것으로 해석하고 있다.

다만, 본 논문에서는 주로 CTC/AED 시스템에 집중하여 검증하였으며, Neural Transducer 등 다른 E2E 구조로의 전이 가능성은 언급만 되었을 뿐 실제 실험 결과는 제시되지 않았다. 또한, ConvNeXt 블록의 하이퍼파라미터(커널 사이즈 $7 \times 7$ 등)가 음성 데이터의 특성에 최적화된 것인지에 대한 심층적인 분석보다는 기존 CV 논문의 권장 사항을 따른 경향이 있다.

## 📌 TL;DR

Nextformer는 Conformer의 단순한 서브샘플링 모듈을 **ConvNeXt 기반의 CNTF 모듈**로 대체하고, 중간에 **경량 다운샘플링 층**을 추가하여 음성 특징의 시간-주파수 정보를 극대화한 모델이다. 이를 통해 계산 비용(FLOPs)은 Conformer와 비슷하게 유지하면서도 AISHELL-1(CER 4.06%)과 WenetSpeech에서 SOTA 성능을 달성하였다. 이 연구는 향후 E2E ASR 인코더 설계 시 시간-주파수 도메인의 정밀한 모델링이 성능 향상의 핵심 키가 될 수 있음을 시사한다.
