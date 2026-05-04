# MambaOut: Do We Really Need Mamba for Vision?

Weihao Yu, Xinchao Wang (2024)

## 🧩 Problem to Solve

본 논문은 최근 자연어 처리(NLP) 분야에서 Transformer의 quadratic complexity 문제를 해결하기 위해 등장한 Mamba(State Space Model, SSM) 아키텍처가 컴퓨터 비전(CV) 작업에서도 실제로 필수적인지에 대한 근본적인 의문을 제기한다.

Mamba는 RNN과 유사한 토큰 믹서(token mixer)를 통해 긴 시퀀스를 효율적으로 처리할 수 있는 능력을 갖추고 있으며, 이를 비전 작업에 적용하려는 다양한 시도(Vision Mamba, VMamba 등)가 이어졌다. 그러나 이러한 모델들의 실제 성능은 최신 Convolutional Neural Networks(CNN)나 Attention 기반 모델들에 비해 기대에 못 미치는 경우가 많았다. 따라서 본 연구의 목표는 Mamba의 핵심 기제인 SSM이 비전 작업, 특히 이미지 분류(Image Classification)와 같은 작업에서 정말로 필요한지 분석하고, 이를 실험적으로 검증하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 직관은 Mamba가 **긴 시퀀스(long-sequence)**와 **자기회귀적 특성(autoregressive characteristics)**을 가진 작업에 최적화되어 있다는 점이다. 저자들은 비전 작업의 특성을 분석하여 다음과 같은 설계 아이디어를 제시한다.

1. **작업 특성 분석**: 이미지 분류는 긴 시퀀스나 자기회귀적 특성을 모두 갖지 않으므로 SSM이 불필요할 것이라는 가설(Hypothesis 1)을 세운다. 반면, 객체 탐지(Object Detection)나 세그멘테이션(Segmentation)은 자기회귀적이지는 않지만 긴 시퀀스 특성을 가지므로 SSM이 잠재적으로 유용할 것이라는 가설(Hypothesis 2)을 세운다.
2. **MambaOut 제안**: Mamba 블록에서 핵심 토큰 믹서인 SSM을 제거하고 Gated CNN 블록만을 쌓아 올린 단순한 모델인 `MambaOut`을 구축하여, SSM의 유무에 따른 성능 차이를 비교 분석한다.
3. **기준점(Baseline) 제시**: Occam's Razor(옥컴의 면도날) 원칙에 따라, 불필요한 복잡성을 제거한 MambaOut을 향후 비전 Mamba 연구의 자연스러운 베이스라인으로 제시한다.

## 📎 Related Works

최근 Transformer의 Attention 메커니즘이 시퀀스 길이에 따라 연산량이 제곱으로 증가하는 문제를 해결하기 위해 Low-rank approach, Kernelization, Token mixing range limitation 등 다양한 선형 복잡도 모델들이 연구되었다. 특히 RWKV와 Mamba 같은 RNN 스타일의 모델들이 LLM에서 뛰어난 성능을 보이며 주목받았다.

이를 비전에 적용한 Vision Mamba, VMamba, LocalMamba, PlainMamba 등의 연구들이 등장했다. 이들은 SSM을 통해 전역적인 수용장(Global Receptive Field)을 확보하면서도 효율적인 연산을 가능하게 하려 했다. 하지만 본 논문은 이러한 기존 연구들이 SSM의 필요성에 대한 근본적인 질문 없이 구조적 개선에만 집중했다는 점을 지적하며, 특정 비전 작업에서는 SSM 없이 단순한 구조만으로도 충분하거나 오히려 더 나은 성능을 낼 수 있음을 강조하며 차별점을 둔다.

## 🛠️ Methodology

### 1. Mamba의 메커니즘 및 적합성 분석

Mamba의 토큰 믹서인 selective SSM은 입력에 의존적인 파라미터 $(\Delta, A, B, C)$를 사용하여 다음과 같이 상태 방정식을 정의한다.

$$h_t = Ah_{t-1} + Bx_t$$
$$y_t = Ch_t$$

여기서 $h$는 고정된 크기의 메모리 역할을 하며, 이는 본질적으로 손실이 발생하는(lossy) 구조이다. 반면 Attention은 모든 이전 토큰의 Key와 Value를 저장하는 lossless 메모리를 가지지만, 시퀀스가 길어질수록 연산 비용이 급격히 증가한다.

또한, SSM의 재귀적 특성으로 인해 Mamba는 현재 토큰이 이전 토큰들의 정보만 참조할 수 있는 **Causal mode**(자기회귀적 모드)로 동작한다. 반면 비전 인식 작업은 이미지 전체를 한 번에 보는 **Fully-visible mode**가 적합하며, 여기에 억지로 Causal 제약을 가하면 성능 저하가 발생한다.

### 2. 시퀀스 길이의 정량적 정의

저자들은 Transformer 블록의 FLOPs 계산식을 통해 시퀀스 길이 $L$과 채널 차원 $D$의 관계를 분석하여, $L > 6D$일 때 quadratic term의 연산량이 linear term을 압도한다고 정의한다.

- **ImageNet 분류**: 입력 토큰 수(약 196개)가 임계값($\tau_{small}=2304$)보다 훨씬 작으므로 long-sequence 작업이 아니다.
- **COCO 탐지 및 ADE20K 세그멘테이션**: 입력 토큰 수가 약 4K개로 임계값에 근접하거나 상회하므로 long-sequence 작업으로 간주한다.

### 3. MambaOut 아키텍처

MambaOut은 Mamba 블록의 모태가 되는 **Gated CNN** 구조를 채택한다. 전체적인 메타 아키텍처는 다음과 같다.

$$X' = \text{Norm}(X)$$
$$Y = (\text{TokenMixer}(X'W_1) \odot \sigma(X'W_2))W_3 + X$$

여기서 $\text{TokenMixer}$의 차이가 핵심이다.

- **Gated CNN (MambaOut)**: $\text{TokenMixer}(Z) = \text{Conv}(Z)$
- **Mamba**: $\text{TokenMixer}(Z) = \text{SSM}(\sigma(\text{Conv}(Z)))$

MambaOut은 SSM을 완전히 제거하고, $7 \times 7$ depthwise convolution을 토큰 믹서로 사용한다. 또한 효율성을 위해 모든 채널이 아닌 일부 채널에 대해서만 convolution을 수행하는 partial convolution 방식을 적용하였다. 전체 구조는 ResNet과 유사한 4단계 계층 구조(Hierarchical architecture)를 따른다.

## 📊 Results

### 1. ImageNet 이미지 분류

실험 결과, SSM을 제거한 **MambaOut이 모든 사이즈에서 기존의 visual Mamba 모델들(Vision Mamba, VMamba 등)보다 일관되게 높은 성능을 기록**하였다.

- MambaOut-Small은 Top-1 정확도 84.1%를 달성하여 LocalVMamba-S보다 0.4% 높았으며, 연산량(MACs)은 79% 수준에 불과했다.
- 이는 이미지 분류 작업에서 SSM이 불필요하다는 **Hypothesis 1**을 강력하게 뒷받침한다.

### 2. COCO 객체 탐지 및 인스턴스 세그멘테이션

Mask R-CNN의 백본으로 사용한 결과, MambaOut이 일부 Mamba 모델보다는 우수했지만, **SOTA visual Mamba 모델(VMamba, LocalVMamba)의 성능에는 미치지 못하였다.**

- MambaOut-Tiny는 VMamba-T에 비해 $AP_b$ 기준 1.4, $AP_m$ 기준 1.1 낮게 측정되었다.
- 이는 긴 시퀀스 처리가 필요한 작업에서는 SSM이 이점을 제공한다는 **Hypothesis 2**를 검증한다.

### 3. ADE20K 시맨틱 세그멘테이션

UperNet의 백본으로 사용한 결과 역시 COCO와 유사한 경향을 보였다. **LocalVMamba-T가 MambaOut-Tiny보다 mIoU 기준 0.5 높게 나타나**, 세그멘테이션 작업에서도 SSM의 잠재적 유용성이 확인되었다.

## 🧠 Insights & Discussion

본 연구는 비전 모델 설계에 있어 무조건적으로 최신 기법(Mamba/SSM)을 도입하기보다, **해당 작업의 데이터 특성(시퀀스 길이, 인과성 여부)을 먼저 분석하는 것이 중요함**을 시사한다.

- **강점**: 단순한 Gated CNN 구조만으로도 복잡한 SSM 기반 모델들을 능가하는 성능을 보임으로써, 비전 분야에서 Mamba의 필요성에 대해 비판적인 시각을 제공하였다.
- **한계 및 논의**: 비록 MambaOut이 일부 Mamba 모델보다 좋았으나, 여전히 TransNeXt와 같은 최신 Hybrid(Conv + Attention) 모델들과는 상당한 성능 격차가 존재한다. 이는 Mamba 계열 모델들이 비전 분야에서 진정한 경쟁력을 갖추려면 단순한 구조 변경 이상의 근본적인 개선이 필요함을 의미한다.
- **결론적 해석**: 이미지 분류와 같은 짧은 시퀀스의 이해 작업에서는 SSM이 오버엔지니어링(Over-engineering)이며, 긴 시퀀스가 필요한 dense prediction 작업에서만 제한적으로 유효할 가능성이 높다.

## 📌 TL;DR

본 논문은 비전 작업에서 Mamba(SSM)의 필요성을 분석하고, SSM을 제거한 단순한 Gated CNN 기반의 **MambaOut** 모델을 제안한다. 실험 결과, **이미지 분류에서는 MambaOut이 기존 Mamba 모델들을 압도**하여 SSM이 불필요함을 입증했으나, **객체 탐지 및 세그멘테이션과 같은 긴 시퀀스 작업에서는 SSM이 여전히 유효함**을 확인했다. 이 연구는 비전 모델 설계 시 작업 특성에 맞는 토큰 믹서를 선택해야 함을 강조하며, MambaOut을 효율적인 베이스라인으로 제시한다.
