# Transformer Utilization in Medical Image Segmentation Networks

Saikat Roy, Gregor Koehler, Michael Baumgartner, Constantin Ulrich, Jens Petersen, Fabian Isensee, and Klaus Maier-Hein (2023)

## 🧩 Problem to Solve

최근 자연어 처리 및 일반 컴퓨터 비전 분야에서 Transformer의 성공에 힘입어, 의료 영상 분할(Medical Image Segmentation) 분야에서도 Transformer 기반 아키텍처가 빠르게 도입되고 있다. 그러나 다양한 형태로 Convolutional 블록과 Transformer 블록이 결합되면서, 실제로 Transformer가 성능 향상에 어느 정도 기여하는지에 대한 정량적인 분석이 부족한 상황이다.

본 논문은 의료 영상 분할 네트워크에서 Transformer가 실제로 얼마나 유용하게 활용되고 있는지를 정량화하는 것을 목표로 한다. 특히, Transformer 블록이 생성하는 표현(representation)이 다른 단순한 연산으로 대체 가능한지, 즉 '표현의 대체 가능성(representability)'을 분석하여 아키텍처 설계의 효율성을 검토하고자 한다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 **Transformer Ablation**이라는 극단적인 절제 실험을 통해 Transformer의 실질적인 기여도를 측정하는 것이다. 연구진은 Transformer 블록을 단순한 선형 연산자(plain linear operators)로 대체하여, Transformer의 복잡한 self-attention 메커니즘이 제거되었음에도 불구하고 네트워크가 성능을 유지할 수 있는지를 확인하였다. 이를 통해 의료 영상 분할 모델에서 성능을 결정짓는 요소가 Transformer 자체의 연산 능력인지, 아니면 이를 둘러싼 전체적인 아키텍처 설계(예: 계층적 구조)인지에 대한 통찰을 제공한다.

## 📎 Related Works

논문은 Vision Transformer (ViT) 및 Swin Transformer와 같은 최신 비전 모델들이 의료 영상 분야로 전이되어 다양한 하이브리드 모델(예: UNETR, SwinUNETR, TransUNet 등)이 제안되었음을 언급한다.

기존 연구들은 주로 새로운 아키텍처를 제안하고 SOTA(State-of-the-art) 성능을 달성하는 데 집중하였으나, 본 논문은 이러한 모델들 내에서 Transformer 모듈이 실제로 어떤 역할을 수행하는지, 그리고 그것이 필수적인지를 비판적으로 분석한다는 점에서 기존의 성능 중심 접근 방식과 차별화된다.

## 🛠️ Methodology

### Transformer Ablation 정의

Transformer의 self-attention 메커니즘은 기본적으로 다음과 같이 표현된다.
$$X = s(QK^T) \cdot V$$
여기서 $Q, K, V \in \mathbb{R}^{N \times d}$이며, $s$는 scaling 함수, $N$은 시퀀스 길이, $d$는 차원 수를 의미한다.

**Transformer Ablation**은 위에서 정의된 Transformer 블록을 완전히 제거하고, 대신 다운스트림 텐서의 호환성(tensor compatibility)을 유지하기 위한 단순한 선형 투영(linear projection)으로 대체하는 것을 의미한다.

- **ViT 기반 모델**: linear projection 기반의 tokenizer로 대체한다.
- **Swin-Transformer 기반 모델**: PatchMerging을 포함한 linear projection으로 대체한다.

### 실험 설계

- **대상 모델**: UNETR, SwinUNETR, TransUNet, TransBTS, TransFuse, CoTr, nnFormer, UTNet 등 총 8개의 모델을 분석하였으며, 기준선(baseline)으로 순수 Convolutional 모델인 nnUNet을 사용하였다.
- **데이터셋**:
  - Kidney Tumor Segmentation (KiTS) 2021: 300개 볼륨, 3개 분할 구조.
  - Multi-Organ Abdominal CT (MultiACT): 90개 볼륨, 8개 분할 구조.
- **평가 지표**: Dice Similarity Coefficient (DSC)와 1mm 허용 오차를 가진 Surface Dice Coefficient (SDC)를 사용하였다.
- **학습 환경**: nnUNet 파이프라인을 기반으로 하되, optimizer를 SGD에서 AdamW로 변경하여 학습하였다.

## 📊 Results

실험 결과, 많은 모델에서 Transformer 블록을 제거하고 선형 연산으로 대체했음에도 불구하고 성능 저하가 거의 없거나, 오히려 성능이 향상되는 결과가 나타났다.

- **표현의 대체 가능성**: UNETR를 제외한 대부분의 네트워크에서 Transformer Ablation 이후에도 성능이 유지되었다. 이는 많은 모델이 Transformer 자체보다는 Convolution이나 Swin-block과 같은 데이터 효율적인 구성 요소에 의해 성능이 주도되고 있음을 시사한다.
- **모델별 특이사항**:
  - **UNETR**: Ablation 시 성능 저하가 가장 뚜렷하게 나타나, Transformer 활용도가 높음을 보였다.
  - **SwinUNETR**: Swin Transformer 블록을 제거했음에도 성능 변화가 거의 없었다.
  - **nnFormer**: Ablation 이후 오히려 성능이 향상되는 결과가 관찰되었다.
- **정량적 결과 (Table 2 참조)**:
  - 많은 경우 $\text{S} - \text{Abl.}$ (Standard 성능과 Ablated 성능의 차이) 값이 매우 작거나 음수로 나타났다.
  - 특히 파라미터 수의 급격한 감소(Ratio $\ll 1$)에도 불구하고 DSC/SDC 수치가 유지되는 경향을 보였다.

## 🧠 Insights & Discussion

본 논문은 실험 결과를 바탕으로 다음과 같은 네 가지 주요 통찰을 제시한다.

1. **표현의 대체 가능성 (Replaceability of Representations)**: 의료 영상 분할 네트워크에서 Transformer가 학습하는 표현은 종종 다른 효율적인 컴포넌트(CNN 등)로 대체 가능하다. 이는 개별 블록의 성능보다 블록들의 조합과 전체적인 아키텍처 스타일이 더 중요하다는 것을 의미한다.
2. **모델 용량과 설계의 관계 (Transformer vs Non-Transformer capacity)**: 단순히 파라미터 수가 많다고 해서 Transformer의 중요성이 커지는 것은 아니다. 예를 들어 nnFormer는 적은 파라미터만으로도 성능을 유지한 반면, CoTr는 상대적으로 많은 용량을 가졌음에도 성능 저하가 발생했다. 이는 모델 용량(capacity)이 아키텍처 설계와 맞물려 작동함을 보여준다.
3. **명시적 계층적 특징 학습의 중요성 (Explicit Hierarchical Feature Learning)**: UNETR(단일 스케일 ViT)보다 SwinUNETR(계층적 구조)의 성능이 더 좋은 이유는 attention 메커니즘 자체보다 **명시적인 계층적 구조(Hierarchical Feature Learning)**라는 귀납적 편향(inductive bias)이 의료 영상 분석에 훨씬 유익하기 때문으로 분석된다.
4. **보틀넥에서의 공간적 다운샘플링 (Spatial Downsampling in the bottleneck)**: TransBTS와 TransUNet처럼 Transformer 입력 전 단계에서 8배 이상의 강한 다운샘플링을 수행하는 경우, Transformer가 학습해야 할 장거리 의존성(long-range dependencies)의 부담이 줄어들어 Ablation 시에도 성능이 유지되는 경향이 있다. 따라서 보틀넥 설계 시 입력 사이즈 설정에 주의가 필요하다.

## 📌 TL;DR

본 논문은 의료 영상 분할 모델에서 Transformer 블록을 단순 선형 연산으로 대체하는 **Transformer Ablation** 실험을 통해, 많은 하이브리드 모델들이 사실상 Transformer의 복잡한 연산 없이도 성능을 유지하는 '표현의 대체 가능성'을 가지고 있음을 밝혀냈다. 특히, 단순한 Attention 메커니즘보다 **계층적 특징 학습 구조**와 **효율적인 아키텍처 설계**가 성능 향상에 더 결정적인 역할을 한다는 점을 시사한다. 이는 향후 불필요하게 복잡한 Transformer 구조를 지양하고, 더 효율적인 의료 영상 분석 모델을 설계하는 데 중요한 근거가 될 수 있다.
