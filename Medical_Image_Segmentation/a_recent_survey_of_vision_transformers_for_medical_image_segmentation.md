# A Recent Survey of Vision Transformers for Medical Image Segmentation

Asifullah Khan et al. (2023/2024)

## 🧩 Problem to Solve

의료 영상 분할(Medical Image Segmentation)은 정확한 진단, 치료 계획 수립 및 질병 모니터링을 위해 매우 중요하다. 전통적으로 이 분야는 Convolutional Neural Networks (CNNs)가 주도해 왔으며, CNN은 국소적 특징 추출(local feature extraction)에 매우 뛰어난 성능을 보였다.

그러나 CNN은 컨볼루션 연산자의 본질적인 특성으로 인해 수용 영역(receptive field)이 제한적이며, 이로 인해 영상 내 서로 떨어진 영역 간의 장거리 의존성(long-range dependencies)을 포착하는 데 한계가 있다. 의료 영상에서는 복잡하고 서로 연결된 구조나 영상 전체에 걸쳐 있는 장기 및 병변을 분할해야 하는 경우가 많으므로, 이러한 전역적 문맥(global context) 파악 능력의 부재는 성능 저하의 주요 원인이 된다.

본 논문의 목표는 최근 등장한 Vision Transformers (ViTs)와 이를 CNN과 결합한 Hybrid Vision Transformers (HVTs)가 의료 영상 분할의 이러한 한계를 어떻게 해결하고 있는지 분석하고, 다양한 의료 영상 모달리티에 적용된 최신 사례들을 체계적으로 분류하여 정리하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 의료 영상 분할을 위한 ViT 및 HVT 기반 접근 방식에 대한 포괄적인 분류 체계(Taxonomy)를 제시한 점이다.

1. **구조적 분류**: ViT 및 HVT의 통합 위치에 따라 다음과 같이 세분화하여 분석하였다.
    * **ViT 기반 방법**: $\text{Encoder}$, $\text{Decoder}$, $\text{Encoder-Decoder}$ 사이, 또는 둘 모두에 ViT를 적용한 사례로 구분하였다.
    * **HVT 기반 방법**: CNN의 국소적 특징 추출 능력과 ViT의 전역적 문맥 파악 능력을 결합한 하이브리드 구조를 분석하였다.
2. **모달리티별 분석**: CT, MRI, 초음파(Ultrasound), X-Ray, 조직병리(Histopathology), 현미경(Microscopy) 영상 등 다양한 의료 영상 모달리티에서 각 모델이 어떻게 적용되었는지와 그 정량적 결과를 상세히 검토하였다.
3. **최신 동향 및 한계 제시**: 데이터 부족, 계산 복잡도, 해석 가능성(interpretability) 등 ViT 적용 시 발생하는 현실적인 문제점과 향후 연구 방향을 제시하였다.

## 📎 Related Works

기존의 의료 영상 분할은 주로 $\text{U-Net}$과 같은 $\text{Encoder-Decoder}$ 구조의 CNN 기반 모델에 의존하였다. $\text{U-Net}$은 $\text{Skip connection}$을 통해 저수준 특징을 보존하며 뛰어난 성과를 거두었으나, 앞서 언급한 전역적 관계 모델링의 한계가 있었다. 이를 해결하기 위해 $\text{Attention}$ 메커니즘을 도입한 CNN 변형 모델들이 제안되었으나, 근본적인 수용 영역의 한계를 완전히 극복하기는 어려웠다.

이후 자연어 처리(NLP)에서 성공을 거둔 $\text{Transformer}$가 컴퓨터 비전 분야로 확장되어 $\text{Vision Transformer (ViT)}$가 등장하였다. $\text{ViT}$는 이미지를 패치(patch) 단위로 나누어 처리함으로써 전역적인 의존성을 학습할 수 있게 하였다. 하지만 $\text{ViT}$는 이미지의 공간적 불변성(translational invariance)과 같은 이미지 특유의 귀납적 편향(inductive bias)이 부족하며, 학습을 위해 매우 방대한 양의 데이터를 필요로 한다는 단점이 있다. 이러한 한계를 보완하기 위해 CNN의 강점(국소 특징 추출)과 $\text{ViT}$의 강점(전역 문맥 파악)을 결합한 $\text{Hybrid Vision Transformers (HVTs)}$ 연구가 활발히 진행되고 있다.

## 🛠️ Methodology

본 논문은 특정 알고리즘을 제안하는 것이 아니라, 기존 연구들을 분석하는 서베이 논문이다. 분석 대상이 되는 $\text{ViT}$ 및 $\text{HVT}$의 핵심 동작 원리와 분류 기준은 다음과 같다.

### 1. Vision Transformer (ViT)의 기본 원리

$\text{ViT}$는 이미지를 고정된 크기의 겹치지 않는 패치(patch)로 분할한 후, 각 패치를 선형 투영하여 임베딩 벡터(token)로 변환한다. 이후 위치 정보 손실을 막기 위해 $\text{Positional Embedding}$을 추가하며, 핵심 연산인 $\text{Multi-head Self-Attention (MSA)}$를 통해 모든 패치 간의 상관관계를 계산한다.
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
이 메커니즘을 통해 모델은 영상 내 거리에 관계없이 중요한 특징에 집중할 수 있다.

### 2. 구조적 분류 방법론

논문은 $\text{Encoder-Decoder}$ 아키텍처 내에서 $\text{ViT}$ 또는 $\text{HVT}$가 배치된 위치에 따라 모델을 분류한다.

* **$\text{ViT}$ in $\text{Encoder}$**: $\text{Encoder}$에서 전역 특징을 추출하고 $\text{CNN-based Decoder}$가 이를 통해 마스크를 생성한다. (예: $\text{UNETR}$)
* **$\text{ViT}$ in $\text{Decoder}$**: $\text{CNN-based Encoder}$가 특징을 추출하고 $\text{ViT-based Decoder}$가 경계선을 정밀하게 예측한다. (예: $\text{ConvTransSeg}$)
* **$\text{ViT}$ in both $\text{Encoder-Decoder}$**: $\text{Encoder}$와 $\text{Decoder}$ 모두에 $\text{ViT}$를 사용하여 전역 문맥을 극대화한다. (예: $\text{Swin-Unet}$)
* **$\text{ViT}$ in between $\text{Encoder-Decoder}$**: $\text{Bottleneck}$ 레이어나 $\text{Skip connection}$에 $\text{ViT}$를 배치하여 특징을 정제한다. (예: $\text{ATTransUNet}$)

### 3. $\text{Hybrid Vision Transformer (HVT)}$의 접근 방식

$\text{HVT}$는 $\text{CNN}$의 컨볼루션 연산과 $\text{ViT}$의 $\text{MSA}$를 결합한다. 주로 $\text{CNN}$을 통해 저수준의 국소 특징을 먼저 추출하고, 이를 $\text{ViT}$에 입력하여 고수준의 전역 특징을 학습하는 파이프라인을 가진다. 이를 통해 데이터 효율성을 높이고 이미지의 세부 디테일(edges, boundaries)을 더 잘 보존할 수 있다.

## 📊 Results

본 논문은 다양한 의료 영상 모달리티별로 $\text{ViT}/\text{HVT}$ 기반 모델들의 성능을 정리한 표를 제공한다. 주요 결과는 다음과 같다.

### 1. 모달리티별 적용 사례 및 성능

* **CT 영상**: $\text{TA-UNet3+}$는 신장 종양 분할에서 $\text{Dice}$ 계수 $0.9638$ (종양), $0.9885$ (신장)라는 높은 성능을 기록하였다. $\text{FocalUNETR}$는 전립선 분할에서 경계면 회귀(label regression) 작업을 추가하여 모호한 경계 문제를 해결하였다.
* **MRI 영상**: 뇌종양 분할(BraTS 데이터셋)에서 $\text{Swin UNETR}$와 $\text{UnetFormer}$ 등이 널리 사용되며, 특히 $\text{UnetFormer}$는 $\text{Dice}$ 계수 기준 $\text{WT}$(전체 종양)에서 $93.22$의 높은 성능을 보였다.
* **현미경 및 조직병리 영상**: $\text{MaxViT-UNet}$은 $\text{MoNuSeg}$ 데이터셋에서 $\text{Dice } 0.8378$를 기록하며 세포 핵 분할에서 유효함을 증명하였다.
* **초음파 및 X-Ray**: $\text{Swin-PNet}$은 유방 병변 분할에 적용되었으며, $\text{ImplantFormer}$는 치근(tooth root) 위치 예측 등에 활용되었다.

### 2. 분석 결과의 요약

* **지표**: 대부분의 연구가 $\text{Dice Similarity Coefficient (Dice)}$, $\text{Intersection over Union (IoU)}$, $\text{F1-score}$, $\text{Accuracy}$를 평가지표로 사용하였다.
* **경향성**: 순수 $\text{ViT}$ 모델보다는 $\text{CNN}$과 결합한 $\text{HVT}$ 모델들이 의료 영상의 특성(작은 데이터셋, 세밀한 경계 필요성)으로 인해 더 안정적이고 우수한 성능을 내는 경향이 있다.

## 🧠 Insights & Discussion

### 1. 강점 및 기회

$\text{ViT}$의 도입은 특히 폐(lungs)와 같이 수용 영역이 커야 하는 장기의 분할에서 획기적인 성능 향상을 가져왔다. 또한, $\text{Swin Transformer}$와 같은 계층적 구조의 $\text{ViT}$는 계산 복잡도를 낮추면서도 다중 스케일(multi-scale) 특징을 추출할 수 있어 의료 영상에 매우 적합하다는 점이 확인되었다.

### 2. 한계 및 비판적 해석

* **데이터 의존성**: $\text{ViT}$는 기본적으로 대규모 데이터로 사전 학습된 모델을 필요로 한다. 하지만 의료 데이터는 개인정보 보호 및 라벨링 비용 문제로 인해 데이터 확보가 어렵다.
* **계산 비용**: 3D 의료 영상(MRI, CT)에 $\text{ViT}$를 직접 적용할 경우, 패치 수의 증가에 따라 $\text{Self-attention}$의 연산 복잡도가 제곱으로 증가하는 문제가 발생한다.
* **해석 가능성**: $\text{Attention map}$을 통해 모델이 어디를 보는지 확인할 수 있으나, 임상 현장에서 의사들이 신뢰할 수 있는 수준의 명확한 설명력(explainability)을 제공하는지는 여전히 미해결 과제이다.

### 3. 논의 사항

본 논문은 다양한 모델을 분류하였으나, 각 모델 간의 직접적인 벤치마크 비교보다는 개별 논문의 결과를 나열하는 방식에 치중되어 있다. 동일한 데이터셋과 동일한 평가 조건 하에서의 정밀한 비교 분석이 추가된다면 더 가치 있는 가이드라인이 될 것이다.

## 📌 TL;DR

본 논문은 의료 영상 분할 분야에서 $\text{CNN}$의 국소적 한계를 극복하기 위해 도입된 $\text{Vision Transformers (ViTs)}$와 $\text{Hybrid Vision Transformers (HVTs)}$의 최신 연구를 체계적으로 분석한 서베이 논문이다. $\text{ViT}/\text{HVT}$의 통합 위치(Encoder, Decoder, Bottleneck)에 따른 분류 체계를 제시하고, CT, MRI, 초음파 등 다양한 모달리티에서의 적용 사례를 정리하였다. 결과적으로 전역적 문맥 파악 능력을 갖춘 $\text{ViT}$와 국소 특징 추출에 강한 $\text{CNN}$을 결합한 하이브리드 구조가 의료 영상 분할의 실질적인 해결책임을 시사하며, 향후 데이터 효율성 향상과 계산 복잡도 감소가 주요 연구 방향이 될 것임을 제시한다.
