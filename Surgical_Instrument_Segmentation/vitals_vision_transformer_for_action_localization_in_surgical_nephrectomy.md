# ViTALS: Vision Transformer for Action Localization in Surgical Nephrectomy

Soumyadeep Chandra, Sayeed Shafayet Chowdhury, Courtney Yong, Chandru P. Sundaram, and Kaushik Roy (2024)

## 🧩 Problem to Solve

본 논문은 수술 비디오에서 수술 단계(surgical phase)를 인식하고 위치를 찾아내는 **Surgical Action Localization** 문제를 해결하고자 한다. 수술 비디오 분석은 수술 절차의 자동화된 교육, 수술 워크플로우 최적화, 그리고 수술 중 보조 도구로서의 활용 가능성 때문에 매우 중요하다.

그러나 이 과제는 다음과 같은 몇 가지 핵심적인 난관이 존재한다.

- **데이터의 희소성**: 개인정보 보호 문제로 인해 고품질의 의료 데이터셋을 확보하기 어렵다.
- **일반 비디오 모델의 한계**: 자연어 비디오와 달리 수술 비디오는 장면의 구성과 하위 작업의 특성이 매우 다르므로, 기존의 일반적인 비디오 분석 모델을 그대로 적용하는 것은 최적이 아니다.
- **Inductive Bias의 부재**: 특히 Vision Transformer (ViT) 계열의 모델은 Inductive Bias가 부족하여, 데이터셋의 규모가 작은 의료 분야에 적용할 경우 심각한 과적합(overfitting)이 발생하기 쉽다.

따라서 본 연구의 목표는 수술 비디오의 특성에 맞게 설계된 새로운 모델인 **ViTALS**를 제안하고, 신장 절제술(Nephrectomy)을 위한 새로운 데이터셋인 **UroSlice**를 구축하여 그 성능을 검증하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 크게 세 가지로 요약할 수 있다.

1. **UroSlice 데이터셋 구축**: 우측 및 좌측 신장의 부분 및 근치 절제술을 포함하는 복잡한 수술 데이터셋을 새롭게 도입하였다. 이 데이터셋은 수술 단계가 균일하지 않고 체계적인 순서 없이 발생하여, 예측 불가능성과 시간적 변동성이 매우 크다는 특징이 있다.
2. **ViTALS 모델 제안**: ViT의 과적합 문제를 해결하기 위해 **Dilated Temporal Convolution** 층을 통합한 모델을 설계하였다. 이는 지역적 연결성(local connectivity)이라는 Inductive Bias를 제공하여 제한된 데이터셋에서도 효과적인 학습을 가능하게 하며, 계층적 구조를 통해 지역적(local) 특징과 전역적(global) 의존성을 동시에 캡처한다.
3. **SOTA 성능 달성**: Cholec80 데이터셋에서 89.8%, UroSlice 데이터셋에서 66.1%의 정확도를 달성하며 수술 단계 인식 분야에서 최첨단(state-of-the-art) 성능을 입증하였다.

## 📎 Related Works

수술 액션 세그멘테이션 연구는 크게 단일 단계(single-stage) 모델과 다단계(multi-stage) 모델로 구분된다.

- **단일 단계 모델**: 초기에는 통계적 접근 방식, Dynamic Time-Wrapping, Hidden Markov Models (HMMs) 등을 활용하여 시각적 특징 기반의 인식을 수행하였다. 이후 딥러닝의 도입으로 CNN 기반의 EndoNet이나 RNN/LSTM 기반의 SV-RCNet과 같은 모델들이 등장하였다.
- **다단계 및 최신 모델**: 최근에는 Attention Regularization을 도입한 OPERA, 하이브리드 시공간 트랜스포머를 사용하는 Trans-SVNet, 그리고 Causal Dilated Convolution을 활용한 AVT 및 ASFormer 등이 제안되었다.

**기존 접근 방식과의 차별점**:
기존의 ViT 기반 모델(예: ASFormer)은 강력한 모델링 능력을 갖추고 있으나, 의료 데이터셋처럼 규모가 작은 경우 Inductive Bias의 부족으로 인한 과적합 문제가 발생한다. ViTALS는 이를 해결하기 위해 **Hierarchical Dilated Temporal Convolution**을 ViT 구조와 결합하여, 모델이 학습해야 할 가설 공간을 제한함으로써 소규모 데이터셋에서도 강건한 성능을 내도록 설계되었다는 점에서 차별화된다.

## 🛠️ Methodology

### 전체 파이프라인

ViTALS의 전체 시스템은 **Spatial Feature Extractor $\rightarrow$ Encoder $\rightarrow$ Decoder**의 순차적 구조로 구성된다.

### 1. Spatial Feature Extractor (FE)

GPU 메모리 제한 문제를 해결하기 위해, 원본 비디오 프레임을 직접 입력하는 대신 사전 학습된 특징 추출기를 사용한다.

- **구조**: ResNet50을 사용하여 각 프레임을 $d=2048$ 차원의 공간적 특징 벡터 $e_i$로 변환한다.
- **샘플링**: 시간적 정보를 유지하면서 계산 효율을 높이기 위해 전체 비디오를 약 15,000개의 프레임으로 균등 샘플링하여 $X'$를 구성하고, 이를 통해 특징 벡터 집합 $E = \{e_i\}_{i=1}^n$을 생성한다.

### 2. ViTALS Encoder

인코더는 지역적 특징에서 전역적 특징으로 확장되는 계층적 구조를 가진다.

- **Dilated Temporal Convolution**: 각 인코더 블록의 시작 부분에 배치되어 'local inductive bias'를 제공한다. 팽창률(dilation rate)은 $d_w = 2^i$ ($i=1, 2, \dots, L$) 형태로 층이 깊어질수록 두 배씩 증가하여 수용 영역(receptive field)을 넓힌다.
- **Self-Attention Layer**: 컨볼루션 층 이후에 배치되어 프레임 간의 상호작용을 모델링한다.
- **Temporal Fusion Head**: 모든 레이어의 중간 특징들을 통합하여 초기 단계 예측값 $p_0 \in \mathbb{R}^{n \times K}$를 생성한다.

### 3. ViTALS Decoder

인코더의 초기 예측값을 정교하게 다듬기(refine) 위해 다단계 디코더를 사용한다.

- **Cross-Attention Mechanism**: 디코더 블록 내에서 **Query(Q)**는 이전 모듈(인코더 또는 이전 디코더)에서 가져오고, **Key(K)**와 **Value(V)**는 현재 레이어의 특징에서 가져와 예측값을 수정한다.
- 이러한 분리 구조는 정제 과정의 안정성과 효과를 높인다.

### 4. 손실 함수 (Loss Functions)

모델은 인코더의 초기 예측과 각 디코더 단계의 정제된 예측을 모두 학습에 활용한다. 전체 손실 함수 $L$은 다음과 같이 정의된다.

$$L = L_{encoder} + \sum_{N} L_{decoder} \approx \sum_{N+1} (L_{ce} + \lambda L_{smooth})$$

- $L_{ce}$: Cross-Entropy Loss로, 예측된 레이블과 실제 정답(ground truth) 간의 차이를 최소화한다.
- $L_{smooth}$: Weighted Smooth Loss로, 과도한 세그멘테이션(over-segmentation)을 방지하여 예측 결과가 시간적으로 더 매끄럽게 이어지도록 돕는다.

## 📊 Results

### 실험 설정

- **데이터셋**: Cholec80(담낭 절제술, 80개 비디오) 및 UroSlice(신장 절제술, 39개 비디오).
- **지표**: 비디오 수준의 Accuracy(AC)와 시퀀스 수준의 Precision(PR), Recall(RE), Jaccard(JA) index를 사용하였다.
- **구현**: PyTorch 기반, ResNet50 백본, Adam optimizer (learning rate $5 \times 10^{-4}$), 150 epoch 학습.

### 정량적 결과

ViTALS는 두 데이터셋 모두에서 SOTA 성능을 기록하였다.

| Dataset | Method | Accuracy ($\uparrow$) | Precision ($\uparrow$) | Recall ($\uparrow$) | Jaccard ($\uparrow$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Cholec80** | Trans-SVNet | 89.1 $\pm$ 7.1 | 88.7 $\pm$ 5.0 | 89.8 $\pm$ 7.4 | 74.3 $\pm$ 6.6 |
| | **ViTALS (Ours)** | **89.8 $\pm$ 4.1** | **82.1 $\pm$ 2.1** | **84.8 $\pm$ 4.8** | **74.8 $\pm$ 5.1** |
| **UroSlice** | Trans-SVNet | 62.2 $\pm$ 6.8 | 57.8 $\pm$ 8.1 | 61.1 $\pm$ 4.8 | - |
| | **ViTALS (Ours)** | **66.1 $\pm$ 2.1** | **59.2 $\pm$ 7.2** | **63.1 $\pm$ 5.4** | - |

특히 UroSlice 데이터셋에서는 Trans-SVNet 대비 정확도가 **5.9% 향상**되었으며, 이는 복잡하고 데이터 규모가 작은 환경에서 ViTALS의 강건함이 더 크게 나타남을 의미한다.

### 소거 연구 (Ablation Study)

1. **특징 추출기(FE) 영향**: ResNet50을 사용했을 때 가장 높은 정확도와 효율적인 GPU 메모리 사용량을 보였다 (Conv layer $\rightarrow$ Patch Embedding $\rightarrow$ ResNet50 순으로 성능 향상).
2. **디코더의 영향**: 인코더만 사용했을 때보다 다단계 디코더를 추가했을 때 정확도가 크게 상승하였다 (예: UroSlice 기준 54.8% $\rightarrow$ 66.1%).

## 🧠 Insights & Discussion

본 논문은 ViT의 강력한 표현력과 Temporal Convolution의 Inductive Bias를 결합함으로써 의료 영상 분석의 고질적인 문제인 '소규모 데이터셋에서의 과적합'을 효과적으로 해결하였다.

**강점 및 해석**:

- **계층적 수용 영역**: Dilated Convolution의 팽창률을 지수적으로 증가시킴으로써, 낮은 층에서는 세밀한 지역적 특징을, 높은 층에서는 전역적인 문맥을 포착하는 구조가 수술 단계 인식에 매우 적합함을 보였다.
- **정제 과정의 중요성**: 다단계 디코더와 Cross-Attention을 통한 반복적 정제 과정이 초기 예측의 오류를 줄이고 최종 정확도를 높이는 데 결정적인 역할을 하였다.

**한계 및 논의**:

- **데이터셋 규모**: UroSlice 데이터셋이 제안되었으나, 여전히 전체적인 데이터 규모가 작아 더 다양한 수술 케이스에 대한 일반화 성능 검증이 필요하다.
- **추론 시간**: 다단계 디코더 구조는 성능을 높이지만, 실시간(real-time) 수술 보조 시스템으로 적용하기 위해서는 추론 속도에 대한 분석과 최적화가 추가로 이루어져야 한다.

## 📌 TL;DR

본 논문은 신장 절제술을 위한 새로운 데이터셋 **UroSlice**를 제안하고, ViT에 Dilated Temporal Convolution과 다단계 Encoder-Decoder 구조를 결합한 **ViTALS** 모델을 개발하였다. 이 모델은 ViT의 과적합 문제를 해결하여 소규모 의료 데이터셋에서도 강건한 성능을 보이며, 특히 복잡한 수술 워크플로우를 가진 데이터셋에서 기존 SOTA 모델들을 유의미하게 능가하였다. 이는 향후 수술 자동화 교육 및 실시간 워크플로우 최적화 연구에 중요한 기반이 될 것으로 보인다.
