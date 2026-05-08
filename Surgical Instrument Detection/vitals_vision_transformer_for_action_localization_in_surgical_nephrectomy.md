# ViTALS: Vision Transformer for Action Localization in Surgical Nephrectomy

Soumyadeep Chandra et al. (2024)

## 🧩 Problem to Solve

본 논문은 수술 비디오에서의 Action Localization(행동 국소화), 특히 수술 단계 인식(Surgical Phase Recognition) 문제를 해결하고자 한다. 수술 비디오 분석은 자동화된 수술 훈련, 수술 워크플로우 최적화 및 내성적 보조 도구 제공 등의 측면에서 매우 중요하다.

그러나 이 작업에는 다음과 같은 주요 문제점들이 존재한다. 첫째, 개인정보 보호 문제로 인해 적절한 의료 데이터셋이 매우 부족하다. 둘째, 자연 영상(Natural videos)과 달리 수술 영상은 장면의 다양성과 구성이 상이하여 기존의 일반적인 비디오 분석 모델을 그대로 적용하기에는 최적화되지 않았다. 셋째, Vision Transformer(ViT)와 같은 강력한 모델은 Inductive Bias가 부족하여 데이터셋의 규모가 작은 의료 영상 데이터에서 심각한 과적합(Overfitting)이 발생할 가능성이 높다.

따라서 본 연구의 목표는 데이터 부족 문제를 해결하기 위한 새로운 신장 절제술(Nephrectomy) 데이터셋인 UroSlice를 구축하고, 적은 데이터로도 효과적으로 학습하며 국소적-전역적 의존성을 모두 포착할 수 있는 ViTALS 모델을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Vision Transformer의 강력한 모델링 능력과 Temporal Convolution의 Local Connectivity Inductive Bias를 결합하여, 데이터 부족 상황에서도 과적합을 방지하고 정밀한 수술 단계 예측을 수행하는 것이다.

주요 기여 사항은 다음과 같다.

- **UroSlice 데이터셋 구축**: 우측 및 좌측 부분/근치적 신장 절제술을 포함하는 복잡한 수술 데이터셋을 처음으로 소개하였다. 이 데이터셋은 단계의 순서가 일정하지 않고 시간적 변동성이 커서 기존 모델들에게 도전적인 과제를 제공한다.
- **ViTALS 모델 제안**: Hierarchical Dilated Temporal Convolution 레이어와 Attention 모듈을 통합한 인코더-디코더 구조를 제안하였다. 이를 통해 수술 영상의 국소적 특성과 전역적 문맥을 동시에 학습하며, 반복적인 Refinement 과정을 통해 예측 정확도를 높인다.
- **SOTA 성능 달성**: Cholec80 데이터셋에서 89.8%, UroSlice 데이터셋에서 66.1%의 정확도를 달성하며 기존 최신 모델(SOTA)들보다 우수한 성능을 입증하였다.

## 📎 Related Works

기존의 수술 행동 분할(Surgical action segmentation) 연구는 크게 단일 단계(Single-stage) 모델과 다단계(Multi-stage) 모델로 나뉜다.

1. **단일 단계 모델**: 초기에는 HMM(Hidden Markov Models), Dynamic Time-Wrapping 등의 통계적 접근 방식이나 센서 데이터를 활용한 방식이 사용되었다. 이후 CNN 기반의 EndoNet이나 RNN/LSTM을 활용한 SV-RCNet 등이 등장하며 비디오 프레임만을 이용한 딥러닝 접근법이 주류가 되었다.
2. **다단계 및 최신 모델**: 최근에는 Attention Regularization을 도입한 OPERA, 하이브리드 spatio-temporal transformer를 사용하는 Trans-SVNet, 그리고 Causal Dilated Convolution을 활용한 ASFormer 등이 제안되었다.

본 논문의 접근 방식은 특히 ASFormer와 같은 Transformer 기반 구조에서 영감을 받았으나, ViT의 Inductive Bias 부족 문제를 해결하기 위해 Dilated Temporal Convolution을 전략적으로 배치하고, 디코더에서 Cross-attention을 통해 예측값을 단계적으로 정밀화한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 전체 파이프라인

ViTALS의 전체 구조는 크게 **Spatial Feature Extractor**와 **Action Segmentation Network(Encoder-Decoder)**의 두 단계로 구성된다.

### 1. Spatial Feature Extractor (FE)

GPU 메모리 제한으로 인해 원본 비디오 프레임을 직접 네트워크에 입력하는 대신, 사전 학습된 ResNet50을 사용하여 각 프레임을 저차원 특징 벡터(Feature vector)로 임베딩한다.

- **데이터 샘플링**: 시간적 정보의 손실을 최소화하면서 효율성을 높이기 위해, 전체 비디오에서 약 15,000개의 프레임을 균등 간격 $w$로 샘플링하여 학습에 사용한다.
- **특징 추출**: 샘플링된 프레임 $X' = (x_1, x_{1+w}, \dots, x_n)$에 대해 다음과 같이 특징 벡터 $E'$를 생성한다.
$$E' = \{e_1, e_{1+w}, \dots, e_n\} = FE(X') = \{FE(x_1), FE(x_{1+w}), \dots, FE(x_n)\}$$
이후 전체 시퀀스에 대해 동일하게 적용하여 $E = \{e_i\}_{i=1}^n$ (크기: $n \times d$, 여기서 $d=2048$)를 생성한다.

### 2. ViTALS Model

#### Encoder

인코더는 수술 단계의 초기 예측값 $p_0$를 생성하는 역할을 하며, 다음과 같은 구조를 가진다.

- **Dilated Temporal Convolution**: Point-wise FC 레이어 대신 Dilated Convolution을 사용하여 Local Inductive Bias를 부여한다. 이는 작은 데이터셋에서 모델이 더 빠르게 수렴하고 과적합을 방지하게 한다.
- **Hierarchical Structure**: 레이어가 깊어질수록 Dilation rate $d_w$를 2배씩 증가시켜 ($d_w = 2^i, i=1, \dots, L$) 수용 영역(Receptive field)을 확장한다. 이를 통해 국소적 특징부터 전역적 의존성까지 단계적으로 학습한다.
- **Self-Attention**: Convolution 레이어 이후 Self-attention 레이어를 배치하여 프레임 간의 상호작용을 모델링한다.
- **Temporal Fusion Head**: 모든 레이어의 중간 특징들을 통합하여 각 프레임에 대한 초기 단계 확률 분포 $p_0 \in \mathbb{R}^{n \times K}$를 출력한다.

#### Decoder

디코더는 인코더의 초기 예측값을 입력받아 이를 더욱 정밀하게 다듬는(Refine) 역할을 수행한다.

- **Cross-Attention Mechanism**: 디코더의 각 블록은 Cross-attention을 사용한다. 이때 Query($Q$)는 이전 단계의 모듈에서 오고, Key($K$)와 Value($V$)는 이전 레이어에서 유도된다. 이러한 구조는 예측 과정의 안정성을 높이고 세밀한 국소 특징을 조정하는 데 도움을 준다.

### 3. Loss Functions

모델은 인코더의 출력과 $N$개 디코더 단계의 출력을 모두 사용하여 학습하며, 전체 손실 함수 $L$은 다음과 같이 정의된다.
$$L = L_{\text{encoder}} + \sum_{i=1}^{N} L_{\text{decoder}_i} \approx \sum_{i=1}^{N+1} (L_{\text{ce}} + \lambda L_{\text{smooth}})$$

- $L_{\text{ce}}$ (Cross-Entropy Loss): 예측 레이블과 실제 정답(Ground Truth) 사이의 차이를 최소화한다.
- $L_{\text{smooth}}$ (Weighted Smooth Loss): 예측값이 급격하게 변하는 Over-segmentation 현상을 방지하여 더 매끄러운 예측 결과를 유도한다.

## 📊 Results

### 실험 설정

- **데이터셋**: Cholec80(담낭 절제술) 및 UroSlice(신장 절제술)를 사용하였다.
- **지표**: 비디오 수준의 Accuracy(AC)와 시퀀스 수준의 Precision(PR), Recall(RE), Jaccard index(JA)를 측정하였다.
- **비교 대상**: EndoNet, SV-RCNet, Trans-SVNet, ASFormer 등 SOTA 모델들과 비교하였다.

### 주요 결과

1. **정량적 성능**:
   - **Cholec80**: Accuracy **89.8%**를 달성하여 기존 모델들보다 약 1.4% 향상된 성능을 보였다.
   - **UroSlice**: Accuracy **66.1%**를 기록하였으며, 이는 Trans-SVNet(62.2%) 및 ASFormer(57.4%)보다 크게 높은 수치이다. 특히 UroSlice에서는 정확도가 약 5.9% 향상되어 복잡한 워크플로우에서도 강건함을 입증하였다.
2. **정성적 결과**: 그림 2의 시각화 결과, 모델의 단계(Stage)가 진행될수록 예측값이 Ground Truth에 가깝게 정밀하게 조정(Refinement)되는 과정이 확인되었다.
3. **Ablation Study**:
   - **FE 네트워크 영향**: ResNet50을 사용했을 때 Accuracy와 GPU 메모리 효율성 면에서 가장 우수한 성능을 보였다. (Conv layer $\rightarrow$ Patch Embedding $\rightarrow$ ResNet50 순으로 성능 향상)
   - **디코더의 필요성**: Encoder만 사용했을 때보다 Decoder를 추가했을 때 Cholec80의 Accuracy가 $84.8\% \rightarrow 89.8\%$로, UroSlice는 $54.8\% \rightarrow 66.1\%$로 크게 상승하였다.

## 🧠 Insights & Discussion

본 논문은 ViT의 전역적 모델링 능력과 CNN의 국소적 Inductive Bias를 적절히 결합함으로써, 데이터가 부족한 의료 영상 분야에서도 높은 성능을 낼 수 있음을 보여주었다. 특히, 단순히 모델의 깊이를 늘리는 것이 아니라 Hierarchical Dilated Convolution을 통해 수용 영역을 체계적으로 확장하고, 디코더의 Cross-attention을 통해 예측값을 반복적으로 정밀화한 설계가 유효했음을 알 수 있다.

**강점**:

- 새로운 복잡한 수술 데이터셋(UroSlice)을 구축하여 학계에 기여하였다.
- Inductive Bias 문제를 해결하여 소규모 데이터셋에서도 과적합 없이 SOTA 성능을 달성하였다.
- 다단계 정제(Multi-stage refinement) 과정을 통해 예측의 신뢰도를 높였다.

**한계 및 논의사항**:

- 논문에서 제안한 샘플링 전략(15,000프레임 고정)이 모든 길이의 수술 영상에 대해 최적인지에 대한 분석은 부족하다.
- ResNet50이라는 고정된 Feature Extractor에 의존하고 있는데, End-to-End 학습이 가능하도록 최적화한다면 더 높은 성능을 기대할 수 있을 것이다. 다만, 이는 논문에서 언급한 GPU 메모리 제한 문제와 상충하는 지점이다.

## 📌 TL;DR

본 연구는 데이터 부족과 복잡한 워크플로우라는 수술 영상 분석의 한계를 극복하기 위해, 새로운 신장 절제술 데이터셋인 **UroSlice**를 제안하고 이를 분석하기 위한 **ViTALS** 모델을 개발하였다. ViTALS는 **Hierarchical Dilated Temporal Convolution**을 통해 국소적 특성을 학습하고, **ViT 기반의 Encoder-Decoder** 구조와 **Cross-attention**을 통해 전역적 문맥을 반영하여 예측값을 정밀하게 수정한다. 그 결과 Cholec80(89.8%)과 UroSlice(66.1%) 데이터셋 모두에서 SOTA 성능을 달성하였으며, 이는 향후 의료 영상의 자동화된 수술 단계 인식 및 워크플로우 최적화 연구에 중요한 기반이 될 것으로 보인다.
