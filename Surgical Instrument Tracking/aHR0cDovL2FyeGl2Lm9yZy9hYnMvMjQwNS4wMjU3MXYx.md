# ViTALS: Vision Transformer for Action Localization in Surgical Nephrectomy

Soumyadeep Chandra, Sayeed Shafayet Chowdhury, Courtney Yong, Chandru P. Sundaram, and Kaushik Roy (2024)

## 🧩 Problem to Solve

본 논문은 수술 비디오에서 특정 동작이 언제 일어나는지를 식별하는 Surgical Action Localization(수술 동작 국지화) 문제를 해결하고자 한다. 이 작업은 수술 절차의 자동화된 교육, 수술 워크플로우 최적화 및 성찰적 보조 도구 제공이라는 측면에서 매우 중요하다.

그러나 수술 비디오 분석에는 다음과 같은 세 가지 주요 난제가 존재한다. 첫째, 개인정보 보호 문제로 인해 의료 데이터셋의 확보가 매우 어렵다. 둘째, 일반적인 자연 영상(natural videos)과 달리 수술 영상은 장면의 다양성과 구성이 매우 상이하여 기존의 일반 비디오 분석 모델을 그대로 적용하기에 한계가 있다. 셋째, 특히 신장 절제술(Nephrectomy)과 같은 복잡한 수술은 단계(phase)의 순서가 일정하지 않고 시간적 변동성이 커 예측이 매우 어렵다.

따라서 본 연구의 목표는 신장 절제술이라는 복잡한 수술 환경에 특화된 새로운 데이터셋인 UroSlice를 구축하고, 데이터 부족 문제와 시간적 복잡성을 해결할 수 있는 새로운 모델인 ViTALS를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 크게 두 가지로 요약할 수 있다.

첫째, 신장 절제술(부분 및 근치 신장 절제술)을 캡처한 새로운 복잡한 데이터셋인 **UroSlice**를 구축하였다. 이 데이터셋은 수술 단계가 균일하지 않고 체계적인 순서 없이 발생하여, 시간적 측면에서 상당한 변동성을 가지는 도전적인 데이터셋이다.

둘째, **ViTALS (Vision Transformer for Action Localization in Surgical Nephrectomy)** 모델을 제안하였다. 이 모델의 중심 아이디어는 Vision Transformer(ViT)가 가진 Inductive Bias의 부재로 인해 발생하는 소규모 데이터셋에서의 Overfitting 문제를 해결하기 위해, Local Connectivity Inductive Bias를 제공하는 **Dilated Temporal Convolution** 층을 통합하는 것이다. 또한, 계층적 구조의 Encoder-Decoder 설계를 통해 국소적(local) 특징과 전역적(global) 의존성을 동시에 캡처하여 예측을 정밀하게 정제한다.

## 📎 Related Works

기존의 수술 동작 분할(Surgical Action Segmentation) 연구는 크게 단일 단계(single-stage) 모델과 다단계(multi-stage) 모델로 나뉜다.

단일 단계 모델은 초기에는 HMM(Hidden Markov Models)이나 통계적 접근 방식을 사용하였으며, 딥러닝의 도입 이후에는 CNN 기반의 EndoNet이나 RNN-LSTM 기반의 SV-RCNet 등이 제안되었다. 최근에는 Trans-SVNet이나 ASFormer와 같이 Transformer 기반의 백본과 Causal Dilated Convolution을 결합하여 공간적, 시간적 특징을 추출하는 방식이 주를 이루고 있다.

기존 접근 방식의 한계점은 ViT 기반 모델들이 강력한 모델링 능력을 갖추고 있음에도 불구하고, 수술 영상과 같은 소규모 데이터셋에서는 Inductive Bias가 부족하여 학습이 어렵고 과적합되기 쉽다는 점이다. ViTALS는 이러한 한계를 극복하기 위해 시간적 합성곱 층을 도입하여 가설 공간(hypothetical space)을 제약함으로써 학습 효율을 높였다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

ViTALS는 크게 **Spatial Feature Extractor**와 **Action Segmentation Network (Encoder-Decoder)**의 두 단계로 구성된다.

### 2. Spatial Feature Extractor

수술 비디오는 매우 길기 때문에 GPU 메모리 제한을 극복하기 위해 사전 학습된 ResNet50을 사용하여 각 프레임을 공간적 특징 벡터로 변환한다. 긴 비디오 시퀀스에서 핵심적인 시간적 정보를 유지하면서 연산량을 줄이기 위해 다음과 같이 다운샘플링을 수행한다.

$$X' = (x_1, x_{1+w}, x_{1+2w}, \dots, x_n), \quad w = \frac{n}{15000}$$

추출된 특징 벡터 집합 $E = \{e_i\}_{i=1}^n$ (여기서 $e_i \in \mathbb{R}^d, d=2048$)은 이후 Action Segmentation Network의 입력으로 사용된다.

### 3. Action Segmentation Network

이 네트워크는 하나의 Encoder와 여러 개의 Decoder 블록으로 구성된다.

**Encoder:**

- **Dilated Temporal Convolution:** 우선 Dilated Convolution을 통해 국소적인 시간적 특징을 분석한다. 이는 Local Inductive Bias를 제공하여 소규모 데이터셋에서도 효과적인 학습을 가능하게 한다.
- **Self-Attention Layer:** 이후 Self-Attention 층을 통해 프레임 간의 전역적 상호작용을 학습한다.
- **Hierarchical Structure:** Dilation rate를 $d_w = 2^i$ 형태로 2배씩 증가시키는 계층적 구조를 채택하여, 낮은 층에서는 국소적 특징을, 높은 층에서는 전역적 정보를 캡처한다.
- **Temporal Fusion Head:** 모든 계층의 중간 특징들을 통합하여 초기 예측 값 $p_0 \in \mathbb{R}^{n \times K}$를 생성한다.

**Decoder:**

- Decoder는 Encoder와 유사한 계층적 구조를 가지지만, **Cross-Attention** 메커니즘을 포함한다.
- Query($Q$)는 이전 단계의 출력에서 가져오고, Key($K$)와 Value($V$)는 현재 층의 특징에서 가져옴으로써 초기 예측 값을 정밀하게 수정(refinement)한다.

### 4. 손실 함수 (Loss Functions)

모델의 최종 손실 함수 $L$은 Encoder의 손실과 $N$개 Decoder 단계의 손실의 합으로 정의된다.

$$L = L_{encoder} + \sum_{i=1}^{N} L_{decoder} \approx \sum_{i=1}^{N+1} (L_{ce} + \lambda L_{smooth})$$

여기서 $L_{ce}$는 예측 값과 정답 레이블 간의 차이를 줄이는 Cross-Entropy Loss이며, $L_{smooth}$는 과도한 분할(over-segmentation)을 방지하여 예측 결과를 부드럽게 만드는 Weighted Smooth Loss이다.

## 📊 Results

### 1. 실험 설정

- **데이터셋:** Cholec80(담낭 절제술) 및 UroSlice(신장 절제술)를 사용하였다. UroSlice는 39개의 비디오로 구성되었으며, 11개의 수술 단계(P1~P11)로 라벨링되었다.
- **지표:** 비디오 수준의 Accuracy(AC)와 시퀀스 수준의 Precision(PR), Recall(RE), Jaccard(JA) 지수를 사용하였다.
- **구현:** PyTorch 기반, ResNet50 백본, Adam Optimizer ($\text{lr}=5 \times 10^{-4}$)를 사용하였다.

### 2. 정량적 결과

ViTALS는 두 데이터셋 모두에서 SOTA(State-of-the-art) 성능을 달성하였다.

- **Cholec80:** Accuracy $89.8\%$를 기록하며 기존 모델들보다 약 $1.4\%$ 향상된 성능을 보였다.
- **UroSlice:** Accuracy $66.1\%$를 기록하였다. 특히 Trans-SVNet($62.2\%$)과 ASFormer($57.4\%$) 대비 각각 $3.9\%$, $8.7\%$ 높은 정확도를 보여, 복잡하고 작은 데이터셋에서 더 강력한 성능을 입증하였다.

### 3. 절제 연구 (Ablation Study)

- **특징 추출기 비교:** ResNet50을 사용했을 때 단순 Conv 층이나 Patch Embedding보다 정확도가 월등히 높았으며, GPU 메모리 효율성 또한 뛰어났다.
- **디코더 효과:** Encoder만 사용했을 때보다 Decoder를 추가했을 때 Cholec80 기준 정확도가 $84.8\% \rightarrow 89.8\%$로 크게 상승하여, 반복적인 정제 과정(Iterative Refinement)의 중요성을 확인하였다.

## 🧠 Insights & Discussion

본 연구는 수술 영상 분석에서 단순히 복잡한 Transformer 모델을 사용하는 것보다, **도메인 특성에 맞는 Inductive Bias(Dilated Convolution)를 주입하는 것이 소규모 데이터셋에서 훨씬 효과적임**을 보여주었다. 특히 신장 절제술과 같이 단계의 순서가 불규칙하고 시간적 변동성이 큰 작업에서 ViTALS의 계층적 구조와 Cross-Attention 기반의 정제 과정이 강점을 보였다.

다만, UroSlice 데이터셋의 경우 Cholec80에 비해 정확도가 상대적으로 낮게 나타났는데, 이는 데이터셋의 크기가 매우 작고 수술 단계 간의 시간적 편차가 극심하기 때문으로 분석된다. 향후 연구에서는 더 많은 데이터 확보나 데이터 증강 기법을 통해 이 격차를 줄일 필요가 있다.

## 📌 TL;DR

본 논문은 신장 절제술 수술 단계 인식을 위한 새로운 데이터셋 **UroSlice**와 모델 **ViTALS**를 제안한다. ViTALS는 ViT의 과적합 문제를 해결하기 위해 **Dilated Temporal Convolution**을 도입하고, **계층적 Encoder-Decoder** 구조를 통해 국소적-전역적 특징을 모두 포착하여 예측을 정밀하게 정제한다. 실험 결과, 특히 복잡한 워크플로우를 가진 소규모 데이터셋에서 기존 SOTA 모델들보다 뛰어난 강건함(Robustness)과 성능을 보였다.
