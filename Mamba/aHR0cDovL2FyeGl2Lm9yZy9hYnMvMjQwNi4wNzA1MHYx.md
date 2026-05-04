# DualMamba: A Lightweight Spectral-Spatial Mamba-Convolution Network for Hyperspectral Image Classification

Jiamu Sheng, Jingyi Zhou, Jiong Wang, Peng Ye, Jiayuan Fan (2024)

## 🧩 Problem to Solve

본 논문은 초분광 이미지(Hyperspectral Image, HSI) 분류에서 복잡한 분광-공간(spectral-spatial) 관계를 모델링할 때 발생하는 **효과성(effectiveness)과 효율성(efficiency) 사이의 트레이드오프** 문제를 해결하고자 한다.

HSI 분류는 수백 개의 분광 밴드를 통해 물질을 식별하는 작업으로, 정교한 모델링이 필수적이다. 기존의 CNN 기반 방법은 국부적 특징(local features) 추출에는 능숙하지만 전역적 문맥(global context) 파악에 한계가 있으며, Transformer 기반 방법은 Self-attention 메커니즘을 통해 전역적 의존성을 모델링할 수 있으나 계산 복잡도가 $O(N^2)$에 달해 메모리와 연산 비용이 매우 높다. 또한, 전역-국부 특징을 모두 잡기 위해 CNN과 Transformer를 순차적으로 결합한 cascading 구조의 경우, 한 단계의 모델링 과정에서 다른 단계의 문맥 정보가 소실되거나 전역-국부 특징의 구분이 명확하지 않은 문제가 존재한다.

따라서 본 연구의 목표는 전역적 특징과 국부적 특징을 동시에 효율적으로 추출할 수 있는 **경량화된 병렬 구조의 분광-공간 Mamba-Convolution 네트워크(DualMamba)**를 설계하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **"분할 정복(Divide and Conquer)"** 철학을 적용하여, 전역적 특징 추출을 위한 Mamba 스트림과 국부적 특징 추출을 위한 CNN 스트림을 병렬로 배치하는 것이다. 구체적인 기여 사항은 다음과 같다.

1.  **Dual-stream Hybrid Network:** Mamba와 경량 CNN을 병렬로 통합하여 전역-국부 분광-공간 관계를 효율적으로 모델링하는 구조를 제안하였다.
2.  **Cross-Attention Spectral-Spatial Mamba Module (CAS2MM):** 동적 위치 임베딩(Dynamic Positional Embedding)과 경량화된 공간/분광 Mamba 블록, 그리고 이들을 융합하는 Cross-attention 메커니즘을 통해 전역적 특징을 효율적으로 추출한다.
3.  **Lightweight Spectral-Spatial Residual Convolution Module (LS2RCMM):** 3D Convolution과 Depthwise Convolution을 활용한 병렬 브랜치를 통해 국부적 분광-공간 특징을 잔차 학습(residual learning) 방식으로 추출한다.
4.  **Adaptive Global-Local Fusion:** 전역 Mamba 특징과 국부 Convolution 특징의 가중치를 입력 데이터의 내용에 따라 동적으로 조절하여 최적의 분광-공간 표현을 학습하는 적응형 융합 모듈을 제안하였다.

## 📎 Related Works

### 1. HSI 분류를 위한 딥러닝 방법론
- **CNN 및 RNN:** CNN은 공간적 텍스처 패턴 인식에 강점이 있고, RNN(특히 Bi-LSTM)은 분광 밴드 간의 순차적 의존성을 파악하는 데 유리하다. 그러나 두 방법 모두 장거리 의존성(long-term dependency) 모델링에 한계가 있다.
- **Transformer:** Self-attention을 통해 전역적 문맥을 효과적으로 캡처하지만, 시퀀스 길이 $N$에 대해 $O(N^2)$의 계산 복잡도를 가져 자원 소모가 극심하다.
- **Hybrid Architecture:** CNN과 Transformer를 결합한 형태가 연구되었으나, 주로 cascading 구조를 취하므로 전역-국부 특징의 독립적인 변별력을 확보하기 어렵다.

### 2. State Space Model (SSM) 및 Mamba
- **SSM 및 S4:** 선형 복잡도로 시퀀스 데이터를 모델링할 수 있는 구조이다. S4는 효율적이지만 파라미터가 입력에 독립적인 정적 구조라는 한계가 있다.
- **Mamba:** 입력 의존적 파라미터화(input-dependent parametrization)와 선택적 스캔(selective scan) 메커니즘을 도입하여 Transformer 수준의 성능을 유지하면서 선형 시간 복잡도와 메모리 효율성을 달성하였다.
- **Vision Mamba (ViM, VMamba 등):** 2D 이미지 데이터를 시퀀스로 변환하기 위해 다방향 스캔(multi-directional scanning) 전략을 사용한다. 하지만 이러한 방식은 HSI의 풍부한 분광 정보를 충분히 활용하지 못하며, 다방향 스캔으로 인한 파라미터 및 FLOPs 증가가 심해 HSI 분류의 경량화 요구사항에 부합하지 않는다.

## 🛠️ Methodology

### 1. 기본 이론: State Space Model 및 Mamba
SSM은 입력 $x(t)$를 상태 변수 $h(t)$를 통해 출력 $y(t)$로 매핑한다.
$$\begin{aligned} h'(t) &= Ah(t) + Bx(t) \\ y(t) &= Ch(t) \end{aligned}$$
Mamba는 이를 이산화하고, 파라미터 $B, C, \Delta$를 입력 $x$의 함수로 만들어 선택적 스캔(S6 모델)을 수행함으로써 입력 데이터에 따라 동적으로 문맥을 파악한다.

### 2. Cross-Attention Spectral-Spatial Mamba Module (CAS2MM)
전역적 특징을 추출하기 위해 다음의 세부 구성 요소를 사용한다.

- **Dynamic Positional Embedding (DPE):** Mamba의 시퀀스 변환 과정에서 소실되는 위치 정보를 보완하기 위해 $3 \times 3$ Depthwise Convolution을 사용하여 입력에 의존적인 위치 임베딩을 생성한다.
  $$\text{DPE}(X) = \text{DWConv}_{3 \times 3}(X)$$
- **Lightweight Spatial Mamba Block:** 연산량을 줄이기 위해 Gated MLP와 대형 MLP를 제거하였다. 또한, HSI의 공간적 특징은 방향에 관계없이 유사하다는 점에 착안하여, 연산 비용이 높은 다방향 스캔 대신 **단방향 스캔(Unidirectional Scan)** 전략을 채택하여 공간 시퀀스를 생성하고 S6 모델에 입력한다.
- **Lightweight Spectral Mamba Block:** 중심 픽셀의 특징만을 추출하여 연산량을 줄였다. 분광 특성의 비대칭성을 고려하여 **양방향 스캔(Bidirectional Scan)**을 수행하며, 전방향과 역방향 시퀀스를 모두 학습한 뒤 이를 병합(Merge)하여 전역 분광 특징을 얻는다.
- **Cross-Attention Spectral-Spatial Fusion:** 추출된 전역 공간 특징 $G_{\text{spa}}$와 전역 분광 특징 $G_{\text{spe}}$를 Softmax 기반의 가중치 $A_{\text{spa}}, A_{\text{spe}}$를 이용해 융합한다.
  $$G = A_{\text{spe}} G_{\text{spa}} + A_{\text{spa}} G_{\text{spe}} + X_{\text{pos}}$$

### 3. Lightweight Spectral-Spatial Residual Convolution Module (LS2RCMM)
국부적 특징을 추출하기 위해 두 개의 병렬 브랜치를 운영한다.

- **Spectral Branch:** $1 \times 1 \times 3$ 크기의 경량 3D Convolution을 사용하여 인접 분광 밴드 간의 정보를 집계한다.
- **Spatial Branch:** $3 \times 3$ Depthwise Convolution을 사용하여 채널 간 혼합 없이 각 분광 채널의 국부적 공간 특징을 추출한다.
- **Fusion:** 두 브랜치의 결과물을 Concatenation한 후 Pointwise Convolution을 통해 원래 채널 수로 복원하며, 최종적으로 국부 분광-공간 특징 $L$을 산출한다.

### 4. Adaptive Global-Local Fusion
전역 특징 $G$와 국부 특징 $L$을 단순 합산하는 대신, 데이터의 특성에 따라 가중치를 동적으로 결정한다.
1. $G+L$ 특징을 Average Pooling하고 MLP를 통과시켜 컴팩트한 벡터 $z$를 생성한다.
2. Sigmoid 함수를 통해 전역 가중치 $W_g$와 국부 가중치 $W_l = 1 - W_g$를 계산한다.
3. 최종 표현 $F$는 다음과 같이 산출된다.
   $$F = G + L + W_g G + W_l L$$

## 📊 Results

### 1. 실험 설정
- **데이터셋:** Indian Pines, WHU-Hi-Longkou, Houston 2018의 세 가지 공인 데이터셋을 사용하였다.
- **비교 대상:** 2-D/3-D CNN, SSRN, AB-LSTM, MSRT, SpectralFormer(SF), SSFTT, GAHT 등 최신 SOTA 모델들과 비교하였다.
- **측정 지표:** Overall Accuracy (OA), Average Accuracy (AA), Kappa coefficient ($\kappa$)를 사용하였다.

### 2. 주요 결과
- **정량적 성능:** DualMamba는 세 데이터셋 모두에서 SOTA 성능을 달성하였다. 특히 Indian Pines 데이터셋에서 OA 99.23%를 기록하며 2위 모델(GAHT, 97.95%)을 크게 앞질렀다.
- **효율성:** 모델 파라미터 수와 FLOPs 측면에서 압도적인 우위를 보였다. SOTA 모델들보다 파라미터와 FLOPs를 1/10 수준 이하로 줄이면서도 더 높은 정확도를 기록하였다.
  - CNN 기반 대비: 파라미터 약 88% 감소, FLOPs 약 95% 감소.
  - Transformer 기반 대비: 파라미터 약 90% 감소, FLOPs 약 98% 감소.
- **데이터 효율성:** 훈련 샘플의 비율을 낮춘 실험에서도 다른 모델 대비 높은 OA를 유지하여, 적은 양의 데이터로도 효과적인 학습이 가능함을 증명하였다.

## 🧠 Insights & Discussion

### 1. 강점 및 분석
- **Parallel vs Cascading:** 본 연구는 전역-국부 모델링을 순차적으로 수행하는 기존 방식보다 병렬로 수행하고 적응형으로 융합하는 것이 특징 추출의 독립성을 보장하고 성능을 높인다는 점을 입증하였다.
- **Scan Strategy의 효율성:** 일반적인 Vision Mamba 모델들이 사용하는 다방향 스캔이 HSI에서는 중복적이며 비효율적임을 밝혀냈다. Nadir view(수직 뷰)에서 촬영된 HSI의 특성상 단방향 공간 스캔만으로도 충분한 전역 특징 추출이 가능하며, 이는 파라미터 수를 획기적으로 줄이는 핵심 요인이 되었다.
- **분광-공간의 상호 보완성:** Mamba의 전역 모델링 능력과 CNN의 국부 특징 추출 능력이 서로의 단점을 완벽히 보완하며, 특히 Adaptive Fusion이 데이터별 특성에 맞게 가중치를 조절함으로써 일반화 성능을 높였다.

### 2. 한계 및 논의사항
- **패치 크기 의존성:** 실험 결과, 데이터셋의 특성에 따라 최적의 패치 크기가 다르게 나타났다(Indian Pines는 7, Houston 2018은 15). 이는 모델이 전역 문맥을 잡는 능력은 뛰어나나, 입력 윈도우 크기에 따른 성능 편차가 존재함을 의미한다.
- **가정:** 본 논문은 HSI가 고도 센서의 Nadir view로 촬영되었다는 가정을 바탕으로 단방향 스캔의 정당성을 부여하였다. 만약 촬영 각도가 다른 데이터셋의 경우, 제안된 단방향 스캔이 유효할지에 대한 추가 검증이 필요하다.

## 📌 TL;DR

본 논문은 HSI 분류를 위해 전역적 문맥을 잡는 **Mamba**와 국부적 특징을 잡는 **경량 CNN**을 병렬로 결합한 **DualMamba** 네트워크를 제안하였다. 특히 HSI 특성에 맞춘 단방향 공간 스캔과 양방향 분광 스캔, 그리고 적응형 가중치 융합 메커니즘을 통해 **기존 SOTA 모델들보다 파라미터와 연산량을 90% 이상 획기적으로 줄이면서도 분류 정확도는 오히려 향상**시키는 성과를 거두었다. 이 연구는 자원이 제한된 에지 디바이스 환경에서의 고성능 HSI 분류 가능성을 제시하였다는 점에서 매우 중요하다.