# Demystify Mamba in Vision: A Linear Attention Perspective

Dongchen Han, Ziyi Wang, Zhuofan Xia, Yizeng Han, Yifan Pu, Chunjiang Ge, Jun Song, Shiji Song, Bo Zheng, Gao Huang (2024)

## 🧩 Problem to Solve

최근 Mamba와 같은 상태 공간 모델(State Space Model, SSM)은 선형 계산 복잡도를 가지면서도 높은 성능을 보여주며 컴퓨터 비전 분야에서 주목받고 있다. 반면, 유사하게 선형 복잡도를 구현하려 했던 Linear Attention Transformer들은 이론적인 효율성에도 불구하고 실제 성능이 기존의 Softmax Attention 기반 Transformer보다 현저히 떨어져 실용성이 낮다는 평가를 받아왔다.

본 논문은 **"왜 Mamba는 성공적이며, Linear Attention Transformer는 성능이 낮은가?"**라는 핵심 질문을 던진다. 연구의 목표는 Mamba와 Linear Attention의 수식적 유사성을 분석하여 Mamba의 성공 요인을 규명하고, 이를 바탕으로 Linear Attention의 한계를 극복한 새로운 아키텍처를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Mamba를 Linear Attention의 특수한 변형으로 재정의하고, 두 모델 사이의 결정적인 차이점을 이론적 및 실험적으로 분석한 점이다.

1.  **통합 공식화(Unified Formulation):** Mamba의 Selective SSM과 Linear Attention을 하나의 통합된 수식 체계 내에서 표현하여 두 모델의 관계를 명확히 정의하였다.
2.  **6가지 핵심 차이점 규명:** Mamba가 Linear Attention과 차별화되는 6가지 설계 요소(Input gate, Forget gate, Shortcut, No attention normalization, Single-head, Modified block design)를 도출하였다.
3.  **성능 기여도 분석:** 실험을 통해 **Forget gate**와 **Block design**이 Mamba의 성능 우위를 결정짓는 핵심 요소임을 밝혀냈으며, 특히 비전 작업에서는 Forget gate의 재귀적 계산을 병렬 가능한 Positional Encoding으로 대체할 수 있음을 증명하였다.
4.  **MILA 모델 제안:** Mamba의 핵심 장점(Forget gate의 효과와 개선된 블록 설계)을 Linear Attention에 이식한 **MILA (Mamba-Inspired Linear Attention)** 모델을 제안하여, 기존 Vision Mamba 모델들보다 우수한 성능과 더 빠른 추론 속도를 달성하였다.

## 📎 Related Works

### Vision Transformer와 Attention
Swin Transformer와 같은 모델들이 비전 작업에서 성공을 거두었으나, Softmax Attention의 $O(N^2)$ 복잡도는 고해상도 이미지 처리에서 큰 병목이 된다. 이를 해결하기 위해 Local window나 Sparsity를 도입하는 방식, 혹은 복잡도를 $O(N)$으로 낮춘 Linear Attention 방식이 제안되었다. 하지만 Linear Attention은 표현력(Expressive power) 부족으로 인해 실제 적용에 한계가 있었다.

### Mamba 및 Vision Mamba
Mamba는 선택적 상태 공간 모델(Selective SSM)을 통해 선형 복잡도와 효율적인 시퀀스 모델링을 동시에 달성하였다. 이를 비전에 적용한 VMamba, LocalMamba 등이 등장하였으나, 이들이 정확히 어떤 메커니즘을 통해 Linear Attention보다 우수한 성능을 내는지에 대한 심층적인 분석은 부족한 상태였다.

## 🛠️ Methodology

### 1. Mamba와 Linear Attention의 통합 관점
본 논문은 Mamba의 Selective SSM과 Linear Attention의 재귀적 형태(Recurrent form)가 매우 유사함을 보인다. 이를 통합 수식으로 표현하면 다음과 같다.

**Selective SSM (Mamba):**
$$h_i = e^{A_i} \odot h_{i-1} + B_i(\Delta_i \odot x_i)$$
$$y_i = C_i h_i + D \odot x_i$$

**Linear Attention:**
$$S_i = 1 \odot S_{i-1} + K_i^\top (1 \odot V_i)$$
$$y_i = \frac{Q_i S_i}{Q_i Z_i} + 0 \odot x_i$$

여기서 $h_i$는 $S_i$에, $B_i$는 $K_i^\top$에, $x_i$는 $V_i$에, $C_i$는 $Q_i$에 대응된다.

### 2. 6가지 주요 차이점 분석
논문은 위 수식을 바탕으로 Mamba가 가진 6가지 특이점을 분석한다.

*   **Input Gate ($\Delta_i$):** 입력값 $x_i$에 곱해져 어떤 정보를 은닉 상태(hidden state)에 저장할지 결정한다.
*   **Forget Gate ($e^{A_i}$):** 이전 상태 $h_{i-1}$을 얼마나 유지하거나 감쇠시킬지 결정하며, 이는 강한 로컬 바이어스(Local bias)와 위치 정보(Positional information)를 제공한다.
*   **Shortcut ($D \odot x_i$):** 입력에서 출력으로 직접 연결되는 경로를 제공하여 학습 안정성을 높인다.
*   **Normalization (부재):** Linear Attention은 가중치 합을 1로 만들기 위해 $Q_i Z_i$로 나누지만, Mamba는 이를 수행하지 않는다. (단, 논문은 분석을 통해 정규화가 학습 안정성에 중요하다고 주장한다.)
*   **Single-head:** Mamba는 구조적으로 단일 헤드 Linear Attention과 유사하며, Multi-head 설계를 사용하지 않는다.
*   **Modified Block Design:** Mamba는 단순한 Transformer 블록이 아닌, depth-wise convolution과 gating mechanism이 결합된 복잡한 구조를 사용한다.

### 3. MILA (Mamba-Inspired Linear Attention) 설계
분석 결과, **Forget gate**와 **Block design**이 가장 중요했다. 하지만 Forget gate의 재귀적 계산은 추론 속도를 늦추고 비전 모델(non-auto-regressive)에 부적합하다. 따라서 MILA는 다음과 같이 설계되었다.

*   **Forget Gate $\rightarrow$ Positional Encoding:** Forget gate가 제공하는 로컬 바이어스와 위치 정보를 제공하기 위해 **LePE, CPE, RoPE**와 같은 병렬 가능한 위치 인코딩을 도입하였다.
*   **Mamba Block Design 채택:** Mamba의 개선된 매크로 아키텍처를 Linear Attention에 적용하였다.
*   **Linear Attention 유지:** 병렬 계산이 가능하고 효율적인 Linear Attention 메커니즘을 기본 뼈대로 사용한다.

## 📊 Results

### 실험 설정
*   **데이터셋:** ImageNet-1K (분류), COCO (객체 탐지), ADE20K (시맨틱 세그멘테이션).
*   **비교 대상:** ConvNeXt, Swin Transformer, VMamba, LocalVMamba, Mamba2D 등 최신 CNN, ViT, Mamba 모델들.

### 주요 결과
1.  **ImageNet-1K 분류 성능:**
    *   MILA-T, S, B 모델 모두 대응하는 크기의 Vision Mamba 모델들을 능가하였다. 특히 **MILA-B는 Top-1 정확도 85.3%**를 달성하며 다른 모델들보다 유의미한 격차로 앞섰다.
2.  **고해상도 밀집 예측(Dense Prediction):**
    *   **COCO Object Detection:** MILA-B는 Mask R-CNN 1x 설정에서 $AP_b$ 50.5%를 기록하며 VMamba-B(49.2%)보다 우수한 성능을 보였다.
    *   **ADE20K Semantic Segmentation:** MILA-B는 mIoU 52.5% (Multi-scale)를 달성하여 SOTA 수준의 성능을 입증하였다.
3.  **추론 속도 (Inference Speed):**
    *   재귀적 계산을 제거하고 병렬 계산을 도입함으로써 속도가 획기적으로 향상되었다. RTX 3090 GPU 기준, **Mamba2D보다 4.5배, VMamba보다 1.5배 빠른 속도**를 기록하면서도 정확도는 더 높았다.

## 🧠 Insights & Discussion

### 강점 및 분석
본 논문은 단순히 새로운 모델을 제안한 것이 아니라, Mamba의 성공 요인을 Linear Attention의 관점에서 '해체(Demystify)'했다는 점에서 학술적 가치가 높다. 특히 Forget gate가 수행하는 역할이 사실상 위치 정보와 로컬 바이어스의 제공임을 밝히고, 이를 Positional Encoding으로 대체할 수 있음을 보인 점은 매우 통찰력 있다.

### 한계 및 논의사항
*   **구현 세부 사항:** 저자들은 Mamba와 Linear Attention 사이의 아주 작은 구현 차이가 더 존재할 수 있으며, 본 논문이 모든 차이점을 완전히 포괄하지는 못했을 수 있음을 명시하였다.
*   **정규화의 역설:** Mamba는 정규화를 하지 않음에도 성능이 좋지만, 실험 결과 Linear Attention에서는 정규화가 없으면 성능이 급락한다($77.6 \rightarrow 72.4$). 이는 Mamba의 다른 요소들(Forget gate 등)이 정규화의 부재로 인한 불안정성을 상쇄하고 있음을 시사한다.

## 📌 TL;DR

본 논문은 **Mamba를 '특수한 형태의 Linear Attention'으로 해석**하여 그 성공 요인이 **Forget gate(로컬 바이어스/위치 정보)**와 **개선된 블록 설계**에 있음을 밝혀냈다. 이를 기반으로 재귀적 계산을 제거하고 병렬 가능한 위치 인코딩을 결합한 **MILA** 모델을 제안하였으며, 이 모델은 **기존 Vision Mamba 모델들보다 더 높은 정확도와 월등히 빠른 추론 속도**를 동시에 달성하였다. 이는 향후 고해상도 이미지 처리를 위한 효율적인 백본 네트워크 설계에 중요한 방향성을 제시한다.