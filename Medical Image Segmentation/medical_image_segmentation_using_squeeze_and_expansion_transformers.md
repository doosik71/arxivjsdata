# Medical Image Segmentation Using Squeeze-and-Expansion Transformers

Shaohua Li, Xiuchao Sui, Xiangde Luo, Xinxing Xu, Yong Liu, Rick Goh (2021)

## 🧩 Problem to Solve

의료 영상 분할(Medical Image Segmentation)의 핵심은 이미지의 전체적인 맥락(Global Context)을 파악하는 동시에 미세한 세부 사항(Fine Details)을 유지하는 것이다. 즉, 높은 공간 해상도를 유지하면서도 넓은 수용 영역(Receptive Field)을 가진 특징을 학습해야 한다.

기존의 CNN 기반 모델인 U-Net 및 그 변형들은 인코더-디코더 구조와 멀티 스케일 특징 융합을 통해 이 문제를 해결하려 했다. 그러나 CNN의 특성상 층이 깊어질수록 먼 거리의 픽셀이 출력에 미치는 영향력이 급격히 감소하며, 이로 인해 이론적인 수용 영역보다 훨씬 작은 **Effective Receptive Field (ERF)**를 갖게 된다. 예를 들어, U-Net과 DeepLabV3+의 ERF는 약 90픽셀 수준에 불과하며, 이는 관심 영역(ROI)의 크기가 200픽셀을 넘는 경우가 많은 의료 영상 작업에서 국소적인 시각적 단서에 매몰되어 분할 오류를 일으키는 원인이 된다.

본 논문의 목표는 Transformer의 무제한적인 ERF 특성을 활용하여, 고해상도 특징 맵에서도 전역적인 맥락을 효과적으로 캡처할 수 있는 새로운 분할 프레임워크인 **Segtran**을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Transformer가 이미지 데이터의 특성에 더 적합하도록 설계한 **Squeeze-and-Expansion Transformer**와 이미지의 공간적 연속성을 부여하는 **Learnable Sinusoidal Positional Encoding**을 제안한 점이다.

중심적인 설계 아이디어는 다음과 같다.

1. **Squeezed Attention Block (SAB):** 거대한 어텐션 행렬($N \times N$)에서 발생하는 노이즈와 과적합 문제를 해결하기 위해, 유도 지점(Inducing points)을 도입하여 연산 복잡도와 메모리 사용량을 줄이고 정규화 효과를 얻는다.
2. **Expanded Attention Block (EAB):** 단일한 특징 변환만으로는 복잡한 데이터 변동성을 모델링하기 어렵다는 점에서, 여러 개의 모드(Mode)를 가진 Mixture-of-Experts 구조를 도입하여 표현력을 확장한다.
3. **Learnable Sinusoidal Positional Encoding:** 픽셀 간의 지역성과 의미적 연속성(Continuity Inductive Bias)을 부여하기 위해, 학습 가능한 파라미터가 포함된 사인/코사인 함수 기반의 위치 인코딩을 제안한다.

## 📎 Related Works

본 연구는 객체 탐지 모델인 DETR에서 영감을 받았으며, Transformer를 이용해 전역 맥락이 포함된 특징을 생성하는 방식을 채택하였다.

기존의 Transformer 기반 분할 모델인 SETR나 TransU-Net은 Vision Transformer(ViT)를 인코더로 사용하여 이미지 특징을 추출하고, 이후 간단한 CNN 디코더를 통해 마스크를 생성한다. 반면 Segtran은 CNN 백본을 통해 먼저 국소적인 시각 특징을 추출하고, 그 위에 Transformer 층을 쌓아 전역 맥락을 구축하며, Feature Pyramid Network(FPN)를 통해 해상도를 복원하는 하이브리드 구조를 취한다.

또한, 기존의 위치 인코딩 방식인 Fixed Sinusoidal Encoding은 적응성이 부족하고, Discretely Learned Encoding은 공간적 연속성을 강제하지 못한다는 한계가 있다. Segtran은 이를 보완하여 학습 가능하면서도 연속적인 형태의 인코딩을 제안함으로써 차별점을 둔다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

Segtran의 파이프라인은 다음과 같은 5가지 주요 구성 요소로 이루어진다.

- **CNN Backbone:** 이미지에서 풍부한 의미를 가진 시각 특징을 추출한다 (예: ResNet-101, EfficientNet-B4).
- **Input FPN:** 백본에서 나온 저해상도 특징 맵을 Transformer에 입력하기 전 적절한 해상도로 업샘플링한다.
- **Learnable Sinusoidal Positional Encoding:** 픽셀 좌표에 기반한 위치 정보를 추가한다.
- **Squeeze-and-Expansion Transformer Layers:** 특징 벡터들 간의 상호작용을 계산하여 전역 맥락을 부여한다.
- **Output FPN & Segmentation Head:** Transformer의 출력을 최종 해상도로 복원하고 픽셀 단위 분류를 수행한다.

### 2. Squeeze-and-Expansion Transformer

전통적인 Self-Attention은 모든 입력 단위 간의 관계를 계산하여 $N \times N$ 행렬을 생성하는데, $N$이 클 경우 과적합 위험이 크다. 이를 해결하기 위해 다음 두 블록을 제안한다.

**Squeezed Attention Block (SAB):**
외부 코드북에 저장된 $M$개의 학습 가능한 임베딩(Inducing points, $M \ll N$)을 사용한다.

1. 입력 $X$와 코드북 $C$ 사이의 어텐션을 통해 코드북 특징 $C'$를 생성한다: $C' = \text{Single-Head}(X, C)$.
2. 생성된 $C'$와 다시 입력 $X$ 사이의 어텐션을 수행하여 최종 출력을 얻는다.
이 과정을 통해 어텐션 행렬의 크기가 $N \times M$으로 축소되어 정규화 효과와 효율성을 동시에 얻는다.

**Expanded Attention Block (EAB):**
$N_m$개의 독립적인 단일 헤드 Transformer(모드)로 구성된 Mixture-of-Experts 모델이다.

1. 각 모드 $k$는 독립적인 출력 $X_{out}^{(k)}$를 생성한다.
2. 각 모드의 특징을 선형 변환하여 가중치 $B^{(k)}$를 구하고, Softmax를 통해 동적 모드 어텐션 $G$를 계산한다:
$$G = \text{softmax}(B^{(1)}, \dots, B^{(N_m)})$$
3. 최종 출력은 각 모드 출력의 가중 합으로 결정된다:
$$X_{out} = (X_{out}^{(1)}, \dots, X_{out}^{(N_m)}) \cdot G^T$$

또한, 이미지 단위 간의 관계는 대칭적이라는 가정하에 Query와 Key 프로젝션을 동일하게 묶어(Tied) 대칭적 어텐션을 구현하였다.

### 3. Learnable Sinusoidal Positional Encoding

이미지의 픽셀 지역성을 반영하기 위해, 좌표 $(x, y)$에 대해 다음과 같은 학습 가능한 위치 인코딩 벡터 $\text{pos}(x, y)$를 정의한다.
$$\text{pos}_i(x, y) =
\begin{cases}
\sin(a_i x + b_i y + c_i) & \text{if } i < C/2 \\
\cos(a_i x + b_i y + c_i) & \text{if } i \ge C/2
\end{cases}$$
여기서 $\{a_i, b_i, c_i\}$는 학습 가능한 가중치이다. 이는 좌표가 조금만 변해도 인코딩 값이 부드럽게 변하게 하여, 인접한 픽셀들이 유사한 위치 정보를 갖도록 하는 **Continuity Inductive Bias**를 제공한다.

## 📊 Results

### 1. 실험 설정
- **데이터셋:** REFUGE'20 (안저 이미지의 시신경 유두/컵 분할), Polyp (대장 내시경 용종 분할), BraTS'19 (MRI 뇌종양 분할).
- **평가 지표:** Dice Score.
- **비교 대상:** U-Net, U-Net++, U-Net3+, PraNet, DeepLabV3+, AttU-Net, nnU-Net, SETR, TransU-Net 등.

### 2. 주요 결과
- **정량적 성능:** 2D 및 3D 모든 작업에서 Segtran이 기존 방법론들보다 일관되게 높은 Dice Score를 기록하였다. 특히 EfficientNet-B4 백본을 사용했을 때 가장 우수한 성능을 보였다.
- **교차 도메인 일반화(Cross-Domain Generalization):** REFUGE20 데이터로 학습하고 완전히 다른 특성을 가진 RIM-One 데이터셋으로 테스트했을 때, U-Net이나 DeepLabV3+에 비해 성능 하락 폭이 가장 적어 강건함을 입증하였다.
- **전역 수용 영역 확인:** ERF 시각화 실험(Fig 2)을 통해 U-Net과 DeepLabV3+는 중심부 주변에만 그라디언트가 집중되는 반면, Segtran은 이미지 전체에 걸쳐 유의미한 그라디언트가 분포함을 확인하였다.

### 3. 효율성 분석
Transformer 기반 모델은 일반적으로 CNN보다 더 많은 메모리와 연산량(FLOPs)을 소모한다. 특히 ResNet 백본을 사용할 때 연산량이 급격히 증가하는 경향이 있으며, EfficientNet 백본을 사용할 때 파라미터 수와 FLOPs 측면에서 최적의 효율을 보였다.

## 🧠 Insights & Discussion

**강점:**
Segtran은 CNN의 국소 특징 추출 능력과 Transformer의 전역 맥락 파악 능력을 효과적으로 결합하였다. 특히 SAB를 통한 정규화와 EAB를 통한 표현력 확장은 Transformer가 의료 영상이라는 특수한 도메인에서 발생할 수 있는 과적합 문제를 완화하고 성능을 끌어올리는 데 기여하였다.

**한계 및 논의사항:**
1. **연산 비용:** 하이브리드 구조임에도 불구하고 Transformer의 특성상 메모리 사용량이 많아, 특히 3D 의료 영상 처리 시 RAM 부족 문제가 발생할 수 있다. 이를 해결하기 위해 3D 작업에서는 Transformer 층을 1개로 줄여 사용하였다.
2. **백본 의존성:** 실험 결과에서 나타나듯, 선택하는 CNN 백본에 따라 모델의 크기와 연산 효율이 크게 달라진다. 이는 Transformer 층 자체보다 이를 감싸는 FPN 및 백본 구조의 영향이 크기 때문이다.

**비판적 해석:**
논문은 전역 수용 영역(ERF)의 확장이 성능 향상의 핵심이라고 주장하며 이를 시각적으로 증명하였다. 그러나 Transformer 층의 개수 실험(Table 3)에서 3층까지는 성능이 오르다 4층부터 떨어지는 현상은, 전역 맥락의 확보뿐만 아니라 적절한 정규화와 모델 용량의 균형이 매우 중요하다는 것을 시사한다.

## 📌 TL;DR

본 논문은 CNN의 제한된 Effective Receptive Field 문제를 해결하기 위해, 효율적인 전역 맥락 학습이 가능한 **Squeeze-and-Expansion Transformer** 기반의 분할 모델 **Segtran**을 제안한다.

**주요 기여:**
- **SAB:** 유도 지점을 통해 어텐션 연산량을 줄이고 과적합을 방지한다.
- **EAB:** Mixture-of-Experts 구조로 다양한 데이터 변동성을 학습한다.
- **Learnable Sinusoidal PE:** 이미지의 공간적 연속성 바이어스를 학습 가능한 형태로 구현한다.

이 연구는 특히 전역적인 구조 파악이 중요한 의료 영상 분할 작업에서 CNN 기반 모델의 한계를 극복할 수 있는 실용적인 아키텍처를 제시하였으며, 향후 다양한 의료 영상 분석 작업에 적용될 가능성이 높다.
