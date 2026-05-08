# Pyramid Medical Transformer for Medical Image Segmentation

Zhuangzhuang Zhang, Weixiong Zhang (n.d.)

## 🧩 Problem to Solve

본 논문은 의료 영상 분할(Medical Image Segmentation)에서 전역적 문맥(Global Context)과 지역적 세부 정보(Local Details)를 동시에 효율적으로 포착해야 하는 문제를 다룬다.

기존의 합성곱 신경망(CNN) 기반 방식은 층을 깊게 쌓거나 필터 크기를 키워 장거리 의존성(Long-range dependencies)을 모델링하려 하지만, 이는 효율성이 떨어지는 한계가 있다. 반면, 최근 도입된 Transformer 기반 모델들은 Self-attention 메커니즘을 통해 전역적 관계를 학습할 수 있으나, 두 가지 주요 문제점을 가진다. 첫째, 전체 이미지에 대해 Self-attention을 수행할 경우 연산 복잡도가 픽셀 수의 제곱에 비례하여 기하급수적으로 증가한다. 둘째, 연산량을 줄이기 위해 이미지를 고정된 크기의 패치(Patch)로 나누는 경향이 있는데, 이러한 경직된 분할 방식(Rigid partitioning scheme)은 객체가 패치 경계에서 잘리게 만들어 유용한 정보를 손실시키고, 다양한 크기와 모양의 의료 영상 객체를 강건하게 포착하지 못하게 한다.

결과적으로 본 논문의 목표는 연산 비용을 효율적으로 관리하면서도, 패치 기반 분할의 한계를 극복하고 다양한 스케일의 특징을 포착할 수 있는 새로운 의료 영상 분할 네트워크를 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **피라미드 구조의 네트워크 아키텍처**와 **적응형 분할 방식(Adaptive partitioning scheme)**을 결합하여 다중 스케일의 어텐션(Multi-scale attention)을 구현하는 것이다.

1. **Pyramid Architecture**: 세 개의 Transformer 브랜치(Short, Mid, Long-range)와 한 개의 CNN 브랜치로 구성된 피라미드 구조를 제안한다. 각 Transformer 브랜치는 서로 다른 해상도의 입력 영상을 처리하여 각각 지역적, 중간 범위, 전역적 관계를 효율적으로 포착한다.
2. **Gated Axial Attention 및 적응형 분할**: 고정된 패치 분할 대신, 각 픽셀을 중심으로 한 윈도우 내에서 Axial Attention을 계산하는 방식을 채택하여 정보 손실을 최소화하고 수용 영역(Receptive field)을 효율적으로 확장한다.
3. **다중 스케일 융합 및 심층 감독(Deep Supervision)**: CNN의 특징 추출 능력과 Transformer의 전역 관계 포착 능력을 Attention Gate를 통해 융합하며, 학습 과정에서 심층 감독을 도입하여 학습 안정성을 높이고 기울기 소실 문제를 완화한다.

## 📎 Related Works

의료 영상 분할은 초기에 Atlas-based, Statistical-based, Shape-based 방식에서 CNN 기반의 U-Net 및 그 변형 모델(Residual U-net, Dense U-net, U-Net++ 등)로 발전하였다. CNN은 국소적 특징 추출에는 뛰어나지만 장거리 의존성 포착에는 한계가 있으며, 이를 해결하기 위해 Atrous Convolution이나 Feature Pyramid 구조 등이 제안되었다.

최근에는 NLP 분야의 Transformer가 CV 분야로 확장되어 Vision Transformer(ViT) 등이 등장하였다. 하지만 ViT는 연산 복잡도가 높고 방대한 양의 사전 학습 데이터가 필요하다는 단점이 있다. 의료 영상 분야에서는 이러한 문제를 해결하기 위해 이미지를 패치로 나누거나(TransUnet, TransFuse), Axial Attention을 사용하는 방식(Medical Transformer)이 제안되었다. 그러나 본 논문은 이러한 기존 Transformer 기반 모델들이 채택한 '경직된 패치 분할' 방식이 의료 영상의 불규칙한 객체 모양을 포착하는 데 방해가 된다는 점을 지적하며 차별점을 둔다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

PMTrans는 총 4개의 브랜치로 구성된다.

- **Transformer Branches**: 입력 영상의 해상도를 $1$, $1/2$, $1/4$로 스케일링하여 각각 Short-range, Mid-range, Long-range 브랜치에 입력한다.
- **CNN Branch**: 입력 영상에서 저수준 문맥(Low-level context)을 추출하기 위해 shallow한 Residual Convolution 블록을 사용한다.

### 2. Gated Axial Attention

전체 픽셀 간의 Attention을 계산하는 대신, 높이($H$)와 너비($W$) 축을 따라 개별적으로 Attention을 수행하는 Axial Attention을 기반으로 한다.

기본적인 Self-attention은 다음과 같이 계산된다:
$$Q = XW_q, \quad K = XW_k, \quad V = XW_v$$
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Gated Axial Attention은 여기에 상대적 위치 인코딩(Relative position encodings)을 추가하고, 이를 제어하는 게이트(Gate) 변수를 도입하여 데이터셋이 적은 의료 영상 환경에서도 위치 정보를 유연하게 학습하게 한다. 수식으로 표현하면 다음과 같다:
$$y_{ij} = \sum_{k=1}^{W} \text{softmax}\left(q_{ij}^T k_{ik} + \mathbf{G}_q q_{ij}^T r_{ik}^q + \mathbf{G}_k r_{ik}^k \right) \left( \mathbf{G}_{v1} v_{ik} + \mathbf{G}_{v2} r_{ik}^v \right)$$
여기서 $\mathbf{G}_q, \mathbf{G}_k, \mathbf{G}_{v1}, \mathbf{G}_{v2}$는 위치 인코딩의 영향을 조절하는 학습 가능한 게이트이다.

### 3. 적응형 분할 및 수용 영역 (Adaptive Partitioning)

본 모델은 이미지를 고정된 그리드로 나누지 않고, 각 픽셀이 자신을 중심으로 한 특정 범위(Span) 내의 픽셀들만 참조하도록 한다.

- **Short-range branch**: 원본 해상도($H \times W$) 입력을 사용하며, $\frac{H}{4} \times \frac{W}{4}$ 범위의 지역적 문맥을 포착한다.
- **Mid-range branch**: $\frac{H}{2} \times \frac{W}{2}$ 해상도 입력을 사용하며, 원본 이미지 기준 $\frac{H}{2} \times \frac{W}{2}$ 범위의 관계를 포착한다.
- **Long-range branch**: $\frac{H}{4} \times \frac{W}{4}$ 해상도 입력을 사용하며, 동일한 Span을 사용하더라도 입력 해상도가 낮으므로 결과적으로 원본 이미지 전체($H \times W$)에 대한 전역적 어텐션을 효율적으로 수행하게 된다.

### 4. 융합 및 학습 절차

- **Fusion**: CNN 브랜치에서 추출된 특징 맵을 Transformer 브랜치의 특징 맵과 융합할 때 Attention U-net의 Attention Gate를 사용하여 중요한 정보를 선택적으로 통합한다.
- **Deep Supervision**: 학습 단계에서 $\frac{1}{2}$ 및 $\frac{1}{4}$ 스케일의 특징 맵 뒤에 보조 분류기(Auxiliary classifier)를 배치하여, 각 스케일의 특징 맵이 의미론적으로 변별력 있게 학습되도록 유도한다.
- **Loss Function**: Binary Cross-Entropy (BCE) 손실 함수를 사용하여 학습한다:
$$\ell_{bce}(\hat{y}, y) = -\sum_{i=1}^{H \times W} [y_i \log \hat{y}_i + (1-y_i) \log (1-\hat{y}_i)]$$

## 📊 Results

### 1. 실험 설정

- **데이터셋**: GLAS(선 조직 분할), MoNuSeg(핵 분할), HECKTOR(두경부 종양 분할)의 세 가지 데이터셋을 사용하였다.
- **평가 지표**: Dice Similarity Coefficient (DSC)를 사용하여 정량적 평가를 수행하였다.
- **비교 대상**: FCN, U-Net, U-Net++, Res-Unet 등 CNN 기반 모델과 Axial Attention U-net, Medical Transformer 등 어텐션 기반 모델들을 비교군으로 설정하였다.

### 2. 주요 결과

- **정량적 성능**: PMTrans는 모든 데이터셋에서 기존 baseline 모델들보다 높은 DSC를 기록하였다. 특히 Medical Transformer 대비 GLAS에서 0.57%, MoNuSeg에서 0.68%, HECKTOR에서 2.21% 향상된 성능을 보였다.
- **정성적 결과**: 시각화 결과, PMTrans는 다양한 크기와 모양의 객체에 대해 Ground Truth에 매우 근접한 정교한 윤곽선을 생성하는 것을 확인하였다.
- **Ablation Study**:
  - 세 가지 범위(Short, Mid, Long)의 브랜치를 모두 사용했을 때 성능이 가장 높았으며, 특히 GLAS와 같이 객체가 큰 데이터셋에서는 Long-range 브랜치의 기여도가 높았다.
  - 적응형 분할 방식이 기존의 단순 패치 분할(Trivial Partitioning) 방식보다 우수함을 입증하였다.
  - CNN 브랜치를 추가했을 때, 단순 Transformer만 사용했을 때보다 성능이 향상되어 CNN의 특징 추출 능력이 보완됨을 확인하였다.

## 🧠 Insights & Discussion

본 논문은 의료 영상 분할에서 Transformer의 전역 모델링 능력을 활용하면서도 연산 비용과 정보 손실이라는 두 마리 토끼를 잡기 위해 피라미드 구조와 적응형 분할을 제안하였다.

**강점**:

- **효율적인 전역 문맥 포착**: 입력 해상도를 단계적으로 낮춘 피라미드 구조를 통해, 연산량을 획기적으로 줄이면서도(Medical Transformer 대비 Long-range 브랜치의 어텐션 쌍이 $1/216$ 수준) 전역적 관계를 학습할 수 있게 설계되었다.
- **유연한 수용 영역**: 고정 패치가 아닌 픽셀 중심의 적응형 윈도우를 사용하여, 객체가 패치 경계에서 잘려 나가는 문제를 근본적으로 해결하였다.

**한계 및 논의**:

- 논문에서는 세 가지 데이터셋에 대해 우수한 성능을 보였으나, 사용된 데이터셋의 규모가 비교적 작다. 특히 MoNuSeg의 경우 학습 데이터가 30장 수준으로 매우 적어, 제안된 Gated Axial Attention이 소규모 데이터셋에서 얼마나 강건하게 작동하는지에 대한 더 광범위한 검증이 필요할 수 있다.
- 또한, 다양한 해상도의 브랜치를 병렬로 운영하므로 메모리 점유율이 증가할 가능성이 있으며, 이에 대한 상세한 메모리 분석은 명시되지 않았다.

## 📌 TL;DR

이 논문은 의료 영상 분할 시 발생하는 연산 비용 문제와 고정 패치 분할로 인한 정보 손실 문제를 해결하기 위해, 세 가지 해상도의 Transformer 브랜치와 한 개의 CNN 브랜치를 결합한 **Pyramid Medical Transformer (PMTrans)**를 제안한다. 적응형 분할 방식의 Gated Axial Attention을 통해 전역적/지역적 특징을 효율적으로 포착하며, 실험을 통해 기존 CNN 및 Transformer 기반 모델보다 우수한 분할 성능을 입증하였다. 이 연구는 의료 영상뿐만 아니라 다양한 스케일의 객체 포착이 필요한 일반 영상 분할 분야에도 적용 가능성이 높다.
