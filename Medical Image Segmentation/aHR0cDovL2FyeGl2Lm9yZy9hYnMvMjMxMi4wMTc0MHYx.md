# MobileUtr: Revisiting the relationship between light-weight CNN and Transformer for efficient medical image segmentation

Fenghe Tang, Bingkun Nian, Jianrui Ding, Quan Quan, Jie Yang, Wei Liu, S.Kevin Zhou (2023)

## 🧩 Problem to Solve

본 논문은 의료 영상 분할(Medical Image Segmentation) 분야에서 모델의 경량화와 성능 유지 사이의 트레이드오프 문제를 해결하고자 한다. 의료 영상 분할은 정밀한 진단을 위해 매우 중요하지만, 최근의 Vision Transformer(ViT) 기반 모델들은 높은 계산 복잡도와 방대한 파라미터 수로 인해 자원이 제한된 모바일 기기나 실시간 진단 환경에서 사용하기 어렵다는 한계가 있다.

특히 의료 영상은 일반 영상에 비해 데이터의 희소성이 높고, 노이즈가 많으며, 경계가 모호한 특성을 가진다. 이러한 특성 때문에 ViT가 장거리 의존성(long-range representation)을 학습하는 데 어려움이 있으며, CNN이 가진 유용한 inductive bias를 효율적으로 활용하지 못하는 문제가 발생한다. 따라서 본 연구의 목표는 CNN의 효율적인 연산 능력과 ViT의 글로벌 문맥 추출 능력을 인프라 설계 수준에서 통합하여, 가볍지만 강력한 범용 의료 영상 분할 네트워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 CNN과 Transformer의 관계를 재정의하여, CNN을 단순한 전처리기나 병렬 구조로 사용하는 것이 아니라 Transformer의 구성 요소(Patch Embedding 등)를 대체하거나 보완하는 형태로 통합하는 것이다.

1. **ConvUtr 블록 설계**: Transformer의 MHSA(Multi-Head Self-Attention)와 FFN(Feed-Forward Network)의 구조적 유사성에 착안하여, Depthwise Convolution과 Inverted Bottleneck 구조를 가진 Transformer-like CNN 블록을 제안하였다. 이를 통해 ViT의 Patch Embedding 단계를 대체함으로써, 노이즈가 제거되고 압축된 시맨틱 정보를 Transformer에 전달한다.
2. **Adaptive Local-Global-Local (LGL) 블록**: CNN의 국소적 특징과 Transformer의 전역적 특징 사이의 원활한 전환을 위해 LGL 모듈을 도입하였다. 특히 데이터셋의 특성에 따라 수용 영역(Receptive Field)을 동적으로 조정하는 Adaptive 방식을 적용하여 정보 손실을 최소화하고 효율적인 정보 교환을 가능케 하였다.
3. **MobileUtr 아키텍처**: 위 요소들을 결합하여 U-자형 구조의 초경량 범용 의료 영상 분할 모델을 제안하였으며, 이는 기존 SOTA 모델 대비 파라미터 수를 획기적으로 줄이면서도 동등하거나 더 높은 성능을 달성하였다.

## 📎 Related Works

기존의 경량화 연구는 주로 CNN 기반의 MobileNetV2나 UNeXt와 같은 모델들이 주도해 왔으며, 이들은 높은 추론 효율성을 가지지만 CNN의 국소적 수용 영역 제한으로 인해 성능 향상에 한계가 있었다. 반면, ViT 기반의 모델들은 전역적 문맥 파악 능력이 뛰어나지만 계산 비용이 매우 높다.

CNN과 Transformer를 결합한 하이브리드 구조(예: TransUnet, Swin-Unet) 또한 제안되었으나, 대부분의 경우 성능 향상을 위해 방대한 파라미터를 사용하므로 실제 임상 환경의 모바일 기기 배포에는 부적합하였다. 또한 MobileViT나 EdgeViT와 같은 경량 ViT 모델들은 일반 자연 영상(Natural Image)에 최적화되어 있어, 데이터 특성이 다른 의료 영상에 그대로 적용했을 때 성능 저하가 심각하게 나타나는 한계가 있었다.

## 🛠️ Methodology

### 1. 전체 시스템 구조
MobileUtr은 전형적인 U-shape 아키텍처를 따른다. Encoder는 **ConvUtr $\rightarrow$ Adaptive LGL $\rightarrow$ Transformer** 순으로 구성되어 있으며, Decoder는 Progressive Cascade Upsampling과 Skip-connection을 통해 특징을 융합한다.

### 2. 주요 구성 요소 및 상세 설명

#### (1) ConvUtr (Transformer-like CNN)
ViT의 Patch Embedding을 대체하기 위해 설계되었으며, Transformer의 연산 흐름을 CNN으로 모방한다. 구체적인 연산 과정은 다음과 같다.

$$Y^l = \text{BN}(\sigma\{\text{DepthwiseConv}(X^l)\}) + X^l$$
$$Z^l = \text{BN}(\sigma\{\text{PointwiseConv}(Y^l)\})$$
$$X^{l+1} = \text{BN}(\sigma\{\text{PointwiseConv}(Z^l)\}) + Y^l$$

여기서 $\sigma$는 GELU 활성화 함수이며, BN은 Batch Normalization이다. Depthwise Convolution은 MHSA의 공간적 정보 추출을 대체하고, 두 번의 Pointwise Convolution(Inverted Bottleneck)은 FFN의 채널 간 정보 통합을 수행한다.

#### (2) Adaptive LGL Bottleneck
CNN의 Local 특징에서 Transformer의 Global 특징으로 넘어가는 과도기적 단계에서 정보 손실을 막기 위해 도입되었다. Local Aggregation $\rightarrow$ Global Sparse Attention $\rightarrow$ Local Propagation 순으로 동작한다.

본 논문에서는 수용 영역 $K$를 데이터셋의 특성에 맞게 사전 계산하여 적용하는 **Adaptive** 방식을 제안한다. 수용 영역 $K$는 다음과 같이 계산된다.

$$K = \frac{\bar{D}}{2^{n+1}}$$

여기서 $\bar{D}$는 데이터셋 $D$ 내 분할 대상 영역의 평균 직경이며, $2^n$은 ViT 층에 도달하기까지 수행된 다운샘플링 횟수이다. 이를 통해 관심 영역을 더 효과적으로 커버할 수 있다.

#### (3) Decoder 및 Skip-connection
Encoder의 저수준 특징(Low-level features)은 노이즈가 많으므로, 이를 Decoder에 직접 연결하지 않고 다운샘플링 연산을 거쳐 노이즈를 제거하고 수용 영역을 맞춘 뒤 융합한다. 업샘플링 과정에서는 Bilinear Interpolation과 $3 \times 3$ Convolution을 반복 사용하는 Progressive Cascade 방식을 채택하여 세밀한 디테일을 보존한다.

#### (4) 학습 절차 및 손실 함수
모델은 Binary Cross Entropy (BCE) 손실과 Dice Loss의 조합으로 학습된다.
$$\mathcal{L} = 0.5 \times \text{BCE}(\hat{y}, y) + \text{Dice}(\hat{y}, y)$$

## 📊 Results

### 1. 실험 설정
- **데이터셋**: CT(Synapse), Ultrasound(BUS, BUSI, TNSCUI), Dermoscopy(ISIC 2018) 등 3가지 모달리티의 5개 공공 데이터셋을 사용하였다.
- **비교 대상**: U-Net, nnUNet, TransUnet, Swin-Unet(무거운 모델), MobileViT, EdgeViT, RepViT(경량 자연영상 모델), UNeXt, MedT(경량 의료모델) 등 총 12개 모델과 비교하였다.
- **지표**: mIoU, Dice, F1 score, HD95(Hausdorff Distance)를 사용하였다.

### 2. 주요 결과
- **정량적 성과**: MobileUtr은 매우 적은 파라미터 수($1.39\text{M}$)로도 많은 경우 SOTA 수준의 성능을 보였다. 특히 TransUnet($105.32\text{M}$) 대비 파라미터를 약 10배 이상 줄이면서도 유사하거나 더 높은 성능을 달성하였다.
- **모달리티별 성능**:
    - **Ultrasound/Dermoscopy**: MobileUtr-L 모델이 nnUNet과 비교해 mIoU 기준 소폭 향상된 결과를 보였으며, 특히 경량 모델들 중에서는 압도적인 성능을 기록하였다.
    - **CT (Synapse)**: mIoU $69.09\%$, Dice $79.90\%$를 기록하며 TransUnet과 대등한 성능을 보였으나, 계산 비용은 극적으로 낮췄다.
- **효율성**: MobileUtr은 높은 FPS(Frames Per Second)와 낮은 GFLOPs를 기록하여 에지 디바이스 배포 가능성을 입증하였다.

## 🧠 Insights & Discussion

### 1. 강점 및 분석
본 논문은 단순히 두 구조를 섞은 것이 아니라, CNN의 구조를 Transformer의 설계 철학(Spatial-Channel mixing)으로 재해석하여 **ConvUtr**을 만든 점이 매우 영리한 접근이다. 또한, 의료 영상의 특성(노이즈 및 저해상도)을 고려하여 Convolutional Downsampling 대신 **Max-pooling**을 선택한 점이 실제 성능 향상에 기여했음을 Ablation Study를 통해 입증하였다.

### 2. 한계 및 비판적 해석
- **Adaptive LGL의 일반성**: 수용 영역 $K$를 계산할 때 데이터셋의 평균 직경 $\bar{D}$라는 사전 지식을 사용한다. 이는 데이터셋의 특성을 미리 알아야 한다는 전제가 필요하므로, 완전히 새로운 도메인의 데이터에 적용할 때 $\bar{D}$를 어떻게 설정할 것인지에 대한 자동화된 방법론이 부족하다.
- **범용성 주장**: 3가지 모달리티에서 좋은 성적을 거두었으나, 'Universal'이라는 표현을 쓰기에는 테스트 된 데이터셋의 범위가 제한적일 수 있다.

### 3. 결론적 논의
MobileUtr은 ViT의 전역 문맥 파악 능력을 유지하면서도 CNN의 효율성을 극대화한 성공적인 사례이다. 특히 경량화된 Patch Embedding(ConvUtr)이 Transformer 뒷단으로 전달되는 정보의 밀도를 높여, Transformer 자체를 가볍게 만들어도 성능이 유지될 수 있음을 보여주었다.

## 📌 TL;DR

본 논문은 CNN의 효율성과 ViT의 전역적 표현 능력을 통합한 초경량 의료 영상 분할 모델 **MobileUtr**을 제안한다. Transformer의 설계를 모방한 **ConvUtr** 블록을 통해 효율적인 Patch Embedding을 구현하고, 데이터 특성에 맞춘 **Adaptive LGL** 블록으로 국소-전역 정보 교환을 최적화하였다. 그 결과, 기존의 무거운 모델(TransUnet 등)보다 파라미터 수를 10배 이상 줄이면서도 SOTA 급의 성능을 달성하여, 실시간 의료 진단 및 모바일 기기 적용 가능성을 크게 높였다.