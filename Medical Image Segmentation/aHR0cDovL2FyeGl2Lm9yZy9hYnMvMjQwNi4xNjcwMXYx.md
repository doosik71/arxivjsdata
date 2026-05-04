# Demystifying the Effect of Receptive Field Size in U-Net Models for Medical Image Segmentation

Vincent Loos, Rohit Pardasani, Navchetan Awasthi (2024)

## 🧩 Problem to Solve

의료 영상 분할(Medical Image Segmentation) 분야에서 U-Net 아키텍처는 매우 널리 사용되고 있으나, 정작 네트워크의 수용 영역(Receptive Field, RF) 크기가 모델 성능에 미치는 영향에 대해서는 충분히 연구되지 않았다. 수용 영역은 CNN 내의 특정 뉴런이 입력 이미지에서 참조하는 영역을 의미하며, 이는 분할 대상인 관심 영역(Region of Interest, RoI)의 전역적 문맥(Global Context)을 파악하는 데 결정적인 역할을 한다.

본 논문의 목표는 U-Net 및 Attention U-Net 아키텍처에서 이론적 수용 영역(Theoretical Receptive Field, TRF)의 크기가 모델 성능, 계산 비용, 그리고 데이터셋의 특성(RoI의 크기 및 대비)과 어떤 관계를 갖는지 정량적으로 분석하는 것이다. 이를 통해 의료 영상 분할 작업에 최적화된 효율적인 네트워크 구조를 설계하기 위한 가이드라인을 제공하고자 한다.

## ✨ Key Contributions

본 연구의 핵심적인 기여는 다음과 같다.

1. **TRF 분석 체계 구축**: U-Net 및 Attention U-Net에서 RF 크기가 성능에 미치는 영향을 포괄적으로 분석하였으며, 특히 데이터 복잡도와 필요한 TRF 크기 사이의 상관관계를 밝혔다.
2. **TRF 계산을 위한 수학적 표기법 제안**: 4차원 텐서를 활용하여 네트워크의 임의의 레이어에서 TRF를 계산할 수 있는 수학적 프레임워크를 제안하였다.
3. **새로운 평가지표 제안**: TRF 대비 실제 기여도가 높은 픽셀의 비율을 측정하는 $\text{ERF rate}$와, TRF 크기 대비 분할 대상 객체의 상대적 크기를 측정하는 $\text{Object rate}$라는 두 가지 새로운 지표를 도입하였다.
4. **효율적 아키텍처 설계 인사이트**: 동일한 파라미터 수 내에서 TRF 크기만을 변경했을 때 성능 변화를 측정하여, 계산 비용과 성능 사이의 최적의 트레이드-오프(Trade-off) 지점이 존재함을 입증하였다.
5. **실용적 도구 제공**: U-Net 설정과 데이터셋에 따라 적절한 TRF 크기를 계산하고 추천하는 오픈소스 도구를 개발하였다.

## 📎 Related Works

기존 연구들에서도 RF 크기의 중요성이 일부 언급된 바 있다. 초음파 영상 분할 연구에서는 네트워크의 깊이나 파라미터 수보다 RF 크기가 더 결정적인 역할을 하며, RF 조절을 통해 얕은 네트워크로도 깊은 네트워크와 유사한 성능을 낼 수 있음을 시사하였다. 또한, 투과 전자 현미경(TEM) 이미지 분석 연구에서는 이미지 해상도와 대비(Contrast) 특성에 따라 RF의 영향력이 달라짐을 발견하였다.

그러나 기존 연구들은 특정 단일 데이터셋이나 표준 U-Net 구조에만 한정되어 분석이 이루어졌다는 한계가 있다. 본 논문은 이를 확장하여 다양한 의료 영상 모달리티(MRI, CT, X-Ray, Ultrasound)와 합성 데이터셋, 그리고 Attention U-Net 구조까지 포함하여 보다 일반화된 분석을 수행함으로써 차별점을 가진다.

## 🛠️ Methodology

### 전체 파이프라인 및 TRF 튜닝
본 연구에서는 네트워크 깊이($d$)와 합성곱 커널 크기($k$)를 조절하여 TRF 크기를 변경하였다. 이때, RF 크기 외의 변수를 통제하기 위해 출력 채널 수를 조정함으로써 모든 모델의 총 파라미터 수를 약 3,100만 개로 일정하게 유지하였다.

### TRF 계산 방법 (Theoretical Receptive Field)
TRF를 엄밀하게 정의하기 위해 입력 이미지 $X \in [0,1]^{h \times w}$에 대해 레이어 $d$에서의 TRF를 4차원 텐서 $T^{(d)} \in \mathbb{R}^{h \times w \times 2 \times 2}$로 표현한다. 여기서 마지막 두 차원은 각각 TRF의 좌상단(top-left)과 우하단(bottom-right) 좌표를 나타낸다.

1. **Convolution**: 패딩이 'same'인 경우, 커널 크기 $k$와 스트라이드 $s$를 고려하여 이전 레이어의 TRF 좌표를 업데이트한다.
2. **Max Pooling**: 윈도우 크기 $k$ 내에서 가장 넓은 영역을 선택하여 좌표를 확장한다.
3. **Upsampling**: 전치 합성곱(Transposed Convolution) 시 발생할 수 있는 중첩 영역을 처리하기 위해 반복적 방법(Algorithm 1)을 사용하여 TRF의 경계를 계산한다.
4. **Concatenations (Skip Connection)**: 인코더에서 온 특징 맵과 디코더의 특징 맵을 결합할 때, 두 TRF 중 더 넓은 영역(최소 좌상단, 최대 우하단)을 선택한다.
5. **Attention Gates**: $1 \times 1$ 합성곱과 요소별 덧셈/곱셈은 TRF 크기를 변화시키지 않으므로, 입력 특징과 게이팅 신호 중 더 큰 TRF 영역을 취한다.

결과적으로 U-Net과 Attention U-Net이 동일한 깊이와 커널 크기를 가진다면 두 모델의 TRF는 동일하다.

### ERF 및 제안 지표 (Effective Receptive Field & Metrics)
$\text{ERF}$는 출력 픽셀 $y_{h/2, w/2}$에 대한 입력 픽셀 $x_{i,j}$의 편미분 $\frac{\partial y_{h/2, w/2}}{\partial x_{i,j}}$을 통해 계산하며, 이는 실제로 결과에 영향을 주는 유효 영역을 의미한다.

- **ERF rate ($r$):** TRF 영역 내에서 임계값 $\epsilon$ 이상의 유의미한 기여를 하는 픽셀의 비율을 측정한다. 임계값 $\epsilon$은 커널 밀도 추정(KDE)을 통해 데이터의 분포에 따라 동적으로 결정한다.
  $$r = \frac{\sum_{y \in E} [|y| > \epsilon] \cdot (1 + |y|)}{m \cdot n}$$
- **Object rate ($\text{OR}$):** 객체를 감싸는 최소 사각형의 면적을 $\text{TRF}$ 면적으로 나눈 값이다.
  $$\text{OR} = \frac{(b-t) \cdot (r-l)}{\text{TRF}^2}$$

## 📊 Results

### 실험 설정
- **데이터셋**: 합성 데이터셋 8종(크기, 대비, 윤곽선 여부에 따라 구분) 및 실제 의료 데이터셋 6종(Fetal head, Kidneys, Lungs, Thyroid, Nerve)을 사용하였다.
- **평가 지표**: Dice Similarity Coefficient (DSC), Sensitivity, Specificity, Accuracy, Jaccard Index 및 제안한 ERF rate, Object rate를 사용하였다.

### 주요 결과
1. **대비(Contrast)의 영향**: Lungs와 같은 고대비(High-contrast) 데이터셋은 TRF 크기가 매우 작더라도 빠르게 최대 성능에 도달하였다. 반면, 저대비(Low-contrast) 데이터셋이나 윤곽선만 존재하는 데이터셋은 TRF 크기가 커질수록 DSC가 점진적으로 상승하는 경향을 보였다.
2. **최적 TRF 크기의 존재**: TRF가 너무 커지면 성능 향상은 정체(Plateau)되는 반면, 학습 시간(epochs)은 증가하는 경향이 확인되었다. 이는 불필요한 계산 비용이 발생함을 의미한다.
3. **Attention Mechanism의 효과**: Attention U-Net은 모든 TRF 크기에서 일반 U-Net보다 높은 절대적인 성능을 보였다. 하지만 Attention 구조를 사용하더라도 TRF 크기가 적절히 확보되어야 성능이 극대화된다는 점에서 TRF의 중요성은 여전하였다.
4. **Object Rate와의 관계**: $\text{Object rate}$가 높아질수록(즉, TRF 대비 객체 크기가 커질수록) DSC가 낮아지는 경향이 있으며, 성능이 포화되는 지점의 TRF 크기는 보통 RoI 크기보다 약간 작은 수준에서 형성되었다.

## 🧠 Insights & Discussion

본 연구는 수용 영역의 크기가 단순한 하이퍼파라미터가 아니라, 해결하고자 하는 문제의 **데이터 복잡도(대비 및 형태)**와 **대상 객체의 크기**에 직접적으로 연결되어 있음을 입증하였다.

- **강점**: 단순히 성능 지표만 제시한 것이 아니라, $\text{ERF rate}$와 $\text{Object rate}$라는 분석 도구를 통해 왜 특정 TRF 크기에서 성능이 변하는지를 정량적으로 설명하였다.
- **비판적 해석**: TRF가 증가함에 따라 전역적 문맥을 더 많이 반영할 수 있지만, 동시에 이미지 내의 불필요한 노이즈나 방해 요소(Distracting features)까지 포함하게 되어 DSC가 미세하게 하락하는 구간이 발생한다. 이는 모델의 변동성(Variability) 관점에서 해석될 수 있다.
- **한계**: 본 연구는 U-Net 계열에 집중되어 있으며, DeepLab이나 PSPNet과 같이 Dilated Convolution 등을 사용하여 RF를 능동적으로 조절하는 다른 최신 아키텍처들과의 비교 분석은 이루어지지 않았다.

## 📌 TL;DR

본 논문은 의료 영상 분할에서 U-Net의 **이론적 수용 영역(TRF) 크기가 성능에 미치는 영향**을 심층 분석하였다. 분석 결과, **고대비 데이터는 작은 TRF로도 충분하지만, 저대비나 복잡한 형태의 데이터는 더 큰 TRF가 필수적**임을 밝혔다. 또한, 무조건 큰 RF가 좋은 것이 아니라 **RoI 크기와 계산 효율성을 고려한 최적의 TRF 지점이 존재**하며, Attention 메커니즘은 TRF 크기와 무관하게 성능을 향상시킨다. 이 연구는 향후 의료 영상 분석 모델 설계 시 데이터셋의 특성에 맞는 최적의 RF 크기를 결정하는 수학적 근거와 도구를 제공한다는 점에서 가치가 크다.