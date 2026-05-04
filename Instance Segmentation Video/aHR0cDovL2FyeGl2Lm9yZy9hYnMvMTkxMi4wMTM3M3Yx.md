# Automatic Video Object Segmentation via Motion-Appearance-Stream Fusion and Instance-aware Segmentation

Sungkwon Choo, Wonkyo Seo, and Nam Ik Cho (2019)

## 🧩 Problem to Solve

비디오 객체 분할(Video Object Segmentation, VOS)은 비디오 내에서 관심 객체에 해당하는 픽셀을 찾아내는 작업으로, 시간적(Temporal) 정보와 공간적(Spatial) 정보 모두를 필요로 한다. 기존의 VOS 방법론은 크게 준지도 학습(Semi-supervised)과 비지도 학습(Unsupervised)으로 나뉜다. 준지도 학습은 첫 번째 프레임에서 사용자가 타겟 마스크를 제공하므로 성능이 높지만, 사용자 개입이 필요하다는 번거로움이 있다. 반면, 비지도 학습은 사용자 개입 없이 자동으로 객체를 검출하므로 편리하지만 성능이 상대적으로 낮다는 한계가 있다.

본 논문의 목표는 사용자 개입 없이도 준지도 학습 수준의 높은 성능을 낼 수 있는 자동 비디오 객체 분할(Automatic Video Object Segmentation) 알고리즘을 개발하는 것이다. 이를 위해 움직임(Motion)과 외형(Appearance) 정보를 효과적으로 융합하고, 인스턴스 수준의 정보를 결합하여 정확한 전경(Foreground) 마스크를 추출하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Motion stream**과 **Appearance stream**을 각각 독립적인 네트워크로 구성하여 장기적인 시간적 정보와 다중 스케일의 공간적 정보를 추출하고, 이를 **원래의 해상도(Original Resolution)**에서 융합하는 것이다. 또한, 픽셀 수준의 분할 결과에 **인스턴스 인식 분할(Instance-aware Segmentation)**을 결합하여 가양성(False Positive)을 제거하고 객체의 경계를 정밀하게 보정함으로써 비지도 학습 환경에서도 높은 정확도를 달성하였다.

## 📎 Related Works

### 1. 비디오 분할의 Recurrent Network
기존의 Recurrent Network 기반 VOS 연구들은 주로 사전 학습된 ResNet이나 VGG와 같은 얕은 네트워크를 사용하여 ConvGRU나 LSTM을 결합하였다. 그러나 이러한 방식은 수용 영역(Receptive Field)이 좁고 장기적인 시간적 의존성을 학습하는 데 한계가 있으며, 해상도 감소로 인해 세부적인 형태 정보를 손실하는 문제가 있다.

### 2. Two-stream Fusion
움직임과 외형 정보를 각각 추출하여 결합하는 Two-stream 방식은 주로 비디오 분류(Action Recognition 등)에서 사용되어 왔다. 기존의 융합 방식은 주로 네트워크 중간 단계의 저해상도 특징 맵에서 융합하는 Late Fusion 방식을 취했는데, 이는 계산 효율성은 높으나 정밀한 분할 경계를 얻는 데 방해가 된다.

### 3. Instance-level Video Segmentation
인스턴스 수준의 분할(예: Mask R-CNN)은 주로 준지도 학습에서 사용되었다. 비지도 학습에서는 어떤 인스턴스가 실제 타겟인지 판별할 기준(Ground Truth)이 없기 때문에 가양성 문제가 발생하여 적용이 어려웠다.

## 🛠️ Methodology

### 1. Motion Stream Network
움직임 스트림은 FlowNet 2.0으로 계산된 Optical Flow를 입력으로 받아 시간적 특징을 추출한다.

*   **Multiscale ConvGRU**: Hourglass 형태의 Encoder-Decoder 구조를 가지며, Skip Connection을 포함한다. 스트라이드(Stride) 2의 컨볼루션을 통해 스케일을 줄였다가 Bilinear Interpolation으로 복원하며 다양한 타임스케일의 정보를 학습한다.
*   **Asymmetric Convolution**: 계산량을 줄이고 수용 영역을 넓히기 위해 $1 \times k$와 $k \times 1$ 컨볼루션을 교차 사용하여 처리한다.
*   **수식 설명**:
    Layer Normalization($\text{LN}$)을 적용한 게이트 $g$ (reset gate $r$ 및 update gate $z$)는 다음과 같이 계산된다.
    $$g^s = \text{LN}([W^g_{s,k\times 1} * W^g_{s,1\times k} * s_t ; W^g_{s,1\times k} * W^g_{s,k\times 1} * s_t])$$
    $$g = \sigma(g_x + g_h)$$
    여기서 $s \in \{\text{input } x, \text{state } h\}$이며, 최종 상태 $h_t$는 다음과 같이 업데이트된다.
    $$\tilde{h} = \tanh(c_x + c_q)$$
    $$h_t = z \odot h_{t-1} + (1-z) \odot \tilde{h}$$
*   **Cascaded Bidirectional Network**: Forward 방향으로 연산한 결과와 Optical Flow 입력을 다시 결합하여 Backward 방향으로 연산하는 계층적 양방향 구조를 채택하여 더 긴 시간적 연결성을 확보한다.

### 2. Appearance Stream Network
외형 스트림은 RGB 프레임을 입력으로 하며, DeepLabv3+ 구조를 사용한다. PASCAL VOC 2012 데이터셋으로 사전 학습된 가중치를 사용하되, 비디오 데이터셋의 특성상 오버피팅이 발생하기 쉬우므로 **Decoder 부분의 가중치만 업데이트**하여 학습시킨다.

### 3. Original Resolution에서의 Two-stream Fusion
기존 방식과 달리, 각 스트림의 Decoder를 통해 특징 맵을 원래 해상도로 복원한 후 융합한다. 두 스트림의 결과물을 채널 축으로 결합(Concatenate)하고 $1 \times 1$ Convolution을 통해 픽셀 단위의 확률 맵(Pixel-level probabilistic foreground map)을 생성한다. 이를 통해 해상도 손실을 최소화하고 정확한 경계를 얻는다.

### 4. Instance-aware Segmentation
픽셀 수준의 결과만으로는 가양성 제거가 어렵기 때문에 Mask R-CNN 기반의 인스턴스 분할을 결합한다.

1.  **IoU 계산**: 각 검출된 인스턴스 마스크 $M^i_{obj}$와 픽셀 수준 분할 결과 $M_{pixel}$ 간의 IoU를 계산한다.
    $$\text{IoU}^i_{obj} = \frac{\sum_x M^i_{obj}(x) \cdot M_{pixel}(x)}{\sum_x \max(M^i_{obj}(x), \text{Bin}(M_{pixel}(x)))}$$
2.  **인스턴스 스코어 결정**: IoU와 Mask R-CNN의 Objectness score를 곱하여 최종 스코어를 산출한다.
    $$\text{Score}^i = \text{IoU}^i_{obj} \cdot \text{objectness}^i$$
3.  **결과 보정(Boosting)**: 가장 높은 스코어를 가진 인스턴스 영역 내의 픽셀 확률값을 해당 영역의 평균값 $\mu_{obj}$로 보강하고, 그 외 영역의 확률값은 절반으로 줄여 가양성을 억제한다.

## 📊 Results

### 1. 실험 설정
*   **데이터셋**: DAVIS(분할 성능 평가), FBMS-59(다중 객체 검출 견고성 평가)
*   **측정 지표**: $\text{J-mean}$ (영역 유사도), $\text{F-score}$ (윤곽선 정확도), $\text{T-mean}$ (시간적 안정성)
*   **비교 대상**: LVO, PDB, MoSal 등 최신 비지도 학습 및 일부 준지도 학습 방법론

### 2. 정량적 결과
*   **DAVIS 데이터셋**: 비지도 학습 방법론 중 가장 높은 성능을 보였으며, 기존 비지도 학습 베이스라인(LVO, PDB) 대비 $\text{J}$와 $\text{F}$ 지표에서 약 $3\% \text{p}$ 이상의 향상을 보였다. 특히 준지도 학습 방법론들과 비교해도 대등하거나 일부 능가하는 수준의 성능을 달성하였다.
*   **FBMS-59 데이터셋**: 다중 객체가 존재하는 환경에서도 $\text{J-mean}$ $78.3$, $\text{F-score}$ $85.1$을 기록하며 기존 비지도 학습 방법들을 상회하였다.
*   **런타임 분석**: Optical Flow 계산 및 인스턴스 제안 시간을 포함하여 프레임당 약 $300\text{ms}$가 소요되며, 이는 다른 최신 VOS 알고리즘들과 비교했을 때 매우 빠른 속도에 해당한다.

### 3. Ablation Study
*   **구조적 영향**: 단순 2D Conv보다 GRU가, GRU보다 양방향 계층 구조(Cas-Bi-GRU)가 성능이 높음을 확인하여 장기 시간적 의존성 학습의 중요성을 입증하였다.
*   **융합 시점**: 저해상도(Encoder) 융합보다 원래 해상도에서 융합했을 때 $\text{J\&F-mean}$ 성능이 향상됨을 확인하였다.
*   **인스턴스 인식**: 인스턴스 정보를 결합했을 때 단순 픽셀 수준 분할보다 약 $1.6\% \text{p}$ 성능이 향상되었으며, 특히 객체 검출이 부분적으로 누락되는 상황에서 강건함을 보였다.

## 🧠 Insights & Discussion

본 연구는 비지도 학습 기반의 비디오 객체 분할에서 고질적인 문제였던 성능 저하를 **고해상도 Two-stream 융합**과 **인스턴스 수준의 정제**라는 두 가지 전략으로 해결하였다. 특히, 단순한 픽셀 확률 값에 의존하지 않고 Objectness score와 IoU를 결합한 인스턴스 선택 전략은 비지도 학습에서도 가양성을 효과적으로 제거할 수 있는 실용적인 방법임을 보여주었다.

다만, 본 방법론은 외부 라이브러리인 FlowNet 2.0을 통해 Optical Flow를 사전에 계산해야 한다는 의존성이 있으며, 인스턴스 분할을 위해 Mask R-CNN이라는 무거운 네트워크를 추가로 사용한다는 점이 한계로 지적될 수 있다. 그럼에도 불구하고 런타임 분석 결과가 매우 효율적으로 나타난 점은 인상적이다.

## 📌 TL;DR

본 논문은 **Multiscale ConvGRU 기반의 Motion stream**과 **DeepLabv3+ 기반의 Appearance stream**을 **원래 해상도에서 융합**하고, 여기에 **Mask R-CNN의 인스턴스 정보**를 결합하여 사용자 개입 없이도 정밀한 비디오 객체 분할을 수행하는 알고리즘을 제안한다. 이 연구는 비지도 학습 기반 VOS의 성능을 준지도 학습 수준으로 끌어올렸으며, 특히 실시간성에 근접한 빠른 추론 속도를 확보하여 향후 자동 다중 객체 분할 연구에 중요한 기여를 할 것으로 보인다.