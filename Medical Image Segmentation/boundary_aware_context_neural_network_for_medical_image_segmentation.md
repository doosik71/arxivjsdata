# Boundary-aware Context Neural Network for Medical Image Segmentation

Ruxin Wang, Shuyuan Chen, Chaojie Ji, Jianping Fan and Ye Li (2020)

## 🧩 Problem to Solve

본 논문은 의료 영상 분할(Medical Image Segmentation)에서 객체의 경계선(Boundary)을 정확하게 예측하지 못해 분할 마스크의 품질이 저하되는 문제를 해결하고자 한다. 의료 영상은 클래스 내 변동성이 크고, 클래스 간 구분이 모호하며, 노이즈가 많이 포함되어 있다는 특성이 있다. 기존의 Convolutional Neural Networks (CNN) 기반 방법론들은 연속적인 Pooling 및 Convolution 연산을 거치면서 세밀한 공간 정보와 식별력 있는 특징 맵(Feature Map)을 손실하는 경향이 있으며, 이로 인해 제한적인 컨텍스트 정보만을 활용하게 되어 불만족스러운 경계선 결과를 생성한다.

따라서 본 연구의 목표는 더 풍부한 컨텍스트 정보를 캡처하고 세밀한 공간 정보를 보존함으로써, 정밀한 경계 인식 능력을 갖춘 2D 의료 영상 분할 네트워크인 BA-Net(Boundary-aware Context Neural Network)을 설계하는 것이다.

## ✨ Key Contributions

BA-Net의 핵심 아이디어는 인코더(Encoder)의 각 단계에서 경계선 정보와 분할 정보를 동시에 학습하고, 서로 다른 레벨의 특징들을 선택적으로 융합하여 컨텍스트를 강화하는 것이다. 이를 위해 다음과 같은 세 가지 핵심 모듈을 제안한다.

1. **Pyramid Edge Extraction (PEE) 모듈**: 다양한 입도(Granularity)의 경계선 특징을 추출하여 초기 단계에서 경계 정보를 강화한다.
2. **Mini Multi-Task Learning (mini-MTL) 모듈**: 객체 마스크 분할과 병변 경계 검출이라는 두 가지 작업을 동시에 학습하며, **Interactive Attention (IA)** 메커니즘을 통해 두 작업 간의 상호 보완적인 정보를 교환한다.
3. **Cross Feature Fusion (CFF) 모듈**: 인코더 전체 네트워크에서 서로 다른 레벨의 특징들을 선택적으로 집계하여, 고수준의 시맨틱 정보와 저수준의 세밀한 공간 정보를 동시에 확보한다.

## 📎 Related Works

기존의 의료 영상 분할 방식은 크게 세 가지 전통적 방법과 최근의 CNN 기반 방법으로 나뉜다.

1. **전통적 방법**: Gray-level 기반, Texture 기반, Atlas 기반 방법들이 존재한다. 이러한 방식들은 픽셀 및 영역 특징을 추출하여 성능을 높였으나, 수작업으로 설계된 저수준 특징(Hand-crafted features)에 의존하므로 복잡한 시나리오에서 성능이 제한적이며, 영상의 아티팩트나 강도 불균일성(Intensity inhomogeneity)에 취약하다는 한계가 있다.
2. **CNN 기반 방법**: U-Net과 FCN으로 대표되는 Encoder-Decoder 구조가 주류를 이루고 있다. 최근에는 Atrous Spatial Pyramid Pooling (ASPP)을 이용한 DeepLab이나 다양한 Pyramid Pooling 모듈을 통해 멀티 스케일 컨텍스트를 캡처하려는 시도가 있었다.

본 논문은 기존 CNN 방법론들이 여전히 경계선 예측에서 어려움을 겪는다는 점에 주목하며, 단순히 특징을 추출하는 것을 넘어 경계선 검출이라는 보조 작업을 통해 분할 성능을 가이드하는 전략을 취함으로써 차별점을 둔다.

## 🛠️ Methodology

### 전체 시스템 구조

BA-Net은 ResNet-101을 백본(Backbone)으로 사용하는 Encoder-Decoder 구조이다. 인코더에서는 ResNet의 마지막 Global Pooling과 Fully-connected 레이어를 제거하고, 수용 영역(Receptive Field)을 넓히기 위해 마지막 두 블록에 Atrous Convolution(rate=2)을 적용한다. 인코더의 각 단계는 **PEE $\rightarrow$ mini-MTL $\rightarrow$ CFF** 순으로 구성된 세 가지 모듈을 통과하며, 최종적으로 ASPP 모듈을 거친 특징 맵이 디코더로 전달되어 최종 분할 마스크를 생성한다.

### 주요 구성 요소 및 상세 설명

#### 1. Pyramid Edge Extraction (PEE) 모듈

경계선 정보의 다양성을 확보하기 위해 피라미드 구조의 특징 추출 방식을 사용한다. 우선 $1 \times 1$ Convolution을 통해 특징 맵 $F_i$의 채널을 축소하여 $F'_i$를 생성한다. 이후 다양한 크기($s$)의 Average Pooling 결과와 지역 컨볼루션 특징 맵의 차이를 계산하여 서로 다른 입도의 경계 특징을 얻는다.

$$F_{i,P}^{(s)} = F'_i - \text{avg}_s(F'_i), \quad s \in \{1, \dots, S\}$$

이렇게 얻어진 여러 입도의 경계 특징들을 $F'_i$와 함께 Concatenation하고, 다시 $1 \times 1$ Convolution을 통해 통합된 경계 특징 맵 $F_{i,P}$를 생성한다.

#### 2. Mini Multi-Task Learning (mini-MTL) 모듈

분할(Segmentation)과 경계 검출(Edge Detection)의 상호 보완성을 활용하기 위해 두 개의 서브 네트워크를 병렬로 구성한다. 각 브랜치는 두 개의 Convolution 레이어와 하나의 Upsampling 레이어로 이루어져 있다.

특히, 첫 번째 Convolution 레이어 이후에 **Interactive Attention (IA)**을 적용하여 두 작업 간의 정보를 교환한다. IA는 Sigmoid 함수를 통해 가중치 마스크를 생성하고, 이를 이용해 상대 작업의 특징 맵에서 유용한 정보만을 선택적으로 가져오는 Gated 메커니즘이다.

$$\text{Edge Branch: } F_{i,E}^{(1)} = F_{i,E}^{(1)} + (1 - \sigma(F_{i,E}^{(1)})) \otimes F_{i,S}^{(1)}$$
$$\text{Seg Branch: } F_{i,S}^{(1)} = F_{i,S}^{(1)} + (1 - \sigma(F_{i,S}^{(1)})) \otimes F_{i,E}^{(1)}$$

여기서 $\sigma$는 Sigmoid 함수이며 $\otimes$는 Element-wise product이다. 두 브랜치의 최종 결과물은 다시 통합되어 $F_{i,M}$이 된다.

#### 3. Cross Feature Fusion (CFF) 모듈

인코더의 서로 다른 단계에서 생성된 특징 맵들 간의 상호 보완성을 극대화하기 위해, 현재 단계의 특징 $F_{i,M}$과 다른 단계의 특징 $F_{j,M}$을 Attention 메커니즘으로 융합한다.

$$F_{i,C} = F_{i,M} + (1 - \sigma(F_{i,M})) \otimes \left[ \sum_{j \neq i} \sigma(F_{j,M}) \otimes F_{j,M} \right]$$

이 과정을 통해 저수준의 세부 공간 정보와 고수준의 시맨틱 정보가 적응적으로 통합된 $F_{i,C}$가 생성된다.

### 학습 절차 및 손실 함수

네트워크는 End-to-End로 학습되며, 최종 디코더의 손실 함수($L_D$)와 각 단계의 mini-MTL 모듈에서 발생하는 경계선 손실($L_{i,E}$) 및 분할 손실($L_{i,S}$)을 함께 최소화한다. 모든 손실 함수는 Binary Cross-Entropy (BCE)를 사용한다.

$$\min L_D + \sum_i \lambda_i (L_{i,E} + L_{i,S})$$

여기서 $\lambda_i$는 밸런스 파라미터로 1.0으로 설정되었다.

## 📊 Results

### 실험 설정

- **데이터셋**: ISIC-2017(피부 병변), Kvasir-SEG 및 CVC-ColonDB(폴립), SZ-CXR(폐), RIM-ONE-R1(시신경 유두) 등 총 5개의 공공 의료 데이터셋을 사용하였다.
- **비교 대상**: FCN, U-Net, MultiResUNet, AG-Net, CE-Net, DeepLabv3 등 최신 SOTA 모델들과 비교하였다.
- **측정 지표**: Dice Similarity Coefficient (DI), Jaccard Index (JA), Accuracy (AC), Sensitivity (SE), Specificity (SP)를 사용하였으며, 특히 JA를 주요 지표로 삼았다.

### 주요 결과

- **정량적 결과**: 모든 데이터셋에서 BA-Net이 가장 우수한 성능을 보였다. 특히 ISIC-2017 데이터셋에서 JA 기준 CE-Net(78.5%) 대비 81.0%로 크게 향상되었으며, Kvasir-SEG와 CVC-ColonDB에서도 SOTA 모델들을 상회하는 결과를 얻었다.
- **정성적 결과**: 시각화 분석 결과, 타 모델들이 경계선이 모호하거나 배경 노이즈를 객체로 인식하는 반면, BA-Net은 배경 억제 능력이 뛰어나고 실제 객체의 경계를 매우 정밀하게 추적함을 확인하였다.
- **Ablation Study**:
  - PEE, mini-MTL, CFF 모듈을 각각 제거했을 때 JA 지수가 하락함을 확인하여, 세 모듈의 결합이 성능 향상에 필수적임을 입증하였다.
  - 특히 mini-MTL 내의 Interactive Attention(IA)을 제거했을 때 성능이 유의미하게 하락하여, 두 작업 간의 정보 교환이 중요함을 확인하였다.

## 🧠 Insights & Discussion

본 논문은 의료 영상 분할의 고질적인 문제인 '경계선 불확실성'을 해결하기 위해 경계선 검출이라는 보조 작업을 네트워크 내부로 통합한 점이 매우 효과적이었다고 판단된다. 단순히 손실 함수에 경계선을 추가하는 것에 그치지 않고, PEE를 통한 다중 입도 특징 추출과 IA를 통한 특징 수준의 상호작용을 설계하여 실제 특징 맵 자체가 경계 인식 능력을 갖추도록 유도한 점이 강점이다.

또한, CFF 모듈을 통해 인코더의 서로 다른 층간 정보를 선택적으로 융합함으로써, 의료 영상 특유의 낮은 대비(Low contrast) 상황에서도 강건한 컨텍스트를 유지할 수 있었다.

다만, 본 연구는 2D 영상 분할에 집중되어 있으며, 실제 임상에서 많이 사용되는 3D 의료 영상(CT, MRI 등)에 대한 확장 가능성과 그에 따른 계산 복잡도 증가 문제는 논문 내에서 깊게 다뤄지지 않았다. 저자들 역시 향후 연구 과제로 3D 확장성을 언급하고 있다.

## 📌 TL;DR

본 논문은 의료 영상 분할 시 발생하는 부정확한 경계선 문제를 해결하기 위해 **Boundary-aware Context Neural Network (BA-Net)**를 제안한다. 이 네트워크는 **Pyramid Edge Extraction (PEE)**, **Mini Multi-Task Learning (mini-MTL)**, **Cross Feature Fusion (CFF)** 모듈을 통해 다중 입도의 경계 정보를 추출하고, 분할-경계 작업 간의 상호작용을 유도하며, 서로 다른 레벨의 특징을 적응적으로 융합한다. 5개의 다양한 의료 데이터셋 실험을 통해 기존 SOTA 모델들을 압도하는 성능을 입증하였으며, 이는 특히 정밀한 병변 경계 추출이 필요한 의료 진단 AI 분야에 크게 기여할 수 있을 것으로 보인다.
