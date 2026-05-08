# Concurrent Segmentation and Localization for Tracking of Surgical Instruments

Iro Laina, Nicola Rieke, Christian Rupprecht, Josué Page Vizcaíno, Abouzar Eslami, Federico Tombari, and Nassir Navab (2017)

## 🧩 Problem to Solve

본 논문은 최소 침습 수술(Minimally Invasive Surgery, MIS) 및 망막 미세수술(Retinal Microsurgery, RM) 환경에서 수술 도구의 실시간 추적(Real-time tracking) 문제를 해결하고자 한다. 수술 도구의 세그멘테이션(Segmentation)과 위치 추정(Localization)은 수술 중 추가 정보의 그래픽 오버레이 제공, 수술 워크플로우 분석, 그리고 망막과의 거리 측정과 같은 컴퓨터 보조 수술 시스템의 핵심적인 기능이다.

그러나 실제 수술 환경(in-vivo)에서는 강한 조명 변화, 거울 반사(Specular reflections), 그리고 도구의 빠른 움직임으로 인한 모션 블러(Motion blur)와 같은 요소들이 발생하여 정확한 추적을 방해한다. 기존의 마커 기반 방식은 수술 흐름을 방해하거나 도구의 수정이 필요하다는 단점이 있어, 본 연구는 마커가 없는(Marker-free) 비전 기반의 강건한 추적 방법론을 제안하는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 수술 도구의 세그멘테이션과 랜드마크 위치 추정 사이의 상호 의존성(Interdependency)을 활용하여 두 작업을 동시에 수행하는 것이다. 주요 기여 사항은 다음과 같다.

- **동시 학습 프레임워크(CSL) 제안**: 세그멘테이션과 포즈 추정(Pose estimation)을 별도의 파이프라인 단계로 처리하지 않고, 단일 딥러닝 모델 내에서 동시에 수행하는 Concurrent Segmentation and Localization (CSL) 구조를 제안하였다.
- **Heatmap Regression으로의 문제 재정의**: 랜드마크의 2D 좌표를 직접 회귀(Direct regression)하는 대신, 각 픽셀이 정답 위치에 가까울수록 높은 신뢰도를 갖는 히트맵(Heatmap)을 예측하도록 문제를 재정의하였다. 이를 통해 세그멘테이션 맵과 동일한 차원으로 표현이 가능해져 두 작업 간의 공간적 의존성을 효율적으로 학습할 수 있게 되었다.
- **심층 신경망 구조 최적화**: Fully Convolutional Residual Networks (FCRN)를 기반으로 하며, 인코더에서 디코더로 이어지는 Long-range skip connection을 도입하여 저수준의 고해상도 특징 정보를 보존함으로써 위치 추정의 정확도를 높였다.

## 📎 Related Works

기존의 수술 도구 추적 방식은 크게 세 가지 접근법으로 나뉜다.

1. **수작업 특징 기반 방식(Handcrafted features)**: Haar wavelets, Gradient, Color 특징 등을 사용한다. Color 특징은 연산 비용이 낮지만 조명 변화에 취약하고, Gradient 특징은 모션 블러에 취약하다는 한계가 있다.
2. **영역 제안 방식(Region proposals)**: 딥러닝을 통해 도구의 바운딩 박스(Bounding box)를 검출하지만, 랜드마크의 정밀한 위치를 추정하는 데는 한계가 있다.
3. **단계적 방식(Two-step methods)**: 먼저 세그멘테이션을 수행한 후, 그 결과로부터 위치를 추정하는 방식이다. 이러한 방식은 세그멘테이션이 전처리 혹은 후처리에 사용될 수 있음을 보여주며, 두 작업이 서로 밀접하게 연관되어 있음을 시사한다.

본 논문은 이러한 단계적 접근을 넘어 두 작업을 통합된 네트워크에서 동시에 학습함으로써 상호 보완적인 효과를 얻고 성능을 향상시켰다는 점에서 차별점을 갖는다.

## 🛠️ Methodology

### 전체 시스템 구조

본 모델은 ResNet-50을 인코더(Encoder)로 사용하고, 그 뒤에 세그멘테이션과 랜드마크 위치 추정을 위한 디코더(Decoder)가 연결된 구조이다. 입력 이미지는 $480 \times 480$ 픽셀 크기로 조정되며, 최종적으로 세그멘테이션 맵과 각 랜드마크별 히트맵이 출력된다.

### 세부 구성 요소 및 학습 절차

연구진은 제안하는 CSL 모델의 우수성을 증명하기 위해 세 가지 모델 변형을 비교 분석하였다.

1. **Localization (L)**: 랜드마크의 2D 좌표 $\tilde{y} \in \mathbb{R}^{2 \times n}$를 직접 예측하며, $L^2$ 손실 함수 $\ell_L(\tilde{y}, y) = \|\tilde{y} - y\|_2^2$를 사용한다.
2. **Segmentation and Localization (SL)**: 세그멘테이션과 좌표 회귀를 동시에 수행하되, 두 경로가 인코더만 공유하고 디코더에서는 분리된다. 손실 함수는 $\ell_{SL} = \lambda_L \ell_L + \ell_S$로 정의된다.
3. **Concurrent Segmentation and Localization (CSL)**: 제안 방법으로, 좌표 대신 가우시안 커널이 적용된 히트맵을 예측한다. 세그멘테이션 스코어를 위치 히트맵의 보조 정보로 활용하기 위해 마지막 단계에서 결합(Concatenation)한다.

### 주요 방정식 및 손실 함수

세그멘테이션을 위한 손실 함수 $\ell_S$는 픽셀 단위의 softmax-log loss를 사용한다.

$$\ell_S(\tilde{S}, S) = -\frac{1}{wh} \sum_{x=1}^w \sum_{y=1}^h \sum_{j=1}^c S(x,y,j) \log \left( \frac{e^{\tilde{S}(x,y,j)}}{\sum_{k=1}^c e^{\tilde{S}(x,y,k)}} \right)$$

CSL 모델의 전체 손실 함수 $\ell_{CSL}$은 다음과 같이 정의된다.

$$\ell_{CSL} = \ell_S(\tilde{S}, S) + \lambda_H \sum_{i=1}^n \sum_{x=1}^w \sum_{y=1}^h \left\| \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{\|y_i - (x,y)^T\|_2^2}{2\sigma^2}} - \tilde{y}^*_{x,y,i} \right\|_2^2$$

여기서 $\sigma$는 랜드마크 주변 가우시안 분포의 퍼짐 정도를 조절하며, $\tilde{y}^*_{x,y,i}$는 예측된 히트맵 값이다. 추론 시에는 예측된 히트맵에서 신뢰도가 최대인 지점을 랜드마크의 최종 위치로 결정한다.

## 📊 Results

### 실험 설정

- **데이터셋**: Retinal Microsurgery (RM) 데이터셋(18개 시퀀스)과 MICCAI EndoVis Challenge 2015 데이터셋을 사용하였다.
- **비교 대상**: 직접 좌표를 예측하는 L 모델, 경로가 분리된 SL 모델, 그리고 U-Net 및 FCN 기반의 기존 방법론들과 비교하였다.
- **측정 지표**: RM 데이터셋에서는 KBB score를, EndoVis 데이터셋에서는 Balanced Accuracy, DICE score, 정밀도(Precision), 재현율(Recall) 및 위치 오차(Localization error)를 측정하였다.

### 정량적 결과

- **모델 전략 비교**: RM 데이터셋 평가 결과, CSL 모델이 가장 높은 정확도를 보였으며, 특히 20픽셀 임계값 기준 랜드마크 팁(Tip)에 대해 90% 이상의 정확도를 달성하였다. 이는 $\text{CSL} > \text{SL} > \text{L}$ 순의 성능 차이를 보여준다.
- **RM 데이터셋**: 제안 방법은 기존의 FPBC, POSE, Online Adaption 방식보다 뛰어난 성능을 보였으며, 평균 KBB score $\alpha=0.15$ 기준 84% 이상의 정확도를 기록하였다.
- **EndoVis Challenge**: 세그멘테이션과 위치 추정 모두에서 기존 SOTA(State-of-the-art) 성능을 유의미하게 상회하였다. 특히 학습 데이터에 없었던 새로운 도구 타입이나 시점에서도 강건하게 작동함을 확인하였다.
- **효율성**: NVIDIA GeForce GTX TITAN X 기준 프레임당 추론 시간은 56ms로, 실시간 적용이 가능한 수준이다.

## 🧠 Insights & Discussion

본 논문은 수술 도구 추적에 있어 세그멘테이션과 위치 추정을 통합하는 것이 개별적으로 수행하는 것보다 훨씬 효과적임을 입증하였다. 특히 Heatmap Regression 방식은 단순히 좌표값 하나만을 정답으로 간주하는 직접 회귀 방식보다 이미지의 컨텍스트(Context)를 더 잘 활용하며, 수동 레이블링 시 발생할 수 있는 몇 픽셀의 오차에 대해 더 유연하게 대응할 수 있다는 강점이 있다.

또한, ResNet-50의 강력한 특징 추출 능력과 Skip connection을 통한 고해상도 정보의 보존이 정밀한 랜드마크 위치 추정의 핵심 요인이 되었음을 알 수 있다.

한계점으로는 EndoVis 데이터셋의 일부 시퀀스(5, 6번)에서 위치 오차가 높게 나타났는데, 이는 모델의 결함보다는 해당 시퀀스의 Ground Truth 데이터 자체가 정밀하지 않았을 가능성이 크다는 점이 언급되었다. 또한, 데이터셋의 규모가 제한적임에도 불구하고 ImageNet으로 사전 학습된 가중치를 활용하여 성공적으로 일반화 성능을 확보하였다.

## 📌 TL;DR

본 연구는 수술 도구의 세그멘테이션과 2D 포즈 추정을 동시에 수행하는 **CSL(Concurrent Segmentation and Localization)** 프레임워크를 제안하였다. 핵심은 위치 추정 문제를 **Heatmap Regression**으로 재정의하여 세그멘테이션 작업과의 공간적 의존성을 극대화하고, ResNet-50 기반의 FCRN 구조와 Skip connection을 통해 정확도를 높인 것이다. 실험 결과, RM 및 EndoVis 벤치마크에서 기존 SOTA 모델들을 능가하는 성능과 실시간성을 입증하였으며, 이는 향후 정밀한 컴퓨터 보조 수술 시스템 구축에 중요한 기여를 할 것으로 평가된다.
