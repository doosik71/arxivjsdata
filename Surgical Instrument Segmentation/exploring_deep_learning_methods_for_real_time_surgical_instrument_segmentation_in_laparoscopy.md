# Exploring Deep Learning Methods for Real-Time Surgical Instrument Segmentation in Laparoscopy

Debesh Jha, Sharib Ali, Nikhil Kumar Tomar, Michael A. Riegler, Dag Johansen, Håvard D. Johansen, Pål Halvorsen (2021)

## 🧩 Problem to Solve

본 연구가 해결하고자 하는 핵심 문제는 최소 침습 수술(Minimally Invasive Surgery, MIS), 특히 복강경 수술(Laparoscopy) 환경에서 수술 도구를 실시간으로 정확하게 분할(Segmentation)하는 것이다. 수술 도구의 위치를 정확히 탐지하고 추적하는 것은 컴퓨터 보조 수술 시스템의 구현을 위한 필수적인 단계이다.

그러나 복강경 영상 환경은 다음과 같은 기술적 난제들을 포함하고 있어 단순한 분할 기법으로는 해결이 어렵다.

- **반사광(Specularity):** 강한 조명으로 인해 조직이나 도구 표면에 발생하는 하얀 반사광이 도구의 색상을 왜곡시켜 다른 표면과 혼동을 일으킨다.
- **시각적 방해 요소:** 수술 중 발생하는 혈액, 연기(Smoke), 그리고 움직임으로 인한 아티팩트(Motion artifacts)가 존재한다.
- **실시간성 요구:** 실제 임상 환경에 적용하기 위해서는 매우 높은 추론 속도가 필수적이지만, 높은 정확도를 가진 딥러닝 모델들은 대개 연산량이 많아 속도가 느린 경향이 있다.

따라서 본 논문의 목표는 다양한 최신 딥러닝 방법론을 평가하고 비교하여, 복강경 영상의 까다로운 조건에서도 강건하게 작동하면서 동시에 실시간 처리가 가능한 최적의 모델을 찾는 것이다.

## ✨ Key Contributions

본 연구의 주요 기여는 실시간 수술 도구 분할을 위해 다양한 최신 딥러닝 아키텍처를 체계적으로 비교 분석하고, 최적의 성능을 보이는 모델을 제시했다는 점이다.

핵심적인 직관은 복잡하고 무거운 하이브리드 네트워크보다, 효율적으로 설계된 경량화 네트워크가 실제 임상 배치에 필요한 실시간 성능과 경쟁력 있는 정확도를 동시에 달성할 수 있다는 것이다. 특히, **DDANet(Dual Decoder Attention Network)**이 공간적 정보를 지속적으로 회복하는 구조를 통해 수술 도구의 경계를 더 날카롭게 포착하며, 정확도와 속도 사이의 최적의 균형점을 제공함을 입증하였다.

## 📎 Related Works

논문에서는 수술 컴퓨터 비전 분야의 발전 과정을 설명하며, 특히 다음과 같은 챌린지들을 언급한다.

- **EndoVis 2015 및 2017 챌린지:** 도구 분할 및 추적에 대한 기초적인 통찰을 제공했으나, 딥러닝 모델의 강건성(Robustness)과 일반화 능력에 대한 분석은 제한적이었다.
- **ROBUST-MIS 2019 챌린지:** 실제 임상 데이터셋을 활용하여 보다 강건한 분할 방법을 개발하는 것을 목표로 하였다. 여기서는 Mask-RCNN, OR-UNet, DeepLabV3+, U-Net, RASNet 등의 방법론이 탐색되었으며, 특히 ImageNet으로 사전 학습된 인코더를 사용한 OR-UNet과 DeepLabv3+가 좋은 성능을 보였다.

기존 연구들은 전반적으로 우수한 성과를 보였으나, 여전히 반사광, 혈액, 연기 등이 포함된 매우 까다로운 영상에서는 성능이 저하되는 한계가 있었다. 본 연구는 이러한 한계를 극복하기 위해 최신 SOTA(State-of-the-Art) 모델들을 동일한 데이터셋 환경에서 광범위하게 비교함으로써 차별점을 둔다.

## 🛠️ Methodology

### 전체 파이프라인 및 시스템 구조

본 연구는 새로운 모델을 제안하는 대신, 검증된 9가지 최신 딥러닝 모델을 사용하여 수술 도구 분할 성능을 벤치마킹하는 구조를 가진다. 입력 영상의 각 픽셀을 '수술 도구' 또는 '배경'으로 분류하는 이진 분할(Binary Segmentation) 작업을 수행한다.

### 비교 대상 모델

다음의 모델들이 분석 대상에 포함되었다.

- **U-Net, FCN8, ResUNet, ResUNet++, DDANet, ColonSegNet, PSPNet**
- **DeepLabv3+** (Backbone으로 MobileNetv2 및 ResNet50 사용)

### 학습 및 구현 절차

- **데이터셋:** ROBUST-MIS 데이터셋(Heidelberg Colorectal 데이터셋의 일부)을 사용하였으며, 총 5,983장의 이미지와 수동으로 작성된 마스크가 포함되어 있다.
- **데이터 전처리:** 원본 이미지 크기 $960 \times 540$을 리사이징하고, 데이터를 훈련(Training), 검증(Validation), 테스트(Testing) 세트로 $8:1:1$ 비율로 분할하였다.
- **학습 설정:**
  - 모든 실험에서 **Adam Optimizer**를 사용하였다.
  - 하드웨어 구성에 따라 온라인 증강(Online augmentation)과 오프라인 증강(Offline augmentation) 전략을 나누어 적용하였다.
  - 일부 모델(ResUNet, ResUNet++, ColonSegNet, DDANet)은 NVIDIA DGX-2에서, 나머지는 NVIDIA Quadro RTX 6000에서 학습되었다.

### 주요 모델의 특성 (DDANet)

논문에서 가장 우수한 성능을 보인 DDANet의 경우, 인코더 분기에서 **Spatial Attention(공간 주의 집중)** 맵을 생성하여 세그멘테이션 분기에 전달함으로써 특성 표현(Feature representation)을 개선하고, 더 정교한 경계선을 추출하는 구조를 가지고 있다.

## 📊 Results

### 실험 설정 및 지표

- **데이터셋:** ROBUST-MIS Challenge 2019
- **평가 지표:**
  - $\text{Dice Coefficient (DSC)}$: 예측 영역과 실제 영역의 중첩도를 측정
  - $\text{mean Intersection over Union (mIoU)}$: 교집합 영역을 합집합 영역으로 나눈 값
  - $\text{Recall}, \text{Precision}, \text{F2-score}, \text{Accuracy}$
  - $\text{FPS (Frames Per Second)}$: 초당 처리 프레임 수 (실시간성 평가)

### 정량적 결과

실험 결과, **DDANet**이 전반적으로 가장 우수한 성능을 기록하였다.

- **DDANet:** $\text{DSC} = 0.8739$, $\text{mIoU} = 0.8183$, $\text{FPS} = 101.36$
- **ColonSegNet:** $\text{Recall}$이 $0.8899$로 가장 높았으며, $\text{FPS}$ 또한 $185.54$로 가장 빨랐다.
- **DeepLabv3+ (ResNet50):** $\text{Precision}$ 측면에서 강점을 보였다.

### 정성적 결과

영상의 난이도를 Easy, Medium, Hard로 나누어 분석한 결과는 다음과 같다.

- **Easy cases:** 모든 모델이 만족스러운 성능을 보였다.
- **Medium cases:** DeepLabv3+, ColonSegNet, DDANet이 우수하였으나, ColonSegNet과 DDANet에서 일부 과분할(Over-segmentation) 현상이 관찰되었다.
- **Hard cases:** DeepLabv3+가 일부 시나리오에서 더 나은 결과를 보였으나, 전반적으로 DDANet이 까다로운 환경에서도 유망한 결과를 보여주었다.

## 🧠 Insights & Discussion

본 연구를 통해 도출된 주요 인사이트는 다음과 같다.

첫째, **모델의 복잡성과 실시간성의 트레이드오프(Trade-off)**이다. DeepLabv3+와 같이 사전 학습된(Pre-trained) 무거운 인코더를 사용하는 모델보다, DDANet이나 ColonSegNet과 같은 경량화된 네트워크가 실시간 추론 속도 면에서 압도적이며 정확도 또한 경쟁력 있음을 확인하였다. 이는 실제 수술실 배포를 위해서는 네트워크의 경량화가 필수적임을 시사한다.

둘째, **Attention 메커니즘의 유효성**이다. DDANet의 공간 주의 집중 구조는 수술 도구의 경계를 더 명확하게 잡아낼 수 있게 하여, 복잡한 배경 속에서도 도구를 효과적으로 분리해낼 수 있었다.

셋째, **여전한 한계점**이다. 최신 모델들조차 도구가 빠르게 움직이는 영상이나 매우 극한의 Hard case에서는 과분할(Over-segmentation) 또는 저분할(Under-segmentation) 문제가 발생한다. 또한, 본 논문에서는 테스트 세트의 Ground Truth가 제공되지 않아 챌린지 참가자들과 직접적인 비교가 불가능했다는 점이 한계로 남는다.

## 📌 TL;DR

본 논문은 복강경 수술 영상에서 실시간 도구 분할을 위해 9가지 딥러닝 모델을 비교 분석하였다. 실험 결과, **DDANet**이 $\text{DSC } 0.8739$와 $101.36 \text{ FPS}$라는 높은 정확도와 실시간 속도를 동시에 달성하여 가장 효율적인 모델임을 입증하였다. 이 연구는 복잡한 하이브리드 모델보다 최적화된 경량 네트워크가 임상 적용에 더 유리함을 보여주며, 향후 수술 도구 추적 및 자동화 시스템 구축을 위한 강력한 베이스라인을 제공한다.
