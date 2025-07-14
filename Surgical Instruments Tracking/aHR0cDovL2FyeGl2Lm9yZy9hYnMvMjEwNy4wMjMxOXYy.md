# 복강경 수술에서 실시간 수술 도구 분할을 위한 딥러닝 방법 탐색
Debesh Jha, Sharib Ali, Nikhil Kumar Tomar, Michael A. Riegler, Dag Johansen, Håvard D. Johansen, Pål Halvorsen

## 🧩 Problem to Solve
최소 침습 수술(MIS)은 기존 개복 수술에 비해 장점이 많지만, 제한된 수술 공간, 내시경의 좁은 시야, 외과의의 손재주 감소, 힘 피드백 부족 등으로 인해 수술 도구의 위치 감지 및 추적에 어려움이 있습니다. 특히 혈액, 연기, 반사광, 모션 아티팩트와 같은 도전적인 환경 조건으로 인해 수술 도구의 정확한 분할(Segmentation)은 매우 어려운 과제입니다. 기존 방법들은 이러한 도전적인 이미지에서 실패하는 경향이 있어, 복강경 이미지에서 수술 도구를 실시간으로 효율적으로 분할할 수 있는 컴퓨터 비전 방법의 개발이 절실합니다.

## ✨ Key Contributions
*   복강경 수술 도구의 자동 분할을 위해 여러 인기 있는 딥러닝 방법들을 평가하고 비교했습니다.
*   Dual Decoder Attention Network(DDANet)가 다른 최신 딥러닝 방법들에 비해 우수한 성능을 보임을 입증했습니다.
*   DDANet은 ROBUST-MIS Challenge 2019 데이터셋에서 0.8739의 다이스 계수(Dice coefficient)와 0.8183의 평균 교차 분할 합(mean Intersection-over-Union)을 달성했습니다.
*   DDANet은 초당 101.36 프레임(FPS)의 실시간 속도로 동작하여, 수술 절차에 매우 중요한 실시간 요구 사항을 충족했습니다.
*   가볍고 효율적인 네트워크 아키텍처(DDANet, ColonSegNet)가 복잡한 하이브리드 네트워크에 필적하는 경쟁력 있는 성능으로 실시간 추론을 달성할 수 있음을 보여주었습니다.

## 📎 Related Works
*   **EndoVis 2015 Instrument sub-challenge 및 Robotic Instrument Segmentation Sub-Challenge 2017:** 수술 도구 분할 및 추적에 대한 통찰을 제공했지만, 딥러닝 방법의 견고성과 일반화 능력에 대한 통찰은 제한적이었습니다.
*   **Robust Medical Instrument Segmentation Challenge 2019 (ROBUST-MIS):** 최소 침습 수술에서 수술 도구를 추적하기 위한 잠재적 방법 개발을 목표로 했습니다.
*   **참가자들이 주로 탐색한 방법:** Mask-RCNN, OR-UNet, DeepLabV3+, U-Net, RASNet 등이 있습니다.
*   **최고 성능 방법:** 이진 도구 분할에서 OR-UNet과 ImageNet 사전 훈련 인코더를 사용한 DeepLabv3+가 가장 좋은 성능을 보였습니다.
*   **최근 연구:** Ceron et al. [16]은 약 45 FPS로 최신 성능을 달성한 어텐션 기반 MIS 도구 분할 방법을 제안했습니다.

## 🛠️ Methodology
*   **목표:** 이미지의 각 픽셀을 '수술 도구' 또는 '배경'으로 분류하여 픽셀 단위의 분할 마스크를 생성합니다.
*   **데이터셋:** Heidelberg Colorectal (HeiCo) 데이터셋 [10], [19]의 일부인 ROBUST-MIS 데이터셋을 사용했습니다. 이 데이터셋은 16가지 다른 수술에서 얻은 총 5983개의 이미지와 수동으로 주석 처리된 분할 마스크를 포함합니다.
*   **데이터 전처리:**
    *   원래 이미지 크기 $960 \times 540$을 사용했습니다.
    *   데이터셋을 훈련, 검증, 테스트용으로 80-10-10 비율로 분할했습니다.
    *   훈련 데이터셋은 증강(augmentation) 및 크기 조절(resizing)을 수행했습니다.
*   **실험 설정 및 구성:**
    *   대부분의 실험은 PyTorch 프레임워크를 사용했으며, ResUNet과 ResUNet++는 TensorFlow [22]로 구현했습니다.
    *   NVIDIA DGX-2 및 NVIDIA Quadro RTX 6000 GPU를 사용하여 모델을 훈련했습니다.
    *   훈련 시 오프라인 또는 온라인 데이터 증강 전략을 채택했습니다.
    *   Adam 옵티마이저를 사용했으며, 하이퍼파라미터는 경험적 평가를 통해 조정했습니다.
*   **비교 모델:** UNet [14], FCN8 [25], ResUNet [20], ResUNet++ [21], DDANet [24], ColonSegNet [23], PSPNet [26], DeepLabv3+ [13] (ResNet50 [28] 및 MobileNetv2 [27] 백본 포함) 등 9가지 최신 딥러닝 분할 방법들을 비교했습니다.

## 📊 Results
*   **정량적 결과:**
    *   **DDANet**이 가장 높은 성능을 보였습니다: 다이스 계수 (DSC) 0.8739, 평균 교차 분할 합 (mIoU) 0.8183, 재현율(Recall) 0.8703, 정밀도(Precision) 0.9348, F2 0.8613, 정확도(Accuracy) 0.9897, **실시간 속도 101.36 FPS**.
    *   **ColonSegNet**은 경쟁력 있는 DSC, mIoU, 정밀도, F2를 보였으며, 가장 높은 재현율(0.8899)과 **최고 FPS (185.54)**를 달성했습니다.
    *   DeepLabv3+ (ResNet50 백본)이 가장 좋은 정밀도를 기록했습니다.
    *   2015년에서 2021년 사이의 방법들을 비교했을 때, 딥러닝 방법의 정확도와 속도 모두에서 상당한 개선이 관찰되었습니다.
*   **정성적 결과 (Figure 1):**
    *   모든 모델은 '쉬운' 케이스(도구가 하나 또는 여러 개인 경우)에서 만족스러운 성능을 보였습니다.
    *   '중간' 케이스에서는 Deeplabv3, ColonSegNet, DDANet이 만족스러웠지만, ColonSegNet과 DDANet에서 과분할(over-segmentation)이 관찰되었습니다.
    *   '어려운' 케이스에서는 Deeplabv3 (두 가지 백본 모두)가 일부 시나리오에서 더 나은 분할 결과를 보였고, ColonSegNet과 DDANet은 과분할을 나타냈습니다.
    *   일부 도전적인 시나리오에서는 DDANet이 유망한 결과를 보였으며, 높은 FPS로 인해 전반적으로 선호되는 선택임을 확인했습니다.
*   **저사양 GPU 성능:** NVIDIA 1060 Ti (6GB) GPU에서 ColonSegNet은 120.05 FPS, DDANet은 95.05 FPS를 달성하여 모델의 효율성을 입증했습니다.

## 🧠 Insights & Discussion
*   수술 도구 분할은 추적 및 자동화 시스템으로 나아가기 위한 중요한 단계이지만, 실시간 요구사항이 많은 최신 딥러닝 방법의 임상 적용을 제한합니다.
*   DDANet은 높은 성능과 실시간 처리 능력(101.36 FPS)을 제공하여 가장 효율적인 아키텍처로 확인되었습니다. DDANet의 인코더-디코더 네트워크는 공간 어텐션(spatial attention) 맵 생성을 통해 더 선명한 수술 도구 경계를 포착하고 특징 표현을 개선합니다.
*   ColonSegNet 및 DDANet과 같은 경량 네트워크는 사전 훈련된 인코더 없이도 경쟁력 있는 성능과 높은 FPS를 달성하여 임상 배포에 효율적임을 보여주었습니다.
*   최고 성능 모델들은 반사광, 혈액, 연기가 있는 샘플에 대해 견고했지만, 일부 영역에서 과분할 문제가 여전히 존재합니다.
*   움직이는 도구가 있는 이미지와 어려운 케이스에서의 과분할/미분할 문제는 추가 연구에서 해결해야 할 과제로 남아있습니다.
*   향후 연구에서는 다중 인스턴스 분할 및 복합 도구 추적 기술을 탐구할 계획입니다.

## 📌 TL;DR
**문제:** 복강경 수술 환경의 복잡성(혈액, 연기, 반사, 움직임)으로 인해 수술 도구를 실시간으로 정확하게 분할하는 것은 어렵지만, 컴퓨터 지원 수술에 필수적입니다.
**방법:** 본 연구는 ROBUST-MIS Challenge 2019 데이터셋을 사용하여 UNet, FCN8, DeepLabv3+, DDANet, ColonSegNet 등 9가지 최신 딥러닝 분할 모델의 성능을 평가하고 비교했습니다.
**결과:** DDANet이 가장 우수한 성능(다이스 계수 0.8739, mIoU 0.8183)을 101.36 FPS의 실시간 속도로 달성하여, 그 탁월함과 실용적 적용 가능성을 입증했습니다. DDANet 및 ColonSegNet과 같은 경량 네트워크는 사전 훈련된 인코더 없이도 실시간 배포에 효율적임을 보여주었습니다.