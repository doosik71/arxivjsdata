# Collaborative Video Object Segmentation by Foreground-Background Integration
Zongxin Yang, Yunchao Wei, Yi Yang

## 🧩 Problem to Solve
기존의 준지도 비디오 객체 분할(Semi-supervised Video Object Segmentation, VOS) 연구들은 주로 전경(foreground) 객체 픽셀만을 활용한 임베딩 학습에 집중했습니다. 이로 인해 배경과 유사한 객체가 존재할 경우 전경 객체 분할에서 배경 혼란(background confusion) 문제가 발생하며, 다양한 객체 스케일에 대한 견고성이 부족하다는 문제가 있었습니다. 본 논문은 이러한 한계를 극복하고 더 정확하고 견고한 VOS를 달성하고자 합니다.

## ✨ Key Contributions
*   **전경-배경 통합(Foreground-Background Integration, FBI) 제안**: 전경 픽셀뿐만 아니라 배경 픽셀의 임베딩 학습을 동등하게 고려하여, 전경과 배경 특징이 대조적으로 학습되도록 유도합니다.
*   **다단계 임베딩 매칭**: 픽셀 수준(pixel-level)과 인스턴스 수준(instance-level)의 임베딩 매칭을 통합하여 다양한 객체 스케일에 견고한 분할을 가능하게 합니다.
*   **다중-로컬 매칭(Multi-Local Matching)**: 다양한 객체 움직임 속도에 대응하기 위해 여러 스케일의 로컬 윈도우를 활용하는 매칭 메커니즘을 도입합니다.
*   **협력적 인스턴스-수준 어텐션(Collaborative Instance-level Attention)**: 인스턴스 수준 정보를 활용하여 픽셀 수준 매칭을 보완하고 대규모 객체 분할을 개선합니다.
*   **협력적 앙상블러(Collaborative Ensembler, CE) 설계**: 학습된 전경/배경 및 픽셀/인스턴스 수준 정보를 통합하고 정밀한 예측을 수행하기 위한 네트워크를 제안합니다.
*   **균형 잡힌 랜덤 크롭(Balanced Random-Crop)**: 전경-배경 픽셀 수 불균형 문제를 해결하여 모델이 배경 특성에 편향되는 것을 방지합니다.
*   **최첨단 성능 달성**: DAVIS 2016, DAVIS 2017, YouTube-VOS 세 가지 인기 벤치마크에서 기존의 모든 최첨단(SOTA) 방법들을 뛰어넘는 성능을 달성했습니다.

## 📎 Related Works
*   **준지도 VOS**:
    *   **미세 조정 기반**: OSVOS, MoNet, OnAVOS, MaskTrack, PReMVOS 등. 높은 정확도를 보이지만 추론 속도가 느립니다.
    *   **미세 조정 불필요**: OSMN, PML, VideoMatch, FEELVOS, RGMP, STMVOS 등. STMVOS는 메모리 네트워크를 사용하고 방대한 시뮬레이션 데이터를 활용하여 높은 성능을 달성했지만, 복잡한 훈련 절차와 시뮬레이션 데이터 의존성이라는 한계가 있습니다. FEELVOS는 픽셀 수준 매칭을 사용하지만 STMVOS보다 성능이 떨어집니다. 이들 중 대부분은 배경 매칭에 충분히 집중하지 않았습니다.
*   **어텐션 메커니즘**: SE-Nets 등 컨볼루션 네트워크의 표현력 향상에 기여한 채널 어텐션 메커니즘에서 영감을 받았습니다.

## 🛠️ Methodology
CFBI(Collaborative video object segmentation by Foreground-Background Integration)는 다음 구성 요소로 이루어집니다:

1.  **협력적 픽셀-수준 매칭 (Collaborative Pixel-level Matching)**:
    *   **전경/배경 정보 통합**: 기존 픽셀 거리 함수를 재설계하여 전경 ($b_F$) 및 배경 ($b_B$) 바이어스를 도입, 전경-배경 픽셀 구분을 강화합니다.
        $$D_t(p, q) = \begin{cases} 1 - \frac{2}{1+\exp(||e_p - e_q||^2 + b_B)} & \text{if } q \in B_t \\ 1 - \frac{2}{1+\exp(||e_p - e_q||^2 + b_F)} & \text{if } q \in F_t \end{cases}$$
    *   **전경-배경 전역 매칭(Foreground-Background Global Matching)**: 현재 프레임 픽셀 $p$와 첫 번째 참조 프레임 픽셀 $q$ 간의 최소 거리 $G_{T,o}(p)$ (전경) 및 $G_{T,o}(p)$ (배경)를 계산합니다.
    *   **전경-배경 다중-로컬 매칭(Foreground-Background Multi-Local Matching)**: 이전 프레임과 현재 프레임 간의 로컬 매칭에 여러 크기의 이웃 윈도우 $K=\{k_1, k_2, ..., k_n\}$를 적용하여 다양한 객체 이동 속도에 견고하게 만듭니다. 계산 오버헤드는 최소화됩니다.
    *   출력은 현재 프레임의 픽셀 수준 임베딩, 이전 프레임의 픽셀 수준 임베딩 및 마스크, 다중-로컬 매칭 맵, 전역 매칭 맵의 연결(concatenation)입니다.

2.  **협력적 인스턴스-수준 어텐션 (Collaborative Instance-level Attention)**:
    *   첫 번째 프레임과 이전 프레임의 전경 및 배경 픽셀 임베딩으로부터 채널별 평균 풀링을 통해 네 가지 인스턴스 수준 임베딩 벡터를 생성합니다.
    *   이 벡터들을 연결하여 '협력적 인스턴스-수준 가이드 벡터'를 생성하고, 이를 완전 연결(FC) 계층과 비선형 활성화 함수를 통해 Collaborative Ensembler의 Res-Block을 조정하는 어텐션 게이트로 활용합니다.

3.  **협력적 앙상블러 (Collaborative Ensembler, CE)**:
    *   ResNets 및 Deeplabs에서 영감을 받아 다운샘플-업샘플 구조를 사용합니다.
    *   3단계의 Res-Blocks와 Atrous Spatial Pyramid Pooling (ASPP) 모듈로 구성됩니다.
    *   확장된 컨볼루션(dilated convolution)을 사용하여 수용 필드(receptive field)를 효율적으로 확장합니다.

4.  **구현 상세 (Implementation Details)**:
    *   **균형 잡힌 랜덤 크롭(Balanced Random-Crop)**: 훈련 시 전경 픽셀이 충분히 포함된 영역을 크롭하여 전경-배경 불균형 문제를 완화합니다.
    *   **순차 훈련(Sequential Training)**: RGMP를 따라 각 SGD 반복에서 연속적인 프레임 시퀀스를 사용하여 훈련하며, 이전 마스크는 네트워크의 최신 예측값을 사용합니다. (FEELVOS와 달리 Ground-truth를 사용하지 않음)
    *   **백본**: DeepLabv3+ 아키텍처를 기반으로 한 확장된 Resnet-101을 사용합니다.
    *   **데이터셋**: DAVIS 2017 훈련 세트와 YouTube-VOS 훈련 세트를 사용하여 모델을 훈련합니다.

## 📊 Results
CFBI는 여러 벤치마크에서 기존 최첨단 방법들을 뛰어넘는 성능을 달성했습니다:

*   **YouTube-VOS**:
    *   시뮬레이션 데이터나 미세 조정 없이 81.4%(J&F)의 평균 점수를 달성, STMVOS (79.4%)를 크게 능가했습니다.
    *   멀티-스케일 및 플립 증강 적용 시 82.7%로 더욱 향상되었습니다.
    *   YouTube-VOS 2019 테스트 세트에서 단일 모델로 Rank 1(EMN) 및 Rank 2(MST) 결과를 능가했습니다.

*   **DAVIS 2016**:
    *   YouTube-VOS 훈련 세트를 추가로 사용했을 때 89.4%(J&F)를 달성, STMVOS (89.3%)보다 약간 우수합니다.
    *   FEELVOS (81.7%)보다 훨씬 높은 정확도를 보이면서도 경쟁력 있는 추론 속도(0.18초 vs 0.45초)를 유지합니다.
    *   멀티-스케일 및 플립 증강 적용 시 90.1%로 향상됩니다.

*   **DAVIS 2017**:
    *   YouTube-VOS 훈련 세트를 사용했을 때 81.9%(J&F)를 달성하여 FEELVOS (71.5%)를 크게 능가합니다.
    *   시뮬레이션 데이터 없이 STMVOS (81.8%)보다 약간 우수합니다.
    *   멀티-스케일 및 플립 증강 적용 시 83.3%로 향상됩니다.
    *   테스트 세트에서 STMVOS (72.2%)를 2.6%p 능가하는 74.8%를 달성했습니다. 증강 적용 시 77.5%까지 향상됩니다.

*   **정성적 결과**: 큰 움직임, 폐색, 블러, 유사 객체 등 어려운 상황에서도 정확한 분할을 수행함을 입증했습니다. (예: 양떼 속 여러 양 추적 성공). 두 명의 유사하고 가까운 인물 간의 분할에서 일부 실패 사례가 있었습니다.

## 🧠 Insights & Discussion
*   **전경-배경 통합의 중요성**: 배경 임베딩을 함께 고려하는 것이 VOS 성능 향상에 필수적임을 입증했습니다. 특히 픽셀 수준 매칭에서 배경 정보가 큰 영향을 미치는 것으로 나타났습니다.
*   **다단계 임베딩의 효과**: 픽셀 수준 정보와 인스턴스 수준 정보의 조합은 모델의 견고성과 정확도를 크게 향상시킵니다. 인스턴스 수준 어텐션은 픽셀 수준 정보만으로는 해결하기 어려운 로컬 모호성을 해소하는 데 도움을 줍니다.
*   **매칭 메커니즘의 개선**: 다중-로컬 윈도우를 사용한 매칭은 다양한 객체 이동 속도에 대한 견고성을 제공합니다.
*   **훈련 전략의 기여**: 균형 잡힌 랜덤 크롭은 전경-배경 픽셀 수 불균형으로 인한 모델 편향을 효과적으로 줄여 성능을 향상시킵니다. 순차 훈련 방식 또한 추론 단계와의 일관성을 높여 성능에 긍정적인 영향을 미칩니다.
*   **단순하지만 효과적인 설계**: CFBI는 복잡한 시뮬레이션 데이터나 추가적인 미세 조정 없이도 최첨단 성능을 달성하며, VOS 연구의 강력한 기준선(baseline) 역할을 할 수 있음을 시사합니다.

## 📌 TL;DR
본 논문은 준지도 비디오 객체 분할(VOS)에서 배경 혼란 문제를 해결하기 위해 **전경-배경 통합(Foreground-Background Integration, FBI)**을 제안하는 **CFBI** 프레임워크를 선보입니다. CFBI는 전경과 배경 픽셀 임베딩을 대조적으로 학습시키고, 픽셀 및 인스턴스 수준의 다단계 매칭 메커니즘을 통합하며, 다양한 객체 스케일과 움직임에 견고하게 대응합니다. 또한, 전경-배경 픽셀 불균형을 해소하는 균형 잡힌 랜덤 크롭 및 순차 훈련과 같은 효과적인 훈련 전략을 도입했습니다. 결과적으로 CFBI는 DAVIS 및 YouTube-VOS 벤치마크에서 기존의 모든 최첨단 방법들을 뛰어넘는 성능을 달성하며, VOS 분야의 강력한 새로운 기준선을 제시했습니다.