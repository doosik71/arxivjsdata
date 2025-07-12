# 대형 커널의 중요성: 전역 합성곱 네트워크를 통한 의미론적 분할 개선
Chao Peng, Xiangyu Zhang, Gang Yu, Guiming Luo, Jian Sun

## 🧩 Problem to Solve
의미론적 분할(Semantic Segmentation)은 픽셀 단위 분류 문제로, 두 가지 상충하는 과제를 동시에 해결해야 합니다:
- **분류(Classification)**: 객체가 다양한 변환(이동, 회전, 크기 조절 등)에 대해 불변하게 올바른 의미론적 개념으로 분류되어야 합니다.
- **지역화(Localization)**: 각 픽셀의 분류 레이블이 출력 스코어 맵에서 정확한 좌표에 맞춰져야 합니다.
기존의 최신 의미론적 분할 모델들은 주로 지역화에 중점을 두어, 객체 변환에 대한 분류기의 강건성이 떨어지고 '유효 수용 영역(Valid Receptive Field, VRF)'이 작아지는 문제가 발생했습니다.

## ✨ Key Contributions
1.  의미론적 분할에서 '분류' 및 '지역화' 문제를 동시에 명시적으로 해결하는 **전역 합성곱 네트워크(Global Convolutional Network, GCN)**를 제안했습니다.
2.  객체 경계 근처의 지역화 성능을 더욱 향상시키는 **경계 정제 블록(Boundary Refinement, BR)**을 도입했습니다.
3.  PASCAL VOC 2012 (82.2%) 및 Cityscapes (76.9%) 두 가지 표준 벤치마크에서 최첨단 성능을 달성했습니다.

## 📎 Related Works
이 연구는 FCN [25]을 기반으로 하며, 의미론적 분할의 다양한 개선 노력들을 언급합니다.
-   **맥락 임베딩(Context Embedding)**: Zoom-out [26], ParseNet [23], Dilated-Net [36], Deeplab-V2 (Atrous Spatial Pyramid Pooling) [7] 등이 있습니다.
-   **해상도 확장(Resolution Enlarging)**: Deconvolution (FCN [25]), Unpooling (Deconv-Net [27], SegNet [3]), Dilated-Convolution (Deeplab [24], Dilated-Net [36])과 같은 기법들이 활용됩니다.
-   **경계 정렬(Boundary Alignment)**: Conditional Random Field (CRF) 기반 방법론 (DenseCRF [6], CRFasRNN [37], DPN [24], Adelaide [21]) 및 Bilateral Solver [4], Bilateral filter [16] 등 다양한 접근 방식이 존재합니다.

## 🛠️ Methodology
이 논문은 분류와 지역화의 상충을 해결하기 위해 새로운 아키텍처를 제안합니다.
-   **GCN (Global Convolutional Network)**:
    -   **설계 원칙**: 1) 지역화 성능 유지를 위한 완전 합성곱(fully-convolutional) 구조, 2) 분류 능력 강화를 위한 큰 커널 크기 채택 (특징 맵과 픽셀별 분류기 간의 조밀한 연결 확보).
    -   **구현**: 단순한 대형 커널 대신, $1 \times k$ + $k \times 1$ 및 $k \times 1$ + $1 \times k$ 합성곱의 조합을 사용합니다. 이는 매개변수 및 계산 비용($O(2k)$ vs. $O(k^2)$)을 크게 줄이며, 대형 커널의 장점을 활용합니다.
-   **전반적인 프레임워크**: ResNet [14]을 특징 추출 네트워크로 사용하는 FCN [25] 기반 구조를 따릅니다. GCN은 다중 스케일 의미론적 스코어 맵을 생성하며, 낮은 해상도의 스코어 맵은 역합성곱(deconvolution)을 통해 업샘플링된 후 더 높은 해상도의 스코어 맵과 합쳐집니다.
-   **BR (Boundary Refinement) 블록**: 경계 정렬을 잔차(residual) 구조로 모델링합니다. $\tilde{S} = S + R(S)$ 형태로, 조악한 스코어 맵 $S$에 잔차 브랜치 $R(\cdot)$를 더해 정제된 스코어 맵 $\tilde{S}$를 얻습니다. 이는 네트워크에 통합되어 엔드-투-엔드 방식으로 학습됩니다.

## 📊 Results
두 가지 표준 벤치마크에서 최첨단 성능을 달성했습니다:
-   **PASCAL VOC 2012**: 82.2% mIoU (기존 최신 결과 80.2% 대비 향상).
-   **Cityscapes**: 76.9% mIoU (기존 최신 결과 71.8% 대비 향상).
**세부 실험 결과**:
-   GCN은 커널 크기 $k$가 증가함에 따라 성능이 꾸준히 향상됨을 확인했습니다 (예: $k=15$일 때 $k=1$ 대비 5.5% 성능 향상).
-   GCN은 매개변수 수가 $k$에 선형적으로 증가하며, 매개변수가 $k^2$에 비례하는 단순 $k \times k$ 합성곱이나 소형 합성곱 스택보다 성능이 우수하고 수렴에 유리합니다.
-   GCN은 주로 객체의 '내부 영역' (분류 능력)의 정확도를 개선하는 반면, BR은 '경계 영역' (지역화 능력)의 정확도를 향상시킵니다.
-   ImageNet 분류에서는 기존 ResNet 대비 약간 낮은 성능을 보였던 ResNet-GCN 사전 학습 모델이 의미론적 분할 미세 조정 시 크게 향상된 성능을 보여주었습니다.

## 🧠 Insights & Discussion
이 연구의 핵심 통찰은 의미론적 분할에서 분류와 지역화 간의 상충 관계를 완화하기 위해 **대형 커널**이 중요하다는 것입니다.
-   GCN은 효과적으로 '유효 수용 영역(VRF)'을 확대하여 분류기가 다양한 변환을 더 잘 처리할 수 있도록 하며, 이는 분류 성능 향상으로 이어집니다.
-   분리 가능한 필터(separable filters)를 사용하여 매개변수와 계산 비용을 효율적으로 관리하면서도 대형 커널의 이점을 얻어, 단순 대형 커널의 과적합 및 수렴 문제를 피했습니다.
-   BR 블록은 경계 지역의 정확도를 특별히 향상시켜 GCN의 내부 영역 개선과 상호 보완적인 역할을 수행합니다.
-   이 결합된 접근 방식은 의미론적 분할의 두 가지 핵심 과제를 효과적으로 해결합니다.

## 📌 TL;DR
**문제**: 의미론적 분할은 분류(변환 불변성)와 지역화(정확한 픽셀 위치)라는 상충되는 과제를 해결해야 합니다.
**해결책**: 효율적인 대형 분리형 커널을 사용하는 **전역 합성곱 네트워크(GCN)**를 제안하여 분류 능력을 향상시키고, 객체 경계의 정확도를 높이는 **경계 정제(BR) 블록**을 도입합니다.
**주요 발견**: GCN은 내부 영역 분류를 크게 개선하고, BR은 경계 정밀도를 높여 PASCAL VOC 2012 (82.2%) 및 Cityscapes (76.9%)에서 최첨단 성능을 달성했습니다. 이 연구는 대형 커널의 중요성을 강조합니다.