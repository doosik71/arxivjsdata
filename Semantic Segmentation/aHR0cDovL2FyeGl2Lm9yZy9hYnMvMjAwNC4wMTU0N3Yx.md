# Context Prior for Scene Segmentation
Changqian Yu, Jingbo Wang, Changxin Gao, Gang Yu, Chunhua Shen, Nong Sang

## 🧩 Problem to Solve
기존의 시맨틱 세그멘테이션(semantic segmentation) 연구들은 컨텍스트 의존성(contextual dependencies)을 활용하여 정확도를 높이려 했지만, 대부분의 접근 방식은 동일 클래스(intra-class) 컨텍스트와 다른 클래스(inter-class) 컨텍스트를 명확하게 구분하지 못했습니다. 이는 혼란스러운 공간 정보 집합을 야기하여 잘못된 예측으로 이어질 수 있습니다. 본 논문은 이러한 컨텍스트의 유형별 구분을 명확히 하는 문제를 해결하고자 합니다.

## ✨ Key Contributions
- **컨텍스트 사전(Context Prior) 개념 도입**: 동일 클래스 및 다른 클래스 컨텍스트 의존성을 명시적으로 학습하고 포착하기 위해 어피니티 손실(Affinity Loss)의 감독하에 컨텍스트 사전을 구성했습니다.
- **컨텍스트 사전 계층(Context Prior Layer) 개발**: 컨텍스트 사전을 신경망에 통합하기 위한 새로운 계층을 제안했으며, 이는 어그리게이션 모듈(Aggregation Module)을 포함하여 효율적으로 공간 정보를 집약합니다.
- **CPNet(Context Prior Network) 설계**: 백본 네트워크와 컨텍스트 사전 계층으로 구성된 효과적인 완전 합성곱 신경망(Fully Convolutional Network, FCN)인 CPNet을 설계했습니다.
- **최첨단 성능 달성**: ADE20K, PASCAL-Context, Cityscapes와 같은 주요 벤치마크 데이터셋에서 기존 최첨단 시맨틱 세그멘테이션 방법론들을 능가하는 성능을 입증했습니다.

## 📎 Related Works
- **컨텍스트 집약(Context Aggregation)**:
    - **피라미드 기반 방법**: PSPNet, DeepLab 시리즈는 피라미드 풀링 또는 atrous spatial pyramid pooling을 사용하여 다양한 스케일의 지역 또는 전역 컨텍스트를 집약합니다.
    - **어텐션 기반 방법**: DANet, OCNet, CCNet, PSANet 등은 자기 유사성(self-similarity)이나 포인트별 어텐션(point-wise attention)을 통해 장거리 공간 정보를 선택적으로 집약합니다. 하지만 명시적인 정규화(regularization)가 부족하여 원치 않는 컨텍스트를 포착할 수 있습니다.
- **어텐션 메커니즘(Attention Mechanism)**: 기계 번역, 이미지/액션 인식, 객체 탐지 등 다양한 분야에 적용되며, 시맨틱 세그멘테이션에서는 채널 어텐션(EncNet, DFN, BiSeNet)이나 공간 어텐션(DANet, OCNet, PSANet)으로 활용되었습니다.

## 🛠️ Methodology
1.  **컨텍스트 사전 개념**: 각 픽셀에 대해 어떤 픽셀이 같은 카테고리에 속하는지(동일 클래스 컨텍스트), 어떤 픽셀이 다른 카테고리에 속하는지(다른 클래스 컨텍스트)를 구분하는 이진 분류기로 컨텍스트 사전을 모델링합니다.
2.  **어피니티 손실(Affinity Loss)**:
    *   **이상적인 어피니티 맵(Ideal Affinity Map, $A$) 구성**: 입력 이미지의 다운샘플링된 정답($\tilde{L}$)을 원-핫 인코딩($\hat{L}$)한 후, 행렬 곱셈 $A = \hat{L}\hat{L}^{\top}$을 통해 각 픽셀이 같은 카테고리에 속하는지 여부를 인코딩하는 $N \times N$ 크기의 맵을 생성합니다. ($N = H \times W$)
    *   **손실 함수**: 예측된 사전 맵($P$)과 이상적인 어피니티 맵($A$) 간의 차이를 줄이도록 학습합니다.
        *   단항 손실($L_u$): 각 픽셀에 대한 이진 교차 엔트로피(binary cross entropy)입니다.
        *   전역 손실($L_g$): 정밀도($T_p$), 재현율($T_r$), 특이도($T_s$)를 포함하여 동일 클래스 및 다른 클래스 픽셀의 관계를 전체적으로 고려합니다.
        *   총 어피니티 손실: $L_p = \lambda_u L_u + \lambda_g L_g$
3.  **컨텍스트 사전 계층(Context Prior Layer)**:
    *   백본 네트워크에서 추출된 특징 $X$를 입력으로 받습니다.
    *   **어그리게이션 모듈(Aggregation Module)**을 사용하여 $X$로부터 공간 정보를 집약합니다.
    *   집약된 공간 정보에 1x1 합성곱, BN, Sigmoid 함수를 적용하여 컨텍스트 사전 맵 $P$를 생성합니다.
    *   $P$를 이용하여 동일 클래스 컨텍스트($Y_{\text{intra}} = P \otimes \tilde{X}$)와 다른 클래스 컨텍스트($Y_{\text{inter}} = (1-P) \otimes \tilde{X}$)를 선택적으로 포착합니다.
    *   원래 특징, 동일 클래스 컨텍스트, 다른 클래스 컨텍스트를 결합(`Concat(X, Y_intra, Y_inter)`)하여 최종 표현을 생성합니다.
4.  **어그리게이션 모듈**: 효율적인 공간 정보 집약을 위해 `k`x`1` 합성곱과 `1`x`k` 합성곱으로 구성된 두 개의 비대칭 완전 분리 가능 합성곱(fully separable convolution, 공간 및 깊이 분리)을 사용합니다. 이는 계산 비용을 줄이면서도 표준 합성곱과 동일한 수용장(receptive field) 크기를 유지합니다.
5.  **네트워크 구조(CPNet)**: ResNet과 같은 백본 네트워크와 컨텍스트 사전 계층으로 구성됩니다. 메인 세그멘테이션 손실($L_s$), 백본 네트워크의 4번째 단계에 적용되는 보조 손실($L_a$), 그리고 컨텍스트 사전 학습을 위한 어피니티 손실($L_p$)을 포함하는 전체 손실 함수는 다음과 같습니다: $L=\lambda_s L_s + \lambda_a L_a + \lambda_p L_p$.

## 📊 Results
- **ADE20K 데이터셋**: ResNet-101 백본 기반으로 46.27% mIoU (평균 IoU) 및 81.85% pixAcc (픽셀 정확도)를 달성하여 최첨단 성능을 기록했습니다. ResNet-50 기반 CPNet50도 44.46% mIoU를 달성하여 PSPNet, PSANet 등 다른 깊은 모델들보다 우수한 성능을 보였습니다.
- **PASCAL-Context 데이터셋**: ResNet-101 백본 기반으로 53.9% mIoU를 달성하여 EncNet 등 기존 최첨단 방법을 1.0% 이상 능가했습니다.
- **Cityscapes 데이터셋**: ResNet-101 백본 기반으로 81.3% mIoU를 달성하여 DenseASPP 등을 능가하는 최첨단 성능을 보였습니다.
- **가시화**: 학습된 사전 맵(Prior Map)이 어피니티 손실의 감독하에 이상적인 어피니티 맵과 유사하게 명시적인 구조 정보를 포착하는 것을 보여주었습니다.

## 🧠 Insights & Discussion
- **컨텍스트 구분의 중요성**: 동일 클래스 및 다른 클래스 컨텍스트를 명확히 구분하는 것이 장면 이해(scene understanding)에 매우 중요하다는 것을 입증했습니다. 기존 방법들이 혼합된 컨텍스트를 사용하여 발생했던 네트워크 혼란을 해소합니다.
- **어그리게이션 모듈의 역할**: 컨텍스트 사전이 관계를 추론하기 위해 적절한 지역 공간 정보가 필요함을 실험적으로 확인했습니다. 어그리게이션 모듈은 효율적으로 이를 제공하며, 이 모듈 단독으로도 좋은 성능을 보입니다.
- **컨텍스트 사전의 일반화 능력**: 제안된 컨텍스트 사전은 PPM(Pyramid Pooling Module)이나 ASPP(Atrous Spatial Pyramid Pooling)와 같은 다른 공간 정보 집약 모듈에도 일반화되어 성능 향상을 가져올 수 있음을 보여주었습니다. 이는 컨텍스트 사전 자체가 강력한 독립적 구성 요소임을 시사합니다.
- **한계 및 개선점**: 특정 필터 크기(`k=11`)에서 성능이 최고점에 달하며, 그 이상으로 증가하면 성능이 떨어진다는 점은 적절한 공간 정보 집약 범위의 중요성을 나타냅니다.

## 📌 TL;DR
장면 세그멘테이션에서 컨텍스트 정보의 혼합 사용은 성능 저하를 야기합니다. 본 논문은 이 문제를 해결하기 위해, 어피니티 손실의 감독하에 동일 클래스 및 다른 클래스 컨텍스트를 명시적으로 구분하는 **컨텍스트 사전(Context Prior)**을 제안합니다. 이 사전은 **컨텍스트 사전 계층(Context Prior Layer)**과 효율적인 **어그리게이션 모듈(Aggregation Module)**을 통해 구현되며, 이를 포함하는 **CPNet**은 ADE20K, PASCAL-Context, Cityscapes 등 주요 벤치마크에서 기존 최첨단 방법론들을 능가하는 우수한 성능을 달성했습니다.