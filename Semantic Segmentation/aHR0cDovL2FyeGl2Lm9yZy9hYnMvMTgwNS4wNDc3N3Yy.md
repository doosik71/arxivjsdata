# Convolutional CRFs for Semantic Segmentation
Marvin T. T. Teichmann, Roberto Cipolla

## 🧩 Problem to Solve
의미론적 이미지 분할(Semantic Image Segmentation)을 위한 기존 모델들은 컨볼루션 신경망(CNN)의 특징 추출 능력과 조건부 랜덤 필드(CRF)의 구조적 모델링 능력을 결합해왔습니다. 하지만 최근 연구에서는 CRF 후처리가 인기를 잃었는데, 이는 주로 CRF의 느린 학습 및 추론 속도와 내부 CRF 파라미터 학습의 어려움 때문입니다. CNN은 지역적 특징 추출과 작은 시야를 활용한 예측에 강력하지만, 전역적인 문맥 정보를 활용하지 못하고 예측 간의 상호작용을 직접 모델링하지 못하는 한계가 있습니다.

## ✨ Key Contributions
*   **조건부 독립성 가정 추가**: 완전 연결 CRF(FullCRFs) 프레임워크에 조건부 독립성 가정을 추가하여 모델의 복잡성을 크게 줄였습니다.
*   **메시지 전달의 컨볼루션 재정의**: CRF 추론의 핵심인 메시지 전달 단계를 컨볼루션 연산으로 재정의하여 GPU에서 매우 효율적으로 구현 가능하게 만들었습니다.
*   **두 자릿수 속도 향상**: 기존 FullCRF에 비해 추론 및 학습 속도를 두 자릿수(약 100배) 향상시켰습니다 (예: 10ms 미만).
*   **모든 파라미터 역전파 학습**: 제안하는 컨볼루션 CRF(ConvCRF)의 모든 파라미터(가우시안 특징 포함)를 역전파(backpropagation)를 통해 쉽게 최적화할 수 있게 하였습니다.
*   **정확도 향상**: permutohedral lattice 근사 없이 정확한 메시지 전달을 통해 정확도가 소폭 개선됨을 입증했습니다.
*   **공개된 구현**: 향후 CRF 연구를 촉진하기 위해 구현 코드를 공개했습니다.

## 📎 Related Works
*   **CNN 기반 의미론적 분할**: FCNs [23], 전치 컨볼루션 레이어, Atrous(dilated) 컨볼루션 [5,38] 등 강력한 딥러닝 아키텍처가 발전을 주도했습니다. 이 모델들은 주로 CNN의 특징 추출 능력에 의존하며, 픽셀 단위 예측은 조건부 독립적인 경향이 있어 구조적 지식을 무시합니다.
*   **CNN-CRF 결합**: CNN 예측 위에 완전 연결 CRF(FullCRF) [17]를 적용하는 방식이 인기를 얻었습니다 (예: DeepLab [5,41,6], CRFasRNN [41]). 피라미드 풀링(Pyramid pooling) [40]도 문맥 정보 통합을 위한 대안으로 제안되었지만, 진정한 구조적 추론은 제공하지 못합니다.
*   **CRF 파라미터 학습**: FullCRF [17]는 쌍(pairwise) 가우시안 커널에 수동으로 제작된 특징을 사용했습니다. 이후 그래디언트 강하 [18] 및 다른 방법(Quadratic optimization [4,36], piecewise training [20])이 제안되었으나, 여전히 학습이 어렵거나 속도가 느리거나 근사적인 문제점이 있었습니다.
*   **CRF 추론 속도**: FullCRF 도입 [17] 이후 추론 속도에 큰 진전이 없었으며, 일부 접근 방식은 예측 해상도를 낮춰 속도를 높였지만 예측 능력 저하를 초래했습니다.

## 🛠️ Methodology
1.  **조건부 독립성 가정**: ConvCRF는 두 픽셀 $i, j$의 레이블 분포가 맨해튼 거리 $d(i,j)$가 특정 `filter-size` $k$보다 클 경우 조건부 독립적이라고 가정합니다. 이는 $k$를 넘어서는 픽셀에 대한 쌍 포텐셜(pairwise potential)을 0으로 만들어 복잡성을 크게 줄입니다.
2.  **효율적인 메시지 전달**:
    *   이 가정을 통해 FullCRF의 메시지 전달 단계를 잘린 가우시안 커널(truncated Gaussian kernel)을 사용한 컨볼루션으로 재정의합니다.
    *   메시지 전달 연산은 다음과 같이 정의됩니다:
        $$Q[b,c,x,y] = \sum_{dx,dy \le k} K[b,dx,dy,x,y] \cdot P[b,c,x+dx,y+dy]$$
        여기서 $K$는 통합된 커널 행렬이고, $P$는 이전 반복의 입력입니다.
    *   이 연산은 일반적인 2D 컨볼루션과 유사하지만, 필터 값이 공간 차원 $x,y$에 따라 달라집니다(locally connected layers [8]와 유사). 채널 차원 $c$에서는 필터가 일정합니다.
    *   GPU에서 효율적인 구현을 위해 `im2col`과 유사한 데이터 재구성 후 채널 차원에 대한 배치 점곱을 수행하는 저수준 구현을 사용했습니다.
3.  **학습 가능한 파라미터**: ConvCRF의 모든 파라미터 ($w^{(m)}$, $\theta_{\alpha}$, $\theta_{\beta}$, $\theta_{\gamma}$)는 역전파를 사용하여 쉽게 최적화될 수 있습니다. 1x1 컨볼루션을 사용하여 호환성 변환(compatibility transformation)도 학습할 수 있습니다.
4.  **실험 설정**: FullCRF와의 비교를 위해 Potts 모델, softmax 정규화, 그리고 수동으로 제작된 가우시안 특징(이후 학습 가능한 특징으로 대체)을 사용합니다. 평균 필드(mean-field) 추론은 5회 반복으로 수행됩니다.
5.  **학습 전략**:
    *   **분리 학습 (Decoupled training)**: 먼저 Unary CNN 모델을 학습시켜 예측을 얻은 후, CNN 파라미터를 고정하고 CRF 파라미터를 최적화합니다. 이는 유연성과 해석 가능성을 제공합니다.
    *   **종단 간 학습 (End-to-End learning)**: CNN과 ConvCRF 파라미터를 함께 학습합니다. 소실 그래디언트(vanishing gradient) 문제를 해결하기 위해 보조 Unary 손실을 도입하고 그래디언트 업데이트 단계를 번갈아 수행합니다.

## 📊 Results
*   **합성 데이터셋**:
    *   ConvCRF는 FullCRF보다 훨씬 뛰어난 성능을 보였습니다 (예: Conv7의 mIoU 93.89% vs FullCRF 84.37%).
    *   속도 면에서 ConvCRF는 FullCRF보다 두 자릿수(예: Conv7 13ms vs FullCRF 684ms) 빨랐습니다.
    *   시각적 비교에서 ConvCRF는 FullCRF에서 나타나는 객체 경계의 근사 오류 아티팩트 없이 더 높은 품질의 출력을 제공했습니다.
*   **PASCAL VOC (분리 학습)**:
    *   CRF 적용은 Unary baseline (mIoU 71.23%)에 비해 성능을 크게 향상시켰습니다 (ConvCRF mIoU 72.04%).
    *   ConvCRF는 DeepLab-CRF와 유사하거나 약간 더 나은 성능을 보였습니다.
    *   학습 가능한 호환성 변환(+C)과 학습 가능한 가우시안 특징(+T)을 모두 사용한 Conv+CT 모델이 가장 좋은 성능(mIoU 72.37%)을 달성했습니다.
*   **PASCAL VOC (종단 간 학습)**:
    *   ConvCRF는 mIoU 72.18%를 달성하여 Unary (70.99%) 및 CRFasRNN (69.6%)보다 우수한 성능을 보였습니다.
    *   학습 시간은 크게 단축되어 (예: 4개의 1080Ti GPU로 250 Epoch에 약 30시간) 기존 CRF 모델보다 훨씬 유연한 연구를 가능하게 했습니다.

## 🧠 Insights & Discussion
*   **가정의 타당성**: 조건부 독립성 가정은 CNN의 지역 특징 처리의 성공을 고려할 때 강력하고 타당합니다.
*   **정확한 메시지 전달의 이점**: permutohedral lattice 근사를 제거하고 정확한 메시지 전달을 구현함으로써 더 나은 정확도(아티팩트 감소)와 GPU 상의 효율적인 연산이 가능해졌습니다.
*   **속도 향상의 중요성**: 두 자릿수 속도 향상은 CRF 기반 접근 방식을 실제 응용 분야에서 실용적으로 만들고, 해당 모델에 대한 심층 연구 및 실험을 용이하게 합니다.
*   **학습 가능성의 확장**: 가우시안 특징을 포함한 모든 CRF 파라미터를 역전파로 학습할 수 있게 되어, 수동으로 특징을 설계해야 하는 제약에서 벗어났습니다.
*   **학습 전략의 유연성**: 분리 학습과 종단 간 학습 모두 효과적이며, 종단 간 학습은 CNN과 CRF의 상호 적응을 통해 최적의 성능을 끌어내고, 분리 학습은 유연성과 해석 가능성을 제공합니다.
*   **향후 연구**: 가우시안 특징 학습의 잠재력, 전역 문맥 정보를 더 잘 포착하는 정교한 CRF 아키텍처 탐구, 인스턴스 분할 및 랜드마크 인식과 같은 다른 구조적 응용 분야에서의 ConvCRF 활용 가능성 탐색이 필요합니다.

## 📌 TL;DR
*   **문제**: 기존 CNN-CRF 모델은 의미론적 분할에서 느린 추론/학습 속도와 수동으로 설정해야 하는 CRF 파라미터 문제로 외면받았습니다.
*   **해결책**: 저자들은 완전 연결 CRF에 조건부 독립성 가정을 추가하여 메시지 전달 단계를 컨볼루션 연산으로 재구성하는 **컨볼루션 CRF (ConvCRF)**를 제안했습니다.
*   **주요 성과**: 이 방법은 추론/학습 속도를 100배 향상시키고, 모든 CRF 파라미터(가우시안 특징 포함)를 역전파로 쉽게 학습할 수 있게 했습니다. 합성 데이터와 PASCAL VOC에서 기존 FullCRF보다 향상된 성능과 정확한 메시지 전달을 통해 아티팩트 감소를 입증하여 CRF 모델의 실용성과 연구 가능성을 크게 높였습니다.