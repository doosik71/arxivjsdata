# Rethinking Atrous Convolution for Semantic Image Segmentation
Liang-Chieh Chen, George Papandreou, Florian Schroff, Hartwig Adam

## 🧩 Problem to Solve
*   **특징 해상도 감소:** 심층 컨볼루션 신경망(DCNN)은 반복적인 풀링 및 스트라이딩 연산으로 인해 특징 맵의 공간 해상도가 감소하며, 이는 상세한 공간 정보가 필수적인 의미론적 이미지 분할(semantic image segmentation)과 같은 밀집 예측(dense prediction) 작업에 불리하다.
*   **다중 스케일 객체 처리:** 이미지 내 객체들이 다양한 스케일로 존재하므로, 네트워크가 다중 스케일 컨텍스트를 효과적으로 캡처하는 것이 중요한 과제이다.

## ✨ Key Contributions
*   **Atrous Convolution 재조명 및 활용:** 필터의 시야(field-of-view)와 특징 응답의 해상도를 명시적으로 조절하는 강력한 도구인 atrous convolution을 의미론적 이미지 분할에 효과적으로 재조명하고 활용한다.
*   **다중 스케일 컨텍스트를 위한 모듈 설계:**
    *   **캐스케이드(Cascade) Atrous Convolution 모듈:** Atrous rate를 점진적으로 증가시키는 캐스케이드 구조를 통해 장거리 컨텍스트를 캡처하는 모듈을 설계한다. (Multi-grid method 포함)
    *   **개선된 Atrous Spatial Pyramid Pooling (ASPP) 모듈:** 다양한 atrous rate를 가진 병렬 atrous convolution을 사용하여 다중 스케일 정보를 탐색하며, 전역 컨텍스트를 인코딩하는 이미지 레벨 특징(image-level features)을 추가하여 성능을 더욱 향상시킨다.
*   **훈련 상세 및 경험 공유:** Batch Normalization의 중요성, `output_stride` 제어, ground truth 대신 로짓(logits) 업샘플링, 희귀 객체 처리를 위한 부트스트래핑(bootstrapping) 등 효과적인 시스템 훈련에 대한 실제적인 노하우를 공유한다.
*   **DeepLabv3 시스템 제안:** 제안된 'DeepLabv3' 시스템은 DenseCRF 후처리 없이도 기존 DeepLab 버전에 비해 성능을 크게 개선하였으며, PASCAL VOC 2012 의미론적 이미지 분할 벤치마크에서 다른 최신 모델들과 비교 가능한 수준의 성능을 달성한다.

## 📎 Related Works
*   의미론적 분할을 위한 컨텍스트 정보 활용에 대한 네 가지 주요 접근 방식을 언급한다 (Fig. 2 참조):
    *   **이미지 피라미드(Image pyramid):** 다양한 스케일의 입력에 모델을 적용하고 특징을 융합 (e.g., Farabet et al. [22], [55,12,11]).
    *   **인코더-디코더(Encoder-decoder):** 인코더에서 공간 정보를 압축하고 디코더에서 복원 (e.g., SegNet [3], U-Net [71], RefineNet [54]).
    *   **컨텍스트 모듈(Context module):** 네트워크 위에 추가 모듈을 캐스케이드하여 장거리 컨텍스트 인코딩 (e.g., DenseCRF [45,10], Dilated Convolution [90]).
    *   **공간 피라미드 풀링(Spatial pyramid pooling):** 여러 범위에서 특징 맵을 탐색하여 다중 스케일 컨텍스트 캡처 (e.g., DeepLabv2의 Atrous Spatial Pyramid Pooling (ASPP) [11], PSPNet [95]).
*   본 연구는 atrous convolution을 컨텍스트 모듈과 공간 피라미드 풀링의 프레임워크 내에서 주로 탐구하며, 모든 네트워크에 적용 가능하도록 일반성을 추구한다.

## 🛠️ Methodology
*   **Atrous Convolution을 이용한 밀집 특징 추출:**
    *   **정의:** 2차원 신호에 대해 출력 $y$, 필터 $w$, 입력 특징 맵 $x$가 주어질 때, atrous convolution은 다음과 같이 적용된다:
        $$y[i] = \sum_k x[i+r \cdot k]w[k]$$
        여기서 atrous rate $r$은 입력 신호를 샘플링하는 보폭에 해당하며, 필터 가중치 사이에 $r-1$개의 0을 삽입하여 업샘플링된 필터를 사용하는 것과 동일하다.
    *   **`output_stride` 제어:** Atrous convolution을 통해 DCNN에서 특징 응답이 계산되는 밀도를 명시적으로 제어할 수 있다. 예를 들어, `output_stride=16`을 위해 다운샘플링하는 마지막 풀링 또는 컨볼루션 레이어의 스트라이드를 1로 설정하고, 이후의 모든 컨볼루션 레이어는 `rate=2`의 atrous convolution으로 대체한다.
*   **Atrous Convolution을 통한 심층화 (Going Deeper):**
    *   ResNet의 마지막 블록(block4)을 `block5`, `block6`, `block7`으로 복제하여 캐스케이드로 배치한다.
    *   `output_stride=16` 설정 시, `block3` 이후부터 `rate>1`인 atrous convolution을 적용하여 신호가 감소되는 것을 방지한다.
    *   **Multi-grid Method:** `block4`에서 `block7`까지의 각 블록 내 3개 컨볼루션 레이어에 대해 단위 atrous rate $ (r_1, r_2, r_3) $를 정의하고, 이를 통해 다양한 atrous rate 조합을 실험한다.
*   **Atrous Spatial Pyramid Pooling (ASPP):**
    *   기존 [11]의 ASPP에 Batch Normalization [38]을 포함하고, 이미지 레벨 특징을 추가하여 개선한다.
    *   `3 \times 3` 필터에 큰 atrous rate를 적용할 경우, 유효한 필터 가중치 수가 줄어들어 사실상 `1 \times 1` 컨볼루션처럼 작동하는 문제(Fig. 4)를 해결하기 위해 이미지 레벨 특징을 도입한다.
    *   개선된 ASPP는 `1 \times 1` 컨볼루션 1개, `output_stride=16`일 때 `rate=(6,12,18)`을 가진 `3 \times 3` 컨볼루션 3개, 그리고 전역 평균 풀링 후 `1 \times 1` 컨볼루션과 이중 선형 업샘플링을 거친 이미지 레벨 특징으로 구성된다. 모든 브랜치는 256개 필터와 Batch Normalization을 사용한다.
*   **훈련 프로토콜 상세:**
    *   **학습률 정책:** `(1 - \frac{iter}{max\_iter})^{0.9}` 형태의 "poly" 정책 사용.
    *   **Crop Size:** `513 \times 513` 크기의 패치를 훈련 및 테스트에 사용하여 큰 atrous rate가 효과적으로 적용되도록 한다.
    *   **Batch Normalization:** ResNet 위에 추가된 모든 모듈에 BN 파라미터를 포함하고 함께 훈련하며, 큰 배치 크기(16)를 사용하여 BN 통계를 계산한다.
    *   **로짓 업샘플링:** Ground truth를 다운샘플링하는 대신 최종 로짓을 업샘플링하여 미세한 어노테이션 정보 손실을 방지한다.
    *   **데이터 증강:** 무작위 스케일링(0.5~2.0) 및 좌우 반전.
    *   **부트스트래핑:** 희귀하거나 세밀하게 어노테이션된 클래스(자전거, 의자, 테이블 등)를 포함하는 이미지를 훈련 세트에서 복제하여 재훈련한다.

## 📊 Results
*   **PASCAL VOC 2012 벤치마크:**
    *   `output_stride`는 `8`이 가장 우수한 성능을 보였으며, `output_stride`가 커질수록(즉, atrous convolution이 덜 적용될수록) 성능이 크게 저하되었다.
    *   ResNet-101이 ResNet-50보다 더 깊은 네트워크 구조에서 성능 이점을 보였다.
    *   Multi-grid 방법(`(1,2,1)` 등)이 기본 설정(`(1,1,1)`)보다 성능을 향상시켰다.
    *   ASPP 모듈에 이미지 레벨 특징을 추가하는 것이 `77.21%` mIOU를 달성하며 가장 효과적이었다.
    *   훈련 시 `output_stride=16`, 추론 시 `output_stride=8`로 전환하고, 다중 스케일 입력 및 좌우 반전 이미지 추론을 결합하여 `79.77%`의 최고 성능을 달성했다.
    *   DeepLabv2 (DenseCRF, MS-COCO 사전 훈련 포함)의 `77.69%`보다 우수한 `79.77%`를 달성하여 DeepLabv2 대비 성능 향상을 입증했다.
    *   MS-COCO 사전 훈련 시 `82.7%`, JFT-300M 사전 훈련 시 `86.9%`로 PASCAL VOC 2012 테스트 세트에서 SOTA 수준의 성능을 기록했다.
    *   부트스트래핑 방법은 자전거와 같은 희귀 클래스의 분할 정확도를 효과적으로 개선했다.
*   **Cityscapes 데이터셋:**
    *   `train_fine` 세트만으로 훈련 시 `79.30%` mIOU (멀티 스케일 및 플립 추론 포함).
    *   `train_val_coarse` 세트를 포함하여 훈련 시 `81.3%` mIOU를 달성하여 경쟁력 있는 성능을 보였다.

## 🧠 Insights & Discussion
*   **Atrous Convolution의 핵심 역할:** Atrous convolution은 DCNN이 특징 해상도를 유지하면서도 넓은 시야와 장거리 컨텍스트를 효과적으로 캡처할 수 있도록 하는 데 필수적이다. 이는 특히 심층 네트워크를 의미론적 분할에 적용할 때 신호 손실을 방지하여 성능을 크게 향상시킨다.
*   **다중 스케일 컨텍스트 인코딩의 중요성:** 캐스케이드 모듈의 Multi-grid 방법과 ASPP의 다양한 atrous rate 및 이미지 레벨 특징의 조합은 다중 스케일 객체들을 효과적으로 인코딩하여 분할 정확도를 높이는 데 결정적인 역할을 한다. 특히, 큰 atrous rate에서 발생하는 필터 유효 가중치 수 감소 문제를 이미지 레벨 특징으로 보완한 것이 중요했다.
*   **훈련 프로토콜의 영향력:** Batch Normalization 파라미터의 미세 조정, 충분히 큰 crop size 사용, 그리고 ground truth 대신 로짓을 업샘플링하는 훈련 기법들이 모델 성능에 매우 중요한 영향을 미친다는 점을 정량적으로 보여주었다.
*   **성능 향상의 주요인:** 기존 DeepLabv2 대비 성능 향상은 주로 Batch Normalization 파라미터의 미세 조정과 다중 스케일 컨텍스트를 인코딩하는 더 효과적인 방법(개선된 ASPP) 덕분이다.
*   **제한 사항:** 일부 객체들(소파 vs. 의자, 식탁과 의자)이나 객체의 희귀한 시점(rare view)을 분할하는 데는 여전히 어려움이 있다.

## 📌 TL;DR
의미론적 이미지 분할에서 DCNN의 특징 해상도 감소 및 다중 스케일 객체 처리 문제를 해결하기 위해, DeepLabv3는 atrous convolution을 재조명한다. 이 논문은 캐스케이드 atrous convolution 모듈과 이미지 레벨 특징이 추가된 개선된 Atrous Spatial Pyramid Pooling (ASPP) 모듈을 제안하여, 다양한 스케일의 컨텍스트를 성공적으로 포착한다. DenseCRF 후처리 없이 PASCAL VOC 2012 벤치마크에서 SOTA 수준의 성능(85.7%, JFT-300M 사전 훈련 시 86.9%)을 달성했으며, Batch Normalization 미세 조정과 충분한 crop size, 로짓 업샘플링 등의 훈련 기법이 성능 향상에 크게 기여했음을 강조한다.