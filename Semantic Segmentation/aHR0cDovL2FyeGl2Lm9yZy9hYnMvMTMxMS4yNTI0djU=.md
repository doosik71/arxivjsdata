# Rich feature hierarchies for accurate object detection and semantic segmentation
Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik

## 🧩 Problem to Solve
기존 객체 탐지 성능은 지난 몇 년간 PASCAL VOC 데이터셋에서 정체되어 있었으며, 주로 복잡한 앙상블 시스템과 저수준 특징(SIFT, HOG 등)에 의존했습니다. 이 논문은 다음의 두 가지 핵심 문제를 해결하고자 합니다:
1.  **심층 네트워크를 이용한 객체 위치 파악**: 심층 컨볼루션 신경망(CNN)이 이미지 분류에 뛰어난 성능을 보였지만, 이미지 내 여러 객체의 정확한 위치를 파악하는 객체 탐지 작업에 어떻게 적용할 것인가?
2.  **훈련 데이터 부족 문제**: 대규모 CNN을 훈련하기에 충분한 주석이 달린 탐지 데이터가 부족할 때, 고용량 모델의 성능을 어떻게 향상시킬 것인가?

## ✨ Key Contributions
*   **새로운 객체 탐지 패러다임 제안 (R-CNN)**: 영역 제안(Region Proposals)과 고용량 CNN을 결합하여 객체 탐지 성능을 획기적으로 개선하는 R-CNN(Regions with CNN features) 방법을 제안했습니다.
*   **성능의 비약적 향상**: PASCAL VOC 2012 데이터셋에서 이전 최고 결과 대비 30% 이상 향상된 53.3%의 mAP를 달성했습니다. ILSVRC2013 탐지 데이터셋에서도 OverFeat을 크게 능가하는 31.4%의 mAP를 기록했습니다.
*   **전이 학습 (Transfer Learning)의 효과 입증**: 레이블된 훈련 데이터가 부족할 때, 대규모 보조 데이터셋(ImageNet)을 이용한 **지도 학습 사전 훈련(Supervised Pre-training)**과 대상 작업(객체 탐지)에 대한 **도메인 특화 미세 조정(Domain-specific Fine-tuning)**이 고용량 CNN 학습에 매우 효과적임을 입증했습니다. 미세 조정을 통해 mAP가 8%p 향상되었습니다.
*   **의미론적 분할(Semantic Segmentation)로의 확장**: 객체 탐지 시스템을 약간만 수정하여 PASCAL VOC 분할 작업에서도 경쟁력 있는 47.9%의 평균 분할 정확도를 달성했습니다.
*   **오류 분석 및 개선**: 탐지 오류의 주요 원인이 잘못된 위치 파악(mislocalization)임을 밝히고, 간단한 바운딩 박스 회귀(Bounding-box Regression) 방법을 통해 이를 크게 개선했습니다.

## 📎 Related Works
*   **전통적인 특징 기반 방법**: SIFT (Scale-Invariant Feature Transform) [29], HOG (Histograms of Oriented Gradients) [7] 등 블록 기반 방향 히스토그램.
*   **변형 가능한 파트 모델 (DPM)**: HOG 기반의 대표적인 객체 탐지 모델 [17, 20].
*   **CNN의 선구적 연구**: 후쿠시마의 Neocognitron [19], LeCun의 CNN 및 역전파 훈련 [26, 27].
*   **ImageNet 성공**: Krizhevsky 등 [25]의 대규모 ImageNet 분류 데이터셋에서의 CNN 성공.
*   **영역 기반 인식 패러다임**: "Recognition using regions" [21], Selective Search [39], CPMC [5] 등.
*   **슬라이딩 윈도우 탐지기**: 얼굴 [32, 40] 및 보행자 [35] 탐지를 위한 CNN 기반 방법.
*   **OverFeat**: 유사한 CNN 아키텍처를 기반으로 한 최근 제안된 슬라이딩 윈도우 탐지기 [34].
*   **오류 분석 도구**: Hoiem 등 [23]의 탐지기 오류 진단 도구.
*   **의미론적 분할**: O$^2$P (Second-Order Pooling) [4] 등.

## 🛠️ Methodology
R-CNN 객체 탐지 시스템은 세 가지 핵심 모듈로 구성됩니다:

1.  **영역 제안 (Region Proposals)**
    *   입력 이미지에서 약 2000개의 카테고리 독립적인 바텀업(bottom-up) 영역 제안을 생성합니다.
    *   논문에서는 주로 **Selective Search** [39]를 "fast mode"로 사용하여 후보 영역을 추출합니다. 이 방식은 이전 탐지 연구와의 비교를 용이하게 합니다.

2.  **CNN 특징 추출 (Feature Extraction)**
    *   각 영역 제안에 대해 고정된 길이의 특징 벡터를 추출하기 위해 **Krizhevsky 등 [25]이 제안한 대규모 CNN 아키텍처** (Caffe [24] 구현)를 사용합니다.
    *   CNN은 227$\times$227 RGB 이미지를 입력으로 받으므로, 임의의 모양을 가진 각 영역 제안의 픽셀 데이터를 필요한 크기로 변환해야 합니다. 이를 위해 해당 영역을 둘러싸는 경계 박스를 **워핑(Warping)**하여 CNN 입력 크기에 맞춥니다. 이때, 원래 박스 주위에 `p=16` 픽셀의 컨텍스트를 포함하도록 경계 박스를 확장한 후 워핑합니다.
    *   CNN의 다섯 개 컨볼루션 레이어와 두 개의 완전 연결 레이어를 통과시켜 4096차원 특징 벡터를 얻습니다.

3.  **클래스별 선형 SVM 분류기 (Class-specific Linear SVMs)**
    *   추출된 각 특징 벡터는 해당 클래스에 대해 훈련된 선형 SVM을 사용하여 점수화됩니다.
    *   이미지 내에서 점수가 매겨진 모든 영역에 대해, 각 클래스별로 독립적으로 **탐욕적인 비최대 억제(Greedy Non-Maximum Suppression, NMS)**를 적용하여 IoU(Intersection-over-Union) 임계값보다 높게 겹치는 저점수 영역을 제거합니다.

**훈련 과정:**
*   **지도 학습 사전 훈련 (Supervised Pre-training)**: CNN은 ImageNet Large Scale Visual Recognition Challenge (ILSVRC2012) 분류 데이터셋에서 이미지 레벨 주석만을 사용하여 사전 훈련됩니다.
*   **도메인 특화 미세 조정 (Domain-specific Fine-tuning)**: 사전 훈련된 CNN은 새로운 작업(탐지)과 새로운 도메인(워핑된 제안 윈도우)에 적응하기 위해 PASCAL VOC 데이터셋에서 **SGD (Stochastic Gradient Descent)**를 사용하여 미세 조정됩니다. CNN의 마지막 1000-way 분류 레이어는 (N+1)-way 분류 레이어 (N은 객체 클래스 수, +1은 배경)로 대체됩니다.
    *   미세 조정 시, 실제 바운딩 박스와 IoU 오버랩이 $\geq0.5$인 영역 제안은 해당 클래스의 긍정 예제로, 나머지는 배경(음성 예제)으로 간주합니다.
*   **객체 카테고리 분류기 훈련**: 각 클래스에 대해 하나의 선형 SVM이 최적화됩니다. 이때, IoU 오버랩 임계값 0.3 미만인 영역은 음성으로, 실제 바운딩 박스는 긍정으로 정의하며, 0.3에서 실제 바운딩 박스 사이의 "회색 영역"에 해당하는 영역은 무시됩니다.
    *   대규모 훈련 데이터를 처리하기 위해 **하드 네거티브 마이닝(Hard Negative Mining)** 기법이 사용됩니다.

**바운딩 박스 회귀 (Bounding-box Regression)**:
*   탐지된 바운딩 박스의 위치 오류를 줄이기 위해, 선택된 영역 제안의 `pool_{5}` 특징을 사용하여 새로운 탐지 윈도우를 예측하는 클래스별 선형 회귀 모델을 훈련합니다.

## 📊 Results
*   **PASCAL VOC 2010-12 객체 탐지**:
    *   VOC 2010 테스트셋에서 R-CNN은 50.2%의 mAP를 달성했으며, 바운딩 박스 회귀를 적용했을 때 **53.7%**로 향상되어, 이전 최고 성능인 SegDPM의 40.4% (UVA의 35.1%)를 크게 뛰어넘었습니다. 이는 기존 방법 대비 30% 이상의 상대적 성능 향상입니다.
    *   VOC 2011/12 테스트셋에서도 53.3% mAP를 달성했습니다.
*   **ILSVRC2013 객체 탐지**:
    *   200개 클래스 ILSVRC2013 탐지 데이터셋에서 R-CNN은 **31.4%**의 mAP를 달성하여, 이전 최고 성능인 OverFeat의 24.3%를 크게 앞섰습니다.
*   **성능 분석 (VOC 2007)**:
    *   **미세 조정의 효과**: 미세 조정을 통해 mAP가 46.2%에서 54.2%로 **8.0%p** 향상되었습니다. `pool_{5}` 특징은 일반적이며, 대부분의 성능 향상은 도메인 특화 비선형 분류기를 학습함으로써 얻어졌습니다.
    *   **층별 성능**: 미세 조정 없이도 CNN의 컨볼루션 레이어(특히 `pool_{5}`)가 높은 표현력을 가지며, 완전 연결 레이어 `fc_{7}`과 `fc_{6}`는 미세 조정 시 더 큰 성능 향상을 보입니다.
    *   **바운딩 박스 회귀 효과**: mAP를 3~4%p 향상시켜 위치 오류를 크게 줄였습니다.
*   **네트워크 아키텍처 영향**: Simonyan and Zisserman [43]의 16-레이어 "O-Net"을 사용했을 때 mAP가 58.5%에서 **66.0%**로 추가 향상되었으나, 계산 시간이 7배 증가하는 단점이 있습니다.
*   **의미론적 분할**:
    *   VOC 2011 테스트셋에서 `fc_{6}` (full+fg) 특징을 사용하여 **47.9%**의 평균 분할 정확도를 달성했으며, O$^2$P와 거의 동등하거나 약간 더 나은 성능을 보였습니다. `full` 특징과 `fg` 특징의 결합이 상호보완적임을 입증했습니다.

## 🧠 Insights & Discussion
*   **심층 학습과 전통적인 컴퓨터 비전의 시너지**: R-CNN은 컨볼루션 신경망이라는 딥러닝 기술과 영역 제안이라는 전통적인 컴퓨터 비전 기법을 성공적으로 결합했습니다. 이는 두 분야가 대립하기보다 상호 보완적인 관계임을 시사합니다.
*   **전이 학습의 중요성**: 대규모 보조 데이터셋(ImageNet)에서의 지도 학습 사전 훈련과 소규모 대상 데이터셋(PASCAL VOC)에서의 미세 조정 패러다임은 데이터 부족 문제를 해결하는 강력한 방법임을 입증했습니다. 이는 다양한 데이터 부족 시각 문제에 효과적인 해결책이 될 수 있습니다.
*   **CNN 특징의 높은 식별력**: 오류 분석 결과, R-CNN의 CNN 특징은 HOG와 같은 기존 특징보다 훨씬 더 식별력이 뛰어나며, 대부분의 오류는 배경 혼동이 아닌 잘못된 위치 파악에서 발생했습니다.
*   **효율성 및 확장성**: CNN의 파라미터가 모든 카테고리에서 공유되고 특징 벡터가 저차원이므로, R-CNN은 수천 개의 객체 클래스로 확장 가능하며, 계산 효율성도 뛰어납니다 (단, OverFeat의 슬라이딩 윈도우 방식보다는 느림).
*   **한계 및 미래 연구**: 현재 R-CNN은 OverFeat 대비 속도 면에서 불리하며, 미세 조정 과정에서의 긍정/음성 예제 정의와 SVM 사용의 최적화 가능성 등 추가 개선의 여지가 있습니다. 특히 미세 조정을 통해 최종 softmax 레이어의 성능을 SVM 수준으로 끌어올릴 수 있다면 시스템이 더욱 간소화될 수 있습니다.

## 📌 TL;DR
객체 탐지 성능의 정체기를 깨기 위해, R-CNN은 "영역 제안"과 "고용량 CNN"을 결합하여 이전 PASCAL VOC 최고 기록 대비 mAP를 30% 이상 향상시켰습니다. 특히, 대규모 데이터셋(ImageNet)에서 사전 훈련된 CNN을 소규모 탐지 데이터셋(PASCAL VOC)에 미세 조정하는 **전이 학습 전략**이 데이터 부족 문제를 해결하는 데 매우 효과적임을 입증했습니다. 이 방법은 객체 탐지뿐만 아니라 의미론적 분할에도 성공적으로 적용되었으며, 전통적인 컴퓨터 비전과 딥러닝의 효과적인 융합을 보여주었습니다.