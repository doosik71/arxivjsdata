# CANet: Class-Agnostic Segmentation Networks with Iterative Refinement and Attentive Few-Shot Learning
Chi Zhang, Guosheng Lin, Fayao Liu, Rui Yao, Chunhua Shen

## 🧩 Problem to Solve
최근 딥러닝 기반의 시맨틱 분할(Semantic Segmentation)은 대규모 레이블링된 데이터셋 덕분에 큰 발전을 이루었습니다. 그러나 픽셀 단위의 레이블링은 매우 지루하고 비용이 많이 들며, 한 번 학습된 모델은 미리 정의된 클래스 내에서만 예측을 수행할 수 있어 새로운 클래스에는 적용하기 어렵습니다. 이 논문은 이러한 한계를 해결하기 위해, 단 몇 개의 주석이 달린 이미지로도 새로운 클래스에 대한 분할을 수행할 수 있는 "Few-shot Semantic Segmentation"이라는 연구 문제를 다룹니다.

## ✨ Key Contributions
*   **새로운 이중 분기 Dense Comparison Module (DCM) 개발:** CNN의 다양한 레벨의 특징 표현을 효과적으로 활용하여 밀집 특징(dense feature) 비교를 수행합니다.
*   **Iterative Optimization Module (IOM) 제안:** 예측된 결과를 반복적으로 개선(refine)하며, 이 반복 개선 능력은 소수의 예시만으로도 보지 못했던 클래스에 대해 미세한 분할 맵을 생성할 수 있도록 일반화됩니다.
*   **어텐션(Attention) 메커니즘 채택:** $k$-shot 설정에서 여러 지원 예시(support example)의 정보를 효과적으로 융합하여, 1-shot 결과의 학습 불가능한(non-learnable) 융합 방식보다 우수한 성능을 보입니다.
*   **약한 주석(weak annotations) 지원:** 지원 세트에 픽셀 단위 주석 대신 바운딩 박스(bounding box) 주석이 주어져도 필셀 단위 주석과 비슷한 성능을 달성하여, 새로운 클래스에 대한 레이블링 노력을 크게 줄일 수 있음을 입증합니다.
*   **최첨단 성능 달성:** PASCAL VOC 2012 데이터셋에서 1-shot 분할에서 55.4%의 평균 Intersection-over-Union (IoU) 점수, 5-shot 분할에서 57.1%를 달성하여 기존 최첨단 방법들을 각각 14.6% 및 13.2%의 큰 차이로 능가합니다.

## 📎 Related Works
*   **시맨틱 분할 (Semantic Segmentation):** FCN(Fully Convolutional Networks) 기반의 방법론들이 주를 이루며, Dilated Convolution 등을 활용하여 특징 맵의 해상도를 유지합니다. 그러나 이러한 모델들은 대량의 픽셀 단위 주석을 요구하며, 학습된 후에는 새로운 카테고리에 대한 분할이 불가능합니다.
*   **Few-shot Learning:** 순환 신경망, 미세 조정(fine-tuning), 네트워크 파라미터 예측, 메트릭 학습(metric learning) 등 다양한 접근 방식이 있습니다. 본 연구는 이미지 분류에 사용되는 Relation Network [37]와 같은 메트릭 학습 기반 방법과 가장 관련이 깊으며, 이를 밀집 예측(dense prediction) 문제로 확장합니다.
*   **Few-shot Semantic Segmentation:** 이전 연구들(Shaban et al. [29], Rakelly et al. [24])은 지원 분기(support branch)와 쿼리 분기(query branch)를 포함하는 이중 분기 구조를 사용합니다. 그러나 이들은 대개 1-shot 설정에 중점을 두었으며, $k$-shot으로 확장할 때는 학습 불가능한 융합 방식을 사용했습니다. CANet은 두 분기가 동일한 백본 네트워크를 공유하고, 학습 가능한 어텐션 메커니즘을 사용하여 여러 지원 예시를 융합하는 차이점을 가집니다.

## 🛠️ Methodology
CANet은 Few-shot Semantic Segmentation 문제를 해결하기 위한 새로운 프레임워크로, Dense Comparison Module (DCM)과 Iterative Optimization Module (IOM)의 두 가지 핵심 모듈로 구성됩니다. $k$-shot 학습을 위해 어텐션 메커니즘을 추가합니다.

*   **Dense Comparison Module (DCM):**
    *   **특징 추출기 (Feature Extractor):** ResNet-50을 백본으로 사용하며, ImageNet에서 사전 학습된 가중치는 학습 중 고정됩니다. 보지 못했던 클래스에서도 유용한 중간 레벨의 특징(`block2`와 `block3`의 특징)을 선택하여 비교에 사용합니다. 특징 맵의 공간 해상도를 유지하기 위해 `block2` 이후의 레이어에서 Dilated Convolution을 사용합니다. 지원 분기와 쿼리 분기는 동일한 특징 추출기를 공유합니다.
    *   **밀집 비교 (Dense Comparison):** 지원 이미지에서 타겟 카테고리에 해당하는 특징만을 추출하기 위해, 전경(foreground) 영역에 대한 전역 평균 풀링(Global Average Pooling)을 사용하여 특징 벡터를 얻습니다. 이 벡터는 쿼리 특징 맵과 동일한 공간 크기로 업샘플링된 후 쿼리 특징과 연결되어 밀집 비교를 수행합니다.

*   **Iterative Optimization Module (IOM):**
    *   초기 예측이 객체의 대략적인 위치에 대한 중요한 단서가 됨에 착안하여, 예측된 결과를 반복적으로 개선합니다.
    *   DCM의 출력 특징 $x$와 이전 반복 단계의 예측 마스크 $y_{t-1}$를 입력으로 받습니다.
    *   잔차(residual) 형태의 구조 $M_t = x + F(x, y_{t-1})$를 사용하여 예측 마스크를 효율적으로 통합합니다. $F(\cdot)$는 특징 $x$와 예측 마스크 $y_{t-1}$의 연결(concatenation) 후 컨볼루션 블록으로 구성됩니다.
    *   Multi-scale 정보를 캡처하기 위해 DeepLab V3의 Atrous Spatial Pyramid Pooling (ASPP) 모듈을 사용합니다.
    *   최종 마스크는 1x1 컨볼루션과 소프트맥스 함수를 통해 전경 및 배경 신뢰도 맵으로 생성되며, 다음 IOM으로 피드백됩니다.
    *   학습 시 과적합을 방지하기 위해, $y_{t-1}$는 확률 $p_r=0.7$로 빈 마스크로 재설정됩니다 (마스크 드롭아웃).

*   **$k$-shot 분할을 위한 어텐션 메커니즘 (Attention Mechanism):**
    *   DCM의 밀집 비교 컨볼루션과 병렬로 어텐션 모듈을 추가합니다.
    *   어텐션 분기는 각 지원 예시에 대한 가중치 $\lambda_i$를 계산하고, 이 가중치들은 소프트맥스 함수 $\hat{\lambda}_i = \frac{e^{\lambda_i}}{\sum_{j=1}^k e^{\lambda_j}}$로 정규화됩니다.
    *   최종 출력은 여러 지원 샘플에서 생성된 특징들의 가중 합(weighted sum)입니다.

*   **바운딩 박스 주석 (Bounding Box Annotations):**
    *   지원 세트에서 픽셀 단위 주석 대신 바운딩 박스 주석을 사용합니다. 바운딩 박스 영역 전체를 전경으로 간주하여 모델의 견고성(robustness)을 평가합니다.

*   **학습 상세:** 평균 교차 엔트로피(cross-entropy) 손실 함수를 사용하며, SGD 옵티마이저로 200 에포크 동안 학습합니다. 추론 시에는 초기 예측 후 4번의 반복 최적화를 수행합니다.

## 📊 Results
*   **평가 데이터셋:** PASCAL VOC 2012 (PASCAL-5i) 및 COCO 데이터셋.
*   **평가 지표:** `meanIoU` (클래스별 IoU 평균)와 `FB-IoU` (전경/배경 IoU 평균). 클래스 불균형과 배경 IoU의 오도 가능성 때문에 `meanIoU`를 주요 분석 지표로 사용합니다.
*   **PASCAL-5i 최첨단 방법 비교:**
    *   **1-shot:** 55.4% meanIoU (기존 SOTA 40.8% 대비 14.6% 향상).
    *   **5-shot:** 57.1% meanIoU (기존 SOTA 43.9% 대비 13.2% 향상).
    *   `FB-IoU` 지표에서도 기존 모델들을 크게 능가합니다.
*   **바운딩 박스 주석 실험:** 픽셀 단위 주석(54.0% meanIoU)과 비교하여 바운딩 박스 주석(52.0% meanIoU)으로도 유사한 성능을 달성하여, 비용 효율적인 레이블링의 가능성을 보여줍니다.
*   **Ablation Study (요소별 성능 분석):**
    *   **비교를 위한 특징 선택:** `block2`와 `block3`의 특징 조합이 가장 좋은 성능(51.2% meanIoU, IOM 적용 전)을 보였습니다. 중간 레벨 특징이 클래스 불특정 객체 부분 매칭에 효과적임을 시사합니다.
    *   **Iterative Optimization Module (IOM):** IOM을 적용하면 초기 예측(CANet-Init, 51.2%) 대비 2.8% 포인트 향상된 54.0% meanIoU를 달성하며, 기존 후처리 방법인 DenseCRF(51.9%)보다 우수한 성능을 보입니다. IOM은 객체 영역을 채우고 불필요한 영역을 제거하는 학습 가능한 방식으로 작동합니다.
    *   **$k$-shot 융합 방식:** 제안된 어텐션 메커니즘(55.8% meanIoU)은 특징 평균(Feature-Avg, 55.0%), 마스크 평균(Mask-Avg, 54.5%), 논리 OR 융합(Mask-OR, 53.4%) 등 학습 불가능한 다른 융합 방식보다 가장 좋은 성능을 보였습니다.
*   **COCO 데이터셋 결과:** PASCAL-5i와 유사하게 반복 최적화 및 어텐션 메커니즘의 효과를 확인했으며, Multi-scale 평가를 통해 추가적인 성능 향상을 달성했습니다.

## 🧠 Insights & Discussion
*   **DCM의 효과:** 중간 레벨의 특징을 활용한 밀집 비교는 보지 못했던 클래스의 공통 객체 부분을 매칭하는 데 매우 효과적입니다. 전경 영역에 대한 전역 평균 풀링은 타겟 카테고리에 집중하는 데 도움을 줍니다.
*   **IOM의 중요성:** IOM은 단순한 후처리 단계를 넘어, 반복적인 개선을 모델 내에 통합하여 객체 영역을 점진적으로 채우고 배경과의 구분을 명확히 하는 학습 가능한 방식을 제공합니다. 이는 특히 Few-shot 설정에서 초기 부정확한 예측을 보완하는 데 중요합니다.
*   **어텐션의 우수성:** $k$-shot 시나리오에서 학습 가능한 어텐션 메커니즘은 여러 지원 예시의 정보를 비학습적(non-learnable) 방법보다 훨씬 효과적으로 융합할 수 있음을 입증했습니다. 이는 모델이 각 지원 예시의 중요도를 동적으로 파악하고 가중치를 부여할 수 있게 해주기 때문입니다.
*   **실용성 향상:** 바운딩 박스 주석으로도 준수한 성능을 달성한 것은 Few-shot Semantic Segmentation의 실제 적용 가능성을 크게 높입니다. 픽셀 단위 레이블링의 높은 비용 없이도 새로운 클래스를 신속하게 추가할 수 있게 됩니다.
*   **한계 및 향후 방향:** 이 논문은 Few-shot Semantic Segmentation 분야에서 상당한 발전을 이루었지만, 여전히 완전 지도 학습(fully supervised learning)과의 성능 차이가 존재합니다. 또한, 학습 과정에서 중간 레벨 특징의 중요성을 강조하고 있지만, 특정 상황에서 최적의 특징 선택에 대한 추가 연구가 필요할 수 있습니다.

## 📌 TL;DR
*   **문제:** 픽셀 단위 시맨틱 분할은 높은 레이블링 비용과 새로운 클래스에 대한 일반화 능력 부족이라는 한계를 가집니다.
*   **방법:** CANet은 Class-Agnostic 분할을 위해 두 가지 핵심 모듈을 제안합니다: 1) 지원 이미지와 쿼리 이미지 간의 다중 레벨 특징을 밀집 비교하는 Dense Comparison Module (DCM), 2) 예측 결과를 반복적으로 개선하는 Iterative Optimization Module (IOM). $k$-shot 학습을 위해 어텐션 메커니즘을 사용하여 여러 지원 예시의 정보를 효과적으로 융합합니다.
*   **주요 결과:** PASCAL VOC 2012 데이터셋에서 1-shot 분할 55.4% mIoU, 5-shot 분할 57.1% mIoU를 달성하여 기존 최첨단 방법들을 크게 능가합니다. 또한, 비용이 저렴한 바운딩 박스 주석으로도 픽셀 단위 주석에 필적하는 성능을 보여 실용성을 높였습니다.