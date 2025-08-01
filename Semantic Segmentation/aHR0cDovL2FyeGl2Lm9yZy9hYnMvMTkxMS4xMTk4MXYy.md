# Class-Conditional Domain Adaptation on Semantic Segmentation
Yue Wang, Yuke Li, James H. Elder, Runmin Wu, Huchuan Lu

## 🧩 Problem to Solve
의미론적 분할(Semantic Segmentation)은 자율 주행 등 다양한 애플리케이션에서 중요하지만, 픽셀 단위의 정답 레이블링은 비용이 많이 들고 훈련 데이터에 과적합되어 일반화 성능이 저하되는 경향이 있습니다. 비지도 도메인 적응(Unsupervised Domain Adaptation, UDA)은 이러한 문제를 해결하여 레이블링된 소스 도메인(예: 합성 데이터셋)에서 학습된 시스템이 레이블이 없는 타겟 도메인으로 적응하도록 돕습니다. 그러나 기존의 UDA 방식은 도메인 표현 분포를 정렬하는 데 초점을 맞추어 빈번하게 나타나지 않는 클래스(low-frequency classes)의 성능을 저하시키는 한계가 있었습니다.

## ✨ Key Contributions
*   클래스 조건부 도메인 이동 학습을 가능하게 하는 새로운 **클래스 조건부 다중 스케일 판별자(Class-Conditional Multi-scale Discriminator)**를 제안했습니다.
*   분할(Segmentation) 및 적응(Adaptation) 모두에 대해 **클래스 조건부 손실(Class-Conditional Loss)**을 균등하게 적용함으로써, 빈번하지 않은 클래스(less frequent classes)에 대한 성능을 더욱 향상시켰습니다.
*   두 가지 의미론적 분할 도메인 적응 시나리오(GTA5→Cityscapes, SYNTHIA→Cityscapes)에서 제안하는 방식이 **최신 알고리즘과 비교할 만한 성능**을 달성함을 입증했습니다.

## 📎 Related Works
*   **적대적 학습(Adversarial Learning) 기반 접근법:** 의미론적 분할을 위한 도메인 적응의 가장 일반적인 방식으로, 판별자(Discriminator)를 사용하여 예측 수준([35], [38]) 또는 특징 수준([13], [5], [21])에서 소스 및 타겟 도메인 표현을 정렬합니다.
*   **픽셀 수준 적응 및 자기 훈련(Pixel-level Adaptation & Self-training):**
    *   픽셀 수준 적응은 도메인 적응을 스타일 변환 문제로 보고, 한 도메인의 이미지를 다른 도메인의 '스타일'로 변환합니다([44], [32]).
    *   자기 훈련은 예측 확률이 높은 레이블 없는 타겟 샘플을 가상 정답 레이블(pseudo ground-truth labels)로 사용하여 모델을 업데이트합니다([48], [3]).
*   **클래스 또는 영역 조건부 적응(Class- or Region-conditioned Adaptation):** 일부 연구([5], [7])는 각 분할 클래스별로 별도의 도메인 분류기를 훈련하거나, 분류기 불일치를 사용하여 정렬이 불량한 영역에 더 많은 적대적 손실 가중치를 부여합니다([22]). [48]은 소수 클래스의 선택 불균형을 막기 위해 클래스 정규화된 신뢰도 점수를 사용합니다.
*   **기존 방법의 한계:** 이전 적대적 학습 적응 방법들은 클래스 정보를 고려하더라도, 클래스별 성능을 동일하게 고려하지 않아 클래스 빈도 불균형을 간과하는 경향이 있었습니다. 본 논문은 이러한 한계를 극복하기 위해 모든 클래스에 대한 동등한 주의를 적용합니다.

## 🛠️ Methodology
본 논문에서는 클래스 조건부 다중 스케일 판별자와 클래스 조건부 손실 함수를 활용하는 Class-Conditional Domain Adaptation (CCDA) 접근 방식을 제안합니다.

1.  **기본 도메인 적응 아키텍처:**
    *   **구성:** 특징 인코더($E$), 분할 디코더($S$), 판별자($D$)로 구성된 적대적 학습 프레임워크를 기반으로 합니다.
    *   **분할 손실($L_{seg}$):** 소스 도메인 이미지에 대한 픽셀 수준 교차 엔트로피 손실을 최소화합니다.
        $$L_{seg}(E,S) = - \sum_{h,w} \sum_{c} Y^{(h,w,c)}_s \log(P^{(h,w,c)}_s)$$
    *   **적대적 정렬 손실($L_{D1}$, $L_{adv1}$):** 판별자 $D$는 소스 및 타겟 도메인의 특징 표현을 구별하려 하고, 인코더 $E$는 $D$를 혼란시키려 합니다. 이는 클래스 빈도 불균형에 취약하며 중간 스케일에서만 도메인 이동을 포착합니다.

2.  **클래스 조건부 다중 스케일 판별자:**
    *   **미세-스케일(Fine-scale) 브랜치:** 픽셀 수준에서 정렬을 측정하며, 기본 아키텍처의 손실 함수(수식 2, 3)를 확장합니다. 각 클래스에 대한 이진 교차 엔트로피 손실($L_{cbce\_s}$)을 클래스별로 평균냅니다. 타겟 도메인의 경우, 픽셀 수준 예측 $P_t$에서 의사 레이블($\hat{Y}_t$)을 생성하고, 불확실한 픽셀($N_t$)에 더 큰 가중치를 부여하여 적응 성능을 향상시킵니다($L_{cbce\_t}$).
        $$L_{D\_fine}(D) = \beta L_{D1} + (1-\beta)L_{D2}$$
        $$L_{adv\_fine}(E,S) = \beta L_{adv1} + (1-\beta)L_{adv2}$$
    *   **거친-스케일(Coarse-scale) 브랜치:** 특징 스케일보다 더 거친 스케일에서 클래스 조건부 정렬을 측정하며, 각 패치(patch) 내의 모든 클래스에 동일한 주의를 기울입니다.
        *   **거친-스케일 클래스 레이블($W$):** 이미지 패치 내 각 클래스의 존재 유무를 나타내는 이진 벡터입니다. 소스 이미지의 경우 정답 레이블 $Y_s$에서 계산하고, 타겟 이미지의 경우 분할 모듈의 예측 $P_t$를 기반으로 특정 임계값($th_w$)을 넘으면 1로 설정합니다. 이는 패치 수준에서 클래스 빈도를 균등화하는 효과가 있습니다.
        *   **손실 함수:** 판별자의 출력은 $C$ 길이의 두 벡터 $O_s, O_t$이며, 도메인과 클래스 정보를 다중화합니다.
            *   **분류 손실:** $O_c = \sigma(O_s + O_t)$와 $W_c$ 간의 이진 교차 엔트로피 손실 $L_{bce}(O_c, W_c)$를 통해 분할 클래스 정보 보존을 장려합니다.
            *   **적대적 적응 손실:** $O_{st} = f([O_s, O_t])$를 사용하여 패치 내 픽셀이 소스 또는 타겟 도메인에서 왔을 확률을 나타냅니다. $L_{D\_coarse}$와 $L_{adv\_coarse}$는 이 손실을 패치에 존재하는 클래스에 대해 합산하여 클래스 전반에 걸쳐 손실을 균등하게 분배합니다.

3.  **클래스 조건부 분할 손실($L_{pred}$):**
    *   픽셀 수준 교차 엔트로피 손실의 단점(소수 클래스 무시)을 해결하기 위해, 다이스 손실(Dice Loss) [24]과 교차 엔트로피 손실을 혼합하여 사용합니다.
    *   다이스 손실은 각 클래스의 기여도를 대략적으로 균등화하여 소수 클래스에 대한 학습 성능을 개선합니다.
    $$L_{dice}(E,S) = 1 - \frac{1}{C} \sum_{c=1}^C \left( \frac{2 \sum_{h,w} Y^{(h,w,c)}_s P^{(h,w,c)}_s}{\sum_{h,w} (Y^{(h,w,c)}_s + P^{(h,w,c)}_s) + \epsilon} \right)$$
    *   최종 분할 예측 손실: $L_{pred}(E,S) = \alpha L_{seg} + (1-\alpha)L_{dice}$.

4.  **완전한 훈련 손실:**
    *   전체 훈련 과정은 클래스 조건부 분할 손실, 미세-스케일 및 거친-스케일 클래스 조건부 도메인 적응 판별자 손실, 그리고 미세-스케일 및 거친-스케일 도메인 적응 적대적 손실을 결합하여 최적화됩니다.

## 📊 Results
*   **데이터셋 및 태스크:** SYNTHIA [29] 및 GTA5 [28] (소스 도메인)와 Cityscapes [6] (타겟 도메인)를 사용하여 GTA5→Cityscapes 및 SYNTHIA→Cityscapes 두 가지 적응 태스크를 수행했습니다.
*   **성능 비교:**
    *   GTA5→Cityscapes 태스크 (표 1): 제안된 CCDA 방식은 선택된 모든 최신 방법보다 평균적으로 더 나은 성능을 보였으며, 특히 로드, 빌딩, 식생, 자동차 등 고빈도 클래스의 성능을 유지하면서도 **사인, 자전거 등 저빈도 클래스의 성능을 크게 향상**시켰습니다.
    *   SYNTHIA→Cityscapes 태스크 (표 2): CCDA는 mIoU (mean Intersection over Union)에서 다른 알고리즘에 비해 유리한 성능을 보였고, 전반적인 성능을 높였습니다. 자기 훈련 및 커리큘럼 학습 방법과 비교했을 때, 가장 빈번하지 않은 몇몇 클래스에서는 비슷한 결과를 보이면서도 전체적인 성능에서는 이들을 능가했습니다.
*   **어블레이션 연구(Ablation Studies) (표 3):**
    *   클래스 조건부 손실을 기본 아키텍처(분할 및 미세-스케일 판별자)에 추가했을 때 2.1%의 mIoU 개선이 있었습니다.
    *   설계된 거친-스케일 브랜치를 추가했을 때 추가로 0.7%의 mIoU 개선이 있었습니다.
    *   이는 클래스 기반 손실과 클래스 기반 거친-스케일 브랜치 모두의 효과를 검증합니다.
*   **정성적 결과 (그림 3):** CCDA 방식이 기준선(baseline) 구조에 비해 도로, 보도 등 고빈도 클래스에서 더 깨끗하고 정확한 예측을 제공하며, 조명, 표지판 등 저빈도 클래스의 성능도 개선함을 시각적으로 보여줍니다.

## 🧠 Insights & Discussion
*   **클래스 불균형 문제 해결:** 이 논문의 핵심 통찰은 의미론적 분할을 위한 도메인 적응에서 빈번하지 않은 클래스의 낮은 성능 문제를 해결하기 위해, 모델의 여러 지점(분할, 거친-스케일 및 미세-스케일 도메인 적응)에 **클래스 조건부(class-conditioning)**를 도입하고 계산의 여러 단계에서 **클래스 간 균등화(equalizing across classes)**를 적용했다는 점입니다.
*   **일반화 및 강건성 향상:** 이 방법을 통해 빈번하지 않은 클래스의 성능을 크게 높이면서 다른 클래스의 성능을 유지할 수 있었으며, 이는 광범위한 클래스에 대한 우수한 성능으로 이어져 궁극적으로 평균적으로 최신 기술을 능가하는 결과를 가져왔습니다. 이는 자율 주행과 같은 실제 애플리케이션에서 UDA의 강건성과 실용성을 높이는 데 기여합니다.
*   **의사 레이블링의 중요성:** 타겟 도메인의 불확실한 픽셀에 더 큰 가중치를 부여하는 전략은 도메인 이동으로 인해 방해받는 분류에 대한 적응을 집중시켜 성능을 향상시키는 데 기여합니다.

## 📌 TL;DR
*   **문제:** 기존 의미론적 분할을 위한 비지도 도메인 적응(UDA) 방법은 빈번하지 않은 클래스(소수 클래스)에 대한 성능이 저조했습니다.
*   **제안 방법:** 논문은 이 문제를 해결하기 위해 클래스 조건부 다중 스케일 판별자와 클래스 조건부 손실 함수를 포함하는 **클래스 조건부 도메인 적응(CCDA)** 방식을 제안합니다. 이 방식은 각 클래스에 대해 동일한 주의를 기울여 도메인 이동을 학습하고 손실을 균등화합니다.
*   **주요 결과:** 두 가지 주요 도메인 적응 태스크(GTA5→Cityscapes, SYNTHIA→Cityscapes)에서 최신 기술과 비교할 만한 성능을 달성했으며, 특히 **소수 클래스에 대한 분할 성능을 크게 향상**시켰습니다.