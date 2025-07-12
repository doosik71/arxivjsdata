# Deep Learning Markov Random Field for Semantic Segmentation
Ziwei Liu, Xiaoxiao Li, Ping Luo,Member, IEEE, Chen Change Loy,Senior Member, IEEE, and Xiaoou Tang,Fellow, IEEE

## 🧩 Problem to Solve
시맨틱 분할(semantic segmentation)은 각 픽셀에 카테고리 라벨을 할당하는 근본적인 컴퓨터 비전 문제이다. 기존의 마르코프 랜덤 필드(MRF) 또는 조건부 랜덤 필드(CRF) 기반 접근법은 픽셀 간의 강력한 상관관계를 모델링하여 문맥 정보를 포착하는 데 효과적이지만, 다음과 같은 한계가 있었다.
*   **얕은 모델의 한계:** Unary term을 모델링하는 데 사용된 SVM이나 Adaboost와 같은 얕은 모델은 학습 능력에 제약이 있었다.
*   **비효율적인 추론:** 복잡한 Pairwise term을 포함하는 MRF의 추론(예: Mean Field 알고리즘)은 반복적인 계산이 필요하여 최적화 및 추론 속도 측면에서 비효율적이었다. 특히, 최근의 CNN과 MRF를 공동으로 학습하는 모델들($[24], [26]$)은 역전파 중 매 학습 이미지마다 MRF의 반복적인 추론을 요구하여 계산 비용이 매우 높았다.
*   **복잡한 Pairwise Term 통합의 어려움:** 이러한 비효율성 때문에 복잡한 Pairwise term을 MRF에 통합하는 것이 비실용적이었다.

본 연구는 이러한 문제들을 해결하고, 고차 관계 및 라벨 컨텍스트 혼합을 MRF에 통합하며, 효율적이고 결정론적인 종단 간 계산을 단일 순방향 패스(forward pass)로 가능하게 하는 딥러닝 기반 시스템을 제안한다.

## ✨ Key Contributions
*   **새로운 DPN(Deep Parsing Network) 제안:** Unary term을 위한 VGG$_{16}$ 네트워크와 풍부한 Pairwise 정보(라벨 컨텍스트 혼합 및 고차 관계)를 공동으로 학습하는 새로운 DPN 모델을 제안한다.
*   **효율적인 추론:** 기존 딥 모델들이 역전파 중 Mean Field(MF) 추론에 많은 반복이 필요했던 것과 달리, DPN은 단 한 번의 MF 추론 근사만으로 높은 성능을 달성하여 계산 비용을 크게 절감한다.
*   **통합 프레임워크:** DPN은 다양한 유형의 Pairwise term을 표현할 수 있어, RNN $[26]$ 및 DeepLab $[23]$과 같은 많은 기존 MRF 기반 딥 모델들이 DPN의 특수한 경우임을 보여준다.
*   **고차원 데이터 처리:** DPN의 Pairwise term은 이미지 및 비디오와 같은 고차원 데이터에서 풍부한 문맥 정보를 인코딩하는 통합 프레임워크를 제공한다.
*   **병렬화 및 가속화 용이성:** DPN은 MF를 합성곱 및 풀링 연산으로 근사화하여 병렬화 및 GPU 가속화가 용이하며, 효율적인 추론을 가능하게 한다.
*   **N-차원 MRF 모델링:** 2D 이미지 분할에서 나아가 N-차원 고차 MRF를 모델링하고 해결하는 일반적인 딥러닝 프레임워크를 제시하며, 동적 노드 연결을 사용하여 N-D 공간에서 그래프를 구성한다.
*   **최첨단 성능 달성:** PASCAL VOC 2012, Cityscapes, CamVid 데이터셋 등 표준 시맨틱 이미지/비디오 분할 벤치마크에서 단일 DPN 모델이 최첨단 분할 정확도를 달성함을 입증한다.

## 📎 Related Works
*   **Markov Random Field (MRF) / Conditional Random Field (CRF) 기반 연구:**
    *   **초기 MRF/CRF:** 시맨틱 분할에서 뛰어난 성공을 거두었으며, 장거리 의존성 $[18], [37]$, 고차 포텐셜 $[19], [38]$, 의미론적 라벨 컨텍스트 $[3], [39], [20]$와 같은 풍부한 정보를 활용하여 라벨링 정확도를 개선했다. 예를 들어, Kr\"{a}henb\"{u}hl et al. $[18]$은 완전 연결 그래프를 통해 정확한 경계 분할을 달성했고, Vineet et al. $[19]$은 고차 및 장거리 픽셀 간 관계를 정의했다.
    *   **한계:** 이러한 방법들은 Unary term을 SVM 또는 Adaboost와 같은 얕은 모델로 모델링하여 학습 능력에 병목 현상이 있었고, 복잡한 Pairwise term의 학습 및 추론 비용이 높았다.
    *   **비디오 시맨틱 분할에서의 MRF/CRF:** 그래프 구조를 시공간 도메인으로 확장하여 활용되었다. (예: Wang et al. $[40]$은 전경 객체 분할, 추적, 가려짐 추론을 MRF 모델로 통합). 그러나 이들은 수동으로 제작된 특징에 기반하여 충분한 학습 능력이 부족했다.
*   **Convolutional Neural Network (CNN) 기반 연구:**
    *   **강력한 Unary 분류기:** 최근 CNN은 강력한 Unary 분류기로 활용되어 단순한 Pairwise 함수를 사용하거나 이를 무시하고도 고무적인 분할 결과를 보여주었다 (예: Long et al. $[6]$은 CNN의 완전 연결 계층을 합성곱 계층으로 변환하여 픽셀별 분류 가능).
    *   **CNN과 MRF의 결합:**
        *   **DeepLab $[23]$:** CNN의 출력을 단순한 Pairwise 잠재력을 가진 MRF에 입력했지만, CNN과 MRF를 별개의 구성 요소로 처리했다.
        *   **공동 학습 노력 $[24], [26]$:** MRF 추론 오류를 CNN으로 역전파하여 CNN과 MRF를 공동으로 훈련하는 진전이 있었다. 하지만, 역전파 과정에서 각 훈련 이미지에 대해 Mean Field(MF) 알고리즘과 같은 MRF의 반복적인 추론이 필요하여 계산 비용이 높았다. Zheng et al. $[26]$은 MF 추론 과정을 순환 신경망(RNN)으로 표현할 수 있음을 보여주었지만, 계산 비용은 여전히 높았다.
    *   **비디오 시맨틱 분할에서의 딥러닝:** SegNet $[44]$와 같은 시도도 있었지만, 시간적 관계를 고려하지 않는 경우가 많았다.

## 🛠️ Methodology
본 논문은 N-차원 고차 MRF를 모델링하고 해결하기 위한 통합 프레임워크인 DPN을 개발한다. DPN은 VGG$_{16}$을 확장하여 Unary term을 모델링하고, 추가 계층을 신중하게 설계하여 Pairwise term을 근사화한다.

1.  **마르코프 랜덤 필드 (MRF) 정식화:**
    *   MRF의 에너지 함수는 Unary term $\Phi(y^u_i)$ (픽셀 $i$에 라벨 $u$를 할당하는 비용)과 Pairwise term $\Psi(y^u_i, y^v_j)$ (픽셀 쌍 $(i,j)$에 라벨 $u,v$를 할당하는 패널티)의 합으로 정의된다:
        $$ E(y) = \sum_{\forall i \in V} \Phi(y^u_i) + \sum_{\forall (i,j) \in E} \Psi(y^u_i, y^v_j) $$
    *   **동적 노드 연결(Dynamic Node Linking):** 기존의 정적 격자(grid) 방식 대신, 시공간 문맥 정보를 더 잘 보존하기 위해 동적 노드 연결을 사용한다. 공간 도메인에서는 2D 구조를 유지하고, 시간 도메인에서는 광학 흐름(optical flow)으로 추정된 동일한 시간 궤적 $\Delta_{i \to j}$ 상에 있는 복셀들을 이웃으로 정의한다:
        $$ (i,j) \in E_t \iff j = i + \Delta_{i \to j} $$
    *   **Unary Term:** VGG$_{16}$으로 모델링되며, 픽셀 $i$에 라벨 $u$가 존재할 확률 $p^u_i$를 기반으로 한다:
        $$ \Phi(y^u_i) = - \ln p(y^u_i = 1|I) $$
    *   **Pairwise Term (고차 관계 및 라벨 컨텍스트 혼합):** 기존의 단순한 Pairwise term의 한계를 극복하기 위해, DPN은 풍부한 복셀 간 정보를 활용하는 Smoothness term을 정의한다:
        $$ \Psi(y^u_i,y^v_j) = \sum_{k=1}^K \lambda_k \mu_k(i,u,j,v) \sum_{\forall z \in N_j} d(j,z)p^v_j p^v_z $$
        *   **Local Label Contexts 혼합 ($ \mu_k(i,u,j,v) $):** 라벨 할당 비용을 지역 큐브에서 학습하며, $K$는 혼합 성분의 수를 나타낸다. 이는 픽셀 $i$와 이웃 $j$ 사이의 상대적 위치에 따른 라벨링 비용을 출력한다.
        *   **Triple Penalty ($ \sum_{\forall z \in N_j} d(j,z)p^v_z $):** 복셀 $i, j$ 및 $j$의 이웃을 포함하는 삼중 패널티를 모델링한다. 이는 $(i,u)$와 $(j,v)$가 호환되면, $(i,u)$가 $j$의 인접 픽셀 $(z,v)$와도 호환되어야 함을 암시한다. $d(j,z)$는 $j$와 $z$ 사이의 시각적/공간적 거리를 나타낸다.

2.  **추론 (Mean Field 근사):**
    *   MF 알고리즘은 MRF의 결합 분포를 추정하며, 최종 닫힌 형태의 해는 $q^u_i$를 반복적으로 계산하여 얻을 수 있다. DPN은 이 MF 업데이트의 단 한 번의 반복을 두 단계로 근사화한다.
        *   **단계 1 (Triple Penalty 근사):** $\sum_{\forall z \in N_j} d(j,z)p^v_j p^v_z$ 항은 $m \times m \times T_m$ 필터를 사용하여 지역 합성곱(local convolution)으로 구현된다. 이는 복셀 $j$의 예측을 이웃 복셀과의 거리에 따라 평활화한다 (Fig. 2(c)).
        *   **단계 2 (Label Contexts 근사):** $ \mu_k(i,u,j,v) $ 항은 $n \times n \times T_n$ 필터를 사용하여 합성곱(convolution)으로 구현된다. 이는 삼중 관계에 대한 패널티를 적용하여 라벨 컨텍스트를 학습한다 (Fig. 2(d)).

3.  **DPN 아키텍처:**
    *   **Unary Term 모델링:** DPN은 VGG$_{16}$의 파라미터를 초기화에 활용한다. VGG$_{16}$의 max pooling 계층(a8, a10)을 제거하여 해상도를 높이고, 완전 연결(fully-connected) 계층(a11)을 합성곱 계층(b9, b10)으로 변환하여 픽셀별 분류를 가능하게 한다. 최종적으로 b11은 각 카테고리의 512x512 확률적 라벨 맵을 생성한다.
    *   **Smoothness Term 모델링 (b12~b15 계층):**
        *   **b12 (3D Locally Convolutional Layer):** 3D 지역 합성곱 계층으로, 각 공간 위치에 고유한 필터를 가지며 Triple Penalty를 구현한다. 이 필터는 픽셀 간의 거리(예: RGB 값)를 기반으로 초기화되며, 픽셀의 확률을 이웃 픽셀의 가중 평균으로 업데이트한다.
        *   **b13 (3D Global Convolutional Layer):** 3D 전역 합성곱 계층으로, 여러 개의 필터를 사용하여 Local Label Contexts의 혼합을 학습한다.
        *   **b14 (Block Min Pooling Layer):** Pairwise term의 결과에서 가장 작은 패널티를 가진 컨텍스트 패턴을 활성화한다.
        *   **b15 (Summation Layer):** Unary term(b11의 출력)과 Smoothness term(b14의 출력)을 Eqn. (15)와 유사하게 결합하고 소프트맥스 정규화를 통해 최종 라벨 확률을 얻는다.

4.  **학습 알고리즘:**
    *   VGG$_{16}$의 처음 10개 그룹은 사전 학습된 가중치로 초기화되고, 마지막 4개 그룹은 무작위로 초기화된다.
    *   **증분 학습(Incremental Learning):** 4단계로 점진적으로 미세 조정된다.
        1.  Unary term 학습 (b1~b11).
        2.  Triple Penalty 학습 (b12 추가 및 파라미터 업데이트).
        3.  Label Contexts 학습 (b13, b14 추가 및 파라미터 업데이트).
        4.  모든 파라미터를 공동으로 미세 조정.
    *   **효율성:** DPN은 MF 추론을 합성곱 및 풀링 연산으로 변환하여 GPU에서 병렬화 및 가속화가 용이하다. 기존 MF 구현 대비 10배 이상의 런타임 단축을 달성한다. 룩업 테이블 기반 필터링으로 지역 합성곱 계산을 가속화한다.

## 📊 Results
DPN은 PASCAL VOC 2012, Cityscapes, CamVid 데이터셋에서 평가되었으며, mIoU(mean Intersection-over-Union), TA(Tagging Accuracy), LA(Localization Accuracy), BA(Boundary Accuracy)를 포함한 다양한 지표로 성능이 측정되었다.

*   **Pairwise Term 효과:**
    *   **Triple Penalty:** b12의 수용 필드 50x50이 가장 좋은 mIoU를 보였으며, `baseline`(VGG$_{16}$ + denseCRF)보다 성능이 향상되었다. 이는 픽셀 간 관계 포착에 50x50 이웃이 충분함을 시사한다.
    *   **Label Contexts:** b13의 '9x9 mixtures' 설정이 가장 좋은 성능을 달성했으며, 이는 지역 공간 컨텍스트를 고려하지 않은 '1x1' 설정보다 mIoU를 크게 개선했다. DPN의 Pairwise term은 DSN 및 DeepLab보다 효과적임을 입증했다.
*   **시공간 DPN의 효과 (CamVid 데이터셋):**
    *   3D Pairwise term(b12: 50x50x3, b13: 7x7x3)은 2D Pairwise term보다 약간 더 나은 성능을 보여주며, 연속 프레임 간의 정보를 효과적으로 포착함을 입증했다.
    *   학습된 3D 라벨 호환성 및 시공간 문맥 패턴 시각화를 통해, DPN이 `sky`, `tree`, `road`와 같이 연속 프레임에서 유연한 모양을 가진 객체들의 시간적 정규화를 성공적으로 수행함을 확인했다.
*   **학습 전략 분석:**
    *   **증분 학습:** 동시 학습(joint learning)보다 더 높은 정확도를 달성하고 국소 최저점에 빠질 가능성이 적음을 보여, 복잡한 모델의 최적화에 증분 학습이 더 안정적임을 시사한다.
    *   **단일 MF 반복:** DPN은 단 한 번의 MF 반복만으로도 좋은 정확도에 도달하며, 이는 기존 CRF $[18]$ 및 다른 딥 모델들(5~10회 반복 필요)에 비해 계산 효율성을 크게 개선했음을 입증한다.
*   **단계별 및 클래스별 분석:** Unary term, Triple Penalty, Label Contexts, Joint Tuning 등 각 단계가 mIoU, LA, BA에서 점진적인 성능 향상에 기여함을 확인했다. 특히 Joint Tuning은 대부분의 클래스에서 이점을 제공했지만, 매우 작은 객체의 경우 분할 정확도를 위해 희생될 수 있음이 관찰되었다. `chair`, `table`, `plant`와 같은 클래스는 경계 및 바운딩 박스 정확도의 어려움 때문에 mIoU가 낮았다.
*   **벤치마크 결과:**
    *   **PASCAL VOC 2012:** 외부 학습 데이터 없이 74.1% mIoU를 달성했으며, COCO 데이터셋으로 사전 학습된 DPN$^\dagger$는 77.5% mIoU로 최첨단 성능을 기록했다.
    *   **Cityscapes:** 66.8% mIoU를 달성하여 기존 방법 중 두 번째로 높은 성능을 보였으며, `road`, `building`, `vegetation` 등 불규칙한 형태의 객체에서 강점을 보였다.
    *   **CamVid:** 60.06% mIoU로 모든 기존 방법을 능가했으며, 시공간 Pairwise term을 적용한 Spatial-temporal DPN은 60.25%로 더욱 개선되었다. 특히 `pole`, `sign`과 같이 좁고 작은 객체에서 우수한 성능을 보였다.

## 🧠 Insights & Discussion
*   **혁신적인 효율성:** DPN의 가장 큰 통찰은 복잡한 MRF 추론(Mean Field)을 단 한 번의 순방향 합성곱 및 풀링 연산으로 근사화하여 딥러닝 프레임워크 내에서 End-to-End 학습 및 추론을 가능하게 했다는 점이다. 이는 기존 CNN-MRF 결합 모델의 주요 단점인 반복적인 추론으로 인한 계산 비용 문제를 해결하며, 실제 애플리케이션에서의 효율적인 배포 가능성을 크게 높였다.
*   **풍부한 문맥 정보 모델링:** DPN의 Pairwise term은 `Triple Penalty`와 `Mixture of Local Label Contexts`를 도입하여 기존 모델들이 놓쳤던 고차 관계 및 지역적 공간 컨텍스트를 효과적으로 포착한다. 이는 복잡한 객체 간의 상호작용과 지역적 패턴을 심층적으로 학습함으로써, 시맨틱 분할의 정확도를 향상시키는 데 결정적인 역할을 했다.
*   **일반화 및 유연성:** DPN은 2D 이미지뿐만 아니라 3D 비디오 데이터에 대해서도 시공간적 문맥 정보를 활용하여 우수한 성능을 보여주었다. 이는 N-차원 MRF를 모델링할 수 있는 일반적인 프레임워크로서 DPN의 강력한 일반화 능력을 시사한다. 또한, Pairwise term의 복잡도를 합성곱의 수용 필드 조정만으로 변경할 수 있어 모델의 유연성이 뛰어나다.
*   **학습 전략의 중요성:** 증분 학습 전략이 동시 학습보다 모델 최적화에 더 안정적이고 효과적임을 보여줌으로써, 복잡한 딥러닝 모델의 훈련 과정에 대한 중요한 통찰을 제공한다.
*   **정확도 분석의 다차원성:** mIoU뿐만 아니라 TA, LA, BA와 같은 다중 지표를 사용하여 모델의 성능을 세부적으로 분석한 점은 시맨틱 분할 문제의 다양한 측면(이미지 레벨 태깅, 객체 위치 파악, 경계 정확도)을 이해하는 데 기여하며, 향후 연구 방향을 제시한다.

**한계 및 향후 연구:**
*   **작은 객체 및 비정형 자세:** 학습 데이터셋에서 드물게 나타나는 매우 작거나 비정형적인 자세의 객체에 대한 분할 정확도는 여전히 도전 과제이다. 잘 훈련된 객체 검출기와 같은 추가적인 Unary potential을 통합하여 이 문제를 해결할 수 있을 것이다.
*   **스케일 및 조명 변화:** 이미지의 스케일 및 조명 변화는 성능에 영향을 미치는 중요한 요소로 지적된다. 이는 훈련 데이터의 증강이나 다양성을 통해 부분적으로 완화될 수 있다.
*   **확장성:** 더 많은 객체 클래스와 상당한 외관/스케일 변화가 있는 시나리오로 DPN의 일반화 가능성을 탐구하는 것이 향후 과제이다.
*   **다른 기법과의 통합:** Dilated convolution을 통한 멀티스케일 문맥 통합과 같은 상호 보완적인 기법을 DPN에 통합하여 성능을 더욱 향상시킬 수 있는 여지가 있다.

## 📌 TL;DR
시맨틱 분할에서 픽셀 분류와 문맥 정보 모델링은 중요하지만, 기존 MRF 기반 모델은 비효율적인 반복 추론과 제한된 Pairwise term 표현력이 문제였다. 본 논문은 이러한 한계를 극복하기 위해 **Deep Parsing Network (DPN)**를 제안한다. DPN은 CNN을 확장하여 Unary term을 모델링하고, **Mean Field (MF) 알고리즘의 단 한 번의 순방향 패스**만으로 **고차 관계 및 라벨 컨텍스트 혼합**을 포함하는 Pairwise term을 효율적으로 근사화한다. 이로써 DPN은 계산 비용을 크게 줄이면서 **종단 간 학습 및 추론**을 가능하게 한다. PASCAL VOC 2012, Cityscapes, CamVid 데이터셋에서 **최첨단 성능을 달성**하여 DPN의 효율성과 정확성을 입증했으며, 특히 기존 모델 대비 10배 이상 빠른 속도로 고품질의 분할 결과를 제공한다.