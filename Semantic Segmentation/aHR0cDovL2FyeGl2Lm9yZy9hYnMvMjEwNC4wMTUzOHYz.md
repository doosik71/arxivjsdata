# Hypercorrelation Squeeze for Few-Shot Segmentation
Juhong Min, Dahyun Kang, Minsu Cho

## 🧩 Problem to Solve
소수(few-shot) 이미지로 타겟 객체를 쿼리 이미지에서 분할하는Few-shot 의미론적 분할(semantic segmentation)은 제한된 주석 데이터로 인해 딥 네트워크의 일반화 능력이 저하되는 근본적인 문제를 안고 있습니다. 특히, 쿼리 및 지원 이미지 간의 다양한 수준의 시각적 단서를 이해하고 미세한 대응 관계를 분석하는 것이 어렵습니다.

## ✨ Key Contributions
*   **Hypercorrelation Squeeze Networks (HSNet) 제안:** 다단계 특징 상관관계와 효율적인 4D 컨볼루션을 활용하여 Few-shot 의미론적 분할 문제를 해결하는 새로운 프레임워크를 제시했습니다.
*   **Center-pivot 4D 컨볼루션 커널 개발:** 기존 4D 컨볼루션보다 정확도와 속도 면에서 효율적인 새로운 4D 커널을 제안하여 실시간 추론을 가능하게 했습니다.
*   **최고 성능 달성:** PASCAL-5$^i$, COCO-20$^i$, FSS-1000 표준 Few-shot 분할 벤치마크에서 새로운 State-of-the-Art (SOTA) 성능을 달성했습니다.

## 📎 Related Works
*   **의미론적 분할(Semantic Segmentation):** 인코더-디코더 구조가 주로 사용되지만, 불충분한 훈련 데이터에 대한 일반화 능력의 한계가 있습니다.
*   **Few-shot 학습(Few-shot Learning):**
    *   매칭 네트워크(Matching Networks) [73] 및 프로토타입 네트워크(Prototypical Networks) [65]와 같이 제한된 주석 예제로 딥 네트워크를 훈련하는 접근 방식이 연구되었습니다.
    *   프로토타입 표현을 활용하는 방법 [10, 36, 37, 46, 63, 75, 80, 87, 89]과 그래프 어텐션(graph attention)을 사용한 쌍별 특징 상관관계 구축 방법 [74, 86]이 있습니다.
*   **시각적 대응(Visual Correspondences):**
    *   중간 특징(intermediate features) 활용 [38, 42, 44] 및 고차원 컨볼루션(4D convolutions) [30, 58, 71]을 사용하여 정확한 대응을 확립하는 연구가 진행되었습니다.
    *   본 연구는 기존 Few-shot 분할 연구가 다양한 CNN 계층의 특징 표현을 충분히 활용하지 않거나 미세한 상관 패턴을 포착하지 못하는 한계를 지적하며, 이 두 가지 강력한 기술을 Few-shot 분할에 결합합니다.

## 🛠️ Methodology
HSNet은 하이퍼 상관관계 구축, 4D 컨볼루션 피라미드 인코더, 2D 컨볼루션 컨텍스트 디코더의 세 가지 주요 부분으로 구성됩니다.

1.  **Hypercorrelation Construction (하이퍼 상관관계 구축):**
    *   쿼리 이미지 $I_q$와 지원 이미지 $I_s$로부터 백본 네트워크를 통해 $L$개의 중간 특징 맵 쌍 $\{(F_q^l, F_s^l)\}_{l=1}^L$을 추출합니다.
    *   지원 특징 맵 $F_s^l$에 지원 마스크 $M_s$를 적용하여 불필요한 활성화를 제거합니다: $\hat{F}_s^l = F_s^l \odot \zeta_l(M_s)$.
    *   각 계층에서 쿼리와 마스킹된 지원 특징 쌍을 사용하여 코사인 유사도를 통해 4D 상관 텐서 $\hat{C}^l \in \mathbb{R}^{H_l \times W_l \times H_l \times W_l}$를 생성합니다. $\hat{C}^l(x_q, x_s) = \text{ReLU}\left(\frac{F_q^l(x_q) \cdot \hat{F}_s^l(x_s)}{\|F_q^l(x_q)\|\|\hat{F}_s^l(x_s)\|}\right)$.
    *   동일한 공간 크기를 가진 4D 텐서들을 채널 차원을 따라 연결하여 하이퍼 상관관계 피라미드 $C = \{C_p\}_{p=1}^P$를 형성합니다.

2.  **4D-convolutional Pyramid Encoder (4D 컨볼루션 피라미드 인코더):**
    *   하이퍼 상관관계 피라미드 $C$를 입력으로 받아 압축된 특징 맵 $Z \in \mathbb{R}^{128 \times H_1 \times W_1}$로 변환합니다.
    *   **Squeezing Block ($f_{\text{sqz}}^p$):** 4D 컨볼루션, 그룹 정규화(Group Normalization), ReLU로 구성되며, 마지막 두 지원 공간 차원을 주기적으로 압축합니다.
    *   **Mixing Block ($f_{\text{mix}}^p$):** 인접한 피라미드 계층의 출력을 업샘플링 후 원소별 덧셈으로 병합하고, 4D 컨볼루션으로 처리하여 상위 계층에서 하위 계층으로 관련 정보를 전파합니다.
    *   가장 낮은 Mixing Block의 출력은 마지막 두 지원 공간 차원에 대한 평균 풀링을 통해 2D 특징 맵 $Z$로 압축됩니다.

3.  **2D-convolutional Context Decoder (2D 컨볼루션 컨텍스트 디코더):**
    *   인코딩된 컨텍스트 $Z$를 입력으로 받아 2D 컨볼루션, ReLU, 업샘플링 계층을 거쳐 예측 마스크 $\hat{M}_q \in [0,1]^{2 \times H \times W}$를 출력합니다.
    *   훈련 시에는 교차 엔트로피 손실로 최적화됩니다.

4.  **Center-pivot 4D Convolution (중심-피봇 4D 컨볼루션):**
    *   기존 4D 컨볼루션의 높은 계산 비용과 과도한 파라미터 문제를 해결합니다.
    *   4D 윈도우 내에서 가장 중요한 위치(쿼리 또는 지원 중심에 인접한 위치)의 활성화에만 집중하여 가중치를 희소화(weight-sparsification)합니다.
    *   수식은 두 개의 분리된 2D 컨볼루션의 합으로 표현될 수 있습니다:
        $$(c \ast k_{\text{CP}})(x,x') = \sum_{p' \in \mathcal{P}(x')} c(x,p')k_{2D}^c(p'-x') + \sum_{p \in \mathcal{P}(x)} c(p,x')k_{2D}^{c'}(p-x)$$
        이는 선형 복잡도를 가지며 효율적인 패턴 인식이 가능합니다.

5.  **K-shot 확장:** $K$개의 지원 이미지-마스크 쌍이 주어질 경우, 각 지원 쌍에 대해 $K$번의 순방향 패스를 수행하여 $K$개의 마스크 예측을 얻고, 픽셀별 투표(voting)를 통해 최종 마스크를 결정합니다.

## 📊 Results
*   **PASCAL-5$^i$ 벤치마크:** VGG16, ResNet50, ResNet101 백본 모두에서 SOTA를 달성했으며, 특히 ResNet101 백본 사용 시 기존 최고 모델 대비 1-shot에서 6.1%p, 5-shot에서 4.8%p의 mIoU 개선을 보였습니다. 학습 가능한 파라미터 수는 2.6M으로 가장 적습니다.
*   **COCO-20$^i$ 벤치마크:** ResNet101 백본에서 1-shot에서 2.7%p, 5-shot에서 6.8%p의 mIoU 개선을 보이며 SOTA를 달성했습니다.
*   **FSS-1000 벤치마크:** 기존 최고 성능을 뛰어넘는 새로운 SOTA를 기록했습니다.
*   **도메인 변화(Domain Shift)에 대한 강건성:** COCO로 훈련된 모델을 PASCAL-5$^i$에 적용했을 때, 데이터 증강 없이도 강력한 성능을 보였습니다.
*   **어블레이션 연구(Ablation Study):**
    *   **하이퍼 상관관계:** 다단계 CNN 계층의 특징 상관관계를 포착하는 것이 중요함을 확인했습니다.
    *   **피라미드 계층:** 의미론적 단서와 기하학적 단서(semantic and geometric cues)를 모두 포착하는 것이 미세한 객체 위치 파악에 필수적임을 입증했습니다.
    *   **4D 커널 비교:** 제안된 center-pivot 4D 커널이 기존 4D 커널(Original [58], Separable [81]) 대비 가장 빠른 추론 시간과 가장 적은 메모리/FLOPs 요구량을 보였습니다.
    *   **백본 네트워크 고정(Freezing Backbone):** 사전 훈련된 백본을 고정하는 것이 새로운 클래스에 대한 과적합을 방지하고 일반화 능력을 향상시키는 데 중요함을 확인했습니다.

## 🧠 Insights & Discussion
*   제한된 지도 학습 환경에서 미세한 분할을 위해서는 "다양한 시각적 측면의 특징 관계(feature relations)" 패턴을 학습하는 것이 매우 효과적임을 입증했습니다.
*   새롭게 제안된 center-pivot 4D 컨볼루션 커널은 4D 커널을 두 개의 2D 커널로 효율적으로 분해하여, 적은 비용으로 4D 컨볼루션 계층을 광범위하게 활용할 수 있게 합니다.
*   모델의 일반화 능력은 새로운 개념을 학습할 때 과거의 경험(예: ImageNet 분류 학습)과 그들의 "관계"를 분석하는 인간의 시각과 유사하게 작용한다는 통찰을 제공합니다.
*   이 연구는 고차원 상관관계를 분석해야 하는 다른 도메인에서 4D 컨볼루션의 활용을 촉진할 것으로 기대됩니다.

## 📌 TL;DR
Few-shot 의미론적 분할에서 소수의 주석 이미지로 객체를 정확히 분할하는 문제를 해결하기 위해, 본 논문은 **Hypercorrelation Squeeze Networks (HSNet)**를 제안합니다. HSNet은 쿼리와 지원 이미지 간의 다단계 특징 상관관계인 "하이퍼 상관관계"를 구축하고, 효율적인 **center-pivot 4D 컨볼루션**을 사용하는 피라미드형 인코더를 통해 이 상관관계를 미세한 분할 마스크로 압축합니다. 결과적으로 PASCAL-5$^i$, COCO-20$^i$, FSS-1000 벤치마크에서 새로운 SOTA 성능을 달성했으며, 동시에 학습 가능한 파라미터 수를 획기적으로 줄여 효율성과 정확도를 모두 입증했습니다.