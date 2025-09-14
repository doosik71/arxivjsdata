# META-LEARNING FOR SEMI-SUPERVISED FEW-SHOT CLASSIFICATION
Mengye Ren, Eleni Triantafillou, Sachin Ravi, Jake Snell, Kevin Swersky, Joshua B. Tenenbaum, Hugo Larochelle & Richard S. Zemel

## 🧩 Problem to Solve
이 논문은 소수 레이블 예시만으로 새로운 클래스를 학습하는 **Few-Shot Classification** 문제를 다룹니다. 기존 Few-Shot Classification 연구는 주로 메타 학습(meta-learning)과 에피소드(episode) 기반 훈련 방식을 사용했으나, 각 에피소드 내에서 **레이블이 없는 예시(unlabeled examples)가 함께 제공되는 시나리오**까지 확장하는 것을 목표로 합니다. 특히, 다음과 같은 두 가지 상황을 고려합니다:
1.  레이블이 없는 예시들이 해당 에피소드의 레이블링된 클래스와 동일한 클래스에 속한다고 가정하는 경우.
2.  도전적인 상황으로, 레이블이 없는 예시들 중에는 분류 대상이 아닌 **방해 클래스(distractor classes)의 예시**가 포함될 수 있는 경우.

## ✨ Key Contributions
*   **새로운 패러다임 제시**: 각 훈련 에피소드에 레이블이 없는 데이터($R$)를 추가하는 **반지도 학습 Few-Shot Learning** 패러다임을 제안하고 정의했습니다.
*   **Prototypical Networks 확장**: 기존 Prototypical Networks (Snell et al., 2017)를 레이블이 없는 예시를 활용하여 프로토타입($p_c$)을 개선할 수 있도록 세 가지 새로운 방식으로 확장했습니다.
    *   Soft k-Means를 활용한 프로토타입 정제.
    *   방해 클래스(distractor cluster)를 위한 추가 클러스터를 포함한 Soft k-Means.
    *   마스킹(masking) 메커니즘을 포함한 Soft k-Means.
*   **End-to-End 훈련**: 제안된 모델들을 레이블이 없는 예시를 성공적으로 활용하도록 에피소드 방식으로 End-to-End 훈련했습니다.
*   **새로운 벤치마크 데이터셋**: 기존 Omniglot, miniImageNet 벤치마크를 레이블 없는 예시를 포함하도록 재구성했으며, 더 크고 계층적인 구조를 가진 새로운 Few-Shot Learning 데이터셋인 **`tieredImageNet`**을 제안했습니다.
*   **실험적 검증**: 제안된 모델들이 레이블이 없는 예시를 통해 예측 성능을 성공적으로 향상시키며, 기존 지도 학습 기반 Prototypical Networks를 능가함을 입증했습니다.

## 📎 Related Works
*   **Few-Shot Learning**: 메타 학습의 에피소드 훈련 방식을 따르는 방법론들, 특히 유사성 측정 학습(metric learning) 접근법에 중점을 둡니다.
    *   **Metric Learning**: Deep Siamese Networks (Koch et al., 2015), Matching Networks (Vinyals et al., 2016), 그리고 본 논문에서 확장하는 **Prototypical Networks (Snell et al., 2017)**가 있습니다. 이들은 임베딩 공간에서 같은 클래스 예시를 가깝게, 다른 클래스 예시를 멀게 학습합니다.
    *   **다른 메타 학습 접근법**: 학습자의 가중치 초기화 및 업데이트 단계를 학습하는 방법 (Ravi & Larochelle, 2017; Finn et al., 2017), 또는 메모리 증강 순환 네트워크 (Santoro et al., 2016)나 시간 컨볼루션 네트워크 (Mishra et al., 2017)와 같은 일반적인 신경망 아키텍처를 훈련하는 방법이 있습니다.
    *   **Active Learning**: Bachman et al. (2017)은 Matching Networks를 활성 학습 프레임워크에 적용했으나, 레이블 없는 데이터를 통해 실제 레이블을 얻을 수 있고 방해 예시를 사용하지 않는다는 점에서 본 연구와 차이가 있습니다.
*   **Semi-Supervised Learning**: 레이블이 있는 데이터와 없는 데이터를 함께 사용하여 학습하는 광범위한 분야입니다.
    *   **Self-Training**: Yarowsky (1995), Rosenberg et al. (2005) 연구와 같이, 초기 모델로 레이블 없는 데이터를 분류하고 가장 확신하는 예측을 레이블로 사용하여 훈련 세트를 확장하는 방식입니다. 본 논문의 Soft k-Means 확장이 이와 유사합니다.
    *   **Transductive Learning**: Vapnik (1998), Joachims (1999), Fu et al. (2015) 연구처럼, 분류기가 레이블이 없는 예시들을 봄으로써 개선되는 방식입니다.
*   **Clustering with Outliers**: Lloyd (1982)의 k-Means와 이상치(outliers) 처리 연구 (Hautamäki et al., 2005; Chawla & Gionis, 2013; Gupta et al., 2017)는 방해 요소(distractors)를 무시하고 클러스터 위치가 잘못 이동하는 것을 방지하는 본 연구의 목표와 관련이 있습니다.

## 🛠️ Methodology
본 논문에서는 Prototypical Networks를 확장하여 레이블이 없는 데이터($R$)를 활용해 프로토타입을 정제($\tilde{p}_c$)하는 세 가지 방법을 제안합니다. 기본 아이디어는 지도 학습으로 초기 프로토타입 $p_c$를 계산한 후, $R$의 예시들을 이용해 이를 수정하는 것입니다.

1.  **Soft k-Means 기반 Prototypical Networks**:
    *   **초기 프로토타입 계산**: 지원 세트 $S$의 레이블된 예시들을 임베딩하여 각 클래스 $c$의 초기 프로토타입 $p_c$를 계산합니다 ($p_c = \frac{\sum_i h(x_i)z_{i,c}}{\sum_i z_{i,c}}$).
    *   **레이블 없는 예시의 소프트 할당**: $R$의 각 레이블 없는 예시 $\tilde{x}_j$에 대해, 기존 프로토타입 $p_c$까지의 유클리드 거리를 기반으로 각 클래스에 대한 소프트 할당 $\tilde{z}_{j,c}$를 계산합니다 ($\tilde{z}_{j,c} = \frac{\exp(-\|h(\tilde{x}_j) - p_c\|^2_2)}{\sum_{c'} \exp(-\|h(\tilde{x}_j) - p_{c'}\|^2_2)}$).
    *   **프로토타입 정제**: 레이블된 예시와 소프트 할당된 레이블 없는 예시들을 모두 사용하여 정제된 프로토타입 $\tilde{p}_c$를 업데이트합니다 ($\tilde{p}_c = \frac{\sum_i h(x_i)z_{i,c} + \sum_j h(\tilde{x}_j) \tilde{z}_{j,c}}{\sum_i z_{i,c} + \sum_j \tilde{z}_{j,c}}$).
    *   훈련 시 하나의 정제 단계만 수행합니다.

2.  **방해 클러스터(Distractor Cluster)를 포함한 Soft k-Means Prototypical Networks**:
    *   위 Soft k-Means 방식은 모든 레이블 없는 예시가 에피소드 내의 $N$개 클래스 중 하나에 속한다고 가정하므로, 방해 클래스 예시가 있을 경우 성능 저하가 발생할 수 있습니다.
    *   이를 해결하기 위해 $N$개의 클래스 외에 **방해 예시들을 위한 추가 클러스터($N+1$번째 클러스터)**를 도입합니다.
    *   방해 클러스터의 프로토타입은 원점(0)으로 가정하며, 이 클러스터에 대한 길이 스케일($r_{N+1}$)을 학습하여 클러스터 내 거리의 변화를 표현합니다.
    *   $\tilde{z}_{j,c}$ 계산 시 길이 스케일 $r_c$를 도입하여 각 클러스터의 특성을 반영합니다 ($\tilde{z}_{j,c} = \frac{\exp(-\frac{1}{r_c^2}\|h(\tilde{x}_j) - p_c\|^2_2 - A(r_c))}{\sum_{c'} \exp(-\frac{1}{r_{c'}^2}\|h(\tilde{x}_j) - p_{c'}\|^2_2 - A(r_{c'}))}$).

3.  **마스킹(Masking)을 포함한 Soft k-Means Prototypical Networks**:
    *   단일 방해 클러스터가 여러 유형의 방해 예시를 포착하기에는 너무 단순할 수 있습니다.
    *   이 모델은 레이블 없는 예시가 합법적인 클래스 프로토타입 주변의 특정 영역 내에 있는지 여부에 따라 기여도를 조절하는 **소프트 마스킹 메커니즘**을 사용합니다.
    *   **정규화된 거리 계산**: $\tilde{x}_j$와 $p_c$ 간의 정규화된 거리 $\tilde{d}_{j,c}$를 계산합니다 ($\tilde{d}_{j,c} = \frac{d_{j,c}}{\frac{1}{M}\sum_j d_{j,c}}$, where $d_{j,c}=\|h(\tilde{x}_j)-p_c\|^2_2$).
    *   **마스크 임계값 및 기울기 예측**: 각 프로토타입에 대해 정규화된 거리 통계(최소, 최대, 분산, 왜도, 첨도)를 MLP에 입력하여 소프트 임계값 $\beta_c$와 기울기 $\gamma_c$를 예측합니다.
    *   **소프트 마스크 계산**: $\tilde{d}_{j,c}$와 예측된 $\beta_c, \gamma_c$를 사용하여 각 예시의 각 프로토타입에 대한 소프트 마스크 $m_{j,c}$를 계산합니다 ($m_{j,c} = \sigma(-\gamma_c(\tilde{d}_{j,c} - \beta_c))$).
    *   **마스킹된 프로토타입 정제**: 레이블 없는 예시의 기여도에 $m_{j,c}$를 곱하여 정제된 프로토타입 $\tilde{p}_c$를 업데이트합니다 ($\tilde{p}_c = \frac{\sum_i h(x_i)z_{i,c} + \sum_j h(\tilde{x}_j) \tilde{z}_{j,c} m_{j,c}}{\sum_i z_{i,c} + \sum_j \tilde{z}_{j,c} m_{j,c}}$).
    *   이 방식을 통해 모델은 MLP를 사용하여 특정 레이블 없는 예시를 포함하거나 완전히 무시하는 방법을 학습합니다.

이 모든 모델은 정제된 프로토타입 $\tilde{p}_c$를 사용하여 쿼리 예시($x^*$)의 클래스를 분류하며, 일반 Prototypical Networks와 동일한 손실 함수(Equation 3)를 사용하여 End-to-End로 훈련됩니다.

## 📊 Results
*   **Omniglot, miniImageNet, tieredImageNet** 세 가지 데이터셋에서 실험을 수행했습니다.
*   **모든 벤치마크에서 일관된 성능 향상**: 제안된 세 가지 반지도 학습 모델 중 적어도 하나는 모든 실험에서 베이스라인(Supervised Prototypical Network, Semi-Supervised Inference)을 능가했습니다. 이는 반지도 학습 메타 학습 절차의 효과를 입증합니다.
*   **방해 클래스가 없는 환경**: 세 가지 제안 모델 모두 대부분의 실험에서 베이스라인을 능가했으며, 특정 모델이 모든 데이터셋과 샷(shot) 수에 걸쳐 명확히 우세하지는 않았습니다.
*   **방해 클래스가 있는 환경**: **Masked Soft k-Means 모델**이 가장 강력하고 안정적인 성능을 보였습니다. 세 가지 데이터셋 중 한 가지 경우를 제외하고 모두 최고의 결과를 달성했으며, 방해 클래스가 없을 때의 성능에 근접했습니다. 이는 이 모델이 방해 요소에 대한 강건성(robustness)을 잘 학습했음을 시사합니다.
*   **레이블 없는 예시 수의 영향**: `tieredImageNet`에서 클래스당 레이블 없는 예시($M$)의 수가 0에서 25로 증가함에 따라 테스트 정확도가 명확하게 향상되는 것을 확인했습니다. 이는 모델이 메타 훈련을 통해 반지도 정제를 통해 개선되는 더 나은 표현을 학습했음을 보여줍니다.

## 🧠 Insights & Discussion
*   이 연구는 Few-Shot Learning 패러다임을 레이블 없는 데이터를 활용하는 **반지도 설정**으로 확장함으로써 실제 적용 가능성을 높였습니다. 특히, 실제 시나리오에서 흔히 발생하는 방해 클래스가 존재하는 경우까지 고려한 점이 중요합니다.
*   **Prototypical Networks의 유연성**: Prototypical Networks가 간단한 구조임에도 불구하고 Soft k-Means, 방해 클러스터, 마스킹과 같은 반지도 학습 기법과 결합하여 성공적으로 확장될 수 있음을 보여주었습니다.
*   **새로운 데이터셋의 가치**: `tieredImageNet`은 기존 데이터셋의 한계(훈련/테스트 클래스 간 유사성)를 극복하고, 계층적 구조를 통해 보다 현실적인 Few-Shot Learning 시나리오를 제공함으로써 향후 연구에 중요한 자원이 될 것으로 기대됩니다.
*   **메타 학습의 효과**: 모델이 메타 훈련을 통해 레이블 없는 데이터를 활용하여 임베딩 공간에서 프로토타입을 효과적으로 정제하고 분류 성능을 향상시키는 방법을 '학습'한다는 점이 핵심적인 통찰입니다. 이는 메타 학습이 "학습하는 방법"을 학습하는 데 매우 효과적임을 다시 한번 확인시켜 줍니다.
*   **한계 및 향후 연구**: 본 연구에서 사용된 MLP의 하이퍼파라미터 튜닝이 충분히 이루어지지 않았으므로, 더 엄격한 튜닝을 통해 성능 향상 여지가 있습니다. 또한, 향후 연구로 에피소드 내 컨텐츠에 따라 다른 임베딩 표현을 가질 수 있도록 고속 가중치(fast weights)를 프레임워크에 통합하는 방안을 제시했습니다.

## 📌 TL;DR
이 논문은 레이블이 있는 소수의 예시뿐만 아니라 **레이블이 없는 예시(심지어 방해 클래스 포함)**가 주어진 **반지도 학습 Few-Shot Classification** 문제를 해결합니다. Prototypical Networks를 Soft k-Means, 방해 클러스터, 마스킹 기법으로 확장하여 레이블 없는 예시를 활용해 프로토타입을 정제하는 방법을 제안합니다. `tieredImageNet`이라는 새로운 계층적 데이터셋을 포함한 여러 벤치마크에서 실험 결과, 제안된 모델들이 베이스라인 대비 일관된 성능 향상을 보였으며, 특히 **Masked Soft k-Means 모델이 방해 클래스가 있는 어려운 시나리오에서 가장 강력한 성능**을 입증했습니다. 이는 메타 학습을 통해 레이블 없는 데이터를 효과적으로 활용하는 방법을 학습함으로써 Few-Shot 분류 성능을 성공적으로 개선할 수 있음을 보여줍니다.