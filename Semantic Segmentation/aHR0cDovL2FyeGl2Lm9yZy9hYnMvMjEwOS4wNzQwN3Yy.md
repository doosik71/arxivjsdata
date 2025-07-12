# Semi-supervised Contrastive Learning for Label-efficient Medical Image Segmentation
Xinrong Hu, Dewen Zeng, Xiaowei Xu, and Yiyu Shi

## 🧩 Problem to Solve
심층 학습 기반의 의료 영상 분할(medical image segmentation)은 방대한 양의 레이블링된 데이터를 필요로 합니다. 그러나 의료 영상 주석(annotation) 작업은 전문 지식을 요구하며 시간과 노력이 많이 소요되어, 현실적으로 충분한 레이블 데이터를 확보하기 어렵습니다. 기존의 대비 학습(contrastive learning) 기반 준지도 학습(semi-supervised learning) 방법들은 주로 레이블이 없는 데이터로 자기 지도(self-supervised) 사전 훈련을 수행한 후, 제한된 레이블 데이터로 미세 조정을 하는 방식을 사용했습니다. 본 논문은 이러한 방식이 사전 훈련 단계에서 제한적으로라도 사용 가능한 레이블 정보를 전혀 활용하지 않는다는 점을 문제로 제기합니다.

## ✨ Key Contributions
*   **레이블 활용 대비 학습 제안:** 제한된 레이블 정보를 사전 훈련 단계에서 활용하여 대비 학습의 성능을 향상시키는 반지도 학습 프레임워크를 제안합니다.
*   **지도 지역 대비 손실(Supervised Local Contrastive Loss) 도입:** 픽셀 단위 주석을 활용하여 동일한 레이블을 가진 픽셀들의 임베딩(embedding)을 가깝게, 다른 레이블을 가진 픽셀들의 임베딩을 멀게 만드는 새로운 손실 함수를 개발했습니다.
*   **계산 효율성 개선 전략:** 대규모 영상에 대한 픽셀 단위 계산의 높은 비용 문제를 해결하기 위해 **다운샘플링(downsampling)**과 **블록 분할(block division)**의 두 가지 전략을 제시합니다.
*   **성능 향상 입증:** 두 가지 공개 의료 영상 데이터셋(MRI, CT)에 대한 실험을 통해, 다양한 레이블 데이터 비율에서 기존 최첨단 대비 학습 기반 및 기타 준지도 학습 방법들을 일관되게 능가하는 성능을 달성했습니다.

## 📎 Related Works
*   의료 영상 분할을 위한 심층 신경망: U-Net [16], V-Net [15] 등.
*   데이터 증강(Data Augmentation): GAN(Generative Adversarial Networks)을 활용한 데이터 합성 [3, 5, 8], Mixup [18].
*   준지도 학습(Semi-supervised Learning): 의료 영상 분야에서의 다양한 준지도 접근법 [2, 12, 14, 19], 특히 TCSM [14].
*   자기 지도 대비 학습(Self-supervised Contrastive Learning): SimCLR [7], MoCo [10] 등 영상 분류를 위한 대표적인 방법들.
*   의료 영상 분할에 대비 학습을 적용한 기존 연구: 제한된 주석으로 전역(global) 및 지역(local) 특징을 학습하는 Chaitanya et al. [4] 및 Zeng et al. [17]의 연구.
*   지도 대비 학습(Supervised Contrastive Learning): Khosla et al. [11].

## 🛠️ Methodology
본 논문은 인코더 $E$와 디코더 $D$로 구성된 2D U-Net을 기반으로 하는 준지도 학습 프레임워크를 제안합니다.

1.  **사전 훈련 단계 (Pre-training Stage):**
    *   **자기 지도 전역 대비 학습 (Self-supervised Global Contrastive Learning):**
        *   레이블이 없는 데이터를 사용하여 인코더 $E$를 훈련합니다.
        *   입력 영상 $x_i$에 두 번의 임의 변환(augmentation) $aug(\cdot)$을 적용하여 증강된 영상 쌍 $a_i, a_{j(i)}$를 생성합니다.
        *   이 쌍들의 특징 임베딩 $z_i = |h_1(E(a_i))|$ (여기서 $h_1(\cdot)$은 투사 헤드(projection head))을 가깝게, 다른 영상들의 임베딩은 멀게 만드는 전역 대비 손실 $L_g$를 사용합니다:
            $$ L_g = - \frac{1}{|A|} \sum_{i \in I} \log \frac{\exp(z_i \cdot z_{j(i)} / \tau)}{\sum_{k \in I-\{i\}} \exp(z_i \cdot z_k) / \tau} $$
    *   **지도 지역 대비 학습 (Supervised Local Contrastive Learning):**
        *   전역 대비 학습 후, 인코더 $E$에 디코더 $D$를 연결하고, 제한된 레이블 데이터를 사용하여 전체 네트워크를 재훈련합니다.
        *   핵심은 픽셀 단위 특징 맵 $f^l_{u,v}$ (디코더의 최상단 블록 출력)에 대한 손실 계산입니다.
        *   **긍정 쌍(Positive Set) $P(u,v)$:** 기준 픽셀 $(u,v)$와 *동일한 레이블*을 가진 배치 내의 모든 특징들로 정의됩니다.
        *   **부정 쌍(Negative Set) $N(u,v)$:** 기준 픽셀 $(u,v)$와 *다른 레이블*을 가진 배치 내의 모든 특징들로 정의됩니다.
        *   손실 계산에 사용되는 픽셀 집합 $\Omega$는 배경(background)이 아닌 레이블을 가진 픽셀만 포함합니다.
        *   지도 지역 대비 손실 $loss(a_i)$는 다음과 같습니다:
            $$ loss(a_i) = - \frac{1}{|\Omega|} \sum_{(u,v) \in \Omega} \frac{1}{|P(u,v)|} \log \frac{\sum_{(u_p,v_p) \in P(u,v)} \exp(f^l_{u,v} \cdot f^l_{u_p,v_p} / \tau)}{\sum_{(u',v') \in N(u,v)} \exp(f^l_{u,v} \cdot f^l_{u',v'} / \tau)} $$
        *   **계산 복잡도 감소 전략:**
            *   **다운샘플링(Downsampling):** 특징 맵에서 고정된 보폭(stride)으로 픽셀을 샘플링합니다 (예: 보폭 4).
            *   **블록 분할(Block Division):** 특징 맵을 작은 블록으로 나누고, 각 블록 내에서만 지역 대비 손실을 계산한 후 평균합니다 (예: 블록 크기 $16 \times 16$).
2.  **미세 조정 단계 (Fine-tuning Stage):**
    *   사전 훈련된 모델을 제한된 레이블 데이터로 미세 조정하여 최종 분할 정확도를 높입니다.

## 📊 Results
*   **데이터셋:** Hippocampus (MRI)와 MMWHS (CT) 두 가지 공개 의료 영상 데이터셋.
*   **평가 지표:** Dice 점수(Dice score).
*   **주요 결과:**
    *   모든 대비 학습 방법은 무작위 초기화(random initialization)보다 높은 Dice 점수를 보였습니다.
    *   제안된 지도 지역 대비 학습 단독(local(stride), local(block))으로도 기존 대비 학습 방법들과 비슷한 성능을 달성했으며, 이는 제한된 레이블 데이터를 효율적으로 활용했음을 시사합니다.
    *   전역 대비 학습과 제안된 지도 지역 대비 학습을 결합한 방법(global+local(stride), global+local(block))은 두 데이터셋 모두에서 **일관되게 최첨단(state-of-the-art) 성능을 능가했습니다**. 이는 사용 가능한 레이블 비율이 낮을수록 더욱 두드러졌습니다.
    *   시각화 결과(그림 3)를 통해 제안된 방법이 더 정확한 분할 결과를 생성함을 확인했습니다.
    *   t-SNE를 이용한 임베딩 특징 시각화(그림 4)에서 제안된 `global+local(block)` 방법은 동일 클래스 특징들이 더 조밀하게 군집되어 있고, 다른 클래스 특징들은 명확하게 분리되어 있음을 보여주어, 효과적인 특징 학습을 입증했습니다.

## 🧠 Insights & Discussion
*   본 연구는 의료 영상 분할 분야에서 레이블 데이터 부족이라는 중대한 문제를 해결하기 위해, 제한된 레이블 정보를 대비 학습의 사전 훈련 단계에 통합하는 새로운 접근 방식이 매우 효과적임을 입증했습니다.
*   자기 지도 전역 대비 학습(영상 레벨 특징)과 지도 지역 대비 학습(픽셀 레벨 특징)은 상호 보완적이며, 이를 결합함으로써 상당한 성능 향상을 이끌어낼 수 있습니다.
*   제안된 다운샘플링 및 블록 분할 전략은 픽셀 단위 대비 손실의 높은 계산 복잡도를 실용적인 수준으로 낮춰, 고해상도 의료 영상에도 적용 가능하게 합니다.
*   학습된 특징들의 시각화는 제안된 방법이 클래스 간 구별력을 높이고 동일 클래스 내에서는 응집력을 갖는 고품질의 임베딩을 학습함을 보여줍니다.

## 📌 TL;DR
*   **문제:** 의료 영상 분할은 레이블 부족 문제가 심각하며, 기존 대비 학습은 사전 훈련 시 레이블을 활용하지 못합니다.
*   **해결책:** 레이블이 없는 데이터로 영상 레벨 특징을 학습하는 자기 지도 전역 대비 학습과, 제한된 레이블 데이터로 픽셀 레벨 특징을 학습하는 **새로운 지도 지역 대비 학습**을 결합한 준지도 학습 프레임워크를 제안합니다. 높은 계산 복잡도를 위해 다운샘플링 또는 블록 분할 전략을 사용합니다.
*   **결과:** 이 방법은 두 가지 의료 영상 데이터셋에서 기존 최첨단 방법들을 일관되게 능가하며, 레이블 부족 환경에서 의료 영상 분할 성능을 크게 향상시킬 수 있음을 입증합니다.