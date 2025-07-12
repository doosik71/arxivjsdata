# Birds of A Feather Flock Together: Category-Divergence Guidance for Domain Adaptive Segmentation

Bo Yuan, Danpei Zhao, Shuai Shao, Zehuan Yuan, and Changhu Wang

## 🧩 Problem to Solve

본 논문은 비지도 도메인 적응(Unsupervised Domain Adaptation, UDA)을 사용하여 합성(source) 도메인에서 현실(target) 도메인으로 시맨틱 분할(semantic segmentation) 모델의 일반화 능력을 향상시키는 문제를 다룹니다. 기존 UDA 모델들은 주로 도메인 간의 특징 불일치(feature discrepancy)를 최소화하여 도메인 시프트(domain shift)를 완화하는 데 초점을 맞추었으나, 범주 혼동(class confusion) 문제를 간과하는 경향이 있었습니다. 특히, 시각적으로 유사하거나 영역 경계에 인접한 픽셀들이 잘못 분류되는 문제가 발생합니다.

## ✨ Key Contributions

* **범주-발산(Category-Divergence) 지침 도입:** 교차 도메인 시맨틱 분할을 위한 범주-발산 지침 접근 방식을 제안합니다. 이는 동일 범주의 특징 표현을 가깝게 당기고, 다른 범주의 특징 표현을 멀리 밀어냄으로써 픽셀 오분류를 효과적으로 줄입니다.
* **다단계 정렬(Multi-level Alignments) 프레임워크 구축:** 이미지 레벨, 특징 레벨, 범주 레벨, 인스턴스 레벨을 포함하는 다단계 정렬을 통해 도메인 간극(domain gap)을 시너지 효과적으로 줄이는 범용 UDA 프레임워크를 구성합니다.
* **원격 감지(Remote-sensing) 장면으로 확장:** 제안된 UDA 방법을 원격 감지 장면으로 확장하여 건물 분할 및 도로 분할을 위한 교차 도메인 데이터셋을 구축하고 해당 시나리오에서의 적용 가능성을 입증합니다.
* **최첨단 성능 달성:** 제안된 UDA 방법은 스트리트 장면(예: GTA5→Cityscapes, SYNTHIA→Cityscapes) 및 원격 감지 이미지 벤치마크 데이터셋에서 최첨단 시맨틱 분할 정확도를 달성합니다.

## 📎 Related Works

* **시맨틱 분할 (Semantic Segmentation):** FCN(Fully Convolutional Network) 기반 모델을 시작으로, 확장/아트러스(dilated/atrous) 컨볼루션, 어텐션 모듈, 인코더-디코더 구조, 피라미드 풀링 등을 통해 수용 필드(receptive field)를 넓히고 문맥 정보를 집약하는 노력이 있었습니다. 최근에는 비전 트랜스포머(vision transformers)도 사용됩니다.
* **적대적 학습 (Adversarial Learning):** GAN(Generative Adversarial Networks)은 생성자와 판별자의 제로섬 게임을 통해 이미지-수준 도메인 매핑 및 스타일 전송에 널리 사용됩니다.
* **시맨틱 분할을 위한 도메인 적응 (Domain Adaptation for Semantic Segmentation):**
  * **이미지-레벨 적응:** 소스 도메인 이미지의 외형을 타겟 도메인과 시각적으로 유사하게 변경합니다 (예: CycleGAN 기반 방법).
  * **특징-레벨 전송:** 소스 및 타겟 도메인 간에 추출된 특징 분포를 GAN 구조를 사용하여 일치시킵니다.
  * **레이블-레벨 적응:** 소스 도메인에서 학습된 지식을 활용하여 타겟 도메인의 의사 레이블(pseudo-labels)을 생성하는 자기 지도(self-supervised) 학습 접근 방식이 주로 사용됩니다.

## 🛠️ Methodology

본 논문은 이미지 레벨, 특징 레벨, 범주 레벨, 인스턴스 레벨의 다단계 정렬을 포함하는 계층적 비지도 도메인 적응 프레임워크를 제안합니다.

1. **전역 특징-레벨 적응 (Global Feature-level Adaptation, GFA):**
    * 이미지-수준 적응을 위해 Cycle-Consistency를 사용하여 타겟 도메인에서 소스 도메인으로 이미지 스타일을 전송합니다.
    * 공유 파라미터를 가진 특징 추출기($F$)와 분류 헤드($C$)로 구성된 생성자 $G = F \circ C$를 사용합니다.
    * 전통적인 GAN 구조를 출력 공간에 적용하여 소스 및 타겟 도메인 간의 특징 분포 불일치($L_{\text{f}}^{\text{adv}}$)를 전역적으로 최소화합니다.
    * 판별자 $D$는 특징이 어느 도메인에서 왔는지 구별하도록 훈련됩니다.
    $$ \min_{G} L_{\text{f}}^{\text{adv}}(G, D) = - \sum_{x_{\text{t}} \in X_{\text{T}}} \log(1 - D(S(G(x_{\text{t}})))) $$
    $$ \min_{D} L_{D}(G, D) = - \sum_{x_{\text{t}} \in X_{\text{T}}} \log(D(S(G(x_{\text{t}})))) - \sum_{x_{\text{s}} \in X_{\text{S}}} \log(1 - D(S(G(x_{\text{s}})))) $$

2. **발산-기반 범주-레벨 정렬 (Divergence-driven Category-level Alignment, ISIA):**
    * **클래스 간 분리 및 클래스 내 집합 (Inter-class Separation and Intra-class Aggregation, ISIA) 메커니즘**을 제안합니다.
    * 코사인 거리를 기반으로 유사성 측정 함수를 구성하여, 동일 범주의 특징 분포는 가깝게 만들고(aggregation), 다른 범주의 특징 분포는 멀리 분리(separation)합니다.
    * 특징 벡터 $c_{i}$와 $c_{j}$ 간의 코사인 유사도는 $D_{\text{cosine}}(c_{i}, c_{j}) = \frac{c_{i} \cdot c_{j}}{||c_{i}|| \times ||c_{j}||}$로 측정되며, 이를 $[0, 1]$ 범위로 정규화합니다.
    * ISIA 손실($L_{\text{ISIA}}$)은 다음과 같이 정의됩니다.
    $$ L_{\text{ISIA}} = \sum_{i=1}^{N} ||c_{\text{s}_{i}} - c_{\text{t}_{i}}||_{1} + \beta \sum_{i=1}^{N_{\text{c}}} \sum_{k=1, k \ne i}^{N_{\text{c}}} D_{\text{sim}}(c_{\text{s}_{i}}, c_{\text{t}_{k}}) $$
    여기서 $c_{\text{s}_{i}}$와 $c_{\text{t}_{i}}$는 소스 및 타겟 도메인의 $i$번째 클래스 특징을 나타내고, $\beta$는 클래스 간 분리의 기여도를 조절하는 가중치입니다.

3. **범주-가이드 인스턴스-레벨 정렬 (Category-guided Instance-level Alignment, AIM):**
    * 전경(foreground) 클래스에 초점을 맞춰 적응 가중 인스턴스 매칭(Adaptive-weighted Instance Matching, AIM) 전략을 설계합니다. 전경 클래스는 도메인 간 편차가 커서 정렬이 어렵기 때문입니다.
    * 범주-레벨 적응의 복잡도를 측정하는 랭킹 리스트 $R_{\text{ac}} = \{\zeta_{k}\}$를 생성하고, 이를 정규화하여 가중치 $\eta_{k}$를 얻습니다.
    * 인스턴스 레벨 특징 표현은 연결된 영역을 통해 생성됩니다.
    * 교차 도메인 인스턴스 매칭 손실($L_{\text{AIM}}$)을 최소화하여 소스 및 타겟 도메인의 인스턴스 특징을 가깝게 만듭니다.
    $$ L_{\text{AIM}} = \sum_{i} \sum_{k \in N_{\text{ins}}} \eta_{k} \frac{1}{|R_{\text{t}_{k}}|} \sum_{r_{\text{t}} \in R_{\text{t}_{k}}} \min_{j} ||L(r_{\text{t}}, f_{\text{t}_{i}}) - s_{k_{j}}||_{1} $$
    여기서 $\eta_{k}$는 $k$번째 클래스의 인스턴스 레벨 정렬에 대한 가중치입니다.

4. **통합 목표 함수 (Integrated Objective):**
    * **초기 단계 (Initial Stage):** 타겟 도메인 레이블이 없는 상태에서 분할 손실($L_{\text{S}}^{\text{seg}}$), 적대적 손실($L_{\text{f}}^{\text{adv}}$), ISIA 손실($L_{\text{ISIA}}$), AIM 손실($L_{\text{AIM}}$) 및 판별자 손실($L_{D}$)을 포함하는 $L_{\text{init}}$를 최적화합니다.
    $$ L_{\text{init}} = \min_{G} (\lambda_{\text{seg}} L_{\text{S}}^{\text{seg}} + \lambda_{\text{adv}} L_{\text{f}}^{\text{adv}} + \lambda_{\text{ISIA}} L_{\text{ISIA}} + \lambda_{\text{AIM}} L_{\text{AIM}}) + \min_{D} \lambda_{D} L_{D} $$
    * **재훈련 단계 (Retraining Stage):** 초기 단계에서 학습된 모델을 사용하여 타겟 도메인 이미지에 대한 의사 레이블($\tilde{y}_{\text{t}}$)을 생성합니다.
    * 최종적으로 $L_{\text{S}}^{\text{seg}}$와 타겟 도메인 의사 레이블에 대한 분할 손실 $L_{\text{T}}^{\text{seg}}$를 포함하는 $L_{\text{total}}$을 최적화합니다.
    $$ L_{\text{total}} = \min_{G} (\lambda_{\text{seg}} (L_{\text{S}}^{\text{seg}} + L_{\text{T}}^{\text{seg}}) + \lambda_{\text{adv}} L_{\text{f}}^{\text{adv}} + \lambda_{\text{ISIA}} L_{\text{ISIA}} + \lambda_{\text{AIM}} L_{\text{AIM}}) + \min_{D} \lambda_{D} L_{D} $$
    $\lambda$ 값들은 각 손실의 가중치를 나타냅니다.

## 📊 Results

* **스트리트 장면 데이터셋 (GTA5→Cityscapes, SYNTHIA→Cityscapes):**
  * GTA5→Cityscapes 전이에서 mIoU 50.7%를 달성하며 최신(state-of-the-art, SOTA) 성능을 기록했습니다. 특히 `building`, `sign`, `bus` 등 혼동하기 쉬운 범주에서 높은 IoU를 보였습니다.
  * SYNTHIA→Cityscapes 전이에서는 13개 공통 클래스에서 mIoU 53.4%를 달성하여 SOTA를 기록했습니다.
  * 제안된 ISIA 및 AIM 모듈이 모델 성능 향상에 크게 기여함이 증명되었습니다 (개별 모듈 추가 시 상당한 mIoU 개선).
  * $\beta$ 및 $\lambda$ 파라미터 연구를 통해 최적의 성능을 위한 값을 확인했습니다.

* **원격 감지 이미지 데이터셋 (MBD↔IAILD, MRD↔DeepGlobe):**
  * **건물 분할 (MBD↔IAILD):** MBD→IAILD에서 81.7% mIoU를 달성하여 Source only 설정 대비 5.6% 개선을 보였습니다. IAILD→MBD에서는 71.9% mIoU로 10.1% 개선되었습니다.
  * **도로 분할 (MRD↔DeepGlobe):** MRD→DeepGlobe에서 63.9% mIoU를 달성했습니다. 도로 대상의 경우, 인스턴스 추출의 어려움과 특징 손실로 인해 AIM 모듈이 성능 저하를 가져올 수 있음을 확인했습니다.
  * 원격 감지 데이터셋에서도 SOTA 모델들과 비교하여 높은 mIoU를 달성했습니다.

* **세분화 모델의 영향:** 다양한 시맨틱 분할 모델(DeepLabv2, DeepLabv3+, FCN with HRNet-w48)을 백본으로 사용하여 도메인 적응 전략의 효과를 정량화하는 새로운 메트릭 `Normalized Adaptability Measure (NAM)`을 제안했습니다. 세분화 모델의 성능이 향상될수록 도메인 적응 전략의 개선도 함께 증가함을 보여주었습니다.

* **훈련 안정성:** 분할 손실, 적대적 손실, ISIA 손실, AIM 손실 등 여러 손실 함수의 결합에도 불구하고 훈련 과정이 안정적으로 수렴함을 보여주었습니다.

* **클래스 간 분리 vs. 클래스 내 집합:** IS(Inter-class Separation)와 IA(Intra-class Aggregation) 모두 도메인 적응에 유익하며, 함께 작동할 때 더 나은 성능을 제공함을 입증했습니다.

## 🧠 Insights & Discussion

* **범주 혼동 문제 해결:** 기존 UDA가 간과했던 범주 혼동 문제를 ISIA 메커니즘을 통해 효과적으로 해결했습니다. 동일 범주는 가깝게, 다른 범주는 멀리 떨어뜨림으로써 픽셀 오분류를 줄였습니다.
* **복잡한 도메인 시프트 처리:** 다단계 정렬 프레임워크는 이미지-수준의 낮은 레벨 적응부터 특징, 범주, 인스턴스-수준의 높은 레벨 적응까지 포괄하여 복잡한 도메인 시프트를 효과적으로 완화합니다.
* **강화된 일반화 성능:** 특히 `building`, `sign`, `bus`와 같이 혼동하기 쉬운 범주에서 IoU가 크게 향상되었는데, 이는 제안된 방법이 실제 시나리오에서 모델의 일반화 성능을 강력하게 개선했음을 시사합니다.
* **원격 감지 분야 확장:** 제안된 방법을 원격 감지 이미지에 적용하여 해당 분야에서의 UDA 가능성을 입증했으며, 이는 특정 도메인에 국한되지 않는 방법론의 범용성을 보여줍니다.
* **세분화 모델 성능의 영향 분석:** `NAM`이라는 새로운 메트릭을 통해 세분화 모델의 학습 능력이 도메인 적응 효율성에 미치는 영향을 분석했습니다. 성능이 좋은 세분화 모델은 도메인 간극의 상당 부분을 커버할 수 있으며, 특히 어려운 도메인 적응 작업에서 도메인 적응 전략의 효과를 증폭시키는 경향이 있음을 발견했습니다.
* **제한 사항 및 향후 과제:**
  * **복잡한 클래스 간 유사성:** `bus`와 `truck`처럼 시각적으로 유사한 클래스 간에는 여전히 분류 오류가 발생할 수 있습니다. 이는 시맨틱 분할 모델 자체의 특징 판별 능력의 한계 때문입니다.
  * **큰 클래스 내 다양성:** `IAILD→MBD` 작업에서와 같이 단일 범주에 속하더라도 모양, 질감, 회색조 등에서 큰 차이를 보이는 경우 픽셀 오분류가 발생할 수 있습니다.
  * **시맨틱 분할 모델의 병목 현상:** 인스턴스 추출 정확도가 세분화 모델에 의해 제한될 수 있으며, 이는 특히 도로와 같이 인스턴스 구분이 어려운 대상에서 AIM 전략의 효율성을 떨어뜨릴 수 있습니다.
  * 향후 연구에서는 더 발전된 시맨틱 분할 모델을 통합하고, 더 복잡한 개방형 세계 문제(open-world problems)로 UDA를 확장할 계획입니다.

## 📌 TL;DR

본 논문은 비지도 도메인 적응 시맨틱 분할에서 범주 혼동 문제를 해결하기 위해 다단계 정렬 프레임워크를 제안합니다. 주요 방법론은 **클래스 간 분리 및 클래스 내 집합(ISIA)**을 통해 동일 범주 특징은 모으고 다른 범주 특징은 분리하며, **적응 가중 인스턴스 매칭(AIM)**으로 어려운 범주에 대한 인스턴스 레벨 적응을 강화하는 것입니다. 이 방식은 GTA5→Cityscapes 및 SYNTHIA→Cityscapes, 그리고 원격 감지 데이터셋에서 최첨단 성능을 달성하여 도메인 간극 및 픽셀 오분류를 효과적으로 줄였음을 입증했습니다.
