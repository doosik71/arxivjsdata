# AlignZeg: Mitigating Objective Misalignment for Zero-shot Semantic Segmentation

Jiannan Ge, Lingxi Xie, Hongtao Xie, Pandeng Li, Xiaopeng Zhang, Yongdong Zhang, and Qi Tian

## 🧩 Problem to Solve

제로샷 시각 인식(Zero-shot Visual Recognition)에서 발생하는 심각한 문제 중 하나는 "목표 불일치(objective misalignment)"입니다. 이는 학습 목표가 학습된 클래스(seen classes)의 인식 정확도 향상에 초점을 맞추지만, 실제 추구해야 할 목표는 학습되지 않은 클래스(unseen classes)의 인식이라는 점입니다. 이 문제는 픽셀 수준의 강력한 감독(supervision)을 요구하는 제로샷 이미지 분할(Zero-shot Image Segmentation)에서 더욱 두드러집니다. 기존 CLIP 기반 방법들은 비효율적이거나 이러한 근본적인 목표 불일치 문제를 간과하여, 모델이 학습된 클래스에 과적합되고 미학습 클래스에 대한 예측 편향이 발생하여 부정확한 분할 결과를 초래합니다 (예: "tree"를 "grass" 또는 "bush"로 오분류).

## ✨ Key Contributions

* **AlignZeg 아키텍처 제안**: 제로샷 분할 목표에 더 잘 부합하도록 제안 추출, 분류, 보정 등 분할 파이프라인의 전반적인 개선을 담은 새로운 아키텍처인 AlignZeg를 제안합니다.
* **Mutually-Refined Proposal Extraction (MRPE)**: 마스크 쿼리(mask queries)와 시각적 특징(visual features) 간의 상호 작용을 통해 상세한 클래스 불가지론적(class-agnostic) 마스크 제안 추출을 용이하게 합니다.
* **Generalization-Enhanced Proposal Classification (GEPC)**: 합성 데이터(synthetic data)와 다중 배경 프로토타입(multiple background prototypes)을 도입하여 더 일반화 가능한 특징 공간을 할당합니다.
* **Predictive Bias Correction (PBC)**: 추론 단계에서 클래스 지표(class indicator)를 사용하여 잠재적인 미학습 클래스 제안을 식별하고 예측 편향을 보정하는 사후 처리(post-process)를 수행합니다.
* **성능 향상**: 제로샷 의미론적 분할 성능을 크게 향상시키며, 특히 hIoU에서 평균 3.8% 증가, 미학습 클래스 식별(mIoU(U))에서 7.1%의 상당한 개선을 달성하여 목표 불일치 문제를 효과적으로 완화함을 입증했습니다.

## 📎 Related Works

* **제로샷 시각 인식 (Zero-shot visual recognition)**:
  * 초기 방법들: 속성(attributes)이나 단어 벡터(word vectors) 같은 의미론적 서술자(semantic descriptors)를 사용하여 학습된/미학습 클래스 간의 간극을 연결했으나, 학습된 클래스에 치중하여 목표 불일치 문제를 겪었습니다.
  * 최근 방법들: CLIP과 같은 대규모 시각-언어 모델을 활용하여 발전했으나, 주로 이미지 수준의 인식에 중점을 두어 픽셀 수준에서는 제한적이었습니다.
* **제로샷 의미론적 분할 (Zero-shot semantic segmentation)**:
  * 초기 방법들: 시각적 내용과 클래스 설명을 공유 공간에서 연결하는 데 중점을 두었습니다.
  * CLIP 시대: SimBaseline, Zegformer와 같은 2단계 접근법(클래스 불가지론적 마스크 제안 추출 후 제로샷 분류)을 도입하여 CLIP의 능력을 픽셀 수준으로 확장했으나 비효율적이었습니다. ZegCLIP, DeOP, SAN과 같은 모델은 CLIP 인코더를 직접 사용하여 효율성을 높였으나, 학습된 클래스에 대한 분류 손실에만 초점을 맞춰 과적합과 목표 불일치 문제가 발생했습니다.
* **예측 편향 (Prediction bias)**: 제로샷 시각 인식에서 분류기의 학습된 클래스에 대한 예측 편향을 줄이기 위한 노력이 있었습니다 (예: 적응형 신뢰도 스무딩, 엔트로피 기반 점수 조정, 거리 기반 게이팅 네트워크). 그러나 이를 픽셀 수준 작업에 적용하는 것은 어려웠습니다. PMOSR은 미학습 마스크 추출을 위한 미지 프로토타입을 제안했으나, 추출 정확도에 민감한 한계가 있었습니다.

## 🛠️ Methodology

AlignZeg는 다음 세 가지 핵심 구성 요소를 통해 제로샷 작업과의 목표 불일치를 완화합니다.

* **3.1 Mutually-Refined Proposal Extraction (MRPE)**
  * **상호 개선 방식**: 마스크 쿼리 $Q_m$와 시각적 특징 $Z_f$를 상호적으로 개선하여 고품질의 클래스 불가지론적 마스크를 추출합니다.
  * **특징 융합**: 학습 가능한 이미지 인코더 ($Z_{enc}$)와 고정된 CLIP 이미지 인코더 ($Z_{clip}$)에서 보완적인 시각적 특징을 추출한 후 $Z_f = Z_{enc} + Z_{clip}$로 융합합니다.
  * **쿼리/이미지 개선**: 교차-어텐션 메커니즘을 사용하여 $Q_m$를 $Z_f$로 개선하고 ($Q'_m$), $Z_f$를 $Q'_m$로 개선합니다 ($Z'_f$).
  * **마스크 제안 생성**: 최종 개선된 쿼리 $Q''_{m}$와 시각적 특징 $Z'_f$를 사용하여 최종 마스크 제안 $M = Q''_{m} \cdot Z'^{\top}_{f}$를 생성합니다.

* **3.2 Generalization-Enhanced Proposal Classification (GEPC)**
  * **클래스 임베딩 및 제안 특징 추출**: CLIP 텍스트 인코더를 사용하여 클래스 프로토타입 $T$와 배경 토큰 $g$를 추출합니다. 제안 특징 $F$를 추출하고 $\phi_{cls}(\cdot)$를 통해 $F_{cls}$로 정제한 뒤, 예측 점수 $P = [F_{cls} \cdot T^{\top}, F_{cls} \cdot g^{\top}]$를 계산합니다.
  * **Feature Expansion Strategy (FES)**:
    * 학습된 클래스 분포를 넘어선 특징을 합성하여 미학습 클래스를 위한 특징 공간을 확장합니다.
    * 미니배치 내 학습된 클래스 제안 특징 $F^{b,s}_{cls}$와 해당 클래스 임베딩 $T^{b,s}$를 Mixup 기법(Beta 분포 $Beta(\alpha, \alpha)$에서 샘플링된 $\beta$ 사용)으로 혼합하여 가상 제안 특징 $F^{v}_{cls}$ 및 가상 프로토타입 $T^{v}$를 생성합니다.
    * 생성된 가상 특징에 대해 $L_{vir} = CE(P_v, Y_v)$ 손실을 적용하여 최적화하며, $Y_v$는 미학습 클래스임을 나타내는 레이블입니다.
  * **Background Diversity Strategy (BDS)**:
    * 단일 고정 프로토타입의 한계를 극복하기 위해 $M$개의 다중 배경 프로토타입 $G=[g_1; \dots; g_M]$를 도입하여 배경 범주의 다양성을 보존합니다.
    * 각 제안 특징에 대해 다중 배경 로짓 $p_{mg}$를 계산하고 가중 합산을 통해 최종 배경 점수 $p_g$를 얻습니다.
    * $L_{reg}$ 손실을 통해 배경 프로토타입 간의 거리를 제약하여 다양성을 유지합니다.

* **3.3 Predictive Bias Correction (PBC)**
  * 학습된 클래스에 대한 예측 편향을 완화하기 위해 각 제안이 잠재적으로 미학습 범주를 포함하는지 식별합니다.
  * **클래스 지표 학습**: 두 개의 완전 연결 레이어와 시그모이드 레이어로 구성된 이진 분류 모델 $\phi_{bc}(\cdot)$를 사용하여 각 제안에 대한 클래스 지표 $I \in [0,1]$를 학습합니다.
  * **긍정/부정 제안 식별**: 학습 과정에서 학습된 클래스 제안을 긍정 제안(레이블 0)으로, 학습된 클래스 영역과 겹치지 않는 배경 제안 중 높은 손실 값을 보이는 상위 K개 제안을 부정 제안(레이블 1)으로 설정합니다.
  * **추론 단계 보정**: 학습된 $\phi_{bc}(\cdot)$에서 얻은 $I$를 사용하여 임계값 $\gamma$를 초과하는 (미학습 클래스일 가능성이 높은) 제안에 대해 학습된 클래스 점수를 최소값 $v_{min}$으로 억제합니다. ($P_{bc}[i,j] = v_{min}$ if $I[i] > \gamma$ and $C_{test}[j] \in S$, else $P[i,j]$).

* **3.4 Optimization**
  * 전체 손실은 $L=L_{bc}+\lambda_1 \cdot L_{ce}+\lambda_2 \cdot L_{mask}+\lambda_3 \cdot L_{vir}+\lambda_4 \cdot L_{reg}$입니다.

## 📊 Results

* **일반화된 제로샷 의미론적 분할 (GZS3)**: PASCAL VOC 2012, COCO-Stuff 164K, PASCAL Context 벤치마크에서 SOTA 성능을 달성했습니다. 특히 핵심 평가 지표인 hIoU에서 각각 5.3%, 3.8%, 2.4% 향상되었으며, 이는 주로 미학습 클래스mIoU(U) 지표에서 각각 7.5%, 8.6%, 5.1%의 탁월한 성능 향상에 기인합니다.
* **제로샷 의미론적 분할 (ZS3)**: 미학습 범주만 고려하는 ZS3 설정에서도 VOC와 COCO 데이터셋 모두에서 mIoU와 pAcc에서 최적의 성능을 달성했습니다. COCO 데이터셋에서 ZegCLIP 대비 mIoU 12.1%, SAN 대비 6.0%의 상대적 개선을 보였습니다.
* **다른 데이터셋으로의 일반화 능력**: 다양한 데이터셋으로의 일반화 능력 평가에서도 AlignZeg는 COCO-Stuff 164K로 학습 후 P-59 및 VOC 데이터셋에서 ZegCLIP 대비 각각 13.1%, 1.1% 향상된 성능을 보였습니다.
* **구성 요소별 효과 분석**: Ablation Study 결과, PBC는 초기 hIoU를 평균 2.15% 증가시켰고, MRPE는 클래스 불가지론적 마스크의 정확도를 높여 pAcc와 hIoU를 향상시켰습니다. FES와 BDS는 각각 평균 1.85%의 hIoU 개선을 추가하여 최종 최적의 성능을 달성했습니다. 이는 각 구성 요소가 목표 불일치 완화에 기여함을 입증합니다.
* **시각화**: 시각화 결과는 Baseline이 미학습 범주를 학습된 범주로 오분류하는 경향을 보이는 반면, AlignZeg는 이러한 편향을 완화하고 학습된/미학습 범주 모두에 대해 더 정확한 분할 마스크를 생성함을 보여줍니다. T-SNE 플롯은 AlignZeg의 특징이 더 넓게 분산되어 있으며 카테고리 간 경계가 더 명확함을 보여주어 GEPC의 효과를 입증합니다.

## 🧠 Insights & Discussion

AlignZeg는 "목표 불일치"라는 핵심 문제를 해결함으로써 제로샷 의미론적 분할의 성능을 크게 향상시켰습니다. MRPE, GEPC, PBC와 같은 구성 요소들은 제안 추출, 특징 공간 일반화, 예측 편향 보정이라는 분할 파이프라인의 핵심 측면을 개선하여 미학습 클래스에 대한 모델의 일반화 능력을 강화합니다. 특히 PBC는 제안 수준에서 잠재적 미학습 클래스를 식별하는 새로운 효과적인 단계를 도입하여 예측 편향을 명시적으로 완화합니다.

**한계점**: AlignZeg가 오분류된 영역을 효과적으로 보정하지만, 일부 분할 경계는 여전히 부정확한 영역을 나타냅니다. 이는 제안의 일부만 보정하기 때문일 수 있으며, 이러한 문제를 완화하기 위해 분할 후처리 기술을 통합할 수 있습니다.

## 📌 TL;DR

* **문제**: 제로샷 의미론적 분할은 학습된 클래스에 대한 모델의 편향(목표 불일치)으로 인해 미학습 클래스에 대한 성능이 저하됩니다.
* **해결책**: AlignZeg는 마스크 쿼리와 시각적 특징 간의 상호 개선을 통한 **Mutually-Refined Proposal Extraction (MRPE)**으로 더 나은 클래스 불가지론적 마스크를 생성하고, 합성 데이터와 다중 배경 프로토타입을 활용하는 **Generalization-Enhanced Proposal Classification (GEPC)**으로 특징 공간을 일반화하며, 추론 시 **Predictive Bias Correction (PBC)**을 통해 잠재적 미학습 제안에 대한 예측 편향을 보정하여 이 문제를 완화합니다.
* **결과**: AlignZeg는 GZS3 및 ZS3 벤치마크에서 SOTA 성능을 달성하며 (GZS3에서 hIoU 3.8%, mIoU(U) 7.1% 증가), 목표 불일치를 효과적으로 완화하여 미학습 클래스 인식률을 크게 향상시켰습니다.
