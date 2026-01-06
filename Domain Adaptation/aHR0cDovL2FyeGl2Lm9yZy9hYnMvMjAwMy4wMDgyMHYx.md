# MADAN: Multi-source Adversarial Domain Aggregation Network for Domain Adaptation

Sicheng Zhao, Senior Member, IEEE, Bo Li, Xiangyu Yue, Pengfei Xu, Kurt Keutzer, Fellow, IEEE

## 🧩 Problem to Solve

도메인 적응(Domain Adaptation, DA)은 레이블링된 소스 도메인과 레이블이 거의 없거나 없는 타겟 도메인 사이의 도메인 시프트(domain shift)를 해결하여 전이 가능한 모델을 학습하는 것을 목표로 합니다. 특히, 레이블링된 데이터가 여러 소스에서 수집될 수 있는 **다중 소스 도메인 적응(Multi-source Domain Adaptation, MDA)** 시나리오가 주목받고 있습니다.

기존 MDA 방법론의 한계점은 다음과 같습니다:

- 주로 특징 레벨(feature-level) 정렬에만 초점을 맞추어 픽셀 단위 예측을 요구하는 정밀한 작업(예: 의미론적 분할)에는 부적합합니다.
- 소스와 타겟 간의 픽셀 레벨 정렬이나, 서로 다른 소스 도메인 간의 불일치(misalignment)를 고려하지 않습니다.
- 주로 이미지 분류에 초점을 맞추고 있으며, 픽셀 단위 예측을 수행하는 의미론적 분할과 같은 구조화된 예측 작업(structured prediction task)으로 직접 확장하기 어렵습니다.

## ✨ Key Contributions

- **MADAN (Multi-source Adversarial Domain Aggregation Network) 프레임워크 제안:** 다중 소스 도메인 적응을 위한 새로운 종단 간(end-to-end) 적대적 모델을 설계했습니다.
  - **픽셀 레벨 정렬 및 동적 의미 일관성(Dynamic Semantic Consistency, DSC) 손실:** 각 소스에 대해 타겟 도메인과 픽셀 레벨에서 사이클 일관성(cycle-consistency)을 유지하면서 동적 의미 일관성을 통해 의미 정보를 보존하는 적응된 도메인을 생성합니다.
  - **도메인 집합(Domain Aggregation) 메커니즘:**
    - **하위 도메인 집합 판별자(Sub-domain Aggregation Discriminator, SAD):** 서로 다른 적응된 도메인들이 구별 불가능하도록 만듭니다.
    - **교차 도메인 사이클 판별자(Cross-domain Cycle Discriminator, CCD):** 각 소스 이미지와 다른 소스에서 변환된 이미지들을 구별합니다.
- **의미론적 분할을 위한 MADAN+ 제안:** MADAN을 확장하여 다중 소스 구조화된 도메인 적응(multi-source structured domain adaptation)을 수행하는 최초의 연구입니다.
  - **카테고리 레벨 정렬(Category-level Alignment, CLA):** 다른 클래스들의 출현 빈도 균형을 맞추기 위해 도입되었습니다.
  - **문맥 인식 생성(Context-aware Generation, CAG):** 멀티 스케일 변환(multi-scale translation) 및 공간 정렬(spatial alignment)을 사용하여 전역적 의미와 지역적 세부 사항을 모두 보존하는 고품질 적응 이미지를 생성합니다.
- **광범위한 실험 및 SOTA 성능 달성:** 숫자 인식, 객체 분류, 시뮬레이션-실제 의미론적 분할 등 다양한 MDA 벤치마크 데이터셋에서 MADAN 및 MADAN+ 모델이 최신 방법론들을 크게 능가함을 입증했습니다.

## 📎 Related Works

- **단일 소스 비지도 도메인 적응 (Single-source Unsupervised Domain Adaptation, UDA):**
  - **비심층(Non-deep) 방법:** 샘플 재가중치(re-weighting) 또는 중간 부분 공간(subspace) 변환.
  - **심층(Deep) 방법:**
    - **불일치 기반(Discrepancy-based):** MMD(Maximum Mean Discrepancies), CORAL(Correlation Alignment) 등.
    - **적대적 생성 기반(Adversarial Generative):** GAN, CycleGAN, CyCADA, PixelDA 등 픽셀 레벨 정렬 고려.
    - **적대적 판별 기반(Adversarial Discriminative):** DANN, ADDA 등 도메인 혼동(domain confusion) 유도.
    - **재구성 기반(Reconstruction-based):** 재구성 손실(reconstruction loss) 최소화.
- **다중 소스 도메인 적응 (Multi-source Domain Adaptation, MDA):**
  - **초기 심층(Shallow) 모델:** 특징 표현 학습, 사전 학습된 분류기 결합.
  - **최신 심층(Deep) MDA 방법:** MDAN, DCTN, MMN, MDDA 등. 주로 이미지 분류를 위한 특징 레벨 정렬에만 초점을 맞추며, 각 소스-타겟 쌍을 정렬하고 단일 또는 여러 개의 태스크 모델을 훈련합니다. MADAN은 픽셀 레벨 정렬과 도메인 집합을 추가로 고려합니다.

## 🛠️ Methodology

MADAN은 크게 세 가지 구성 요소로 이루어져 있으며, MADAN+는 의미론적 분할에 특화된 추가 기능을 포함합니다.

1. **동적 적대적 이미지 생성 (Dynamic Adversarial Image Generation, DAIG)**

   - **픽셀 레벨 GAN 손실 ($L_{S_i \to T}^{\text{GAN}}$, $L_{T \to S_i}^{\text{GAN}}$):** 각 소스 도메인 $S_i$에 대해 생성자 $G_{S_i \to T}$가 타겟 도메인 $T$와 시각적으로 유사한 적응된 이미지 $G_{S_i \to T}(X_i)$를 생성하도록 학습합니다. 판별자 $D_T$는 실제 타겟 이미지 $X_T$와 생성된 이미지를 구별합니다. 역 매핑 $G_{T \to S_i}$와 판별자 $D_i$도 유사하게 학습됩니다.
   - **사이클 일관성 손실 ($L_{S_i \leftrightarrow T}^{\text{cyc}}$):** 매핑이 서로 모순되지 않도록 $G_{T \to S_i}(G_{S_i \to T}(x_i)) \approx x_i$ 및 그 역방향을 강제합니다.
   - **동적 의미 일관성 (Dynamic Semantic Consistency, DSC) 손실 ($L_{S_i}^{\text{DSC}}$):** 적응된 이미지가 원본 소스 이미지와 동일한 의미 정보를 보존하도록 합니다. 사전 학습된 태스크 모델 $F_i$의 소스 예측과 동적으로 업데이트되는 태스크 모델 $F_A (=F)$의 적응된 이미지 예측 간의 KL 발산(KL divergence)을 최소화합니다.

2. **적대적 도메인 집합 (Adversarial Domain Aggregation, ADA)**

   - **하위 도메인 집합 판별자 (Sub-domain Aggregation Discriminator, SAD) ($L_{S_i}^{\text{SAD}}$):** 서로 다른 적응된 도메인들($G_{S_i \to T}(X_i)$)이 서로 구별 불가능하도록 만듭니다.
   - **교차 도메인 사이클 판별자 (Cross-domain Cycle Discriminator, CCD) ($L_{S_i}^{\text{CCD}}$):** 각 소스 도메인 $S_i$의 실제 이미지 $X_i$와 다른 소스 도메인의 적응된 이미지를 다시 $S_i$로 전송한 이미지($G_{T \to S_i}(G_{S_j \to T}(X_j))$)를 구별합니다.

3. **특징 정렬 태스크 학습 (Feature-aligned Task Learning, FTL)**
   - **태스크 손실 ($L_{\text{task}}$):** ADA를 통해 응집된 적응 도메인 $X'$와 해당 레이블 $Y$를 사용하여 단일 태스크 모델 $F$ (분류 또는 분할)를 학습합니다.
   - **특징 레벨 정렬 (Feature-level Alignment, FLA) 손실 ($L^{\text{FLA}}$):** 특징 판별자 $D_F^f$를 사용하여 응집된 도메인의 특징 $F_f(X')$와 타겟 도메인의 특징 $F_f(X_T)$를 정렬합니다.

**MADAN+를 위한 의미론적 분할 적응 (Segmentation Adaptation):**

- **카테고리 레벨 정렬 (Category-level Alignment, CLA) ($L^{\text{CLA}}$):** FLA를 확장하여 각 클래스에 대한 지역 특징 정렬을 수행합니다. 그리드 단위(grid-wise) (의사) 레이블 $\tilde{\aleph}_n^l(x_d)$을 사용하여 각 클래스 $l$에 대해 판별자 $D_l^C$가 적응된 이미지와 타겟 이미지의 클래스별 출현 빈도를 균형 잡히게 합니다.
- **문맥 인식 생성 (Context-aware Generation, CAG) ($L^{\text{CAG}}$):** 이미지 생성 품질을 향상시킵니다.
  - **멀티 스케일 변환:** 적응된 및 타겟 이미지를 여러 크기 $\{C_1, \dots, C_K\}$로 자르고 고정 해상도로 크기를 조정합니다.
  - **공간 정렬:** 공통 중심점을 기준으로 균일하게 잘라 클래스의 공간 분포를 유지합니다.

**전체 학습 목표 함수:**
$$ L*{\text{MADAN}} = \sum*{i=1}^{M} \left[ L_{S_i \to T}^{\text{GAN}} + L_{T \to S_i}^{\text{GAN}} + L_{S_i \leftrightarrow T}^{\text{cyc}} + L_{S_i}^{\text{DSC}} + L_{S_i}^{\text{SAD}} + L_{S_i}^{\text{CCD}} \right] + L*{\text{task}} + L^{\text{FLA}} $$
$$ L*{\text{MADAN+}} = L^{\text{CAG}} + \sum*{i=1}^{M} \left[ L*{S*i}^{\text{SAD}} + L*{S*i}^{\text{CCD}} \right] + L*{\text{task}} + L^{\text{FLA}} + L^{\text{CLA}} $$
훈련은 여러 단계로 나누어 수행되며, 생성자, 판별자 및 태스크 네트워크를 번갈아 최적화합니다.

## 📊 Results

- **숫자 인식, 객체 분류 (Digits-five, Office-31, Office+Caltech-10, Office-Home):**
  - MADAN은 최신 MDA 방법론 대비 **2.8% ~ 4.6%**의 분류 정확도 향상을 달성하며 SOTA 성능을 기록했습니다.
  - 단순히 여러 소스를 결합하여 단일 소스 DA를 수행하는 것보다 MADAN의 명시적인 도메인 집합 전략이 더 효과적이었습니다.
- **의미론적 분할 (GTA, SYNTHIA → Cityscapes, BDDS):**
  - **MADAN+**는 FCN-VGG16 백본에서 Cityscapes에 대해 SOTA MDA 방법론 대비 **1.4%** (MDAN 대비), BDDS에 대해 **5.3%**의 mIoU(mean Intersection-over-Union) 향상을 보였습니다.
  - DeepLabV2-ResNet101 백본에서도 MADAN+는 Cityscapes에서 **2.3%**, BDDS에서 **3.1%**의 mIoU 향상을 달성했습니다.
  - MADAN+는 16개 카테고리 중 6~9개에서 최고 cwIoU(class-wise IoU)를 달성하여 픽셀 단위 적응에 대한 우수성을 입증했습니다.
- **절제 연구 (Ablation Study):**
  - 제안된 DSC 손실은 기존 SC(Semantic Consistency) 손실보다 더 나은 mIoU 결과를 가져왔습니다.
  - SAD, CCD, DSC, FLA, CLA, CAG를 포함한 모든 구성 요소들이 성능 향상에 긍정적으로 기여하며, 상호 보완적인 역할을 함을 확인했습니다.
- **정성적 결과 (Qualitative Results):**
  - 의미론적 분할 시각화에서 MADAN+의 예측은 원본보다 보행자, 자전거 운전자 등의 윤곽이 훨씬 명확하고 실제 정답에 가깝게 개선되었습니다.
  - 이미지 변환 시각화에서는 적응된 이미지가 타겟 도메인의 스타일을 따르면서도 의미론적 정보를 잘 보존함을 보여주었습니다.
- **특징 시각화 (Feature Visualization, t-SNE):**
  - 적응 후 소스와 타겟 도메인 간의 특징 분포가 더 잘 정렬되어, 학습된 특징의 전이 가능성이 향상되었음을 보여줍니다.
- **모델 해석 가능성 (Model Interpretability, Grad-Cam):**
  - 적응 후 생성된 관심 영역(attention map)은 모델이 배경의 복잡한 영역 대신 계산기나 헬멧 등 작업에 더 중요한 판별적이고 전이 가능한 영역에 집중함을 보여주었습니다.

## 🧠 Insights & Discussion

- **다중 소스 DA의 중요성:** 단순히 소스 도메인을 결합하거나 단일 소스 DA를 사용하는 것보다, 여러 소스의 상호 보완성을 탐색하는 다중 소스 DA가 우수한 성능을 제공합니다. 이는 특히 서로 다른 소스 간의 잠재적 간섭을 해결하는 데 중요합니다.
- **픽셀 레벨 정렬의 효과:** 기존 MDA 방법론이 특징 레벨 정렬에만 집중했던 것과 달리, MADAN이 픽셀 레벨 정렬과 동적 의미 일관성을 결합함으로써 특히 의미론적 분할과 같은 픽셀 단위 예측 작업에서 성능을 크게 향상시켰습니다.
- **도메인 집합의 가치:** SAD와 CCD를 통한 적응된 도메인들의 명시적인 집합은 여러 소스 도메인의 데이터를 통합하여 보다 통일된 학습 공간을 제공함으로써 모델의 견고성을 높입니다.
- **MADAN+의 전문성:** MADAN+는 카테고리 레벨 정렬과 문맥 인식 생성을 통해 픽셀 단위 적응의 고유한 문제를 해결하며, MADAN 대비 의미론적 분할에서 현저한 성능 향상을 보였습니다.
- **오라클 모델과의 성능 격차:** 현재 모든 DA 알고리즘과 타겟 도메인에 직접 학습된 오라클(oracle) 모델 간에는 여전히 큰 성능 격차가 존재하며, 이는 DA 연구의 추가 노력이 필요함을 시사합니다.
- **향후 연구 방향:**
  - 멀티 모달 DA(예: 이미지와 LiDAR 데이터 결합)를 통한 적응 성능 향상.
  - 신경망 아키텍처 탐색(Neural Architecture Search)과 같은 기술을 활용하여 MADAN의 계산 효율성 개선.
  - 다양한 소스 및 각 소스 내 샘플의 상대적 중요성을 자동으로 가중치 부여하는 방법 연구.

## 📌 TL;DR

다중 소스 도메인 적응(MDA)은 여러 소스 도메인의 레이블 데이터를 사용하여 레이블 없는 타겟 도메인에 일반화할 수 있는 모델을 학습하는 것을 목표로 하지만, 기존 방법은 픽셀 레벨 정렬이나 소스 간 도메인 불일치를 제대로 다루지 못했습니다. 본 논문은 픽셀 레벨에서 사이클 일관성을 가진 이미지 생성을 통해 각 소스에 대한 적응 도메인을 만들고, 동적 의미 일관성 손실로 의미를 보존하는 **MADAN (Multi-source Adversarial Domain Aggregation Network)**을 제안합니다. MADAN은 하위 도메인 집합 판별자와 교차 도메인 사이클 판별자를 사용하여 적응된 도메인들을 더욱 응집시키며, 특징 레벨 정렬을 통해 타겟 도메인과의 간극을 좁힙니다. 특히, 의미론적 분할을 위해 카테고리 레벨 정렬과 문맥 인식 생성을 추가한 **MADAN+**는 미세한 픽셀 단위 예측에 탁월한 성능을 보입니다. 광범위한 실험을 통해 MADAN과 MADAN+가 숫자 인식, 객체 분류 및 의미론적 분할에서 최신 방법론을 크게 능가하는 SOTA 성능을 달성함을 입증했습니다.
