# Centering the Value of Every Modality: Towards Efficient and Resilient Modality-agnostic Semantic Segmentation

Xu Zheng, Yuanhuiyi Lyu, Jiazhou Zhou, and Lin Wang

## 🧩 Problem to Solve

기존 멀티모달 시맨틱 분할(semantic segmentation) 연구는 임의의 수의 모달리티를 융합하는 데 충분히 탐구되지 않았습니다. 특히, RGB 모달리티를 중심으로 다른 모달리티를 보조적으로 사용하는 비대칭 아키텍처가 일반적이었으나, 야간과 같은 특정 환경에서 RGB 모달리티가 취약할 수 있다는 문제가 있습니다. 따라서, 융합 모델이 견고한(robust) 모달리티와 취약한(fragile) 모달리티를 구별하고, 가장 견고하고 취약한 모달리티를 모두 통합하여 더욱 탄력적인(resilient) 멀티모달 프레임워크를 학습하는 것이 중요합니다.

## ✨ Key Contributions

* **MAGIC(Modality-agnostic) 프레임워크 제안:** 다양한 백본(compact에서 고성능 모델까지)에 유연하게 결합할 수 있는 효율적이고 견고한 시맨틱 분할 프레임워크인 MAGIC을 제안합니다.
* **두 가지 플러그 앤 플레이 모듈:**
  * **Multi-modal Aggregation Module (MAM) 개발:** 멀티모달 배치에서 특징을 효율적으로 처리하고 특정 모달리티에 의존하지 않고 상보적인(complementary) 장면 정보를 추출합니다.
  * **Arbitrary-modal Selection Module (ASM) 제안:** MAM에서 집계된 특징을 벤치마크로 활용하여 유사성 점수를 기반으로 멀티모달 특징을 랭킹화하고, 훈련 중 동적으로 모달리티 불가지론적(modality-agnostic) 장면 특징을 융합하며, 추론 시 임의의 모달리티 입력에 대한 모델의 견고성을 향상시킵니다.
* **RGB 의존성 제거 및 센서 실패 극복:** 제안된 방법은 RGB 모달리티에 대한 의존성을 제거하고 센서 실패에 더 잘 대처하면서도 분할 성능을 크게 향상시킵니다.
* **최첨단 성능 달성 및 효율성:**
  * 일반적인 멀티모달 설정에서 최첨단 성능을 달성하며 모델 파라미터를 60% 감소시켰습니다.
  * 새로운 모달리티 불가지론적 설정에서 기존 방법보다 mIoU에서 +19.41%의 큰 폭으로 성능을 뛰어넘었습니다.

## 📎 Related Works

* **Semantic Segmentation:** FCN, Multi-scale features, Attention blocks, Edge cues, Context priors, Transformer-based methods (SegFormer, Swin Transformer 등)와 같은 기존 시맨틱 분할 기법을 백본으로 활용합니다.
* **Multi-modal Semantic Segmentation:**
  * RGB와 Depth, Thermal, Polarization, Event, LiDAR 등 보완적 모달리티를 융합하는 다양한 초기 연구들이 존재했습니다.
  * 최근에는 듀얼 모달리티 융합에서 다중 모달리티 융합으로 확장하려는 시도(예: MCubeSNet)가 있었습니다.
  * 기존 아키텍처는 주로 RGB를 중심으로 다른 모달리티를 보조적으로 사용하는 비대칭 구조(예: CMNeXt)를 가집니다. 특히 CMNeXt는 RGB 브랜치에 의존하여 보조 모달리티 정보를 융합하지만, 본 연구는 특정 센서에 의존하지 않고 모든 모달리티를 동등하게 취급하여 견고성을 높이는 데 중점을 둡니다.

## 🛠️ Methodology

MAGIC 프레임워크는 Multi-modal Aggregation Module (MAM)과 Arbitrary-modal Selection Module (ASM) 두 가지 핵심 모듈로 구성됩니다.

1. **입력:** RGB, Depth, LiDAR, Event의 네 가지 모달리티에서 미니 배치 ${r, d, l, e}$를 입력으로 받습니다.
2. **특징 추출:** 각 모달리티의 입력은 가중치를 공유하는 인코더($F_r, F_d, F_l, F_e$)를 통해 고수준 특징 ${f_r, f_d, f_l, f_e}$로 변환됩니다:
    $$ \{f_r, f_d, f_l, f_e\} = F_r(r), F_d(d), F_l(l), F_e(e) $$
3. **Multi-modal Aggregation Module (MAM):**
    * 추출된 특징들은 `Conv` 레이어와 병렬 풀링 레이어(3x3, 7x7, 11x11)를 거쳐 다양한 스케일의 공간 정보를 탐색합니다.
    * 이 특징들은 집계되어 `Sigmoid` 활성화가 적용된 `Conv` 레이어를 거칩니다.
    * 병렬 MLP 레이어가 이 멀티모달 특징들과 원본 특징을 집계하여 의미론적 특징 $f_{se}$를 생성합니다.
    * $f_{se}$는 분할 헤드(SegHead)를 통해 MAM 예측 $P_m$을 생성하며, 이는 GT(Ground Truth) $Y$에 의해 다음과 같이 교차 엔트로피 손실 $L_M$로 감독됩니다:
        $$ L_M = -\sum_{0}^{K-1} Y \cdot \log(P_m) $$
4. **Arbitrary-modal Selection Module (ASM):**
    * **Cross-modal Semantic Similarity Ranking:** MAM에서 얻은 의미론적 특징 $f_{se}$를 벤치마크로 사용하여 각 모달리티 특징 ${f_r, f_d, f_l, f_e}$와의 코사인 유사성을 계산하고 랭킹화합니다.
        $$ f_{rf}, f_{rm} = \text{Rank}(\text{Cos}(\{f_r, f_d, f_l, f_e\}, f_{se})) $$
        여기서 $f_{rf}$는 가장 견고한(Top-1) 특징과 가장 취약한(Last-1) 특징을 포함하며, $f_{rm}$은 나머지 특징들입니다. $f_{rf}$는 다른 MAM으로 전달되어 최종 salient 특징 $f_{sa}$를 생성하고, 이는 SegHead를 통해 ASM 예측 $P_s$를 생성합니다.
    * **Prediction-level Arbitrary-modal Selection Training (선택적 GT 스무딩):**
        $P_m$의 로짓에 `argmax` 연산을 적용하고, 이를 GT 라벨 $y$와 통합하여 마스크 $M$을 생성합니다. $M$은 `argmax(P_m)`과 $y$의 픽셀별 범주 예측이 일치할 때 로짓을 유지하고, 불일치할 때는 버립니다. 이 마스크 $M$은 $P_s$의 감독 신호로 사용되며, Arbitrary-modal Selection Loss $L_S$는 다음과 같습니다:
        $$ L_S = -\sum_{0}^{K-1} M \cdot \log(P_s) $$
    * **Cross-modal Semantic Consistency Training:** $f_{rm}$에 대해 의미론적 일관성 훈련을 적용합니다. 이는 나머지 특징들과 의미론적 특징 $f_{sa}$ 간의 코사인 유사성(상관관계 $c_1, c_2$)을 정렬하는 것을 목표로 합니다. 일관성 훈련 손실 $L_C$는 다음과 같습니다:
        $$ L_C = \sum_{0}^{K-1} \left(c_1 \log \frac{c_1}{\frac{1}{2}(c_1+c_2)} + c_2 \log \frac{c_2}{\frac{1}{2}(c_1+c_2)}\right) $$
5. **총 손실:** 전체 프레임워크는 $L_M, L_S, L_C$의 선형 조합인 총 손실 $L$을 최소화하여 훈련됩니다:
    $$ L = L_M + \lambda L_S + \beta L_C $$
    훈련 시에만 ASM이 활용되며, 추론 시에는 백본과 MAM이 사용됩니다.

## 📊 Results

* **Dataset:** DELIVER (RGB, Depth, Event, LiDAR) 및 MCubeS (Image, AoLP, DoLP, NIR) 데이터셋에서 평가되었습니다.
* **Multi-modal Semantic Segmentation:**
  * DELIVER 데이터셋에서 CMNeXt 대비 mIoU +1.33% 향상, MCubeS에서 +1.47% 향상.
  * MAGIC은 CMNeXt 파라미터의 42%만 사용(24.73M vs 58.73M)하면서도 우수한 성능을 보였습니다.
* **Modality-agnostic Semantic Segmentation:**
  * 임의의 모달리티 입력(RGB 데이터 부재 시 포함)에 대해 CMNeXt 대비 DELIVER에서 +19.41% mIoU, MCubeS에서 +14.97% mIoU의 큰 폭 성능 향상을 달성했습니다.
  * 특히 Depth-only 시나리오(RGB 없음)에서 CMNeXt 0.81% mIoU 대비 MAGIC 57.59% mIoU로 +56.78% 압도적인 성능 차이를 보였습니다.
  * SegFormer-B0 백본(CMNeXt 파라미터의 6%에 불과)을 사용해서도 CMNeXt를 각각 +15.24%, +9.39% mIoU로 능가했습니다.
* **시각화 결과:** 야간, 과노출, 안개와 같은 까다로운 조건에서도 MAGIC은 임의의 입력에 대해 일관되게 잘 작동하는 반면, CMNeXt는 대부분의 시나리오에서 취약함을 보였습니다.

## 🧠 Insights & Discussion

* **모든 모달리티의 가치:** 기존 연구들이 RGB 모달리티의 필수성을 강조했던 것과 달리, 본 연구는 모든 모달리티가 고유한 가치를 지니며 어떤 것도 간과해서는 안 된다는 점을 입증했습니다. 각 모달리티를 추가할 때마다 성능 향상이 관찰되었습니다.
* **특징 시각화:** RGB 특징에 비해 MAM에서 추출된 의미론적 특징과 ASM에서 추출된 Salient 특징이 훨씬 더 풍부한 장면 세부 정보를 포착하는 것을 시각적으로 확인했습니다. 이는 MAM과 ASM이 멀티모달 입력의 잠재력을 효과적으로 활용함을 의미합니다.
* **손실 함수의 효과:** 제안된 MAM ($L_M$), ASM ($L_S$), 그리고 일관성 훈련 손실 ($L_C$)이 각각의 백본 모델에서 mIoU 성능을 일관되게 향상시키는 것으로 나타났습니다.
* **한계 및 향후 방향:** 특정 데이터 특성으로 인해 일부 시나리오에서 최적의 성능을 보이지 않는 경우가 있었습니다. 향후에는 모달리티 불가지론적 플러그 앤 플레이 모듈의 일관성과 성능 향상에 초점을 맞출 예정입니다.

## 📌 TL;DR

이 논문은 기존 멀티모달 시맨틱 분할 모델의 RGB 의존성과 임의의 모달리티 융합 한계를 해결하기 위해 **MAGIC**이라는 새로운 프레임워크를 제안합니다. MAGIC은 **Multi-modal Aggregation Module (MAM)**로 상보적인 장면 정보를 효율적으로 추출하고, **Arbitrary-modal Selection Module (ASM)**로 견고하고 취약한 특징을 선별적으로 융합하여 모델의 견고성을 강화합니다. 결과적으로 MAGIC은 파라미터를 60% 감소시키면서도 기존 멀티모달 및 특히 모달리티 불가지론적 설정에서 mIoU를 크게 향상시키며 최첨단 성능을 달성했습니다.
