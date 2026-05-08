# CNN in MRF: Video Object Segmentation via Inference in A CNN-Based Higher-Order Spatio-Temporal MRF

Linchao Bao, Baoyuan Wu, Wei Liu

## 🧩 Problem to Solve

본 논문은 비디오 객체 분할(Video Object Segmentation) 문제를 다룹니다. 특히, 입력 비디오의 첫 번째 프레임에 객체 마스크가 주어지는 준지도 학습(semi-supervised) 설정에 초점을 맞춥니다. 이 작업은 객체의 의미론적 클래스를 미리 알 수 없는 '클래스 불가지(class-agnostic)' 접근 방식을 따릅니다.

기존 연구들은 크게 두 가지 한계를 가집니다:

1. **CNN 기반 방법의 한계:** 일반적으로 각 비디오 프레임을 개별적으로 처리하거나, 시간 축을 따라 정보를 전파하기 위한 단순한 휴리스틱만 사용합니다. 이로 인해 시간에 따른 일관성 유지가 어렵습니다.
2. **전통적인 그래프 기반(MRF/CRF) 방법의 한계:** 심층 신경망의 강력한 표현 능력을 충분히 활용하지 못합니다.

따라서, 이 연구는 CNN의 강력한 객체 외형/형태 정보 활용 능력과 MRF 모델의 시공간적 일관성 모델링 능력을 결합하여, 복잡한 객체 상호작용, 폐색(occlusion), 움직임, 객체 변형 등이 빈번한 실제 환경(예: DAVIS 2017 챌린지)에서의 비디오 객체 분할 성능 저하 문제를 해결하고자 합니다.

## ✨ Key Contributions

- **새로운 시공간 마르코프 랜덤 필드(MRF) 모델 제안:** 비디오 객체 분할을 위해 CNN을 통해 공간적 포텐셜을 인코딩하는 혁신적인 MRF 모델을 제안합니다. 이는 픽셀 간의 고차원적 의존성을 모델링하여 객체 인스턴스의 전체적인 분할(holistic segmentation)을 강화합니다.
- **효율적인 반복 추론 알고리즘 개발:** Temporal Fusion(시간적 융합) 단계와 Mask Refinement(마스크 정제) CNN Feed-Forward 단계를 번갈아 수행하여 점진적으로 분할 결과를 개선하는 알고리즘을 제안합니다. 이 알고리즘은 기존의 단일 샷(one-shot) 비디오 객체 분할 CNN으로 초기화될 때, 공개 벤치마크에서 최첨단 성능을 달성합니다.

## 📎 Related Works

- **비디오 객체 분할 (Video Object Segmentation):**
  - **그래프 기반 방법:** 픽셀 [45], 패치 [4], 슈퍼픽셀 [18, 46], 객체 제안 [37] 등 위에 그래프 구조를 구축하여 레이블을 전파합니다. 시간적 연결은 광학 흐름(optical flow) [18] 등을 사용합니다.
  - **CNN 기반 방법:** OSVOS [13], MaskTrack [35], OnAVOS [48] 등은 심층 학습을 통해 획기적인 성능 향상을 보였습니다. 그러나 DAVIS 2017 데이터셋 [38]과 같이 복잡한 시나리오에서는 성능이 크게 저하되는 한계가 있었습니다.
- **CNN + MRF/CRF 결합:**
  - **느슨하게 결합된 방법 (Loosely-coupled):** DeepLab [14]은 CNN 출력 후 CRFs [29]를 후처리 단계로 사용합니다. Jang and Kim [23]은 MRF 최적화를 통해 CNN 출력을 융합합니다.
  - **공동 훈련 (Joint Training):** Schwing and Urtasun [41], Arnab et al. [2]은 CNN과 MRF를 공동으로 훈련하려고 시도했습니다. 본 연구는 이들과 달리 CNN을 MRF 내 고차원 포텐셜 모델링에 활용하는 데 중점을 둡니다.
  - **MRF 추론을 신경망으로 근사화 (MRF Inference as RNN/NN):** CRF-RNN [53]은 CRF의 평균 필드(mean-field) 근사 추론을 RNN으로 공식화하여 CNN과 통합합니다. DPN [32]은 MRF의 평균 필드 추론을 한 번의 패스(pass)로 근사화합니다. 본 연구는 MRF 추론을 신경망으로 근사화하는 것과는 다른 방향으로, CNN을 MRF 내 고차원 포텐셜 모델링에 활용합니다.

## 🛠️ Methodology

본 연구는 픽셀 단위로 이진 레이블(0 또는 1)을 할당하는 단일 객체 분할을 가정하며, 다중 객체는 별도로 처리합니다.

1. **모델 구조 및 에너지 함수:**
   전체 에너지는 다음과 같이 정의됩니다:
   $$E(x) = \sum_{i \in V} E_u(x_i) + \sum_{(i,j) \in N_T} E_t(x_i,x_j) + \sum_{c \in S} E_s(x_c)$$

   - **단항 에너지 ($E_u$):** 각 픽셀 레이블의 음의 로그 우도($-\theta_u \log p(X_i=x_i)$)로 정의됩니다.
   - **시간적 에너지 ($E_t$):** 광학 흐름(optical flow)을 사용하여 이웃 프레임의 픽셀 간 연결($N_T$)을 설정합니다. `Forward-Backward consistency check`로 신뢰할 수 있는 움직임 벡터를 필터링합니다. 현재 프레임에서 $\pm 1, \pm 2$ 프레임까지의 연결을 고려하며, $E_t(x_i,x_j) = \theta_t w_{ij} (x_i-x_j)^2$ 와 같이 정의되어 시간적 일관성을 장려합니다.
   - **공간적 에너지 ($E_s$):** 한 프레임 내의 모든 픽셀을 하나의 클리크($c \in S$)로 정의하고, 마스크 정제 CNN인 $g_{CNN}(\cdot)$을 사용하여 품질을 평가합니다. $E_s(x_c) = \theta_s f(x_c)$ 이고, $f(x_c) = ||x_c - g_{CNN}(x_c)||_2^2$ 로 정의됩니다. 이는 주어진 마스크 $x_c$를 $g_{CNN}(\cdot)$을 통과시켰을 때의 결과가 자기 자신과 유사할수록 낮은 에너지를 할당합니다. CNN은 객체의 외형과 형태를 인코딩하여 고차원적인 공간적 의존성을 모델링합니다.

2. **추론 알고리즘:**
   MRF 모델의 MAP(Maximum A Posteriori) 추론은 CNN 기반 고차원 에너지 함수로 인해 매우 어렵습니다. 이를 해결하기 위해, 보조 변수 $y$를 도입하여 문제의 근사치를 최소화하는 반복 알고리즘을 제안합니다:
   $$\hat{E}(x,y) = \sum_{i \in V} E_u(x_i) + \sum_{(i,j) \in N_T} E_t(x_i,x_j) + \frac{\beta}{2} ||x-y||_2^2 + \sum_{c \in S} E_s(y_c)$$
   알고리즘은 두 단계를 번갈아 수행합니다:

   - **Temporal Fusion Step (TF):** $y$를 고정하고 $x$를 업데이트합니다 ($x^{(k)} \leftarrow \text{arg min}_x \hat{E}(x,y^{(k-1)})$). 이는 $E_u + E_t + \frac{\beta}{2}||x-y||_2^2$를 최소화하는 문제로, 공간적 의존성을 무시한 정규화된 에너지 함수입니다. ICM(Iterated Conditional Modes) [11]을 사용하여 근사 해를 찾습니다.
   - **Mask Refinement Step (MR):** $x$를 고정하고 $y$를 업데이트합니다 ($y^{(k)} \leftarrow \text{arg min}_y \hat{E}(x^{(k)},y)$). 이는 각 프레임 $c$에 대해 $\frac{\beta}{2}||x_c^{(k)}-y_c||_2^2 + E_s(y_c)$를 최소화하는 문제와 같습니다. 이 비볼록(non-convex) 문제는 $y_c^{(k)} \leftarrow g_{CNN}(x_c^{(k)})$ 로 근사화됩니다. 즉, CNN의 Feed-Forward 패스를 통해 마스크를 정제합니다.

3. **$g_{CNN}(\cdot)$ 구현:**

   - DeepLab 프레임워크를 사용하며, VGG-Net [44]을 백본으로 합니다.
   - 입력은 4채널(RGB 이미지 + 1채널 이진 마스크)입니다.
   - `Multi-level feature fusion`을 위해 중간 풀링 레이어에서 최종 출력 컨볼루션 레이어로 `skip connections`를 추가합니다.
   - **2단계 훈련:**
     1. **오프라인 훈련:** DAVIS 2017 훈련 세트를 사용하여 일반적인 객체 분할 모델을 훈련합니다.
     2. **온라인 미세 조정:** 주어진 테스트 비디오의 첫 번째 프레임 `ground-truth mask`를 사용하여 오프라인 모델을 미세 조정합니다. 훈련 입력 마스크는 `Lucid data dreaming` [26]과 같은 데이터 증강(data augmentation) 기술로 오염(contaminated)된 `ground-truth mask`를 사용합니다.

4. **세부 구현:**
   - **초기화 및 픽셀 우도:** OSVOS [13]를 사용하여 초기 레이블링 및 픽셀 우도를 얻습니다. 선형 움직임 모델과 이전 프레임에서 워핑된 응답 맵을 융합하여 초기 마스크를 생성합니다.
   - **다중 객체 처리:** 각 객체를 개별적으로 처리하고, 겹치는 영역은 `connected pixel blobs`로 분할하여 `Eq. (10)`을 최소화하는 레이블을 할당합니다.
   - **광학 흐름:** FlowNet2 [20]를 사용합니다.
   - **에너지 균형 매개변수:** $\theta_u = \theta_t = 1$, $\beta = 1.5$로 초기 설정되고 매 반복마다 $1.2$배 증가합니다.
   - **반복 횟수:** `Outer iterations` $K=3$, `inner iterations` $L=5$로 설정합니다.

## 📊 Results

- **Ablation Study (DAVIS 2017 Validation Set):**
  - 기존 OSVOS 기반 `baseline` 성능(0.596 J & F mean) 대비:
    - `Temporal Fusion (TF)`만 수행 시: 오히려 성능이 약간 저하됩니다(0.590). 시간적 정보만으로는 노이즈가 전파될 수 있음을 보여줍니다.
    - `Mask Refinement (MR)`만 수행 시: `baseline` 대비 약 5%의 성능 향상을 보입니다(0.649).
    - `TF`와 `MR`을 모두 활성화하여 반복 수행 시: `baseline` 대비 최대 11%의 성능 향상을 달성합니다(0.707). 이는 TF가 누락된 부분을 복구하고 MR이 이를 정제하는 상호 보완적 효과를 보여줍니다.
- **벤치마크 결과 (DAVIS 2017 test-dev set):**
  - 본 알고리즘은 0.675 J & F mean을 달성하여, DAVIS 2017 챌린지에서 상위권을 차지했던 `apata` (0.666) 및 `lixx` (0.661) 등 기존의 고도로 엔지니어링된 시스템(모델 앙상블, 멀티스케일 학습/테스트, 전용 객체 감지기 등 활용)을 능가하는 성능을 보였습니다. 본 방법은 이러한 추가 기술 없이도 더 우수한 성능을 달성합니다.
- **레거시 데이터셋 결과 (DAVIS 2016, Youtube-Objects, SegTrack v2):**
  - DAVIS 2016 (0.842 J), Youtube-Objects (0.784 J), SegTrack v2 (0.771 J) 등 세 가지 데이터셋 모두에서 최첨단(state-of-the-art) 성능을 달성했습니다. 이는 CNN의 표현 능력과 MRF의 시공간 연결 모델링 이점을 모두 활용했기 때문입니다.

## 🧠 Insights & Discussion

- **CNN과 MRF의 시너지:** 이 연구는 CNN의 강력한 객체 외형 및 형태 표현 능력과 MRF의 시공간적 관계 모델링 능력을 성공적으로 결합했음을 보여줍니다. 이는 특히 폐색, 재등장 객체, 복잡한 객체 변형 등 DAVIS 2017과 같은 도전적인 시나리오에서 강점을 발휘합니다.
- **고차원 포텐셜의 효과:** CNN을 통해 공간적 고차원 포텐셜을 모델링함으로써, 기존의 쌍별(pairwise) 또는 제한된 고차원 포텐셜로는 불가능했던 복잡한 픽셀 간 의존성을 포착하고 전체적인 객체 마스크 품질을 향상시킬 수 있음을 입증했습니다.
- **효율적인 추론 전략:** CNN이 포함된 고차원 MRF의 추론이 어렵다는 문제를, 보조 변수 도입과 반복적인 Temporal Fusion 및 CNN 기반 Mask Refinement 단계로 근사화하여 효과적으로 해결했습니다. 이는 MRF 모델 내부에 CNN의 feed-forward 패스를 직접 임베딩하는 새로운 추론 패러다임을 제시합니다.
- **일반화 능력:** 특정 객체 감지기(예: 사람 감지기)에 의존하지 않고 클래스 불가지 분할이 가능하다는 점은 모델의 일반화 능력을 높입니다.
- **한계 및 시사점:** 본 연구의 성공은 CNN의 feed-forward 패스를 MRF 추론 과정에 내장하는 새로운 방향을 제시하며, 향후 연구에 영감을 줄 수 있습니다.

## 📌 TL;DR

본 논문은 준지도 학습 비디오 객체 분할에서 시공간적 일관성 부족 문제를 해결하기 위해 **CNN을 통해 고차원 공간 포텐셜을 인코딩한 새로운 시공간 MRF 모델**을 제안합니다. 복잡한 MRF 추론 문제는 **Temporal Fusion과 CNN 기반 Mask Refinement를 번갈아 수행하는 효율적인 반복 알고리즘**으로 근사화됩니다. 이 방법은 DAVIS 2017 및 기타 벤치마크에서 **기존 최첨단 방법을 능가하는 성능**을 달성하며, CNN의 표현력과 MRF의 시공간 모델링 능력을 효과적으로 결합했음을 입증합니다.
