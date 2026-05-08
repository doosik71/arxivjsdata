# Tracking without Label: Unsupervised Multiple Object Tracking via Contrastive Similarity Learning

Sha Meng, Dian Shao, Jiacheng Guo, Shan Gao

## 🧩 Problem to Solve

다중 객체 추적(MOT)은 객체 간 간섭, 가려짐(occlusion) 등 고유의 문제에 직면하며, 특히 레이블이 없는 비지도(unsupervised) 환경에서는 더욱 어렵습니다. 기존의 클러스터 기반 비지도 MOT 방법들은 훈련 과정에서 오류가 누적되기 쉽습니다. 본 논문은 이러한 레이블 부재 및 MOT 고유의 문제로 인해 식별 가능한(discriminative) 객체 표현을 학습하기 어려운 점을 해결하고자 합니다.

## ✨ Key Contributions

- **비지도 MOT를 위한 대비 유사성 학습(Contrastive Similarity Learning) 방법인 UCSL 제안**: ID 정보 없이 ReID 모듈의 샘플 특징만을 기반으로 잠재적인 객체 일관성을 학습하는 방법을 제시합니다.
- **세 가지 핵심 대비 모듈 설계**: 다양한 상황에서 객체 간의 연관성을 모델링하기 위해 다음 세 가지 모듈을 제안합니다:
  - **자기 대비(Self-Contrast) 모듈**: 프레임 내 객체 및 인접 프레임 간 객체를 일치시켜 식별력 있는 표현을 얻고 자기 유사성을 최대화합니다.
  - **교차 대비(Cross-Contrast) 모듈**: 교차 프레임(cross-frame) 매칭 결과와 연속 프레임(continuous-frame) 매칭 결과를 정렬하여 객체 가려짐으로 인한 지속적인 부정적 영향을 완화합니다.
  - **모호성 대비(Ambiguity Contrast) 모듈**: 가려지거나, 사라지거나, 새로 나타나는 객체 등 모호한 객체들을 다시 매칭하여 후속 객체 연관성의 확실성을 높입니다.
- **우수한 성능 입증**: MOT15, MOT17, MOT20 데이터셋에서 제안된 UCSL 방법의 효과를 입증했습니다. UCSL은 최신 비지도 MOT 방법들을 능가하며, 심지어 많은 완전 지도(fully supervised) MOT 방법들과 유사하거나 더 나은 성능을 달성합니다.

## 📎 Related Works

- **다중 객체 추적 (Multi-Object Tracking, MOT)**:
  - 초기에는 추적-탐지(tracking-by-detection) 패러다임(e.g., SORT [3])이 주를 이뤘으나, 탐지기의 성능에 크게 의존했습니다.
  - 최근에는 공동 탐지 및 ReID 임베딩(joint detection and ReID embeddings) 패러다임(e.g., JDE [39], FairMOT [45])이 강세를 보였으며, 본 연구는 FairMOT을 기반으로 합니다. Transformer 기반 아키텍처(e.g., TrackFormer [23], TransTrack [31])도 있습니다.
- **비지도 추적 (Unsupervised Tracking)**:
  - SimpleReID [12]는 비디오와 탐지 결과를 사용하여 추적 결과를 시뮬레이션하고 이를 유사 레이블(pseudo-labels)로 활용하여 ReID 네트워크를 훈련했습니다.
  - OUTrack [21]은 비지도 ReID 학습 모듈과 지도 가려짐 추정(occlusion estimation) 모듈을 결합하여 추적 성능을 향상시켰습니다.
- **재식별 (Re-Identification, ReID)**:
  - 비지도 ReID는 도메인 적응(domain adaptation), 클러스터링(e.g., MMCL [34], Caron et al. [5], Lin et al. [20])을 통해 널리 사용되었으나, 클러스터링 기반 방법은 오류 누적 문제가 있습니다.
  - CycAs [38]와 같은 자기 지도 학습(self-supervised learning) 방법은 다중 객체 추적의 데이터 연관성 개념에서 영감을 받아 자체 지도 신호를 제약 조건으로 사용하여 특징 표현 능력을 강화했습니다.
- **순환 일관성 (Cycle Consistency)**: GAN에서 유래하여 분할(segmentation), 추적 등에 널리 사용됩니다. Jabri et al. [10]은 비디오에서 시공간 그래프를 구성하고, Wang et al. [37]은 시간의 순환 일관성을 시각적 표현 학습을 위한 지도 신호로 사용했습니다.
- **대비 학습 (Contrastive Learning)**: 자기 지도 학습에서 큰 잠재력을 보였습니다. QDTrack [26]은 수백 개의 영역 제안(region proposals)을 밀집하게 샘플링하여 대비 학습에 사용했으며, Yu et al. [42]는 다중 시점 궤적 대비 학습을 제안했습니다.

## 🛠️ Methodology

UCSL은 FairMOT [45]를 기반으로 하며, 세 가지 핵심 대비 모듈로 구성됩니다.

1. **입력 및 특징 추출**:

   - 세 개의 연속된 이미지 $I_t, I_{t-1}, I_{t-2}$를 백본 네트워크에 입력합니다.
   - 탐지 브랜치와 ReID 헤드를 통해 탐지 결과와 ReID 특징 맵을 얻습니다.
   - 각 객체에 해당하는 특징 임베딩을 추출하여 임베딩 행렬 $X_1, X_2, X_3$를 구성합니다.

2. **대비 유사성 학습 모듈**:

   - **자기 대비 모듈 (Self-Contrast Module)**:

     - **직접 자기 대비(Direct Self-Contrast)**: 동일 프레임 내 객체들은 서로 다른 클래스에 속하므로, $I_1$ 내 객체들의 자기 유사성 행렬 $S_{dsc} = \psi_{row}(X_1^T X_1)$를 계산하고, 대각 성분($\text{diag}(S_{dsc})$)이 1에 가까워지도록 최대화합니다.
     - **간접 자기 대비(Indirect Self-Contrast)**: $I_1$과 $I_2$ 간의 유사성 행렬 $S_{1 \to 2}$와 $S_{2 \to 1}$을 계산합니다. 순환 연관성 일관성(cycle association consistency)에 따라, $S_{isc} = S_{1 \to 2} S_{2 \to 1}$의 대각 성분이 1에 가까워지도록 최대화합니다.
     - **손실 함수**: $L_{sc} = -\frac{1}{N} \left[ \sum \log (\text{diag}(S_{dsc})) + \sum \log (\text{diag}(S_{isc})) \right]$

   - **교차 대비 모듈 (Cross-Contrast Module)**:

     - 목표: 객체 가려짐으로 인한 부정적 영향을 완화합니다. 직접 교차 프레임 매칭 결과와 연속 프레임을 통한 간접 매칭 결과가 일치해야 합니다.
     - $I_1, I_2, I_3$ 세 프레임을 사용하며, 직접 매칭 결과($S_{1 \to 3}, S_{3 \to 1}$)와 간접 매칭 결과($S_{1 \to 3}^* = \psi_{row}(S_{1 \to 2} S_{2 \to 3})$, $S_{3 \to 1}^* = \psi_{row}(S_{3 \to 2} S_{2 \to 1})$)를 계산합니다.
     - **손실 함수**: 두 매칭 분포의 차이를 측정하기 위해 JS divergence를 사용합니다: $L_{cc} = \frac{1}{N} JSD(S_{1 \to 3}^* \| S_{1 \to 3}) + \frac{1}{K} JSD(S_{3 \to 1}^* \| S_{3 \to 1})$.

   - **모호성 대비 모듈 (Ambiguity Contrast Module)**:
     - 목표: 가려지거나 사라지거나 새로 나타나는 등 모호한 객체(유사성 임계값 $\theta=0.7$ 미만)를 처리합니다.
     - $I_1$과 $I_2$에서 모호한 객체들을 식별하고, 이들 모호한 객체들 간의 유사성 행렬 $S_{r_{1 \to 2}}$와 $S_{r_{2 \to 1}}$을 다시 계산합니다.
     - **손실 함수**: 최소 엔트로피(minimum entropy)를 사용하여 계산합니다: $L_{ac} = -\frac{1}{|N_r - M_r| + 1} \left[ \frac{1}{N_r} S_{r_{1 \to 2}} \log (S_{r_{1 \to 2}}) + \frac{1}{M_r} S_{r_{2 \to 1}} \log (S_{r_{2 \to 1}}) \right]$. 두 프레임의 모호한 객체 수가 같을 경우 가려짐만 있다고 가정하고 엔트로피를 최소화하며, 다를 경우 사라지거나 나타난 객체가 있음을 고려하여 손실을 동적으로 약화합니다.

3. **총 손실 함수**:

   - $L(I_t, I_{t-1}, I_{t-2}) = L_{sc} + L_{cc} + L_{ac}$

4. **추론 단계 (Inference)**:
   - FairMOT [45]와 유사하게 칼만 필터(Kalman Filter)를 사용하여 객체 위치를 예측하고, 임베딩 거리(embedding distance) 및 IoU 거리(IoU distance)를 사용하여 2단계 매칭(two-stage matching)을 수행합니다.
   - 매칭되지 않은 탐지 결과는 새 객체로 초기화하고, 매칭되지 않은 궤적은 30 프레임 동안 저장되었다가 다시 나타나면 매칭됩니다.

## 📊 Results

- **MOT17, MOT15, MOT20 벤치마크 평가**:
  - **MOT17**: 비지도 방식인 UCSL은 73.0 MOTA, 70.4 IDF1, 58.4 HOTA를 달성하여, 기존의 비지도 SimpleReID [12]를 크게 능가하고, UTrack [21]보다 MOTA에서 1.2%p 향상된 성능을 보입니다. 또한, 많은 지도 학습 기반 방법들과 비교할 만한 성능을 달성합니다.
  - **MOT15**: UCSL은 59.1 MOTA를 기록하여, 지도 학습 기반 방법들(EAMTT [28], TubeTK [25], FairMOT [45])보다 MOTA에서 우수하며 다른 지표에서도 유사한 성능을 보입니다.
  - **MOT20**: UCSL은 62.4 MOTA, 63.0 IDF1, 52.3 HOTA를 달성하여, 비지도 SimpleReID [12]를 크게 앞서며 지도 학습 기반 방법들과도 경쟁력 있는 성능을 보여줍니다.
- **JDE(Joint Detection and Embeddings) 패러다임 내 비교**: JDE [39]와 같은 다른 JDE 기반 방법에도 적용 가능하며, 지도 학습 방법들과 유사한 결과를 보여줍니다.
- **Ablation Studies**:
  - **자기 대비 ($L_{sc}$)**: IDF1, HOTA, IDS(Identity Switches) 지표를 크게 개선하여, 더 식별력 있는 ReID 임베딩을 추출함을 입증합니다.
  - **교차 대비 ($L_{cc}$)**: 자기 대비에 더해 IDF1과 HOTA를 각각 68.4, 55.6으로 향상시키며, 가려짐으로 인한 영향을 효과적으로 완화합니다.
  - **모호성 대비 ($L_{ac}$)**: 위 두 손실에 모호성 대비를 추가했을 때 MOTA, MT(Mostly Tracked objects), IDS에서 더욱 뚜렷한 개선을 보여 객체 궤적 유지에 긍정적인 효과가 있음을 나타냅니다.
  - **입력 프레임 간격**: 간격이 좁을수록(예: 1) 성능이 가장 좋게 나타났습니다. 레이블이 없는 환경에서는 긴 간격이 객체의 급격한 변화를 유발하여 매칭을 어렵게 하고 오류를 누적시킬 수 있습니다.
  - **ReID 임베딩 차원**: 128차원이 64차원보다 MOTA와 MT 지표에서 더 나은 성능을 보였으며, 256차원은 128차원과 유사한 개선 효과를 보이지만 더 많은 공간을 소비하고 속도를 저하시키므로 128차원이 최적임이 확인되었습니다.

## 🧠 Insights & Discussion

- UCSL은 레이블 없이도 객체 특징의 잠재적 일관성을 학습함으로써 비지도 MOT의 핵심 문제를 성공적으로 해결합니다.
- 세 가지 대비 모듈(자기 대비, 교차 대비, 모호성 대비)의 조합은 네트워크가 가려지거나, 사라지거나, 새로 나타나는 객체와 같은 어려운 경우를 포함하여 일관되고 신뢰할 수 있는 식별력 있는 특징을 학습할 수 있도록 합니다.
- UCSL은 기존 비지도 방법들을 뛰어넘을 뿐만 아니라, 비용이 많이 드는 주석(annotation)을 사용하는 많은 지도 학습 방법들보다 높은 정확도를 제공하여 비지도 학습의 잠재력을 강력하게 보여줍니다.
- 프레임 간격에 대한 분석은 비지도 학습 환경에서 장기적인 시간적 관계를 명시적으로 모델링하는 것보다 적절한 단기적 관계 학습이 중요할 수 있음을 시사합니다.
- IDS 지표에서 다른 방법보다 개선되지 않은 점은 UCSL이 더 많은 궤적을 추적하고 더 높은 재현율(recall)을 가지기 때문일 수 있으며, 이는 모델이 더 많은 객체를 추적하려 시도할 때 발생할 수 있는 trade-off를 보여줍니다.

## 📌 TL;DR

본 논문은 레이블이 없는 환경에서 다중 객체 추적(MOT)의 어려운 문제를 해결하기 위해 **비지도 대비 유사성 학습(UCSL)** 방법을 제안합니다. UCSL은 **자기 대비(self-contrast)**, **교차 대비(cross-contrast)**, **모호성 대비(ambiguity contrast)**의 세 가지 모듈을 통해 객체 특징의 잠재적 일관성을 학습하여 식별력 있는 ReID 임베딩을 생성합니다. 실험 결과, UCSL은 기존 비지도 MOT 방법들을 크게 능가하며, 심지어 다수의 지도 학습 기반 방법들과도 필적하는 성능을 달성하여 레이블 없는 환경에서의 MOT 가능성을 입증합니다.
