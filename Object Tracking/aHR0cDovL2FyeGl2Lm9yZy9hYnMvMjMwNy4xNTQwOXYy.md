# Uncertainty-aware Unsupervised Multi-Object Tracking

Kai Liu, Sheng Jin, Zhihang Fu, Ze Chen, Rongxin Jiang, Jieping Ye (2023)

## 🧩 Problem to Solve

본 논문은 수동으로 어노테이션된 ID 정보 없이 학습하는 비지도 다중 객체 추적(Unsupervised Multi-Object Tracking, MOT)에서 발생하는 **불확실성(Uncertainty)** 문제를 해결하고자 한다.

비지도 MOT의 핵심은 신뢰할 수 있는 특징 임베딩(Feature Embedding)을 학습하는 것이나, 정답 ID가 없는 상태에서 유사도 기반의 프레임 간 연관(Association) 단계는 오류에 취약하다. 이러한 오류가 프레임별로 누적되면 Pseudo-tracklets의 품질이 저하되며, 결과적으로 시간적 변화에 강건한 일관된 특징 임베딩을 학습하는 것을 방해한다.

최근의 자기지도 학습(Self-supervised learning) 기법들이 이를 해결하려 했으나, 주로 단일 프레임 기반의 증강(Augmentation)에 의존하여 프레임 간의 시간적 관계(Temporal relation)를 포착하지 못한다는 한계가 있었다. 따라서 본 논문의 목표는 불확실성 자체를 역으로 이용하여 Pseudo-tracklets의 정확도를 높이고, 이를 통해 시간적 정보가 반영된 효과적인 데이터 증강 전략을 구축함으로써 임베딩의 일관성을 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 중심 아이디어는 **불확실성을 정량화하여 이를 검증(Verification)과 증강(Augmentation)의 가이드로 활용**하는 것이다.

1.  **Uncertainty-aware Tracklet-Labeling (UTL):** 연관 단계에서 발생하는 불확실성을 측정하는 메트릭을 도입하여, 위험한 연관을 식별하고 이를 수정함으로써 매우 정확한 Pseudo-tracklets를 생성한다.
2.  **Tracklet-Guided Augmentation (TGA):** 생성된 신뢰할 수 있는 Pseudo-tracklets의 공간적 변위(Spatial displacement)를 활용해 실제 객체의 움직임을 시뮬레이션하는 증강 전략을 제안한다. 이때 계층적 불확실성 샘플링(Hierarchical uncertainty-based sampling)을 통해 학습에 도움이 되는 하드 샘플(Hard sample)을 효율적으로 마이닝한다.
3.  **U2MOT 프레임워크:** 위 두 가지 핵심 요소(UTL, TGA)를 결합하여, 지도 학습 기반 추적기나 기존 비지도 추적기보다 뛰어난 성능을 보이는 통합 프레임워크를 제안한다.

## 📎 Related Works

### 기존 비지도 MOT 접근 방식 및 한계
- **Pseudo-label 기반 방법:** 운동 정보 기반(SimUMOT), 클러스터링 기반, 유사도 기반(Cycas, OUTrack) 방법들이 존재한다. 그러나 운동 기반 방법은 불규칙한 카메라 움직임에 취약하며, 클러스터링 기반은 시간적 연관성을 무시하고, 유사도 기반은 시간이 지남에 따라 노이즈가 누적되는 문제가 있다.
- **불확실성 추정(Uncertainty Estimation):** 주로 분류 모델의 캘리브레이션이나 Out-of-distribution 탐지에서 연구되었으나, MOT의 특성인 객체 가림(Occlusion)과 유사한 외관으로 인한 오매칭 문제를 해결하기 위한 MOT 전용 메트릭 연구는 부족했다.
- **데이터 증강(Augmentation):** 무작위 관점 변환(Random perspective transformation)이나 GAN 기반 증강이 사용되었으나, MOT의 핵심인 프레임 간 시간적 연속성과 실제 객체의 움직임을 반영하지 못했다.

## 🛠️ Methodology

### 전체 파이프라인 및 학습 목표
U2MOT는 대비 학습(Contrastive Learning) 기법을 사용하여 학습된다. 동일한 트랙렛 내의 객체들은 서로 가깝게(Pull), 서로 다른 트랙렛의 객체들은 멀게(Push) 배치하는 것을 목표로 하며, $\text{InfoNCE}$ 손실 함수를 사용한다.

$$L_{cl}(q; k^+; k^-) = -\log \frac{\exp(q \cdot k^+ / \epsilon)}{\sum_{i} \exp(q \cdot k_i / \epsilon)}$$

여기서 $q$는 쿼리, $k^+$는 긍정 샘플(동일 ID), $k^-$는 부정 샘플(타 ID)이며, $\epsilon = 0.07$은 온도 파라미터이다.

### Uncertainty-aware Tracklet-Labeling (UTL)
정확한 Pseudo-tracklets를 생성하기 위해 다음의 4단계 절차를 거친다.

1.  **Association:** $\ell_2$-정규화된 임베딩 $f$를 사용하여 코사인 유사도 행렬 $C$를 생성하고, 헝가리안 알고리즘을 통해 초기 연관을 수행한다.
    $$c_{i,j} = f^t_i \cdot f^{t-1}_j$$
2.  **Verification:** 연관 결과의 위험도(Risk)를 측정하여 불확실성을 정량화한다.
    - 위험도 $\sigma_{i,j}$는 할당된 유사도 $c_{i,j}$가 낮거나, 두 번째로 높은 유사도 $c_{i,j}^2$와의 차이가 적을 때 높아진다.
    $$\sigma_{i,j} = -\log c_{i,j} - \log(1 - c_{i,j}^2)$$
    - 적응형 임계값 $\gamma_{i,j}$를 설정하여 연관 불확실성 $\delta_{i,j}$를 계산한다.
    $$\gamma_{i,j} = -\log m_1 - \log(1 + m_2 - c_{i,j}), \quad \delta_{i,j} = \sigma_{i,j} - \gamma_{i,j}$$
    ($m_1=0.5, m_2=0.05$ 사용)
3.  **Rectification:** 불확실하다고 판단된 샘플들에 대해 재연관을 수행한다. 이때 단순 프레임 간 유사도가 아닌, 이전 $K$개 프레임의 평균 유사도와 IoU 제약 조건을 결합하여 더 견고한 판단을 내린다.
    $$c'_{i,j} = \left( \frac{1}{K} \sum_{\hat{t}=t-K}^{t-1} f^t_i \cdot f^{\hat{t}}_j \right) \times \mathbb{I}(\text{IoU}(b^t_i, b^{t-1}_j) > \beta)$$
4.  **Propagation:** 검증 및 수정이 완료된 ID를 프레임별로 전파하여 일관된 Pseudo-tracklets를 유지한다.

### Tracklet-Guided Augmentation (TGA)
신뢰할 수 있는 Pseudo-tracklets를 바탕으로 실제 객체의 움직임을 모사하는 증강을 수행한다.

- **공간 변환:** 현재 프레임 $I_t$의 객체 $o^t_a$와 과거 프레임 $I_{t-\tau}$의 객체 $o^{t-\tau}_a$ 사이의 아핀 변환(Affine Transformation) 행렬 $M^{t-\tau}_t$를 DLT 알고리즘으로 구하여 증강 이미지 $\tilde{I}_t$를 생성한다.
- **계층적 불확실성 샘플링:**
    1.  **소스 앵커(Source Anchor) 선택:** 트랙렛 수준의 불확실성 $\Omega_i$가 낮은(신뢰도가 높은) 트랙렛을 선택하여 잘못된 Pseudo-label로 인한 과도한 변형을 방지한다.
        $$\Omega_i = \frac{1}{n_t} \sum_{s=t_0}^t \exp(\delta^s_i), \quad p(a=i|t) = \frac{\exp(-\Omega_i)}{\sum \exp(-\Omega_{\hat{i}})}$$
    2.  **타겟 앵커(Target Anchor) 선택:** 해당 트랙렛 내에서 연관 불확실성 $\delta$가 높은 시점을 타겟으로 선택하여, 학습이 어려운 하드 네거티브(Hard negative) 샘플을 집중적으로 생성한다.
        $$p(\pi=t-\tau|a) = \frac{\exp(\delta^{t-\tau}_a)}{\sum \exp(\delta^{t-\hat{\tau}}_a)}$$

## 📊 Results

### 실험 설정
- **데이터셋:** MOT17, MOT20, VisDrone-MOT.
- **지표:** HOTA, MOTA, IDF1, IDS.
- **구현:** YOLOX 기반에 ReID head를 통합하여 학습.

### 주요 결과
1.  **MOT-Challenge:** MOT17에서 HOTA 64.2%, MOT20에서 HOTA 62.7%를 달성하여 기존의 비지도 추적기(UEANet 등)는 물론, 일부 지도 학습 기반 추적기보다 우수한 성능을 보였다.
2.  **VisDrone-MOT:** 불규칙한 카메라 움직임이 심한 환경에서도 IDF1과 FPS 면에서 ByteTrack 등의 기존 방법보다 우수한 성능을 입증했다.
3.  **Ablation Study:**
    - **UTL의 효과:** 단순한 장기 의존성(LTD) 추가보다 UTL을 적용했을 때 HOTA와 IDF1이 유의미하게 상승했다.
    - **TGA의 효과:** 무작위 증강보다 계층적 불확실성 샘플링을 적용한 TGA가 성능 향상에 더 기여했다.
    - **Inference Boosting:** U2MOT의 UTL 전략을 FairMOT, DeepSORT 등 다른 추적기에 적용했을 때도 HOTA와 IDF1이 일관되게 상승하는 범용성을 확인했다.

## 🧠 Insights & Discussion

### 강점
본 논문은 비지도 학습의 고질적 문제인 '노이즈 섞인 Pseudo-label'을 단순히 피하려 하지 않고, **불확실성이라는 지표로 정량화하여 이를 정제(Refinement)와 학습 가이드(Hard sample mining)로 활용**했다는 점에서 매우 영리한 접근 방식을 취하고 있다. 특히, 학습 단계뿐만 아니라 추론 단계에서도 다른 기존 모델들의 성능을 높일 수 있는 UTL의 범용성이 돋보인다.

### 한계 및 비판적 해석
1.  **오프라인 처리의 한계:** 저자가 명시했듯이, 불확실성 평가가 오프라인으로 수행되어 네트워크 학습 과정과 실시간으로 연동되지 않는다. 이는 모델이 학습 중에 실시간으로 불확실성을 피드백 받아 최적화될 수 없음을 의미한다.
2.  **학습 시간 증가:** 오프라인 불확실성 평가 단계로 인해 학습 시간이 약 2배 증가한다. 대규모 데이터셋을 다룰 때 이는 상당한 비용적 부담이 될 수 있다.
3.  **IDS의 증가 (MOT20):** MOT20과 같이 매우 혼잡한 장면에서는 IDS가 다소 증가하는 경향이 있는데, 이는 임베딩 기반의 비지도 방식이 가림 현상이 심한 환경에서는 여전히 한계가 있음을 시사한다.

## 📌 TL;DR

U2MOT는 비지도 MOT에서 발생하는 연관 불확실성을 정량화하여, **정확한 Pseudo-tracklets 생성(UTL)**과 **시간적 정보가 반영된 하드 샘플 증강(TGA)**에 활용하는 프레임워크이다. 이를 통해 지도 학습 기반 추적기에 근접하거나 능가하는 SOTA 성능을 달성했으며, 제안된 불확실성 메트릭은 타 추적 모델의 추론 성능을 높이는 데도 적용 가능하다. 향후 실시간 불확실성 추정 및 학습 시간 단축이 주요 개선 방향이 될 것이다.