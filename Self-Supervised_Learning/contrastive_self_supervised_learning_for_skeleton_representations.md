# Contrastive Self-Supervised Learning for Skeleton Representations

Nico Lingg, Miguel Sarabia, Luca Zappella, Barry-John Theobald (2022)

## 🧩 Problem to Solve

본 논문은 인간의 행동을 자동으로 분류하고 예측하기 위해 널리 사용되는 Skeleton Point Clouds의 효율적인 표현 학습(Representation Learning) 문제를 다룬다. Skeleton 데이터는 깊이 카메라(Depth Camera)나 모션 캡처(MoCap) 시스템을 통해 쉽게 획득할 수 있다는 장점이 있으나, 실제 환경에서는 가려짐(Occlusions), 노이즈(Noisy data), 관절 누락(Missing joints), 그리고 부정확한 스켈레톤 피팅(Poor skeleton fittings)과 같은 고유한 문제점들이 존재한다.

이러한 문제들을 해결하기 위해, 저자들은 레이블이 없는 대규모 데이터셋을 활용하여 노이즈와 누락된 데이터에 강건한 표현을 학습하는 Self-Supervised Learning(SSL) 기법을 적용하고자 한다. 구체적인 목표는 Skeleton Reconstruction, Motion Prediction, Activity Classification이라는 세 가지 서로 다른 다운스트림 태스크(Downstream tasks)에 범용적으로 적용될 수 있는 스켈레톤 표현을 학습하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 기여는 SimCLR라는 Contrastive Learning 프레임워크를 스켈레톤 데이터에 맞게 최적화하고, 학습 결과에 영향을 미치는 알고리즘적 결정 요소들을 체계적으로 분석한 것에 있다. 주요 설계 아이디어는 다음과 같다.

1. **SimCLR의 스켈레톤 데이터 적용**: 입력 데이터로부터 두 개의 서로 다른 뷰(View)를 생성하고, 동일 샘플에서 유래한 뷰 간의 거리는 좁히고 서로 다른 샘플 간의 거리는 멀게 하는 Contrastive Learning 방식을 채택하였다.
2. **체계적인 Ablation Study**: 데이터 증강(Augmentation), 데이터셋 구성, 백본 아키텍처(Backbone Architecture)가 학습된 표현의 질에 미치는 영향을 정량적으로 평가하였다.
3. **대규모 데이터 통합**: 총 6개의 서로 다른 데이터셋을 통합하여 약 4,000만 프레임의 방대한 데이터를 구축함으로써 표현의 일반화 성능을 높였다.

## 📎 Related Works

본 논문은 Multi-view Self-Supervised Learning과 Skeleton Representation Learning이라는 두 가지 분야의 접점에 위치한다.

- **Multi-view SSL**: CPC, MoCO v3, DINO, SimCLR 등이 대표적이다. 특히 SimCLR는 학습이 안정적이고 Negative sample을 마이닝하기 위한 추가적인 복잡한 로직이 필요 없다는 장점이 있어 본 연구에서 채택되었다.
- **Skeleton Representation Learning**: 기존 연구들은 Motion consistency나 continuity와 같은 특정 Pretext task를 설계하거나, BERT에서 영감을 받은 Transformer 구조를 사용하여 SOTA 성능을 달성하고자 하였다. 또한 CP-STN과 같이 Contrastive learning과 Pretext task를 결합한 형태의 연구도 존재한다.

본 논문은 특정 도메인에 특화된 복잡한 Pretext task를 설계하는 대신, 범용적인 SimCLR 프레임워크를 사용하되 스켈레톤 데이터의 특성에 맞는 증강 기법과 아키텍처를 탐색함으로써 차별점을 둔다.

## 🛠️ Methodology

### 전체 파이프라인

시스템은 크게 전처리, Pre-training, 그리고 Downstream evaluation의 세 단계로 구성된다.

1. **전처리 (Preprocessing)**:
    - 모든 데이터셋의 관절 표현을 3D 전역 좌표계로 변환한다.
    - 관절 공간을 15개의 기본 관절로 통일한다.
    - 신장(Height)을 2m로 스케일링하며, $z$축 범위를 $[-1.0, 1.0]$으로 설정한다.
    - 몸통(Torso)이 좌표계의 중심에 오도록 이동시키고, 스켈레톤이 카메라를 정면으로 바라보도록 회전시킨다.

2. **Pre-training**:
    - **입력**: 50프레임으로 구성된 스켈레톤 시퀀스를 입력으로 사용한다.
    - **데이터 증강 (Augmentation)**: 8가지의 공간적(Spatial) 및 시간적(Temporal) 증강을 통해 하나의 시퀀스에서 두 개의 뷰를 생성한다.
        - **Spatial**: Axis mirroring, Random scaling, Joint jitter, Frame dropout, Joint dropout.
        - **Temporal**: Speed-up, Slow-down.
    - **인코더 (Encoder)**: ST-GCN (Spatial Temporal Graph Convolutional Network)을 사용하여 128차원의 벡터를 추출한다.
    - **프로젝션 헤드 (Projection Head)**: 3층의 MLP를 통해 최종 표현을 투영한다.
    - **손실 함수**: NT-Xent (Normalized Temperature-scaled Cross Entropy) 손실 함수를 사용하여 동일 샘플의 뷰는 서로 가깝게, 다른 샘플의 뷰는 멀게 학습한다.
    $$ \mathcal{L} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k)/\tau)} $$
    (여기서 $z$는 프로젝션된 벡터, $\tau$는 온도 파라미터, $\text{sim}$은 코사인 유사도를 의미한다.)

3. **추론 및 다운스트림 적용**: Pre-training이 완료되면 MLP 헤드는 제거하고, 학습된 ST-GCN 인코더만을 고정(Frozen)하거나 미세 조정(Fine-tuning)하여 하위 태스크에 사용한다.

## 📊 Results

### 실험 설정

- **데이터셋**: NTU RGB+D 60/120, TrinitySpeech-Gesture, Talking With Hands, DanceDB, HDM05 등 총 6개 데이터셋 사용.
- **다운스트림 태스크**:
  - **Skeleton Reconstruction**: 80% 프레임 제거 후 복원. 지표: MAE (Mean Absolute Error, mm).
  - **Motion Prediction**: 이전 50프레임을 통해 다음 50프레임 예측. 지표: MAE (mm).
  - **Activity Classification**: NTU-60 레이블을 이용한 행동 분류. 지표: Accuracy (%).

### 주요 결과

- **백본의 영향**: ST-GCN을 MLP로 대체했을 때, 행동 분류 정확도가 16.5% 포인트 하락하여 그래프 기반 인코더의 중요성이 입증되었다.
- **데이터 다양성**: NTU-60 데이터셋만으로 사전 학습했을 때보다 6개 데이터셋을 모두 사용했을 때 행동 분류 정확도가 6.6% 포인트 향상되었다.
- **증강 기법의 영향**:
  - 공간적 증강만 사용하거나 시간적 증강만 사용했을 때보다, 두 가지를 모두 결합했을 때 성능이 가장 좋았다.
  - 특히 시간적 증강만 사용한 경우, 행동 분류 정확도가 22.8% 포인트 급락하였다.
  - 개별 증강 중에서는 Random scaling과 Joint dropout이 성능 유지에 매우 중요한 역할을 하였다.
- **Fine-tuning의 효과**: 행동 분류에서는 성능이 향상되었으나, 복원 및 예측 태스크에서는 오히려 성능이 저하되었다. 이는 하위 태스크의 학습 데이터셋 규모가 모델 전체를 미세 조정하기에는 부족하기 때문으로 분석된다.

## 🧠 Insights & Discussion

본 논문은 스켈레톤 데이터의 표현 학습을 위해 Contrastive Learning을 적용할 때 고려해야 할 핵심 요소들을 명확히 제시하였다.

첫째, **데이터 다양성의 힘**이다. 서로 다른 센서와 환경에서 수집된 이질적인 데이터셋들을 통합하여 학습시키는 것이 모델의 일반화 성능을 높이는 데 결정적인 역할을 한다.

둘째, **Inductive Bias로서의 증강 기법**이다. 저자들은 증강 기법의 선택이 모델이 어떤 불변성(Invariance)을 학습할지를 결정한다고 보았다. 실험 결과, 노이즈에 대한 강건함(Spatial)과 동작 속도에 대한 불변성(Temporal)을 동시에 학습시키는 것이 필수적임을 확인하였다.

셋째, **아키텍처의 적합성**이다. 단순한 MLP보다 스켈레톤의 기하학적 구조를 반영할 수 있는 ST-GCN이 훨씬 우수한 표현을 추출함을 보였다.

한계점으로는, 행동 분류 태스크에서 기존 SOTA 모델들과 직접적인 성능 비교가 어렵다는 점이 언급되었다. 이는 본 연구가 사용한 관절의 수(15개)가 기존 연구들보다 적고, 템포럴 수용 영역(Temporal receptive field)이 다르기 때문이다.

## 📌 TL;DR

본 연구는 SimCLR 프레임워크를 활용하여 인간 스켈레톤의 범용적 표현을 학습하는 방법을 제안하고, 이에 영향을 주는 요소들을 체계적으로 분석하였다. 결론적으로 **ST-GCN 인코더**를 사용하고, **대규모의 다양한 데이터셋**을 통합하며, **공간적·시간적 증강을 동시에 적용**하는 것이 최적의 표현을 학습하는 방법임을 입증하였다. 이 연구는 향후 스켈레톤 기반의 행동 인식 및 예측 모델의 사전 학습(Pre-training) 전략을 수립하는 데 중요한 가이드라인을 제공한다.
