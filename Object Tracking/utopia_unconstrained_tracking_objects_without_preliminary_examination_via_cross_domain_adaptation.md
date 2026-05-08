# UTOPIA: Unconstrained Tracking Objects without Preliminary Examination via Cross-Domain Adaptation

Pha Nguyen, Kha Gia Quach, John Gauch, Samee U. Khan, Bhiksha Raj, Khoa Luu (2023)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 Multiple Object Tracking (MOT) 모델의 **도메인 일반화(Generalization)** 능력 부족이다. 일반적으로 fully-supervised MOT 방법론들은 기존 데이터셋에서 높은 정확도를 보이지만, 학습 시 보지 못한 새로운 데이터셋이나 새로운 도메인(Unseen Domain)에 적용했을 때 성능이 급격히 저하되는 현상이 발생한다.

이러한 성능 저하의 주된 원인은 소스 도메인(Source Domain)과 타겟 도메인(Target Domain) 사이의 **도메인 갭(Domain Gap)** 때문이다. 예를 들어, 카메라의 시점(Perspective), 합성 데이터와 실제 데이터의 차이, 가림(Occlusion) 정도 및 객체의 외형(Appearance) 차이 등이 이에 해당한다.

또한, MOT를 위한 학습 데이터를 수동으로 어노테이션(Annotation)하는 작업은 막대한 시간과 비용이 소모되는 매우 힘든 작업이다. 기존의 self-supervised MOT 연구들은 대부분 강력한 객체 검출기(Robust Detector)가 이미 존재한다는 가정하에 특징 추출기(Feature Extractor) 학습에만 집중하였으며, 검출 단계부터 추적 단계까지를 모두 포함하는 end-to-end 방식의 cross-domain adaptation에 대한 연구는 부족한 실정이다. 따라서 본 논문의 목표는 인간의 사전 지식(Pre-defined human knowledge) 없이도 타겟 도메인의 피드백을 통해 스스로 학습하고 업데이트할 수 있는 self-supervised cross-domain MOT 프레임워크를 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 소스 도메인의 레이블 정보와 타겟 도메인의 무레이블 데이터를 동시에 활용하여 도메인 간의 불일치를 최소화하는 것이다. 이를 위해 다음과 같은 핵심 설계 아이디어를 제안한다.

1. **Object Consistency Agreement (OCA)**: 서로 다른 데이터 증강(Data Augmentation)이 적용된 두 뷰(View) 사이의 일관성을 측정하여, 타겟 도메인에서 레이블이 없는 샘플에 대해 객체의 존재 여부를 전파하는 메커니즘이다.
2. **Optimal Proposal Assignment (OPA)**: Sinkhorn-Knopp 알고리즘 기반의 Optimal Transport를 도입하여, 객체 간의 유사도 학습을 위한 최적의 매칭(Assignment) 전략을 구축한다. 이는 One-to-One 및 One-to-Many 매칭을 모두 지원하여 학습의 효율성과 정확도를 높인다.
3. **End-to-End Self-supervised Framework**: 객체 검출(Detection)부터 유사도 학습(Similarity Learning)까지 전체 파이프라인을 통합하여 학습함으로써, 타겟 도메인에 적응적인 검출기와 추적기를 동시에 최적화한다.

## 📎 Related Works

### 기존 연구 및 한계

- **Fully-supervised MOT**: Siamese network를 이용한 ID 할당이나 Optical flow, Kalman filter 등을 이용한 모션 모델링 연구가 주를 이룬다. 그러나 이러한 방법들은 방대한 양의 레이블 데이터가 필수적이며, 새로운 도메인에 대한 적응력이 떨어진다.
- **Unsupervised MOT**: 주로 휴리스틱(Heuristics)이나 Kalman filter와 같은 비매개변수적 모션 모델에 의존한다. 외형 정보 없이 모션 모델만으로 추적을 수행하는 경우가 많아 복잡한 환경에서 한계가 있다.
- **Self-supervised MOT**: 최근 cross-input consistency를 이용해 특징 추출기를 학습하는 연구들이 등장했다. 하지만 이들은 대개 '강력한 검출기가 있다'는 가정을 전제로 하여, 실제 환경에서 가장 병목이 되는 검출 단계의 도메인 갭 문제를 해결하지 못했다.

### 차별점

UTOPIA는 기존 self-supervised 연구들과 달리 검출 단계(Detection step)를 생략하지 않고 end-to-end로 학습한다. 또한, 단순한 특징 학습을 넘어 cross-domain evaluation protocol을 통해 실제 데이터 획득 과정과 유사한 환경에서 모델의 일반화 성능을 검증했다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조

UTOPIA는 소스 도메인($X_{src}$)과 타겟 도메인($X_{tgt}$)의 데이터를 동시에 처리하는 **Two-branch deep network** 구조를 가진다. 소스 브랜치는 정답(Ground Truth)을 사용하여 지도 학습을 수행하고, 타겟 브랜치는 제안된 OCA와 OPA를 통해 자기지도 학습(Self-supervised learning)을 수행한다.

### 주요 구성 요소 및 작동 원리

#### 1. Object Consistency Agreement (OCA)

타겟 도메인에서 객체의 존재를 확신하기 위해, 동일한 이미지에 두 가지 서로 다른 증강 $\text{aug}(\cdot)$와 $\text{aug}'(\cdot)$를 적용한다. 두 결과물 사이의 일관성을 GIoU(Generalized IoU) 기반의 비용 행렬로 측정한다.

$$ \text{agr}(x_t^{src}) = \text{GIoU}(\text{D}(\text{aug}(x_t^{src})), \text{D}(\text{aug}'(x_t^{src}))) $$

이에 따른 Agreement Loss는 다음과 같이 정의되며, 모델이 입력 변화에 관계없이 일관된 예측을 하도록 유도한다.

$$ L_{agr} = \text{avg}_i (1 - \max_j (\text{agr}(x_t^{src}))) $$

타겟 도메인에서는 이 agreement metric을 활용해 낮은 신뢰도($\epsilon\gamma$)를 가진 검출 결과 중에서도 실제 객체일 가능성이 높은 후보($\tilde{D}_{tgt}$)를 선별한다.

#### 2. Optimal Proposal Assignment (OPA)

객체 ID 할당 및 유사도 학습을 위해 **Optimal Transport (OT)** 이론을 도입한다. 두 특징 벡터 간의 코사인 거리를 기반으로 운송 비용 행렬(Transportation cost matrix) $C$를 생성한다.

$$ c[i,j] = 1 - \frac{F(tr_{t-1}^i)^\top \cdot F(d_j^\top)}{\|F(tr_{t-1}^i)\| \|F(d_j^\top)\|} $$

OT 문제는 다음과 같이 운송 비용을 최소화하는 최적의 플랜 $\pi$를 찾는 문제로 정의된다.

$$ \min_{\pi \in \Pi(p,q)} \sum_i^N \sum_j^M c[i,j]\pi[i,j] $$

여기서 계산 복잡도를 줄이기 위해 **Sinkhorn-Knopp Iteration** 알고리즘을 사용하여 convex 형태로 근사화하고 효율적으로 최적 플랜 $\bar{\pi}$를 도출한다.

#### 3. 학습 전략 및 손실 함수

- **One-to-One (1:1) Assignment**: 타겟 도메인에서 적용한다. $\bar{\pi}$를 통해 가장 가능성이 높은 샘플을 positive($o^+$), 가장 낮은 샘플을 negative($o^-$)로 선택하여 soft-label을 생성하고 $L_{det}$와 $L_{sim}$을 통해 학습한다.
- **One-to-Many (1:M) Assignment**: 소스 도메인에서 적용한다. IoU sampler를 통해 다수의 positive/negative 샘플을 생성하며, 다음과 같은 Multiple-Positive loss($L_{MP}$)를 사용한다.

$$ L_{MP} = \log \left( 1 + \sum_{o^+} \sum_{o^-} \left[ \exp(F(tr_i) \cdot F(o^-)) - \exp(F(tr_i) \cdot F(o^+)) \right] \right) $$

또한, 최적 플랜 $\bar{\pi}$와 실제 정답 $c$ 사이의 차이를 줄이는 보조 손실 함수 $L_{aux}^{1:M} = \bar{\pi}[i,j] - c$를 추가하여 유사도 학습을 가속화한다.

## 📊 Results

### 실험 설정

- **데이터셋**: MOT17, MOT20, MOTSynth, VisDrone, DanceTrack.
- **시나리오**:
    1. MOTSynth $\to$ MOT17 (합성 $\to$ 실제)
    2. MOT17 $\to$ MOT20 (희소 $\to$ 밀집)
    3. MOT17 $\to$ VisDrone (감시 카메라 $\to$ 드론 시점)
    4. MOT17 $\to$ DanceTrack (구분 가능한 외형 $\to$ 유사한 외형)
- **지표**: MOTA (Multiple Object Tracking Accuracy), IDF1 (Identification F1 Score), mAP.

### 주요 결과

- **정량적 성과**: UTOPIA는 cross-domain 설정에서 fully-supervised, unsupervised, self-supervised SOTA 방법론들보다 우수한 MOTA 및 IDF1 성능을 보였다. 특히 MOT17 $\to$ MOT20 시나리오에서 OCA는 MOTA를 17.8% 상승시켰으며, OPA는 IDF1을 14.3% 상승시키는 효과를 보였다.
- **Ablation Study**:
  - **증강 전략**: 타겟 도메인의 특성(예: MOT17의 모션 블러, VisDrone의 시점 변화)을 반영한 증강 기법을 사용했을 때 성능이 극대화됨을 확인하였다.
  - **FP/FN Trade-off**: OCA를 적용했을 때, 낮은 신뢰도의 객체들을 효과적으로 복구함으로써 False Negative를 줄이면서도 False Positive의 증가를 억제하는 강건함을 보였다.

## 🧠 Insights & Discussion

### 강점

본 연구는 MOT 분야에서 드물게 **Cross-domain Adaptation** 문제를 정면으로 다루었으며, 특히 검출기와 추적기를 동시에 적응시키는 end-to-end 프레임워크를 제안했다는 점이 매우 고무적이다. Sinkhorn-Knopp 알고리즘을 통해 효율적인 매칭 최적화를 구현함으로써 self-supervised 환경에서도 안정적인 유사도 학습이 가능함을 입증하였다.

### 한계 및 비판적 해석

1. **모션 모델의 부재**: 논문에서도 언급되었듯, 최신 MOT 프레임워크에서 필수적인 모션 모델(Motion Model)이 self-supervised 방식으로 통합되지 않았다. 이로 인해 DanceTrack과 같이 비선형적 움직임이 심한 데이터셋에서는 IDF1 성능이 ByteTrack 같은 모션 기반 추적기보다 낮게 나타나는 한계가 있다.
2. **객체 타입의 제한**: 현재의 프레임워크는 소스 도메인에 존재하는 객체 타입만 타겟 도메인에서 인식할 수 있다. 새로운 종류의 객체를 발견(Open-vocabulary)하는 기능은 포함되어 있지 않다.
3. **하이퍼파라미터 의존성**: 타겟 도메인의 객체를 선별하는 임계값 $\epsilon\gamma$ 등의 설정이 성능에 민감하게 작용하며, 이에 대한 자동 최적화 방안은 제시되지 않았다.

## 📌 TL;DR

본 논문은 새로운 도메인의 데이터에 대해 레이블 없이도 스스로 적응하는 self-supervised cross-domain MOT 프레임워크인 **UTOPIA**를 제안한다. **Object Consistency Agreement(OCA)**를 통해 타겟 도메인의 객체 존재 여부를 판단하고, **Optimal Proposal Assignment(OPA)**와 Sinkhorn-Knopp 알고리즘을 통해 최적의 ID 매칭 및 유사도 학습을 수행한다. 실험을 통해 합성 데이터나 드론 시점 등 극한의 도메인 변화 상황에서도 기존 SOTA 방법론보다 뛰어난 추적 성능을 보임을 증명하였으며, 이는 향후 레이블 비용을 최소화하면서 다양한 환경에 즉시 적용 가능한 MOT 시스템 구축에 중요한 기여를 할 것으로 평가된다.
