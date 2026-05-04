# Instance-specific and Model-adaptive Supervision for Semi-supervised Semantic Segmentation

Zhen Zhao, Sifan Long, Jimin Pi, Jingdong Wang, Luping Zhou (2022)

## 🧩 Problem to Solve

본 논문은 Semi-supervised Semantic Segmentation (SSS) 분야에서 발생하는 핵심적인 문제, 즉 **미라벨링된(unlabeled) 데이터의 개별적 특성과 학습 난이도 차이를 무시하는 경향**을 해결하고자 한다.

기존의 SSS 연구들은 대부분 모든 unlabeled 데이터를 동일하게 취급하여 처리한다. 그러나 저자들은 다음과 같은 두 가지 구체적인 문제점을 지적한다:

1. **과도한 섭동(Over-perturbation):** 모든 데이터에 무차별적으로 강한 데이터 증강(strong augmentation)을 적용할 경우, 학습 난이도가 높은 샘플은 모델의 일반화 능력을 벗어나는 수준으로 왜곡되어 오히려 학습을 방해하고 데이터 분포를 해칠 수 있다.
2. **노이즈에 대한 과적합(Overfitting to noise):** 서로 다른 난이도를 가진 샘플들의 consistency loss를 단순히 평균 내어 최소화하는 방식은, 학습하기 어려운 샘플(noisy supervision)에 모델이 과도하게 집중하게 만들어 성능 저하를 초래할 수 있다.

따라서 본 논문의 목표는 각 unlabeled 인스턴스의 특성에 맞춘 **인스턴스 특화적(instance-specific)이며 모델 적응적인(model-adaptive) 감독 체계인 iMAS**를 제안하여 SSS 성능을 높이는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 모델의 학습 상태에 따라 각 unlabeled 샘플의 **'정량적 난이도(Quantitative Hardness)'**를 평가하고, 이를 기반으로 **데이터 증강 강도와 손실 함수 가중치를 동적으로 조절**하는 것이다.

주요 기여 사항은 다음과 같다:

- **정량적 난이도 분석:** Teacher-Student 모델 간의 예측 불일치를 기반으로 한 'Class-weighted Symmetric IoU' 지표를 통해 unlabeled 인스턴스의 난이도를 수치화하였다.
- **모델 적응적 데이터 증강:** 평가된 난이도에 따라 Intensity-based 증강의 강도를 조절하고, CutMix 적용 시 '어려운 샘플-쉬운 샘플' 쌍을 전략적으로 구성하여 학습 효율을 높였다.
- **인스턴스 특화적 손실 함수:** Curriculum Learning의 개념을 도입하여, 쉬운 샘플에 더 높은 가중치를 부여하는 방식의 unsupervised loss를 설계하였다.
- **추가 비용 없는 성능 향상:** 별도의 추가 네트워크나 복잡한 학습 절차 없이, 기존의 Consistency Regularization 프레임워크 내에서 SOTA 성능을 달성하였다.

## 📎 Related Works

SSS 연구는 크게 두 가지 흐름으로 나뉜다:

1. **Self-training (ST):** 모델이 생성한 pseudo-label을 통해 모델을 다시 학습시키는 방식이다.
2. **Consistency Regularization (CR):** 데이터나 모델에 섭동(perturbation)을 가하고, 서로 다른 뷰(view) 간의 예측 일관성을 강제하는 방식이다. 최근에는 CutMix와 같은 강한 증강이나 Contrastive Learning을 결합하여 성능을 높이는 추세이다.

**기존 연구의 한계 및 iMAS의 차별점:**

- 기존의 CR 기반 방식들은 unlabeled 샘플을 무차별적으로 섭동시키고 평균 손실을 계산한다.
- 인스턴스 난이도(Instance Hardness) 연구는 주로 Ground Truth가 존재하는 Supervised 학습 환경에서 손실 값을 기준으로 이루어졌다.
- **iMAS**는 Label이 없는 상황에서도 Teacher-Student 모델의 합의(consensus) 정도를 이용하여 정량적인 난이도를 측정함으로써, 이를 SSS 과정에 직접적으로 주입했다는 점에서 기존 방식과 차별화된다.

## 🛠️ Methodology

### 1. 전체 파이프라인 및 구조

iMAS는 기본적인 Teacher-Student 프레임워크를 따른다. Student 모델($\theta_s$)은 학습 가능한 가중치를 가지며, Teacher 모델($\theta_t$)은 Student 모델 가중치의 지수 이동 평균(EMA)으로 업데이트된다.

전체 학습 손실 함수는 다음과 같다:
$$L = L_x + \lambda_u L_u$$
여기서 $L_x$는 labeled 데이터에 대한 supervised loss이며, $L_u$는 unlabeled 데이터에 대한 unsupervised loss이다.

### 2. 정량적 난이도 분석 ($\phi$)

iMAS는 Teacher와 Student 모델의 예측 결과 사이의 불일치를 통해 난이도를 측정한다. 클래스 불균형 문제를 해결하기 위해 Class-weighted IoU를 사용하며, 계산의 비가환성을 해결하기 위해 symmetric한 구조를 취한다.

먼저, 각 모델의 고신뢰도 픽셀 비율 $\rho$를 계산한다:
$$\rho_{s/t}^i = \frac{1}{H \times W} \sum_{j=1}^{H \times W} \mathbb{1}(\max(p_{s/t}^i(j)) \ge \tau)$$

이후, $i$번째 인스턴스의 symmetric hardness $\gamma_i$는 다음과 같이 정의된다:
$$\gamma_i = \phi(p_t^i, p_s^i) = 1 - \left[ \frac{\rho_{s}^i}{2} wIoU(p_s^i, p_t^i) + \frac{\rho_{t}^i}{2} wIoU(p_t^i, p_s^i) \right]$$
$\gamma_i$ 값이 클수록 학습하기 어려운(Hard) 샘플이며, 작을수록 쉬운(Easy) 샘플임을 의미한다.

### 3. 모델 적응적 감독 (Model-adaptive Supervision)

#### (1) Intensity-based 강도 조절

강한 증강이 데이터 분포를 해치는 것을 방지하기 위해, 난이도 $\gamma_i$에 따라 강한 증강($A_s^I$)과 약한 증강($A_w$) 결과를 혼합한다:
$$A_s^I(u_i) \leftarrow \gamma_i A_s^I(u_i) + (1 - \gamma_i) A_w(u_i)$$

- **Hard 샘플 ($\gamma_i$ 높음):** 상대적으로 섭동이 약해져 Out-of-distribution 사례로 빠지는 것을 방지한다.
- **Easy 샘플 ($\gamma_i$ 낮음):** 강한 섭동을 적용하여 모델이 더 견고하게 학습하도록 유도한다.

#### (2) 모델 적응적 CutMix

- **트리거 확률:** mini-batch 내의 평균 난이도 $\bar{\gamma}$를 기준으로 CutMix 적용 여부를 결정한다.
- **Hard-Easy 페어링:** batch 내 샘플들을 난이도 순으로 정렬한 뒤, 가장 어려운 샘플과 가장 쉬운 샘플을 짝지어 CutMix를 수행함으로써 학습 효율을 극대화한다.

#### (3) 모델 적응적 Unsupervised Loss

Curriculum Learning의 원리를 적용하여, 쉬운 샘플에 더 많은 가중치를 부여한다. 각 인스턴스의 가중치는 $(1 - \gamma_i)$로 설정된다:
$$L_u = \frac{1}{|B_u|} \sum_{i=1}^{|B_u|} (1 - \gamma_i) \cdot \text{ConsistencyLoss}(f_{\theta_s}(A_s(u_i)), p_t^i)$$
즉, 모델이 이미 잘 이해하고 있는 쉬운 샘플로부터 먼저 안정적으로 학습하고, 어려운 샘플의 영향력은 낮추어 노이즈로 인한 과적합을 방지한다.

## 📊 Results

### 실험 설정

- **데이터셋:** Pascal VOC 2012 (Classic 및 Blended), Cityscapes.
- **백본:** DeepLabv3+ (ResNet-50, ResNet-101).
- **평가지표:** mean Intersection-over-Union (mIoU).
- **비교 대상:** Supervised baseline, MT, CCT, CutMix-Seg, ST++, $U^2PL$ 등.

### 주요 결과

1. **SOTA 달성:** iMAS는 다양한 labeled data 비율(partition protocols)에서 기존 SOTA 방법론들을 능가하였다.
2. **Label 효율성:** 특히 labeled 데이터가 매우 적은 상황(label-scarce scenarios)에서 성능 향상폭이 컸다.
    - Blended VOC 2012에서 662개의 label만 사용했을 때 75.9% mIoU를 기록하며, $U^2PL$이 1323개의 label로 얻은 성능(75.2%)을 앞질렀다.
3. **데이터셋별 성능:**
    - **Pascal VOC 2012:** ResNet-101 기준, 1/8 파티션에서 supervised baseline 대비 상당한 성능 향상을 보였다.
    - **Cityscapes:** ResNet-50 기준, 1/16 파티션에서 supervised baseline 대비 10.3% 향상, $U^2PL$ 대비 6.5% 향상된 결과를 보였다.

### Ablation Study

- **구성 요소별 기여:** Unsupervised loss 가중치 조절, Intensity-based 증강, CutMix 증강 각각이 단독으로도 성능을 향상시키며, 모두 결합했을 때 가장 높은 성능(Supervised 대비 +5.8% mIoU)을 보였다.
- **하이퍼파라미터 민감도:** $\lambda_u$와 $\tau$ 값의 변화에 따라 성능이 변하며, 특히 Cityscapes와 같이 어려운 데이터셋에서는 낮은 $\tau$ 값(0.7)이 더 효과적임이 확인되었다.

## 🧠 Insights & Discussion

**강점:**

- **단순성과 효율성:** 별도의 보조 네트워크(correcting network)나 복잡한 손실 함수를 추가하지 않고도, 데이터 증강과 가중치 조절이라는 간단한 메커니즘만으로 성능을 비약적으로 높였다.
- **동적 적응력:** 모델이 학습됨에 따라 샘플의 난이도가 변한다는 점($\gamma_i$의 감소)을 잘 포착하여 학습 과정에 반영하였다.

**한계 및 비판적 해석:**

- **계산 비용:** 난이도를 평가하기 위해 unlabeled 샘플에 대해 Student와 Teacher 모델 모두 forward pass를 수행해야 하므로, 연산량이 증가하는 측면이 있다. (저자 역시 이를 한계점으로 명시함)
- **난이도 지표의 타당성:** Teacher-Student 간의 IoU 불일치를 '난이도'의 대리 지표(proxy)로 사용하였는데, 이것이 실제 ground truth 기반의 난이도와 얼마나 일치하는지에 대한 심층적인 분석은 부족해 보인다. 다만, 실험 결과가 이를 뒷받침하고 있다.

## 📌 TL;DR

본 논문은 Semi-supervised Semantic Segmentation에서 모든 unlabeled 데이터를 동일하게 처리하는 기존 방식의 한계를 지적하고, 인스턴스별 난이도를 정량화하여 학습에 반영하는 **iMAS** 프레임워크를 제안한다. Teacher-Student 모델 간의 예측 불일치(Symmetric IoU)를 통해 난이도를 측정하고, 이를 바탕으로 **데이터 증강 강도를 조절**하고 **손실 함수에 가중치를 부여**함으로써, 추가 네트워크 없이도 label-scarce 상황에서 SOTA 성능을 달성하였다. 이 연구는 SSS 분야에서 모델의 상태와 데이터의 특성을 결합한 적응적 학습 전략의 중요성을 시사한다.
