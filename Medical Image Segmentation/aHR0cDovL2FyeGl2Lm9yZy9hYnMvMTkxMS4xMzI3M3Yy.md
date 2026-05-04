# Confidence Calibration and Predictive Uncertainty Estimation for Deep Medical Image Segmentation

Alireza Mehrtash, William M. Wells III, Clare M. Tempany, Purang Abolmaesumi, and Tina Kapur (2020)

## 🧩 Problem to Solve

본 논문은 의료 영상 분할(Medical Image Segmentation) 분야에서 널리 사용되는 Fully Convolutional Neural Networks(FCNs), 특히 U-Net 구조의 모델들이 가진 **신뢰도 보정(Confidence Calibration)** 문제를 해결하고자 한다.

딥러닝 모델은 예측값과 함께 해당 예측이 맞을 확률(신뢰도)을 제공하지만, 실제로는 맞든 틀리든 지나치게 확신하는 **과잉 확신(Overconfidence)** 경향이 있다. 특히 의료 분야에서는 모델이 틀린 예측을 하면서도 높은 신뢰도를 보일 경우, 임상적 의사결정에서 치명적인 위험을 초래할 수 있다.

또한, 모델의 성능을 높이기 위해 흔히 사용되는 **Batch Normalization(BN)**과 **Dice loss**가 학습의 안정성과 분할 품질은 향상시키지만, 결과적으로 모델의 Calibration 품질을 악화시켜 불확실성 추정(Uncertainty Estimation)을 어렵게 만든다는 점이 주요 문제로 제기된다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 모델 앙상블(Model Ensembling)을 통해 Dice loss와 BN을 사용한 FCN의 신뢰도를 보정하고, 이를 통해 신뢰할 수 있는 불확실성 추정치를 얻는 것이다. 주요 기여 사항은 다음과 같다.

1. **손실 함수 비교 분석**: Segmentation 품질과 불확실성 추정 관점에서 Cross-entropy loss와 Dice loss의 성능을 체계적으로 비교하였다.
2. **신뢰도 보정을 위한 앙상블 제안**: Dice loss와 BN으로 학습된 모델들의 Calibration 문제를 해결하기 위해, 서로 다른 초기값과 데이터 셔플링으로 학습된 여러 모델의 예측값을 평균 내는 앙상블 기법을 제안하였다.
3. **세그먼트 수준의 품질 예측 및 OOD 탐지**: 예측된 분할 영역(Segment) 내의 평균 엔트로피(Average Entropy)를 계산하여, 정답(Ground Truth) 없이도 분할 품질을 예측하고 분포 외 데이터(Out-of-distribution, OOD)를 탐지하는 방법론을 제시하였다.

## 📎 Related Works

기존의 불확실성 추정 방식은 크게 Bayesian 접근법과 Non-Bayesian 접근법으로 나뉜다.

- **Bayesian 접근법**: 모델 파라미터를 확률 분포로 처리하며, 대표적으로 **MC dropout**이 있다. 추론 시 Dropout을 활성화한 상태로 여러 번 실행하여 그 변동성을 통해 불확실성을 측정한다.
- **Non-Bayesian 접근법**: **Deep Ensembles**가 대표적이며, 서로 다르게 초기화된 여러 모델을 독립적으로 학습시켜 그 결과의 평균과 분산을 이용한다.

논문은 기존 연구들이 주로 일반 이미지 분류에 집중했거나, 의료 영상 분할에서는 MC dropout 등의 기법이 사용되었음을 언급한다. 하지만 본 논문은 특히 Dice loss와 BN이 Calibration에 미치는 악영향을 구체적으로 분석하고, 이를 해결하기 위한 실용적인 앙상블 레시피를 제공한다는 점에서 기존 연구와 차별화된다.

## 🛠️ Methodology

### 1. 모델 구조 및 학습 설정
기본 아키텍처로 2D U-Net을 사용하며, 입력 및 출력 크기는 $224 \times 224$ 픽셀이다. 모든 모델은 Batch Normalization을 포함하며, Adam 옵티마이저를 사용하여 학습한다.

### 2. 손실 함수 (Loss Functions)
논문에서는 두 가지 손실 함수를 비교한다.

- **Cross-Entropy (CE) Loss**: 각 픽셀의 클래스 확률 분포와 실제 라벨 간의 로그 가능도를 최대화한다.
$$L_{CE} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{k=1}^{K} \omega_k \ln(p(\hat{y}_i=k|x_i, \theta)) \cdot (y_i=k)$$
여기서 $\omega_k$는 클래스 불균형을 해결하기 위한 가중치이다.

- **Dice Loss**: 예측된 영역과 실제 영역의 겹침 정도(Overlap)를 직접 최적화한다.
$$L_{DSC} = -2 \sum_{k=1}^{K} \omega_k \frac{\sum_{i=1}^{N} [p(\hat{y}_i=k|x_i, \theta) \cdot (y_i=k)]}{\sum_{i=1}^{N} [p(\hat{y}_i=k|x_i, \theta) + (y_i=k)] + \epsilon}$$
$\epsilon$은 수치적 안정성을 위한 smoothing factor이다.

### 3. 신뢰도 보정 및 앙상블 (Confidence Calibration via Ensembling)
Dice loss로 학습된 단일 모델의 과잉 확신 문제를 해결하기 위해, $M$개의 모델을 독립적으로 학습시킨 후 그 확률값의 산술 평균을 최종 예측값으로 사용한다.
$$p^E(y_j=k|x_j) = \frac{1}{M} \sum_{m=1}^{M} p(y_j=k|x_j, \theta^*_m)$$
이 방식은 네트워크 구조를 수정할 필요가 없으며, 경험적으로 $M \ge 5$일 때 유의미한 보정 효과가 나타남을 확인하였다.

### 4. 세그먼트 수준의 불확실성 측정 (Segment-level Uncertainty)
픽셀 단위의 신뢰도를 넘어, 분할된 객체 전체의 품질을 평가하기 위해 예측된 전경(Foreground) 영역 $\hat{S}_k$ 내의 평균 엔트로피 $H(\hat{S}_k)$를 계산한다.
$$H(\hat{S}_k) = -\frac{1}{|\hat{S}_k|} \sum_{i \in \hat{S}_k} [p \ln p + (1-p) \ln (1-p)]$$
여기서 $p$는 해당 픽셀이 클래스 $k$에 속할 확률이다. 엔트로피가 높을수록 모델이 해당 영역의 예측에 대해 불확실함을 의미하며, 이는 실제 분할 품질(Dice score)의 저하와 강한 상관관계를 가진다.

## 📊 Results

### 1. 실험 설정
- **데이터셋**: 뇌종양(BraTS), 심장 벤트리클(ACDC), 전립선(PROSTATEx, PROMISE12) MRI 영상.
- **평가 지표**:
    - Calibration 품질: Negative Log-Likelihood (NLL), Brier Score, Expected Calibration Error (ECE%).
    - Segmentation 품질: Dice Score, 95th Hausdorff Distance (HD).

### 2. 주요 결과
- **CE vs Dice**: 단일 모델 기준, CE loss는 Calibration 품질(NLL, ECE)이 우수하지만 Segmentation 품질(Dice)은 낮았다. 반면, Dice loss는 Segmentation 성능은 뛰어나지만 Calibration 품질이 매우 나빴다.
- **앙상블의 효과**: Dice loss로 학습된 모델들을 앙상블($M=50$)했을 때, Calibration 품질과 Segmentation 품질 모두가 비약적으로 향상되었다. 이는 MC dropout보다 더 우수한 성능을 보였다.
- **품질 예측 및 OOD 탐지**: 제안한 세그먼트 평균 엔트로피 $H(\hat{S})$와 logit-transformed Dice score 간의 상관계수가 $0.77 \le r \le 0.92$로 매우 높게 나타났다.
- **OOD 탐지 사례**: 전립선 데이터셋에서 학습 데이터와 다른 프로토콜(Endorectal coil 사용)로 촬영된 영상(OOD)의 경우, 엔트로피 값이 높게 측정되어 모델이 스스로 "잘 모른다"고 판단함을 확인하였다.

## 🧠 Insights & Discussion

본 연구는 의료 영상 분할에서 성능(Dice score)과 신뢰도(Calibration) 사이의 상충 관계(Trade-off)를 명확히 드러냈다. 특히 Dice loss가 유발하는 과잉 확신 문제는 앙상블 기법을 통해 효과적으로 완화될 수 있음을 입증하였다.

**강점 및 시사점:**
- 단순히 성능을 높이는 것이 아니라, 모델의 예측이 얼마나 믿을만한지를 정량화함으로써 의료 현장에서의 안전성을 높일 수 있는 실무적 방법(Recipe)을 제공한다.
- 세그먼트 수준의 엔트로피를 통해 정답 없이도 추론 시점에 품질을 예측할 수 있다는 점은 매우 실용적이다.

**한계 및 논의 사항:**
- **계산 비용**: 앙상블은 여러 모델을 처음부터 다시 학습시켜야 하므로 시간과 자원 소모가 매우 크다. 학습 없이 보정할 수 있는 Temperature Scaling 등의 대안에 대한 심도 있는 연구가 필요하다.
- **가정**: 본 연구의 세그먼트 엔트로피 지표는 이진 분류를 가정하고 계산되었으며, 다중 클래스 상황에서의 일반화 가능성에 대해서는 추가 연구가 필요하다.
- **데이터 범위**: MRI 영상에 한정된 실험이므로, CT 등 다른 모달리티에서도 동일한 경향이 나타나는지 확인이 필요하다.

## 📌 TL;DR

의료 영상 분할 모델(U-Net)에서 성능을 높이는 Dice loss와 Batch Normalization이 모델을 과잉 확신하게 만들어 신뢰도를 떨어뜨리는 문제를 다룬다. 이를 해결하기 위해 **모델 앙상블**을 적용하면 분할 정확도와 신뢰도 보정을 동시에 달성할 수 있으며, **세그먼트 평균 엔트로피**를 통해 정답 없이도 분할 품질을 예측하고 분포 외 데이터(OOD)를 효과적으로 탐지할 수 있다. 이 연구는 의료 AI의 신뢰성을 확보하기 위한 실질적인 가이드라인을 제공한다.