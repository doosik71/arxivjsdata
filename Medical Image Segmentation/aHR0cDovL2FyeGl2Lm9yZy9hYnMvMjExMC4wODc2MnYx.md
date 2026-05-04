# Inconsistency-aware Uncertainty Estimation for Semi-supervised Medical Image Segmentation

Yinghuan Shi, Jian Zhang, Tong Ling, Jiwen Lu, Yefeng Zheng, Qian Yu, Lei Qi, Yang Gao (2021)

## 🧩 Problem to Solve

본 논문은 의료 영상 분할(Medical Image Segmentation) 분야에서 레이블이 지정된 데이터의 부족 문제를 해결하기 위한 반지도 학습(Semi-supervised Learning, SSL) 방법론을 다룬다. 의료 영상의 정밀한 픽셀 단위 레이블링은 전문가의 많은 시간과 노력이 소요되며, 주관적인 요인으로 인해 품질이 일정하지 않을 수 있다는 문제가 있다.

기존의 반지도 학습 기반 분할 모델들은 주로 Softmax 층의 출력값(Entropy)을 기반으로 불확실성(Uncertainty)을 추정하여 Pseudo-label을 생성한다. 하지만 저자들은 다음과 같은 두 가지 핵심 문제를 제기한다.
1. **초기 Pseudo-label의 중요성**: 초기 단계에서 생성된 Pseudo-label에 오류가 많을 경우, 학습 과정에서 오류가 전파(Error Propagation)되어 최종 성능을 저하시킨다. 기존의 Softmax 기반 신뢰도 추정 방식은 정밀도(Precision)와 재현율(Recall) 사이의 단순한 트레이드-오프 관계에 머물러 있어 안전하고 신뢰할 수 있는 초기화가 어렵다.
2. **영역별 특성 무시**: 기존 방식은 신뢰도가 높은 Certain region과 낮은 Uncertain region을 동일한 네트워크에 입력하여 처리한다. 이는 신뢰도 높은 영역의 이점을 충분히 활용하지 못하게 하며, 불확실한 영역의 복잡성을 과소평가하는 결과를 초래한다.

따라서 본 논문의 목표는 새로운 불확실성 추정 방식을 통해 더 신뢰할 수 있는 Pseudo-label을 생성하고, Certain region과 Uncertain region을 분리하여 처리하는 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **오분류 비용(Misclassification Cost)의 설정에 따른 예측 결과의 불일치(Inconsistency)를 이용하여 불확실성을 추정**하는 것이다. 

구체적으로, 클래스 간의 비용 가중치를 다르게 설정한 두 모델(Conservative-Radical)을 학습시켰을 때, 두 모델의 예측 결과가 서로 다르다면 해당 픽셀은 불확실한 영역으로 간주하고, 결과가 일치한다면 확실한 영역으로 간주한다. 이러한 직관을 바탕으로 **CoraNet (Conservative-Radical Network)**이라는 새로운 반지도 학습 프레임워크를 제안하며, Certain region과 Uncertain region에 대해 서로 다른 학습 전략을 적용하는 Separate Self-training 전략을 도입하였다.

## 📎 Related Works

### 관련 연구 및 한계
1. **의료 영상 분할**: U-Net과 같은 Encoder-Decoder 구조의 FCN이 주류를 이루고 있으며, Attention mechanism이나 정교한 Loss function을 통해 성능을 개선해 왔다. 그러나 대부분 완전 지도 학습(Fully Supervised)에 의존한다는 한계가 있다.
2. **불확실성 추정**: Bayesian modeling, Monte Carlo Dropout, Input Augmentation 등을 통해 예측의 신뢰도를 측정한다. 하지만 의료 영상 SSL에서는 구현이 쉬운 Softmax 기반의 Confidence threshold 방식을 주로 사용하며, 이는 앞서 언급한 오류 전파 문제에 취약하다.
3. **반지도 학습(SSL)**: $\Pi$-model, Mean Teacher와 같이 일관성 규제(Consistency Regularization, CR)나 엔트로피 최소화(Entropy Minimization, EM)를 사용하는 방법들이 제안되었다.

### 기존 방식과의 차별점
기존 방법들이 주로 동일한 모델에 섭동(Perturbation)을 주거나 단순한 Softmax 확률값을 사용하는 것과 달리, CoraNet은 **서로 다른 오분류 비용을 가진 모델들 사이의 예측 불일치**를 통해 영역 수준(Region-level)의 마스크를 생성한다. 또한, 생성된 마스크를 기반으로 Certain region에는 직접적인 Self-training을, Uncertain region에는 Mean Teacher 기반의 정교한 일관성 학습을 적용함으로써 두 전략의 장점을 모두 취한다.

## 🛠️ Methodology

### 전체 시스템 구조
CoraNet은 크게 세 가지 구성 요소로 이루어져 있으며, 이들은 end-to-end 방식으로 교차 학습된다.
1. **Conservative-Radical Module (CRM)**: 불확실성 영역을 식별하기 위한 모듈.
2. **Certain Region Segmentation Network (C-SN)**: 확실한 영역을 전문적으로 학습하는 네트워크.
3. **Uncertain Region Segmentation Network (UC-SN)**: 불확실한 영역을 전문적으로 학습하는 네트워크.

### 상세 방법 및 알고리즘

#### 1. CRM을 통한 불확실성 추정
CRM은 하나의 공유 Encoder $E$와 세 개의 서로 다른 Decoder ($D, D_{con}, D_{rad}$)를 가진다. 여기서 $D_{con}$과 $D_{rad}$는 각각 다음과 같은 비용 민감도 설정을 가진다.
- **Object Conservative Setting**: 배경을 객체로 오분류하는 비용을 매우 높게 설정하여, 매우 확실한 객체의 중심부만 예측하도록 유도한다.
- **Object Radical Setting**: 객체를 배경으로 오분류하는 비용을 높게 설정하여, 배경에 대해 매우 신중하게(즉, 객체 영역을 넓게) 예측하도록 유도한다.

이때 사용되는 손실 함수는 다음과 같다.
$$L^{ce-con}(X^l_i, Y^l_i; E, D_{con}) = -\sum_{k=1}^2 \sum_{z=1}^{H_0 \times W_0} w^{con}_k q_{z,k} \log p_{z,k}$$
$$L^{ce-rad}(X^l_i, Y^l_i; E, D_{rad}) = -\sum_{k=1}^2 \sum_{z=1}^{H_0 \times W_0} w^{rad}_k q_{z,k} \log p_{z,k}$$
여기서 $w^{con}$은 배경 클래스($k=0$)에 $\alpha$배의 가중치를, $w^{rad}$는 객체 클래스($k=1$)에 $\alpha$배의 가중치를 부여한다 ($\alpha=5$). 전체 CRM 손실 함수는 일반적인 $L_{ce}$와 위 두 손실의 합으로 정의된다.

추출된 예측 결과 $Y_{u,con}$과 $Y_{u,rad}$를 이용하여 마스크를 다음과 같이 생성한다.
- **Uncertain Mask ($M_{u,uc}$)**: 두 예측의 XOR 연산을 통해 불일치하는 영역을 추출한다.
  $$M_{u,uc} = Y_{u,rad} \oplus Y_{u,con}$$
- **Certain Mask ($M_{u,c}$)**: $M_{u,uc}$의 반전 영역이다.
  $$M_{u,c} = 1 - M_{u,uc}$$

#### 2. Separate Self-training (C-SN & UC-SN)
- **C-SN (Certain region)**: Certain mask에 해당하는 영역은 예측 신뢰도가 매우 높다고 판단하여, 해당 예측값을 Pseudo-label로 직접 사용하여 모델을 학습시킨다.
  $$L_{c-sn}(E, D) = \frac{1}{n} \sum_{j=1}^n \sum_{k=1}^2 \sum_{z=1}^{H_0 \times W_0} M_{u,c,j,z} q_{z,k} \log p_{z,k}$$
- **UC-SN (Uncertain region)**: 불확실한 영역은 Pseudo-label을 직접 믿기 어려우므로, **Mean Teacher** 프레임워크를 도입한다. Student 모델과 Teacher 모델(Student의 가중치 이동 평균) 사이의 예측 일관성을 강제하는 손실 함수를 사용한다.
  $$L_{cnt} = \frac{1}{n} \sum_{j=1}^n \| M_{u,uc,j} \otimes (F(X^u_j; E, D) - F'(X^u_j; E', D')) \|^2$$

### 학습 및 추론 절차
1. Labeled data를 통해 $E, D, D_{con}, D_{rad}$를 초기화한다.
2. 반복적으로 다음 과정을 수행한다:
   - C-SN을 통해 Certain region 학습.
   - UC-SN을 통해 Uncertain region 학습.
   - CRM을 통해 Encoder와 Decoder들을 업데이트하고 Mask를 갱신한다.
   - Teacher 모델의 가중치를 EMA(Exponential Moving Average) 방식으로 업데이트한다.
3. **추론(Test)** 단계에서는 $D_{con}$과 $D_{rad}$를 폐기하고, 메인 모델인 $E$와 $D$만을 사용하여 분할 결과를 도출한다.

## 📊 Results

### 실험 설정
- **데이터셋**: CT Pancreas, MR Endocardium, ACDC (Multi-class)
- **지표**: Dice Score (DSC), Precision, Recall, Hausdorff Distance (HD)
- **비교 대상**: U-Net (Supervised), $\Pi$-model, Temporal Ensembling, Mean Teacher, UA-MT

### 주요 결과
1. **CT Pancreas (2D)**: 50%의 레이블 데이터를 사용했을 때, CoraNet은 **67.01%의 DSC**를 기록하며 UA-MT(63.82%)를 포함한 모든 베이스라인을 능가하였다. 특히 HD(Hausdorff Distance)에서 15.90 voxel로 가장 낮은 수치를 기록하여 경계 예측 능력이 뛰어남을 보였다.
2. **레이블 비율 변화**: 레이블과 언레이블 데이터의 비율을 $1:4, 1:2, 1:1$로 변경하며 실험한 결과, CoraNet은 모든 조건에서 타 모델보다 일관되게 높은 DSC를 유지하였다.
3. **3D 분할 성능**: V-Net을 백본으로 사용하여 3D CT Pancreas 데이터셋을 평가한 결과, DSC 79.67%를 달성하여 기존 V-Net 기반 SOTA 모델들(SASSNet, DTC 등)보다 우수한 성능을 보였다.
4. **불확실성 추정 방법 비교**: MC Dropout, Aleatoric Uncertainty 등 다른 추정 방식을 적용했을 때보다 본 논문이 제안한 Inconsistency-aware 방식이 2D/3D 모두에서 더 높은 DSC를 기록하였다.
5. **MR Endocardium**: DSC 86.67%를 기록하며 UA-MT(82.02%) 대비 큰 폭의 성능 향상을 보였다.
6. **ACDC (Multi-class)**: RV, Myo, LV 세 클래스에 대해 평균 86.10%의 DSC를 기록하여, 반지도 학습 모델 중 가장 우수했으며 완전 지도 학습(88.79%)에 근접한 성능을 보였다.

## 🧠 Insights & Discussion

### 강점
- **강건한 초기화**: CRM을 통한 불확실성 추정이 기존 Softmax 기반 방식보다 Precision과 Recall 사이의 균형을 잘 잡으며, 이는 초기 Pseudo-label의 품질을 높여 오류 전파를 효과적으로 억제한다.
- **효율적인 파라미터 운용**: 학습 시에는 보조 디코더($D_{con}, D_{rad}$)를 사용하지만, 추론 시에는 이를 제거하므로 U-Net과 동일한 추론 속도와 메모리 사용량을 가진다.
- **영역별 맞춤 전략**: 신뢰도에 따라 Self-training과 Mean Teacher 전략을 분리 적용함으로써, 학습의 효율성과 안정성을 동시에 확보하였다.

### 한계 및 논의사항
- **하이퍼파라미터 $\alpha$**: 오분류 비용을 결정하는 $\alpha$ 값에 따라 성능이 변하며, 본 논문에서는 $\alpha=5$가 최적임을 보였다. $\alpha$가 너무 크거나 작으면 노이즈가 발생할 수 있다는 점이 지적되었다.
- **계산 비용**: Mean Teacher 구조와 세 개의 디코더를 학습시켜야 하므로, 단순 U-Net보다는 학습 시간이 소요된다 (다만, MT 모델과는 유사한 수준이다).

### 비판적 해석
본 논문은 불확실성을 '모델의 예측 확률'이 아닌 '조건 변화에 따른 결과의 일관성'으로 재정의함으로써 의료 영상 SSL의 고질적인 문제인 Pseudo-label 노이즈 문제를 효과적으로 해결하였다. 특히, 불확실한 영역이 학습 과정에서 고정된 것이 아니라 모델의 발전과 함께 변화한다는 점을 시각적으로 증명하여, 동적인 마스크 업데이트의 정당성을 부여한 점이 인상적이다.

## 📌 TL;DR

이 논문은 오분류 비용(Misclassification Cost)을 다르게 설정한 두 모델의 예측 불일치를 이용하여 픽셀 단위의 불확실성을 추정하는 **CoraNet**을 제안한다. 확실한 영역(Certain region)은 직접적인 Pseudo-labeling으로, 불확실한 영역(Uncertain region)은 Mean Teacher 기반의 일관성 학습으로 분리 처리하는 전략을 통해, 의료 영상 분할에서 데이터 부족 문제를 극복하고 SOTA 성능을 달성하였다. 이 연구는 향후 레이블이 극도로 부족한 의료 영상 분석 환경에서 매우 실용적인 가이드라인을 제공할 것으로 기대된다.