# Boundary Difference Over Union Loss For Medical Image Segmentation

Fan Sun, Zhiming Luo, and Shaozi Li (2023)

## 🧩 Problem to Solve

의료 영상 분할(Medical Image Segmentation)은 임상 진단에서 매우 중요한 역할을 하지만, 기존의 손실 함수(Loss Function)들은 주로 전체적인 분할 결과의 정확도에 집중하는 경향이 있다. 경계 영역의 분할 성능을 높이기 위해 제안된 일부 손실 함수들은 다른 손실 함수와 결합하여 사용해야만 하거나, 학습 과정에서 불안정하여 효과적인 결과를 내지 못하는 한계가 존재한다. 특히, 경계 영역은 해부학적 구조의 정확한 식별을 위해 필수적임에도 불구하고, 이를 직접적으로 가이드할 수 있는 단순하고 안정적인 손실 함수의 부재가 문제로 지적된다. 따라서 본 논문의 목표는 경계 영역의 분할을 효과적으로 가이드하면서도 구현이 쉽고 학습이 안정적인 새로운 손실 함수인 Boundary Difference over Union (Boundary DoU) Loss를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Boundary IoU 메트릭에서 영감을 얻어, 미분 불가능한 연산을 배제하고 영역 계산 기반으로 설계된 **Boundary DoU Loss**를 제안한 것이다. 이 손실 함수는 예측값과 정답(Ground Truth) 사이의 차이 집합(Difference Set)을 활용하여 경계 영역의 오차를 줄이는 방식이다. 또한, 타겟 객체의 크기에 따라 경계 영역에 부여하는 가중치를 적응적으로 조절하는 **Adaptive Size Strategy**를 도입하여, 객체의 크기에 관계없이 최적의 분할 성능을 낼 수 있도록 설계하였다.

## 📎 Related Works

논문에서는 의료 영상 분할에 사용되는 손실 함수를 세 가지 범주로 분류하여 설명한다.

1. **Cross-Entropy 기반:** 예측 확률 분포와 정답 간의 차이를 계산하는 방식으로, Focal Loss 등이 하드 샘플 문제를 해결하기 위해 제안되었다.
2. **Dice 기반:** 정답과 예측의 교집합과 합집합을 이용하는 Dice Loss와, 정밀도(Precision)와 재현율(Recall)의 균형을 맞춘 Tversky Loss, 다중 클래스 분할로 확장한 Generalized Dice Loss 등이 있다.
3. **경계 집중 기반:** Hausdorff Distance Loss나 Boundary Loss 등이 있으며, 특히 Boundary Loss는 윤곽선 상의 점들 사이의 거리를 가중치로 사용하여 계산한다.

기존의 경계 중심 접근 방식들은 학습의 불안정성이나 타 손실 함수와의 복합적인 사용이 강제된다는 한계가 있다. 또한, 평가 지표로 사용되는 Boundary IoU는 침식(Erode) 연산을 포함하고 있어 미분 불가능하므로, 이를 직접적인 손실 함수로 사용할 수 없다는 차이점이 있다.

## 🛠️ Methodology

### 전체 파이프라인 및 Boundary DoU Loss

본 논문은 미분 불가능한 Boundary IoU 메트릭을 학습 가능한 형태의 손실 함수로 변환하기 위해, 영역 계산 방식을 도입하였다. 제안된 Boundary DoU Loss는 다음과 같은 방정식으로 정의된다.

$$L_{DoU} = \frac{G \cup P - G \cap P}{G \cup P - \alpha * G \cap P}$$

여기서 $G$는 Ground Truth, $P$는 예측 결과(Prediction)를 의미한다. 분자 부분인 $G \cup P - G \cap P$는 두 집합의 대칭 차집합(Symmetric Difference)으로, 정답과 예측이 일치하지 않는 '미스매치' 영역을 나타낸다. 분모 부분은 전체 합집합에서 교집합의 일부($\alpha * G \cap P$)를 제외한 부분 합집합(Partial Union)이다. $\alpha$는 부분 합집합 영역의 영향력을 조절하는 하이퍼파라미터이다.

### 적응적 $\alpha$ 조절 전략 (Adaptive Size Strategy)

타겟 객체의 크기에 따라 경계 영역이 차지하는 비중이 다르므로, $\alpha$ 값을 고정하지 않고 객체의 크기에 따라 적응적으로 계산한다.

$$\alpha = 1 - 2 \times \frac{C}{S}, \quad \alpha \in [0, 1)$$

여기서 $C$는 타겟의 경계 길이(Boundary Length)이고, $S$는 타겟의 전체 크기(Size)이다.

- **타겟이 큰 경우:** 내부 영역은 비교적 쉽게 분할되므로, $\alpha$ 값을 크게 설정하여 경계 영역에 더 집중하도록 유도한다.
- **타겟이 작은 경우:** 내부와 경계를 명확히 구분하기 어려우므로, $\alpha$ 값을 작게 설정하여 내부와 경계 영역 모두를 동시에 고려하도록 한다.

### Dice Loss와의 관계

본 논문은 Boundary DoU Loss를 다음과 같이 재작성하여 Dice Loss와의 수학적 유사성을 논의한다.

$$L_{DoU} = 1 - \frac{\alpha' * S_I}{S_D + \alpha' * S_I}$$

($S_D$는 차집합 영역의 넓이, $S_I$는 교집합 영역의 넓이, $\alpha' = 1 - \alpha$)

반면, Dice Loss는 다음과 같이 표현된다.

$$L_{Dice} = 1 - \frac{2 * S_I}{S_D + 2 * S_I}$$

두 함수 모두 $S_I$를 증가시키고 $S_D$를 감소시키는 방향으로 작동하지만, Boundary DoU Loss는 $\alpha < 1$ 임을 이용하여 Dice Loss보다 경계 영역의 오차 비율($S_D/S_I$)에 더 민감하게 반응하며 패널티를 부여한다.

## 📊 Results

### 실험 설정

- **데이터셋:** Synapse(복부 CT 스캔 30개), ACDC(심장 MRI 150명 데이터)
- **사용 모델:** UNet, TransUNet, Swin-UNet
- **비교 대상:** Dice Loss, CE Loss, Dice + CE, Tversky Loss, Boundary Loss
- **평가 지표:** Dice Similarity Coefficient (DSC $\uparrow$), Hausdorff Distance (HD $\downarrow$), Boundary IoU (B-IoU $\uparrow$)

### 정량적 결과

- **Synapse 데이터셋:** 모든 모델에서 제안된 Loss가 Dice Loss보다 우수한 성능을 보였다. 특히 UNet에서 DSC가 2.30% 향상되었으며, B-IoU에서 가장 높은 성능을 기록하여 경계 영역 분할 능력이 입증되었다.
- **ACDC 데이터셋:** DSC 수치에서 소폭의 향상이 있었으며, 특히 Boundary IoU 지표에서 다른 모든 손실 함수를 압도하는 결과를 보였다. 이는 제안 방법이 임상적으로 중요한 정밀한 경계 식별에 유리함을 시사한다.
- **객체 크기별 성능:** 타겟 크기를 Large($C/S < 0.2$)와 Small로 나누어 실험한 결과, 두 경우 모두에서 기존 방식보다 높은 DSC를 달성하여 적응적 $\alpha$ 전략의 유효성을 확인하였다.

### 정성적 결과

시각화 결과, 제안 방법은 위, 췌장과 같이 형태가 복잡하고 크기가 작은 장기들에 대해 더 정확한 국소화(Localization) 및 분할 성능을 보였다. 또한, ACDC 데이터셋의 RV(우심실) 영역처럼 형태 변화가 심한 부위에서도 과소 분할(Under-segmentation) 문제를 효과적으로 해결하고 완전성을 유지하는 모습이 관찰되었다.

## 🧠 Insights & Discussion

본 논문은 기존의 Boundary Loss들이 가졌던 학습 불안정성 문제를 해결하기 위해, 영역 기반의 단순한 비율 계산 방식을 도입함으로써 구현의 용이성과 학습 안정성을 동시에 확보하였다. 특히 주목할 점은 타겟의 크기에 따라 손실 함수의 집중도를 조절하는 $\alpha$ 전략이다. 이는 객체의 스케일이 다양한 의료 영상의 특성을 잘 반영한 설계라고 평가할 수 있다.

다만, 본 논문에서 $\alpha$를 계산하기 위해 사용된 $C$(경계 길이)와 $S$(크기)가 정확히 어떤 방식으로 계산되었는지(예: Ground Truth 기반인지, 학습 중 동적으로 계산되는지)에 대한 상세한 설명이 부족하다. 또한, 하이퍼파라미터 $\alpha$의 범위가 $[0, 1)$로 설정되었으나, 실제 데이터셋에서 $\alpha$ 값이 어떤 분포를 가지며 성능에 영향을 주었는지에 대한 분석이 추가되었다면 더 설득력이 있었을 것이다.

## 📌 TL;DR

이 논문은 의료 영상 분할에서 경계 영역의 정확도를 높이기 위해, 미분 가능한 영역 계산 기반의 **Boundary DoU Loss**와 객체 크기에 따라 가중치를 조절하는 **Adaptive Size Strategy**를 제안하였다. UNet, TransUNet, Swin-UNet 등 다양한 아키텍처에서 기존 Dice Loss 및 다른 경계 손실 함수보다 뛰어난 Boundary IoU 성능을 보였으며, 이는 특히 크기가 작거나 형태가 복잡한 장기의 정밀 분할에 기여할 가능성이 높다.
