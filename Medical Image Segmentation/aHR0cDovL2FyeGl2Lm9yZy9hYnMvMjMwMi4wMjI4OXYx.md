# Selecting the Best Optimizers for Deep Learning based Medical Image Segmentation

Aliasghar Mortazi, Vedat Cicek, Elif Keles, Ulas Bagci (2023)

## 🧩 Problem to Solve

본 연구는 심장 영상 분할(Cardiac Image Segmentation) 분야에서 딥러닝 모델의 성능을 최적화하기 위한 가장 효과적인 Optimizer를 식별하고, 효율적인 최적화 전략을 설계하는 가이드를 제공하는 것을 목표로 한다.

심장 MRI 및 CT 영상에서 심장 구조를 정확하게 분할하는 것은 박출률(Ejection Fraction, EF) 측정과 같은 임상적 진단에 매우 중요하다. 하지만 딥러닝 모델을 학습시킬 때 사용하는 Optimizer의 선택에 따라 수렴 속도와 일반화 성능(Generalization performance)이 크게 달라지며, 특히 분할(Segmentation) 작업은 분류(Classification) 작업에 비해 출력 차원이 매우 높기 때문에 기존의 분류 작업 중심의 Optimizer 평가 결과를 그대로 적용하기에는 한계가 있다.

따라서 본 논문은 Adaptive learning rate 방법과 Accelerated schemes(Momentum) 간의 상호작용을 탐구하여, 의료 영상 분할 작업에 최적화된 새로운 최적화 방법을 제안하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 **CLMR(Cyclic Learning/Momentum Rate)**이라는 새로운 최적화 방법을 제안한 것이다.

핵심 아이디어는 학습률(Learning Rate, LR)뿐만 아니라 모멘텀률(Momentum Rate, MR) 또한 주기적으로 변화시키는 Cyclic 함수(삼각형 함수)를 적용하는 것이다. 저자들은 기존의 Adaptive Optimizer(예: Adam)가 학습 초기 수렴 속도는 빠르지만, 특정 지역 최솟값(Local minima)에 갇혀 일반화 성능이 떨어질 수 있다는 점에 주목하였다. 이를 해결하기 위해 Nesterov Accelerated Gradient(NAG) Optimizer를 기반으로 LR과 MR을 동시에 주기적으로 가변시킴으로써, 탐색 범위를 넓히고 더 나은 일반화 성능을 달성하고자 하였다.

## 📎 Related Works

논문에서는 기존의 Optimizer를 크게 두 가지 범주로 나누어 설명한다.

1. **Fixed LR/MR Optimizer**: SGD와 Momentum, NAG가 이에 해당한다. SGD는 단순하지만 학습률 설정에 매우 민감하며, Momentum은 이전 기울기의 가중치를 사용하여 수렴을 가속화한다. NAG는 기울기를 계산하기 전 파라미터를 미리 업데이트하는 방식을 통해 Momentum보다 더 나은 일반화 성능과 수렴 속도를 보인다.
2. **Adaptive LR/MR Optimizer**: AdaGrad, RMSProp, Adam 등이 있으며, 과거의 기울기 정보를 바탕으로 파라미터별 학습률을 동적으로 조정한다. 학습 속도가 매우 빠르다는 장점이 있으나, 최근 연구들에 따르면 고전적인 SGD 기반 방식보다 일반화 성능이 떨어질 수 있다는 한계가 지적되고 있다.

또한, 학습률을 주기적으로 변화시키는 **CLR(Cyclic Learning Rate)**이 제안되었으나, 이는 MR을 고정된 값으로 사용한다는 제약이 있다. 본 논문은 이 지점을 보완하여 LR과 MR을 모두 가변시키는 CLMR을 제안하며 기존 방식과의 차별점을 둔다.

## 🛠️ Methodology

### 전체 파이프라인 및 아키텍처

본 연구에서는 제안한 CLMR의 효과를 검증하기 위해 세 가지 주요 CNN 아키텍처를 사용하였다.

- **Encoder-Decoder**: 기본적인 인코더-디코더 구조이다.
- **U-Net**: 인코더와 디코더 사이에 Skip Connection을 추가하여 세밀한 정보를 복원한다.
- **DenseNet (Tiramisu)**: 모든 레이어를 서로 연결하는 Dense Block을 사용한다. 특히 **DenseNet2**는 $1 \times 1$ Convolution 레이어를 통해 채널 수를 조절하며 더 높은 Growth Rate(GR=24)를 적용하여 효율성을 높였다.

### CLMR Optimizer 설계

CLMR은 Nesterov Accelerated Gradient(NAG)를 기반으로 하며, 업데이트 식은 다음과 같다.
$$\theta_i = \theta_{i-1} - \alpha \nabla_{\theta_i} J(\theta_i - \beta(\theta_{i-1} - \theta_{i-2})) - \beta(\theta_{i-1} - \theta_{i-2})$$
여기서 $\theta$는 네트워크 파라미터, $\alpha$는 학습률(LR), $\beta$는 모멘텀률(MR), $J$는 비용 함수이다.

CLMR의 핵심은 $\alpha$와 $\beta$를 다음과 같은 삼각형 주기 함수로 정의하는 것이다.

- **주기 정의**: $\text{cycle}_{lr} = C_{lr} \times I_t$, $\text{cycle}_{mr} = C_{mr} \times I_t$ ($I_t$는 에포크당 반복 횟수)
- **LR ($\alpha$) 결정**: $\min_{lr}$과 $\max_{lr}$ 사이를 삼각형 형태로 왕복한다.
- **MR ($\beta$) 결정**: $\min_{mr}$과 $\max_{mr}$ 사이를 삼각형 형태로 왕복한다.

구체적인 수식은 반복 횟수 $i$에 따라 다음과 같이 정의된다.
$$\text{LR} = \begin{cases} 2 \times \frac{\max_{lr} - \min_{lr}}{C_{lr} \times I_t} \times i + \min_{lr} & \text{for } N \times \text{cycle}_{lr} \le i < \frac{2N+1}{2} \times \text{cycle}_{lr} \\ -2 \times \frac{\max_{lr} - \min_{lr}}{C_{lr} \times I_t} \times i + 2\max_{lr} - \min_{lr} & \text{for } \frac{2N+1}{2} \times \text{cycle}_{lr} \le i < (N+1) \times \text{cycle}_{lr} \end{cases}$$
MR 또한 동일한 형태의 함수 구조를 가진다. 저자들은 $\min/\max$ 값들을 고정한 상태에서 $C_{lr}$과 $C_{mr}$에 대한 2D 휴리스틱 탐색을 통해 최적의 주기 값을 찾았다.

### 학습 절차

- **데이터셋**: ACDC (MICCAI 2017) 데이터셋의 Cine-MRI 영상 150건 사용.
- **손실 함수**: Cross Entropy Loss를 사용하였다.
- **전처리**: B-spline 보간법을 통한 $200 \times 200$ 리사이징, Anisotropic filtering 및 Histogram matching 적용.

## 📊 Results

### 실험 설정

- **평가 지표**: Dice Index (DI) 및 Cross Entropy (CE) Loss.
- **비교 대상**: Adam, Nesterov, CLR, CLMR.
- **작업**: 심장의 세 가지 구조(RV, Myo, LV)에 대한 단일 및 다중 객체 분할.

### 주요 결과

1. **정량적 결과**: 테스트 세트 평가 결과, CLMR이 다른 Optimizer들에 비해 전반적으로 더 높은 Dice Index를 기록하였다. 특히 U-Net 아키텍처에서 CLMR은 CLR 대비 약 2% 이상의 DI 향상을 보였다.
2. **수렴 특성**: Adam은 학습 초기 매우 빠르게 수렴하지만, 이후 성능이 정체되는 경향을 보였다. 반면 CLMR은 초기 수렴 속도는 Adam보다 느리지만, 최종적으로는 더 높은 정확도와 낮은 Loss에 도달하여 우수한 일반화 능력을 입증하였다.
3. **아키텍처 영향**: DenseNet2(GR=24)가 가장 좋은 성능을 보였으며, 이는 단순한 파라미터 수의 증가보다 Dense Connection의 Growth Rate를 높이는 것이 성능 향상에 더 효과적임을 시사한다.
4. **정성적 결과**: 정성적 분석에서 CLMR과 DenseNet2의 조합이 특히 심장 정지기(End-Systole, ES)와 같이 분할이 어려운 상황에서도 Ground-truth와 가장 유사한 결과를 생성하였다.

## 🧠 Insights & Discussion

### 강점 및 해석

본 논문은 Adaptive Optimizer의 빠른 수렴 속도라는 장점과 SGD 기반 방식의 뛰어난 일반화 성능이라는 장점을 주기적 가변(Cyclic) 전략을 통해 결합하였다. 특히 LR뿐만 아니라 MR을 함께 주기적으로 변화시키는 것이 의료 영상 분할과 같은 고차원 출력 작업에서 최적의 해를 찾는 데 결정적인 역할을 한다는 것을 보여주었다.

### 한계 및 비판적 논의

- **파라미터 설정의 어려움**: $\min/\max$ 값과 주기 $C$ 값을 결정하는 과정이 여전히 휴리스틱한 탐색에 의존하고 있다. 저자들도 이를 언급하며 향후 강화학습(Policy Gradient) 등을 통해 이러한 하이퍼파라미터를 자동으로 학습하는 방법이 필요함을 제안하였다.
- **데이터 일반성**: 본 실험은 Cine-MRI라는 특정 모달리티와 ACDC 데이터셋에 국한되어 수행되었다. 영상의 노이즈 수준이나 클래스 불균형이 심한 다른 데이터셋에서도 동일한 경향이 나타날지는 추가 검증이 필요하다.
- **계산 비용**: Adaptive 방식보다 계산 비용이 낮다고 주장하지만, 최적의 $C_{lr}, C_{mr}$을 찾기 위한 사전 탐색 비용에 대해서는 구체적으로 언급되지 않았다.

## 📌 TL;DR

본 연구는 의료 영상 분할 성능을 높이기 위해 학습률(LR)과 모멘텀률(MR)을 동시에 주기적으로 변화시키는 **CLMR(Cyclic Learning/Momentum Rate)** Optimizer를 제안하였다. 실험 결과, CLMR은 Adam과 같은 Adaptive Optimizer보다 수렴 속도는 느리지만, 최종적인 일반화 성능과 Dice Index에서 우위를 점하였다. 특히 DenseNet2 아키텍처와 결합했을 때 가장 뛰어난 성능을 보였으며, 이는 향후 의료 영상 분석 모델의 최적화 전략 수립에 중요한 가이드를 제공한다.
