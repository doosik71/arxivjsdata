# GELU Activation Function in Deep Learning: A Comprehensive Mathematical Analysis and Performance

Minhyeok Lee

## 🧩 Problem to Solve

심층 학습 모델의 효과는 활성화 함수(activation function)의 선택에 크게 좌우됩니다. 기존의 ReLU(Rectified Linear Unit)와 같은 활성화 함수는 "죽은 ReLU(dying ReLU)" 문제와 같은 한계를 가지고 있으며, 이에 대한 대안으로 GELU(Gaussian Error Linear Unit) 함수가 등장하여 다양한 최신 아키텍처에서 우수한 성능을 보여왔습니다. 그러나 GELU 활성화 함수와 정규화(normalization) 기법의 결합된 효과에 대한 포괄적인 수학적 이해는 여전히 부족하며, 이는 훈련 역학(training dynamics) 및 일반화 성능(generalization performance)에 미치는 영향을 명확히 하는 데 방해가 됩니다.

## ✨ Key Contributions

- **GELU의 수학적 특성 심층 분석:** GELU 활성화 함수의 미분 가능성(differentiability), 유계성(boundedness), 정상성(stationarity), 부드러움(smoothness) 등 핵심 수학적 특성을 엄밀하게 분석했습니다.
- **정규화 기법과의 시너지 효과 규명:** 정규화 기법(예: Batch Normalization, Layer Normalization, Group Normalization)과 결합 시 GELU의 상한 유계성(upper-boundedness)이 강화되어 기울기 소실/폭주(vanishing/exploding gradient) 문제 완화에 기여함을 수학적으로 증명했습니다.
- **광범위한 실험적 비교:** ResNet 아키텍처와 CIFAR-10, CIFAR-100, STL-10 데이터셋을 활용하여 GELU를 포함한 20가지 활성화 함수의 성능을 비교했습니다.
- **GELU의 우수성 입증:** 실험 결과 GELU가 모든 데이터셋에서 가장 낮은 테스트 손실과 가장 높은 테스트 정확도를 달성하며 다른 활성화 함수보다 뛰어난 성능을 보임을 입증했습니다.

## 📎 Related Works

이 연구는 딥러닝에서 활성화 함수의 중요성을 강조하며, 기존 연구에서 제안된 다양한 활성화 함수들을 참조합니다.

- **ReLU (Rectified Linear Unit) [16]:** 가장 널리 사용되지만, `dying ReLU` 문제를 겪을 수 있습니다.
- **GELU (Gaussian Error Linear Unit) [17]:** BERT [18], ViT [19], GPT [20]와 같은 최신 신경망 아키텍처에 성공적으로 통합된 활성화 함수입니다.
- **다른 활성화 함수 [14, 15]:** ELU, Hardshrink, Hardsigmoid, Hardtanh, Hardswish, LeakyReLU, LogSigmoid, PReLU, ReLU6, RReLU, SELU, CELU, Sigmoid, Softplus, Softshrink, Softsign, Tanh, Tanhshrink 등이 비교 대상에 포함되었습니다.
- **정규화 기법:** Batch Normalization (BN) [30], Layer Normalization (LN) [31], Group Normalization (GN) [32] 등 내부 공변량 변화(internal covariate shift)를 완화하는 기법들이 언급됩니다.
- **최적화 알고리즘:** Adam [25]과 같은 최적화 알고리즘 및 손실 함수(Cross-Entropy, MSE, MAE, Huber, Hinge, Triplet loss)에 대한 배경 지식이 제공됩니다.

## 🛠️ Methodology

본 연구는 GELU 활성화 함수에 대한 수학적 분석과 포괄적인 실험적 비교를 통해 진행되었습니다.

### 수학적 분석

GELU 활성화 함수의 핵심 특성들을 상세히 분석했습니다.

- **미분 가능성(Differentiability):**
  - GELU 함수는 가우시안 누적 분포 함수(CDF) $\Phi(x)$와 입력 $x$의 곱 $x \cdot \Phi(\alpha x)$ 형태로 정의될 수 있습니다. (여기서 $\alpha=1$)
  - 연쇄 법칙(chain rule)을 사용하여 GELU의 1차 미분을 계산했습니다.
    $$ \frac{dGELU(x)}{dx} = \frac{\alpha x}{\sqrt{2\pi}} e^{-\frac{(\alpha x)^2}{2}} + \Phi(\alpha x) $$
  - 특히, $x \approx -0.75$ 부근에서 미분값이 최소 약 -0.17을 가지는 지점이 존재하며, 이는 초기 학습 단계에서 모델이 지역 최솟값(local minima)을 탈출하는 데 도움이 될 수 있음을 시사합니다.
- **유계성(Boundness):**
  - $\lim_{x \to -\infty} GELU(x) = 0$ 이고 $\lim_{x \to \infty} GELU(x) = \infty$ 입니다.
  - 최솟값은 약 $x \approx -0.75$에서 약 -0.17로 하한(lower bound)이 존재하지만, 양의 방향으로는 비유계(unbounded)입니다.
  - **상한 유계성(Upper-boundness) (정규화와 결합 시):** 정규화 기법이 GELU 활성화 함수 전에 적용되면, 정규화된 입력 $z''_{i}$는 $|z''_{i}|_{\infty} \leq K$와 같은 유한한 범위 내에 존재합니다. GELU($x$) $\leq x$이므로, $GELU(z''_{i}) \leq z''_{i} \leq K$가 되어 활성화 값이 상한 $K$에 의해 유계됩니다. 이는 깊은 신경망에서 활성화 값의 발산(exploding)을 방지하는 데 중요합니다.
- **정상성(Stationarity):**
  - GELU는 모든 실수 $x$에 대해 연속(continuous)이며 미분 가능(differentiable)합니다.
  - **립시츠 연속성(Lipschitz Continuity):** GELU의 1차 미분 $|\frac{dGELU(x)}{dx}|$가 $L \approx 1.084$로 유계됨을 증명하여 립시츠 연속성을 입증했습니다. 이는 기울기 변화율에 상한이 있어 훈련 안정성에 기여합니다.
- **특징 공간의 부드러움(Smoothness of Feature Space):**
  - GELU는 2차 미분 가능하며, 이는 함수의 곡률(curvature)과 부드러움을 나타냅니다.
  - 립시츠 연속성은 Hölder 연속성 ($\alpha=1$)을 의미하며, 2차 미분 가능성은 $\alpha \in (1,2]$ 범위의 Hölder 연속성을 암시하여 특징 공간의 부드러움을 보여줍니다.

### 실험적 비교

- **네트워크 아키텍처:** BN과 비선형 활성화가 통합된 잔여 블록(residual block)으로 구성된 ResNet 기반의 사전 활성화(pre-activated) 잔여 컨볼루션 네트워크(residual convolutional network)를 사용했습니다 (총 14개 레이어).
- **데이터셋:** CIFAR-10, CIFAR-100, STL-10 데이터셋을 사용했습니다.
- **활성화 함수:** ELU, Hardshrink, Hardsigmoid, Hardtanh, Hardswish, LeakyReLU, LogSigmoid, PReLU, ReLU, ReLU6, RReLU, SELU, CELU, GELU, Sigmoid, Softplus, Softshrink, Softsign, Tanh, Tanhshrink 등 총 20가지 활성화 함수를 비교했습니다.
- **훈련 설정:** 교차 엔트로피 손실(cross-entropy loss), Adam optimizer, 20 에포크, 배치 크기 128, 학습률 0.001을 사용했습니다.

## 📊 Results

실험 결과는 GELU 활성화 함수의 우수한 성능을 일관되게 보여줍니다.

- **CIFAR-10 데이터셋:**
  - GELU: 테스트 손실 0.3685, 테스트 정확도 **89.52%** (가장 우수)
  - Hardswish: 88.77%, ReLU6: 88.70% (다음으로 우수)
  - Sigmoid: 3.2102, 33.90% (가장 낮은 성능)
- **CIFAR-100 데이터셋:**
  - GELU: 테스트 정확도 **64.71%** (가장 우수)
  - Hardswish: 64.12% (다음으로 우수)
- **STL-10 데이터셋:**
  - GELU: 테스트 정확도 **58.48%** (가장 우수)
  - LeakyReLU: 56.26% (다음으로 우수)

GELU는 모든 벤치마크 데이터셋에서 가장 높은 테스트 정확도를 달성하며 다른 활성화 함수들을 능가했습니다. 이는 GELU가 다양한 딥러닝 애플리케이션에 적합하다는 것을 입증합니다.

## 🧠 Insights & Discussion

- **GELU의 우수성 원인:** GELU의 부드러움(smoothness)과 미분 가능성(differentiability)은 기울기 기반 최적화 알고리즘에 필수적인 잘 작동하는 최적화 환경(well-behaved optimization landscape)을 제공합니다. 이는 특히 초기 학습 단계에서 지역 최솟값을 벗어나는 데 유리할 수 있습니다.
- **정규화와의 시너지:** 정규화 기법과의 결합은 GELU의 활성화 값을 효과적으로 제어하여, 양의 방향으로 비유계(unbounded)인 GELU의 특성에도 불구하고 전체 레이어 출력이 유계되도록 돕습니다. 이는 기울기 폭주 문제를 완화하고 훈련 안정성을 높이는 데 결정적인 역할을 합니다.
- **기존 활성화 함수의 한계 극복:** Sigmoid와 같이 기울기 소실(vanishing gradient) 문제가 있는 함수와 달리, GELU는 양의 영역에서 ReLU와 유사한 선형적 특성을 가지면서도 부드러움을 유지하여 학습 효율성을 높입니다.
- **실용적 함의:** 본 연구의 결과는 딥러닝 실무자들이 특정 목표와 제약 조건에 가장 적합한 활성화 함수를 선택하는 데 귀중한 통찰력을 제공합니다. GELU의 일관된 우수성은 광범위한 딥러닝 시나리오에서 기본 선택으로 고려될 수 있음을 시사합니다.

## 📌 TL;DR

이 논문은 딥러닝에서 GELU(Gaussian Error Linear Unit) 활성화 함수의 수학적 특성과 성능을 종합적으로 분석합니다. 미분 가능성, 유계성, 정상성, 부드러움을 엄밀하게 분석하고, 정규화 기법과의 시너지를 통해 GELU가 활성화 값의 상한 유계성을 보장하여 기울기 소실/폭주 문제를 완화함을 수학적으로 증명합니다. CIFAR-10, CIFAR-100, STL-10 데이터셋에서 ResNet을 사용한 광범위한 실험을 통해 GELU가 다른 19가지 활성화 함수보다 일관되게 우수한 성능을 보임을 입증하여, 딥러닝 애플리케이션에서 GELU의 강력한 적용 가능성을 확인했습니다.
