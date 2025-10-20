# Continuously Differentiable Exponential Linear Units

Jonathan T. Barron

## 🧩 Problem to Solve

기존의 Exponential Linear Units (ELU) 활성화 함수는 심층 학습 아키텍처에서 유용한 정류기(rectifier)이지만, 형상 매개변수 $\alpha$가 1이 아닐 때 입력 $x$에 대해 연속적으로 미분 가능하지 않습니다. 또한, 음수 $x$ 값에 대해 $\alpha$ 값이 커지면 기울기가 폭발적으로 증가(exploding gradient)하여 학습을 어렵게 만들 수 있습니다. 이로 인해 ELU 함수를 이해하고 $\alpha$를 조정하기가 복잡해집니다.

## ✨ Key Contributions

- **CELU(Continuously Differentiable ELU) 제안:** 모든 $\alpha$ 값에 대해 $C^1$ 연속인 ELU의 대안적 매개변수화를 제시합니다.
- **유계(Bounded) 미분:** 입력 $x$에 대한 미분값이 유계입니다. 이는 기울기 폭발 문제를 완화합니다.
- **특수 사례 포함:** 선형 전달 함수(linear transfer function)와 ReLU를 특수 사례로 포함합니다.
- **스케일-유사성(Scale-similarity):** $\alpha$에 대해 스케일-유사(scale-similar)한 속성을 가집니다.
- $\alpha$ 조정 용이성: ELU를 더 쉽게 추론하고 $\alpha$를 더 쉽게 조정할 수 있도록 합니다.

## 📎 Related Works

- Clevert, D., Unterthiner, T., & Hochreiter, S. (2015). "Fast and accurate deep network learning by exponential linear units (elus)." CoRR, abs/1511.07289. (이 논문은 원본 ELU 활성화 함수를 제안했습니다.)

## 🛠️ Methodology

저자들은 기존 ELU의 문제를 해결하기 위해 "CELU"라는 새로운 매개변수화를 제안했습니다.

1. **기존 ELU 함수 및 미분:**
   $$ ELU(x,\alpha) = \begin{cases} x & \text{if } x \ge 0 \\ \alpha(\exp(x)-1) & \text{otherwise} \end{cases} $$
    $$ \frac{d}{dx} ELU(x,\alpha) = \begin{cases} 1 & \text{if } x \ge 0 \\ \alpha\exp(x) & \text{otherwise} \end{cases} $$
    $\alpha \ne 1$일 때 $x=0$에서 미분값이 불연속임을 지적합니다.

2. **CELU 함수 정의:**
   음수 값에 대한 활성화 함수를 수정하여 $x=0$에서 모든 $\alpha$ 값에 대해 미분값이 1이 되도록 합니다.
   $$ CELU(x,\alpha) = \begin{cases} x & \text{if } x \ge 0 \\ \alpha\left(\exp\left(\frac{x}{\alpha}\right)-1\right) & \text{otherwise} \end{cases} $$

3. **CELU 미분:**
   $$ \frac{d}{dx} CELU(x,\alpha) = \begin{cases} 1 & \text{if } x \ge 0 \\ \exp\left(\frac{x}{\alpha}\right) & \text{otherwise} \end{cases} $$
    이 수정으로 $x=0$에서 미분값이 1로 연속성을 유지합니다.

4. **효율적인 계산:** ELU와 유사하게 $\exp(x/\alpha)$를 미리 계산하여 활성화 함수와 미분 계산에 효율적으로 사용할 수 있습니다.

## 📊 Results

이 논문은 주로 제안된 CELU 활성화 함수의 수학적 특성과 장점을 설명하며, 광범위한 실험적 결과는 포함하지 않습니다. 주요 결과는 다음과 같습니다.

- **연속 미분 가능성:** CELU의 미분은 모든 $\alpha$ 값에 대해 $x=0$에서 연속적입니다. (그림 1d에서 시각적으로 확인 가능)
- **유계 미분:** CELU의 미분($\exp(x/\alpha)$)은 항상 1보다 작거나 같으므로 유계(bounded)입니다.
- **스케일-유사성:** $CELU(x,\alpha) = \frac{1}{c} CELU(cx,c\alpha)$ 관계가 성립합니다.
- **특수 사례:**
  - $\alpha \to 0^+$일 때, CELU($x,\alpha$)는 ReLU($\max(0,x)$)로 수렴합니다.
  - $\alpha \to \infty$일 때, CELU($x,\alpha$)는 선형 함수($x$)로 수렴합니다.

## 🧠 Insights & Discussion

CELU는 기존 ELU의 핵심 이점(사라지는 기울기 없음, 평균 활성화 값이 0에 가까움)을 유지하면서 주요 단점인 미분 불연속성을 해결했습니다. 이는 ELU의 속성을 분석하고 튜닝하는 것을 더 쉽게 만듭니다.

- **학습 안정성 향상:** 미분값이 유계이므로 "기울기 폭발" 문제를 방지하여 학습 안정성을 높일 수 있습니다.
- **모델 유연성:** $\alpha$ 값을 조절함으로써 CELU는 ReLU와 선형 함수 사이를 보간할 수 있는 강력한 해석을 제공합니다. 이는 신경망 설계에서 더 큰 유연성을 제공합니다.
- **쉬운 튜닝:** 연속적인 미분 가능성과 스케일-유사성 덕분에 $\alpha$ 매개변수를 더 직관적이고 효과적으로 튜닝할 수 있습니다.
- 논문은 실제 딥러닝 모델에서 CELU의 성능 향상에 대한 직접적인 벤치마크 결과를 제시하지는 않지만, 이러한 수학적, 이론적 이점들이 실제 학습 과정에서 더 나은 안정성과 성능으로 이어질 것임을 시사합니다.

## 📌 TL;DR

기존 ELU 활성화 함수는 $\alpha \ne 1$일 때 미분 불연속성 및 폭발하는 기울기 문제를 가집니다. 본 논문은 이를 해결하기 위해 **CELU(Continuously Differentiable ELU)**를 제안합니다. CELU는 음수 $x$에 대한 정의를 수정하여 모든 $\alpha$에 대해 $C^1$ 연속성을 보장하며, 미분값이 유계입니다. 또한, CELU는 스케일-유사성을 가지며, $\alpha$ 값에 따라 ReLU와 선형 함수 사이를 보간하여 모델의 유연성과 학습 안정성을 향상합니다.
