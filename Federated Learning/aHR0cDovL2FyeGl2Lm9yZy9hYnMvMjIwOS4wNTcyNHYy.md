# Concealing Sensitive Samples against Gradient Leakage in Federated Learning

Jing Wu, Munawar Hayat, Mingyi Zhou, Mehrtash Harandi (2024)

## 🧩 Problem to Solve

본 논문은 연합 학습(Federated Learning, FL) 환경에서 발생하는 Gradient Leakage 문제, 특히 모델 인버전 공격(Model Inversion Attacks)으로 인해 클라이언트의 민감한 데이터가 복원되는 보안 취약점을 해결하고자 한다. 연합 학습은 raw data를 서버로 전송하지 않고 모델 업데이트 값(gradient)만 공유함으로써 프라이버시를 보호한다고 알려져 있으나, 최근 연구들은 공격자가 공유된 gradient 정보를 가로채어 원래의 입력 데이터를 재구성할 수 있음을 보여주었다.

특히 저자들은 이러한 공격이 성공하는 핵심 요인이 확률적 최적화(Stochastic Optimization) 과정에서 배치 내 개별 데이터 간의 gradient 얽힘(entanglement) 정도가 낮기 때문이라는 가설을 세운다. 이로 인해 공격자는 개별 샘플의 정보를 분리하여 추출할 수 있게 된다. 본 논문의 목표는 모델의 성능 저하를 최소화하면서도, 공격자가 민감한 데이터를 재구성하지 못하도록 gradient 수준에서 데이터를 은폐하는 효과적인 방어 전략을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 민감한 데이터($x_s$)를 직접 보호하는 대신, gradient 수준에서는 민감한 데이터와 매우 유사하지만 시각적으로는 전혀 다른 '은폐 샘플(Concealed Samples, $\tilde{x}_c$)'을 합성하여 삽입하는 것이다.

공격자가 gradient를 통해 데이터를 복원하려 할 때, 실제 민감한 데이터의 gradient와 합성된 은폐 샘플의 gradient가 서로 매우 유사하게 설계되어 있어, 공격자는 어떤 데이터가 실제 민감한 데이터인지 구분할 수 없게 된다. 결과적으로 공격자가 복원한 결과물은 실제 데이터가 아닌 시각적으로 무관한 은폐 샘플이 될 가능성이 높아지며, 이를 통해 민감한 정보의 유출을 효과적으로 차단한다.

## 📎 Related Works

### 관련 연구 및 한계

기존의 모델 인버전 공격은 크게 두 가지 방향으로 발전해 왔다. 첫째는 Deep Leakage from Gradients (DLG)와 같이 최적화 기반으로 dummy 데이터를 조정하여 실제 gradient와 일치시키는 방식이며, 둘째는 Imprint 공격과 같이 서버가 모델 구조를 수정하여 특정 데이터의 유출을 가속화하는 방식이다.

이에 대응하는 기존 방어 기법들은 다음과 같이 분류된다:

1. **Gradient Compression 및 Perturbation**: Gradient에 노이즈를 추가하는 DP-Gaussian이나 특정 레이어의 gradient를 제거하는 Soteria 등이 있다.
2. **Data Encryption 및 Transformation**: 데이터에 변형을 가하는 ATS나 InstaHide 등이 있다.
3. **Architectural Modifications**: 네트워크에 병목(bottleneck) 구조를 추가하는 PRECODE 등이 있다.

### 차별점

기존 방어 기법들은 모든 데이터를 동일하게 보호하려 하여 프라이버시와 모델 성능 간의 트레이드오프(trade-off)가 심하거나, 최근의 적응형 공격(Adaptive Attacks)에 쉽게 무너지는 한계가 있다. 반면, 본 논문의 DCS$^2$ 방식은 보호해야 할 '민감한 샘플'에 집중하여 적응적으로 은폐 샘플을 생성하며, Gradient Projection 기법을 통해 모델의 학습 성능 손실을 방지한다는 점에서 기존 연구와 차별화된다.

## 🛠️ Methodology

### 전체 파이프라인

본 논문에서 제안하는 DCS$^2$ (Defense by Concealing Sensitive Samples)의 전체 과정은 크게 두 단계로 구성된다: 1) 민감한 데이터를 모방하는 은폐 샘플의 합성, 2) 합성된 샘플로 인해 변형된 gradient를 원래 방향으로 정렬하는 Gradient Projection이다.

### 1. 은폐 샘플 합성 (Synthesizing Concealed Samples)

민감한 샘플 $x_s$에 대응하는 은폐 샘플 $\tilde{x}_c$를 생성하기 위해 다음과 같은 세 가지 목표를 설정한다.

- **목표 1**: $\tilde{x}_c$와 $x_s$의 시각적 차이($\| \tilde{x}_c - x_s \|$)는 최대화하되, gradient 수준의 유사도(Cosine Similarity)는 최대화한다.
- **목표 2**: 잠재 표현(latent representation)의 거리 $\| f_\theta(\tilde{x}_c) - f_\theta(x_s) \|$가 임계값 $\epsilon$ 이하가 되도록 하여 안정성을 유지한다.

이를 위해 다음과 같은 목적 함수 $L_{obj}$를 최소화하는 $\tilde{x}_c$를 학습한다:

$$L_{obj} = \left( 1 - \frac{\langle \nabla_\theta L(f_\theta(\tilde{x}_c), \tilde{y}_c), \nabla_\theta L(f_\theta(x_s), y_s) \rangle}{\| \nabla_\theta L(f_\theta(\tilde{x}_c), \tilde{y}_c) \| \times \| \nabla_\theta L(f_\theta(x_s), y_s) \| } \right) + e^{-\lambda_x \| \tilde{x}_c - x_s \|} + \lambda_z \left( \frac{\| f_\theta(\tilde{x}_c) - f_\theta(x_s) \|}{\| f_\theta(x_s) \|} - \epsilon \right)$$

여기서 첫 번째 항은 gradient 유사도를, 두 번째 항은 시각적 거리 극대화를, 세 번째 항은 잠재 공간에서의 거리 제한을 의미한다.

### 2. Gradient Projection

은폐 샘플이 추가되면 전체 배치의 gradient가 변하여 모델 성능이 하락할 수 있다. 이를 방지하기 위해, 은폐 샘플이 포함된 새로운 gradient $g_c$를 원래의 gradient $g$와 정렬시킨다.

먼저, Mixup 개념을 도입하여 다음과 같이 새로운 gradient $g_c$를 계산한다:
$$g_c \triangleq \nabla_\theta [ L(f_\theta(x_s), y_s) + \lambda_g L(f_\theta(\tilde{x}_c), \tilde{y}_c) + (1 - \lambda_g) L(f_\theta(\tilde{x}_c), y_s) ]$$

이후, $\langle g, g_c \rangle \ge 0$ 조건을 만족하도록 하여 $g_c$를 $g$ 방향으로 투영(projection)한다. 구체적으로는 다음과 같은 Quadratic Programming (QP) 문제를 풀어 최적의 투영된 gradient $\hat{g}_c$를 구한다:

$$\arg \min_{v} \frac{1}{2} g^\top g v + g^\top c g v, \quad \text{s.t. } v \ge 0$$

최종적으로 서버에 전송되는 gradient는 $\hat{g}_c = g v^* + g_c$가 된다.

## 📊 Results

### 실험 설정

- **데이터셋**: MNIST, CIFAR10, CelebA, TinyImageNet.
- **공격 모델**: GS attack (최적화 기반), GGL attack (GAN 기반), Imprint attack (모델 수정 기반).
- **비교 대상**: DP-Gaussian, Prune (Gradient Compression), Soteria.
- **평가 지표**:
  - 재구성 품질: PSNR, SSIM (낮을수록 방어 성능 우수), LPIPS (높을수록 방어 성능 우수).
  - 모델 성능: Classification Accuracy.

### 주요 결과

1. **프라이버시 보호 성능**: MNIST와 CIFAR10 실험에서 DCS$^2$는 PSNR과 SSIM을 기존 방어 기법보다 더 낮게 유지하여, 공격자가 원본 데이터를 복원하는 것을 효과적으로 차단하였다. 특히 TinyImageNet의 Imprint 공격에 대해 Soteria는 무력했으나, DCS$^2$는 LPIPS를 0.00에서 0.22로 높이며 유의미한 방어력을 보였다.
2. **모델 성능 유지**: 대부분의 방어 기법이 성능 저하를 야기하는 것과 달리, DCS$^2$는 원래 모델의 정확도를 거의 그대로 유지하거나, CelebA 데이터셋의 경우 오히려 약간의 성능 향상을 보이기도 하였다.
3. **적응형 공격에 대한 강건성**: 공격자가 각 클래스의 평균 이미지(AvgImg)를 prior로 사용하는 강력한 적응형 공격 상황에서도 DCS$^2$는 민감한 데이터를 효과적으로 보호하였다.
4. **초기값 영향**: 은폐 샘플 생성 시 랜덤 노이즈나 타 도메인(CIFAR10 $\rightarrow$ CelebA)의 이미지에서 시작하더라도 일관된 방어 성능을 보였다.

## 🧠 Insights & Discussion

### 강점 및 성과

본 연구는 단순히 노이즈를 추가하거나 정보를 삭제하는 기존의 수동적 방어에서 벗어나, '유사하지만 다른' 샘플을 생성하여 공격자를 기만하는 능동적 방어 전략을 제시하였다. 특히 Gradient Projection을 통해 보안 강화로 인한 유틸리티(정확도) 손실 문제를 수학적으로 해결하여 실용성을 높였다.

### 한계 및 비판적 해석

- **계산 복잡도**: 은폐 샘플을 생성하기 위해 추가적인 최적화 과정이 필요하며, 이는 계산 비용 증가로 이어진다. 논문에 따르면 NVIDIA RTX A100 GPU 기준, 방어 적용 시 업데이트 시간이 102초에서 172초로 약 68% 증가한다.
- **가정의 제약**: 본 방법은 보호해야 할 '민감한 샘플'이 미리 정의되어 있다는 가정을 전제로 한다. 모든 데이터를 보호해야 하는 상황에서는 계산 비용이 기하급수적으로 증가할 수 있다.
- **잠재적 취약점**: 저자들도 언급했듯이, 공격자가 은폐 샘플 자체를 복원하고 필터링하는 메커니즘을 설계한다면 다시 취약해질 가능성이 있다.

## 📌 TL;DR

본 논문은 연합 학습에서 gradient 정보를 통해 민감한 데이터를 복원하는 모델 인버전 공격을 막기 위해, **민감한 데이터와 gradient는 비슷하지만 시각적으로는 전혀 다른 '은폐 샘플'을 합성하여 삽입하는 DCS$^2$ 기법을 제안**한다. 이 방법은 공격자가 복원 결과물에서 실제 데이터와 은폐 샘플을 구분하지 못하게 만들어 프라이버시를 보호하며, Gradient Projection을 통해 모델 성능 저하를 방지한다. 실험 결과, 기존 SOTA 방어 기법들보다 강력한 보호 성능을 보이면서도 모델의 정확도를 유지함을 입증하였으며, 이는 향후 고도의 프라이버시가 요구되는 의료 및 금융 분야의 연합 학습 시스템에 적용될 가능성이 높다.
