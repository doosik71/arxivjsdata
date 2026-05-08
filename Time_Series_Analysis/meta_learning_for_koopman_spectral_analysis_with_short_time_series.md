# Meta-Learning for Koopman Spectral Analysis with Short Time-series

Tomoharu Iwata, Yoshinobu Kawahara (2021)

## 🧩 Problem to Solve

본 논문은 비선형 동적 시스템(nonlinear dynamical systems)을 분석하기 위한 Koopman spectral analysis의 데이터 효율성 문제를 해결하고자 한다. Koopman 연산자 이론의 핵심은 비선형 시스템의 상태를 적절한 비선형 함수를 통해 Koopman 공간이라는 선형 공간으로 임베딩(embedding)하여, 복잡한 비선형 역학을 선형 체계로 분석하는 것이다.

이 과정에서 가장 중요한 점은 데이터를 선형 공간으로 보내는 적절한 임베딩 함수를 찾는 것인데, 기존의 신경망 기반 방법들은 이러한 함수를 학습하기 위해 매우 긴 시계열 데이터(long time-series)를 필요로 한다. 이는 짧은 시계열 데이터만 얻을 수 있는 실제 응용 분야에서 Koopman spectral analysis를 적용하는 것을 불가능하게 만든다. 따라서 본 논문의 목표는 관련이 있지만 서로 다른 여러 시계열 데이터로부터 학습된 지식을 활용하여, 처음 보는 짧은 시계열 데이터(short time-series)만으로도 효율적으로 임베딩 함수를 추정하는 메타 학습(meta-learning) 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **시계열 표현(time-series representation)**을 통해 각 태스크(시계열)의 고유한 특성을 추출하고, 이를 임베딩 함수와 매핑 함수의 입력으로 사용하여 신경망이 각 시계열에 맞게 동적으로 적응하도록 하는 것이다.

구체적으로, Bidirectional LSTM을 통해 짧은 시계열의 전역적인 특성을 담은 벡터 $r_S$를 생성하고, 이를 Feed-forward 신경망에 함께 입력함으로써 동일한 신경망 구조를 공유하면서도 각 시계열의 특성에 맞는 개별적인 임베딩 함수를 구현한다. 이를 통해 새로운 짧은 시계열이 주어졌을 때 신경망을 재학습(retraining)하지 않고도 즉각적으로 Koopman 공간으로의 임베딩과 미래 예측이 가능하도록 설계하였다.

## 📎 Related Works

기존의 Koopman spectral analysis의 대표적인 알고리즘인 DMD(Dynamic Mode Decomposition)는 데이터가 이미 Koopman 공간에 있다고 가정하므로, 사용자가 수동으로 임베딩 함수를 정의해야 한다는 한계가 있다. 이를 자동화하기 위해 기저 함수(basis functions)나 커널(kernels)을 사용하는 방법, 혹은 신경망을 이용해 임베딩 함수를 학습하는 NDMD(Neural network-based DMD) 등이 제안되었다. 그러나 NDMD를 포함한 신경망 기반 방식들은 학습을 위해 방대한 양의 시계열 데이터가 필요하다는 치명적인 단점이 있다.

전이 학습(transfer learning)이나 메타 학습(meta-learning) 같은 기법들이 다른 도메인의 데이터를 활용해 성능을 높이는 시도가 있었으나, 이는 주로 회귀나 분류, 일반적인 시계열 예측에 국한되었으며 Koopman spectral analysis에 적용된 사례는 없었다. 본 논문은 특히 Neural Processes와 유사한 인코더-디코더 스타일의 메타 학습 구조를 Koopman 임베딩 함수 추정에 도입했다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 전체 파이프라인

제안된 모델은 짧은 시계열 $Y_S$가 입력되었을 때, 이를 Koopman 공간으로 투영하고 선형 연산자인 Koopman 행렬을 추정하여 미래의 값을 예측하는 구조를 가진다. 전체 과정은 다음과 같은 5단계로 진행된다.

**1. 시계열 표현 추출 (Time-series Representation)**
입력된 짧은 시계열 $Y_S = [y_{S1}, \dots, y_{ST}]$에 대해 Bidirectional LSTM을 적용한다. 순방향(forward) 및 역방향(backward) 은닉 상태 $r_{S t}^F, r_{S t}^B$를 계산한 후, 이를 전체 시간 단계 $T$에 대해 평균 내어 해당 시계열의 전역적 특성을 대표하는 벡터 $r_S \in \mathbb{R}^{2K}$를 생성한다.
$$r_S = \frac{1}{T} \sum_{t=1}^T [r_{S t}^F, r_{S t}^B]$$

**2. Koopman 임베딩 (Koopman Embedding)**
각 측정 벡터 $y_{St}$와 위에서 구한 시계열 표현 $r_S$를 신경망 $\phi$의 입력으로 넣어 Koopman 공간의 좌표 $g_{St}$를 얻는다.
$$g_{St} = \phi([y_{St}, r_S])$$
이 구조를 통해 신경망 $\phi$는 $r_S$에 따라 서로 다른 임베딩 함수로 동작하게 된다.

**3. Koopman 행렬 추정 (Koopman Matrix Estimation)**
임베딩된 벡터들을 이용하여 $G_{S1} = [g_{S1}, \dots, g_{S,T-1}]$과 $G_{S2} = [g_{S2}, \dots, g_{ST}]$ 행렬을 구성한다. Koopman 행렬 $\hat{K}_S$는 최소자승법(least squares)을 통해 닫힌 형태(closed form)로 추정된다.
$$\hat{K}_S = \arg \min_{K_S} \|G_{S2} - K_S G_{S1}\|^2 = G_{S2} G_{S1}^\dagger$$
여기서 $\dagger$는 의사 역행렬(pseudo-inverse)을 의미하며, 이는 미분 가능하므로 신경망 학습 시 역전파가 가능하다.

**4. 미래 임베딩 예측 (Future Embedding Prediction)**
추정된 $\hat{K}_S$를 마지막 상태 $g_{ST}$에 $\tau-T$번 곱하여 미래 시점 $\tau$의 임베딩 $\hat{g}_{S\tau}$를 예측한다.
$$\hat{g}_{S\tau} = \hat{K}_S^{\tau-T} g_{ST}$$

**5. 측정 공간으로의 복원 (Mapping to Measurement Space)**
예측된 임베딩 $\hat{g}_{S\tau}$와 시계열 표현 $r_S$를 신경망 $\psi$에 입력하여 최종적인 미래 측정값 $\hat{y}_{S\tau}$를 얻는다.
$$\hat{y}_{S\tau} = \psi([\hat{g}_{S\tau}, r_S])$$

### 학습 절차 및 손실 함수

본 모델은 **에피소드 학습(episodic training)** 프레임워크를 사용하여 테스트 시의 예측 오차를 최소화하도록 학습된다. 학습 데이터셋 $D$에서 무작위로 시계열 $Y_d$를 샘플링하고, 이를 다시 짧은 서포트 시계열(support time-series, $Y_S$)과 그 뒤를 잇는 쿼리 시계열(query time-series, $Y_Q$)로 나눈다.

손실 함수는 쿼리 시계열의 실제 값과 예측 값 사이의 평균 제곱 오차(MSE)로 정의된다.
$$L(Y_Q, Y_S) = \frac{1}{T_Q} \sum_{\tau=1}^{T_Q} \|\hat{y}_{Q,\tau}(Y_S) - y_{Q,\tau}\|^2$$
최종 목표는 모든 학습 시계열에 대한 기대 테스트 예측 오차를 최소화하는 파라미터 $\Theta$를 찾는 것이다.

## 📊 Results

### 실험 설정

- **데이터셋**: Synthetic(합성 데이터), Van-der-Pol(비보존 진동자), Lorenz(카오스 시스템), Cylinder-wake(유체 역학)의 4가지 데이터를 사용하였다.
- **비교 대상**: DMD, NDMD, NDMD의 미세 조정(Finetune), MAML(Model-Agnostic Meta-Learning)을 적용한 NDMD.
- **지표**: Koopman 연산자의 고유값(eigenvalue) 추정 오차 및 미래 값 예측의 RMSE.

### 주요 결과

1. **고유값 추정 성능**: Synthetic 데이터에 대해 제안 방법이 가장 낮은 고유값 추정 오차를 기록하였다(Table 1). 이는 짧은 시계열만으로도 Koopman 연산자의 특성을 정확히 파악했음을 의미한다.
2. **미래 예측 성능**: 모든 데이터셋에서 제안 방법이 가장 낮은 예측 오차를 보였다(Table 2). 특히 NDMD는 공통 함수만 학습하여 개별 시계열 특성을 반영하지 못해 성능이 낮았으며, Finetune은 짧은 데이터로 인한 과적합(overfitting) 문제로 인해 성능이 저하되었다.
3. **MAML과의 비교**: MAML은 초기 파라미터를 공유하여 빠른 적응을 돕지만, 동적 시스템의 특성이 매우 다양할 경우 단일 초기값만으로는 한계가 있다. 반면 제안 방법은 $r_S$를 통해 함수 자체를 유연하게 변경하므로 더 우수한 성능을 보였다.
4. **Ablation Study**:
   - 학습 오차를 최소화한 `OursT`보다 테스트 오차를 최소화한 제안 방법이 더 우수했다.
   - 시계열 표현 $r_S$를 제거한 `OursN`의 성능이 크게 떨어져, $r_S$가 태스크 적응에 핵심적인 역할을 함이 증명되었다.

## 🧠 Insights & Discussion

본 연구는 메타 학습을 Koopman spectral analysis에 도입하여, 데이터 부족 문제라는 고질적인 한계를 극복하였다. 특히 **시계열 표현($r_S$)**이라는 매개체를 통해 신경망이 각 시스템의 역학적 특성을 인지하고, 그에 맞는 임베딩 함수를 즉각적으로 생성하게 만든 설계가 매우 효과적이었다.

**강점**:

- 새로운 시계열이 들어왔을 때 재학습 없이 즉시 추론이 가능하다(Test computational time이 매우 짧음).
- 단순한 파라미터 초기화 공유(MAML)보다 훨씬 더 유연한 적응 능력을 보여주었다.

**한계 및 논의**:

- 모델이 Bi-LSTM에 의존하고 있어, 시계열의 길이가 너무 짧을 경우 $r_S$의 추정 정확도가 떨어져 성능이 저하될 수 있다(Figure 6).
- 현재는 미래 예측에 집중하고 있으나, 이를 제어(control) 문제로 확장하는 연구가 필요하다.
- LSTM 외에 Attention 메커니즘 등을 활용하여 시계열 특성을 더 정교하게 추출할 가능성이 남아 있다.

## 📌 TL;DR

이 논문은 짧은 시계열 데이터만으로 비선형 동적 시스템을 선형 공간으로 투영하는 **Koopman 임베딩 함수를 추정하는 메타 학습 방법**을 제안한다. Bi-LSTM으로 추출한 시계열 표현을 신경망에 입력함으로써 각 시스템의 특성에 맞는 임베딩을 동적으로 생성하며, 에피소드 학습을 통해 일반화 성능을 높였다. 실험 결과, 기존의 NDMD나 MAML 기반 방식보다 고유값 추정과 미래 예측에서 월등한 성능을 보였으며, 이는 데이터가 제한적인 환경에서의 비선형 시스템 분석 및 제어 연구에 중요한 기여를 할 것으로 보인다.
