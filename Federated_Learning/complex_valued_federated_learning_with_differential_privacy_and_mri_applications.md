# Complex-valued Federated Learning with Differential Privacy and MRI Applications

Anneliese Riess, Alexander Ziller, Stefan Kolek, Daniel Rueckert, Julia Schnabel, and Georgios Kaissis (2024)

## 🧩 Problem to Solve

본 논문은 의료 데이터, 특히 자기공명영상(MRI)과 같이 복소수(Complex-valued, CV) 신호 처리 기술이 필수적인 도메인에서 데이터 프라이버시를 보호하며 모델을 학습시키는 방법을 다룬다.

MRI의 k-space 데이터나 심전도 등의 생체 신호는 본질적으로 복소수 형태로 표현된다. 이러한 민감한 의료 데이터를 처리하기 위해 연합 학습(Federated Learning, FL)이 제안되었으나, FL만으로는 데이터 재구성 공격(Data Reconstruction Attacks)과 같은 프라이버시 침해 위험을 완전히 제거할 수 없다. 이를 해결하기 위해 수학적으로 엄격한 프라이버시 보장을 제공하는 차분 프라이버시(Differential Privacy, DP)를 결합하는 것이 필요하다. 하지만 기존의 DP 연구는 대부분 실수(Real-valued) 데이터에 집중되어 있으며, 복소수 데이터 및 복소수 신경망(Complex-valued Neural Networks, CVNNs)에 DP를 어떻게 적용하고 그 프라이버시 비용을 어떻게 계산할 것인지에 대한 이론적, 방법론적 연구가 부족한 상태이다. 따라서 본 논문의 목표는 복소수 영역에서의 DP 메커니즘을 이론적으로 정립하고, 이를 CVNN에 적용하여 프라이버시가 보장되는 연합 학습 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 복소수 데이터에 최적화된 DP 프레임워크를 제안하여 프라이버시와 유틸리티(정확도) 사이의 균형을 맞춘 점에 있다.

1. **복소수 가우시안 메커니즘(Complex-valued Gaussian Mechanism, cGM) 도입**: 복소수 쿼리 함수에 적용 가능한 가우시안 메커니즘을 정의하고, 이를 $\mu$-Gaussian DP($\mu$-GDP) 관점에서 이론적으로 분석하였다. 특히, 실수부와 허수부의 상관관계가 없는 circular cGM이 프라이버시-유틸리티 트레이드오프 관점에서 최적임을 증명하였다.
2. **$\zeta$-DP-SGD 알고리즘 제안**: 실수 영역의 DP-SGD를 복소수 신경망으로 확장한 $\zeta$-DP-SGD를 제안하였다. 이를 위해 Wirtinger calculus를 도입하여 복소수 가중치에 대한 켤레 기울기(Conjugate Gradient)를 계산하고 이를 클리핑(Clipping)한 후 노이즈를 추가하는 절차를 정립하였다.
3. **새로운 CVNN 프리미티브 개발**: DP 학습 시 배치 정규화(Batch Normalization)를 사용할 수 없는 문제를 해결하기 위해 복소수 그룹 정규화(Complex GroupNorm)를 제안하였으며, 기존 활성화 함수보다 성능이 뛰어난 $\text{ConjMish}$ 활성화 함수를 새롭게 설계하였다.
4. **실제 MRI 데이터셋을 통한 검증**: MRI 펄스 시퀀스 분류(Pulse Sequence Classification)라는 실제 의료 태스크의 k-space 데이터에 제안 방법론을 적용하여, 강력한 프라이버시 보장 하에서도 중앙 집중식 학습(Centralized Learning)에 근접하는 높은 정확도를 달성함을 입증하였다.

## 📎 Related Works

기존 연구들은 MRI 재구성 등에서 CVNN의 효용성을 입증해 왔으며, 최근 자동 미분 시스템의 발전과 Wirtinger calculus의 도입으로 CVNN의 구현이 용이해졌다. 그러나 CVNN을 연합 학습(FL)에 적용한 사례는 극히 드물며, 특히 의료 분야에서 DP를 결합한 연구는 본 논문이 처음이라고 명시하고 있다.

DP 분야에서는 DP-SGD가 실수 기반 신경망의 표준으로 자리 잡았으나, 복소수 데이터에 DP를 적용한 기존의 소수 연구들은 일반적인 프레임워크를 제공하지 못했다. 본 논문은 이러한 공백을 메우기 위해 복소수 영역에서의 정밀한 프라이버시 회계(Privacy Accounting)가 가능한 이론적 토대를 마련함으로써 기존 접근 방식과 차별화된다.

## 🛠️ Methodology

### 1. Complex Gaussian Mechanism (cGM)

복소수 쿼리 함수 $q: \mathcal{X} \rightarrow \mathbb{C}^n$에 대해, 노이즈 $\psi \sim \mathcal{N}_{\mathbb{C}}(0, \Gamma, C)$를 더하는 메커니즘을 정의한다. 여기서 $\Gamma$는 공분산 행렬, $C$는 관계 행렬(Relation Matrix)이다. 특히 $\rho=0$인 circular cGM의 경우, 실수부와 허수부가 독립적인 동일 분포(i.i.d.) 가우시안 분포를 따르게 된다.

이 메커니즘의 프라이버시 특성은 $\mu$-GDP로 표현되며, 이때 $\mu$ 값은 다음과 같이 결정된다:
$$\mu = \frac{d}{\Delta^2(q)}$$
여기서 $\Delta(q)$는 쿼리 함수의 $\ell_2$-민감도이다. 논문은 상관계수 $|\rho|$가 증가할수록 $\mu$가 증가하여 프라이버시 보호 수준이 낮아짐을 보이며, $\rho=0$일 때 최적의 효율을 가짐을 수학적으로 증명하였다.

### 2. $\zeta$-DP-SGD 알고리즘

복소수 신경망을 학습시키기 위해 제안된 $\zeta$-DP-SGD의 절차는 다음과 같다.

1. **켤레 기울기 계산**: 복소수-실수 손실 함수 $L: \mathbb{C}^n \rightarrow \mathbb{R}$에 대해 Wirtinger calculus를 사용하여 켤레 기울기 $\nabla L$을 계산한다.
2. **클리핑(Clipping)**: 각 샘플의 기울기 노름이 임계값 $B$를 넘지 않도록 다음과 같이 클리핑한다.
    $$\hat{g} = \frac{g}{\max(1, \|g\|_2 / B)}$$
3. **노이즈 추가**: 클리핑된 기울기의 평균에 circular cGM을 통해 생성된 가우시안 노이즈를 추가하여 프라이버시를 보장한다.
4. **가중치 업데이트**: 노이즈가 섞인 기울기를 이용하여 가중치를 업데이트한다.

### 3. CVNN Primitives

- **Complex GroupNorm**: DP 환경에서는 샘플 간 정보 유출을 방지하기 위해 Batch Norm을 사용할 수 없다. 이를 대체하기 위해 그룹별로 화이트닝(Whitening)을 수행하는 Complex GroupNorm을 제안하였다. 이 과정에서는 SVD(Singular Value Decomposition)를 사용하여 공분산 행렬의 역제곱근을 구함으로써 데이터를 정규화한다.
- **ConjMish 활성화 함수**: 기존의 CReLU나 Cardioid보다 높은 성능을 보이는 새로운 활성화 함수 $\text{ConjMish}$를 제안하였다. 정의는 다음과 같다:
    $$\text{ConjMish}(z) = (1+i)\text{Mish}(\Re(z)) - (1-i)\text{Mish}(\Im(z))$$
    이는 단순한 위상 통과가 아니라 위상 비선형성(Phase Non-linearity) 효과를 제공하여 모델의 수렴 속도와 정확도를 향상시킨다.

## 📊 Results

### 실험 설정

- **데이터셋**: Medical Segmentation Decathlon의 뇌(brain) 서브 챌린지 데이터.
- **태스크**: MRI 펄스 시퀀스 분류 (FLAIR, T1w, T1wGD, T2w의 4가지 클래스).
- **입력 데이터**: k-space(주파수 영역)의 복소수 데이터 ($32 \times 32$ 픽셀).
- **환경**: 11개의 노드로 구성된 i.i.d. 연합 학습 환경, Flower 프레임워크 및 Opacus 라이브러리 사용.
- **평가 지표**: 프라이버시 예산 $\epsilon \in \{1, 3, 5, 8, 10\}$ 및 $\delta=0.001$에 따른 분류 정확도.

### 주요 결과

실험 결과, 제안된 방법론은 강력한 프라이버시 보호 수준에서도 우수한 성능을 보였다.

- **정확도**: $\epsilon=3$ (상당히 엄격한 프라이버시 수준)에서 약 $87.98\%$의 정확도를 기록하였다.
- **프라이버시-유틸리티 관계**: $\epsilon$이 3에서 10으로 증가하더라도 정확도 상승폭은 약 2% 내외로 적었다. 이는 매우 엄격한 DP 설정을 적용하더라도 성능 저하가 거의 없음을 의미한다.
- **중앙 집중식 학습(CL)과의 비교**: CL의 정확도가 약 $82.85\% \sim 90.89\%$ 범위인 반면, FL 기반의 $\zeta$-DP-SGD는 $\epsilon=10$에서 $90.12\%$를 달성하여 CL과 거의 대등한 성능을 보였다.

## 🧠 Insights & Discussion

본 논문은 복소수 데이터라는 특수한 도메인에서 DP의 이론적 기반을 마련하고, 이를 실제 의료 영상 태스크에 성공적으로 적용하였다는 점에서 큰 강점을 가진다. 특히 복소수 가우시안 메커니즘의 최적 조건을 수학적으로 도출하여 불필요한 시행착오를 줄였으며, $\text{ConjMish}$와 같은 새로운 프리미티브를 통해 CVNN의 표현력을 높였다.

다만, 몇 가지 한계점과 논의 사항이 존재한다. 첫째, CVNN은 파라미터당 두 개의 실수값이 필요하므로 통신 오버헤드가 실수 기반 모델보다 높다. 논문에서도 이를 언급하며 향후 혼합 정밀도(Mixed-precision)나 양자화(Quantization) 기법의 필요성을 제안하고 있다. 둘째, 실험이 i.i.d. 설정에서 진행되었으므로, 실제 의료 현장에서 발생할 수 있는 Non-i.i.d. 데이터 분포 상황에서의 강건성은 추가적인 검증이 필요하다.

결론적으로, 본 연구는 의료 데이터의 민감성을 고려할 때, 단순히 데이터를 분산시키는 FL을 넘어 수학적 보장을 제공하는 DP-CVNN 프레임워크가 필수적임을 시사한다.

## 📌 TL;DR

본 논문은 복소수(Complex-valued) 데이터를 처리하는 신경망에 차분 프라이버시(DP)를 적용하기 위한 이론적 프레임워크와 $\zeta$-DP-SGD 알고리즘을 제안하였다. 복소수 가우시안 메커니즘(cGM)의 수학적 성질을 규명하고, 복소수 그룹 정규화 및 $\text{ConjMish}$ 활성화 함수를 도입하여 성능을 최적화하였다. MRI 펄스 시퀀스 분류 실험을 통해, 강력한 프라이버시 보장 하에서도 중앙 집중식 학습에 근접하는 높은 정확도를 달성할 수 있음을 입증하였으며, 이는 향후 프라이버시가 중요한 의료 AI 및 신호 처리 연구에 중요한 기초가 될 것으로 기대된다.
