# U-Net Fixed-Point Quantization for Medical Image Segmentation

MohammadHossein AskariHemmat et al. (2019)

## 🧩 Problem to Solve

의료 영상 분할(Medical Image Segmentation) 분야에서 널리 사용되는 U-Net 아키텍처는 높은 정밀도의 픽셀 수준 재구성을 위해 많은 메모리 자원을 소모한다. 특히 의료 영상은 해상도가 매우 높고 3차원 볼륨 데이터인 경우가 많아, 모델의 파라미터와 활성화 함수(activation) 값을 32비트 부동 소수점(floating point) 정밀도로 저장하고 처리하는 것은 하드웨어적으로 큰 부담이 된다. 이러한 메모리 및 계산 비용 문제는 스마트폰과 같은 엣지 디바이스로의 모델 배포를 어렵게 만들며, 추론 속도를 저하시키는 요인이 된다. 본 논문의 목표는 U-Net의 성능 저하를 최소화하면서 모델의 메모리 점유율과 계산 시간을 획기적으로 줄일 수 있는 효율적인 고정 소수점 양자화(Fixed-Point Quantization) 방법을 제안하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 U-Net의 가중치(weights)와 활성화 값(activations)을 낮은 비트 해상도의 고정 소수점으로 표현하여 하드웨어 효율성을 극대화하는 것이다. 구체적인 기여 사항은 다음과 같다.

- 의료 영상 분할을 위한 U-Net 아키텍처에 적용 가능한 첫 번째 고정 소수점 양자화 방법론을 제시하였다.
- 가중치 4비트, 활성화 값 6비트 설정을 통해 성능 손실을 최소화하면서 메모리 요구량을 8배 감소시킬 수 있음을 입증하였다.
- 기존의 BNN(Binary Neural Network)이나 TernaryNet과 달리, 현재 상용화된 CPU 및 GPU 하드웨어에서 더 효율적으로 작동하는 정수 기반 연산 구조를 채택하였다.
- 양자화가 강한 규제(regularizer) 역할을 수행하므로, 기존 U-Net에서 사용하던 Dropout을 제거하는 것이 오히려 성능 향상에 도움이 된다는 통찰을 제공하였다.

## 📎 Related Works

기존의 신경망 양자화 연구는 결정론적 양자화와 확률적 양자화로 나뉘며, 주로 일반적인 이미지 분류 작업에 집중되어 있었다. 의료 영상 분할 분야에서는 Fully Convolutional Networks(FCN)에 Incremental Quantization(INQ)을 적용하여 7비트 수준에서 성능을 유지한 연구가 있었다. 하지만 FCN은 특징 맵을 최종 단계에서 한 번에 업샘플링하는 구조인 반면, U-Net은 점진적인 해상도 증가와 Skip Connection을 통해 더 높은 정확도와 빠른 수렴 속도를 보인다.

또한, U-Net을 위한 기존 양자화 시도인 TernaryNet은 가중치를 3진수(-1, 0, 1)로 표현하고 Hamming 공간에서의 행렬 곱셈을 통해 효율성을 높이려 했다. 그러나 저자들은 TernaryNet이 $\tanh$ 활성화 함수를 사용하며, 부동 소수점 스케일링 팩터가 필요하다는 점 때문에 실제 상용 CPU/GPU 하드웨어에서는 연산 효율이 낮다는 한계점을 지적하며 차별성을 두었다.

## 🛠️ Methodology

### 전체 시스템 구조 및 양자화 함수
본 논문은 32비트 부동 소수점 모델을 베이스라인으로 하여, 추론 경로의 파라미터를 다음과 같은 고정 소수점 양자화 함수로 변환한다.

$$\text{quantize}(x, n) = (\text{round}(\text{clamp}(x, n)) \ll n) \gg n$$

여기서 $\text{round}$ 함수는 입력을 가장 가까운 정수로 투영하며, $\ll$와 $\gg$는 각각 비트 시프트 왼쪽/오른쪽 연산자이다. $\text{clamp}$ 함수는 값을 특정 범위 내로 제한하며 다음과 같이 정의된다.

$$\text{clamp}(x, n) = \begin{cases} 2^{n-1} & \text{when } x \geq 2^{n-1} \\ x & \text{when } 0 < x < 2^{n-1} \\ 0 & \text{when } x \leq 0 \end{cases}$$

### 고정 소수점 표현 방식
임의의 수 $x$를 고정 소수점으로 매핑하기 위해 먼저 정수부($x_i$)와 소수부($x_f$)를 분리한다.

$$x_f = \text{abs}(x) - \text{floor}(\text{abs}(x)), \quad x_i = \text{floor}(\text{abs}(x))$$

이후 정수부 비트 수($ibits$)와 소수부 비트 수($fbits$)를 지정하여 최종 값을 계산한다.

$$\text{to\_fixed\_point}(x, ibits, fbits) = \text{sign}(x) * \text{quantize}(x_i, ibits) + \text{sign}(x) * \text{quantize}(x_f, fbits)$$

본 논문에서는 $Q^p_{i.f}$ 표기법을 사용하여 파라미터 $p$의 정수부 $i$비트, 소수부 $f$비트 양자화를 나타낸다. 실험 결과, 가중치($w$)는 대부분 $[-1, 1]$ 범위에 분포하므로 정수부가 필요 없는 $Q^w_{0.f}$ 형태를 사용하였다.

### 학습 절차 및 최적화
1. **미분 가능성 확보**: $\text{clamp}$ 함수와 $\text{round}$ 함수는 임계값에서 미분이 불가능하다. 이를 해결하기 위해 Straight-Through Estimator(STE)를 도입하여 역전파 시 그래디언트가 임계값과 반올림 함수를 그대로 통과하도록 설계하였다.
2. **Dropout 제거**: 양자화 자체가 강력한 규제 효과를 가지므로, Dropout을 함께 사용할 경우 성능이 크게 하락함을 발견하여 모든 레이어에서 Dropout을 제거하였다.
3. **부분적 정밀도 유지**: 모든 레이어를 양자화하는 대신, 마지막 레이어는 Full Precision(FP32)으로 유지하는 것이 분할 성능 유지에 결정적인 영향을 미친다는 점을 확인하였다.
4. **Batch Normalization Folding**: 학습 시에는 BN을 부동 소수점으로 계산한 후 양자화 블록을 통과시키지만, 추론 시에는 BN 파라미터를 가중치에 통합(folding)하여 양자화된 가중치의 일부로 처리함으로써 연산량을 줄였다.

## 📊 Results

### 실험 환경
- **데이터셋**: 
    - Spinal Cord Gray Matter (GM): 척수 회백질 분할.
    - ISBI Electron Microscopic (EM): 뉴런 구조 분할.
    - NIH Pancreas: 복부 CT 내 췌장 분할 (512x512 2D 슬라이스 추출).
- **평가 지표**: Dice Overlap Score.
- **손실 함수**: GM과 NIH 데이터셋은 Dice Loss를 사용하였으며, EM 데이터셋은 Weighted Cross Entropy와 Dice Loss의 가중 합을 사용하였다.

### 주요 결과
- **정량적 결과**: 가중치 4비트($Q^w_{0.4}$), 활성화 값 6비트($Q^a_{6.0}$) 설정을 사용했을 때, FP32 모델 대비 메모리 요구량을 8배 감소시키면서도 Dice score의 하락폭은 EM 2.21%, GM 0.57%, NIH 2.09%로 매우 낮았다.
- **비교 분석**:
    - **BNN 및 TernaryNet 대비**: EM 및 GM 데이터셋에서 제안 방법이 BNN과 TernaryNet보다 우수한 성능을 보였다. 특히 BNN은 성능 저하가 매우 심했다.
    - **하드웨어 효율성**: $\tanh$를 사용하는 TernaryNet보다 $\text{ReLU}$를 사용하는 제안 방법이 추론 속도 면에서 최대 8배 더 빠름을 Intel OpenVino 벤치마크를 통해 입증하였다.

## 🧠 Insights & Discussion

본 논문은 단순한 비트 수 감소를 넘어, 실제 하드웨어 구현 관점에서의 효율성을 깊이 있게 다루었다. 특히 $\tanh$와 $\text{ReLU}$의 연산 시간 차이를 정량적으로 분석하여, 수학적 최적화보다 하드웨어 친화적인 연산자 선택이 실질적인 추론 속도 향상에 더 중요함을 보여주었다.

또한, 양자화가 모델의 복잡도를 제한함으로써 Dropout과 같은 추가적인 규제 기법 없이도 과적합을 방지할 수 있다는 점은 주목할 만한 발견이다. 다만, NIH 췌장 데이터셋의 경우 해부학적 변동성이 커서 다른 데이터셋에 비해 양자화에 따른 성능 저하가 상대적으로 더 크게 나타났는데, 이는 매우 복잡한 형태의 객체를 분할할 때 저정밀도 표현이 가질 수 있는 한계를 시사한다.

## 📌 TL;DR

이 논문은 U-Net 기반 의료 영상 분할 모델을 위해 하드웨어 친화적인 **고정 소수점 양자화(Fixed-Point Quantization)** 방법을 제안하였다. 가중치 4비트, 활성화 값 6비트 설정을 통해 **메모리 사용량을 8배 줄이면서도 성능 하락을 최소화**하였으며, $\text{ReLU}$ 기반의 정수 연산을 통해 기존 TernaryNet보다 빠른 추론 속도를 달성하였다. 이 연구는 고해상도 의료 영상을 처리해야 하는 임베디드 시스템이나 실시간 진단 도구에 U-Net 모델을 효율적으로 배포하는 데 중요한 가이드라인을 제공한다.