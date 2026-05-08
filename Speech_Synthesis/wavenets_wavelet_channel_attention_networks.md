# WaveNets: Wavelet Channel Attention Networks

Hadi Salman, Caleb Parks, Shi Yin Hong, Justin Zhan (2022)

## 🧩 Problem to Solve

본 논문은 컴퓨터 비전 분야에서 널리 사용되는 Channel Attention(CA) 메커니즘, 특히 SENet에서 제안된 방식의 한계점을 해결하고자 한다. 기존의 Channel Attention은 Global Average Pooling(GAP)을 사용하여 각 채널의 특징을 하나의 스칼라 값으로 압축하는데, 이 과정에서 특징 학습에 필요한 정교한 세부 정보가 손실되는 정보 손실(information loss) 문제가 발생한다.

GAP의 단순한 차원 축소 방식은 채널 간의 상호 의존성(inter-channel dependencies)을 모델링하는 능력을 제한하며, 이는 결국 모델의 전체적인 분류 성능을 저하시키는 요인이 된다. 따라서 본 연구의 목표는 채널 정보를 압축하는 과정에서 특징 보존(feature preservation) 능력을 향상시켜, 계산 효율성을 유지하면서도 더 정교한 세부 정보를 캡처할 수 있는 새로운 채널 압축 메커니즘을 설계하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Discrete Wavelet Transform(DWT)을 활용하여 채널 정보를 압축하는 것이다. 주요 기여 사항은 다음과 같다.

1. **GAP와 Haar Wavelet의 관계 규명**: 이론적 증명을 통해 기존의 GAP가 Discrete Haar Wavelet Transform(DHWT)의 재귀적 근사 성분(recurrent approximate component)의 특수한 사례임을 밝혔다.
2. **WaveNet 제안**: DWT의 주파수 기반 압축 방식을 채널 어텐션에 도입하여 GAP를 대체하는 WaveNet 프레임워크를 제안하였다.
3. **WaveNet-C 제안**: Haar Wavelet을 넘어, Gram-Schmidt 과정을 통해 생성된 직교 독립 필터(orthogonal linearly independent filters)를 사용하는 WaveNet-C를 제안하여 압축 과정에서 정보의 다양성을 극대화하였다.
4. **효율성 및 성능 입증**: ImageNet 데이터셋 실험을 통해 SENet 및 SOTA 모델인 FcaNet-34보다 우수한 성능을 보였으며, 파라미터 수와 계산 비용은 SENet과 유사한 수준임을 입증하였다.

## 📎 Related Works

### 기존의 Visual Attention 연구

SENet은 Squeeze-and-Excitation 구조를 통해 효율적인 채널 어텐션을 도입하였다. 이후 CBAM은 GAP의 한계를 극복하기 위해 Global Max Pooling을 병합하였고, ECANet은 불필요한 차원 축소 없이 채널 간 상호작용을 캡처하도록 구조를 개선하였다. FcaNet은 주파수 분석 관점에서 GAP와 Discrete Cosine Transform(DCT)의 관계를 설명하며 멀티 스펙트럼 성분을 추가하였다.

### Wavelet Transform의 활용

DWT는 이미지 압축, 복원, 분류 등 디지털 신호 처리 분야에서 널리 사용되어 왔다. 최근 딥러닝 연구에서도 AWNet이나 WAEN 등이 어텐션 메커니즘과 DWT를 결합하려는 시도를 하였으나, 대부분 특정 도메인(ISP, 비디오 초해상도 등)의 응용 단계에서 결합되었다. 반면, WaveNet은 채널 어텐션의 가장 근본적인 아키텍처 수준에서 DWT를 통합하여 일반적인 성능 향상을 꾀한다는 점에서 차별점을 갖는다.

## 🛠️ Methodology

### Discrete Wavelet Transform (DWT)

DWT는 입력 신호를 저주파 통과 필터(Low-pass filter)와 대역 통과 필터(Band-pass filter)를 통해 분해한다. 2D DWT를 적용하면 입력 $X$는 다음과 같이 네 개의 서브 밴드로 분해된다.

$$X_{output} = \text{DWT}(X) = \begin{bmatrix} A & V \\ H & D \end{bmatrix}$$

여기서 $A$는 근사(Approximation), $V$는 수직 차이(Vertical difference), $H$는 수평 차이(Horizontal difference), $D$는 대각선 차이(Diagonal difference)를 의미한다. 본 논문에서는 이를 컨볼루션 형태로 구현하며, 스트라이드 2와 패딩 없음 설정을 통해 출력 크기를 $\frac{H}{2} \times \frac{W}{2}$로 줄인다.

### Wavelet Channel Attention의 구조

기존 SENet의 프로세스는 다음과 같다.
$$\text{att} = \text{sigmoid}(\text{fc}(\text{GAP}(X)))$$

WaveNet은 여기서 $\text{GAP}(X)$ 부분을 DWT 기반의 압축 함수 $C(X)$로 대체한다.

#### 1. 이론적 배경 (Theorem 1)

저자들은 GAP가 Haar Wavelet Transform의 재귀적 근사 성분과 동일하다는 것을 증명하였다. 즉, GAP는 DWT의 여러 성분 중 오직 '근사 성분'만을 사용하는 특수한 경우이며, 나머지 세부 정보(V, H, D)를 모두 버리는 행위임을 수학적으로 보였다.

#### 2. WaveNet-C: 직교 독립 필터링

GAP의 정보 손실을 막기 위해, WaveNet-C는 무작위로 초기화된 필터들에 Gram-Schmidt 과정을 적용하여 서로 직교(orthogonal)하고 선형 독립인 필터 세트를 생성한다.

- **Basic Block**: 채널 크기에 맞는 4개의 직교 필터를 사용하여 정보를 압축한다.
- **Bottleneck Block**: 채널 크기가 매우 큰 경우(1024, 2048), 채널을 512 크기의 청크(chunk)로 나누고 각각에 대해 새로운 직교 필터를 적용하여 정보의 다양성을 확보한다.

최종적인 어텐션 벡터 생성 과정은 다음과 같다.
$$\text{Attention}(X) = \text{sigmoid}(\text{fc}(C(X)))$$
여기서 $C(X)$는 제안된 직교 웨이브렛 압축 모듈을 의미한다.

## 📊 Results

### 실험 설정

- **Backbone**: ResNet-34
- **Dataset**: ImageNet
- **Optimizer**: SGD (Momentum 0.9, Learning Rate 0.2)
- **Training**: 100 epochs, Cosine Annealing Warm Restarts, Mixed Precision Training (Nvidia APEX) 적용

### 정량적 결과

실험 결과, WaveNet-C는 비교 대상 모델들 중 가장 높은 성능을 기록하였다.

| Method | Parameters | FLOPs | Top-1 Acc | Top-5 Acc |
| :--- | :---: | :---: | :---: | :---: |
| ResNet-34 | 21.80 M | 3.68 G | 74.58% | 92.05% |
| SENet | 21.95 M | 3.68 G | 74.83% | 92.23% |
| ECANet | 21.80 M | 3.68 G | 74.65% | 92.21% |
| FcaNet-TSI | 21.95 M | 3.68 G | 75.02% | 92.07% |
| **WaveNet-C** | **21.95 M** | **3.68 G** | **75.06%** | **92.37%** |

- **성능 향상**: WaveNet-C는 Top-1 정확도 75.06%를 달성하여 SOTA인 FcaNet-TSI(75.02%)와 SENet(74.83%)를 모두 상회하였다.
- **계산 효율성**: 파라미터 수와 FLOPs가 SENet과 거의 동일하며, 실제 계산 비용 증가량은 SENet 대비 약 0.05%로 매우 미미하다.

## 🧠 Insights & Discussion

### 강점 및 해석

본 논문의 가장 큰 강점은 GAP라는 단순한 연산이 가진 수학적 의미를 DWT 관점에서 재해석하여, 정보 손실 문제를 이론적으로 접근했다는 점이다. 단순한 평균값(GAP) 대신 다양한 주파수 성분을 캡처하는 DWT를 도입함으로써, 네트워크가 채널 간의 더 풍부하고 독특한(unique) 특징을 학습할 수 있게 되었다. 특히 직교 필터를 사용한 WaveNet-C는 채널 간의 중복성을 줄이고 정보의 다양성을 강제함으로써 성능을 더욱 끌어올렸다.

### 한계 및 논의사항

- **범용성 검증 부족**: 실험이 주로 ResNet-34와 ImageNet 분류 작업에 집중되어 있어, 더 깊은 네트워크(ResNet-50, 101 등)나 다른 태스크(객체 탐지, 세그멘테이션)에서의 성능 향상 폭이 어느 정도일지는 명시적으로 확인되지 않았다 (저자들도 향후 과제로 언급함).
- **필터 학습 가능성**: 현재 DWT 필터는 상수로 고정되어 사용되지만, 이 필터들을 학습 가능한 파라미터로 전환했을 때 추가적인 성능 향상이 있을 가능성이 크다.

## 📌 TL;DR

본 논문은 Channel Attention의 핵심인 GAP가 정보 손실을 유발한다는 점을 지적하고, 이를 Discrete Wavelet Transform(DWT) 기반의 압축 방식으로 대체한 **WaveNet**을 제안한다. 특히 직교 독립 필터를 적용한 **WaveNet-C**는 계산 비용의 증가 없이 ImageNet 분류 작업에서 SOTA 수준의 성능(Top-1 75.06%)을 달성하였다. 이 연구는 단순한 픽셀 평균화 대신 주파수 도메인의 정보를 보존하는 것이 채널 어텐션의 효율성을 높이는 핵심임을 시사하며, 기존의 다양한 CA 모델에 쉽게 통합될 수 있는 범용적인 프레임워크를 제공한다.
