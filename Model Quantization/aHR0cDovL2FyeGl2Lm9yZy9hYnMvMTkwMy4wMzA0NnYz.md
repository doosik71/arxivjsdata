# Focused Quantization for Sparse CNNs

Yiren Zhao, Xitong Gao, Daniel Bates, Robert Mullins, Cheng-Zhong Xu (2019)

## 🧩 Problem to Solve

심층 합성곱 신경망(CNN)은 강력한 성능을 제공하지만, 방대한 메모리와 계산 리소스 요구량으로 인해 자원이 제한된 장치(constrained devices)에 배포하는 데 어려움이 있다. 이를 해결하기 위해 가지치기(Pruning)와 양자화(Quantization) 같은 모델 압축 기술이 사용되어 왔다. 

특히 가중치를 2의 거듭제곱 형태로 표현하여 곱셈 연산을 비트 시프트(bit-shift) 연산으로 대체하는 Shift Quantization은 하드웨어 효율성 측면에서 매우 유리하다. 그러나 세밀한 가지치기(fine-grained pruning)를 거친 희소 CNN(Sparse CNN)의 경우, 각 레이어마다 희소도(sparsity)가 다르게 나타나며 가중치 분포의 통계적 특성이 변화한다. 

실제 경험적으로, 가지치기 후의 일부 레이어에서는 가중치가 0 근처에 집중되지 않고 오히려 0을 피하는 경향이 나타난다. 기존의 Shift Quantization은 0 근처에 정밀한 양자화 레벨이 밀집되어 있어, 이러한 희소 레이어에 적용할 경우 많은 양자화 레벨이 사용되지 않고 낭비되는 문제가 발생한다. 결과적으로 이는 양자화 오차를 증가시키고 모델의 정확도를 떨어뜨리는 원인이 된다. 본 논문의 목표는 희소 CNN의 가중치 분포 특성을 활용하여 양자화 레벨의 활용도를 높이고, 계산 비용을 최소화하면서도 높은 압축률과 정확도를 유지하는 새로운 양자화 전략을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Focused Quantization(FQ)**으로, 이는 기존의 Shift Quantization과 새롭게 제안하는 **Recentralized Quantization**을 결합한 하이브리드 방식이다. 

중심적인 직관은 가중치 분포에서 확률 밀도가 높은 영역(high probability regions)에 양자화 자원을 집중시키는 것이다. 가중치 분포가 0 근처에 집중되어 있지 않다면, 분포의 중심을 확률 밀도가 높은 곳으로 강제로 이동(recentralize)시킨 후 양자화를 수행하고, 다시 원래 위치로 되돌리는 방식을 통해 양자화 레벨의 활용도를 극대화할 수 있다는 설계 아이디어를 제시한다.

## 📎 Related Works

모델 압축을 위한 기존 연구들은 크게 가지치기와 양자화로 나뉜다.
- **Pruning:** 세밀한 가지치기(Fine-grained pruning)는 개별 가중치를 제거하여 높은 압축률을 제공하지만, 요소별 희소성(element-wise sparsity)을 유발한다.
- **Quantization:** 가중치를 낮은 비트로 표현하여 메모리와 계산량을 줄인다. 특히 가중치를 2의 거듭제곱으로 표현하는 방식은 곱셈을 비트 시프트로 대체할 수 있어 하드웨어 효율성이 높다.
- **Existing Pipelines:** Deep Compression과 같이 가지치기, 양자화, 허프만 코딩(Huffman encoding)을 체인 형태로 연결하여 압축률을 극대화하려는 시도가 있었다.

그러나 기존 연구들은 가지치기와 양자화 사이의 통계적 관계를 충분히 탐구하지 않았다. 특히 희소 모델에서 나타나는 가중치의 분포 변화가 양자화 효율에 미치는 영향에 대한 분석이 부족했으며, 본 논문은 이 지점을 공략하여 통계적 특성에 기반한 적응형 양자화 방법을 제안함으로써 기존 방식과 차별화를 꾀한다.

## 🛠️ Methodology

### 1. Shift Quantization (기초)
기본이 되는 Shift Quantization은 가중치 $v$를 다음과 같이 표현한다.
$$v = s \cdot 2^{e-b}$$
여기서 $s \in \{-1, 0, 1\}$은 부호 또는 0을 나타내고, $e$는 $[0, 2^k - 1]$ 범위의 정수이며, $b$는 레이어별 상수인 bias이다. 이 방식은 하드웨어에서 곱셈을 비트 시프트 연산으로 대체할 수 있게 한다.

### 2. Recentralized Quantization
희소 레이어의 가중치 분포에서 확률 밀도가 높은 영역에 집중하기 위해 설계된 방식이다. 가중치 $\theta$에 대한 양자화 함수 $Q[\theta]$는 다음과 같이 정의된다.
$$Q[\theta] = z_\theta \alpha \sum_{c \in C} \delta_{c,m_\theta} Q^{rec}_c[\theta]$$
여기서 $z_\theta$는 가지치기 여부를 나타내는 이진 값이며, $\alpha$는 학습 가능한 레이어별 스케일링 인자이다. 핵심이 되는 $Q^{rec}_c[\theta]$는 다음과 같다.
$$Q^{rec}_c[\theta] = Q^{shift}_{n,b} \left[ \frac{\theta - \mu_c}{\sigma_c} \right] \sigma_c + \mu_c$$
이 과정은 가중치 $\theta$에서 평균 $\mu_c$를 빼서 중심을 0으로 옮기고, 표준편차 $\sigma_c$로 나누어 스케일을 조정한 뒤 Shift Quantization을 적용하고, 다시 역과정을 통해 원래의 스케일과 중심 위치로 복원하는 절차를 거친다.

### 3. 최적화 및 하이퍼파라미터 결정
$\mu_c, \sigma_c, \lambda_c$ 등의 하이퍼파라미터를 최적화하기 위해 가중치 분포를 가우시안 혼합 모델(Gaussian Mixture Model, GMM)로 근사한다.
- **EM 알고리즘:** 최대 우도 추정(Maximum Likelihood Estimate, MLE)을 위해 EM 알고리즘을 사용하여 가중치 분포 $p(\theta|D)$에 가장 잘 맞는 GMM을 찾는다. 일반적으로 두 개의 가우시안 성분($C = \{-, +\}$)을 사용한다.
- **컴포넌트 할당:** 각 가중치 $\theta$가 어떤 성분 $c$에 속할지는 범주형 분포(categorical distribution)에서 샘플링하여 결정한다.

### 4. 적응형 양자화 선택 (Wasserstein Separation)
모든 레이어가 반드시 여러 개의 고확률 영역을 가지는 것은 아니다. 두 가우시안 성분이 너무 많이 겹쳐 있다면 굳이 Recentralized 방식을 쓸 필요 없이 일반 Shift Quantization을 쓰는 것이 더 효율적이다. 이를 판단하기 위해 **2-Wasserstein 거리**를 기반으로 한 Wasserstein separation $W(c_1, c_2)$를 도입한다.
$$W(c_1, c_2) = \frac{1}{\sigma^2} \left( (\mu_{c_1} - \mu_{c_2})^2 + (\sigma_{c_1} - \sigma_{c_2})^2 \right)$$
여기서 $\sigma^2$은 전체 가중치 분포의 분산이다. $W(c_1, c_2) < w_{sep}$ (실험적으로 2.0 사용)인 경우 일반 Shift Quantization을 적용하고, 그렇지 않은 경우 Recentralized Quantization을 적용한다.

### 5. 학습 절차 및 MDL 관점
모델은 다음과 같은 파이프라인으로 최적화된다.
1. 초기 GMM 파라미터 및 컴포넌트 마스크 $m_\theta$ 계산.
2. 가중치를 양자화하여 순전파(forward pass)를 수행하고, 역전파(backward pass)에서는 양자화를 항등 함수로 취급하여 부동 소수점 가중치를 업데이트하는 Fine-tuning 수행.
3. $k$ 에포크마다 현재 가중치 분포를 반영하여 하이퍼파라미터를 업데이트하며, $k$를 지수적으로 증가시켜 학습 안정성을 높인다.

이론적으로 이는 최소 기술 길이(Minimum Description Length, MDL) 최적화 문제로 정식화되며, 에러 비용(Error cost)과 복잡도 비용(Complexity cost, KL-divergence)의 합인 변분 자유 에너지(variational free energy)를 최소화하는 것과 같다.

## 📊 Results

### 1. 모델 크기 감소 및 정확도
ImageNet 데이터셋의 ResNet-18, ResNet-50, MobileNet-V1/V2 모델에 대해 실험을 진행하였다.
- **ResNet-50:** 5-bit 양자화를 적용했을 때, Top-5 정확도 손실을 단 0.24%로 억제하면서 **18.08배의 압축률(CR)**을 달성하였다.
- **ResNet-18:** 다른 최신 압축 기법(TTQ, INQ, ADMM, Coreset 등)과 비교했을 때, 동일하거나 더 높은 압축률에서 가장 높은 정확도를 기록하였다.
- **MobileNets:** 구조적 특성상 ResNet보다는 압축률이 낮았으나(약 8배), 여전히 효율적인 메모리 풋프린트를 보여주었다.

### 2. 계산 복잡도 및 하드웨어 효율성
가중치뿐만 아니라 활성화 함수(activations)와 Batch Normalization(BN) 파라미터를 정수 양자화함으로써 모든 곱셈 연산을 비트 시프트와 정수 덧셈으로 대체하였다.
- **하드웨어 리소스:** 커스텀 가속기 설계 시, FQ(5-bit)는 ABC-Net(5-bit)이나 LQ-Net(2-bit)보다 훨씬 적은 수의 로직 게이트(Logic Gates)를 필요로 한다. 이는 ABC-Net 등이 여러 개의 병렬 이진 합성곱을 수행하고 이를 고정밀도 합산하는 오버헤드가 크기 때문이며, FQ는 단일 비트 시프트 경로만 필요로 하여 하드웨어 구현 효율이 압도적으로 높다.

### 3. Wasserstein 임계값 분석
$w_{sep}$ 값의 변화에 따른 Top-1 에러를 분석한 결과, 임계값이 너무 낮으면 불필요하게 많은 레이어에 Recentralized 방식이 적용되어 성능이 저하되고, 너무 높으면 Shift Quantization의 한계로 인해 성능이 떨어진다. 실험 결과 $w_{sep} = 2.0$ 근처에서 최적의 성능이 나타남을 확인하였다.

## 🧠 Insights & Discussion

본 논문은 가지치기로 인해 발생하는 가중치 분포의 비정형성을 양자화 단계에서 어떻게 처리할 것인가에 대한 명확한 해답을 제시한다. 단순히 고정된 양자화 레벨을 사용하는 대신, GMM과 Wasserstein 거리를 이용해 레이어별 특성에 맞게 양자화 전략을 선택하는 적응형 접근 방식이 매우 효과적임을 입증하였다.

특히 주목할 점은 이론적 기반(MDL)과 실제 하드웨어 구현 효율성(Logic Gate 수)을 동시에 고려했다는 점이다. 대부분의 양자화 논문이 정확도와 압축률에만 집중하는 반면, 본 연구는 실제 하드웨어 가속기에서의 연산 패턴과 리소스 사용량을 구체적으로 분석하여 실용성을 높였다.

다만, 하이퍼파라미터를 주기적으로 업데이트하는 과정이 학습 시간을 증가시킬 수 있으며, GMM이 모든 가중치 분포를 완벽하게 근사할 수 있는지에 대한 일반화 가능성에 대해서는 추가적인 논의가 필요할 수 있다.

## 📌 TL;DR

이 논문은 희소 CNN의 가중치 분포 특성을 분석하여, 확률 밀도가 높은 영역에 양자화 레벨을 집중시키는 **Focused Quantization(FQ)**을 제안한다. GMM을 통해 가중치 분포를 분석하고 Wasserstein 거리로 양자화 방식을 적응적으로 선택하며, 이를 통해 ResNet-50에서 정확도 손실 0.24%만으로 18.08배의 압축률을 달성하였다. 특히 모든 곱셈을 비트 시프트로 대체 가능하여 하드웨어 리소스 사용량을 획기적으로 줄였으며, 이는 향후 IoT 기기용 저전력 CNN 가속기 설계에 중요한 기여를 할 것으로 보인다.