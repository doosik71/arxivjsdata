# CosSGD: Communication-Efficient Federated Learning with a Simple Cosine-Based Quantization

Yang He, Hui-Po Wang, Maximilian Zenk, Mario Fritz (2021)

## 🧩 Problem to Solve

본 논문은 Federated Learning(FL) 시스템에서 서버와 클라이언트 간의 통신 비용이 전체 시스템의 배포 및 성공적인 학습을 가로막는 주요 병목 현상(bottleneck)이 된다는 문제를 해결하고자 한다. 

기존의 Gradient Compression 기법들이 많은 진전을 이루었으나, 특히 low-bits compression(저비트 압축)을 적용할 때 모델 성능이 크게 저하되는 경향이 있다. 특히 모델 가중치(weights)와 그래디언트(gradients) 모두를 압축하는 양방향 압축(double directions compression)을 적용할 경우, 시스템의 전반적인 성능 저하가 심화되는 문제가 발생한다. 따라서 본 연구의 목표는 모델 성능을 최대한 유지하면서도 통신 비용을 획기적으로 줄일 수 있는 효율적인 양자화(Quantization) 방법을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 코사인 함수(cosine function)의 비선형성(non-linearity)을 활용하여, 중요한 정보가 더 많이 담겨 있는 큰 절대값의 가중치와 그래디언트를 더 정밀하게 양자화하는 것이다.

핵심적인 직관은 모든 값의 분포를 균일하게 처리하는 대신, magnitude가 큰 값들이 학습에 더 결정적인 역할을 한다는 점에 착안하여, 큰 값에는 더 좁은 양자화 간격을 제공하고 작은 값에는 더 넓은 간격을 제공함으로써 양자화 오차를 최적화하는 것이다. 이는 데이터의 밀집도에 집중하는 기존의 비선형 양자화 방식과는 대조적인 접근 방식이다.

## 📎 Related Works

기존 연구들은 주로 다음과 같은 접근 방식을 취해왔다.
- **Federated Learning 최적화**: FedAvg와 같은 고전적 방법부터, 서버 측 모멘텀을 사용하는 적응형 최적화(Adaptive Optimization)나 Non-IID 데이터 문제를 해결하기 위한 클라이언트 그룹화(FedCD) 등이 제안되었다.
- **Gradient Compression**: 
    - **Quantization**: float 형태의 그래디언트를 저비트로 표현하며, 편향을 줄이기 위해 확률적 비편향 양자화(probabilistic unbiased quantization)나 random Hadamard rotations를 사용하여 오차를 줄이려는 시도가 있었다.
    - **Sparsification**: 전체 그래디언트 중 일부(random 또는 top-K)만 전송하여 통신량을 줄이는 방식이 연구되었다.
- **전체 통신 비용 절감**: 모델 아키텍처를 단순화하거나(Federated Dropout), 로컬 특징 추출기와 글로벌 분류기를 분리하여 전송량을 줄이는 방식 등이 제안되었다.

기존의 비선형 양자화 방법들은 주로 데이터가 밀집된 영역(dense distribution)에서 오차를 줄이려 했으나, 본 논문은 데이터의 분포보다는 값의 중요도(magnitude)에 집중하여 차별성을 둔다.

## 🛠️ Methodology

### 전체 파이프라인 및 구조
CosSGD는 서버에서 클라이언트로 전송되는 모델 가중치($M^t$)와 클라이언트에서 서버로 전송되는 그래디언트($\nabla M_i$) 모두에 양자화를 적용하여 왕복 통신 비용(round-trip communication costs)을 절감한다.

### 코사인 기반 양자화 절차
1. **각도 공간으로의 매핑**: 벡터 $v = (v_1, \dots, v_n)^T$가 주어졌을 때, 각 성분 $v_i$를 표준 좌표축과의 각도 $\theta_i$로 변환한다.
   $$\cos(\theta_i) = \frac{v_i}{\|v\|_2} \implies \theta_i = \arccos\left(\frac{v_i}{\|v\|_2}\right), \quad \theta_i \in [0, \pi]$$
2. **양자화 범위 설정**: 각도 벡터 $\Theta = (\theta_1, \dots, \theta_n)^T$ 전체를 양자화하는 대신, 효율성을 위해 경계값 $b_\theta$를 계산하여 $[b_\theta, \pi - b_\theta]$ 범위 내에서만 양자화한다. 이때 $b_\theta = \min(\min(\Theta), 1 - \max(\Theta))$로 설정하거나, 특정 차원이 지배적인 경우 top dimensions를 clipping 하여 $b_\theta$를 구함으로써 양자화 공간의 낭비를 막는다.
3. **양자화 수행**: 
   - **Biased Quantization**: 각도 공간에서 선형 양자화를 수행한다.
   - **Unbiased Quantization**: 확률적 절차를 통해 기대값이 원래 값과 같도록 설계한다. $s$-bits 양자화 함수 $Q_\theta(\Theta; b, s)$는 다음과 같이 정의된다.
     $$Q_\theta(\Theta; b, s) = \begin{cases} \lfloor v \rfloor & \text{with probability } 1-p \\ \lfloor v \rfloor + 1 & \text{otherwise} \end{cases}$$
     여기서 $v = \frac{\Theta - b}{\pi - 2b} \times (2^s - 1)$이며, $p = v - \lfloor v \rfloor$이다.
4. **복원**: 서버는 전송받은 양자화된 벡터 $Q_\theta(\Theta)$, 원래 벡터의 노름 $\|v\|_2$, 그리고 경계값 $b_\theta$를 이용하여 원래의 그래디언트나 가중치를 복원한다.

### 양자화 오차 분석
코사인 함수의 대칭적 특성으로 인해, 본 방법은 $\sin(\cdot)$ 함수의 단조 증가 성질을 이용하여 큰 magnitude를 가진 값일수록 양자화 오차가 작아짐을 수학적으로 증명하였다. 구체적으로 $|v_1| > |v_2|$일 때 $|v_1 - Q_v(v_1)| < |v_2 - Q_v(v_2)|$가 성립한다.

### 계산 복잡도 및 효율성
- **시간/공간 복잡도**: Closed-form 수식을 사용하므로 $O(m)$의 복잡도를 가지며, 이는 k-means($O(m^2)$)나 TinyScript($O(mn)$)보다 월등히 효율적이다.
- **추가 비용**: 복원을 위해 $\|v\|_2$와 $b_\theta$라는 두 개의 float 숫자만 추가로 전송하면 되므로 통신 오버헤드가 매우 적다.
- **결합 기법**: 추가적인 비용 절감을 위해 무손실 데이터 압축 알고리즘인 `Deflate`와 Random Sparsification을 결합하여 적용한다.

## 📊 Results

### 실험 설정
- **데이터셋 및 작업**: 
    - CIFAR-10: 이미지 분류 (CNN 모델, 파라미터 122,570개)
    - BraTS 2019: 3D 뇌종양 시맨틱 세그멘테이션 (3D-UNet 모델, 파라미터 9,451,567개)
- **비교 대상**: Float32(baseline), Linear Quantization (U, R), k-means, TinyScript, signSGD (+EF, +CSEA).
- **지표**: Accuracy (CIFAR-10), Dice Score (BraTS 2019), Communication Cost.

### 주요 결과
- **저비트 압축 성능**: CIFAR-10 실험에서 2-bit 및 1-bit 압축 시에도 float32 기반 학습과 거의 유사한 성능을 보였다. 특히 2-bit linear quantization은 성능 저하가 뚜렷한 반면, CosSGD는 이를 극복하였다.
- **비선형 양자화 비교**: k-means 기반 방식보다 높은 성능을 기록하였는데, 이는 데이터의 분포보다 큰 그래디언트 값들을 정밀하게 복원하는 것이 학습에 더 중요하다는 가설을 뒷받침한다.
- **세그멘테이션 작업**: BraTS 2019 데이터셋에서도 1-bit, 2-bit 설정에서 다른 방법들보다 높은 Dice Score를 기록하였으며, 동일 성능 대비 통신 비용을 크게 낮추었다.
- **Sparsification 결합**: Random Sparsification과 결합했을 때, 2-bit 설정에서 1000배 이상의 압축률을 달성하면서도 안정적인 성능을 유지하였다.
- **양방향 압축**: 가중치와 그래디언트를 동시에 압축하는 시나리오에서, CosSGD는 4-bit 가중치 압축까지도 성공적으로 수행하며 타 방법 대비 우월한 통신 효율과 성능을 보였다.

## 🧠 Insights & Discussion

### 강점
본 논문은 단순한 코사인 함수를 이용하여 비선형 양자화를 구현함으로써, 계산 복잡도를 $O(m)$으로 유지하면서도 저비트 압축에서의 성능 저하 문제를 해결하였다. 특히, 기존 연구들이 데이터의 '분포(distribution)'에 집중했던 것과 달리 '중요도(magnitude)'에 집중하여 정밀도를 배분한 점이 주효했다.

### 한계 및 비판적 해석
- **데이터 의존성**: $b_\theta$를 설정할 때 top 1% 그래디언트를 clipping 하는 방식을 사용하는데, 이는 그래디언트 분포가 극단적으로 튀는 경우에 최적화된 설정이다. 만약 그래디언트 분포가 매우 균일하다면 본 방법의 이점이 희석될 가능성이 있다.
- **가정**: 본 논문은 "큰 magnitude의 값이 더 중요하다"는 가정을 전제로 한다. 이는 많은 딥러닝 연구에서 통용되는 직관이지만, 모든 신경망 아키텍처나 모든 최적화 문제에서 항상 성립하는지에 대한 일반화 가능성 검토가 더 필요하다.

## 📌 TL;DR

본 논문은 Federated Learning의 통신 병목 문제를 해결하기 위해 **코사인 함수 기반의 비선형 양자화 기법인 CosSGD**를 제안한다. 이 방법은 중요한 정보가 담긴 큰 값들을 더 정밀하게 양자화함으로써, 1~2비트의 극심한 압축 상황에서도 모델 성능을 float32 수준으로 유지한다. $O(m)$의 낮은 계산 복잡도와 적은 메모리 사용량 덕분에 리소스가 제한된 엣지 디바이스 환경에 매우 적합하며, 가중치와 그래디언트 모두에 적용 가능한 범용적인 통신 효율화 솔루션을 제공한다.