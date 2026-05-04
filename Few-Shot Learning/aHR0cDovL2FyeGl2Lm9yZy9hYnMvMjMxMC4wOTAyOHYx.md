# Subspace Adaptation Prior for Few-Shot Learning

Mike Huisman, Aske Plaat, Jan N. van Rijn (2023)

## 🧩 Problem to Solve

본 논문은 Few-Shot Learning 상황에서 Gradient-based meta-learning 기법들이 겪는 과적합(overfitting) 문제와 학습 효율성 저하 문제를 해결하고자 한다. 기존의 MAML과 같은 알고리즘들은 새로운 태스크를 학습할 때 네트워크의 모든 학습 가능 파라미터를 업데이트한다. 그러나 태스크당 제공되는 데이터가 극히 제한적인 Few-Shot 설정에서 모든 파라미터를 조정하는 것은 데이터의 노이즈까지 학습하게 만들어 과적합을 유발할 가능성이 크며, 특정 태스크 분포에 최적화된 효율적인 학습 전략을 반영하지 못한다는 한계가 있다.

따라서 본 연구의 목표는 모든 파라미터를 업데이트하는 대신, 각 레이어에서 어떤 파라미터 서브스페이스(subspace)를 조정하는 것이 가장 효율적인지를 메타 학습함으로써, 일반화 성능을 높이고 과적합 위험을 줄이는 새로운 알고리즘을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Subspace Adaptation Prior (SAP)**라는 알고리즘을 통해 좋은 초기화 파라미터(prior knowledge)와 더불어, 각 레이어별로 적응시켜야 할 파라미터 서브스페이스를 '연산 서브셋(operation subsets)'의 형태로 함께 학습하는 것이다.

단순히 고정된 서브스페이스를 사용하는 것이 아니라, DARTS와 유사하게 다양한 후보 연산(candidate operations)들의 가중치 조합을 학습함으로써, 주어진 태스크 분포에 따라 어떤 연산 조합을 조정하는 것이 빠른 적응에 유리한지를 스스로 찾아내도록 설계하였다. 이는 일종의 정규화(regularization) 역할을 하며, 네트워크가 문제의 내재적 구조(예: 사인파의 위상 변화 $\rightarrow$ 입력 shift)를 파악하여 효율적으로 적응할 수 있게 한다.

## 📎 Related Works

본 논문은 다음과 같은 관련 연구들을 기반으로 하며, 그 차별점을 제시한다.

1. **Optimization-based Meta-learning**: MAML이나 Reptile과 같이 초기화 파라미터를 학습하는 방식이다. SAP는 MAML의 구조를 따르지만, 모든 파라미터가 아닌 선택된 서브스페이스만을 업데이트한다는 점에서 차별화된다.
2. **Neural Architecture Search (NAS)**: DARTS는 후보 연산들 중 최적의 아키텍처를 찾는다. SAP는 DARTS의 가중치 그래프 개념을 차용하여 레이어별 연산 서브셋을 찾지만, 테스트 타임에 아키텍처를 고정하고 특정 파라미터 서브스페이스 내에서만 gradient descent를 수행한다는 점이 다르다.
3. **Gradient Modulation**: T-Net이나 Warp-MAML은 implicit gradient modulation을 통해 손실 곡면을 변형(warp)하여 빠른 수렴을 돕는다. SAP는 이러한 구조를 포함하고 있어, 서브스페이스를 결정함과 동시에 해당 서브스페이스 내에서의 그래디언트 흐름을 최적화하는 효과를 얻는다.

## 🛠️ Methodology

### 전체 시스템 구조

SAP는 기본 네트워크(base-learner)의 각 레이어 $W_\ell$ 앞에 후보 연산 집합 $O_\ell$을 삽입한 구조를 가진다. 네트워크의 최종 출력은 다음과 같이 표현된다.

$$f_{\Theta}(x) = O_{L+1} W_L O_L \sigma(\dots \sigma(W_2 O_2 \sigma(W_1 O_1 x)))$$

여기서 $\Theta = \{\theta, \phi, \lambda\}$이며, $\theta$는 base-learner의 가중치, $\phi$는 각 후보 연산의 파라미터, $\lambda$는 각 연산의 활성화 강도(activation strength)를 의미한다.

### 주요 구성 요소 및 연산

각 레이어의 연산 층 $O_\ell$은 여러 후보 연산 $o_{\ell,i}$들의 convex combination으로 구성된다.

$$O_\ell z_\ell = \sum_{i=1}^{n_\ell} w_{\ell,i} o_{\ell,i}(z_\ell)$$

이때 $\sum w_{\ell,i} = 1$이며, 후보 연산으로는 Identity, Matrix multiplication, SVD-matrix multiplication, Element-wise scale, Scalar scale, Vector shift, Scalar shift 등이 포함된다. 이러한 연산들은 모두 full-rank matrix multiplication의 부분 집합으로 볼 수 있으므로, 전체 네트워크의 표현력(expressivity)은 기존 네트워크와 동일하게 유지된다.

### 학습 절차 및 손실 함수

학습은 Inner-loop와 Outer-loop의 두 단계로 진행된다.

1. **Inner-loop (Task Adaptation)**: 새로운 태스크 $T_j$가 주어지면, base-learner 파라미터 $\theta$와 활성화 강도 $\lambda$는 고정하고, 오직 후보 연산 파라미터 $\phi$만을 업데이트한다.
    $$\phi^{(t+1)}_j \leftarrow \phi^{(t)} - \alpha \nabla_{\phi^{(t)}} L_{T_j}(\theta, \phi^{(t)}, \lambda)$$
2. **Outer-loop (Meta-update)**: 여러 태스크에 대해 inner-loop를 수행한 후, 쿼리 셋(query set)에서의 성능을 최대화하도록 초기 파라미터 $\Theta$를 업데이트한다.
    $$\Theta \leftarrow \Theta - \beta \nabla_{\Theta} \sum_{T_j \in B} L_{T_j}(\theta, \phi^{(T)}_j, \lambda)$$

### Gradient Modulation의 역할

SAP는 $W_\ell$와 $O_\ell$를 분리하여 배치함으로써 implicit gradient modulation 효과를 얻는다. 연산 파라미터 $\phi$에 대한 그래디언트가 계산될 때, 고정된 $W_\ell$를 통과하며 다음과 같이 변형된다.

$$\Delta v = v_{new} - v = -\alpha (W \nabla_O L_{T_j}) x$$

즉, $W$가 그래디언트를 warp하여 gradient descent가 더 빠르게 최적해에 도달할 수 있도록 돕는다.

## 📊 Results

### 실험 설정

- **데이터셋**: Sine wave regression, miniImageNet, tieredImageNet, CUB.
- **지표**: Mean Squared Error (MSE, 회귀), Accuracy (분류).
- **비교 대상**: MAML, T-Net, MT-Net, Warp-MAML.

### 주요 결과

1. **Sine Wave Regression**: SAP는 모든 베이스라인보다 일관되게 낮은 MSE를 기록하였다. 특히 $T=1$일 때 SAP의 성능 우위가 두드러졌다.
2. **Few-Shot Image Classification**: miniImageNet과 tieredImageNet에서 SAP는 베이스라인 대비 약 $1.1\% \sim 3.3\%$의 정확도 향상을 보였다.
3. **Cross-Domain**: miniImageNet $\rightarrow$ CUB 설정에서 Warp-MAML과 대등하거나 더 우수한 성능을 기록하여, 학습되지 않은 도메인에서도 서브스페이스 적응이 유효함을 입증하였다.
4. **서브스페이스 분석**: 학습된 활성화 강도 $\lambda$를 분석한 결과, 고차원 연산(Conv 3x3 등)보다 저차원 연산(Scalar shift, Vector shift, MTL scale)에 더 높은 가중치가 부여되었다. 이는 제한된 데이터 환경에서 저차원 서브스페이스를 조정하는 것이 과적합을 방지하고 효율적임을 시사한다.

## 🧠 Insights & Discussion

### 강점 및 분석

SAP의 가장 큰 강점은 **학습 가능한 정규화(learnable regularization)**를 구현했다는 점이다. 모든 파라미터를 업데이트하는 대신 태스크 분포에 적합한 최소한의 서브스페이스만을 선택적으로 업데이트함으로써, Few-Shot 환경의 고질적인 문제인 과적합을 효과적으로 억제하였다. 특히 사인파 실험에서 입력 shift와 출력 scale 연산이 중요하게 선택된 것은, 모델이 문제의 수학적 구조를 메타 학습을 통해 파악했음을 보여준다.

### 한계 및 비판적 해석

1. **계산 복잡도**: 메타 학습 단계에서 second-order gradient를 계산해야 하므로 파라미터 수 $N$에 대해 $O(N^2)$의 메모리 비용이 발생한다. first-order approximation을 사용할 경우 메모리 문제는 해결되나, 성능이 $0.2\% \sim 7.3\%$ 가량 하락하는 트레이드오프가 존재한다.
2. **수동 설계된 후보군**: 후보 연산 집합 $O_\ell$을 연구자가 수동으로 정의했다는 점은 한계로 지적될 수 있다. 데이터로부터 최적의 연산 형태를 스스로 발견하는 구조로 확장될 필요가 있다.
3. **파라미터 증가**: 동일한 표현력을 가짐에도 불구하고 구조적 분리로 인해 전체 학습 가능 파라미터 수가 증가한다.

## 📌 TL;DR

본 논문은 모든 파라미터를 업데이트하는 기존 gradient-based meta-learning의 과적합 문제를 해결하기 위해, **어떤 파라미터 서브스페이스를 업데이트할지 함께 학습하는 SAP(Subspace Adaptation Prior)**를 제안한다. SAP는 후보 연산들의 가중치 조합을 통해 최적의 적응 경로를 찾으며, 특히 저차원 연산(shift, scale) 위주의 적응이 Few-Shot 학습의 성능을 향상시킨다는 것을 실험적으로 증명하였다. 이 연구는 향후 딥러닝 모델이 더 적은 데이터로 빠르게 적응하기 위해 '무엇을 수정해야 하는가'에 대한 구조적 가이드를 제공하는 데 중요한 기여를 한다.
