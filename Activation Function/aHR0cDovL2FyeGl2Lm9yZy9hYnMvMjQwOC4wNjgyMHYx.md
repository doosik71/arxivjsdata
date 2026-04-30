# Efficient Search for Customized Activation Functions with Gradient Descent

Lukas Strack, Mahmoud Safari, Frank Hutter (2024)

## 🧩 Problem to Solve

딥러닝 모델의 성능은 활성화 함수(activation function)의 선택에 의해 크게 좌우된다. 현재 ReLU, GELU, SiLU와 같은 표준 활성화 함수들이 널리 사용되고 있으나, 이는 일반적인 성능을 보장할 뿐 특정 모델이나 작업(task)에 최적화된 것은 아니다. 

특정 문제에 최적화된 맞춤형 활성화 함수를 설계하는 것은 매우 까다로운 작업이며, 기존의 자동화된 탐색 방법들은 수천 번의 함수 평가를 요구하는 블랙박스(black-box) 최적화나 진화 전략(evolutionary strategies)에 의존하여 계산 비용이 매우 높다는 한계가 있다.

본 논문의 목표는 모델의 가중치를 학습시키는 것과 유사한 수준의 낮은 비용으로, 특정 애플리케이션에 최적화된 고성능 활성화 함수를 효율적으로 찾아내는 gradient 기반의 탐색 방법을 제안하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 신경망 구조 탐색(Neural Architecture Search, NAS)의 gradient 기반 원샷(one-shot) 방법론을 활성화 함수 설계에 적용하는 것이다.

1. **활성화 함수 전용 탐색 공간 설계**: 기본적인 수학 연산(unary 및 binary operations)을 조합하여 새로운 활성화 함수를 생성할 수 있는 fine-grained search cell을 제안하였다.
2. **GRAFS (GRadient-based Activation Function Search) 제안**: DrNAS의 분포 학습(distribution learning) 개념을 도입하고, 활성화 함수 탐색의 특성에 맞게 Warmstarting, Unbounded operations 제약, Progressive shrinking 등의 기법을 추가하여 탐색의 안정성과 효율성을 극대화하였다.
3. **범용적 성능 향상 및 전이 가능성 입증**: ResNet(이미지 분류), ViT(비전 트랜스포머), GPT(언어 모델) 등 서로 다른 구조의 모델에서 맞춤형 활성화 함수가 기존 baseline보다 우수한 성능을 보임을 확인하였으며, 작은 모델에서 찾은 함수가 더 큰 모델이나 새로운 데이터셋에서도 효과적으로 작동하는 전이 가능성(transferability)을 입증하였다.

## 📎 Related Works

기존의 자동화된 활성화 함수 설계 연구는 크게 두 가지 방향으로 나뉜다.

1. **학습 가능한 활성화 함수 (Learnable Activations)**: Gradient descent를 사용하여 네트워크 가중치와 함께 활성화 함수의 파라미터를 동시에 학습하는 방식이다. Piecewise linear unit, 다항식 기저 함수(polynomial basis elements), Padé approximant 등이 사용되었다.
2. **NAS 기반 탐색 (NAS-style Search)**: 활성화 함수를 하이퍼파라미터로 간주하고 심볼릭(symbolic) 조합으로 표현하여 탐색하는 방식이다. 주로 RNN 컨트롤러를 이용한 강화학습이나 진화 전략을 사용한다. 하지만 이러한 방식은 수천 번의 함수 평가가 필요하여 계산 비용이 매우 높다.

본 논문은 심볼릭 조합이라는 NAS의 유연성과 gradient descent라는 최적화의 효율성을 결합하여, 기존의 블랙박스 방식보다 수십 배 이상 효율적인 탐색을 가능하게 함으로써 차별점을 가진다.

## 🛠️ Methodology

### 1. 활성화 함수 탐색 공간 (Search Space)
활성화 함수 $f$는 unary 연산과 binary 연산의 조합으로 구성된 계산 그래프(computational graph)로 정의된다. 
- **Unary operations**: $\sinh(x), \tanh(x), \text{arcsinh}(x), \arctan(x), \sqrt{x}, \text{erf}(x), \exp(x), \min(0, x), \max(0, x), \sigma(x), \text{GELU}(x), \text{SiLU}(x), \text{ELU}(x), \text{LeakyReLU}(x), \log(1+e^x)$ 등이 포함된다.
- **Binary operations**: $x_1+x_2, x_1-x_2, x_1 \cdot x_2, x_1 / x_2, \max(x_1, x_2), \min(x_1, x_2), \sigma(\gamma)x_1 + (1-\sigma(\gamma))x_2, \text{L}(x_1, x_2), \text{R}(x_1, x_2)$ 등이 포함된다.

이산적인 탐색 공간을 연속적으로 완화(continuous relaxation)하기 위해, 각 엣지와 정점에 가중합(weighted sum)을 할당한다.
- Unary 엣지 $(i, j)$의 출력: $\sum_u \upsilon_{u}^{(i,j)} u$
- Binary 정점 $i$의 출력: $\sum_b \beta_{b}^{(i)} b$
이때 가중치 $\upsilon, \beta$는 심플렉스(simplex) 제약 조건 $\sum \upsilon = 1, \sum \beta = 1$을 따른다.

### 2. GRAFS (GRadient-based Activation Function Search)
본 논문은 Bi-level optimization 구조를 통해 활성화 함수 파라미터 $\alpha = (\upsilon, \beta)$와 네트워크 가중치 $w$를 최적화한다.

$$
\min_{\alpha} \mathcal{L}_{\text{val}}(w^*(\alpha), \alpha) \quad \text{s.t.} \quad w^*(\alpha) = \arg\min_w \mathcal{L}_{\text{train}}(w, \alpha)
$$

탐색의 효율성과 안정성을 위해 다음의 네 가지 기법을 적용한다.

1. **Warmstarting**: 초기 단계에서 모델 가중치는 기존 활성화 함수(예: ReLU)로 업데이트하고, search cell의 파라미터만 외곽 루프에서 최적화하여 초기 설정값을 안정화한다.
2. **Constraining Unbounded Operations**: $\exp(x)$와 같이 출력이 무한히 커질 수 있는 연산이 gradient 폭주를 일으키는 것을 방지하기 위해, 출력값 $y$의 크기가 임계값 $\ell=10$을 넘으면 $y = \ell \cdot \text{sign}(y)$로 클리핑(clipping)한다.
3. **Progressive Shrinking**: 탐색 과정 중 가중치가 낮은 연산을 점진적으로 제거(drop)하는 로그 스케줄(log schedule)을 적용한다. 최종적으로는 각 위치에 단 하나의 연산만 남게 되어 자연스럽게 이산화(discretization)가 이루어진다.
4. **Variance Reduction Sampling**: DrNAS의 디리클레 분포(Dirichlet distribution) 샘플링 시, 네트워크 내의 각 활성화 셀마다 독립적인 샘플을 추출하여 샘플링으로 인한 분산을 줄인다.

## 📊 Results

### 1. 실험 설정
- **대상 모델**: ResNet20/32, ViT-tiny/small, mini/tiny/small GPT.
- **데이터셋**: CIFAR10, CIFAR100, SVHN Core, TinyStories.
- **평가 지표**: Test Accuracy (이미지), Test Loss (언어 모델).
- **비교 대상 (Baselines)**: ReLU, GELU, SiLU, ELU, LeakyReLU.

### 2. 주요 결과
- **ResNet (CIFAR10)**: 5개의 서로 다른 시드로 탐색한 결과, 발견된 5개 함수 모두 ReLU보다 우수한 성능을 보였으며, 그 중 2개는 모든 baseline을 능가하였다.
- **Vision Transformer (ViT)**: ViT-tiny에서 탐색된 모든 맞춤형 활성화 함수가 기존 baseline(GELU 포함)보다 일관되게 높은 성능을 기록하였다. 특히 더 큰 모델인 ViT-small에서도 이 성능 향상이 유지되었다.
- **GPT (TinyStories)**: miniGPT에서 탐색된 함수들이 GELU보다 낮은 test loss를 기록하였으며, 이는 tinyGPT와 smallGPT로 모델 크기를 키웠을 때도 동일하게 나타났다. 특히 $\text{ReLU}(x)^2$ 형태의 거동을 보이는 함수가 효과적임이 확인되었다.

### 3. 효율성 및 전이 가능성
- **탐색 비용**: 탐색에 소요되는 시간이 실제 모델을 한 번 평가하는 시간의 수 배(ResNet의 경우 2.2~4.1배, ViT의 경우 1배 미만)에 불과하여, 기존의 수천 번 평가가 필요했던 블랙박스 방식보다 압도적으로 효율적이다.
- **전이 가능성**: 작은 모델(ResNet20, ViT-tiny, miniGPT)에서 찾은 최적의 활성화 함수가 더 큰 모델(ResNet32, ViT-small, smallGPT)과 다른 데이터셋에서도 매우 잘 작동함을 확인하였다.

## 🧠 Insights & Discussion

본 연구는 gradient 기반의 NAS 기법을 활성화 함수 탐색에 성공적으로 이식하여, 매우 적은 비용으로 모델 맞춤형 활성화 함수를 찾을 수 있음을 보여주었다. 

**강점 및 시사점**:
- **실용적 효율성**: 탐색 오버헤드가 극히 적어 실제 딥러닝 파이프라인에 즉시 적용 가능하다.
- **구조적 인사이트**: 단순한 함수 교체가 아니라, 수학적 연산의 조합을 통해 새로운 형태의 비선형성을 발견할 수 있음을 시사한다. 특히 GPT 실험에서 $\text{ReLU}(x)^2$ 형태가 발견된 것은 기존의 진화적 탐색 결과와 일치하며, gradient 기반 방법의 유효성을 뒷받침한다.

**한계 및 논의 사항**:
- **최적 파이프라인의 미완성**: 저자들은 본 연구가 gradient 기반 탐색의 가능성을 보여준 첫 사례이며, 이것이 최적의 파이프라인은 아닐 수 있음을 명시하였다.
- **탐색 공간의 제약**: 제안된 search cell 내의 연산들로만 조합이 가능하므로, 이 공간 밖에 존재하는 완전히 새로운 형태의 함수는 찾을 수 없다.
- **Scaling Behavior**: 향후 연구로 더 큰 네트워크로 확장했을 때 활성화 함수의 성능이 어떻게 변하는지에 대한 scaling behavior 분석이 필요할 것으로 보인다.

## 📌 TL;DR

본 논문은 gradient 기반의 신경망 구조 탐색(NAS) 기법을 활용하여 특정 모델과 작업에 최적화된 활성화 함수를 효율적으로 찾는 **GRAFS** 방법론을 제안한다. 수학적 연산들의 조합으로 구성된 search cell을 정의하고, Bi-level optimization과 점진적 축소(Progressive shrinking) 기법을 통해 기존 블랙박스 탐색 방식보다 수십 배 빠른 속도로 고성능 활성화 함수를 찾아낸다. 실험 결과, ResNet, ViT, GPT 등 다양한 아키텍처에서 기존 표준 함수(ReLU, GELU 등)보다 우수한 성능을 보였으며, 작은 모델에서 찾은 함수가 큰 모델로 전이되는 특성을 확인하였다. 이 연구는 딥러닝 아키텍처 최적화 과정에서 활성화 함수를 자동으로 설계하는 매우 실용적인 경로를 제시한다.