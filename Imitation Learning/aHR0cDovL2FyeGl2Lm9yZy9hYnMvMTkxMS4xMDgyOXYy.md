# Neural Random Forest Imitation

Christoph Reinders, Bodo Rosenhahn (2024)

## 🧩 Problem to Solve

본 논문은 Random Forest(RF)의 강점과 Neural Network(NN)의 강점을 결합하고자 한다. 일반적으로 Neural Network는 복잡한 데이터 모델링에 탁월하지만, 학습을 위해 막대한 양의 레이블링된 데이터가 필요하다는 치명적인 단점이 있다. 반면, Random Forest는 여러 결정 트리의 앙상블 구조 덕분에 데이터가 매우 적은 상황에서도 오버피팅(Overfitting)에 강하며 우수한 성능을 보인다. 하지만 Random Forest는 미분 불가능(Non-differentiable)하므로 경사 하강법(Gradient-based optimization)을 통한 미세 조정(Fine-tuning)이 불가능하다.

기존 연구들은 Random Forest를 Neural Network로 변환하기 위해 결정 트리의 노드를 뉴런으로 직접 매핑하는 Direct Mapping 방식을 사용하였다. 그러나 이 방식은 트리의 깊이가 증가함에 따라 노드 수가 지수적으로 증가하여 네트워크 파라미터 수가 방대해지며, 많은 가중치가 0으로 설정되어 매우 비효율적인 구조를 생성한다. 결과적으로 기존의 직접 매핑 방식은 복잡한 Random Forest 모델에 적용하기 어렵다는 한계가 있다. 따라서 본 논문의 목표는 Random Forest의 결정 경계를 효율적으로 학습하여, 파라미터 수를 획기적으로 줄이면서도 미분 가능하고 최적화 가능한 효율적인 Neural Network로 변환하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Direct Mapping 대신 **Imitation Learning(모방 학습)** 접근 방식을 사용하는 것이다. Random Forest를 직접 변환하는 것이 아니라, Random Forest의 행동을 모방하도록 하는 학습 데이터를 생성하고, 이를 통해 Neural Network를 학습시키는 '암시적 변환(Implicit Transformation)' 방식을 제안한다.

주요 기여 사항은 다음과 같다.
- Random Forest로부터 입력-타겟 쌍(Input-target pairs)을 생성하여 Neural Network가 RF의 결정 경계를 학습하게 하는 새로운 모방 학습 프레임워크를 제안한다.
- 결정 트리의 리프 노드에서 루트 노드로 정보를 전파하고, 이를 기반으로 타겟 클래스에 부합하는 샘플을 생성하는 Guided Routing 기법을 도입하였다.
- 직접 매핑 방식과 달리, 제안 방법은 복잡하고 깊은 Random Forest 모델에 대해서도 확장 가능하다.
- 생성된 모델은 완전히 미분 가능하므로, 추가적인 End-to-End 미세 조정이나 특성 추출(Feature Extraction) 파이프라인에 통합하여 사용할 수 있는 Warm Start 지점을 제공한다.

## 📎 Related Works

기존의 Random Forest $\to$ Neural Network 변환 연구들은 주로 다음과 같은 방식을 취했다.
- **Sethi (1990):** 결정 트리의 각 분할 노드(Split node)와 리프 노드(Leaf node)에 각각 뉴런을 할당하는 2-은닉층 네트워크 매핑 방식을 제안하였다.
- **Welbl (2014) & Biau et al. (2019):** 유사한 매핑 전략을 사용하며, 이후 역전파(Backpropagation)를 통해 네트워크를 미세 조정하는 방식을 제안하였다. 이들은 개별 트리 네트워크를 학습시키는 Independent training과 모든 트리를 하나의 네트워크로 통합하는 Joint training 방식을 구분하여 다루었다.
- **Massiceti et al. (2017):** 결정 트리를 여러 개의 서브트리(Subtree)로 나누어 매핑함으로써 파라미터 수를 줄이는 네트워크 분할 전략을 도입하였다.

**기존 방식의 한계 및 차별점:**
위의 방법들은 모두 트리의 구조를 물리적으로 뉴런의 연결 구조로 옮기는 방식이다. 이로 인해 트리의 깊이가 깊어지면 파라미터 수가 기하급수적으로 증가하여 메모리 소비가 극심해지며, 실제 GPU 연산 시 효율성이 떨어진다. 반면, 본 논문의 NRFI(Neural Random Forest Imitation)는 RF를 '함수'로서 취급하여 그 결과값을 모방하는 데이터를 생성하므로, RF의 복잡도와 관계없이 사용자가 정의한 효율적인 NN 구조(예: 단순한 MLP)를 사용할 수 있다는 점에서 근본적인 차이가 있다.

## 🛠️ Methodology

NRFI의 전체 파이프라인은 **데이터 생성 $\to$ 모방 학습**의 단계로 구성된다.

### 1. 데이터 생성 (Data Generation)
Random Forest로부터 학습 데이터를 생성하는 과정은 다음과 같다.

**가. 클래스 가중치 전파 (Bottom-up Propagation):**
먼저 각 결정 트리의 리프 노드에서 저장된 클래스 확률 $P^{leaf}(l)$을 루트 노드 방향으로 합산하여 모든 노드 $n$에 대한 클래스 가중치 $W(n)$을 계산한다.
$$W(n) = \begin{cases} P^{leaf}(n) & \text{if } n \in N_{leaf} \\ W(c_{left}(n)) + W(c_{right}(n)) & \text{if } n \in N_{split} \end{cases}$$

**나. 가이드 라우팅 기반 샘플 생성 (Top-down Guided Routing):**
타겟 클래스 $t$에 해당하는 샘플 $x$를 생성하기 위해 루트 노드에서 시작하여 리프 노드까지 내려가며 데이터를 수정한다.
1. 현재 노드 $n$에서 타겟 클래스 $t$에 대한 왼쪽 자식과 오른쪽 자식의 가중치 $w_{left}^t, w_{right}^t$를 확인하고 이를 정규화하여 $\hat{w}_{left}, \hat{w}_{right}$를 구한다.
2. 이 확률에 따라 다음 노드 $n_{next}$를 무작위로 선택한다.
3. 선택된 노드로 이동하기 위해 현재 샘플 $x$의 특징값 $x_{f(n)}$이 분할 임계값 $\theta(n)$을 만족하는지 확인한다.
4. 만약 조건을 만족하지 않는다면, 다음과 같이 해당 범위 내에서 새로운 값을 무작위로 샘플링하여 값을 강제로 수정한다.
   - 왼쪽 자식 선택 시: $x_{f(n)} \sim U(f_{min, f(n)}, \theta(n))$
   - 오른쪽 자식 선택 시: $x_{f(n)} \sim U(\theta(n), f_{max, f(n)})$
5. 이 과정을 리프 노드에 도달할 때까지 반복한다.

**다. Random Forest로의 확장 및 신뢰도 분포 최적화:**
단일 트리가 아닌 RF 전체를 위해, 여러 개의 트리 부분집합 $RF_{sub}$를 무작위 순서로 처리하여 샘플을 생성한다. 또한, 데이터의 다양성을 위해 생성되는 샘플의 신뢰도(Confidence)가 균등하게 분포하도록 다음의 최적화 문제를 통해 트리 선택 가중치 $w_D$를 결정한다.
$$\min_{w_D} \left\| \begin{bmatrix} \sum_{j=1}^{n_T} w_{D_j} h_j^1 & \dots & \sum_{j=1}^{n_T} w_{D_j} h_j^H \end{bmatrix}^T - \begin{bmatrix} 1 \\ \dots \\ 1 \end{bmatrix} \right\|^2 \quad \text{s.t. } \forall j, 0 \le w_{D_j}$$
여기서 $h_j^i$는 $j$개의 트리를 사용하여 생성한 데이터의 $i$번째 빈(bin)에 해당하는 히스토그램 값이다.

### 2. 모방 학습 (Imitation Learning)
생성된 입력-타겟 쌍 $(x, y)$를 사용하여 Neural Network를 학습시킨다.
- **아키텍처:** ReLU 활성화 함수를 가진 하나 이상의 은닉층을 포함한 Fully Connected Network를 사용하며, 출력층에는 Softmax를 적용한다.
- **학습 절차:** Random Forest가 예측한 결과 $y = RF(x)$를 정답으로 하여 Cross-Entropy 손실 함수를 통해 학습한다. 오버피팅을 방지하기 위해 학습 데이터를 미리 저장하지 않고 On-the-fly 방식으로 생성하여 매번 고유한 샘플을 제공한다.

## 📊 Results

### 실험 설정
- **데이터셋:** UCI Machine Learning Repository의 9개 분류 데이터셋 (Car, Covertype, Iris 등).
- **설정:** 클래스당 학습 샘플 수를 5, 10, 20, 50개로 제한하여 소량 데이터 상황을 시뮬레이션하였다.
- **비교 대상:** DT, SVM, RF, 일반 NN, 그리고 직접 매핑 방식인 Sethi, Welbl, Massiceti 방법론.
- **지표:** 테스트 정확도(Accuracy) 및 네트워크 파라미터 수.

### 주요 결과
- **정확도:** NRFI는 생성된 데이터만으로 학습했을 때 RF 정확도의 약 99.18%에 도달하였으며, 원본 데이터와 생성 데이터를 혼합하여 학습한 $\text{NRFI(gen+ori)}$는 많은 경우 RF와 비슷하거나 오히려 더 높은 정확도를 보였다. 특히 데이터가 매우 적은 상황(클래스당 5~20개)에서 기존 방법들보다 우수한 성능을 나타냈다.
- **효율성:** 파라미터 수에서 압도적인 차이를 보였다. Direct Mapping 방식(Sethi, Welbl 등)이 수십만에서 수백만 개의 파라미터를 필요로 하는 반면, NRFI는 동일하거나 더 높은 성능을 내면서도 단 **2,676개**의 파라미터(은닉층 뉴런 32개 기준)만으로 구현 가능했다.
- **데이터 생성 전략:** 무작위 생성보다 제안된 NRFI Dynamic 샘플링 방식이 신뢰도 분포를 더 균등하게 생성하며, 이는 최종 모델의 정확도 향상으로 이어졌다.

## 🧠 Insights & Discussion

본 연구의 가장 큰 성과는 Random Forest의 '지식'을 데이터 형태로 추출하여 Neural Network에 전이시킴으로써, 구조적 제약 없이 모델의 크기를 획기적으로 줄였다는 점이다.

**강점 및 해석:**
- **일반화 능력:** 단순한 모방을 넘어, 생성된 데이터를 통해 학습한 NN이 RF보다 약간 더 높은 정확도를 보이는 경우가 있는데, 이는 NN의 연속적인 결정 경계가 RF의 계단식 결정 경계보다 더 나은 일반화(Generalization) 성능을 제공하기 때문으로 해석된다.
- **확장성:** 트리의 깊이나 노드 수에 영향을 받지 않으므로 매우 복잡한 RF 모델도 작은 NN으로 압축할 수 있다.

**한계 및 논의사항:**
- **데이터 생성 비용:** 학습 과정에서 데이터를 On-the-fly로 생성해야 하므로, 학습 초기 단계의 계산 시간이 늘어날 수 있다.
- **가정:** 본 논문은 주로 분류(Classification) 작업에 집중하였으나, 저자는 회귀(Regression) 작업으로의 확장 가능성을 언급하였다. 다만 실제 회귀 작업에서의 성능 검증은 본문에 명시되지 않았다.

## 📌 TL;DR

본 논문은 Random Forest를 Neural Network로 변환하기 위해, RF의 결정 경계를 모방하는 학습 데이터를 생성하고 이를 NN이 학습하게 하는 **Neural Random Forest Imitation (NRFI)** 방법을 제안한다. 이 방식은 기존의 직접 매핑 방식이 가졌던 파라미터 폭증 문제를 완전히 해결하여, **파라미터 수를 수백만 개에서 수천 개 수준으로 줄이면서도 RF와 대등하거나 더 뛰어난 정확도를 달성**한다. 특히 데이터가 극히 적은 환경에서 RF의 강점과 NN의 미분 가능성을 동시에 확보할 수 있어, 향후 효율적인 모델 압축 및 End-to-End 파이프라인 구축에 중요한 역할을 할 것으로 기대된다.