# Zero-Shot Neural Architecture Search: Challenges, Solutions, and Opportunities

Guihong Li, Duc Hoang, Kartikeya Bhardwaj, Ming Lin, Zhangyang Wang, Radu Marculescu (2024)

## 🧩 Problem to Solve

딥러닝 모델의 성능을 최적화하기 위한 Neural Architecture Search (NAS)는 매우 유용하지만, 후보 아키텍처들을 실제로 학습시켜 성능을 검증해야 하는 과정에서 막대한 계산 비용(GPU 시간 및 탄소 배출)이 발생한다는 치명적인 문제가 있다. 이를 해결하기 위해 학습 과정 없이 네트워크의 성능을 예측하는 Zero-Shot NAS가 제안되었다.

본 논문은 기존의 Zero-Shot NAS 프록시(Proxy)들이 실제로 얼마나 효과적인지, 특히 실제 하드웨어 제약 조건이 포함된 Hardware-aware NAS 시나리오에서 어떻게 작동하는지를 종합적으로 분석하는 것을 목표로 한다. 특히, 많은 연구가 단순한 벤치마크에서만 성능을 입증했을 뿐, 실제 대규모 데이터셋이나 복잡한 하드웨어 환경에서의 효용성에 대한 분석이 부족하다는 점을 해결하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 단순히 기존 프록시들을 나열하는 것을 넘어, 이론적 배경을 분석하고 대규모 실험을 통해 그 한계를 명확히 규명했다는 점에 있다. 구체적인 기여 사항은 다음과 같다.

- **프록시의 체계적 분류 및 이론적 분석**: 기존의 Zero-Shot 프록시들을 Gradient-based와 Gradient-free로 분류하고, 각 프록시가 모델의 표현 능력(Expressive Capacity), 일반화 능력(Generalization Capacity), 학습 가능성(Trainability & Convergence) 중 어떤 측면을 측정하는지 이론적으로 분석하였다.
- **대규모 벤치마크를 통한 성능 검증**: ImageNet-1K, COCO, ADE20K와 같은 대규모 데이터셋과 다양한 비전 작업(분류, 검출, 세그멘테이션)에 걸쳐 프록시들의 상관관계를 분석하였다.
- **제약 조건부 탐색(Constrained Search)에서의 한계 발견**: 모든 아키텍처를 대상으로 할 때와 달리, 성능이 상위 5%인 고성능 네트워크들만을 대상으로 할 때는 기존 프록시들의 상관관계가 급격히 떨어진다는 근본적인 한계를 발견하였다.
- **Hardware-aware NAS 적용성 평가**: 실제 Edge-AI 디바이스(Jetson TX2 등)에서의 에너지 소비 및 지연 시간(Latency) 제약 하에서 Pareto-optimal 네트워크를 찾을 수 있는지 평가하였다.

## 📎 Related Works

기존의 NAS 접근 방식은 크게 세 가지로 분류된다.

1. **Multi-shot NAS**: 여러 후보 네트워크를 각각 완전히 학습시키는 방법으로, 비용이 매우 높다.
2. **One-shot NAS**: 하이퍼 네트워크(Hyper-network)를 구축하여 가중치를 공유(Weight-sharing)함으로써 탐색 비용을 줄이지만, 여전히 학습 과정이 필요하다.
3. **Zero-shot NAS**: 학습 없이 초기화 단계에서 프록시를 통해 성능을 예측하여 학습 비용을 완전히 제거한다.

기존의 Zero-shot NAS 관련 서베이들이 존재하지만, 본 논문은 이론적 근거를 깊게 분석하고, Vision Transformer(ViT)로의 확장 가능성을 탐색하며, 특히 하드웨어 인식(Hardware-awareness) 관점에서 대규모 실험을 수행했다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 1. Zero-Shot 프록시의 설계 원칙

이상적인 프록시는 다음 세 가지 요소를 모두 반영해야 한다.

- **Expressive Capacity**: 복잡한 데이터 패턴을 캡처할 수 있는 능력.
- **Generalization Capacity**: 학습되지 않은 데이터에 대해 일반화하는 능력.
- **Trainability and Convergence**: 효율적으로 최소 손실 값에 수렴하는 속도.

### 2. 주요 프록시 상세 설명

#### A. Gradient-based Proxies (기울기 기반)

기울기 기반 프록시는 역전파를 통해 파라미터나 활성화 함수의 기울기를 계산한다.

- **Gradient Norm**: 각 레이어 기울기 벡터의 $\ell_2$-norm의 합으로 계산한다.
$$G \triangleq \sum_{i=1}^{D} \|\nabla_{\theta_i} L\|_2$$
- **SNIP**: 파라미터 값과 해당 기울기의 내적의 절대값 합을 측정하여 연결성의 중요도를 평가한다.
$$\text{SNIP} \triangleq \sum_{i}^{D} |\langle \theta_i, \nabla_{\theta_i} L \rangle|$$
- **Synflow**: SNIP와 유사하나 절대값을 취하지 않고 부호를 유지한다.
$$\text{Synflow} \triangleq \sum_{i}^{D} \langle \theta_i, \nabla_{\theta_i} L \rangle$$
- **GraSP**: 1차 및 2차 미분(Hessian 행렬)을 모두 고려하여 파라미터 중요도를 측정한다.
$$\sum_{i}^{D} -\langle H_i \nabla_{\theta_i} L, \theta_i \rangle$$
- **GradSign**: 여러 샘플에 대해 기울기의 부호(sign)가 얼마나 일관된지를 측정하여 수렴 특성을 평가한다.
$$\text{GradSign} \triangleq \sum_{\theta_k \in \Theta} \left| \sum_{i=1}^{B} \text{sign}[\nabla_{\theta_k} L(f(x_i), y_i)] \right|$$
- **Jacobian Covariant (Jacobcov)**: 입력 데이터 $x$에 대한 출력 $y$의 야코비안 행렬의 공분산을 이용하여 모델의 표현력을 측정한다.
- **Zen-score**: 가우시안 랜덤 벡터를 이용해 모델의 Gaussian complexity를 측정함으로써 표현 능력을 평가한다.
- **NTK Condition Number (NTKCond)**: Neural Tangent Kernel의 고유값(eigenvalue) 비율 $\lambda_m / \lambda_1$을 통해 학습 역학의 균형 잡힌 수렴도를 측정하며, 값이 작을수록 성능이 좋은 경향이 있다.

#### B. Gradient-free Proxies (기울기 미사용)

기울기 계산조차 생략하여 효율성을 극대화한 방식이다.

- **Number of Linear Regions**: ReLU 네트워크가 입력 공간을 얼마나 많은 선형 영역으로 분할하는지를 측정하여 표현력을 평가한다.
- **Logdet**: 선형 영역의 해밍 거리(Hamming distance) 기반 행렬 $H$의 행렬식(determinant) 로그 값을 사용한다.
$$\text{Logdet} \triangleq \log|H|$$
- **NN-Mass**: 네트워크 토폴로지(연결성)를 분석하여 학습 가능성을 측정한다.
$$\text{NN-Mass} \triangleq \sum_{\text{each cell } c} \rho_c w_c d_c$$
여기서 $\rho_c$는 실제 연결된 skip-connection의 비율, $w_c$는 너비, $d_c$는 깊이다.

### 3. 하드웨어 성능 예측 모델

Hardware-aware NAS를 위해 지연 시간(Latency)을 예측하는 세 가지 모델을 비교하였다.

- **BRP-NAS**: GCN(Graph Convolution Network)이나 MLP를 사용하여 레이어 수준에서 예측한다.
- **HELP**: 하드웨어 정보(메모리 크기 등)를 추가 입력으로 사용하여 새로운 하드웨어에 대한 전이 가능성(Transferability)을 높였다.
- **NN-Meter**: 컴파일 과정에서 생성되는 커널(Kernel) 수준의 세밀한 분석을 통해 가장 높은 예측 정확도를 보였다.

## 📊 Results

### 1. 실험 설정 및 지표

- **데이터셋/벤치마크**: NASBench-101/201, NATS-Bench, TransNAS-Bench-101, ImageNet-1K, COCO, ADE20K.
- **평가 지표**: Spearman's $\rho$ (SPR)와 Kendall's $\tau$ (KT)를 사용하여 프록시 값과 실제 테스트 정확도 간의 상관관계를 측정하였다.

### 2. 주요 결과

- **단순 프록시의 강세**: 일반적인 탐색 공간(Unconstrained search space)에서는 놀랍게도 파라미터 수($\#Params$)와 연산량($\#FLOPs$)이라는 단순한 지표가 대부분의 정교한 Zero-shot 프록시보다 높은 상관관계를 보였다.
- **고성능 영역에서의 상관관계 붕괴**: 테스트 정확도 상위 5%인 네트워크들만 대상으로 분석했을 때, $\#Params$를 포함한 모든 프록시의 상관관계가 급격히 하락하였다. 이는 기존 프록시들이 '매우 좋은 모델'과 '최고의 모델'을 구분하는 능력이 부족함을 시사한다.
- **특정 구조에서의 유효성**: ResNet이나 MobileNet-v2와 같은 특정 네트워크 패밀리 내에서는 SNIP, Zen-score, NN-Mass 등이 $\#Params$보다 더 나은 성능을 보였다.
- **대규모 작업 결과**: ImageNet-1K 분류, COCO 검출, ADE20K 세그멘테이션 모두에서 $\#Params$와 $\#FLOPs$가 지배적인 성능을 보였으며, Zero-shot 프록시 기반 NAS는 One-shot NAS 대비 약 1% 미만의 성능 저하가 있었으나 탐색 비용은 수십 배 이상 절감하였다.
- **Hardware-aware NAS**: 에너지 제약이 엄격할 때는 대부분의 프록시가 잘 작동하지만, 제약이 완화되어 고성능 모델을 찾아야 하는 상황에서는 $\#Params, \#FLOPs$, 그리고 일부 프록시만이 실제 Pareto-optimal에 근접한 결과를 냈다.

## 🧠 Insights & Discussion

### 1. $\#Params$가 왜 잘 작동하는가?

본 논문은 단순한 파라미터 수가 잘 작동하는 이유를 세 가지 관점에서 설명한다.

- **표현 능력**: 일반적으로 파라미터가 많을수록 더 복잡한 함수를 표현할 수 있다.
- **일반화 능력**: 적절한 학습 설정 하에서 더 많은 파라미터는 더 높은 테스트 정확도로 이어지는 경향이 있다.
- **학습 가능성**: 동일 깊이에서 너비가 넓은(파라미터가 많은) 네트워크가 수렴 속도가 더 빠르고 학습이 용이하다.
즉, $\#Params$는 표현력과 학습 가능성을 동시에 암시적으로 캡처하는 반면, 기존의 정교한 프록시들은 이 중 어느 한 가지 측면에만 집중하는 경향이 있다.

### 2. 한계 및 향후 방향

- **통합적 지표의 필요성**: 표현 능력, 일반화 능력, 학습 가능성을 모두 동시에 고려하는 새로운 프록시 설계가 필요하다.
- **벤치마크의 다양성 부족**: 현재의 NAS 벤치마크는 대부분 Cell-based 구조에 치중되어 있어, 실제 산업계에서 쓰이는 Inverted Bottleneck과 같은 더 실용적인 탐색 공간을 포함해야 한다.
- **맞춤형 프록시**: 모든 공간에서 작동하는 범용 프록시보다는, 탐색 공간을 하위 공간으로 나누고 각 영역에 특화된 맞춤형 프록시를 적용하는 전략이 유효할 수 있다.

## 📌 TL;DR

본 논문은 Zero-Shot NAS의 다양한 프록시들을 이론적으로 분석하고 대규모 실험을 통해 검증하였다. 실험 결과, 일반적인 상황에서는 단순한 파라미터 수($\#Params$)가 가장 강력한 예측 지표였으며, 기존의 정교한 프록시들은 고성능 모델들 사이의 미세한 성능 차이를 구분하는 데 한계가 있음이 밝혀졌다. 이는 향후 Zero-Shot NAS 연구가 모델의 표현력, 일반화력, 학습 가능성을 통합적으로 측정할 수 있는 방향으로 나아가야 함을 시사한다.
