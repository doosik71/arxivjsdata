# Buffer-based Gradient Projection for Continual Federated Learning

Shenghong Dai, Jy-yong Sohn, Yicong Chen, S M Iftekharul Alam, Ravikumar Balakrishnan, Suman Banerjee, Nageen Himayat, Kangwook Lee (2024)

## 🧩 Problem to Solve

본 논문은 **Continual Federated Learning (CFL)** 환경에서 발생하는 **Catastrophic Forgetting (치명적 망각)** 문제를 해결하고자 한다. CFL은 분산된 여러 클라이언트가 연속적인 데이터 스트림으로부터 적응적으로 학습하는 환경을 의미하며, 새로운 정보를 학습할 때 이전에 습득한 지식을 소실하는 문제가 핵심적인 장애물이다.

특히 기존의 CFL 접근 방식들은 다음과 같은 현실적인 제약 조건들로 인해 한계가 있다:
1. **제한된 저장 용량**: 엣지 디바이스의 저장 공간 부족으로 인해 이전 태스크의 데이터를 충분히 저장할 수 없다.
2. **데이터의 비균질성 (Non-IID)**: 클라이언트 간의 데이터 분포가 달라, 로컬 수준에서 망각을 방지하더라도 이를 집계(aggregate)했을 때 전역적인 망각 방지 효과가 나타나지 않는다.
3. **태스크 경계(Task Boundary)에 대한 의존성**: 많은 기존 알고리즘들이 새로운 태스크가 언제 시작되는지에 대한 명시적인 정보(task boundary)가 있다는 비현실적인 가정을 전제로 한다.

따라서 본 논문의 목표는 태스크 경계에 대한 정보 없이도(General Continual Learning 설정), 적은 메모리 사용량과 통신 오버헤드로 전역적인 망각을 효과적으로 억제하는 **Fed-A-GEM** 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Global Buffer Gradient**와 **Local Gradient Projection**의 결합이다.

- **Global Buffer Gradient**: 각 클라이언트가 자신의 로컬 버퍼 데이터를 사용하여 전역 모델에 대한 그래디언트를 계산하고, 서버가 이를 평균 내어 전역 버퍼 그래디언트를 생성한다. 이는 개별 클라이언트의 제한된 데이터가 아닌, 전체 네트워크가 경험한 과거 지식의 방향성을 대표하는 참조점 역할을 한다.
- **Local Gradient Projection**: 클라이언트가 로컬 모델을 업데이트할 때, 현재 태스크의 그래디언트 방향이 전역 버퍼 그래디언트 방향과 충돌(angle $> 90^\circ$)하는지 확인한다. 충돌이 발생할 경우, 전역 버퍼 그래디언트에 수직인 방향으로 그래디언트를 투영(projection)하여 과거의 지식을 훼손하지 않으면서 새로운 지식을 학습하게 한다.

## 📎 Related Works

### 기존 연구 및 한계
1. **Continual Learning (CL)**: Regularization-based (EWC 등), Architecture-based (Progressive NN 등), Replay-based (GEM, A-GEM 등) 방법론이 존재한다. 특히 Replay-based 방법은 버퍼를 통해 과거 데이터를 재사용하여 성능이 좋지만, FL 환경에서는 저장 공간과 Non-IID 문제가 발생한다.
2. **Federated Learning (FL)**: 데이터 프라이버시를 유지하며 협업 학습을 가능케 하는 FedAvg 등이 대표적이다. 그러나 대부분의 FL 연구는 데이터 분포가 시간에 따라 변하지 않는 정적 상황을 가정한다.
3. **Continual Federated Learning (CFL)**: FedCurv, FedProx, CFeD, GLFC, FOT 등이 제안되었다. 하지만 이들 중 상당수는 명시적인 태스크 경계가 필요하거나, 대리 데이터셋(surrogate dataset) 생성에 많은 자원이 소모되며, 통신 오버헤드가 크다는 단점이 있다.

### 차별점
Fed-A-GEM은 태스크 경계 정보 없이도 동작하는 **General Continual Learning** 원칙을 FL에 통합하였다. 또한, 서버 측에서 그래디언트를 투영하는 FOT와 달리 클라이언트 측에서 투영을 수행하며, 복잡한 구조 없이 단순한 그래디언트 투영만으로 기존 CFL 기법들의 성능을 보완할 수 있는 플러그인 형태로 설계되었다.

## 🛠️ Methodology

### 전체 시스템 구조
Fed-A-GEM의 동작 과정은 다음과 같은 파이프라인으로 구성된다.
1. **모델 집계**: 서버가 클라이언트들로부터 로컬 모델 $w_k$를 수집하여 전역 모델 $w$를 생성한다.
2. **전역 버퍼 그래디언트 계산**: 각 클라이언트는 자신의 로컬 버퍼 $M^k$를 사용하여 전역 모델 $w$에 대한 로컬 버퍼 그래디언트 $g^k_{ref}$를 계산한다. 서버는 이를 Secure Aggregation(SecAgg)을 통해 평균 내어 전역 버퍼 그래디언트 $g_{ref}$를 산출한다.
3. **로컬 업데이트 및 투영**: 클라이언트는 현재 태스크 데이터로 그래디언트 $g$를 계산하고, 이를 $g_{ref}$를 기준으로 투영하여 수정된 그래디언트 $\tilde{g}$를 얻어 모델을 업데이트한다.
4. **버퍼 업데이트**: Reservoir Sampling을 통해 로컬 버퍼 $M^k$를 최신 데이터로 갱신한다.

### 주요 방정식 및 절차

#### 1. 그래디언트 투영 (Gradient Projection)
현재 배치 데이터의 그래디언트 $g$와 전역 버퍼 그래디언트 $g_{ref}$ 사이의 내적이 0 이하(즉, 각도가 $90^\circ$ 이상)일 때, 다음과 같이 그래디언트를 수정한다.

$$\tilde{g} = g - \text{proj}_{g_{ref}} g \cdot \mathbb{1}(g^\top g_{ref} \le 0)$$

여기서 $\text{proj}_{g_{ref}} g$는 $g$를 $g_{ref}$ 방향으로 투영한 성분으로, 다음과 같이 정의된다.

$$\text{proj}_{g_{ref}} g = \frac{g^\top g_{ref}}{\|g_{ref}\|^2} g_{ref}$$

이 수식의 의미는 현재 학습 방향 $g$가 과거 지식을 유지하는 방향 $g_{ref}$와 충돌할 때, 충돌하는 성분을 제거하여 **과거 지식에 직교하는 방향**으로만 업데이트를 수행하겠다는 것이다.

#### 2. Reservoir Sampling
태스크 경계 없이도 데이터 스트림에서 균일하게 샘플을 유지하기 위해 Reservoir Sampling을 사용한다.
- 버퍼 크기가 $|M^k|$일 때, $n$번째 들어오는 데이터는 $\frac{|M^k|}{n}$의 확률로 버퍼 내 기존 샘플 중 하나를 대체한다.
- 이를 통해 모든 데이터가 버퍼에 포함될 확률을 동일하게 유지하여 태스크 간 균형 잡힌 대표성을 확보한다.

## 📊 Results

### 실험 설정
- **데이터셋**: rotated-MNIST, permuted-MNIST (Domain-IL), sequential-CIFAR10/100 (Class-IL, Task-IL), sequential-YahooQA (Text), CARLA (Object Detection).
- **비교 대상**: FL (FedAvg), FL+CL (A-GEM, DER, iCaRL, L2P), CFL (FedCurv, FedProx, CFeD, GLFC, FOT).
- **평가 지표**: 평균 정확도 ($\text{Acc}_T$), 망각도 ($\text{Fgt}_T$), Backward Transfer (BWT), Forward Transfer (FWT).

### 주요 결과
1. **성능 향상**: Fed-A-GEM을 기존 방법론에 결합했을 때 거의 모든 시나리오에서 정확도가 향상되었다. 특히 sequential-CIFAR100 Task-IL 설정에서 정확도를 최대 **27%**까지 끌어올렸다.
2. **단순 결합의 강력함**: 단순한 FL(FedAvg)에 Fed-A-GEM만 추가한 모델이 복잡한 CFL 기법인 GLFC나 CFeD보다 더 높은 성능을 보였다.
3. **태스크 경계 불필요**: 태스크 경계 정보가 필요한 iCaRL, FOT 등과 달리, 경계 정보 없이도 경쟁력 있는 성능을 달성하였다.
4. **강건성 검증**: 
    - **버퍼 크기**: 버퍼 크기가 커질수록 성능이 향상되지만, $B=200$과 같은 매우 작은 크기에서도 baseline 대비 유의미한 성능 향상을 보였다.
    - **비동기 환경**: 클라이언트들이 서로 다른 시점에 태스크를 완료하는 비동기 상황에서도 성능이 유지되거나 오히려 향상되는 경향을 보였다.
    - **확장성**: 클라이언트 수를 $K=20$으로 늘리거나, Tiny-ImageNet과 같은 더 큰 데이터셋에서도 일관된 성능 향상을 입증하였다.

## 🧠 Insights & Discussion

### 강점 및 분석
- **효율적인 지식 보존**: 전역 버퍼 그래디언트를 참조점으로 사용함으로써, 개별 클라이언트의 local buffer가 가진 편향성을 극복하고 네트워크 전체의 집단적 경험을 활용하여 망각을 방지하였다.
- **낮은 오버헤드**: 그래디언트 투영 연산은 $O(P)$ (P는 파라미터 수)의 복잡도를 가지며, 전체 학습 시간 증가분은 약 7.2%로 매우 적다. 통신 횟수를 조절함으로써 추가적인 통신 비용 문제도 완화 가능하다.
- **Ablation Study 결과**: 
    - 그래디언트 수정 방식 중 'Project' 방식이 'Average'나 'Rotate'보다 우수하며, 특히 충돌 상황($>90^\circ$)에서만 적용하는 조건부 투영이 가장 효과적임을 확인하였다.
    - 버퍼 업데이트 전략 중 Reservoir Sampling이 가장 균형 잡힌 샘플 분포를 제공하여 성능이 높았다.

### 한계 및 향후 과제
- **버퍼 유지 비용**: 망각을 억제하기 위해 최소한의 버퍼를 유지해야 하며, 이는 저장 공간이 극도로 제한된 디바이스에서는 여전히 부담이 될 수 있다.
- **Secure Aggregation 비용**: 프라이버시 보호를 위한 SecAgg 도입 시 계산 오버헤드가 발생하며, 이는 초대규모 네트워크에서 확장성 문제가 될 수 있다.

## 📌 TL;DR

본 논문은 Continual Federated Learning에서 발생하는 치명적 망각 문제를 해결하기 위해, 전역적으로 집계된 버퍼 그래디언트를 참조하여 로컬 그래디언트를 투영하는 **Fed-A-GEM**을 제안한다. 이 방법은 **명시적인 태스크 경계 정보 없이도** 작동하며, 기존의 다양한 CFL 기법과 결합하여 성능을 크게 향상시킬 수 있다. 특히 CIFAR-100에서 최대 27%의 정확도 향상을 보였으며, 낮은 계산/통신 오버헤드로 인해 실제 분산 환경 및 엣지 디바이스 적용 가능성이 매우 높다.