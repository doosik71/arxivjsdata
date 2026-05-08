# Parameterizing Federated Continual Learning for Reproducible Research

Bart Cox, Jeroen Galjaard, Aditya Shankar, Jérémie Decouchant, and Lydia Y. Chen (2024)

## 🧩 Problem to Solve

현대의 Federated Learning (FL) 시스템은 이질적이고 끊임없이 변화하는 환경에서 작동하며, 실제 배포 환경에서는 클라이언트가 수행해야 하는 학습 태스크(Task) 또한 시간에 따라 진화한다. 이러한 시나리오를 해결하기 위해 Continual Learning (CL) 방법론을 FL에 통합한 Federated Continual Learning (FCL)의 필요성이 대두되었다.

그러나 기존의 연구 환경은 다음과 같은 문제점을 가지고 있다. 첫째, 대부분의 FL 시뮬레이션 프레임워크는 정적인 환경을 가정하며, 데이터 및 하드웨어의 이질성(Heterogeneity)이나 동적인 태스크 변화를 정밀하게 모사하지 못한다. 둘째, 기존의 CL 프레임워크들은 단일 머신 환경에 집중되어 있어 FL의 특성인 간접적 협력 학습을 지원하지 않는다. 결과적으로, 실제 배포 환경과 유사한 복잡한 FCL 시나리오를 구성하고 그 결과를 재현(Reproducibility)하는 것이 매우 어렵다는 문제가 존재한다. 본 논문의 목표는 이러한 제약을 극복하여 복잡한 학습 시나리오를 정밀하게 캡처하고 에뮬레이션할 수 있는 확장 가능하고 유연한 FCL 프레임워크를 제공하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Federated Continual Learning (FCL)을 위한 최초의 완전히 설정 가능한 오픈소스 프레임워크인 **Freddie**를 제안한 것이다. Freddie의 주요 설계 아이디어는 다음과 같다.

- **확장 가능한 인프라 구축**: Kubernetes와 컨테이너화(Containerization) 기술을 도입하여, 단일 머신 시뮬레이션부터 대규모 클라우드 환경의 에뮬레이션까지 유연하게 확장할 수 있도록 설계하였다.
- **이질성 파라미터화**: 데이터의 통계적 이질성뿐만 아니라, 컴퓨팅 파워, 네트워크 지연 시간(Latency), 처리량(Throughput)과 같은 자원 이질성(Resource Heterogeneity)을 설정할 수 있게 하여 실제 환경과 유사한 실험을 가능하게 한다.
- **FCL 전용 벤치마킹 방법론**: 클라이언트 간 태스크 순서의 차이가 모델 성능에 미치는 영향을 분석하기 위해 Column, Balanced, Shuffled라는 세 가지 태스크 분할 스킴(Partitioning Scheme)을 제안하였다.

## 📎 Related Works

기존의 FL 프레임워크(Flower, Fate, TorchX, TensorFlow Federated 등)는 대부분 고정된 태스크 세트를 기반으로 동작하며, 시간이 지남에 따라 출력 타입이 변하는 CL 시나리오를 지원하는 데 한계가 있다. 특히 Fate와 같은 도구는 보안과 프로덕션 환경에 최적화되어 있어 연구 단계의 빠른 프로토타이핑과 유연한 실험 구성에는 부적합한 측면이 있다.

CL 분야의 프레임워크(FACIL, PyCIL, Avalanche 등)는 EWC, GEM과 같은 다양한 알고리즘을 제공하지만, 대부분 단일 머신 기반으로 동작하며 분산 환경에서의 모델 업데이트 및 집계 과정을 고려하지 않는다. FedWEIT와 같은 일부 FCL 연구가 존재하지만, 태스크 시퀀스가 글로벌 모델의 품질에 미치는 영향에 대한 체계적인 분석과 이를 재현하기 위한 프레임워크 수준의 지원은 부족한 상태이다.

## 🛠️ Methodology

### 시스템 아키텍처

Freddie는 Kubernetes 상에서 Kubeflow의 Training Operators를 통해 배포된다. 전체 시스템 구조는 다음과 같은 주요 구성 요소로 이루어져 있다.

- **Orchestrator**: 사용자가 제출한 시스템 설정 및 하이퍼파라미터 설명을 바탕으로 실험을 배포하고 모니터링한다.
- **Extractor**: NFS(Network File System) 프로비저너 및 서버를 통해 페더레이터(Federator)와 클라이언트가 생성한 통계 데이터 및 아티팩트를 수집하고 저장한다.
- **Federator & Client Nodes**: 실제 학습이 수행되는 노드들로, 두 주체 간의 통신은 비동기(Asynchronous) 방식으로 이루어져 Non-blocking 상호작용이 가능하다.

### FCL 지원 메커니즘

Freddie는 최신 FCL 알고리즘과 더불어, CL의 핵심인 Task-Interactive Learning (Task-IL)과 Domain-Interactive Learning (Domain-IL)을 지원하기 위해 세 가지 윈도우 메커니즘을 구현하였다.

1. **Sliding Window**: 현재 평가 시점 $t$의 태스크에 해당하는 출력 클래스로만 제한한다. (Task-IL 지원)
2. **Expanding Window**: 태스크 ID를 사용하지 않고, 현재까지 학습한 모든 클래스를 포함하여 출력한다. (Domain-IL 지원)
3. **Full Window**: 출력 제한 없이 표준 FL 시나리오와 동일하게 동작한다.

### 태스크 분할 스킴 (Task Partitioning Schemes)

클라이언트마다 학습하는 태스크의 순서를 다르게 설정하여 Catastrophic Forgetting의 영향을 분석한다.

- **Column**: 모든 클라이언트가 동일한 순서로 태스크를 학습한다. 이는 가장 단순한 분할 방식이며, 심각한 Catastrophic Forgetting이 발생할 가능성이 높다.
- **Balanced**: 특정 태스크가 한 시점에 최대 한 명의 클라이언트에 의해서만 학습되도록 배치하여 단기적인 망각 효과를 완화한다.
- **Shuffled**: 사전 정의된 시드(Seed)를 바탕으로 각 클라이언트의 태스크 순서를 무작위로 배치한다.

## 📊 Results

### 실험 설정

- **데이터셋**: CIFAR10(소규모 실험) 및 Overlapping CIFAR100(FCL 실험, 10~20개 태스크로 분할)을 사용하였다.
- **인프라**: Google Kubernetes Engine (GKE)의 `e2-standard-8` 노드를 사용하였다.
- **지표**: CL 문헌에 따라 Average Accuracy를 측정하였다.

### 주요 결과

1. **확장성(Scalability) 분석**:
   - 클라이언트 수(World Size, WS)가 증가함에 따라 클라이언트와 페더레이터의 라운드 소요 시간이 모두 증가하는 양의 상관관계를 보였다.
   - 이는 동일한 노드에 여러 클라이언트가 함께 배치(Co-scheduling)됨에 따라 발생하는 자원 경합(Resource Contingency)과 통신 오버헤드에 기인한 것으로 분석된다.

2. **Task-IL vs Domain-IL**:
   - Task ID를 알고 있는 Sliding Window(Task-IL) 방식이 Expanding Window(Domain-IL) 방식보다 훨씬 높은 정확도를 보였다.
   - 이는 Task-IL의 경우 출력 클래스가 해당 태스크의 5개 클래스로 제한되어 정답 확률이 높아지기 때문이다.

3. **태스크 이질성의 영향**:
   - 태스크 분할 스킴 중 **Column** 방식이 가장 낮은 정확도를 기록하였으며, Shuffled 및 Balanced 방식에 비해 평균적으로 약 4%의 테스트 정확도 하락이 관찰되었다.
   - 이는 모든 클라이언트가 동일한 순서로 학습할 때 Catastrophic Forgetting의 영향이 더 극명하게 나타남을 시사한다.

## 🧠 Insights & Discussion

본 연구는 FCL 연구에서 단순한 알고리즘 성능 측정보다, **어떻게 실험 환경을 파라미터화하고 제어하느냐**가 재현성에 결정적인 영향을 미친다는 점을 강조한다. 특히 Kubernetes 기반의 에뮬레이션 환경을 통해 자원 이질성이 실제 학습 시간에 미치는 영향을 정량적으로 보여준 점이 고무적이다.

논문의 결과에서 나타난 Task-IL과 Domain-IL의 성능 차이는 실무적인 관점에서 중요하다. 실제 애플리케이션에서 태스크 ID를 명시적으로 알 수 없는 경우가 많으므로, Domain-IL 환경에서의 성능 저하를 해결하는 것이 향후 FCL 연구의 핵심 과제가 될 것이다. 또한, Column 스킴에서 발생하는 극심한 망각 현상은 글로벌 모델의 업데이트 방향이 특정 태스크에 편향될 때 발생하는 문제임을 시사하며, 이를 해결하기 위한 더 정교한 집계(Aggregation) 전략이 필요함을 보여준다.

## 📌 TL;DR

본 논문은 확장 가능하고 설정 가능한 FCL 프레임워크인 **Freddie**를 제안하여, 데이터 및 자원의 이질성을 포함한 실제적인 FCL 실험 환경의 재현 가능성을 높였다. 특히 태스크 분할 방식(Column, Balanced, Shuffled)에 따라 Catastrophic Forgetting의 정도가 크게 달라짐을 입증하였으며, 이는 향후 FCL 시스템 설계 시 태스크 스케줄링과 자원 관리가 핵심 요소가 될 것임을 시사한다.
