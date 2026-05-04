# ELIS: Efficient LLM Iterative Scheduling System with Response Length Predictor

Seungbeom Choi*, Jeonghoe Goo*, Eunjoo Jeon, Mingyu Yang, Minsung Jang (2025)

## 🧩 Problem to Solve

본 논문은 Large Language Model (LLM) 서빙 시스템에서 발생하는 **Head-of-Line (HOL) blocking** 문제를 해결하고자 한다. 현재 많은 LLM 서빙 시스템은 First-Come-First-Served (FCFS) 스케줄링 전략을 채택하고 있는데, 이는 짧은 응답 길이의 작업이 긴 응답 길이의 작업 뒤에 배치될 경우 불필요하게 긴 대기 시간을 갖게 되는 비효율성을 초래한다.

이 문제를 해결하기 위해서는 Shortest Job First (SJF)와 같은 우선순위 스케줄링이 필요하며, 이를 위해 각 요청의 추론 시간(정확히는 생성될 토큰 수)을 미리 예측해야 한다. 그러나 LLM의 **Auto-regressive(자기회귀)** 특성으로 인해 생성될 출력 토큰의 길이를 정확히 예측하는 것은 매우 어렵다. 또한, 예측 모델 자체가 무거울 경우 추론 오버헤드가 발생하여 전체 시스템의 병목 지점이 될 수 있다는 점이 주요 해결 과제이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 LLM의 출력 길이를 반복적으로 예측하여 작업의 우선순위를 동적으로 조정하는 **Iterative Shortest Remaining Time First (ISRTF)** 스케줄러를 제안한 것이다.

핵심 직관은 LLM이 토큰을 순차적으로 생성한다는 점을 이용하여, **생성이 진행됨에 따라(iterative) 더 정확한 잔여 길이 예측이 가능**하다는 것이다. 이를 위해 가벼운 Encoder 기반의 BGE 모델을 활용한 Response Length Predictor를 설계하였으며, 이를 기반으로 잔여 시간이 가장 짧은 작업을 우선적으로 처리함으로써 평균 Job Completion Time (JCT)을 단축시켰다. 또한, 이를 Kubernetes 기반의 클라우드 네이티브 환경에서 구현하여 실제 산업 현장에서의 확장성과 실용성을 검증하였다.

## 📎 Related Works

기존의 LLM 서빙 최적화 연구들은 주로 다음과 같은 접근 방식을 취했다.

1.  **Iteration-level Batching (예: ORCA):** 전체 배치가 끝날 때까지 기다리지 않고 반복 단위로 배치를 구성하는 Continuous Batching을 도입했으나, 기본적으로 FCFS 방식을 사용하여 HOL blocking 문제를 완전히 해결하지 못했다.
2.  **단일 예측 기반 SJF (예: Qiu et al.):** BERT 모델을 사용하여 출력 길이를 한 번 예측하고 이를 기반으로 SJF 스케줄링을 수행했다. 하지만 예측 정확도가 낮고, 예측이 틀렸을 때의 대응 방안이 부족하여 여전히 HOL blocking이 발생할 가능성이 있었다.
3.  **LLM 자체 예측 (예: Zheng et al.):** LLM이 응답과 동시에 길이를 예측하도록 Instruction Fine-tuning을 수행했다. 그러나 이 방식은 LLM 본연의 모델 정확도에 영향을 줄 수 있으며, 예측된 길이에 맞추어 응답이 제한되는 부작용이 있을 수 있다.

ELIS는 이러한 한계를 극복하기 위해 **별도의 가벼운 예측 모델**을 사용하며, 단 한 번의 예측이 아닌 **생성 과정 중 반복적인 예측**을 통해 정확도를 높이고 스케줄링 효율을 극대화했다는 점에서 차별점을 갖는다.

## 🛠️ Methodology

### 1. 전체 시스템 아키텍처
ELIS는 **Frontend Scheduler**와 여러 개의 **Backend Worker**로 구성된다.
- **Frontend Scheduler:** 요청을 수신하고, Predictor를 통해 우선순위를 결정하며, Load Balancer를 통해 작업을 백엔드 워커에 분배한다.
- **Backend Worker:** vLLM 실행 엔진을 탑재하여 실제 추론을 수행하며, 스케줄러의 결정에 따라 배치를 구성하고 결과를 반환한다.

### 2. Response Length Predictor
예측 모델은 Encoder 기반의 **BGE (BAAI/bge-base-en-v1.5)** 모델과 8개의 Fully Connected (FC) 레이어로 구성된다.
- **구조:** 입력 프롬프트 $\rightarrow$ BGE Model $\rightarrow$ CLS Token Embedding $\rightarrow$ Mean Pooling $\rightarrow$ 8 FC Layers $\rightarrow$ Predicted Length.
- **특징:** BGE 모델의 파라미터는 Frozen 상태로 유지하고 FC 레이어만 학습시켜 연산 비용을 낮추었다.
- **Iterative Prediction:** 고정된 윈도우 크기(예: 50토큰)마다 예측을 갱신한다. $t$번째 반복에서 예측 모델은 $\text{Prompt} + \text{Previously Generated Tokens}$를 입력으로 받아 남은 토큰 수를 예측한다.

### 3. ISRTF (Iterative Shortest Remaining Time First) 스케줄링
ISRTF는 예측된 잔여 토큰 수가 가장 적은 작업에 가장 높은 우선순위를 부여한다.

- **동작 절차:**
    1. 요청 도착 시 초기 예측을 통해 우선순위를 부여하고 $\text{Job Pool}$에 저장한다.
    2. $\text{Priority Buffer}$에서 우선순위가 높은 작업부터 배치(Batch)를 구성한다.
    3. 백엔드 워커는 해당 배치를 50토큰(윈도우 크기)만큼만 실행하고 결과를 반환한다.
    4. 생성된 부분 응답을 바탕으로 예측 모델이 잔여 길이를 다시 계산하여 우선순위를 갱신하고 다시 큐에 삽입한다.

### 4. 구현 상세
- **Kubernetes 배포:** Backend Worker를 $\text{StatefulSet}$으로 관리하여 고유 ID를 부여하고, Frontend Scheduler와 통신하도록 구현했다.
- **vLLM 수정:** vLLM의 기본 FCFS 정책을 오버라이드하여 ISRTF 우선순위를 적용하고, 정해진 토큰 수($K$ tokens)만큼만 실행 후 제어권을 반환하는 Iteration-wise execution 기능을 추가했다.

## 📊 Results

### 1. 실험 설정
- **GPU:** NVIDIA A100 (80GB) 및 H100.
- **LLM 모델:** OPT-6.7B/13B, LlaMA2-7B/13B, Vicuna-13B 등 다양한 크기의 모델 사용.
- **워크로드:** 실제 삼성 SDS의 FabriX 서비스 트레이스 데이터를 분석하여, 요청 간격이 포아송 분포가 아닌 **감마 분포** ($\alpha=0.73, \beta=10.41$)를 따른다는 것을 발견하고 이를 시뮬레이션에 적용했다.

### 2. 정량적 결과
- **JCT 단축:** ISRTF는 FCFS 대비 평균 JCT를 최대 **19.58%**까지 감소시켰다. (Table 5 및 Figure 5 참조)
- **원인 분석:** JCT 감소의 주된 요인은 **Queuing Delay(대기 시간)의 감소**에 있었다. 특정 케이스(LlaMA2-13B, 5.0x RPS)에서 ISRTF는 FCFS보다 대기 시간을 16.75% 줄였으며, 이는 전체 JCT 감소분과 거의 일치한다.
- **오버헤드:** BERT 기반 예측 및 스케줄링에 소요되는 오버헤드는 약 11.04ms로, 전체 추론 시간(예: LlaMA2-13B의 경우 8610.2ms) 대비 약 0.13% 수준으로 매우 미미하다.

### 3. 확장성 평가 (Scalability)
- NVIDIA H100 GPU 환경에서 워커 수를 10개에서 50개까지 늘렸을 때, Peak Throughput이 **2.31 RPS에서 18.77 RPS까지 거의 선형적으로 증가**하는 것을 확인하였다. 이는 Load Balancer의 효율적인 분배와 비동기 스케줄링 구조 덕분이다.

## 🧠 Insights & Discussion

### 1. 강점 및 통찰
- **반복적 예측의 효용성:** 생성된 토큰이 많아질수록 예측 모델의 MAE(Mean Absolute Error)가 감소하는 것을 확인했다. 이는 autoregressive한 LLM의 특성을 스케줄링에 잘 반영한 설계이다.
- **실제 데이터 기반 분석:** 단순히 이론적인 포아송 분포를 가정하지 않고, 실제 산업 현장의 트레이스 데이터를 분석하여 Gamma 분포를 적용함으로써 실험의 신뢰성을 높였다.
- **실용적 구현:** Kubernetes와 vLLM이라는 업계 표준 도구를 활용하여 실제 배포 가능한 수준의 시스템을 구축했다는 점이 높게 평가된다.

### 2. 한계 및 논의사항
- **Preemption(선점) 가능성:** 논문에서는 메모리 부족으로 인한 Preemption 가능성을 조사하였으나, 실제 FabriX의 요청률(3 RPS 미만) 수준에서는 Preemption이 발생할 확률이 매우 낮음을 확인했다. 따라서 실제 실험에서는 iterative scheduling에 집중했으나, 초고부하 환경에서의 Preemption 전략에 대한 심도 있는 실험은 부족하다.
- **예측 모델의 일반화:** BGE 모델을 사용해 높은 성능을 냈으나, 모델마다 응답 패턴이 다를 수 있으므로 다양한 도메인의 LLM에 대해 예측 모델을 어떻게 빠르게 적응(Adaptation)시킬 것인지에 대한 논의가 필요하다.

## 📌 TL;DR

본 논문은 LLM 서빙 시 발생하는 HOL blocking 문제를 해결하기 위해, BGE 모델 기반의 응답 길이 예측기와 이를 활용한 **ISRTF(Iterative Shortest Remaining Time First)** 스케줄러를 제안한다. 생성 과정 중 반복적으로 잔여 길이를 예측하여 우선순위를 조정함으로써 **평균 JCT를 최대 19.6% 단축**시켰으며, Kubernetes 기반 구현을 통해 뛰어난 확장성을 입증했다. 이 연구는 실제 산업 현장의 워크로드 분석을 바탕으로 하여, 향후 고효율 LLM 서빙 인프라 구축에 중요한 기초 자료가 될 것으로 보인다.