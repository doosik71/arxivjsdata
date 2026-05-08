# LLM Assisted Anomaly Detection Service for Site Reliability Engineers: Enhancing Cloud Infrastructure Resilience

Nimesh Jha, Shuxin Lin, Srideepika Jayaraman, Kyle Frohling, Christodoulos Constantinides, Dhaval Patel (2025)

## 🧩 Problem to Solve

본 논문은 현대 클라우드 인프라 운영의 핵심인 Site Reliability Engineers(SRE)가 직면한 모니터링의 한계를 해결하고자 한다. 클라우드 컴퓨팅의 도입 증가로 인해 인프라의 안정성과 신뢰성에 대한 요구가 높아졌으나, 수만 대의 가상 서버 인스턴스(VSI)를 수동으로 모니터링하는 것은 사실상 불가능하다.

기존의 룰 기반(Rule-based) 알림 시스템은 특정 임계치(예: CPU 사용률 80% 초과)를 넘어야만 작동하므로, 임계치 이하에서 발생하는 미세하지만 유의미한 이상 징후나 일시적인 스파이크(Spike)를 감지하지 못하는 한계가 있다. 따라서 SRE가 장애가 확대되기 전에 선제적으로 문제를 식별하고 해결할 수 있도록 돕는 확장 가능하고 일반화된 이상 탐지(Anomaly Detection) 서비스의 구축이 필수적이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 대규모 언어 모델(LLM)을 활용한 이상 모델링과 이를 실제 서비스 형태로 구현한 Anomaly Detection API의 제공이다.

1. **LLM 기반 이상 모델링**: 도메인 지식이 부족한 상태에서 단순히 데이터를 분석하는 대신, LLM을 사용하여 클라우드 구성 요소 $\rightarrow$ 장애 모드(Failure Modes) $\rightarrow$ 모니터링 지표 $\rightarrow$ 이상 행동 패턴으로 이어지는 지식 맵을 생성하고, 이를 실제 데이터셋의 변수와 매핑하는 혁신적인 접근 방식을 제안한다.
2. **확장 가능한 API 서비스 구현**: 단변량(Univariate) 및 다변량(Multivariate) 시계열 데이터를 모두 지원하며, 회귀 기반, 혼합 모델 기반, 준지도 학습 기반 등 다양한 알고리즘을 제공하는 일반화된 API를 구축하여 실제 산업 현장에 배포하였다.
3. **SRE 중심의 실용적 프레임워크**: 데이터 수집부터 전처리, 이상 탐지, Grafana 기반의 시각화까지 이어지는 엔드-투-엔드 파이프라인을 구축하여 SRE의 업무 효율성을 실질적으로 향상시켰다.

## 📎 Related Works

논문은 클라우드 인프라 모니터링 및 이상 탐지 분야의 기존 연구들을 언급하며, 특히 다변량 시계열 데이터에서의 비지도 학습 기반 이상 탐지 연구들에 주목한다. 벤치마크 분석을 위해 최신 프레임워크인 DAEMON(Adversarial Autoencoder Anomaly Detection Interpretation)을 비교 대상으로 설정하였다.

기존 접근 방식과의 차별점은 단순히 모델의 성능 향상에 그치지 않고, LLM을 통해 도메인 지식을 체계적으로 추출하여 모델링에 반영했다는 점과, 이를 API 형태로 서비스화하여 다양한 페르소나(Ops SRE, Alert Developer, Service SRE)가 즉각적으로 활용할 수 있는 인프라를 제공했다는 점에 있다.

## 🛠️ Methodology

### 1. LLM-Assisted Anomaly Modelling

LLM(Llama, Mistral 등)을 활용하여 다음과 같은 워크플로우로 이상 모델을 설계한다.

- **단계 1**: 클라우드 인프라 내의 구성 요소(서버, 네트워크, 보안, 전원 시스템 등) 식별.
- **단계 2**: 각 구성 요소별 발생 가능한 장애 모드 생성.
- **단계 3**: 각 장애 모드별로 모니터링해야 할 지표와 이상 행동(예: 갑작스러운 스파이크 또는 지속적인 고부하) 정의.
- **단계 4**: 생성된 지식을 실제 데이터셋의 변수명(예: `ibm.isinstanceaveragecpuusagepercentage`)과 매핑.

### 2. Anomaly Detection System Pipeline

전체 시스템은 **Data Capture $\rightarrow$ Data Store $\rightarrow$ Anomaly Detection $\rightarrow$ System Visualization**의 단계로 구성된다.

- **데이터 수집 및 저장**: IaaS 멀티존 지역(MZR)에서 네트워크 트래픽, 메모리 사용량, 디스크 I/O, CPU 이용률 등의 지표를 수집하여 Cloud Object Storage(COS)에 저장한다.
- **ReconstructAD 알고리즘**: 본 시스템의 핵심인 ReconstructAD는 $\text{DNN AutoEncoder}$ 기반의 딥러닝 모델을 사용한다. AutoEncoder가 정상 데이터의 패턴을 학습하여 재구성(Reconstruction)하고, 입력 데이터와 재구성 데이터 사이의 차이인 재구성 오차(Reconstruction Error)를 통해 이상치를 탐지한다.
- **이상 점수 산출 및 판별**:
  - 재구성 오차 배열의 평균과 표준편차를 계산한다.
  - $\text{Chi-Squared Distribution}(\chi^2 \text{ 분포})$를 사용하여 p-value를 계산하며, 이를 이상 점수(Anomaly Score)로 활용한다.
  - 설정된 임계치(Threshold)를 기준으로 p-value가 특정 기준보다 낮을 경우 이를 이상치로 라벨링한다.
- **다변량 분석**: 다변량 이상 탐지 시나리오에서는 주성분 분석(PCA)을 사용하여 어떤 지표가 이상치 발생에 가장 큰 영향을 주었는지 식별한다.

### 3. Anomaly Detection Service API

사용자가 다양한 알고리즘을 선택해 사용할 수 있도록 5가지 API 엔드포인트를 제공한다.

- **Univariate**: 단일 지표의 이상 탐지.
- **Multi-variate**: 여러 지표 간의 상관관계를 고려한 탐지.
- **Regression-based**: 입력 변수와 타겟 변수 간의 회귀 관계를 학습하여 예측값과 실제값의 차이로 탐지.
- **Mixture-Model based**: 정상 작동 모드를 학습하고, 학습되지 않은 새로운 모드(Unseen mode)를 탐지.
- **Semi-Supervised**: 정상 데이터로 학습하되, 극소수의 고장 데이터를 사용하여 파라미터를 튜닝.

## 📊 Results

### 1. 벤치마크 성능 평가

SMD, MSL, SMAP의 세 가지 공공 벤치마크 데이터셋을 사용하여 $F1\text{-score}$를 측정하였다.

- **결과**: 본 서비스의 알고리즘 스위트는 SOTA 모델인 DAEMON과 경쟁 가능한 수준의 성능을 보였다.
- **상세**: 특히 $\text{GMML0}$와 $\text{GMML1}$ 모델은 MSL과 SMAP 데이터셋에서 매우 높은 $F1\text{-score}$를 기록하며 DAEMON을 상회하는 성능을 보이기도 했다. $\text{DeepAD}$ 또한 높은 경쟁력을 보였다.

### 2. 실제 서비스 적용 및 사용량

- **사용 규모**: 2022년부터 현재까지 500명 이상의 사용자가 서비스를 이용했으며, 총 500,000회 이상의 API 호출이 발생하였다.
- **실제 적용**: SRE들이 Grafana 대시보드를 통해 실시간으로 $\text{CPU Usage Percent}$ 등의 지표에서 발생하는 스파이크를 탐지하고 대응하는 데 성공적으로 적용되었다.
- **확장성**: 2023년 6월과 같은 특정 시점에 API 호출 수가 급증했음에도 시스템이 안정적으로 작동하여 구현체의 확장성(Scalability)이 검증되었다.

## 🧠 Insights & Discussion

### 강점

본 연구는 이론적인 모델 제안에 그치지 않고, 실제 클라우드 환경에서 SRE가 겪는 고충을 해결하기 위한 **실용적인 서비스 아키텍처**를 구축했다는 점이 매우 강력하다. 특히 LLM을 활용해 도메인 지식을 정형화하고 이를 모델링에 연결한 점은, 데이터 기반의 접근 방식과 지식 기반의 접근 방식을 효과적으로 결합한 사례라고 평가할 수 있다.

### 한계 및 향후 과제

- **Zero-shot 탐지**: 현재 시스템은 데이터에 기반한 학습이 필요하다. 저자들은 향후 시계열 파운데이션 모델(Time series foundation models)을 도입하여 학습 데이터 없이도 이상을 탐지하는 $\text{Zero-shot anomaly detection}$ 기능을 추가할 계획임을 밝혔다.
- **가정**: 본 보고서에서는 API의 성능 지표(F1)는 제시되었으나, 실제 SRE의 업무 시간이 얼마나 단축되었는지에 대한 정량적인 분석(예: MTTD - Mean Time To Detect 감소량)은 명시되지 않았다.

### 비판적 해석

LLM을 통한 지표 매핑 과정이 매우 유용해 보이지만, LLM이 생성한 지식에 오류(Hallucination)가 있을 경우 잘못된 지표를 모니터링하게 될 위험이 있다. 따라서 LLM의 제안을 인간 전문가가 검증하는 루프(Human-in-the-loop)가 필수적으로 포함되어야 할 것이다.

## 📌 TL;DR

본 논문은 SRE의 클라우드 인프라 관리를 돕기 위해 **LLM 기반의 도메인 지식 추출**과 **다양한 딥러닝 알고리즘을 탑재한 확장 가능한 API 서비스**를 제안한다. $\text{DNN AutoEncoder}$ 기반의 $\text{ReconstructAD}$와 $\chi^2$ 분포를 이용한 이상 점수 산출 방식을 사용하며, 벤치마크 결과 SOTA 모델인 DAEMON과 대등하거나 더 우수한 성능을 보였다. 이 연구는 클라우드 모니터링의 자동화를 통해 다운타임을 줄이고 인프라 회복탄력성을 높이는 데 중요한 역할을 할 것으로 기대된다.
