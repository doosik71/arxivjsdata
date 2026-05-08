# MedSegNet10: A Publicly Accessible Network Repository for Split Federated Medical Image Segmentation

Chamani Shiranthika, Zahra Hafezi Kafshgari, Hadi Hadizadeh, Parvaneh Saeedi (2025)

## 🧩 Problem to Solve

의료 영상 분할(Medical Image Segmentation)은 질병의 정확한 진단과 치료 계획 수립에 필수적인 작업이다. 하지만 딥러닝 모델을 학습시키기 위해서는 대량의 정밀하게 라벨링된 데이터가 필요하며, 의료 데이터의 특성상 환자의 개인정보 보호 문제로 인해 데이터를 한곳에 모으는 중앙 집중식 학습이 어렵다.

이를 해결하기 위해 연합 학습(Federated Learning, FL)과 분할 학습(Split Learning, SL), 그리고 이 둘을 결합한 분할 연합 학습(Split Federated Learning, SplitFed/SFL)과 같은 분산 학습 방식이 제안되었다. 그러나 이러한 분산 학습 프레임워크를 실제 의료 영상 분할 작업에 적용하기 위해서는 잘 설계되고 검증된 네트워크 아키텍처가 필요함에도 불구하고, 연구자들이 즉시 활용하고 벤치마킹할 수 있는 재사용 가능한 SplitFed 전용 네트워크 저장소(Repository)가 부족한 실정이다.

따라서 본 논문의 목표는 다양한 의료 영상 유형에 최적화된 10가지 주요 세그멘테이션 네트워크의 SplitFed 버전을 구현하고, 이를 공개 저장소인 **MedSegNet10**으로 제공하여 의료 영상 분할 연구의 효율성을 높이고 데이터 프라이버시를 보호하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 기여는 의료 영상 분할을 위한 최초의 문서화된 SplitFed 네트워크 저장소인 **MedSegNet10**을 구축한 것이다.

단순히 기존 모델을 나열한 것이 아니라, 최신 시맨틱 세그멘테이션 트렌드를 반영하여 성능이 검증된 10가지 모델(UNet, SegNet, DeepLabV3, DeepLabV3+, RefineNet, CGNet, SUNet, DUCK-Net, Attention UNet, Swin-UNet)을 선정하고, 이를 SplitFed 구조에 맞게 최적화하여 구현하였다. 이를 통해 연구자들은 모델을 처음부터 다시 설계할 필요 없이, 통합된 플랫폼에서 다양한 아키텍처의 차이를 분석하고 비교 실험을 수행할 수 있다.

## 📎 Related Works

### 분산 학습 접근 방식

- **Federated Learning (FL):** 중앙 서버가 여러 클라이언트의 데이터를 직접 가져오지 않고 모델 가중치만을 주고받으며 글로벌 모델을 학습시킨다. 하지만 클라이언트가 전체 모델을 학습시켜야 하므로 계산 자원이 부족한 환경에서는 부담이 크다.
- **Split Learning (SL):** 모델을 여러 부분으로 나누어 서로 다른 위치(클라이언트와 서버)에 배치한다. 클라이언트는 모델의 일부만 처리하므로 계산 부담이 줄어든다.
- **Split Federated Learning (SplitFed):** FL의 글로벌 모델 업데이트 방식과 SL의 모델 분할 구조를 결합한 형태이다. 데이터 프라이버시를 유지하면서 클라이언트의 계산 부담을 최소화하고 협력 학습을 가능하게 한다.

### 기존 저장소의 한계

TensorFlow Federated, FATE, FedCT 등 다양한 연합 학습 관련 저장소가 존재하지만, 대다수가 일반적인 FL에 집중되어 있다. 특히 SplitFed 아키텍처를 기반으로 하여 모델 간의 성능을 포괄적으로 비교하고 벤치마킹할 수 있는 전용 네트워크 저장소는 부재한 상태였다.

## 🛠️ Methodology

### 전체 시스템 구조 및 파이프라인

MedSegNet10의 SplitFed 구조는 모델을 세 부분으로 나누어 배치하는 방식을 취한다.

1. **Front-End (FE) sub-model:** 클라이언트 측에 위치하며, 원본 의료 영상 데이터에 직접 접근하여 초기 특징을 추출한다.
2. **Server sub-model:** 서버 측에 위치하며, 모델의 대부분의 레이어를 포함하여 주된 계산 부담을 담당한다.
3. **Back-End (BE) sub-model:** 클라이언트 측에 위치하며, 서버로부터 전달받은 특징을 바탕으로 최종 예측을 수행하고 Ground Truth(GT)와 비교하여 손실을 계산한다.

이러한 구조는 원본 데이터와 정답 라벨(GT)이 모두 클라이언트에 머물게 함으로써 데이터 프라이버시를 보장하며, 서버는 모델의 중간 표현(Intermediate representation)만을 처리하게 한다.

### 학습 절차 및 손실 함수

- **손실 함수:** 클래스 불균형이 심한 의료 영상의 특성을 고려하여 $\text{Soft Dice Loss}$를 사용하였다.
- **학습 흐름:**
  - 각 클라이언트는 로컬 데이터를 사용하여 FE, BE 및 서버 모델의 복사본을 가지고 일정 에폭(local epochs) 동안 학습한다.
  - 로컬 학습 후, 클라이언트는 FE와 BE의 가중치를 서버로 전송하며, 서버는 이를 집계(Aggregation)하여 업데이트된 글로벌 모델을 생성한다.
  - 업데이트된 모델은 다시 클라이언트로 배포되어 검증 과정을 거친다.
- **최적화:** Adam Optimizer를 사용하였으며, 데이터셋과 네트워크 특성에 맞는 학습률을 적용하였다.

### Split Point(분할 지점) 선정 기준

모델을 어디서 나눌 것인가는 통신 효율과 성능에 결정적인 영향을 미친다. 본 논문에서는 다음의 기준을 적용하였다.

- **태스크 특성:** 컴퓨터 비전 작업의 경우, 고수준 시각 특징(High-level visual features)을 캡처하는 레이어에서 분할한다.
- **통신 제약:** 전송되는 데이터 양을 줄여 지연 시간과 통신 오버헤드를 최소화하는 지점을 선택한다.
- **프라이버시:** 민감한 원본 데이터와 GT가 서버로 전송되지 않도록 FE와 BE를 독립적으로 구성한다.
- **계산 능력:** 클라이언트의 계산 부담을 최소화하기 위해 서버 모델이 대부분의 연산을 수행하도록 설계한다.

## 📊 Results

### 실험 설정

- **데이터셋:**
  - **Blastocyst dataset:** 배아 영상 (5개 클래스 분할)
  - **HAM10K dataset:** 피부 병변 영상 (이진 분할)
  - **KVASIR-SEG dataset:** 폴립 영상 (이진 분할)
- **비교 대상:**
  - **Centralized (C):** 전체 데이터를 한곳에 모아 학습한 경우 (Upper bound)
  - **Locally Centralized (L):** 각 클라이언트가 가진 소량의 데이터로만 개별 학습한 경우 (Lower bound)
  - **SplitFed (S):** 제안하는 방식으로 협력 학습한 경우
- **평가 지표:** Average Intersection Over Union (Average IoU) 및 FLOPs(계산 복잡도).

### 정량적 결과 분석

실험 결과, 모든 데이터셋에서 $\text{Centralized} > \text{SplitFed} > \text{Locally Centralized}$ 순으로 성능이 나타났다.

- **Blastocyst 데이터셋:** 중앙 집중식(C) 대비 SplitFed(S)의 성능 차이는 매우 적었으나, 로컬 학습(L)보다는 약 $10.61\%$ 높은 IoU를 기록하였다.
- **HAM10K 데이터셋:** C와 S의 차이는 약 $0.87\%$에 불과하여, SplitFed가 중앙 집중식 학습에 근접한 성능을 낼 수 있음을 보였다.
- **KVASIR-SEG 데이터셋:** L 대비 S의 성능 향상이 두드러졌으며, 중앙 집중식 학습과의 격차는 약 $2.01\%$였다.

### 계산 복잡도 및 모델 효율성

- **가장 가벼운 모델:** **CGNet**이 학습 가능한 파라미터 수와 FLOPs 모두에서 가장 낮은 수치를 기록하여 효율성이 매우 높았다.
- **가장 무거운 모델:** **RefineNet**은 파라미터 수가 가장 많았으며, **Attention UNet**은 FLOPs 수치가 가장 높게 측정되었다.

## 🧠 Insights & Discussion

### 강점 및 해석

본 연구는 SplitFed가 로컬 데이터 부족 문제를 효과적으로 해결할 수 있음을 입증하였다. 특히 $\text{S}$ (SplitFed) 결과가 $\text{C}$ (Centralized)에 근접한다는 점은, 데이터를 중앙으로 모으지 않고도 협력 학습을 통해 거의 동일한 수준의 모델 성능을 확보할 수 있음을 의미한다.

### 한계 및 비판적 논의

- **통신 오버헤드:** SplitFed는 매 반복마다 클라이언트와 서버 간에 특징 맵(Feature maps)과 그래디언트를 교환해야 하므로, 통신 비용이 매우 크다는 단점이 있다.
- **범용 아키텍처의 한계:** 현재 MedSegNet10은 기존의 일반적인 세그멘테이션 모델을 '분할'하여 사용하고 있다. 하지만 SplitFed 구조 자체에 최적화된, 즉 통신 효율과 분산 처리 능력을 극대화한 전용 아키텍처 설계가 필요하다.

### 향후 연구 방향

- **Transformer의 통합:** Swin-UNet 등의 결과에서 보듯, Transformer 기반 구조가 복잡한 의료 영상 패턴 캡처에 유리하므로 이를 더 적극적으로 도입할 필요가 있다.
- **개인화 연합 학습 (Personalized FL):** 클라이언트마다 데이터 분포가 다른 Non-IID 문제를 해결하기 위해 각 클라이언트의 특성에 맞게 모델을 조정하는 개인화 기법의 적용이 필요하다.
- **프라이버시 강화:** 동형 암호(Homomorphic Encryption)나 차분 프라이버시(Differential Privacy)를 네트워크 구조 자체에 내장하는 설계가 요구된다.

## 📌 TL;DR

본 논문은 의료 영상 분할을 위한 Split Federated Learning(SplitFed) 전용 네트워크 저장소인 **MedSegNet10**을 제안한다. UNet, DeepLabV3, Swin-UNet 등 10가지 주요 모델의 SplitFed 버전을 구현하여 제공함으로써, 데이터 프라이버시를 유지하면서도 중앙 집중식 학습에 근접한 성능을 낼 수 있는 협력 학습 환경을 구축하였다. 이 연구는 분산 의료 AI 연구자들이 모델 설계의 부담 없이 즉시 벤치마킹할 수 있는 기반을 마련했다는 점에서 큰 의의가 있다.
