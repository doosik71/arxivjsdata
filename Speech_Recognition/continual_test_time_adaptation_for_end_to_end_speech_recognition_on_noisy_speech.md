# Continual Test-time Adaptation for End-to-end Speech Recognition on Noisy Speech

Guan-Ting Lin, Wei-Ping Huang, Hung-yi Lee (2024)

## 🧩 Problem to Solve

딥러닝 기반의 종단간(End-to-end) 자동 음성 인식(Automatic Speech Recognition, ASR) 시스템은 학습 데이터와 동일한 도메인에서는 매우 낮은 오류율을 보이지만, 실제 환경에서 발생하는 도메인 시프트(Domain Shift) 문제로 인해 학습 데이터 외의 샘플(Out-of-domain samples)에 대해서는 성능이 크게 저하되는 문제가 있다.

이를 해결하기 위해 추론 시점에 테스트 샘플을 사용하여 모델을 적응시키는 Test-time Adaptation (TTA) 방법론들이 제안되었다. 그러나 기존의 ASR TTA 연구들은 주로 각 샘플을 독립적으로 처리하고 다시 초기 모델로 리셋하는 Non-continual TTA에 집중되어 있었다. 이러한 방식은 샘플 간의 지식을 학습할 수 없다는 한계가 있다. 반면, Continual TTA (CTTA)는 지식을 누적하여 학습할 수 있으나, 데이터 스트림이 길어질 경우 모델이 붕괴(Model collapse)되어 성능이 급격히 떨어지는 불안정성 문제가 존재한다.

따라서 본 논문의 목표는 CTTA의 지식 누적 이점과 Non-continual TTA의 안정성을 동시에 확보하여, 시간에 따라 도메인이 변하는 노이즈 섞인 음성 데이터 환경에서도 강건하게 작동하는 ASR TTA 프레임워크를 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Fast-slow TTA 프레임워크**를 도입하여 모델 파라미터를 두 가지 속도로 업데이트하는 것이다.

1. **Fast-slow TTA Framework**: 느리게 업데이트되는 메타 파라미터($\phi_t$)를 통해 도메인 간 공통 지식을 캡처하고, 각 샘플에 대해 빠르게 적응하는 파라미터($\hat{\phi}_t$)를 통해 개별 샘플의 특성을 반영함으로써 모델 붕괴 위험을 줄이면서도 성능을 극대화한다.
2. **Dynamic SUTA (DSUTA)**: 위 프레임워크를 바탕으로 Entropy Minimization 기반의 SUTA 방법론을 확장한 CTTA 모델을 제안한다.
3. **Dynamic Reset Strategy**: 도메인이 시간에 따라 변하는 환경에서 모델의 안정성을 높이기 위해, 통계적 방법으로 도메인 시프트를 감지하고 자동으로 모델을 초기 상태로 리셋하는 전략을 제안한다.

## 📎 Related Works

### Non-continual TTA for ASR

SUTA, SGEM과 같은 방법론들이 대표적이다. 이들은 Entropy Minimization (EM)이나 Minimum Class Confusion (MCC) 같은 비지도 학습 목적 함수를 사용하여 각 발화(Utterance)에 대해 모델을 최적화한 후, 다음 샘플이 들어오면 다시 원본 모델로 리셋한다. 이러한 방식은 안정적이지만, 스트림 데이터 전체에서 얻을 수 있는 지식을 활용하지 못한다는 단점이 있다.

### Continual TTA (CTTA)

컴퓨터 비전 분야에서는 모델 리셋이나 샘플 효율적 EM 등을 통해 CTTA의 안정성을 높이는 연구가 진행되었다. ASR 분야에서는 AWMC가 Pseudo-labeling과 앵커 모델을 사용하여 CTTA를 시도하였으나, Pseudo-labeling 방식은 EM 기반 방식보다 효과가 떨어지는 경향이 있으며, 다중 도메인이 혼재된 긴 데이터 스트림에서의 성능 검증이 부족했다.

## 🛠️ Methodology

### 1. Fast-slow TTA Framework

본 프레임워크는 모델 파라미터를 두 가지 경로로 운용한다.

- **Slow Update (Meta-parameters $\phi_t$)**: 메타 파라미터 $\phi_t$는 버퍼에 저장된 여러 샘플들을 통해 천천히 업데이트되며, 도메인의 일반적인 지식을 학습한다.
- **Fast Adaptation ($\hat{\phi}_t$)**: 실제 예측을 수행할 때는 $\phi_t$에서 시작하여 현재 입력 샘플 $x_t$에 대해 빠르게 최적화된 $\hat{\phi}_t$를 생성하여 예측값 $\hat{y}_t$를 도출한다.

이 구조는 Non-continual TTA(업데이트 알고리즘 $U$가 항등 함수인 경우)와 Continual TTA(적응 알고리즘 $A$와 업데이트 알고리즘 $U$가 동일한 경우)를 모두 일반화하는 형태이다.

### 2. Dynamic SUTA (DSUTA)

DSUTA는 위 프레임워크를 SUTA의 손실 함수를 사용하여 구현한 것이다.

- **손실 함수 ($L_{suta}$)**: 엔트로피 최소화($L_{em}$)와 클래스 간 혼동 최소화($L_{mcc}$)의 가중 합으로 구성된다.
  $$L_{suta} = \alpha L_{em} + (1-\alpha) L_{mcc}$$
- **학습 절차**:
  1. 새로운 샘플 $x_t$가 들어오면 $\phi_t$로부터 $N$단계 동안 $L_{suta}$를 통해 빠르게 적응하여 $\hat{\phi}_t$를 만들고 예측을 수행한다.
  2. 샘플을 버퍼 $B$에 저장하며, 버퍼가 크기 $M$에 도달하면 $M$개 샘플의 평균 $L_{suta}$를 사용하여 메타 파라미터 $\phi_t$를 업데이트한다.

### 3. Dynamic Reset Strategy

시간에 따라 도메인이 바뀔 때 모델 붕괴를 막기 위해 제안된 전략이다.

#### Loss Improvement Index (LII)

도메인 시프트를 측정하기 위해 LII라는 지표를 정의한다. 이는 도메인 특화 모델($\phi_D$)의 손실에서 사전 학습된 모델($\phi_{pre}$)의 손실을 뺀 값이다.
$$LII_t = L(\phi_D, x_t) - L(\phi_{pre}, x_t)$$
데이터 자체의 난이도로 인한 영향을 제거하기 위해 사전 학습 모델의 손실을 빼주는 정규화 과정을 거친다.

#### 작동 단계

1. **Domain Construction Stage**: 초기 $K$개의 샘플을 통해 기반 도메인 $D$를 구축하고, $\phi_D$를 생성하며 LII의 통계적 분포 $G_D = N(\mu, \sigma^2)$를 계산한다.
2. **Shift Detection Stage**: 이후 들어오는 샘플들의 평균 LII를 계산하여 Z-score 기반의 가설 검정을 수행한다.
    $$\frac{\frac{1}{M} \sum_{i \in B} LII_i - \mu}{\sigma / \sqrt{M}} > 2$$
    위 조건이 만족되면 큰 도메인 시프트가 발생한 것으로 판단하여 메타 파라미터를 $\phi_{pre}$로 리셋하고 다시 구축 단계로 돌아간다.

## 📊 Results

### 실험 설정

- **데이터셋**:
  - 단일 도메인: Librispeech에 10종의 노이즈를 섞은 LS-C (SNR 5dB).
  - 다중 도메인: MD-Easy, MD-Hard, MD-Long (노이즈 종류와 순서를 다르게 배치한 스트림).
  - 실제 데이터: CHiME-3 (카페, 버스, 거리 등 실제 소음 환경).
- **비교 대상**: SUTA, SGEM (Non-continual), CSUTA, AWMC (Continual).
- **지표**: Word Error Rate (WER).

### 주요 결과

- **단일 도메인 성능**: DSUTA는 대부분의 노이즈 환경에서 기존 방법론들을 압도하는 성능을 보였다. 특히 사전 학습 모델의 에러율이 매우 높았던 NB 도메인에서 SUTA는 WER 100%를 넘겼으나, DSUTA는 36.3%를 기록하며 강력한 적응 능력을 입증했다.
- **다중 도메인 및 시간 가변 데이터**:
  - MD-Long 시나리오에서 Dynamic Reset을 적용한 DSUTA는 WER 35.8%를 달성했다. 이는 정답 도메인 경계에서 리셋하는 Oracle Boundary(39.5%)보다 오히려 더 좋은 성능인데, 이는 Dynamic Reset이 무조건적인 리셋보다 유용한 지식을 더 오래 유지했기 때문으로 분석된다.
- **실제 데이터**: CHiME-3 데이터셋에서도 DSUTA(21.7%)가 SUTA(23.3%)나 AWMC(22.4%)보다 낮은 WER을 기록하며 일반화 성능을 보였다.
- **효율성**: DSUTA는 SUTA보다 더 적은 적응 단계($N$)로도 더 좋은 성능을 내며, 전체 런타임 시간 또한 기존 방법론들보다 빨랐다.

## 🧠 Insights & Discussion

### LII 지표의 유효성

논문은 왜 평균 LII를 지표로 선택했는지 분석한다. 실험 결과, 단순 LII나 사전 학습 모델의 손실만 사용했을 때보다, 평균 LII를 사용했을 때 인-도메인(In-domain)과 아웃-도메인(Out-of-domain) 샘플의 분포가 가장 명확하게 분리됨을 확인하였다.

### 도메인 전환 속도의 영향

도메인이 빠르게 변할 때($s=20$), 고정된 주기나 정답 경계에서 리셋하는 방식은 성능이 크게 저하된다. 반면 Dynamic Reset은 유연하게 리셋 시점을 결정함으로써 불필요한 리셋을 줄이고 다른 도메인의 지식을 효율적으로 활용하여 강건한 성능을 유지한다.

### 한계 및 비판적 해석

1. **도메인 범위의 제한**: 본 연구는 주로 배경 소음(Background Noise)에 의한 도메인 시프트에 집중했다. 억양(Accent), 화자의 특성, 말하기 스타일 등 더 복잡한 도메인 시프트에 대해서는 추가 검증이 필요하다.
2. **모델 망각 문제**: 본 논문은 추론 시점의 적응에 집중하므로, 도메인이 변함에 따라 과거의 지식을 완전히 잊어버리는 Catastrophic Forgetting 문제가 발생할 수 있다. 다만, 새로운 샘플에 즉각적으로 적응하는 구조이므로 최종 성능에는 큰 영향이 없다고 주장한다.

## 📌 TL;DR

본 논문은 ASR의 도메인 시프트 문제를 해결하기 위해 **Fast-slow TTA 프레임워크** 기반의 **DSUTA**를 제안한다. 메타 파라미터의 느린 업데이트와 개별 샘플의 빠른 적응을 결합하여 CTTA의 성능과 Non-continual TTA의 안정성을 모두 잡았으며, 특히 **Dynamic Reset Strategy**를 통해 도메인 변화를 자동으로 감지하고 리셋함으로써 시간에 따라 변하는 다중 도메인 환경에서도 최적의 성능을 낸다. 이 연구는 향후 실시간으로 환경이 변하는 실제 음성 인식 시스템의 강건성을 높이는 데 중요한 기여를 할 것으로 보인다.
