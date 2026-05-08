# Precursor-of-Anomaly Detection for Irregular Time Series

Sheo Yon Jhin, Jaehoon Lee, Noseong Park (2023)

## 🧩 Problem to Solve

본 논문은 시계열 데이터에서 발생하는 이상치 탐지(Anomaly Detection)의 패러다임을 확장하여, 이상 징후가 실제로 발생하기 전에 이를 미리 예측하는 **Precursor-of-Anomaly (PoA) Detection**이라는 새로운 태스크를 정의하고 이를 해결하는 것을 목표로 한다.

기존의 이상치 탐지 연구는 주어진 관측치 내에서 현재 상태가 정상인지 혹은 이상인지(Anomaly vs Normal)를 판별하는 데 집중하였다. 그러나 현실 세계의 금융, 의료, 지질학 등의 분야에서는 이상 현상이 발생한 후 대응하는 것보다, 발생 전 징후(Precursor)를 포착하여 예방 조치를 취하는 것이 훨씬 중요하다. 또한, 실제 시계열 데이터는 관측 간격이 일정하지 않은 **Irregular Time Series**인 경우가 많으며, 기존의 Regular Time Series 기반 모델들은 이러한 불규칙한 구조에서 성능이 저하되는 한계가 있다.

따라서 본 논문은 불규칙한 다변량 시계열 데이터에서 '현재의 이상치 탐지'와 '미래의 이상 징후(PoA) 탐지'라는 두 가지 문제를 동시에 해결하는 통합 프레임워크를 구축하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같다.

1. **새로운 태스크 정의**: 일반적인 도메인에서 이상치 발생 전의 징후를 포착하는 Precursor-of-Anomaly (PoA) Detection 태스크를 최초로 제안하였다.
2. **PAD 프레임워크 제안**: Neural Controlled Differential Equations (NCDEs)를 기반으로 하여, 이상치 탐지와 PoA 탐지를 동시에 수행하는 통합 프레임워크인 PAD를 제안하였다.
3. **학습 전략 설계**: 두 태스크 간의 지식 전이를 위해 Multi-task Learning (MTL)과 Knowledge Distillation (KD) 기반의 학습 알고리즘을 설계하였다.
4. **자기지도 학습(Self-supervised Learning) 적용**: 레이블이 부족한 이상치 탐지 문제의 특성을 극복하기 위해, 데이터 증강(Data Augmentation)을 통해 인위적인 이상치를 생성하고 이를 활용한 자기지도 학습 방식을 도입하였다.
5. **강건성 입증**: 불규칙한 시계열 환경(데이터 누락 상황)에서도 NCDE의 연속적 특성을 활용하여 기존 SOTA 모델들보다 뛰어난 성능을 보임을 입증하였다.

## 📎 Related Works

### 1. 시계열 이상치 탐지 (Time Series Anomaly Detection)

기존 연구들은 크게 네 가지 범주로 나뉜다.

- **Classical Methods**: OCSVM, Isolation Forest 등이 있으며 전통적인 통계 및 머신러닝 방식을 사용한다.
- **Clustering-based**: Deep-SVDD, ITAD 등이 있으며 정상 데이터의 클러스터에서 벗어난 지점을 탐색한다. 복잡한 데이터에서는 이상 패턴을 정의하기 어렵다는 한계가 있다.
- **Density-estimation-based**: LOF, DAGMM 등이 있으며 확률 밀도 함수를 추정하여 저밀도 영역을 이상치로 간주한다. 비정상성(Non-stationary) 패턴이 강한 데이터에서는 성능이 낮고 계산 비용이 높다.
- **Reconstruction-based**: USAD, OmniAnomaly 등이 있으며 데이터를 저차원으로 압축 후 복원할 때 발생하는 복원 오차(Reconstruction Error)를 이용한다. 시계열의 시간적 의존성을 모델링하는 데 한계가 있을 수 있다.

### 2. Neural Controlled Differential Equations (NCDEs)

NCDE는 RNN의 연속 시간 버전으로 볼 수 있으며, 이산적인 관측치를 연속적인 경로 $\mathcal{X}(t)$로 보간(Interpolation)하여 처리한다. 기존의 Neural ODE가 고정된 함수를 따라 상태를 변화시킨다면, NCDE는 입력 데이터의 경로 $\mathcal{X}(t)$에 의해 제어되는 미분 방정식을 사용하므로 불규칙한 시계열 데이터를 처리하는 데 매우 강력한 도구가 된다.

### 3. Multi-task Learning 및 Knowledge Distillation

- **MTL**: 여러 관련 태스크를 동시에 학습하여 공유 파라미터를 통해 일반화 성능을 높이는 기법이다.
- **KD**: 복잡한 Teacher 모델의 지식을 단순한 Student 모델로 전이하는 기법이다. 본 논문에서는 이상치 탐지 모델(Teacher)의 판단 능력을 PoA 탐지 모델(Student)에게 전이하는 데 사용한다.

## 🛠️ Methodology

### 1. 전체 파이프라인 및 문제 정의

입력 시계열 $x_{0:T}$를 윈도우 크기 $b$인 겹치지 않는 윈도우 $w_i$들로 나눈다.

- **Anomaly Detection**: 현재 윈도우 $w_i$에 이상치가 포함되어 있는지 판별하는 이진 분류 문제이다.
- **PoA Detection**: 현재 윈도우 $w_i$를 보고, 바로 다음 윈도우 $w_{i+1}$에 이상치가 발생할지 예측하는 이진 분류 문제이다.

### 2. Co-evolving NCDEs 아키텍처

PAD는 서로 상호작용하며 진화하는 두 개의 NCDE 레이어를 사용한다. 하나는 이상치 탐지($h$)를 위한 것이고, 다른 하나는 PoA 탐지($z$)를 위한 것이다.

두 상태 벡터의 변화는 다음과 같은 미분 방정식으로 정의된다.
$$h(T) = h(0) + \int_{0}^{T} f(h(t); \theta_f, \theta_c) \frac{d\mathcal{X}(t)}{dt} dt$$
$$z(T) = z(0) + \int_{0}^{T} g(z(t); \theta_g, \theta_c) \frac{d\mathcal{X}(t)}{dt} dt$$

여기서 $\theta_f$와 $\theta_g$는 각 태스크 전용 파라미터이며, $\theta_c$는 두 네트워크가 공유하는 파라미터이다. 이를 통해 두 모델은 서로 다른 목표를 가지면서도 시계열의 핵심 패턴을 캡처하는 공통 지식을 공유하는 **Task-specific parameter sharing** 구조를 가진다.

최종 출력은 각각 Sigmoid 함수를 통과하여 확률값으로 도출된다.
$$\hat{y}_i^a = \sigma(FC_{\theta_a}(h(T))), \quad \hat{y}_i^p = \sigma(FC_{\theta_p}(z(T)))$$

### 3. 학습 절차 및 손실 함수

본 모델은 **Knowledge Distillation**을 통해 PoA 탐지 능력을 학습한다. PoA NCDE($z$)는 현재 윈도우($i$)를 입력받아, Anomaly NCDE($h$)가 다음 윈도우($i+1$)를 보고 내린 판단을 모방하도록 학습된다.

**손실 함수**:

- **이상치 탐지 손실**: $\mathcal{L}_a = CE(\hat{y}_i^a, y_i)$ (정답 레이블 $y_i$와의 교차 엔트로피)
- **지식 증류 손실 (PoA)**: $\mathcal{L}_{KD} = CE(\hat{y}_{i+1}^a, \hat{y}_{i+1}^p)$ (Teacher인 Anomaly NCDE의 예측치와 Student인 PoA NCDE의 예측치 간의 차이)

학습 시에는 메모리 효율을 위해 **Adjoint Sensitivity Method**를 사용하여 그래디언트를 계산하며, 이는 메모리 복잡도를 $O(T + L)$ 수준으로 유지하게 해준다.

### 4. 자기지도 학습을 위한 데이터 증강

훈련 데이터에 이상치 레이블이 없는 경우가 많으므로, 정상 데이터를 리샘플링하여 인위적인 이상치를 생성하는 방식을 사용한다.

- 전체 데이터에서 특정 비율 $\gamma$만큼 이상치를 생성한다.
- 무작위 시작점과 무작위 길이(100~500 시퀀스)를 정해 기존 데이터를 복사-붙여넣기(Copy-and-paste) 하여 이상 패턴을 만든다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: MSL (NASA 우주선 상태 데이터), SWaT (산업용 수처리 시설), WADI (SWaT 확장 버전)
- **지표**: Precision, Recall, F1-score
- **비교 대상**: 17개의 베이스라인 (Classical, Clustering, Density-estimation, Reconstruction-based 모델들)

### 2. 정량적 결과

- **Regular Time Series**: PAD(Anomaly)는 MSL, SWaT, WADI 모든 데이터셋에서 가장 높은 F1-score를 기록하였다. 특히 WADI 데이터셋에서는 InterFusion이 두 번째로 좋은 성능을 보였다.
- **Irregular Time Series (Dropping Ratio)**: 데이터를 30%, 50%, 70% 무작위로 삭제하여 불규칙한 환경을 조성했을 때, 다른 베이스라인들은 성능이 급격히 하락하는 반면, PAD는 모든 조건에서 F1-score 90% 이상을 유지하며 압도적인 강건성을 보였다.
- **PoA Detection**: PoA 태스크에서 PAD는 LSTM, LSTM-VAE, USAD 등 재구성 기반 모델들보다 월등히 높은 성능을 보였다. (예: WADI 데이터셋에서 PAD PoA F1-score 92.71% vs USAD 57.40%)

### 3. 분석 및 시각화

시각화 결과, PAD는 실제 이상치가 발생하기 전 구간(붉은색 표시)을 정확하게 예측하여 PoA 탐지 능력이 실질적으로 작동함을 확인하였다.

## 🧠 Insights & Discussion

### 강점

- **NCDE의 도입**: 시계열을 연속적인 경로로 처리함으로써 데이터 누락이 심한 불규칙 시계열에서도 성능 저하 없이 강건한 탐지가 가능하다는 점이 가장 큰 강점이다.
- **통합 프레임워크**: AD와 PoA를 별개의 태스크로 보지 않고, KD와 MTL을 통해 상호 보완적으로 학습시킴으로써 단일 모델로 두 가지 목적을 달성하였다.

### 한계 및 논의사항

- **데이터 증강 의존성**: 본 연구는 self-supervised 학습을 위해 인위적인 이상치 생성 방식을 사용하였다. 하지만 실제 현실의 이상치 패턴이 단순한 리샘플링/복사-붙여넣기 방식으로 생성된 패턴과 얼마나 일치하는지에 대한 검증이 더 필요하다. 저자 또한 결론에서 향후 전처리가 필요 없는 완전한 비지도(Unsupervised) PoA 탐지 연구가 필요함을 언급하였다.
- **계산 복잡도**: 두 개의 NCDE를 동시에 풀어야 하므로, 단일 NCDE 모델보다는 계산 비용이 증가한다. 비록 Adjoint method로 메모리 문제를 해결했지만, 추론 속도에 대한 상세 분석은 부족하다.

## 📌 TL;DR

본 논문은 이상 현상이 발생하기 전 징후를 포착하는 **Precursor-of-Anomaly (PoA) Detection**이라는 새로운 태스크를 제안하고, 이를 위해 **NCDE 기반의 통합 프레임워크 PAD**를 개발하였다. NCDE의 연속 시간 모델링 특성과 Multi-task Learning, Knowledge Distillation을 결합하여 불규칙한 시계열 데이터에서도 기존 SOTA 모델들을 압도하는 이상치 및 징후 탐지 성능을 달성하였다. 이 연구는 특히 데이터 누락이 빈번한 실제 산업 현장의 시계열 모니터링 시스템에 적용되어 사고를 예방하는 데 기여할 가능성이 높다.
