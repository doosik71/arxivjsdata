# TOWARDS FOUNDATION MODELS FOR ZERO-SHOT TIME SERIES ANOMALY DETECTION: LEVERAGING SYNTHETIC DATA AND RELATIVE CONTEXT DISCREPANCY

Tian Lan, Hao Duong Le, Jinbo Li, Wenjun He, Meng Wang, Chenghao Liu, Chen Zhang (2025)

## 🧩 Problem to Solve

본 논문은 시계열 이상치 탐지(Time Series Anomaly Detection, TSAD) 분야에서 새로운 데이터셋에 대해 추가 학습 없이 바로 적용 가능한 Zero-shot 일반화 성능을 확보하는 문제를 해결하고자 한다.

기존의 TSAD 파운데이션 모델들은 주로 재구성 기반(Reconstruction-based) 목적 함수에 의존한다. 하지만 이러한 방식은 '목적 함수의 불일치(Objective Mismatch)'라는 근본적인 한계를 가진다. 구체적으로, 재구성 모델은 지배적인 패턴을 학습하려는 경향이 있어 미세하거나 문맥적인 이상치(Subtle/Contextual anomalies)를 정상 패턴으로 매끄럽게 재구성하여 탐지하지 못하는 거짓 음성(False Negative) 문제가 발생한다. 반대로, 학습 데이터에서 보지 못한 복잡한 정상 패턴을 제대로 재구성하지 못해 이를 이상치로 오인하는 거짓 양성(False Positive) 문제 역시 빈번하게 발생한다.

또한, 실제 세계의 데이터는 라벨링 된 이상치 샘플이 매우 희소하며, 데이터의 다양성이 부족하여 Zero-shot 환경에서 본 적 없는 새로운 유형의 정상/이상 패턴에 대응하기 어렵다는 문제가 있다. 따라서 본 연구의 목표는 재구성 방식의 한계를 극복하는 새로운 사전 학습 패러다임을 제시하고, 이를 지원할 대규모 합성 데이터셋을 구축하여 강건한 Zero-shot TSAD 파운데이션 모델을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 입력을 단순 재구성하는 대신, 인접한 시간 윈도우 간의 유의미한 차이를 식별하는 **Relative Context Discrepancy (RCD)** 전략을 사용하는 것이다.

RCD의 중심 직관은 많은 이상치, 특히 문맥적 이상치는 단일 윈도우 내의 절대적인 값보다는 주변 문맥과의 상대적인 차이(Discrepancy)를 통해 더 효과적으로 식별될 수 있다는 점이다. 이를 위해 표준 Transformer 아키텍처의 Self-attention 메커니즘을 활용하여 윈도우 간의 관계적 차이를 캡처하도록 설계하였다.

또한, 이러한 RCD 학습을 가능하게 하기 위해 토큰 레벨의 이상치 라벨이 포함된 대규모의 다양한 합성 코퍼스(Synthetic Corpus)를 구축하여, 모델이 정교한 감독 신호(Supervisory signal)를 통해 이상치 탐지 전략을 직접 학습할 수 있도록 하였다.

## 📎 Related Works

기존의 TSAD 접근 방식은 크게 두 가지로 나뉜다. 첫째는 도메인 특화 데이터로 학습하는 전통적인 비지도 학습 방식(재구성 기반 및 대조 학습 기반)이며, 둘째는 다양한 태스크를 수행하려는 일반 목적 파운데이션 모델 또는 TSAD 전용 파운데이션 모델이다.

기존의 파운데이션 모델들은 주로 실제 데이터를 기반으로 재구성 오차를 통해 이상치를 간접적으로 추론한다. 그러나 본 논문은 앞서 언급한 '목적 함수의 불일치'로 인해 이러한 방식이 Zero-shot 설정에서 신뢰도가 낮음을 지적한다. 일부 연구에서 데이터 증강(Data Augmentation)을 통해 인위적인 이상치를 주입하는 시도가 있었으나, 이는 여전히 기반이 되는 실제 데이터의 다양성에 의존한다는 한계가 있다. 본 연구는 실제 데이터에 의존하지 않고 전적으로 정교하게 설계된 합성 데이터만을 사용하여 사전 학습을 수행함으로써 기존 접근 방식과 차별화된다.

## 🛠️ Methodology

### 1. TimeRCD 아키텍처 및 RCD 전략

TimeRCD는 별도의 구조적 수정 없이 표준 **Encoder-only Transformer**를 기반으로 한다.

* **Variate-Window Tokenization**: 다변량 시계열 데이터를 처리하기 위해 모든 변수를 하나의 시퀀스로 평탄화(Flatten)하여 윈도우 단위로 분할하고, 이를 선형 투영을 통해 입력 토큰 임베딩 $H^{inp} \in \mathbb{R}^{\lceil n/W \rceil d \times D_v}$로 변환한다. 여기서 $W$는 윈도우 길이, $d$는 변수 개수, $D_v$는 임베딩 차원이다.
* **RCD 메커니즘**: Transformer의 Self-attention은 각 토큰(윈도우)이 다른 토큰들과의 관계를 계산하므로, 자연스럽게 윈도우 간의 상대적 문맥 차이(RCD)를 캡처하는 기능을 수행하게 된다.
* **출력 헤드 (Anomaly & Reconstruction Heads)**:
  * **Anomaly Head**: 학습된 변별적 특징을 사용하여 윈도우 레벨의 이상치 점수를 출력한다.
  * **Reconstruction Head**: 마스킹 된 입력 부분의 재구성을 예측하는 보조 태스크를 수행한다. 추론 시에는 제거되지만, 학습 과정에서 모델이 단순한 관계적 차이를 넘어 시계열의 풍부하고 안정적인 임베딩을 학습하도록 돕는다.

### 2. 합성 데이터 생성 엔진 (Synthetic Data Generation)

RCD 학습을 위해 총 4단계의 계층적 프로세스로 합성 데이터를 생성한다.

* **Stage 1: 단변량 문맥 생성**: 다음과 같은 가법 모델(Additive template)을 통해 기본 신호를 생성한다.
    $$x_{base}(t) = T(t) + S(t) + \epsilon(t)$$
    여기서 $T(t)$는 결정론적/확률적 추세(Trend), $S(t)$는 다양한 파형(sinusoid, square, triangle, wavelet)의 혼합인 계절성(Seasonality), $\epsilon(t)$는 가변 변동성을 가진 노이즈이다.
* **Stage 2: 다변량 인과 문맥 융합**: DAG(Directed Acyclic Graph)와 ARX(AutoRegressive with eXogenous inputs) 시스템을 사용하여 변수 간 인과 관계를 설정한다.
    $$z_i[t] = a_i z_i[t-1] + \sum_{j \in P(i)} b_{ij} x_j[t - \ell_{ij}] + c_i$$
    최종 관측값 $x_i[t]$는 기본 신호 $x_{base,i}(t)$와 인과 채널 $z_i[t]$의 혼합으로 생성된다.
* **Stage 3: 인과-문맥적 이상치 주입**:
  * **외생적 주입(Exogenous)**: 생성된 정상 시스템의 특정 구간을 단순 덮어쓰기 하는 방식이다.
  * **내생적 주입(Endogenous)**: 부모 노드의 기본 신호에 변동을 주어 ARX 동역학을 통해 자식 노드로 이상치가 유기적으로 전파되게 하는 방식이다. 이는 실제 시스템의 내부 고장을 모사하며, 루트 원인(Root-cause)과 전파 효과를 구분하여 학습하게 한다.
* **Stage 4: 라벨링 및 마스킹**: 토큰 레벨의 이진 라벨을 생성하며, 내생적 주입의 경우 루트 원인 윈도우와 전파된 효과 윈도우를 모두 라벨링 한다.

## 📊 Results

### 1. 실험 설정

* **데이터셋**: univariate(IOPS, NAB, YAHOO 등) 및 multivariate(SWaT, SMD, MSL 등)를 포함한 총 14개의 공개 데이터셋을 사용하였다.
* **비교 대상**: Zero-shot 모델(MOMENT, Chronos, TimesFM 등) 및 각 데이터셋에 최적화하여 학습된 Full-shot 모델(TranAD, USAD, IForest 등)과 비교하였다.
* **평가 지표**: Affiliation-F1, F1-T, Standard-F1, VUS-PR의 4가지 지표를 사용하였다.

### 2. 주요 결과

* **Zero-shot 성능**: TimeRCD는 모든 Zero-shot 모델 중 SOTA 성능을 달성하였다. 특히 단변량 사례 56건 중 41건에서 1위를 기록하였다.
* **Full-shot 모델과의 비교**: 대상 데이터로 직접 학습한 Full-shot 모델들과 비교했을 때도 매우 경쟁력 있는 성능을 보였으며, 28개 항목에서 1위를 차지하였다. 이는 사전 학습된 파운데이션 모델이 도메인 특화 모델에 근접하는 일반화 능력을 갖췄음을 시사한다.
* **문맥적 이상치 탐지 능력**: 포인트 이상치(Point anomalies)에서는 다른 모델들과 유사한 성능을 보였으나, 문맥적 이상치(Contextual anomalies)에서는 다른 모델들이 성능 급락을 보이는 반면 TimeRCD는 Standard-F1 0.827이라는 압도적인 성능을 유지하였다.
* **윈도우 크기 영향**: 시계열의 패턴이 긴 데이터셋(UCR, Power 등)에서는 입력 윈도우 크기가 커질수록 성능이 향상되는 경향을 보였다.
* **데이터 스케일링 법칙**: 사전 학습 데이터의 양을 350M $\rightarrow$ 700M $\rightarrow$ 2.5B로 늘림에 따라 모든 지표에서 성능이 일관되게 향상되는 긍정적인 스케일링 법칙(Positive scaling law)을 확인하였다.

## 🧠 Insights & Discussion

본 연구는 재구성 기반의 목적 함수가 가진 근본적인 한계를 RCD라는 새로운 패러다임으로 성공적으로 해결하였다. 특히, 단순한 데이터 증강이 아니라 시스템 수준의 인과 관계와 다양한 문맥적 위반 사례를 포함한 합성 데이터 커리큘럼을 구축함으로써, 모델이 '상대적 차이'를 인식하는 법을 배우게 한 점이 주효했다.

실험을 통해 밝혀진 흥미로운 점은 **재구성 헤드(Reconstruction Head)**의 역할이다. 비록 추론 시에는 사용되지 않지만, 이 보조 태스크가 없으면 이상치 탐지 헤드의 성능이 거의 완전히 붕괴된다는 것이 확인되었다. 이는 재구성 태스크가 시계열의 기본 구조에 대한 기초적인 표현 학습을 제공하며, 그 기반 위에서 RCD가 세밀한 이상치를 식별할 수 있음을 의미한다.

한계점으로는 현재 표준 Transformer 백본을 사용하고 있다는 점이 있으며, 향후 RCD를 더 효율적으로 캡처할 수 있는 특화된 네트워크 구조를 탐색한다면 추가적인 성능 향상이 가능할 것으로 보인다.

## 📌 TL;DR

TimeRCD는 재구성 오차에 의존하던 기존 TSAD 방식에서 벗어나, 인접 윈도우 간의 **상대적 문맥 차이(Relative Context Discrepancy)**를 명시적으로 학습하는 새로운 Zero-shot 파운데이션 모델이다. 대규모로 정교하게 설계된 합성 데이터셋(2.5B points)을 통해 사전 학습되었으며, 특히 기존 모델들이 어려워하던 **문맥적 이상치 탐지**에서 압도적인 성능을 보인다. 이 연구는 합성 데이터 커리큘럼과 관계적 학습 전략이 시계열 이상치 탐지의 일반화 성능을 극대화할 수 있음을 입증하였다.
