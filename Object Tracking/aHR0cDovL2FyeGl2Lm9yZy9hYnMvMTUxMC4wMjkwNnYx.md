# Temporal Dynamic Appearance Modeling for Online Multi-Person Tracking

Min Yang, Yunde Jia (2015)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 복잡한 환경에서의 **Online Multi-Person Tracking (MOT)** 시 발생하는 부정확한 **Data Association** 문제이다. 

일반적으로 Online MOT는 검출된 객체(Detection)를 기존의 궤적(Trajectory)에 실시간으로 연결하는 방식으로 동작한다. 이때 궤적과 검출 결과 사이의 유사도(Affinity)를 측정하는 것이 매우 중요한데, 기존의 많은 방법론은 외형(Appearance)의 공간적 구조(Spatial Structure)만을 고려하였다. 그러나 실제 환경에서는 서로 다른 사람이 매우 유사한 외형을 가질 수 있으며, 특히 사람들이 서로 가깝게 상호작용하거나 겹치는 상황에서 외형의 공간적 정보만으로는 개별 객체를 식별하는 데 한계가 있어 Identity Switch(IDS)가 빈번하게 발생하는 문제가 있다.

따라서 본 논문의 목표는 외형의 공간적 특성뿐만 아니라, 시간이 흐름에 따라 외형이 어떻게 변하는지에 대한 **Temporal Dynamic(시계열 동적 특성)**을 모델링하여, 보다 정확하고 신뢰할 수 있는 외형 유사도를 측정함으로써 데이터 연관(Data Association)의 정확도를 높이는 것이다.

## ✨ Key Contributions

본 논문의 중심 아이디어는 한 사람의 외형 변화를 단순한 특징점들의 집합이 아니라, 시간적 의존성을 가진 **시계열 데이터(Temporal Sequence)**로 간주하는 것이다.

1.  **Temporal Dynamic Appearance Model (TDAM) 제안**: Hidden Markov Model (HMM)을 도입하여 외형의 공간적 구조와 시간적 전이 특성을 동시에 캡처하는 모델을 설계하였다.
2.  **Mid-level Semantic Feature Selection**: 고차원의 저수준(Low-level) 특징(HOG 등)은 노이즈에 취약하고 중복 정보가 많아 HMM 학습이 어렵다. 이를 해결하기 위해 저수준 특징을 의미론적 해석이 가능한 저차원의 **Mid-level feature space**로 매핑하는 특징 선택 알고리즘을 제안하였다.
3.  **Online Incremental Learning**: 실시간 추적을 위해 새로운 외형 관측치가 들어올 때마다 모델 파라미터를 점진적으로 업데이트하는 **Online Expectation-Maximization (EM)** 알고리즘을 적용하였다.
4.  **Robust Online Tracking Framework**: 제안한 TDAM을 기존의 Tracking-by-detection 프레임워크에 통합하여, 복잡한 상호작용 상황에서도 강건한 다중 인원 추적 성능을 입증하였다.

## 📎 Related Works

기존의 다중 인원 추적 연구들은 크게 두 가지 방향으로 나뉜다.

-   **Offline/Batch Approach**: 넓은 시간 윈도우 내에서 전역 최적화(Linear Programming, CRF 등)를 통해 궤적을 생성한다. 이러한 방식은 정확도가 높지만, 연산 시간이 매우 길고 결과 출력에 시간 지연(Temporal Delay)이 발생하여 실시간 응용 분야에 적용하기 어렵다.
-   **Online Approach**: 현재 프레임까지의 관측치만을 사용하여 즉각적으로 결과를 출력한다. 주로 컬러 히스토그램과 같은 Descriptor 기반의 유사도 측정이나, 온라인 학습 기반의 분류기(Classifier)를 사용하여 외형을 모델링한다.

**기존 방식의 한계 및 차별점**: 기존의 온라인 방법론들은 외형의 **공간적 구조(Spatial Structure)**만을 고려한다. 하지만 추적 대상이 모두 '사람'이라는 동일 카테고리에 속하기 때문에, 정적인 시점에서는 서로 다른 사람이라도 특징 공간 상에서 매우 유사하게 나타날 수 있다. 본 논문은 이러한 정적 분석의 한계를 극복하기 위해, 외형의 변화 과정인 **Temporal Dynamic**을 명시적으로 모델링한다는 점에서 기존 연구들과 차별화된다.

## 🛠️ Methodology

### 1. TDAM (Temporal Dynamic Appearance Model)
TDAM은 외형 시퀀스가 관찰 불가능한 마르코프 체인(Markov Chain)과 연결되어 있다는 가정하에 **HMM**을 사용하여 모델링한다.

-   **모델 파라미터**: $\theta = (\pi, A, F)$
    -   $\pi = \{\pi_i\}$: 초기 상태 분포 ($P(s_1 = i)$)
    -   $A = \{a_{ij}\}$: 상태 전이 행렬 ($P(s_{t+1} = j | s_t = i)$)
    -   $F = \{f_i(\cdot)\}$: 각 상태에 연결된 관측 밀도 함수. 본 논문에서는 **Gaussian Mixture Models (GMMs)**를 사용한다.
    $$f_i(o_t) = \sum_{k=1}^M \omega_{ik} \cdot \mathcal{N}(o_t; \mu_{ik}, \Sigma_{ik})$$
    여기서 $M$은 가우시안 성분의 개수이며, $\omega, \mu, \Sigma$는 각각 가중치, 평균, 공분산 행렬이다.

### 2. Feature Selection (Low-level $\rightarrow$ Mid-level)
HMM의 학습 효율을 높이기 위해 HOG 특징(2976차원)을 $d$차원의 의미론적 특징 공간으로 매핑한다.

-   **Clustering**: INRIA 데이터셋의 정적 이미지들을 Recursive Normalized Cuts를 통해 클러스터링하고, 각 클러스터를 대표하는 선형 분류기(Detector)를 학습한다.
-   **Selection**: 학습된 후보 Detector들 중, 궤적 내에서 특정 외형 카테고리가 얼마나 확실하게 나타나는지를 **Entropy** $H(\Omega | X, w)$로 측정하여 상위 $d$개의 '의미 있는' Detector를 선택한다.
-   **결과**: 최종적으로 HOG 특징 벡터는 선택된 $d$개 Detector들의 검출 점수(Detection Score)로 구성된 저차원 벡터로 변환된다.

### 3. Appearance Matching
새로운 관측치 $o_{t+1}$이 들어왔을 때, 최근 $L$개의 관측치로 구성된 윈도우 $W_t$를 기반으로 다음의 우도(Likelihood)를 계산한다.
$$P(o_{t+1} | W_t, \theta_t) = \sum_{j=1}^N \phi^{(t)}(j) \cdot f^{(t)}_j(o_{t+1})$$
여기서 $\phi^{(t)}(j)$는 $W_t$를 통한 Forward Procedure로 계산된 **상태 예측 확률**이며, $f^{(t)}_j(o_{t+1})$는 해당 상태의 **관측 우도**이다. 즉, 이전의 외형 변화 흐름($\phi$)과 현재의 외형 특성($f$)을 모두 고려하여 매칭을 수행한다.

### 4. Incremental Learning
모델 파라미터를 실시간으로 업데이트하기 위해 **Online EM 알고리즘**을 사용한다.

-   **E-step**: Forward-Backward Procedure를 통해 기대 충분 통계량(Expected Sufficient Statistics) $\xi, \gamma, m, C$를 계산한다.
-   **Accumulation**: 과거의 지식을 유지하기 위해 학습률 $\eta$를 사용하여 통계량을 누적 업데이트한다.
    $$\hat{\xi}^{(t+1)}_{ij} = (1-\eta) \cdot \hat{\xi}^{(t)}_{ij} + \eta \cdot \xi^{(t)}_{ij}$$
-   **M-step**: 누적된 통계량을 사용하여 전이 행렬 $A$와 GMM 파라미터 $\omega, \mu, \Sigma$를 갱신한다.

### 5. Online Tracking Framework
전체 시스템은 **Tracking-by-detection** 구조를 따르며, 헝가리안 알고리즘(Hungarian Algorithm)을 통해 데이터 연관을 수행한다.

-   **Association Cost**: $\Psi_{pq} = -\log(\rho(X^p_t, z^q_{t+1}))$
-   **Total Affinity**: $\rho = \rho^A \cdot \rho^M \cdot \rho^S$
    -   $\rho^A$: TDAM을 통한 외형 유사도 $\rho^A = P(o_{q_{t+1}} | \theta^p_t)$
    -   $\rho^M$: Constant Velocity 모델 기반의 모션 유사도 (Gaussian distribution)
    -   $\rho^S$: 높이와 너비를 이용한 형상 유사도 (Exponential decay)

## 📊 Results

### 1. 실험 설정
-   **데이터셋**: MOTChallenge 2015 (2D MOT)
-   **지표**: MOTA(Accuracy), MOTP(Precision), IDS(Identity Switches), MT(Mostly Tracked), ML(Mostly Lost) 등
-   **파라미터**: $d=64$ (특징 차원), $N=8$ (HMM 상태 수), $M=3$ (GMM 성분 수), $L=8$ (윈도우 길이)

### 2. 정량적 결과
-   **진단 분석 (Diagnosis Analysis)**: 
    -   단순 거리 측정(a1)이나 temporal dependency를 제거한 모델(a2)보다 TDAM(p1)이 MOTA, MT, ML 모든 지표에서 월등한 성능을 보였다. 특히 IDS를 크게 낮추었다.
    -   특징 공간 비교 시, PCA(a3)나 LLE(a4)보다 제안한 Mid-level semantic feature가 Temporal Dynamic을 캡처하는 데 훨씬 유리함이 증명되었다.
-   **SOTA 비교**:
    -   Online 방법론(TCODAL, RMOT)뿐만 아니라 많은 Offline 방법론들보다도 높은 MOTA(33.0%)를 기록하였다.
    -   특히 **IDS(Identity Switch) 수치**에서 압도적인 우위를 점하며, 이는 외형의 동적 특성을 이용한 식별 능력이 매우 강력함을 시사한다.

### 3. 분석 및 속도
-   **속도**: Intel Core i7 PC에서 약 5 fps로 동작한다. 가장 연산량이 많은 부분은 TDAM의 파라미터 업데이트(전체 연산의 50%)이며, 이는 병렬 처리를 통해 개선 가능하다.
-   **한계**: 온라인 방식의 특성상 장기 폐색(Long-term occlusion)이나 갑작스러운 움직임이 발생할 경우 궤적이 단절되어 FG(Fragmentation) 점수가 오프라인 방식보다 다소 높게 나타나는 경향이 있다.

## 🧠 Insights & Discussion

본 논문은 외형 모델링에서 **'시간적 흐름'**이라는 요소를 도입하여 MOT의 고질적인 문제인 Identity Switch를 효과적으로 억제하였다.

**강점**:
-   정적인 외형 유사도가 높은(비슷하게 생긴) 객체들이 섞여 있을 때, 각 객체가 가지는 고유한 외형 변화 패턴(Temporal Dynamic)을 통해 이를 구분해 낼 수 있다는 점을 이론적, 실험적으로 입증하였다.
-   저수준 특징을 의미론적 공간으로 매핑하여 HMM의 학습 가능성(Trainability)과 효율성을 동시에 확보하였다.

**한계 및 비판적 해석**:
-   **Online vs Offline**: IDS는 매우 낮으나 FG(단절)가 높은 이유는 미래 프레임을 보지 못하는 온라인 모델의 근본적 한계이다. 저자 또한 이를 인정하며, 짧은 지연(Latency)을 허용하고 미래 프레임을 일부 고려하는 방식으로 확장할 가능성을 언급하였다.
-   **독립적 학습**: 현재 모델은 각 사람의 TDAM을 독립적으로 학습한다. 하지만 실제로는 사람 간의 외형 차이(Discriminative learning)를 명시적으로 학습하는 것이 데이터 연관의 모호성을 더 줄일 수 있을 것이다.

## 📌 TL;DR

본 연구는 온라인 다중 인원 추적에서 외형의 공간적 특성뿐만 아니라 **시간적 변화 패턴(Temporal Dynamic)**을 캡처하기 위해 **HMM 기반의 TDAM**을 제안하였다. 저차원의 의미론적 특징 공간(Mid-level features)과 점진적 학습(Incremental Learning)을 통해 실시간성을 확보하였으며, MOTChallenge 2015 벤치마크에서 SOTA 성능(특히 매우 낮은 IDS)을 달성하였다. 이 연구는 외형의 시계열적 특성이 다중 객체 식별에 결정적인 역할을 함을 보였으며, 향후 실시간 추적 시스템의 강건성을 높이는 데 기여할 가능성이 크다.