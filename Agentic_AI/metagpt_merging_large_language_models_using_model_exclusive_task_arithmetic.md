# MetaGPT: Merging Large Language Models Using Model Exclusive Task Arithmetic

Yuyan Zhou, Liang Song, Bingning Wang, Weipeng Chen (2024)

## 🧩 Problem to Solve

본 논문은 거대 언어 모델(LLM) 환경에서 여러 태스크를 동시에 수행할 수 있는 다중 작업 학습(Multi-Task Learning, MTL)의 효율적인 구현 문제를 다룬다. 일반적으로 MTL은 여러 태스크의 데이터를 모아 공동 학습(Joint Training)을 진행하지만, LLM의 경우 파라미터 규모가 너무 커서 계산 비용이 막대하며, 데이터 소유자의 개인정보 보호 문제로 인해 원시 데이터(Raw Data)를 수집하는 것이 어렵다.

이를 해결하기 위해 사전 학습된 모델에 각 태스크별 파인튜닝 모델의 가중치 차이인 Task Vector를 더하는 Task Arithmetic 방식이 제안되었다. 그러나 기존의 Task Arithmetic 방법론들은 다음과 같은 **트릴레마(Trilemma)**에 직면해 있다:

1. **최적 성능(Optimal Performance):** 최적의 하이퍼파라미터를 찾기 위해 추가 학습이나 그리드 서치가 필요하며, 이는 계산 비용이 매우 높다.
2. **계산 효율성(Computational Efficiency):** 고정된 계수(예: 0.3)를 사용하는 방식은 효율적이지만 성능이 최적이 아니다.
3. **데이터 프라이버시(Data Privacy):** 검증 셋을 이용한 최적화는 데이터 유출 위험이 있으며, 태스크 수가 늘어날수록 차원의 저주로 인해 탐색 공간이 기하급수적으로 증가한다.

따라서 본 논문의 목표는 데이터 없이(Data-agnostic), 계산 효율적이면서도, 최적의 성능을 낼 수 있는 모델 배타적(Model-exclusive) Task Arithmetic 방법론을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Task Arithmetic의 최적화 목적 함수를 수학적으로 분석하여, 데이터 없이도 최적의 스케일링 계수(Scaling Coefficient)를 결정할 수 있는 **폐쇄형 해(Closed-form solution)**를 도출한 것이다.

주요 기여 사항은 다음과 같다:

- Task Arithmetic의 최적화 목표에 대한 수학적 정식화와 성능 상한(Performance Bound)에 대한 이론적 분석을 제공한다.
- 데이터 항(Data term)과 스케일링 계수 항을 분리함으로써, 데이터 없이 모델의 가중치 정보만으로 최적의 계수를 계산할 수 있는 MetaGPT 알고리즘을 제안한다.
- 제안 방법이 기존의 Task Vector 개선 방법론(Ties-Merging, DARE 등)과 직교(Orthogonal)함을 입증하여, 이들과 결합해 성능을 추가로 향상시킬 수 있음을 보여준다.

## 📎 Related Works

**1. 모델 병합(Model Merging):**
단일 태스크 성능 향상이나 도메인 일반화를 위해 가중치 평균(Weight Averaging)이나 Fisher Merging 등이 사용되었다. 최근에는 Task Vector를 활용한 Task Arithmetic이 주목받았으며, Ties-Merging이나 DARE와 같이 Task Vector 간의 간섭을 줄이고 중복 성분을 제거하여 성능을 높이는 기법들이 제안되었다.

**2. 다중 작업 학습(Multi-Task Learning):**
전통적인 MTL은 공유 표현(Shared Representation)을 학습하기 위한 아키텍처 설계나 그래디언트 충돌 해결 등의 최적화 방법에 집중했다. 하지만 이러한 방식은 모든 태스크의 원시 데이터가 필요하므로, 데이터 프라이버시와 계산 비용 문제로 인해 LLM에 적용하기에는 한계가 있다.

**MetaGPT의 차별점:**
기존 방법들이 데이터 기반의 하이퍼파라미터 서치나 단순한 휴리스틱(Fixed value)에 의존한 반면, MetaGPT는 이론적 근거를 바탕으로 데이터 없이 모델 가중치만으로 최적의 병합 계수를 도출한다.

## 🛠️ Methodology

### 전체 파이프라인

MetaGPT는 사전 학습된 모델 $\theta_0$와 각 태스크별로 파인튜닝된 모델 $\theta_1, \dots, \theta_T$가 있을 때, 다음과 같이 병합 모델 $\theta_{final}$을 생성한다:
$$\theta_{final} = \theta_0 + \sum_{i=1}^T \lambda_i \tau_i$$
여기서 $\tau_i = \theta_i - \theta_0$는 Task Vector이며, $\lambda_i$는 각 태스크의 중요도를 결정하는 스케일링 계수이다.

### 최적화 목표 (Optimization Objective)

MetaGPT의 목표는 병합 모델과 각 개별 파인튜닝 모델 간의 평균 손실 차이인 **Average Loss Difference (ALD)**를 최소화하는 $\lambda_i$를 찾는 것이다:
$$\arg \min_{\lambda_1, \dots, \lambda_T} \frac{1}{T} \sum_{t=1}^T (L^t(\theta_{final}, x) - L^t(\theta_t, x))$$

### 이론적 근거 및 단순화 과정

데이터 없이 해를 구하기 위해 논문은 두 가지 핵심 가정을 사용한다:

1. **NTK Linearization (Property 5):** 신경망의 너비가 매우 넓을 때, 파인튜닝 과정이 선형 영역(Linear regime)에서 일어난다는 성질이다. 이는 LLM 규모의 모델에서 유효하며, 손실 함수를 2차 형식(Quadratic form)으로 근사할 수 있게 한다.
2. **Task Vector Orthogonality (Property 6):** 서로 다른 태스크의 Task Vector $\tau_i$와 $\tau_j$는 서로 직교($\tau_i^\top \tau_j = 0$)한다는 성질이다.

### 폐쇄형 해(Closed-form Solution) 도출

위의 가정들을 바탕으로 ALD를 분석하면, 데이터와 무관한 모델 전용의 최적해를 도출할 수 있다. 각 태스크 $t$에 대한 최적의 스케일링 계수 $\lambda_t$는 다음과 같이 결정된다:
$$\lambda_t = \frac{\|\theta_t - \theta_0\|^2}{\sum_{k=1}^T \|\theta_k - \theta_0\|^2}$$
즉, **특정 태스크의 계수는 해당 태스크의 Task Vector의 L2-노름 제곱을 모든 태스크 Vector들의 L2-노름 제곱의 합으로 나눈 값**으로 결정된다. 이는 별도의 데이터나 반복적인 최적화 과정 없이 가중치 값만으로 즉시 계산 가능하다.

## 📊 Results

### 실험 설정

- **모델:** Llama-2 (7B, 13B), Mistral-7B.
- **태스크:** 일반 지식(WinoGrande, AGIEval), 수학(GSM8K, MATH), 코드 생성(MBPP, HumanEval).
- **비교 대상:** Weight Average, Task Arithmetic (Fixed $\lambda=0.3$), Ties-Merging, DARE.
- **평가 지표:** 절대 평균 성능(Absolute Avg) 및 정규화된 평균 정확도(Normalized Avg).

### 주요 결과

1. **전반적 성능 향상:** Llama-2-7B 및 Mistral-7B 실험에서 MetaGPT는 대부분의 태스크에서 기존 방법론보다 우수한 성능을 보였으며, 특히 평균 지표(Abs. Avg, Nor. Avg)에서 1위를 기록하였다.
2. **모델 규모 및 아키텍처 강건성:** Llama-2-13B와 같은 더 큰 모델과 Mistral과 같은 다른 아키텍처에서도 일관되게 최적의 성능을 달성함을 확인하였다.
3. **기존 기법과의 결합:** Task Vector 자체를 개선하는 Ties-Merging 및 DARE와 결합했을 때 성능이 추가로 향상됨을 확인하였다. 이는 MetaGPT가 계수 최적화라는 독립적인 영역을 다루기 때문에 기존 기법들과 직교함을 의미한다.
4. **OOD 일반화 능력:** 학습에 사용되지 않은 외부 데이터셋(JEC-QA, FinanceIQ, MedQA)에서도 타 방법론 대비 높은 성능을 보여, 분포 외(Out-of-Distribution) 일반화 능력이 뛰어남을 입증하였다.

## 🧠 Insights & Discussion

**강점:**
MetaGPT는 이론적인 분석을 통해 Task Arithmetic의 최적해를 수학적으로 도출함으로써, 데이터 프라이버시 문제를 완벽히 해결함과 동시에 계산 복잡도를 $O(1)$ 수준으로 낮추었다. 특히 LLM의 특성인 NTK 선형성과 Task Vector의 직교성을 활용하여 실용적인 폐쇄형 해를 찾아낸 점이 탁월하다.

**한계 및 가정:**

- **공통 초기화 필요:** 본 방법론은 모든 모델이 동일한 사전 학습 모델($\theta_0$)에서 시작했다는 가정이 필수적이다.
- **모델 크기 의존성:** NTK 선형화 가정을 기반으로 하므로, 모델의 크기가 작은 경우(Wide network 가정이 깨지는 경우) 성능이 저하될 가능성이 있다.
- **직교성 가정:** 이론적으로는 Task Vector가 완전히 직교한다고 가정하지만, 실제로는 '거의' 직교하는 상태이다. 이 미세한 차이가 실제 성능에 미치는 영향에 대한 추가 논의가 필요할 수 있다.

**비판적 해석:**
제안된 $\lambda_t$ 공식은 단순하게 가중치 변화량($\|\tau_t\|^2$)이 큰 태스크에 더 많은 가중치를 부여하는 방식이다. 이는 직관적으로 '학습이 많이 된 태스크'가 중요하다고 보는 관점인데, 이것이 항상 최적의 성능으로 이어진다는 점을 수학적으로 증명하고 실험으로 입증한 것이 본 논문의 핵심 가치이다.

## 📌 TL;DR

MetaGPT는 LLM의 모델 병합 시 발생하는 데이터 프라이버시와 계산 비용 문제를 해결하기 위해, 데이터 없이 가중치만으로 최적의 병합 계수를 찾는 **Model Exclusive Task Arithmetic** 방법을 제안한다. NTK 선형화와 벡터 직교성 가정을 통해 $\lambda_t = \frac{\|\tau_t\|^2}{\sum \|\tau_k\|^2}$라는 단순한 폐쇄형 해를 도출하였으며, 이를 통해 Llama-2 및 Mistral 모델에서 기존 SOTA 병합 방법론들을 능가하는 성능과 뛰어난 OOD 일반화 능력을 보여주었다. 이 연구는 향후 데이터 없이 다중 능력을 가진 LLM을 효율적으로 구축하는 데 중요한 역할을 할 것으로 기대된다.
