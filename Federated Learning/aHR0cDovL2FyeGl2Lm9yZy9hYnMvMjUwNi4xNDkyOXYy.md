# FedOne: Query-Efficient Federated Learning for Black-box Discrete Prompt Learning

Ganyu Wang, Jinjie Fang, Maxwell J. Yin, Bin Gu, Xi Chen, Boyu Wang, Yi Chang, Charles Ling (2025)

## 🧩 Problem to Solve

본 논문은 클라우드 기반의 대규모 언어 모델(Large Language Model, LLM)을 활용한 **Black-box Discrete Prompt Learning (BDPL)** 환경에서 연합 학습(Federated Learning, FL)을 적용할 때 발생하는 **쿼리 비용(Query Cost) 문제**를 해결하고자 한다.

일반적인 Black-box prompt tuning은 모델의 내부 파라미터나 그래디언트에 접근할 수 없는 상태에서 입력 프롬프트만을 최적화하는 방식이다. 이를 연합 학습 체계로 확장하면 다양한 소스의 데이터를 활용해 프롬프트 성능을 높일 수 있다는 장점이 있다. 그러나 기존의 연합 학습 기반 Black-box 프롬프트 튜닝 연구들은 클라우드 LLM 서비스 API를 호출할 때마다 발생하는 상당한 비용과 자원 소모를 간과하였다. 특히, 매 라운드 여러 클라이언트를 참여시키는 기존의 $\text{FedAvg}$ 방식은 쿼리 횟수를 선형적으로 증가시켜 실제 서비스 적용 시 막대한 비용 부담을 초래한다.

따라서 본 논문의 목표는 클라우드 LLM 서비스와의 상호작용 시 **쿼리 효율성(Query Efficiency)을 극대화**할 수 있는 연합 학습 프레임워크를 제안하고, 이에 대한 이론적 근거를 제시하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **"매 라운드 단 한 명의 클라이언트만 활성화하는 것이 쿼리 효율성 측면에서 최적이다"**라는 직관이다. 이를 바탕으로 다음과 같은 기여를 한다.

1. **쿼리 효율성 분석**: 연합 Black-box 프롬프트 학습에서 클라이언트 활성화 수($K^*$)와 수렴 속도, 그리고 전체 쿼리 비용 간의 상관관계를 이론적으로 분석하였다.
2. **FedOne 프레임워크 제안**: 분석 결과를 바탕으로 매 라운드 단 하나의 클라이언트만 선택하여 학습 및 집계하는 $\text{FedOne}$ 프레임워크를 제안하였다.
3. **최초의 수렴성 증명**: Federated BDPL 환경에서 이산 프롬프트를 최적화할 때의 수렴성 분석을 최초로 수행하여, $\text{FedOne}$이 이론적으로 타당함을 입증하였다.

## 📎 Related Works

### 1. White-box vs Black-box Prompt Tuning

- **White-box**: 모델의 중간 표현(intermediate representations)이나 그래디언트에 접근 가능하다. $\text{Prompt-tuning}$이나 $\text{P-tuning v2}$가 대표적이며, 상대적으로 계산 효율이 좋으나 폐쇄형 LLM(Closed-source LLM)에는 적용할 수 없다.
- **Black-box**: 모델 내부 구조나 가중치에 접근할 수 없고 오직 API 출력값만을 이용한다. $\text{BDPL}$과 같이 이산 프롬프트를 최적화하는 방식은 이식성이 높고 클라우드 API 환경에 적합하지만, 수많은 forward pass가 필요하여 계산 비용이 매우 높다.

### 2. Federated Learning (FL) 및 기존 접근법의 한계

기존 연합 학습 연구들은 주로 데이터 이질성(Heterogeneity)이나 프라이버시 보호에 집중하였다. 연합 프롬프트 튜닝의 경우 대부분 White-box 시나리오를 가정하였으며, Black-box 시나리오를 다룬 일부 연구($\text{Lin et al., 2023}$ 등)조차도 클라우드 API 호출 비용이라는 실질적인 경제적/자원적 제약 조건을 고려하지 않았다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

시스템은 하나의 중앙 집계 서버(Aggregation Server)와 $K$개의 클라이언트로 구성된다. 각 클라이언트는 로컬 데이터셋 $\mathcal{D}_k = \{\Psi^k, \mathcal{Y}^k\}$를 가지며, 서버와 협력하여 전역 손실 함수 $\mathcal{L}(\Phi; \Psi)$를 최소화하는 최적의 프롬프트 $\Phi$를 찾는다.

### 2. Black-box Discrete Prompt Learning (BDPL) 과정

클라이언트는 LLM의 파라미터를 수정하는 대신, 학습 가능한 파라미터 $\alpha_k \in \mathbb{R}^{n \times N}$를 통해 이산 프롬프트 $\Phi_k$를 생성한다.

- **프롬프트 생성 (Gumbel-Softmax)**:
  이산적인 토큰 선택 과정은 미분이 불가능하므로, $\text{Gumbel-Softmax}$ 기법을 사용하여 재매개변수화(Re-parameterization)한다. 토큰 선택 확률 $p_{k,i,j}$는 다음과 같이 계산된다.
  $$p_{k,i,j} = \frac{\exp\left(\frac{\log(\alpha_{k,i,j}) + g_{k,i,j}}{\tau}\right)}{\sum_{l=1}^{N} \exp\left(\frac{\log(\alpha_{k,i,l}) + g_{k,i,l}}{\tau}\right)}$$
  여기서 $\tau$는 temperature 파라미터, $g$는 Gumbel noise이다.

- **그래디언트 추정 (MB-SVRP)**:
  이산 프롬프트 샘플링으로 인한 높은 분산을 줄이기 위해 $\text{MB-SVRP (Mini-batch Stochastic Variance-Reduced Policy)}$ 추정량을 사용한다. $I$번의 샘플링을 통해 평균 손실 $\ell_{\text{avg}}$를 구하고, 이를 이용해 그래디언트를 계산한다.
  $$\hat{\nabla}_{\alpha_{k,i}} f_k(\alpha_k, B_k) = \frac{1}{I-1} \sum_{r=1}^{I} \left[ (\ell(\Phi_{k,r}; B_k) - \ell_{\text{avg}}) \nabla_{\alpha_{k,i}} \log P(\phi_{k,r,i}) \right]$$
  이 과정에서 클라이언트는 샘플링된 프롬프트 $\Phi_k$와 미니배치 $B_k$를 클라우드 LLM 서버에 전송하고, 서버로부터 손실 값 $\ell$을 반환받는다.

### 3. FedOne 알고리즘

$\text{FedOne}$은 $\text{FedAvg}$의 구조를 따르되, 매 라운드 선택되는 클라이언트의 수 $K^*$를 **1**로 고정한다.

- **절차**:
  1. 서버가 $\alpha$를 초기화하고 1명의 클라이언트를 무작위로 선택한다.
  2. 선택된 클라이언트는 로컬 데이터와 클라우드 LLM API를 이용하여 $\alpha_k$를 $E$번의 에포크 동안 업데이트한다.
  3. 업데이트된 $\alpha_k$를 서버로 전송하면, 서버는 이를 새로운 전역 $\alpha$로 설정한다.
  4. 이 과정을 $S$번 반복한다.

### 4. 이론적 분석 (Query Efficiency)

논문은 $\epsilon$-솔루션에 도달하기 위해 필요한 최소 쿼리 수 $Q_\epsilon(K^*)$를 다음과 같이 도출하였다.
$$Q_\epsilon(K^*) = c \left[ \frac{c_1}{\sqrt{K^*}} + \frac{c_2}{K^*} \right]^2$$
분석 결과, $Q_\epsilon(K^*)$는 $K^* \ge 1$ 범위에서 단조 증가하는 함수임이 밝혀졌다. 즉, 참여 클라이언트 수를 늘려 수렴 속도를 약간 높이는 것보다, 단 한 명의 클라이언트만 참여시켜 쿼리 횟수를 획기적으로 줄이는 것이 전체 시스템 비용 관점에서 훨씬 효율적이라는 결론에 도달한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: GLUE 벤치마크 (MNLI, QQP, SST-2, MRPC, CoLA, QNLI, RTE).
- **모델**: RoBERTa-large 및 실제 클라우드 모델인 GPT-3.5 Turbo.
- **비교 대상**: Manual Prompt, In-Context Learning, Fine-tuning, White-box 방법 ($\text{Prompt-Tuning}, \text{P-Tuning v2}$), Black-box 방법 ($\text{BBT}, \text{BDPL}, \text{GS-BDPL}$).

### 2. 주요 결과

- **정확도 (Accuracy)**: $\text{FedOne-GS-BDPL}$은 RoBERTa-large 실험에서 평균 54.90%의 정확도를 기록하며, White-box 기반의 연합 프롬프트 튜닝 방법들과 대등하거나 오히려 상회하는 성능을 보였다.
- **자원 효율성**:
  - **메모리**: White-box 방법은 클라이언트가 LLM 전체를 로드해야 하므로 막대한 GPU 메모리가 필요하지만, $\text{FedOne}$ (Black-box)은 프롬프트 파라미터만 유지하므로 GPU 메모리 사용량이 거의 없다.
  - **쿼리 및 통신 비용**: $\text{FedOne}$은 $K^*=10$인 일반 연합 학습 대비 LLM 서버 쿼리 횟수와 서버 통신 비용을 획기적으로 감소시켰다.
- **실제 LLM 적용**: GPT-3.5 Turbo를 이용한 실험에서도 $\text{FedOne-GS-BDPL}$이 단순 프롬프트나 훈련되지 않은 프롬프트보다 월등한 성능을 보였으며, 특히 GS-BDPL 방식이 가장 효과적임을 확인하였다.

## 🧠 Insights & Discussion

### 1. 비용-수렴 트레이드오프 (Cost-Convergence Trade-off)

전통적인 연합 학습에서는 더 많은 클라이언트가 참여할수록 수렴이 빨라진다고 믿어왔다. 하지만 본 논문은 **"단위 클라이언트당 비용(API 쿼리 비용 등)이 매우 높은 경우"**에는 이 상식이 깨진다는 점을 시사한다. 수렴 속도의 이득은 서브리니어(sub-linear)한 반면, 비용 증가는 리니어(linear)하기 때문에 비용 효율적인 지점은 $K^*=1$이 된다.

### 2. 범용적 적용 가능성

$\text{FedOne}$의 논리는 단순히 LLM 쿼리 비용에만 국한되지 않는다. 클라이언트의 계산 자원이 극도로 제한적이거나, 통신 비용이 매우 비싼 환경, 혹은 외부 오라클(Oracle)을 호출해야 하는 모든 연합 학습 시나리오에 적용 가능한 일반적인 최적화 전략이 될 수 있다.

### 3. 한계 및 고려사항

클라이언트 수를 1명으로 줄이면 매 라운드의 수렴 속도는 느려질 수밖에 없다. 따라서 총 훈련 시간(Wall-clock time)과 총 비용 사이의 정밀한 균형점을 찾는 추가 연구가 필요할 것이다.

## 📌 TL;DR

본 논문은 클라우드 LLM API를 사용하는 연합 프롬프트 튜닝에서 발생하는 **막대한 쿼리 비용 문제**를 해결하기 위해, 매 라운드 단 한 명의 클라이언트만 활성화하는 **$\text{FedOne}$** 프레임워크를 제안한다. 이론적 분석을 통해 $K^*=1$일 때 쿼리 효율성이 최적임을 증명하였으며, 실제 실험을 통해 White-box 방식에 필적하는 성능을 내면서도 자원 소모와 비용을 획기적으로 줄일 수 있음을 입증하였다. 이는 자원이 제한된 엣지 디바이스 환경에서 거대 모델을 효율적으로 튜닝하는 실무적인 방향성을 제시한다.
