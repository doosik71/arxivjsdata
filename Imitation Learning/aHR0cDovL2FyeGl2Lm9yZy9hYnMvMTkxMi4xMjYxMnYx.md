# Hierarchical Variational Imitation Learning of Control Programs

Roy Fox, Richard Shin, William Paul, Yitian Zou, Dawn Song, Ken Goldberg, Pieter Abbeel, Ion Stoica (2019)

## 🧩 Problem to Solve

본 논문은 자율 에이전트가 교사의 시연(demonstration)을 통해 행동을 학습하는 Imitation Learning(IL)에서 데이터 효율성(data efficiency)과 일반화(generalization) 능력을 향상시키는 문제를 해결하고자 한다.

기존의 평탄한(flat) 제어 정책, 예를 들어 일반적인 LSTM 기반의 정책은 구조화된 과업을 수행할 때 복잡한 상태 공간과 환경 역학으로 인해 매우 많은 양의 데이터를 필요로 한다. 반면, 계층적 제어(hierarchical control)는 복잡한 과업을 더 단순한 하위 과업(sub-tasks)으로 분해함으로써 각 모듈이 단순한 행동에 집중하게 하여 학습 효율을 높일 수 있는 잠재력이 있다.

논문의 핵심 목표는 제어 정책을 프로그램과 유사한 계층적 구조인 Parametrized Hierarchical Procedures(PHP)로 모델링하고, 데이터 내에 숨겨진(latent) 계층적 구조를 자동으로 발견하여 학습할 수 있는 Variational Inference(VI) 기반의 학습 방법을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 다음과 같다.

1. **Hierarchical Variational Imitation Learning (HVIL) 제안**: 제어 정책을 PHP라는 프로그램 구조로 정의하고, 변분 추론(Variational Inference)을 통해 관측-행동 궤적(observation-action traces)으로부터 잠재적인 프로시저 호출 시퀀스를 추론하여 학습하는 프레임워크를 제시하였다.
2. **계층적 추론 모델 아키텍처 설계**: 전체 시연 데이터를 양방향(bidirectional) RNN으로 처리하여 문맥을 생성하고, 이를 통해 현재 단계에서 어떤 프로시저를 호출하거나 종료할지를 결정하는 추론 모델을 설계하였다.
3. **분산 감소 기법 적용**: 이산적(discrete) 확률 변수를 포함하는 stochastic RNN 학습의 불안정성을 해결하기 위해 Analytic KL computation과 Rao-Blackwellization 기법을 도입하였다.
4. **Acausal 정보 활용의 이점 발견**: 추론 모델이 미래의 정보를 활용(acausal information)함으로써, 생성 모델(generative model)이 학습할 수 없는 정보를 이용해 복잡한 정책을 단순한 프로시저로 효율적으로 분해할 수 있음을 이론적/실험적으로 입증하였다.

## 📎 Related Works

본 논문은 다음과 같은 기존 연구들과 차별점을 갖는다.

- **계층적 제어 프레임워크**: Options framework, StackRNN, Neural Programmers–Interpreters(NPI) 등은 명시적 또는 묵시적인 호출 스택(call-stack)을 사용한다. 하지만 본 연구의 PHP는 프로시저 프로그래밍에서 영감을 받아 각 프로시저마다 프로그램 카운터(program counter)를 유지하며, 이를 통해 더 정교한 구조적 제어가 가능하다.
- **PHP (Fox et al., 2018)**: 이전의 PHP 연구는 수준별(level-by-level)로 정확한 추론(exact-inference) 방식을 적용해 학습했으나, 본 논문은 Variational Inference를 도입하여 다층 계층 구조를 동시에(jointly) 학습할 수 있게 하였다.
- **SRNN (Fraccaro et al., 2016)**: 본 연구는 비구조적 연속 잠재 변수를 다루는 SRNN을 계층적 구조의 이산 잠재 변수로 확장하여 제어 정책 학습에 적용하였다.

## 🛠️ Methodology

### 1. Parametrized Hierarchical Procedures (PHP)

PHP는 제어 정책을 하나의 프로그램처럼 모델링한다. 각 프로시저 $h$는 현재 상태 $o_t$와 실행 단계 $\tau$를 입력받아 다음의 세 가지 행동 중 하나를 선택하는 확률 분포 $p_\theta(u | h, \tau, o)$를 갖는다.

- **Sub-procedure call ($u \in \text{PH}$)**: 하위 프로시저를 호출하여 스택(stack)에 푸시(push)한다.
- **Elementary action ($u \in \text{PA}$)**: 환경에서 직접적인 제어 행동 $a_t$를 수행한다.
- **Termination ($u = H$)**: 현재 프로시저를 종료하고 스택에서 팝(pop)하여 호출자로 돌아간다.

### 2. Variational Inference Framework

제어 정책의 구조(어떤 프로시저가 호출되었는지)는 데이터에 명시되지 않은 잠재 변수 $z$이다. 따라서 본 논문은 Evidence Lower Bound (ELBO)를 최대화하는 방식으로 학습한다.

$$ \log p_\theta(x) \geq \mathbb{E}_{z|x \sim q_\phi} [\log p_\theta(z, x)] - D_{KL}(q_\phi(z|x) || p_\theta(z|x)) $$

여기서 $q_\phi(z|x)$는 추론 모델(inference model)이며, 시연 데이터 $x$가 주어졌을 때 잠재 변수 $z$의 사후 분포를 근사한다.

### 3. Inference Model Architecture

추론 모델 $q_\phi$는 다음과 같이 구성된다.

- **Bidirectional RNN**: 전체 궤적 $x$를 입력받아 각 시점 $t$에 대한 사후 문맥(posterior context) $b_t$를 생성한다.
- **Step-wise Distribution**: 현재 스택의 최상단 프로시저 $h_i$와 단계 $\tau_i$ 및 문맥 $b_t$를 사용하여 $q_\phi(u_i | h_i, \tau_i, b_t)$를 계산한다.
- **Masking**: 추론 모델이 시연 데이터에 나타난 실제 행동 $a_t$와 모순되는 선택을 하지 않도록 로짓(logit) 값을 $-\infty$로 마스킹하여 일관성을 유지한다.

### 4. Variance Reduction

이산 잠재 변수 학습 시 발생하는 높은 분산을 줄이기 위해 다음 기법을 사용한다.

- **Analytic KL**: 샘플링 대신 각 PHP 단계에서 KL 발산(KL divergence)을 분석적으로 계산하여 합산함으로써, $q_\phi$가 0으로 수렴할 때 발생하는 그래디언트 편향(bias) 문제를 해결한다.
- **Rao-Blackwellization**: score function gradient estimator를 계산할 때, $f$를 계산하는 데 사용되지 않는 잠재 변수 요소를 제외하여 분산을 감소시킨다.

## 📊 Results

### 1. Bubble Sort 실험

- **설정**: 메모리 배열의 숫자를 정렬하는 태스크. 4층 계층 구조(6개 프로시저)의 PHP와 4층 LSTM을 비교하였다.
- **결과**: HVIL로 학습된 PHP는 LSTM보다 훨씬 적은 데이터로 더 낮은 에러율을 보였다. 특히 24% 에러율에 도달하기까지 필요한 데이터 양이 LSTM의 절반 미만이었다.

### 2. Karel 실험

- **설정**: 교육용 언어 Karel의 프로그램을 모방하는 태스크. 1~3층 깊이의 PHP와 LSTM을 비교하였다.
- **데이터 효율성**: 대부분의 프로그램에서 PHP가 LSTM보다 낮은 에러율을 보였으며, 적은 수의 시연(10~80개)만으로도 효과적으로 학습하였다.
- **일반화 능력**: 학습 때 본 적 없는 더 긴 실행 경로의 테스트 데이터에 대해 평가한 결과, PHP가 LSTM보다 훨씬 뛰어난 일반화 성능을 보였다. 이는 계층적 구조가 강력한 사전 지식(prior)으로 작용했음을 시사한다.

### 3. MNIST Parity 실험 (Acausal Information 검증)

- **설정**: 첫 번째 관측값 $o_0$(이미지)을 보고 짝홀수 $a_0$를 결정한 뒤, 두 번째 관측값 $o_1$(숫자 정답)을 보고 종료하는 태스크.
- **결과**: $o_1$은 $a_0$를 결정하는 데 인과적(causal) 관계가 없으므로 일반 RNN은 이를 학습에 활용하지 못한다. 그러나 HVIL은 추론 모델이 미래의 $o_1$을 보고 $z$를 추론함으로써 생성 모델이 더 빠르게 학습하도록 돕는 "acausal information" 활용 능력을 보여주었다.

## 🧠 Insights & Discussion

### 강점

- **구조적 유도 편향(Inductive Bias)**: 정책을 프로시저 계층으로 모델링함으로써 복잡한 태스크를 단순한 모듈의 조합으로 분해하여 학습 효율을 극대화하였다.
- **추론 모델의 역할**: 단순히 $z$를 예측하는 것을 넘어, 미래 정보를 활용해 생성 모델의 학습 방향을 가이드하는 추론 모델의 효용성을 입증하였다.

### 한계 및 논의사항

- **사전 지식 의존성**: 본 연구에서는 일반적인 5-ary tree 구조를 사용했으나, 실제 도메인 지식을 반영한 호출 그래프(call-graph)를 제공한다면 효율성이 더 높아질 가능성이 있다.
- **최적화 기법의 확장 가능성**: 저자들은 RELAX와 같은 더 정교한 제어 변수(control variates)나 IWAE와 같은 더 타이트한 하한선(bound)을 도입하면 수렴 속도와 강건성을 더 높일 수 있을 것이라고 언급하였다.
- **표현력 확장**: 현재의 PHP는 인자와 반환값이 없으나, 이를 추가하거나 재귀적 상태(recurrent state)를 보강하면 더 복잡한 프로그램 표현이 가능할 것이다.

## 📌 TL;DR

본 논문은 제어 정책을 계층적 프로시저(PHP) 구조로 설계하고, 변분 추론(Variational Inference)을 통해 데이터 속의 잠재적 계층 구조를 발견하는 **HVIL** 프레임워크를 제안한다. HVIL은 추론 모델이 미래의 정보를 활용해 정책 분해를 돕게 함으로써, 기존 LSTM 기반 모델 대비 **데이터 효율성을 2배 이상 높이고**, 학습하지 않은 **긴 시퀀스에 대한 일반화 성능을 크게 향상**시켰다. 이 연구는 구조화된 복잡한 제어 태스크를 효율적으로 학습시키기 위한 프로그래밍적 접근법의 중요성을 시사한다.
