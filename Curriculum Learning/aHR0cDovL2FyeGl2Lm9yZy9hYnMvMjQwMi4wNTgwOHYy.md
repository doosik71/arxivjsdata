# Training Large Language Models for Reasoning through Reverse Curriculum Reinforcement Learning

Zhiheng Xi, Wenxiang Chen, Boyang Hong, Senjie Jin, Rui Zheng, Wei He, Yiwen Ding, Shichun Liu, Xin Guo, Junzhe Wang, Honglin Guo, Wei Shen, Xiaoran Fan, Yuhao Zhou, Shihan Dou, Xiao Wang, Xinbo Zhang, Peng Sun, Tao Gui, Qi Zhang, Xuanjing Huang (2024)

## 🧩 Problem to Solve

복잡한 다단계 추론(multi-step reasoning)을 수행하는 거대 언어 모델(LLM)을 학습시킬 때, 강화 학습(Reinforcement Learning, RL)을 적용하는 과정에서 발생하는 보상 설계의 딜레마를 해결하고자 한다. 

현재 LLM 추론 학습을 위한 보상 체계는 크게 두 가지로 나뉜다. 첫째, 결과 기반 감독(Outcome Supervision, OS)은 최종 정답의 정답 여부만 확인하므로 비용이 적게 들지만, 보상이 매우 희소(sparse)하여 모델이 어느 단계에서 오류를 범했는지 파악하기 어렵고 탐색 공간이 넓어 학습 효율이 떨어진다. 둘째, 과정 기반 감독(Process Supervision, PS)은 각 추론 단계마다 보상을 제공하여 정밀한 가이드가 가능하지만, 모든 단계에 대해 숙련된 전문가의 세밀한 주석(annotation)이 필요하므로 비용이 매우 높고 인간의 편향이 개입될 수 있다.

따라서 본 논문의 목표는 추가적인 세밀한 주석 없이, 결과 기반 감독(OS)만을 활용하면서도 과정 기반 감독(PS)과 유사한 단계별 학습 효과를 얻을 수 있는 새로운 강화 학습 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **역방향 커리큘럼 강화 학습(Reverse Curriculum Reinforcement Learning, $R^3$)**을 도입하는 것이다.

핵심 직관은 모델이 처음부터 전체 추론 경로를 생성하게 하는 대신, 정답이 포함된 기존의 데모(demonstration) 경로의 끝부분(정답에 가까운 상태)부터 시작하여 역순으로 탐색 범위를 넓혀가는 것이다. 즉, 정답에 아주 가까운 상태에서 시작해 성공 경험을 먼저 쌓게 하고, 점차 시작 지점을 앞당겨 더 어려운 문제에 도전하게 함으로써 탐색의 난이도를 점진적으로 높이는 커리큘야(curriculum)를 구성한다. 이를 통해 결과 기반 보상(OS)만으로도 사실상 단계별 보상(PS)을 제공하는 것과 유사한 효과를 낼 수 있다.

## 📎 Related Works

기존의 LLM 추론 학습 연구는 주로 다음과 같은 방향으로 진행되었다.
- **Prompting 및 CoT**: Chain-of-Thought(CoT)와 같은 프롬프팅 기법은 모델의 추론 능력을 끌어올렸으나, 프롬프트 구성에 민감하며 모델 의존성이 높다는 한계가 있다.
- **Supervised Fine-Tuning (SFT)**: 인간이 작성한 정답 경로를 모방 학습하는 방식이지만, 일반화 성능을 높이기 위해서는 대규모의 고품질 주석 데이터가 필수적이다.
- **Outcome-Supervised RL**: 최종 결과에만 보상을 주는 방식이나, 앞서 언급했듯 보상의 희소성 문제로 인해 복잡한 추론 작업에서 수렴 속도가 매우 느리다.
- **Process-Supervised RL**: 각 단계에 보상을 부여하여 효율을 높였으나, 단계별 보상을 산출하기 위한 Reward Model 학습에 막대한 비용과 데이터가 소요된다.

$R^3$는 이러한 기존 방식들과 달리, 추가 데이터 구축 없이 기존의 정답 데모를 활용해 탐색 시작 지점을 조절함으로써 OS의 경제성과 PS의 효율성을 동시에 달성한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조 및 파이프라인
$R^3$의 전체 학습 과정은 정답 경로(demonstration)에서 중간 상태들을 추출하고, 이를 기반으로 역방향 커리큘럼을 구성하여 모델을 점진적으로 학습시키는 구조이다.

1. **중간 상태 추출**: 정답 데모 경로 $\tau = \{s_0, a_1, s_1, a_2, \dots, s_{T-1}, a_T\}$에서 $M$개의 중간 상태 $S_{Inter}$를 샘플링한다.
2. **역방향 커리큘럼 구성**: 가장 정답에 가까운 상태(마지막 단계)부터 시작하여 점차 초기 상태 $s_0$ 방향으로 탐색 시작 지점을 옮긴다.
3. **혼합 단계 학습(Mixed Stages)**: 단순한 단계별 순차 학습은 이전 단계의 지식을 잊어버리는 과적합 문제가 발생할 수 있으므로, 서로 다른 난이도의 시작 상태들을 섞어서 학습하는 Multi-task Learning 방식을 채택한다.

### 주요 방정식 및 최적화 절차
본 연구는 정책 경사(Policy Gradient) 방법론을 기반으로 하며, 특히 PPO(Proximal Policy Optimization) 알고리즘을 사용하여 안정성을 확보한다.

**1. 정책 경사 목적 함수**
역방향 커리큘럼 하에서 모델은 샘플링된 중간 상태 $s_k$부터 시작하여 최종 상태까지의 경로를 생성하며, 다음과 같은 목적 함수를 최적화한다.
$$\mathbb{E}_{s_k \sim S_{Inter}} \left[ \mathbb{E}_{\tau \sim \pi^{RL}_\theta (\cdot | s_k)} \left[ \sum_{t=k+1}^{T} \nabla_\theta \log \pi^{RL}_\theta (a_t | s_{t-1}) R_o (s_{t-1}, a_t) \right] \right]$$
여기서 $R_o$는 결과 기반 보상 함수이며, 모델은 $s_k$ 이후의 액션들에 대해 보상을 받게 된다.

**2. 최종 보상 함수 설계**
학습의 안정성을 위해 초기 모델 $\pi^{Init}_\theta$와의 KL Divergence를 페널티로 추가한 최종 보상을 사용한다.
$$r_{final}(s_{t-1}, a_t) = r_o(s_{t-1}, a_t) - \beta KL \left( \pi^{RL}_\theta (\cdot | s_{t-1}), \pi^{Init}_\theta (\cdot | s_{t-1}) \right)$$
여기서 $\beta$는 KL 페널티의 강도를 조절하는 계수이다.

**3. 부분 보상(Partial Reward) $\epsilon$**
수학적 추론 작업에서 정답이 숫자 형태인 경우, 완전히 정답은 아니더라도 숫자를 출력한 경우에 대해 작은 보상 $\epsilon$을 부여하여 보상의 희소성을 완화한다.

## 📊 Results

### 실험 설정
- **데이터셋**: 수학 추론(GSM8K, SVAMP), 논리 추론(BGQA), 자연어 추론(SNLI, MNLI), 독해(race@High, race@Middle) 등 8개 작업.
- **모델**: Llama2-7B를 기본 백본으로 사용하였으며, 프로그램 기반 추론(P-CoT)을 위해 Glactica, Codellama-7B를 추가로 사용하였다.
- **기준선(Baseline)**: Few-shot CoT, SFT, Vanilla RL.

### 주요 결과
1. **CoT 추론 성능**: $R^3$는 모든 작업에서 SFT 및 Vanilla RL보다 우수한 성능을 보였다. 평균적으로 SFT 대비 5.4포인트, RL 대비 4.1포인트 향상된 결과를 얻었다. 특히 Vanilla Staged RL(단순 단계별 학습)보다 Mixed Stage 방식의 $R^3$가 훨씬 안정적인 성능 향상을 보였다.
2. **프로그램 기반 추론(P-CoT)**: GSM8K 데이터셋에서 Codellama-7B + $R^3$ 조합은 추가 데이터 없이도 MAmmoTH-Coder(7B, 34B)와 같은 데이터 증강 모델이나 GPT-3.5-Turbo와 같은 폐쇄형 모델에 근접하는 성능을 기록하였다.
3. **학습 안정성**: 학습 곡선 분석 결과, Vanilla RL은 보상 값의 변동이 심하고 불안정한 모습을 보인 반면, $R^3$는 매우 안정적으로 보상과 정확도가 상승하는 양상을 보였다.

## 🧠 Insights & Discussion

### 강점 및 분석
- **효율적인 탐색**: 역방향 커리큘럼을 통해 모델이 성공 확률이 높은 지점부터 학습하게 함으로써, 지수적으로 증가하는 탐색 공간의 문제를 선형적인 복잡도로 해결하였다.
- **데이터 효율성**: 실험을 통해 $R^3$가 전체 데이터의 일부만 사용하고도 기존의 SFT나 RL 전체 학습 결과와 유사한 성능을 낼 수 있음을 확인하였다.
- **난이도의 중요성**: 데이터 구성 분석 결과, 단순한 경로보다는 어려운 경로(시작 지점이 정답에서 먼 데이터)를 학습시키는 것이 최종 성능 향상에 더 결정적인 역할을 한다는 것을 발견하였다.

### 한계 및 비판적 해석
- **데모 데이터 의존성**: 본 방법론은 기본적으로 '정답 경로'가 존재하는 데모 데이터가 필요하다. 만약 고품질의 데모 데이터를 확보하지 못한 상황이라면 $R^3$의 효과는 제한적일 수 있다.
- **하이퍼파라미터 민감도**: KL 계수 $\beta$나 부분 보상 $\epsilon$의 설정 값에 따라 성능 변화가 관찰되었다. 특히 $\beta$가 너무 크면 최적화가 저해되는 경향이 있었다.
- **보상 함수 설계**: 난이도에 따라 보상 값을 다르게 주는 방식(Linear, Square 등)을 시도했으나, 오히려 성능이 하락했다. 이는 모든 시작 상태에 대해 동일한 정답 보상을 주는 것이 가장 공정하고 효과적임을 시사한다.

## 📌 TL;DR

본 논문은 결과 기반 보상(Outcome Supervision)의 희소성 문제를 해결하기 위해, 정답 데모의 끝에서부터 시작 지점을 역순으로 넓혀가는 **Reverse Curriculum RL ($R^3$)** 방법론을 제안한다. 이를 통해 추가적인 단계별 주석 없이도 과정 기반 감독(Process Supervision)과 유사한 학습 효과를 거두었으며, Llama2-7B 및 Codellama-7B 모델에서 수학 및 논리 추론 성능을 크게 향상시켰다. 이 연구는 고비용의 데이터 구축 없이도 LLM의 복잡한 추론 능력을 효율적으로 강화할 수 있는 실질적인 방법론을 제시했다는 점에서 향후 추론 모델 학습에 중요한 역할을 할 것으로 보인다.