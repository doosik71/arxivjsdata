# ViSurf: Visual Supervised-and-Reinforcement Fine-Tuning for Large Vision-and-Language Models

Yuqi Liu, Liangyu Chen, Jiazhen Liu, Mingkang Zhu, Zhisheng Zhong, Bei Yu, Jiaya Jia (2025)

## 🧩 Problem to Solve

Large Vision-and-Language Models(LVLMs)의 사후 학습(post-training)을 위한 기존 패러다임은 크게 Supervised Fine-Tuning(SFT)과 Verifiable Rewards를 이용한 Reinforcement Learning(RLVR)으로 나뉜다. SFT는 전문가가 작성한 데이터를 통해 외부 가이드라인을 주입함으로써 새로운 지식을 학습시키기에 유리하지만, 성능이 최적이 아니거나 사전 학습된 지식을 잃어버리는 Catastrophic Forgetting 현상이 발생할 수 있다. 반면, RLVR은 모델 내부의 피드백을 통해 추론 능력을 강화하고 성능을 높이는 데 효과적이지만, 해결하려는 과제가 모델이 이미 보유한 내부 지식 베이스(internal knowledge base)를 벗어나는 경우 성능이 급격히 저하되는 한계가 있다.

특히, 본 논문은 '객체가 없는 시나리오(Non-Object Scenarios)'에서의 분석을 통해 RLVR 모델이 정답이 '없음'이어야 하는 상황에서도 강제로 마스크를 생성하는 경향이 있음을 발견하였다. 이는 RLVR이 자체 롤아웃(self-rollouts)에만 의존하여 정답을 수정할 수 있는 외부 교정 메커니즘이 부족하기 때문이다. 기존의 2단계 학습 방식인 $\text{SFT} \rightarrow \text{RLVR}$ 파이프라인은 두 방식의 장점을 모두 취하려 하지만, 계산 비용이 두 배로 증가하며 SFT 단계에서 발생하는 Catastrophic Forgetting 문제에 여전히 취약하다. 따라서 본 연구의 목표는 SFT의 외부 지도 학습과 RLVR의 내부 강화 학습의 장점을 단일 단계(single-stage)로 통합하여 효율적으로 성능을 극대화하는 새로운 학습 패러다임을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 SFT와 RLVR을 단일 단계로 통합한 **ViSurf(Visual Supervised-and-Reinforcement Fine-Tuning)** 프레임워크를 제안한 것이다.

- **통합 목적 함수 설계**: SFT와 RLVR의 그래디언트(gradient)가 유사한 형태를 띤다는 이론적 분석을 바탕으로, Ground-truth(GT) 레이블을 RLVR의 롤아웃 세트에 고보상 샘플로 포함시키는 통합 목적 함수를 설계하였다.
- **Reward Control 전략**: 학습 안정성을 위해 GT 레이블에 대해 (1) 롤아웃 선호도에 맞춘 정렬(Aligning), (2) 사고 과정 보상 제거(Eliminating Thinking Reward), (3) 보상 평활화(Smoothing)라는 세 가지 제어 전략을 도입하였다.
- **효과 검증**: 다양한 벤치마크에서 ViSurf가 SFT, RLVR 및 $\text{SFT} \rightarrow \text{RLVR}$ 방식보다 우수한 성능을 보임을 입증하였으며, 특히 모델의 지식 범위를 벗어나는 과제에서 강력한 성능 향상을 보였다.

## 📎 Related Works

### 1. Supervised Fine-tuning for LVLMs

LLaVA를 필두로 LLaVA-series, QwenVL-series, InternVL 등 많은 모델이 SFT를 채택하고 있다. SFT는 모델을 다양한 다운스트림 애플리케이션에 적응시키는 데 효과적이지만, 앞서 언급한 것처럼 지식 망각 문제와 성능의 하한선 문제가 존재한다.

### 2. Reinforcement Learning for LVLMs

DPO나 PPO 같은 방식은 인간의 선호도 데이터나 별도의 보상 모델이 필요하다는 비용적 한계가 있다. 최근에는 GRPO나 DAPO와 같이 객관적인 기준(Verifiable Rewards)으로 모델 출력을 평가하는 RLVR 알고리즘이 주목받고 있으며, SegZero나 VisualRFT 등이 이를 LVLM에 적용하였다.

본 논문의 ViSurf는 기존의 단순한 목적 함수 합산 방식과 달리, 이론적 분석을 통해 두 패러다임을 하나의 통합된 관점에서 접근하여 단일 단계에서 상호 보완적으로 작동하게 한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 전체 파이프라인 및 시스템 구조

ViSurf는 단일 학습 단계 내에서 Ground-truth(GT) 데모네스트레이션과 모델이 직접 생성한 on-policy 롤아웃을 교차로 배치하여 학습한다. 기본적으로 GRPO(Group Relative Policy Optimization) 알고리즘의 구조를 따르되, 보상 계산 및 목적 함수 부분에 GT 데이터를 통합하였다.

### 상세 방법론 및 방정식

#### 1. SFT와 RLVR의 목적 함수 분석

SFT의 목적 함수는 다음과 같이 레이블 $y$의 음의 로그 가능도(NLL)를 최소화하는 것이다.
$$\mathcal{L}_{\text{SFT}}(\theta) = -\mathbb{E}_{(v,t)\sim\mathcal{D}_{\text{input}}, y\sim\mathcal{D}_{\text{label}}} [\log \pi_\theta(y | v,t)]$$

RLVR(GRPO 기준)은 모델이 생성한 $G$개의 롤아웃 $\{o_j\}_{j=1}^G$에 대해 상대적 이득(Advantage) $\hat{A}_j$를 계산하여 최적화한다.
$$\hat{A}_j = \frac{r(o_j) - \text{mean}(\{r(o_j)\}_{j=1}^G)}{\text{std}(\{r(o_j)\}_{j=1}^G)}$$

#### 2. ViSurf 목적 함수

ViSurf의 핵심 아이디어는 GT 레이블 $y$를 RLVR 롤아웃 세트에 포함시켜 증강된 세트 $y \cup \{o_j\}_{j=1}^G$를 구성하는 것이다. 이에 따라 이득 계산 식이 다음과 같이 변경된다.

- **롤아웃 $o_j$의 이득**:
$$\hat{A}_j = \frac{r(o_j) - \text{mean}(r(y) \cup \{r(o_j)\}_{j=1}^G)}{\text{std}(r(y) \cup \{r(o_j)\}_{j=1}^G)}$$
- **GT 레이블 $y$의 이득**:
$$\hat{A}_y = \frac{r(y) - \text{mean}(r(y) \cup \{r(o_j)\}_{j=1}^G)}{\text{std}(r(y) \cup \{r(o_j)\}_{j=1}^G)}$$

최종 목적 함수 $\mathcal{L}_{\text{ViSurf}}(\theta)$는 롤아웃들과 GT 레이블에 대한 clipped objective의 합으로 정의된다.
$$\mathcal{L}_{\text{ViSurf}}(\theta) = -\mathbb{E} \left[ \frac{1}{G+1} \left( \sum_{j=1}^G \min(\dots \hat{A}_j) + \min(\dots \hat{A}_y) \right) \right]$$

이 구조를 통해 ViSurf의 그래디언트는 $\text{RLVR Term}$과 $\text{SFT Term}$의 합으로 나타나며, 모델이 정답을 생성하지 못할 때는 SFT 항이 지배적으로 작용하여 외부 가이드를 제공하고, 모델이 충분히 잘 생성할 때는 RLVR 항이 지배적으로 작용하여 내부 성능을 강화한다.

### Reward Control 전략

GT 레이블 $y$에 의한 보상 해킹(reward hacking) 및 엔트로피 붕괴를 막기 위해 세 가지 전략을 사용한다.

1. **Aligning**: GT 데이터의 JSON 형식을 모델이 생성하는 롤아웃의 스타일(예: 쉼표 뒤 공백 추가)과 일치시켜 토큰화 차이로 인한 분포 변화를 최소화한다.
2. **Eliminating Thinking Reward**: GT 데이터에는 추론 과정(thinking trace)이 없으므로, 추론 형식에 대한 보상을 0으로 설정하여 모델이 오직 self-rollouts를 통해서만 추론 과정을 배우게 한다.
3. **Smoothing**: 생성된 롤아웃 중 최대 보상이 GT 보상보다 크거나 같으면($\max\{r(o_j)\} \ge r(y)$), $r(y)$를 롤아웃들의 평균 보상으로 설정한다. 이는 모델이 이미 정답을 낼 수 있을 때 불필요한 외부 감독 신호를 제거하여 $\hat{A}_y \approx 0$이 되게 함으로써 학습을 안정화한다.

## 📊 Results

### 실험 설정

- **데이터셋**: gRefCOCO(비객체 세그멘테이션), ReasonSeg(추론 세그멘테이션), OmniACT(GUI Grounding), RealIAD(이상 탐지), ISIC 2018(피부 병변 세그멘테이션), MathVista(수학 추론) 등.
- **기준 모델**: Qwen2.5VL-7B 및 SAM2 사용.
- **평가 지표**: gIoU, N-Acc(비객체 탐지 정확도), ROC AUC, BboxAcc 등.

### 주요 결과

1. **성능 우위**: Table 1에 따르면, ViSurf는 모든 도메인에서 SFT, RLVR, $\text{SFT} \rightarrow \text{RLVR}$보다 높은 성능을 기록하였다. 특히 모델의 기본 성능이 낮은 Non-Object 및 Anomaly 탐지 과제에서 향상 폭이 매우 컸다.
2. **망각 방지**: ChartQA 및 DocVQA 실험(Table 2)에서 SFT 및 $\text{SFT} \rightarrow \text{RLVR}$은 성능 저하(Catastrophic Forgetting)가 나타났으나, ViSurf와 RLVR은 베이스라인 성능을 유지하거나 향상시켰다.
3. **SOTA 비교**: gRefCOCO와 ReasonSeg 과제에서 LISA, GSVA, SegZero 등 기존 최신 모델(SoTA)들을 제치고 가장 높은 성능을 달성하였다.
4. **학습 비용**: Table 6에 따르면, ViSurf는 RLVR과 유사한 메모리 효율성을 보였으며, 2단계 학습 방식인 $\text{SFT} \rightarrow \text{RLVR}$보다 전체 학습 시간을 크게 단축하였다.

## 🧠 Insights & Discussion

### 강점 및 분석

ViSurf의 가장 큰 강점은 **자기 적응적(self-adaptive) 학습 메커니즘**에 있다. 학습 초기나 모델이 정답을 내지 못하는 상황에서는 GT 레이블의 이득($\hat{A}_y$)이 높아져 SFT-like 학습이 주도하며 모델을 가이드한다. 이후 모델의 생성 능력이 향상되면 Reward Smoothing 전략에 의해 SFT 항의 영향력이 줄어들고 RLVR-like 학습이 주도하여 추론 능력을 극대화한다. 이는 엔트로피 붕괴를 방지하고 학습의 안정성을 높이는 결과로 이어진다.

### 한계 및 논의

본 연구에서는 GT 레이블을 최종 정답으로만 제한하여 사용하였다. 하지만 저자들은 제안된 프레임워크가 명시적인 추론 경로(reasoning traces)나 대형 모델로부터의 지식 증류(knowledge distillation) 결과물을 통합하는 것과도 호환될 수 있음을 시사하였다. 또한, 베이스라인 모델의 초기 성능이 이미 매우 높은 경우($>50\%$), ViSurf의 상한선이 RLVR과 일치하게 된다는 점은 본 방법론이 특히 '지식 부족' 상태의 모델을 끌어올리는 데 최적화되어 있음을 의미한다.

## 📌 TL;DR

본 논문은 LVLM의 사후 학습에서 SFT의 외부 가이드와 RLVR의 내부 강화를 단일 단계로 통합한 **ViSurf**를 제안하였다. 이론적 분석을 통해 GT 레이블을 RLVR의 고보상 샘플로 처리하는 통합 목적 함수를 도출하였으며, 세 가지 보상 제어 전략을 통해 학습 안정성을 확보하였다. 실험 결과, ViSurf는 기존의 2단계 학습법보다 효율적이면서도 성능이 뛰어나며, 특히 모델의 지식 범위를 벗어나는 난이도 높은 시각적 인지 및 추론 과제에서 탁월한 성능과 안정성을 입증하였다. 이는 향후 LVLM의 효율적인 post-training을 위한 새로운 표준 패러다임이 될 가능성이 높다.
