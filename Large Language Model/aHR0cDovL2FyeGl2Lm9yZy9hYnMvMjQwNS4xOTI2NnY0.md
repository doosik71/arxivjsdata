# PediatricsGPT: Large Language Models as Chinese Medical Assistants for Pediatric Applications

Dingkang Yang, Jinjie Wei, Dongling Xiao, et al. (2024)

## 🧩 Problem to Solve

본 연구는 특히 중국과 같이 의료 자원이 부족한 환경에서 진단 효율성을 높이기 위한 지능형 소아과 상담 시스템 구축을 목표로 한다. 최근 중국어 의료분야를 위한 거대 언어 모델(LLM)들이 발전하고 있으나, 소아과 응용 분야에서는 여전히 성능이 최적화되지 않은 상태이다. 이러한 문제의 주요 원인은 소아과 특화 지침 데이터(Instruction Data)의 부족과 취약한 학습 절차에 있다. 기존의 의료 LLM들은 일반적인 의료 말뭉치를 단순하게 재구성하거나 의사-환자 대화를 수집하는 수준에 그쳐, 소아과 특유의 전문성과 세밀한 진단 요구사항을 충족하지 못하며, 이는 모델의 일반화 능력을 제한하고 환각(Hallucination) 현상을 유발하는 결과로 이어진다. 따라서 본 논문은 고품질의 소아과 특화 데이터셋인 PedCorpus를 구축하고, 이를 활용한 체계적이고 강건한 학습 파이프라인을 갖춘 소아과 전문 LLM인 PediatricsGPT를 제안한다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 소아과 전문 지식과 일반 의료 지식을 동시에 갖춘 최초의 중국어 소아과 LLM인 PediatricsGPT를 개발한 것이다. 이를 위해 다음과 같은 설계 아이디어를 도입하였다.

첫째, 소아과 교과서, 가이드라인, 지식 그래프(Knowledge Graph) 및 실제 의사-환자 대화를 통합하여 30만 개 이상의 다중 작업 지침을 포함하는 고품질 데이터셋 **PedCorpus**를 구축하였다.

둘째, 모델 학습 과정에서 내부 지식과 외부 주입 지식 간의 불일치를 완화하기 위해 **Hybrid Instruction Pre-training** 메커니즘을 도입하여 Continuous Pre-training(CPT) 단계를 수행하였다.

셋째, 인간의 선호도에 부합하는 인간 중심적인 응답 생성을 위해 **Direct Following Preference Optimization (DFPO)** 기법을 설계하여 응답의 강건성을 높였다.

넷째, 일반 의료 지식의 숙련도와 소아과 전문 지식 사이의 역량 충돌을 해결하기 위해 LoRA(Low-Rank Adaptation) 기반의 **Mixture of Universal-specific Experts (MUE)** 전략을 제안하여 모델의 적응력을 강화하였다.

## 📎 Related Works

최근 Baichuan, GLM, Qwen과 같은 중국어 LLM들이 발전하며 중국어 지시어 수행 능력이 향상되었으며, ChatDoctor, DoctorGLM, HuatuoGPT 등 의료 특화 LLM들이 등장하였다. 이러한 기존 연구들은 주로 Supervised Fine-Tuning(SFT)이나 의료 관련 말뭉치 수집을 통해 의료 능력을 강화하려 하였다.

그러나 기존 접근 방식은 다음과 같은 한계가 있다. 우선, 대부분의 의료 데이터셋이 일반적인 의료 지식에 치중되어 있어 소아과와 같은 특정 세부 분과(Specialty)에 대한 전문성이 부족하다. 또한, 단순한 SFT 방식은 모델이 지식을 진정으로 이해하기보다 단순한 역할 수행(Role-playing)에 그치게 하는 경향이 있으며, RLHF(Reinforcement Learning from Human Feedback) 기반의 방법론은 Actor-Critic 구조의 불안정성과 온라인 샘플링 편향 문제로 인해 성능 향상에 제약이 있다. PediatricsGPT는 이러한 한계를 극복하기 위해 전문 데이터셋 구축과 단계별 최적화 파이프라인을 통해 차별화를 꾀하였다.

## 🛠️ Methodology

### 1. PedCorpus 데이터셋 구축
PedCorpus는 세 가지 주요 의료 작업인 지식 질의응답(MedKQ&A), 근거 기반 진단(EviDiag), 치료 추천(TreRecom)을 수행할 수 있도록 설계되었다.
- **소아과 전문 데이터**: 교과서, 가이드라인, 지식 그래프에서 데이터를 추출하고, GPT-4를 활용하여 '질문자'와 '전문 소아과 의사' 역할을 부여한 롤플레잉 기반의 지침을 생성하였다.
- **실제 의사-환자 대화**: 실제 상담 데이터를 수집하고, GPT-4의 In-context learning을 통해 정제하여 의사답고 환자 친화적인 응답 스타일을 학습시켰다.
- **증류된 의료 데이터셋**: 기존의 공개 벤치마크에서 고품질의 데이터를 샘플링하고, 점진적 지침 재구성 규칙(Progressive Instruction Reconstruction Rule)을 통해 논리적이고 정보량이 많은 응답으로 정제하였다.

### 2. 학습 파이프라인
PediatricsGPT는 다음의 네 단계 과정을 통해 학습된다.

**가. Continuous Pre-training (CPT)**
기초 모델에 대규모 의료 지식을 주입하는 단계이다. 이때 **Hybrid Instruction Pre-training** 메커니즘을 사용하여, 입력-출력 형태의 지침 데이터를 텍스트 완성(Completion) 형태로 변환하여 일반 텍스트와 함께 학습시킨다. 이는 기초 모델의 기존 지식과 새롭게 주입되는 지식 간의 형식 차이로 인한 성능 저하(Catastrophic Forgetting)를 방지한다. 학습 목표는 다음 토큰 예측 확률을 최대화하는 것이며, 손실 함수는 다음과 같다.
$$L_{CPT}(\theta, D_{cpt}) = \mathbb{E}_{t \sim D_{cpt}} \left[ - \sum_{i=1}^{|t|} \log p(t_i | t_0, t_1, \dots, t_{i-1}; \theta) \right]$$

**나. Full-parameter Supervised Fine-tuning (FSFT)**
모델의 의료 지시어 수행 능력을 활성화하기 위해 전체 파라미터를 튜닝한다. 일반 의료 데이터, 일반 도메인 데이터(Alpaca, ShareGPT), 그리고 안전성 및 자기 인식(Self-cognition)을 위한 특수 데이터를 함께 사용한다. 손실 함수는 다음과 같다.
$$L_{FSFT}(\theta, D_{fsft}) = \mathbb{E}_{(x,y) \sim D_{fsft}} \left[ - \sum_{i=1}^{|y|} \log p(y_i | x, y_{<i}; \theta) \right]$$

**다. Direct Following Preference Optimization (DFPO)**
인간의 선호도를 반영하여 유해성을 줄이고 인간 중심적인 응답을 생성하게 하는 단계이다. 선호 응답 $y^w$와 비선호 응답 $y^l$의 쌍을 사용하여 학습하며, 최적화 목표는 다음과 같다.
$$L_{DFPO}(\theta, D_{dfpo}) = - \mathbb{E}_{(x, y^w, y^l) \sim D_{dfpo}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y^w|x)}{\pi_r(y^w|x)} - \beta \log \frac{\pi_\theta(y^l|x)}{\pi_r(y^l|x)} \right) \right] + \mu \Phi(x, y^w)$$
여기서 $\pi_\theta$는 최적화할 정책, $\pi_r$은 참조 정책이며, $\Phi(x, y^w)$는 선호 응답에 대한 로그 확률을 최대화하는 정규화 항이다.

**라. Parameter-efficient SFT 및 Mixture of Universal-specific Experts (MUE)**
마지막으로 LoRA를 활용하여 파라미터 효율적인 SFT를 수행한다. 이때 일반 지식과 소아과 전문 지식 간의 충돌을 막기 위해 **MUE** 전략을 사용한다.
- **구조**: 여러 개의 특정 전문가(Specific Experts, $E_{s_j}$)와 하나의 보편적 전문가(Universal Expert, $E_u$)를 배치한다.
- **라우팅 게이팅**: 입력 $x$에 대해 어떤 전문가를 활성화할지 결정하는 $G(x)$ 함수를 사용한다.
$$G(x) = \text{Softmax}(xW_g + S(\phi(xW_n)))$$
- **최종 출력**: 보편적 전문가의 출력과 선택된 특정 전문가들의 가중 합을 결합하여 최종 결과 $z$를 생성한다.
$$z = \alpha_r \left( \sum_{j=1}^T G(x)_j E_{s_j}(x) + E_u(x) \right)$$

## 📊 Results

### 1. 실험 설정
- **모델**: Baichuan2-Base (7B, 13B)를 기반으로 개발하였다.
- **평가 벤치마크**: 소아과 작업(MedKQ&A, EviDiag, TreRecom)과 일반 의료 작업(CMD, webMedQA)을 모두 사용하였다.
- **측정 지표**: ROUGE, BLEU, GLEU, Distinct 등의 정량적 지표와 함께 GPT-4 및 전문 의사의 정성적 평가(Win-rate)를 수행하였다.

### 2. 주요 결과
- **정량적 성능**: PediatricsGPT-13B는 거의 모든 지표에서 기존 중국어 의료 LLM 및 베이스라인 모델들을 압도하였다. 특히 소아과 특화 작업에서 매우 높은 성능을 보였으며, GPT-3.5-turbo와 경쟁 가능한 수준에 도달하였다.
- **GPT-4 및 의사 평가**:
    - **MedKQ&A**: 지식 집약적인 CPT의 효과로 인해 타 모델 대비 매우 높은 승률을 기록하였다.
    - **EviDiag & TreRecom**: 다회차 상담과 치료 추천 작업에서도 Zhongjing과 같은 SOTA 모델들을 상회하는 승률을 보였다.
    - **전문가 평가**: 실제 의사들은 PediatricsGPT의 응답이 전문성, 사실성, 안전성 면에서 더 우수하다고 판단하였다.
- **일반화 능력**: CMD와 webMedQA 벤치마크 결과, 소아과뿐만 아니라 산부인과, 내과 등 다양한 진료과에서도 우수한 성능을 보여 '의료 제너럴리스트'로서의 역량과 '소아과 전문가'로서의 역량을 동시에 확보했음을 입증하였다.

## 🧠 Insights & Discussion

### 1. 강점 및 분석
- **데이터 품질의 중요성**: 방대한 양의 데이터보다 정교하게 설계된 고품질의 지침 데이터(Instruction Data)가 모델 성능 향상에 더 결정적인 영향을 미친다는 것을 확인하였다.
- **MUE 전략의 유효성**: 분석 결과, 특정 전문가(Specific Experts)들이 각각 다른 작업(예: Expert 1은 진단, Expert 2는 치료 추천)에 특화되어 활성화됨을 확인하였다. 또한 보편적 전문가(Universal Expert)가 일반 의료 지식을 유지함으로써 전문 지식 학습 시 발생하는 일반 능력 저하 문제를 해결하였다.
- **DFPO의 효과**: 단순한 RLHF보다 가벼우면서도 안정적인 DFPO를 통해 의사다운 정중하고 인간 중심적인 응답 스타일을 효과적으로 학습시켰다.

### 2. 한계 및 향후 과제
- **보안 리스크**: 온라인 배포 시 모델의 출력을 조작하려는 적대적 공격에 취약할 수 있으며, 이를 위한 다층 보안 체계 구축이 필요하다.
- **언어 확장성**: 현재는 중국어 기반으로 구축되어 있어, 글로벌 적용을 위해서는 다국어 지원 학습이 필요하다.

## 📌 TL;DR

본 연구는 소아과 전문 지식과 일반 의료 능력을 동시에 갖춘 중국어 LLM인 **PediatricsGPT**를 제안하였다. 고품질 소아과 데이터셋인 **PedCorpus**를 구축하고, **Hybrid CPT $\rightarrow$ Full SFT $\rightarrow$ DFPO $\rightarrow$ MUE-based PSFT**로 이어지는 체계적인 학습 파이프라인을 통해 성능을 극대화하였다. 실험 결과, 본 모델은 기존의 중국어 의료 LLM들을 상회하며 전문 의사들로부터 높은 평가를 받았다. 이 연구는 특정 의료 분과에 특화된 전문 LLM을 구축하는 효과적인 방법론을 제시하였으며, 향후 의료 AI의 전문성 강화 및 실제 진단 보조 시스템 적용에 중요한 역할을 할 것으로 기대된다.