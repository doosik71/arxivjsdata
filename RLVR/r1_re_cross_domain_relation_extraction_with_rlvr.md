# R1-RE: Cross-Domain Relation Extraction with RLVR

Runpeng Dai, Tong Zheng, Run Yang, Kaixian Yu, Hongtu Zhu (2025)

## 🧩 Problem to Solve

본 논문은 자연어 처리의 핵심 과제인 관계 추출(Relation Extraction, RE)에서 발생하는 **도메인 외(Out-of-Domain, OOD) 일반화 성능 저하 문제**를 해결하고자 한다.

기존의 전통적인 RE 방식이나 최근의 지도 미세 조정(Supervised Fine-Tuning, SFT) 방식은 문맥을 레이블로 직접 매핑하는 방식에 의존한다. 이러한 접근법은 학습 데이터와 유사한 도메인 내(In-domain)에서는 높은 성능을 보이지만, 새로운 도메인의 데이터가 입력되었을 때는 성능이 급격히 떨어지는 한계가 있다. 저자들은 SFT 방식이 모델로 하여금 진정한 주석 능력(annotation ability)을 습득하게 하기보다 단순히 학습 데이터를 암기(memorization)하게 만들기 때문에 이러한 일반화 문제가 발생한다고 분석한다.

따라서 본 연구의 목표는 소형 언어 모델(Small LLM)이 인간 주석자와 유사한 추론 과정을 거쳐 관계를 추출하게 함으로써, 도메인에 구애받지 않고 강건한(robust) 성능을 내는 관계 추출 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **관계 추출을 단순한 분류 문제가 아닌, 주석 가이드라인(Annotation Guidelines)에 기반한 추론 과제로 재정의**하는 것이다.

인간 주석자는 단순히 정답을 맞히는 것이 아니라, 가이드라인을 참고하여 가설을 세우고 이를 검증하는 반복적인 추론 과정을 거친다. 이러한 통찰을 바탕으로 저자들은 **검증 가능한 보상 기반의 강화 학습(Reinforcement Learning with Verifiable Reward, RLVR)** 프레임워크인 **R1-RE**를 제안한다. 이를 통해 모델이 명시적인 정답 레이블뿐만 아니라, 정답에 도달하기 위한 논리적인 사고 과정(Chain-of-Thought, CoT)을 스스로 학습하게 하여 OOD 상황에서도 높은 적응력을 갖게 한다.

## 📎 Related Works

### 기존 연구 및 한계

1. **전통적 RE 및 SFT 방식**: BERT, BART와 같은 모델을 사용한 지도 학습 방식은 가벼우나 OOD 시나리오에 대응하기 위해 막대한 양의 미세 조정 데이터가 필요하다.
2. **LLM 기반 RE**: Few-shot 프롬프팅이나 API 기반 모델(GPT-4 등)을 사용하는 방식이 제안되었으나, 소형 오픈소스 모델을 SFT로 학습시켰을 때 여전히 암기 편향 문제가 발생한다.
3. **LLM 추론 학습**: DeepSeek-R1과 같이 RLVR을 통해 수학이나 코딩 분야에서 추론 능력을 끌어올린 연구들이 있었으나, 이를 관계 추출과 같은 지식 추출 작업에 적용한 사례는 부족했다.

### 차별점

R1-RE는 기존의 '문장 $\rightarrow$ 관계'의 직접 매핑 방식에서 벗어나, **'주석 가이드라인 $\rightarrow$ 추론 과정 $\rightarrow$ 관계'**의 경로를 학습시킨다. 특히, 프로세스 기반의 보상 모델(PRM) 대신 규칙 기반의 검증 가능한 보상을 사용하여 보상 해킹(reward hacking) 문제를 방지하고 순수한 추론 능력을 배양했다는 점이 차별화된다.

## 🛠️ Methodology

### 1. 인간 중심의 RE 패러다임 (Human-Inspired RE Paradigm)

모델이 인간 주석자처럼 동작하도록 다음과 같은 구조의 프롬프트를 설계한다.

- **입력 구성**: 대상 문장, 엔티티 태그($\langle e1 \rangle, \langle e2 \rangle$), 그리고 관계 유형에 대한 상세 정의가 담긴 **주석 가이드라인(Annotation guide)**을 함께 제공한다.
- **출력 구조**: 모델은 반드시 $\langle \text{think} \rangle$ 태그 내에 추론 과정을 작성하고, $\langle \text{answer} \rangle$ 태그 내에 최종 관계와 방향을 작성해야 한다.

### 2. R1-RE 프레임워크 및 GRPO 알고리즘

모델의 정책 $\pi_\theta$를 최적화하기 위해 **Group Relative Policy Optimization (GRPO)** 알고리즘을 사용한다.

**학습 절차:**

1. 하나의 프롬프트 $q$에 대해 모델이 $G$개의 후보 출력 $\{o_1, o_2, \dots, o_G\}$을 생성한다.
2. 각 출력에 대해 규칙 기반 보상 $r_i$를 계산한다.
3. 그룹 내의 평균과 표준편차를 이용해 어드밴티지(Advantage) $A_i$를 계산한다:
   $$A_i = \frac{r_i - \text{mean}(r_1, \dots, r_G)}{\text{std}(r_1, \dots, r_G)}$$
4. 다음의 목적 함수를 최대화하여 모델을 업데이트한다:
   $$\mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^G \left( \min \left( \frac{\pi_\theta}{\pi_{\theta_{\text{old}}}} A_i, \text{clip}\left(\frac{\pi_\theta}{\pi_{\theta_{\text{old}}}}, 1-\epsilon, 1+\epsilon\right) A_i \right) - \beta D_{KL}(\pi_\theta || \pi_{\text{ref}}) \right) \right]$$
   여기서 KL 발산($D_{KL}$) 페널티는 업데이트된 모델이 참조 모델($\pi_{\text{ref}}$)에서 너무 멀어지지 않도록 제한한다.

### 3. 다단계 보상 설계 (Multi-stage Reward Design)

보상은 크게 **형식 보상(Format Reward)**과 **지표 보상(Metric Reward)**으로 나뉜다.

- **형식 보상 ($r_{\text{format}}$)**: 응답이 $\langle \text{think} \rangle$와 $\langle \text{answer} \rangle$ 태그를 올바르게 사용했는지, 그리고 정해진 형식(예: `Relation(e1, e2)`)을 지켰는지 확인한다.
  - 정답 형식 준수 시: $+1$
  - 형식 오류 시: $-3$
- **지표 보상 ($r_{\text{metric}}$)**: 형식이 올바른 경우에만 계산하며, 최종 예측값이 정답 레이블과 일치하는지 확인한다.
  - 정답 일치 시: $+2$
  - 정답 불일치 시: $-1.5$
- **최종 보상 ($r$)**: 형식이 틀리면 형식 보상만 부여하고, 형식이 맞으면 두 보상의 합을 부여한다.

## 📊 Results

### 실험 설정

- **데이터셋**: 공개 데이터셋인 **SemEval-2010 Task 8 (Sem2010)**과 비공개 데이터셋인 **MDKG (정신질환 지식 그래프)**를 사용한다.
- **베이스라인**: Qwen2.5-7B-Instruct를 기반 모델로 사용하며, GPT-4o, Claude 3.5 Sonnet 등 상용 모델 및 SFT 모델과 비교한다.
- **평가 지표**: Avg@4 accuracy (4개 샘플의 Pass@1 평균 정확도)를 사용한다.

### 주요 결과

- **OOD 성능의 비약적 향상**: R1-RE-7B 모델은 MDKG 데이터셋에서 SFT 모델 대비 OOD 정확도를 약 15~16%p 향상시켰다.
- **상용 모델 수준의 성능**: 특히 비공개 데이터셋인 MDKG에서 R1-RE-7B는 GPT-4o와 유사한 수준의 OOD 성능을 달성하였다.
- **데이터 누수(Data Leakage) 확인**: Sem2010 데이터셋에서는 상용 모델들이 압도적으로 높은 성능을 보였는데, 이는 상용 모델의 학습 데이터에 해당 벤치마크가 포함되었을 가능성(데이터 누수)을 시사한다. 반면, 비공개 데이터인 MDKG에서는 R1-RE의 효율성이 더 명확하게 드러났다.

## 🧠 Insights & Discussion

### 1. 추론 능력의 창발 (Emergent Reasoning)

학습 과정에서 모델의 응답 길이가 초기 200 토큰에서 최대 1,000 토큰까지 증가하는 현상이 관찰되었다. 이는 모델이 단순히 정답을 맞히기 위해 고민하는 '과잉 사고(overthinking)'가 아니라, **엔티티 식별 $\rightarrow$ 가이드라인 비교 $\rightarrow$ 가설 설정 $\rightarrow$ 검증 $\rightarrow$ 결론 도출**이라는 인간의 주석 프로세스를 스스로 체득했음을 의미한다.

### 2. 일반화 능력 및 치명적 망각 방지

SFT 학습은 특정 작업의 성능은 높이지만 다른 작업의 성능을 떨어뜨리는 경향이 있는 반면, R1-RE(RL 방식)는 MATH-500, IFEval, GPQA와 같은 일반 추론 및 지식 벤치마크에서 오히려 성능이 소폭 향상되는 결과를 보였다. 이는 RL이 단순 암기가 아닌 '추론 기술' 자체를 학습시켜 모델의 전반적인 지적 능력을 강화했음을 시사한다.

### 3. 추가 데이터의 효과

Sem-2018과 같은 추가적인 RE 데이터셋을 학습에 포함시켰을 때 OOD 성능이 추가적으로 약 4%p 향상되었다. 이는 제안된 RLVR 프레임워크가 다양한 도메인의 데이터를 통합하여 일반화 능력을 확장하는 데 매우 유연함을 보여준다.

## 📌 TL;DR

본 논문은 관계 추출(RE)을 단순 분류가 아닌 **주석 가이드라인 기반의 추론 과제**로 정의하고, 이를 위해 **GRPO 알고리즘과 검증 가능한 보상(RLVR)**을 적용한 **R1-RE** 프레임워크를 제안한다. 7B 규모의 소형 모델임에도 불구하고, 강화 학습을 통해 인간과 유사한 다단계 추론 과정(CoT)을 학습함으로써 **도메인 외(OOD) 일반화 성능을 획기적으로 높였으며, 특정 도메인에서는 GPT-4o 수준의 성능**을 보여주었다. 이 연구는 지식 추출 분야에서도 RLVR을 통해 모델의 추론 능력을 극대화할 수 있음을 입증하였다.
