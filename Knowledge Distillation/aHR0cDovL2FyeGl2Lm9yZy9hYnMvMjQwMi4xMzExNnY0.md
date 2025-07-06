# 대규모 언어 모델의 지식 증류에 대한 조사 연구
Xiaohan Xu, Ming Li, Chongyang Tao, Tao Shen, Reynold Cheng, Jinyang Li, Can Xu, Dacheng Tao, Tianyi Zhou

## 🧩 해결할 문제
대규모 언어 모델(Large Language Models, LLMs) 시대에 GPT-4와 같은 선도적인 독점 LLM들은 뛰어난 성능을 보이지만, 높은 비용과 제한된 접근성이라는 단점을 가지고 있습니다. 반면 LLaMA, Mistral과 같은 오픈소스 LLM들은 접근성이 좋고 커스터마이징이 용이하지만, 독점 LLM에 비해 규모와 성능 면에서 뒤처집니다. 이 논문은 이러한 독점 LLM의 고급 역량을 오픈소스 LLM에 효율적으로 전이시키는 방법, 즉 지식 증류(Knowledge Distillation, KD)의 중요성을 강조합니다. 주요 문제점은 다음과 같습니다.
*   **성능 격차 해소**: 강력한 독점 LLM의 지식과 역량을 접근성 좋은 오픈소스 LLM에 어떻게 효과적으로 전달할 것인가?
*   **모델 압축 및 효율성**: 거대한 LLM을 어떻게 압축하여 자원 제약적인 환경에서도 효율적으로 배포할 수 있도록 할 것인가?
*   **자체 개선**: 오픈소스 LLM이 스스로를 교사 모델로 활용하여 역량을 향상시키는 메커니즘은 무엇인가?

## ✨ 주요 기여
*   **포괄적인 KD 조사**: LLM 영역에서 지식 증류의 역할을 상세히 분석한 종합적인 조사를 제공합니다.
*   **세 가지 핵심 축**: 알고리즘, 스킬, 수직화라는 세 가지 핵심 축을 중심으로 KD 메커니즘, 특정 인지 능력 향상, 다양한 분야에서의 실질적인 적용을 체계적으로 탐구합니다.
*   **데이터 증강(DA)과 KD의 상호작용 강조**: DA가 KD 프레임워크 내에서 LLM의 성능을 강화하는 강력한 패러다임으로 작용함을 보여줍니다. DA를 활용하여 문맥이 풍부하고 스킬에 특화된 훈련 데이터를 생성함으로써, 오픈소스 모델이 독점 모델의 문맥 적응성, 윤리적 정렬, 깊은 의미론적 통찰력을 근사하도록 만듭니다.
*   **연구 지침 및 미래 방향 제시**: 연구자와 실무자에게 통찰력 있는 가이드를 제공하며, 현재의 KD 방법론에 대한 상세한 개요를 제공하고 미래 연구 방향을 제안합니다.
*   **접근성 및 효율성 증대**: 독점 LLM과 오픈소스 LLM 간의 격차를 줄여 보다 접근 가능하고 효율적이며 강력한 AI 솔루션의 잠재력을 강조합니다.
*   **윤리적 및 법적 준수 강조**: LLM 사용을 규제하는 법적 조건 준수를 강력히 옹호하며, LLM KD의 윤리적이고 합법적인 적용을 보장합니다.

## 📎 관련 연구
*   **전통적인 지식 증류**: Gou et al. (2021), Gupta and Agrawal (2022) 등. 복잡한 모델에서 작고 효율적인 모델로 지식을 전이하는 데 중점.
*   **LLM의 emergent abilities**: Wei et al. (2022a,b), Xu et al. (2024a), OpenAI et al. (2023), Liang et al. (2022). LLM이 명시된 훈련 목표를 넘어선 능력을 발휘하는 현상.
*   **독점 LLM의 한계**: OpenAI et al. (2023), Wu et al. (2023a). 접근성, 비용, 데이터 프라이버시 문제.
*   **오픈소스 LLM의 장단점**: Touvron et al. (2023) (LLaMA), Jiang et al. (2023a) (Mistral) 등.
*   **LLM 지식 증류의 주요 흐름**:
    *   **데이터 증강(DA)과의 연관성**: Feng et al. (2021), Wang et al. (2022a), Ye et al. (2022).
    *   **모델 자체 개선**: Yuan et al. (2024a), Chen et al. (2024a).
*   **주요 증류 기법 및 모델**:
    *   **명령어 추종**: Self-Instruct (Wang et al., 2022a), Alpaca (Taori et al., 2023), WizardLM (Xu et al., 2023a), Orca (Mukherjee et al., 2023).
    *   **정렬(Alignment)**: RLAIF (Bai et al., 2022a), UltraFeedback (Cui et al., 2023a), Zephyr (Tunstall et al., 2023).
    *   **특징 기반 증류**: MiniLLM (Gu et al., 2024), BabyLlama (Timiryasov and Tastet, 2023).
    *   **수직 도메인 적용**: LawyerLLaMA (Huang et al., 2023b), HuatuoGPT (Zhang et al., 2023c), XuanYuan (Zhang and Yang, 2023).

## 🛠️ 방법론
본 조사는 LLM 지식 증류 파이프라인을 네 가지 주요 단계와 두 가지 추상화된 정식화로 설명하고, 지식 추출 및 증류 알고리즘을 탐구합니다.

### 1. 지식 증류 파이프라인 (The General Distillation Pipeline)
1.  **교사 LLM 조작 (Target Skill or Domain Steering Teacher LLM)**: 교사 LLM을 특정 스킬 또는 도메인(예: 헬스케어, 추론)으로 유도하기 위해 신중하게 구성된 명령어 또는 템플릿을 사용합니다.
2.  **시드 지식 입력 (Seed Knowledge as Input)**: 교사 LLM에 시드 지식(작은 데이터셋 또는 특정 데이터 단서)을 제공하여 더 정교하고 상세한 출력을 생성하도록 유도합니다.
3.  **증류 지식 생성 (Generation of Distillation Knowledge)**: 교사 LLM은 시드 지식과 유도 명령어에 따라 질문-답변(QA) 대화, 서술적 설명, 또는 로그잇, 히든 피처 형태의 지식 예시를 생성합니다.
4.  **학습 목표를 통한 학생 모델 훈련 (Training the Student Model with a Specific Learning Objective)**: 생성된 지식 예시를 사용하여 손실 함수를 통해 학생 모델을 훈련하여 교사 모델의 지식을 모방하고 유사한 능력을 습득하게 합니다.

이 네 단계는 두 가지 주요 정식화로 요약됩니다.
*   **지식 추출**: $$D^{(kd)}_I = \{ \text{Parse}(o,s) | o \sim p_T(o|I \oplus s), \forall s \sim S \}$$
    *   $I$: 특정 작업, 스킬 또는 도메인을 유도하고 지식을 추출하는 명령어.
    *   $s \sim S$: LLM이 새로운 지식을 탐색하고 생성할 수 있는 시드 지식 예시.
    *   $\text{Parse}(o,s)$: 교사 LLM의 출력 $o$에서 증류 예시($x,y$)를 파싱하는 과정.
    *   $p_T$: 매개변수 $\theta_T$를 가진 교사 LLM.
*   **학습 목표 정의**: $$\mathcal{L} = \sum_I \mathcal{L}_I(D^{(kd)}_I; \theta_S)$$
    *   $\sum_I$: 여러 작업 또는 스킬이 하나의 학생 모델로 증류될 수 있음을 나타냅니다.
    *   $\mathcal{L}_I(\cdot;\cdot)$: 특정 학습 목표.
    *   $\theta_S$: 학생 모델의 매개변수.

### 2. 지식 추출 (Knowledge Elicitation)
교사 LLM으로부터 지식을 추출하는 다양한 방법은 다음과 같습니다.
*   **레이블링 (Labeling)**: 교사 LLM이 주어진 입력 $x$에 대해 출력 $y$를 레이블링합니다. (예: Orca, CoT-Distill).
    *   $$D^{(lab)} = \{ (x,y) | x \sim X, y \sim p_T(y|I \oplus c \oplus x) \}$$
*   **확장 (Expansion)**: 교사 LLM이 시드 데모 $c$를 기반으로 새로운 입력 $x$와 출력 $y$를 생성하여 데이터셋을 확장합니다. (예: Self-Instruct, Alpaca, WizardLM).
    *   $$D^{(exp)} = \{ (x,y) | x \sim p_T(x|I \oplus c), y \sim p_T(y|I \oplus x) \}$$
*   **데이터 큐레이션 (Data Curation)**: 교사 LLM이 다양한 메타 정보 $m$을 활용하여 고품질의 대규모 데이터를 처음부터 합성합니다. (예: UltraChat, Phi 시리즈).
    *   $$D^{(cur)} = \{ (x,y) | x \sim p_T(x|I \oplus m), y \sim p_T(y|I \oplus x) \}$$
*   **특징 (Feature)**: 화이트-박스(White-box) 교사 LLM의 내부 표현(출력 분포, 중간 특징)을 추출합니다. (예: MiniLLM, BabyLlama).
    *   $$D^{(feat)} = \{ (x,y,\phi_{feat}(x,y;\theta_T)) | x \sim X, y \sim Y \}$$
*   **피드백 (Feedback)**: 교사 LLM이 학생 모델이 생성한 출력에 대해 선호도, 평가, 교정 정보를 제공합니다. (예: RLAIF, UltraFeedback).
    *   $$D^{(fb)} = \{ (x,y,\phi_{fb}(x,y;\theta_T)) | x \sim X, y \sim p_S(y|x) \}$$
*   **자기 지식 (Self-Knowledge)**: 학생 모델이 스스로 교사이자 학생이 되어 이전에 생성된 출력을 증류하고 개선하며 반복적으로 스스로를 향상시킵니다. (예: Self-Instruct, Self-Align, Self-Rewarding).
    *   $$D^{(sk)} = \{ (x,y,\phi_{sk}(x,y)) | x \sim S, y \sim p_S(y|I \oplus x) \}$$

### 3. 증류 알고리즘 (Distillation Algorithms)
추출된 지식을 학생 모델에 주입하는 방법은 다음과 같습니다.
*   **지도 미세 조정 (Supervised Fine-Tuning, SFT)**: 교사 LLM이 생성한 시퀀스의 우도를 최대화하여 학생 모델을 미세 조정합니다. (예: Alpaca, Vicuna).
    *   $$L_{SFT} = E_{x \sim X, y \sim p_T(y|x)} [-\log p_S(y|x)]$$
*   **발산 및 유사성 (Divergence and Similarity)**:
    *   **발산**: 교사 및 학생 모델의 확률 분포 간 발산을 최소화합니다. (예: Kullback-Leibler Divergence).
        *   $$L_{Div} = E_{x \sim X, y \sim Y} [D(p_T(y|x), p_S(y|x))]$$
    *   **유사성**: 학생 모델의 히든 상태나 특징을 교사 모델의 그것과 정렬합니다. (예: L2-Norm, Cross-Entropy).
        *   $$L_{Sim} = E_{x \sim X, y \sim Y} [L_F(\Phi_T(f_T(x,y)),\Phi_S(f_S(x,y)))]$$
*   **강화 학습 (Reinforcement Learning, RL)**:
    *   **증류된 보상 모델 훈련**: 교사 LLM이 생성한 피드백 데이터로 보상 모델 $r_\phi$를 훈련합니다.
        *   $$L_{RM}(r_\phi, D^{(fd)}) = -E_{(x,y_w,y_l)\sim D^{(fd)}} [\log\sigma(r_\phi(x,y_w) - r_\phi(x,y_l))]$$
    *   **강화 학습 최적화**: 훈련된 보상 모델에 따라 학생 모델 정책 $\pi_\theta$를 최적화합니다. (예: Constitutional AI, UltraFeedback).
        *   $$\max_{\pi_\theta} E_{x \sim X, y \sim \pi_\theta(y|x)} [r_\phi(x,y)] - \beta D_{KL}[\pi_\theta(y|x) \parallel \pi_{ref}(y|x)]$$
*   **순위 최적화 (Ranking Optimization)**: 고정된 선호도 데이터셋에서 언어 모델에 직접 순위 정보를 통합합니다. (예: Direct Preference Optimization (DPO), RRHF, PRO).
    *   DPO: $$E_{(x,y_w,y_l)\sim D^{(fd)}} [\log\sigma(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)})]$$
    *   RRHF: $$L_{RRHF} = \sum_{r_i < r_j} \max(0, p_i - p_j)$$
    *   PRO: $$L_{PRO} = -\sum_{k=1}^{n-1} \log \frac{\exp(p_k)}{\sum_{i=k}^{n} \exp(p_i)}$$

## 📊 결과
지식 증류(KD)는 독점 LLM과 오픈소스 LLM 간의 성능 격차를 크게 줄여 컴퓨팅 요구 사항을 간소화하고 AI 접근성을 높이는 데 중요한 역할을 합니다.

*   **맥락 이해 (Context Following)**:
    *   **명령어 추종**: Self-Instruct, Alpaca, WizardLM, Orca 시리즈는 교사 LLM으로부터 생성된 데이터를 통해 작은 모델들이 복잡하고 다양한 명령어를 효과적으로 따르도록 훈련시켰습니다. WizardLM은 ChatGPT보다 높은 성능을 보이기도 했으며, Phi 시리즈(Phi-1, Phi-1.5, Phi-2)는 "교과서 품질" 데이터를 통해 작은 모델임에도 Mistral이나 Llama-2보다 뛰어난 성능을 달성했습니다.
    *   **다중 턴 대화**: Vicuna, Baize, UltraLLaMA는 ShareGPT와 같은 실제 인간 대화 데이터 또는 자기 대화(self-chat)를 통해 다중 턴 대화 능력을 크게 향상시켰습니다.
    *   **RAG (검색 증강 생성) 능력**: SAIL, KARD, Self-RAG는 검색 결과 노이즈 제거, 정확한 응답 생성, 그리고 검색 필요성을 스스로 판단하는 적응형 RAG 능력(Self-RAG의 critic model)을 학생 모델에 성공적으로 증류했습니다.

*   **정렬 (Alignment)**:
    *   **사고 패턴**: SelFee, Reflection-Tuning, Orca, Orca 2는 교사 모델의 추론 과정(예: 단계별 설명, 자체 수정 과정)을 모방하여 학생 모델의 추론 능력을 향상시켰습니다.
    *   **선호도**: RLAIF(Reinforcement Learning from AI Feedback)는 인간 피드백을 모방하여 유용성, 무해성, 진실성 등 인간의 선호도에 모델을 정렬시켰으며, UltraFeedback과 Zephyr 같은 대규모 데이터셋과 기법을 통해 효율적으로 이루어졌습니다.
    *   **가치**: Constitutional AI, SANDBOX 등은 사회적 상호작용 시뮬레이션 및 원칙 기반 지침을 통해 LLM을 인간의 가치(도움, 무해, 정직)에 맞춰 정렬하는 가능성을 보여주었습니다.

*   **에이전트 능력 (Agent)**:
    *   **도구 사용**: Toolformer, Graph-ToolFormer, Gorilla, ToolLLM 등은 LLM이 외부 API 도구를 활용하여 복잡한 계산이나 전문 작업을 수행하는 능력을 증류했습니다.
    *   **계획**: FireAct, AgentTuning, Lumos, AUTOACT는 LLM이 고수준 작업을 실행 가능한 단계로 분해하는 계획 능력을 학습할 수 있음을 입증했습니다.

*   **NLP 작업 전문화 (NLP Task Specialization)**:
    *   **자연어 이해 (NLU)**: AugGPT, TDG, AnnoLLM 등은 텍스트 분류, 감성 분석, 자연어 추론 등에서 성능을 향상시켰습니다.
    *   **자연어 생성 (NLG)**: InheritSumm, Impossible Distillation, Genie 등은 요약, 기계 번역 등에서 교사 모델에 필적하는 고품질 텍스트 생성을 가능하게 했습니다.
    *   **정보 검색 (IR)**: QUILL, InPars, RankGPT 시리즈는 질의어 재작성, 검색기 및 재순위화 모델의 성능을 향상시켜 효율적인 정보 검색을 가능하게 했습니다.
    *   **추천**: NDR, InstructRec, ONCE는 LLM 지식을 활용하여 사용자 맞춤형 추천 시스템의 성능을 높였습니다.
    *   **텍스트 생성 평가**: PandaLM, Prometheus, InstructScore 등은 LLM을 활용하여 생성된 텍스트를 인간과 유사하게 평가하고 오류를 분석하는 효율적인 평가 모델을 구축했습니다.
    *   **코드**: Code Alpaca, Magicoder, WaveCoder 등은 코드 생성, 코드 완성 등 다양한 코드 관련 작업을 수행하는 전문화된 모델을 개발했습니다.

*   **멀티모달리티 (Multi-Modality)**: LLaVA, SVIT, Macaw-LLM 등은 이미지, 오디오, 비디오를 포함한 다양한 모달리티에서 지시를 이해하고 따르는 MLLM을 훈련하는 데 성공했습니다.

*   **수직화 증류 (Verticalization Distillation)**:
    *   **법률**: LawyerLLaMA, LawGPT, Fuzi는 법률 문서 이해, 질의응답, 사례 검색 등 법률 분야에 특화된 LLM을 구축했습니다.
    *   **의료 및 헬스케어**: HuatuoGPT, DoctorGLM, AlpaCare는 의료 상담, 진단, 약물 추천 등 의료 전문 지식을 갖춘 모델을 개발했습니다.
    *   **금융**: XuanYuan는 금융 도메인에 특화된 명령어 생성 및 훈련을 통해 금융 LLM의 가능성을 보여주었습니다.
    *   **과학**: DARWIN, SciGLM, WizardMath, MAmmoTH 등은 수학적 추론, 천문학 질의응답, 화학/재료 과학 예측, 생물학적 서열 분석 등 다양한 과학 분야에서 뛰어난 성능을 보였습니다.

## 🧠 통찰 및 논의
*   **KD의 중요성**: 지식 증류는 독점 LLM과 오픈소스 LLM 간의 격차를 해소하고, 고급 AI 기능에 대한 접근성을 민주화하며, AI 연구 및 개발의 효율성과 지속 가능성을 높이는 데 핵심적인 역할을 합니다.
*   **데이터 증강의 핵심 역할**: 단순히 데이터 양을 늘리는 것을 넘어, 맥락이 풍부하고 특정 스킬에 특화된 고품질 데이터를 생성하는 것이 KD의 효과를 극대화하는 데 결정적입니다. 합성 데이터 생성은 AI의 핵심 기술로 부상하고 있습니다.
*   **모방을 넘어선 전이**: KD는 단순히 교사 모델의 출력 동작을 모방하는 것을 넘어, 추론 패턴, 선호도 정렬, 가치 정렬과 같은 추상적인 인지 능력을 학생 모델에 전이시키는 방향으로 발전하고 있습니다.
*   **윤리적 및 법적 고려사항**: LLM KD를 수행할 때 모델 제공자의 이용 약관(예: 경쟁 제품 개발 제한)과 데이터 프라이버시, 편향성, 안전성 등 신뢰성과 관련된 윤리적 문제를 준수하는 것이 중요합니다.
*   **개방형 문제 및 미래 연구 방향**:
    *   **최적의 데이터 선택**: 효율적인 KD를 위해 얼마나 많은 데이터가 필요한지, 그리고 고품질의 최적화된 증류 데이터를 자동으로 선별하는 방법은 여전히 해결되지 않은 문제입니다.
    *   **증류 비용 절감**: 모델 압축(양자화, 가지치기, 저랭크 근사) 및 효율적인 미세 조정(PEFT, 메모리 효율적 방법)을 통해 증류 비용을 더욱 줄이는 연구가 필요합니다.
    *   **다중 교사 증류**: 여러 교사 모델의 지식과 강점을 하나의 학생 모델로 효과적으로 증류하는 방법은 아직 충분히 탐구되지 않았습니다.
    *   **더 풍부한 지식 탐색**: SFT의 하드 레이블링을 넘어, 교사 LLM으로부터 피드백(선호도, 비판) 및 내부 특징 지식(로그잇, 히든 상태)과 같은 더 풍부하고 심층적인 지식을 추출하는 방법론이 중요합니다.
    *   **치명적인 망각 극복**: 지속적인 미세 조정 또는 증류 과정에서 이전에 학습한 지식과 능력을 잃지 않도록 하는 효과적인 방법(예: 리허설, 정규화, 동적 아키텍처) 개발이 필요합니다.
    *   **신뢰성 있는 지식 증류**: 진실성, 안전성, 공정성, 견고성, 프라이버시 등 LLM의 신뢰성 전반을 KD 과정에서 고려하고 향상시키는 연구가 필수적입니다.
    *   **약-강 증류 (Weak-to-strong Distillation)**: 약한 감독으로도 더 강력한 모델의 고급 역량을 이끌어낼 수 있는 가능성을 탐구하는 것으로, 이론적/실용적 한계, 최적의 약한 감독자 식별, 그리고 다양한 모델 크기 및 유형 전반에서의 전이 및 확장 가능성에 대한 연구가 필요합니다.
    *   **자기 정렬 (Self-Alignment)**: 교사 모델의 제약을 넘어, 학생 모델 자체가 피드백을 생성하고 비판하며 스스로를 개선하고 정렬하는 자율적인 능력을 강화하는 방향으로 나아가야 합니다.

## 📌 TL;DR
**문제**: 강력하지만 고비용/비접근성 독점 LLM과 접근성 좋지만 성능이 부족한 오픈소스 LLM 간의 격차를 해소하고, 모델 압축 및 자체 개선을 효율적으로 달성해야 합니다.

**제안 방법**: 이 논문은 LLM 지식 증류(Knowledge Distillation, KD)에 대한 포괄적인 조사 연구입니다. KD는 지식 추출(레이블링, 확장, 데이터 큐레이션, 특징, 피드백, 자기 지식) 및 증류 알고리즘(지도 미세 조정, 발산 및 유사성, 강화 학습, 순위 최적화)을 기반으로 하며, 명령어 추종, 정렬, 에이전트 능력, NLP 작업 전문화, 멀티모달리티 등의 스킬을 전이하고 법률, 의료, 금융, 과학 등 특정 도메인에 특화됩니다. 특히, 데이터 증강(DA)이 고품질 스킬별 훈련 데이터를 생성하여 LLM 성능을 향상시키는 핵심 패러다임으로 강조됩니다.

**주요 발견**: KD를 통해 오픈소스 LLM은 복잡한 명령어 추종, 인간 선호도/가치 정렬, 전문적인 NLP 및 멀티모달 작업을 효율적으로 수행하여 독점 LLM과의 성능 격차를 크게 줄일 수 있습니다. Phi 시리즈와 Orca와 같은 모델들은 고품질 데이터와 사고 과정 증류의 중요성을 입증했습니다. 그러나 데이터 선택, 증류 비용 절감, 다중 교사 증류, 더 풍부한 지식 활용, 치명적인 망각 극복, 신뢰성 있는 KD, 그리고 약-강 증류 및 자기 정렬과 같은 분야에서 여전히 해결해야 할 과제가 많습니다.