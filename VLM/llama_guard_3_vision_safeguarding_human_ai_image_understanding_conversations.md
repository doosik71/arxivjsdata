# Llama Guard 3 Vision: Safeguarding Human-AI Image Understanding Conversations

Jianfeng Chi et al. (2024)

## 🧩 Problem to Solve

최근 대규모 언어 모델(LLM)의 발전으로 텍스트와 이미지를 동시에 처리하는 Vision-Language Multimodal 모델들이 등장하였으며, 이는 시각적 질의응답(VQA), 이미지 캡셔닝 등 다양한 분야에서 전문가 수준의 성능을 보여주고 있다. 그러나 이러한 모델들이 온라인 사용자와 상호작용할 때 유해한 콘텐츠를 생성하거나 전파할 위험이 존재한다.

기존의 시스템 가드레일(System Guardrails) 연구들은 대부분 텍스트 전용(Text-only)으로 설계되어 있어, 이미지가 포함된 멀티모달 대화의 안전성을 평가하는 데 한계가 있다. 따라서 이미지 이해가 포함된 인간-AI 대화에서 멀티모달 입력(Prompt)과 그에 따른 텍스트 출력(Response) 모두를 분류하여 안전하게 관리할 수 있는 전용 세이프가드 도구가 필요하다. 본 논문의 목표는 멀티모달 LLM 기반의 세이프가드인 **Llama Guard 3 Vision**을 개발하여 이미지 추론 유스케이스에서의 유해 콘텐츠 탐지 능력을 확보하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Llama 3.2-Vision 모델을 기반으로 하여, 텍스트와 이미지가 결합된 입력과 그에 대한 모델의 응답을 안전하게 분류하는 멀티모달 세이프가드 모델을 구축한 것이다.

주요 설계 아이디어는 MLCommons의 Hazard Taxonomy를 적용하여 13가지 유해 카테고리를 정의하고, 이를 기반으로 모델이 "safe" 또는 "unsafe" 여부를 판단하게 하는 것이다. 특히, 이미지 내의 실존 인물을 식별하는 행위를 개인정보 침해(Privacy violation)로 간주하는 등 이미지 이해 작업의 특성을 반영한 안전 정책을 수립하였다.

## 📎 Related Works

LLM의 유해 출력 방지 전략은 크게 두 가지로 나뉜다.

1. **Model-level Mitigation**: Instruction-tuning이나 RLHF(Reinforcement Learning from Human Feedback)를 통해 모델의 행동을 인간의 가치관에 정렬(Alignment)시키는 방법이다. 최근에는 이러한 정렬을 적대적 공격으로부터 보호하려는 연구들이 진행되고 있다.
2. **System-level Mitigation**: 기본 모델 외부에 안전 분류기(Safety Classifier)를 배치하여 입력 프롬프트나 출력 응답을 필터링하는 방식이다.

기존의 시스템 레벨 분류기들은 대부분 텍스트 전용으로 구현되어 있어, 멀티모달 환경에서의 안전성 확보에는 한계가 있었다. Llama Guard 3 Vision은 이러한 간극을 메우기 위해 멀티모달 입출력을 모두 지원하는 베이스라인을 제공함으로써 차별점을 가진다.

## 🛠️ Methodology

### 전체 파이프라인 및 구조

Llama Guard 3 Vision은 **Llama 3.2-Vision 11B** 모델을 기반으로 Supervised Fine-Tuning(SFT)을 통해 구축되었다. 이 모델은 크게 두 가지 작업(Task)을 수행한다.

- **Prompt Classification**: 사용자가 입력한 이미지와 텍스트 프롬프트의 유해성 판단.
- **Response Classification**: 사용자의 멀티모달 입력과 그에 대해 AI 에이전트가 생성한 텍스트 응답의 유해성 판단.

### 주요 구성 요소 및 학습 절차

모델의 입력은 다음의 네 가지 요소로 구성된다.

1. **Guidelines**: 유해 카테고리에 대한 번호가 매겨진 상세 설명서.
2. **Classification Type**: 프롬프트를 분류할 것인지, 응답을 분류할 것인지에 대한 지정.
3. **Conversation**: 사용자 제공 이미지 및 사용자/에이전트 간의 대화 기록(단일 또는 다회차).
4. **Output Format**: 기본적으로 "safe" 또는 "unsafe"를 출력하며, "unsafe"일 경우 위반된 카테고리 번호를 함께 출력한다.

### 데이터 수집 및 학습 상세

- **데이터셋**: 인간이 작성한 프롬프트-이미지 쌍과 Llama 모델을 통해 합성된 벤지인(Benign) 및 유해 응답(Violating responses) 데이터를 혼합하여 사용하였다. 유해 응답 생성에는 Jailbreaking 기법이 활용되었으며, Llama 3.1 405B 모델이 레이블링에 참여하였다.
- **데이터 규모**: 프롬프트 분류 데이터 22,500개, 응답 분류 데이터 40,034개를 사용하였다. 또한, 텍스트 전용 Llama Guard 3 데이터에 더미 이미지를 추가하여 성능과 강건성을 높였다.
- **학습 설정**:
  - Sequence length: 8192
  - Learning rate: $1 \times 10^{-5}$
  - Training steps: 3600
  - 이미지 처리: 이미지를 $560 \times 560$ 픽셀 크기의 4개 청크(Chunk)로 리스케일링하여 입력한다.
- **데이터 증강(Data Augmentation)**: 위반되지 않은 카테고리를 무작위로 삭제하거나 카테고리 인덱스를 셔플링하여 모델이 형식을 암기하는 것을 방지하고 일반화 능력을 향상시켰다.

## 📊 Results

### 성능 평가 (Internal Benchmark)

MLCommons Hazard Taxonomy를 기반으로 GPT-4o 및 GPT-4o mini와 성능을 비교하였다.

- **정량적 결과**: Llama Guard 3 Vision은 특히 **Response Classification**에서 GPT-4o 계열보다 높은 $\text{F1 score}$를 기록하였으며, 오탐지율(False Positive Rate, FPR)을 훨씬 낮게 유지하였다.
- **작업별 차이**: Prompt Classification의 성능이 Response Classification보다 낮게 나타났다. 이는 "이걸 어떻게 사나요?"라는 텍스트와 복잡한 이미지가 함께 있을 때, 사용자가 가리키는 대상에 따라 유해성 여부가 달라지는 모호성(Ambiguity) 때문으로 분석된다. 이에 따라 저자들은 응답 분류 작업에 모델을 사용할 것을 권장한다.
- **카테고리별 성능**: Indiscriminate Weapons 및 Elections 카테고리에서 매우 높은 성능을 보였으며, 전 카테고리에서 $\text{F1 score} > 0.69$를 달성하였다.

### 적대적 강건성 (Adversarial Robustness)

White-box 공격인 PGD(이미지 공격)와 GCG(텍스트 공격)를 통해 모델을 스트레스 테스트하였다.

- **PGD 공격**: 이미지 픽셀을 최적화하여 "safe" 판정을 유도하는 공격이다.
  - Prompt Classification은 매우 취약하여, 작은 섭동($8/255$)만으로도 유해 프롬프트의 오분류율이 $21\%$에서 $70\%$로 급증하였다.
  - 반면, Response Classification은 이미지 공격에 훨씬 강건하였다. 이는 모델이 판단 시 이미지보다 텍스트 응답에 더 의존하기 때문이다.
- **GCG 공격**: 텍스트 접미사(Suffix)를 최적화하여 공격하는 방식이다.
  - Prompt Classification의 경우 $72\%$가 오분류될 정도로 취약하였다.
  - Response Classification에서는 공격자가 프롬프트만 수정할 수 있을 때는 비교적 강건하였으나, 에이전트의 응답 내용 자체에 GCG 접미사가 포함될 경우 오분류율이 $75\%$까지 상승하였다.

## 🧠 Insights & Discussion

### 강점 및 통찰

Llama Guard 3 Vision은 멀티모달 입출력을 모두 처리할 수 있는 범용 세이프가드로서, 특히 **응답 분류(Response Classification)** 단계에서 강력한 성능과 강건성을 보여준다. 이는 공격자가 입력 프롬프트를 조작하더라도, 최종 생성된 텍스트 응답이 유해하다면 이를 효과적으로 차단할 수 있음을 시사한다.

### 한계 및 비판적 해석

1. **적대적 취약성**: 이미지 기반의 PGD 공격이나 텍스트 기반의 GCG 공격에 여전히 취약하다. 특히 프롬프트 분류 단계에서의 취약성은 멀티모달 모델이 시각적 노이즈에 민감함을 보여준다.
2. **데이터 및 도메인 제약**: 영어 언어에 최적화되어 있으며, 현재는 프롬프트당 단 한 장의 이미지만 지원한다. 또한 이미지 리스케일링 과정에서 정보 손실이 발생하여 성능에 영향을 줄 수 있다.
3. **지식 의존성**: 명예훼손(S5), 지식재산권(S8), 선거(S13)와 같은 카테고리는 최신 팩트 체크 능력이 필요하므로, 단순 분류기보다는 더 복잡한 시스템적 보완이 필요하다.

## 📌 TL;DR

본 논문은 멀티모달 LLM 대화의 안전성을 확보하기 위한 세이프가드 모델인 **Llama Guard 3 Vision**을 제안한다. Llama 3.2-Vision을 기반으로 13가지 유해 카테고리를 분류하도록 학습되었으며, 실험 결과 응답 분류 작업에서 GPT-4o보다 우수한 $\text{F1 score}$와 낮은 오탐지율을 보였다. 비록 적대적 공격에 일부 취약점이 있으나, 프롬프트와 응답 분류를 동시에 적용하고 Perplexity 필터 등을 결합함으로써 멀티모달 AI 시스템의 안전성을 크게 향상시킬 수 있는 중요한 베이스라인을 제공한다.
