# Multimodal Foundation Models: From Specialists to General-Purpose Assistants

Chunyuan Li, Zhe Gan, Zhengyuan Yang, Jianwei Yang, Linjie Li, Lijuan Wang, Jianfeng Gao (2023)

## 🧩 Problem to Solve

본 논문은 컴퓨터 비전(CV) 및 비전-언어(Vision-Language, VL) 분야에서 모델들이 개별 작업에 특화된 '전문의(Specialist)' 형태에서 다양한 작업을 수행할 수 있는 '범용 어시스턴트(General-Purpose Assistant)'로 진화하는 과정과 그 분류 체계(Taxonomy)를 분석하는 것을 목표로 한다.

과거의 컴퓨터 비전 연구는 이미지 분류, 객체 탐지, 세그멘테이션 등 특정 작업(Task)을 해결하기 위해 개별적인 모델을 학습시키는 방식에 집중하였다. 그러나 최근 대규모 언어 모델(Large Language Models, LLMs)의 성공으로 인해, 텍스트 분야에서 나타난 '하나의 모델이 다양한 작업을 수행하는' 패러다임을 비전 분야에도 적용하려는 시도가 급증하고 있다. 따라서 본 연구는 비전 이해(Visual Understanding), 비전 생성(Visual Generation), 통합 비전 모델(Unified Vision Models), LLM 기반의 다중모달 모델(LMMs), 그리고 다중모달 에이전트(Multimodal Agents)라는 다섯 가지 핵심 주제를 통해 이 진화 과정을 체계적으로 정리하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다중모달 파운데이션 모델의 발전 궤적을 '전문의 모델 $\rightarrow$ 통합 모델 $\rightarrow$ 범용 어시스턴트'의 흐름으로 정의하고, 이를 상세히 분석한 종합 서베이를 제공한다는 점이다.

주요 직관은 LLM이 보여준 In-context Learning, Chain-of-Thought, Instruction-following과 같은 능력을 비전-언어 모델에서도 구현함으로써, 사용자의 복잡한 의도(Human Intent)를 이해하고 실행하는 범용 시각 어시스턴트를 구축할 수 있다는 것이다. 이를 위해 본 논문은 단순한 문헌 리뷰를 넘어, 모델 아키텍처의 통합 방식, LLM을 활용한 데이터 생성 및 학습 전략, 그리고 외부 도구를 체이닝(Chaining)하는 에이전트 구조까지 포괄하는 분석 프레임워크를 제시한다.

## 📎 Related Works

논문은 다중모달 파운데이션 모델을 크게 두 가지 클래스로 분류하여 관련 연구를 설명한다.

첫째, 특정 목적을 위해 사전 학습된 전문 모델들이다. 여기에는 이미지 백본 학습을 위한 Supervised Pre-training, CLIP으로 대표되는 Contrastive Language-Image Pre-training, 그리고 MAE와 같은 Image-only Self-Supervised Learning이 포함된다. 기존 연구들은 주로 특정 벤치마크 성능 향상에 집중했으며, 이는 모델이 학습 데이터에 포함된 닫힌 집합(Closed-set) 내에서만 작동한다는 한계가 있었다.

둘째, 범용 어시스턴트를 지향하는 최신 연구들이다. LLM의 성공에 영감을 받은 Unified Vision Models, LMMs(Large Multimodal Models), 그리고 Multimodal Agents가 이에 해당한다. 기존의 VLP(Vision-Language Pre-training) 연구들이 단순한 이미지-텍스트 매칭이나 캡셔닝에 치중했다면, 최신 접근 방식은 LLM의 추론 능력을 결합하여 복잡한 지시사항을 수행하거나 외부 API를 호출하는 능력을 갖추는 방향으로 차별화된다.

## 🛠️ Methodology

본 논문은 다중모달 모델의 구현 방법론을 크게 세 가지 경로로 설명한다.

### 1. 비전-언어 정렬 및 통합 (Visual Understanding & Generation)

비전 이해 모델은 CLIP과 같이 이미지와 텍스트를 공통의 임베딩 공간에 매핑하는 Contrastive Learning을 핵심으로 한다. 이미지 생성 모델, 특히 Stable Diffusion과 같은 Diffusion Model은 다음과 같은 구조를 가진다.

- **VAE**: 이미지를 저차원의 잠재 공간(Latent Space) $z$로 압축하고 다시 복원한다.
- **Text Encoder**: CLIP의 텍스트 인코더를 사용하여 텍스트 쿼리 $y$를 특징 벡터 $\tau(y)$로 변환한다.
- **Denoising U-Net**: 가우시안 노이즈가 섞인 잠재 변수 $z_t$에서 노이즈 $\epsilon$을 예측하여 제거함으로써 이미지를 생성한다. 이때 Cross-attention 메커니즘을 통해 텍스트 특징과 시각 특징을 융합하며, 수식은 다음과 같다.
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) \cdot V$$

### 2. LLM 기반의 다중모달 학습 (Training with LLM)

LMM(Large Multimodal Model)은 이미지 인코더와 LLM을 연결 모듈(Connection Module)로 결합한 구조를 가진다. LLaVA와 같은 모델은 다음과 같은 2단계 학습 절차를 따른다.

- **Stage 1 (Feature Alignment)**: CLIP 이미지 인코더와 LLM 사이의 선형 투영 층(Linear Projection Layer)만을 학습시켜 시각 특징을 LLM이 이해할 수 있는 텍스트 임베딩 공간으로 정렬한다.
- **Stage 2 (Instruction Tuning)**: GPT-4와 같은 강력한 모델을 이용하여 생성한 '지시어-응답' 쌍(Instruction-following data)을 통해 LLM과 연결 모듈을 함께 미세 조정(Fine-tuning)한다.

### 3. 도구 체이닝 기반의 에이전트 (Chaining Tools with LLM)

학습 없이 LLM의 계획(Planning) 능력을 활용하는 방식이다. MM-ReAct와 같은 시스템은 다음과 같은 루프를 수행한다.

- **Planning**: 사용자의 요청을 분석하여 어떤 외부 도구(OCR, 객체 탐지, 캡셔닝 등)를 어떤 순서로 사용할지 결정한다.
- **Execution**: 결정된 도구를 호출하여 결과(Observation)를 텍스트 형태로 얻는다.
- **Response**: 얻어진 모든 관찰 결과와 대화 기록을 종합하여 LLM이 최종 답변을 생성한다.

## 📊 Results

논문은 다양한 벤치마크를 통해 모델들의 성능과 능력을 분석한다.

- **LMM 성능**: LLaVA는 LLaVA-Bench(COCO 및 In-the-Wild)에서 GPT-4가 평가한 상대 점수 기준으로 각각 85.1%, 73.5%를 기록하며, 오픈소스 모델로서 강력한 시각적 대화 능력을 보여주었다.
- **제로샷 OCR**: LMM들이 명시적인 OCR 학습 없이도 텍스트 인식 능력을 갖추고 있음을 확인하였다. 특히 BLIP-2와 같은 모델은 대규모 데이터 학습을 통해 복잡한 텍스트 인식에서도 강점을 보였다.
- **에이전트의 능력**: MM-ReAct는 단일 모델로는 불가능한 '다중 이미지 추론(Multi-image Reasoning)', '시각적 수학 풀이', '비디오 요약' 등의 복잡한 작업을 수행할 수 있음을 정성적으로 입증하였다.
- **비교 분석**: 전문 모델은 특정 작업에서 효율적이지만 확장성이 낮고, LMM은 범용성이 높지만 학습 비용이 크며, 에이전트는 추가 학습 없이 빠르게 구축 가능하지만 도구 호출의 정확도에 의존한다는 트레이드오프(Trade-off)를 제시한다.

## 🧠 Insights & Discussion

본 논문은 다중모달 모델이 단순한 성능 향상을 넘어 '인간의 의도에 정렬(Alignment)'되는 방향으로 나아가고 있음을 강조한다.

**강점 및 기회**:
LLM의 Instruction Tuning 기법을 비전 분야에 도입함으로써, 모델이 단순한 캡셔닝을 넘어 복잡한 추론과 대화가 가능해졌다. 특히, 데이터 중심(Data-centric) 접근법을 통해 고품질의 지시어 데이터를 생성하는 것이 모델의 성능을 결정짓는 핵심 요소가 되었다.

**한계 및 도전 과제**:
첫째, 오픈소스 모델과 GPT-4와 같은 폐쇄형 모델 사이에는 여전히 거대한 성능 격차가 존재하며, 이는 주로 데이터의 규모와 추론 능력의 차이에서 기인한다. 둘째, 비전 데이터는 텍스트에 비해 저장 비용이 매우 높고 라벨링 비용이 비싸, LLM 수준의 Scaling Law를 적용하는 데 제약이 있다. 셋째, 환각(Hallucination) 문제, 특히 이미지에 없는 객체를 있다고 주장하는 현상이 여전히 해결해야 할 과제로 남아 있다.

**비판적 해석**:
현재의 LMM들은 주로 '이미지 $\rightarrow$ 텍스트' 생성에 치중되어 있다. 진정한 범용 어시스턴트가 되기 위해서는 텍스트, 이미지, 오디오, 비디오를 동시에 입력받고 출력할 수 있는 'Any-to-Any' 모델로의 진화가 필수적이다.

## 📌 TL;DR

본 논문은 다중모달 파운데이션 모델이 특정 작업 전용 모델(Specialists)에서 범용 시각 어시스턴트(General-Purpose Assistants)로 진화하는 과정을 체계적으로 분석한 서베이 보고서이다.

핵심은 **시각-언어 정렬 $\rightarrow$ LLM 결합 및 Instruction Tuning $\rightarrow$ 외부 도구 체이닝**으로 이어지는 기술적 흐름이며, 이를 통해 인간의 복잡한 의도를 이해하고 실행하는 AI 에이전트 구축의 가능성을 제시한다. 이 연구는 향후 다중모달 AI가 단순한 인식을 넘어 계획, 기억, 도구 사용 능력을 갖춘 자율적 에이전트로 발전하는 데 중요한 이론적 토대를 제공한다.
