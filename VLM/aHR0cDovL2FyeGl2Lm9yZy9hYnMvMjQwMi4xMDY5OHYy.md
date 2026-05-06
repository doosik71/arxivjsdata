# Question-Instructed Visual Descriptions for Zero-Shot Video Question Answering

David Romero and Thamar Solorio (2024)

## 🧩 Problem to Solve

본 논문은 비디오 질의응답(Video Question Answering, Video QA) 분야에서 기존 모델들이 가진 복잡성과 높은 비용 문제를 해결하고자 한다. 비디오 데이터는 단순한 이미지와 달리 여러 프레임 간의 관계를 파악해야 하며, 객체 인식뿐만 아니라 시공간적(temporal), 인과적(causal), 그리고 의미론적 추론 능력이 필수적으로 요구된다.

기존의 Video QA 접근 방식은 크게 두 가지 한계를 가지고 있다. 첫째, 모델 아키텍처가 매우 복잡하거나 연산 비용이 많이 드는 학습 파이프라인을 필요로 한다. 둘째, GPT와 같은 폐쇄형(closed) 모델에 의존하여 접근성이 떨어지고 내부 동작을 알 수 없는 경우가 많다. 따라서 본 연구의 목표는 복잡한 학습 과정 없이 오픈 소스 기반의 Vision-Language Model(VLM)을 활용하여, Zero-shot 환경에서도 경쟁력 있는 성능을 내는 단순하고 효율적인 Video QA 프레임워크를 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **'비디오 QA 문제를 텍스트 QA 문제로 변환'**하는 것이다. 이를 위해 단순히 비디오의 일반적인 묘사를 생성하는 것이 아니라, 주어진 질문에 따라 프레임에서 필요한 정보를 선택적으로 추출하는 **'질문 지시형 시각적 묘사(Question-Instructed Visual Descriptions)'** 방식을 제안한다.

중심적인 설계 직관은 다음과 같다. 질문과 관련된 구체적인 지침(instruction)을 VLM에 제공하여 각 프레임에서 질문 답변에 결정적인 힌트가 될 수 있는 텍스트 묘사를 생성하게 하고, 이렇게 생성된 텍스트 기반의 비디오 설명을 LLM(Large Language Model)의 입력으로 사용하여 최종 정답을 추론하게 하는 것이다.

## 📎 Related Works

### 기존 연구 및 한계

1. **Multimodal Pretraining for Video QA**: Flamingo, SeViLa, Frozen-Bilm 등은 비디오 프레임을 입력으로 받아 모달리티 정렬(modality alignment)을 통해 추론을 수행한다. 그러나 이들은 복잡한 아키텍처를 가지거나, 특정 모듈(Perceiver Resampler, Q-former 등)을 학습시켜야 하는 번거로움이 있다.
2. **Image Captions for Video Understanding**: 이미지 캡셔닝 능력을 활용해 시각적 내용을 텍스트로 변환 후 LLM으로 추론하는 방식이 제안되었다. 하지만 대다수의 기존 연구(ViperGPT, LLoVi 등)는 GPT-3.5와 같은 폐쇄형 모델의 API를 사용하여 캡션을 생성하거나 요약하는 방식을 취하고 있다.

### Q-ViD의 차별점

Q-ViD는 추가적인 경사 하강법(gradient-based) 학습이 전혀 필요 없는 **Gradient-free** 접근 방식을 취하며, 폐쇄형 모델 대신 오픈 소스 모델인 **InstructBLIP**을 활용한다. 특히, 일반적인 캡션이 아닌 '질문에 의존적인(question-dependent)' 캡션을 생성함으로써 텍스트 변환 과정에서 정보 손실을 최소화하고 추론의 정확도를 높였다는 점에서 차별화된다.

## 🛠️ Methodology

### 전체 파이프라인

Q-ViD의 전체 프로세스는 [비디오 프레임 샘플링 $\rightarrow$ 질문 지시형 캡션 생성 $\rightarrow$ 텍스트 기반 추론]의 순서로 진행된다.

### 상세 구성 요소 및 절차

**1. 질문 지시형 프레임 묘사 생성 (Frame Description Generation)**
입력 비디오 $V$에서 균등 샘플링(uniform sampling)을 통해 $n$개의 프레임 $\{f_1, f_2, \dots, f_n\}$을 추출한다. 이후 오픈 소스 모델인 InstructBLIP($I^b$)을 사용하여 각 프레임에 대한 캡션 $c_i$를 생성한다. 이때 사용되는 프롬프트 $E$는 다음과 같이 구성된다.

$$E = \text{concat}(B, Q)$$

여기서 $B$는 "Provide a detailed description of the image related to the question:"와 같은 기본 캡셔닝 지시어(base prompt)이며, $Q$는 사용자의 질문이다. 각 프레임의 캡션 $c_i$는 다음과 같이 계산된다.

$$c_i = I^b(f_i, E)$$

이 과정을 통해 비디오 $V$는 질문에 특화된 텍스트 시퀀스 $C = (c_1, c_2, \dots, c_n)$으로 변환된다.

**2. 추론 모듈 (Reasoning Module)**
생성된 캡션 집합 $C$를 비디오의 시간 순서대로 연결하여 전체 비디오 묘사를 형성한다. 이후 InstructBLIP의 언어 모델 부분인 Flan-T5를 reasoning module로 재사용한다. 최종 입력 프롬프트 $L$은 다음과 같이 구성된다.

$$L = \text{concat}(C, Q, A, T)$$

- $C$: 질문 지시형 프레임 캡션들의 집합
- $Q$: 질문
- $A$: 객관식 선택지 (Options)
- $T$: "Considering the information presented in the captions, select the correct answer in one letter (A, B, C) from the options."와 같은 작업 지시어(task description)

최종적으로 LLM은 $L$을 입력받아 객관식 정답 중 하나의 알파벳을 출력한다.

### 아키텍처 특징

- **InstructBLIP 활용**: Instruction-aware Q-former를 통해 지시어에 따라 가변적인 시각적 특징을 추출할 수 있다.
- **Frozen Model**: 모든 모델 파라미터는 고정(frozen)된 상태로 사용되며, 추가 학습을 진행하지 않는다.

## 📊 Results

### 실험 설정

- **데이터셋**: NExT-QA, STAR, How2QA, TVQA, IntentQA 총 5종의 다지선다형 Video QA 벤치마크를 사용하였다.
- **비교 대상**: SeViLa, FrozenBILM, VideoChat2(오픈 모델 기반) 및 ViperGPT, LloVi(GPT 기반) 등과 비교하였다.
- **구현 상세**: InstructBLIP-Flan-T5 XXL (12.1B 파라미터) 모델을 사용하였으며, 비디오당 64개의 프레임을 추출하였다.

### 주요 결과

- **종합 성능**: Q-ViD는 복잡한 아키텍처를 가진 SeViLa, VideoChat2 등보다 경쟁력 있거나 더 높은 성능을 보였다. 특히 TVQA에서는 VideoChat2를 제치고 가장 높은 성능을 기록하였다.
- **NExT-QA 분석**: 인과적(Causal) 및 시간적(Temporal) 질문에서 GPT 기반 모델이 아닌 모델 중 가장 우수한 성능을 보였으며, 이는 행동 추론 능력이 뛰어남을 입증한다.
- **IntentQA 분석**: 지도 학습(Supervised) 방식의 모델들을 압도하였으며, GPT-3.5 기반의 LloVi 모델에 근접하는 성능을 달성하였다.

## 🧠 Insights & Discussion

### 강점 및 발견

- **질문 지시형 프롬프트의 중요성**: Ablation study 결과, 일반적인 묘사를 생성하는 것보다 질문에 특화된 캡션을 생성하는 것이 성능 향상에 결정적인 역할을 함을 확인하였다. 특히 모델의 크기가 커질수록(XL $\rightarrow$ XXL) 이러한 질문 지시형 프롬프트의 효과가 극대화되었다.
- **단순성의 효율성**: 복잡한 정렬 모듈이나 추가 학습 없이도, 단순히 적절한 텍스트 변환 과정을 거치는 것만으로 최신 SOTA 모델들과 대등한 성능을 낼 수 있음을 보였다.
- **추론 프롬프트의 영향**: 최종 정답을 내는 QA 프롬프트를 복잡하게 만드는 것보다, 초기 캡션을 생성하는 프롬프트를 정교하게 만드는 것이 훨씬 더 중요하다는 점을 발견하였다.

### 한계 및 비판적 해석

- **환각 현상(Hallucinations)**: VLM이 캡션을 생성하는 과정에서 실제 영상에 없는 내용을 만들어내거나, 상세 묘사 대신 단답형 답변을 생성하는 경우가 발생한다.
- **메모리 및 토큰 제약**: 비디오가 매우 길어질 경우, 모든 프레임의 상세 캡션을 저장하고 LLM에 입력하는 과정에서 메모리 사용량이 급증하며, LLM의 최대 입력 토큰 제한(context window)으로 인해 처리가 어려울 수 있다.

## 📌 TL;DR

Q-ViD는 비디오 QA를 **[비디오 $\rightarrow$ 질문 특화 텍스트 묘사 $\rightarrow$ 텍스트 QA]** 과정으로 단순화한 Zero-shot 프레임워크이다. InstructBLIP을 통해 질문에 최적화된 프레임 캡션을 생성하고 이를 LLM으로 추론함으로써, 복잡한 학습이나 폐쇄형 GPT 모델 없이도 SOTA 수준의 성능을 달성하였다. 이 연구는 VLM의 instruction-following 능력을 극대화하여 비디오 이해 문제를 텍스트 영역으로 성공적으로 전이시켰으며, 향후 효율적인 비디오 추론 시스템 설계에 중요한 지침을 제공한다.
