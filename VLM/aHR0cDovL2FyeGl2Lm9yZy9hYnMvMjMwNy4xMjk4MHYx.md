# A Systematic Survey of Prompt Engineering on Vision-Language Foundation Models

Jindong Gu, Zhen Han, Shuo Chen, Ahmad Beirami, Bailan He, Gengyuan Zhang, Ruotong Liao, Yao Qin, Volker Tresp, Philip Torr (2023)

## 🧩 Problem to Solve

본 논문은 최근 급격히 발전하고 있는 Vision-Language Foundation Models (VLMs)에서의 Prompt Engineering 기법들에 대한 체계적인 정리와 분석의 부재라는 문제를 해결하고자 한다.

Prompt Engineering은 사전 학습된 거대 모델에 태스크 특화된 힌트인 '프롬프트(Prompt)'를 추가하여 모델을 새로운 태스크에 적응시키는 기술이다. 기존의 전통적인 머신러닝 패러다임은 대량의 레이블링된 데이터가 필요하고, 모델 전체 또는 일부 파라미터를 미세 조정(Fine-tuning)해야 하므로 계산 자원 소모가 크고 확장성이 떨어진다는 한계가 있다. 반면, Prompt Engineering은 모델 파라미터를 업데이트하지 않고도 예측을 수행할 수 있게 하여 실세계 적용 가능성을 높인다.

따라서 본 논문의 목표는 Multimodal-to-Text Generation, Image-Text Matching, Text-to-Image Generation이라는 세 가지 주요 VLM 유형을 중심으로 최신 프롬프트 엔지니어링 연구를 종합적으로 조사하고, 방법론적 분류 및 책임감 있는 AI(Responsible AI) 관점에서의 분석을 제공하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Vision-Language 모델을 위한 프롬프트 엔지니어링의 체계적인 Taxonomy(분류 체계)를 구축한 것에 있다.

1. **프롬프트 유형의 이분법적 분류**: 프롬프트를 가독성에 따라 사람이 읽을 수 있는 **Hard Prompt**(이산적 토큰)와 최적화 가능한 벡터 형태인 **Soft Prompt**(연속적 벡터)로 정의하고 세부 분류를 제시하였다.
2. **VLM 유형별 분석**:
    - **Multimodal-to-Text Generation**: 생성 기반 모델의 융합 모듈 구조와 프롬프트 적용 방식 분석.
    - **Image-Text Matching**: 대조 학습(Contrastive Learning) 기반 모델에서의 텍스트, 시각, 통합 프롬프트 기법 분석.
    - **Text-to-Image Generation**: 확산 모델(Diffusion Models)을 중심으로 한 시맨틱 설계 및 제어 가능 생성 기법 분석.
3. **단일 모달 모델과의 비교**: NLP 및 Pure Vision 모델의 프롬프트 기법과 VLM의 공통점 및 차이점을 심층 논의하였다.
4. **책임감 있는 AI 논의**: 프롬프트 엔지니어링 과정에서 발생할 수 있는 편향(Bias), 강건성(Robustness), 개인정보 보호 및 백도어 공격 등의 보안 이슈를 체계적으로 정리하였다.

## 📎 Related Works

논문은 Prompt Engineering이 처음으로 대중화된 Natural Language Processing (NLP) 분야의 연구들을 언급한다. NLP에서는 In-Context Learning, Chain-of-Thought 등이 연구되었으며, 이후 이러한 흐름이 Computer Vision 및 VLM 분야로 확장되었다.

기존의 접근 방식은 모델 파라미터를 직접 수정하는 Fine-tuning 중심이었으나, 이는 데이터 확보 비용이 높고 모델의 일반화 능력을 저해할 수 있다는 한계가 있었다. 본 논문에서 다루는 프롬프트 기반 접근 방식은 모델의 구조를 유지하면서 입력단에서 힌트를 제공함으로써 파라미터 효율적인 적응(Parameter-efficient adaptation)을 가능하게 한다는 점에서 차별점을 갖는다.

## 🛠️ Methodology

본 논문은 서베이 논문으로서 특정 알고리즘을 제안하기보다, 기존 연구들의 방법론을 다음과 같은 구조로 분석한다.

### 1. Multimodal-to-Text Generation

이 모델들은 일반적으로 텍스트 특징, 시각 특징, 그리고 이를 통합하는 Fusion Module로 구성된다.

- **Fusion Module 구조**:
  - **Encoder-Decoder**: 융합 인코더 $E$가 시각 및 텍스트 정보를 통합하여 공동 표현을 만들고, 이를 생성 모듈 $G$에 전달한다. 수식으로는 $y = G(E(x_{input}))$으로 표현된다.
  - **Decoder-only**: 공동 표현을 미리 만들지 않고 디코딩 단계에서 직접 결합한다. 수식으로는 $y = G(x_{input})$이다.
- **프롬프트 기법**:
  - **Hard Prompt**: Task Instruction, In-context Learning, Retrieval-based Prompting, Chain-of-Thought Prompting으로 나뉜다.
  - **Soft Prompt**: Prompt Tuning과 Prefix Token Tuning이 있으며, 특히 Prompt Tuning의 목적 함수는 다음과 같이 정의된다.
    $$\text{argmin}_{x_p} L(F(y_i, x_p) | y_{<i}, x_{input})$$
    여기서 $x_p$는 학습 가능한 프롬프트 파라미터이며, 모델 출력과 실제 정답 사이의 손실 $L$을 최소화하는 방향으로 최적화된다.

### 2. Image-Text Matching

CLIP과 같은 모델들이 대표적이며, 이미지와 텍스트 임베딩을 정렬시키는 대조 학습(Contrastive Learning)을 사용한다.

- **손실 함수**: 이미지-텍스트 손실 $L_{i2t}$와 텍스트-이미지 손실 $L_{t2i}$의 합으로 구성된다.
    $$L_{i2t} = -\frac{1}{N} \sum_{i=1}^N \log \left( \frac{\exp \text{sim}(f^l_v(v_i), f^l_t(t_i))}{\sum_{j=1}^N \exp \text{sim}(f^l_v(v_i), f^l_t(t_j))} \right)$$
    여기서 $\text{sim}(\cdot)$은 코사인 유사도를 의미하며, 정답 쌍의 유사도는 높이고 오답 쌍의 유사도는 낮추는 방향으로 학습한다.
- **프롬프트 적용 범위**: 텍스트 인코더 프롬프팅, 시각 인코더 프롬프팅(Patch-wise 또는 Annotation 방식), 그리고 두 가지를 모두 사용하는 Unified Prompting으로 분류된다.

### 3. Text-to-Image Generation

최근의 Diffusion Model(확산 모델)을 중심으로 분석한다.

- **학습 절차**:
  - **Forward Process**: 깨끗한 이미지 $x_0$에 점진적으로 가우시안 노이즈를 추가하여 $x_T$를 만드는 과정이다.
  - **Reverse Process**: 학습된 노이즈 예측기 $\epsilon_\theta$를 통해 노이즈를 제거하여 원본 이미지를 복원하는 과정이다.
- **조건부 생성 목적 함수**: 텍스트 프롬프트 $c$가 주어졌을 때의 손실 함수는 다음과 같다.
    $$L(\theta) = E_{t, x_0, \epsilon, c} \left[ \| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t, c) \|^2 \right]$$
    여기서 $\epsilon_\theta$는 입력 이미지, 타임스텝 $t$, 그리고 조건 $c$(프롬프트)를 바탕으로 추가된 노이즈 $\epsilon$을 예측한다.

## 📊 Results

본 논문은 개별 실험 결과보다는 다수 논문의 결과를 종합한 분석 내용을 제시한다.

1. **Multimodal-to-Text**:
    - **In-Context Learning**: Flamingo 등의 모델에서 소수의 예시(Few-shot)만으로도 새로운 개념을 빠르게 학습하는 'Fast Concept Binding' 능력이 확인되었다.
    - **Prompt Tuning**: 일부 연구에서 프롬프트 튜닝이 전체 미세 조정보다 강건성(Robustness) 면에서 더 우수한 성능을 보였음이 나타났다.
2. **Image-Text Matching**:
    - **Soft Prompting**: CoOp와 같은 학습 가능한 프롬프트 기법이 고정된 텍스트 템플릿(예: "a photo of a [CLASS]")보다 제로샷 성능을 유의미하게 향상시켰다.
    - **Visual Prompting**: VPT(Visual Prompt Tuning)가 이미지 패치 형태로 프롬프트를 추가했을 때, 전체 모델 튜닝에 근접하는 성능을 내면서도 효율적임이 입증되었다.
3. **Text-to-Image**:
    - **Semantic Design**: 명사 기반의 구체적인 설명과 조명/화풍 관련 키워드가 생성 이미지의 품질과 분위기에 결정적인 영향을 미침이 분석되었다.
    - **ControlNet**: 텍스트 프롬프트 외에 에지 맵(Edge map)과 같은 추가 조건을 부여함으로써 생성 결과의 정밀한 제어가 가능해졌다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 논문은 파편화되어 있던 VLM 프롬프트 연구를 세 가지 모델 유형과 두 가지 프롬프트 성격(Hard/Soft)으로 명확히 체계화하였다. 특히 단순히 성능 향상 기법만 나열한 것이 아니라, Responsible AI 관점에서 편향성과 보안 취약점을 함께 다룸으로써 학술적 깊이를 더했다.

### 한계 및 논의사항

- **메커니즘의 불투명성**: In-context learning이나 Instruction tuning이 실제 모델 내부에서 어떻게 작동하는지에 대한 이론적 설명은 여전히 부족하며, 이는 향후 연구의 핵심 과제이다.
- **상호 운용성 문제**: Multimodal-to-Text 모델과 Text-to-Image 모델이 서로 다른 개념 공간을 가지고 있어, 하나의 모델에서 생성한 프롬프트가 다른 모델에서도 동일하게 작동하는 'Universal Prompt' 구축이 어렵다는 점이 지적된다.
- **보안 취약성**: 프롬프트 튜닝 과정에서 주입된 백도어(Backdoor)가 모델의 하위 태스크로 전이될 수 있는 위험성이 존재한다.

## 📌 TL;DR

본 논문은 Vision-Language Foundation Models의 프롬프트 엔지니어링 기법을 **Hard/Soft 프롬프트**라는 기준과 **생성/매칭/합성**이라는 세 가지 모델 유형을 중심으로 집대성한 종합 서베이 보고서이다.

이 연구는 거대 VLM을 효율적으로 적응시키기 위한 방법론적 가이드라인을 제공하며, 향후 **멀티모달 통합 프롬프트 개발**, **모델 내부 작동 기법의 해석**, 그리고 **편향 및 보안 문제 해결**이 VLM 연구의 핵심 방향이 될 것임을 제시한다.
