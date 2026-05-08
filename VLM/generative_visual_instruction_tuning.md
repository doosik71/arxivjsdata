# GENERATIVE VISUAL INSTRUCTION TUNING

Jefferson Hernandez, Ruben Villegas, Vicente Ordonez (2024)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 거대 다중 모달 모델(Large Multimodal Model, LMM)에서 시각적 이해(Visual Understanding) 능력과 시각적 생성(Visual Generation) 능력을 동시에 확보하는 것이다.

기존의 많은 LMM들은 이미지 생성 능력을 추가했을 때, 기존에 보유하고 있던 시각적 이해나 언어 이해 능력이 저하되는 성능 퇴보 현상이 빈번하게 발생하였다. 또한, 시각적 이해와 생성이라는 서로 다른 성격의 작업을 하나의 통합된 프레임워크 내에서 효율적으로 수행하는 모델의 부재가 문제로 지적되었다.

따라서 본 연구의 목표는 이미지 이해, 이미지 생성, 그리고 이미지 편집(Image Editing)이라는 세 가지 기능을 동시에 수행하면서도, 각 기능의 성능을 서로 희생시키지 않는 통합 모델인 **GenLLaVA**를 개발하는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 아이디어는 **자동 생성된 명령어 수행 데이터(Automatically generated instruction-following data)**를 활용하여 모델을 튜닝함으로써, 추가적인 성능 저하 없이 생성 능력을 통합하는 것이다. 주요 기여 사항은 다음과 같다.

1. **생성형 다중 모달 명령어 데이터셋 구축**: GPT-4V와 기존의 이미지 생성 및 편집 데이터셋을 활용하여 이미지 이해, 생성, 편집을 모두 아우르는 새로운 명령어 튜닝 세트를 큐레이션하였다.
2. **GenLLaVA 모델 제안**: Mistral(언어 모델), SigLIP(시각 인코더), Stable Diffusion(이미지 생성기)이라는 세 가지 강력한 오픈소스 모델을 결합한 단일 복합 모델을 구축하였다.
3. **단일 단계 학습 레시피(Single-stage training recipe)**: 기존 LLaVA의 2단계 학습 방식 대신, 프로젝트와 언어 모델, 생성 헤드를 동시에 학습시키는 단일 단계 파이프라인을 도입하여 학습 비용을 줄이고 성능 저하 문제를 해결하였다.

## 📎 Related Works

논문은 LMM 연구를 세 가지 방향으로 분류하여 설명한다.

1. **Large Multimodal Models (LMMs)**: BLIP-2, LLaVA와 같이 시각 인코더와 LLM을 결합하여 이미지-텍스트 정렬을 수행하는 연구들이다. 최근 LLaVA-NeXT 등 성능 향상이 있었으나, 여전히 텍스트 응답에 국한된 경우가 많다.
2. **Diffusion-based LMMs**: GILL, MGIE 등 확산 모델(Diffusion Model)을 결합하여 시각적 생성을 수행하는 연구들이다. 이들은 주로 LLM의 은닉 표현을 확산 모델의 임베딩으로 매핑하는 방식을 사용한다.
3. **Token-based LMMs**: AnyGPT, Unified-IO 2, GPT-4o와 같이 시각적 콘텐츠를 이산 토큰(Discrete Tokens)으로 변환하여 언어 모델의 차기 토큰 예측(Next-token prediction) 방식으로 생성하는 연구들이다.

**차별점**: 기존 모델들은 특정 기능(예: 편집)에 특화되어 이해 능력이 떨어지거나, 생성 능력을 위해 너무 많은 토큰을 소모하는 경향이 있다. GenLLaVA는 오픈소스 모델들의 조합과 정교하게 설계된 데이터셋을 통해 이해와 생성을 균형 있게 통합하였다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 1. 시스템 구조 및 파이프라인

GenLLaVA는 시각적 이해와 생성을 동시에 수행하기 위해 다음과 같은 구조를 가진다.

- **시각 인코더 (Vision Encoder)**: SigLIP을 사용하여 이미지 $V$로부터 시각적 특징 $f$를 추출한다.
- **비전-언어 프로젝트 (Vision-Language Projector)**: 2층 GELU MLP를 사용하여 $f$를 언어 모델의 임베딩 공간으로 매핑한다.
- **언어 모델 (Language Model)**: Mistral-7B를 사용하여 텍스트 생성 및 생성용 특수 토큰을 예측한다.
- **시각 생성 헤드 (Visual Generation Head)**: LMM이 생성한 $[IMG]$ 토큰의 워드 임베딩 $e$와 은닉 상태 $h$를 입력받아, Stable Diffusion의 가이드가 될 시각적 잠재 변수(Visual Latent) $U$를 생성하는 4층 Encoder-Decoder Transformer이다.
- **확산 이미지 디코더 (Diffusion Image Decoder)**: Frozen 상태의 Stable Diffusion v1.4를 사용하여 $U$를 조건으로 이미지를 최종 생성한다.

### 2. 주요 방정식 및 작동 원리

**시각적 이해 단계**:
입력 이미지 $V$와 텍스트 $C=\{x_1, \dots, x_l\}$에 대해, 모델은 다음과 같이 다음 토큰을 예측한다.
$$f = \text{Enc}_{\text{vis}}(V)$$
$$x_t = \text{LMM}(\{x_1, \dots, x_{t-1}\} | W(f))$$
여기서 $W$는 프로젝트(Projector)를 의미한다.

**시각적 생성 단계**:
LMM이 생성한 $[IMG]$ 토큰들을 생성 헤드 $T$에 통과시켜 시각적 가이드 $U=\{u_1, \dots, u_L\}$를 얻는다.
$$u_t = T(\{u_1, \dots, u_{t-1}\} | \{e_{[IMG]} + h_{[IMG]}\})$$
이후, 확산 모델의 UNet $\epsilon_\theta$는 Cross-attention 메커니즘을 통해 $U$를 조건으로 노이즈를 예측하며 이미지를 생성한다. 이때 Attention 식은 다음과 같다.
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{\dim}}\right) \cdot V$$
여기서 $Q$는 이미지 잠재 변수 $z_t$에서, $K$와 $V$는 생성 헤드가 만든 $U$에서 유도된다.

### 3. 학습 절차

- **단일 단계 학습**: 다단계 학습 시 발생하는 성능 저하를 막기 위해 프로젝트, 언어 모델, 생성 헤드를 한꺼번에 파인튜닝한다.
- **태스크 토큰(Task Tokens)**: 모델이 사용자의 의도를 명확히 구분하도록 $[T2I]$ (Text-to-Image), $[I2T]$ (Image-to-Text), $[SOI]$ (Start of Image), $[EOI]$ (End of Image)와 같은 특수 토큰을 도입하였다.
- **데이터 구성**: ShareGPT(자연어), IPr2Pr(이미지 편집), LLaVA-Pretrain(이미지 생성), LLaVA-Finetune 및 LVIS-INSTRUCT4V(이미지 이해) 등을 혼합하여 약 207만 개의 샘플로 구성된 GVIT-mix-2076K 데이터셋을 사용하였다.

## 📊 Results

### 1. 실험 설정

- **지표**:
  - 시각적 이해: MathVista, MMMU, MM-Vet, SEED-B, MMB (정확도 기반)
  - 시각적 생성: FID (이미지 사실성), CLIP Similarity (텍스트-이미지 정렬도)
  - 이미지 편집: DINOScore (EVR 데이터셋)
- **비교 대상**: GILL, AnyGPT, MGIE, Unified-IO 2 등

### 2. 정량적 결과

- **시각적 이해**: GenLLaVA는 MathVista와 MMMU 같은 고급 지식 작업에서 다른 모델들을 상회하는 최고 점수를 기록하였다. 특히 SEED-B와 MMB에서 경쟁 모델 대비 월등한 성능을 보였다.
- **시각적 생성 및 편집**: Unified-IO 2와 대등한 수준의 CC3M 및 CLIP Similarity 결과를 보였으며, 편집 작업(EVR)에서도 범용 모델 중 최상위권의 성능을 기록하였다.
- **VQA 성능**: VQAv2(79.3%)와 GQA(62.9%)에서 강한 성능을 보였으며, 특히 Unified-IO 2와 함께 전반적으로 높은 성능을 유지하였다.

### 3. 절제 연구 (Ablation Study)

- **데이터 영향**: 생성 능력 추가 시 이해 능력이 하락하는 현상이 발견되었으나, 이해 관련 데이터(LVIS-INSTRUCT4V 등)의 비중을 높여 이를 상쇄하고 균형을 맞출 수 있었다.
- **비전 인코더**: CLIP보다 SigLIP을 사용했을 때 전반적인 성능이 더 높게 나타났다.
- **생성 토큰 수 ($N$)**: $N=16$일 때 최적의 성능을 보였다. 이는 생성과 편집이라는 복잡한 작업을 동시에 수행하기 위해 더 많은 시각적 토큰이 필요함을 시사한다.

## 🧠 Insights & Discussion

**강점**:
GenLLaVA는 서로 다른 성격의 모델(Mistral, SigLIP, Stable Diffusion)을 효율적으로 결합하여, 단일 모델이 이해와 생성이라는 두 마리 토끼를 모두 잡을 수 있음을 증명하였다. 특히 단일 단계 학습 레시피를 통해 학습 효율성을 높이면서도 기존 능력의 상실을 막은 점이 인상적이다.

**한계 및 논의사항**:

- **모델 크기의 한계**: GPT-4o나 Gemini와 같은 거대 폐쇄형 모델에 비해서는 여전히 성능 격차가 존재한다.
- **환각 현상 (Hallucination)**: 정성적 분석 결과, 여전히 일부 이미지 묘사에서 환각 현상이 발생하며, 이는 향후 개선해야 할 과제이다.
- **학습 데이터 의존성**: 생성 능력을 추가함에 따라 이해 능력이 떨어지는 현상을 데이터 양으로 해결했다는 점은, 모델 구조 자체의 해결책보다는 데이터 큐레이션에 크게 의존하고 있음을 의미한다.

## 📌 TL;DR

본 논문은 시각적 이해와 생성/편집 능력을 모두 갖춘 통합 LMM인 **GenLLaVA**를 제안한다. Mistral-7B, SigLIP, Stable Diffusion을 결합하고, 정교하게 설계된 다중 모달 명령어 데이터셋을 통해 **단일 단계 학습**을 수행함으로써, 성능 저하 없이 이해와 생성 기능을 통합하는 데 성공하였다. 이는 향후 범용 시각 비서(General-purpose visual assistant) 구축을 위한 효율적인 경로를 제시하며, 오픈소스 모델들의 조합만으로도 매우 경쟁력 있는 다중 모달 시스템을 만들 수 있음을 보여준다.
