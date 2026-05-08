# VisionLLM: Large Language Model is also an Open-Ended Decoder for Vision-Centric Tasks

Wenhai Wang et al. (2023)

## 🧩 Problem to Solve

최근 대규모 언어 모델(Large Language Models, LLMs)은 사용자 맞춤형 지시어(instructions)를 통해 제로샷(zero-shot) 능력을 발휘하며 다양한 NLP 과업에서 혁신적인 성과를 거두었다. 그러나 컴퓨터 비전 분야의 시각 기반 모델(Vision Foundation Models, VFMs)은 여전히 사전에 정의된(pre-defined) 태스크 형식에 국한되어 있으며, LLM과 같은 개방형(open-ended) 과업 수행 능력을 갖추지 못하고 있다.

기존의 범용 비전 모델들은 멀티태스크 통합 방식을 사용하더라도 정의된 태스크의 한계를 극복하기 어려웠으며, 최근 등장한 시각적 프롬프트 튜닝(visual prompt tuning)은 LLM의 언어 지시어 형식과 일치하지 않아 LLM의 추론 능력과 세계 지식을 직접적으로 활용하는 데 한계가 있다. 따라서 본 논문의 목표는 LLM의 유연한 지시어 기반 제어 능력을 비전 중심 과업에 통합하여, 정해진 형식을 넘어 사용자가 정의한 대로 과업을 수행할 수 있는 통합 범용 프레임워크인 VisionLLM을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 이미지를 하나의 '외국어'로 취급하여 비전 과업을 언어 모델이 처리할 수 있는 형식으로 정렬하는 것이다. 이를 통해 비전 중심 과업을 언어 지시어로 유연하게 정의하고 관리할 수 있도록 설계하였다.

주요 기여 사항은 다음과 같다:

1. **통합 언어 지시어(Unified Language Instruction) 설계**: 비전 전용 과업과 비전-언어 과업을 모두 포괄하며, LLM의 형식에 맞춘 통합 인터페이스를 제안하였다.
2. **언어 가이드 이미지 토크나이저(Language-Guided Image Tokenizer) 개발**: 주어진 언어 프롬프트에 따라 시각 정보를 유연하게 인코딩하여 LLM이 이해할 수 있는 토큰 형태로 변환한다.
3. **LLM 기반 개방형 태스크 디코더(LLM-based Open-Ended Task Decoder) 구축**: Alpaca-7B 모델을 기반으로, 확장된 어휘집(vocabulary)과 '출력 형식을 쿼리로 사용하는(output-format-as-query)' 디코딩 방식을 통해 다양한 비전 과업을 수행한다.

## 📎 Related Works

### 관련 연구 및 한계

- **Large Language Models**: GPT 시리즈와 LLaMA 등의 모델은 뛰어난 추론 능력을 갖추고 있다. 이를 활용한 API 기반 애플리케이션(Visual ChatGPT, HuggingGPT 등)이 등장했으나, 이는 개별 API를 호출하는 방식이므로 세밀한 시각적 세부 사항을 포착하거나 복잡한 시각적 문맥을 이해하는 데 한계가 있다.
- **Vision Generalist Models**: Uni-Perceiver, Pix2Seq v2 등은 다양한 태스크를 시퀀스 생성 문제로 통합하려 시도했다. 하지만 이들은 여전히 사전에 정의된 태스크 셋에 의존하며, LLM처럼 언어 지시어를 통한 유연한 과업 맞춤화(customization)가 불가능하다.
- **Instruction Tuning**: Flamingo, BLIP-2, LLaVA 등은 이미지-텍스트 생성 과업에 지시어 튜닝을 적용했다. 하지만 이들은 주로 캡셔닝이나 VQA 같은 이미지-텍스트 과업에 집중하며, 객체 탐지(object detection)나 인스턴스 분할(instance segmentation)과 같은 정밀한 시각적 인지(visual perception) 과업은 다루지 않는다.

## 🛠️ Methodology

### 전체 아키텍처

VisionLLM은 **통합 언어 지시어 $\rightarrow$ 언어 가이드 이미지 토크나이저 $\rightarrow$ LLM 기반 개방형 태스크 디코더**로 이어지는 엔드-투-엔드 파이프라인을 가진다.

### 주요 구성 요소 및 작동 원리

**1. 통합 언어 지시어 (Unified Language Instruction)**
비전-언어 과업(Captioning, VQA)은 기존 NLP 방식과 유사하게 처리한다. 반면, 비전 전용 과업(Detection, Segmentation)은 다음과 같은 튜플 형식 $(C, P)$로 정의한다:

- $C$: 카테고리 셋 $\langle \text{class} \rangle$ 내의 클래스 인덱스.
- $P$: 객체의 위치를 나타내는 $N$개의 포인트 좌표 $\{x_i, y_i\}$.
- 좌표값은 $[-\langle \text{range} \rangle, \langle \text{range} \rangle]$ 범위 내의 정수로 이산화(discretize)되어 토큰화된다.

**2. 언어 가이드 이미지 토크나이저 (Language-Guided Image Tokenizer)**
이미지를 고정된 패치 임베딩으로 표현하는 대신, 언어 프롬프트에 반응하는 토큰을 생성한다:

- **특징 추출**: ResNet 등의 백본을 통해 다중 스케일 시각 특징 $F_v$를 추출하고, BERT를 통해 언어 특징 $F_l$을 추출한다.
- **정렬**: Cross-attention을 통해 시각 특징에 언어 정보를 주입하여 '언어 인식 시각 특징'을 생성한다.
- **토큰화**: Deformable DETR 기반의 트랜스포머 네트워크를 사용하여 $M$개의 랜덤 쿼리를 통해 최종적으로 $M$개의 이미지 토큰 $T = \{(e_i, l_i)\}$ (세만틱 및 위치 정보 포함)를 추출한다.

**3. LLM 기반 개방형 태스크 디코더 (LLM-based Open-Ended Task Decoder)**
Alpaca-7B를 기반으로 하며, 비전 과업 수행을 위해 다음과 같은 수정을 가했다:

- **어휘집 확장**:
  - **위치 토큰**: $\langle p_{-512} \rangle, \dots, \langle p_{512} \rangle$를 추가하여 연속적인 좌표 예측을 이산적인 빈(bin) 분류 문제로 변환하였다.
  - **분류 토큰**: $\langle c_0 \rangle, \dots, \langle c_{511} \rangle$를 추가하여 클래스 이름 대신 인덱스 토큰을 사용함으로써 효율성을 높였다.
- **Output-Format-as-Query**: 인과적(causal) 모델의 비효율적인 토큰-바이-토큰 생성을 피하기 위해, 지시어에서 파싱한 출력 형식(예: $\langle \text{cls} \rangle \langle x_1 \rangle \langle y_1 \rangle \dots$)을 쿼리로 직접 입력하여 결과를 생성한다.
- **LoRA (Low-Rank Adaptation)**: 효율적인 파라미터 튜닝을 위해 LoRA를 적용하여 시각-언어 토큰 간의 정렬을 돕고 수렴 속도를 향상시켰다.

### 학습 절차 및 손실 함수

전체 손실 함수는 다음과 같다:
$$L = L_{tok} + L_{dec}$$
여기서 $L_{tok}$는 이미지 토크나이저의 손실(Focal loss 및 $L_1$ loss)이며, $L_{dec}$는 디코더의 교차 엔트로피(cross-entropy) 손실이다.

학습은 2단계로 진행된다:

- **Stage 1**: 이미지 토크나이저와 LoRA 파라미터만 학습하며, 단순한 객체 탐지 과업을 통해 시각-언어 정렬에 집중한다.
- **Stage 2**: 시각 백본을 동결하고, 다양한 비전 중심 과업에 대해 통합 감독 학습을 수행한다.

## 📊 Results

### 실험 설정

- **데이터셋**: COCO2017 (Detection, Segmentation, Captioning), RefCOCO/+/g (Grounding), LLaVA-Instruct-150K (VQA).
- **백본**: ResNet-50 및 InternImage-H.
- **지표**: mAP, P@0.5, BLEU-4, CIDEr.

### 주요 결과

- **범용 성능**: InternImage-H 백본을 사용한 generalist 모델은 COCO 객체 탐지에서 $60.2\%$ mAP를 기록하며, 특정 과업 전용 모델들과 대등한 수준의 성능을 보였다.
- **과업별 성능 (ResNet-50 기준)**:
  - **Object Detection**: $44.6\%$ mAP.
  - **Visual Grounding**: $80.6\%$ P@0.5.
  - **Image Captioning**: BLEU-4 $31.0$, CIDEr $112.5$.
- **맞춤화 능력(Customization)**:
  - **객체 수준**: 클래스 수를 10개에서 80개로 변경해도 안정적인 성능을 유지하였다.
  - **출력 형식**: 인스턴스 분할 시 포인트 수를 8개에서 24개로 늘리면 마스크의 정교함이 증가하는 것을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 해석

VisionLLM은 LLM의 지시어 이해 능력을 비전 과업에 성공적으로 이식하였다. 특히 '출력 형식을 쿼리로 사용하는 방식'은 기존 Seq2Seq 방식보다 수렴이 빠르고 효율적이며, 다양한 비전 과업을 토큰 분류 문제로 통합하여 해결할 수 있음을 보여주었다. 또한, LoRA가 시각-언어 토큰 사이의 가교 역할을 하여 랜덤한 과업 정의 상황에서도 모델이 수렴할 수 있도록 돕는 필수적인 요소임을 입증하였다.

### 한계 및 비판적 논의

- **인스턴스 분할 성능 저하**: Mask R-CNN 같은 전용 모델에 비해 $\text{AP}_{75}$ 성능이 낮게 나타났다. 이는 좌표값을 정수로 이산화하는 과정에서 발생하는 정보 손실과, 메모리 제약으로 인해 포인트 수를 제한적으로 사용했기 때문으로 분석된다.
- **멀티태스크 충돌**: 단일 태스크로 학습한 모델($\text{VisionLLM-R50}_{\text{sep}}$)이 통합 모델보다 약간 높은 성능을 보였다. 이는 일반적인 범용 모델에서 나타나는 '정확도와 일반화 능력 간의 트레이드오프' 현상으로 해석된다.

## 📌 TL;DR

VisionLLM은 이미지를 외국어로 취급하고 LLM의 지시어 기반 제어 방식을 도입하여, 비전 중심 과업을 개방형으로 수행할 수 있게 한 프레임워크이다. 언어 가이드 이미지 토크나이저와 확장된 어휘집을 갖춘 LLM 디코더를 통해 객체 탐지, 분할, 캡셔닝 등을 하나의 모델로 통합하였으며, 특히 사용자가 지시어를 통해 탐지 대상이나 출력 형식을 자유롭게 지정할 수 있다는 점이 핵심이다. 이 연구는 향후 진정한 의미의 시각-언어 범용 지능 모델을 구축하는 데 중요한 기준점이 될 가능성이 높다.
