# Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond

Jinze Bai, Shuai Bai, Shusheng Yang, et al. (2023)

## 🧩 Problem to Solve

최근 Large Language Models (LLMs)는 텍스트 생성과 이해 능력에서 비약적인 발전을 이루었으나, 기본적으로 텍스트 데이터만을 처리할 수 있다는 한계가 있다. 이를 해결하기 위해 이미지나 비디오와 같은 시각적 신호를 이해하는 Large Vision Language Models (LVLMs) 연구가 활발히 진행되고 있다.

그러나 기존의 오픈 소스 LVLMs는 다음과 같은 두 가지 주요 문제점을 가지고 있다. 첫째, 학습 및 최적화의 부족으로 인해 폐쇄형(proprietary) 모델에 비해 성능이 크게 뒤처진다. 둘째, 대부분의 모델이 이미지를 거시적인(coarse-grained) 관점에서만 인식하며, 객체 Localization(Grounding)이나 텍스트 읽기(OCR)와 같은 세밀한(fine-grained) 시각적 이해 능력이 부족하다.

본 논문의 목표는 일반적인 이미지 이해뿐만 아니라, 정밀한 객체 위치 지정과 텍스트 읽기 능력을 모두 갖춘 다재다능한 시각-언어 기반 모델인 Qwen-VL 시리즈를 개발하는 것이다.

## ✨ Key Contributions

Qwen-VL의 핵심 아이디어는 Qwen-7B라는 강력한 언어 모델을 기반으로, 정교하게 설계된 Visual Receptor와 3단계 학습 파이프라인을 통해 시각적 인지 능력을 부여하는 것이다.

가장 중심적인 설계 특징은 고해상도 입력 지원과 Position-aware Vision-Language Adapter를 도입하여 시각적 특징의 손실을 최소화하면서 LLM이 처리 가능한 효율적인 길이로 압축하는 것이다. 또한, 이미지-캡션-박스 튜플(tuple) 형태의 데이터를 학습시켜 단순한 설명을 넘어 이미지 내 특정 영역을 정확히 짚어내고 읽어낼 수 있는 Grounding 및 Text-reading 능력을 구현하였다.

## 📎 Related Works

기존의 시각-언어 모델 연구는 크게 두 가지 방향으로 진행되었다. 하나는 CoCa나 OFA와 같이 여러 시각-언어 작업을 통합된 프레임워크에서 처리하는 Generalist 모델이며, 다른 하나는 CLIP나 BEIT-3와 같이 강력한 시각-언어 표현(representation)을 학습하는 모델이다.

최근에는 LLM을 기반으로 한 LVLMs(예: BLIP-2, LLaVA, Mini-GPT4)가 등장하여 지시어 이행(instruction following) 능력을 강화하는 추세이다. 일부 모델(Kosmos-2, Shikra)은 시각적 Grounding 능력을 추가하려는 시도를 하였으나, 여전히 많은 오픈 소스 모델들이 고해상도 이미지 처리와 세밀한 텍스트 인식 능력에서 한계를 보인다. Qwen-VL은 이러한 한계를 극복하기 위해 다국어 말뭉치와 다단계 학습 전략을 사용하여 일반적인 이해도와 세밀한 인지 능력을 동시에 확보하였다.

## 🛠️ Methodology

### 1. 모델 아키텍처 (Model Architecture)

Qwen-VL은 크게 세 가지 구성 요소로 이루어져 있다.

- **Visual Encoder**: OpenCLIP의 ViT-bigG를 사용하며, 입력 이미지를 패치 단위로 분할하여 시각적 특징(image features)을 추출한다.
- **Position-aware Vision-Language Adapter**: ViT에서 추출된 방대한 양의 시각적 특징 시퀀스를 LLM이 효율적으로 처리할 수 있도록 고정된 길이인 256로 압축하는 역할을 한다. 단일 레이어의 Cross-attention 모듈을 사용하며, 학습 가능한 쿼리 벡터(Learnable Query Embeddings)를 쿼리로, ViT의 특징을 키(key)로 사용한다. 이때 위치 정보 손실을 막기 위해 2D Absolute Positional Encodings를 추가한다.
- **Large Language Model (LLM)**: Qwen-7B를 기반으로 하며, 어댑터를 통해 압축된 시각적 특징 시퀀스를 입력받아 텍스트를 생성한다.

### 2. 입력 및 출력 인터페이스 (Inputs and Outputs)

- **이미지 입력**: 이미지 특징의 시작과 끝을 알리는 특수 토큰 `<img>`와 `</img>`를 사용한다.
- **Bounding Box**: 세밀한 인지를 위해 좌표 값을 $[0, 1000)$ 범위로 정규화한 후, `(Xtopleft, Ytopleft), (Xbottomright, Ybottomright)` 형태의 문자열로 변환한다.
- **특수 토큰**:
  - `<box>` 및 `</box>`: 좌표 문자열의 시작과 끝을 구분한다.
  - `<ref>` 및 `</ref>`: 해당 좌표가 가리키는 대상이 되는 단어나 문장을 표시한다.

### 3. 학습 절차 (Training Pipeline)

학습은 총 3단계로 진행된다.

**단계 1: Pre-training (사전 학습)**

- **목표**: 대규모 이미지-텍스트 쌍을 통해 기본적인 시각-언어 정렬을 학습한다.
- **데이터**: 14억 개의 정제된 이미지-텍스트 쌍을 사용한다.
- **방법**: LLM은 동결(freeze)하고, Vision Encoder와 VL Adapter만 최적화한다. 입력 해상도는 $224 \times 224$이다. 손실 함수로는 텍스트 토큰의 Cross-entropy를 최소화하는 방식을 사용한다.

**단계 2: Multi-task Pre-training (다중 작업 사전 학습)**

- **목표**: 세밀한 인지 능력과 고해상도 처리 능력을 확보한다.
- **데이터**: Captioning, VQA, Grounding, OCR 등 7가지 작업을 동시에 학습한다.
- **방법**: 입력 해상도를 $448 \times 448$로 높이고, LLM을 포함한 전체 모델의 가중치를 업데이트한다.

**단계 3: Supervised Fine-tuning (SFT, 지도 미세 조정)**

- **목표**: 사용자의 의도를 이해하고 대화할 수 있는 Qwen-VL-Chat 모델을 구축한다.
- **데이터**: 멀티모달 지시어 데이터 및 사람이 직접 구축한 대화 데이터를 사용한다.
- **방법**: Vision Encoder는 동결하고, LLM과 Adapter 모듈만 최적화한다.

## 📊 Results

### 1. 일반 시각 이해 및 캡셔닝

Qwen-VL은 Image Captioning(Flickr30K)과 General VQA(VQAv2, OKVQA, GQA)에서 기존 Generalist 모델들을 크게 상회하는 성능을 보였다. 특히 Flickr30K에서 85.8 CIDEr 점수를 기록하며, 파라미터 수가 훨씬 많은 Flamingo-80B보다 우수한 성능을 입증하였다.

### 2. 텍스트 중심 시각 질의응답 (Text-oriented VQA)

TextVQA, DocVQA, ChartQA 등 텍스트 읽기 능력이 필수적인 벤치마크에서 압도적인 성능을 기록하였다. 이는 고해상도 입력과 OCR 특화 데이터 학습의 결과로 분석된다.

### 3. Referring Expression Comprehension (REC)

특정 묘사를 바탕으로 객체의 위치를 찾는 REC 작업(RefCOCO, GRIT 등)에서도 최상위권(top-tier) 성적을 거두어, 세밀한 Localization 능력을 증명하였다.

### 4. 실세계 사용자 행동 및 지시어 이행 (Instruction Following)

TouchStone, SEED-Bench, MME 벤치마크 결과, Qwen-VL-Chat은 다른 LVLMs보다 뛰어난 성능을 보였으며, 특히 한국어와 중국어를 포함한 다국어 능력과 차트 분석, 텍스트 인식 분야에서 강점을 보였다.

## 🧠 Insights & Discussion

### 1. 모델 설계의 유효성

본 논문은 Vision-Language Adapter의 쿼리 개수에 대한 Ablation Study를 통해, 쿼리 수가 너무 적으면 정보 손실이 발생하고 너무 많으면 수렴 속도가 느려짐을 확인하여 256이라는 최적의 수치를 도출하였다. 또한, ViT에서 Window Attention보다 Global Attention을 사용했을 때 수렴 성능이 더 좋음을 확인하여 Vanilla Attention을 채택하였다.

### 2. 텍스트 능력 유지 (Preventing Catastrophic Forgetting)

멀티모달 학습이 LLM 본연의 텍스트 처리 능력을 저하시키지 않는지 확인하기 위해 MMLU, C-Eval 등의 벤치마크를 수행하였다. 학습 과정에서 순수 텍스트 데이터를 함께 학습시킨 결과, 텍스트 성능의 저하 없이 오히려 일부 지표에서 향상된 결과를 보였다.

### 3. 한계 및 향후 과제

현재 모델은 정지 이미지 위주로 학습되었으나, 향후 오디오 및 비디오와 같은 더 다양한 모달리티를 통합할 계획이다. 또한, 더 높은 해상도와 더 큰 모델 규모를 통해 복잡한 시각적 관계를 더 정밀하게 파악하고자 한다.

## 📌 TL;DR

Qwen-VL은 Qwen-7B LLM을 기반으로 ViT-bigG 인코더와 Position-aware Adapter를 결합하여, **일반적인 이미지 이해, 정밀한 객체 Localization(Grounding), 그리고 고성능 OCR 능력**을 동시에 갖춘 다재다능한 LVLM이다. 3단계의 체계적인 학습 파이프라인과 다국어 데이터셋을 통해 오픈 소스 모델임에도 불구하고 폐쇄형 모델에 근접하는 성능을 달성하였으며, 특히 세밀한 시각 인지 작업에서 탁월한 능력을 보여 향후 멀티모달 AI 에이전트 연구에 중요한 기여를 할 것으로 평가된다.
