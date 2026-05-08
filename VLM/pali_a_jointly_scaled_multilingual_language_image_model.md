# PALI: A JOINTLY-SCALED MULTILINGUAL LANGUAGE-IMAGE MODEL

Xi Chen, Xiao Wang, Soravit Changpinyo, AJ Piergiovanni, Piotr Padlewski, Daniel Salz, Sebastian Goodman, Adam Grycner, Basil Mustafa, Lucas Beyer, Alexander Kolesnikov, Joan Puigcerver, Nan Ding, Keran Rong, Hassan Akbari, Gaurav Mishra, Linting Xue, Ashish Thapliyal, James Bradbury, Weicheng Kuo, Mojtaba Seyedhosseini, Chao Jia, Burcu Karagol Ayan, Carlos Riquelme, Andreas Steiner, Anelia Angelova, Xiaohua Zhai, Neil Houlsby, Radu Soricut

## 🧩 Problem to Solve

이 연구는 대규모 언어 모델(LLM)의 성공적인 스케일링 및 유연한 태스크 인터페이스 접근 방식을 언어와 비전의 **공동 모델링(joint modeling)**으로 확장하는 것을 목표로 합니다. 기존 비전 및 언어 모델은 각 모달리티의 컴포넌트 간 파라미터 불균형이 존재했으며, 특히 비전 백본의 용량이 언어 백본에 비해 상대적으로 작았습니다. 또한, **다국어 환경**에서 다양한 비전, 언어, 멀티모달 태스크를 효과적으로 처리할 수 있는 통합된 모델이 필요했습니다.

## ✨ Key Contributions

* **단순하고 모듈화된 확장 가능한 시퀀스-투-시퀀스 학습 아키텍처:** 기존의 트랜스포머 기반 단일 모달리티 체크포인트(mT5, ViT)를 재활용하여 효율적으로 훈련할 수 있도록 설계했습니다.
* **언어 및 비전 컴포넌트의 공동 스케일링 효과 입증:** 광범위한 파라미터에 걸쳐 두 컴포넌트를 함께 스케일링했을 때 성능 포화가 없음을 보였습니다. 특히, 기존 최대 ViT 모델을 넘어서 비전 컴포넌트를 스케일링(40억 파라미터 ViT-e 도입)하는 것이 멀티모달 성능에 큰 이점을 제공하며, 미래 V&L 모델 스케일링에 대한 로드맵을 제시합니다.
* **혼합 목적 함수(mixture-of-objectives)의 효과 검증:** 다양한 사전 훈련 태스크의 혼합이 대규모 V&L 모델의 성능에 긍정적인 영향을 미침을 경험적으로 입증했습니다.
* **대용량 다국어 멀티모달 모델 훈련 및 검증:** 100개 이상의 언어를 포함하는 대규모 사전 훈련 데이터셋을 구축하고, 이를 통해 적절히 스케일링된 모델이 다양한 언어를 잘 처리하면서도 영어 전용 태스크에서 최첨단(SOTA) 성능을 달성할 수 있음을 보여주었습니다.

## 📎 Related Works

* **언어 모델:** T5, GPT-3, Megatron-Turing, GLaM, Chinchilla, PaLM 등 대규모 텍스트 데이터에 대한 대규모 트랜스포머 훈련의 이점을 보여준 선행 연구.
* **비전 모델:** CNN, Vision Transformer (ViT), MLP-Mixer 등 스케일링 이점을 보여준 연구 (Zhai et al., 2022a).
* **언어-비전 모델:** SimVLM, Florence, CoCa, GIT, BEiT-3, Flamingo 등 언어-비전 모델링의 스케일링 트렌드를 따르는 연구들.
* **이미지-텍스트 사전 훈련 방식:**
  * **대조 학습(Contrastive Learning):** CLIP, ALIGN (Radford et al., 2021; Jia et al., 2021)과 같이 이미지와 텍스트 임베딩을 정렬하는 방식.
  * **텍스트 생성(Text Generation):** SimVLM, OFA, Unified-IO 등 모든 V&L 태스크를 텍스트 생성 문제로 통일하여 접근하는 방식 (Wang et al., 2021; Wang et al., 2022b; Lu et al., 2022).
* **대규모 이미지-텍스트 데이터셋:** Conceptual Captions (CC3M, CC12M), LEMON, ALIGN 등 대규모 데이터셋 구축 및 활용 연구. 기존 데이터셋의 대부분이 영어 전용이었다는 점이 PaLI의 WebLI 데이터셋 개발 동기.

## 🛠️ Methodology

PaLI는 단일 "이미지-및-텍스트를 텍스트로(image-and-text to text)" 인터페이스를 사용하여 이미지 전용, 언어 전용, 이미지+언어 태스크를 여러 언어로 수행합니다.

1. **아키텍처:**
   * 핵심적으로 **인코더-디코더 트랜스포머(mT5)**를 사용하며, 입력 텍스트 인코더는 ViT의 출력 패치 특징인 시각 토큰(visual tokens)을 순서대로 받습니다.
   * **사전 훈련된 단일 모달리티 백본 재활용:** 언어 컴포넌트에는 사전 훈련된 mT5 모델 (mT5-Large 10억 파라미터, mT5-XXL 130억 파라미터)을, 비전 컴포넌트에는 대규모 ViT 모델 (ViT-G 18억 파라미터)을 재활용합니다.
   * **새로운 ViT-e 도입:** 40억 파라미터의 "enormous" ViT-e 모델을 새로 훈련하여 비전 백본의 용량을 대폭 확장했습니다. PaLI-17B 모델은 mT5-XXL과 ViT-e를 결합하여 약 170억 파라미터를 가집니다.
   * **태스크 인터페이스:** 모든 태스크를 "이미지+쿼리 → 답변(image+query to answer)" 형식으로 통일하여, 쿼리와 답변 모두 텍스트 토큰으로 표현됩니다. 텍스트 기반 프롬프트를 사용하여 모델에 어떤 태스크를 수행할지 지시합니다.
2. **데이터:**
   * **WebLI 데이터셋:** 100억 개의 이미지와 120억 개의 alt-text, 290억 개의 이미지-OCR 쌍을 포함하는 새로운 대규모 다국어 이미지-언어 데이터셋을 구축했습니다. 100개 이상의 언어를 지원합니다. PaLI 훈련에는 품질 상위 10% (약 10억 개 예제)를 사용합니다.
   * **사전 훈련 태스크 혼합:** PaLI는 8가지 사전 훈련 태스크의 혼합을 사용하여 다양한 능력을 학습합니다.
     * 텍스트 전용 데이터의 스팬 손상(Span corruption).
     * WebLI alt-text 데이터의 분할 캡셔닝(Split-captioning).
     * CC3M-35L의 캡셔닝(Captioning).
     * WebLI OCR 텍스트 데이터의 OCR.
     * VQ$ {^2} $A-CC3M의 영어 및 교차 언어 VQA.
     * VQ$ {^2} $A-CC3M의 영어 및 교차 언어 시각 질문 생성(VQG).
     * 객체 지향(Object-Aware) VQA (Open Images 기반).
     * 객체 탐지(Object detection).
3. **모델 훈련:**
   * 모든 PaLI 모델은 사전 훈련 데이터셋(16억 개 예제)에 대해 224x224 이미지 해상도로 1 epoch 훈련됩니다. 이 단계에서는 비전 컴포넌트는 고정하고 언어 컴포넌트의 파라미터만 업데이트됩니다.
   * 가장 큰 모델인 PaLI-17B의 경우, 588x588의 고해상도 사전 미세 조정 단계를 10k 스텝 동안 추가로 수행하며, 이때 모든 PaLI 파라미터가 업데이트됩니다.

## 📊 Results

PaLI는 여러 비전 및 언어 벤치마크에서 SOTA 성능을 달성했습니다.

* **이미지 캡셔닝:**
  * COCO Captions (Karpathy split): 149.1 CIDEr 점수 (SOTA 달성, CIDEr 최적화 없이).
  * NoCaps: 124.4 CIDEr 점수, 장기 객체 인식 및 설명에서 이전 모델 능가.
  * TextCaps, VizWiz-Cap: OCR 문자열을 입력으로 사용하여 높은 성능 달성.
  * Crossmodal-3600 (다국어 캡셔닝): 35개 언어 평균 CIDEr에서 이전 SOTA를 큰 폭으로 능가 (예: 영어 98.1, 프랑스어 75.5, 태국어 72.1).
* **시각 질문 답변 (VQA):**
  * VQAv2: 개방형 어휘 생성 설정(open-vocabulary generation setting)에서 84.3% 정확도 (SOTA 달성, 심지어 고정 어휘 분류 설정 모델보다 우수).
  * OKVQA: 64.5% 정확도 (외부 지식 필요). 이전 SOTA보다 10.1%p 향상.
  * TextVQA, VizWiz-QA, ST-VQA: 이미지 내 텍스트 이해 능력 요구 태스크에서 높은 성능 달성 (OCR 문자열 활용).
  * xGQA 및 MaXM (교차 언어 및 다국어 VQA): 13개 언어에서 모두 상당한 성능 향상.
* **언어 이해 능력:** SuperGLUE, XTREME (XNLI, XQuAD, TyDiQA-GoldP) 벤치마크에서 mT5-XXL과 동등한 수준의 높은 언어 이해 능력 유지.
* **제로샷 이미지 분류:**
  * ImageNet 및 OOD (ImageNet-R, -A, -Sketch, -v2, ObjectNet) 데이터셋에서 강력한 제로샷 분류 성능 (PaLI-17B는 ImageNet에서 72.11% Top-1 정확도).
  * LiT ViT-e는 ObjectNet에서 84.9% 제로샷 정확도로 SOTA 달성.

## 🧠 Insights & Discussion

* **스케일링의 중요성:** 언어 모델과 비전 모델 컴포넌트 모두를 공동으로 스케일링하는 것이 성능 향상에 필수적임을 입증했습니다. 특히, 비전 컴포넌트(ViT-G에서 ViT-e로 스케일링)는 전체 모델 크기 증가 대비 더 높은 성능 향상(파라미터/FLOP당 정확도)을 보여, 비전 모델의 추가적인 스케일링 잠재력을 시사합니다.
* **다국어 능력:** WebLI와 같은 다국어 데이터셋으로 모델을 훈련함으로써, 모델은 100개 이상의 언어를 효과적으로 처리할 수 있게 되며, 이는 영어 전용 태스크의 성능에도 긍정적인 영향을 미칠 수 있음을 보였습니다.
* **제한 사항:**
  * 복잡한 장면에서 많은 객체를 상세하게 묘사하는 데 한계가 있을 수 있습니다.
  * 영어 전용 데이터로 미세 조정 시 다국어 능력이 일부 손실될 수 있습니다.
  * 개방형 어휘 생성 설정에서의 평가는 동의어나 다른 표현을 "오답"으로 처리할 수 있어 실제 능력보다 낮게 평가될 수 있습니다.
  * 평가 벤치마크가 서구 중심적 편향을 가질 수 있습니다.

## 📌 TL;DR

* PaLI는 대규모 사전 훈련된 언어 및 비전 모델 컴포넌트를 **균형 있게 공동 스케일링**하고 **WebLI 다국어 데이터셋**으로 훈련된 멀티모달 언어-이미지 모델입니다.
* "이미지+텍스트 쿼리 → 텍스트 답변"의 **단일 생성 인터페이스**를 통해 이미지 캡셔닝, VQA, 장면 텍스트 이해 등 다양한 비전, 언어, 멀티모달 태스크 및 다국어 태스크를 처리합니다.
* 특히 40억 파라미터 **ViT-e 비전 백본**을 도입하여 비전 스케일링의 중요성을 강조하며, COCO Captions, VQAv2 등 여러 벤치마크에서 **SOTA 성능**을 달성하고 뛰어난 다국어 이해 능력을 보여줍니다.
