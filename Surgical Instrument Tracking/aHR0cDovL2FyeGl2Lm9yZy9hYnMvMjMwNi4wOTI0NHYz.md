# Text Promptable Surgical Instrument Segmentation with Vision-Language Models

Zijian Zhou, Oluwatosin Alabi, Meng Wei, Tom Vercauteren, Miaojing Shi (2023)

## 🧩 Problem to Solve

본 논문은 최소 침습 수술(Minimally Invasive Surgeries, MIS) 환경에서 발생하는 수술 도구 분할(Surgical Instrument Segmentation)의 한계를 해결하고자 한다. 구체적으로 다음과 같은 세 가지 주요 문제를 다룬다.

첫째, 수술 도구의 다양성과 데이터 부족 문제이다. 수술 도구의 종류가 빠르게 증가하고 있으나, 이를 학습시키기 위한 대규모 데이터셋이 부족하다. 기존의 지도 학습 기반 모델들은 미리 정의된 카테고리(Predefined categories)에 의존하므로, 새로운 도구가 도입될 때마다 데이터를 다시 라벨링하고 모델을 재학습시켜야 하는 비효율성이 존재한다.

둘째, 도구 간의 유사성으로 인한 구분 어려움이다. 서로 다른 카테고리의 도구들이 시각적으로 매우 유사한 외형을 가지고 있으며, 수술 cavity 내부의 열악한 조명 및 촬영 조건으로 인해 모델이 도구들을 정확히 구별해내는 데 어려움을 겪는다.

따라서 본 논문의 목표는 Vision-Language Model(VLM)을 활용하여 텍스트 프롬프트(Text Prompt)를 통해 도구를 분할하는 방식을 제안함으로써, 새로운 도구 유형에 대한 적응력을 높이고 분할 정밀도를 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 수술 도구 분할 작업을 '텍스트 프롬프트 가능(Text Promptable)'한 작업으로 재정의하는 것이다. 이를 통해 고정된 클래스 분류에서 벗어나 텍스트 설명을 통해 도구를 식별함으로써 유연성과 일반화 성능을 확보한다. 주요 기여 사항은 다음과 같다.

1. **Text Promptable Mask Decoder 설계**: CLIP의 이미지 및 텍스트 인코더를 백본으로 사용하며, Attention 기반의 글로벌 디코딩과 Convolution 기반의 로컬 디코딩을 순차적으로 결합하여 정교한 마스크를 생성한다.
2. **Mixture of Prompts (MoP) 메커니즘**: 단일 프롬프트의 한계를 극복하기 위해 클래스 명칭, 템플릿, GPT-4 생성 설명 등 다양한 프롬프트를 동시에 입력하고, 시각-텍스트 게이팅 네트워크(Visual-textual Gating Network)를 통해 픽셀 단위로 가중치를 조절하여 결과를 융합한다.
3. **Hard Instrument Area Reinforcement (HIAR) 모듈**: Masked Autoencoder(MAE) 구조를 응용하여, 분할 오류가 자주 발생하는 '어려운 영역(Hard area)'을 집중적으로 마스킹하고 이를 재구성(Reconstruction)하게 함으로써 이미지 특징 추출 능력을 강화한다.

## 📎 Related Works

기존의 수술 도구 분할 연구는 주로 TernausNet(U-Net 기반), ISINet, MF-TapNet과 같은 전통적인 비전 기반 모델에 의존해 왔다. 최근에는 MATIS와 같이 Transformer 기반의 방법론이 등장하여 성능을 높였으나, 여전히 학습 데이터에 포함된 특정 카테고리에만 국한되어 동작한다는 한계가 있다. 즉, 새로운 도구가 추가되면 모델을 처음부터 다시 학습시켜야 한다.

반면, CLIP과 같은 대규모 사전 학습 Vision-Language Model은 이미지와 텍스트를 동일한 임베딩 공간에 정렬시켜 Zero-shot 성능을 보여주었다. CRIS나 CLIPSeg 같은 모델들이 이를 일반적인 이미지 분할에 적용했으나, 수술 도구와 같이 매우 특수한 도메인에 특화된 설계는 부족했다. 본 논문은 이러한 VLM의 강점을 수술 도메인에 최적화하여 적용했다는 점에서 기존 연구와 차별화된다.

## 🛠️ Methodology

### 1. 전체 파이프라인

전체 시스템은 $\text{Input Image } I$와 $\text{Text Description } T$를 입력받아 수술 도구의 마스크 $M$을 생성한다. 구조는 크게 이미지/텍스트 인코더, 프롬프트 가능 마스크 디코더, MoP 모듈, HIAR 모듈의 네 가지 단계로 구성된다.

### 2. Image and Text Encoders

- **Image Encoder**: CLIP의 ViT-B-16을 사용하며, 다양한 스케일의 특징을 추출하기 위해 **Multi-scale Feature Augmentation (MSFA)**를 적용한다. ViT의 4번째, 8번째, 12번째 레이어의 출력을 추출하고 Feature Pyramid Network (FPN)를 통해 융합하여 다중 스케일 특징 $F^I \in \mathbb{R}^{N \times D}$를 얻는다.
- **Text Encoder**: CLIP의 Transformer 인코더를 사용하며, 가중치는 고정(Frozen) 상태로 유지한다. 텍스트의 전체적인 정보를 대표하는 $[CLS]$ 토큰의 특징 $F^T \in \mathbb{R}^{1 \times D}$를 추출한다.

### 3. Text Promptable Mask Decoder

텍스트 특징 $F^T$를 이용하여 이미지 특징 $F^I$로부터 스코어 맵 $S$를 생성한다.

- **Attention-based Prompting**: Self-Attention(SA)으로 전경 영역을 강조하고, Cross-Attention(CA)을 통해 텍스트 $F^T$에 해당하는 도구 영역을 전역적으로 로컬라이즈한다.
- **Convolution-based Prompting**: 전역적으로 추출된 특징을 정교화하기 위해, $F^T$를 Fully Connected(FC) 레이어에 통과시켜 컨볼루션 커널 가중치 $w$와 바이어스 $b$로 변환한다. 이를 통해 이미지 특징에 로컬 컨볼루션을 수행하여 정밀한 경계를 찾아낸다.
$$S = \text{Sigmoid}(\text{Conv}(\tilde{F}^I_A | w, b))$$

### 4. Mixture of Prompts (MoP)

다양한 텍스트 설명(단순 클래스명, 템플릿, GPT-4 생성 문장)을 사용하여 각각의 스코어 맵 $S_i$를 생성한다. 이후 **Visual-textual Gating Network $G$**가 이미지와 텍스트 특징을 입력받아 각 픽셀별 가중치 맵을 생성하며, 이를 통해 최종 스코어 맵 $S_{set}$을 도출한다.

### 5. Hard Instrument Area Reinforcement (HIAR)

분할이 어려운 영역의 표현력을 높이기 위해 MAE 구조를 도입한다.

- **Hard Area Mining**: 예측 마스크 $M_{et}$와 정답 마스크 $M_{gt}$를 비교하여 오차가 큰 영역을 'Hard area'로 정의하고 마스킹한다.
- **Reinforcement**: 마스킹된 이미지를 인코더에 넣고 다시 원래 이미지를 복구하도록 학습함으로써, 모델이 어려운 영역의 세부 특징을 더 잘 이해하게 만든다.

### 6. 학습 절차 및 손실 함수

텍스트 인코더는 고정하고 이미지 인코더만 파인튜닝한다. 손실 함수는 분할 손실 $L_{seg}$ (Binary Cross Entropy)와 재구성 손실 $L_{rec}$ (L2 Loss)의 합으로 정의된다.
$$L = L_{seg} + \lambda L_{rec}$$
여기서 $\lambda$는 손실 가중치이다.

## 📊 Results

### 실험 설정

- **데이터셋**: EndoVis2017, EndoVis2018, EndoVis2019, Cholecseg8k.
- **평가 지표**: Ch_IoU, ISI_IoU, mc_IoU (IoU 기반 지표), DSC, NSD.
- **비교 대상**: 기존 지도 학습 모델(ISINet, MATIS, S3Net 등) 및 프롬프트 기반 모델(CRIS, CLIPSeg), SAM 변형 모델.

### 주요 결과

- **정량적 성과**: EndoVis2017 데이터셋에서 Ch_IoU 기준 기존 SOTA 대비 약 7.36% 향상된 성능을 보였다. 특히- a smaller difference between Ch_IoU and ISI_IoU indicates fewer misclassifications.
- **일반화 능력**: EndoVis2018에서 학습하고 2017에서 테스트하는 교차 데이터셋(Cross-dataset) 실험에서도 경쟁력 있는 성능을 유지하여, 새로운 도구에 대한 적응력이 높음을 입증하였다.
- **SAM과의 비교**: 텍스트 프롬프트가 가능한 SAM 변형 모델(lang-segment-anything)보다 월등히 높은 성능을 보였는데, 이는 SAM이 의료 도메인의 특수한 개념에 대해 파인튜닝되지 않았기 때문으로 분석된다.
- **효율성**: FLOPs 및 FPS 분석 결과, 실시간 임상 적용이 가능한 수준의 추론 속도를 달성하였다 (EndoVis2017 기준 약 22 FPS).

## 🧠 Insights & Discussion

### 강점

본 연구는 단순한 클래스 분류를 넘어 '텍스트 프롬프트'라는 유연한 인터페이스를 도입함으로써, 의료 현장에서 새로운 수술 도구가 도입될 때마다 모델을 재학습시켜야 하는 고질적인 문제를 해결할 수 있는 가능성을 제시하였다. 특히 GPT-4를 이용한 상세 묘사 프롬프트가 성능 향상에 기여했다는 점은 도메인 지식을 텍스트 형태로 주입하는 것이 효과적임을 시사한다.

### 한계 및 비판적 해석

실험 결과 분석 중, 모델이 특히 어려워하는 영역이 도구의 **Clasper(집게 부분)**와 **Shaft(자루 부분)**임이 밝혀졌다. Clasper는 조직과 밀접하게 상호작용하여 배경과 구분이 어렵고, Shaft는 서로 다른 도구들이 매우 유사한 외형을 가지기 때문이다.
저자들은 외과의의 관점에서는 Shaft가 어렵지 않지만, 모델은 Clasper와 Shaft의 관계를 유추하지 못해 오류가 발생한다고 분석한다. 이는 단순한 픽셀 수준의 특징 추출을 넘어, 도구의 구조적 관계(Structural relationship)를 모델링하는 추가적인 연구가 필요함을 의미한다.

## 📌 TL;DR

본 논문은 CLIP과 같은 Vision-Language Model을 수술 도구 분할에 도입하여, 텍스트 프롬프트만으로 도구를 식별하고 분할하는 **Text Promptable Surgical Instrument Segmentation** 방식을 제안한다. Multi-scale 특징 추출, 다중 프롬프트 융합(MoP), 그리고 어려운 영역을 집중 학습하는 HIAR 모듈을 통해 기존 SOTA 모델들을 능가하는 성능과 우수한 일반화 능력을 보여주었다. 이 연구는 향후 새로운 수술 도구에 즉각적으로 대응해야 하는 로봇 보조 수술 시스템의 자동화에 중요한 기여를 할 것으로 기대된다.
