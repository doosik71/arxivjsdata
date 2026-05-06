# Enhance Image-to-Image Generation with LLaVA-generated Prompts

Zhicheng Ding, Panfeng Li, Qikai Yang, Siyang Li (2024)

## 🧩 Problem to Solve

본 논문은 Image-to-Image (Img2Img) 생성 과정에서 발생하는 제어력 부족과 충실도(fidelity) 저하 문제를 해결하고자 한다. 일반적으로 Stable Diffusion과 같은 모델을 사용하여 이미지를 생성할 때, 입력 이미지에만 의존할 경우 생성된 결과물이 사용자의 의도에서 크게 벗어나거나 원본 이미지의 핵심 특징을 제대로 유지하지 못하는 한계가 존재한다. 또한, 대규모 언어 모델(LLM) 기반의 이미지 생성에서 나타나는 부정확성과 불안정성 역시 해결해야 할 주요 과제이다. 따라서 본 연구의 목표는 입력 이미지의 시각적 정보를 정확하게 텍스트로 변환하여 생성 파이프라인에 제공함으로써, 원본 이미지와 생성 결과물 사이의 시각적 유사성과 일관성을 높이는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 멀티모달 능력을 갖춘 LLaVA (Large Language and Vision Assistant)를 이미지 해석기로 활용하여, 이미지 생성 모델을 위한 최적의 Prompt를 자동으로 생성하는 프레임워크를 구축하는 것이다. 단순히 이미지를 입력하는 대신, LLaVA가 이미지의 핵심 특징과 개념을 분석하여 텍스트 형태의 Positive Prompt와 Negative Prompt를 생성하게 하고, 이를 Stable Diffusion의 Img2Img 파이프라인에 함께 입력함으로써 생성 프로세스를 정교하게 가이드한다. 즉, '시각적 이해(LLaVA) $\rightarrow$ 텍스트 묘사(Prompt) $\rightarrow$ 이미지 생성(Stable Diffusion)'으로 이어지는 연결 고리를 통해 제어력을 강화한 것이 핵심이다.

## 📎 Related Works

논문에서는 Stable Diffusion과 같은 Latent Diffusion Model과 LLaVA와 같은 멀티모달 LLM의 발전상을 언급하고 있다. 기존의 Image-to-Image 접근 방식은 주로 입력 이미지의 픽셀 정보나 잠재 공간(latent space)의 특징에 의존하였으나, 이는 텍스트 기반의 세밀한 제어가 어렵다는 한계가 있었다. 본 논문은 이러한 한계를 극복하기 위해 LLaVA의 이미지 이해 능력을 결합함으로써, 기존의 단순 이미지 기반 생성 방식보다 더 높은 수준의 충실도와 제어 가능성을 제공한다는 점에서 차별점을 갖는다.

## 🛠️ Methodology

### 전체 시스템 구조

제안된 프레임워크는 다음과 같은 단계적 파이프라인으로 구성된다.

1. **Input Image Analysis**: 입력 이미지를 LLaVA 모델에 제공한다.
2. **Prompt Generation**: LLaVA에게 "Generate prompt and negative prompt for this image"라는 지시어(Instruction)를 전달하여, 이미지의 내용을 상세히 묘사하는 긍정 프롬프트와 생성에서 제외해야 할 요소를 정의하는 부정 프롬프트를 생성한다.
3. **Image-to-Image Generation**: 생성된 텍스트 프롬프트들과 원본 이미지를 함께 Stable Diffusion의 `Img2ImgPipeline`에 입력하여 최종 이미지를 생성한다.

### 주요 구성 요소 및 절차

- **LLaVA (v1.6-34b)**: 이미지의 전체적인 분위기, 배경 세부 사항, 제외 요소 등을 분석하여 텍스트로 변환하는 역할을 수행한다.
- **Stable Diffusion v2**: LLaVA가 생성한 프롬프트를 가이드라인으로 삼아, 원본 이미지의 구조를 유지하면서도 프롬프트에 명시된 세부 사항을 반영한 새로운 이미지를 생성한다.
- **Diffusion-Denoising Mechanism**: Stable Diffusion 2.0의 디노이징 메커니즘을 통해 이미지 편집 과정에 대한 미세한 제어를 수행한다.

### 특이사항

논문 내에서 SVM(Support Vector Machine)을 언급하며 모델의 성능을 높이고 제약 조건을 해결하는 데 도움이 되었다고 서술하고 있으나, 구체적으로 SVM이 파이프라인의 어느 단계에서 어떻게 적용되었는지에 대한 상세한 방정식이나 알고리즘 설명은 명시되지 않았다.

## 📊 Results

### 실험 설정

- **데이터셋 및 시나리오**: 다양한 시나리오(개, 우주비행사, 비행기, 마천루 등)의 입력 이미지를 사용하였다.
- **비교 대상**: LLaVA 생성 프롬프트를 사용한 경우(with prompts)와 사용하지 않은 경우(w/o prompt)를 비교하였다.
- **평가 지표**: 이미지 유사도를 측정하기 위해 다음의 정량적 지표를 사용하였다.
  - $\text{RMSE}$ (Root Mean Square Error) $\downarrow$
  - $\text{PSNR}$ (Peak Signal-to-Noise Ratio) $\uparrow$
  - $\text{FSIM}$ (Feature-based Similarity Index) $\uparrow$
  - $\text{SSIM}$ (Structural Similarity Index) $\uparrow$
  - $\text{UIQ}$ (Universal Image Quality Index) $\uparrow$
  - $\text{SRE}$ (Signal to Reconstruction Error Ratio) $\uparrow$

### 정량적 결과

Table I과 Table II의 결과에 따르면, 모든 시나리오에서 LLaVA 프롬프트를 사용했을 때 유사도 지표가 크게 향상되었다. 예를 들어, 전반적인 지표 비교(Table I)에서 $\text{RMSE}$는 $0.0193$에서 $0.0100$으로 감소하였고, $\text{SSIM}$은 $0.7850$에서 $0.9219$로 크게 상승하였다. 개별 객체(개, 우주비행사, 비행기 등)에 대한 실험에서도 프롬프트를 적용한 경우 $\text{PSNR}$과 $\text{SSIM}$ 값이 일관되게 높게 나타났다.

### 정성적 결과

프롬프트 없이 생성된 이미지는 원본과 테마는 유사하나, 도로의 차선 수가 변하거나 배경 산의 가시성이 떨어지는 등의 노이즈 및 왜곡이 발생하였다. 반면, LLaVA 프롬프트를 적용한 결과물은 원본 이미지의 구조적 특징을 더 잘 유지하면서도 시각적 일관성이 높은 이미지가 생성됨을 확인하였다.

## 🧠 Insights & Discussion

### 강점

본 연구는 별도의 추가 학습 없이 기존의 강력한 모델인 LLaVA와 Stable Diffusion을 효율적으로 결합하여 Image-to-Image 생성의 품질을 높였다는 점이 강점이다. 특히, 텍스트 프롬프트가 이미지 생성 과정에서 일종의 '앵커(Anchor)' 역할을 하여, 확산 모델이 가질 수 있는 무작위성을 억제하고 원본 이미지에 대한 충실도를 높일 수 있음을 입증하였다.

### 한계 및 비판적 해석

1. **Negative Prompt의 정확성**: 저자 스스로 언급했듯이, LLaVA가 생성하는 부정 프롬프트에 때때로 오도하는(misleading) 요소가 포함될 수 있으며 이에 대한 정밀한 검증이 부족하다.
2. **방법론의 구체성 결여**: SVM을 사용했다는 언급이 있으나, 그것이 구체적으로 어떤 수식이나 모듈로 구현되었는지 설명되지 않아 재현 가능성이 떨어진다.
3. **창의성과의 트레이드-오프**: 유사도를 높이는 데 집중했기 때문에, 생성 모델의 본래 목적인 '창의적인 변형'보다는 '원본 복제'에 가까운 결과가 나올 가능성이 있다.

## 📌 TL;DR

본 논문은 LLaVA의 멀티모달 이해 능력을 활용해 입력 이미지로부터 최적의 긍정/부정 프롬프트를 자동 생성하고, 이를 Stable Diffusion의 Img2Img 파이프라인에 주입하여 결과물의 시각적 유사성과 충실도를 높이는 프레임워크를 제안한다. 실험을 통해 프롬프트 기반 생성 방식이 기존의 이미지 전용 방식보다 $\text{SSIM}, \text{PSNR}$ 등 주요 유사도 지표에서 우수함을 증명하였다. 이 연구는 향후 AI 기반의 정밀한 이미지 편집 및 제어 가능한 콘텐츠 생성 분야에 기여할 가능성이 높다.
