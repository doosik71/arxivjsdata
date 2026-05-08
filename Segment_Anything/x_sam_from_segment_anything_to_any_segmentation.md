# X-SAM: From Segment Anything to Any Segmentation

Hao Wang et al. (2025)

## 🧩 Problem to Solve

최근 대규모 언어 모델(LLMs)은 광범위한 지식 표현 능력을 보여주었으나, 픽셀 수준의 지각적 이해(pixel-level perceptual understanding) 능력이 본질적으로 부족하다. Segment Anything Model (SAM)의 등장으로 시각적 프롬프트 기반의 이미지 분할(image segmentation) 분야에서 큰 진전이 있었으나, SAM은 여전히 다음과 같은 한계점을 가지고 있다.

첫째, 다중 마스크 예측(multi-mask prediction)과 카테고리 특정적 분할(category-specific segmentation) 작업에서 뚜렷한 한계를 보인다. 둘째, 시각적 프롬프트에 대한 의존도가 높아 범용 분할(generic segmentation), 참조 분할(referring segmentation), 오픈 보캐브러리(open-vocabulary) 분할 등 다양한 분할 작업을 하나의 통합된 모델 구조 내에서 처리하지 못한다.

본 논문의 목표는 이러한 한계를 극복하여, '어떤 것이든 분할(segment anything)'하는 수준을 넘어 '어떤 분할 작업이든 수행(any segmentation)'할 수 있는 통합된 멀티모달 거대 언어 모델(MLLM) 프레임워크인 X-SAM을 제안하는 것이다.

## ✨ Key Contributions

X-SAM의 핵심 아이디어는 텍스트 쿼리와 시각적 쿼리를 모두 처리할 수 있는 통합 입력 포맷을 설계하고, 이를 처리할 수 있는 MLLM 구조와 SAM의 강력한 마스크 생성 능력을 결합하는 것이다. 주요 기여 사항은 다음과 같다.

1. **통합 분할 프레임워크 제안**: 다양한 이미지 분할 작업을 표준화된 포맷으로 변환하여 하나의 모델에서 처리할 수 있도록 하는 X-SAM 프레임워크를 구축하였다.
2. **Visual GrounDed (VGD) 분할 작업 정의**: 사용자가 제공한 상호작용 시각적 프롬프트(점, scribble, 박스, 마스크 등)를 통해 이미지 내의 모든 인스턴스 객체를 분할하는 새로운 벤치마크를 제안하여, MLLM에 시각적 가이드 모달리티를 도입하였다.
3. **다단계 통합 학습 전략**: 세그멘터 미세 조정, 정렬 사전 학습, 혼합 미세 조정으로 이어지는 3단계 학습 전략을 통해 다양한 데이터셋에서 최적의 성능을 낼 수 있도록 하였다.

## 📎 Related Works

논문에서는 크게 세 가지 관련 연구 분야를 다룬다.

1. **Multi-modal Large Language Model**: LLaVA와 같은 모델들이 시각적 특성 토큰화를 통해 발전해 왔으나, 대부분의 진전은 특정 작업에 국한되어 있으며 픽셀 수준의 정밀한 분할 능력을 통합적으로 갖춘 모델은 부족한 실정이다.
2. **Multi-modal Grounded Segmentation**: SAM과 그 확장 모델들이 시각적 그라운딩 신호를 사용하여 성능을 높였으나, 그라운딩된 입력을 텍스트 입력처럼 자유롭게 다루어 분할 작업에 활용하는 능력은 부족하였다.
3. **Unified Segmentation Model**: Mask2Former와 같은 end-to-end 마스크 분류 프레임워크가 등장하며 범용 분할 능력이 향상되었으나, MLLM에서 볼 수 있는 상호작용적 텍스트 및 시각적 프롬프트를 통합적으로 처리하는 구조는 결여되어 있었다.

X-SAM은 SAM의 마스크 생성 능력과 MLLM의 추론 능력을 결합함으로써, 기존 모델들이 수행하지 못했던 텍스트-시각 쿼리 통합 처리를 가능하게 하여 차별점을 갖는다.

## 🛠️ Methodology

### 1. 통합 포맷 정의 (Formulation)

X-SAM은 다양한 분할 작업을 처리하기 위해 입력 포맷을 **Text Query Input**과 **Vision Query Input** 두 가지로 정의한다.

* **Text Query Input**: 사용자 요청에 따른 언어적 프롬프트만 포함한다. 범용 분할, 참조 분할, 추론 분할 등이 이에 해당한다. 특정 구절의 시작과 끝을 알리는 특수 토큰 $\langle\text{p}\rangle$와 $\langle/\text{p}\rangle$를 사용하여 "$\langle\text{p}\rangle \text{category/phrase/sentence} \langle/\text{p}\rangle$" 형태로 표준화한다.
* **Vision Query Input**: 언어적 프롬프트와 함께 점, 박스, 마스크 등의 시각적 프롬프트를 포함한다. 시각적 프롬프트는 $\langle\text{region}\rangle$ 토큰으로 표시하며, 최종적으로 "$\langle\text{p}\rangle\langle\text{region}\rangle\langle/\text{p}\rangle$" 형태로 입력된다.
* **출력**: 모든 분할 결과는 특수 토큰 $\langle\text{SEG}\rangle$를 통해 출력된다.

### 2. 모델 아키텍처 (Architecture)

X-SAM은 크게 dual encoders, dual projectors, LLM, segmentation connector, segmentation decoder로 구성된다.

* **Dual Encoders**: 글로벌 이미지 특징을 추출하는 **Image Encoder** (SigLIP2-so400m)와 세밀한 특징을 추출하는 **Segmentation Encoder** (SAM-L)를 동시에 사용한다.
* **Dual Projectors**: 두 인코더에서 나온 특징을 LLM의 임베딩 공간으로 투영한다. 특히 세그멘테이션 인코더의 특징은 크기가 매우 크기 때문에 **Pixel Shuffle** 연산을 통해 공간 크기를 줄인 후 MLP projector를 통해 투영한다.
* **Segmentation Connector**: SAM의 인코더 출력은 단일 스케일($1/16$)이므로, 이를 multi-scale 특징으로 확장하기 위해 Pixel Shuffle을 이용한 patch-merge($0.5$배)와 patch-expand($2.0$배)를 수행하여 $1/32, 1/16, 1/8$ 스케일의 특징을 생성한다.
* **Segmentation Decoder**: 기존 SAM의 디코더를 Mask2Former 스타일의 새로운 디코더로 교체하였다. 이를 통해 단일 객체가 아닌 이미지 내의 모든 객체를 한 번의 추론으로 예측할 수 있으며, LLM의 $\langle\text{SEG}\rangle$ 토큰 임베딩을 조건으로 마스크와 카테고리 확률을 예측한다.

### 3. 학습 절차 (Training)

학습은 총 3단계로 진행된다.

**Stage 1: Segmentor Fine-tuning**
새로 설계된 디코더가 단일 패스에서 모든 객체를 분할할 수 있도록 COCO-Panoptic 데이터셋으로 학습시킨다. 손실 함수 $\mathcal{L}_{\text{seg}}$는 다음과 같이 정의된다.
$$\mathcal{L}_{\text{seg}} = \mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{mask}} + \mathcal{L}_{\text{dice}}$$
여기서 $\mathcal{L}_{\text{cls}}$는 분류 손실, $\mathcal{L}_{\text{mask}}$와 $\mathcal{L}_{\text{dice}}$는 마스크의 정확도를 높이기 위한 손실 함수이다.

**Stage 2: Alignment Pre-training**
시각적 임베딩과 언어 임베딩을 정렬하기 위해 LLaVA-558K 데이터셋을 사용하여 dual projectors만 학습시킨다. 이때의 목표는 auto-regressive loss $\mathcal{L}_{\text{regressive}}$를 최소화하는 것이다.
$$\mathcal{L}_{\text{regressive}} = -\sum_{i=1}^{N} \log p_{\theta} (Y_{[P+i]}^q | Y_{[:i-1]}^q, X_{[:i-1]}^q)$$

**Stage 3: Mixed Fine-tuning**
다양한 작업의 데이터셋을 사용하여 end-to-end로 공동 학습(co-training)을 수행한다. 대화 작업에는 $\mathcal{L}_{\text{regressive}}$를, 분할 작업에는 $\mathcal{L}_{\text{regressive}}$와 $\mathcal{L}_{\text{seg}}$를 모두 사용하여 학습한다.
$$\mathcal{L}_{\text{total}} = \begin{cases} \mathcal{L}_{\text{regressive}}, & \text{conversation} \\ \mathcal{L}_{\text{regressive}} + \mathcal{L}_{\text{seg}}, & \text{segmentation} \end{cases}$$

## 📊 Results

### 실험 설정

* **데이터셋 및 작업**: Generic, Open-Vocabulary, Referring, Reasoning, GCG, Interactive, VGD Segmentation 등 7가지 작업에 대해 평가하였다. 특히 COCO-VGD는 본 논문에서 제안한 벤치마크이다.
* **측정 지표**: 작업별로 PQ, mIoU, mAP (범용/OV), cIoU, gIoU (참조/추론), METEOR, CIDEr (GCG) 등을 사용하였다.
* **비교 대상**: SAM-L, Mask2Former, SEEM, LISA, GLaMM, PSALM 등 최신 SOTA 모델들과 비교하였다.

### 주요 결과

1. **통합 성능**: X-SAM은 단일 모델임에도 불구하고 거의 모든 이미지 분할 벤치마크에서 SOTA 성능을 달성하였다 (Tab. 2).
2. **Referring Segmentation**: RefCOCO, RefCOCO+, RefCOCOg 모든 셋에서 PSALM과 Sa2VA를 제치고 가장 높은 cIoU를 기록하였다.
3. **GCG Segmentation**: 이미지 묘사와 마스크 생성을 동시에 수행하는 작업에서 GLaMM과 OMG-LLaVA보다 우수한 픽셀 수준 이해도(mIoU, AP)를 보였다.
4. **VGD Segmentation**: 새롭게 제안된 VGD 작업에서 PSALM 대비 압도적인 성능 향상(약 45% 이상의 AP 증가)을 보이며, 시각적 프롬프트 기반의 인스턴스 분할 능력을 입증하였다.
5. **이미지 수준 벤치마크**: MME, MMBench 등 일반적인 MLLM 벤치마크에서도 OMG-LLaVA를 능가하는 성능을 보여, 분할 능력 강화가 전체적인 이미지 이해 능력 향상으로 이어졌음을 보여주었다.

## 🧠 Insights & Discussion

### 강점 및 의의

X-SAM은 텍스트 쿼리와 시각적 쿼리를 하나의 통합된 프레임워크 내에서 처리함으로써, 기존의 task-specific 모델들이 가졌던 파편화 문제를 해결하였다. 특히 SAM의 강력한 마스크 생성 능력과 LLM의 고차원적 추론 능력을 성공적으로 결합하여, 복잡한 추론이 필요한 Reasoning Segmentation과 정교한 시각적 가이드가 필요한 VGD Segmentation 모두에서 우수한 성능을 냈다.

### 한계 및 논의사항

1. **데이터 불균형 문제**: 다양한 데이터셋을 함께 학습시키는 mixed fine-tuning 과정에서 일부 분할 데이터셋의 성능이 약간 하락하는 현상이 관찰되었다. 이는 멀티소스 학습 시 데이터 간의 밸런스를 맞추는 것이 매우 까다로운 과제임을 시사한다.
2. **모델 크기와 성능의 트레이드-오프**: 모든 작업에서 완벽한 최적 성능을 내는 것은 여전히 어려우며, 이는 향후 모델 파라미터 규모 확장(scaling up)과 더 방대한 학습 데이터 확보를 통해 해결해야 할 문제로 남아 있다.
3. **비판적 해석**: 본 논문은 SAM의 인코더를 활용하여 효율성을 높였으나, SAM2와 같은 최신 모델로의 확장이 필요하다. 또한 VGD 작업의 정의가 주로 COCO 데이터셋에 기반하고 있어, 더 복잡하고 다양한 실제 환경에서의 일반화 성능에 대한 추가 검증이 필요해 보인다.

## 📌 TL;DR

X-SAM은 SAM의 픽셀 수준 분할 능력과 MLLM의 언어 이해 능력을 통합하여, **텍스트 및 시각적 프롬프트 모두에 대응하는 'Any Segmentation' 모델**을 구현하였다. 특히 새로운 **VGD(Visual GrounDed) 분할 작업**을 제안하고 다단계 학습 전략을 통해 7가지 이상의 다양한 분할 작업에서 SOTA 성능을 달성하였다. 이 연구는 향후 MLLM이 단순한 텍스트 출력을 넘어 정밀한 픽셀 수준의 시각적 조작 및 이해를 수행하는 통합 모델로 발전하는 데 중요한 기준점(baseline)을 제시한다.
