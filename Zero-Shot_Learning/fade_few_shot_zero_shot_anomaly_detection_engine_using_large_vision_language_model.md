# FADE: Few-shot/zero-shot Anomaly Detection Engine using Large Vision-Language Model

Yuanwei Li, Elizaveta Ivanova, Martins Bruveris (2024)

## 🧩 Problem to Solve

본 논문은 제조 산업의 품질 검사를 위한 자동 이미지 이상 탐지(Anomaly Detection) 문제를 다룬다. 이상 탐지는 크게 이상 여부를 분류하는 Anomaly Classification (AC)과 이상 부위를 식별하는 Anomaly Segmentation (AS)으로 나뉜다.

기존의 비지도 학습(Unsupervised) 기반 이상 탐지 방식은 각 객체 클래스마다 정상 샘플 데이터셋을 사용하여 모델을 개별적으로 학습시켜야 한다. 그러나 실제 산업 현장에서는 다음과 같은 두 가지 현실적인 제약이 존재한다. 첫째, 정상 학습 샘플의 수가 매우 적거나 아예 없는 경우가 많다. 둘째, 객체 클래스가 증가할 때마다 클래스별 모델을 새로 학습시키는 것은 확장성(Scalability) 측면에서 매우 비효율적이다.

따라서 본 논문의 목표는 정상 샘플이 거의 없거나 없는 상황에서도 작동하는 Zero-shot 또는 Few-shot 이상 탐지 엔진인 FADE를 구축하여, 추가적인 학습이나 미세 조정(Fine-tuning) 없이도 높은 성능의 AC 및 AS를 달성하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 대규모 시각-언어 모델(Vision-Language Model)인 CLIP을 활용하고, 이를 산업용 이상 탐지 작업에 최적화하기 위해 다음과 같은 세 가지 전략을 제안하는 것이다.

1. **언어 가이드 기반 세그멘테이션 개선**: CLIP의 패치 임베딩과 텍스트 간의 정렬(Alignment) 문제를 해결하기 위해 GEM (Grounding Everything Module)을 도입하고, 다양한 크기의 이상치를 탐지하기 위해 Multi-scale 접근 방식을 적용한다.
2. **LLM을 통한 프롬프트 자동 생성**: 수동으로 프롬프트를 설계하는 번거로움을 없애기 위해 ChatGPT(LLM)를 이용하여 산업용 이상 탐지에 특화된 다양하고 방대한 텍스트 프롬프트 앙상블을 자동으로 생성한다.
3. **시각 가이드 기반 보완**: 언어 기반 방법의 한계를 극복하기 위해 쿼리 이미지 및 참조 이미지의 시각적 특징을 비교하는 Vision-guided 접근 방식을 결합하여 Zero-shot 및 Few-shot 성능을 동시에 향상시킨다.

## 📎 Related Works

기존의 이상 탐지 연구는 PatchCore와 같은 메모리 뱅크 기반의 비지도 학습 방법이 주를 이루었으나, 이는 많은 양의 정상 샘플을 필요로 한다는 한계가 있다. 최근에는 Metaformer나 RegAD와 같이 Few-shot 설정을 다루는 연구들이 등장했지만, 여전히 일정 수준의 모델 학습이 필요하다.

CLIP을 활용한 연구 중 WinCLIP은 텍스트 프롬프트를 통해 Zero-shot 이상 탐지를 시도하였으나, 픽셀 단위의 AS를 위해 겹치는 윈도우(Overlapping windows) 방식을 사용하여 계산 효율성과 컨텍스트 파악에 한계가 있었다. 또한 AnomalyCLIP이나 APRIL-GAN 같은 최신 기법들은 추가적인 학습 가능 층(Learnable layers)이나 프롬프트 튜닝을 도입하여 성능을 높였으나, 이는 추가 학습 데이터가 필요하다는 단점이 있다.

FADE는 이러한 기존 연구들과 달리 **추가적인 학습이나 미세 조정 없이(Zero-training)** 오직 사전 학습된 모델과 프롬프트 엔지니어링, 그리고 구조적 개선(GEM)만으로 성능을 극대화했다는 점에서 차별점을 가진다.

## 🛠️ Methodology

FADE는 언어 가이드(Language-guided)와 시각 가이드(Vision-guided)라는 두 가지 파이프라인으로 구성된다.

### 1. Language-Guided Anomaly Classification (AC)

Zero-shot AC는 WinCLIP의 방식을 따른다. 정상($t^-$)과 이상($t^+$)을 나타내는 텍스트 프롬프트 앙상블을 구성하고, CLIP의 텍스트 인코더 $g$를 통해 각각의 평균 임베딩 $h^-$와 $h^+$를 계산한다.

$$h^+ = \frac{1}{N_{t^+}} \sum_{t \in t^+} g(t), \quad h^- = \frac{1}{N_{t^-}} \sum_{t \in t^-} g(t)$$

이미지 $x$의 CLS 토큰 임베딩 $f^{clip}_{cls}(x)$와 위 텍스트 임베딩 간의 코사인 유사도를 계산하여 최종 이상 점수 $s^{lang}$를 산출한다.

$$s^{lang} = \frac{\exp(\langle f^{clip}_{cls}(x), h^+ \rangle / \tau)}{\exp(\langle f^{clip}_{cls}(x), h^+ \rangle / \tau) + \exp(\langle f^{clip}_{cls}(x), h^- \rangle / \tau)}$$

여기서 $\tau$는 $0.01$로 고정된 온도 파라미터이다.

### 2. Language-Guided Anomaly Segmentation (AS)

CLIP의 기본 패치 임베딩은 텍스트와 잘 정렬되지 않아 AS 성능이 떨어진다. 이를 해결하기 위해 본 논문은 **GEM (Grounding Everything Module)**을 도입한다. GEM은 기존의 Query-Key attention 대신 Self-self attention(Query-Query, Key-Key, Value-Value)을 사용하여 픽셀 간의 응집력을 높인다.

$$Attn_{self-self} = \text{softmax}(h^{clip}_{Patches} W \cdot (h^{clip}_{Patches} W)^T)$$

추출된 GEM 패치 임베딩 $f^{gem}_p(x)$를 사용하여 각 패치 $p$에 대한 이상 점수 $M^{lang}_p$를 계산하며, 이를 통해 픽셀 단위의 세그멘테이션 맵 $M^{lang}$을 생성한다.

$$M^{lang}_p = \frac{\exp(\langle f^{gem}_p(x), h^+ \rangle / \tau)}{\exp(\langle f^{gem}_p(x), h^+ \rangle / \tau) + \exp(\langle f^{gem}_p(x), h^- \rangle / \tau)}$$

또한, 다양한 크기의 이상치를 잡기 위해 이미지를 $240 \times 240, 448 \times 448, 896 \times 896$ 세 가지 크기로 리사이징하여 결과를 얻은 후 이를 평균내는 **Multi-scale aggregation**을 적용한다.

### 3. LLM-based Prompt Engineering

수동 프롬프트의 한계를 극복하기 위해 ChatGPT 3.5를 사용하여 산업용 이상 탐지에 적합한 텍스트 프롬프트를 자동으로 생성한다. 객체 이름은 언급하지 않도록 지시하여 객체 불가지론적(Object-agnostic)인 프롬프트를 생성하며, 최종적으로 486개의 이상 프롬프트와 423개의 정상 프롬프트를 확보하여 WinCLIP의 프롬프트와 결합해 사용한다.

### 4. Vision-Guided Anomaly Detection

시각적 단서를 직접 비교하는 방법으로, 설정에 따라 두 가지 방식으로 작동한다.

- **Few-shot 설정**: $k$개의 정상 참조 이미지에서 패치 임베딩을 추출하여 메모리 뱅크 $R$을 구축한다. 쿼리 이미지 패치 $f^{clip}_p(x)$와 메모리 뱅크 내 가장 가까운 이웃 간의 코사인 거리를 계산하여 점수를 매긴다.
  $$M^{vis,k}_p = \min_{r \in R} \frac{1}{2}(1 - \langle f^{clip}_p(x), r \rangle)$$
- **Zero-shot 설정**: 참조 이미지가 없으므로 쿼리 이미지 자체를 메모리 뱅크로 사용한다. 이때 자기 자신과의 거리가 0이 되므로, 두 번째로 가까운 이웃(second nearest neighbour)과의 거리를 계산한다. 이 과정에서는 GEM 임베딩을 사용한다.

## 📊 Results

### 실험 설정

- **데이터셋**: MVTec-AD, VisA
- **지표**: AC는 AUROC, AUPR, F1-max를 사용하며, AS는 pAUROC, PRO, F1-max를 사용한다.
- **비교 대상**: PatchCore (Unsupervised), WinCLIP (Zero/Few-shot)

### 주요 결과

1. **Anomaly Classification (AC)**: Few-shot 설정에서 FADE는 MVTec-AD와 VisA 모두에서 WinCLIP과 PatchCore보다 우수한 성능을 보였다. 특히 VisA 데이터셋에서 향상 폭이 컸다.
2. **Anomaly Segmentation (AS)**: Zero-shot 설정에서 FADE는 pAUROC 기준 MVTec-AD에서 $89.6\%$, VisA에서 $91.5\%$를 기록하며 기존 방법들을 크게 상회하였다. 1-shot 설정에서도 각각 $95.4\%$와 $97.5\%$라는 높은 성능을 달성하였다.
3. **Ablation Study**:
    - GEM 임베딩을 사용하지 않고 일반 CLIP 임베딩을 사용했을 때, 정상/이상 영역이 반대로 시각화되는 현상이 발생하며 pAUROC가 $50\%$ 미만으로 급격히 떨어진다.
    - ChatGPT 프롬프트를 추가했을 때 AS 성능이 눈에 띄게 향상되었으나, 이미지 레벨의 AC 성능에는 큰 영향이 없었다.
    - Multi-scale aggregation은 다양한 크기의 이상치를 탐지하는 데 필수적이며, 단일 스케일보다 항상 우수한 성능을 보였다.

## 🧠 Insights & Discussion

본 논문은 학습 없이 모델의 구조적 특성과 프롬프트를 최적화하는 것만으로도 강력한 이상 탐지가 가능함을 입증하였다. 특히 GEM 임베딩이 언어-시각 정렬을 개선하여 Zero-shot 세그멘테이션의 고질적인 문제(반전 시각화 등)를 해결했다는 점이 고무적이다.

**한계점 및 논의사항**:

- **재현성 문제**: ChatGPT를 통해 생성된 프롬프트는 모델의 무작위성으로 인해 완전히 동일하게 재현하기 어렵다. 오픈소스 LLM을 통한 해결책이 제시되었으나, LLM 종류나 프롬프트 민감도에 대한 추가 연구가 필요하다.
- **임베딩 선택의 모호성**: 실험 결과, 언어 가이드 AC에는 CLIP 임베딩이, AS에는 GEM 임베딩이 적합하며, 시각 가이드 Few-shot AS에는 다시 CLIP 임베딩이 적합하다는 결과가 나왔다. 왜 상황마다 적합한 임베딩이 다른지에 대한 이론적 분석이 부족하다.
- **객체 이미지의 취약성**: Zero-shot 시각 가이드 방식은 텍스처 이미지(예: 가죽)에서는 잘 작동하지만, 패치 간 변동성이 큰 객체 이미지(예: 트랜지스터)에서는 오탐(False Positive)이 발생하는 경향이 있다.

## 📌 TL;DR

FADE는 추가 학습 없이 CLIP 모델을 활용하여 Zero-shot 및 Few-shot 산업 이상 탐지를 수행하는 엔진이다. **GEM 임베딩**을 통한 픽셀-텍스트 정렬 개선, **LLM 기반 자동 프롬프트 생성**, **Multi-scale aggregation**, 그리고 **시각 가이드 메모리 뱅크**를 결합하여 MVTec-AD와 VisA 벤치마크에서 SOTA 수준의 성능을 달성하였다. 이 연구는 데이터 수집이 어려운 산업 현장에서 즉시 적용 가능한 범용적 이상 탐지 프레임워크로서의 가능성을 제시한다.
