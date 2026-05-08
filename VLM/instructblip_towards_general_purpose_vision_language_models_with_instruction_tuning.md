# InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning

Wenliang Dai et al. (2023)

## 🧩 Problem to Solve

본 논문은 범용 시각-언어 모델(Vision-Language Models, VLMs)을 구축하는 과정에서 발생하는 핵심적인 난제를 해결하고자 한다. 자연어 처리(NLP) 분야에서는 Instruction Tuning을 통해 광범위한 능력을 갖춘 범용 언어 모델(LLMs)이 성공적으로 개발되었으나, 시각-언어 모델의 경우 시각적 입력의 풍부한 분포와 매우 다양한 태스크 특성으로 인해 범용성을 확보하는 것이 훨씬 더 어렵다.

기존의 접근 방식인 다중 작업 학습(Multitask Learning)은 다양한 태스크를 동일한 입출력 형식으로 통합하지만, 학습 과정에서 보지 못한 새로운 데이터셋이나 태스크에 대한 제로샷(Zero-shot) 일반화 성능이 떨어진다는 한계가 있다. 또한, 단순히 이미지 캡션 데이터로 시각적 구성 요소를 학습시키는 방식은 단순 묘사 이상의 복잡한 추론이 필요한 태스크를 수행하기에는 데이터의 다양성이 부족하다. 따라서 본 연구의 목표는 통합된 자연어 인터페이스를 통해 광범위한 시각-언어 태스크를 해결할 수 있는 범용 모델인 InstructBLIP을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 크게 세 가지로 요약될 수 있다.

첫째, 시각-언어 Instruction Tuning에 대한 체계적이고 종합적인 연구를 수행하였다. 이를 위해 26개의 공개 데이터셋을 수집하여 11개의 태스크 카테고리로 분류하고, 이를 Instruction Tuning 형식으로 변환하였다. 특히 13개의 데이터셋은 학습(Held-in)에 사용하고, 나머지 13개는 제로샷 평가(Held-out)에 사용하여 모델의 일반화 능력을 엄격하게 검증하였다.

둘째, **Instruction-aware Visual Feature Extraction** 메커니즘을 제안하였다. 기존의 BLIP-2와 같은 모델은 태스크와 상관없이 정적인 시각적 특징을 추출하여 LLM에 전달하는 Instruction-agnostic 방식을 취했다. 반면, InstructBLIP은 텍스트 지시어(Instruction)를 Q-Former의 입력으로 함께 제공함으로써, 주어진 지시어에 최적화된 정보성 시각 특징을 동적으로 추출하도록 설계하였다.

셋째, 데이터셋 간의 크기 차이로 인한 오버피팅 및 언더피팅 문제를 해결하기 위해 **Balanced Sampling Strategy**를 도입하였다. 또한, FlanT5와 Vicuna라는 서로 다른 두 가지 계열의 LLM을 기반으로 모델을 구현하여 다양한 환경에서의 성능을 입증하고 이를 오픈소스로 공개하였다.

## 📎 Related Works

본 논문은 기존의 시각-언어 모델 접근 방식을 두 가지 방향으로 구분하여 설명한다.

1. **Multitask Learning:** 다양한 태스크를 하나의 형식으로 통합하여 학습하는 방식이다. 그러나 본 논문의 실험 결과, 지시어(Instruction) 없이 수행되는 다중 작업 학습은 학습되지 않은 데이터셋에 대해 낮은 일반화 성능을 보였다.
2. **LLM 확장 방식:** BLIP-2, MiniGPT-4, LLaVA와 같이 사전 학습된 LLM에 시각적 컴포넌트를 추가하는 방식이다. BLIP-2는 frozen LLM을 활용해 시각 정보를 텍스트로 생성하며, LLaVA는 시각 엔코더의 출력을 LLM의 입력으로 투영하고 대화형 데이터로 튜닝한다.

InstructBLIP은 이러한 기존 연구들과 차별화되어, 훨씬 더 광범위한 범위의 Instruction Tuning 데이터를 사용하며, 특히 Q-Former 단계에서부터 지시어를 반영하여 시각 특징을 추출하는 구조적 차별점을 갖는다.

## 🛠️ Methodology

### 전체 시스템 구조

InstructBLIP은 사전 학습된 BLIP-2 모델을 초기값으로 사용하며, 크게 세 가지 구성 요소로 이루어져 있다.

- **Image Encoder:** 이미지를 특징 벡터로 변환하는 고정된(Frozen) 모델로, ViT-g/14가 사용되었다.
- **Q-Former (Query Transformer):** 이미지 엔코더의 출력과 텍스트 지시어 사이의 상호작용을 통해 태스크에 적합한 시각적 특징을 추출하는 가변적인 모듈이다.
- **LLM:** 추출된 시각 특징을 입력받아 최종 응답을 생성하는 고정된 모델로, FlanT5 또는 Vicuna가 사용되었다.

### Instruction-aware Visual Feature Extraction

기존 BLIP-2의 Q-Former는 학습 가능한 $K$개의 쿼리 임베딩만을 사용하여 시각 특징을 추출했다. 하지만 InstructBLIP은 텍스트 지시어 토큰을 Q-Former의 입력으로 함께 제공한다. 지시어는 Q-Former 내의 Self-attention 레이어를 통해 쿼리 임베딩과 상호작용하며, 이를 통해 모델은 현재 수행해야 할 태스크(예: "상세 묘사" vs "공간 추론")에 따라 이미지의 서로 다른 영역이나 특징에 집중하여 정보를 추출할 수 있게 된다.

### 학습 절차 및 손실 함수

모델은 표준적인 언어 모델링 손실 함수(Language Modeling Loss)를 사용하여 학습된다. 주어진 이미지와 지시어를 바탕으로 정답 응답을 직접 생성하도록 최적화하며, 이때 이미지 엔코더와 LLM은 고정시키고 **Q-Former의 파라미터만 미세 조정(Finetuning)** 한다.

### Balanced Sampling Strategy

데이터셋의 크기가 서로 매우 다르기 때문에, 단순히 균등하게 샘플링할 경우 작은 데이터셋에 오버피팅되거나 큰 데이터셋을 충분히 학습하지 못하는 문제가 발생한다. 이를 해결하기 위해 데이터셋 $d$에서 샘플이 선택될 확률 $p_d$를 데이터셋 크기 $S_d$의 제곱근에 비례하도록 설정하였다.

$$p_d = \frac{\sqrt{S_d}}{\sum_{i=1}^{D} \sqrt{S_i}}$$

## 📊 Results

### 실험 설정

- **데이터셋:** 26개의 데이터셋을 13개(학습용)와 13개(제로샷 평가용)로 분리하였다.
- **비교 대상:** BLIP-2, Flamingo-80B 등.
- **지표:** 태스크에 따라 CIDEr, AUC, MRR 및 Top-1 Accuracy(%)를 사용하였다.

### 주요 결과

1. **제로샷 성능:** InstructBLIP은 모든 13개의 held-out 데이터셋에서 기존 SOTA(State-of-the-art) 성능을 달성하였다. 특히 가장 작은 모델인 InstructBLIP (FlanT5-XL, 4B)이 훨씬 거대한 Flamingo-80B보다 모든 공통 평가 데이터셋에서 우수한 성능을 보였다.
2. **Ablation Study:** Instruction-aware 기능을 제거했을 때 성능이 크게 하락하였으며, 특히 ScienceQA(공간 추론)나 iVQA(시간 추론)와 같이 정밀한 시각적 집중이 필요한 태스크에서 하락 폭이 컸다.
3. **다중 작업 학습과의 비교:** 동일한 조건에서 지시어 없이 학습한 Multitask 모델은 학습 데이터(Held-in)에 대해서는 InstructBLIP과 유사한 성능을 보였으나, 처음 보는 데이터(Held-out)에 대해서는 BLIP-2 제로샷 수준에 머물렀다. 이는 **Instruction Tuning이 제로샷 일반화 능력 향상의 핵심**임을 시사한다.
4. **다운스트림 파인튜닝:** InstructBLIP을 초기 모델로 사용하여 특정 태스크에 파인튜닝했을 때, BLIP-2보다 더 높은 성능을 보였으며 ScienceQA, OCR-VQA 등에서 SOTA를 기록하였다.

## 🧠 Insights & Discussion

### 강점 및 가치

InstructBLIP의 가장 큰 강점은 **효율성과 일반화 능력의 조화**이다. 전체 모델을 튜닝하지 않고 Q-Former라는 작은 모듈만을 튜닝함으로써 파라미터 업데이트 비용을 획기적으로 줄이면서도(1.2B $\rightarrow$ 188M), 광범위한 지시어 데이터셋을 통해 높은 제로샷 성능을 확보하였다. 또한, LLM의 종류(Encoder-Decoder인 FlanT5 vs Decoder-only인 Vicuna)에 따라 다지선다형 문제와 오픈엔드 생성 문제에 각각 강점을 보인다는 분석을 통해, 기반 LLM의 특성이 VLM의 최종 성능에 직접적인 영향을 미친다는 점을 밝혔다.

### 한계 및 비판적 해석

본 모델은 Frozen LLM을 그대로 사용하기 때문에, LLM이 본래 가지고 있는 고질적인 문제인 **환각(Hallucination) 현상이나 편향성(Bias)**을 그대로 계승한다. 논문에서는 데이터 다양성을 통해 이를 완화하려 했으나, 시각적 근거가 부족한 상태에서 LLM의 내부 지식만으로 답변을 생성하는 위험이 여전히 존재한다. 또한, 영상 데이터(Video QA)의 경우 프레임을 단순히 샘플링하여 개별 처리한 후 연결하는 방식을 취하고 있어, 영상의 연속적인 시간적 맥락을 완벽하게 포착하는 데는 한계가 있을 것으로 판단된다.

## 📌 TL;DR

InstructBLIP은 BLIP-2 구조를 기반으로, **지시어(Instruction)를 통해 시각 특징 추출을 동적으로 제어하는 Instruction-aware Q-Former**와 **데이터 균형 샘플링 전략**을 도입한 범용 시각-언어 모델이다. 26개의 다양한 데이터셋을 활용한 Instruction Tuning을 통해, 학습하지 않은 새로운 태스크에 대해서도 매우 강력한 제로샷 일반화 성능을 보였으며, 이는 향후 적은 비용으로 다양한 멀티모달 태스크를 수행할 수 있는 범용 AI 모델 설계에 중요한 이정표가 될 것으로 기대된다.
