# Tarsier2: Advancing Large Vision-Language Models from Detailed Video Description to Comprehensive Video Understanding

Liping Yuan, Jiawei Wang, Haomiao Sun, Yuchen Zhang, Yuan Lin (2025)

## 🧩 Problem to Solve

본 연구는 Large Vision-Language Model (LVLM)이 비디오 이해 영역에서 겪고 있는 근본적인 한계를 해결하고자 한다. 현재 GPT-4o나 Gemini 1.5 Pro와 같은 최첨단 폐쇄형 모델들이 뛰어난 성능을 보이고 있으나, 여전히 인간 수준의 비디오 이해에는 미치지 못하고 있다. 특히 시간적 역동성(temporal dynamics)의 정확한 인지, 시공간적 추론(spatial-temporal reasoning), 그리고 모델의 환각(hallucination) 현상이 주요 문제로 지적된다.

또한, 오픈소스 모델들의 경우 복잡하고 개방형인 생성 작업, 특히 상세한 비디오 묘사(detailed video description) 작업에서 폐쇄형 모델들에 비해 크게 뒤처지는 경향이 있다. 따라서 본 논문의 목표는 상세하고 정확한 비디오 묘사 능력을 갖춤과 동시에, 전반적인 비디오 이해 능력을 극대화한 7B 규모의 LVLM인 Tarsier2를 구축하는 것이다.

## ✨ Key Contributions

Tarsier2는 기존 Tarsier 모델에서 세 가지 핵심적인 업그레이드를 통해 성능을 향상시켰다.

첫째, 사전 학습(Pre-training) 데이터의 규모를 1,100만 개에서 4,000만 개의 비디오-텍스트 쌍으로 확장하였다. 특히 영화나 TV 프로그램의 분석 내용이 담긴 '해설 비디오(commentary videos)'를 대량으로 수집하여 데이터의 양과 다양성을 모두 확보하였다.

둘째, 지도 미세 조정(Supervised Fine-Tuning, SFT) 단계에서 세밀한 시간적 정렬(fine-grained temporal alignment)을 수행하였다. 단순한 캡션-비디오 쌍이 아니라, 묘사된 각 이벤트가 비디오의 어느 프레임에 해당하는지를 명시함으로써 모델이 시간적 흐름을 정확히 파악하게 하였다.

셋째, 모델 기반 샘플링을 통해 자동으로 선호도 데이터(preference data)를 구축하고, 이를 Direct Preference Optimization (DPO) 학습에 적용하여 생성 결과물의 품질을 최적화하였다. 특히 비디오를 의도적으로 훼손(corrupt)하여 부정적 샘플을 생성하는 기법을 도입하였다.

## 📎 Related Works

기존의 Video-LLM 연구들은 주로 모델 아키텍처의 개선이나 데이터 수집에 집중해 왔다. 하지만 많은 오픈소스 모델들이 사용하는 데이터셋은 규모가 작거나, 데이터의 질이 낮고 묘사가 지나치게 단순하다는 한계가 있었다. 예를 들어 LLaVA-Video는 약 130만 개의 쌍을 사용하며, InternVL2.5 등은 500만 개 미만의 데이터를 사용한다.

비디오 묘사(Video Description) 분야에서는 최근 LVLM을 통해 상세한 출력을 생성하려는 시도가 있었으나, GPT-4V 등을 이용한 자동 어노테이션 방식은 지나치게 장황하거나 환각 현상이 발생하는 문제가 있었다. Tarsier2는 이러한 한계를 극복하기 위해 실제 해설 비디오 데이터를 활용하여 저수준의 시각적 요소부터 고수준의 플롯(plot) 정보까지 아우르는 정렬을 꾀하였다.

## 🛠️ Methodology

Tarsier2는 Qwen2-VL의 가중치를 초기값으로 사용하며, 비전 인코더, 비전 어댑터, 그리고 LLM으로 구성된 단순한 구조를 가진다. 전체 학습 프로세스는 사전 학습, SFT, RL(DPO)의 3단계로 진행된다.

### 1. Pre-training

총 4,000만 개의 비디오-텍스트 쌍을 사용하여 학습하였다. 데이터는 공개 데이터셋 2,000만 개와 자체 수집 데이터 2,000만 개로 구성된다. 특히 자체 수집 데이터 중 '해설 비디오'는 OCR 도구를 통해 자막을 추출하고, BERT 기반 모델로 시각적 대응 관계가 낮은 클립을 필터링하여 1,100만 개를 확보하였다. 비디오당 16~128 프레임을 샘플링하여 약 2,000억 개의 토큰을 학습하였다.

### 2. Supervised Fine-Tuning (SFT)

SFT는 두 단계로 나누어 진행된다.

- **SFT-1 (Fine-grained Grounding):** 15만 개의 비디오 클립에 대해 상세 묘사와 함께 각 이벤트가 발생하는 프레임 범위($\langle \text{frame:i-j} \rangle$)를 명시하여 학습시킨다. 이는 모델이 특정 이벤트의 시간적 위치를 정확히 인식하게 하여 환각을 줄이는 역할을 한다.
- **SFT-2 (Human-like Style):** SFT-1의 결과물이 너무 세분화되어 부자연스러워지는 문제를 해결하기 위해, 다양한 지시어(Instruction)와 자연스러운 형태의 묘사 데이터를 통해 인간과 유사한 스타일로 정제한다.

### 3. Direct Preference Optimization (DPO)

모델의 묘사 정확도를 더욱 높이기 위해 자동화된 선호도 데이터 구축 및 DPO 학습을 수행한다.

**부정적 샘플링(Negative Sampling):**
단순 샘플링의 변동성 문제를 해결하기 위해, 원본 비디오 $x$를 다음과 같은 섭동(perturbation)을 통해 훼손된 비디오 $\tilde{x}$로 변환한다.

- **Clip-switching:** 비디오를 4등분 하여 2개의 클립 순서를 무작위로 바꾼다.
- **Clip-reversing:** 특정 클립의 재생 방향을 반전시킨다.
- **Clip-cropping:** 비디오의 절반 길이의 무작위 클립만 추출한다.
- **Down-sampling:** 프레임의 절반을 무작위로 삭제한다.

이렇게 생성된 $\tilde{x}$를 모델에 입력하여 얻은 응답 $\tilde{y}$를 부정적 응답($y_l$)으로, 원본 $x$에 대한 응답 $y$를 긍정적 응답($y_w$)으로 설정한다.

**선호도 데이터 필터링:**
AutoDQ 스코어러를 사용하여 긍정/부정 응답 간의 차이가 명확한 데이터만 선택한다. 필터링 조건은 다음과 같다.
$$\Delta DQ^R \ge 0 \quad \text{and} \quad \Delta DQ^P \ge 0 \quad \text{and} \quad \Delta DQ^R + \Delta DQ^P \ge \delta \quad (1)$$
여기서 $\Delta DQ^R$과 $\Delta DQ^P$는 각각 Recall과 Precision 점수의 차이를 의미하며, $\delta$는 임계값이다.

최종적으로 다음과 같은 DPO 손실 함수를 최소화하여 최적화한다.
$$L_{DPO} = -\mathbb{E}_{(x, y_w, y_l) \sim D} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right] \quad (2)$$

## 📊 Results

### 1. 정량적 결과

- **비디오 묘사:** DREAM-1K 벤치마크에서 Tarsier2-7B는 F1 스코어 42.0%를 기록하며 GPT-4o(39.2%)와 Gemini-1.5-Pro를 능가하였다. 특히 Recall 점수에서 처음으로 40%를 돌파하며 동적인 이벤트 포착 능력을 입증하였다.
- **비디오 질의응답(VQA):** MVBench(71.5%), TVBench(54.7%), TOMATO(42.0%) 등 15개 벤치마크에서 SOTA 혹은 그에 준하는 성능을 달성하였다. 특히 시간적 추론이 필요한 작업에서 강세를 보였다.
- **환각 테스트:** EventHallusion에서 GPT-4o(84.1%)보다 높은 84.6%의 정확도를 보였으며, 상세 묘사 일치 작업에서는 GPT-4o를 7.1% 차이로 앞섰다.
- **Embodied QA:** EgoTaskQA에서 인간 수준의 정확도(77.5%)에 근접하는 성과를 거두었다.

### 2. 정성적 결과 및 인간 평가

인간 평가(Side-by-Side) 결과, Tarsier2-7B는 GPT-4o 대비 +8.6%, Gemini-1.5-Pro 대비 +24.9%의 선호도 우위를 보였다. 이는 Tarsier2가 단순히 길게 쓰는 것이 아니라, 실제 비디오 내용에 부합하는 정확하고 상세한 묘사를 생성함을 의미한다.

### 3. 소거 연구 (Ablation Study)

- **사전 학습:** base 모델을 Qwen2-VL로 업그레이드하고 데이터를 4,000만 개로 확장했을 때 모든 지표에서 유의미한 상승이 있었다.
- **SFT:** 세밀한 시간적 정렬(grounding) 데이터가 없을 경우 DREAM-1K F1 점수가 3.4% 하락하는 등 성능 저하가 뚜렷하였다.
- **DPO:** DPO 학습, 특히 부정적 샘플링(NS)과 필터링(PF) 전략이 묘사 작업의 정밀도를 높이는 데 핵심적인 역할을 하였다.

## 🧠 Insights & Discussion

Tarsier2의 성공 요인은 단순히 데이터의 양을 늘린 것이 아니라, **'데이터의 질적 구성'**과 **'단계적인 정렬 전략'**에 있다. 해설 비디오를 통해 고수준의 문맥을 학습하고, SFT-1에서 프레임 단위의 정밀한 정렬을 수행하며, SFT-2에서 다시 인간의 언어 스타일로 정제하는 과정이 모델의 환각을 줄이고 상세도를 높였다.

특히 DPO 단계에서 비디오를 인위적으로 훼손하여 부정적 샘플을 만든 아이디어는 매우 효과적이었다. 모델이 '잘못된 시간적 순서'나 '누락된 정보'가 무엇인지 명시적으로 학습하게 함으로써, 결과적으로 더 정확한 묘사를 가능케 하였다.

한계점으로는 학습 데이터에 아주 긴 비디오 데이터가 충분하지 않아 초장기 비디오 이해에는 개선의 여지가 있다는 점이 언급되었다. 또한, 실시간 스트리밍 비디오 분석이나 오디오-텍스트-비디오 간의 더 깊은 상호작용 연구가 향후 과제로 남아 있다.

## 📌 TL;DR

Tarsier2는 4,000만 개의 대규모 비디오-텍스트 데이터 확장, 세밀한 시간적 정렬(SFT), 그리고 비디오 훼손 기반의 선호도 최적화(DPO)를 통해 구축된 7B 규모의 LVLM이다. 본 모델은 상세 비디오 묘사 작업에서 GPT-4o와 같은 폐쇄형 모델을 능가하는 성능을 보였으며, VQA, 환각 테스트, Embodied QA 등 15개 이상의 벤치마크에서 SOTA를 달성하였다. 또한, 이 모델로 생성한 고품질 캡션 데이터셋(Tarsier2-Recap-585K)을 통해 타 모델의 성능까지 향상시킬 수 있음을 입증하며 비디오 이해 분야의 새로운 기준을 제시하였다.
