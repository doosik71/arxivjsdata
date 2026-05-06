# Cross-Modal Safety Mechanism Transfer in Large Vision-Language Models

Shicheng Xu, Liang Pang, Yunchang Zhu, Huawei Shen, Xueqi Cheng (2025)

## 🧩 Problem to Solve

본 논문은 Large Vision-Language Models(LVLMs)에서 텍스트 모달리티에 대해 구축된 안전 메커니즘(Safety Mechanism)이 시각 모달리티로 효과적으로 전이되지 않는 문제를 해결하고자 한다.

기존의 LVLM들은 기본적으로 안전 정렬(Safety Alignment)이 완료된 대규모 언어 모델(LLM)을 기반으로 시각-언어 정렬(Vision-Language Alignment) 과정을 거치지만, 실제 실험 결과 동일한 유해 시맨틱을 가졌음에도 불구하고 텍스트 입력보다 이미지 입력에 대해 훨씬 더 취약한 모습을 보인다. 이는 LLM이 이미 가지고 있는 텍스트 기반의 안전 메커니즘이 시각 데이터로 확장되지 않았음을 의미한다. 따라서 본 연구의 목표는 추가적인 시각적 안전 미세 조정(Safety Fine-tuning) 없이, LLM의 기존 안전 메커니즘을 시각 모달리티로 전이시키는 방법을 제시하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 **Cross-Modal Safety Mechanism Transfer**라는 새로운 관점을 제시하고, 이를 실현하기 위한 **Text-Guided Alignment(TGA)** 방법론을 제안한 것이다.

핵심 직관은 LVLM의 안전 메커니즘이 특정 Transformer 레이어의 Hidden States에 의해 활성화된다는 점에 착안하여, 시각 입력의 Hidden States를 해당 시맨틱을 가진 텍스트의 Hidden States 공간으로 정렬시킴으로써 LLM의 내장된 안전 메커니즘을 그대로 활용하도록 만드는 것이다.

## 📎 Related Works

기존의 LVLM 정렬 방식(예: LLaVA, InstructBLIP)은 주로 이미지-명령어-출력의 쌍을 이용해 출력 텍스트를 최적화하는 'Input-to-Output Alignment'에 집중했다. 그러나 이러한 방식은 LLM 내부의 표현 공간(Internal Representation Space)에서 시각 정보가 텍스트 정보와 실제로 정렬되었는지를 간과한다.

또한, 기존의 LVLM 안전성 연구들은 유해한 시각 데이터를 직접 수집하여 안전 미세 조정을 수행하는 방식에 의존했다. 하지만 이러한 접근법은 유해한 멀티모달 데이터를 대량으로 수집하는 데 막대한 비용과 인력이 소요되며, 특히 오디오나 비디오와 같은 다른 모달리티로 확장할 때 한계가 명확하다. 본 논문은 이러한 데이터 의존성을 탈피하여 모델 내부의 메커니즘 전이에 집중한다는 점에서 차별성을 갖는다.

## 🛠️ Methodology

### 1. 안전 메커니즘의 작동 원리 분석

연구진은 안전 메커니즘이 어디서, 어떻게 작동하는지 분석하기 위해 'Sorry' 의미를 가진 토큰의 분포 변화를 측정하여 안전 메커니즘이 활성화되는 특정 레이어를 탐색하는 알고리즘을 제안했다.

특정 레이어 $j$에서의 단어 분포 변화 $D^j$는 다음과 같이 정의된다.
$$D^j(x|t, s) = \log \frac{P^j(x|t, s)}{P^{j-1}(x|t, s)}$$
여기서 $P^j$는 $j$번째 레이어의 Hidden State $H^j$와 Vocabulary Head $W$를 통해 계산된 확률 분포이다. 'Sorry' 계열의 토큰이 $D^j$에서 가장 높은 순위를 차지하기 시작하는 지점을 안전 메커니즘의 활성화 지점으로 정의했다. 분석 결과, 안전 메커니즘은 특정 레이어에서 유해 토큰들의 Hidden States에 대한 Attention이 급증할 때 활성화됨을 확인했다.

### 2. 실패 원인: Hidden States의 불일치

실험을 통해 유해한 텍스트와 그에 대응하는 유해 이미지의 Hidden States 간 코사인 유사도를 측정했다. 그 결과, 안전 메커니즘이 활성화되어야 하는 특정 레이어에서 텍스트와 이미지의 Hidden States 간 정렬이 매우 부족함을 발견했다. 이러한 시맨틱 시프트(Semantic Shift)로 인해, 안전 레이어들이 이미지 입력의 유해성을 올바르게 판단하지 못하고 안전 메커니즘이 붕괴되는 것이다.

### 3. Text-Guided Alignment (TGA)

이를 해결하기 위해 본 논문은 Hidden State 레벨에서 시각-언어를 정렬하는 TGA 방법론을 제안한다.

**가. 데이터 구성**
각 이미지 $X_{image}$에 대해 두 가지 텍스트를 준비한다.

- $X_{caption}$: LLaVA-1.5-13B를 통해 생성한 이미지의 텍스트 설명.
- $X_{retrieval}$: BEIT-3를 이용해 대규모 코퍼스에서 검색한 관련 텍스트.

**나. 학습 절차 및 손실 함수**
TGA는 이미지를 LLM의 Hidden State 공간으로 투영할 때, 텍스트의 Hidden State를 가이드로 사용한다.

- $I^j, C^j, R^j$를 각각 $j$번째 레이어에서 이미지, 캡션, 검색된 텍스트의 Mean Pooled Hidden State 벡터라고 정의한다.
- Pair-wise Loss $\mathcal{L}_{guide}$를 도입하여 이미지의 표현이 검색된 텍스트보다 캡션의 표현에 더 가깝게 위치하도록 유도한다.
$$\mathcal{L}_{guide} = \sum_{j=1}^{N} -\cos(I^j, C^j) + \log \left[ 1 + \exp(-( \cos(I^j, C^j) - \cos(R^j, C^j) )) \right]$$
- 최종 손실 함수는 위 가이드 손실과 일반적인 언어 모델링을 위한 Cross-Entropy 손실 $\mathcal{L}_{CE}$의 합으로 구성된다.
$$\mathcal{L} = \mathcal{L}_{guide} - \frac{1}{N} \sum_{i=1}^{N} \log P(X_{a,i} | X_{retrieval}, X_{image}, X_{inst}, X_{a,<i})$$

## 📊 Results

### 1. 실험 설정

- **데이터셋**: 유해 이미지 데이터셋인 HOD(음주, 흡연, 무기 등)와 ToViLaG(잔인함, 음란물)를 사용했다.
- **지표**: 모델이 유해한 응답을 거부하는 비율인 Defence Success Rate(DSR)를 사용했으며, 응답의 유해성 판단은 LLaMA-2-7B를 통해 수행했다.
- **비교 대상**: LLaVA-1.6, InstructBlip, Qwen-VL-Chat 및 텍스트 언러닝 기반의 Unlearn-FigS.

### 2. 정량적 결과

- **안전성 향상**: TGA는 시각 모달리티에 대한 추가적인 안전 미세 조정 없이도 유해 이미지에 대한 DSR을 획기적으로 높였다. 예를 들어, Mistral-7B 기반의 LLaVA-1.6보다 모든 유해 시나리오에서 월등히 높은 방어 성공률을 보였다.
- **일반 성능 유지**: SciQA, POPE, SEED-Bench, MM-Vet 등 다양한 벤치마크에서 기존 SoTA 모델들과 대등하거나 더 우수한 성능을 기록하여, 안전성 강화가 일반적인 시각 이해 능력을 저해하지 않음을 입증했다.
- **강건성 테스트**: 캡션에 10% 정도의 노이즈가 섞여도 안정적인 성능을 유지했으며, Role-Play, ICA, FigStep과 같은 탈옥(Jailbreak) 공격에 대해서도 기존 모델들보다 높은 방어력을 보였다.

## 🧠 Insights & Discussion

본 논문은 LVLM의 안전성 문제를 단순히 '데이터의 부족'으로 보지 않고, '모달리티 간 내부 표현의 불일치'라는 아키텍처 및 정렬의 관점에서 해석했다는 점이 매우 인상적이다.

**강점 및 통찰**

- 안전 메커니즘이 특정 레이어에서 활성화된다는 점을 정량적으로 증명함으로써, 블랙박스에 가까운 LLM의 안전 작동 원리를 일부 규명했다.
- 유해 데이터를 직접 학습시키지 않고도 기존 LLM의 능력을 '전이'시키는 방식은 데이터 수집의 윤리적 문제와 비용 문제를 동시에 해결할 수 있는 효율적인 방향이다.

**한계 및 논의사항**

- TGA는 초기 캡션 생성 모델(LLaVA-1.5-13B)의 성능에 의존한다. 비록 노이즈 테스트를 통해 강건성을 입증했으나, 캡션 생성 단계에서 치명적인 오류가 발생할 경우 정렬 성능이 저하될 가능성이 있다.
- 잔인함(Bloody)과 같은 특정 도메인에서는 다른 도메인에 비해 DSR 상승폭이 상대적으로 낮았는데, 이는 해당 도메인의 시각-텍스트 간 시맨틱 갭이 더 크거나 텍스트 안전 메커니즘 자체가 해당 도메인에서 약할 가능성을 시사한다.

## 📌 TL;DR

본 논문은 LVLM이 텍스트보다 이미지의 유해성에 취약한 이유가 **특정 Transformer 레이어에서 시각-언어 간 Hidden State 정렬이 부족하기 때문**임을 밝혀냈다. 이를 해결하기 위해 검색된 텍스트와 캡션을 가이드로 삼아 Hidden State 레벨에서 정렬을 수행하는 **TGA(Text-Guided Alignment)** 방법을 제안하였다. 결과적으로 TGA는 추가적인 유해 이미지 학습 없이도 LLM의 기존 안전 메커니즘을 시각 모달리티로 성공적으로 전이시켰으며, 일반적인 비전 작업 성능 또한 유지하거나 향상시켰다.
