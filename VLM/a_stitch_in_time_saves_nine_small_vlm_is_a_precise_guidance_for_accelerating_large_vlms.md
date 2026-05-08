# A Stitch in Time Saves Nine: Small VLM is a Precise Guidance for Accelerating Large VLMs

Wangbo Zhao, Yizeng Han, Jiasheng Tang, et al. (2024)

## 🧩 Problem to Solve

본 논문은 대규모 Vision-Language Model(VLM)이 수많은 시각적 토큰(visual tokens)을 처리함으로써 발생하는 심각한 추론 효율성 저하 문제를 해결하고자 한다.

기존의 가속화 방법론들은 특정 레이어의 Attention map과 같은 부분적인 정보만을 사용하여 토큰의 중요도를 평가하고 덜 중요한 토큰을 제거(pruning)하는 방식을 사용했다. 그러나 저자들은 실험을 통해 다음과 같은 세 가지 핵심 문제점을 발견하였다.

1. **부분적 정보의 한계**: 단일 레이어의 Attention 정보만으로는 중요한 시각적 토큰을 정확히 식별하기 어려우며, 특히 토큰 유지 비율(retention ratio)이 낮을 때 성능이 급격히 저하된다.
2. **전역 정보의 비용 문제**: 모든 레이어의 Attention map을 합산한 전역(Global) 정보는 토큰 제거 시 성능 유지 능력이 뛰어나지만, 이를 얻기 위해서는 전체 추론 과정을 한 번 거쳐야 하므로 계산 비용이 너무 높아 실용적이지 않다.
3. **모델 크기 간의 상관관계**: 놀랍게도 작은 VLM에서 얻은 전역 Attention map이 큰 VLM의 그것과 매우 유사하다는 점을 발견하였다.

따라서 본 연구의 목표는 작은 VLM을 가이드로 활용하여 큰 VLM의 시각적 토큰을 효율적으로 제거하고, 불필요한 큰 VLM의 호출을 줄여 추론 속도를 높이는 **SGL(Small VLM Guidance for Large VLMs)** 방법론을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **"작은 모델이 어디를 보아야 할지는 알고 있다"**는 직관에 기반한다. 비록 작은 VLM이 정답을 맞히지 못하더라도, 정답과 관련된 중요한 시각적 영역을 식별하는 능력은 큰 모델과 유사하다는 점을 활용한다. 이를 위해 두 가지 핵심 기술을 제안한다.

1. **SGP (Small VLM-Guided visual token Pruning)**: 작은 VLM의 모든 레이어에서 합산된 전역 Attention map을 생성하고, 이를 가이드로 삼아 큰 VLM에서 중요도가 낮은 시각적 토큰을 제거한다.
2. **SEE (Small VLM Early Exiting)**: 작은 VLM의 예측 결과에 대한 확신도(confidence)를 측정하여, "쉬운" 문제의 경우 큰 VLM을 호출하지 않고 즉시 추론을 종료함으로써 계산량을 극단적으로 줄인다.

## 📎 Related Works

본 논문은 다음과 같은 관련 연구들과의 차별점을 가진다.

- **Visual Token Compression**: ToMe나 FastV와 같은 기존 연구들은 토큰 병합(merging)이나 단일 레이어 기반의 제거(pruning)를 수행한다. 하지만 ToMe는 시각-언어 간의 상호작용을 간과할 수 있고, FastV는 낮은 토큰 유지 비율에서 정확도가 급격히 떨어진다. 반면 SGL은 작은 모델의 전역 정보를 활용하여 훨씬 더 공격적인 pruning(최대 91%)에서도 성능을 유지한다.
- **Confidence Estimation**: 언어 모델의 불확실성을 측정하여 상위 모델로 전달하는 Cascade 방식의 연구들이 존재한다. SGL의 SEE 메커니즘은 단순한 확률값뿐만 아니라, 토큰 제거 후의 일관성(consistency)을 함께 측정하는 새로운 기준을 제시한다.

## 🛠️ Methodology

SGL은 학습이 필요 없는(training-free) 방법론으로, 다음과 같은 파이프라인으로 구성된다.

### 1. SGP: Small VLM-Guided Visual Token Pruning

작은 VLM($VLM_S$)을 먼저 실행하여 시각적 토큰의 중요도 점수 $A$를 계산한다.

**Attention Map 합산 과정:**

- **Pre-filling 단계**: 프롬프트 토큰이 시각적 토큰에 주는 Attention 점수를 추출한다. 각 레이어 $j$와 헤드 $k$에 대해 $\tilde{A}^P_{j,k} \in \mathbb{R}^{N_T \times N_I}$를 얻고, 이를 합산하여 $A^P$를 구한다.
$$A^P = \sum_{j=1}^{L} \sum_{k=1}^{H} \bar{A}^P_{j,k}$$
- **Decoding 단계**: 생성된 토큰 $x^G$들이 시각적 토큰에 주는 Attention 점수를 누적 합산하여 $A^D$를 구한다.
$$A^D = \sum_{i=1}^{N_G} \sum_{j=1}^{L} \sum_{k=1}^{H} A^D_{i,j,k}$$
- **최종 중요도 점수**: 두 단계의 합을 통해 전역 Attention 점수를 산출한다.
$$A = A^P + A^D$$

이렇게 계산된 $A$를 기준으로 시각적 토큰의 순위를 매기고, 큰 VLM($VLM_L$)의 초기 레이어(예: 2번째 레이어)에서 상위 $R\%$의 토큰만 남기고 나머지를 제거한다.

### 2. SEE: Small VLM Early Exiting

작은 VLM의 예측이 충분히 신뢰할 만한 경우 큰 VLM을 사용하지 않고 종료한다. 이를 위해 결정 점수 $S$를 다음과 같이 정의한다.

- **Confidence Score ($S_{confidence}$)**: 생성된 시퀀스의 길이 정규화 확률값을 사용한다.
$$S_{confidence} = \exp \left( \frac{1}{N_G} \sum_{i=1}^{N_G} \log P(x^i_G | \dots) \right)$$
- **Consistency Score ($S_{consistency}$)**: SGP를 통해 토큰을 제거한 상태의 작은 모델($LM'_S$)에서도 동일한 정답이 나오는지 확인하여 일관성을 측정한다.
$$S_{consistency} = \prod_{i=1}^{N_G} P(x^i_G | LM'_S(x^I, x^T, x^{1:i-1}_G))$$
- **최종 결정 점수**: 두 점수의 평균을 사용하며, 이 값이 임계값(threshold)보다 높으면 Early Exit을 수행한다.
$$S = \frac{1}{2}(S_{confidence} + S_{consistency})$$

## 📊 Results

### 실험 설정

- **모델**: InternVL2-2B(Small) $\rightarrow$ InternVL2-26B/40B/76B(Large)
- **벤치마크**: TextVQA, ChartQA, DocVQA, GQA, RefCOCO 시리즈, SEED, MMBench, MM-Vet, MME 등 총 11개.
- **비교 대상**: ToMe, FastV 및 Original 모델.

### 주요 결과

1. **Pruning 성능**: 시각적 토큰을 9%만 남긴 극단적인 상황에서도 SGP는 Original 모델 성능의 89% 이상을 유지하였다. 이는 FastV나 ToMe가 완전히 붕괴되는 것과 대조적이다.
2. **효율성-성능 트레이드오프**: SGP와 SEE를 결합했을 때, 큰 VLM의 사이즈가 커질수록(40B, 76B) FastV 대비 더 빠른 속도와 월등한 성능을 보였다.
3. **범용성**: InternVL2 외에도 Qwen2-VL-72B와 LLaVa-OV-72B 모델에 적용했을 때, 토큰 9% 유지 시에도 원래 성능의 약 96%를 유지하며 높은 범용성을 입증하였다.
4. **메모리 오버헤드**: 작은 VLM을 추가로 로드함에도 불구하고, 전체 peak memory 증가량은 5% 미만으로 매우 적었다.

## 🧠 Insights & Discussion

**강점 및 발견:**

- 작은 VLM이 정답을 맞히는 능력(reasoning/perception)은 부족할지라도, 정답과 관련된 핵심 영역을 짚어내는 능력(localization)은 매우 뛰어나다는 점을 정량적/정성적으로 증명하였다.
- 전역 Attention map을 사용하는 것이 단일 레이어 정보를 사용하는 것보다 훨씬 정확한 가이드를 제공한다.
- 작은 VLM의 크기가 1B, 2B, 4B로 달라져도 가이드 성능에 큰 차이가 없음을 확인하여, 매우 작은 모델로도 충분히 가이딩이 가능함을 시사한다.

**한계 및 논의사항:**

- 본 연구는 주로 이해(Understanding) 작업에 집중되어 있다. 최근 등장한 이해와 생성(Generation)이 통합된 VLM 아키텍처에서의 효율성은 추가적인 연구가 필요하다.
- SEE의 임계값(threshold) 설정에 따라 효율성과 정확도의 균형이 결정되는데, 이를 자동화하는 최적의 방법론에 대한 논의가 더 필요할 수 있다.

## 📌 TL;DR

본 논문은 작은 VLM의 전역 Attention map을 이용해 큰 VLM의 불필요한 시각적 토큰을 제거하는 **SGP**와, 쉬운 문제는 작은 VLM에서 즉시 처리하는 **SEE**를 제안하였다. 이 방법은 추가 학습 없이도 큰 VLM의 추론 속도를 획기적으로 높이면서 성능 저하를 최소화하며, 특히 토큰을 91%까지 제거하는 공격적인 pruning 상황에서도 강건한 성능을 보여준다. 향후 거대 멀티모달 모델의 추론 비용을 낮추는 데 중요한 역할을 할 것으로 기대된다.
