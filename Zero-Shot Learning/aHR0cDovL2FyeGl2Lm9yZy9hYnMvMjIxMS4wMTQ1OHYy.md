# TOWARDS ZERO-SHOT CODE-SWITCHED SPEECH RECOGNITION

Brian Yan, Matthew Wiesner, Ondřej Klejch, Preethi Jyothi, Shinji Watanabe (2022)

## 🧩 Problem to Solve

본 논문은 훈련 단계에서 전사된 Code-Switched (CS, 코드 스위칭) 음성 데이터가 전혀 없는 **Zero-shot setting** 환경에서 효과적인 자동 음성 인식(ASR) 시스템을 구축하는 것을 목표로 한다. 코드 스위칭이란 한 문장 내에서 두 개 이상의 언어가 혼용되어 사용되는 현상을 말하며, 실제 환경에서는 매우 빈번하게 발생하지만, 이를 학습시키기 위한 데이터셋을 구축하는 것은 비용이 많이 들고 언어 쌍의 조합이 너무 다양하여 현실적으로 어렵다.

기존의 연구들은 이진 과업(Bilingual task)을 개별 단일 언어 과업(Monolingual parts)으로 조건부 분해(Conditionally factorize)하여 단일 언어 데이터를 효율적으로 활용하려 했다. 그러나 이러한 방식들은 각 단일 언어 모듈이 **Language Segmentation**을 수행해야 한다는 치명적인 한계가 있다. 즉, 각 모듈이 음성 세그먼트가 자신의 언어인지 아닌지를 동시에 판단하여, 자신의 언어인 경우에만 전사하고 다른 언어인 경우에는 이를 무시해야 한다. Zero-shot 환경에서 이러한 정밀한 언어 구분 능력을 갖추는 것은 매우 어렵으며, 여기서 발생하는 오류가 후속 단계로 전파되어 전체 시스템의 성능을 저하시키는 문제가 발생한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 단일 언어 모듈의 역할을 '언어 구분 및 전사'에서 단순한 **Transliteration(음역)**으로 변경하는 것이다.

기존 방식이 외국어 구간을 감지하여 무시($\langle \text{NULL} \rangle$ 토큰 생성)하도록 강제했다면, 제안된 방식은 단일 언어 모듈이 입력 음성이 어떤 언어인지 상관없이 자신의 언어 스크립트를 사용하여 무조건 전사하도록 한다. 외국어 음성이 입력되면 이를 자신의 언어에서 발음이 유사한 단위로 매핑하는 transliteration 결과가 나오게 된다.

이렇게 함으로써 CS 포인트(언어가 전환되는 지점)를 탐지해야 하는 무거운 책임을 단일 언어 단계에서 제거하고, 이를 최종 Bilingual 모듈로 지연시킨다. 최종 모듈은 여러 단일 언어 모듈의 transliteration 결과와 외부 언어 모델(Language Model)의 정보를 종합적으로 고려하여 최종 출력을 결정한다. 이를 위해 **Cross-lingual Pseudo-labeling** 기법을 도입하여, 단일 언어 데이터만으로도 transliteration 능력을 학습할 수 있도록 설계하였다.

## 📎 Related Works

기존의 다국어 음성 인식 연구들은 대규모 신경망을 통해 언어 간 공유 표현을 학습하거나, 언어 식별(LID)과 ASR을 공동 모델링하여 어느 정도의 CS 능력을 확보하려 했다. 그러나 이러한 모델들은 주로 자원이 풍부한 고자원 언어에 치우쳐 있으며, 특정 언어 쌍 간의 문장 내 코드 스위칭을 직접적으로 최적화하지는 않았다.

특히, 본 논문이 기반으로 하는 조건부 분해(Conditionally factorized) 프레임워크는 단일 언어 데이터를 직접 통합하여 CS 성능을 높이려 했다. 하지만 앞서 언급했듯이, 기존의 이러한 접근 방식들은 단일 언어 모듈이 $\langle \text{NULL} \rangle$ 토큰을 사용하여 언어 세그먼트 분리를 수행하도록 학습되었기에, Zero-shot 상황에서 일반화 성능이 크게 떨어진다는 한계가 있었다. 본 연구는 이 지점을 정확히 타격하여, '분리'가 아닌 '음역'이라는 새로운 관점을 제시함으로써 차별점을 갖는다.

## 🛠️ Methodology

### 1. Transliteration 기반의 모델링

기존의 모델은 입력 프레임 $x_t$에 대해 언어 식별(LID) 결과가 자신의 언어일 때만 전사하고, 아닐 때는 $\langle \text{NULL} \rangle$을 출력했다. 반면, 제안된 방식의 단일 언어 모듈(예: Mandarin 모듈)은 다음과 같이 단순화된다.

$$z^M_t = \text{argmax}_{m \in V^M \cup \{\emptyset\}} p(z^M_t = m | X, z^M_{1:t-1})$$

여기서 $V^M$은 중국어 어휘집이며, 입력 $X$에 어떤 언어가 섞여 있든 상관없이 중국어 스크립트로 전사한다. 이에 따라 최종 Bilingual 결정 식 또한 다음과 같이 변경된다.

$$z_t = \text{argmax}_{b \in V^M \cup V^E \cup \{\emptyset\}} p(z_t = b | Z^M, Z^E, z_{1:t-1})$$

이 구조는 단일 언어 모듈의 예측값에 맹목적으로 의존하지 않고, 두 언어 모듈의 예측치($Z^M, Z^E$)를 조건으로 하여 최종 토큰을 결정하므로 오류 전파 위험을 줄인다.

### 2. Cross-lingual Pseudo-labeling

단일 언어 모듈이 외국어를 자신의 언어로 음역하도록 학습시키기 위해, 가상의 전사 데이터(Pseudo-labels)를 생성한다. 예를 들어, 영어 음성 $X^E$를 중국어 ASR 모델 $\text{ASR}_M(\cdot)$에 통과시켜 얻은 결과물을 학습 타겟 $Y^M_{\text{TRA}}$로 사용한다.

$$Y^M_{\text{TRA}} \leftarrow \text{ASR}_M(X^E)$$
$$Y^E_{\text{TRA}} \leftarrow \text{ASR}_E(X^M)$$

### 3. Conditional CTC 아키텍처

본 논문은 **Conditional CTC**라는 신경망 구조를 제안한다. 전체 시스템은 다음과 같은 구성 요소를 가진다.

- **Speech Encoders**: 입력 음성 $X$를 각각의 단일 언어 잠재 표현 $h^M$과 $h^E$로 매핑하는 두 개의 Conformer 인코더를 사용한다.
- **CTC Networks**: 세 가지 CTC 네트워크를 운용한다.
  - 단일 언어 CTC: $P_{MCTC}(z^M_t | X)$ 및 $P_{ECTC}(z^E_t | X)$
  - Bilingual CTC: 두 잠재 표현의 합($h^M_t + h^E_t$)을 입력으로 받아 $P_{BCTC}(z_t | h^M, h^E)$를 출력한다.

학습 시에는 다음과 같은 다중 작업 손실 함수(Multi-task objective)를 사용하여 공동 최적화한다.

$$L = \lambda_1 L_{BCTC} + (1-\lambda_1) \frac{L_{MCTC} + L_{ECTC}}{2}$$

### 4. 추론 및 디코딩 절차

추론 단계에서는 병합된 CTC 확률 $P_{CTC}(Z|X)$와 외부 bilingual 언어 모델 $P_{BLM}(Y)$를 결합하여 다음과 같은 결정 식에 따라 Time-synchronous beam search를 수행한다.

$$\text{argmax}_{Y \in \{V^M \cup V^E\}^*} \lambda_2 (\prod_{Z \in Z} \log P_{CTC}(\cdot)) + (1-\lambda_2) \log P_{BLM}(\cdot)$$

## 📊 Results

### 실험 설정

- **데이터셋**: SEAME (Mandarin-English CS corpus)를 사용하였으며, 훈련 데이터에서 CS 음성을 모두 제거하여 Zero-shot 환경을 구축했다.
- **평가 지표**: 영어의 단어 수준과 중국어의 문자 수준을 모두 고려한 **Mixed Error Rate (MER)**를 측정하였다.
- **비교 대상**: Vanilla CTC 및 기존의 Language Segmentation 기반 Conditional CTC 모델과 비교하였다.

### 주요 결과

실험 결과, CS 음성 데이터가 없는 Zero-shot 설정에서 Transliteration 기반 방식이 Language Segmentation 방식보다 **절대 MER 기준 약 5%p의 성능 향상**을 보였다. 특히 CS utterances(코드 스위칭이 발생한 문장)에서 월등한 성능 개선이 나타났다.

- **데이터 조건에 따른 영향**: CS 음성 데이터가 전혀 없을 때(Zero-shot)는 제안 방법이 압도적이었으나, CS 음성 데이터가 약 2시간 이상 확보될 경우 기존의 Language Segmentation 방식이 더 우세해지는 경향을 보였다. 이는 Pseudo-labeling으로 생성된 데이터에 어느 정도의 노이즈가 포함되어 있음을 시사한다.
- **Ablation Study**: Bilingual LM을 제거했을 때 성능 저하가 가장 심했으며, 이는 Zero-shot 환경에서 외부 텍스트 기반 언어 모델이 CS 포인트 결정에 결정적인 역할을 함을 보여준다.

## 🧠 Insights & Discussion

본 논문은 단일 언어 모듈에 부여되었던 '언어 식별'이라는 과도한 부담을 제거하고, 이를 '단순 음역'으로 전환함으로써 Zero-shot CS ASR의 강건성을 확보할 수 있음을 증명하였다. 이는 복잡한 탐지 알고리즘보다 단순한 정보의 중첩과 최종 단계에서의 통합적 결정이 더 효과적일 수 있다는 통찰을 제공한다.

**강점 및 가능성**:

- 별도의 CS 음성 데이터 없이 단일 언어 데이터와 텍스트 데이터만으로 구현 가능하다.
- Bilingual CTC 모듈 없이 단일 언어 CTC들과 CS LM만으로도 어느 정도 성능이 유지된다는 점은, 향후 3개 이상의 다국어 CS ASR로 확장할 때 매우 높은 확장성(Scalability)을 가질 수 있음을 의미한다.

**한계 및 비판적 해석**:

- Pseudo-labeling의 품질이 전체 성능의 상한선을 결정한다. 현재의 Greedy decoding 기반 pseudo-label 생성 방식은 노이즈가 많으며, 이를 제약 조건 기반 디코딩(Constrained decoding) 등으로 개선할 필요가 있다.
- 2시간이라는 매우 적은 양의 CS 데이터만으로도 기존 방식이 역전한다는 점은, 본 제안 방식이 '완벽한 해결책'이라기보다 '데이터가 극도로 부족한 상황에서의 최적의 우회로'에 가깝다고 해석할 수 있다.

## 📌 TL;DR

본 연구는 Zero-shot 환경의 코드 스위칭 음성 인식(CS ASR)을 위해, 단일 언어 모듈이 언어를 구분하지 않고 무조건 전사하게 만드는 **Transliteration 기반의 Conditional CTC** 프레임워크를 제안하였다. 이를 통해 언어 구분 오류가 전파되는 문제를 해결했으며, 결과적으로 SEAME 데이터셋에서 기존 방식 대비 MER을 5%p 낮추는 성과를 거두었다. 이 연구는 특히 CS 데이터 확보가 어려운 희귀 언어 쌍의 음성 인식 시스템 구축에 중요한 기여를 할 것으로 기대된다.
