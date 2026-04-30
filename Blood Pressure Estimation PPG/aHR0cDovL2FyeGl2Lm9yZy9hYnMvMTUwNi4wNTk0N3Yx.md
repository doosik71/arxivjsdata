# Comparison of methods of automatic blood pressure measurement in the same device

Sikorskyi M.V., Soroka A.O., Mosiychuk V.S., Sharpan O.B. (Year not explicitly stated in text, though references go up to 2014)

## 🧩 Problem to Solve

본 연구는 자동 혈압 측정 장치에서 사용되는 서로 다른 측정 방법들의 정확성과 신뢰성을 비교 분석하는 것을 목표로 한다. 혈압 측정은 심혈관 질환의 예방과 관리에 필수적이지만, 현재 상용화된 자동 혈압계에서 주로 사용하는 tacho-oscillographic 방법은 경험적 기준(empirical criteria)에 의존하여 매개변수를 결정하므로 측정의 정확도가 떨어지는 문제가 있다.

또한, 팔의 지속적인 압박(occlusion)은 환자에게 스트레스를 주거나 부종 및 혈액 정체와 같은 부작용을 일으킬 수 있으며, 특히 신체 움직임으로 인한 아티팩트(actuated artifacts)가 발생할 경우 tacho-oscillographic 방법으로는 정확한 측정이 어렵다. 따라서 본 논문은 동일한 장치와 조건에서 서로 다른 혈압 측정 알고리즘을 구현하여, 측정의 정확도, 신뢰성, 그리고 아티팩트에 대한 저항성을 비교하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 동일한 하드웨어 시스템 내에서 tacho-oscillographic 방법과 oscillometric 방법을 동시에 구현하여 실시간으로 비교 분석할 수 있는 측정 시스템 및 신호 처리 소프트웨어를 개발한 것이다. 

특히, 광전용적맥파(Photoplethysmographic, PPG) 센서를 이용한 oscillometric 방법에서 상관관계 분석(correlation analysis)을 적용함으로써, 기존의 tacho-oscillographic 방법이 가졌던 경험적 기준에 의한 매개변수 탐색 문제를 해결하고 측정의 객관성을 높일 수 있음을 제시하였다.

## 📎 Related Works

논문에서는 혈압 측정 방법을 다음과 같이 분류하여 설명한다.

1.  **침습적 방법 (Invasive method):** 동맥에 직접 압력 센서를 삽입하여 측정하며, 가장 정확하여 표준(etalon)으로 사용되지만, 출혈, 혈종, 혈전 등의 위험이 있어 병원 내 수술 상황에서만 제한적으로 사용된다.
2.  **비침습적 방법 (Non-invasive method):** 
    *   **청진법 (Auscultatory method):** 커프(cuff)의 압력을 낮출 때 발생하는 코로트코프 음(Korotkov’s tones)을 측정한다. WHO에서 인정하는 표준 비침습적 방법이며 움직임에 강하지만, 주변 소음과 마이크 배치에 매우 민감하며 '청진 간극(auscultatory gap)' 등의 현상으로 인해 정확도가 떨어질 수 있다.
    *   **Tacho-oscillographic 방법:** 커프 내의 압력 진동을 분석한다. 추가 센서가 필요 없어 간편하지만, 실제 혈류가 흐르기 전부터 진동이 감지되는 특성 때문에 수축기 혈압(SBP)이 실제보다 높게 측정되는 경향이 있으며, 이를 보정하기 위해 경험적 알고리즘을 사용한다.
    *   **Oscillometric 방법 (PPG 이용):** 적외선 파장의 흡수율 변화를 통해 혈류의 맥동을 측정하는 PPG 센서를 사용한다. 기준 채널(reference channel)을 두어 두 신호의 일치 여부를 분석함으로써 혈압을 결정한다.

## 🛠️ Methodology

### 1. 시스템 구조 및 하드웨어 구성
본 연구에서 개발한 장치는 마이크로컨트롤러와 ADC, USB 인터페이스를 포함하며, 총 3개의 아날로그 측정 채널을 가진다.
- **PPG 채널 (2개):** 1.5 kHz로 변조된 광원을 사용하여 생체 조직을 통과한 신호를 수집한다. 신호 처리 경로는 `트랜스임피던스 컨버터 $\rightarrow$ 입력 증폭기 $\rightarrow$ 밴드 필터` 순으로 구성되어 전원 노이즈 및 공통 모드 간섭을 제거하고 ADC의 다이내믹 레인지를 최적화한다.
- **커프 압력 채널 (1개):** 커프 내의 압력을 측정하며, 프리앰프와 2차 저역 통과 필터(LFF)를 거친다.

### 2. 혈압 측정 알고리즘

#### A. Oscillometric 방법 (상관관계 분석)
두 개의 PPG 채널(주 채널과 기준 채널) 간의 신호 유사성을 분석한다.
- **원리:** 커프 압력이 수축기 혈압보다 높으면 주 채널의 맥동이 사라져 상관관계가 $0\%$가 되며, 이완기 혈압보다 낮아지면 두 채널의 신호가 거의 동일해져 상관관계가 $100\%$에 도달한다.
- **상관함수 공식:** 두 PPG 신호 $S_1(t)$와 $S_2(t)$의 상관관계는 다음과 같은 교차 상관 함수(Cross-Correlation Function, CCF)로 정의된다.
$$B(\tau) = \int_{t_0}^{t_0+T} \frac{S_1(t) S_2(t+\tau)}{\sigma_1 \sigma_2} dt$$
여기서 $\sigma_1, \sigma_2$는 각 신호의 표준편차, $\tau$는 상대적 시간 차이, $T$는 적분 시간이다.
- **판정 기준:** 규격화된 CCF 값이 $10\%$가 되는 시점을 수축기 혈압($t_{sys}$), $90\%$가 되는 시점을 이완기 혈압($t_{dias}$)으로 결정한다.

#### B. Tacho-oscillographic 방법
커프 압력 센서에서 획득한 데이터에 $0.5 \sim 2\text{ Hz}$ 대역의 디지털 협대역 필터(narrow-band filter)를 적용하여 혈압 변동 성분만을 추출하고 분석한다.

## 📊 Results

### 1. 실험 설정
- **대상 및 방법:** 6명의 피험자를 대상으로 각각 10회씩 측정하였다.
- **기준(Etalon):** 청진법(Auscultative method)을 표준으로 설정하였다.
- **조건:** WHO 1999 가이드라인을 준수하여 누운 자세에서, 식후 2시간 뒤, 5분 간격으로 측정하였다.

### 2. 정량적 결과
실험 결과, oscillometric 방법과 tacho-oscillographic 방법 모두 청진법 대비 평균 오차가 $3\text{ mmHg}$를 넘지 않아 충분한 정확도를 보였다.

- **오차 분석 (Table I 요약):**
    - **Oscillometric ($\text{P1, P2}$):** 수축기 및 이완기 혈압 모두에서 비교적 낮은 편차를 보였다 (대략 $1 \sim 2\text{ mmHg}$ 수준).
    - **Tacho-oscillographic ($\text{P3, P4}$):** 수축기 혈압($\text{P3}$)에서 상대적으로 더 높은 편차가 관찰되었다 (일부 피험자의 경우 $2.8\text{ mmHg}$까지 발생).

### 3. 주요 발견
Tacho-oscillographic 방법은 수축기 혈압을 실제보다 과대평가(overvaluation)하는 경향이 나타났다. 이는 혈류가 완전히 회복되기 전, 커프 내에서 이미 진동이 감지되는 물리적 특성 때문인 것으로 분석된다.

## 🧠 Insights & Discussion

본 연구는 동일 장치에서 세 가지 방법을 동시에 측정함으로써 방법론 간의 편차를 명확히 규명하였다. 특히 tacho-oscillographic 방법의 고질적인 문제인 수축기 혈압 과대평가 경향을 실험적으로 재확인하였다.

**강점 및 시사점:**
- PPG 기반의 oscillometric 방법은 상관관계 분석을 통해 경험적 기준 없이도 혈압을 결정할 수 있어, 측정의 객관성이 높다.
- 실제 외래 진료 환경과 같이 움직임 아티팩트가 빈번하거나 코로트코프 음이 지속적으로 발생하는 경우, 기준 채널을 활용하는 oscillometric 방법이 tacho-oscillographic 방법보다 더 높은 정확도와 신뢰성을 제공할 것으로 판단된다.

**한계 및 향후 과제:**
- 실험이 통제된 정적인 환경(누운 자세)에서 수행되었으므로, 실제 움직임 아티팩트가 존재하는 상황에서의 정량적 비교 데이터는 부족하다. 
- 논문에서는 향후 연구로 움직임 아티팩트가 존재하는 상황에서의 동시 측정 및 커프가 없는(cuff-free) 방법과의 비교 분석을 제시하고 있다.

## 📌 TL;DR

본 논문은 동일 장치에서 **tacho-oscillographic** 방법과 **PPG 기반 oscillometric** 방법을 구현하여 혈압 측정 정확도를 비교하였다. 실험 결과 두 방법 모두 $3\text{ mmHg}$ 이내의 오차를 보였으나, tacho-oscillographic 방법은 수축기 혈압을 과대평가하는 경향이 있었다. 반면, **상관관계 분석을 적용한 oscillometric 방법**은 경험적 기준 없이도 정확한 측정이 가능하여, 향후 아티팩트가 많은 실제 환경에서 더 유용한 대안이 될 가능성이 높다.