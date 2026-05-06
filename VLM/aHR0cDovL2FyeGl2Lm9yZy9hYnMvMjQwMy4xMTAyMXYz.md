# Towards Neuro-Symbolic Video Understanding

Minkyu Choi, Harsh Goel, Mohammad Omama, Yunhao Yang, Sahil Shah, and Sandeep Chinchali (2024)

## 🧩 Problem to Solve

최근 비디오 데이터의 폭발적인 증가로 인해 방대한 분량의 영상에서 의미 있는 프레임을 효율적으로 추출하고 검색하는 도구의 필요성이 증대되었다. 특히, 단순한 객체 인식을 넘어 "사건 A가 발생하고, 사건 B는 발생하지 않았으며, 몇 시간 뒤에 사건 C가 발생하는" 장면을 찾는 것과 같은 **장기적 시간 추론(Long-term temporal reasoning)** 능력이 필수적이다.

그러나 VideoLLaMA나 ViCLIP과 같은 최신 Foundation Model들은 단기적인 의미 이해에는 능숙하지만, 프레임 간의 장기적인 추론에서는 한계를 보인다. 논문은 이러한 실패의 핵심 원인이 **단일 딥러닝 네트워크 내에서 개별 프레임의 인지(Perception)와 시간적 추론(Temporal reasoning)이 서로 얽혀 있기 때문**이라고 분석한다. 따라서 효율적인 장면 식별을 위해서는 의미론적 이해와 시간적 추론을 분리하여 설계하는 것이 중요하다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 신경망의 인지 능력과 기호적(Symbolic) 추론 능력을 결합한 **Neuro-Symbolic 접근 방식**을 통해 비디오 이해 문제를 해결하는 것이다.

구체적으로, 개별 프레임의 의미론적 이해는 Vision-Language Model(VLM)과 같은 신경망 모델에 맡기고, 이벤트의 장기적인 진화 과정에 대한 추론은 메모리 능력을 본질적으로 갖춘 **상태 머신(State Machines)**과 **시간 논리(Temporal Logic, TL)** 공식을 사용하여 수행한다. 이를 통해 인지와 추론을 분리함으로써 매우 긴 비디오에서도 복잡한 시간적 쿼리를 정확하게 처리할 수 있는 NSVS-TL(Neuro-Symbolic Video Search with Temporal Logic) 시스템을 제안한다.

## 📎 Related Works

기존의 비디오 이벤트 탐지 연구는 주로 딥러닝을 이용해 객체의 모션이나 위치와 같은 잠재 표현(Latent representations)을 학습하는 방식에 집중해 왔다. 하지만 이러한 방식은 학습에 막대한 계산 자원이 소모되며, 결과에 대한 해석 가능성이 떨어진다는 단점이 있다.

최근의 Video-Language Model(예: Video-Llama, Video-ChatGPT)은 GPT-4나 Llama와 같은 대규모 언어 모델(LLM)을 통합하여 제로샷 이벤트 인식 등을 수행하지만, 비디오 정보를 잠재 벡터로 집계(Aggregate)하는 과정에서 긴 비디오의 정확한 프레임 식별 능력이 저하되는 문제가 발생한다.

반면, 본 논문에서 제안하는 방식은 완전히 해석 가능한 **오토마톤(Automaton) 기반 표현**을 사용한다. 이는 잠재 공간에 의존하는 기존의 심볼릭 비디오 이해 방식과 차별화되며, 하위 작업인 비디오 검색에 대해 형식적 보장(Formal guarantees)을 제공할 수 있다는 점에서 강점을 가진다.

## 🛠️ Methodology

NSVS-TL 시스템은 다음의 4단계 파이프라인을 통해 관심 장면을 식별한다.

### 1. Neural Perception Model Calibration (신경망 인지 모델 보정)

신경망 모델 $f_v$가 출력하는 신뢰도 $\hat{y}$를 확률적 검증에 사용할 수 있는 정확도 지표로 변환하기 위해 일반화된 로지스틱 함수 $z(\hat{y})$를 사용하여 보정한다.
$$z(\hat{y}) = \frac{1}{1 + e^{-k(\hat{y}-\hat{y}_0)}}$$
여기서 $k$는 민감도를 조절하는 스케일링 인자이며, $\hat{y}_0$는 변곡점이다. 최종적으로 거짓 양성(FP) 임계값 $\gamma_{fp}$와 진 양성(TP) 임계값 $\gamma_{tp}$를 적용한 함수 $g$를 정의한다.
$$g(\hat{y};\gamma_{fp},\gamma_{tp}) = \begin{cases} 0 & \text{if } \hat{y} < \gamma_{fp} \\ 1 & \text{if } \hat{y} > \gamma_{tp} \\ z(\hat{y}) & \text{otherwise} \end{cases}$$

### 2. Frame Validation (프레임 검증)

모든 프레임을 처리하는 대신, 쿼리와 관련 있는 프레임만 선별하기 위해 두 가지 검증 함수를 사용한다.

- **Detection Verification ($V_c$):** 쿼리에 포함된 원자 명제(Atomic propositions)들이 프레임 내에 존재하는지 확인한다.
- **Symbolic Verification ($V_{sv}$):** 프레임이 시간 논리 식 $\Phi$의 1차 논리 연산자($\Psi$: $\wedge, \vee, \neg$ 등) 및 시간 연산자($\Theta$: $\Box, \Diamond, U$ 등)와 일치하는지 확인한다.
최종적으로 $V(F_t, P) = V_c \wedge V_{sv}$ 가 1인 프레임만 오토마톤 구성 단계로 넘어간다.

### 3. Dynamic Automaton Construction (동적 오토마톤 구축)

검증된 프레임들을 기반으로 비디오의 모델인 **확률적 오토마톤(Probabilistic Automaton, PA)**을 구축한다.

- **상태 정의:** 원자 명제 집합 $P$의 모든 가능한 조합($2^{|P|}$개)에 대해 상태를 생성한다.
- **전이 확률:** 이전 상태에서 현재 프레임 $F_t$의 상태 $\omega$로 전이될 확률 $P_{t, \omega}$를 다음과 같이 계산한다.
$$P_{t, \omega} = \prod_{p_k \in P} g(f_v(F_t, p_k); \gamma_{fp}, \gamma_{tp}) \quad (\text{부정 명제 } \neg p_k \text{의 경우 } 1 - g \text{ 사용})$$
이 과정을 통해 비디오의 시간적 진화 과정을 확률적 상태 전이 모델로 추상화한다.

### 4. Model Checking (모델 체킹)

구축된 오토마톤이 사용자의 시간 논리 사양 $\Phi$를 만족하는지 확인하기 위해 확률적 모델 체커인 **STORM**을 이용한다. 이때 사양은 PCTL(Probabilistic Computation Tree Logic)로 표현된다. 오토마톤이 $\Phi$를 만족하면 해당 구간의 프레임들을 '관심 장면'으로 식별하며, 이후 오토마톤을 리셋하고 다음 프레임들을 처리한다.

## 📊 Results

### 실험 설정

- **데이터셋:** 합성 데이터셋(COCO, ImageNet 이미지 조합) 및 실제 자율주행 데이터셋(Waymo, NuScenes)으로 구성된 **TLV(Temporal Logic Video)** 데이터셋을 제안하고 사용하였다.
- **비교 대상:** Video-Llama, ViCLIP 및 LLM(GPT-3.5, GPT-4) 기반의 추론 벤치마크.
- **지표:** 정밀도(Precision), 재현율(Recall), F1-score를 사용하였다.

### 주요 결과

- **인지 모델의 영향:** Mask R-CNN과 YOLOv8x 등이 높은 성능을 보였으며, 이는 하위 인식 모델의 성능이 전체 시스템의 장면 식별 정확도에 직접적인 영향을 미침을 보여준다.
- **시간 논리 vs LLM 추론:** 단순한 이벤트(Always/Eventually A)에서는 두 방식 모두 우수했으나, 복잡한 멀티 이벤트 시나리오("A and B until C" 등)에서는 TL 기반 추론이 LLM 기반 방식보다 월등히 높은 성능을 기록하였다. 특히 Waymo와 NuScenes 데이터셋에서 F1-score가 9~15% 향상되었다.
- **비디오 길이에 따른 강건성:** LLM 기반 벤치마크(특히 GPT-4)는 비디오 길이가 1,000초를 넘어가면 성능이 급격히 하락하는 반면, NSVS-TL은 2,400초(40분) 이상의 긴 영상에서도 일관된 성능을 유지하였다.

## 🧠 Insights & Discussion

### 강점

본 연구는 인지와 추론을 분리하는 Neuro-Symbolic 구조를 통해, 기존 딥러닝 모델들이 해결하지 못한 **장기적 시간 의존성 문제**를 성공적으로 해결하였다. 특히 확률적 오토마톤과 형식 검증(Formal Verification)을 도입함으로써, 결과에 대한 수학적 근거를 제공하고 해석 가능성을 확보한 점이 매우 뛰어나다.

### 한계 및 비판적 해석

본 모델의 가장 큰 한계는 개별 프레임 기반의 인지에 의존한다는 점이다. "말에서 떨어지는 사람"과 같이 **여러 프레임에 걸쳐 일어나는 동적인 행위(Multi-frame semantics)**는 현재의 시스템으로는 포착할 수 없으며, 단순히 "말 위에 있는 사람"과 같은 정적인 상태의 조합으로만 이해할 수 있다. 따라서 향후 연구에서는 단일 프레임 인지를 넘어 짧은 비디오 클립 단위의 의미론적 이해를 통합하는 방향으로 확장되어야 할 것이다.

## 📌 TL;DR

본 논문은 비디오의 의미 이해(Neural Perception)와 시간적 추론(Symbolic Reasoning)을 분리한 **NSVS-TL** 프레임워크를 제안한다. VLM으로 프레임 내 객체를 인식하고, 이를 확률적 오토마톤으로 구축한 뒤 시간 논리(TL)를 통해 검증함으로써, 기존 LLM 기반 모델들이 실패하는 **초장기 비디오(40분 이상)의 복잡한 이벤트 검색**을 성공적으로 수행한다. 이 연구는 향후 자율주행 영상 분석이나 대규모 비디오 아카이브 검색 시스템의 신뢰성을 높이는 데 기여할 가능성이 크다.
