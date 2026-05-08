# HecVL: Hierarchical Video-Language Pretraining for Zero-shot Surgical Phase Recognition

Kun Yuan, Vinkle Srivastav, Nassir Navab, and Nicolas Padoy (2024)

## 🧩 Problem to Solve

전통적인 수술 단계 인식(Surgical Phase Recognition) 모델들은 특정 작업에 특화된 task-specific 모델 형태를 띠어 왔다. 이러한 방식은 수술 영상의 복잡성으로 인해 수많은 프레임에 대해 수동으로 레이블을 지정해야 하는 막대한 양의 주석(annotation) 작업이 필요하며, 이는 결과적으로 다른 수술 절차나 다른 의료 센터의 데이터에 적용했을 때 전이 가능성(transferability)이 떨어지는 문제를 야기한다.

본 논문의 목표는 수동 주석 없이도 다양한 수술 절차와 의료 센터에 적용 가능한 일반ist 수술 모델(generalist surgical model)을 구축하는 것이다. 이를 위해 자연어 텍스트를 감독 신호로 활용하여, 모델이 수술 영상의 계층적 구조를 이해하고 zero-shot 환경에서 수술 단계를 인식할 수 있도록 하는 것을 목표로 한다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 수술 영상과 텍스트 간의 관계를 세 가지 계층(clip, phase, video)으로 정의하고, 이를 통해 세밀한 동작부터 거시적인 수술 맥락까지 동시에 학습하는 **HecVL(Hierarchical Encoded Contrastive Video-language pretraining)** 프레임워크를 제안하는 것이다.

가장 중심적인 설계 직관은 서로 다른 입도(granularity)를 가진 텍스트 정보를 단일 임베딩 공간에 투영하는 대신, 각 계층별로 분리된 임베딩 공간을 구축함으로써 의미적 모호성을 제거하고 단기 및 장기 수술 개념을 효과적으로 인코딩하는 것이다.

## 📎 Related Works

기존의 수술 영상 분석 모델들은 대부분 수동으로 정의된 카테고리와 대량의 프레임 주석에 의존하는 task-specific 모델이었다. 최근 SurgVLP와 같은 연구가 수술 강의 영상의 오디오를 텍스트로 변환한 SVL 데이터셋을 통해 비디오-언어 사전 학습을 시도하였으나, 이는 주로 clip-level(짧은 구간)의 텍스트 쌍에만 의존한다는 한계가 있었다.

또한, 일반적인 컴퓨터 비전 분야의 CLIP과 같은 모델이나 최근 제안된 수술 foundation 모델들이 존재하지만, 전자는 수술 도메인의 특수성을 반영하지 못하며 후자는 여전히 하위 작업으로의 전이를 위해 파인튜닝(fine-tuning)이 필요하다는 차별점이 있다. HecVL은 이러한 한계를 극복하기 위해 계층적 구조의 텍스트 감독 신호를 도입하여 파인튜닝 없는 zero-shot 전이를 가능케 한다.

## 🛠️ Methodology

### 전체 파이프라인 및 시스템 구조

HecVL은 수술 강의 영상에서 추출한 세 가지 수준의 텍스트-비디오 쌍을 활용하여 시각 인코더 $F_v$와 텍스트 인코더 $F_t$를 학습시킨다. 전체 구조는 세밀한 수준에서 거친 수준으로 나아가는 **fine-to-coarse contrastive learning** 전략을 따른다.

### 계층적 비디오-텍스트 쌍 (Hierarchical Video-Text Pairs)

데이터셋 $D = \{(V_i, N_i, C_i, A_i)\}$는 다음과 같이 세 단계로 구성된다.

1. **Clip-level (Narration texts, $N_i$):** 수초 분량의 짧은 영상 클립과 이에 대응하는 음성 인식(ASR) 기반 내레이션 텍스트의 쌍이다. 원자적 동작(atomic actions)을 캡처한다.
2. **Phase-level (Concept texts, $C_i$):** 더 긴 영상 세그먼트와 수술 단계에 대한 개념적 요약 텍스트의 쌍이다. 고수준의 수술 활동을 캡처한다.
3. **Video-level (Abstract texts, $A_i$):** 전체 수술 영상과 수술의 목표 및 주요 포인트가 담긴 초록(abstract) 문단의 쌍이다. 전체적인 수술 맥락을 캡처한다.

### Fine-to-Coarse Contrastive Learning

모델은 각 계층에 대해 서로 다른 임베딩 공간 $S_{narration}, S_{concept}, S_{abstract}$를 구축한다.

- **Clip-level:** $F_v(v_{ij})$와 $F_t(n_{ij})$를 통해 $S_{narration}$ 공간에서 학습한다.
- **Phase-level:** 특정 단계에 해당하는 클립들의 시각/텍스트 임베딩 세트 $V_c, N_c$를 수집하고, 이를 aggregator 함수 $\text{Agg}(\cdot)$(평균 풀링)를 통해 집계하여 $S_{concept}$ 공간에서 학습한다.
- **Video-level:** 전체 영상에서 고르게 샘플링된 클립 세트 $V_a, N_a$를 집계하여 $S_{abstract}$ 공간에서 학습한다.

### 손실 함수 및 학습 절차

학습에는 $\text{InfoNCE}$ 손실 함수를 사용하여 매칭되는 쌍의 유사도는 높이고, 매칭되지 않는 쌍의 유사도는 낮춘다.

$$L_{phase} = -\frac{1}{B} \sum_{i=1}^{B} \log \left( \frac{\exp(\text{Agg}(F_v(V_c))^T \cdot F_t(c_i)/\tau)}{\sum_{j=1}^{B} \exp(\text{Agg}(F_v(V_c))^T \cdot F_t(c_j)/\tau)} + \frac{\exp(\text{Agg}(F_t(N_c))^T \cdot F_t(c_i)/\tau)}{\sum_{j=1}^{B} \exp(\text{Agg}(F_t(N_c))^T \cdot F_t(c_j)/\tau)} \right)$$

$$L_{video} = -\frac{1}{B} \sum_{i=1}^{B} \log \left( \frac{\exp(\text{Agg}(F_v(V_A))^T \cdot F_t(A_i)/\tau)}{\sum_{j=1}^{B} \exp(\text{Agg}(F_v(V_A))^T \cdot F_t(A_j)/\tau)} + \frac{\exp(\text{Agg}(F_t(N_A))^T \cdot F_t(A_i)/\tau)}{\sum_{j=1}^{B} \exp(\text{Agg}(F_t(N_A))^T \cdot F_t(A_j)/\tau)} \right)$$

여기서 $B$는 배치 크기, $\tau$는 온도 하이퍼파라미터이다. Clip-level 학습에는 SurgVLP에서 제안된 $L_{clip}$을 사용한다.

학습 시에는 **교차 학습 전략(alternating training strategy)**을 사용하여 $L_{clip}$을 $m$ 배치, $L_{phase}$를 $n$ 배치, $L_{video}$를 $l$ 배치만큼 순차적으로 반복 학습함으로써 치명적 망각(catastrophic forgetting) 문제를 방지하고 수렴 속도를 높였다.

## 📊 Results

### 실험 설정

- **데이터셋:** 사전 학습에는 SVL 데이터셋(25,578개 clip, 10,304개 phase, 1,076개 video 쌍)을 사용하였다. 하위 평가 작업으로는 Cholec80(담낭 절제술), AutoLaparo(자궁 절제술), StrasBypass70/BernBypass70(위 우회술) 데이터셋을 사용하였다.
- **평가 방식:** Zero-shot 설정으로, 클래스 레이블을 텍스트 프롬프트로 변환하여 시각 임베딩과 매칭하는 방식으로 수행하였다.
- **구현:** 시각 인코더로 ResNet-50, 텍스트 인코더로 BioClinicalBert를 사용하였다.

### 주요 결과

- **Zero-shot 성능:** HecVL은 모든 데이터셋에서 SOTA 성능을 달성하였다. 예를 들어, Cholec80에서 Top-1 정확도 41.7%, F1 스코어 26.3%를 기록하여 SurgVLP 및 CLIP 대비 우수한 성능을 보였다.
- **전이 가능성:** 서로 다른 수술 종류(담낭, 자궁, 위 우회술)에 대해서도 일관된 성능 향상을 보여, 학습된 표현이 일반화되었음을 입증하였다.
- **다기관 평가 (Multi-center):** 위 우회술 데이터셋에서 Strasbourg 병원과 Bern 병원 데이터를 비교했을 때 HecVL이 가장 우수했으나, Bern 병원의 성능이 상대적으로 낮게 나타났다. 이는 두 병원의 수술 워크플로우 차이로 인해 Strasbourg 기반의 텍스트 프롬프트가 완벽히 적용되지 않았기 때문으로 분석된다.

## 🧠 Insights & Discussion

### 강점 및 분석

HecVL은 수술 영상의 계층적 특성을 반영한 사전 학습을 통해, 추가적인 레이블링 없이도 새로운 수술 도메인에 적응할 수 있음을 보여주었다. 특히 Ablation study를 통해 phase-level 정보의 추가가 성능 향상에 크게 기여함을 확인하였으며, 단일 임베딩 공간(Single embedding space)을 사용하는 것보다 계층별로 분리된 공간을 사용하는 것이 의미적 모호성을 줄여 성능을 높인다는 점을 입증하였다.

### 한계 및 비판적 해석

본 연구의 가장 큰 한계는 zero-shot 성능이 텍스트 프롬프트의 설계에 의존한다는 점이다. 다기관 평가에서 나타난 성능 저하는 수술 센터마다의 프로토콜 차이가 크다는 것을 시사하며, 이를 해결하기 위해서는 센터별 특성이 반영된 텍스트 프롬프트 구축 전략이 추가로 필요할 것으로 보인다. 또한, 시각 인코더로 ResNet-50과 같은 비교적 오래된 아키텍처를 사용하였기에, 최신 Vision Transformer(ViT) 계열의 인코더를 적용했을 때의 성능 향상 가능성이 남아 있다.

## 📌 TL;DR

HecVL은 수술 영상의 내레이션(Clip), 개념 요약(Phase), 전체 초록(Video)이라는 **세 가지 계층적 텍스트 정보를 활용한 사전 학습 프레임워크**이다. 서로 다른 입도의 임베딩 공간을 분리하여 학습함으로써 수술의 단기 동작과 장기 맥락을 동시에 파악하며, 이를 통해 **수동 주석 없이도 다양한 수술 및 의료 센터에 적용 가능한 zero-shot 수술 단계 인식**을 구현하였다. 이 연구는 수술 AI 모델이 task-specific 모델에서 벗어나 일반ist 모델로 나아가는 중요한 방향성을 제시한다.
