# VLM-GUARD: Safeguarding Vision-Language Models via Fulfilling Safety Alignment Gap

Qin Liu, Fei Wang, Chaowei Xiao, Muhao Chen (2025)

## 🧩 Problem to Solve

본 논문은 시각-언어 모델(Vision-Language Models, VLMs)에서 발생하는 안전성 정렬 격차(Safety Alignment Gap) 문제를 해결하고자 한다. 일반적으로 VLM은 텍스트 안전성 정렬이 완료된 거대 언어 모델(Large Language Models, LLMs)을 기반으로 구축된다. 그러나 텍스트 전용 모델에서는 안전하게 처리되던 쿼리가 시각적 모달리티(이미지)가 결합되는 순간, 모델의 안전 메커니즘이 무너지고 유해한 응답을 생성하는 현상이 발생한다.

연구진은 이러한 문제의 근본 원인을 Modality Gap으로 분석한다. 이는 공유 표현 공간(shared representation space) 내에서 이미지와 텍스트 표현이 서로 분리되어 존재하는 현상을 의미하며, 이로 인해 LLM에서는 명확했던 '유해한 쿼리'와 '무해한 쿼리'의 구분이 VLM에서는 모호해지게 된다. 심지어 의미 없는 빈 이미지(blank image)를 추가하는 것만으로도 안전 정렬이 깨질 수 있다는 점이 핵심적인 문제로 지적된다. 따라서 본 논문의 목표는 추론 단계에서의 개입을 통해 VLM의 안전성을 강화하고 LLM 수준의 안전 정렬을 회복하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 VLM의 기반이 되는 LLM 구성 요소가 이미 가지고 있는 안전 정렬 지식을 활용하여 VLM의 추론 과정을 감독하는 것이다. 이를 위해 **VLM-GUARD**라는 추론 시점 개입 전략을 제안한다.

VLM-GUARD의 중심 직관은 안전 정렬이 된 LLM에서 '안전성 조향 방향(Safety Steering Direction, SSD)'을 추출하고, VLM의 표현(representation)을 이 SSD와 직교하는 부분 공간(orthogonal subspace)으로 투영함으로써 시각적 모달리티가 안전성을 훼손하는 영향을 최소화하는 것이다. 또한, 유해한 의도가 감지된 쿼리의 경우 표현을 SSD의 반대 방향으로 이동시켜 모델이 거부 응답을 생성할 확률을 높이는 방식을 취한다.

## 📎 Related Works

기존의 VLM 안전성 강화 방법은 크게 두 가지 방향으로 나뉜다. 첫째는 지도 미세 조정(Supervised Fine-Tuning, SFT)이나 인간 피드백 기반 강화 학습(RLHF)과 같은 학습 단계의 정렬 방법이다. 둘째는 표현 공학(Representation Engineering)과 같은 추론 시점의 개입 방법이다.

기존 연구들은 주로 적대적 공격이나 탈옥(jailbreaking) 공격에 대응하는 방어책을 제시했으나, 본 논문은 특히 LLM과 VLM 사이의 '모달리티 격차'로 인해 발생하는 안전성 저하 현상에 집중한다. VLM-GUARD는 별도의 추가 학습 없이, 이미 정렬된 LLM의 내부 메커니즘을 VLM으로 전이시켜 안전성을 확보한다는 점에서 기존 접근 방식과 차별화된다.

## 🛠️ Methodology

VLM-GUARD의 전체 프로세스는 안전성 조향 방향 추출, 부분 공간 투영, 그리고 추론 시점 정렬의 세 단계로 구성된다.

### 1. 안전성 조향 방향(SSD) 추출

먼저, 모델의 거부 행동과 상관관계가 있는 유해성 특징을 포착하기 위해 LLM의 저차원 표현 공간을 앵커링한다. 100쌍의 유해한 쿼리($q^-$)와 무해한 쿼리($q^+$)로 구성된 앵커 데이터를 사용하여 $l$번째 레이어의 활성화 차이 행렬 $A^l$을 계산한다.

$$A^l = \begin{bmatrix} h^l(q^-_1), h^l(q^-_2), \dots, h^l(q^-_N) \end{bmatrix} - \begin{bmatrix} h^l(q^+_1), h^l(q^+_2), \dots, h^l(q^+_N) \end{bmatrix}$$

여기서 $h^l(\cdot)$은 $l$번째 레이어의 마지막 입력 토큰의 hidden state이다. 이 행렬 $A$를 특이값 분해(SVD)하면 다음과 같다.

$$A = U\Sigma V^T$$

이때 $V$의 상위 $m$개의 우측 특이 벡터(right singular vectors)를 해당 레이어의 안전성 조향 방향 $V_{m,l}$로 정의한다.

### 2. 부분 공간 투영 (Subspace Projection)

추출된 SSD를 바탕으로, VLM의 hidden state $h^l(q)$를 SSD와 직교하는 부분 공간으로 투영한다. 이를 통해 시각적 모달리티가 안전성 판단을 흐리는 영향을 제거한다. 투영된 상태 $h'^l(q)$는 다음과 같이 계산된다.

$$h'^l(q) = h^l(q) - h^l(q) V_{m,l}^T V_{m,l}$$

여기서 $V_{m,l}^T V_{m,l}$은 SSD가 생성하는 부분 공간으로의 직교 투영 행렬이다.

### 3. 추론 시점 정렬 (Inference-Time Alignment)

모든 입력에 개입하는 대신, 유해한 의도가 있는 입력에만 선택적으로 개입하기 위해 이진 게이트 $g^l$을 사용한다. 게이트는 $h^l(q)V_{1,l} > 0$일 때 활성화된다. 최종적으로 개입이 적용된 hidden state $h^*_l(Q)$는 다음과 같다.

$$h^*_l(Q) = h^l(Q) + \alpha \cdot g^l \cdot h^l(q) V_{m,l}^T V_{m,l}$$

여기서 $\alpha$는 개입 강도를 조절하는 하이퍼파라미터이며, $L_G$는 개입이 적용될 레이어의 집합이다.

## 📊 Results

### 실험 설정

- **대상 모델**: LLaVA-1.5-7B-HF
- **데이터셋**:
  - **MaliciousInstruct**: 10가지 유해 의도를 포함한 100개의 쿼리.
  - **Jailbreak Instructions**: 5가지 탈옥 프롬프트가 적용된 100개의 쿼리.
  - **MM-Harmful Bench**: 이미지와 텍스트가 결합되어야 응답 가능한 100개의 유해 쿼리.
- **측정 지표**:
  - **Attack Success Rate (ASR)**: 모델이 거부하지 못하고 유해한 응답을 생성한 비율 (LlamaGuard-7b로 판정).
  - **Perplexity (PPL)**: 응답의 품질과 유창성을 측정 (Llama-2-7b-chat 사용).

### 주요 결과

- **안전성 강화 효과**: VLM-GUARD는 모든 테스트 설정에서 Baseline(Self-Reminder, Goal Priority)보다 낮은 ASR을 기록하며 가장 강력한 방어 성능을 보였다.
- **안전성 정렬 격차 해소**: Vanilla LLaVA의 경우, 텍스트 전용 쿼리에서는 ASR이 15%였으나 빈 이미지를 추가하자 34%로 급증했다. 하지만 VLM-GUARD를 적용하면 이 격차가 크게 줄어들며 이미지 유무와 상관없이 낮은 ASR을 유지했다.
- **품질 유지**: PPL 측정 결과, VLM-GUARD는 Vanilla LLaVA와 유사한 수준의 언어 생성 능력을 유지했으며, 이는 모델이 단순히 헛소리를 하는 것이 아니라 적절하게 거부 응답을 생성하고 있음을 시사한다.

## 🧠 Insights & Discussion

본 연구는 VLM의 안전성 취약점이 단순히 데이터의 부족이 아니라, 모달리티 간의 표현 공간 차이(Modality Gap)에서 기인한다는 점을 실험적으로 보여주었다. 특히 빈 이미지라는 무의미한 입력만으로도 기존의 텍스트 안전 정렬이 무력화될 수 있다는 발견은 VLM 보안의 심각성을 일깨워준다.

VLM-GUARD의 강점은 추가적인 학습 비용 없이 LLM의 기존 지식을 활용해 실시간으로 개입한다는 점이다. 다만, 다음과 같은 한계점이 존재한다. 첫째, 추론 단계의 개입일 뿐 학습 단계에서의 근본적인 정렬 문제를 해결한 것은 아니다. 둘째, 안전성 외에 추론이나 이해 능력과 같은 일반적인 성능에 모달리티 격차가 어떤 영향을 주는지에 대해서는 추가 연구가 필요하다. 셋째, 실험에서는 빈 이미지를 주로 사용했으나, 실제 일반 이미지에서도 동일한 안전성 저하 현상이 지속된다는 점을 명시하고 있다.

## 📌 TL;DR

VLM-GUARD는 VLM이 이미지 입력 시 안전 정렬이 무너지는 '안전성 정렬 격차' 문제를 해결하기 위해, 기반 LLM에서 추출한 안전성 조향 방향(SSD)을 활용해 추론 시점의 표현을 조절하는 방법이다. 실험 결과, 탈옥 공격 및 유해 쿼리에 대해 매우 효과적인 방어 성능을 보였으며, 모델의 생성 품질을 유지하면서도 LLM 수준의 안전성을 회복할 수 있음을 증명했다. 이는 향후 멀티모달 모델의 실시간 보안 필터링 및 안전한 배포에 중요한 기여를 할 것으로 보인다.
