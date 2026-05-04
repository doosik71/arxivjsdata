# MFA-KWS: Effective Keyword Spotting with Multi-head Frame-asynchronous Decoding

Yu Xi, Haoyu Li, Xiaoyu Gu, Yidi Jiang, and Kai Yu (2025)

## 🧩 Problem to Solve

본 논문은 음성 기반 인터페이스의 핵심 기술인 Keyword Spotting (KWS) 시스템의 정확도와 효율성을 동시에 개선하는 것을 목표로 한다. 전통적인 ASR(Automatic Speech Recognition) 기반 KWS 방법론인 Greedy search나 Beam search는 전체 탐색 공간을 탐색하므로, 키워드 검출이라는 특수한 목적에 우선순위를 두지 않아 성능이 최적화되지 않는 경향이 있다.

또한, 기존의 Transducer 기반 시스템은 현재 프레임의 예측이 이전 예측 결과에 의존하는 자기회귀(Autoregressive) 특성 때문에 에러 누적(Error accumulation) 문제가 발생하며, 이는 특히 소음이 심한 환경이나 임의의 키워드를 검출해야 하는 복잡한 상황에서 성능 저하를 야기한다. 따라서 본 연구는 온디바이스(On-device) 배포가 가능할 만큼 가볍고 효율적이면서도, 노이즈에 강건하고 정확한 키워드 전용 디코딩 프레임워크를 구축하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **CTC(Connectionist Temporal Classification)**와 **Transducer**의 상호 보완적인 특성을 결합한 **Multi-head Frame-asynchronous (MFA)** 디코딩 구조를 제안하는 것이다.

1.  **Joint CTC-Transducer 학습**: CTC의 조건부 독립성(Condition-independence)을 활용해 Transducer의 에러 누적 문제를 완화하고 학습 수렴 속도를 높였다.
2.  **Frame-asynchronous 디코딩**: CTC 브랜치에는 **PSD (Phone-synchronous Decoding)**를, Transducer 브랜치에는 **TDT (Token-and-Duration Transducer)**를 적용하여 불필요한 프레임을 건너뛰고 연산 효율을 극대화하였다.
3.  **새로운 스코어 융합 전략**: 두 브랜치에서 비동기적으로 생성된 점수를 효과적으로 결합하기 위해 **CDC-Last (Cross-layer Discrimination Consistency)**를 포함한 다양한 융합 전략을 제안하였다.

## 📎 Related Works

기존의 KWS 접근 방식은 크게 두 가지로 나뉜다.
- **패턴 매칭 및 다중 라벨 분류**: 엔드-투-엔드(E2E) 방식으로 단순하지만, 데이터 민감도가 높고 소음 환경에서 강건성이 떨어진다는 한계가 있다.
- **ASR 기반 프레임워크**: 음향 모델 학습 후 디코딩 알고리즘을 통해 키워드를 찾는 방식이다. WFST(Weighted Finite-State Transducer) 기반 방식은 구현과 유지보수가 복잡하며, CTC나 RNN-T 기반의 일반적인 ASR 디코딩은 키워드 특화 우선순위가 없어 성능이 최적화되지 않는다.

본 논문은 저자들의 이전 연구인 TDT-KWS를 확장하여, 단일 브랜치의 한계를 극복하기 위해 CTC와 Transducer를 결합한 멀티태스크 학습 및 멀티헤드 디코딩 방식을 채택함으로써 기존 방식과 차별화를 꾀하였다.

## 🛠️ Methodology

### 1. 전체 시스템 구조 및 학습 목표
MFA-KWS는 공유 인코더(Shared Encoder)를 기반으로 CTC 헤드와 Transducer 헤드가 병렬로 구성된 구조이다. 인코더로는 가벼운 **DFSMN (Deep Feedforward Sequential Memory Network)**을 사용한다.

학습 시에는 두 손실 함수를 결합한 Joint loss를 사용하여 최적화한다.
$$L_{MFA} = L_{TDT}(x, y) + \alpha L_{CTC}(x, y)$$
여기서 $\alpha$는 CTC의 영향력을 조절하는 계수이며, 기본값으로 0.3이 설정되었다.

### 2. 주요 구성 요소 및 메커니즘

#### (1) Token-and-Duration Transducer (TDT)
기존 RNN-T가 토큰 $v$의 확률만 예측했다면, TDT는 토큰의 지속 시간(Duration) $d$를 함께 예측한다.
$$P(v, d|t, u) = P^T(v|t, u) P^D(d|t, u)$$
이를 통해 예측된 지속 시간만큼 프레임을 건너뛸 수 있어 추론 속도가 향상된다.

#### (2) Phone-synchronous Decoding (PSD)
CTC 브랜치에서는 블랭크(Blank) 확률이 임계값 $\lambda_\phi$보다 높은 프레임을 무시하는 PSD를 적용한다. CTC의 출력 특성인 'Peaky property'(확률 값이 특정 토큰에 매우 집중되는 현상)를 이용하여, 정보량이 적은 블랭크 프레임을 건너뜀으로써 연산량을 줄인다.

#### (3) Multi-head Joint Decoding 및 융합 전략
두 브랜치가 서로 다른 프레임을 건너뛰기 때문에, 시간축 상의 불일치를 해결하기 위해 **Placeholder (PH)** 상태를 도입하고 다음과 같은 융합 전략을 사용한다.
- **Domination 전략 (CTC-Dom, Transducer-Dom)**: 한쪽 브랜치의 점수를 우선시하고, 해당 브랜치가 PH일 때만 다른 쪽 점수를 사용한다.
- **Equivalence-Dom**: 두 브랜치의 점수를 동일하게 취급하여 평균을 낸다.
- **CDC (Cross-layer Discrimination Consistency)**: 슬라이딩 윈도우 내에서 두 브랜치 점수 추세의 코사인 유사도($w^{CDC}_t$)를 계산하여 가중합을 구한다.
  $$S^{Fused}_t = \frac{S^{Trans}_t + w^{CDC}_t \cdot S^{CTC}_t}{1 + w^{CDC}_t}$$
- **CDC-Last**: PH 발생 시 가장 최근의 유효한 점수로 채워 넣는(Padding) 방식으로, 본 논문에서 가장 우수한 성능을 보였다.

## 📊 Results

### 1. 실험 설정
- **데이터셋**: Hey Snips (영어 고정 키워드), MobvoiHotwords (중국어 고정 키워드), LibriKWS-20 (임의 키워드 20종) 및 소음 평가를 위한 WHAM! 데이터셋.
- **지표**: FAR (False Alarm Rate) 대비 Recall(재현율) 및 Macro-accuracy.
- **모델 크도**: 약 3.3M 파라미터의 경량 모델.

### 2. 주요 결과
- **정확도**: Hey Snips 데이터셋에서 FAR=0.05/h 기준 Recall 99.80%를 달성하며 SOTA 성능을 기록하였다. 중국어 MobvoiHotwords에서도 기존 모델 대비 Miss rate를 최대 68%까지 줄였다.
- **임의 키워드 검출**: LibriKWS-20 평가에서 MFA-KWS는 평균 정확도 83.50%를 기록하며, 단일 CTC나 Transducer 기반 스트리밍 디코딩보다 우수한 성능을 보였다.
- **소음 강건성**: SNR(신호 대 잡음비) 변화 실험에서 MFA-KWS가 단일 브랜치 시스템보다 훨씬 안정적인 성능을 유지하였다. 특히 프레임 스킵 메커니즘이 소음 간섭을 줄여 성능을 높이는 것으로 분석되었다.
- **추론 효율성**: 프레임 동기 방식(MFS) 대비 **47%~63%의 속도 향상(1.47$\times$ ~ 1.63$\times$ speed-up)**을 달성하였다.

## 🧠 Insights & Discussion

### 강점
본 연구는 CTC의 '비자기회귀적 안정성'과 Transducer의 '문맥 모델링 능력'을 결합하여 KWS의 고질적인 문제인 에러 누적을 효과적으로 해결하였다. 또한, TDT와 PSD라는 두 가지 비동기 프레임 스킵 메커니즘을 통해 정확도 손실 없이 추론 속도를 획기적으로 높였다는 점이 매우 고무적이다. 특히 CDC-Last 융합 전략이 비동기 상태에서도 상태 정보를 유지하며 최적의 점수를 산출함을 입증하였다.

### 한계 및 논의사항
PSD의 성능은 블랭크 임계값 $\lambda_\phi$ 설정에 민감하게 반응한다. 실험 결과 약 35%의 프레임을 스킵할 때 효율과 성능의 균형이 가장 좋았으나, 이는 데이터셋마다 다를 수 있어 최적의 임계값을 찾는 추가적인 튜닝 과정이 필요하다. 또한, 본 논문은 3.3M의 매우 작은 모델을 사용했으므로, 더 큰 모델 구조에서도 동일한 효율성 증폭 효과가 나타날지는 명시되지 않았다.

## 📌 TL;DR

MFA-KWS는 CTC와 TDT(Token-and-Duration Transducer)를 결합한 멀티태스크 학습 프레임워크로, PSD(Phone-synchronous Decoding)와 TDT의 비동기 프레임 스킵 기능을 통해 **추론 속도를 최대 63% 향상**시키면서도 **SOTA 수준의 키워드 검출 정확도**를 달성하였다. 특히 CDC-Last 융합 전략을 통해 노이즈 환경에서도 강건한 성능을 보여, 실제 온디바이스 웨이크워드(Wake-word) 시스템 적용 가능성이 매우 높다.