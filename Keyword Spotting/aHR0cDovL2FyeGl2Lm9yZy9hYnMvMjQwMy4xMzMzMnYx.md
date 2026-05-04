# TDT-KWS: Fast and Accurate Keyword Spotting using Token-and-Duration Transducer

Yu Xi, Hao Li, Baochen Yang, Haoyu Li, Hainan Xu, Kai Yu (2024)

## 🧩 Problem to Solve

본 논문은 자원이 제한된 엣지 디바이스(edge devices)에서 동작하는 효율적인 키워드 검출(Keyword Spotting, KWS) 시스템 구축을 목표로 한다. 기존의 KWS 검색 알고리즘은 대부분 프레임 동기식(frame-synchronous) 접근 방식을 취하는데, 이는 대부분의 프레임이 키워드와 무관함에도 불구하고 매 프레임마다 검색 결정을 내려야 하므로 계산 낭비가 심하다는 문제가 있다.

또한, 기존의 Transducer 기반 KWS 시스템들은 단순히 일반적인 자동 음성 인식(ASR) 디코딩을 수행한 후 결과물에 키워드가 포함되어 있는지를 확인하는 방식을 사용한다. 이러한 방식은 검색 공간이 특정 키워드로 제한되지 않으며, KWS 작업의 특성에 맞게 최적화되지 않았기 때문에 효율성과 정확도 면에서 최적이 아니다. 따라서 본 연구의 목표는 Token-and-Duration Transducer(TDT)를 KWS에 도입하고, KWS 작업에 특화된 새로운 디코딩 알고리즘을 통해 추론 속도를 높이면서도 높은 정확도를 유지하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같다.

1.  **KWS 특화 디코딩 알고리즘 제안**: 스트리밍 연속 음성 환경에서 Transducer 모델이 키워드의 시작 지점을 동적으로 탐지할 수 있도록 검색 공간을 특정 키워드로 제한하는 효율적인 디코딩 알고리즘을 제안하였다.
2.  **TDT-KWS 시스템 구축**: TDT 모델을 KWS에 적용하여, 기존 RNN-T 기반 KWS 대비 유사하거나 더 뛰어난 성능을 달성함과 동시에 추론 속도를 2~4배 향상시켰다.
3.  **노이즈 강건성 입증**: 실험을 통해 TDT-KWS가 낮은 신호 대 잡음비(SNR) 환경에서 기존 RNN-T 시스템보다 더 강건한 성능을 보이며, 오경보(false alarm)를 줄이는 데 효과적임을 증명하였다.

## 📎 Related Works

기존의 KWS 연구들은 주로 RNN-T(Recurrent Neural Network Transducer)와 같은 Transducer 구조를 활용해 왔다. 특히 Tiny Transducer는 DFSMN(Deep Feed-forward Sequential Memory Network) 인코더와 stateless predictor, linear joiner를 사용하여 파라미터 수를 줄인 경량화 모델로, 많은 소규모 KWS 시스템의 기반이 되었다.

그러나 RNN-T의 자기회귀(auto-regressive) 디코딩은 계산 집약적이며 지연 시간이 발생한다는 한계가 있다. 최근 제안된 Token-and-Duration Transducer(TDT)는 토큰과 그 토큰이 지속되는 기간(duration)을 동시에 예측함으로써 추론 과정에서 불필요한 입력 프레임을 건너뛸 수 있게 하여 이를 해결하고자 하였다. 본 논문은 이러한 TDT의 특성을 KWS에 접목하여 프레임 비동기식(frame-asynchronous) 검색을 구현함으로써 효율성을 극대화하였다.

## 🛠️ Methodology

### 1. Token-and-Duration Transducer (TDT) 구조
기본적인 Transducer는 인코더, 예측기(predictor), 조이너(joiner)로 구성된다. TDT는 조이너의 출력 단계에서 토큰 $v$뿐만 아니라 해당 토큰의 지속 시간 $d$를 함께 예측한다. 이때 토큰과 지속 시간의 조건부 독립 가정을 통해 다음과 같이 결합 확률 분포를 정의한다.

$$P(v,d|t,u) = P^T(v|t,u)P^D(d|t,u)$$

여기서 $P^T(\cdot)$는 토큰 출력 분포, $P^D(\cdot)$는 지속 시간 출력 분포를 의미한다. 예측된 $d$ 값에 따라 디코딩 과정에서 입력 프레임을 건너뜀으로써 추론 속도를 획기적으로 높인다.

### 2. KWS 특화 스트리밍 디코딩 알고리즘
본 논문은 ASR과 달리 예측기(predictor)에 전체 가설을 넣지 않고, 오직 디코딩된 키워드 토큰 시퀀스만을 입력으로 제공한다. 이는 KWS의 목적이 전체 문장 생성보다는 특정 키워드의 존재 여부 판단에 있기 때문이다.

음성 특징을 $x = \{x_0, x_1, \dots, x_T\}$, 키워드 토큰 시퀀스를 $y = \{y_0=\phi, y_1, \dots, y_U\}$ ($\phi$는 blank 심볼)라고 할 때, 토큰 및 blank 방출 확률은 각각 다음과 같다.

$$y(t,u) = P(y_{u+1}|x_{[1:t]}, y_{[0:u]})$$
$$\phi(t,u) = P(\phi|x_{[1:t]}, y_{[0:u]})$$

이때 $\delta(t,u)$를 노드 $(t,u)$에 도달하는 경로 중 가장 높은 점수로 정의하며, 동적 계획법(Dynamic Programming)을 통해 다음과 같이 계산한다.

$$\delta(t,u) = \max(\delta(t,u-1) \cdot y(t,u-1), \delta(t-d,u) \cdot \phi(t-d,u))$$

특히, 스트리밍 환경에서 키워드가 언제든 시작될 수 있도록 모든 시점 $t$에 대해 $\delta(t,0) = 1$로 설정한다. 최종적으로 시점 $t$에서의 키워드 신뢰도(Confidence Score)는 다음과 같이 산출된다.

$$\text{Score}[t] = \delta(t,U) \cdot \phi(t,U)$$

TDT 모델의 경우, 위 과정에서 예측된 지속 시간 $d$를 사용하여 $\delta(t-d, u)$와 같이 프레임을 건너뛰며 계산을 수행한다.

## 📊 Results

### 1. 실험 설정
- **데이터셋**: Hey Snips (단일 키워드), LibriKWS-20 (LibriSpeech 기반 20개 키워드), WHAM! (배경 소음 데이터).
- **모델 구성**: 6개 DFSMN 레이어 인코더, stateless predictor, 약 2M 개의 파라미터 규모.
- **평가 지표**: 특정 오경보율(FAR)에서의 Macro-recall, 상대적 실행 속도 향상(Relative running speed-up), 상대적 검색 속도 향상(Relative search speed-up).

### 2. 주요 결과
- **디코딩 알고리즘 비교**: 제안된 KWS 특화 알고리즘이 기존 ASR 디코딩(Greedy/Beam Search)보다 월등히 높은 Recall을 기록하였다. 이는 검색 공간을 키워드로 한정함으로써 정확도가 향상되었음을 보여준다.
- **TDT vs RNN-T**: TDT-KWS는 RNN-T KWS와 대등하거나 더 높은 성능을 보이면서도 추론 속도는 2~4배 더 빨랐다. 특히 최대 건너뛰기 길이 $D_{max}$가 커질수록 속도 향상 폭이 증가하였다.
- **노이즈 강건성**: SNR이 낮아질수록(소음이 심해질수록) TDT-KWS와 RNN-T KWS의 성능 격차가 벌어졌으며, TDT-KWS가 훨씬 더 높은 Recall을 유지하였다.

## 🧠 Insights & Discussion

본 연구 결과는 TDT-KWS가 효율성과 정확성이라는 두 마리 토끼를 잡았음을 보여준다. 특히 $D_{max}$ 하이퍼파라미터에 따른 성능 변화가 흥미로운데, Hey Snips와 같이 단순한 키워드셋에서는 $D_{max}$가 커도 성능 저하가 거의 없었으나, LibriKWS-20과 같은 복잡한 데이터셋에서는 $D_{max}$가 너무 크면 필요한 음성 토큰을 건너뛰어 성능이 하락하는 경향을 보였다. 이는 모델이 유의미한 음성 정보와 불필요한 간섭 프레임을 구분하여 선택적으로 수용하는 능력을 갖추었음을 시사한다.

또한, 노이즈 환경에서 TDT가 더 강건한 이유는 프레임을 건너뛰는 메커니즘이 일종의 필터링 역할을 하여, 소음이 섞인 구간의 영향을 덜 받고 핵심적인 음성 특징에 더 집중할 수 있기 때문으로 해석된다. 다만, 본 논문에서는 고정된 키워드에 대한 검색만을 다루었으므로, 동적으로 변경되는 키워드나 오픈 보캐브러리(Open-vocabulary) 환경에서의 확장 가능성에 대해서는 추가적인 연구가 필요할 것으로 보인다.

## 📌 TL;DR

본 논문은 TDT(Token-and-Duration Transducer)를 이용한 효율적인 KWS 시스템인 **TDT-KWS**를 제안한다. 키워드 탐색 공간을 제한하는 전용 디코딩 알고리즘을 통해 **정확도를 높였으며**, 프레임 건너뛰기 기능을 통해 **추론 속도를 2~4배 향상**시켰다. 특히 **소음 환경에서의 강건성**이 기존 RNN-T 모델보다 뛰어나, 자원이 제한된 엣지 디바이스의 웨이크 워드(Wake-word) 검출 시스템에 매우 적합한 구조임을 입증하였다.