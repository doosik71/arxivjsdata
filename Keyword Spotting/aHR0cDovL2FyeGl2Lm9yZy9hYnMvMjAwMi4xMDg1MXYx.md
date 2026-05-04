# Small-Footprint Open-Vocabulary Keyword Spotting with Quantized LSTM Networks

Theodore Bluche, Maël Primet, Thibault Gisselbrecht (2020)

## 🧩 Problem to Solve

본 논문은 저전력 IoT 기기와 같은 초소형 디바이스(micro-controllers)에서 동작 가능한 **Open-Vocabulary Keyword Spotting (KWS)** 시스템 구축을 목표로 한다.

기존의 대규모 단어 인식(LVCSR) 시스템은 성능은 뛰어나지만, 모델 크기가 보통 100MB를 초과하여 메모리와 계산 자원이 제한적인 소형 디바이스에 탑재하기 어렵고, 클라우드 의존성으로 인한 프라이버시 및 지연 시간 문제가 발생한다. 반면, 기존의 소형 KWS 모델들은 특정 키워드 세트에 대해서만 학습되는 Closed-Vocabulary 방식이어서, 사용자가 임의로 키워드를 정의하려면 모델을 다시 학습시켜야 하며 해당 키워드에 특화된 학습 데이터가 필요하다는 한계가 있다.

따라서 본 연구는 사용자가 별도의 재학습이나 추가 데이터 없이도 임의의 키워드를 정의하여 사용할 수 있으면서, 전체 모델 크기가 500KB 미만인 초경량 Open-Vocabulary KWS 시스템을 개발하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 제한된 자원 환경에서 유연하게 동작하는 KWS를 구현하기 위해 다음과 같은 설계 아이디어를 제안한 것이다.

1.  **양자화된 LSTM 네트워크**: 모델 크기를 극단적으로 줄이기 위해 가중치와 활성화 함수 값을 8비트로 양자화하는 전략을 도입하였다.
2.  **CTC 출력 특성에 최적화된 신뢰도 점수(Confidence Score)**: Connectionist Temporal Classification (CTC) 학습 네트워크의 출력 특성(Blank 클래스의 빈번한 예측 등)을 고려하여, 키워드 검출의 정확도를 높이는 **No-blank normalization** 기법을 제안하였다.
3.  **고속 디코딩 전략**: 연산량을 줄이기 위해 프레임 스킵(Frame skipping), 가지치기(Pruning), 그리고 Blank 확률이 높은 프레임을 무시하는 기법을 통해 추론 속도를 대폭 향상시켰다.

## 📎 Related Works

기존의 KWS 접근 방식은 크게 세 가지로 분류된다.

1.  **Closed-Vocabulary KWS**: 특정 키워드 데이터셋으로 학습하며 주로 Cross-entropy 손실 함수를 사용한다. 모델 크기는 작지만, 새로운 키워드를 추가하려면 재학습이 필수적이다.
2.  **Acoustic KWS (HMM 기반)**: 음소(Phone) 단위의 모델링을 통해 임의의 키워드 구성이 가능하며, 키워드 외의 음성을 처리하기 위해 **Filler model**을 사용한다. 하지만 HMM 기반 방식은 최신 딥러닝 방식에 비해 성능이 낮다.
3.  **End-to-End Open-Vocabulary KWS**: 오디오와 텍스트를 벡터 공간에 임베딩하여 거리 기반으로 검출하는 방식이다. 그러나 이러한 방식은 주로 단일 키워드 쿼리에만 적용 가능하며, 본 논문이 목표로 하는 '자연어 문장 내 여러 키워드 검출(mini-SLU)' 시나리오에는 적합하지 않다.

본 논문은 CTC 기반의 신경망을 활용하여 LVCSR의 유연성과 소형 KWS의 효율성을 동시에 달성하며, 특히 Filler model 방식보다 우수한 성능을 내는 것을 목표로 한다.

## 🛠️ Methodology

### 1. 전체 시스템 구조
시스템은 **MFCC 특징 추출 $\rightarrow$ 양자화된 LSTM 음향 모델 $\rightarrow$ CTC 기반 키워드 검출 $\rightarrow$ 후처리(Post-processing)** 단계로 구성된다.

### 2. Acoustic Model 및 양자화
- **아키텍처**: Stack&Skip LSTM 구조를 사용한다. 입력은 5개의 연속적인 MFCC 프레임 스택이며, 여러 층의 LSTM 레이어와 최종 Affine 레이어로 구성된다.
- **양자화 기법**: 모든 가중치와 활성화 값을 8비트로 양자화한다.
    - **가중치**: 학습 후(Post-training) 양자화를 수행하며, $[-8, +8]$ 범위로 클리핑한 후 2의 거듭제곱 경계값을 갖는 대칭 양자화를 적용한다.
    - **활성화 값**: 학습 중에 양자화(Quantization-aware training)를 수행한다. 특히 LSTM의 포화 활성화 함수 특성을 이용하여 입력과 출력 범위를 $[-4, +4]$ 및 $(-1, +1)$로 고정함으로써, 추론 시 룩업 테이블(Lookup table)을 사용할 수 있게 하여 속도를 높였다.
- **학습**: Librispeech 데이터셋을 사용하여 CTC 손실 함수로 학습한다. CTC는 입력 시퀀스와 타겟 시퀀스의 길이가 달라도 정렬(Alignment) 없이 학습 가능한 손실 함수이다.
$$\mathcal{L}_{CTC} = -\sum_{(x^{(i)}, \pi^{(i)}) \in \mathcal{D}} \log p(\pi^{(i)} | x^{(i)})$$

### 3. Keyword Spotting 메커니즘
음향 모델이 출력한 음소 확률 시퀀스에서 사용자가 정의한 키워드 음소 시퀀스를 탐색한다.

- **신뢰도 점수 계산**:
    - **Raw Confidence ($C_{raw}$)**: Viterbi 근사를 통해 계산하며, 세그먼트 길이가 길어질수록 확률 값이 급격히 낮아지는 문제가 있다.
    - **No-blank normalization ($C_{nb}$)**: CTC 네트워크가 Blank($\nu$)를 많이 예측하는 특성을 반영하여, 실제 유의미한 프레임 수(전체 길이 - Blank 예측 수)로 정규화한다.
    $$\log C_{nb}(k, t_s, t_e) = \frac{\log C_{raw}(k, t_s, t_e)}{\sum_{t=t_s}^{t_e} (1 - p(l_t = \nu | x))}$$
- **후처리(Post-processing)**:
    - **Greedy**: 가장 먼저 검출된 키워드를 선택하고 겹치는 후보를 제거한다.
    - **Sequence**: 겹치지 않는 키워드 시퀀스 중 누적 신뢰도 합이 최대가 되는 조합을 찾는다.

### 4. 디코딩 최적화
- **Boundaries Subsampling**: 세그먼트의 시작/끝 지점을 매 프레임이 아닌 3프레임 단위로 확인하여 연산량을 1/3로 줄인다.
- **Pruning**: 접두사(Prefix)의 확률이 일정 임계치 이하로 낮으면 해당 경로를 즉시 폐기한다.
- **Ignoring Blank Frames**: Blank 확률이 매우 높은(예: 0.95 이상) 프레임은 정보량이 적다고 판단하여 계산에서 제외한다. 실험 결과, 성능 저하 거의 없이 약 60%의 프레임을 제거할 수 있었다.

## 📊 Results

### 1. 실험 설정
- **데이터셋**: 스마트 조명(Smart lights) 및 세탁기(Washing machine) 시나리오를 위해 구축한 Crowd-sourced 데이터셋을 사용하였다. (Clean 및 Noisy 환경 각각 측정)
- **비교 대상(Baselines)**: 
    - LVCSR 기반(Viterbi, Lattice decoding)
    - Keyword-Filler 모델 (전통적인 KWS 방식)
    - 기타 CTC 기반 방식 (PSD-MED, Hwang et al.)
- **지표**: F1 Score 및 Exact Parse Rate(쿼리 내 모든 키워드 시퀀스가 정확히 검출된 비율)

### 2. 주요 결과
- **모델 크기 및 성능**: 5층 LSTM, 96유닛 모델(약 395KB)이 성능과 크기 면에서 가장 효율적이었다.
- **신뢰도 점수 영향**: $C_{nb}$ (No-blank normalization) 방식이 가장 변별력이 높았으며, 특히 Ratio 기반 점수보다 우수한 성능을 보였다.
- **후처리 비교**: Sequence 방식이 Greedy 방식보다 False Alarm을 효과적으로 제거하여 더 높은 정확도를 기록하였다.
- **최종 비교**: 제안 방법은 동일한 신경망을 사용한 Filler model 방식보다 우수한 성능을 보였으며, 일부 데이터셋에서는 훨씬 큰 모델인 TDNN-LSTM 기반 Filler model보다도 나은 결과를 보였다.

## 🧠 Insights & Discussion

본 연구는 모델의 크기를 극단적으로 줄이면서도 Open-Vocabulary 기능을 유지할 수 있음을 입증하였다. 

- **강점**: 500KB 미만의 모델로 마이크로 컨트롤러에서 실시간 동작이 가능하며, 사용자가 자유롭게 키워드를 설정할 수 있다는 점이 매우 실용적이다. 특히 CTC의 특성을 이용한 $C_{nb}$ 정규화와 Blank 프레임 제거 기법은 연산 효율과 정확도를 동시에 잡은 핵심 전략이다.
- **한계**: 양자화된 모델이 부동 소수점 모델에 비해 Clean 데이터셋에서 성능이 다소 하락하는 경향이 확인되었다. 이는 8비트 양자화로 인한 정밀도 손실이 영향을 미친 것으로 보인다.
- **비판적 해석**: LVCSR 베이스라인이 매우 낮은 성능을 보인 것은 소형 모델의 단어 인식률 자체가 낮기 때문이며, 이는 KWS 관점에서는 Filler model이나 제안 방법처럼 '특정 패턴'에 집중하는 접근 방식이 훨씬 효율적임을 시사한다.

## 📌 TL;DR

이 논문은 **8비트 양자화된 LSTM**과 **CTC 학습**을 결합하여 **500KB 미만**의 초소형 Open-Vocabulary KWS 시스템을 제안한다. 특히 CTC 출력의 특성을 반영한 **No-blank 신뢰도 정규화**와 **Blank 프레임 스킵** 기법을 통해, 추가 학습 없이도 사용자 정의 키워드를 효율적으로 검출할 수 있게 하였다. 이 연구는 클라우드 연결 없이 기기 자체에서 음성 명령을 처리해야 하는 초소형 IoT 기기의 SLU(Spoken Language Understanding) 구현에 중요한 기반을 제공한다.