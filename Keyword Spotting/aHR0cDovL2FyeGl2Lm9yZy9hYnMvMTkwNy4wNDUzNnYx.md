# Multi-layer Attention Mechanism for Speech Keyword Recognition

Ruisen Luo, Tianran Sun, Chen Wang, Miao Du, Zuodong Tang, Kai Zhou, Xiaofeng Gong, and Xiaomei Yang (Preprint)

## 🧩 Problem to Solve

본 논문은 제한된 인프라와 계산 자원을 가진 환경(예: 차량 내 음성 명령 인식, 로봇 상호작용)에서 효율적으로 동작해야 하는 **자동 음성 키워드 인식(Automatic Speech Keyword Recognition, KWS)** 문제를 해결하고자 한다. 

기존의 메인스트림 방식은 Attention mechanism이 결합된 Long Short-Term Memory (LSTM) 네트워크를 사용한다. 그러나 전통적인 Attention 구조는 일반적으로 네트워크의 마지막 레이어 출력값만을 사용하여 Attention weight를 계산한다. 이 과정에서 특성 추출(Feature Extraction) 단계인 CNN 레이어 등을 거치며 필연적으로 발생하는 정보 손실로 인해, 계산된 Attention weight가 편향(biased)되거나 부정확해지는 문제가 발생한다. 따라서 본 논문의 목표는 특성 추출 단계와 LSTM의 중간 단계 정보를 모두 활용하는 **Multi-layer Attention Mechanism**을 통해 더 정확한 Attention weight를 산출하고 키워드 인식 성능을 향상시키는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 Attention weight를 계산할 때 마지막 레이어의 정보만 사용하는 것이 아니라, **특성 추출 전 단계와 LSTM의 각 레이어에서 발생하는 정보를 단계적으로 통합하여 활용**하는 것이다. 

단일 레이어 기반의 Attention이 가지는 정보 왜곡 문제를 해결하기 위해, 입력 데이터(MFCC)부터 CNN 출력, LSTM의 중간 레이어, 그리고 최종 레이어에 이르기까지 여러 수준의 출력을 시너지 효과(synergy) 있게 결합함으로써, 모델이 더 정밀하고 집중된 영역을 파악할 수 있도록 설계하였다.

## 📎 Related Works

논문에서는 다음과 같은 기존 연구들을 언급한다.
- **GMM-HMM 및 DNN**: DNN의 특성 추출 능력을 GMM-HMM에 결합하여 에러율을 낮춘 연구들이 존재한다.
- **LSTM 및 Bi-directional LSTM**: 순환 신경망의 망각 문제를 해결하기 위해 LSTM 유닛을 도입하였으며, 특히 Bi-directional LSTM은 음성 인식 성능을 크게 향상시켰다.
- **CNN 기반 인식**: CNN과 가중치 공유(weight sharing)를 통해 다중 키워드 인식 정확도를 높인 사례가 있다.
- **RHN (Recurrent Highway Networks)**: Jump connection을 통해 그래디언트 폭주 및 소실 문제를 완화한 깊은 LSTM 구조가 제안되었다.
- **Attention Mechanism**: 긴 시퀀스 입력에서 순차적 특성 기억의 강건성을 높이기 위해 Attention을 도입한 연구들이 있으며, 특히 키워드 인식에 전용 Attention을 적용한 사례([8])가 있다. 

본 논문은 이러한 기존 Attention 기반 모델들이 단일 레이어의 출력에만 의존한다는 한계를 지적하며, 이를 다층 구조로 확장하여 차별성을 둔다.

## 🛠️ Methodology

### 1. 데이터 전처리: MFCC
음성 샘플은 표준 음성 인식 절차에 따라 **Mel-frequency Cepstral Coefficients (MFCC)**로 전처리된다. 과정은 다음과 같다.
1. 원본 오디오에 Framing, Pre-emphasis, Windowing 적용.
2. Fourier Transform(푸리에 변환) 수행.
3. Mel Filter Banks를 통해 신호를 재처리.
4. Discrete Cosine Transform (DCT)을 통해 최종 MFCC 추출.

이때, Mel spectrum의 계산식은 다음과 같이 정의된다.
$$\text{MELSPEC}(M) = \sum_{k=f(m-1)}^{f(m+1)} \frac{\text{mask}_m(k) * |X(k)|^2}{f(m+1)}$$
여기서 $\text{mask}_m(k)$는 Mel frequency cepstrum의 마스크이며, $X(k)$는 신호의 푸리에 변환 결과, $f(m)$은 삼각형 밴드패스 필터 함수이다.

### 2. Multi-layer Attention Mechanism 구조
제안된 모델은 정보 손실을 최소화하기 위해 다음과 같은 단계적 융합(fusion) 과정을 거쳐 Attention weight를 생성한다.

1. **1차 융합**: MFCC(입력 정보)와 CNN 레이어의 출력 벡터를 내적(dot product)하고, 이를 Fully Connected (FC) 레이어에 통과시킨다.
2. **2차 융합**: 1차 융합 결과물과 LSTM의 중간 레이어(intermediate layer) 출력을 내적하고, 다시 FC 레이어에 통과시킨다.
3. **3차 융합**: 2차 융합 결과물과 LSTM의 최종 레이어 출력을 내적하여 최종 Attention parameter를 얻는다.

최종적으로 산출된 Attention weight는 LSTM의 출력 레이어에 할당되어 가중치가 적용된 기억 정보(memory information)를 생성하며, 이는 최종 FC 레이어로 전달되어 키워드를 분류한다.

### 3. 학습 절차
- **최대 Epoch**: 40 / **Batch Size**: 64.
- **Optimizer**: Learning rate attenuation이 적용된 Adam Optimizer를 사용한다.
- **조기 종료 (Early Stopping)**: 그래디언트 폭주 또는 무의미한 계산을 방지하기 위해 Early stopping 전략을 사용하며, 가장 성능이 좋은 모델을 저장한다.

## 📊 Results

### 1. 실험 설정
- **데이터셋**: Google Speech Command datasets V2 (20개 키워드).
- **데이터 분할**: 학습 세트(84,849개), 검증 세트(9,981개), 테스트 세트(1,105개)로 약 8:1:1 비율로 분할하였다.
- **비교 대상**: $\text{CNN} \rightarrow \text{Bi-LSTM} \rightarrow \text{Attention-based LSTM (Single-layer)} \rightarrow \text{Proposed Multi-layer Attention LSTM}$.

### 2. 주요 결과
- **CNN 기반 모델**: 기본 성능을 보였으나 후속 모델들에 비해 정확도가 낮다.
- **Bi-directional LSTM**: 테스트 세트에서 $94.34\%$의 정확도를 기록하며 CNN보다 월등한 성능을 보였으나, LSTM 유닛의 과도한 누적으로 인해 학습 속도가 느려지거나 성능 저하(degradation) 문제가 발생할 가능성이 확인되었다.
- **Attention-based LSTM (Single-layer)**: 수렴 속도는 빨라졌으나, 인식 정확도의 향상은 미미했다. 특히 특정 개별 단어에 대한 인식률이 낮게 나타나는 한계가 있었다.
- **제안 모델 (Multi-layer Attention)**: 
    - **전체 인식률**: $95.07\%$를 달성하여 기존 Attention 모델보다 $0.7\%$ 향상되었다.
    - **단어별 분석**: 'down' 명령(89.2%)을 제외한 나머지 모든 키워드에서 $93\%$ 이상의 높은 인식률을 보였다.
    - **안정성**: 다른 테스트 세트에서도 성능이 비교적 일정하게 유지되는 안정성을 보였다.

## 🧠 Insights & Discussion

본 논문은 Attention mechanism의 입력 소스를 단순화된 최종 출력에서 다층적 구조로 확장함으로써, 특성 추출 과정에서 발생하는 정보 손실을 효과적으로 보완할 수 있음을 증명하였다. 특히, 입력 데이터의 원시 정보(MFCC)와 중간 단계의 특징들이 직접적으로 Attention 계산에 참여함으로써, 모델이 더 정확하게 중요한 음성 구간에 집중할 수 있게 되었다.

**비판적 해석 및 한계점:**
1. **성능 향상폭의 미미함**: 전체 정확도 향상치가 $0.7\%$ 수준으로 수치상으로는 작다. 다만, 개별 단어의 인식률을 상향 평준화시켰다는 점이 실용적 가치로 평가된다.
2. **계산 복잡도**: 다층적으로 내적과 FC 레이어를 추가했으므로, 단일 레이어 Attention 모델에 비해 연산 비용이 증가했을 가능성이 크나 이에 대한 정량적 분석(FLOPs나 추론 시간 비교)이 부족하다.
3. **데이터셋 한정**: Google Speech Command V2라는 단일 데이터셋에서만 실험이 진행되어, 다양한 소음 환경이나 다른 언어셋에서의 범용성은 검증되지 않았다.

## 📌 TL;DR

본 논문은 음성 키워드 인식에서 Attention weight의 편향 문제를 해결하기 위해, MFCC $\rightarrow$ CNN $\rightarrow$ LSTM 중간층 $\rightarrow$ LSTM 최종층의 정보를 순차적으로 융합하는 **Multi-layer Attention Mechanism**을 제안한다. 실험 결과, Google Speech Command V2 데이터셋에서 $95.07\%$의 정확도를 달성하며 기존 모델들보다 높은 성능과 안정적인 단어별 인식률을 보였다. 이 연구는 향후 자원 제한적 환경에서의 고성능 키워드 스포팅(Keyword Spotting) 시스템 설계에 기여할 수 있다.