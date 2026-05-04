# Feature learning for efficient ASR-free keyword spotting in low-resource languages

Ewald van der Westhuizen, Herman Kamper, Raghav Menon, John Quinn, Thomas Niesler (2021)

## 🧩 Problem to Solve

본 논문은 언어 자원이 매우 부족한(severely under-resourced) 환경에서 효율적인 키워드 탐지(Keyword Spotting, KWS)를 수행하기 위한 특징 학습(Feature Learning) 방법을 다룬다. 특히 아프리카 지역의 인도적 구호 프로그램을 지원하기 위해, 전사 데이터(transcribed speech data)가 거의 없는 상황에서 특정 키워드를 빠르게 탐지해야 하는 문제를 해결하고자 한다.

일반적인 키워드 탐지 시스템은 자동 음성 인식(Automatic Speech Recognition, ASR)에 의존하여 음성을 텍스트로 변환한 뒤 키워드를 검색한다. 그러나 ASR 시스템을 구축하기 위해서는 대량의 전사된 데이터가 필요하며, 자원이 부족한 언어의 경우 이러한 코퍼스를 구축하는 데 막대한 시간과 비용, 그리고 언어학적 전문 지식이 요구된다는 치명적인 한계가 있다. 따라서 본 연구의 목표는 대규모 전사 데이터 없이, 소량의 키워드 템플릿과 전사되지 않은 도메인 내 음성 데이터만을 활용하여 ASR-free 방식의 효율적인 키워드 탐지 시스템을 구축하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 계산 비용이 높은 Dynamic Time Warping(DTW)의 성능을 유지하면서, 추론 속도가 빠른 Convolutional Neural Network(CNN)의 효율성을 결합하는 **CNN-DTW** 프레임워크를 제안하는 것이다.

구체적으로는, 소량의 키워드 템플릿을 사용하여 대규모의 전사되지 않은 데이터에 대해 DTW 정렬 점수를 계산하고, 이 점수를 소프트 라벨(soft label)로 활용하여 CNN을 학습시킨다. 이를 통해 CNN이 DTW의 스코어링 동작을 모사하도록 하여, 실제 적용 단계에서는 무거운 DTW 정렬 과정 없이 CNN만으로 빠르게 키워드를 탐지할 수 있게 한다.

또한, 성능 극대화를 위해 다음과 같은 계층적 특징 학습 파이프라인을 제안한다:
1. **Multilingual Bottleneck Features (BNF)**: 자원이 풍부한 타 언어들로부터 일반적인 음성 특징을 추출한다.
2. **Autoencoder (AE)**: 도메인 내 전사되지 않은 데이터를 통해 비지도 학습 방식으로 특징을 정제한다.
3. **Correspondence Autoencoder (CAE)**: 소량의 레이블된 키워드 데이터를 활용하여, 동일 키워드의 서로 다른 발화 간의 공통점을 학습함으로써 화자나 채널의 영향을 제거하고 단어의 정체성(identity)을 강화한다.

## 📎 Related Works

기존의 ASR-free 키워드 탐지는 주로 Query-by-Example(QbE) 방식의 DTW를 사용하였다. DTW는 적은 양의 템플릿만으로도 매칭이 가능하다는 장점이 있지만, 모든 검색 대상과 템플릿을 일일이 정렬해야 하므로 계산 복잡도가 매우 높아 대규모 실시간 응용에 부적합하다.

또한, "Zero-resource" 음성 처리 분야에서는 전사 데이터 없이 강건한 특징 표현을 학습하려는 시도가 많았다. 특히 다양한 언어로 학습된 BNF 추출기는 MFCC보다 우수한 성능을 보인다는 점이 알려져 있다. 본 논문은 이러한 BNF의 범용성과 CAE의 도메인 특화 학습을 결합하여, 기존의 단순 BNF나 MFCC 기반 접근 방식보다 더욱 정교한 특징 표현을 생성함으로써 DTW 및 CNN-DTW의 성능을 끌어올린다.

## 🛠️ Methodology

### 1. Keyword Spotting Approaches

본 논문은 세 가지 KWS 접근 방식을 비교한다.

**A. CNN Keyword Classifier (Baseline)**
소량의 레이블된 키워드 템플릿을 사용하여 직접적으로 키워드 유형을 분류하는 지도 학습 모델이다. 입력은 $M \times D$ 차원의 특징 행렬이며, 최종 출력층은 Softmax를 통해 $K$개 키워드 중 하나를 분류한다. 손실 함수로는 Categorical Cross Entropy를 사용한다:
$$\ell = -\log f_i(X_{i,j})$$

**B. DTW Keyword Classifier (Topline)**
템플릿과 검색 음성 간의 코사인 유사도를 기반으로 DTW 정렬 비용을 계산한다. 모든 템플릿과 윈도우에 대해 가장 높은 유사도 점수 $\hat{J}_i$를 추출하여 키워드 존재 여부를 결정한다.

**C. CNN-DTW Keyword Classifier (Proposed)**
DTW의 점수를 타겟으로 하여 CNN을 학습시키는 방식이다. 전사되지 않은 데이터 $Y$에 대해 DTW로 계산된 점수 벡터 $\hat{J} = [\hat{J}_1, \hat{J}_2, \dots, \hat{J}_K]$를 정답으로 두고, CNN의 출력 $g(Y)$가 이를 예측하도록 학습한다. 최종 출력층은 Sigmoid를 사용하며, 손실 함수로는 Summed Cross-Entropy를 사용한다:
$$\ell = -\sum_{k=1}^{K} \{ \hat{J}_k \log g_k(Y) + (1 - \hat{J}_k) \log(1 - g_k(Y)) \}$$
이 구조는 $K$개의 이진 분류기가 입력층을 공유하는 형태와 같다.

### 2. Feature Extractors

**A. MFCC & BNF**
- **MFCC**: 13차원 MFCC에 $\Delta, \Delta\Delta$를 추가한 39차원 특징을 기본선으로 사용한다.
- **BNF**: 10개의 풍부한 언어 데이터로 학습된 TDNN(Time-Delay Neural Network)의 bottleneck layer에서 추출한 특징이다.

**B. Autoencoder (AE)**
비지도 학습 기반의 Stacked AE를 사용하여 입력 $x$를 재구성하도록 학습한다. 손실 함수는 $\|x - \hat{x}\|^2$이며, 중간의 39차원 hidden layer를 특징 추출기로 사용한다.

**C. Correspondence Autoencoder (CAE)**
AE를 사전 학습한 후, 동일한 키워드의 서로 다른 두 발화 $x^{(a)}$와 $x^{(b)}$를 DTW로 정렬하여 입력-출력 쌍으로 구성해 학습한다. 이를 통해 화자, 성별, 채널 등 부가적인 정보는 무시하고 단어의 정체성만을 캡처하도록 유도한다. 손실 함수는 다음과 같다:
$$\text{loss} = \|\hat{x} - x^{(b)}\|^2$$

### 3. Overall Pipeline
최종적으로 $\text{BNF} \rightarrow \text{AE} \rightarrow \text{CAE}$ 순으로 특징을 정제하여 $\text{CAE}_{\text{BNF}}$ 특징을 생성하고, 이를 CNN-DTW 모델의 입력으로 사용하여 학습 및 추론을 수행한다.

## 📊 Results

### 실험 설정
- **데이터셋**: South African English (SABN) 및 Luganda (우간다 언어).
- **지표**: AUC (Area Under the ROC Curve), EER (Equal Error Rate), P@10, P@N.
- **비교 대상**: MFCC, BNF, $\text{CAE}_{\text{MFCC}}$, $\text{CAE}_{\text{BNF}}$ 특징 조합과 CNN, DTW, CNN-DTW 모델.

### 주요 결과
1. **특징 학습의 효과**:
   - $\text{CAE}_{\text{BNF}}$ 특징이 모든 모델에서 가장 우수한 성능을 보였다.
   - 영어 개발 세트에서 $\text{CAE}_{\text{BNF}}$는 MFCC 대비 AUC가 약 18.9% 개선되었으며, P@10 및 P@N 지표에서 가장 가까운 경쟁자보다 최소 1.65배 높은 정밀도를 기록했다.
   - 특히 BNF를 CAE의 입력으로 사용했을 때, MFCC 기반 CAE보다 월등한 성능 향상이 나타났다.

2. **모델 성능 및 효율성**:
   - **정확도**: DTW가 가장 높은 성능(Topline)을 보였으나, $\text{CAE}_{\text{BNF}}$ 특징을 사용한 CNN-DTW는 DTW와 거의 대등한 수준의 성능을 달성했다. 반면 단순 CNN 분류기는 가장 낮은 성능을 보였다.
   - **계산 효율성**: 15초 음성 세그먼트 기준, CNN-DTW는 DTW보다 수천 배 이상 빠르다.
     - **CPU 실행 시간**: DTW ($\approx 227\text{s}$) $\gg$ CNN ($\approx 82.5\text{s}$) $\gg$ CNN-DTW ($\approx 164.74\text{ms}$).
     - **GPU 실행 시간**: CNN-DTW는 단 $3.40\text{ms}$ 만에 처리가 가능하여 실시간 응용에 매우 적합함을 입증했다.

## 🧠 Insights & Discussion

본 연구는 자원이 극도로 부족한 환경에서 **'외부의 풍부한 데이터(BNF) $\rightarrow$ 내부의 대량 무라벨 데이터(AE) $\rightarrow$ 내부의 소량 라벨 데이터(CAE)'**로 이어지는 특징 정제 과정이 매우 효과적임을 보여주었다. 이는 서로 다른 성격의 데이터 자원을 상호 보완적으로 활용하여 최적의 특징 표현을 학습할 수 있음을 시사한다.

특히, CNN-DTW 방식은 DTW의 데이터 효율성(소량의 템플릿으로 가능)과 CNN의 추론 효율성(빠른 속도) 사이의 트레이드오프를 성공적으로 해결했다. DTW 점수를 소프트 라벨로 사용하는 전략은 라벨 데이터가 부족한 상황에서도 CNN이 학습할 수 있는 충분한 감독 신호를 제공하는 효과적인 방법이었다.

한계점으로는, Luganda와 같은 실제 저자원 언어의 경우 영어보다 성능이 낮게 나타났는데, 이는 데이터의 노이즈와 전송 왜곡 등이 영향을 미친 것으로 보인다. 또한 하이퍼파라미터 최적화가 영어 데이터셋에서만 이루어졌음에도 Luganda에서 준수한 성능을 보였으나, 각 언어별 특화된 튜닝이 이루어진다면 더 높은 성능을 기대할 수 있을 것이다.

## 📌 TL;DR

본 논문은 전사 데이터가 없는 저자원 언어 환경을 위해, DTW의 스코어링을 모사하는 **CNN-DTW** 기반의 효율적인 키워드 탐지 시스템을 제안한다. 특히 **Multilingual BNF $\rightarrow$ AE $\rightarrow$ CAE**로 이어지는 특징 학습 파이프라인을 통해 화자 독립적인 강건한 특징을 추출함으로써, DTW 수준의 높은 정확도를 유지하면서도 추론 속도를 수천 배 이상 향상시켰다. 이 연구는 데이터가 극도로 제한된 환경에서 실시간 음성 모니터링 시스템을 구축하는 데 중요한 방법론을 제시한다.