# An Anchor-Free Detector for Continuous Speech Keyword Spotting

Zhiyuan Zhao, Chuanxin Tang, Chengdong Yao, Chong Luo (2022)

## 🧩 Problem to Solve

본 논문은 연속적인 음성 데이터 내에서 미리 정의된 키워드를 탐지하는 Continuous Speech Keyword Spotting (CSKWS) 문제를 해결하고자 한다. 일반적인 Keyword Spotting (KWS)은 크게 두 가지 방향으로 발전해 왔으나, 각각의 한계가 존재한다. 첫째, Trigger Word Detection은 주로 엣지 디바이스에서 단일 호출어(예: "Hey Siri")를 찾는 것에 집중한다. 둘째, Speech Command Recognition은 키워드가 독립적으로 존재하며 경계가 명확한 상황을 가정한다.

반면, CSKWS는 실제 회의 기록이나 금지어 필터링과 같이 여러 타겟 키워드가 등장하고, 주변 문맥(context)의 간섭이 심한 연속 음성 환경을 다룬다. 특히 기존 연구에서 사용된 데이터셋(예: FKD)은 개별 단어를 수집한 뒤 인위적으로 합성하여 연속 음성을 만들었기 때문에, 단순한 진본/합성 판별기만으로도 높은 성능을 낼 수 있다는 취약점이 있다. 따라서 본 논문의 목표는 실제 연속 음성 환경에서의 어려움을 해결할 수 있는 새로운 벤치마크 데이터셋을 구축하고, 이를 효과적으로 탐지할 수 있는 새로운 모델 구조를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 CSKWS를 단순한 분류(Classification) 문제가 아닌, 1차원 객체 탐지(1D Object Detection) 문제로 정의하는 것이다. 컴퓨터 비전의 Anchor-free 객체 탐지 방식에서 영감을 얻어, 고정된 앵커(Anchor) 없이 키워드의 중심 위치와 길이를 직접 예측하는 AF-KWS 모델을 제안한다.

가장 중요한 설계적 특징은 보조 클래스인 'unknown' 클래스를 도입한 것이다. 일반적인 객체 탐지에서는 정의되지 않은 객체를 모두 배경(Background)으로 처리하지만, 음성 데이터에서는 '정의되지 않은 다른 단어'와 '침묵 또는 소음'을 구분하는 것이 성능 향상에 필수적이라는 직관을 바탕으로 이를 설계에 반영하였다.

## 📎 Related Works

기존의 KWS 연구들은 주로 트리거 워드 탐지나 단순 명령 인식에 치중되어 있었다. 트리거 워드 탐지는 단일 단어 탐지에 최적화되어 있으며, 명령 인식은 키워드가 분리되어 있다는 가정하에 수행된다. 본 논문에서 언급한 FKD(Football Keyword Dataset)는 연속 음성을 다루려 했으나, 단일 단어 오디오를 합성하여 만들었기 때문에 실제 연속 음성의 특성을 완전히 반영하지 못했다는 한계가 있다.

AF-KWS는 이러한 기존 방식들이 연속 음성 내의 다중 키워드 탐지와 문맥 간섭 문제를 충분히 해결하지 못한다는 점을 지적하며, 이를 위해 CV 분야의 Anchor-free 탐지 기법을 1차원 음성 신호 영역으로 확장하여 차별성을 둔다.

## 🛠️ Methodology

### 전체 시스템 구조

AF-KWS는 단일 단계(Single-stage) 딥러닝 네트워크로 구성된다. 전체 파이프라인은 다음과 같다.

1. **입력**: 16,000Hz 모노 오디오를 입력받아 Short-Time Fourier Transform (STFT) 스펙트로그램을 생성한다.
2. **Backbone**: ResNet34를 사용하여 특징 맵 $F \in \mathbb{R}^{T \times (C+1) \times N_{ch}}$를 추출한다. 여기서 $T$는 시간 해상도, $N_{ch}$는 채널 수이다.
3. **Prediction Heads**: 추출된 특징 맵을 바탕으로 세 가지 헤드가 병렬적으로 작동한다.
    - **Heatmap Head**: 키워드의 중심 위치 확률 $\hat{Y} \in [0, 1]^{T \times (C+1)}$를 예측한다.
    - **Length Head**: 키워드의 길이 $\hat{L} \in \mathbb{R}^T$를 예측한다.
    - **Offset Head**: 중심 위치의 정밀도를 높이기 위한 오프셋 $\hat{O} \in \mathbb{R}^T$를 예측한다.

여기서 $C$는 정의된 키워드 수이며, $(C+1)$번째 클래스는 앞서 언급한 'unknown' 클래스이다.

### 학습 목표 및 손실 함수

모델은 지도 학습 방식으로 훈련되며, Forced Alignment 도구를 통해 얻은 정답(Ground Truth)을 사용한다.

1. **Heatmap Loss ($L_h$)**:
키워드 중심 위치에 가우시안 커널을 적용하여 smoothed heatmap $Y$를 생성한다. 손실 함수로는 Focal Loss를 사용하여 클래스 불균형 문제를 해결한다.
$$L_h = -\frac{1}{N} \begin{cases} \sum_{t,c} (1-\hat{Y}_{t,c})^\alpha \log(\hat{Y}_{t,c}) & \text{if } Y_{t,c}=1 \\ \sum_{t,c} (1-Y_{t,c})^\beta (\hat{Y}_{t,c})^\alpha \sum_{t,c} \log(1-\hat{Y}_{t,c}) & \text{otherwise} \end{cases}$$

2. **Length Loss ($L_{len}$)**:
중심 위치에서의 실제 길이와 예측 길이 사이의 $L_1$ loss를 사용한다.
$$L_{len} = \frac{1}{N} \sum_{i=1}^N ||\hat{L}_{t_i} - L_{t_i}||$$

3. **Offset Loss ($L_{offset}$)**:
중심 위치의 정밀한 보정을 위한 오프셋에 대해 $L_1$ loss를 사용한다.
$$L_{offset} = \frac{1}{N} \sum_{i=1}^N ||\hat{O}_{loc(w_i)} - O_{loc(w_i)}||$$

**전체 손실 함수**:
$$L_{cskws} = L_h + \lambda_{len}L_{len} + \lambda_{offset}L_{offset}$$
(설정값: $\lambda_{len} = 0.1, \lambda_{offset} = 1$)

### 추론 절차 (Inference)

추론 단계에서는 먼저 Heatmap $\hat{Y}$에서 각 클래스별 피크(Peak) 지점을 찾고, 점수가 높은 상위 $M$개의 위치를 선택한다. 선택된 각 위치 $t$에 대해 다음과 같이 최종 결과를 도출한다.

- **클래스**: $\hat{Y}_{t,c}$에서 가장 높은 점수를 가진 클래스 $c$
- **정밀 위치**: $loc_{pc} = t + \hat{O}_t$
- **길이**: $len = \hat{L}_t$
- **Bounding Box**: $[loc_{pc} - len/2, loc_{pc} + len/2]$

## 📊 Results

### 데이터셋 구성

논문에서는 두 가지 벤치마크 데이터셋을 제안하였다.

- **LibriTop-20**: LibriSpeech에서 유도되었으며, 2개 음절 이상의 가장 빈번한 단어 20개를 선정하였다. 총 256k의 발화와 530k의 키워드를 포함한다.
- **CMAK (Continuous Meeting Analysis Keywords)**: 회의 구조 분석을 위한 24개 키워드를 포함하며, 실제 녹음 데이터와 합성 데이터가 혼합된 하이브리드 데이터셋이다.

### 실험 결과 및 분석

AF-KWS의 성능을 측정하기 위해 mAP(mean Average Precision), FRR(False Rejection Rate), FAs per hour 등의 지표를 사용하였다. 비교 대상으로는 SOTA 트리거 워드 탐지 알고리즘인 DSTC-ResNet와 MHAtt-RNN에 슬라이딩 윈도우(Sliding-window) 모듈을 추가한 모델을 사용하였다.

- **정량적 결과**: Table 3에 따르면, AF-KWS는 mAP 0.860을 기록하여 DSTC-ResNet(0.398)과 MHAtt-RNN(0.426)을 압도하였다. 특히 AP@75 지표에서 매우 높은 성능(0.886)을 보였는데, 이는 AF-KWS가 위치 예측에서 매우 높은 정밀도를 가짐을 의미한다.
- **추론 속도**: RTF(Real Time Factor) 관점에서도 기존 방식과 유사하거나 경쟁력 있는 속도를 보여 실용성을 입증하였다.
- **Ablation Study**:
  - 'unknown' 클래스를 제거했을 때 AP@5가 0.952에서 0.867로 크게 하락하여, 배경 소음과 다른 단어를 구분하는 설계의 중요성이 확인되었다.
  - 동일한 Backbone을 사용하더라도 분류(Classification) 헤드를 사용한 모델보다 탐지(Detection) 헤드를 사용한 AF-KWS의 성능이 훨씬 뛰어났다.

## 🧠 Insights & Discussion

본 논문은 CSKWS가 단순한 KWS의 확장판이 아니라, 독자적인 특성을 가진 '탐지 문제'임을 입증하였다. 기존의 슬라이딩 윈도우 방식은 시간 해상도가 낮거나 연산 비용이 지나치게 높아지는 트레이드-오프가 존재하지만, Anchor-free 방식은 이를 효율적으로 해결하였다.

특히, 음성 신호의 특성상 '말소리가 들리지만 타겟 키워드는 아닌 상태'가 존재하는데, 이를 'unknown' 클래스로 명시하여 학습시킨 점이 모델의 강건성을 높이는 결정적인 요인이 되었다.

다만, CMAK 데이터셋의 경우 개인정보 보호 문제로 인해 실제 음성 데이터 전체를 공개하지 못하고 합성 데이터를 중심으로 공개한다는 점이 한계로 언급된다. 또한, 제안된 모델이 매우 다양한 소음 환경이나 다양한 언어의 연속 음성에서도 동일한 강건성을 유지하는지에 대한 추가 검증이 필요할 것으로 보인다.

## 📌 TL;DR

본 논문은 연속 음성 내 키워드 탐지(CSKWS)를 1차원 객체 탐지 문제로 재정의하고, 이를 해결하기 위한 **Anchor-free 검출기인 AF-KWS**를 제안하였다. 중심점, 길이, 오프셋을 직접 회귀하는 구조와 더불어 **'unknown' 클래스**를 도입하여 타겟 외 단어와 배경 소음을 구분함으로써 성능을 극대화하였다. 또한 LibriTop-20과 CMAK라는 새로운 벤치마크 데이터셋을 통해 그 효용성을 입증하였으며, 이는 향후 회의 분석 및 음성 필터링 시스템의 기초 연구에 중요한 역할을 할 것으로 기대된다.
