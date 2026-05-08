# DeepPhase: Surgical Phase Recognition in CATARACTS Videos

Odysseas Zisimopoulos, Evangello Flouty, Imanol Luengo, Petros Giataganas, Jean Nehme, Andre Chow, and Danail Stoyanov (2018)

## 🧩 Problem to Solve

본 논문은 백내장 수술(Cataract Surgery) 비디오에서 수술 단계(Surgical Phase)를 자동으로 인식하는 시스템을 구축하고자 한다. 수술 워크플로우 분석의 자동화는 수술 절차의 표준화를 돕고, 수술 후 평가 및 인덱싱을 개선하며, 수술 중 실시간 모니터링을 통해 수술 팀의 협업과 환자의 안전을 증진시키는 데 매우 중요하다.

특히, 수술 단계 인식은 다음 단계의 예측이나 필요한 도구의 준비, 조기 경고 메시지 제공 등을 가능하게 하여 수술실 내의 효율성을 극대화할 수 있다. 또한, 의료진이 수동으로 보고서를 작성하는 부담을 줄이고 교육 목적으로 비디오를 효율적으로 인덱싱하는 기능을 제공하는 것을 목표로 한다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 수술 도구의 인식 결과가 수술 단계의 결정적인 단서가 된다는 직관에 기반하여, **CNN(Convolutional Neural Network)과 RNN(Recurrent Neural Network)을 결합한 2단계 파이프라인**을 설계한 것이다.

먼저 ResNet-152를 사용하여 비디오 프레임 내의 수술 도구를 검출하고, 여기서 추출된 도구 정보(이진 존재 여부 또는 특징 벡터)를 RNN의 입력으로 사용하여 시간적 흐름에 따른 수술 단계를 분류하는 구조이다. 이는 이미지의 정적 특징과 비디오의 시계열적 특성을 모두 활용하여 수술 워크플로우를 정밀하게 추론하려는 시도이다.

## 📎 Related Works

과거의 수술 단계 인식 연구들은 주로 외과의의 손 움직임이나 도구의 존재 여부를 모니터링하는 방식이었다. 초기에는 Random Forest나 Conditional Random Fields(CRF) 모델을 사용하여 도구 사용 패턴을 통해 단계를 인식하였다. 이후 시각적 특징(Visual Features)을 직접 활용하는 방식이 등장하였으나, 대부분 수작업으로 설계된(Hand-crafted) 특징에 의존하여 강건함(Robustness)이 떨어진다는 한계가 있었다.

최근에는 딥러닝의 발전으로 복강경 수술(Laparoscopy) 분야에서 EndoNet과 같은 모델이 제안되었다. EndoNet은 AlexNet을 특징 추출기로 사용하고 Hierarchical Hidden Markov Model(HHMM)을 통해 단계를 추론하는 구조를 가졌으며, 이후 LSTM 등으로 발전하였다. 하지만 이러한 접근 방식은 도메인 적응(Domain Adaptation) 문제와 환경 변화에 대한 회복력 문제로 인해 백내장 수술과 같은 다른 수술 절차로의 확장이 제한적이었다.

## 🛠️ Methodology

### 전체 시스템 구조

시스템은 크게 두 단계의 네트워크로 구성된다. 첫 번째 단계는 비디오 프레임에서 수술 도구를 인식하는 **Tool Recognition Network**이며, 두 번째 단계는 도구 정보를 바탕으로 수술 단계를 분류하는 **Phase Recognition Network**이다.

### 1. Tool Recognition (CNN)

수술 도구 인식을 위해 **ResNet-152** 아키텍처를 사용하였다. 이 모델은 50개의 residual block으로 구성되어 있으며, 다중 레이블 분류(Multi-label Classification)를 수행한다. 출력층에는 Sigmoid 활성화 함수를 사용하여 21종의 도구 각각에 대한 존재 확률을 계산한다.

학습을 위한 손실 함수로는 Sigmoid Cross-Entropy를 사용하며, 방정식은 다음과 같다.

$$L_{CNN} = -\frac{1}{N_t} \sum_{i=1}^{N_t} \sum_{c=1}^{C_t} p_{ic} \log \hat{p}_{ic} + (1 - p_{ic}) \log(1 - \hat{p}_{ic})$$

여기서 $p_{ic} \in \{0, 1\}$는 프레임 $i$에서 클래스 $c$의 정답 레이블이며, $\hat{p}_{ic} = \sigma(p_{ic})$는 예측값, $N_t$는 미니배치 내 총 프레임 수, $C_t = 21$은 전체 도구 클래스 수이다.

### 2. Phase Recognition (RNN)

수술 단계는 시간적 순서에 따라 전개되므로, temporal information을 캡처하기 위해 RNN 기반의 접근 방식을 사용하였다. RNN의 입력으로는 CNN에서 추출된 두 가지 형태의 정보를 사용한다.

1. **Binary Presence**: CNN 출력층의 도구 존재 여부(이진 값).
2. **Tool Features**: CNN의 마지막 pooling layer에서 추출된 특징 벡터(도구의 움직임, 방향, 조명, 색상 정보 포함).

본 연구에서는 **LSTM**(은닉층 1개, 256개 노드)과 **GRU**(은닉층 2개, 레이어당 128개 노드) 두 가지 모델을 실험하였다. 출력층은 14개의 수술 단계 클래스에 대해 Softmax 활성화 함수를 사용한다.

손실 함수로는 Cross-Entropy Loss를 사용하며, 정의는 다음과 같다.

$$L_{LSTM} = -\frac{1}{N_p} \sum_{i=1}^{N_p} \sum_{c=1}^{C_p} p_{ic} \log[\phi(p_{ic})], \quad \phi(p_c) = \frac{e^{p_c}}{\sum_{c=1}^{C_p} e^{p_c}}$$

여기서 $N_p$는 미니배치 크기, $C_p = 14$는 수술 단계 클래스 수이다.

### 추론 절차

추론 시에는 약 33초 분량에 해당하는 100개 프레임 배치를 구성하고, 이를 슬라이딩 윈도우(Sliding-window) 방식으로 입력하여 각 시점의 수술 단계를 분류한다.

## 📊 Results

### 실험 설정

- **데이터셋**: CATARACTS 데이터셋 (훈련/테스트 비디오 각 25개).
- **평가 지표**:
  - 도구 인식: Area Under the ROC Curve (AUC), Subset Accuracy (sAcc), Hamming Accuracy (hAcc).
  - 단계 인식: Per-frame Accuracy, Mean Class Precision/Recall, F1-score.

### 도구 인식 결과

ResNet-152는 매우 높은 성능을 보였으며, Hold-out 테스트 세트에서 **Hamming Accuracy 99.07%**, **AUC 99.59%**를 기록하였다. 특히 CATARACTS 챌린지 테스트 세트에서도 97.69%의 AUC를 달성하여 State-of-the-art 수준에 근접한 결과를 보였다. 주요 오류는 비디오 프레임의 품질 저하(Blurry frames)나 도구가 안구 표면에 닿지 않았을 때 발생하였다.

### 수술 단계 인식 결과

단계 인식 모델의 성능은 입력 데이터의 종류에 따라 차이를 보였다.

- **LSTM**: Binary input보다 **Tool features input을 사용했을 때 더 높은 성능**을 보였으며, 일반화 테스트 세트에서 **78.28%의 정확도**를 달성하였다.
- **GRU**: Binary input에서는 LSTM보다 우수한 성능(테스트 세트 89.85%)을 보였으나, Feature input에서는 오히려 성능이 하락하여 LSTM보다 낮은 결과를 나타냈다.

결과적으로, 단순한 도구 존재 여부 이상의 시각적 특징(Feature)들이 LSTM의 단계 추론 능력을 향상시키는 데 중요한 역할을 했음을 확인하였다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 연구는 CNN을 통한 정적 특징 추출과 RNN을 통한 동적 시퀀스 분석을 효과적으로 결합하였다. 특히 도구 인식의 높은 정확도가 단계 인식의 기반이 되었으며, 단순한 이진 정보뿐만 아니라 CNN의 중간 특징 맵(Feature map)이 수술 단계의 미세한 변화를 구분하는 데 유용하다는 점을 입증하였다.

### 한계점 및 비판적 해석

1. **데이터 불균형(Class Imbalance)**: 일부 수술 단계(예: Phase 3, 4)는 단 2개의 비디오에서만 등장하여 모델이 충분히 학습하지 못했다. 이는 검증 세트와 테스트 세트 간의 성능 차이를 유발하는 주요 원인이 되었다.
2. **단방향 추론**: 현재 모델은 과거의 정보만을 활용하는 단방향 RNN을 사용한다. 수술의 전체 맥락을 파악하기 위해서는 미래의 프레임 정보까지 함께 고려하는 Bidirectional RNN이나 Temporal Convolutional Networks(TCN)의 도입이 필요할 것으로 보인다.
3. **도구 의존성**: 실험 결과에서 도구가 화면에서 사라지면 단계 예측이 불안정해지는 경향이 확인되었다. 이는 시스템이 도구의 존재에 너무 과도하게 의존하고 있음을 시사하며, 도구가 없는 상태에서의 해부학적 특징(Anatomical cues)을 추가로 학습할 필요가 있다.

## 📌 TL;DR

본 논문은 백내장 수술 비디오에서 **ResNet-152(도구 인식) $\rightarrow$ LSTM/GRU(단계 분류)**로 이어지는 딥러닝 파이프라인을 제안하였다. 도구 인식에서는 99% 이상의 높은 정확도를 달성하였으며, 이를 바탕으로 수술 단계를 최대 78.28%의 정확도로 인식하였다. 특히 단순한 도구 존재 여부보다 CNN에서 추출된 고차원 특징 벡터가 단계 인식 성능을 높이는 데 기여함을 밝혔다. 이 연구는 향후 수술 자동화 및 의료 교육 시스템을 위한 워크플로우 분석의 기초가 될 가능성이 높다.
