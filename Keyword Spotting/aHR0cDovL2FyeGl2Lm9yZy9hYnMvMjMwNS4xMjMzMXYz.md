# DCCRN-KWS: An Audio Bias Based Model for Noise Robust Small-Footprint Keyword Spotting

Shubo Lv, Xiong Wang, Sining Sun, Long Ma, Lei Xie (2023)

## 🧩 Problem to Solve

본 논문은 저신호 대 잡음비(low Signal-to-Noise Ratio, SNR)를 갖는 복잡한 실제 음향 환경에서 소형 풋프린트(small-footprint) 키워드 스포팅(Keyword Spotting, KWS) 시스템이 겪는 성능 저하 문제를 해결하고자 한다. 특히 웨이크업 워드(Wake-up Word, WuW) 검출과 같이 스마트 기기 인터랙션의 첫 단계가 되는 작업에서, 사용자-기기 간의 거리로 인한 신호 감쇠, 환경 소음 및 실내 잔향은 시스템의 강건성(robustness)을 심각하게 저해한다. 따라서 본 연구의 목표는 잡음에 강건하면서도 연산 효율성을 유지하는 KWS 모델을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 신경망 기반의 음성 향상(Speech Enhancement)과 음성 인식의 문맥 편향(Context Bias) 개념을 KWS에 통합하는 것이다. 구체적인 기여 사항은 다음과 같다.

1.  **Multi-task Learning (MTL) 프레임워크**: DCCRN(Deep Complex Convolution Recurrent Network) 인코더를 공유하여 잡음 제거(Denoising)와 키워드 검출(KWS)을 동시에 수행하는 구조를 제안하여, 두 작업이 서로를 보완하도록 설계하였다.
2.  **Audio Context Bias 모듈**: 실제 키워드 오디오 샘플에서 임베딩을 추출하여 네트워크에 제공함으로써, 잡음 환경에서도 키워드를 더 잘 식별할 수 있도록 편향(bias)을 부여한다.
3.  **Feature Merge 모듈**: DCCRN 인코더의 각 계층 출력물에서 키워드 영역의 에너지가 강조되는 특성을 이용해, 여러 계층의 특징을 가중 평균하여 키워드 변별력을 강화한다.
4.  **Complex Context Linear 모듈**: 복소수 특징의 실수부와 허수부를 분리하고 이전 프레임들의 문맥 정보를 효율적으로 통합하여, 모델 복잡도 증가를 억제하면서도 인식 성능을 향상시킨다.

## 📎 Related Works

기존의 KWS 성능 향상을 위해 신호 향상 프런트엔드(front-end)를 도입하는 방식이 주로 사용되었다. 전통적인 통계적 신호 처리 기반의 방법들이 있었으나, 최근에는 비정상 잡음(non-stationary noise) 제거 능력이 뛰어난 딥러닝 기반의 음성 향상이 주류를 이루고 있다. 특히 DCCRN과 같은 구조는 복소 스펙트럼(complex spectrum)을 직접 모델링하여 우수한 성능을 보였다.

기존 연구에서는 음성 향상 모듈을 독립적으로 최적화하거나 ASR(Automatic Speech Recognition) 모델과 결합하는 시도가 있었다. 하지만 본 논문은 KWS가 사전에 정의된 제한된 수의 키워드만을 인식한다는 특성에 주목하여, 제약된 언어적/음향적 정보를 최대한 활용하는 Audio Context Bias 방식을 제안함으로써 기존의 일반적인 음성 향상 결합 방식과 차별점을 둔다.

## 🛠️ Methodology

### 전체 시스템 구조
본 모델은 DCCRN 인코더를 공유하는 Multi-task Learning 구조를 가진다. 입력된 노이즈 음성은 DCCRN 인코더를 통과하며, 그 출력은 두 갈래로 나뉜다. 하나는 음성 향상을 위한 DCCRN 디코더로 이어져 깨끗한 음성을 복원하고, 다른 하나는 제안된 Audio Context Bias, Feature Merge, Complex Context Linear 모듈을 거쳐 KWS 모듈로 전달된다. 추론 단계에서는 음성 향상 디코더 부분을 제거하고 KWS 결과만을 사용한다.

### 주요 구성 요소 및 절차

**1. KWS Module**
Dilated Temporal Convolutional (DTC) 블록으로 구성된다. Dilated depthwise 1D-convolution으로 시간적 문맥을 파악하고, 두 층의 pointwise convolution으로 채널 간 특징을 통합한 뒤, FC(Fully Connected) 층과 Softmax 함수를 통해 키워드 존재 확률을 추정한다.

**2. Audio Context Bias**
키워드 코퍼스에서 선택된 오디오 샘플 리스트를 ECAPA-TDNN 임베딩 추출기에 입력하여 키워드 임베딩을 얻는다. 리스트 내 모든 벡터의 평균을 구한 뒤, 이를 DCCRN 인코더의 마지막 출력과 연결(concatenate)하여 KWS 모듈의 입력으로 사용한다.

**3. Feature Merge (FM)**
인코더의 각 계층 $i$의 출력 $E_i$에서 키워드 영역의 에너지가 높게 나타나는 현상을 활용한다. 학습 가능한 가중치 $w_i$를 사용하여 다음과 같이 가중 평균을 수행한다.
$$E' = \frac{\sum_{i} w_i \cdot \text{downsample}(E_i)}{\sum_{i} |w_i|}$$
이 과정을 통해 키워드 부분의 특성을 더욱 두드러지게 만든다.

**4. Complex Context Linear (CCL)**
단순한 연결(concatenation)으로 인한 연산량 폭발을 막기 위해, 인코더 출력을 실수부($\Re$)와 허수부($\Im$)로 분리한다. 각각의 부분에 Bias 임베딩을 연결하고, 현재 프레임($t$)과 이전 두 프레임($t-1, t-2$)의 정보를 결합하여 FC 층에 입력한다.

### 학습 목표 및 손실 함수
음성 향상 작업에는 시간 영역의 SI-SNR loss를 사용하고, KWS 작업에는 Binary Cross Entropy (BCE) loss를 사용한다. 최종 손실 함수 $L$은 다음과 같다.
$$L = L_{\text{SI-SNR}} + L_{\text{BCE}}$$
여기서 $L_{\text{BCE}}$는 다음과 같이 정의된다.
$$L_{\text{BCE}} = -y_i^* \ln(y_i) - (1-y_i^*) \ln(1-y_i)$$
($y_i$는 예측 확률, $y_i^*$는 정답 레이블)

## 📊 Results

### 실험 설정
- **데이터셋**: 내부 데이터셋(Mandarin, "ding1-dang2-ding1-dang2") 및 공개 데이터셋인 HIMIYA(Mandarin, "ni2-hao3-mi1-ya4")를 사용하였다.
- **환경**: SNR -6 ~ -2 dB(내부), -5, 0, 5 dB(HIMIYA)의 잡음 환경을 시뮬레이션하였으며, RT60 0.05s ~ 0.95s의 잔향을 추가하였다.
- **측정 지표**: 내부 데이터셋은 ROC 곡선(False Reject rate vs False Alarm rate)을 통해 평가하였으며, HIMIYA 데이터셋은 10시간 노출 시 최대 1회 오작동을 허용하는 Wake-up Accuracy로 측정하였다.

### 주요 결과
1.  **모듈별 효과**: 내부 데이터셋 실험 결과, DCCRN 도입 $\rightarrow$ Audio Bias 추가 $\rightarrow$ Feature Merge 적용 $\rightarrow$ Complex Context Linear 적용 순으로 성능이 단계적으로 향상됨을 확인하였다.
2.  **Audio Bias의 우수성**: 학습 가능한(learnable) 임베딩보다 실제 오디오 샘플에서 추출한 임베딩이 더 효과적이었으며, 이는 실제 오디오가 키워드의 구조적 정보를 더 많이 포함하고 있기 때문으로 분석된다.
3.  **사용자 의존성**: HIMIYA 데이터셋에서 사용자 의존적(speaker-dependent) 오디오 리스트를 사용했을 때 성능이 더 높았는데, 이는 임베딩에 키워드 정보와 화자 정보가 동시에 포함되어 변별력이 높아졌기 때문이다.
4.  **추론 효율성**: RK3326 플랫폼 측정 결과, 제안 모델은 RTF(Real-Time Factor)와 CPU 사용량 측면에서 합리적인 수준을 유지하였다. 반면, 단순하게 DCCRN과 KWS를 직렬로 연결하여 공동 학습시킨 모델(DCCRN-KWS-joint-train)은 연산 비용이 너무 높아 실사용이 불가능한 수준이었다.

## 🧠 Insights & Discussion

본 논문은 음성 향상과 KWS를 MTL 구조로 통합함으로써, 단순한 파이프라인 연결보다 효율적이고 강력한 성능을 낼 수 있음을 보여주었다. 특히 Audio Context Bias는 텍스트 기반 bias보다 오디오-오디오 유사도 측정이 더 직관적이며, 잡음 환경에서의 강건성을 높이는 데 기여한다.

한계점으로는 HIMIYA 데이터셋에서 내부 데이터셋만큼의 성능 향상이 나타나지 않았는데, 이는 HIMIYA의 등록(enrollment) 오디오 길이가 상대적으로 짧아 강건한 임베딩 추출에 제약이 있었기 때문으로 보인다. 즉, 편향을 제공할 키워드 오디오의 길이가 충분할수록 모델의 성능이 더욱 향상될 가능성이 있다.

비판적으로 해석하자면, 제안된 구조는 특정 키워드 세트에 최적화된 형태이므로, 키워드가 빈번하게 변경되는 환경에서는 매번 임베딩을 새로 추출하거나 업데이트해야 하는 오버헤드가 발생할 수 있다.

## 📌 TL;DR

본 논문은 잡음이 심한 환경에서도 작동하는 소형 KWS 모델인 **DCCRN-KWS**를 제안한다. DCCRN 인코더 기반의 **Multi-task Learning** 구조를 채택하고, 실제 키워드 오디오 샘플을 활용한 **Audio Context Bias**, 계층별 특징을 통합하는 **Feature Merge**, 효율적인 문맥 통합을 위한 **Complex Context Linear** 모듈을 도입하였다. 실험을 통해 낮은 SNR 환경에서도 높은 인식률을 보였으며, 저사양 임베디드 플랫폼에서 실시간 추론이 가능한 효율성을 입증하였다. 이 연구는 향후 엣지 디바이스의 웨이크업 워드 시스템의 강건성을 높이는 데 중요한 역할을 할 것으로 기대된다.