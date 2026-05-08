# UNIFIED KEYWORD SPOTTING AND AUDIO TAGGING ON MOBILE DEVICES WITH TRANSFORMERS

Heinrich Dinkel, Yongqing Wang, Zhiyong Yan, Junbo Zhang and Yujun Wang (2023)

## 🧩 Problem to Solve

본 연구는 모바일 기기에서 효율적으로 동작할 수 있는 통합 Keyword Spotting(KWS) 및 Audio Tagging(AT) 모델을 설계하는 것을 목표로 한다.

Keyword Spotting(KWS)은 사용자가 특정 키워드를 말했을 때 시스템을 깨우는 전단(front-end) 작업으로, 항상 켜져 있어야 하므로 모델 크기가 작아야 하고 추론 속도가 매우 빨라야 하며 지연 시간(latency)이 최소화되어야 한다. 반면 Audio Tagging(AT)은 오디오 콘텐츠를 특정 소리 이벤트 클래스로 분류하는 작업으로, 기존 연구들은 주로 성능 향상에만 집중하여 모바일 기기에 배포하기에는 모델의 크기와 연산량이 너무 크다는 문제가 있었다.

이전 연구에서 KWS와 AT를 하나의 프레임워크(UniKW-AT)로 통합했을 때 소음 강건성이 향상된다는 점이 증명되었으나, 이를 실제 모바일 환경에 적용하기 위한 경량화 및 최적화 연구는 부족한 상태였다. 따라서 본 논문은 성능을 유지하면서도 모바일 기기의 제약 사항을 만족하는 경량 Transformer 기반의 통합 모델을 제안한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 모바일 배포에 최적화된 **Unified Transformers (UiT)** 모델 시리즈를 제안한 것이다.

중심적인 설계 아이디어는 Transformer의 연산 복잡도를 결정하는 주요 요인인 임베딩 차원($D$)과 패치 수($N$)를 효율적으로 제어하는 것이다. 이를 위해 저자는 **Subsampling Stem**을 통한 패치 감소와 **Bottleneck Attention (BN-A)** 메커니즘을 도입하여, 연산 오버헤드를 획기적으로 줄이면서도 KWS와 AT 두 작업 모두에서 경쟁력 있는 성능을 확보하였다.

## 📎 Related Works

기존 KWS 연구는 주로 모델 파라미터 축소, 추론 속도 향상, 오인식률(false-acceptance rate) 감소에 집중해 왔으며, 구조적으로는 CNN, Transformer, 그리고 MLP-Mixer 등이 연구되었다.

반면 AT 분야의 연구는 주로 AudioSet과 같은 대규모 벤치마크에서 최첨단(SOTA) 성능을 달성하는 것에 초점을 맞추었으며, CNN이나 Vision Transformer(ViT) 기반의 모델들이 사용되었다. 하지만 AT 연구의 대부분은 실제 기기 배포를 위한 모델 크기나 추론 속도에 대한 고려가 부족했다는 한계가 있다.

본 연구는 KWS의 효율성과 AT의 성능이라는 두 가지 요구사항을 동시에 만족시키는 UniKW-AT 프레임워크를 모바일 환경으로 확장했다는 점에서 기존 연구와 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조 및 통합 프레임워크

UniKW-AT는 KWS 레이블 집합($L_{KWS}$)과 AT 레이블 집합($L_{AT}$)을 합쳐 하나의 통합 레이블 집합 $L = L_{KWS} \cup L_{AT}$를 구성한다. 학습 데이터는 KWS와 AT 데이터셋에서 무작위로 크롭(crop)되어 사용되며, 최종적으로 Binary Cross Entropy (BCE) 손실 함수를 통해 최적화된다.

### Vision Transformer의 적용 및 연산 복잡도

본 모델은 오디오 스펙트로그램을 2차원 이미지처럼 처리하는 ViT 구조를 따른다. 입력 스펙트로그램을 $N$개의 겹치지 않는 패치로 나누고, 이를 $L$개의 동일한 블록(Multi-Head Attention(MHA) + MLP)에 통과시킨다. MHA의 기본 연산은 다음과 같다.

$$A = \text{softmax}\left(\frac{QK^T}{\sqrt{D}}\right)V$$

여기서 $Q, K, V \in \mathbb{R}^{N \times D}$이며, $D$는 모델 차원, $N$은 패치의 수이다. 이 연산의 복잡도는 $O(N^2D + D^2N)$으로, $N$과 $D$가 커질수록 연산량이 급격히 증가한다.

### UiT의 핵심 최적화 기법

연산량을 줄이기 위해 UiT는 다음 두 가지 기법을 도입한다.

1. **Subsampling Stem**: 일반적인 CNN Stem은 입력 데이터를 확장하여 메모리 사용량을 늘리지만, UiT는 입력 패치 $P$를 저차원 공간 $D$ ($D < P$)로 직접 매핑하는 subsampling 방식을 사용하여 메모리 요구량을 줄인다.
2. **Bottleneck Attention (BN-A)**: 스펙트로그램의 각 패치 임베딩에 중복 정보가 많다는 직관에 기반하여, 셀프 어텐션 단계에서 차원 $D$를 더 낮은 차원 $U$ ($U = D/4$)로 축소하여 연산한다.

### 모델 아키텍처 및 학습 절차

모델은 크기에 따라 UiT-XS, UiT-2XS, UiT-3XS의 세 가지 버전으로 제안되었으며, 패치 크기는 $16 \times 16$으로 설정하여 지연 시간을 16프레임으로 제한하였다. 추론 속도를 높이기 위해 활성화 함수로는 GeLU 대신 ReLU를 사용하였으며, MLP 내부의 임베딩 차원을 $3D$로 설정하여 메모리 풋프린트를 줄였다.

학습 시에는 AT 데이터와 KWS 데이터를 50:50 비율로 섞어 배치(batch)를 구성하며, AT 데이터의 경우 1초 단위로 크롭하여 사용한다. 이때 발생하는 학습/평가 길이 불일치 문제는 MobileNetV2 기반의 교사 모델을 통해 생성한 Pseudo Strong Labels (PSL)를 사용하여 완화하였다.

## 📊 Results

### 실험 설정

- **데이터셋**: Google Speech Commands V1 (GSCV1) 및 AudioSet.
- **측정 지표**: KWS는 Accuracy(정확도)를, AT는 mAP(mean Average Precision)를 사용한다.
- **하드웨어**: Qualcomm Snapdragon 865, 888 및 MediaTek Helio G90T, Dimensity 700 등 4종의 모바일 SoC에서 CPU(float32) 기준으로 측정하였다.

### 주요 결과

정량적 분석 결과, 가장 성능이 좋은 **UiT-XS** 모델은 GSCV1에서 97.76%의 정확도를, AudioSet에서 34.09 mAP를 기록하였다. 이는 기존의 UniKW-AT 베이스라인인 MobileNetV2(MBv2)와 유사하거나 더 우수한 성능이다.

추론 속도 면에서는 획기적인 향상을 보였다. Snapdragon 865 기준, MBv2의 추론 시간이 8.0ms인 반면 UiT-XS는 3.4ms로 약 2.3배 빨랐으며, 모델 크기에 따라 최대 6배까지 속도가 향상되었다. 또한 MBv2의 지연 시간이 320ms인 것에 비해 UiT 모델들은 160ms 이내로 반응하여 실시간성에 더 유리함을 입증하였다.

### 어블레이션 연구 (Ablation Study)

BN-A 메커니즘과 ReLU 활성화 함수의 효과를 검증한 결과, BN-A를 적용하지 않았을 때보다 추론 속도가 최소 20% 향상되었으며 메모리 사용량 또한 크게 감소하였다. 특히 GeLU를 사용했을 때는 성능 향상이 미미한 반면 추론 속도가 크게 느려지고(3.4ms $\to$ 5.7ms) 메모리 사용량이 증가하여 모바일 배포에 부적합함이 확인되었다.

## 🧠 Insights & Discussion

본 논문은 KWS와 AT라는 서로 다른 성격의 두 작업을 통합하면서도, 모바일 기기라는 극한의 제약 환경에서 동작 가능하도록 모델을 최적화하였다는 점에서 강점이 있다. 특히 통합 모델을 사용함으로써 AT 브랜치의 출력이 VAD(Voice Activity Detector) 역할을 수행하거나 ASR(Automatic Speech Recognition) 파이프라인의 강건성을 높이는 데 기여할 수 있다는 점은 실용적인 가치가 높다.

다만, AT 성능 향상을 위해 1초 단위의 크롭 데이터를 사용하고 PSL로 보완하였으나, 이는 원래 10초 단위로 학습하던 기존 AT 모델들에 비해 정보 손실이 있을 수 있다는 가정을 내포하고 있다. 또한, 제안된 모델들이 TC-ResNet8과 같은 극소형 KWS 전용 모델보다는 여전히 느리다는 점은, AT 기능을 추가함으로써 발생하는 연산 비용의 트레이드-오프(trade-off)로 해석된다.

## 📌 TL;DR

본 연구는 모바일 기기 배포를 위해 최적화된 통합 KWS 및 AT 모델인 **UiT (Unified Transformers)**를 제안하였다. **Subsampling Stem**과 **Bottleneck Attention**을 통해 연산량과 메모리 사용량을 획기적으로 줄였으며, 결과적으로 MobileNetV2 기반 모델 대비 파라미터 수는 절반으로 줄이면서 추론 속도는 2~6배 향상시켰다. 이 연구는 향후 모바일 기기에서 더 지능적이고 강건한 음성 인터페이스를 구현하는 데 중요한 기초가 될 것으로 보인다.
