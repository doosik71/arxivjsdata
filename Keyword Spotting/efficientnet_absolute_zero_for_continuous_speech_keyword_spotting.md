# EFFICIENTNET-ABSOLUTEZERO FOR CONTINUOUS SPEECH KEYWORD SPOTTING

Amir Mohammad Rostami, Ali karimi, Mohammad Ali Akhaee (2021)

## 🧩 Problem to Solve

본 논문은 연속적인 음성 데이터에서 특정 단어나 구절을 찾아내는 Keyword Spotting (KWS) 문제를 해결하고자 한다. 특히 페르시아어(Persian) 환경에서 축구 관련 키워드를 탐지하는 시스템을 구축하는 것에 집중한다.

KWS 시스템이 스마트폰이나 임베디드 시스템과 같은 실제 환경에서 작동하기 위해서는 모델의 경량화가 필수적이며, 훈련 및 추론 시간이 짧아야 한다. 또한, 기존의 많은 KWS 데이터셋은 단일 단어(isolated words) 위주로 구성되어 있어, 실제 환경과 유사한 연속 음성(continuous speech)에서의 성능을 보장하기 어렵다는 문제가 있다. 따라서 본 연구의 목표는 페르시아어 축구 키워드 데이터셋(FKD)을 구축하고, 이를 통해 연속 음성에서도 효율적으로 작동하는 초경량 모델인 EfficientNet-A0를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 다음과 같이 세 가지로 요약할 수 있다.

첫째, 크라우드소싱을 통해 수집한 페르시아어 축구 키워드 데이터셋인 Football Keywords Dataset (FKD)을 구축하였다. 이 데이터셋은 다양한 연령, 성별, 억양, 감정 및 녹음 환경을 포함하여 일반화 성능을 높였다.

둘째, 단일 단어 데이터셋을 연속 음성 데이터로 변환할 수 있는 Continuous Speech Synthesis Method (CSSM)를 제안하였다. 이는 배경 소음이나 실제 리포터의 음성에 키워드를 자연스럽게 합성하여, 모델이 실제 연속 음성 환경에 적응할 수 있도록 돕는다.

셋째, EfficientNet-B0의 Compound Scaling 방식을 적용하여 KWS 작업에 최적화된 초경량 아키텍처인 EfficientNet-A0(Absolute Zero)를 설계하였다. 이를 통해 파라미터 수를 획기적으로 줄이면서도 높은 추론 효율성을 달성하였다.

## 📎 Related Works

KWS 분야에서는 전통적으로 Convolutional Neural Networks (CNN)가 널리 사용되어 왔으나, 연산 복잡도가 높다는 단점이 있다. 이를 해결하기 위해 Small-footprint 아키텍처나 Residual architecture (ResNet) 등이 제안되었으며, 일부 연구에서는 Feed-forward DNN이나 Recurrent Neural Networks (RNN)를 사용하기도 하였다. 그러나 RNN은 높은 계산 복잡도와 실행 시간으로 인해 실시간 시스템 적용에 제약이 있다.

또한, Google Speech Command (GSC)와 같은 공개 데이터셋이 KWS 연구의 발전을 이끌었으나, 페르시아어와 같은 특정 언어에 특화된, 특히 연속 음성을 지원하는 데이터셋은 부족한 실정이다. 본 연구는 이러한 데이터셋의 공백을 메우고, 이미지 처리 분야에서 성능이 검증된 EfficientNet을 KWS에 도입하여 효율성을 극대화하고자 한다.

## 🛠️ Methodology

### 1. Football Keywords Dataset (FKD) 구축

FKD는 총 18개의 축구 관련 클래스로 구성되며, 약 31,000개의 샘플을 포함한다. 데이터의 다양성을 위해 1,700명 이상의 고유 화자로부터 7가지 감정 상태(normal, emphasized, upset, surprised, emotional, fast, stretched)의 발화를 수집하였다. 수집된 데이터는 Res-8 아키텍처 기반의 클리닝 시스템을 통해 정제되었으며, 실제 환경을 모사하기 위해 11가지 종류의 소음(white, pink, restaurant 등)을 추가하여 일반화 성능을 높였다.

### 2. Continuous Speech Synthesis Method (CSSM)

단일 단어 샘플을 연속 음성으로 변환하기 위해, 배경 음성 파일($s_g$)에서 2초 길이의 슬라이스($b_g$)를 무작위로 추출한다. 이후 modified zeroth-order Bessel function ($I_0$)을 이용한 윈도우 함수를 적용하여 키워드와 배경음을 합성한다.

키워드 윈도우 $w_{kw}$와 배경 윈도우 $w_{bg}$는 다음과 같이 정의된다.

$$w_{kw}(j) = \frac{I_0\left(1.5\sqrt{1-\frac{4j^2}{(M-1)^2}}\right)}{I_0(1.5)}$$
$$w_{bg}(j) = 1.05 - \frac{I_0\left(2.5\sqrt{1-\frac{4j^2}{(M-1)^2}}\right)}{I_0(2.5)}$$

최종 합성 음성 $s_s$는 배경 음성에 키워드 윈도우가 적용된 샘플 $s'$를 더함으로써 생성된다.

### 3. EfficientNet-A0 아키텍처

저자들은 EfficientNet-B0의 Compound Scaling 개념을 반대로 적용하여 모델을 축소하였다. 목표 파라미터 수는 200,000 ~ 250,000개로 설정하였으며, 입력 해상도($\gamma$)는 1로 고정하고 깊이($\alpha$)와 너비($\beta$)를 조정하였다.

$$\alpha \cdot \beta^2 \cdot \gamma^2 \approx 0.05 \pm 0.003$$

최종적으로 EfficientNet-B0의 4,032,595개 파라미터를 238,250개로 줄인 EfficientNet-A0를 도출하였다. 전체 구조는 MBConv 블록과 Global Average Pooling, Fully Connected 레이어로 구성된다.

### 4. 학습 절차

- **입력 데이터**: 20Hz/4kHz 대역 통과 필터 적용 후, 30ms 프레임 및 10ms 스트라이드로 40차원 MFCC를 추출하여 2차원 입력으로 사용한다.
- **데이터 증강**: SpecAugment를 적용하여 주파수 및 시간 마스킹을 수행한다.
- **학습 전략**: GSC 데이터셋으로 사전 학습된 가중치를 사용하는 Transfer Learning (TL)을 적용하였으며, 데이터 불균형을 해소하기 위해 Weighted Cross-Entropy 손실 함수를 사용하였다.

## 📊 Results

### 1. 실험 설정

- **비교 모델**: Res8, Res15, Res26, trad-fpool13, tpool12, one-srtide1 등 기존 GSC에서 성능이 검증된 모델들과 비교하였다.
- **평가 지표**: Test Accuracy, CSS (Continuous Speech Samples) Accuracy, 파라미터 수, GFLOPS를 측정하였다.

### 2. 정량적 결과

- **정확도**: EfficientNet-A0에 SpecAugmentation (SA)과 Transfer Learning (TL)을 적용했을 때, CSS 정확도는 $76.84 \pm 0.240\%$, Test 정확도는 $95.83 \pm 0.401\%$로 가장 높은 성능을 보였다.
- **효율성**: EfficientNet-A0의 GFLOPS는 0.01로, 유사한 정확도를 보이는 Res26에 비해 압도적으로 낮다. 이는 추론 속도가 매우 빠름을 의미한다.
- **CSSM의 효과**: CSSM을 사용하지 않고 학습한 EfficientNet-A0의 CSS 정확도는 $63.45 \pm 0.412\%$에 그쳐, 제안된 합성 방법이 연속 음성 인식 성능 향상에 기여했음을 확인하였다.

## 🧠 Insights & Discussion

본 연구는 매우 작은 모델 크기로도 충분히 높은 KWS 성능을 낼 수 있음을 입증하였다. 특히 EfficientNet의 스케일링 기법을 축소 방향으로 적용하여 GFLOPS를 획기적으로 줄인 점은 실시간 임베디드 시스템 적용 가능성을 크게 높인 부분이다.

또한, 데이터셋 구축 단계에서 단순히 단어만 수집한 것이 아니라, CSSM이라는 합성 메커니즘을 통해 데이터의 도메인을 확장한 접근 방식이 유효했다. 이는 실제 연속 음성 데이터를 대량으로 수집하기 어려운 상황에서 유용한 대안이 될 수 있다.

다만, 실험 결과에서 ResNet 계열 모델들이 Test Accuracy 측면에서는 EfficientNet-A0와 유사한 성능을 보였다는 점은, 매우 단순한 키워드 탐지 작업에서는 EfficientNet의 복잡한 구조가 주는 이득이 상대적으로 적을 수 있음을 시사한다. 그럼에도 불구하고 연산량(GFLOPS)의 압도적인 차이는 EfficientNet-A0의 실용적 우위를 뒷받침한다.

## 📌 TL;DR

본 논문은 페르시아어 축구 키워드 데이터셋(FKD)을 구축하고, 이를 연속 음성으로 확장하는 합성 방법(CSSM)과 초경량 모델인 EfficientNet-A0를 제안하였다. 실험 결과, EfficientNet-A0는 Transfer Learning과 SpecAugment를 통해 높은 정확도를 달성함과 동시에, 기존 ResNet 대비 매우 낮은 연산 복잡도(0.01 GFLOPS)를 보여 실시간 KWS 시스템에 최적화된 솔루션임을 증명하였다.
