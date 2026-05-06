# Enabling Calibration in the Zero-Shot Inference of Large Vision-Language Models

Will Levine, Benjamin Pikus, Pranav Raja & Fernando Amat Gil (2023)

## 🧩 Problem to Solve

본 논문은 딥러닝 모델의 신뢰성과 안전한 사용을 위해 필수적인 **Calibration**(교정) 문제를 다룬다. Calibration이란 모델이 예측한 확신도(Confidence)가 실제 정답 확률(Accuracy)과 일치하는 정도를 의미한다. 예를 들어, 모델이 0.8의 확신도로 예측한 샘플들이 100개 있다면, 그 중 실제로 80개가 정답이어야 잘 교정된 모델이라고 할 수 있다.

전통적인 지도 학습 기반의 분류 모델에서는 Calibration 연구가 활발히 진행되었으나, CLIP과 같은 **Vision-Language Model(VLM)**의 **Zero-shot inference** 설정에서의 Calibration에 대한 포괄적인 연구는 부족한 실정이다. Zero-shot inference는 추론 시점에 클래스가 동적으로 정의되므로, 고정된 검증 세트를 사용하는 기존의 Calibration 방법론을 그대로 적용하기 어렵다. 따라서 본 논문의 목표는 CLIP의 Zero-shot 설정에서 발생하는 Miscalibration을 분석하고, Zero-shot의 특성을 유지하면서도 이를 해결할 수 있는 효율적인 Calibration 방법을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 CLIP의 Zero-shot 추론 과정에서 모델의 확신도를 실제 정확도와 일치시키기 위한 **"Zero-Shot-Enabled Temperature Scaling"** 방법을 제안한 것이다.

핵심 직관은 특정 CLIP 모델(특정 아키텍처와 사전 학습 데이터셋의 조합)이 갖는 **Temperature($T$) 파라미터가 추론 시 사용되는 데이터셋이나 프롬프트(Prompt)의 선택에 관계없이 일정하게 유지된다**는 점이다. 즉, 보조 데이터셋(Auxiliary dataset)을 통해 한 번 학습된 단일 Temperature 값만으로도 다양한 다운스트림 Zero-shot 작업에서 일반화된 Calibration 성능을 얻을 수 있다는 설계 아이디어를 제시한다.

## 📎 Related Works

기존의 Calibration 연구들은 주로 지도 학습 환경에서 수행되었으며, 대표적으로 **Temperature Scaling (TS)**, **Isotonic Regression**, **Histogram Binning** 등이 있다. 이러한 방법들은 모델의 Logit 값을 조정하여 확률 분포를 완만하게 하거나 비선형적으로 매핑함으로써 Calibration을 개선한다.

그러나 기존 방식들은 다음과 같은 한계가 있다:

1. **데이터 의존성**: 추론하려는 작업과 동일한 분포를 가진 별도의 Calibration 데이터셋(Validation set)이 필요하다.
2. **Zero-shot 특성 상실**: CLIP의 가장 큰 장점은 사전 학습된 모델을 그대로 사용하여 새로운 클래스에 대해 추론하는 것인데, 작업마다 별도의 Calibration 과정을 거쳐야 한다면 이는 더 이상 순수한 Zero-shot 추론이라고 보기 어렵다.

본 논문은 이러한 제약 조건을 해결하여, 추론 시점에 추가적인 학습이나 튜닝 없이 적용 가능한 방식을 제안함으로써 기존 연구와 차별화된다.

## 🛠️ Methodology

### 1. CLIP의 Logit 계산 방식

CLIP은 이미지 임베딩 $E_{im}(x_i)$와 텍스트 임베딩 $E_{lang}(y_c)$ 간의 코사인 유사도를 기반으로 Logit을 계산한다.

$$L_{CLIP}^c(x_i) = 100 \times \frac{E_{im}(x_i) \cdot E_{lang}(y_c)}{\|E_{im}(x_i)\| \|E_{lang}(y_c)\|}$$

여기서 100은 OpenAI CLIP에서 기본적으로 사용하는 스칼라 값이다. 이후 Softmax 함수를 통해 클래스 확률을 도출한다.

$$\hat{f}_c(x_i) = \frac{e^{L_{CLIP}^c(x_i)}}{\sum_{j=1}^{C} e^{L_{CLIP}^j(x_i)}}$$

### 2. Zero-Shot-Enabled Temperature Scaling

제안 방법은 기존의 Temperature Scaling을 Zero-shot 설정에 맞게 변형한 것이다. Temperature Scaling은 Logit을 스칼라 $T$로 나누어 확률 분포를 조정한다.

$$L_{calibrated}^c(x_i; T) = \frac{L_{CLIP}^c(x_i)}{T}$$

**학습 및 적용 절차:**

- **학습**: 특정 아키텍처와 사전 학습 데이터셋을 가진 CLIP 모델에 대해, 보조 데이터셋(본 논문에서는 ImageNet-1k 사용)과 기본 프롬프트("a photo of{}")를 이용하여 Cross-Entropy Loss를 최소화하는 최적의 $T$를 찾는다.
- **추론**: 학습된 $T$ 값을 저장해 두었다가, 이후 어떠한 새로운 데이터셋이나 프롬프트를 사용하여 Zero-shot 추론을 수행하든 관계없이 모든 Logit을 해당 $T$로 나누어 최종 확률을 계산한다.

이 과정은 CLIP의 사전 학습 방식과 유사하게, 모델의 파라미터(여기서는 $T$)가 특정 아키텍처/데이터셋 조합에 종속될 뿐, 실제 추론 대상인 다운스트림 작업과는 독립적이라는 점을 이용한 것이다.

## 📊 Results

### 1. 실험 설정

- **지표**: **Expected Calibration Error (ECE)**를 사용하여 Miscalibration 정도를 정량화한다. ECE는 신뢰도 빈(Bin)별 평균 신뢰도와 실제 정확도의 차이를 가중 평균한 값이다.
  $$ECE = \sum_{m=1}^{M} \frac{|B_m|}{|D|} |\hat{p}(\hat{f}, B_m) - acc(\hat{f}, B_m)|$$
- **데이터셋**: CIFAR-10, CIFAR-100, SUN397 등 다양한 규모와 분포의 데이터셋을 사용하였다.
- **비교 대상**: Vanilla CLIP(교정 없음), Supervised Temperature Scaling(대상 데이터셋으로 직접 교정), 제안 방법(Zero-Shot-Enabled TS).
- **모델**: ViT-B-16, ViT-L-14 등 다양한 OpenCLIP 아키텍처와 LAION-400M, LAION-2B 등 서로 다른 사전 학습 데이터를 사용한 모델들을 평가하였다.

### 2. 주요 결과

- **Miscalibration 확인**: Vanilla CLIP은 대부분의 설정에서 심각한 Miscalibration(주로 Overconfidence)을 보였다.
- **성능 개선**: 제안 방법은 모든 설정에서 Vanilla CLIP보다 ECE를 유의미하게 낮추어 Calibration 성능을 향상시켰다.
- **강건성(Robustness)**: 실험 결과, 최적의 $T$ 값은 추론 데이터셋의 종류나 프롬프트의 문구 변화에 관계없이 매우 유사하게 유지됨이 확인되었다. 이는 보조 데이터셋에서 학습한 $T$가 일반적인 Zero-shot 상황에서도 유효함을 시사한다.
- **한계점**: 다만, 대상 데이터셋의 레이블을 직접 사용하여 교정한 Supervised TS보다는 Calibration 성능이 낮게 나타났다.

## 🧠 Insights & Discussion

본 논문은 CLIP과 같은 대규모 VLM이 Zero-shot 설정에서 신뢰할 수 없는 확신도를 출력한다는 점을 실험적으로 입증하였다. 제안된 Zero-Shot-Enabled Temperature Scaling은 모델의 아키텍처와 사전 학습 데이터라는 정적인 특성에 기반하여 $T$를 결정함으로써, Zero-shot의 핵심인 '추론 시 추가 학습 없음'이라는 패러다임을 유지하면서도 신뢰도를 개선했다는 점에서 강점이 있다.

특히, 최적의 $T$ 값이 프롬프트나 데이터셋에 관계없이 일정하다는 발견은 매우 흥미롭다. 이는 모델이 출력하는 Logit의 스케일 특성이 작업의 내용보다는 모델 자체의 학습 상태(Training Dynamics)에 더 큰 영향을 받는다는 것을 의미한다.

다만, Supervised TS와의 성능 격차는 여전히 존재한다. 이는 보조 데이터셋(ImageNet-1k)이 모든 가능한 Zero-shot 분포를 완벽히 대표하지 못하기 때문일 수 있다. 향후 연구에서는 단순한 스칼라 $T$ 외에 더 정교한 Zero-shot Calibration 방법론이 필요할 것으로 보인다.

## 📌 TL;DR

- **요약**: CLIP의 Zero-shot 추론 시 발생하는 확신도-정확도 불일치(Miscalibration) 문제를 분석하고, 보조 데이터셋에서 학습한 단일 Temperature 파라미터를 적용하는 **Zero-Shot-Enabled Temperature Scaling** 방법을 제안하였다.
- **의의**: 이 방법은 추론 시점에 추가적인 튜닝이나 레이블 데이터 없이도 모델의 확신도를 개선할 수 있어, VLM의 신뢰성을 높이고 실제 서비스 적용 시 안전한 임계값(Threshold) 설정을 가능하게 한다.
