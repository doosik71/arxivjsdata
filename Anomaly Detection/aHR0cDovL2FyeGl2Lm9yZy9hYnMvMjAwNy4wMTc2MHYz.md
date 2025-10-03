# EXPLAINABLE DEEP ONE-CLASS CLASSIFICATION

Philipp Liznerski, Lukas Ruff, Robert A. Vandermeulen, Billy Joe Franks, Marius Kloft, Klaus-Robert Müller

## 🧩 Problem to Solve

심층 특이치 탐지(Deep Anomaly Detection, AD) 모델은 명목 샘플(nominal samples)을 특징 공간의 특정 지점에 집중시키고 이상 샘플(anomalous samples)을 멀리 매핑하는 방식으로 작동합니다. 그러나 이러한 매핑 과정이 고도로 비선형적이어서 모델의 예측을 해석하기 어렵다는 문제가 있습니다. 특히 산업 응용 분야에서는 안전, 보안, 공정성 및 전문가 의사결정 지원을 위해 모델의 예측이 어떻게 도출되었는지에 대한 설명이 필수적입니다. 이 논문은 기존의 심층 단일 클래스 분류(Deep One-Class Classification) 방법의 해석 불가능성 문제를 해결하고, 동시에 경쟁력 있는 탐지 성능을 유지하는 설명 가능한 방법을 제시하고자 합니다.

## ✨ Key Contributions

- **설명 가능한 심층 단일 클래스 분류 방법인 FCDD(Fully Convolutional Data Description) 제안:** FCDD는 출력 특징 자체가 다운샘플링된 특이치 히트맵(anomaly heatmap)으로 작동하도록 설계되어 공간 정보를 보존하고 해석을 내재적으로 제공합니다.
- **경쟁력 있는 탐지 성능 달성:** CIFAR-10 및 ImageNet과 같은 표준 특이치 탐지 벤치마크에서 기존 SOTA(State-of-the-Art) 방법에 근접하는 성능을 보였습니다.
- **MVTec-AD 데이터셋에서 SOTA 달성:** 실제 제조 결함 데이터셋인 MVTec-AD에서 미세한 결함을 정확하게 설명하며 비지도 학습(unsupervised setting) 환경에서 새로운 SOTA 성능(0.92 pixel-wise mean AUC)을 기록했습니다.
- **적은 수의 레이블링된 이상 샘플 활용 능력:** 학습 중 그라운드 트루스 이상(ground-truth anomaly) 설명을 통합할 수 있으며, 단 몇 개(예: $\sim$5개)의 레이블링된 샘플만으로도 성능을 크게 향상시킬 수 있음을 입증했습니다.
- **심층 단일 클래스 분류 모델의 취약성 분석:** FCDD의 설명을 통해 심층 단일 클래스 분류 모델이 이미지 워터마크와 같은 가짜 특징(spurious image features)에 취약하며 "영리한 한스(Clever Hans)" 효과를 보일 수 있음을 시연했습니다.

## 📎 Related Works

- **오토인코더(Autoencoders, AEs):** 재구성 오류(reconstruction error)를 특이치 점수 및 픽셀 단위 히트맵으로 사용하여 설명 가능성을 제공하지만, 알려진 특이치를 학습에 통합하기 어렵습니다.
- **심층 단일 클래스 분류(Deep One-Class Classification):** DSVDD(Deep Support Vector Data Description) (Ruff et al., 2018), Deep SAD (Ruff et al., 2020b), HSC(Hypersphere Classifier) (Ruff et al., 2020a) 등이 있으며, 명목 데이터를 특징 공간에 집중시키고 이상 데이터를 멀리 매핑합니다. 이미지에 대한 설명 접근 방식은 제한적입니다 (예: Deep Taylor decomposition).
- **자기 지도 학습(Self-supervision):** 명목 샘플에 변환을 적용하고 네트워크가 사용된 변환을 예측하도록 훈련하여 예측의 신뢰도를 통해 특이치 점수를 제공합니다. 이 방법에서는 아직 설명 가능성 접근 방식이 고려되지 않았습니다.
- **일반적인 설명 가능 방법론 (XAI):** LIME (Ribeiro et al., 2016)과 같은 모델 불가지론적(model-agnostic) 방법이나 그래디언트 기반 기법(Simonyan et al., 2013; Sundararajan et al., 2017)이 있습니다. FCDD와 관련하여 완전 컨볼루션 아키텍처는 지도 방식의 분할(segmentation) 작업에 사용되었습니다 (Long et al., 2015; Noh et al., 2015).

## 🛠️ Methodology

FCDD는 심층 단일 클래스 분류와 완전 컨볼루션 네트워크(FCN)의 장점을 결합하여 출력 특징 자체가 공간 정보를 보존하는 특이치 히트맵으로 기능하게 합니다.

1. **심층 단일 클래스 분류 (Deep One-Class Classification):**

   - HSC(Hypersphere Classifier)를 기반으로 합니다.
   - 목표는 신경망 $\phi$가 명목 샘플 $X_i$를 중심 $c$ 근처에 매핑하고, 이상 샘플을 $c$에서 멀리 매핑하도록 학습하는 것입니다.
   - 다음과 같은 유사-휴버(pseudo-Huber) 손실 함수를 사용합니다: $h(a) = \sqrt{\Vert a \Vert^{2}_{2} + 1} - 1$.
   - 목표 함수는 다음과 같습니다:
     $$
     \min_{W,c} \frac{1}{n} \sum_{i=1}^{n} (1-y_i)h(\phi(X_i;W)-c) - y_i \log (1-\exp (-h(\phi(X_i;W)-c)))
     $$
     여기서 $y_i=0$은 명목 샘플, $y_i=1$은 이상 샘플입니다. 실제 구현에서는 $c$가 네트워크 $\phi$의 마지막 레이어 바이어스 항에 포함됩니다.

2. **완전 컨볼루션 아키텍처 (Fully Convolutional Architecture, FCN):**

   - 모든 완전 연결(fully connected) 레이어 없이 컨볼루션 및 풀링 레이어만 사용하여 이미지를 특징 행렬 $\phi: \mathbb{R}^{c \times h \times w} \to \mathbb{R}^{1 \times u \times v}$로 매핑합니다.
   - 각 출력 픽셀의 수용 영역(receptive field)이 입력 이미지의 작은 영역에만 의존하며, 공간 정보를 보존합니다.

3. **완전 컨볼루션 데이터 설명 (Fully Convolutional Data Description, FCDD):**

   - FCN $\phi$와 HSC 손실 함수를 결합합니다.
   - FCN의 출력 특징 행렬 $A(X) = (\sqrt{\phi(X;W)^2+1}-1)$에 대해 유사-휴버 손실을 적용합니다 (모든 연산은 요소별(element-wise)로 적용).
   - FCDD의 목적 함수는 다음과 같습니다:
     $$
     \min_{W} \frac{1}{n} \sum_{i=1}^{n} (1-y_i) \frac{1}{u \cdot v} \Vert A(X_i) \Vert_1 - y_i \log \left( 1-\exp \left( -\frac{1}{u \cdot v} \Vert A(X_i) \Vert_1 \right) \right)
     $$
   - $\Vert A(X) \Vert_1$은 특이치 점수로 사용되며, $A(X)$의 값은 입력 이미지의 특이치 영역에 해당합니다.
   - **아웃라이어 노출(Outlier Exposure, OE):** 학습 시 명목 샘플과 함께 보조 이상 샘플(예: 무작위 이미지, ImageNet)을 활용하여 모델 성능을 향상시킵니다. "진정한" 이상 샘플을 사용할 경우 소수만으로도 효과적입니다.

4. **히트맵 업샘플링 (Heatmap Upsampling):**

   - FCN 출력 $A(X)$는 저해상도 특이치 히트맵입니다.
   - 학습 시 픽셀 단위 그라운드 트루스 특이치 맵이 없으므로, 역 컨볼루션(deconvolution) 방식으로 업샘플링할 수 없습니다.
   - 대신, 수용 영역의 효과가 중심에서 멀어질수록 가우시안(Gaussian) 방식으로 감소한다는 속성을 활용합니다 (Luo et al., 2016).
   - **고정된 가우시안 커널을 사용한 스트라이드 전치 컨볼루션(strided transposed convolution)으로 업샘플링을 수행합니다 (Algorithm 1 참조).** 커널 크기는 FCDD의 수용 영역 범위, 스트라이드는 FCDD의 누적 스트라이드로 설정됩니다. 가우시안 분산($\sigma$)은 경험적으로 선택됩니다.

5. **반지도 학습 FCDD (Semi-Supervised FCDD):**
   - MVTec-AD와 같이 픽셀 단위 그라운드 트루스 특이치 맵 $Y_i$가 소수의 이상 샘플에 대해 제공될 때 사용됩니다.
   - 픽셀 단위 목적 함수를 통해 학습하여 설명 정확도를 높입니다.
   - $$
     \min_{W} \frac{1}{n} \sum_{i=1}^{n} \left( \frac{1}{m} \sum_{j=1}^{m} (1-(Y_i)_j)A'(X_i)_j \right) - \log \left( 1-\exp \left( -\frac{1}{m} \sum_{j=1}^{m} (Y_i)_j A'(X_i)_j \right) \right)
     $$

## 📊 Results

- **표준 특이치 탐지 벤치마크 (Fashion-MNIST, CIFAR-10, ImageNet):**

  - FCDD는 제한된 FCN 아키텍처에도 불구하고 기존 SOTA 방법에 근접하는 AUC 성능을 달성했습니다 (표 1 참조). 특히 복잡한 데이터셋에서는 오토인코더보다 뛰어난 성능을 보였습니다.
  - 시각적 분석 결과, FCDD의 히트맵은 그래디언트 기반 방법보다 덜 노이즈가 많고, 오토인코더보다 더 구조적인 정보를 제공했습니다 (그림 4, 5, 6 참조).

- **MVTec-AD 제조 결함 탐지:**

  - **비지도 학습(Unsupervised):** FCDD는 0.92의 픽셀 단위 평균 AUC를 달성하여 경쟁 모델들을 능가하며 새로운 SOTA를 기록했습니다 (표 2 참조). "콘페티 노이즈(confetti noise)"와 같은 합성 이상 샘플을 활용했습니다.
  - **반지도 학습(Semi-supervised):** 각 결함 유형당 단 하나의 이상 샘플(총 3-8개)만 사용하여 학습했을 때, 픽셀 단위 평균 AUC는 0.96으로 더욱 향상되었습니다 (표 2 참조). FCDD는 클래스 전반에 걸쳐 가장 일관된 성능을 보였습니다.

- **영리한 한스 효과 (Clever Hans Effect) 시연:**
  - PASCAL VOC 데이터셋에서 "말" 클래스를 이상 샘플로, ImageNet을 명목 샘플로 설정하여 FCDD를 훈련시켰습니다.
  - 그 결과, FCDD의 히트맵은 말 자체보다 이미지 왼쪽 하단의 워터마크와 같은 가짜 특징을 높은 특이치로 강조했습니다 (그림 8 참조). 이는 심층 단일 클래스 모델도 훈련 시에는 유용하지만 배포 시에는 바람직하지 않은 가짜 특징을 학습할 수 있음을 보여줍니다.

## 🧠 Insights & Discussion

- **다목적성:** FCDD는 CIFAR-10, ImageNet과 같은 **의미론적 탐지 작업**뿐만 아니라 MVTec-AD와 같은 **미세한 결함 탐지 작업**에서도 뛰어난 성능을 보였습니다.
- **내재적 설명 가능성:** FCDD는 특이치 점수에 직접 설명을 연결함으로써, 사후(a posteriori) 설명 방법론보다 **적대적 공격(adversarial attacks)에 덜 취약할 수 있다는 잠재력**을 가집니다 (향후 연구 과제).
- **모델 진단 및 개선:** FCDD의 투명한 설명은 **"영리한 한스" 효과**와 같이 모델이 바람직하지 않은 특징을 학습하는 경우를 실무자가 인식하고, 훈련 데이터 정제나 확장 등을 통해 해결할 수 있도록 돕습니다. 이는 모델의 신뢰성과 공정성 확보에 중요합니다.
- **수용 영역 크기의 영향:** 수용 영역이 작을수록 히트맵의 집중도가 높아지고 MVTec-AD에서 픽셀 단위 AUC 점수가 향상되는 경향이 있습니다 (부록 A 참조).
- **업샘플링 하이퍼파라미터의 중요성:** 가우시안 커널의 분산($\sigma$)과 같은 업샘플링 하이퍼파라미터는 설명의 품질에 영향을 미치며, 경험적 조정을 통해 최적화될 수 있습니다 (부록 B 참조).

## 📌 TL;DR

**문제:** 기존 심층 특이치 탐지 모델은 높은 비선형성으로 인해 예측을 해석하기 어렵습니다.

**해결책:** FCDD(Fully Convolutional Data Description)는 완전 컨볼루션 네트워크(FCN)와 변형된 HSC(Hypersphere Classifier) 손실을 결합하여, 모델의 출력 특징 자체가 다운샘플링된 특이치 히트맵으로 기능하게 합니다. 저해상도 히트맵은 고정된 가우시안 커널을 사용한 전치 컨볼루션으로 업샘플링되어 전체 해상도 설명을 제공합니다. FCDD는 아웃라이어 노출(OE) 및 소수의 레이블링된 이상 샘플을 활용한 반지도 학습도 지원합니다.

**핵심 발견:** FCDD는 표준 특이치 탐지 벤치마크에서 경쟁력 있는 성능을 보였고, MVTec-AD 데이터셋에서는 비지도 학습 환경에서 0.92의 픽셀 단위 평균 AUC로 새로운 SOTA를 달성했습니다. 특히, FCDD의 설명은 모델이 워터마크와 같은 가짜 특징을 학습하는 "영리한 한스" 효과를 진단하는 데 유용하며, 이는 심층 특이치 탐지 모델의 취약성을 밝히는 데 기여합니다.
