# Medical Image Segmentation Using a U-Net type of Architecture

Eshal Zahra, Bostan Ali, and Wajahat Siddique

## 🧩 Problem to Solve

이 논문은 의료 영상 분할(Medical Image Segmentation) 작업에서 U-Net 아키텍처의 인코더(encoder) 부분이 더 효과적인 특징(feature)을 인코딩하도록 개선하는 문제를 다룹니다. 기존 U-Net은 자체 지도(self-supervised) 방식으로 인코더를 훈련하지만, 이 방식이 병목(bottleneck) 계층에서 의미론적 정보(semantic information)를 충분히 포착하지 못할 수 있다는 점을 지적하며, 이를 개선하기 위한 새로운 훈련 전략을 제안합니다.

## ✨ Key Contributions

- U-Net의 인코더 브랜치 병목 계층에 완전 지도(fully supervised) 방식의 FC(Fully Connected) 계층 기반 픽셀 단위 손실(pixel-wise loss)을 도입했습니다.
- 두 계층으로 구성된 FC 서브넷(sub-net)이 병목 표현(bottleneck representation)이 더 많은 의미론적 정보를 포함하도록 훈련시킵니다.
- 이러한 수정된 아키텍처가 원래 U-Net과 비교할 만한 성능을 낼 수 있음을 보여줍니다.
- FC 계층 기반 서브넷은 픽셀 단위 교차 엔트로피 손실(pixel-wise cross entropy loss)을 사용하여 훈련하고, U-Net 아키텍처는 L1 손실(L1 loss)을 사용하여 훈련합니다.

## 📎 Related Works

- **기존 의료 영상 분할 기법:** 임계값 기반(thresholding based), 영역 성장(region growing), 클러스터링 기반(clustering based) 방법 등이 있습니다.
- **심층 학습 기반 기법:**
  - **U-Net [18]:** 생물학적 현미경 이미지 분할에 사용되는 인코더-디코더(encoder-decoder) 기반 네트워크로, 스킵 연결(skip connections)과 데이터 증강(data augmentation)을 활용합니다.
  - **V-Net [15]:** 3D 의료 영상 분할을 위해 설계되었으며, Dice 계수(Dice coefficient) 기반 손실 함수를 사용합니다.
  - **Progressive Dense V-net (PDV-Net) [10]:** CT 이미지에서 폐엽을 빠르게 분할하는 방법으로, 사전 정보 없이 자동 분할이 가능합니다.
  - **3D-CNN 인코더 [2]:** U-Net과 CEN [3]의 장점을 결합하여 병변 분할에 사용됩니다.

## 🛠️ Methodology

제안하는 네트워크는 세 부분으로 구성됩니다: 인코더, 병목 훈련 부분, 디코더.

- **인코더 부분:**
  - 일반적인 CNN 설계에 기반하며, $3 \times 3$ 필터를 가진 합성곱 블록(convolutional blocks)을 포함합니다.
  - 각 합성곱 블록 뒤에는 ReLU(Rectified Linear Unit)와 스트라이드(stride) 2를 가진 $2 \times 2$ 최대 풀링(max pooling) 연산을 사용하여 다운샘플링(down-sampling) 계층이 옵니다.
  - 이는 이미지의 공간적 크기를 줄이면서 채널 수를 늘려 유용한 정보를 인코딩합니다.
- **병목 훈련 부분:**
  - 두 개의 완전 연결(fully connected) 계층으로 구성됩니다.
  - 입력 이미지와 예측된 분할 맵이 정합되도록 선형 변환을 사용하여 실제 분할 맵(ground-truth segmentation map)을 예측합니다.
  - 이 부분은 픽셀 단위 교차 엔트로피 손실로 훈련되어 병목 계층에 명시적으로 지도 신호를 제공합니다.
- **디코더 부분:**
  - 업샘플링(up-sampling) 역합성곱(deconvolutional) 블록에 기반합니다.
  - 중간 역합성곱 계층에서 특징 맵의 크기를 늘리기 위해 $2 \times 2$ 업컨볼루션(up-convolutions)을 사용합니다.
  - U-Net의 스킵 연결 아키텍처를 따라 인코더 계층의 특징 맵을 디코더 네트워크의 해당 계층에 연결합니다.
  - 그 후, $3 \times 3$ 합성곱 필터와 ReLU를 사용하여 비선형성(non-linearity)을 추가합니다.
- **손실 함수:** 병목 부분의 FC 계층 서브넷은 픽셀 단위 교차 엔트로피 손실로 훈련되며, 주 U-Net 아키텍처는 L1 손실로 훈련됩니다.

## 📊 Results

제안된 CNN 기반 방법은 민감도(Sensitivity), 특이도(Specificity), 정확도(Accuracy) 기준으로 평가되었습니다. 각 지표는 다음과 같이 정의됩니다:

- Sensitivity $= N_{tp} / N_p$
- Specificity $= N_{tn} / N_n$
- Accuracy $= (N_{tp} + N_{tn}) / (N_{tp} + N_{tn} + N_{fn} + N_{fp})$

**MRI 이미지 성능:**

- 특이도(Specificity): 0.926
- 민감도(Sensitivity): 0.939
- 정확도(Accuracy): 0.913

**CT-scan 이미지 성능:**

- 특이도(Specificity): 0.961
- 민감도(Sensitivity): 0.972
- 정확도(Accuracy): 0.976

실험 결과는 MRI 및 CT 스캔 이미지에서 유망한 성능을 보여줍니다.

## 🧠 Insights & Discussion

- 병목 계층에 명시적인 지도 신호를 제공함으로써 인코더가 입력 이미지에서 더욱 효과적인 문맥 정보를 추출할 수 있습니다. 이는 인코더의 특징 추출 능력을 향상시킵니다.
- U-Net의 핵심 요소인 스킵 연결은 인코더의 중간 계층에서 인코딩된 정보를 디코더로 직접 전달하여 정보 손실을 방지하고 풍부한 문맥 정보를 활용하는 데 기여합니다.
- 제안된 기법은 MRI 및 CT 스캔 이미지에서 좋은 분할 결과를 달성하여 의료 영상 분석에서 실질적인 적용 가능성을 시사합니다.
- 논문은 이 방법이 기존 U-Net과 "비슷한 결과"를 낸다고 주장하지만, 동일한 데이터셋에서 기존 U-Net과의 직접적인 정량적 비교나 통계적 유의성 분석은 제시되지 않았습니다.

## 📌 TL;DR

**문제:** 의료 영상 분할을 위한 U-Net 인코더의 특징 추출 능력 향상.
**방법:** U-Net 아키텍처의 병목 계층에 두 개의 완전 연결 계층과 픽셀 단위 교차 엔트로피 손실을 사용하는 완전 지도 학습 전략을 도입하여 의미론적 정보 인코딩을 강화했습니다. 주 U-Net은 L1 손실로 훈련됩니다.
**결과:** 제안된 방법은 MRI 및 CT 스캔 이미지에서 유망한 분할 성능을 보여주며, 병목 계층의 지도 학습이 효과적인 특징 인코딩에 기여함을 시사합니다.
