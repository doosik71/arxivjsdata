# Self-Supervised Instance Segmentation by Grasping

YuXuan Liu, Xi Chen, Pieter Abbeel (2023)

## 🧩 Problem to Solve

로봇 공학의 다양한 응용 분야(물체 파지, 조작, 배치 등)에서 Instance Segmentation은 핵심적인 기술이다. 하지만 기존의 딥러닝 기반 Segmentation 모델들은 COCO와 같은 대규모의 정답 라벨링 데이터셋(Labeled Dataset)에 크게 의존한다. 

실제 로봇이 작동하는 환경에서는 지속적으로 새로운 물체가 유입되는데, 매번 사람이 수작업으로 수천 개의 객체를 라벨링하는 것은 비용과 시간이 매우 많이 소요되어 비현실적이다. 따라서 본 논문은 **인간의 개입(Human Annotation) 없이 로봇이 스스로 물체와의 상호작용을 통해 Instance Segmentation을 학습할 수 있는 자기지도학습(Self-supervised Learning) 방법론**을 제안하는 것을 목표로 한다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 **"로봇이 물체를 집어 올리면 해당 물체가 장면에서 사라진다"**는 물리적 직관을 이용하는 것이다.

1. **Grasp Segmentation Model (GSM) 제안**: 파지 전(Before)과 후(After)의 이미지, 그리고 파지 위치 정보를 이용하여 파지된 물체의 마스크를 예측하는 모델을 구축하였다.
2. **자기지도 학습 루프 구축**: GSM을 통해 수집한 수많은 물체 마스크를 활용하여, 이를 다른 이미지에 합성하는 "Cut-and-Paste" 방식과 Stable Diffusion 기반의 "Inpainting" 기술을 결합해 대량의 가상 학습 데이터를 자동으로 생성하였다.
3. **성능 향상 및 실효성 검증**: 제안 방법론을 통해 학습된 모델이 10배 더 많은 라벨링 데이터를 사용한 모델보다 우수한 성능을 보였으며, 실제 로봇 파지 시스템에서 파지 실패율을 3배 이상 낮추었음을 입증하였다.

## 📎 Related Works

### 1. Instance Segmentation
Mask2Former와 같이 Transformer 기반의 최신 모델들이 높은 성능을 보이고 있으나, 이들은 모두 대규모 라벨링 데이터셋을 전제로 한다. 로봇 환경처럼 새로운 객체가 계속 추가되는 상황에서는 한계가 명확하다.

### 2. Self-Supervised Segmentation
기존에는 파지 전후 이미지의 차이를 이용하는 Image Subtraction 방식이 제안되었다. 하지만 이 방식은 가림(Occlusion), 반사(Reflection), 또는 파지 대상 외의 다른 물체가 움직였을 때 매우 취약하다는 한계가 있다. 또한 Optical Flow 방식은 고대역폭의 비디오 데이터가 필요하며, 밀어내기(Pushing) 작업에 최적화되어 있어 일반적인 파지 환경에는 부적합하다.

### 3. Moving Object Detection
배경 제거(Background Subtraction) 방식은 움직이는 모든 객체를 탐지하지만, 로봇이 파지한 '특정 객체'만을 분리해내지는 못한다. 특히 배경이 계속 변하는 로봇 작업 환경에서는 배경 모델을 학습시키는 것이 어렵다.

## 🛠️ Methodology

### 1. Grasp Segmentation Model (GSM)
GSM은 파지 전 이미지 $i_b$, 파지 후 이미지 $i_a$, 파지 마스크 $g$, 그리고 두 이미지의 차분 이미지 $(i_b - i_a)$를 입력으로 받아 물체의 가시적 마스크 $m_v$와 가려진 부분까지 포함한 Amodal 마스크 $m_a$를 예측한다.

- **아키텍처**: ResNet-50 백본과 Feature Pyramid Network (FPN)를 사용하며, COCO 데이터셋으로 사전 학습된 Mask-RCNN의 가중치로 초기화하였다.
- **Upsampling**: FPN의 stride-4 특징 맵에 Gated Residual Convolution 블록과 Transposed Convolution을 적용하여 원본 해상도로 복원한다. 이때 Gating 연산 $y = x + \sigma(C_2(x))C_1(x)$를 통해 반사광과 같이 불필요한 특징을 효과적으로 제거한다.
- **손실 함수**: 가중치가 적용된 Binary Cross Entropy (BCE) 손실 함수를 사용한다.
$$w^{CE}(\hat{m}, m, w) = \frac{1}{n} \sum_{i=1}^{n} w_i (m_i \log \hat{m}_i + (1-m_i) \log(1-\hat{m}_i))$$
- **가중치 전략**: 클래스 불균형을 해소하기 위해 6가지 가중치 $w^{(1)} \dots w^{(6)}$를 사용한다. 특히 물체 경계면의 정확도를 높이기 위해 Max-pooling을 적용한 경계 영역에 높은 가중치를 부여한다.

### 2. Instance Segmentation 모델 학습 파이프라인
GSM을 통해 얻은 데이터를 바탕으로 일반적인 Instance Segmentation 모델(Mask2Former)을 학습시키는 과정은 다음과 같다.

1. **마스크 수집 및 필터링**: 
    - **가림 필터**: $\frac{S_v}{S_a} > 0.95$ (가시적 영역과 전체 영역의 비율이 높은 것만 선택)
    - **불확실성 필터**: Binary Entropy $H(\hat{m}) = -\hat{m}\log\hat{m} - (1-\hat{m})\log(1-\hat{m})$를 계산하여 평균 엔트로피가 $0.1$ 미만인 높은 확신도의 마스크만 유지한다.
    - **연결성 필터**: OpenCV의 `findContours`를 이용해 1~2개의 연속된 덩어리로 이루어진 마스크만 선택한다.
2. **Cut-and-Paste 증강**: 필터링된 물체 마스크를 무작위 이미지에 합성한다. 회전(0~360도), 스케일(0.75x~1.25x), 위치 변화를 주어 복잡한 장면을 생성한다.
3. **Stable Diffusion Inpainting**: 단순 합성은 경계면이 부자연스러워 모델이 가짜 특징을 학습할 위험이 있다. 이를 해결하기 위해 합성된 객체의 경계면을 5픽셀 확장하여 마스크를 만들고, 사전 학습된 Stable Diffusion 모델로 4단계의 denoising 과정을 거쳐 경계면을 자연스럽게 다듬는다.
4. **모델 학습**: 이렇게 생성된 고품질의 합성 이미지와 마스크를 사용하여 Mask2Former 모델을 학습시킨다.

## 📊 Results

### 1. Grasp Segmentation 성능 평가
전통적인 Image Subtraction, MOG, LSBP 방식과 비교한 결과, 제안된 GSM(Grasp-200-Filter)이 $92.6\%$의 mIOU를 기록하며 압도적인 성능을 보였다. 특히 traditional 방식들이 반사광이나 주변 물체의 움직임에 취약한 반면, GSM은 이를 견고하게 처리하였다.

### 2. Instance Segmentation 성능 평가 (Table II)
- **데이터 효율성**: 제안 방법(Paste-Grasp-Inpaint)으로 학습한 모델은 라벨링 데이터 100장만 사용했음에도 불구하고, 라벨링 데이터 1,000장을 사용한 Mask2Former-1000 모델과 대등하거나 더 높은 AP(Average Precision) 성능을 보였다.
- **Inpainting의 효과**: 단순 합성(Paste-Grasp)보다 Inpainting을 적용했을 때 성능이 크게 향상되었는데, 이는 실제 이미지와 유사한 데이터 분포가 학습에 결정적임을 시사한다.

### 3. 실제 로봇 파지 실험 (Table III)
실제 로봇 시스템에 적용하여 파지 에러율(Grasp Error Rate)을 측정한 결과:
- **Paste-Subtract-Robust**: $29.02\%$
- **Mask2Former-1000**: $8.78\%$
- **Paste-Grasp-Inpaint**: $8.22\%$
결과적으로 제안 방법론이 Image Subtraction 기반 방식보다 에러율을 3배 이상 낮췄으며, 대량의 정답 데이터를 학습한 모델과 유사한 수준의 성능을 달성하였다.

## 🧠 Insights & Discussion

### 강점 및 분석
- **상호작용을 통한 학습**: 물리적 파지라는 상호작용을 통해 정답 라벨을 스스로 생성함으로써, 데이터 수집 비용 문제를 근본적으로 해결하였다.
- **합성 데이터의 현실화**: Diffusion 모델을 이용한 Inpainting이 합성 데이터의 고질적인 문제인 '부자연스러운 경계선' 문제를 해결하여, 모델이 실제 세계의 특징을 더 잘 학습하게 만들었다.
- **Robust Set Loss의 한계**: 기존 연구에서는 노이즈가 많은 마스크를 위해 Robust Set Loss를 사용했으나, 본 논문에서는 GSM의 정확도가 매우 높기 때문에 일반적인 Cross Entropy Loss를 사용하는 것이 더 효과적임을 밝혔다.

### 한계 및 논의사항
- **초기 학습 데이터**: GSM 자체를 학습시키기 위해 100~200장의 소규모 라벨링 데이터가 필요하다. 완전한 Zero-label 학습은 아니며, 최소한의 부트스트래핑 데이터가 필요하다는 가정이 존재한다.
- **실패 사례**: 분석 결과, 하나의 물체를 두 개로 분리하여 인식하거나, 반대로 두 물체를 하나로 인식하는 경우가 여전히 발생한다. 이는 Segmentation 모델의 근본적인 한계이며 향후 개선 과제로 보인다.

## 📌 TL;DR

본 논문은 로봇이 물체를 파지하는 행위를 통해 스스로 Instance Segmentation 마스크를 생성하고, 이를 Diffusion 모델 기반의 정교한 합성 이미지로 확장하여 학습하는 자기지도학습 프레임워크를 제안한다. 이 방법은 사람이 라벨링한 데이터가 10배 적음에도 불구하고 대등한 성능을 내며, 실제 로봇의 파지 에러율을 획기적으로 낮추어 로봇의 자가 학습 가능성을 입증하였다.