# U-Net Based Multi-instance Video Object Segmentation

Heguang Liu, Jingle Jiang (2019)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 문제는 **Multi-instance Video Object Segmentation (VOS)**이다. 이는 비디오 시퀀스의 첫 번째 프레임에 제공된 어노테이션(annotation)만을 이용하여, 이후 모든 프레임에서 특정 객체 인스턴스들을 픽셀 수준으로 세그멘테이션하는 작업이다.

이 문제는 특히 여러 인스턴스가 서로를 가리는 occlusion(폐색) 현상이 발생할 때 트래킹 실패로 이어지기 쉽다는 점에서 매우 도전적인 과제이다. VOS는 자율주행 자동차, 모션 트래킹, 비디오 요약 등 다양한 분야에서 중요하게 활용되고 있다. 본 연구의 목표는 실시간 처리 요구사항을 충족하기 위해 시간적 정보(temporal information)를 사용하지 않으면서도 효율적인 다중 인스턴스 세그멘테이션 모델을 구현하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 아이디어는 **OSVOS(One-Shot Video Object Segmentation)의 fine-tuned layer 위에 U-Net 구조의 Fully Convolutional Network(FCN)를 결합**하는 것이다. 

가장 중요한 설계 전략은 **Instance Isolation** 기법을 도입한 것이다. 이는 복잡한 다중 인스턴스 세그멘테이션 문제를 여러 개의 단순한 이진 라벨링(binary labeling) 문제로 변환하여 처리하는 방식이다. 또한, 배경에 비해 객체 영역이 매우 작은 데이터셋의 특성을 고려하여 Weighted Cross Entropy Loss를 사용함으로써 클래스 불균형 문제를 해결하고자 하였다.

## 📎 Related Works

### 이미지 세그멘테이션 (Image Segmentation)
최신 이미지 시맨틱 세그멘테이션은 주로 Encoder-Decoder 구조를 따른다. Encoder는 VGG나 ResNet과 같은 사전 학습된 분류 네트워크를 사용하며, Decoder는 저해상도의 특징 맵을 픽셀 공간으로 투영하여 정밀한 분류를 수행한다. SegNet은 Transposed Convolutional layer를 도입했고, U-Net은 여기서 더 나아가 Skip Connection을 통해 세밀한 정보를 보존하도록 개선되었다. 한편, Mask R-CNN과 같은 Region-Based 방식은 객체 검출 후 세그멘테이션을 수행하는 2단계 구조를 가진다.

### 비디오 객체 세그멘테이션 (Video Object Segmentation)
본 논문은 세 가지 주요 접근 방식을 언급한다.
1. **OSVOS**: VGG-16 기반의 FCN을 사용하며, 비디오의 첫 프레임으로 모델을 fine-tuning 하여 해당 시퀀스 전용 모델을 생성한다. 시간적 정보를 사용하지 않는 현재의 State-of-the-Art(SOTA) 방식이다.
2. **MaskTrack**: 이전 프레임의 예측 마스크와 광학 흐름(optical flow) 정보를 입력으로 사용하는 2-스트림 네트워크 방식이다.
3. **Recurrent Mask Propagation**: 심층 순환 신경망(deep recurrent network)을 통해 시간적 연속성을 학습하며, 장기간의 occlusion 이후에도 객체를 재식별할 수 있는 능력을 갖추고 있다.

본 연구는 실시간성을 위해 temporal information을 사용하지 않는 OSVOS의 접근 방식을 계승하되, 디코더 구조를 U-Net으로 개선하여 성능을 높이고자 하였다.

## 🛠️ Methodology

### 전체 시스템 구조
본 모델은 그림 2에서 제시된 바와 같이 **U-Net 기반의 Fully Convolutional Network** 구조를 가진다. 전체 파이프라인은 OSVOS의 사전 학습된 레이어를 기반으로 하며, 그 위에 U-Net의 인코더-디코더 구조가 얹혀진 형태이다.

### 주요 구성 요소 및 역할
1. **Instance Isolation**: 다중 인스턴스 마스크를 처리하기 위해 세 가지 방법을 실험하였으며, 최종적으로 **각 인스턴스를 개별적인 이진 라벨 입력($H \times W \times 1$)으로 분리**하여 처리하는 방식이 가장 효과적임을 확인하였다. 이를 통해 다중 인스턴스 문제를 단일 인스턴스 이진 세그멘테이션 문제로 단순화하였다.
2. **OSVOS Layer**: ImageNet으로 사전 학습된 VGG-16의 FC 레이어를 제거하여 FCN으로 변환한 후 DAVIS 데이터셋으로 학습시킨다. 이후 각 비디오의 첫 프레임을 사용하여 500회 반복 학습(fine-tuning)함으로써 해당 비디오에 최적화된 맞춤형 모델을 생성한다.
3. **U-Net Architecture**: 
    - **Contracting Path**: Convolutional layer와 Max pooling layer를 통해 이미지의 컨텍스트를 캡처한다.
    - **Expanding Path**: Up-sampling layer를 통해 해상도를 높인다.
    - **Skip Connection**: Contracting path의 고해상도 특징을 Expanding path의 출력과 병합하여 더욱 정밀한 경계선을 생성한다.

### 손실 함수 (Loss Function)
모델 학습 및 평가를 위해 다음 두 가지 손실 함수를 탐색하였다.

- **Weighted Cross Entropy Loss**:
$$L = -\sum_{x} \omega(x) p(x) \log q(x)$$
여기서 $p(x)$는 실제 분포, $q(x)$는 예측 분포이다. $\omega(x)$는 배경 대비 전경(foreground)의 픽셀 수 비율로 설정하여, 모델이 모든 픽셀을 배경으로만 예측하는 편향을 방지한다.

- **Dice Coefficient Loss**:
$$L = 1 - \frac{2|X \cap Y|}{|X| + |Y|}$$
두 영역의 겹침 정도를 측정하는 지표로, 0(불일치)에서 1(완전 일치) 사이의 값을 가진다.

실험 결과, 두 함수는 양의 상관관계를 보였으며 Weighted Cross Entropy가 수치적으로 더 안정적이었기에 학습에는 이를 사용하였다.

## 📊 Results

### 실험 설정
- **데이터셋**: DAVIS 2017 Challenge 데이터셋 (다중 인스턴스 챌린지 포함).
- **평가 지표**:
    - **Region Similarity ($\mathcal{J}$ mean)**: 예측 마스크 $M$과 정답 $G$ 사이의 IoU(Intersection over Union).
    - **Contour Accuracy ($\mathcal{F}$ mean)**: 경계선의 Precision과 Recall의 조화 평균.
- **환경**: TensorFlow 1.8, Python 3.6, NVIDIA Tesla P100 GPU 사용.

### 주요 결과
- **정량적 결과**: 
    - **Best U-Net**: $\mathcal{J}$ mean: $0.424$, $\mathcal{F}$ mean: $0.467$
    - **OSVOS**: $\mathcal{J}$ mean: $0.499$, $\mathcal{F}$ mean: $0.592$
    - **SegNet**: $\mathcal{J}$ mean: $0.347$, $\mathcal{F}$ mean: $0.214$
- **정성적 분석**: U-Net은 OSVOS보다 $\mathcal{F}$ 값(경계 정밀도)은 낮지만, **인스턴스 커버리지가 더 넓고 경계선이 더 부드러운(smoother contour)** 특성을 보인다. 따라서 정밀도보다는 재현율(Recall)이 중요한 시나리오(예: 보행자 세그멘테이션)에 더 적합하다.
- **구조적 분석**: SegNet의 성능이 저조한 이유는 Skip Connection의 부재로 인해 contracting 과정에서 고해상도 세부 정보를 소실하기 때문인 것으로 분석되었다.

### 하이퍼파라미터 튜닝
필터 수 [64, 128, 256, 512], 학습률 $4e-5$, 배치 사이즈 8일 때 최적의 성능을 보였으며, 이때 파라미터 수는 약 31M 개이다.

## 🧠 Insights & Discussion

### 강점 및 성과
본 모델은 급격한 외형 변화, 거리 변화, 배경 안개 등이 포함된 시퀀스에서도 객체를 안정적으로 추적하며, 유사한 움직임을 보이는 여러 객체가 겹쳐 있는 상황에서도 비교적 명확한 경계를 그려내는 능력을 보여주었다.

### 한계 및 실패 사례
1. **Multi-instance Occlusion**: 타겟 객체가 다른 객체를 지나칠 때, 이후 두 객체 모두를 추적하는 경향이 발생한다.
2. **Boundary Exit**: 객체가 이미지 경계 밖으로 나갔다가 다시 들어올 경우, 타겟을 놓치고 일부(예: 라이더 대신 오토바이만)만 추적하는 현상이 나타난다.

### 비판적 해석 및 논의
- **Mask R-CNN의 실패 이유**: Mask R-CNN은 인식(Recognition) 기반 알고리즘이므로, 학습 데이터(COCO)에 없는 클래스(예: 낙타)가 등장하면 이를 학습된 다른 클래스(예: 말)로 오인하여 형태를 강제로 수정하려는 경향을 보인다. 또한, 첫 프레임의 반지도 학습(semi-supervised) 정보를 활용하는 메커니즘이 부족하다.
- **결론적 고찰**: 본 연구는 U-Net 구조가 VOS의 디코더로서 매우 강력한 성능을 낼 수 있음을 입증하였다. 다만, 시간적 연속성을 고려하지 않은 한계를 극복하기 위해 향후 Recurrent Neural Network나 Adaptive re-identification 기법의 도입이 필요할 것으로 보인다.

## 📌 TL;DR

본 논문은 OSVOS의 fine-tuning 레이어 위에 **U-Net 기반의 FCN 구조를 결합**하고, **Instance Isolation** 기법과 **Weighted Cross Entropy Loss**를 적용하여 다중 인스턴스 비디오 객체 세그멘테이션 문제를 해결하였다. 제안된 모델은 SOTA인 OSVOS와 대등한 수준의 성능을 보이며, 특히 **더 부드러운 경계선과 높은 커버리지(Recall focus)**를 제공한다는 장점이 있다. 이는 실시간 처리가 필요한 환경에서 객체의 누락 없는 검출이 중요한 응용 분야에 유용하게 적용될 가능성이 높다.