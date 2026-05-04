# Mask2Former for Video Instance Segmentation

B. Cheng, A. Choudhuri, I. Misra, A. Kirillov, R. Girdhar, A. G. Schwing (2021)

## 🧩 Problem to Solve

본 논문은 Video Instance Segmentation (VIS) 문제를 해결하고자 한다. VIS는 비디오 내의 객체들을 시간축에 따라 동시에 세그멘테이션(Segmentation)하고 추적(Tracking)해야 하는 과제이다.

기존의 VIS 연구들은 이미지 세그멘테이션 모델을 확장하여 사용해 왔으나, 대부분 비디오 데이터만을 처리하기 위해 특별히 설계된 구조를 채택했다. 이로 인해 범용적인 이미지 세그멘테이션 연구와 비디오 세그멘테이션 연구가 서로 단절되어 발전해 왔다는 문제점이 있다. 본 논문의 목표는 범용 이미지 세그멘테이션 아키텍처인 Mask2Former가 아키텍처, 손실 함수, 혹은 학습 파이프라인의 수정 없이도 비디오 세그멘테이션 작업으로 간단히 확장(Trivially generalize)될 수 있음을 증명하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 비디오 시퀀스를 $T \times H \times W$ 크기의 3D 시공간 볼륨(Spatio-temporal volume)으로 취급하고, 이를 통해 3D 세그멘테이션 볼륨을 직접 예측하게 하는 것이다. 

즉, 이미지 세그멘테이션을 위해 설계된 Mask2Former의 쿼리(Query)를 프레임 간에 공유함으로써, 별도의 복잡한 추적 알고리즘 없이도 여러 프레임에 걸쳐 동일한 객체 인스턴스를 세그멘테이션하고 추적할 수 있다는 직관을 제시한다. 이를 통해 범용 이미지 세그멘테이션 모델이 비디오 도메인에서도 매우 강력한 성능을 낼 수 있음을 보여주었다.

## 📎 Related Works

비디오 인스턴스 세그멘테이션을 위한 기존 접근 방식은 크게 두 가지로 나뉜다.

1. **Per-frame methods (Online methods):** 각 프레임을 독립적으로 세그멘테이션한 후, 후처리 단계에서 인스턴스 마스크들을 연결하는 방식이다. MaskTrack R-CNN, TrackR-CNN, VPSNet 등이 있으며, 주로 Mask R-CNN에 track embedding 예측 기능을 추가하여 사용한다. 하지만 이러한 방식은 아키텍처 수정과 추가적인 손실 함수가 필요하여 이미지 모델을 그대로 적용하기 어렵다.
2. **Per-clip methods (Offline methods):** 비디오 클립을 하나의 3D 시공간 볼륨으로 처리하여 3D 마스크를 직접 예측하는 방식이다. 최근에는 DETR의 성공에 힘입어 VisTR, IFC, SeqFormer와 같은 Transformer 기반 아키텍처들이 Cross-attention을 통해 3D 볼륨을 처리하는 방식을 제안하였다.

본 논문은 이러한 기존 방법들이 비디오 전용으로 설계된 것과 달리, 범용 이미지 세그멘테이션 모델인 Mask2Former를 거의 수정 없이 적용함으로써 이미지와 비디오 세그멘테이션 연구의 간극을 좁혔다는 점에서 차별점을 가진다.

## 🛠️ Methodology

Mask2Former를 비디오 세그멘테이션에 적용하기 위해 저자들은 비디오 시퀀스를 $T \times H \times W$ 차원의 3D 볼륨으로 정의하고 다음과 같은 세 가지 주요 변경 사항을 적용하였다.

### 1. Joint Spatio-temporal Masked Attention
Transformer decoder에서 시공간 특징(Spatio-temporal features)에 대해 Masked Attention을 적용한다. 수식은 다음과 같다.

$$X^{l} = \text{softmax}(M^{l-1} + Q^{l}(K^{l})^{T})V^{l} + X^{l-1}$$

여기서 $l$은 레이어 인덱스이며, $X^{l} \in \mathbb{R}^{N \times C}$는 $l$번째 레이어의 $N$개 쿼리 특징을 의미한다. $K^{l}, V^{l} \in \mathbb{R}^{T H^{l} W^{l} \times C}$는 시공간 특징이다. 

이때 사용하는 3D attention mask $M^{l-1}$은 이전 레이어($l-1$)에서 예측된 3D 마스크를 0.5 임계값으로 이진화하여 생성하며, 다음과 같이 정의된다.

$$M^{l-1}(t, x, y) = \begin{cases} 0 & \text{if } M^{l-1}(t, x, y) = 1 \\ -\infty & \text{otherwise} \end{cases}$$

### 2. Temporal Positional Encoding
이미지 세그멘테이션 모델과의 호환성을 위해 시간축 위치 인코딩과 공간축 위치 인코딩을 분리하여 합산하는 방식을 사용한다.

$$e_{pos} = e_{pos-t} \oplus e_{pos-xy}$$

여기서 $e_{pos-t} \in \mathbb{R}^{T \times 1 \times 1 \times C}$는 시간적 위치 인코딩이고, $e_{pos-xy} \in \mathbb{R}^{1 \times H^{l} \times W^{l} \times C}$는 공간적 위치 인코딩이다. 두 인코딩 모두 임의의 길이를 처리할 수 있는 non-parametric sinusoidal positional encoding을 사용하며, $\oplus$는 numpy 스타일의 broadcasting 합산을 의미한다.

### 3. Joint Spatio-temporal Mask Prediction
마지막으로, $n$번째 쿼리에 대한 3D 마스크를 마스크 임베딩과 픽셀 임베딩의 단순 내적(Dot product)을 통해 예측한다.

$$M(n, t, h, w) = \text{sigmoid}(E_{mask}(:, n)^{T} \cdot E_{pixel}(:, t, h, w))$$

이 과정에서 클래스 분류(Classification)를 위한 연산은 이미지 세그멘테이션과 동일하게 유지된다.

## 📊 Results

### 실험 설정
- **데이터셋:** YouTubeVIS-2019 및 YouTubeVIS-2021.
- **훈련 세부사항:** AdamW 옵티마이저, step learning rate schedule 사용. 모든 모델은 COCO 인스턴스 세그멘테이션 모델로 초기화되었다. 훈련 시 효율성을 위해 비디오 클립은 $T=2$ 프레임으로 구성하였다. 특히, SeqFormer와 달리 COCO 이미지 데이터를 증강(Augmentation)으로 사용하지 않고 YouTubeVIS 훈련 데이터만 사용하였다.
- **추론:** 전체 비디오 시퀀스를 입력으로 넣어 후처리 없이 3D 마스크를 직접 예측하며, 상위 10개의 예측 결과만 유지한다.
- **백본(Backbone):** ResNet(R50V, R101V) 및 Swin Transformer(T, S, B, L)를 사용하였다.

### 정량적 결과
Mask2Former는 기존의 SOTA 모델들을 상회하는 성능을 보였다.

- **YouTubeVIS-2019:** Swin-L 백본을 사용했을 때 **60.4 AP** (최대 60.7 AP)를 기록하며 SOTA를 달성하였다.
- **YouTubeVIS-2021:** Swin-L 백본 기준 **52.6 AP** (최대 53.0 AP)를 기록하였다.
- **비교 분석:** 동일한 훈련 파라미터 설정 하에 IFC 모델보다 6 AP 이상 높은 성능을 보였으며, 데이터 증강 없이도 SeqFormer보다 우수한 성능을 입증하였다.

## 🧠 Insights & Discussion

본 논문은 범용 이미지 세그멘테이션 아키텍처가 매우 적은 수정만으로도 비디오 인스턴스 세그멘테이션 작업으로 확장될 수 있음을 보여주었다. 이는 복잡한 추적 메커니즘이나 비디오 전용 모듈 없이도, 쿼리를 시간축으로 공유하고 3D 볼륨을 직접 예측하는 것만으로 충분하다는 점을 시사한다.

**강점 및 가능성:**
- 아키텍처의 단순함과 범용성이 매우 높다.
- 이미지 세그멘테이션에서의 성공적인 설계가 비디오 도메인으로 전이될 수 있음을 증명하였다.
- 저자들은 Mask2Former의 범용성을 고려할 때, 비디오 시맨틱 세그멘테이션(Video Semantic Segmentation)이나 비디오 파놉틱 세그멘테이션(Video Panoptic Segmentation)에도 적용 가능할 것으로 전망한다.

**한계 및 논의사항:**
- 훈련 시 $T=2$라는 매우 짧은 프레임 길이를 사용하였는데, 이 설정이 실제 긴 비디오 시퀀스 추론 시의 성능에 어떤 영향을 미치는지에 대한 심층적인 분석은 부족하다.
- 추론 시 전체 비디오를 입력으로 넣기 때문에 비디오 길이에 따른 메모리 제약 문제가 발생할 수 있으며, 실제로 Swin-L 모델의 경우 메모리 제약으로 인해 해상도를 낮춰 평가하였다.

## 📌 TL;DR

본 논문은 범용 이미지 세그멘테이션 모델인 **Mask2Former**가 아키텍처나 손실 함수의 변경 없이, 비디오를 3D 시공간 볼륨으로 처리함으로써 **Video Instance Segmentation (VIS)** 작업에서 SOTA 성능을 달성할 수 있음을 보여주었다. 이는 이미지와 비디오 세그멘테이션 연구의 통합 가능성을 제시하며, 향후 범용적인 시공간 세그멘테이션 아키텍처 설계에 중요한 이정표가 될 것으로 보인다.