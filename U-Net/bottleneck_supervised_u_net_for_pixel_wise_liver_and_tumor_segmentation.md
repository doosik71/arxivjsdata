# Bottleneck Supervised U-Net for Pixel-wise Liver and Tumor Segmentation

Song LI, Geoffrey K. F. Tso (2019)

## 🧩 Problem to Solve

본 논문은 CT 영상에서 간(Liver)과 간 종양(Liver Tumor)을 픽셀 단위로 정밀하게 분할(Segmentation)하는 문제를 해결하고자 한다. 의료 영상 분할은 임상 분석 및 의료적 개입을 위해 매우 중요하지만, 다음과 같은 기술적 난점이 존재한다.

첫째, 의료 영상 내 타겟 객체는 크기, 모양, 강도(Intensity)가 매우 불규칙하여 모델이 일관된 패턴을 학습하기 어렵다. 둘째, 타겟 조직이 주변의 다른 조직과 유사한 특징(강도, 크기, 모양 등)을 가지는 경우가 많아 위양성(False Positive) 문제가 심각하게 발생한다. 셋째, 의료 영상의 정답지(Annotation)를 작성하는 비용이 매우 높기 때문에 학습 데이터셋 내에서 양성 사례와 음성 사례의 불균형이 심하며, 이는 다시 위양성 문제를 악화시키는 원인이 된다.

따라서 본 연구의 목표는 이러한 불규칙성과 데이터 불균형 문제를 극복하고, 형태 왜곡을 제어하며 위양성 및 위음성(False Negative) 사례를 줄일 수 있는 정밀한 분할 모델을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 두 가지 측면에서 제시된다.

첫째, 기존 U-Net의 인코딩 경로(Encoding Path)를 강화한 **Base U-Net**을 제안한다. 여기에는 다양한 스케일의 정보를 캡처하기 위한 Inception modules, 파라미터 수를 줄이고 기울기 소실 문제를 완화하는 Dense modules, 그리고 수용 영역(Receptive Field)을 확장하여 세그멘테이션 성능을 높이는 Dilated convolution이 통합되어 있다.

둘째, 인코딩 U-Net과 세그멘테이션 U-Net으로 구성된 **Bottleneck Supervised (BS) U-Net** 구조를 제안한다. 핵심 직관은 정답 맵(Label map)으로부터 해부학적 정보(모양, 위치 등)를 학습한 인코딩 U-Net의 Bottleneck 특징 벡터를 생성하고, 이를 세그멘테이션 U-Net의 학습 가이드로 사용하여 타겟 객체의 해부학적 특징을 유지하도록 강제하는 것이다.

## 📎 Related Works

의료 영상 분할 연구는 크게 비지도 학습(Unsupervised) 방법과 지도 학습(Supervised) 방법으로 나뉜다.

비지도 학습 방법으로는 Watershed segmentation, Level set model, Cahn-Hilliard 방정식 기반의 위상 분리 접근법 등이 사용되었다. 이러한 방법들은 특정 인구 집단에 의존하지 않아 일반화 성능이 좋다는 장점이 있으나, 정답지를 통한 감독이 없기에 전반적인 성능이 지도 학습보다 떨어지는 한계가 있다.

지도 학습 방법으로는 SVM(Support Vector Machine)이나 FCN(Fully Convolutional Network), 그리고 U-Net과 같은 심층 신경망이 주로 사용되었다. 특히 U-Net은 수축 경로(Contracting path)와 확장 경로(Expanding path) 사이의 Skip connection을 통해 손실된 공간 정보를 복원함으로써 정밀한 로컬라이제이션을 가능하게 하여 의료 영상 분야에서 널리 쓰이고 있다. 본 논문은 이러한 U-Net 구조를 기반으로 하되, Bottleneck 지점에 추가적인 감독(Supervision)을 부여함으로써 기존 U-Net이 가진 형태 왜곡 및 위양성 문제를 해결하고자 차별화를 두었다.

## 🛠️ Methodology

### 1. Base U-Net 아키텍처

Base U-Net은 BS U-Net의 기본 구성 요소이며, 기존 U-Net의 인코더 부분에 세 가지 모듈을 추가하여 성능을 개선하였다.

- **Inception Module**: 다양한 크기의 필터를 사용하여 이미지의 전체적인 윤곽(Big picture)과 세부 디테일을 동시에 파악함으로써, 주변 장기와 유사한 간/종양의 경계를 더 정확히 식별하고 위양성을 줄인다.
- **Dense Module**: 이전 레이어의 특징 맵을 이후 레이어에 연결함으로써 특징 전파를 강화하고 파라미터 수를 줄여 과적합(Overfitting)을 방지한다.
- **Dilated Convolution**: 풀링(Pooling) 없이 수용 영역을 확장하여 세그멘테이션에 필요한 광범위한 문맥 정보를 효율적으로 획득한다.

### 2. Bottleneck Supervised (BS) U-Net 구조 및 학습 절차

BS U-Net은 **Encoding U-Net**과 **Segmentation U-Net**의 두 단계로 구성된다.

1. **Encoding U-Net 학습**: 먼저 Skip connection이 제거된 U-Net(Auto-encoder 형태)을 사용하여 정답 맵(Label maps)을 입력받아 다시 복원하도록 학습시킨다. 이 과정이 완료되면 Bottleneck 지점의 특징 벡터는 해당 객체의 해부학적 정보(Anatomical information)를 압축하여 가지고 있게 된다.
2. **Segmentation U-Net 학습**: 실제 강도 영상(Intensity image)을 입력으로 하여 분할을 수행하는 네트워크를 학습시킨다. 이때, 이 네트워크의 Bottleneck 특징 벡터가 앞서 학습된 Encoding U-Net의 특징 벡터와 유사해지도록 유도한다.

### 3. 가중치 맵 (Weight Map)

간의 경계 영역에서 분할 성능이 저하되는 문제를 해결하기 위해, 경계 부분에 더 많은 가중치를 부여하는 Weight map $W$를 설계하였다.
먼저 거리 맵 $D$(픽셀에서 간 윤곽선까지의 최단 거리)를 계산한 후, 다음과 같이 가중치 $A$와 최종 정규화된 가중치 $W$를 구한다.

$$A = (w \times F + 1)e^{-\frac{D^2}{2\sigma^2}}$$
$$W = \frac{A - \min A}{\max A - \min A}$$

여기서 $F$는 관심 영역(ROI)을 나타내는 이진 행렬이며, $w$는 중요도, $\sigma$는 분산이다.

### 4. 손실 함수 (Loss Function)

학습은 다음과 같은 손실 함수들을 통해 이루어진다.

- **Dice Loss**: 데이터 불균형 문제를 완화하기 위해 사용하며, 다음과 같이 정의된다.
$$\text{Dice loss} = 1 - \frac{2|A \cap B|}{|A| + |B|}}$$
- **Weighted Dice Loss**: 위에서 정의한 가중치 맵 $W$를 적용하여 경계 영역의 오차를 더 크게 반영한다.
$$\text{Weighted Dice loss} = 1 - \frac{2|W \times A \cap B|}{|W \times A| + |W \times B|}$$
- **Euclidean Loss**: Encoding U-Net의 특징 벡터 $T_1$과 Segmentation U-Net의 특징 벡터 $T_2$ 사이의 거리를 측정한다.
$$\text{Euclidean Loss} = \sum_{i} (\bar{T}_1^i - \bar{T}_2^i)^2$$

최종적으로 BS U-Net의 전체 손실 함수는 두 손실의 가중 평균으로 계산된다.
$$\text{Total loss} = w_1 \times \text{Dice loss} + w_2 \times \text{Euclidean Loss}$$

## 📊 Results

### 실험 설정

- **데이터셋**: LiTS(Liver Tumor Segmentation) 공개 데이터셋 (훈련 131개, 테스트 70개 3D CT 스캔).
- **전처리**: HU(Hounsfield Units) 값을 $[-200, 250]$ 범위로 클리핑하고 MinMax 정규화를 수행하였다. 3D 데이터를 2D 슬라이스로 분해하여 사용하였으며, 1채널 및 3채널 접근 방식을 모두 테스트하였다.
- **지표**: Dice per case (DPC), Dice global (DG), Volume Overlap Error (VOE), Relative Volume Difference (RVD) 등을 사용하였다.

### 주요 결과

1. **간 분할 (Liver Segmentation)**:
   - BS U-Net이 Original U-Net과 Base U-Net보다 우수한 성능을 보였다. 특히 1채널 접근 방식의 BS U-Net은 DG(Dice Global) 지표에서 Base U-Net보다 0.1 높고, DPC(Dice per case)에서 0.2 높은 수치를 기록하며 리더보드 3위에 올랐다.
   - 정성적 분석 결과, BS U-Net은 위양성 및 위음성 사례를 크게 줄였으며, 저해상도 이미지에서도 간의 전체 형태를 성공적으로 예측하였다.

2. **종양 분할 (Tumor Segmentation)**:
   - 간 분할 결과물과 종양 분할 네트워크를 연결한 **Cascaded structure**를 적용하였다.
   - 결과적으로 BS U-Net이 Base U-Net보다 DPC와 DG 측면에서 모두 우수하여, Bottleneck 지점의 감독 학습이 종양 분할에서도 유효함을 입증하였다.

## 🧠 Insights & Discussion

본 연구의 가장 큰 강점은 단순한 아키텍처 개선(Base U-Net)에 그치지 않고, **Bottleneck Supervision**이라는 학습 전략을 도입하여 모델이 해부학적 제약 조건을 학습하게 만든 점이다. 이는 딥러닝 모델이 흔히 겪는 형태 왜곡(Shape distortion) 문제를 효과적으로 억제하며, 의료 영상의 고질적인 문제인 위양성 사례를 줄이는 데 기여하였다.

다만, BS U-Net을 학습시키기 위해서는 추가적인 Encoding U-Net 학습 과정이 필요하다는 점이 잠재적 한계로 보일 수 있으나, 저자는 이 과정이 매우 빠르게 수렴(500회 반복 내 Dice 0.98 달성)하므로 연산 부담이 충분히 감내할 수준이라고 주장한다. 또한, 본 모델의 구조는 Base U-Net의 아키텍처를 다른 최신 모델로 교체함으로써 쉽게 확장 가능하다는 범용성을 가지고 있다.

## 📌 TL;DR

본 논문은 간 및 종양 분할을 위해 **Base U-Net**(Inception, Dense, Dilated Conv 통합)과 이를 활용한 **Bottleneck Supervised (BS) U-Net**을 제안하였다. 정답 맵의 특징을 학습한 인코더가 세그멘테이션 네트워크의 Bottleneck을 가이드함으로써 해부학적 일관성을 유지하고 위양성을 획기적으로 줄였다. LiTS 데이터셋 실험 결과, 기존 U-Net 대비 정량적 지표(DPC, DG) 및 정성적 형태 유지 능력이 크게 향상되었음을 확인하였다.
