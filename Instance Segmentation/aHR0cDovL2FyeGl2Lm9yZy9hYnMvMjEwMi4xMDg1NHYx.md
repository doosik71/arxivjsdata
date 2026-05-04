# Contour Loss for Instance Segmentation via k-step Distance Transformation Image

Xiaolong Guo, Xiaosong Lan, Kunfeng Wang, Shuxiao Li (2021)

## 🧩 Problem to Solve

본 논문은 인스턴스 세그멘테이션(Instance Segmentation)에서 예측된 마스크의 경계선(Contour) 부분이 불분명하고 부정확하게 생성되는 문제를 해결하고자 한다. Mask R-CNN과 같은 기존의 대표적인 방법론들은 전반적인 객체의 영역은 잘 잡아내지만, 경계선 근처에서의 픽셀 단위 정밀도가 떨어지는 경향이 있다.

이러한 문제는 단순한 시각적 품질 저하를 넘어, 비전 기반의 로봇 그리핑(Robot Grabbing)과 같이 객체의 정확한 외곽선 정보가 필수적인 실제 응용 분야에서 치명적인 성능 저하를 야기할 수 있다. 따라서 본 연구의 목표는 예측된 마스크와 정답(Ground-truth) 마스크가 전체적으로 일치할 뿐만 아니라, 특히 경계선 영역에서 최대한 일치하도록 최적화하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 거리 변환 이미지(Distance Transformation Image, DTI)를 활용하여 경계선 전용 감독 신호를 제공하는 **Contour Loss**를 제안하는 것이다. 주요 기여 사항은 다음과 같다.

1. **Contour Loss 제안**: DTI를 기반으로 예측된 경계선과 정답 경계선 사이의 거리를 측정하는 새로운 손실 함수를 설계하였다. 이를 통해 네트워크가 마스크의 외곽선 부분을 집중적으로 최적화하도록 유도한다.
2. **미분 가능한 k-step DTI 모듈 설계**: 기존의 DTI 계산 방식은 미분이 불가능하여 딥러닝 프레임워크 내에서 직접 학습하기 어려웠다. 본 논문은 미분 가능한 형태의 근사치 계산 모듈인 **kSDT(k-Step Distance Transformation)**를 설계하여, 학습 과정 중에 온라인으로 truncated DTI를 계산하고 역전파(Backpropagation)를 통해 학습할 수 있게 하였다.
3. **높은 범용성**: 제안된 방법은 추론 네트워크의 구조를 변경하거나 추가적인 학습 파라미터를 늘리지 않고도 기존의 Mask R-CNN 및 HTC(Hybrid Task Cascade)와 같은 프레임워크에 통합하여 사용할 수 있다.

## 📎 Related Works

논문은 인스턴스 세그멘테이션 방법론을 크게 두 가지 방향으로 분류하여 설명한다.

- **주류 방법론**: 세그멘테이션 기반 방식(FCN, Instance-FCN, SOLO 등)과 검출 기반 방식(Mask R-CNN, PA-Net, HTC 등)이 있으며, 일반적으로 검출 기반 방식이 더 우수한 성능을 보인다.
- **경계선 정보 활용 방법론**:
  - **에지 확장 방식**: 정답 마스크의 에지를 $k$ 픽셀만큼 확장하여 학습시키는 방법이 있으나, 하이퍼파라미터 $k$에 매우 민감하며 이론적 근거가 부족하다는 한계가 있다.
  - **Sobel 연산자 기반 MSE**: Sobel 연산자로 에지 이미지를 추출하고 MSE(Mean Square Error) 손실을 사용하는 방식이 제안되었으나, 단순한 위치 정보만 포함할 뿐 거리 정보가 부족하다.
  - **DTI 예측 방식**: DTI 자체를 예측하는 복잡한 네트워크 분기를 추가하는 방법이 있으나, 추론 과정에서 별도의 후처리가 필요하고 외곽선에서 먼 지역의 노이즈에 취약한 불안정성이 존재한다.

본 연구는 이러한 기존 방식과 달리, **미분 가능한 truncated DTI**를 손실 함수에 직접 통합함으로써 네트워크 구조의 변경 없이 외곽선 최적화를 달성하며, 거리 정보를 인코딩하여 단순 에지 매칭보다 정교한 최적화를 수행한다는 점에서 차별성을 가진다.

## 🛠️ Methodology

### 1. 전체 파이프라인

Contour Loss의 계산 흐름은 다음과 같다.

1. 마스크 분기(Mask Branch)에서 출력된 예측 마스크 응답($M^R$)을 선택한다.
2. 미분 가능한 시뮬레이션 이진화 함수 $B(\cdot)$를 통해 예측 마스크 $M^P$를 생성한다.
3. Sobel 연산자를 적용하여 예측된 경계선 응답($\Omega^{PCR}$)과 정답 경계선 응답($\Omega^{GCR}$)을 추출한다.
4. kSDT 모듈을 통해 각각의 $k$-step DTI($\Gamma_P^k, \Gamma_{GT}^k$)를 계산한다.
5. 두 경계선 간의 거리 차이를 계산하여 Contour Loss를 산출하고, 기존 손실 함수와 함께 공동 학습한다.

### 2. 주요 구성 요소 및 방정식

#### (1) 시뮬레이션 이진화 (Simulated Binarization)

추론 단계의 이진화 연산을 학습 과정에서도 미분 가능한 형태로 모사하기 위해 다음과 같은 시그모이드 기반 함수를 사용한다.
$$B(x) = \frac{1}{1 + e^{-\gamma(x-T)}}$$
여기서 $\gamma=20, T=0.5$로 설정되어 있으며, 이를 통해 $M^P = B(M^R)$을 얻는다.

#### (2) 경계선 응답 계산 (Contour Response)

Sobel 연산자($Sobel_x, Sobel_y$)를 사용하여 $x, y$ 방향의 기울기를 계산하고 그 절대값의 평균을 통해 경계선 응답을 구한다.
$$\Omega = \frac{1}{2} \left( |M \ast Sobel_x| + |M \ast Sobel_y| \right)$$

#### (3) 미분 가능한 k-step DTI (kSDT)

DTI는 각 픽셀에서 가장 가까운 배경 픽셀까지의 거리를 나타낸다. 본 논문은 이를 구현하기 위해 반복적인 팽창(Dilation) 연산을 사용한다.

- **Dilation 연산**: $3 \times 3$ Smooth 커널로 컨볼루션을 수행한 후 다시 $B(\cdot)$ 함수로 이진화하여 픽셀을 확장한다.
- **k-step 계산**:
    $$I_{Mask}^k = D(I_{Mask}^{k-1})$$
    $$I_{OPDT}^k = I_{Mask}^k \oplus I_{OPDT}^{k-1}$$
    $$\Gamma^k = (k+1) - I_{OPDT}^k$$
여기서 $D(\cdot)$는 미분 가능한 1단계 팽창 연산자이며, $\oplus$는 요소별 덧셈이다. 최종적으로 $k$라는 임계값으로 거리를 제한(Truncation)하여 경계선 근처의 최적화에 집중하게 한다.

#### (4) Contour Loss 정의

두 경계선 사이의 거리를 측정하기 위해, 한 경계선의 응답을 다른 경계선의 DTI 위에 투영하여 누적합을 구한다. 연속적인 값으로 계산하기 위해 Hadamard product($\otimes$)와 Global Average Pooling(GAP)을 사용한다.
$$d(\Omega^{PCR}, \Omega^{GCR}) = \frac{1}{2} \left[ \frac{GAP(\Omega^{PCR} \otimes \Gamma_{GT}^k) + \epsilon}{GAP(\Omega^{PCR}) + \epsilon} + \frac{GAP(\Omega^{GCR} \otimes \Gamma_P^k) + \epsilon}{GAP(\Omega^{GCR}) + \epsilon} \right]$$
최종 손실 함수는 다음과 같이 정의된다.
$$L = L_{cls} + L_{box} + L_{mask} + L_{Contour}$$

## 📊 Results

### 1. 실험 설정

- **데이터셋**: COCO 2014 (학습 82,783장, 검증 5,000장)
- **백본 및 프레임워크**: Res-50+FPN, Res-101+FPN, Res-X-101 기반의 Mask R-CNN 및 HTC.
- **평가 지표**: COCO AP (mAP, $AP_{50}, AP_{75}, AP_s, AP_m, AP_l$).
- **특이사항**: 초기 120K 반복 학습 후, Contour Loss를 활성화하여 180K까지 추가 학습하는 보조적(Auxiliary) 방식으로 진행하였다.

### 2. 주요 결과

- **하이퍼파라미터 $k$의 영향**: $k=1 \sim 5$ 범위에서 성능이 안정적이었으며, $k=2$일 때 가장 높은 성능을 보였다. (mAP 기준 baseline 34.28% $\rightarrow$ 34.54%로 상승).
- **Ablation Study**: 단순 MSE Edge Loss나 MSE Contour Loss보다 제안된 Contour Loss가 더 높은 정확도를 기록하였다. 이는 단순 픽셀 값의 차이가 아니라 거리 기반의 정교한 최적화가 이루어졌음을 시사한다.
- **범용성 검증**: 다양한 백본(Res-50, Res-101, Res-X-101)과 다른 프레임워크(HTC) 모두에서 mAP가 상승하는 결과를 보였다. 특히 $AP_{75}$와 $AP_l$(대형 객체)에서 유의미한 상승이 관찰되었다.
- **정성적 결과**: 시각화 결과, 기존 Mask R-CNN보다 객체의 외곽선이 훨씬 더 뚜렷하고 정교하게 세그멘테이션되는 것이 확인되었다.

## 🧠 Insights & Discussion

본 논문은 인스턴스 세그멘테이션의 고질적인 문제인 '경계선 부정확성'을 해결하기 위해 고전적인 이미지 처리 기법인 DTI를 딥러닝의 손실 함수로 성공적으로 통합하였다.

**강점**은 다음과 같다.
첫째, 네트워크 구조를 전혀 수정하지 않고 손실 함수만 추가함으로써 즉시 적용 가능한 높은 범용성을 확보하였다.
둘째, DTI 계산 과정을 미분 가능하게 설계하여 end-to-end 학습이 가능하게 만들었다.
셋째, truncated DTI를 통해 배경의 불필요한 정보는 배제하고 경계선 근처의 픽셀들에만 집중하여 최적화 효율을 높였다.

**한계 및 논의사항**은 다음과 같다.
학습 전략 면에서, 초기 단계부터 Contour Loss를 적용하지 않고 120K iteration 이후에 적용했다는 점이다. 이는 초기 학습 단계에서 마스크 분기가 불안정할 때 Contour Loss가 학습을 방해할 수 있다는 가정을 전제로 한 것인데, 이는 최적의 학습 스케줄링에 대한 추가적인 연구가 필요함을 의미한다. 또한, 본 논문은 COCO 데이터셋에 국한되어 실험이 진행되었으므로, 매우 복잡한 배경이나 극소형 객체에서도 동일한 효과가 나타날지는 추가 검증이 필요하다.

## 📌 TL;DR

본 연구는 인스턴스 세그멘테이션 마스크의 경계선 정밀도를 높이기 위해, 미분 가능한 **k-step 거리 변환 이미지(kSDT)** 모듈과 이를 활용한 **Contour Loss**를 제안하였다. 제안된 방법은 Mask R-CNN 등의 기존 구조 변경 없이 통합 가능하며, COCO 데이터셋 실험을 통해 경계선 영역의 세그멘테이션 정확도를 유의미하게 향상시킴을 입증하였다. 이 연구는 정밀한 외곽선 정보가 필수적인 의료 영상 분석이나 로봇 제어 분야의 세그멘테이션 성능 향상에 기여할 가능성이 높다.
