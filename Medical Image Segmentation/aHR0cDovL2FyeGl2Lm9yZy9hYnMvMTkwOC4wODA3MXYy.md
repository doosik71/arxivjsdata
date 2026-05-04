# End-to-End Boundary Aware Networks for Medical Image Segmentation

Ali Hatamizadeh, Demetri Terzopoulos, and Andriy Myronenko (2019)

## 🧩 Problem to Solve

본 연구는 의료 영상 분할(Medical Image Segmentation) 분야에서 딥러닝 모델, 특히 Convolutional Neural Networks(CNN)가 가진 근본적인 한계점을 해결하고자 한다. 일반적인 CNN 아키텍처는 이미지의 형태(Shape)보다는 질감(Texture) 정보에 편향되어 객체를 인식하는 경향이 있다. 그러나 의료 영상 분석의 실제 현장에서 전문의들은 해부학적 구조의 경계(Boundary)를 먼저 식별하고 이를 바탕으로 내부 영역을 분할하는 방식을 사용한다.

따라서 본 논문의 목표는 네트워크가 학습 과정에서 경계 정보를 명시적으로 학습하게 함으로써, 질감 편향성을 극복하고 의료 영상 분할의 정확도를 높이는 Boundary Aware CNN을 설계하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 세그멘테이션을 위한 메인 스트림 외에 경계 정보만을 전문적으로 처리하는 **Shape Stream(Edge Branch)**을 추가하고, 이를 **Edge-aware loss**로 감독하여 네트워크 전체에 구조적 정규화(Structured Regularization) 효과를 주는 것이다.

단순히 경계를 예측하는 것이 목적이 아니라, 경계 학습을 통해 메인 스트림의 인코더가 경계의 중요성을 내면화하도록 유도함으로써 최종적인 세그멘테이션 성능을 향상시키는 것이 이 설계의 핵심이다.

## 📎 Related Works

기존의 세그멘테이션 방법론들은 픽셀 단위의 분류에 집중하였으며, 최근에는 객체의 경계를 직접 예측하여 다양한 비전 작업의 성능을 높이려는 시도들이 있었다. 논문에서는 다음과 같은 관련 연구들을 언급한다.

- **Yu et al. [11, 12]:** 카테고리별 경계 활성화를 위해 skip-layer 아키텍처를 제안하거나, 레이블 정렬(Label alignment) 문제를 해결하여 경계 학습의 질을 높였다.
- **Acuna et al. [1]:** 클래스 경계에 속하는 픽셀을 식별하고, 경계의 법선 방향(Normal direction)으로 최대 응답을 갖도록 강제하는 손실 함수를 제안하였다.
- **Takikawa et al. [10]:** Gated-shape CNN을 통해 고수준 활성화 함수의 노이즈를 제거하고 경계 관련 정보를 별도로 처리하는 게이트 메커니즘을 도입하였다.
- **Hu et al. [6]:** 객체 검출, 세그멘테이션, 인스턴스 경계 검출을 하나의 네트워크에서 통합하여 학습하는 다중 브랜치 프레임워크를 제시하였다.

본 연구는 이러한 경계 학습 아이디어를 의료 영상 분할에 적용하되, 경계 예측 자체가 목적이 아니라 세그멘테이션 성능 향상을 위한 보조 수단으로 활용한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

### 전체 시스템 구조

제안된 네트워크는 세그멘테이션을 담당하는 **Main Encoder-Decoder Stream**과 경계 정보를 처리하는 **Shape Stream**으로 구성된다.

1. **Encoder 단계:** 메인 스트림의 각 해상도 레벨에는 두 개의 Residual Block이 있으며, 이들의 출력값은 Shape Stream의 대응하는 해상도 레벨로 전달된다.
2. **Shape Stream:** $1 \times 1$ Convolution을 거친 후 Attention Layer를 통과한다. 이후 Connection Residual Block을 거쳐 최종적으로 Dilated Spatial Pyramid Pooling(DSPP) 레이어로 전달된다.
3. **Fusion:** Shape Stream의 출력과 메인 스트림 인코더의 출력이 결합되어 최종 세그멘테이션 맵을 생성한다.

### Attention Layer

Attention Layer는 메인 스트림의 특징 맵($m^l$)과 이전 Shape Stream의 출력($s^l$)을 결합하여 경계 정보를 정제한다.

먼저, 두 입력을 Concatenation 한 후 $1 \times 1$ Convolution($C_{1\times1}$)과 시그모이드 함수($\sigma$)를 적용하여 Attention Map $\alpha^l$을 생성한다.
$$\alpha^l = \sigma(C_{1\times1}(s^l \parallel m^l))$$

이후, 입력 $s^l$에 $\alpha^l$을 요소별 곱셈(Element-wise multiplication)하여 최종 출력 $o^l$을 얻는다.
$$o^l = s^l \cdot \alpha^l$$

### 학습 목표 및 손실 함수

본 모델은 세그멘테이션 결과와 경계 예측 결과를 동시에 감독하는 Joint Learning 방식을 사용한다. 전체 손실 함수 $L_{total}$은 다음과 같이 세 가지 항의 합으로 정의된다.
$$L_{total} = \lambda_1 L_{Dice}(y_{pred}, y_{true}) + \lambda_2 L_{Dice}(s_{pred}, s_{true}) + \lambda_3 L_{Edge}(s_{pred}, s_{true})$$

여기서 $y$는 세그멘테이션 예측값, $s$는 경계 예측값이며, $s_{true}$는 정답 마스크 $y_{true}$의 공간적 기울기(Spatial gradient)를 계산하여 얻는다.

- **Dice Loss:** 클래스 불균형 문제를 해결하기 위해 사용되며, 다음과 같이 정의된다.
$$L_{Dice} = 1 - \frac{2 \sum y_{true} y_{pred}}{\sum y_{true}^2 + \sum y_{pred}^2 + \epsilon}$$

- **Edge Loss:** 경계 픽셀과 비경계 픽셀 사이의 심한 불균형을 처리하기 위해 가중치 기반의 Binary Cross Entropy loss를 사용한다.
$$L_{Edge} = -\beta \sum_{j \in y^+} \log P(y_{pred,j}=1|x;\theta) - (1-\beta) \sum_{j \in y^-} \log P(y_{pred,j}=0|x;\theta)$$
여기서 $\beta$는 전체 픽셀 대비 비경계 픽셀의 비율이며, $y^+$와 $y^-$는 각각 경계와 비경계 픽셀 집합을 의미한다.

## 📊 Results

### 실험 설정

- **데이터셋:** BraTS 2018 데이터셋을 사용하였으며, 다중 모달리티 중 T1c 모달리티만을 입력으로 사용하였다.
- **분할 대상:** Tumor Core (TC) 영역을 대상으로 하였으며, 3D 볼륨에서 추출한 2D axial slice를 사용하였다.
- **비교 모델:** U-Net, V-Net.
- **평가 지표:** Dice Score, Jaccard Index, Hausdorff Distance.

### 정량적 결과

실험 결과, 제안된 모델이 모든 지표에서 기존 모델들을 상회하였다.

| Model | Dice Score | Jaccard Index | Hausdorff Distance |
| :--- | :---: | :---: | :---: |
| U-Net | $0.731 \pm 0.230$ | $0.805 \pm 0.130$ | $3.861 \pm 1.342$ |
| V-Net | $0.769 \pm 0.270$ | $0.837 \pm 0.140$ | $3.667 \pm 1.329$ |
| Ours (no edge loss) | $0.768 \pm 0.236$ | $0.832 \pm 0.136$ | $3.443 \pm 1.218$ |
| **Ours** | $\mathbf{0.822 \pm 0.176}$ | $\mathbf{0.861 \pm 0.112}$ | $\mathbf{3.406 \pm 1.196}$ |

### 분석 및 고찰

- **Edge Loss의 영향:** Edge loss 없이 동일한 아키텍처로 학습했을 때의 성능은 V-Net과 유사한 수준으로 떨어진다. 이는 단순히 브랜치를 추가하는 것보다, 경계 정보를 강제하는 손실 함수를 통해 인코더를 정규화하는 것이 핵심임을 시사한다.
- **정성적 결과:** U-Net은 위양성(False Positive)이 많고 경계가 부정확한 반면, 제안 모델은 세밀한 경계선을 생성하며 종양의 작은 구조적 디테일을 효과적으로 포착하였다.

## 🧠 Insights & Discussion

본 논문의 가장 큰 강점은 경계 인식이라는 인간 전문가의 직관을 네트워크 구조와 손실 함수에 성공적으로 통합했다는 점이다. 특히 Boundary Stream이 단순히 경계를 예측하는 도구가 아니라, 메인 스트림의 하위 레이어(인코더)가 경계 특징에 집중하도록 만드는 정규화 장치로 작동한다는 점이 인상적이다.

다만, 실험이 3D 데이터를 2D 슬라이스로 변환하여 진행되었다는 점은 한계로 볼 수 있다. 의료 영상의 특성상 3D 공간 정보가 중요하므로, 향후 3D Boundary Aware Network로 확장했을 때 어느 정도의 성능 향상이 있을지가 중요한 미해결 질문으로 남는다. 또한, 단일 모달리티(T1c)만을 사용했는데, BraTS 데이터셋이 제공하는 다른 모달리티와의 융합(Fusion) 시나리오에 대한 분석이 부재하다.

## 📌 TL;DR

이 논문은 CNN의 질감 편향성을 해결하기 위해 **별도의 Shape Stream(경계 브랜치)**과 **Edge-aware Loss**를 도입한 의료 영상 분할 네트워크를 제안한다. 경계 정보를 명시적으로 학습함으로써 모델이 구조적으로 정규화되어, 기존 U-Net 및 V-Net 대비 훨씬 정교한 경계 분할 성능을 보였다. 이 접근법은 경계 식별이 중요한 다양한 의료 영상 분석 작업에 효과적으로 적용될 가능성이 높다.
