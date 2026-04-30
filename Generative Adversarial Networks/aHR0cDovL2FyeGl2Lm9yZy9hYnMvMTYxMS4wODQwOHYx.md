# Semantic Segmentation using Adversarial Networks

Pauline Luc, Camille Couprie, Soumith Chintala, Jakob Verbeek (2016)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 핵심 문제는 Semantic Segmentation 모델이 예측하는 결과물에서 나타나는 **고차원적 일관성(Higher-order consistency)의 부족**이다.

기존의 대부분의 CNN 기반 Semantic Segmentation 방식들은 각 픽셀의 클래스 라벨을 서로 독립적으로 예측하는 경향이 있다. 이러한 방식은 훈련 과정에서 픽셀 간의 공간적 관계를 명시적으로 고려하지 않기 때문에, 결과물에서 공간적 연속성이 떨어지거나 현실적이지 않은 형태의 분할 맵(Segmentation map)이 생성되는 문제가 발생한다.

이를 해결하기 위해 Conditional Random Fields (CRFs)와 같은 후처리 기법들이 사용되어 왔으나, 대부분의 CRF 모델은 픽셀 쌍(Pairwise) 간의 관계만을 다루는 한계가 있다. 슈퍼픽셀(Superpixel) 기반의 고차원 포텐셜(Higher-order potentials)을 사용하는 방법도 제안되었지만, 이는 특정 클래스의 고차원 포텐셜만 학습할 수 있다는 제약이 있다. 따라서 본 논문의 목표는 특정 형태에 국한되지 않고, 학습 가능한 방식으로 **범용적인 고차원 공간 일관성을 강제할 수 있는 훈련 프레임워크를 구축하는 것**이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 Generative Adversarial Networks (GAN)의 구조를 Semantic Segmentation에 도입하여, **Adversarial training(적대적 학습)을 통해 분할 맵의 구조적 정교함을 높이는 것**이다.

중심적인 설계 직관은 다음과 같다. Segmentation 네트워크가 생성한 결과물이 실제 Ground Truth(GT) 분할 맵과 구별되지 않도록 만드는 Discriminator(판별기)를 함께 학습시킨다면, 이 Discriminator가 일종의 '학습 가능한 고차원 손실 함수' 역할을 수행하게 된다는 점이다. Discriminator는 이미지 전체 또는 넓은 영역의 픽셀 배치를 확인하므로, 단일 픽셀 단위의 Cross-entropy loss로는 잡을 수 없는 형태적 부조화나 공간적 불일치를 감지하고 이를 Segmentation 네트워크에 피드백할 수 있다.

## 📎 Related Works

### 관련 연구 및 한계
1.  **Adversarial Learning (GAN):** Goodfellow 등이 제안한 GAN은 고정된 분포에서 샘플을 생성하여 실제 데이터 분포를 근사한다. 이후 Conditional GAN이나 이미지 인페인팅(Inpainting) 등에 적용되어 픽셀 단위의 회귀 손실(Regression loss)이 만드는 'Blurry'한 결과를 개선하는 데 사용되었다.
2.  **Semantic Segmentation:** FCN (Fully Convolutional Networks) 이후 Dilated Convolution, Skip connection 등이 도입되어 해상도 손실을 줄이고 수용 영역(Receptive field)을 넓히는 방향으로 발전했다.
3.  **CRFs 및 구조적 모델:** CNN의 단일 픽셀 예측(Unary potentials)과 CRF의 쌍별 포텐셜(Pairwise potentials)을 결합하여 세밀한 경계를 복원하려는 시도가 많았으나, 계산 복잡도가 높거나 고차원 제약 조건을 수동으로 설계해야 한다는 한계가 있었다.

### 기존 방식과의 차별점
본 논문의 접근 방식은 CRF와 달리 **추론(Inference) 시점에 추가적인 연산이나 복잡한 후처리가 필요 없다**. 고차원 제약 조건은 오직 훈련 단계의 Adversarial loss를 통해 Segmentation 네트워크의 가중치에 내재화되기 때문이다. 또한, 수동으로 고차원 포텐셜을 설계하는 대신 CNN 기반의 Discriminator가 데이터로부터 직접 일관성 기준을 학습한다는 점에서 유연성이 높다.

## 🛠️ Methodology

### 전체 파이프라인
본 시스템은 두 개의 네트워크, 즉 **Segmentation 네트워크($s$)**와 **Adversarial 네트워크($a$)**로 구성된다.
1.  Segmentation 네트워크는 입력 이미지 $x$를 받아 픽셀별 클래스 확률 맵 $s(x)$를 생성한다.
2.  Adversarial 네트워크는 생성된 분할 맵(또는 GT 맵)을 입력받아, 이것이 실제 GT인지 아니면 네트워크가 생성한 가짜(Synthetic)인지를 판별하는 이진 분류(0 또는 1)를 수행한다.

### 손실 함수 및 학습 절차
전체 목적 함수는 표준적인 Multi-class Cross-Entropy (MCE) 손실과 Adversarial 손실의 가중 합으로 정의된다.

$$ \ell(\theta_s, \theta_a) = \sum_{n=1}^N \ell_{mce}(s(x^n), y^n) - \lambda [ \ell_{bce}(a(x^n, y^n), 1) + \ell_{bce}(a(x^n, s(x^n)), 0) ] $$

여기서 $\ell_{mce}$는 픽셀 단위의 정확도를 높이는 표준 손실 함수이며, $\ell_{bce}$는 이진 교차 엔트로피(Binary Cross-Entropy) 손실이다.

*   **Adversarial 네트워크 학습:** $\theta_a$에 대해 위 식을 최소화하여 GT 맵($y^n$)은 1로, 생성된 맵($s(x^n)$)은 0으로 정확히 구분하도록 학습한다.
*   **Segmentation 네트워크 학습:** $\theta_s$에 대해 $\ell_{mce}$를 최소화함과 동시에, Discriminator가 생성된 맵을 GT라고 믿게끔(즉, $\ell_{bce}(a(x^n, s(x^n)), 1)$을 최소화) 유도한다.

### 네트워크 아키텍처 및 입력 방식
실험에서는 특히 PASCAL VOC 2012 데이터셋을 위해 세 가지 Adversarial 입력 방식을 제안하였다.
1.  **Basic:** Segmentation 네트워크가 출력한 확률 맵을 그대로 입력으로 사용한다.
2.  **Product:** 입력 RGB 이미지와 각 클래스 확률 맵을 원소별로 곱하여 $3C$ 채널의 입력을 생성한다. 이는 Discriminator가 라벨과 이미지 콘텐츠 간의 관계를 함께 볼 수 있게 한다.
3.  **Scaling:** GT 맵을 1-hot 인코딩하는 대신, Segmentation 네트워크의 출력과 유사하면서도 정답 라벨에 최소 $\tau$ 만큼의 질량을 가진 분포로 변환하여 입력한다. 이는 Discriminator가 단순히 '0과 1로만 구성되었는가'를 보고 GT를 구별하는 편법(Trivial solution)을 막기 위함이다.

또한, 수용 영역의 크기에 따라 **LargeFOV**($34 \times 34$ 픽셀)와 **SmallFOV**($18 \times 18$ 픽셀) 두 가지 설정을 통해 광역적 패턴과 국소적 세부 사항 중 무엇이 더 효과적인지 분석하였다.

## 📊 Results

### 실험 설정
- **데이터셋:** Stanford Background, PASCAL VOC 2012
- **기준 모델:** Farabet et al.의 Multi-scale network (Stanford), Yu et al.의 Dilated-8 (PASCAL)
- **평가 지표:** Per-class Accuracy, Per-pixel Accuracy, mean IoU (mIoU), BF measure (경계선 정확도 측정)

### 주요 결과
1.  **Stanford Background 데이터셋:**
    - Adversarial training을 적용했을 때 mIoU가 $51.3\% \rightarrow 54.3\%$로 향상되었으며, 픽셀 정확도 또한 증가하였다.
    - 정성적으로는 클래스 확률 맵이 더 매끄러워지고(smooth), 불필요한 작은 영역의 오분류(spurious labels)가 제거되는 효과가 확인되었다.
    - 특히 훈련 데이터에 대한 과적합(Overfitting)이 줄어드는 정규화(Regularization) 효과가 뚜렷하게 나타났다.

2.  **PASCAL VOC 2012 데이터셋:**
    - Baseline(Dilated-8)의 mIoU $71.8\%$ 대비, Adversarial training 적용 시 $72.0\% \sim 72.0\%$ 수준으로 소폭이지만 일관된 성능 향상을 보였다.
    - 특히 **LargeFOV** 구조가 전반적으로 가장 효과적이었으며, 이는 광역적인 클래스 배치 패턴을 학습하는 것이 segmentation 품질 향상에 유리함을 시사한다.
    - 경계선 정확도를 측정하는 BF measure에서도 향상이 관찰되어, 객체의 외곽선이 더 정교하게 예측됨을 확인하였다.

## 🧠 Insights & Discussion

본 논문은 Adversarial network를 단순한 생성 도구가 아니라, **매개변수화된 가변적 손실 함수(Variational loss)**로 활용하여 모델을 정규화했다는 점에서 큰 의의가 있다.

### 강점 및 해석
- **효율적인 추론:** 학습 시에는 복잡한 적대적 구조를 사용하지만, 추론 시에는 일반적인 CNN과 동일한 속도로 작동한다.
- **데이터셋 규모에 따른 효과 차이:** Stanford Background와 같은 작은 데이터셋에서 더 큰 성능 향상이 나타났는데, 이는 Adversarial loss가 강력한 정규화 도구로 작용하여 과적합을 방지했기 때문으로 해석된다. 반면 PASCAL VOC는 이미 강력한 아키텍처(Dilated-8)를 사용하고 데이터 양이 많아 향상 폭이 상대적으로 작았다.

### 한계 및 논의사항
- **학습 안정성:** GAN 계열의 특성상 학습이 불안정할 수 있으며, 본 논문에서는 이를 해결하기 위해 업데이트를 500회마다 교체하는 'Slow alternating scheme'을 사용하였다.
- **입력 방식의 영향:** Basic, Product, Scaling 방식 간의 성능 차이가 크지 않았는데, 이는 Discriminator가 입력 형태와 무관하게 어느 정도 수준의 구조적 특징을 잡아낼 수 있음을 의미하지만, 동시에 최적의 입력 표현에 대한 더 깊은 연구가 필요함을 시사한다.

## 📌 TL;DR

본 연구는 Semantic Segmentation 모델의 고질적인 문제인 '픽셀 간 독립적 예측으로 인한 구조적 불일치'를 해결하기 위해 **Adversarial Training**을 도입하였다. Discriminator가 GT 맵과 예측 맵을 구분하도록 학습시킴으로써, Segmentation 네트워크가 고차원의 공간적 일관성을 학습하게 유도하였다. 그 결과, 추가적인 추론 비용 없이도 **mIoU 향상, 경계선 정교화, 과적합 방지**라는 성과를 거두었으며, 특히 수용 영역이 넓은(LargeFOV) 판별기가 더 효과적임을 입증하였다. 이 연구는 적대적 학습이 단순한 이미지 생성을 넘어 정교한 구조적 정규화 도구로 활용될 수 있음을 보여준다.