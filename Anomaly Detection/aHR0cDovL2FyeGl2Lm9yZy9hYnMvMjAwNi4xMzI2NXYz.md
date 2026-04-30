# Anomaly Detection in Medical Imaging with Deep Perceptual Autoencoders

Nina Shvetsova, Bart Bakker, Irina Fedulova, Heinrich Schulz, and Dmitry V. Dylov (2021)

## 🧩 Problem to Solve

본 논문은 의료 영상 분야에서의 이상치 탐지(Anomaly Detection) 문제를 해결하고자 한다. 의료 영상에서의 이상치 탐지는 정상 데이터만을 학습하여 비정상적인 입력(예: 전이암, 폐 질환 등)을 식별하는 작업이다. 

의료 영상의 이상치 탐지가 특히 어려운 이유는 다음과 같다. 첫째, 의료 데이터의 이상 징후는 매우 미세하여 정상 영상과 매우 유사한 특성을 갖는다. 예를 들어, 흉부 X-ray의 희미한 병변이나 병리 슬라이드의 전이 세포는 숙련된 전문가조차 구분하기 어려울 정도로 정상 조직과 닮아 있다. 둘째, 기존의 딥러닝 기반 이상치 탐지 방법론들은 주로 자연 이미지(Natural Images) 벤치마크에 최적화되어 있어, 복잡하고 고해상도인 의료 영상의 미세한 변화를 포착하는 데 한계가 있다. 

따라서 본 연구의 목표는 고해상도의 복잡한 의료 영상에서도 효과적으로 작동하며, 실제 임상 환경의 제약 사항을 반영한 강력한 이상치 탐지 베이스라인 모델을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 Autoencoder의 구조를 단순화하되, 손실 함수와 학습 전략을 고도화하여 모델이 정상 데이터의 '콘텐츠(Content)' 정보를 효율적으로 학습하게 하는 것이다.

1.  **Deep Perceptual Autoencoder (DPA) 제안**: 픽셀 단위의 단순 비교(L1 loss)나 현실적인 이미지를 생성하게 강제하는 적대적 손실(Adversarial loss)을 배제하고, 오직 Perceptual Loss만을 사용하여 Autoencoder를 학습시킨다. 이를 통해 모델이 시각적으로 완벽한 이미지를 복원하는 대신, 정상 데이터의 핵심적인 특징(Perceptive information)을 유연하게 포착하도록 한다.
2.  **Progressive Growing 학습 기법 도입**: 고해상도 의료 영상의 세부 정보를 효과적으로 학습하기 위해, 저해상도 네트워크에서 시작하여 점진적으로 레이어를 추가하고 해상도를 높이는 Progressive Growing 방식을 적용한다. 특히, Autoencoder의 해상도 증가에 맞춰 Perceptual Loss의 특징 추출 깊이(Depth)를 동기화하여 점진적으로 정밀한 정보를 학습하게 한다.
3.  **약지도 학습(Weakly-supervised) 기반 하이퍼파라미터 최적화**: 완전히 비지도(Unsupervised) 방식인 기존 설정의 비현실성을 지적하며, 매우 적은 양의 이상치 데이터(전체 학습 데이터의 0.5% 미만)를 검증 세트로 사용하여 하이퍼파라미터를 최적화하는 실용적인 파이프라인을 제안한다.

## 📎 Related Works

이상치 탐지 방법론은 크게 두 가지 범주로 나뉜다.
- **분포 기반 방법(Distribution-based methods)**: 데이터의 확률 밀도를 모델링하거나 정상 샘플 주변에 경계를 생성하는 방식이다. 최근에는 Deep SVDD나 Deep IF와 같이 딥러닝 특징 추출기를 결합한 방식이 사용되지만, 학습 데이터와 타겟 데이터 간의 도메인 시프트(Domain shift)가 발생할 경우 성능이 급격히 저하되는 한계가 있다.
- **복원 기반 방법(Reconstruction-based methods)**: 정상 데이터만으로 학습된 모델이 이상치 데이터를 제대로 복원하지 못한다는 점을 이용한다. PCA, Autoencoder, GAN(AnoGAN 등) 기반 방식이 있으며, 복원 오차(Reconstruction Error)를 이상치 점수로 사용한다. 

본 논문은 특히 PIAD와 같은 최신 복원 기반 방법론과 차별점을 둔다. PIAD는 Perceptual metric을 사용하지만 GAN 구조를 채택하고 있어 학습이 불안정하고 자원 소모가 크다. 반면, 제안된 DPA는 단순한 Autoencoder 구조에 Perceptual Loss를 결합하여 학습의 안정성과 효율성을 높이면서도 유사한 혹은 더 뛰어난 성능을 달성한다.

## 🛠️ Methodology

### 전체 시스템 구조
본 모델은 입력 이미지 $x$를 받아 복원된 "콘텐츠 텐서" $\check{x}$를 생성하는 Autoencoder $g$와, 이미지의 깊은 특징을 추출하는 사전 학습된 특징 추출기(Feature Extractor) $f$로 구성된다.

### 손실 함수: Relative-Perceptual-L1 Loss
DPA는 픽셀 값의 직접적인 비교 대신, 사전 학습된 VGG19 네트워크의 특징 공간에서 거리(Distance)를 측정하는 Perceptual Loss를 사용한다. 구체적으로는 노이즈와 대비 변화에 강건한 Relative-Perceptual-L1 Loss를 채택하며, 수식은 다음과 같다.

$$L_{rec}(x, \check{x}) = \frac{\| \hat{f}(x) - \hat{f}(\check{x}) \|_1}{\| \hat{f}(x) \|_1}$$

여기서 $\hat{f}(x)$는 정규화된 특징 벡터를 의미하며, 다음과 같이 계산된다.
$$\hat{f}(x) = \frac{f(x) - \mu}{\sigma}$$
($\mu$와 $\sigma$는 대규모 데이터셋에서 미리 계산된 평균과 표준편차이다.)

이 방식의 핵심은 $\check{x}$가 반드시 시각적으로 그럴듯한 이미지가 될 필요가 없다는 점이다. $\check{x}$는 단지 $x$의 콘텐츠 정보를 담고 있는 텐서이면 충분하며, 이는 모델이 정상 데이터의 '정상성'을 판단하는 핵심 큐(Cue)를 더 유연하게 학습하게 한다.

### Progressive Growing 학습 절차
고해상도 이미지 학습의 불안정성을 줄이기 위해 다음과 같은 단계적 성장 전략을 사용한다.
1. **저해상도 시작**: $8 \times 8$ 픽셀의 매우 작은 해상도와 특징 추출기의 얕은 레이어(Coarse layer)에서 학습을 시작한다.
2. **해상도 및 깊이 확장**: 점진적으로 레이어를 추가하여 해상도를 높이며, 이에 맞춰 Perceptual Loss에 사용되는 특징 $f$의 깊이 또한 함께 증가시킨다.
3. **부드러운 전이(Smooth Transition)**: 새로운 해상도 레이어를 추가할 때 파라미터 $\alpha$를 $0$에서 $1$까지 선형적으로 증가시켜 이전 단계의 출력과 새 단계의 출력을 혼합한다. 손실 함수 역시 다음과 같이 가중 합으로 계산하여 급격한 변화를 방지한다.

$$L_{rec} = \alpha \cdot L_{rec}(f^2(x), f^2(\check{x})) + (1-\alpha) \cdot L_{rec}(f^1(\text{down}(x)), f^1(\text{down}(\check{x})))$$

### 하이퍼파라미터 최적화 (Weakly-supervised)
비지도 학습에서는 검증 세트가 없어 최적의 하이퍼파라미터(Bottleneck 크기, Conv 레이어 수 등)를 정하기 어렵다. 본 논문은 아주 적은 수(예: 20장)의 이상치 샘플을 사용하여 ROC AUC를 측정하고, 이를 통해 최적의 설정을 찾는 방식을 제안한다.

## 📊 Results

### 실험 설정
- **데이터셋**: 
    - **Camelyon16**: 림프절 전이암 탐지 (디지털 병리 영상)
    - **NIH ChestX-ray14**: 흉부 X-ray 질환 탐지
    - **CIFAR10 / SVHN**: 일반 자연 이미지 벤치마크 (비교용)
- **지표**: ROC AUC (Receiver Operating Characteristic Area Under the Curve)
- **비교 대상**: Deep GEO, Deep IF, PIAD, AnoGAN, GANomaly 등 SOTA 모델

### 주요 결과
1. **의료 영상 성능**: 제안 방법은 의료 데이터셋에서 SOTA 모델들을 압도하였다.
    - **Camelyon16**: **93.4%** (PIAD 90.6%, Deep IF 89.5% 대비 우수)
    - **NIH Subset**: **92.6%** (PIAD 88.0%, Deep IF 87.3% 대비 우수)
2. **자연 이미지 성능**: CIFAR10과 SVHN에서는 Deep GEO나 Deep IF와 유사하거나 약간 낮은 성능을 보였다. 이는 자연 이미지의 높은 다양성이 Autoencoder의 과잉 일반화(Overgeneralization)를 유발하여 이상치까지 정상으로 복원하기 때문으로 분석된다.
3. **학습 효율성**: DPA는 GAN 기반 모델(PIAD 등)보다 구현이 훨씬 간단하며, 학습 속도가 매우 빠르고 자원 소모가 적다 (Table 3 참조).

### Ablation Study 결과
- **Perceptual Loss의 중요성**: 단순 L1 Loss를 사용했을 때보다 Perceptual Loss를 사용했을 때 성능이 비약적으로 향상되었다.
- **추가 손실의 역효과**: 현실적인 이미지를 만들기 위해 Adversarial Loss나 L1 Loss를 추가했을 때, 오히려 성능이 저하되었다. 이는 모델이 시각적 복원에 집착할 때보다 콘텐츠 정보에 집중할 때 이상치 탐지 능력이 더 높아짐을 시사한다.
- **Progressive Growing 효과**: 이를 적용했을 때 의료 영상에서 약 1%의 추가 성능 향상이 있었다.

## 🧠 Insights & Discussion

본 연구는 의료 영상의 이상치 탐지에서 '완벽한 복원'보다 '의미 있는 콘텐츠의 포착'이 더 중요하다는 점을 입증하였다. 

**강점**: 
- 단순한 Autoencoder 구조임에도 불구하고 Perceptual Loss와 Progressive Growing을 결합하여 복잡한 의료 영상의 미세한 이상 징후를 매우 정확하게 잡아냈다.
- 실무적으로 적용 가능한 하이퍼파라미터 튜닝 가이드라인(약지도 학습)을 제시하여 결과의 재현성을 높였다.

**한계 및 비판적 해석**: 
- 자연 이미지 데이터셋에서의 성능 저하는 본 모델이 데이터의 다양성(Variability)이 낮은 도메인에 특화되어 있음을 보여준다. 즉, 의료 영상처럼 획득 방식이 표준화된 데이터에는 강력하지만, 일반적인 사진 데이터에는 취약할 수 있다.
- 하지만 저자들의 주장처럼, 실제 임상 환경에서는 범용적인 모델보다 특정 의료 장비나 모달리티(Modality)에 최적화된 민감한 모델이 더 선호되므로, 이는 실용적인 관점에서 큰 단점이 아니라고 볼 수 있다.

## 📌 TL;DR

본 논문은 의료 영상의 미세한 이상치를 탐지하기 위해 **Perceptual Loss 기반의 Autoencoder**와 **Progressive Growing 학습법**을 결합한 **Deep Perceptual Autoencoder (DPA)**를 제안한다. 특히 시각적 복원보다는 콘텐츠 정보의 일치에 집중함으로써, 복잡한 의료 영상에서 SOTA 모델들을 능가하는 성능(Camelyon16 93.4%, NIH 92.6%)을 달성하였다. 또한, 소량의 이상치 데이터를 활용한 하이퍼파라미터 튜닝 방식을 도입하여 실무적인 적용 가능성을 높였다. 이 연구는 향후 고해상도 의료 영상 분석의 새로운 베이스라인으로 활용될 가능성이 크다.