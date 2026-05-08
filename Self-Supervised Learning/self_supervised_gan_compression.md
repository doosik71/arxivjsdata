# Self-Supervised GAN Compression

Chong Yuan and Jeff Pool (2020)

## 🧩 Problem to Solve

딥러닝 모델의 크기가 거대해짐에 따라 연산 및 메모리 집약적인 특성이 강해졌으며, 이는 낮은 지연 시간(latency)과 저장 공간 요구량을 필요로 하는 실제 배포 환경에서 큰 도전 과제가 된다. 특히 Generative Adversarial Networks (GANs)는 복잡한 작업을 수행하며 수백만 개의 파라미터를 포함하고 있어 압축의 필요성이 매우 높다.

기존의 모델 압축 기술인 Weight Pruning은 이미지 분류나 탐지 모델에서는 성공적으로 적용되었으나, 복잡한 작업을 수행하는 GAN의 Generator에 적용했을 때는 성능 저하가 심각하게 나타난다. 본 논문은 표준적인 Pruning 기법들이 GAN의 생성 성능을 유지하지 못하며, 특히 복잡한 Image-to-Image Translation 작업에서 기존 방식들이 한계를 보인다는 점을 지적한다. 따라서 본 연구의 목표는 복잡한 GAN 모델에서도 품질 저하 없이 높은 압축률을 달성할 수 있는 새로운 압축 프레임워크를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 이미 학습이 완료된 **Discriminator를 '자기 지도(Self-Supervised)' 감독자로 활용**하여 압축된 Generator의 학습을 가이드하는 것이다. 

기존의 Pruning 방식이 단순한 Fine-tuning이나 새로운 Discriminator를 학습시키는 방식에 의존했다면, 제안된 방법은 이미 대상 데이터셋에 대해 최적화된 강력한 Discriminator를 그대로 사용하여, 압축된 Generator가 생성한 결과물이 원본 Generator의 결과물 및 실제 데이터와 구별되지 않도록 강제한다. 이를 통해 인간의 주관적인 평가를 정량적인 손실 함수로 대체하고, GAN 학습의 고질적인 문제인 Mode Collapse를 방지하며 효율적인 압축을 가능하게 한다.

## 📎 Related Works

논문에서는 다음과 같은 기존 압축 연구들을 언급하며 그 한계를 분석한다.

1.  **Network Pruning & Quantization**: 가중치 중 작은 값을 제거하는 Pruning(Han et al., 2015)이나 가중치를 양자화하는 Quantization 등이 있으며, 이는 주로 분류 모델에서 효과적이었다.
2.  **GAN Compression**: 단순한 이미지 합성(예: MNIST)을 위한 GAN 압축 연구는 존재했으나, 복잡한 Image-to-Image Translation 작업에 적용했을 때는 성능이 급격히 떨어진다.
3.  **Knowledge Distillation**: Teacher 모델의 지식을 Student 모델로 전이하는 방식이 제안되었으나, GAN의 경우 Teacher Discriminator를 적절히 활용하지 않거나 새로운 Discriminator를 처음부터 학습시켜야 하는 문제로 인해 복잡한 작업에서 실패하는 경우가 많다.

본 논문은 기존 방식들이 복잡한 작업에서 실패하는 이유가 GAN의 학습 불안정성, 출력값의 높은 엔트로피(Entropy), 그리고 적절한 평가 지표의 부재 때문이라고 분석하며, 이를 해결하기 위해 Pre-trained Discriminator를 고정하여 사용하는 차별점을 둔다.

## 🛠️ Methodology

### 전체 파이프라인
본 연구의 핵심은 원본 Generator($G^O$)와 압축된 Generator($G^C$) 사이의 출력 차이를 줄이는 동시에, 원본 Discriminator($D^O$)가 두 생성자의 결과물을 동일하게 판단하도록 만드는 것이다.

### 주요 구성 요소 및 학습 절차
1.  **입력 및 출력**: 입력 이미지 $x$에 대해 원본 생성자는 $\hat{y}_o = G^O(x)$를, 압축 생성자는 $\hat{y}_c = G^C(x)$를 생성한다.
2.  **Self-Supervisor**: 학습이 완료된 원본 Discriminator $D^O$를 고정(Freeze)하여 사용한다.
3.  **손실 함수 구성**:
    *   **Generative Consistent Loss ($L^{GC}$)**: 원본 생성자와 압축 생성자가 내뱉는 생성 손실(Generative Loss) 값들 사이의 차이를 측정한다.
    *   **Discriminative Consistent Loss ($L^{DC}$)**: 원본 Discriminator가 원본 생성물의 결과물($\hat{y}_o$)과 압축 생성물의 결과물($\hat{y}_c$)을 보았을 때 느끼는 판별 손실(Discriminative Loss)의 차이를 측정한다.

### 주요 방정식 설명
본 논문은 가중치가 적용된 정규화된 유클리드 거리(Weighted Normalized Euclidean Distance)를 사용하여 손실을 정의한다.

**1. Generative Consistent Loss ($L^{GC}$)**
$$L^{GC}(l\text{-}G^O, l\text{-}G^C) = \frac{|l\text{-}\text{Gen}^O - l\text{-}\text{Gen}^C|}{|l\text{-}\text{Gen}^O|} + \alpha \frac{|l\text{-}\text{Cla}^O - l\text{-}\text{Cla}^C|}{|l\text{-}\text{Cla}^O|} + \beta \frac{|l\text{-}\text{Rec}^O - l\text{-}\text{Rec}^C|}{|l\text{-}\text{Rec}^O|}$$
여기서 $l\text{-}\text{Gen}$은 생성 손실, $l\text{-}\text{Cla}$는 분류 손실, $l\text{-}\text{Rec}$은 재구성 손실을 의미하며, $\alpha, \beta$는 가중치이다.

**2. Discriminative Consistent Loss ($L^{DC}$)**
$$L^{DC}(l\text{-}D^O, l\text{-}D^C) = \frac{|l\text{-}\text{Dis}^O - l\text{-}\text{Dis}^C|}{|l\text{-}\text{Dis}^O|} + \delta \frac{|l\text{-}\text{GP}^O - l\text{-}\text{GP}^C|}{|l\text{-}\text{GP}^O|}$$
여기서 $l\text{-}\text{Dis}$는 판별 손실, $l\text{-}\text{GP}$는 Gradient Penalty 손실을 의미하며, $\delta$는 가중치이다.

**3. 최종 전체 손실 함수**
$$L^{\text{Overall}} = L^{GC}(l\text{-}G^O, l\text{-}G^C) + \lambda L^{DC}(l\text{-}D^O, l\text{-}D^C)$$
$\lambda$는 두 손실 간의 비중을 조절하는 하이퍼파라미터이다.

## 📊 Results

### 실험 설정
- **데이터셋 및 모델**: StarGAN (CelebA), DCGAN (MNIST), Pix2Pix (Sat$\leftrightarrow$Map), CycleGAN (Monet$\leftrightarrow$Photo, Zebra$\leftrightarrow$Horse), SRGAN (DIV2K).
- **압축 설정**: Generator의 모든 Convolution 및 Deconvolution 레이어에 대해 50% Sparsity를 목표로 Pruning을 수행하였다.
- **지표**: FID (Frechet Inception Distance), PSNR, SSIM을 사용하여 정량적으로 평가하였다.

### 주요 결과
1.  **기존 기법 대비 성능**: StarGAN 실험 결과, 표준 Pruning 및 Fine-tuning 방식은 얼굴 형태가 뭉개지거나(Facial artifacts) Mode Collapse가 발생하여 FID 점수가 매우 높게(나쁘게) 나타났다. 반면, 제안된 Self-Supervised 방식은 Dense baseline(6.113)과 매우 유사한 FID(6.929)를 기록하며 시각적으로도 거의 구별 불가능한 품질을 보여주었다.
2.  **범용성 확인**: DCGAN, Pix2Pix, CycleGAN, SRGAN 등 다양한 아키텍처와 작업에 적용했을 때 모두 50% 압축률에서도 성능 하락이 매우 적음을 확인하였다.
3.  **Pruning 입도(Granularity) 분석**:
    - **Fine-grained (0D) Pruning**: 90%의 매우 높은 Sparsity에서도 세부 디테일이 일부 사라질 뿐, 전반적인 품질이 유지되었다.
    - **Filter (3D) Pruning**: 단 25%의 Sparsity만으로도 심각한 색상 왜곡(Color shift)과 품질 저하가 발생하였다. 이는 분류 모델에서는 효과적이었던 필터 제거 방식이 GAN에서는 매우 위험할 수 있음을 시사한다.

## 🧠 Insights & Discussion

### 강점 및 분석
본 논문은 GAN 모델 압축이 왜 어려운지에 대해 세 가지 통찰을 제시한다. 첫째, GAN은 명확한 수치적 지표(Softmax 확률 등)가 부족하여 주관적 평가에 의존해야 한다는 점, 둘째, Generator와 Discriminator 사이의 정교한 균형이 깨지기 쉽다는 점, 셋째, 출력물의 엔트로피가 높아 정보 손실에 민감하다는 점이다. 제안된 방법은 Pre-trained Discriminator를 고정된 '심판'으로 사용하여 이 세 가지 문제를 동시에 해결하였다.

### 한계 및 비판적 해석
- **하드웨어 가속의 실효성**: 논문에서 성능이 가장 좋았던 Fine-grained Pruning은 비정형 희소 행렬(Unstructured Sparse Matrix)을 생성하므로, 특수한 하드웨어 가속기가 없다면 실제 추론 속도 향상으로 이어지기 어렵다. 
- **필터 Pruning의 실패**: 필터 단위 Pruning이 GAN에서 특히 취약하다는 점을 발견한 것은 의미 있으나, 이를 해결하기 위한 구조적 압축 방안까지는 제시하지 못하였다.
- **학습 시간**: 원본 학습 시간의 1~10%만 소요된다고 명시하였으나, 이는 이미 학습된 모델들이 있다는 가정 하의 수치이므로 전체 파이프라인 관점에서의 비용 분석이 더 필요하다.

## 📌 TL;DR

이 논문은 복잡한 작업을 수행하는 GAN의 Generator를 압축하기 위해, **학습된 Discriminator를 고정된 감독자로 활용하는 Self-Supervised Compression 기법**을 제안한다. 기존의 Pruning 방식들이 GAN에서 Mode Collapse나 품질 저하를 일으키는 문제를 해결하였으며, 다양한 GAN 아키텍처에서 50% 이상의 높은 압축률에서도 원본에 가까운 품질을 유지함을 입증하였다. 특히, GAN 압축에서는 필터 단위의 거친 Pruning보다 세밀한 가중치 단위의 Pruning이 훨씬 효과적이라는 점을 실험적으로 보여주었다.