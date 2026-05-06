# Learning Temporal Distribution and Spatial Correlation Towards Universal Moving Object Segmentation

Guanfang Dong, Chenqiu Zhao, Xichen Pan and Anup Basu (2024)

## 🧩 Problem to Solve

본 논문이 해결하고자 하는 문제는 비디오 내에서 정지된 배경으로부터 움직이는 객체를 분리하는 Moving Object Segmentation (MOS)의 **보편성(Universality)** 확보이다.

기존의 딥러닝 기반 방법론들은 특정 데이터셋이나 장면(Scene)에서는 매우 높은 성능을 보이지만, 학습되지 않은 새로운 장면(Unseen scenes)에 적용할 경우 성능이 급격히 저하되는 경향이 있다. 이를 해결하기 위해 데이터 증강(Data augmentation)이나 네트워크 재학습(Retraining)과 같은 튜닝 과정이 필수적이었으나, 이는 실제 환경에서 막대한 계산 비용과 정답 데이터(Ground-truth) 확보의 어려움을 야기한다. 반면, 전통적인 비-딥러닝 방법론들은 보편성은 어느 정도 갖추었으나 복잡한 배경이나 조명 변화 등의 까다로운 시나리오에서 정확도가 떨어진다는 한계가 있다.

따라서 본 연구의 목표는 추가적인 튜닝이나 재학습 없이, 다양한 자연 장면의 비디오에 직접 적용 가능한 **보편적인(Universal) MOS 모델**을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **장면의 외형적 정보(Scene information)보다는 픽셀들의 시간적 분포(Temporal distribution)가 서로 다른 비디오 간에도 유사성이 높다**는 직관에서 출발한다. 이를 바탕으로 다음과 같은 세 가지 핵심 기여를 제시한다.

1. **Defect Iterative Distribution Learning (DIDL) 네트워크 제안**: 시간적 픽셀 분포를 학습하여 장면 독립적인(Scene-independent) 세그멘테이션을 수행한다. 특히, 전체 데이터셋의 일부만을 사용하여 효율적으로 학습하는 결함 반복 학습 전략을 도입하였다.
2. **개선된 Product Distribution Layer 설계**: 기존 분포 학습 네트워크에서 발생하던 '0으로 나누기(Zero division error)' 문제를 수학적으로 분석하고 이를 해결하는 새로운 역전파(Back-propagation) 방정식을 도출하여 수치적 안정성과 정확도를 높였다.
3. **Stochastic Bayesian Refinement (SBR) 네트워크 제안**: DIDL이 간과하는 공간적 상관관계(Spatial correlation)를 학습하여 결과 마스크의 노이즈를 제거한다. 이때, 특정 장면에 과적합(Overfitting)되는 것을 방지하기 위해 다중 스케일의 무작위 샘플링(Multiscale random sampling) 기법을 적용하여 보편성을 유지한다.

## 📎 Related Works

### 1. 비-딥러닝 방법론 (Non-Deep-Learning Methods)

통계적 방법(Gaussian Mixture Model 등)과 샘플 기반 방법(RPCA 등)으로 나뉜다. 이들은 정해진 규칙이나 통계 모델을 사용하므로 어느 정도의 보편성을 가지지만, 복잡한 환경에서 성능이 낮고 수동 튜닝 비용이 높다는 단점이 있다.

### 2. 딥러닝 방법론 (Deep Learning Methods)

CNN, GAN, RNN, GNN 기반의 다양한 구조가 제안되었다. FgSegNet과 같은 모델은 표준 데이터셋에서 거의 완벽한 결과를 보이지만, 장면 정보에 크게 의존하기 때문에 학습되지 않은 비디오에서는 성능이 매우 낮아진다. BSUV나 AE-NE 같은 모델들이 보편성을 높이려 시도했으나, 여전히 복잡한 데이터 증강이나 사전 학습 과정이 필요하다는 한계가 있다.

### 3. 차별점

본 연구는 ADNN(Arithmetic Distribution Neural Network)에서 영감을 받았으나, (1) 공간적 상관관계를 학습하는 SBR 네트워크를 추가하고, (2) 효율적인 DIDL 학습 전략을 도입했으며, (3) Product Distribution Layer의 수치적 오류를 해결함으로써 기존 ADNN 대비 정확도를 15% 이상 향상시켰다.

## 🛠️ Methodology

### 1. Defect Iterative Distribution Learning (DIDL)

DIDL은 장면의 외형이 달라도 시간적 픽셀 분포는 유사하다는 가정하에, 픽셀의 시간적 분포를 학습한다.

- **학습 전략**: 전체 학습 셋 $H$를 모두 학습하는 대신, 대표성 있는 서브셋 $H_i$를 통해 파라미터 $\theta$를 근사한다.
  $$\theta = \text{argmax}_{\hat{\theta}} L(\hat{\theta}, H) \simeq E_{H_i \sim H}(\text{argmax}_{\hat{\theta}_i} L(\hat{\theta}_i, H_i))$$
- **결함 반복 학습 (Defect Iterative Process)**:
  1. 초기 서브셋 $H_1$으로 학습한다.
  2. 학습된 모델로 전체 셋 $H$를 검증하여 잘못 분류된 '결함 샘플($H_d$)'을 추출한다.
  3. $H_2 = H_1 \cup H_d$와 같이 결함 샘플을 추가하여 다시 학습함으로써 효율적으로 전체 분포를 학습한다.
- **입력 데이터**: 현재 픽셀과 과거 픽셀 간의 차이에 대한 히스토그램을 입력으로 사용한다.
- **구조**: Product Distribution Layer $\rightarrow$ Sum Distribution Layer $\rightarrow$ Convolution $\rightarrow$ Fully Connected Layer 순으로 구성되며, 최종적으로 배경(Black), 전경(White), 기타(Gray)로 분류한다.

### 2. Improved Product Distribution Layer

분포 학습의 핵심인 Product Distribution Layer에서 $w \to 0$일 때 발생하는 수치적 불안정성을 해결하였다.

- **문제점**: 전방향 계산 시 $f_Z(z) = \int_{-\infty}^{\infty} f_W(w) f_X(\frac{z}{w}) \frac{1}{|w|} dw$ 식에서 $w$가 0에 가까워지면 $\frac{1}{|w|}$이 무한대로 발산하여 Zero division error가 발생한다.
- **해결책**: $z \to 0$일 때의 세 가지 케이스를 분석하여 $f_Z(0)$ 값을 명시적으로 정의하였다.
  - $w=0, x \neq 0$ 또는 $w \neq 0, x=0$인 경우: $f_Z(0) = f_W(0) E(\frac{1}{|x|})$
  - $w=0, x=0$인 경우: $f_Z(0) = \infty$
이러한 수학적 정의를 통해 역전파 시 발생하던 오류를 방지하고 정확도를 높였다.

### 3. Stochastic Bayesian Refinement (SBR) Network

DIDL의 결과물은 공간적 정보가 없어 노이즈가 많다. SBR은 베이지안 추론을 통해 이를 정교화한다.

- **핵심 원리**: 픽셀 $(x, y)$의 라벨을 주변 픽셀 $G$의 정보와 현재 픽셀 값 $I(x, y)$를 이용하여 결정한다.
- **구조 및 절차**:
  - **Simplified U-Net**: 인코더-디코더 구조의 가벼운 U-Net을 Refine Block으로 사용한다.
  - **Stochastic Sampling**: 특정 장면에 과적합되는 것을 막기 위해, 전체 이미지에서 $16 \times 16, 32 \times 32, 64 \times 64$ 크기의 패치를 무작위로 샘플링하여 학습 및 추론을 수행한다.
  - **Heatmap 생성**: 샘플링된 패치들을 다시 원래 위치에 쌓고 정규화하여 히트맵을 생성한 뒤, 0.5 임계값을 적용해 최종 전경 마스크를 도출한다.

## 📊 Results

### 1. 실험 설정

- **데이터셋**: LASIESTA, CDNet2014, BMC, SBMI2015 및 실제 환경에서 촬영된 128개의 비디오.
- **평가 지표**: F-measure (Fm).
- **비교 대상**: 전통적인 방법론(GMM, SuBSENSE 등) 및 최신 딥러닝 방법론(FgSegNet, AE-NE, ADNN 등).

### 2. 주요 결과

- **보편성 검증 (Unseen Video)**: CDNet2014에서 학습하고 LASIESTA에서 테스트한 결과($LTS\text{-}U$), FgSegNet(0.37)과 같은 기존 모델보다 월등히 높은 성능(0.8648)을 보였다. 이는 고정된 파라미터만으로도 새로운 장면에 잘 적응함을 의미한다.
- **정량적 성능**: CDNet2014 데이터셋에서 $LTS\text{-}A$는 0.9431, $LTS\text{-}D$는 0.9484의 Fm 점수를 기록하며, 고정 파라미터를 사용하는 방법론 중 최상위권의 성능을 달성하였다.
- **효율성**: 모델 크기가 매우 작으며(약 1.37MB), RTX 3090 기준 프레임당 0.15초의 빠른 추론 속도를 보였다.

### 3. 절제 실험 (Ablation Study)

SBR 네트워크에서 **다중 스케일 무작위 샘플링**의 중요성이 확인되었다. 샘플링 없이 학습했을 때보다 무작위 샘플링을 적용했을 때 Fm 점수가 0.7372에서 0.9431로 급격히 상승하였으며, 이는 공간적 정보 학습 시 과적합을 방지하는 것이 필수적임을 입증한다.

## 🧠 Insights & Discussion

### 강점

본 논문은 딥러닝 모델의 고질적인 문제인 '장면 의존성'을 '시간적 분포 학습'이라는 관점으로 해결하였다. 특히 SBR 네트워크를 통해 정확도를 높이면서도, 무작위 샘플링 기법을 통해 보편성을 잃지 않도록 설계한 점이 탁월하다. 또한, 단순한 모델 크기로도 방대한 데이터를 효율적으로 학습할 수 있는 DIDL 전략을 제시하여 실용성을 확보하였다.

### 한계 및 논의사항

1. **데이터 의존성**: BMC 데이터셋에서는 성능 향상이 미미했는데, 이는 제공된 프레임 수가 너무 적어 고품질의 히스토그램을 생성하지 못했기 때문이다. 이는 본 방법론이 일정 수준 이상의 시간적 샘플(프레임)이 확보되어야 함을 시사한다.
2. **계산 비용**: 이론적으로는 모든 위치와 크기에 대해 샘플링해야 하지만, 실제 구현에서는 3가지 고정 크기($16, 32, 64$)만을 사용하였다. 이는 효율성을 위한 타협안이며, 향후 더 최적화된 샘플링 전략에 대한 연구가 필요하다.
3. **해상도 문제**: 고해상도 이미지 처리 시 연산 시간이 증가하는 문제가 언급되었으며, 이에 대한 최적화 방안이 필요할 것으로 보인다.

## 📌 TL;DR

본 논문은 다양한 환경에서 추가 튜닝 없이 즉시 사용 가능한 **보편적 움직이는 객체 분할(Universal MOS)** 모델인 **LTS**를 제안한다. 시간적 분포를 효율적으로 학습하는 **DIDL 네트워크**와 수치적 안정성을 높인 **Product Distribution Layer**, 그리고 공간적 상관관계를 학습하여 노이즈를 제거하는 **SBR 네트워크**를 결합하였다. 실험 결과, 기존의 딥러닝 모델보다 훨씬 뛰어난 보편성과 정확도를 보였으며, 매우 가벼운 모델 크기로 실시간 응용 가능성을 입증하였다. 이 연구는 향후 실시간 보안 관제 및 교통 분석 시스템 등 다양한 실제 환경에 직접 적용될 가능성이 매우 높다.
