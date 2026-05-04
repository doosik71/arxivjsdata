# Modeling the Probabilistic Distribution of Unlabeled Data for One-shot Medical Image Segmentation

Yuhang Ding, Xin Yu, Yi Yang (2021)

## 🧩 Problem to Solve

의료 영상 분할(Medical Image Segmentation)은 전문적인 지식을 가진 전문가가 3D 이미지의 해부학적 영역을 직접 라벨링해야 하므로 매우 높은 비용과 시간이 소요된다. 이로 인해 딥러닝 모델 학습에 필요한 대규모 라벨링 데이터셋을 구축하는 것이 큰 병목 현상이 된다.

이 문제를 해결하기 위해 단 하나의 라벨링된 이미지(Atlas)와 소수의 라벨링되지 않은(Unlabeled) 이미지만을 사용하는 **One-shot Medical Image Segmentation** 방식이 제안되어 왔다. 하지만 기존의 데이터 증강(Data Augmentation) 방식들은 다음과 같은 한계가 있다:
1. **Hand-crafted 증강**: 무작위 탄성 변형(Random elastic deformations) 등을 사용하지만, 실제 의료 영상의 분포를 고려하지 않아 비현실적인 이미지를 생성하며, 이는 모델의 일반화 성능을 저하시킨다.
2. **결정론적 등록(Deterministic Registration)**: 이미지 등록(Registration)을 통해 Atlas와 타겟 이미지 간의 변형을 학습하여 데이터를 생성하지만, 학습된 변형이 제한적이고 결정론적이어서 생성된 데이터의 다양성이 부족하다.

본 논문의 목표는 실제 뇌 MRI 영상의 형태(Shape)와 강도(Intensity) 분포를 따르는 다양하고 현실적인 학습 데이터를 생성하여, 단 하나의 Atlas만으로도 높은 성능을 내는 분할 네트워크를 학습시키는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 **두 개의 3D Variational Autoencoders (VAEs)를 사용하여 Atlas와 Unlabeled 이미지들 사이의 변형 분포를 확률적으로 모델링**하는 것이다.

- **확률적 데이터 증강**: 단순히 기존의 변형을 재조합하는 것이 아니라, 형태 변형(Shape deformation)과 강도 오프셋(Intensity offset)의 확률 분포를 학습함으로써, 기존 데이터셋에 존재하지 않는 새로운 변형을 생성할 수 있다.
- **$\beta$-VAE 도입**: KL Divergence의 영향을 조절하는 $\beta$ 하이퍼파라미터를 통해 잠재 공간(Latent space)을 확장함으로써, 데이터가 부족한 상황에서도 더 다양하고 풍부한 변형을 생성하도록 유도한다.
- **새로운 벤치마크 제안**: 다양한 소스(Imaging sources)에서 수집된 MRI 데이터셋인 ABIDE 벤치마크를 제안하여, 학습 시 보지 못한(Unseen) 데이터에 대한 모델의 일반화 성능을 평가한다.

## 📎 Related Works

### Atlas-based Segmentation
- **Single Atlas-based**: 하나의 Atlas를 타겟 이미지에 정렬(Registration)하여 라벨을 전이하는 방식이다. 하지만 정렬 오차(Misalignment errors)가 발생할 경우 분할 결과가 부정확해지는 문제가 있다.
- **Data Augmentation 기반**: 정렬을 통해 얻은 변형 필드를 이용해 Atlas를 변형시켜 가상 학습 데이터를 생성하고, 이를 통해 분할 네트워크를 학습시켜 정렬 오차를 완화한다. 그러나 변형이 결정론적이어서 다양성이 부족하다.

### Medical Image Data Augmentation
- **전통적 방식**: 가우시안 노이즈, 블러링, 밝기 조절, 탄성 변형 등을 사용한다.
- **GAN 기반 방식**: CycleGAN이나 Conditional GAN을 통해 현실적인 영상을 합성한다. 하지만 라벨링된 데이터가 매우 적은 One-shot 상황에서는 모드 붕괴(Mode collapse) 현상이 발생하여 모든 출력이 동일해지는 문제가 있다.

본 논문은 GAN 대신 모드 붕괴 문제에서 자유로운 **VAE**를 채택하여 이 문제를 해결하며, 단순한 이미지 합성이 아닌 '변형(Deformation)'의 분포를 학습한다는 점에서 차별점을 가진다.

## 🛠️ Methodology

전체 파이프라인은 크게 세 단계로 구성된다: (1) 이미지 등록을 통한 변형 추출 $\rightarrow$ (2) VAE를 이용한 변형 분포 학습 $\rightarrow$ (3) 가상 데이터 생성 및 분할 네트워크 학습.

### 1. Learning Deformations from Image Registration
먼저 Atlas 이미지 $x_a$와 Unlabeled 이미지 $x_{u,i}$ 사이의 형태 및 강도 차이를 학습한다.

- **Shape Registration Network**: U-Net 기반 네트워크가 $x_a$를 $x_{u,i}$로 변형시키는 shape deformation $S_i$를 추정한다.
  - 손실 함수 $\mathcal{L}_{srn}$은 국소 상호 상관관계(Local Cross-Correlation, $L_{CC}$)와 변형의 매끄러움(Smoothness, $L_{reg}^S$)의 합으로 정의된다.
  $$\mathcal{L}_{srn} = -L_{CC} + L_{reg}^S$$
- **Intensity Alignment Network**: 형태가 정렬된 상태에서 Atlas와 타겟 이미지 간의 강도 차이인 intensity deformation $I_i$를 학습한다.
  - 손실 함수 $\mathcal{L}_{irn}$은 픽셀 단위 재구성 손실($L_{sim}$)과 강도 변화의 매끄러움($L_{reg}^I$)을 사용한다.
  $$\mathcal{L}_{irn} = L_{sim} + \lambda L_{reg}^I$$

### 2. Diverse Image Generation via VAEs
추출된 $N$개의 $\{S_i, I_i\}$ 쌍을 사용하여 각각의 분포를 학습하는 두 개의 VAE를 구축한다.

- **Shape VAE**: 형태 변형 $S_i$의 분포를 학습한다.
  - **손실 함수**: $\mathcal{L}_{S} = (L_{d}^{S} + L_{i}^{S}) + \beta L_{kl}^{S}$
    - $L_{kl}^S$: 잠재 변수 $z$가 표준 정규 분포를 따르도록 강제한다.
    - $L_{d}^S$: 변형 필드 자체의 픽셀 단위 재구성 손실이다.
    - $L_{i}^S$: 변형된 이미지의 강도 차이를 측정하는 손실으로, 이미지 윤곽선(Contour)의 일관성을 유지하기 위해 도입되었다.
    - $\beta$: $\beta$ 값을 작게 설정(0.1)하여 KL Divergence의 제약을 줄임으로써 잠재 공간을 확장하고 생성 데이터의 다양성을 높인다.

- **Intensity VAE**: 강도 변형 $I_i$의 분포를 학습한다.
  - **손실 함수**: $\mathcal{L}_{I} = L_{d}^{I} + \beta L_{kl}^{I}$ (단순 픽셀 재구성 손실과 KL 손실 사용)

### 3. Data Synthesis and Segmentation
학습된 VAE의 디코더($D_S, D_I$)에 가우시안 분포 $\mathcal{N}(0, \sigma)$에서 샘플링한 잠재 벡터를 입력하여 새로운 변형 $S_g$와 $I_g$를 생성한다. 이를 통해 가상 이미지 $x_g$와 라벨 $y_g$를 합성한다.
$$x_g = (x_a + I_g) \circ S_g, \quad y_g = y_a \circ S_g$$
여기서 $\circ$는 워핑(Warping) 연산을 의미한다. 이렇게 생성된 대량의 데이터를 사용하여 2D U-Net 기반의 분할 네트워크를 Cross-Entropy 손실 함수($\mathcal{L}_{CE}$)로 학습시킨다.

## 📊 Results

### 실험 설정
- **데이터셋**: CANDI 데이터셋(T1-weighted brain MRI)과 자체 구축한 ABIDE 벤치마크를 사용한다.
- **지표**: Dice Coefficient를 사용하여 분할 성능을 측정한다.
- **비교 대상**: VoxelMorph, DataAug (Zhao et al. 2019), LT-Net (Wang et al. 2020) 및 Fully Supervised 학습 모델.

### 주요 결과
- **CANDI 데이터셋**: 제안 방법이 Mean Dice score 85.1%를 기록하며 SOTA 모델인 LT-Net(82.3%)보다 2.8% 향상된 성능을 보였다. 특히 표준 편차가 가장 낮아 성능의 안정성이 높음을 입증하였다.
- **ABIDE 벤치마크 (Generalization)**: 
  - **Seen set**: 제안 방법이 기존 One-shot 방법들보다 월등히 높은 성능을 보였다.
  - **Unseen set**: 보지 못한 데이터 소스에서도 성능 저하가 가장 적었다(타 방법 대비 약 3~5% 더 높은 성능 유지). 이는 VAE를 통한 확률적 증강이 모델의 강건함(Robustness)을 높였음을 시사한다.

### Ablation Study
- **VAE의 효과**: 단순 등록 기반 증강보다 VAE 기반 증강이 훨씬 높은 성능을 보였으며, 특히 강도 변형(Intensity deformation)의 다양성이 성능 향상에 크게 기여함을 확인하였다.
- **재구성 손실 조합**: Shape VAE에서 $L_d^S$와 $L_i^S$를 함께 사용했을 때 가장 좋은 성능을 보였는데, 이는 픽셀 단위의 변형뿐만 아니라 이미지 윤곽선의 일관성이 중요하기 때문이다.
- **하이퍼파라미터**: $\beta$ 값이 작을수록, 그리고 샘플링 시 $\sigma$ 값이 적절히 클 때(예: 10) 더 다양한 데이터가 생성되어 분할 성능이 향상되었다.

## 🧠 Insights & Discussion

본 논문은 의료 영상 분야에서 데이터 부족 문제를 해결하기 위해 '데이터 자체'가 아닌 '데이터 간의 변형'을 확률적으로 모델링했다는 점에서 매우 영리한 접근 방식을 취하고 있다. 

**강점**:
- GAN의 고질적인 문제인 모드 붕괴를 피하면서도, VAE의 $\beta$ 조절을 통해 생성 데이터의 다양성을 확보하였다.
- 형태(Shape)와 강도(Intensity)를 분리하여 모델링함으로써 실제 MRI 영상에서 발생하는 복합적인 변동성을 효과적으로 캡처하였다.
- 추론 단계에서는 오직 분할 네트워크만 사용하므로, 학습 시 VAE를 사용했음에도 불구하고 추론 속도나 연산량에는 영향이 없다.

**한계 및 논의사항**:
- VAE의 잠재 공간 샘플링 시 $\sigma$ 값에 의존적인 경향이 있으며, 최적의 $\sigma$를 찾는 과정이 실험적으로 이루어졌다.
- 본 연구는 뇌 MRI에 집중되어 있으나, 장기(Organ)의 형태 변화가 극심한 다른 의료 영상(예: 위장관 등)에서도 동일한 수준의 효과를 거둘 수 있을지는 추가적인 검증이 필요하다.
- $\beta$ 값을 낮추어 다양성을 확보하는 전략이 재구성 품질(Reconstruction quality)과 생성 다양성 사이의 Trade-off를 어떻게 최적화했는지에 대한 더 깊은 분석이 있었다면 좋았을 것이다.

## 📌 TL;DR

본 논문은 단 하나의 라벨링된 Atlas와 소수의 Unlabeled 이미지만을 사용하는 **One-shot 의료 영상 분할**을 위해 **3D VAE 기반의 확률적 데이터 증강 방법**을 제안한다. 형태 변형과 강도 변화의 분포를 각각의 VAE로 학습하고, 여기서 샘플링한 다양한 변형을 Atlas에 적용하여 대량의 가상 학습 데이터를 생성한다. 이를 통해 기존의 결정론적 증강 방식의 한계를 극복하고, 특히 새로운 데이터 소스에 대한 **일반화 성능을 크게 향상**시켰으며, 제안한 ABIDE 벤치마크를 통해 그 우수성을 입증하였다.