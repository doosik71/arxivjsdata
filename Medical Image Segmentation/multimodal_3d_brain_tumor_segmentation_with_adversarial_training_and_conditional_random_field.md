# Multimodal 3D Brain Tumor Segmentation with Adversarial Training and Conditional Random Field

Lan Jiang, Yuchao Zheng, Miao Yu, Haiqing Zhang, Fatemah Aladwani, and Alessandro Perelli (2024)

## 🧩 Problem to Solve

본 논문은 뇌종양, 특히 신경교종(glioma)의 구조적 복잡성과 개인별 형태 차이로 인해 발생하는 3D 뇌종양 분할(Segmentation)의 어려움을 해결하고자 한다. 뇌종양 분할에서 정확한 경계를 찾아내는 것은 매우 중요하지만, 수동으로 주석(Annotation)을 단 데이터가 부족한 상황에서 딥러닝 모델이 생성하는 분할 결과의 경계가 뭉툭하거나 부정확한 문제가 빈번히 발생한다.

따라서 본 연구의 목표는 멀티모달 MRI(T1, T1c, T2, FLAIR) 데이터를 활용하여 뇌종양의 세 가지 하위 영역인 전체 종양(Whole Tumor, WT), 종양 핵심(Tumor Core, TC), 그리고 강화 종양(Enhancing Tumor, ET)을 정밀하게 분할할 수 있는 3D-vGAN 모델을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 V-Net의 공간 특징 추출 능력과 Conditional Random Field(CRF)의 세부 디테일 복원 능력을 Generative Adversarial Network(GAN) 구조 내에서 결합하는 것이다. 주요 기여 사항은 다음과 같다.

1. **3D-vGAN 모델 제안**: V-Net을 기반으로 한 Generator와 CNN 기반의 Discriminator를 결합하여, 복잡한 뇌종양의 형태에 대응하고 분할 경계의 정밀도를 높였다.
2. **Pseudo-3D Convolution 적용**: 연산 효율성을 높이기 위해 완전한 3D 컨볼루션 대신 2D 필터와 1D 필터를 순차적으로 적용하는 Pseudo-3D 구조를 도입하여 파라미터 수를 줄이고 비선형성을 증가시켰다.
3. **CRF-RNN 모듈 통합**: Generator의 후처리 단계에 CRF-RNN 레이어를 추가하여 출력 라벨의 공간적 연속성을 강화하고 국소적인 픽셀 위치를 정확하게 수정함으로써 부드럽고 정밀한 분할 결과를 얻도록 설계하였다.
4. **Discriminator 가이드 강화**: Discriminator의 입력으로 생성된 분할 이미지뿐만 아니라 원본 이미지를 추가 정보(Supplemental guidance)로 제공하여 식별 능력을 향상시켰다.

## 📎 Related Works

논문에서는 GAN과 V-Net(3D U-Net)이 뇌종양 분할에 널리 사용되고 있음을 언급한다. 기존 연구들은 주로 2D 분할이나 단순한 3D 네트워크에 의존해 왔으나, 다음과 같은 한계가 존재한다.

- **2D 분할의 한계**: 2D 슬라이스 단위의 분할은 결과물 간의 불일치가 발생할 수 있으며, 표면 정보가 결여되고 중요한 3D 문맥(Context)을 상실하는 문제가 있다.
- **경계 부정확성**: 기존 3D 네트워크들은 종양의 복잡한 경계를 정밀하게 묘사하는 데 한계가 있으며, 특히 수동 주석 데이터가 부족할 때 경계가 거칠게 나타나는 경향이 있다.

본 연구는 이러한 한계를 극복하기 위해 3D 공간 정보를 보존하는 V-Net에 적대적 학습(Adversarial Training)의 정밀함과 CRF의 공간 최적화 능력을 결합하여 차별성을 둔다.

## 🛠️ Methodology

### 전체 파이프라인

3D-vGAN의 전체 구조는 Generator와 Discriminator가 서로 경쟁하며 학습하는 GAN 구조를 따른다. 입력으로는 4가지 모달리티(T1, T1c, T2, FLAIR)의 3D MRI 이미지가 사용된다. Generator가 생성한 분할 맵과 원본 이미지가 Discriminator로 전달되며, Discriminator의 피드백을 통해 Generator가 더욱 실제와 유사한 분할 맵을 생성하도록 학습된다.

### Generator 및 Pseudo-3D

Generator의 백본은 Residual Nets를 포함한 V-Net 구조이다. Encoder는 4개의 다운샘플링 단계로 구성되며, 각 단계는 Residual Block, Instance Normalization, Leaky ReLU 활성화 함수로 이루어진다. Decoder는 3D Transposed Convolution을 통해 업샘플링을 수행한다.

특히, 연산 비용을 줄이기 위해 Pseudo-3D 컨볼루션을 적용하였다. 이는 3D 차원(가로, 세로, 깊이)을 2D 슬라이스 차원($S$)과 1D 깊이 차원($D$)으로 분리하여 처리하는 방식이며, 다음과 같은 방정식으로 표현된다.
$$(I+DS)x_t := x_t + D(S(x_t)) = x_{t+1}$$
여기서 $x_t$와 $x_{t+1}$은 각각 유닛의 입력과 출력이며, $S$는 2D 필터, $D$는 1D 필터를 의미한다.

### Discriminator

Discriminator는 다층 3D CNN으로 구성되어 있으며, 생성된 이미지 $\hat{y}$와 실제 정답(Ground Truth) $y$를 판별한다. 이때 원본 이미지 $x$를 함께 입력받아 가이드로 활용함으로써 판별 성능을 높인다.

### 손실 함수 (Loss Function)

학습은 Generator 손실($L_G$)과 Discriminator 손실($L_D$)의 최적화를 통해 이루어진다.

Generator의 손실 함수는 다음과 같다.
$$L_G = L_2[D(x, \hat{y}), 1] + \alpha GDL(y, \hat{y})$$
여기서 $L_2$는 Discriminator가 생성된 이미지를 진짜라고 판단하게 만드는 손실이며, $GDL$은 Generalized Dice Loss로 예측값 $\hat{y}$와 정답 $y$ 사이의 직접적인 유사도를 측정한다. $\alpha$는 두 손실의 균형을 조절하는 가중치 계수이다.

Discriminator의 손실 함수는 다음과 같다.
$$L_D = L_2[D(x, y), 1] + L_2[D(x, \hat{y}), 0]$$
이는 실제 정답 $y$는 1(진짜)로, 생성된 이미지 $\hat{y}$는 0(가짜)으로 판별하도록 학습하는 것을 목표로 한다.

### Conditional Random Field (CRF) Module

CRF-RNN은 CNN의 출력 결과에 대해 공간적 연속성을 부여하기 위한 후처리 모듈이다. 다음의 5단계 반복 과정을 거친다.

1. **Message Passing**: 가우시안 필터를 통해 주변 픽셀의 정보를 전달한다.
2. **Re-weighting**: 필터 출력값의 가중 합을 구한다.
3. **Compatibility Transform**: 라벨 간의 호환성을 고려하여 값을 변환한다.
4. **Unary Addition**: 초기 단일 포텐셜(Unary Potentials)을 더한다.
5. **Normalizing**: 소프트맥스 함수와 유사하게 정규화하여 확률 분포를 생성한다.

## 📊 Results

### 실험 설정

- **데이터셋**: BraTS-2018 (100개의 3D 이미지, 4개 모달리티)
- **전처리**: z-score 표준화, $128 \times 128 \times 128$ 크기의 패치 추출, 3D 회전 및 탄성 변형을 통한 데이터 증강.
- **평가 지표**: Dice Similarity Coefficient (DSC), Hausdorff Distance (HD), Sensitivity, Specificity.
- **하이퍼파라미터**: Adam Optimizer ($\lambda=2 \times 10^{-4}$), 학습률 $0.0001$, 배치 크기 4, 200 Epoch 학습.

### 주요 결과

정량적 평가 결과, 3D-vGAN은 기존 모델들(U-net, GAN, FCN, 3D V-net)보다 모든 지표에서 우수한 성능을 보였다.

- **DSC**: 82.13%로 가장 높음.
- **Sensitivity**: 84.42%를 기록하여, 0.8 미만이었던 기존 모델들에 비해 크게 향상됨.
- **Specificity**: 99.97%로 매우 높은 정확도를 보임.
- **Hausdorff Distance**: 11.89mm로 기존 모델들(26~38mm)에 비해 획기적으로 감소하여 경계 예측의 정밀도가 향상되었음을 입증하였다.

### 파라미터 $\alpha$ 분석

$\alpha$ 값에 따른 성능 변화 실험 결과, $\alpha=5$일 때 최적의 성능을 보였다. 이는 $\alpha$가 증가함에 따라(즉, Discriminator의 가이드 역할이 커질수록) 모델의 성능이 향상되는 경향이 있음을 보여준다.

## 🧠 Insights & Discussion

본 논문은 적대적 학습 구조에 Pseudo-3D 컨볼루션과 CRF-RNN을 효과적으로 통합하여 3D 의료 영상 분할의 고질적인 문제인 경계 부정확성을 해결하였다. 특히 Discriminator에 원본 이미지를 함께 입력한 설계는 생성기가 단순한 형태를 모방하는 것을 넘어, 원본 이미지의 해부학적 구조를 반영하도록 강제하는 효과적인 전략으로 판단된다.

다만, 몇 가지 논의할 점이 있다. 첫째, 사용된 데이터셋의 규모가 100개로 매우 작아, 데이터 증강을 적용했음에도 불구하고 더 큰 규모의 데이터셋에서 일반화 성능을 검증할 필요가 있다. 둘째, $\alpha$ 값에 따라 성능 편차가 크게 나타나는데, 이를 자동으로 최적화할 수 있는 방법론에 대한 논의가 부족하다. 셋째, CRF-RNN의 반복 계산으로 인한 추론 시간 증가 문제가 구체적으로 분석되지 않았다.

## 📌 TL;DR

본 연구는 뇌종양의 정밀한 3D 분할을 위해 **V-Net, Pseudo-3D Convolution, CRF-RNN을 결합한 3D-vGAN 모델**을 제안하였다. 이 모델은 적대적 학습을 통해 경계 정밀도를 높이고 CRF로 세부 디테일을 복원함으로써, BraTS-2018 데이터셋에서 기존 모델 대비 높은 DSC(82.13%)와 낮은 Hausdorff Distance(11.89mm)를 달성하였다. 이는 향후 멀티모달 MRI 기반의 정밀 의료 영상 분석 및 자동 진단 시스템 구축에 중요한 기여를 할 수 있을 것으로 보인다.
