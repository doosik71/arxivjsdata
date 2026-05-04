# Cross-dimensional transfer learning in medical image segmentation with deep learning

Hicham Messaoudi, Ahror Belaid, Douraied Ben Salem, Pierre-Henri Conze (2023)

## 🧩 Problem to Solve

본 연구가 해결하고자 하는 핵심 문제는 의료 영상 분석 분야에서 딥러닝 모델을 학습시키기 위해 필요한 고품질의 주석(annotated) 데이터가 절대적으로 부족하다는 점이다. 특히 의료 영상의 특성상 3D 볼륨 데이터의 경우 수동으로 주석을 다는 비용이 매우 높고 시간이 많이 소요되며, 전문가마다 결과가 다른 inter-observer variability 문제 또한 존재한다.

이러한 데이터 부족 현상은 모델의 일반화 성능을 저하시키며, 특히 고차원 데이터(3D)로 갈수록 심화된다. 따라서 본 논문의 목표는 자연 이미지(natural images)로 학습된 대규모 2D 분류 네트워크의 효율성을 2D 및 3D, 그리고 단일 및 다중 모달(uni- and multi-modal) 의료 영상 분할(segmentation) 작업으로 전이(transfer)하는 효율적인 방법을 제안하는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 설계 아이디어는 **'차원 간 전이 학습(Cross-dimensional transfer learning)'**이다. 저자들은 2D pre-trained 네트워크의 가중치를 단순 활용하는 것을 넘어, 이를 고차원 구조로 확장하는 두 가지 핵심 원칙을 제안한다.

1. **Weight Transfer Learning (WTL):** 이미 학습된 2D 인코더를 2D 또는 더 높은 차원의 U-Net 구조 내부에 임베딩하여 가중치를 재사용하는 방식이다.
2. **Dimensional Transfer Learning (DTL):** 2D 네트워크의 가중치를 3D 공간으로 확장(extrapolation)하여 3D 네트워크의 초기값으로 사용하는 방식이다. 구체적으로는 2D 가중치를 깊이(depth) 방향으로 반복적으로 연결(concatenating)하여 3D 가중치로 변환한다.

## 📎 Related Works

기존의 의료 영상 분할은 주로 U-Net과 그 변형 구조(Attention U-Net, U-Net++, U-Net3+)에 의존해 왔다. 이러한 모델들은 데이터 부족 문제를 완화하기 위해 인코더-디코더 구조를 사용하지만, 여전히 많은 경우 무작위 초기화(random initialization)를 통해 학습되므로 계산 비용이 높고 수렴 속도가 느리다는 한계가 있다.

이를 해결하기 위해 TernausNet이나 v16U-Net과 같이 ImageNet으로 학습된 VGG-11, VGG-16 등의 2D 가중치를 인코더에 적용하는 Transfer Learning 방식이 제안된 바 있다. 또한, EfficientNet과 같은 최신 스케일링 아키텍처를 분할 네트워크의 백본으로 사용하는 연구들이 진행되었다. 그러나 기존 연구들은 대부분 2D 가중치를 2D 작업에 사용하거나, 3D 작업 시 단순히 2D 슬라이스 단위로 처리하는 수준에 머물렀으며, 본 논문에서 제안하는 것처럼 2D 가중치를 3D 가중치로 직접 확장하여 초기화하는 '차원 간 전이'에 대한 탐구는 거의 이루어지지 않았다.

## 🛠️ Methodology

본 연구에서는 전이 학습 원칙에 따라 세 가지 네트워크 구조(Omnia-Net, DS-Net, DX-Net)를 제안한다. 모든 네트워크의 기본 인코더로는 EfficientNet을 사용한다.

### 1. Omnia-Net (Weight Transfer Learning)

Omnia-Net은 pre-trained 2D 분류 네트워크를 인코더로 사용하는 U-Net 스타일의 아키텍처이다.

- **구조:** 입력 이미지의 전체 스케일 특성을 활용하기 위해 인코더 시작 부분에 컨볼루션 블록을 추가하였다. 디코더는 최근접 이웃 보간(nearest neighbor interpolation), 연결(concatenation), 그리고 두 개의 $3 \times 3$ 컨볼루션 층(Batch Normalization 및 ReLU 포함)으로 구성된다.
- **적용:** 2D 심초음파 영상 및 3D 복부 장기 영상의 2D axial slice 분할에 사용된다.

### 2. DS-Net (Dimensionally-Stacked Network)

DS-Net은 3D 데이터를 처리하기 위해 2D pre-trained 네트워크를 내부에 통합한 고차원 구조이다.

- **파이프라인:**
    1. **3D $\rightarrow$ 2D 인코딩:** 3D 데이터를 높이와 너비는 유지한 채 깊이(depth) 차원을 채널 수로 압축하여 2D 데이터로 변환한다.
    2. **2D 처리:** 압축된 데이터를 pre-trained 2D EfficientNet 인코더와 2D 디코더에 통과시킨다.
    3. **2D $\rightarrow$ 3D 디코딩:** 2D 디코더의 출력을 다시 3D 디코더로 전달하여 최종 3D 분할 맵을 생성한다. 이때 3D 디코더는 Batch Normalization 대신 Instance Normalization을 사용한다.

### 3. DX-Net (Dimensionally-eXpanded Network)

DX-Net은 2D 가중치를 3D로 직접 확장하여 초기화하는 3D U-Net 스타일의 네트워크이다.

- **Dimensional Transfer:** ImageNet(noisy-student training 방식)으로 학습된 2D 가중치를 깊이 방향으로 반복 연결하여 3D 가중치로 변환하고, 이를 3D EfficientNet 인코더의 초기값으로 설정한다.
- **구조:** 인코더 전단에 공간적 특징을 캡처하기 위한 컨볼루션 층을 배치하고, 디코더에서는 Transposed 3D Convolution과 SiLU 활성화 함수를 사용한다.

### 4. 학습 절차 및 손실 함수

- **손실 함수:** 클래스 불균형 문제를 해결하기 위해 Dice Loss와 Binary Cross-Entropy (BCE)를 결합한 복합 손실 함수 $\mathcal{L}_{\text{loss}}$를 사용한다.
$$ \mathcal{L}_{\text{loss}} = 1 - \frac{2}{N} \sum_{n=0}^{N} \frac{\sum_{i=0}^{I} y_{i,n} g_{i,n}}{\sum_{i=0}^{I} y_{i,n} + \sum_{i=0}^{I} g_{i,n} + \epsilon} - \frac{1}{N} \sum_{n=0}^{N} \sum_{i=0}^{I} [y_{i,n} \log(g_{i,n}) + (1-y_{i,n}) \log(1-g_{i,n})] $$
여기서 $y_{i,n}$은 네트워크의 출력값, $g_{i,n}$은 정답(ground truth)이며, $\epsilon$은 0으로 나누는 것을 방지하는 상수이다.
- **최적화:** Nadam 옵티마이저를 사용하며, 학습률은 에포크마다 5%씩 감소시키는 decay 방식을 적용한다.

## 📊 Results

### 1. 데이터셋 및 지표

- **데이터셋:** CAMUS (심초음파), CHAOS (복부 CT/MR), BraTS 2022 (뇌종양 MR).
- **평가 지표:** Dice Score, MAD (Mean Absolute Distance), HD (Hausdorff Distance), RAVD (Relative Absolute Volume Difference), ASSD, MSSD.

### 2. 주요 정량적 결과

- **CAMUS (2D):** Omnia-Net은 심내막(Endocardium)과 심외막(Epicardium) 분할 모두에서 SOTA 성능을 보이며 챌린지 1위를 기록하였다. 특히 심실 용적 추정의 상관 계수가 기존 방법 대비 크게 향상되었다.
- **CHAOS (2D/3D):** Omnia-Net은 2D 기반 방법들 중 가장 우수한 평균 점수를 기록했으며, 온라인 평가 플랫폼에서 3위를 차지하였다. 특히 다중 모달 설정인 Task 1에서 타 2D 네트워크 대비 Dice Score 기준 11% 이상의 성능 격차를 보였다.
- **BraTS 2022 (3D):**
  - **DS-Net:** Whole Tumor(WT) 영역에서 평균 Dice 91.69%를 달성하며 부종(Edema) 영역 분할에 강점을 보였다.
  - **DX-Net:** Tumor Core(TC)와 Enhancing Tumor(ET) 영역에서 각각 84.77%, 83.88%의 Dice score를 기록하며, DS-Net보다 종양의 핵심 부위 식별 능력이 뛰어남을 입증하였다.

### 3. 실험적 분석

- DX-Net은 3D 가중치를 직접 사용하므로 볼륨 특징(volumetric features)을 더 잘 캡처하며, DS-Net보다 학습 속도가 더 빨랐다(이미지당 1.5초 vs 2초).

## 🧠 Insights & Discussion

### 강점 및 해석

본 연구는 2D pre-trained 모델의 지식을 고차원(3D)으로 전이시키는 구체적인 방법론(WTL, DTL)을 제시하여, 의료 영상의 고질적인 문제인 데이터 부족 문제를 효과적으로 완화하였다. 특히 DX-Net의 결과는 2D 가중치를 3D로 확장하는 단순한 아이디어가 실제 3D 의료 영상의 복잡한 구조를 학습하는 데 매우 유효한 초기값으로 작용할 수 있음을 보여준다.

### 한계 및 논의사항

- **학습 시간 및 데이터 증강:** 저자들은 시간과 자원 제약으로 인해 데이터 증강(data augmentation)을 충분히 적용하지 못했으며, 학습 에포크 수가 100회 미만으로 제한적이었다는 점을 언급한다. 이를 확장한다면 더 높은 성능 향상이 가능할 것으로 보인다.
- **ET 클래스의 낮은 점수:** BraTS 결과에서 Enhancing Tumor(ET)의 점수가 상대적으로 낮은데, 이는 BraTS 평가 플랫폼의 엄격한 이진 점수 계산 방식(작은 false positive에도 큰 감점) 때문인 것으로 분석된다.
- **가중치 변환의 단순성:** 2D 가중치를 단순히 반복 연결하여 3D로 확장하는 방식 외에, 더 정교한 차원 확장 기법이 존재할 가능성이 있으며 이는 향후 연구 과제로 남아있다.

## 📌 TL;DR

본 논문은 자연 이미지로 학습된 2D EfficientNet의 가중치를 의료 영상 분할 모델로 전이하는 **차원 간 전이 학습(Cross-dimensional transfer learning)** 방법론을 제안한다. 2D 가중치를 네트워크 내부에 임베딩하는 **WTL(Omnia-Net, DS-Net)**과 2D 가중치를 3D로 확장하여 초기화하는 **DTL(DX-Net)**을 통해, 데이터가 부족한 의료 영상 환경에서도 SOTA 수준의 분할 성능을 달성하였다. 이 연구는 특히 3D 의료 영상 모델 학습 시 2D pre-trained 가중치를 활용한 초기화가 학습 효율과 정확도를 획기적으로 높일 수 있음을 시사한다.
