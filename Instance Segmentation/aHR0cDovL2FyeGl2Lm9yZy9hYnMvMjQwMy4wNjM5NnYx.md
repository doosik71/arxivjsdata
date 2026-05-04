# A Segmentation Foundation Model for Diverse-type Tumors

Jianhao Xie, Ziang Zhang, Guibo Luo, and Yuesheng Zhu (2024)

## 🧩 Problem to Solve

본 논문은 의료 영상 분석에서 다양한 유형의 종양(Tumor)을 효과적으로 분할(Segmentation)하기 위한 거대 사전 학습 모델의 부재 문제를 해결하고자 한다. 일반적으로 의료 영상 데이터셋은 공개된 데이터의 양이 적어, 자연어 처리나 일반 컴퓨터 비전 분야와 같은 대규모 파운데이션 모델(Foundation Model)을 구축하는 데 어려움이 있다. 

기존의 $\text{nnU-Net}$과 같은 모델들은 매우 강력한 성능을 보이지만, 특정 작업이나 단일 데이터셋에 맞춰 학습되는 경향이 있어 새로운 데이터셋에 적용할 때마다 처음부터 다시 학습시켜야 하는 번거로움과 성능 불안정성 문제가 존재한다. 또한, 종양은 구조적 복잡성과 가변성이 매우 크기 때문에 기존의 다기관(Multi-organ) 분할 모델로는 높은 정확도를 달성하기 어렵다. 따라서 본 연구의 목표는 대규모 데이터셋 풀을 구축하고, 이를 통해 다양한 종양 유형에 범용적으로 적용 가능하며 하위 작업(Downstream tasks)으로의 전이 능력이 뛰어난 종양 분할 파운데이션 모델인 $\text{TSFM(Tumor Segmentation Foundation Model)}$을 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심적인 기여는 크게 세 가지로 요약할 수 있다.

첫째, 16억 개($1.6\text{ billion}$)의 파라미터를 가진 대규모 종양 분할 모델인 $\text{TSFM}$을 제안하였다. 이 모델은 $\text{Resblock-backbone}$과 $\text{Transformer-bottleneck}$을 결합한 효율적인 구조를 통해 복잡한 특징을 학습하고 높은 일반화 성능을 확보하였다.

둘째, 종양과 주변 장기 사이의 강한 공간적 상관관계(Spatial correlation)를 활용하기 위해 7개의 종양 데이터셋과 3개의 다기관 데이터셋을 통합하여 대규모 3D 의료 영상 데이터셋 풀을 구축하였다. 총 2,779개의 케이스와 30만 장의 이미지를 포함하며, 이는 기존의 단일 공개 데이터셋 규모를 크게 상회한다.

셋째, 사전 학습된 $\text{TSFM}$이 다양한 하위 작업에서 매우 빠른 수렴 속도와 높은 정확도를 보임을 입증하였다. 특히 $\text{nnU-Net}$ 대비 훨씬 적은 학습 횟수(Epoch)만으로도 동등하거나 더 우수한 성능을 달성할 수 있음을 보여주었다.

## 📎 Related Works

논문에서는 $\text{U-Net}$, $\text{nnU-Net}$, $\text{3D-TransUNet}$, $\text{SwinUNETR}$와 같은 딥러닝 기반의 의료 영상 분할 모델들을 언급한다. $\text{nnU-Net}$은 범용적인 프레임워크로서 매우 우수한 성능을 보이지만, 새로운 작업마다 처음부터 다시 학습(Train from the ground up)해야 한다는 한계가 있다. 

또한, 다기관 분할을 위한 기존 모델들이 존재하지만, 종양의 복잡한 구조적 가변성으로 인해 종양 분할 정확도는 여전히 낮다는 점을 지적한다. $\text{STU-Net}$과 같이 확장 가능하고 전이 가능한 모델이 제안된 바 있으나, 종양 영상 분할 분야에 특화된 대규모 모델은 여전히 부족한 실정이다. 본 연구는 $\text{MultiTalent}$의 데이터 통합 방식을 참고하여 데이터 부족 문제를 해결하고, $\text{CNN}$의 효율성과 $\text{Transformer}$의 전역적 특징 추출 능력을 결합하여 기존 모델들과 차별화를 꾀하였다.

## 🛠️ Methodology

### 1. 데이터셋 풀(Dataset Pool) 구축
데이터 부족 문제를 해결하기 위해 $\text{MultiTalent}$와 유사한 데이터 통합 방식을 사용하였다. 서로 다른 데이터셋 간에 동일한 대상에 대해 다른 라벨 ID가 부여되는 문제를 해결하기 위해 라벨 재매핑 테이블(Label remapping table)을 구축하여 고유 라벨을 부여하였다.

학습 시 사용된 손실 함수는 $\text{Binary Cross-Entropy (BCE)}$ 손실과 수정된 $\text{Dice loss}$를 결합하여 사용하였다. 또한, 각 데이터셋의 샘플 수 차이로 인한 불균형을 해소하기 위해, 각 케이스의 샘플링 확률을 해당 소스 데이터셋의 케이스 수 $n$의 제곱근에 반비례하게 설정하였다. 즉, 샘플링 확률 $P \propto \frac{1}{\sqrt{n}}$으로 설정하여 학습의 균형을 맞추었다.

### 2. 네트워크 구조 ($\text{TSFM}$ Architecture)
$\text{TSFM}$은 기본적으로 $\text{U-shaped}$ 구조와 스킵 연결(Skip connections)을 유지하며, $\text{Resblock-backbone}$과 $\text{Transformer-bottleneck}$을 결합하였다.

- **$\text{Resblock-Backbone}$**: $\text{Conv-IN-LeakyReLU}$ 층 두 개와 하나의 잔차 연결(Residual connection)로 구성된다. 입력 파라미터가 모두 양수가 되는 것을 방지하기 위해 $\text{pre-activation}$ 구조를 채택하였으며, $3\times3\times3$ 합성곱 커널을 사용한다.
- **$\text{Down-sampleblock}$**: $\text{Resblock}$을 기반으로 하되, 출력 채널 수를 2배로 늘리고 스트라이드(Stride)를 조정하여 벡터 크기를 입력의 절반으로 줄인다.
- **$\text{Upsampleblock}$**: $1\times1\times1$ 기반의 보간(Interpolation) 연산을 통해 크기를 2배로 늘린 후, $1\times1\times1$ 합성곱 층을 통해 채널 수를 다시 절반으로 줄인다.
- **$\text{Transformer-Bottleneck}$**: $\text{Transformer}$의 막대한 메모리 소비를 고려하여 $\text{U-Net}$의 가장 최하단 층(Bottleneck)에만 배치하였다. $\text{ViT-base}$ 버전을 사용하였으며, 12개의 $\text{Transformer encoder}$ 층으로 구성된다.

### 3. 전이 학습 ($\text{Transfer Learning}$)
$\text{TSFM}$은 $\text{CNN}$과 $\text{Transformer}$를 결합한 구조 덕분에 하위 작업에서 입력 영상의 크기가 달라지더라도 $\text{CNN}$ 부분의 학습된 가중치를 유지할 수 있다. $\text{Transformer-bottleneck}$ 부분만 미세 조정(Fine-tuning)함으로써 오버피팅 위험을 줄이고 전이 효율을 높였다.

## 📊 Results

### 1. 종양 데이터셋 성능 비교
$\text{TSFM}$을 $\text{nnU-Net}$, $\text{SwinUNETR}$, $\text{SAM-Med3D}$와 비교한 결과, 대부분의 종양 유형에서 $\text{TSFM}$이 가장 높은 $\text{Dice}$ 점수를 기록하였다. 특히 $\text{nnU-Net}$ 대비 평균적으로 약 $3\%$의 성능 향상을 보였으며, $\text{SAM-Med3D}$보다 월등히 높은 성능을 나타냈다.

### 2. 전이 학습 성능 분석
$\text{BraTS2020}$, $\text{KiTS2019}$, $\text{Abdomen-1K}$ 데이터셋을 통해 전이 능력을 평가하였다. 실험 결과, 사전 학습된 $\text{TSFM}$은 $\text{nnU-Net}$이 $1,000\text{ epochs}$ 동안 학습했을 때의 성능을 단 $100\text{ epochs}$($10\%$의 학습 기간)만으로 추월하거나 대등한 수준으로 달성하였다. 또한 $50\text{ epochs}$($5\%$ 수준)만 학습해도 $\text{nnU-Net}$과 유사한 성능을 보여, 매우 강력한 전이 효율성을 입증하였다.

### 3. 제거 실험 ($\text{Ablation Study}$)
- **모델 구조 검증**: 동일한 데이터셋 풀에서 학습한 $\text{nnU-Net}$보다 $\text{TSFM}$의 성능이 더 높게 나타났다. 이는 $\text{TSFM}$의 방대한 파라미터 수가 대규모 데이터셋의 다양한 특징을 더 잘 포착할 수 있게 함을 시사한다.
- **데이터셋 풀 검증**: 단일 데이터셋(예: $\text{Pancreas}$ 또는 $\text{AMOS22}$)으로 사전 학습한 모델보다, 통합 데이터셋 풀로 학습한 모델의 전이 성능이 훨씬 우수함을 확인하였다. 이는 파운데이션 모델 구축에 있어 다양하고 충분한 데이터 확보가 필수적임을 보여준다.

## 🧠 Insights & Discussion

본 논문은 대규모 파라미터와 정교하게 구축된 데이터셋 풀이 의료 영상 분할 모델의 일반화 성능을 비약적으로 상승시킬 수 있음을 보여주었다. 특히 $\text{CNN}$의 국소적 특징 추출 능력과 $\text{Transformer}$의 전역적 문맥 파악 능력을 적절히 배치하여, 메모리 효율성을 챙기면서도 종양과 같은 복잡한 구조를 효과적으로 분할할 수 있었다.

다만, 모델의 파라미터 수가 $1.6\text{ billion}$개에 달해 학습 시 매우 높은 계산 자원이 요구된다는 점이 잠재적인 한계로 보인다. 또한, 본 논문에서는 데이터셋 풀을 통한 성능 향상을 입증했지만, 구체적으로 어떤 데이터셋 조합이 특정 종양 분할에 가장 결정적인 영향을 미쳤는지에 대한 세밀한 분석은 부족하다. 하지만 전이 학습 단계에서 학습 시간을 획기적으로 단축(10% 수준)시킨 결과는 실제 임상 환경에서 새로운 데이터에 빠르게 적응해야 하는 모델 개발에 매우 중요한 시사점을 제공한다.

## 📌 TL;DR

본 연구는 16억 개의 파라미터를 가진 $\text{TSFM}$이라는 종양 분할 파운데이션 모델을 제안하였다. $\text{Resblock-backbone}$과 $\text{Transformer-bottleneck}$ 구조를 채택하고, 10개의 다양한 의료 영상 데이터셋을 통합한 대규모 데이터셋 풀로 사전 학습을 진행하였다. 그 결과, $\text{nnU-Net}$보다 우수한 분할 성능을 보였으며, 특히 전이 학습 시 $\text{nnU-Net}$ 학습 시간의 $10\%$만으로도 더 높은 성능을 달성하는 압도적인 효율성을 입증하였다. 이는 향후 의료 영상 분야에서 데이터 부족 문제를 극복하고 범용적인 분할 모델을 구축하는 데 중요한 기반이 될 것으로 기대된다.