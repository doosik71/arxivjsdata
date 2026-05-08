# Bridging 2D and 3D Segmentation Networks for Computation-Efficient Volumetric Medical Image Segmentation: An Empirical Study of 2.5D Solutions

Yichi Zhang, Qingcheng Liao, Le Ding, Jicong Zhang (2022)

## 🧩 Problem to Solve

본 논문은 의료 영상 분석에서 핵심적인 volumetric medical image segmentation(체적 의료 영상 분할)의 효율성과 정확도 사이의 트레이드오프 문제를 해결하고자 한다. MRI나 CT와 같은 의료 영상은 본질적으로 3D 데이터이다. 이를 처리하기 위해 기존에는 두 가지 극단적인 접근 방식이 사용되었다.

첫째, 3D 데이터를 2D 슬라이스로 나누어 처리하는 2D CNN 방식은 연산 비용이 낮고 추론 속도가 빠르지만, 인접 슬라이스 간의 공간적 정보(inter-slice information)를 무시하므로 분할 정확도가 떨어지며 3D 공간 상에서 결과가 불연속적으로 나타나는 문제가 있다.

둘째, 3D convolution을 사용하는 3D CNN 방식은 체적 공간 정보를 충분히 활용하여 정확도가 높지만, 연산 비용과 추론 시간이 매우 높고 파라미터 수가 많아 데이터셋이 작은 의료 영상 분야에서 overfitting(과적합) 위험이 크다는 단점이 있다. 또한, 높은 GPU 메모리 요구량은 실제 임상 적용에 큰 걸림돌이 된다.

따라서 본 연구의 목표는 2D CNN의 효율성과 3D CNN의 공간 문맥 파악 능력을 동시에 확보할 수 있는 '2.5D' segmentation 방법론들을 체계적으로 분석하고, 다양한 데이터셋과 모달리티에 대해 어떤 방법이 가장 효과적인지 실증적으로 비교하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 2.5D segmentation 방법론에 대한 체계적인 분류 체계를 제안하고, 이를 동일한 조건에서 대규모로 비교 분석했다는 점이다. 구체적인 기여 사항은 다음과 같다.

1. **2.5D 방법론의 범주화**: 2.5D 접근 방식을 $\text{Multi-view fusion}$, $\text{Incorporating inter-slice information}$, $\text{Fusing 2D/3D features}$의 세 가지 카테고리로 정의하고 최신 동향을 정리하였다.
2. **광범위한 실증적 비교**: 서로 다른 모달리티(CT, MRI)와 타겟 장기(심장, 비장, 전립선)를 포함하는 세 가지 대표 공공 데이터셋을 활용하여, 2D 및 3D CNN과 2.5D 방법론들의 성능을 동일한 백본 구조와 설정 하에서 비교 평가하였다.
3. **데이터 특성에 따른 가이드라인 제시**: 특히 anisotropic voxel(비등방성 복셀, z축 해상도가 x, y축보다 현저히 낮은 경우)을 가진 데이터에서 3D CNN보다 2.5D 방법론이 더 유리할 수 있음을 입증하여, 실무적인 모델 선택 기준을 제시하였다.

## 📎 Related Works

논문에서는 체적 의료 영상 분할을 위해 2D와 3D의 간극을 메우려는 2.5D(또는 pseudo 3D) 방법론들을 다음과 같이 소개한다.

- **Multi-view Fusion (MVF)**: Axial, Sagittal, Coronal의 세 가지 직교 평면에서 2D 예측을 수행한 뒤 이를 결합하는 방식이다. 단순 Majority Voting 외에도 얕은 3D CNN을 이용해 결과를 융합하는 Volumetric Fusion Net (VFN) 등이 제안되었다.
- **Incorporating Inter-slice Information**: 2D CNN의 입력 단계에서 인접 슬라이스를 함께 입력하거나, RNN(특히 Bi-CLSTM)을 통해 슬라이스 간 시퀀스 정보를 학습하거나, Attention mechanism을 사용하여 인접 슬라이스의 중요 영역을 가이드하는 방식이다.
- **Fusing 2D/3D Features**: 2D와 3D 인코더를 모두 사용하여 추출된 특성 맵(feature map)을 융합하는 방식이다. 2D 결과로 3D CNN에 사전 형태 정보를 제공하거나, Encoder 단계 또는 Output 단계에서 특성을 결합하여 효율성을 높인다.

기존 연구들은 각자 서로 다른 데이터셋과 백본 네트워크를 사용했기에, 어떤 2.5D 전략이 일반적으로 우수한지에 대한 직접적인 비교가 부족했다는 점이 본 논문의 차별점이다.

## 🛠️ Methodology

### 전체 시스템 구조 및 백본

본 연구는 공정한 비교를 위해 모든 실험에서 $\text{U-Net}$을 기본 백본으로 사용하였다. 2D CNN의 경우 2D U-Net을, 3D CNN의 경우 3D U-Net을 적용하였으며, 기본 convolution 블록의 feature map 수는 8개로 설정하였다.

### 학습 절차 및 손실 함수

- **최적화**: Adam optimizer를 사용하였으며, 초기 learning rate는 $0.001$로 설정하고 20 epoch 동안 loss 개선이 없을 시 $0.5$배씩 감소시켰다.
- **손실 함수**: 영역의 일치도를 높이는 $\text{Dice loss}$와 픽셀 단위 분류를 위한 $\text{Cross-entropy loss}$를 결합하여 사용하였다.
$$\text{Loss} = \text{Cross-Entropy Loss} + \text{Dice Loss}$$

### 평가 지표

- **Dice Coefficient (DSC)**: 분할 결과와 ground truth 간의 영역 중첩도를 측정하며, 1에 가까울수록 성능이 좋다.
- **95% Hausdorff Distance (95HD)**: 경계선 간의 최대 거리 중 상위 5%를 제외한 값으로, 경계 오류를 측정하며 0에 가까울수록 성능이 좋다.

### 실험 설계

1. **Multi-view Fusion 실험**: 서로 다른 평면(S, C, A)의 결과와 이를 $\text{Majority Voting (MV)}$, $\text{Weighted Voting (WV)}$, $\text{VFN}$으로 융합한 결과를 비교한다.
2. **Inter-slice Information 실험**: 입력 슬라이스 수(1, 3, 5, 7개)에 따른 변화를 측정하고, $\text{RNN}$, $\text{Attention (Shape/Border)}$ 방식의 효율성을 비교한다.
3. **2D/3D Feature Fusion 실험**: 융합 단계(Encoder vs Output)와 융합 방법($\text{Add}$ vs $\text{Squeeze-and-Excitation}$)에 따른 성능을 비교한다.

## 📊 Results

### 1. Multi-view Fusion 결과

Cardiac MRI와 Spleen CT에서는 MVF가 단순 2D보다 성능이 좋았으며, 특히 VFN을 사용했을 때 가장 높은 성능을 보였다. 그러나 Prostate MRI와 같이 $\text{Anisotropic}$한 데이터에서는 일부 평면(S, C)의 2D 결과가 매우 좋지 않아 단순 MV는 오히려 성능이 떨어졌다. 이를 보완하기 위해 해상도가 높은 단축 슬라이스(short-axis)에 가중치를 주는 $\text{Weighted Voting (WV)}$이 효과적임을 확인하였다.

### 2. Inter-slice Information 결과

인접 슬라이스를 추가 입력으로 사용하는 경우, 1개보다 3개를 사용할 때 성능이 향상되었으나 5개 이상부터는 오히려 성능이 정체되거나 하락하였다. 이는 너무 많은 인접 슬라이스가 오히려 노이즈로 작용하거나 불필요한 연산을 증가시키기 때문으로 분석된다. $\text{RNN}$ 및 $\text{Attention}$ 기반 방법들은 단순 multi-slice 입력보다 우수하였으며, 특히 **Prostate MRI에서는 모든 2.5D 방법론이 3D U-Net보다 높은 성능**을 기록하였다.

### 3. 2D/3D Feature Fusion 결과

모든 융합 방법이 단일 2D CNN보다는 우수한 성능을 보였다. Cardiac MRI의 경우 Output 단계에서의 융합이 효과적이었으나, 다른 데이터셋에서는 결과가 가변적이었다. 전반적으로 Encoder 단계에서의 융합이 연산 비용 대비 효율적인 특성 추출을 가능하게 하였다.

### 4. 모델 복잡도 비교

실험 결과, 모든 2.5D 방법론은 3D CNN에 비해 파라미터 수와 학습 시간이 현저히 적었다. 3D U-Net의 파라미터 수는 $1.403\text{M}$이며 학습 시간이 2D 대비 $4.14\text{x}$ 높았던 반면, 2.5D 방법들은 대부분 $0.5\text{M}$ 내외의 파라미터와 $1.1\text{x} \sim 2.3\text{x}$ 수준의 학습 시간만을 소모하였다.

## 🧠 Insights & Discussion

### 강점 및 핵심 발견

본 논문은 3D CNN이 항상 최선의 선택이 아님을 입증하였다. 특히 **Anisotropic volumetric images**의 경우, z축 해상도가 낮아 3D convolution 과정에서 downsampling이 일어날 때 중요한 정보가 손실될 위험이 크다. 이 경우 2.5D 방법론이 3D CNN보다 더 정확한 분할 결과를 낼 수 있으며, 연산 효율성까지 챙길 수 있다는 점이 핵심적인 통찰이다.

### 한계점 및 비판적 해석

- **백본의 제한**: 모든 실험이 U-Net 기반으로 진행되었기에, 최신 Transformer 기반 구조나 다른 아키텍처에서도 동일한 경향성이 나타날지는 미지수이다.
- **재현성의 한계**: 저자들은 모든 방법론을 완벽하게 재현했다고 주장하기 어렵다고 명시하였으며, 하이퍼파라미터 튜닝의 영향이 결과에 포함되었을 가능성이 있다.
- **데이터셋 규모**: 3개의 데이터셋만으로 모든 의료 영상의 특성을 일반화하기에는 다소 부족함이 있다.

## 📌 TL;DR

본 연구는 3D 의료 영상 분할에서 2D의 효율성과 3D의 정확성을 동시에 잡기 위한 **2.5D 방법론들을 체계적으로 분류하고 실증적으로 비교 분석**하였다. 실험 결과, 2.5D 방법론은 3D CNN보다 훨씬 적은 연산 자원을 사용하면서도 2D CNN보다는 월등한 성능을 보였으며, 특히 **z축 해상도가 낮은 비등방성(Anisotropic) 데이터에서는 3D CNN보다 더 우수한 성능**을 나타내기도 하였다. 이는 향후 계산 효율적인 의료 영상 분할 모델을 설계할 때 데이터의 해상도 특성에 따라 2.5D 접근 방식을 적극적으로 고려해야 함을 시사한다.
