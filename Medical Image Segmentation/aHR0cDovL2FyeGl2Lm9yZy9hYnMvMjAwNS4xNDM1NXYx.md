# Enhancing Foreground Boundaries for Medical Image Segmentation

Dong Yang, Holger Roth, Xiaosong Wang, Ziyue Xu, Andriy Myronenko, Daguang Xu (2020)

## 🧩 Problem to Solve

현대 의료 영상 분석에서 객체 분할(Object Segmentation)은 임상 연구, 질병 진단 및 수술 계획 수립에 필수적인 역할을 수행한다. 다양한 의료 영상 모달리티(Modality)에 따라 장기, 뼈, 종양 등의 관심 영역(ROI, Region-of-Interest)을 식별하기 위해 자동 또는 반자동 분할 방식이 사용되고 있다.

그러나 기존의 분할 방식들은 관심 영역의 경계 부분(Boundary areas)을 정확하게 예측하는 데 어려움을 겪는 경향이 있다. 이는 영상 획득 과정에서 스캐너 설정, 호흡, 또는 신체 움직임 등으로 인해 경계 지역의 외관 대비(Appearance contrast)가 본질적으로 모호하게 나타나는 'Fuzzy appearance contrast' 현상 때문이다. 기존의 $\text{Multi-class weighted cross-entropy}$나 $\text{Soft Dice loss}$와 같은 손실 함수들은 클래스 불균형 문제는 해결할 수 있으나, 모든 픽셀 또는 복셀(Voxel)을 동일하게 취급하기 때문에 경계 지역의 특수성을 충분히 반영하지 못한다는 한계가 있다. 따라서 본 논문의 목표는 경계 지역에 추가적인 제약 조건을 부여하여 분할 품질을 향상시키는 새로운 손실 함수를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 학습 과정에서 경계 영역에 명시적으로 집중하도록 강제하는 **Boundary Enhancement (BE) Loss**를 도입하는 것이다.

이 방법의 중심적인 직관은 라플라시안 필터(Laplacian filter)를 적용하여 경계 영역에서는 강한 응답을 생성하고 그 외의 영역에서는 응답을 제거함으로써, 예측된 마스크와 실제 정답(Ground Truth) 마스크 간의 경계 차이를 직접적으로 최적화하는 것이다. 제안된 BE Loss는 별도의 전처리나 후처리가 필요 없으며, 특수한 네트워크 구조를 요구하지 않는 경량화된 설계라는 점이 주요 특징이다. 이를 통해 기존의 어떤 3D 백본 네트워크(Backbone networks)에도 쉽게 통합하여 사용할 수 있다.

## 📎 Related Works

경계 분할 성능을 높이기 위한 기존의 연구들(Chen et al., 2016; Oda et al., 2018; Karimi and Salcudean, 2019; Kervadec et al., 2018)이 존재한다. 하지만 이러한 접근 방식들은 다음과 같은 한계점을 가진다.

1. 일부 연구는 경계 계산을 위해 특수한 네트워크 아키텍처를 설계해야 하므로 모델의 복잡도가 증가한다.
2. 또 다른 연구들은 손실 함수를 계산하기 위해 복잡한 전처리나 후처리 과정을 요구하여 연산 부담을 초래한다.

본 논문에서 제안하는 BE Loss는 이러한 복잡한 요구사항 없이 단순한 필터링 연산만으로 경계를 강화하며, 계산 비용이 매우 적어 기존 방식들과 차별화된다.

## 🛠️ Methodology

### 전체 시스템 구조 및 원리

본 방법론은 3D 이진 분할 마스크 $S$에 라플라시안 필터 $L(\cdot)$를 적용하여 경계 영역을 강조하는 방식을 취한다. 라플라시안 필터는 경계 부분에서 강한 반응을 보이고 평탄한 영역에서는 0에 가까운 값을 출력한다.

### 주요 방정식 및 손실 함수

라플라시안 필터의 정의는 다음과 같다.

$$L(x,y,z) = \frac{\partial^2 S}{\partial x^2} + \frac{\partial^2 S}{\partial y^2} + \frac{\partial^2 S}{\partial z^2}$$

이 필터링 연산은 표준 3D 컨볼루션 연산을 통해 이산적으로 구현될 수 있다. $\text{Boundary Enhancement Loss}$ ($l_{BE}$)는 신경망의 예측 결과 $F(X)$에 필터를 적용한 값과 정답 라벨 $Y$에 필터를 적용한 값 사이의 $l^2$-norm 차이로 정의된다.

$$l_{BE} = \|L(F(X)) - L(Y)\|^2_2 = \left\| \frac{\partial^2 (F(X) - Y)}{\partial x^2} + \frac{\partial^2 (F(X) - Y)}{\partial y^2} + \frac{\partial^2 (F(X) - Y)}{\partial z^2} \right\|^2_2$$

이 손실 함수는 경계 지역에서 벗어난 거짓 양성(False positives)이나 원거리 이상치(Remote outliers)를 효과적으로 억제하는 특성을 가진다.

### 구현 상세 및 학습 절차

실제 구현에서 BE Loss는 다음과 같은 비학습성(Non-trainable) 컨볼루션 연산의 연속으로 구성된다.

1. **Smoothing**: 동일한 상수 값 $1/27$을 가진 3개의 $3 \times 3 \times 3$ 컨볼루션 레이어를 연속적으로 적용하여 스무딩을 수행한다.
2. **Edge Detection**: 마지막 레이어에 표준 3D 이산 라플라시안 커널을 적용한다. 이는 결과적으로 $\text{Laplacian of Gaussian (LoG)}$ 필터링과 유사한 동작을 수행한다.

최종적인 전체 손실 함수 $l_{overall}$은 $\text{Soft Dice loss}$ ($l_{dice}$)와 $\text{BE loss}$의 가중 합으로 정의된다.

$$l_{overall} = \lambda_1 \cdot l_{dice} + \lambda_2 \cdot l_{BE}$$

여기서 $\lambda_1$과 $\lambda_2$는 각 손실 함수의 가중치이다. BE Loss는 필터링 후 내부와 외부를 구분할 수 없기 때문에, 반드시 $\text{Soft Dice loss}$와 함께 사용되어야 한다.

## 📊 Results

### 실험 설정

- **데이터셋**: Medical Decathlon Challenge (MSD)의 Task 01(뇌종양 MRI 분할)과 Task 09(비장 CT 분할)를 사용하였다.
- **백본 네트워크**: 3D residual blocks를 사용하는 $\text{SegResNet}$ (Myronenko, 2018)을 베이스라인으로 사용하였다.
- **하이퍼파라미터**: 가중치는 $\lambda_1 = 1, \lambda_2 = 1000$으로 설정하였으며, $\text{Adam}$ 옵티마이저를 사용하였다.
- **평가 지표**: $\text{Dice score}$를 통해 검증 정확도를 측정하였다.

### 정량적 결과

실험 결과, 제안된 BE Loss를 적용했을 때 다음과 같은 성능 향상이 관찰되었다 (Table 1 기준).

| Method | Task 01 (Brain Tumor) | Task 09 (Spleen) |
| :--- | :---: | :---: |
| U-Net | 0.72 | 0.94 |
| AH-Net | 0.81 | 0.95 |
| SegResNet (Baseline) | 0.83 | 0.95 |
| SegResNet + Boundary Loss | 0.85 | 0.94 |
| SegResNet + Focal Loss | 0.85 | 0.95 |
| **SegResNet + Proposed BE Loss** | **0.85** | **0.96** |

### 결과 해석

제안된 방식은 뇌종양(비정형 객체)과 비장(정형 객체) 모두에서 우수한 성능을 보였으며, MRI와 CT라는 서로 다른 모달리티에서도 효과적임이 입증되었다. 특히 비장 분할(Task 09)에서는 기존의 모든 베이스라인 및 다른 손실 함수 적용 모델보다 높은 Dice score를 기록하였다.

## 🧠 Insights & Discussion

본 논문은 복잡한 아키텍처 변경 없이 손실 함수만으로 의료 영상의 고질적인 문제인 경계 모호성을 해결하려 했다는 점에서 실용적인 강점을 가진다. 특히 $\text{LoG}$ 필터링과 유사한 구조를 통해 연산 효율성을 확보하면서도 경계 제약 조건을 명시적으로 부여한 점이 효과적이었다.

다만, 본 논문에서 제시된 결과는 특정 백본 네트워크($\text{SegResNet}$)에 기반하고 있으며, 가중치 $\lambda_2$의 값이 $\lambda_1$에 비해 매우 크게($1000$배) 설정되어 있다. 이는 BE Loss의 출력값이 매우 작기 때문에 발생하는 현상으로 추정되나, 다른 네트워크나 데이터셋에서도 동일한 가중치 설정이 최적인지에 대한 추가 분석이 필요하다. 또한, BE Loss가 단독으로 사용될 수 없고 반드시 $\text{Dice loss}$와 결합되어야 한다는 제약 사항은 필터링 연산의 특성상 내부/외부 구분 능력이 없기 때문이라는 점이 명확히 기술되어 있다.

## 📌 TL;DR

본 연구는 의료 영상 분할 시 발생하는 경계 영역의 모호함 문제를 해결하기 위해, 라플라시안 필터를 이용한 경량화된 **Boundary Enhancement (BE) Loss**를 제안하였다. 이 손실 함수는 별도의 전/후처리나 구조 변경 없이 기존 3D 네트워크에 쉽게 통합 가능하며, MSD 데이터셋 실험을 통해 정형/비정형 객체 및 다양한 모달리티(CT, MRI)에서 분할 정확도(Dice score)를 향상시킴을 입증하였다. 이 연구는 향후 다양한 의료 영상 분할 모델의 경계 정밀도를 높이는 플러그인 형태로 적용될 가능성이 높다.
