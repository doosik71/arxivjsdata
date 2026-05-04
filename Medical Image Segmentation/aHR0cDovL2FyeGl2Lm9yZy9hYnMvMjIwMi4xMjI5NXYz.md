# Factorizer: A Scalable Interpretable Approach to Context Modeling for Medical Image Segmentation

Pooya Ashtari, Diana M. Sima, Lieven De Lathauwer, Dominique Sappey-Marinier, Frederik Maes, and Sabine Van Huffel (2022)

## 🧩 Problem to Solve

의료 영상 분할(Medical Image Segmentation) 분야에서 Convolutional Neural Networks(CNN), 특히 U-Net 구조는 매우 효과적이지만, 컨볼루션 연산의 내재적인 지역성(locality)으로 인해 전역적 문맥(global context)을 충분히 활용하지 못하는 한계가 있다. 뇌 병변과 같이 크기와 모양이 매우 다양하고 광범위한 구조를 인식하기 위해서는 전역적 문맥 모델링이 필수적이다.

최근 Vision Transformer(ViT)와 같은 Transformer 기반 모델들이 long-range dependency를 모델링하는 능력 덕분에 유망한 성과를 보이고 있으나, Self-attention 메커니즘의 계산 복잡도가 시퀀스 길이의 제곱에 비례하는 $O(N^2)$이라는 치명적인 단점이 있다. 이로 인해 기존의 Transformer 모델들은 이미지 해상도를 상당히 낮춘 후에만 attention 레이어를 적용할 수 있으며, 이는 고해상도 단계에서 존재하는 전역적 문맥 정보를 소실시키는 결과를 초래한다.

본 논문의 목표는 계산 복잡도를 낮추면서도 전역적 문맥을 효과적으로 모델링할 수 있고, 동시에 해석 가능성(interpretability)을 제공하는 새로운 분할 모델인 **Factorizer**를 제안하는 것이다.

## ✨ Key Contributions

본 연구의 핵심 아이디어는 저차원 행렬 근사(Low-Rank Matrix Approximation, LRMA), 그 중에서도 **Nonnegative Matrix Factorization(NMF)**을 미분 가능한 레이어로 구성하여 딥러닝 아키텍처에 통합하는 것이다.

주요 기여 사항은 다음과 같다.
1. 의료 영상 분할을 위해 행렬 분해(Matrix Factorization) 레이어를 포함한 최초의 엔드-투-엔드(end-to-end) 딥러닝 모델을 제시하였다.
2. Block Coordinate Descent(BCD) 솔버를 사용하여 문맥 정보를 효율적으로 모델링하는 미분 가능한 NMF 레이어를 구축하였다.
3. 지역적 문맥을 충분히 활용하기 위해 **Shifted Window (SW) Matricize** 연산을 도입하고 이를 NMF와 결합하였다.
4. NMF 기반의 확장 가능하고 해석 가능한 U-shaped 분할 모델을 제안하였으며, BraTS 및 ISLES’22 데이터셋에서 SOTA(State-of-the-art) 성능을 달성하였다.

## 📎 Related Works

**CNN 기반 분할 모델**
U-Net 및 그 변형 모델(3D U-Net, UNet++, nnU-Net 등)은 의료 영상 분할의 표준으로 자리 잡았다. 하지만 작은 커널 사이즈를 사용하는 컨볼루션 연산은 지역적인 정보만을 집계하므로, 뇌종양과 같이 전역적인 해부학적 구조에 대한 이해가 필요한 작업에서는 한계를 보인다.

**Visual Transformers**
ViT는 이미지 패치를 시퀀스로 처리하여 전역적 의존성을 학습하지만, 데이터 요구량이 많고 계산 복잡도가 매우 높다. 이를 해결하기 위해 Swin Transformer와 같은 계층적 구조나 PvT, CvT 등이 제안되었다. UNETR, nnFormer와 같은 3D 의료 영상용 Transformer 모델들도 등장했으나, 여전히 계산 비용 문제로 인해 저해상도 단계에서만 Transformer를 적용하는 경향이 있어 고해상도에서의 전역 문맥 활용이 제한적이다.

**행렬 분해 모델**
NMF는 데이터 압축 및 해석 가능한 희소 요인(sparse factors) 추출 능력이 뛰어나 비지도 학습 기반의 뇌종양 분할에 사용된 바 있다. 최근 'Hamburger'와 같은 연구가 행렬 완성을 통해 전역 문맥을 모델링하려 시도했으나, 본 논문은 이를 U-shaped 아키텍처에 통합하고 HALS 솔버 및 다양한 Matricization 기법을 도입하여 차별점을 둔다.

## 🛠️ Methodology

### 전체 아키텍처
Factorizer는 전형적인 U-Net 스타일의 인코더-디코더 구조를 따른다. 입력 영상은 Stem 레이어를 거쳐 채널 수가 $C=32$로 확장되며, 이후 4단계의 스테이지를 통해 해상도가 $1/16$까지 감소했다가 다시 복원된다. 특히 디코더의 상위 3개 해상도 단계에서는 Deep Supervision을 적용하여 학습의 안정성을 높였다.

### Factorizer Block
Factorizer 블록은 ViT 블록의 Multi-head Self-attention 모듈을 **Wrapped NMF 모듈**로 대체한 구조이다. 구조는 다음과 같다:
$$Y = \text{WrappedNMF}(\text{LayerNorm}(X)) + X$$
$$Z = \text{MLP}(\text{LayerNorm}(Y)) + Y$$
여기서 MLP는 두 개의 Pointwise Convolution과 GELU 활성화 함수로 구성된다.

### Wrapped NMF Module
Wrapped NMF 모듈은 텐서를 행렬로 변환하여 저차원 근사를 수행한 뒤 다시 텐서로 되돌리는 과정을 거친다. 절차는 다음과 같다:
1. **Pointwise Conv**: 입력 특징을 선형 투영한다.
2. **Matricize**: 텐서를 행렬의 배치(batch of matrices)로 변환한다.
3. **ReLU**: 모든 요소를 양수로 제한하여 NMF 적용 조건을 충족시킨다.
4. **NMF**: 저차원 행렬 근사를 수행한다.
5. **Dematricize**: 근사된 행렬을 원래 텐서 크기로 복원한다.
6. **Pointwise Conv**: 최종 출력을 생성한다.

#### Matricize 연산의 종류
입력 데이터를 어떻게 행렬로 변환하느냐에 따라 세 가지 변형이 존재한다:
- **Global Matricize**: 공간 차원을 완전히 평탄화(flatten)하여 전역 문맥을 모델링한다. 지역성 편향(locality bias)이 없다.
- **Local Matricize**: 입력을 겹치지 않는 패치(patch) 그리드로 나누어 각 패치 내부의 상호작용을 모델링한다.
- **Shifted Window (SW) Matricize**: Local Matricize의 경계 부분 정보 손실을 막기 위해, 입력을 일정 오프셋만큼 이동(roll)시킨 뒤 다시 패치화하여 기존 패치와 결합한다. 이후 Dematricize 단계에서 이들의 평균을 취해 부드러운 특징 맵을 생성한다.

### Nonnegative Matrix Factorization (NMF)
NMF는 주어진 비음수 행렬 $\mathbf{X} \in \mathbb{R}^{M \times N}_{\ge 0}$를 두 개의 비음수 행렬 $\mathbf{F} \in \mathbb{R}^{M \times R}_{\ge 0}$와 $\mathbf{G} \in \mathbb{R}^{N \times R}_{\ge 0}$의 곱으로 근사하는 것이다:
$$\mathbf{X} \approx \mathbf{FG}^T$$
여기서 $R$은 랭크(rank)이며, 본 논문에서는 학습 시 $R=1$을 사용한다. 목적 함수는 제곱 오차를 최소화하는 것이다:
$$\text{minimize}_{\mathbf{F} \ge 0, \mathbf{G} \ge 0} \|\mathbf{X} - \mathbf{FG}^T\|_2^2$$

이 문제는 비볼록(non-convex) 문제이므로 BCD(Block Coordinate Descent) 기반의 반복적 알고리즘을 사용한다. 본 논문은 수렴 속도가 빠른 **HALS(Hierarchical Alternating Least Squares)** 솔버를 사용하며, 각 요인 행렬의 열을 다음과 같이 폐쇄형 해(closed-form solution)로 업데이트한다:
$$f_r^* \leftarrow \max(0, \frac{\mathbf{E}_r \mathbf{g}_r}{\|\mathbf{g}_r\|^2})$$
(여기서 $\mathbf{E}_r$은 잔차 행렬이다.) 계산 복잡도는 반복당 $O(MNR)$로, Self-attention의 $O(N^2)$보다 훨씬 효율적인 선형 확장성을 갖는다.

## 📊 Results

### 실험 설정
- **데이터셋**: BraTS (뇌종양 MRI), ISLES’22 (뇌졸중 병변 MRI).
- **평가 지표**: Dice score ($\uparrow$), Hausdorff Distance 95% (HD95, $\downarrow$).
- **비교 대상**: nnU-Net, Res-U-Net, Performer(선형 attention), UNETR, Swin UNETR, nnFormer 등.

### 정량적 결과
- **BraTS 데이터셋**: **Swin Factorizer**가 평균 Dice score $84.21\%$, HD95 $6.89\text{mm}$로 가장 우수한 성능을 보였다. 특히 ET(Enhancing Tumor)에서 $79.33\%$의 높은 Dice score를 기록했다. 주목할 점은 nnFormer보다 파라미터 수는 95% 이상 적고 FLOPs는 60% 이상 낮음에도 불구하고 성능은 더 높았다는 것이다.
- **ISLES’22 데이터셋**: Swin Factorizer가 Dice $76.49\%$, HD95 $11.96\text{mm}$로 SOTA를 달성하였으며, Transformer 기반 모델들과 nnU-Net을 유의미하게 능가하였다.

### 정성적 결과 및 해석 가능성
NMF의 성분(components)을 시각화한 결과, 각 성분이 실제 해부학적 의미(예: WT - Whole Tumor, TC - Tumor Core)를 가진 영역을 명확히 구분해내는 것이 확인되었다. 이는 내부 동작 과정을 알 수 없는 CNN이나 Transformer와 달리, Factorizer가 어떤 시각적 특징을 기반으로 판단하는지 설명할 수 있는 강력한 해석 가능성을 제공함을 시사한다.

## 🧠 Insights & Discussion

**강점 및 효율성**
Factorizer는 전역 문맥 모델링 능력을 갖추면서도 계산 효율성이 극도로 높다. 특히 실험을 통해 추론 단계에서 NMF의 반복 횟수 $T$를 줄이거나 특정 NMF 레이어를 제거(short-circuit)하더라도 정확도 손실이 적으면서 추론 속도를 획기적으로 높일 수 있음을 발견하였다. 이는 추가적인 재학습 없이도 모델의 경량화가 가능하다는 매우 독특한 장점이다.

**한계 및 가정**
본 연구에서는 랭크 $R=1$을 사용했는데, 이는 행렬이 매우 '뚱뚱한(fat)' 형태(열의 수가 행의 수보다 훨씬 많은 경우)일 때 1차 근사만으로도 충분하다는 가정에 기반한다. 하지만 모든 영역에 동일한 랭크를 적용하는 것이 최선인지에 대해서는 의문이 남으며, 향후 영역별 최적 랭크를 자동으로 선택하는 기법이 필요할 것으로 보인다.

**비판적 해석**
Swin Factorizer가 Global Factorizer보다 성능이 좋은 이유는 CNN처럼 초기 단계에서는 지역적 특징을 학습하고, 층이 깊어질수록 수용 영역(receptive field)을 넓혀가는 계층적 구조가 의료 영상의 특성에 더 적합하기 때문으로 해석된다. 또한, NMF 레이어의 미분 가능성을 위해 반복 횟수 $T$를 제한(예: $T=5$)해야 하는데, 이는 이론적 최적해와 실제 구현된 레이어의 출력 값 사이에 간극이 존재할 수 있음을 의미한다.

## 📌 TL;DR

본 논문은 Transformer의 고비용 Attention 메커니즘을 대체하기 위해 **미분 가능한 NMF(Nonnegative Matrix Factorization) 레이어**를 제안하였다. 이를 U-Net 구조에 통합하고 **Shifted Window Matricize** 기법을 적용함으로써, 계산 복잡도를 선형 수준으로 낮추면서도 전역적 문맥을 효과적으로 포착하는 **Factorizer** 모델을 구축하였다. 결과적으로 BraTS와 ISLES’22 데이터셋에서 SOTA 성능을 기록하였으며, 특히 NMF 성분을 통한 뛰어난 **해석 가능성**과 추론 단계에서의 **유연한 속도 조절 능력**을 입증하였다. 이 연구는 고해상도 3D 의료 영상 처리를 위한 효율적이고 투명한 딥러닝 아키텍처의 새로운 방향성을 제시한다.