# CLIP-Mamba: CLIP Pretrained Mamba Models with OOD and Hessian Evaluation

Weiquan Huang, Yifei Shen, Yifan Yang (2024)

## 🧩 Problem to Solve

본 논문은 최근 다양한 도메인에서 뛰어난 성능을 보이고 있는 State Space Model(SSM) 기반의 Mamba 아키텍처를 Foundation Model 수준으로 확장하여, 전이 가능성(transferability)과 일반화 능력을 확보하는 것을 목표로 한다. 

기존의 Mamba 기반 모델들은 주로 미리 정의된 고정된 객체 범주(fixed array of predetermined object categories)에 대해 학습되었으며, 이로 인해 새로운 데이터셋에 대해 추가 학습 없이 성능을 내는 Zero-shot generalization 능력이 부족하다는 한계가 있었다. 따라서 본 연구는 Contrastive Language-Image Pretraining(CLIP) 프레임워크를 Mamba 모델에 적용하여, 텍스트와 이미지 간의 정렬을 통해 범용적인 시각 표현을 학습하고 이를 통해 Zero-shot 성능을 확보하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 Mamba 아키텍처를 활용한 최초의 CLIP 사전 학습 모델을 제안하고, 이를 Vision Transformer(ViT)와 다각도로 비교 분석한 점이다. 주요 기여 사항은 다음과 같다.

- **CLIP-Mamba 모델의 구현 및 공개**: 다양한 크기의 Mamba 모델을 CLIP 방식으로 학습시켜 공개하였다.
- **파라미터 효율성 입증**: 50M 파라미터의 Mamba 모델이 84M 파라미터의 ViT 모델보다 우수한 성능을 보이며, 특히 67M 파라미터 모델이 307M 파라미터의 ViT-L 모델과 대등한 Zero-shot 성능을 달성함으로써 Mamba의 압도적인 파라미터 효율성을 증명하였다.
- **OOD 일반화 능력 분석**: 16개의 Out-of-Distribution(OOD) 데이터셋 평가를 통해 Mamba 모델이 ViT보다 강건함을 보였으며, 특히 이미지 대비(contrast) 변화나 High-pass filtering 환경에서 탁월한 성능을 보임을 확인하였다.
- **학습 랜드스케이프 분석**: Hessian 분석을 통해 Mamba 모델의 손실 함수 지형(loss landscape)이 ViT보다 더 날카롭고(sharper) 비볼록(non-convex)함을 밝혀내어, 최적화의 난이도가 더 높음을 시사하였다.

## 📎 Related Works

기존의 Foundation Model들은 주로 Transformer 아키텍처를 기반으로 하며, 그 핵심인 Self-attention 메커니즘은 모든 토큰 쌍 간의 정보 흐름을 가능하게 하여 In-context learning과 OOD 강건성을 높이는 데 기여하였다. 그러나 Self-attention은 연산 복잡도가 시퀀스 길이의 제곱에 비례하는 $O(N^2)$의 특성을 가지므로, 윈도우 길이가 길어질 때 확장성(scalability) 문제가 발생한다. 이를 해결하기 위해 sub-quadratic 시간 복잡도를 가진 효율적인 attention 연구들이 진행되었으나, 일반적으로 표준 Transformer보다 성능이 낮다는 단점이 있었다.

반면, Selective State Space Models(Mamba)는 선형 시간 복잡도($O(N)$)를 가지면서도 Transformer보다 우수한 Scaling law를 보여 차세대 백본 아키텍처로 주목받고 있다. 최근 이미지 분류, 객체 검출, 세그멘테이션 등 다양한 컴퓨터 비전 작업에서 Mamba 기반 모델들이 SOTA 성능을 기록하고 있으나, 앞서 언급한 바와 같이 Zero-shot 일반화 능력을 갖춘 대규모 언어-이미지 사전 학습 연구는 부족한 상태였다.

## 🛠️ Methodology

### 모델 구조 및 학습 파이프라인
본 연구에서는 기존에 제안된 Mamba 기반 아키텍처인 VMamba(30M, 50M, 89M)와 Simba-L(66.6M)을 백본으로 사용하였다. 학습 방식은 표준 CLIP 사전 학습 파이프라인을 그대로 따르며, 텍스트 인코더와 이미지 인코더(Mamba) 간의 대조 학습(contrastive learning)을 통해 공통의 임베딩 공간을 학습한다.

### 평가 방법론
1. **Zero-shot Classification**: 26개의 다양한 데이터셋을 사용하여 추가 미세 조정 없이 모델의 전이 능력을 측정하였다.
2. **OOD Robustness**: Geirhos et al. (2021)의 방법론을 따라 16개의 OOD 데이터셋에서 성능을 평가하였다. 여기에는 Silhouette, Stylized, Sketch, Edge, Color, Contrast, High-pass/Low-pass filter 등이 포함된다.
3. **Hessian Analysis**: 모델의 학습 지형을 분석하기 위해 3,000개의 샘플과 배치 사이즈 15를 사용하여 top-5 Hessian eigenvalue spectra를 계산하였다.

### 주요 이론적 배경 (Frequency Bias)
논문은 Mamba 모델이 ViT나 인간의 시각 체계보다 High-pass filter 환경에서 강건한 이유를 다음과 같이 설명한다. ViT와 인간의 시각은 저주파(low-frequency) 성분에 편향된 경향이 있어 저주파 성분이 제거된 환경에서 성능이 급격히 저하된다. 그러나 Mamba(SSM)의 hidden states는 직교 다항식(orthogonal polynomials)의 계수로 표현되므로, 이러한 주파수 편향이 ViT에 비해 덜 나타난다.

## 📊 Results

### Zero-shot 성능
- **파라미터 효율성**: VMamba-S (50M) 모델이 ViT-B (84M) 모델보다 대부분의 데이터셋에서 우수한 성능을 보였다.
- **최상위 성능 비교**: Simba-L (66.6M) 모델은 일부 데이터셋에서 ViT-L (307M)과 대등하거나 오히려 능가하는 결과를 보여, 매우 적은 파라미터로도 거대 모델의 성능을 낼 수 있음을 입증하였다.

### OOD 일반화 및 강건성
- **전반적 성능**: 16개의 OOD 데이터셋 전반에서 Mamba 기반 모델들이 ViT 모델들을 일관되게 상회하였다.
- **Shape Bias**: Mamba 모델은 텍스처보다 객체의 형태(shape)를 인식하려는 경향이 강하며, 이는 인간의 시각 인식 방식과 더 유사한 특성이다.
- **특이 사항**: 특히 High-pass filter가 적용된 이미지나 대비가 극심한 조건에서 Mamba는 ViT뿐만 아니라 인간의 성능까지 능가하는 결과를 보였다.

### Hessian 분석 결과
- **비볼록성(Non-convexity)**: VMamba 모델은 ViT 모델에 비해 음의 고유값(negative eigenvalues)의 개수가 더 많았으며, 이는 손실 지형이 더 비볼록함을 의미한다.
- **날카로움(Sharpness)**: 큰 크기의 고유값이 더 많이 관찰되었으며, 이는 Mamba의 손실 지형이 ViT보다 더 날카롭다는 것을 의미한다. 결과적으로 Mamba 모델이 ViT보다 최적화(optimization) 과정에서 더 많은 어려움이 있을 수 있음을 시사한다.

## 🧠 Insights & Discussion

본 연구는 Mamba 아키텍처가 단순한 연산 효율성을 넘어, 시각적 표현 학습 관점에서 ViT와는 다른 독특한 이점을 가지고 있음을 보여준다. 특히 인간과 유사한 Shape bias를 보이고 주파수 편향이 적다는 점은, Mamba가 더 본질적인 객체의 특징을 학습할 가능성이 높음을 시사한다.

하지만 Hessian 분석을 통해 드러난 '날카롭고 비볼록한' 학습 지형은 Mamba 모델의 잠재적 약점이다. 이는 모델의 크기가 커지거나 학습 데이터가 복잡해질 때 수렴 안정성이 떨어질 수 있음을 의미하며, 향후 Mamba 기반 Foundation Model을 개발하기 위해서는 더 정교한 최적화 기법이나 정규화 방법이 필요할 것으로 판단된다.

## 📌 TL;DR

본 논문은 최초로 CLIP 프레임워크를 Mamba 모델에 적용하여, **매우 적은 파라미터(약 67M)로도 거대 ViT 모델(307M) 수준의 Zero-shot 성능을 달성**할 수 있음을 보여주었다. 또한 Mamba 모델이 인간과 유사한 Shape bias를 가지며 OOD 환경(특히 고주파 성분 환경)에서 ViT보다 월등히 강건함을 입증하였다. 다만, 손실 지형이 더 날카롭고 비볼록하여 학습 최적화가 더 어렵다는 분석 결과가 함께 제시되었다. 이 연구는 향후 효율적인 시각 Foundation Model 설계에 중요한 지표가 될 것이다.