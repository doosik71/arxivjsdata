# VIS-MAE: An Efficient Self-supervised Learning Approach on Medical Image Segmentation and Classification

Zelong Liu, Andrew Tieu, Nikhil Patel, George Soultanidis, Louisa Deyer, Ying Wang, Sean Huver, Alexander Zhou, Yunhao Mei, Zahi A. Fayad, Timothy Deyer, and Xueyan Mei (2024)

## 🧩 Problem to Solve

본 연구가 해결하고자 하는 핵심 문제는 의료 영상 분석 AI 모델 개발 시 직면하는 **데이터 가용성의 한계와 일반화 능력의 부족**이다. 의료 영상 분야에서 고성능 모델을 구축하기 위해서는 대량의 정밀한 레이블링 데이터가 필요하지만, 전문의의 수작업에 의존하는 레이블링 과정은 비용과 시간이 매우 많이 소요된다. 또한, 특정 질환이나 특정 모달리티(Modality)에 최적화된 기존의 지도 학습(Supervised Learning) 모델들은 다른 작업이나 데이터셋에 적용했을 때 성능이 크게 저하되는 일반화 문제(Generalization problem)를 겪는다.

따라서 본 논문의 목표는 대규모의 레이블 없는(Unlabeled) 의료 영상 데이터를 활용하여 다양한 하위 작업(Downstream tasks)에 유연하게 적응할 수 있는 **기반 모델(Foundation Model)인 VIS-MAE**를 구축함으로써, 레이블 데이터에 대한 의존도를 낮추고 세그멘테이션(Segmentation) 및 분류(Classification) 성능을 동시에 향상시키는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **Swin Transformer 기반의 Masked AutoEncoder(MAE)를 의료 영상의 특성에 맞게 최적화하여 대규모 다중 모달리티 데이터셋으로 사전 학습(Pre-training)시키는 것**이다. 

가장 중점적인 설계 아이디어는 다음과 같다:
1. **대규모 다중 모달리티 사전 학습**: CT, MR, PET, X-ray, Ultrasound 등 5가지 서로 다른 모달리티에서 수집된 250만 장의 방대한 무레이블 데이터를 사용하여 범용적인 특징 추출 능력을 확보하였다.
2. **계층적 가중치 전략**: 모든 데이터를 통합 학습한 `VIS-MAE-Generic` 모델과 각 모달리티별 특성을 극대화한 `VIS-MAE-Modality` 모델을 동시에 개발하여, 범용성과 특수성을 모두 확보하고자 하였다.
3. **의료 영상 맞춤형 마스킹 전략**: 기존 MAE의 단순한 패치 마스킹이 의료 영상의 단순한 구조로 인해 너무 쉽게 복원되는 문제를 해결하기 위해, 최소 $16 \times 16$ 픽셀 크기의 윈도우 기반 마스킹 방식을 도입하여 모델이 더 복잡한 문맥적 특징을 학습하도록 강제하였다.

## 📎 Related Works

논문에서는 다음과 같은 관련 연구들을 언급하며 VIS-MAE의 차별점을 제시한다:
- **기반 모델(Foundation Models)**: SAM(Segment Anything Model)과 MedSAM과 같은 모델들이 대규모 데이터셋을 통해 범용 세그멘테이션 가능성을 보여주었으며, RETFound는 망막 영상에 특화된 기반 모델의 가능성을 제시하였다.
- **자기지도 학습(Self-supervised Learning, SSL)**: SimCLR와 같은 대조 학습(Contrastive Learning) 방식이 의료 영상 분류에서 효과적임이 입증되었다. 하지만 SimCLR는 주로 인코더 가중치만을 제공하는 반면, MAE(Masked AutoEncoder)는 인코더와 디코더 가중치를 모두 제공하므로 세그멘테이션과 분류 작업 모두에 더 유리하다는 점을 강조한다.
- **Swin MAE**: Swin Transformer를 백본으로 사용하는 MAE가 작은 데이터셋에서도 효율적임을 보여주었으며, 본 연구는 이를 의료 영상의 대규모 데이터셋으로 확장하여 적용하였다.

## 🛠️ Methodology

### 전체 파이프라인 및 아키텍처
VIS-MAE는 **Swin Transformer**를 백본으로 하는 인코더-디코더 구조를 가진다. 사전 학습 단계에서는 이미지의 일부를 마스킹하고 이를 복원하는 Pretext task를 수행하며, 이후 하위 작업에 따라 디코더를 수정하여 사용한다.
- **세그멘테이션 작업**: 인코더와 디코더 사이에 Skip connection 레이어를 추가하여 고해상도 특징을 보존한다.
- **분류 작업**: 디코더 부분을 제거하고 분류를 위한 Classification head 레이어를 추가한다.

### 주요 구성 요소 및 연산
백본으로 사용된 Swin Transformer 블록은 LayerNorm(LN), Window-based Multi-head Self-Attention(W-MSA), 그리고 MLP 레이어로 구성된다. 각 블록의 연산 과정은 다음과 같은 방정식으로 표현된다:

$$\hat{z}_l = \text{W-MSA}(\text{LN}(z_{l-1})) + z_{l-1}$$
$$z_l = \text{MLP}(\text{LN}(\hat{z}_l)) + \hat{z}_l$$

여기서 $z_l$은 $l$번째 블록의 출력값이며, 잔차 연결(Residual connection)을 통해 그래디언트 소실 문제를 방지하고 깊은 네트워크 학습을 가능하게 한다.

### 학습 절차 및 손실 함수
1. **마스킹 전략**: 이미지의 최대 75%까지 마스킹하며, 최소 마스크 크기를 $16 \times 16$ 픽셀로 설정하여 모델이 단순 픽셀 보간이 아닌 의미론적 복원을 수행하게 한다.
2. **손실 함수**: 원본 이미지 패치 $X$와 모델이 복원한 이미지 패치 $\hat{X}$ 사이의 픽셀 단위 복원 오차를 최소화하는 Squared $L_2$ norm(MSE)을 사용한다.
$$\text{Loss} = \| X - \hat{X} \|_2^2$$
3. **훈련 설정**: AdamW 옵티마이저를 사용하며, 800 epoch 동안 학습을 진행하였다.

## 📊 Results

### 실험 설정
- **사전 학습 데이터**: 2005~2022년 사이 수집된 2,486,425장의 이미지 (MR: 1.2M, CT: 0.57M, X-ray: 0.44M, US: 0.21M, PET/CT: 0.06M).
- **평가 작업**: 8개의 세그멘테이션 데이터셋(BTCV, ACDC, AMOS, Glioma, Prostate, TUCC, BUSI, ISIC)과 6개의 분류 데이터셋(COVID-19, Sarcoidosis, ACL tear, Knee OA, BUSI, NIH Chest X-ray)을 사용하였다.
- **비교 대상**: nnU-Net, TransUNet, RadImageNet, ImageNet-1k pre-trained weights, SimCLR.
- **측정 지표**: 세그멘테이션은 DSC(Dice Similarity Coefficient), Precision, Recall을 사용하였고, 분류는 ROC AUC 및 PR AUC를 사용하였다.

### 주요 결과
- **성능 우위**: VIS-MAE는 대부분의 세그멘테이션 및 분류 작업에서 기존 벤치마크 모델들과 비교하여 동등하거나 더 우수한 성능을 보였다. 특히 모달리티 특화 모델인 `VIS-MAE-Modality`가 특정 작업에서 강세를 보였다.
- **레이블 효율성(Label Efficiency)**: 본 연구의 가장 중요한 결과 중 하나로, 학습 데이터의 양을 5%, 10%, 25%, 50%, 80%로 제한하여 실험했을 때, VIS-MAE는 다른 모델들보다 훨씬 적은 양의 데이터만으로도 유사하거나 더 높은 성능에 도달하였다. 이는 사전 학습된 가중치가 하위 작업에서 매우 강력한 초기값 역할을 함을 시사한다.

## 🧠 Insights & Discussion

### 강점 및 가치
VIS-MAE는 의료 영상의 도메인 특성을 반영한 대규모 SSL 학습을 통해 **레이블 효율성**을 획기적으로 높였다. 이는 전문의의 레이블링 비용이 매우 높은 의료 분야에서 실질적인 이점을 제공한다. 또한, 단일 모달리티가 아닌 다중 모달리티 데이터를 학습함으로써 범용적인 의료 영상 특징 추출기로서의 가능성을 입증하였다.

### 한계 및 논의사항
- **계산 자원**: 8대의 NVIDIA DGX A100 GPU를 사용하여 최대 516시간까지 학습시키는 등 매우 높은 계산 비용이 소요되었다.
- **데이터 편향**: 데이터가 특정 의료 기관(RadImageNet LLC)에서 수집되었으므로, 다른 기관의 데이터에 대해서도 동일한 일반화 성능이 나타날지에 대한 추가 검증이 필요하다.
- **마스킹 비율**: 75%라는 높은 마스킹 비율이 모든 의료 영상 작업에서 최적인지에 대한 세부 분석은 본문에 명시되지 않았다.

## 📌 TL;DR

본 논문은 250만 장의 다중 모달리티 의료 영상을 활용하여 Swin Transformer 기반의 Masked AutoEncoder로 사전 학습된 기반 모델 **VIS-MAE**를 제안한다. 이 모델은 의료 영상의 특성에 맞는 윈도우 마스킹 전략을 도입하였으며, 실험 결과 세그멘테이션과 분류 작업 모두에서 뛰어난 성능과 높은 레이블 효율성을 보였다. 특히 적은 양의 레이블 데이터만으로도 고성능 모델을 구축할 수 있게 함으로써 의료 AI의 실용적인 배포 가능성을 높였다.