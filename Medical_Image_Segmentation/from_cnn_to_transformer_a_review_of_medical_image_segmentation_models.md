# From CNN to Transformer: A Review of Medical Image Segmentation Models

Wenjian Yao, Jiajun Bai, Wei Liao, Yuheng Chen, Mengjuan Liu, Yao Xie (2023)

## 🧩 Problem to Solve

본 논문은 의료 영상 분석의 핵심 단계인 의료 영상 분할(Medical Image Segmentation) 모델들의 발전 과정을 분석하고 그 성능을 정량적으로 평가하는 것을 목표로 한다. 의료 영상 분할은 CT나 MRI와 같은 영상에서 장기나 병변의 픽셀을 식별하는 작업으로, 질병의 진단과 치료 계획 수립에 필수적이다.

기존의 전통적인 분할 방법(임계값 설정, 엣지 검출 등)은 수동 특징 추출에 의존하여 효율성과 정확도가 떨어지며, 의료진의 주관적 판단에 영향을 받는 한계가 있었다. 이를 해결하기 위해 딥러닝, 특히 Convolutional Neural Networks(CNN) 기반의 모델들이 도입되었으나, CNN은 이미지 내의 장거리 의존성(Long-range dependencies)을 모델링하는 능력이 부족하여 전역적인 문맥 정보를 충분히 활용하지 못하는 문제가 존재한다.

따라서 본 연구는 CNN 기반의 대표적 모델인 U-Net과 UNet++, 그리고 이를 개선하여 Transformer 구조를 도입한 TransUNet과 Swin-Unet의 이론적 특성을 분석하고, 두 가지 벤치마크 데이터셋을 통해 이들의 성능을 직접 비교함으로써 연구자들이 특정 영역에 적합한 분할 모델을 신속하게 구축할 수 있도록 돕고자 한다.

## ✨ Key Contributions

본 논문의 핵심 기여는 최근 의료 영상 분할 분야에서 가장 영향력 있는 네 가지 모델(U-Net, UNet++, TransUNet, Swin-Unet)을 선정하여 이론적 분석과 정량적 평가를 동시에 수행했다는 점이다.

단순한 구조 설명에 그치지 않고, CNN에서 Transformer로의 아키텍처 변화가 실제 의료 영상의 특징 추출 및 전역 정보 활용 능력에 어떠한 영향을 미치는지 분석하였다. 특히, 폐 결핵 흉부 X-ray 데이터셋과 난소 종양 CT 데이터셋이라는 서로 다른 특성을 가진 두 데이터셋을 활용하여, 작업의 난이도에 따른 모델별 성능 차이를 명확히 제시하였다.

## 📎 Related Works

논문에서는 의료 영상 분할의 흐름을 전통적 방법에서 딥러닝 기반 방법으로 구분하여 설명한다.

1. **전통적 방법:** Thresholding, Edge Detection, Morphological Operations 등이 사용되었으며, 해석 가능성은 높으나 복잡하고 다양한 의료 영상의 특성을 반영하지 못해 정확도와 효율성이 낮다는 한계가 있다.
2. **CNN 기반 모델:** Fully Convolutional Networks(FCN)를 기반으로 한 U-Net과 SegNet 등이 등장하며 비약적인 발전을 이루었다. 특히 U-Net은 대칭적 구조와 Skip Connection을 통해 세밀한 정보를 보존하는 능력을 보여주었다.
3. **Transformer 기반 모델:** NLP에서 성공한 Transformer를 비전 분야에 적용한 TransUNet, Swin-Unet 등이 제안되었다. 이들은 Self-attention 메커니즘을 통해 CNN의 한계인 장거리 의존성 문제를 해결하고자 하였다.

기존의 리뷰 논문들이 주로 CNN 기반 모델만을 다루거나(문헌 [12], [13]), 정량적 평가 없이 구조적 분석에만 치중했다는 점([14])을 지적하며, 본 논문은 최신 Transformer 기반 모델을 포함한 정량적 비교 분석을 제공함으로써 차별성을 갖는다.

## 🛠️ Methodology

### 1. 분석 대상 모델 아키텍처

본 논문에서 분석한 네 가지 모델의 상세 구조와 특성은 다음과 같다.

* **U-Net:** 대칭적인 인코더(Contracting path)와 디코더(Expanding path) 구조를 가진다. 인코더에서 추출된 저수준 특징 맵을 디코더의 상응하는 층으로 직접 전달하는 Skip Connection을 사용하여, 업샘플링 과정에서 손실될 수 있는 세부적인 위치 정보를 보존한다.
* **UNet++:** U-Net의 단순한 Skip Connection을 Dense Connection 형태로 확장하였다. 인코더와 디코더 사이에 더 많은 짧은 연결 경로와 업샘플링 컨볼루션 블록을 추가하여 새로운 인코더 레벨을 형성함으로써, 서로 다른 스케일의 특징을 더 효과적으로 융합한다. 이는 기울기 소실(Gradient vanishing) 문제를 완화하고 세그멘테이션 정확도를 높인다.
* **TransUNet:** CNN과 Transformer가 결합된 하이브리드 인코더 구조를 가진다. 먼저 CNN을 통해 특징 맵을 추출한 뒤, 이를 패치(Patch) 단위로 나누어 12개의 Transformer 모듈에 입력한다. Multi-head Self-attention 메커니즘을 통해 이미지 전체의 전역적 문맥 정보를 캡처하며, 디코더에서는 다시 CNN 특징 맵과 결합하여 정밀한 위치 기반 분할을 수행한다.
* **Swin-Unet:** 순수 Transformer 기반의 U자형 구조를 가진다. Swin Transformer 블록을 사용하여 계층적 특징을 추출하며, 특히 Shifted Window 메커니즘을 도입하여 계산 복잡도를 줄이면서도 인접 영역 간의 상호작용을 효과적으로 모델링한다.

### 2. 학습 절차 및 손실 함수

모델 학습을 위해 Binary Cross-Entropy(BCE) 손실 함수와 Dice Loss를 함께 사용하였다.

* **Binary Cross-Entropy Loss:** 픽셀 단위의 이진 분류 성능을 측정하며, 수식은 다음과 같다.
    $$L_{BCE} = -[(1-y)\log(1-\hat{y}) + y\log\hat{y}]$$
    여기서 $y$는 실제 라벨, $\hat{y}$는 모델의 예측값이다.
* **Dice Loss:** 예측 영역과 실제 영역의 겹침 정도(Overlap)를 측정하여 클래스 불균형 문제를 완화한다.
    $$L_{DSC} = 1 - \frac{2TP}{2TP + FP + FN}$$
    ($TP$: True Positive, $FP$: False Positive, $FN$: False Negative)

### 3. 구현 세부 사항

* **Optimizer:** U-Net 및 UNet++는 Adam을 사용하였고, TransUNet과 Swin-Unet은 SGD(Momentum 0.9)를 사용하였다.
* **입력 크기:** U-Net, UNet++, TransUNet은 $512 \times 512$ 크기를 사용하였으나, Swin-Unet은 $224 \times 224$ 크기를 사용하였다.
* **초기화:** Transformer 기반 모델들은 ImageNet으로 사전 학습된(Pre-trained) 가중치를 사용하여 초기화하였다.

## 📊 Results

### 1. 실험 설정

* **데이터셋:**
  * **Tuberculosis Chest X-rays:** 흉부 X-ray 영상 566장 (학습 452, 테스트 114). 폐 영역 분할을 목표로 한다.
  * **Ovarian Tumors:** 난소 종양 CT 영상 4,050장 (학습 3,092, 테스트 958). 종양 부위 분할을 목표로 하며, 영상의 크기와 형태가 매우 다양하여 난이도가 높다.
* **평가 지표:** Dice Coefficient (DSC), 95% Hausdorff Distance (HD95), Intersection over Union (IoU), Accuracy (Acc), Precision, Recall을 사용하였다.

### 2. 정량적 결과 분석

**A. Tuberculosis Chest X-rays (상대적으로 쉬운 작업)**
모든 모델이 $mIoU > 91\%$ 이상의 우수한 성능을 보였다.

* **최고 성능:** TransUNet ($\text{DSC: } 96.45\%, \text{HD95: } 10.75$)
* **성능 순위:** TransUNet $\approx$ Swin-Unet $\approx$ UNet++ $\approx$ U-Net (모델 간 차이가 크지 않음)

**B. Ovarian Tumors (상대적으로 어려운 작업)**
데이터셋의 복잡성으로 인해 모델 간 성능 차이가 뚜렷하게 나타났다.

* **최고 성능:** TransUNet ($\text{DSC: } 89.18\%, \text{IoU: } 82.73\%, \text{Acc: } 99.02\%$)
* **성능 순위:** TransUNet $\gg$ Swin-Unet $>$ U-Net $>$ UNet++
* 특히 TransUNet은 Precision 면에서 타 모델보다 월등히 높은 성능을 보여, 종양 영역을 매우 정확하게 식별함을 입증하였다.

**C. 오류 분석 (Dice < 20% 샘플 수)**
완전히 잘못된 예측을 수행한 샘플 수를 집계한 결과, Transformer 기반 모델(TransUNet, Swin-Unet)이 CNN 기반 모델(U-Net, UNet++)보다 훨씬 적은 수의 실패 사례를 보였다. 이는 전역 정보를 캡처하는 능력이 모델의 안정성을 높였음을 시사한다.

## 🧠 Insights & Discussion

### 1. 모델 구조의 영향력

U-Net의 U자형 구조는 의료 영상 분할의 표준으로서 매우 강력한 성능을 제공하며, 확장성이 뛰어나다. 하지만 CNN 단독으로는 전역적인 문맥 정보를 파악하는 데 한계가 있다.

### 2. Transformer의 역할과 한계

Transformer는 전역 정보를 캡처하여 분할의 안정성을 높이지만, 이미지 내의 세밀한 위치 정보(Localization)를 파악하는 능력은 부족하다.

* **TransUNet**이 가장 좋은 성능을 보인 이유는 CNN의 국소적 특징 추출 능력과 Transformer의 전역적 문맥 파악 능력을 결합한 하이브리드 구조 덕분이다.
* 반면, 순수 Transformer 기반인 **Swin-Unet**은 TransUNet보다 낮은 성능을 보였는데, 이는 CNN의 보조 없이는 정밀한 위치 정보를 복원하는 것이 어렵다는 점을 시사한다.

### 3. 데이터셋 특성에 따른 난이도

폐 영역 분할은 배경과의 대비가 뚜렷하여 모든 모델이 잘 수행하지만, 난소 종양 분할은 크기, 모양, 질감이 매우 다양하여 모델의 표현력(Representation capacity)이 결과에 큰 영향을 미친다.

### 4. 일반적인 한계와 과제

의료 영상 분야의 고질적인 문제인 **데이터 라벨링의 어려움**(전문 의료진 필요)과 **데이터 불균형**(양성/악성 비율 차이)이 언급되었다. 이를 해결하기 위해 Data Augmentation(회전, 반전, 색상 변형)과 전이 학습(Transfer Learning)이 대안으로 제시된다.

## 📌 TL;DR

본 논문은 U-Net, UNet++, TransUNet, Swin-Unet 네 가지 모델을 의료 영상 분할 작업에서 비교 분석하였다. 실험 결과, CNN과 Transformer를 결합한 **TransUNet**이 전역적 문맥 파악과 정밀한 위치 식별 능력을 동시에 갖추어 가장 우수한 성능을 보였다. 특히 난이도가 높은 데이터셋일수록 Transformer의 전역 정보 활용 능력이 모델의 안정성과 정확도를 결정짓는 핵심 요소임을 확인하였다. 향후 연구로는 SAM(Segment Anything Model)과 같은 거대 파운데이션 모델을 의료 분야에 적용하는 MedSAM과 같은 제로샷(Zero-shot) 전이 학습 방향이 유망할 것으로 전망한다.
