# LViT: Language meets Vision Transformer in Medical Image Segmentation

Zihan Li, Yunxiang Li, Qingde Li, Puyang Wang, Dazhou Guo, Le Lu, Dakai Jin, You Zhang, Qingqi Hong (2023)

## 🧩 Problem to Solve

의료 영상 분할(Medical Image Segmentation)은 임상 진단에서 매우 중요한 작업이지만, 고품질의 레이블링된 데이터를 확보하는 데 드는 비용이 매우 높다는 고질적인 문제가 있다. 특히 조직 구조가 복잡하거나 COVID-19와 같은 새로운 질병의 경우, 전문가의 정밀한 어노테이션을 얻는 것이 매우 어렵다.

기존의 반지도 학습(Semi-supervised learning) 방식들은 레이블이 없는 데이터에 대해 의사 레이블(Pseudo label)을 생성하여 학습하지만, 이 의사 레이블의 품질이 낮을 경우 모델의 전체적인 성능이 저하되는 한계가 있다. 따라서 본 논문은 의료 영상과 함께 생성되는 의료 텍스트 기록(Medical notes)이라는 보완적 정보를 활용하여 영상 데이터의 품질 부족을 해결하고, 의사 레이블의 신뢰도를 높여 분할 성능을 향상시키는 것을 목표로 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 텍스트 정보를 통해 모델에 도메인 전문가의 지식을 주입함으로써, 영상 데이터만으로는 부족한 세밀한 가이드를 제공하는 것이다. 이를 위해 다음과 같은 설계를 제안한다.

1. **LViT 아키텍처**: CNN의 국소 특징 추출 능력과 Transformer의 전역 특징 및 교차 모달리티(Cross-modality) 처리 능력을 결합한 Double-U 구조를 제안한다.
2. **PLAM (Pixel-Level Attention Module)**: Transformer가 전역 특징에 치중하여 국소 특징을 놓치는 경향을 보완하기 위해, 채널 및 공간 주의 집중(Attention)을 통해 국소 특징을 보존하는 모듈을 설계하였다.
3. **EPI (Exponential Pseudo label Iteration)**: 지수 이동 평균(EMA) 방식을 도입하여 의사 레이블을 점진적으로 정제함으로써, 노이즈가 많은 레이블에 대한 강건성을 확보하였다.
4. **LV (Language-Vision) Loss**: 텍스트 정보와 예측된 의사 레이블 간의 코사인 유사도를 이용해 레이블이 없는 이미지의 학습을 직접적으로 감독하는 손실 함수를 제안하였다.

## 📎 Related Works

### 의료 영상 분할 및 반지도 학습

U-Net과 UNet++와 같은 CNN 기반 모델들이 널리 사용되어 왔으나, 데이터의 양과 질에 매우 민감하여 일반화 성능이 떨어진다는 단점이 있다. 이를 해결하기 위해 반지도 학습이 연구되었지만, 앞서 언급했듯이 의사 레이블의 신뢰도 문제가 주요 병목 구간으로 작용하고 있다.

### Vision-Language Model (VLM)

CLIP, ViLT, LAVT와 같은 VLM들이 자연어-이미지 정렬을 통해 성능을 높이고 있다. 그러나 자연 영상과 달리 의료 영상은 경계가 모호하고 그레이스케일 값의 차이가 매우 작아, 기존의 자연 영상 기반 VLM을 그대로 적용하기에는 무리가 있다. 특히 LAVT나 VLT는 명시적인 텍스트-이미지 정렬(Alignment)에 집중하는데, 의료 영상에서는 이러한 엄격한 정렬을 달성하기가 매우 어렵다.

## 🛠️ Methodology

### 전체 시스템 구조

LViT는 U-자형 CNN 브랜치와 U-자형 Transformer(ViT) 브랜치가 결합된 **Double-U 구조**를 가진다.

- **CNN 브랜치**: 이미지 입력을 받고 최종 분할 마스크를 출력하는 역할을 수행하며, 국소 특징(Local features)을 추출한다.
- **ViT 브랜치**: BERT-Embed를 통해 벡터화된 텍스트 특징과 CNN에서 추출된 이미지 특징을 융합하여 전역적 문맥(Global context)을 학습한다.

### 주요 구성 요소 및 작동 원리

#### 1. U-shape CNN & ViT Branch

CNN 브랜치의 DownCNN 모듈은 다음과 같이 정의된다.
$$D_i = \text{DownCNN}_i = \text{Relu}(\text{BN}_i(\text{Conv}_i(\cdot)))$$
$$Y_{\text{DownCNN},i+1} = \text{MaxPool}(D_i(Y_{\text{DownCNN},i}))$$

ViT 브랜치는 텍스트 벡터와 이미지 패치 임베딩을 융합하여 처리한다. 첫 번째 레이어의 융합 과정은 다음과 같다.
$$Y_{\text{DownViT},1} = \text{ViT}(x_{\text{img},1} + \text{CTBN}(x_{\text{text}}))$$
여기서 $\text{CTBN}$은 텍스트와 이미지 특징의 차원을 맞추기 위한 블록이며, 이후 레이어에서는 이전 ViT 레이어의 출력과 현재 CNN 레이어의 특징을 더해 처리한다.

#### 2. Pixel-Level Attention Module (PLAM)

PLAM은 Transformer의 전역 특징 편향을 억제하고 국소 특징을 보존하기 위해 설계되었다. Global Average Pooling(GAP)과 Global Max Pooling(GMP)을 병렬로 사용하여 채널 및 공간 주의 집중을 수행하며, 이를 통해 CNN 레이어가 더 강력한 국소 표현을 생성하도록 돕는다.

#### 3. Exponential Pseudo label Iteration (EPI)

반지도 학습 설정에서 의사 레이블 $P$를 다음과 같이 업데이트하여 급격한 품질 저하를 방지한다.
$$P_t = \beta \cdot P_{t-1} + (1-\beta) \cdot P_t$$
여기서 $\beta$는 모멘텀 파라미터(0.99)이다. 이는 모델 가중치가 최적점 주변에서 진동할 때, 의사 레이블 역시 마스크 주변에서 진동하게 하여 최종적으로 신뢰도 높은 레이블을 얻게 하는 원리이다.

#### 4. LV (Language-Vision) Loss

텍스트 간의 코사인 유사도($\text{TextSim}$)를 통해 가장 유사한 대비 레이블(Contrastive label)을 찾고, 예측된 의사 레이블($x_{\text{img},p}$)과 대비 레이블($x_{\text{img},c}$) 간의 유사도를 계산한다.
$$\text{ImgSim} = \frac{x_{\text{img},p} \cdot x_{\text{img},c}}{|x_{\text{img},p}| \times |x_{\text{img},c}|}$$
$$\mathcal{L}_{\text{LV}} = 1 - \text{ImgSim}$$
이 손실 함수는 레이블이 없는 데이터에 대해서만 적용되며, 완전히 잘못된 위치에 분할이 발생하는 것을 방지하는 가이드 역할을 한다.

## 📊 Results

### 실험 설정

- **데이터셋**: MosMedData+ (폐 감염 CT), QaTa-COV19 (COVID-19 X-ray), ESO-CT (식도암 CT)
- **지표**: Dice Score, mIoU
- **비교 대상**: U-Net, UNet++, nnUNet, TransUNet, Swin-Unet, GLoRIA, LAVT 등

### 정량적 결과

QaTa-COV19 데이터셋에서 LViT-T는 Dice score 83.66%, mIoU 75.11%를 달성하여 기존 SOTA 모델인 nnUNet보다 우수한 성능을 보였다. 특히, **훈련 데이터 레이블의 1/4만 사용했을 때(LViT-T 1/4)에도 완전 지도 학습 기반의 기존 모델들과 대등하거나 더 높은 성능**을 보였다는 점이 주목할 만하다.

| Dataset | Metric | LViT-T (Full) | LViT-T (1/4 Labels) | nnUNet (Full) |
| :--- | :--- | :---: | :---: | :---: |
| QaTa-COV19 | Dice (%) | **83.66** | 80.95 | 80.42 |
| MosMedData+ | Dice (%) | **74.57** | 72.48 | 72.59 |

### 정성적 및 일반화 결과

- **시각적 분석**: Grad-CAM을 통한 해석 가능성 연구 결과, LViT는 텍스트 정보를 통해 병변 영역을 더 정확하게 활성화(Activation)하는 것으로 나타났다. 특히 텍스트가 없는 LViT-TW보다 LViT-T가 병변 위치를 훨씬 더 잘 포착하였다.
- **일반화 연구**: 자체 구축한 ESO-CT 데이터셋에서도 LViT-T가 다른 모델들을 큰 차이로 앞서며, 텍스트 기반의 대략적인 위치 정보가 실제 분할 성능 향상에 기여함을 입증하였다.

## 🧠 Insights & Discussion

### 강점 및 분석

본 논문은 단순한 모델 구조의 변경이 아니라, 의료 현장에서 쉽게 얻을 수 있는 '텍스트'라는 보조 정보를 어떻게 효과적으로 분할 작업에 녹여낼 것인가에 대한 해답을 제시했다. 특히, 무거운 Text Encoder 대신 Embedding Layer를 사용하여 파라미터 수를 줄이면서도 성능을 유지한 점이 효율적이다. 또한, EPI와 LV Loss를 통해 반지도 학습의 고질적인 문제인 의사 레이블의 불안정성을 성공적으로 해결하였다.

### 한계 및 비판적 해석

- **텍스트 의존성**: 추론 단계에서 텍스트 입력이 반드시 필요하다는 점은 실제 적용 시 제약 사항이 될 수 있다. (저자들 또한 이를 인지하고 향후 자동 텍스트 생성 기능을 추가하겠다고 언급함)
- **2D 기반**: 현재 모델은 2D 슬라이스 기반으로 작동한다. 의료 영상의 특성상 3D 볼륨 정보가 중요하므로, 3D로의 확장이 필수적이다.
- **텍스트 구조**: 실험에서 구조화된 텍스트(Structured text)가 비구조화된 텍스트보다 성능이 좋음을 보였는데, 이는 실제 의료 현장의 다양하고 정제되지 않은 리포트를 처리할 때 성능 저하가 발생할 가능성을 시사한다.

## 📌 TL;DR

LViT는 의료 영상의 레이블 부족 문제를 해결하기 위해 **텍스트 정보를 결합한 Double-U 구조의 분할 모델**이다. CNN-ViT 하이브리드 구조와 국소 특징 보존을 위한 PLAM, 의사 레이블 정제를 위한 EPI, 그리고 텍스트 기반의 LV Loss를 통해 **데이터 효율성을 극대화**하였다. 실험 결과, 적은 양의 레이블(25%)만으로도 기존 SOTA 모델의 성능을 능가하였으며, 이는 의료 텍스트가 병변의 위치를 가이드하는 강력한 보조 지표가 될 수 있음을 시사한다.
