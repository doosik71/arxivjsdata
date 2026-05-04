# STPNet: Scale-aware Text Prompt Network for Medical Image Segmentation

Dandan Shan, Zihan Li, Yunxiang Li, Qingde Li, Jie Tian, Qingqi Hong (2021/2025)

## 🧩 Problem to Solve

의료 영상 분석 및 진단에서 병변(lesion)의 정확한 분할(segmentation)은 매우 중요하다. 그러나 의료 영상은 낮은 대비(low contrast), 모호한 경계, 한정된 데이터셋이라는 특성이 있으며, 특히 병변의 분포 위치와 크기가 매우 가변적이라는 불확실성을 가지고 있다.

기존의 시각적 특징(visual features)에만 의존하는 분할 방식은 이러한 병변의 불확실한 분포와 다양한 크기를 처리하는 데 한계가 있다. 최근 시각-언어 모델링(vision-language modeling)을 통해 텍스트 정보를 결합하려는 시도가 있었으나, 대부분의 기존 방식은 추론(inference) 단계에서도 쌍을 이룬 텍스트 입력이 필요하다는 치명적인 단점이 있다. 실제 임상 현장에서는 모든 영상에 대해 정교하게 작성된 텍스트 보고서가 항상 준비되어 있지 않으므로, 추론 시 텍스트 입력 없이도 텍스트의 세만틱 지식을 활용할 수 있는 모델이 필요하다.

본 논문의 목표는 학습 단계에서는 텍스트 지식을 활용하여 모델의 성능을 높이되, 추론 단계에서는 텍스트 입력 없이 영상만으로도 정교한 분할이 가능한 Scale-aware Text Prompt Network(STPNet)를 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **'Retrieval-Segmentation Joint Learning'**이다. 학습 과정에서 전문 의료 텍스트 저장소(medical text repository)로부터 영상과 가장 관련성이 높은 텍스트 정보를 검색(retrieval)하여 학습에 활용하고, 이 과정을 통해 영상 인코더가 텍스트의 세만틱 정보를 내재적으로 학습하도록 설계하였다. 이를 통해 추론 시에는 별도의 텍스트 입력 없이도 텍스트 가이드가 적용된 것과 같은 효과를 얻을 수 있다.

또한, 병변의 '어디에나 존재할 수 있는(anywhere)' 문제와 '어떤 크기로든 존재할 수 있는(anysize)' 문제를 해결하기 위해, 전역적 정보에서 국소적 정보로 이어지는 다중 스케일 텍스트 특징(multi-scale textual features)을 구축하고 이를 시각적 특징과 결합하는 구조를 제안하였다.

## 📎 Related Works

**1. 의료 영상 분할 (Medical Image Segmentation):**
UNet과 같은 Encoder-Decoder 구조와 Transformer 기반 모델이 주류를 이룬다. UNet은 다양한 스케일의 특징을 캡처하는 데 강점이 있고, Transformer는 Self-attention 메커니즘을 통해 전역적인 문맥(global context)을 파악하는 데 유리하다. 그러나 이러한 방법들은 오직 영상 데이터에만 의존하므로, 성능 향상에 한계가 있다.

**2. 다중 스케일 학습 (Multi-scale Learning):**
Atrous Convolution(PASPP)이나 Pyramid Convolution(MSD-Net) 등을 통해 다양한 크기의 병변을 탐지하려는 시도가 있었다. 하지만 이 역시 영상 특징 내에서만 스케일을 다루었을 뿐, 방사선 보고서(radiology reports)와 같은 가치 있는 텍스트 정보의 다중 스케일을 활용하지는 못했다.

**3. 텍스트 가이드 학습 (Text-guided Learning):**
LViT, TGANet 등 시각-언어 정렬을 통해 분할 성능을 높인 연구들이 존재한다. 그러나 이들은 학습과 추론 모든 단계에서 텍스트 입력이 필수적이며, 텍스트 입력의 다양성이나 부정확성으로 인해 오히려 성능 저하가 발생할 수 있는 위험이 있다.

## 🛠️ Methodology

### 1. 전체 시스템 구조
STPNet은 크게 **Text Retrieval Network**와 **Segmentation Network**라는 두 개의 네트워크로 구성된다. 학습 시에는 두 네트워크가 공동 학습(joint learning)을 수행하며, 추론 시에는 Text Retrieval Network를 제외한 Segmentation Network만 사용하여 효율성을 높인다.

### 2. Text Retrieval Network
이 네트워크는 영상과 가장 유사한 텍스트 설명을 찾는 역할을 한다.

- **Image Encoder ($\text{Enc}_v$):** 사전 학습된 ResNet-101을 백본으로 사용한다. Layer 3와 Layer 4의 특징을 추출하여 업샘플링 후 결합하고, MLP와 MaxPooling을 거쳐 최종 영상 특징 벡터 $F_v$를 생성한다.
  $$F_v = \text{MP}(\text{MLP}([\text{Upsample}(v_h); v_l]))$$
- **Text Encoder ($\text{Enc}_t$):** 사전 학습된 BioClinicalBERT를 사용하며, 가중치는 고정(frozen) 상태로 유지한다. 텍스트 데이터베이스는 다음 네 가지 카테고리로 분류된다: $\text{Infection text}$ (감염 분포), $\text{Num text}$ (병변 수), $\text{Left Loc text}$ (왼쪽 위치), $\text{Right Loc text}$ (오른쪽 위치).
- **Text Retrieval:** 영상 특징 $F_v$와 텍스트 특징 $F_{t,i,j}$ 사이의 코사인 유사도(cosine similarity)를 계산하여 가장 높은 점수를 가진 텍스트를 검색한다.
  $$s_{ij} = \frac{F_v \cdot F_{t,i,j}}{\|F_v\| \cdot \|F_{t,i,j}\|}, \quad f_{t,i} = F_{t,i, \text{argmax}_j(s_{ij})}$$

### 3. Text Features Recombination
검색된 네 가지 종류의 텍스트 특징($f_{t,1} \dots f_{t,4}$)은 전역적 정보에서 국소적 정보로 이어지는 계층 구조를 가진다. 이를 평균값 계산을 통해 재조합하여 다중 스케일 텍스트 특징 $F_{\text{text},i}$를 생성한다.
$$F_{\text{text},i} = \frac{1}{i} \sum_{j=1}^i f_{t,j}$$

### 4. Segmentation Network
하이브리드 CNN-Transformer 구조를 채택하고 있으며, 주요 구성 요소는 다음과 같다.

- **MTBlock (Multi-scale Text-guided Block):** 영상 특징 $F_{\text{img},i}$와 재조합된 텍스트 특징 $\bar{F}_{\text{text},i}$를 결합한다. 텍스트 특징을 영상 특징과 동일한 크기의 맵으로 확장하여 Concatenate한 뒤, Conv-BN-ReLU 과정을 거쳐 융합 특징 $F_{\text{mix},i}$를 생성한다.
- **SSM (Spatial Scale-aware Module):** 융합된 특징에서 공간적 및 다중 스케일 정보를 캡처한다.
    - **공간 특징 추출:** Depth-wise Conv와 $1\times 1$ Conv 후 Sigmoid를 적용해 중요한 영역에 집중한다.
    - **다중 스케일 특징 추출:** 서로 다른 dilation rate ($6, 12, 18$)를 가진 Dilated Convolution을 사용하여 다양한 크기의 병변 특징을 추출한다.
- **UTrans:** Vision Transformer 구조를 기반으로 하며, 영상 특징과 텍스트 특징을 함께 입력받아 Self-attention을 수행한 뒤, 최종적으로는 영상 관련 특징만 추출하여 다음 단계로 전달함으로써 텍스트의 세만틱 정보를 시각적 특징에 주입한다.

### 5. 학습 목표 및 손실 함수
최종 손실 함수 $L_{\text{mix}}$는 세 가지 손실의 합으로 정의된다 ($\lambda_1=\lambda_2=\lambda_3=1$).
$$L_{\text{mix}} = \lambda_1 L_{\text{seg}} + \lambda_2 L_{\text{retrieval}} + \lambda_3 L_{\text{focal}}$$

- **$L_{\text{seg}}$:** Dice Loss와 Cross Entropy Loss의 합으로, 픽셀 단위의 분할 정확도를 최적화한다.
- **$L_{\text{retrieval}}$:** Contrastive Loss를 사용하여 영상 특징과 정답 텍스트 특징 사이의 거리를 좁히고, 오답 텍스트와의 거리는 멀게 한다.
- **$L_{\text{focal}}$:** 검색된 카테고리를 기반으로 영상 특징 $F_v$를 최적화하여 분류 능력을 높인다.

## 📊 Results

### 1. 실험 설정
- **데이터셋:** COVID-CT, COVID-Xray (폐렴), Kvasir-SEG (폴립) 세 가지 데이터셋을 사용하였다.
- **지표:** Dice score, IoU (Intersection over Union), Precision, Recall을 측정하였다.

### 2. 정량적 결과
STPNet은 모든 데이터셋에서 기존 SOTA(State-of-the-art) 모델들을 상회하는 성능을 보였다.

- **COVID-CT:** Dice 76.18%를 기록하며, 텍스트 입력 없이 작동함에도 불구하고 텍스트 입력을 사용하는 CMIRNet(73.69%)이나 Foundation 모델 기반의 SAM2UNet(72.91%)보다 높은 성능을 보였다.
- **COVID-Xray:** Dice 80.63%를 기록하여 UniLSeg(79.99%)를 앞섰다.
- **Kvasir-SEG:** Dice 98.19%라는 매우 높은 성능을 기록하였으며, 특히 SAM2UNet(92.80%) 대비 Dice 5.39%, IoU 8.55%의 큰 폭의 향상을 보였다.

### 3. 기타 분석
- **추론 속도:** Kvasir-SEG 기준 평균 추론 시간은 0.0393s로, 텍스트 가이드 방식인 UniLSeg(0.0707s)나 CMIRNet(0.1270s)보다 훨씬 빨랐다. 이는 추론 시 텍스트 검색 과정을 생략했기 때문이다.
- **시각화:** Saliency map 분석 결과, STPNet은 병변의 크기가 변하더라도 이에 유연하게 대응하여 경계를 정확하게 포착하는 능력이 뛰어남이 확인되었다.

## 🧠 Insights & Discussion

**강점:**
본 연구는 의료 영상 분할에서 텍스트 정보가 주는 강력한 가이드를 유지하면서도, 실무적인 제약 사항인 '추론 시 텍스트 입력 필요성'을 완전히 제거했다는 점에서 매우 실용적이다. 특히 다중 스케일 텍스트 특징을 계층적으로 주입함으로써 병변의 크기와 위치에 따른 불확실성을 효과적으로 줄였다.

**한계 및 비판적 해석:**
실험 결과에서 Kvasir-SEG(폴립)의 성능이 폐렴 데이터셋보다 월등히 높게 나타났다. 이는 폴립과 같이 형태가 비교적 정형화된 병변에서는 텍스트 가이드가 매우 효과적이지만, 형태가 매우 불규칙하고 경계가 모호한 폐렴 병변에서는 텍스트 정보만으로는 한계가 있음을 시사한다. 또한, 모델의 성능이 사전 구축된 '의료 텍스트 저장소'의 품질에 크게 의존한다는 점이 잠재적인 취약점이다. 저장소가 부실하거나 편향되어 있다면 검색 결과가 틀려 오히려 분할 성능을 저해할 가능성이 있다.

## 📌 TL;DR

STPNet은 학습 시에만 의료 텍스트 저장소에서 관련 정보를 검색하여 영상 특징과 정렬시키는 **Retrieval-Segmentation Joint Learning** 방식을 제안한 모델이다. 이를 통해 추론 시에는 텍스트 입력 없이 영상만으로도 텍스트의 세만틱 지식을 활용한 정교한 분할이 가능하다. 특히 다중 스케일 텍스트 가이드와 SSM 모듈을 통해 병변의 다양한 크기와 위치 문제를 해결하였으며, COVID-CT, COVID-Xray, Kvasir-SEG 데이터셋에서 SOTA 성능을 달성하였다. 이 연구는 텍스트-영상 다중 모달 학습의 이점을 유지하면서 실무적 적용 가능성을 극대화했다는 점에서 향후 의료 AI 진단 시스템 구축에 중요한 역할을 할 것으로 보인다.