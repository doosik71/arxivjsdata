# Medical Image Segmentation Using Deep Learning: A Survey

Risheng Wang, Tao Lei, Ruixia Cui, Bingtao Zhang, Hongying Meng and Asoke K. Nandi (2021)

## 🧩 Problem to Solve

의료 영상 분할(Medical Image Segmentation)은 영상 내의 해부학적 또는 병리학적 구조의 변화를 명확하게 하여 컴퓨터 보조 진단(CAD) 및 스마트 의료의 진단 효율성과 정확성을 높이는 핵심적인 역할을 수행한다. 그러나 의료 영상은 일반적인 RGB 영상과 달리 블러(blur), 노이즈, 낮은 대비(low contrast) 등의 문제가 빈번하며, 특히 전문의의 정교한 레이블링이 필요하므로 대량의 학습 데이터를 확보하는 데 막대한 비용과 시간이 소요된다는 한계가 있다.

기존의 의료 영상 분할 관련 서베이 논문들은 주로 문헌들을 시간 순서대로 나열하거나 단순 그룹화하여 소개하는 경향이 있으며, 딥러닝의 기술적 분기나 실제 태스크의 특성(예: few-shot learning, imbalance learning)을 깊게 다루지 않는 문제가 있었다. 따라서 본 논문의 목표는 딥러닝 기반의 의료 영상 분할 기술을 '거시적 관점에서 미시적 관점(coarse to fine)'으로 재분류하여 제시함으로써, 연구자들이 관련 논리를 쉽게 이해하고 적절한 개선 방향을 설정할 수 있도록 가이드를 제공하는 것이다.

## ✨ Key Contributions

본 논문의 중심적인 설계 아이디어는 기존의 단순 나열식 서베이에서 벗어나, 기술적 계층 구조를 기반으로 딥러닝 방법론을 체계적으로 분류하는 것이다.

1. **다층적 분류 체계 구축**: 의료 영상 분할 기술을 Supervised Learning과 Weakly Supervised Learning의 두 가지 큰 축으로 나누고, 이를 다시 네트워크 백본, 블록 설계, 손실 함수 등의 세부 요소로 세분화하여 분석하였다.
2. **Supervised Learning의 세밀한 분석**: 네트워크 아키텍처의 개선 방향을 Backbone Networks, Network Blocks, Loss Functions라는 세 가지 관점에서 분석하여, 모델 설계의 동기와 전략을 명확히 제시하였다.
3. **Weakly Supervised Learning의 전략 제시**: 데이터 부족 문제를 해결하기 위한 Data Augmentation, Transfer Learning, Interactive Segmentation의 관점에서 최신 문헌을 검토하였다.
4. **최신 트렌드 및 데이터셋 정리**: Neural Architecture Search(NAS), Graph Convolutional Networks(GCN), Multi-modality Data Fusion 등 최신 경향을 분석하고, 현재 사용 가능한 공공 의료 영상 데이터셋을 집대성하여 제공하였다.

## 📎 Related Works

논문에서는 기존의 여러 서베이 연구들을 언급하며 본 논문과의 차별점을 제시한다.

- **기존 연구의 한계**: Shen et al. [22], Litjens et al. [23] 등은 딥러닝의 전반적인 의료 영상 분석을 다루었으나 분할 태스크에 특화된 심층 분석이 부족했다. 또한 Taghanaki et al. [24]이나 Seo et al. [25]과 같은 연구들은 방법론을 분류하긴 했으나, 여전히 시간순 요약에 치중하거나 태스크 중심의 특성(예: 클래스 불균형 문제)을 간과하는 경향이 있었다.
- **차별점**: 본 논문은 단순한 문헌 요약이 아니라, 모델의 구성 요소(Backbone $\rightarrow$ Block $\rightarrow$ Loss)라는 기술적 흐름에 따라 분류함으로써, 독자가 새로운 모델을 설계할 때 어떤 구성 요소를 조합해야 하는지 직관적으로 이해할 수 있게 한다.

## 🛠️ Methodology

본 논문은 서베이 논문으로서 특정 알고리즘을 제안하는 대신, 의료 영상 분할의 기술적 파이프라인을 다음과 같이 체계화하여 분석한다.

### 1. Supervised Learning (지도 학습)
가장 높은 정확도가 요구되는 의료 영상 분할에서 가장 널리 사용되는 방법이다.

#### A. Backbone Networks
- **U-Net**: 인코더-디코더 구조와 Skip Connection을 통해 저해상도(Semantic) 특징과 고해상도(Detail) 특징을 융합하는 벤치마크 모델이다.
- **3D Net (3D U-Net, V-Net)**: CT, MRI와 같은 3D 볼륨 데이터의 공간적 상관관계를 학습하기 위해 3D 합성곱 커널을 사용한다. 특히 V-Net은 Residual Connection을 도입하여 더 깊은 네트워크를 가능하게 했다.
- **RNN**: 영상 시퀀스의 시간적 의존성이나 슬라이스 간 관계를 모델링하기 위해 LSTM 등을 결합한다.
- **Cascade Models**: Coarse-to-Fine 전략을 사용하여, 첫 번째 네트워크가 대략적인 영역을 찾고 두 번째 네트워크가 세부 영역을 정밀하게 분할하는 방식이다. 2D와 3D 모델을 하이브리드로 결합하여 연산 효율성과 정확도를 동시에 잡으려는 시도가 많다.

#### B. Network Function Blocks
- **Dense Connection**: 모든 이전 레이어의 출력을 입력으로 받아 풍부한 특징을 추출한다 (예: U-Net++).
- **Inception**: 다양한 크기의 커널을 병렬로 배치하여 멀티스케일 특징을 추출한다.
- **Depthwise Separable Convolution**: 연산량을 줄여 3D 데이터 처리에 효율적인 경량 네트워크를 구축한다.
- **Attention Mechanism**: 
    - **Spatial Attention**: 공간 도메인에서 픽셀의 중요도를 계산한다.
    - **Channel Attention**: 채널 간의 관계를 통해 유용한 특징을 강조하고 불필요한 특징을 억제한다 (예: SE-Net).
    - **Non-local Attention**: Self-attention을 통해 국소적인 합성곱의 한계를 넘어 전역적인(global) 맥락 정보를 추출한다.
- **Multi-scale Fusion**: Atrous Convolution(Dilated Convolution)과 ASPP(Atrous Spatial Pyramid Pooling)를 사용하여 파라미터 증가 없이 수용 영역(Receptive Field)을 확장한다.

#### C. Loss Functions
클래스 불균형(Class Imbalance) 문제를 해결하는 것이 핵심이다.

- **Cross Entropy (CE)**: 픽셀 단위의 분류 오차를 측정한다.
  $$CE(p, \hat{p}) = -(p \log(\hat{p}) + (1-p) \log(1-\hat{p}))$$
- **Weighted Cross Entropy (WCE)**: 클래스 불균형을 해소하기 위해 양성 샘플에 가중치 $\beta$를 부여한다.
  $$WCE(p, \hat{p}) = -(\beta p \log(\hat{p}) + (1-p) \log(1-\hat{p}))$$
- **Dice Loss**: 예측 결과와 Ground Truth 간의 겹침 정도(Overlap)를 직접 최적화한다.
  $$DL(p, \hat{p}) = 1 - \frac{2 \langle p, \hat{p} \rangle}{\|p\|_1 + \|\hat{p}\|_1}$$
- **Tversky Loss**: False Positive(FP)와 False Negative(FN)의 기여도를 조절하여 Dice Loss를 일반화한 형태이다.
- **Generalized Dice Loss (GDL)**: 클래스별 가중치를 부여하여 심각한 클래스 불균형 상황에서도 안정적인 학습을 가능하게 한다.
- **Boundary Loss**: 분할된 경계와 실제 경계 사이의 거리를 최소화하여 정밀한 경계 추출을 돕는다.

### 2. Weakly Supervised Learning (약지도 학습)
레이블 부족 문제를 해결하기 위한 전략이다.

- **Data Augmentation**: 전통적인 기하학적 변환(회전, 확대/축소) 외에도 cGAN(Conditional GAN)을 이용해 가상의 의료 영상을 생성하여 데이터를 증강한다.
- **Transfer Learning**: ImageNet으로 사전 학습된 모델을 인코더로 사용하여 Fine-tuning하거나, CycleGAN을 통해 서로 다른 도메인(예: CT $\rightarrow$ MRI) 간의 특징을 전이하는 Domain Adaptation을 수행한다.
- **Interactive Segmentation**: 사용자가 마우스 클릭이나 바운딩 박스로 수정을 가하면 모델이 이를 반영하여 결과를 업데이트하는 방식이다 (예: DeepIGeoS, BIFSeg).

### 3. 최신 연구 방향 (Popular Directions)
- **NAS (Neural Architecture Search)**: 전문가의 수동 설계 대신 알고리즘이 최적의 네트워크 구조를 자동으로 탐색한다.
- **GCN (Graph Convolutional Network)**: 영상을 그래프 구조로 변환하여 비유클리드 공간에서의 객체 간 관계를 학습한다.
- **Interpretability**: Saliency Map이나 Attention Map을 통해 모델이 왜 그렇게 판단했는지 시각적으로 설명하는 연구가 진행 중이다.
- **Multi-modality Fusion**: CT, MRI 등 서로 다른 모달리티의 데이터를 입력, 레이어, 또는 결정 단계에서 융합하여 더 풍부한 정보를 활용한다.

## 📊 Results

본 논문은 서베이 논문이므로 새로운 실험 결과보다는 기존 연구들의 평가 방법과 데이터셋을 정리하여 제시한다.

### 1. 주요 평가 지표 (Metrics)
- **Pixel Accuracy (PA)**: 전체 픽셀 중 정확히 분류된 픽셀의 비율이다.
- **Dice Score**: 예측 영역과 실제 영역의 겹침 정도를 측정하며, 의료 영상 분할에서 가장 널리 쓰인다.
  $$Dice = \frac{2|A \cap B|}{|A| + |B|}$$
- **VOE (Volume Overlap Error)**: Jaccard 지수의 보수로, 영역 간의 불일치 정도를 측정한다.
- **Surface Distance Metrics**: ASD(Average Symmetric Surface Distance)와 MSD(Maximum Symmetric Surface Distance, 또는 Hausdorff Distance)를 통해 경계면의 거리 오차를 정밀하게 측정한다.

### 2. 공공 데이터셋
논문에서는 Table I을 통해 간, 췌장, 대장, 심장, 폐, 전립선, 뇌(BRATS 등), 신장 등 장기별로 사용 가능한 공개 데이터셋과 URL을 상세히 정리하여 제공한다.

## 🧠 Insights & Discussion

### 강점 및 기여
본 논문은 의료 영상 분할이라는 방대한 분야를 'Backbone $\rightarrow$ Block $\rightarrow$ Loss'라는 논리적 흐름으로 재구성함으로써, 연구자가 자신의 문제 상황(예: 데이터 부족 $\rightarrow$ Weakly Supervised, 경계 불분명 $\rightarrow$ Boundary Loss)에 맞는 솔루션을 빠르게 찾을 수 있도록 돕는다. 특히 3D 데이터 처리의 어려움과 클래스 불균형이라는 의료 영상 특유의 문제를 기술적으로 어떻게 해결해왔는지 체계적으로 분석하였다.

### 한계 및 비판적 해석
- **방법론의 깊이**: 광범위한 서베이인 만큼 개별 논문의 세부적인 실험 수치나 하이퍼파라미터에 대한 분석보다는 아키텍처의 분류에 집중되어 있어, 특정 기법의 실제 성능 향상 폭을 파악하기에는 한계가 있다.
- **Transformers의 비중**: 논문 작성 시점에 Transformer가 급부상하고 있었으나, 본문의 주된 내용은 여전히 CNN 기반의 구조에 치중되어 있다. 다만, 마지막 섹션에서 CNN과 Transformer의 결합이 미래의 핵심 방향이 될 것임을 명시하고 있다.

### 향후 연구 방향
- **NAS와 수동 설계의 조화**: 완전 자동화된 NAS보다는 전문가의 도메인 지식을 바탕으로 백본을 설계하고 세부 모듈을 NAS로 최적화하는 하이브리드 접근법이 필요하다.
- **해석 가능성(Interpretability)**: 의료 현장에서는 결과의 정확도만큼이나 '왜' 그런 결과가 나왔는지에 대한 근거가 중요하므로, XAI(Explainable AI) 기술의 통합이 필수적이다.

## 📌 TL;DR

이 논문은 딥러닝 기반 의료 영상 분할 기술을 **Supervised**와 **Weakly Supervised** 학습으로 나누고, 이를 다시 **백본-블록-손실 함수**라는 계층적 구조로 분석한 종합 서베이 보고서이다. 특히 클래스 불균형 해결을 위한 다양한 손실 함수와 데이터 부족 문제를 극복하기 위한 약지도 학습 전략을 체계화하였다. 본 연구는 향후 연구자들이 자신의 태스크에 맞는 최적의 네트워크 구성 요소를 선택하고 조합하는 데 있어 매우 유용한 가이드라인을 제공하며, 특히 **CNN과 Transformer의 결합, GCN의 도입, 그리고 모델의 해석 가능성 확보**가 차세대 의료 영상 분할의 핵심 과제가 될 것임을 시사한다.