# 3D Kidneys and Kidney Tumor Semantic Segmentation using Boundary-Aware Networks

Andriy Myronenko and Ali Hatamizadeh (2019)

## 🧩 Problem to Solve

본 논문은 복부 3D CT 스캔 영상에서 신장(Kidneys)과 신장 종양(Kidney Tumor)을 자동으로 분할(Segmentation)하는 문제를 다룬다.

신장암 치료 계획 시, 종양의 형태학적 세부 사항을 정량화하는 것은 질병의 진행 상황을 모니터링하고 치료 방법을 결정하는 데 매우 중요하다. 특히, 종양과 영향을 받은 신장 전체를 제거하는 근치적 신장 절제술(Radical Nephrectomy, RN)과 종양만 제거하고 신장을 보존하는 부분 신장 절제술(Partial Nephrectomy, PN) 사이의 선택을 위해 정확한 분할 정보가 필수적이다.

그러나 전문가가 수동으로 영역을 구분하는 방식은 시간이 많이 소요될 뿐만 아니라 오류가 발생하기 쉽고, 분석가마다 결과가 다를 수 있는 주관성의 문제가 존재한다. 따라서 본 연구의 목표는 신장과 종양의 경계 정보를 효과적으로 학습하여 신뢰할 수 있는 자동 분할을 수행하는 end-to-end Boundary-Aware Fully Convolutional Neural Networks(CNNs)를 개발하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **경계 인식 전용 스트림(Dedicated Boundary Stream)**을 구축하여, 주 분할 네트워크가 놓치기 쉬운 장기와 종양의 가장자리(Edge) 정보를 명시적으로 학습하도록 하는 것이다.

단순히 단일 네트워크로 분할을 수행하는 것이 아니라, 인코더에서 추출된 특징 맵을 바탕으로 경계 정보를 전문적으로 처리하는 별도의 브랜치를 두고, 이를 경계 인식 손실 함수(Edge-aware loss terms)로 감독함으로써 분할 결과의 정밀도를 높였다.

## 📎 Related Works

전통적으로 신장 분할에는 Deformable models, GrabCuts, Region growing, Atlas-based methods 등이 사용되었다. 하지만 이러한 방식들은 자동화 수준이 낮고, 장기의 형태에 대한 사전 통계 정보에 크게 의존한다는 한계가 있다.

최근에는 딥러닝, 특히 CNN 기반의 접근 방식이 의료 영상 분석에서 두드러진 성과를 보이고 있다. 3D U-Net을 이용한 신장 검출 및 분할, Pyramid pooling module을 갖춘 modified residual FCN, 그리고 신장 영역을 먼저 찾고 내부를 분할하는 Cascaded approach 등이 제안되었다. 본 논문은 이러한 기존 CNN 기반 방식들에 '경계 인식'이라는 명시적인 구조적 장치를 추가하여 정밀도를 개선하고자 하였다.

## 🛠️ Methodology

### 1. 전체 시스템 구조

본 네트워크는 크게 **주 분할 브랜치(Main Segmentation Branch)**와 **경계 인식 스트림(Boundary Stream)**의 두 가지 경로로 구성된다.

* **주 분할 브랜치**: 비대칭 인코더-디코더(Asymmetric encoder-decoder) 구조를 따른다.
  * **Encoder**: $176 \times 176 \times 176$ 크기의 입력을 받아 $3 \times 3 \times 3$ 합성곱(Convolution)과 Residual block, 그리고 다운샘플링을 위한 Strided convolution을 통해 특징 맵을 추출한다.
  * **Decoder**: Bilinear interpolation을 통해 특징 맵을 업샘플링하며, 인코더의 특징 맵을 더해주는 스킵 연결(Skip connection) 구조를 가진다.
* **경계 인식 스트림**: 메인 인코더에서 추출된 특징 맵을 입력으로 받아 경계 정보를 강조하는 Attention-driven decoder 역할을 수행한다.
* **최종 출력**: 디코더의 출력과 경계 스트림의 출력을 결합(Concatenate)한 후, $1 \times 1 \times 1$ 합성곱과 Sigmoid 활성화 함수를 통해 각 복셀(Voxel)이 신장 및 종양인지, 혹은 종양인지에 대한 확률을 결정한다.

### 2. Boundary Stream 및 Attention Gate

경계 스트림의 목적은 메인 인코더의 특징 맵에서 엣지 정보를 강조하는 것이다. 이를 위해 각 해상도 단계에서 **Attention Gate**를 사용한다.

* Attention Gate는 인코더의 특징 맵과 이전 단계의 출력을 입력으로 받아, $3 \times 3 \times 3$ 합성곱 $\rightarrow$ ReLU $\rightarrow$ $1 \times 1 \times 1$ 합성곱 $\rightarrow$ Sigmoid 과정을 거쳐 Attention map을 생성한다.
* 생성된 Attention map을 특징 맵에 원소별 곱셈(Element-wise multiplication)하여 최종적인 경계 특징을 추출한다.

### 3. 손실 함수 (Loss Functions)

본 모델은 주 스트림과 경계 스트림 모두에 대해 Dice Loss를 적용하며, 경계 스트림에는 추가적으로 Weighted BCE Loss를 적용한다.

**Dice Loss**는 예측값($y_{pred}$)과 실제값($y_{true}$) 사이의 겹침 정도를 측정하며 다음과 같이 정의된다.
$$L_{Dice} = 1 - \frac{2 \sum y_{true} * y_{pred}}{\sum y_{true}^2 + \sum y_{pred}^2 + \epsilon}$$
여기서 $\epsilon$은 0으로 나누는 것을 방지하기 위한 작은 상수이다.

**Weighted Binary Cross Entropy (BCE) Loss**는 경계 복셀과 비경계 복셀 간의 심한 불균형을 해결하기 위해 경계 스트림에 적용된다.
$$L_{BCE} = -\beta \sum_{j \in y^+} \log P(y_{pred,j} = 1|x; \theta) - (1-\beta) \sum_{j \in y^-} \log P(y_{pred,j} = 0|x; \theta)$$
여기서 $y^+$는 엣지 복셀 집합, $y^-$는 비엣지 복셀 집합이며, $\beta$는 전체 복셀 수 대비 비엣지 픽셀의 비율을 나타낸다.

최종 손실 함수는 '종양 전용(tumor-only)' 예측과 '전경(foreground)' 예측에 대한 손실 값들의 평균으로 계산된다.

## 📊 Results

### 실험 설정

* **데이터셋**: KiTS 2019 데이터셋 (총 300명의 환자 중 210명 학습, 90명 테스트).
* **전처리**: CT 강도 값을 $[-1, 1]$ 범위로 정규화하고, $1 \times 1 \times 1\text{mm}$ 등방성(Isotropic) 해상도로 리샘플링하였다.
* **학습 세부사항**: 8장의 NVIDIA Tesla V100 GPU 사용, Adam optimizer, 배치 크기 8, 총 300 에포크 학습. 학습 시 종양 영역을 더 자주 샘플링하는 전략을 사용하였다.
* **추론 전략**: Test Time Augmentation(TTA)과 5개 모델의 앙상블(Ensemble)을 적용하여 성능을 극대화하였다.
* **평가 지표**: Kidneys Dice (신장+종양), Tumor Dice (종양만), 그리고 이 둘의 평균인 Composite Dice를 사용하였다.

### 주요 결과

KiTS 2019 테스트 세트에서 얻은 최종 결과는 다음과 같다.

| Metric | Score |
| :--- | :--- |
| **Kidneys Dice** | 0.9742 |
| **Tumor Dice** | 0.8103 |
| **Composite Dice** | 0.8923 |

본 모델은 KiTS 2019 챌린지 참여자 100명 중 **Composite Dice 기준 종합 9위**를 기록하였다. 특히 신장(Kidneys) 분할 성능이 매우 우수한 것으로 나타났다.

## 🧠 Insights & Discussion

본 논문은 의료 영상 분할에서 가장 까다로운 부분 중 하나인 '경계 영역'을 별도의 스트림으로 분리하여 처리함으로써 분할 정밀도를 높일 수 있음을 입증하였다. 특히, Weighted BCE Loss를 통해 경계 복셀의 희소성 문제를 해결하고, Attention Gate를 통해 유의미한 특징만을 강조한 점이 유효했다.

다만, 결과 분석에서 신장(Kidneys Dice: 0.9742)에 비해 종양(Tumor Dice: 0.8103)의 분할 성능이 상대적으로 낮게 나타난다. 이는 종양의 크기가 매우 다양하고, 신장 조직과의 대비(Contrast)가 낮은 경우가 많아 경계 인식 스트림만으로는 완전히 해결되지 않는 복잡성이 존재함을 시사한다. 또한, $176^3$ 크기의 크롭(Crop) 영역을 사용한 학습 방식이 전체 장기의 전역적 문맥(Global context)을 파악하는 데 일부 제약이 있었을 가능성이 있다.

## 📌 TL;DR

본 연구는 3D CT 영상에서 신장과 종양을 정밀하게 분할하기 위해 **경계 인식 전용 스트림(Boundary-aware stream)**과 **Attention Gate**, 그리고 **Weighted BCE Loss**를 결합한 CNN 아키텍처를 제안하였다. KiTS 2019 챌린지에서 종합 9위를 기록하며 그 효과를 입증하였으며, 특히 경계 정보를 명시적으로 학습하는 구조가 의료 영상의 정밀 분할에 매우 중요하다는 것을 보여주었다. 이 연구는 향후 다른 장기나 병변의 정밀 분할 모델 설계에 있어 경계 인식 모듈의 필요성을 제시하는 중요한 참고 자료가 될 수 있다.
