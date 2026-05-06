# Frustratingly Simple but Effective Zero-shot Detection and Segmentation: Analysis and a Strong Baseline

Siddhesh Khandelwal, Anirudth Nambirajan, Behjat Siddiquie, Jayan Eledath, Leonid Sigal (2023)

## 🧩 Problem to Solve

본 논문은 **Zero-shot Object Detection (ZSD)** 및 **Zero-shot Instance Segmentation (ZSI)** 분야에서 직면한 문제를 다룬다. 객체 탐지 및 세그멘테이션 모델을 학습시키기 위해서는 방대한 양의 인스턴스 수준 어노테이션(bounding boxes, masks)이 필요하며, 이는 수집 비용이 매우 높고 시간이 많이 소요된다는 한계가 있다.

따라서 학습 과정에서 한 번도 본 적 없는 카테고리, 즉 **unseen categories**에 대해 객체를 식별하고 정밀하게 로컬라이즈(localize)하는 능력을 갖추는 것이 본 연구의 핵심 목표이다. 기존 연구들은 이를 위해 매우 복잡한 아키텍처나 생성 모델(generative models)을 도입하는 경향이 있었으나, 저자들은 이러한 복잡성이 반드시 성능 향상으로 이어지는지에 대해 의문을 제기하며, 단순하지만 효과적인 베이스라인을 구축하고자 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 **"복잡한 모델 구조보다 설계 선택지(design choices)에 대한 세밀한 분석과 적절한 시맨틱 임베딩(semantic embedding)의 활용이 더 중요하다"**는 것이다.

- **단순한 2단계 학습 프레임워크 제안**: 복잡한 모듈 대신, 표준 탐지기 학습 후 시맨틱 공간으로의 투영 층(projection layer)만을 학습시키는 단순한 구조를 제안한다.
- **체계적인 어블레이션 연구**: 백본 용량, 임베딩 소스, 배경 임베딩 방식, 전이 메커니즘, 학습 동역학 등 ZSD/ZSI 성능에 영향을 주는 핵심 설계 요소들을 광범위하게 분석하였다.
- **강력한 베이스라인 제공**: 제안한 단순한 구조가 기존의 복잡한 SOTA(State-of-the-art) 모델들보다 우수한 성능을 보임을 입증함으로써, 향후 연구의 기준점이 될 수 있는 강력한 베이스라인을 제시한다.

## 📎 Related Works

### Zero-shot Detection (ZSD)

기존 ZSD 연구들은 주로 모델 구조를 수정하여 성능을 높이려 했다. 일부 연구는 시각적 공간과 시맨틱 공간을 분리하여 앙상블하거나, 배경(background) 카테고리의 임베딩을 정교하게 학습하여 unseen categories와의 구별력을 높이려 했다. 또한, 생성 모델을 통해 unseen categories의 특징을 합성하여 분류기를 학습시키는 방식이 사용되었다.

### Zero-shot Segmentation (ZSI)

ZSI는 ZSD보다 덜 탐구된 분야이며, 주로 픽셀 단위의 semantic segmentation에 집중되어 왔다. 일부 연구는 WordNet의 계층 구조를 활용하거나, 가상 특징을 생성하여 전이 학습을 수행한다. 인스턴스 수준의 ZSI 연구는 드물며, 대부분 unseen categories에 대한 이미지 수준의 캡션이나 레이블이 있다는 가정을 전제로 한다.

### 차별점

본 논문은 복잡한 모델 설계나 생성 모델에 의존하는 대신, **시맨틱 임베딩의 품질**과 **투영 기반의 단순한 전이 학습**이 성능에 더 결정적인 영향을 미친다는 점을 강조하며, 이를 통해 매우 단순한 구조로도 높은 성능을 낼 수 있음을 보여준다.

## 🛠️ Methodology

### 전체 파이프라인

본 모델은 Faster R-CNN(탐지) 또는 Mask R-CNN(세그멘테이션)을 기반으로 하며, 학습은 다음의 **2단계 과정**으로 진행된다.

1. **Step 1: Feature Representation Learning**
   - seen categories에 대한 인스턴스 수준 데이터 $D_s$를 사용하여 Faster/Mask R-CNN의 모든 학습 가능 파라미터를 표준 방식으로 학습시킨다.
2. **Step 2: Information Transfer Learning**
   - Step 1에서 학습된 모델의 파라미터를 모두 **동결(freeze)**한다.
   - 이미지 특징을 시맨틱 임베딩 공간으로 매핑하는 투영 행렬($W_{cls}, W_{reg}, W_{seg}$)만을 학습시킨다.

### 상세 구성 요소 및 방정식

#### 1. Classifier (분류기)

이미지 제안 영역(proposal)의 특징 $z_{i,j}$를 시맨틱 공간으로 투영한 뒤, 정규화된 카테고리 임베딩 $E$와의 내적을 통해 유사도를 측정한다.

- **Seen categories**:
  $$f_{W_{cls}^{seen}}(z_{i,j}) = (W_{cls} z_{i,j}) \left( \frac{E_s}{\|E_s\|_2} \right)^T$$
- **Unseen categories**:
  $$f_{W_{cls}^{unseen}}(z_{i,j}) = (W_{cls} z_{i,j}) \left( \frac{E_u}{\|E_u\|_2} \right)^T$$
최종 확률 $p_{i,j}$는 두 결과의 연결(concatenation)에 소프트맥스($\sigma$) 함수를 적용하여 산출한다.

#### 2. Regressor (회귀기)

바운딩 박스의 좌표를 정밀하게 조정하기 위해 4개의 투영 행렬 $W_{reg}^r$ ($r \in [1,4]$)을 사용한다.

- **Seen/Unseen Regressor**:
  $$f_{W_{reg}}(z_{i,j}) = \left\{ (W_{reg}^r z_{i,j}) \left( \frac{E}{\|E\|_2} \right)^T \right\}; r \in [1,4]$$

#### 3. Segmentor (세그멘터)

Mask R-CNN의 세그멘테이션 헤드에서 추출된 공간적 특징 $z_{m_{i,j}}[x,y]$에 대해 동일한 투영 방식을 적용한다.

- **Seen/Unseen Segmentor**:
  $$f_{W_{seg}}(z_{m_{i,j}}[x,y]) = (W_{seg} z_{m_{i,j}}[x,y]) \left( \frac{E}{\|E\|_2} \right)^T$$

### 학습 및 추론 절차

- **손실 함수**: 분류기는 Cross-Entropy Loss, 회귀기는 Smooth-L1 Loss, 세그멘테이션 헤드는 Pixel-level Binary Cross-Entropy Loss를 사용하여 학습한다.
- **추론 시 bias 조절**: 모델이 학습 데이터인 seen categories에 편향되는 것을 막기 위해 임계값 $\beta$를 도입한다. seen category의 예측 결과 중 신뢰도가 $\beta$보다 낮은 경우 제거함으로써, 상대적으로 unseen categories의 예측 결과가 더 많이 선택되도록 조절한다.

## 📊 Results

### 실험 설정

- **데이터셋**: MSCOCO 2014 (80개 카테고리).
- **Split**: 48/17 split 및 65/15 split (seen/unseen 비율).
- **지표**: mAP @ IoU=0.5, Recall@100, 및 Generalized ZSD/ZSI 설정에서의 Harmonic Mean (HM).

### 주요 결과

1. **ZSD 성능**: 제안된 방법은 기존의 복잡한 모델(예: RRFS)을 크게 상회한다. 특히 CLIP 임베딩을 사용했을 때 mAP가 비약적으로 상승하였으며, 48/17 split에서 RRFS 대비 mAP가 최대 37.3% 향상되는 결과를 보였다.
2. **ZSI 성능**: baseline 모델(ZSI [48])과 비교하여 mAP와 Recall 모두에서 압도적인 성능 향상을 기록하였다.
3. **Generalized ZSD/ZSI**: seen과 unseen을 동시에 탐지해야 하는 GZSD/GZSI 설정에서도 CLIP 기반 모델이 HM mAP 및 HM Recall에서 최상위 성능을 기록하였다.

### 어블레이션 분석 결과

- **Backbone**: ResNet-50보다 ResNet-101과 같은 깊은 모델이 더 풍부한 특징을 제공하여 성능이 향상된다.
- **Embedding Source**: Word2Vec $\rightarrow$ ConceptNet $\rightarrow$ CLIP 순으로 성능이 증가한다. 특히 **CLIP**의 시각-언어 통합 임베딩이 가장 효과적이다.
- **Background Embedding**: 고정된 벡터나 평균 벡터보다 **학습 가능한(learned)** 배경 임베딩을 사용하는 것이 unseen categories 구분 능력을 높인다.
- **Transfer Mechanism**: 휴리스틱한 방식(가장 유사한 클래스 복제 등)보다 투영 행렬을 통한 **학습된 전이(learned transfer)** 방식이 우수하다.
- **Fine-tuning**: 2단계 학습 시 R-CNN 파라미터를 동결하지 않고 함께 학습시키면 seen categories에 과적합(overfitting)되어 일반화 성능이 떨어진다.

## 🧠 Insights & Discussion

### 강점 및 통찰

본 논문은 ZSD/ZSI 분야에서 최근의 트렌드가 '모델의 복잡성 증가'에 매몰되어 있었음을 지적한다. 실험 결과, 아키텍처의 복잡함보다는 **시맨틱 임베딩의 품질(예: CLIP)**과 **단순한 투영 기반의 정렬(alignment)**이 훨씬 더 큰 성능 이득을 준다는 점을 입증하였다. 이는 모델 구조를 복잡하게 만들기보다, 더 나은 사전 학습된 임베딩을 찾고 이를 효율적으로 연결하는 것이 더 효율적인 방향임을 시사한다.

### 한계 및 논의사항

- **임베딩 의존성**: 모델의 성능이 사용되는 시맨틱 임베딩의 품질에 매우 크게 의존한다. 이는 모델 자체의 구조적 혁신보다는 외부 사전 학습 모델의 성능에 기대는 측면이 있다.
- **$\beta$ 하이퍼파라미터**: 추론 시 $\beta$ 값에 따라 seen/unseen 성능의 트레이드오프가 발생하며, 이를 최적으로 설정하는 기준에 대한 추가적인 논의가 필요할 수 있다.

## 📌 TL;DR

본 논문은 복잡한 구조 대신 **"표준 탐지기 학습 $\rightarrow$ 시맨틱 공간 투영 층 학습"**이라는 매우 단순한 2단계 프로세스를 통해 Zero-shot 탐지 및 세그멘테이션 성능을 극대화하였다. 특히 CLIP과 같은 고품질 임베딩의 중요성을 강조하며, 기존의 복잡한 SOTA 모델들을 뛰어넘는 성능을 달성함으로써 향후 ZSD/ZSI 연구의 강력하고 단순한 베이스라인을 제시하였다.
