# A Survey on Deep Learning for Skin Lesion Segmentation

Zahra Mirikharaji, Kumar Abhishek, Alceu Bissoto, Catarina Barata, Sandra Avila, Eduardo Valle, M. Emre Celebi, Ghassan Hamarneh (2023)

## 🧩 Problem to Solve

본 논문은 피부암, 특히 멜라노마(Melanoma)의 조기 진단을 위한 피부 병변 분할(Skin Lesion Segmentation) 분야에서 딥러닝(Deep Learning, DL) 기술의 적용 현황을 체계적으로 분석하고자 한다. 피부 병변 분할은 컴퓨터 보조 진단(Computer-Aided Diagnosis, CAD) 시스템의 핵심 단계로, 병변의 비대칭성, 경계 불규칙성, 크기 등을 측정하는 ABCD 알고리즘의 기반이 된다. 또한, 머신러닝 기반 진단 시스템에서 모델이 병변 내부 영역에 집중하게 함으로써 분류(Classification) 성능을 높이고, 블랙박스 형태인 딥러닝 모델의 해석 가능성을 제공하여 의료진의 신뢰를 얻는 데 중요한 역할을 한다.

그러나 실제 환경에서 피부 병변 분할은 다음과 같은 이유로 매우 도전적인 과제이다.

- **인공적/자연적 아티팩트:** 털(Hair), 혈관, 공기 방울, 수술 마커 등이 이미지에 포함되어 경계 구분을 방해한다.
- **내재적 요인:** 병변의 크기와 모양의 다양성, 피부색의 차이, 낮은 대비(Low contrast) 및 모호한 경계선 등이 존재한다.
- **데이터 부족:** 전문가가 정밀하게 생성한 Ground-truth 세그멘테이션 마스크 데이터셋의 양이 매우 적으며, 구축 비용이 높다.

따라서 본 연구의 목표는 2014년부터 2022년까지 발표된 177편의 연구 논문을 교차 분석하여, 입력 데이터, 모델 설계, 평가 방법이라는 세 가지 차원에서 딥러닝 기반 피부 병변 분할 기술의 트렌드와 한계를 종합적으로 정리하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 기여는 피부 병변 분할 분야의 방대한 문헌을 체계적인 분류 체계(Taxonomy)에 따라 분석하여 학술적 가이드라인을 제시한 점에 있다. 주요 기여 사항은 다음과 같다.

- **광범위한 문헌 분석:** 총 177편의 논문을 분석하여 입력 데이터(데이터셋, 전처리, 합성 데이터), 모델 설계(아키텍처, 모듈, 손실 함수), 평가 측면(어노테이션 요구사항, 성능 지표)을 상세히 조사하였다.
- **상세한 기술 분류 체계 구축:** 단순한 나열이 아니라, Single-network, Multiple-network, Hybrid-feature, Transformer 모델 등으로 아키텍처를 분류하고, 각 모델에 사용된 핵심 모듈과 손실 함수 간의 관계를 분석하였다.
- **데이터셋 및 벤치마크 정리:** ISIC Archive를 포함하여 공개된 주요 데이터셋의 특성, 클래스 분포, 이미지 모달리티(Dermoscopic vs Clinical)를 표 형태로 정리하여 향후 연구자들이 쉽게 비교할 수 있도록 하였다.
- **미래 연구 방향 제시:** 모바일 환경 적용, 데이터 다양성(다양한 피부톤) 확보, 비지도 학습(Unsupervised learning)의 필요성 등 실무적/학술적 관점에서의 향후 과제를 도출하였다.

## 📎 Related Works

기존에도 피부 병변 분할에 관한 서베이 논문들이 존재하였으나, 본 논문은 다음과 같은 차별점을 가진다.

- **이전 서베이와의 차이점:** Celebi et al. (2009, 2015)의 연구들은 딥러닝 혁명 이전의 고전적 이미지 처리 및 머신러닝 기법에 집중하였다. 반면, Adegun and Viriri (2020)의 서베이는 딥러닝을 다루었으나 ISIC 챌린지(2018, 2019)의 상위 알고리즘에 집중하여 분석 범위가 상대적으로 좁았다.
- **본 논문의 차별성:** 딥러닝 기반의 최신 기법들을 훨씬 더 넓은 범위(177편)에서 다루며, 특히 모델의 세부 모듈(Attention, Dilated Convolution 등)과 다양한 손실 함수(Loss functions)의 수학적 설계 의도를 깊이 있게 분석하였다.

## 🛠️ Methodology

본 논문은 딥러닝 기반 분할 파이프라인의 구성 요소를 중심으로 분석을 진행하였다.

### 1. 입력 데이터 및 전처리 (Input Data)

- **데이터 모달리티:** 확대경을 사용해 피부 하부 구조를 보는 Dermoscopic 이미지와 일반 카메라로 촬영한 Clinical 이미지로 구분한다.
- **데이터 증강(Data Augmentation):**
  - **전통적 방식:** 회전, 반전, 스케일링, 색상 변환 등 기하학적/광도적 변환을 적용한다.
  - **현대적 방식:** GAN(Generative Adversarial Networks)을 이용하여 실제와 유사한 합성 이미지를 생성한다. 특히 $\text{pix2pixHD}$와 같은 Image-to-Image translation 기법이 고품질 이미지 생성에 사용된다.
- **전처리 기법:** Downsampling, 색 공간 변환(RGB $\rightarrow$ HSV, CIELAB), 대비 향상, 털 제거(Hair removal) 등이 포함된다.

### 2. 모델 아키텍처 (Model Architecture)

분석 결과, 모델은 크게 네 가지 범주로 나뉜다.

- **Single Network Models:** 주로 Encoder-Decoder 구조의 FCN 또는 U-Net 기반이다.
  - **Shortcut Connections:** Gradient vanishing 문제를 해결하기 위한 Residual connection, 저수준 특징을 보존하는 Skip connection, 특징 재사용을 극대화하는 Dense connection이 주로 사용된다.
  - **Convolutional Modules:** 수용 영역(Receptive field)을 넓히는 Dilated convolution, 파라미터 수를 줄이는 Separable convolution 등이 핵심이다.
  - **Multi-scale Modules:** Image Pyramid, Parallel multi-scale convolution, Pyramid pooling 등을 통해 다양한 크기의 병변을 포착한다.
  - **Attention Modules:** Spatial 및 Channel attention을 통해 병변 영역에 집중하고 배경 노이즈를 억제한다.
- **Multiple Network Models:** 모델 앙상블(Ensemble), 분류와 분할을 동시에 수행하는 Multi-task learning, 그리고 생성자와 판별자가 경쟁하는 GAN 기반 모델이 포함된다.
- **Hybrid Feature Models:** 딥러닝 특징과 수작업으로 설계된 Hand-crafted feature(예: LBP, Wavelet)를 결합하여 도메인 지식을 반영한다.
- **Transformer Models:** 최근 ViT(Vision Transformer)의 성공 이후 $\text{TransUNet}$, $\text{Swin-Unet}$ 등이 도입되어 전역적 문맥(Global context)을 더 잘 파악하는 경향을 보인다.

### 3. 손실 함수 (Loss Functions)

모델의 최적화 목표를 정의하는 손실 함수는 다음과 같이 구분된다.

- **픽셀 기반 손실:** 가장 일반적인 Cross-Entropy(CE)와 $L_1, L_2$ norm이 있다.
- **오버랩 기반 손실:** 예측 마스크와 실제 마스크의 겹침 정도를 측정하는 Dice Loss와 Jaccard Loss가 대표적이다.
  - Dice Loss 수식: $L_{\text{dice}} = 1 - \frac{2 \sum y_i \hat{y}_i}{\sum y_i + \sum \hat{y}_i}$
- **특수 목적 손실:**
  - **Tversky Loss:** False Positive(FP)와 False Negative(FN)에 서로 다른 가중치를 두어 클래스 불균형 문제를 해결한다.
  - **Star-Shape Loss:** 병변이 중심에서 방사형으로 뻗어나가는 특성을 반영하여 공간적 일관성을 강제한다.
  - **End-Point Error Loss:** 마스크의 1차 미분값을 사용하여 경계선(Boundary)의 정확도를 높인다.
  - **Adversarial Loss:** 판별자(Discriminator)를 통해 생성된 마스크가 실제 정답 마스크의 분포와 유사하도록 학습시킨다.

## 📊 Results

### 1. 실험 설정 및 데이터셋

분석 대상 논문들은 주로 다음과 같은 공개 데이터셋을 사용하였다.

- **ISIC Archive:** 세계 최대의 저장소로, ISIC 2016, 2017, 2018 챌린지 데이터셋이 가장 많이 사용되었다. 특히 ISIC 2017이 가장 대중적이다.
- **HAM10000:** 10,015장의 대규모 데이터셋으로, 최근 세그멘테이션 마스크가 공개되어 많이 활용되고 있다.
- **PH2:** Dermoscopic 이미지 데이터셋으로, 초기 연구들에서 벤치마크로 자주 사용되었다.

### 2. 성능 지표 (Metrics)

분할 성능은 주로 다음과 같은 지표로 측정된다.

- **Jaccard Index (J):** $\text{IoU (Intersection over Union)}$라고도 하며, 본 서베이에서 성능 비교의 주 지표로 사용되었다.
- **Dice Coefficient (F):** Jaccard와 단조 증가 관계에 있으며, 의료 영상 분할에서 매우 선호된다.
- **Accuracy (AC), Sensitivity (SE), Specificity (SP):** 기본적인 픽셀 단위 분류 정확도를 측정한다.
- **Matthews Correlation Coefficient (MCC):** 클래스 불균형이 심한 데이터에서 더 신뢰할 수 있는 지표로 평가된다.

### 3. 주요 결과 및 분석

- **아키텍처 경향:** 대부분의 연구가 U-Net 구조를 기본으로 하며, 여기에 Attention 모듈이나 Dilated convolution을 추가하여 성능을 높이는 방향으로 발전하고 있다.
- **손실 함수 조합:** 단일 손실 함수보다는 $\text{CE + Dice}$ 또는 $\text{CE + Jaccard}$와 같이 픽셀 단위 손실과 영역 단위 손실을 결합하여 사용하는 것이 일반적이다.
- **데이터 증강의 효과:** 단순한 기하학적 변환보다 GAN을 통한 합성 데이터 생성이나 Test-time augmentation(TTA)이 일반화 성능 향상에 크게 기여함을 확인하였다.

## 🧠 Insights & Discussion

### 1. 강점 및 성과

딥러닝의 도입으로 인해 과거의 수작업 특징 추출 방식보다 훨씬 더 강건(Robust)하고 정확한 분할이 가능해졌다. 특히 Transformer의 도입으로 전역적 특징 추출 능력이 향상되었으며, 다양한 Loss function의 설계로 의료 영상 특유의 클래스 불균형 문제를 효과적으로 다루기 시작했다.

### 2. 한계 및 미해결 질문

- **데이터의 편향성:** 대부분의 데이터셋이 밝은 피부톤(Fair-skinned) 환자 위주로 구성되어 있어, 어두운 피부톤에 대한 모델의 전이 가능성(Transferability)과 공정성(Fairness) 문제가 제기된다.
- **어노테이션의 불확실성:** 전문가 사이에서도 병변의 경계에 대한 의견이 일치하지 않는 경우가 많아(Inter-annotator variability), 단일 Ground-truth에 의존하는 평가 방식의 한계가 있다.
- **재현성 부족:** 분석 대상 논문 중 코드 공개 비율이 21.47%로 매우 낮아, 연구 결과의 객관적인 검증과 재현이 어렵다.

### 3. 비판적 해석

현재의 연구들은 주로 ISIC와 같은 벤치마크 데이터셋에서 Jaccard Index를 높이는 데 치중해 있다. 하지만 실제 임상 환경에서는 다양한 노이즈가 포함된 Clinical 이미지에서의 성능이 더 중요함에도 불구하고, 관련 데이터셋과 연구가 Dermoscopic 이미지에 비해 현저히 부족하다는 점은 실용적 관점에서 큰 간극이다.

## 📌 TL;DR

본 논문은 177편의 문헌을 통해 딥러닝 기반 피부 병변 분할 기술을 집대성한 서베이 보고서이다. U-Net 기반 아키텍처에서 Transformer와 Attention 모듈로의 진화, 그리고 단순 CE 손실에서 영역 기반 및 구조적 손실 함수로의 발전을 체계적으로 분석하였다. 특히 데이터셋의 편향성과 코드 공개 부족, 임상 이미지 연구의 부재를 지적하며, 향후 연구가 단순 성능 수치 향상을 넘어 모바일 배포 가능성, 피부톤 다양성 확보, 비지도 학습으로 확장되어야 함을 강조하고 있다.
