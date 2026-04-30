# Seismic Facies Analysis: A Deep Domain Adaptation Approach

M Quamer Nasim, Tannistha Maiti, Ayush Srivastava, Tarry Singh, and Jie Mei (2021)

## 🧩 Problem to Solve

본 연구가 해결하고자 하는 핵심 문제는 지진파 영상(Seismic Images) 분석에서 발생하는 **데이터 부족(Data Scarcity)**과 **도메인 변화(Domain Shift)** 문제이다. 딥러닝 모델, 특히 심층 신경망(DNN)은 대량의 레이블링된 데이터가 필요하지만, 지진파 해석 분야에서는 전문가의 수동 작업으로 인해 레이블링된 데이터를 확보하는 비용과 시간이 매우 많이 소요된다.

또한, 특정 지역(Source Domain, SD)에서 학습된 모델을 다른 지역(Target Domain, TD)의 데이터에 적용할 때, 지질학적 설정, 퇴적 환경, 암석 밀도 및 공극률의 차이로 인해 데이터 분포가 달라지는 Domain Shift 현상이 발생한다. 이로 인해 모델의 일반화 성능이 급격히 떨어지며, 결과적으로 서로 다른 지역의 지진파 영상에 대해 일관된 세그멘테이션 및 분류 성능을 확보하는 것이 매우 어렵다. 본 논문의 목표는 데이터가 부족한 상황에서도 높은 성능을 내는 네트워크 구조인 EarthAdaptNet(EAN)을 제안하고, 레이블이 없는 타겟 도메인에 적응시키기 위한 Unsupervised Deep Domain Adaptation(DDA) 기법을 적용하는 것이다.

## ✨ Key Contributions

본 논문의 주요 기여 사항은 다음과 같다.

1.  **EarthAdaptNet (EAN) 아키텍처 제안**: 지진파 상(Seismic Facies)의 정밀한 묘사를 위해 설계된 네트워크로, 특히 데이터가 적은 소수 클래스(Minority Classes)에 대해 기존 베이스라인보다 향상된 성능을 보인다.
2.  **CORAL (Correlation Alignment) 기반의 DDA 도입**: 레이블이 없는 타겟 도메인의 데이터를 활용하기 위해, 소스 도메인과 타겟 도메인의 특징 맵(Feature Map) 간의 상관관계(Correlation)를 정렬하는 CORAL 손실 함수를 도입하여 EAN-DDA 네트워크를 구축하였다.
3.  **지질학적 유사성에 기반한 도메인 전이 검증**: 서로 다른 지역(네덜란드 F3 블록 $\rightarrow$ 캐나다 Penobscot)임에도 불구하고 유사한 반사 패턴(Reflection Patterns)을 가진 클래스들을 정의하고, 이를 통해 도메인 적응의 가능성을 실험적으로 증명하였다.

## 📎 Related Works

기존의 지진파 해석 자동화 연구들은 주로 CNN 기반의 딥러닝 모델을 사용해 왔으나, 다음과 같은 한계가 존재한다.

*   **데이터 부족 문제**: 레이블링된 데이터가 적어 모델 학습이 어렵다. 이를 해결하기 위해 Weakly-supervised learning, 유사도 기반 데이터 검색 등이 제안되었으며, 최근에는 Dilated Convolution을 Transposed Residual Unit으로 대체하여 필요한 데이터 양을 줄이려는 시도(Danet-FCN 시리즈)가 있었다.
*   **전이 학습(Transfer Learning)의 한계**: 이미 학습된 모델을 재사용하는 전이 학습이 대안으로 제시되었으나, 지구과학이나 의료 영상과 같이 도메인 간의 분포 차이가 극심한 분야에서는 여전히 대량의 타겟 레이블 데이터가 필요하다는 한계가 있다.
*   **DDA의 필요성**: 타겟 도메인의 레이블 없이 지식을 전이하는 Unsupervised DDA가 더 현실적인 대안으로 꼽히지만, 지진파 영상 분야에서 효과적인 일반화 성능을 연구한 사례는 부족한 실정이다.

## 🛠️ Methodology

### 1. EarthAdaptNet (EAN) 아키텍처
EAN은 U-Net의 Encoder-Decoder 구조에서 영감을 받았으며, vanishing gradient 문제를 해결하기 위해 Residual Block(RB)과 Transposed Residual Block(TRB)을 사용한다.

*   **Encoder**: 여러 개의 RB로 구성된다. 각 RB는 두 개의 Convolution 레이어와 Batch Normalization, 그리고 $1 \times 1$ Convolution을 이용한 downsampling residual connection으로 이루어져 있다.
*   **Decoder**: Encoder와 대칭되는 구조로 TRB를 사용한다. TRB는 일반 Convolution 대신 Transposed Convolutional 레이어를 사용하여 이미지를 업샘플링한다.
*   **ASPP (Atrous Spatial Pyramid Pooling) 모듈**: Encoder와 Decoder 사이의 병목(Bottleneck) 지점에 위치하며, 서로 다른 dilatation rate(6, 12, 18)를 가진 3개의 Atrous Convolution, $1 \times 1$ Convolution, Image Pooling 레이어를 병렬로 배치하여 다양한 스케일의 특징을 캡처한다.
*   **Skip Connection**: Encoder의 RB와 Decoder의 TRB 사이에 연결되어 저수준의 세부 정보와 고수준의 문맥 정보를 결합한다.

### 2. EarthAdaptNet-DDA (EAN-DDA)
EAN의 인코더 부분을 백본으로 사용하고, 소스 도메인(SD)과 타겟 도메인(TD)의 분포 차이를 줄이기 위해 DeepCORAL 방법론을 적용하였다.

**CORAL Loss의 핵심 아이디어**는 두 도메인 특징 맵의 2차 통계량인 공분산(Covariance) 행렬을 일치시키는 것이다. 특징 공간의 차원을 $d$, 샘플 수를 $n_S, n_T$라고 할 때, CORAL 손실 함수는 다음과 같이 정의된다.

$$\mathcal{L}_{CORAL} = \frac{1}{4d^2} \| C_S - C_T \|_F^2$$

여기서 $\| \cdot \|_F^2$는 Frobenius norm이며, $C_S$와 $C_T$는 각각 소스 및 타겟 도메인의 공분산 행렬이다. 공분산 행렬은 다음과 같이 계산된다.

$$C_S = \frac{1}{n_S - 1} ( X_S^T X_S - \frac{1}{n_S} (\mathbf{1}^T X_S)^T (\mathbf{1}^T X_S) )$$
$$C_T = \frac{1}{n_T - 1} ( X_T^T X_T - \frac{1}{n_T} (\mathbf{1}^T X_T)^T (\mathbf{1}^T X_T) )$$

최종 학습 목표는 분류 손실(Classification Loss)과 CORAL 손실의 가중합을 최소화하는 것이다.

$$\mathcal{L} = \mathcal{L}_{classification} + \sum_{i=1}^{t} \lambda_i \mathcal{L}_{CORAL}^i$$

여기서 $t$는 CORAL 손실이 적용되는 레이어의 수이며, $\lambda$는 가중치 계수이다. 이 과정을 통해 모델은 도메인에 불변하는 특징(Domain-invariant features)을 학습하게 된다.

## 📊 Results

### 1. 데이터셋 및 설정
*   **Source Domain (SD)**: 네덜란드 F3 블록 데이터 (레이블 있음).
*   **Target Domain (TD)**: 캐나다 Penobscot 데이터 (레이블 없음/검증용으로만 사용).
*   **분석 대상**: 반사 패턴이 유사한 3가지 대표 클래스(Class 1, 2, 3)를 선정하여 비교 분석하였다.
*   **평가 지표**: Pixel Accuracy (PA), Mean Class Accuracy (MCA), Intersection over Union (IoU) 및 F1-Score 등을 사용하였다.

### 2. 세그멘테이션 결과 (EAN)
EAN-ASPP 모델은 기존 Baseline 및 DeepLab V3+보다 우수한 성능을 보였다.
*   **전체 정확도**: 픽셀 수준 정확도 84% 이상을 달성하였다.
*   **소수 클래스 성능**: 데이터가 부족한 소수 클래스에 대해 약 70%의 정확도를 보였으며, 특히 Zechstein 클래스의 경우 baseline 대비 정확도가 최대 39%까지 향상되었다.
*   **효율성**: Baseline 모델 대비 파라미터 수를 약 8배 감소시켰음에도 불구하고(약 84M $\rightarrow$ 11.9M), 학습 수렴 시간은 16시간에서 6시간으로 단축되었다.

### 3. 도메인 적응 결과 (EAN-DDA)
단순히 SD에서 학습된 모델을 TD에 적용한 'Direct Test'와 CORAL 손실을 적용한 'DDA' 모델을 비교하였다.
*   **Direct Test**: 최대 정확도가 0.07~0.68 수준으로 매우 낮아, 도메인 시프트 문제가 매우 심각함을 확인하였다.
*   **EAN-DDA**: CORAL 손실을 적용한 결과, 타겟 도메인에서 성능이 비약적으로 상승하였다.
    *   Class 3 (Zechstein $\leftrightarrow$ H4-H3): 정확도 **91%** 달성.
    *   Class 2 (Scruff $\leftrightarrow$ H5-H4): 정확도 **75%** 달성.
    *   Class 1 (Chalk/Rijnland $\leftrightarrow$ H5-H6): 정확도 **19%**로 낮게 나타났는데, 이는 저자들이 언급했듯이 두 도메인 간의 반사 패턴이 실제로는 완전히 일치하지 않았기 때문으로 분석된다.

## 🧠 Insights & Discussion

### 강점 및 성과
본 연구는 지진파 영상이라는 특수한 도메인에서 Unsupervised DDA의 가능성을 입증하였다. 특히 ASPP 모듈을 통해 유효 수용 영역(Valid Receptive Field)을 확장함으로써 소수 클래스의 인식률을 높인 점과, CORAL 손실을 통해 레이블 없이도 타겟 도메인의 특징 분포를 소스 도메인에 맞게 정렬한 점이 고무적이다.

### 한계 및 비판적 해석
1.  **패치 기반 모델의 한계**: 본 연구는 연산 효율성을 위해 패치 기반(Patch-based) 모델을 사용하였다. 하지만 이로 인해 전체 지진파 섹션의 공간적 맥락(Spatial Context) 정보가 손실되어, 인접한 클래스 간의 오분류(예: Upper N.S. $\leftrightarrow$ Middle N.S.)가 발생하는 문제가 관찰되었다.
2.  **반사 패턴의 불일치**: Class 1의 낮은 정확도는 단순히 알고리즘의 문제가 아니라, 도메인 간의 '유사성'이라는 가정이 완벽하지 않았음을 시사한다. 이는 DDA를 적용하기 전, 도메인 간의 시맨틱 유사성을 정밀하게 분석하는 과정이 선행되어야 함을 의미한다.
3.  **단순 분류 문제로의 치환**: DDA 실험 시 세그멘테이션이 아닌 패치 단위의 분류(Classification) 문제로 접근하였다. 이를 실제 픽셀 수준의 세그멘테이션으로 확장하는 것은 더 어려운 과제이며, 향후 Adversarial-based DDA 등의 도입이 필요해 보인다.

## 📌 TL;DR

본 논문은 지진파 영상 해석의 고질적인 문제인 **데이터 부족**과 **도메인 시프트**를 해결하기 위해 **EarthAdaptNet (EAN)** 아키텍처와 **DeepCORAL** 기반의 도메인 적응 기법을 제안하였다. EAN은 ASPP와 Residual 구조를 통해 소수 클래스의 세그멘테이션 성능을 높였으며, EAN-DDA는 소스-타겟 도메인 간의 공분산 행렬을 일치시킴으로써 레이블 없는 타겟 데이터에 대해서도 최대 91%의 분류 정확도를 달성하였다. 이 연구는 향후 레이블 확보가 어려운 다양한 지질학적 도메인에 대해 딥러닝 모델을 효율적으로 전이시키는 기초 틀을 제공한다.