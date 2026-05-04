# Fréchet Radiomic Distance (FRD): A Versatile Metric for Comparing Medical Imaging Datasets

Nicholas Konz, Richard Osuala, Preeti Verma, et al. (2025)

## 🧩 Problem to Solve

본 논문은 두 의료 영상 데이터셋이 동일한 분포(Distribution) 또는 도메인(Domain)에 속하는지 판단하는 문제에 집중한다. 이러한 분석은 의료 영상 생성 모델의 품질 평가, Out-of-Domain (OOD) 데이터 탐지, 그리고 영상 간 변환(Image-to-Image Translation) 모델의 성능 측정 등 현대 의료 영상 분석 및 딥러닝 분야에서 매우 중요하다.

기존에 사용되던 지표들은 크게 두 가지 한계를 가진다. 첫째, 세그멘테이션과 같은 특정 다운스트림 태스크(Downstream Task)의 성능에 의존하는 방식은 태스크 선택에 따른 편향이 발생할 수 있으며, 막대한 양의 레이블링 작업과 학습 비용이 요구된다. 둘째, natural imaging 분야에서 널리 쓰이는 FID (Fréchet Inception Distance)와 같은 인지적 지표(Perceptual Metrics)는 ImageNet과 같은 일반 이미지로 학습된 특징(Feature)을 사용하므로, 의료 영상의 핵심인 해부학적 특징과 임상적 의미를 충분히 포착하지 못한다. 따라서 본 연구의 목표는 의료 영상의 특성을 반영하면서도 태스크에 독립적이고, 임상적으로 해석 가능한 새로운 인지적 지표인 FRD (Fréchet Radiomic Distance)를 제안하는 것이다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 딥러닝 기반의 학습된 특징(Learned Features) 대신, 의료 영상 분석에서 이미 표준화되고 임상적으로 의미가 검증된 **Radiomic Features**를 사용하여 데이터셋 간의 거리를 측정하는 것이다. 

주요 기여 사항은 다음과 같다.
1. **FRD 제안**: Radiomic 특징을 기반으로 하여 태스크 독립적이며 해석 가능한 의료 영상 분포 비교 지표인 FRD를 제안한다. 특히, 주파수 영역의 특성을 포착하기 위해 Wavelet 필터를 도입하여 기존 버전(FRD v0)보다 표현력을 높였다.
2. **광범위한 검증**: OOD 탐지, 영상 변환 및 생성 모델 평가, 이상 징후 예측 등 다양한 시나리오에서 FRD가 FID, RadFID 등 기존 지표보다 우수함을 입증하였다.
3. **임상적 정렬 확인**: 전문 방사선 전문의(Radiologist)가 느끼는 이미지 품질 및 현실성(Realism)과 FRD 수치가 강한 상관관계를 가짐을 보였다.
4. **분석 프레임워크 및 코드 공개**: 의료 영상 유사도 지표를 다각도로 평가할 수 있는 프레임워크와 구현 코드를 공개하여 향후 연구의 기반을 마련하였다.

## 📎 Related Works

기존의 이미지 분포 비교 방식은 일반적으로 두 데이터셋을 저차원 특징 공간으로 인코딩한 후, 두 가우시안 분포 사이의 2-Wasserstein 거리인 Fréchet distance를 계산하는 방식을 취한다.

- **FID (Fréchet Inception Distance)**: ImageNet으로 사전 학습된 Inception v3 네트워크의 특징을 사용한다. 일반 이미지에서는 효과적이지만, 의료 영상의 해부학적 일관성을 측정하기에는 부적절하다는 지적이 많다.
- **RadFID (Radiology FID)**: FID의 의료 영상 버전으로, RadImageNet이라는 대규모 의료 영상 데이터셋으로 학습된 모델의 특징을 사용한다. 자연 이미지 기반 지표보다는 개선되었으나, 여전히 특징 추출 과정이 '블랙박스' 형태여서 임상적 해석이 불가능하고 소규모 데이터셋에서 불안정하다는 한계가 있다.
- **Radiomics**: 수작업으로 설계된(Hand-crafted) 특징들로, 암 진단이나 치료 반응 평가 등 임상 현장에서 널리 사용되어 왔다. 본 논문은 이 Radiomics를 진단 목적이 아닌 '데이터셋 분포 비교'라는 새로운 관점에서 적용하였다.

## 🛠️ Methodology

### 전체 파이프라인
FRD의 기본 흐름은 다음과 같다: $\text{Input Images} \rightarrow \text{Wavelet Filtering} \rightarrow \text{Radiomic Feature Extraction} \rightarrow \text{Normalization} \rightarrow \text{Fréchet Distance Calculation}$.

### 주요 구성 요소 및 절차
1. **특징 추출 (Feature Extraction)**:
   - PyRadiomics 라이브러리를 사용하여 1차 통계량(First-order statistics), GLCM (Gray Level Co-occurrence Matrix), GLRLM (Gray Level Run Length Matrix), GLSZM (Gray Level Size Zone Matrix) 등 총 464개의 특징을 추출한다.
   - **Wavelet 필터 적용**: 단순 영상뿐만 아니라, 공간 푸리에 변환 후 저주파/고주파 필터 조합(LL, LH, HL, HH)을 적용한 영상에서도 특징을 추출하여 주파수 영역의 세밀한 차이를 포착한다.

2. **정규화 (Normalization)**:
   - 이상치(Outlier)에 대한 강건함을 높이기 위해 Min-Max 정규화 대신 **Z-score 정규화**를 사용한다. 이때, 기준이 되는 데이터셋 $D_1$의 분포를 바탕으로 두 데이터셋 $D_1, D_2$의 특징을 모두 정규화한다.

3. **거리 계산 (Distance Calculation)**:
   - 추출된 특징 분포를 가우시안 분포라고 가정하고, 두 분포의 평균 벡터 $\mu_1, \mu_2$와 공분산 행렬 $\Sigma_1, \Sigma_2$를 이용해 Fréchet distance $d_F$를 계산한다.
   $$d_F(F_1, F_2) = \left( ||\mu_1 - \mu_2||_2^2 + \text{tr} [ \Sigma_1 + \Sigma_2 - 2(\Sigma_1 \Sigma_2)^{1/2} ] \right)^{1/2}$$
   - 최종적인 FRD 값은 계산의 안정성을 위해 로그 변환을 적용하여 정의한다.
   $$\text{FRD}(D_1, D_2) := \log d_F(f_{\text{radio}}(D_1), f_{\text{radio}}(D_2))$$

## 📊 Results

### 실험 설정
- **데이터셋**: Breast MRI (DBC), Brain MRI (BraTS), Lumbar Spine MRI/CT, Abdominal MRI/CT (CHAOS) 등 다양한 양상(Modality)과 도메인을 포함한다.
- **비교 대상**: FID, RadFID, KID, CMMD, FRD v0.
- **측정 지표**: OOD 탐지의 경우 AUC, Accuracy, Sensitivity, Specificity를 사용하며, 영상 변환 평가의 경우 다운스트림 태스크(세그멘테이션 Dice, 분류 AUC 등)와의 피어슨 상관계수($r$)를 측정한다.

### 주요 결과
1. **OOD 탐지**: FRD는 특히 도메인 차이가 미세한 Breast MRI 사례에서 타 지표보다 월등한 AUC와 민감도를 보였다. 이는 Wavelet 특징이 미세한 시각적 차이를 잘 포착했기 때문이다.
2. **영상 변환 평가**: FRD는 다운스트림 태스크 성능과 가장 강한 상관관계($r = -0.43$)를 보였다. 특히 해부학적 일관성을 측정하는 세그멘테이션 성능과의 상관관계가 매우 높았으며, 이는 FRD가 단순한 시각적 유사성을 넘어 구조적 보존 상태를 잘 반영함을 의미한다.
3. **전문가 평가와의 일치**: 방사선 전문의 3인이 평가한 이미지 현실성 점수와 FRD 값 사이에는 강한 음의 상관관계가 나타났다. 즉, 전문가가 보기에 더 실제 같은 이미지일수록 FRD 값이 낮게 측정되었다. 반면, FID와 RadFID는 오히려 전문가의 판단과 반대로 움직이는 경향(양의 상관관계)을 보였다.
4. **효율성 및 안정성**: FID 계열 지표들이 샘플 수 $N$이 적을 때 값이 크게 요동치는 반면, FRD는 $N=10$과 같은 매우 작은 샘플 사이즈에서도 매우 안정적인 수치를 유지하였다.
5. **적대적 공격 탐지**: 시각적으로는 거의 구분이 불가능한 FGSM 기반 적대적 공격(Adversarial Attack)이 가해졌을 때, FRD는 이를 민감하게 탐지하여 거리 값이 증가하는 양상을 보였다.

## 🧠 Insights & Discussion

### 강점 및 해석
FRD의 가장 큰 강점은 **임상적 해석 가능성(Interpretability)**이다. 학습된 특징을 사용하는 FID와 달리, FRD는 어떤 Radiomic 특징(예: 텍스처, 강도 등)이 두 분포의 차이를 만들었는지 정량적으로 분석할 수 있다. 논문에서는 $\Delta h$ (특징 변화 벡터)를 통해 MRI $\rightarrow$ CT 변환 시 어떤 텍스처 특징이 가장 크게 변했는지 분석함으로써 모델의 동작을 해석하는 가능성을 제시하였다.

또한, 소규모 의료 데이터셋 환경에서 FID의 불안정성을 해결했다는 점이 실무적으로 매우 중요하다. 이는 Radiomic 특징 공간의 유효 차원(Effective Dimensionality)이 학습된 특징 공간보다 낮아, 적은 샘플로도 공분산 행렬을 안정적으로 추정할 수 있기 때문으로 분석된다.

### 한계 및 비판적 논의
본 연구는 주로 방사선 영상(Radiology)에 집중되어 있다. 저자들도 언급하였듯이, 조직 병리 영상(Histopathology)이나 피부과 영상 등으로 확장하기 위해서는 각 도메인에 특화된 바이오마커(Biomarker)로 특징 세트를 조정해야 할 필요가 있다. 

또한, FRD의 절대값 자체는 기준 데이터셋 $D_1$이 무엇인지에 따라 달라지므로, 서로 다른 실험 환경 간의 절대적인 수치 비교보다는 동일 데이터셋 내에서의 상대적 비교에 한정하여 사용해야 한다는 제약이 있다.

## 📌 TL;DR

본 논문은 의료 영상의 특성을 반영하지 못하는 기존의 일반 영상 기반 지표(FID 등)를 대체하기 위해, 표준화된 **Radiomic 특징**을 활용한 **Fréchet Radiomic Distance (FRD)**를 제안한다. FRD는 OOD 탐지, 생성 모델 평가 등에서 기존 지표보다 뛰어난 성능을 보였으며, 특히 **전문가의 인지적 판단과 높은 일치도**를 보이고 **소규모 데이터셋에서도 안정적**이며 **결과에 대한 임상적 해석이 가능**하다는 독보적인 장점을 가진다. 이 연구는 향후 의료 영상 생성 AI의 정량적 평가 표준을 제시하고, 실제 임상 현장에서 모델의 도메인 적합성을 판단하는 도구로 활용될 가능성이 높다.