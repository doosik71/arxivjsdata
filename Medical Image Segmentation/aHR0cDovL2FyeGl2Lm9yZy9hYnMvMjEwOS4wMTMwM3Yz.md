# Self-supervised Pseudo Multi-class Pre-training for Unsupervised Anomaly Detection and Segmentation in Medical Images

Yu Tian, Fengbei Liu, Guansong Pang, Yuanhong Chen, Yuyuan Liu, Johan W Verjans, Rajvinder Singh, Gustavo Carneiro (2023)

## 🧩 Problem to Solve

본 논문은 의료 영상 분석(Medical Image Analysis, MIA)에서 비지도 이상 탐지(Unsupervised Anomaly Detection, UAD) 및 세그멘테이션의 성능을 향상시키는 것을 목표로 한다. 의료 영상 스크리닝 데이터셋은 일반적으로 정상 영상은 매우 많으나, 비정상(질환) 영상은 매우 적고 그 종류 또한 다양하여 모든 질병 하위 클래스를 대표하는 학습 세트를 구축하기 어렵다는 특성이 있다.

기존의 UAD 방법들은 정상 영상만을 사용하여 정상 데이터의 분포를 학습하고, 여기서 벗어난 데이터를 이상치로 판별하는 One-Class Classifier (OCC) 방식에 의존한다. 그러나 정상 영상만으로 학습할 경우, 모델이 정상 데이터에 과적합(Overfitting)되어 크기가 작거나 형태가 모호한 미세 병변을 탐지하지 못하는 저차원 표현 학습의 한계가 발생한다. 이를 해결하기 위해 ImageNet으로 사전 학습된 모델을 사용하는 경우가 많으나, 자연 영상과 의료 영상의 도메인 차이로 인해 최적의 성능을 내지 못하는 문제가 존재한다.

결과적으로 본 연구는 의료 도메인 지식을 활용한 자기지도학습(Self-Supervised Learning, SSL) 기반의 사전 학습 방법을 통해, 정상 데이터만으로도 효과적인 특징 표현을 학습하여 미세한 병변 탐지 및 세그멘테이션 능력을 높이고자 한다.

## ✨ Key Contributions

본 논문의 핵심 아이디어는 정상 영상만을 이용하여 **가상의 비정상 클래스(Pseudo abnormal classes)**를 생성하고, 이를 정상 클래스와 대조하여 학습함으로써 모델이 '정상'과 '비정상'의 경계를 더 명확하게 인식하도록 만드는 것이다.

주요 기여 사항은 다음과 같다.
1. **MedMix Augmentation**: 정상 영상에서 패치를 추출하고 변형하여 붙이는 방식으로, 다양한 크기와 외형의 가상 병변을 시뮬레이션하는 강력한 데이터 증강 기법을 제안하였다.
2. **PMSACL (Pseudo Multi-class Strong Augmentation via Contrastive Learning)**: 정상 클래스와 여러 개의 가상 비정상 클래스를 구분하여 특징 공간에서 각각 밀집된 클러스터를 형성하도록 하는 새로운 대조 학습 최적화 방법을 제안하였다.
3. **손실 함수 개선**: 클러스터를 더 조밀하게 만들기 위한 Multi-centring loss와, 동일 클래스 내의 밀어내기 강도를 조절하는 온도 스케일링(Temperature Scaling)이 적용된 새로운 Contrastive loss를 도입하였다.
4. **범용적 사전 학습**: 제안 방법이 특정 모델에 국한되지 않고 PaDiM, IGD와 같은 다양한 SOTA UAD 방법론의 성능을 일관되게 향상시킴을 입증하였다.

## 📎 Related Works

기존의 UAD 접근 방식은 크게 두 가지로 나뉜다.
- **Predictive-based**: DSVDD나 OC-SVM처럼 정상 데이터의 분포를 학습하여 거리를 측정하는 방식이다.
- **Generative-based**: Auto-encoder나 GAN을 통해 정상 영상을 재구성하고, 재구성 오차(Reconstruction error)를 통해 이상치를 탐지하는 방식이다.

두 방식 모두 효과적인 특징 표현(Feature representation) 학습이 필수적이지만, 의료 영상에서는 병변이 매우 미세하여 일반적인 방법으로는 한계가 있다. 이를 위해 SSL 기반의 사전 학습이 시도되었으며, 특히 저자들의 이전 연구인 CCD(Constrained Contrastive Distribution learning)가 있었다. 하지만 CCD는 일반적인 컴퓨터 비전의 증강 기법을 사용하여 의료 영상의 특성(병변의 형태 등)을 충분히 반영하지 못했으며, 다운스트림 작업에서 정상 클래스 하나와 소수의 비정상 하위 클래스를 구분해야 한다는 점을 고려하지 않았다는 한계가 있었다.

## 🛠️ Methodology

### 전체 파이프라인
전체 시스템은 1) **PMSACL 기반의 자기지도 사전 학습** 단계와 2) 사전 학습된 인코더 $f_\theta$를 활용한 **UAD 모델의 파인튜닝** 단계로 구성된다.

### MedMix Augmentation
의료 영상의 이상치는 보통 비정상 조직의 비정상적인 성장으로 나타난다는 점에 착안하여, 정상 영상에서 작은 패치를 잘라내어 변형시킨 후 다시 붙이는 방식을 사용한다.
- **변형 과정**: 잘라낸 패치에 Color jittering, Gaussian noise, 비선형 강도 변환(Fisheye, Horizontal wave)을 적용하여 시각적 변형을 준다.
- **클래스 구성**: 증강 강도에 따라 $|A|=4$개의 분포를 설정한다. $A_0$는 약한 증강이 적용된 정상 이미지이며, $A_1, A_2, A_3$는 각각 1개, 2개, 3개의 가상 병변이 포함된 비정상 이미지 분포를 의미한다.

### PMSACL Pre-training 및 손실 함수
학습 목표는 정상 이미지($A_0$)와 가상 비정상 이미지들($A_1, A_2, A_3$)을 특징 공간에서 서로 다른 밀집된 클러스터로 분리하는 것이다. 전체 손실 함수는 다음과 같다.
$$\ell(D;\theta,\beta,\gamma) = \ell_{ctr}(D;\theta) + \ell_{PMSACL}(D;\theta) + \ell_{aug}(D;\beta) + \ell_{pos}(D;\gamma)$$

1. **Multi-centring Loss ($\ell_{ctr}$)**:
   각 증강 분포 $A_n$에 대한 평균 표현 $c_n$을 계산하고, 해당 클래스의 샘플들이 이 중심점으로 모이게 하여 유클리드 공간에서 클러스터를 밀집시킨다.
   $$\ell_{ctr}(D;\theta) = \mathbb{E}_{x\in D, n, a\sim A_n} \|f_\theta(a(x)) - c_n\|^2$$

2. **PMSACL Contrastive Loss ($\ell_{PMSACL}$)**:
   코사인 유사도를 기반으로 동일 클래스 샘플 간의 유사도는 높이고, 다른 클래스 간의 유사도는 낮춘다. 특히, 중심점 $c_n$으로 정규화된 특징량을 사용하여 하이퍼스피어 상에서 클러스터링을 수행한다.
   - **온도 스케일링 $\kappa(n,m)$**: 동일 클래스($n=m$)일 때는 온도를 낮추어 밀어내는 힘을 약하게 하고, 다른 클래스($n \neq m$)일 때는 표준 온도를 사용하여 더 강하게 밀어내도록 설계하였다.
   $$\kappa(n,m) = \begin{cases} 1/(\alpha\tau), & \text{if } n=m \\ 1/\tau, & \text{otherwise} \end{cases}$$

3. **Regularization Losses**:
   - $\ell_{aug}$: 모델이 현재 적용된 증강 함수가 무엇인지 분류하게 하여 학습을 정규화한다.
   - $\ell_{pos}$: 이미지 패치 간의 상대적 위치를 예측하게 하여 지역적 텍스처와 위치 특성을 학습하게 한다.

### Downstream UAD Methods
사전 학습된 인코더는 다음 두 가지 SOTA 방법론으로 파인튜닝된다.
- **IGD**: Auto-encoder 구조를 가지며, 재구성 오차와 가우시안 이상 분류기(Gaussian Anomaly Classifier)의 점수를 결합하여 이상치를 탐지한다.
- **PaDiM**: 각 패치 위치별로 다변량 가우시안 분포를 학습하고, 테스트 시 마할라노비스 거리(Mahalanobis distance)를 측정하여 이상 점수를 산출한다.

## 📊 Results

### 실험 설정
- **데이터셋**: Hyper-Kvasir (대장내시경), LAG (안저 영상-녹내장), Liu et al. (대장내시경), Covid-X (흉부 X-ray).
- **평가 지표**: AUROC, Specificity, Sensitivity, Accuracy (탐지 성능), IoU, Dice, Pro-score (세그멘테이션 성능).
- **비교 대상**: DAE, f-AnoGAN, ADGAN, PaDiM, IGD, CutPaste, 그리고 이전 연구인 CCD.

### 주요 결과
- **이상 탐지 성능**: 모든 데이터셋에서 PMSACL 사전 학습이 ImageNet이나 CCD보다 우수한 성능을 보였다. 특히 Hyper-Kvasir에서 PaDiM과 IGD의 AUROC를 각각 99.6%, 99.5%까지 끌어올려 SOTA를 달성하였다.
- **세그멘테이션 성능**: Hyper-Kvasir와 LAG 데이터셋에서 IoU와 Dice score가 유의미하게 상승하였다. PaDiM-PMSACL은 Hyper-Kvasir에서 40.6% IoU를 기록하며 SOTA를 달성했다.
- **데이터셋별 특성**: 흉부 X-ray(Covid-X)와 같이 병변이 매우 미세한 경우, 생성 기반 모델인 IGD가 PaDiM보다 월등히 높은 성능(AUROC 87.2%)을 보였으며, 이는 PMSACL이 미세한 특징 표현을 잘 학습했음을 시사한다.

## 🧠 Insights & Discussion

### 강점 및 분석
본 논문은 단순히 데이터 양을 늘리는 것이 아니라, **의료 영상의 병변 특성을 모사한 Pseudo-class**를 설계함으로써 UAD의 고질적인 문제인 '정상 데이터 과적합'을 효과적으로 해결하였다. 특히 $\ell_{ctr}$와 $\ell_{PMSACL}$의 결합은 특징 공간에서 정상 샘플들을 매우 조밀하게 응집시켜, 아주 작은 편차만으로도 이상치를 잡아낼 수 있는 민감도를 확보하게 했다.

### 한계 및 논의사항
1. **가상 클래스 수의 영향**: 실험을 통해 가상 클래스 수 $|A|=4$일 때 최적의 성능이 나타남을 확인하였다. 클래스 수가 너무 적으면 학습 정보가 부족하고, 너무 많으면 정상 영역이 가려져 모델이 과신(Over-confident)하는 문제가 발생한다. 이는 도메인마다 최적의 가상 클래스 수가 다를 수 있음을 의미한다.
2. **Centering 전략**: 클래스 중심점 $c_n$을 학습 중에 계속 업데이트하는 것보다, 학습되지 않은 인코더에서 추출한 초기 중심점을 고정하여 사용하는 것이 안정성과 효율성 면에서 훨씬 유리함이 밝혀졌다. 이는 UAD 모델에서 흔히 발생하는 'Catastrophic Collapse(모든 샘플이 한 점으로 모이는 현상)'를 방지하는 효과적인 전략이다.

## 📌 TL;DR

본 연구는 의료 영상의 비지도 이상 탐지를 위해 **가상 병변을 생성하는 MedMix 증강**과 **다중 가상 클래스를 대조 학습하는 PMSACL 사전 학습** 방법을 제안하였다. 이를 통해 정상 영상만으로도 효과적인 특징 표현을 학습하여, 미세 병변 탐지 및 세그멘테이션 성능을 획기적으로 향상시켰으며, 다양한 UAD 모델(PaDiM, IGD)에 범용적으로 적용 가능하다는 점을 입증하였다. 이 연구는 데이터 부족 문제가 심각한 의료 AI 분야에서 합성 데이터를 통한 SSL 사전 학습의 중요성을 제시하였다.