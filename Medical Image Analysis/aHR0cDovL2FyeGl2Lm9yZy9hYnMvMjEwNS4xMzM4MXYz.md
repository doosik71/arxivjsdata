# Recent Advances and Clinical Applications of Deep Learning in Medical Image Analysis

Xuxin Chen, Ximin Wang, Ke Zhang, Kar-Ming Fung, Theresa C. Thai, Kathleen Moore, Robert S. Mannel, Hong Liu, Bin Zheng, Yuchen Qiu

## 🧩 Problem to Solve

의료 영상 분석 분야에서 딥러닝 모델의 발전은 크게 **대규모의 잘 주석된(annotated) 데이터셋 부족**으로 인해 어려움을 겪고 있습니다. 기존의 지도 학습(supervised learning) 방식은 충분한 양의 레이블링된 데이터가 필요하지만, 의료 영상 데이터는 수집 및 전문의에 의한 주석 작업이 시간과 비용이 많이 드는 어려운 작업입니다. 또한, 숙련된 임상 의사(예: 방사선과 의사, 병리학자)의 전문성에 의존하는 현재 임상 관행은 판독 간의 높은 가변성을 초래하며, 보다 효율적이고 정확하며 객관적인 컴퓨터 지원 진단(CAD) 시스템의 필요성이 증대되고 있습니다.

## ✨ Key Contributions

- **최신 딥러닝 기법 종합 분석**: 지난 5년간 의료 영상 분석에 적용된 딥러닝 연구들을 광범위하게 검토하고 요약했습니다.
- **비지도 및 준지도 학습 강조**: 특히 데이터 부족 문제를 해결하기 위한 최첨단 비지도 학습(unsupervised learning) 및 준지도 학습(semi-supervised learning) 방법론의 진행 상황과 기여를 중점적으로 다루었습니다.
- **다양한 응용 분야별 요약**: 분류(classification), 분할(segmentation), 탐지(detection), 정합(registration) 등 네 가지 주요 의료 영상 분석 작업에 대한 딥러닝 적용을 포괄적으로 설명했습니다.
- **최신 아키텍처 및 전략 소개**: Transformer 모델과 같은 최신 신경망 아키텍처와 주의 메커니즘(attention mechanism), 도메인 지식 통합, 불확실성 추정(uncertainty estimation) 등 성능 향상 전략을 논의했습니다.
- **주요 기술적 과제 및 미래 방향 제시**: 딥러닝 모델 개선의 주요 기술적 과제를 논의하고, 향후 연구 노력과 대규모 임상 적용을 위한 가능한 해결책을 제안했습니다.

## 📎 Related Works

이 연구는 의료 영상 분석 분야의 딥러닝 적용에 대한 기존의 다양한 리뷰 논문들과 차별점을 두며, 다음 연구들을 참조 및 비교하고 있습니다:

- **초기 딥러닝 기법**: Litjens et al. (2017) 및 Shen et al. (2017)은 주로 지도 학습 기반의 초기 딥러닝 기술을 다루었습니다.
- **GANs 활용**: Yi et al. (2019) 및 Kazeminia et al. (2020)은 다양한 의료 영상 작업에서의 GANs 적용을 검토했습니다.
- **준지도 및 다중 인스턴스 학습**: Cheplygina et al. (2019)는 진단 및 분할 작업에서 준지도 학습과 다중 인스턴스 학습의 사용법을 조사했습니다.
- **제한된 데이터셋 처리**: Tajbakhsh et al. (2020)은 의료 영상 분할에서 데이터셋 제한(희소하거나 약한 주석)을 다루는 다양한 방법을 연구했습니다.

본 논문은 이들 연구와 달리 '완전한 지도 학습이 아닌' 학습(self-supervised, unsupervised, semi-supervised learning)의 광범위한 적용과 Transformer 같은 최신 아키텍처에 중점을 둡니다.

## 🛠️ Methodology

이 논문은 의료 영상 분석을 위한 딥러닝 방법론을 다음 세 가지 주요 학습 패러다임과 성능 향상 전략으로 구분하여 검토합니다.

1. **지도 학습 (Supervised Learning)**

   - **CNNs**: 의료 영상 분석에서 가장 널리 사용되는 아키텍처로, 컨볼루션 계층(convolutional layers)과 풀링 계층(pooling layers)으로 구성됩니다.
   - **전이 학습 (Transfer Learning)**: ImageNet과 같은 대규모 자연 영상 데이터셋으로 사전 훈련된 모델을 의료 영상 작업에 미세 조정(fine-tuning)하여 데이터 부족 문제를 해결합니다.

2. **비지도 학습 (Unsupervised Learning)**

   - **오토인코더(Autoencoders)**: 차원 축소 및 특징 학습에 사용되며, 입력 데이터를 잠재 표현(latent representation)으로 인코딩하고 이를 다시 원본으로 디코딩하여 재구성 손실(reconstruction loss)을 최소화합니다. VAE(Variational Autoencoder)는 확률론적 방식으로 잠재 공간을 학습합니다.
   - **생성적 적대 신경망(Generative Adversarial Networks, GANs)**: 생성자(generator)와 판별자(discriminator)의 적대적 학습을 통해 실제와 유사한 데이터를 생성하며, 데이터 증강 및 반지도 학습에 활용됩니다.
   - **자기지도 학습(Self-supervised Learning)**: 레이블 없는 데이터 자체에서 입력과 레이블을 생성하여 유용한 특징 표현을 학습합니다.
     - **프리텍스트 태스크(Pretext Tasks)**: 이미지 인페인팅(inpainting), 색상화(colorization), 지그소 퍼즐(jigsaw puzzles) 등 보조 작업을 통해 특징을 학습합니다.
     - **대조 학습(Contrastive Learning)**: 유사한 쌍(positive pairs) 간의 유사도를 최대화하고 dissimilar한 쌍(negative pairs) 간의 유사도를 최소화하여 판별적인 특징을 학습합니다 (예: MoCo, SimCLR).

3. **준지도 학습 (Semi-supervised Learning)**

   - **일관성 정규화(Consistency Regularization)**: 레이블 없는 데이터에 작은 섭동(perturbation)을 가해도 모델의 예측이 크게 변하지 않도록 하는 손실 함수를 사용합니다 (예: Mean Teacher).
   - **가상 레이블링(Pseudo Labeling)**: 모델이 레이블 없는 데이터에 가상 레이블을 생성하고, 이를 레이블된 데이터와 함께 사용하여 모델을 훈련하는 과정을 반복합니다.
   - **생성 모델 기반 접근법**: GANs나 VAEs를 분류와 같은 특정 목표 작업에 더 집중하도록 변형하여 사용합니다.

4. **성능 향상 전략**
   - **주의 메커니즘 (Attention Mechanisms)**: 모델이 입력 데이터의 특정 부분에 집중하도록 하여 중요한 특징을 선별적으로 학습합니다 (예: 공간 주의, 채널 주의, 자기 주의).
   - **도메인 지식 통합 (Domain Knowledge Integration)**: 해부학적 정보, 3D 공간 문맥, 환자 메타데이터 등 의료 특유의 지식을 모델 설계에 통합하여 성능을 최적화합니다.
   - **불확실성 추정 (Uncertainty Estimation)**: 모델 예측의 신뢰도를 정량화하여 임상 환경에서 안전성 요구 사항을 충족하고 예측을 신뢰할 수 있게 합니다 (예: Bayesian 근사, 앙상블 기법).

## 📊 Results

이 논문은 다양한 의료 영상 분석 작업에서 딥러닝 모델의 성공적인 적용 사례와 최신 동향을 요약합니다.

- **분류 (Classification)**:

  - **지도 학습**: AlexNet, VGG, ResNet, DenseNet과 같은 CNN 기반 모델이 의료 영상 분류에서 뛰어난 성능을 보이며, 전이 학습이 데이터 부족 문제를 해결하는 데 핵심적인 역할을 합니다.
  - **비지도/자기지도 학습**: GAN을 이용한 데이터 증강(Frid-Adar et al., 2018a)과 자기지도 학습(특히 대조 학습 기반 SimCLR, MoCo)을 통한 사전 훈련이 제한된 데이터 환경에서 ImageNet 사전 훈련보다 우수한 성능을 나타냈습니다 (Azizi et al., 2021). 환자 메타데이터를 활용한 대조 학습도 성능을 크게 향상시켰습니다 (Vu et al., 2021).
  - **준지도 학습**: 반지도 GAN(Madani et al., 2018a) 및 Mean Teacher 모델(Liu et al., 2020a)이 레이블된 데이터가 제한적인 상황에서 분류 성능을 높이는 데 효과적이었습니다.

- **분할 (Segmentation)**:

  - **U-Net 및 변형**: U-Net은 의료 영상 분할의 가장 보편적인 아키텍처이며, Skip Connection, 잔여 블록(residual block) (V-Net), 재귀적 컨볼루션(recurrent convolution) (R2U-Net) 등의 변형을 통해 성능을 향상시켰습니다.
  - **Transformer 기반 모델**: TransUNet (Chen et al., 2021b), Swin-UNet (Cao et al., 2021) 등 CNN과 Transformer를 결합하거나 순수 Transformer 기반의 아키텍처가 장거리 의존성 학습 능력을 바탕으로 최첨단 성능을 달성했습니다.
  - **Mask R-CNN**: Mask R-CNN은 인스턴스 분할에 사용되며, 3D 볼륨 주의(volumetric attention) (Wang et al., 2019b) 또는 UNet++ 설계와의 결합을 통해 성능을 개선했습니다.
  - **비지도/자기지도 학습**: 해부학적 위치 예측(Bai et al., 2019)과 같은 프리텍스트 태스크 및 지역 대조 손실(local contrastive loss)을 활용한 자기지도 학습(Chaitanya et al., 2020)이 제한된 주석으로도 높은 분할 정확도를 보였습니다.
  - **준지도 학습**: Mean Teacher (Yu et al., 2019), 가상 레이블링(Fan et al., 2020) 및 VAE/GAN과 같은 생성 모델을 이용한 보조 작업(Sedai et al., 2017)이 분할 정확도를 높였습니다.

- **탐지 (Detection)**:

  - **최신 프레임워크**: Faster R-CNN, YOLO, RetinaNet 등 최신 객체 탐지 프레임워크가 의료 객체 탐지에 적용되었습니다.
  - **특정 유형 병변 탐지**: 3D 공간 문맥 정보 활용(Ding et al., 2017), 도메인 특화된 특징(Rijthoven et al., 2018), 준지도 학습(Wang et al., 2020c) 및 불확실성 추정(Nair et al., 2020)을 통해 작은 병변 탐지 및 오탐 감소에서 좋은 성능을 보였습니다.
  - **범용 병변 탐지**: Mask R-CNN 기반 ULDor (Tang et al., 2019), 다중 작업을 공동으로 수행하는 MULAN (Yan et al., 2019) 등 다양한 유형의 병변을 한 번에 탐지하는 모델이 개발되었습니다.
  - **비지도 병변 탐지**: VAE 및 GAN 기반 모델(AnoGAN, AnoVAEGAN)은 정상 분포를 학습하고 이탈하는 영역을 이상으로 간주하여 레이블 없는 병변 탐지(이상 감지)에 활용되었습니다 (Schlegl et al., 2017; Baur et al., 2018).

- **정합 (Registration)**:
  - **깊은 반복적 정합 (Deep Iterative Registration)**: 딥러닝 모델이 이미지 유사도 척도(similarity metric)를 학습하고, 이를 전통적인 최적화기와 결합하여 반복적으로 정합 파라미터를 업데이트합니다 (Simonovsky et al., 2016).
  - **지도 정합 (Supervised Registration)**: 변형 필드(deformation fields)를 직접 예측하는 방식으로, ground truth 변형 필드가 필요하며 이중 지도 학습(dual supervision)을 통해 정확도를 높일 수 있습니다 (Fan et al., 2019a).
  - **비지도 정합 (Unsupervised Registration)**: ground truth 변형 필드 없이 이미지 간의 유사도 척도와 정규화 항(regularization term)을 사용하여 변형 필드를 예측합니다 (VoxelMorph, Balakrishnan et al., 2018). GAN 기반 적대적 유사도(adversarial similarity) 학습(Fan et al., 2019b)은 다중 모달 정합에서 유망한 결과를 보였습니다.

## 🧠 Insights & Discussion

이 논문은 딥러닝이 의료 영상 분석에 가져온 엄청난 발전에도 불구하고, 임상 현장 적용에는 여전히 많은 과제와 개선점이 남아있음을 강조합니다.

- **과제별 관점**:

  - **분류**: 클래스 간 유사성이 높은 경우(예: 유방암 분류) 미세한 특징을 추출하는 것이 중요하며, 미세 분류(Fine-Grained Visual Classification, FGVC) 기법의 적용 가능성이 높습니다.
  - **탐지**: 작은 객체 탐지 및 클래스 불균형(class imbalance) 문제가 주요 과제입니다. 다중 스케일 특징(multi-scale features)과 Focal Loss 등이 도움이 되지만, anchor-free 방식의 원-스테이지(one-stage) 탐지기의 잠재력이 주목됩니다.
  - **분할**: 작은 병변 및 장기 분할과 클래스 불균형이 문제입니다. Dice 계수와 같은 영역 기반(region-based) 메트릭 외에 구조, 모양, 윤곽선과 같은 비영역 기반(non-region-based) 메트릭 개발의 필요성이 제기됩니다. Transformer 모델은 장거리 의존성(long-range dependencies) 모델링에 강점을 보여 유망합니다.
  - **정합**: 신뢰할 수 있는 ground truth 정합을 얻기 어렵고, 다단계 프레임워크의 복잡성이 과제입니다. 엔드 투 엔드(end-to-end) 학습이 가능한 비지도 정합 프레임워크 개발이 필요합니다.

- **학습 패러다임별 관점**:

  - **GAN**: 데이터 증강 및 반지도 학습에 유망하지만, 생성자와 목표 작업 간의 연결 강화와 적은 데이터에서의 학습 능력 개선이 필요합니다. 최신 증강 기법(예: Differentiable Augmentation)의 적용이 기대됩니다.
  - **자기지도 학습**: 특히 대조 학습은 지도 학습 전이 학습보다 우수한 특징 표현 학습 가능성을 보여주지만, 대규모 레이블 없는 데이터에 대한 의존성을 줄이고, 레이블 정보를 활용하여 긍정-부정 쌍을 구성하며, 작업에 맞는 데이터 증강 전략을 맞춤화하는 것이 중요합니다. 높은 계산 비용 문제도 해결해야 합니다.
  - **준지도 학습**: 데이터 증강 전략에 대한 의존도가 높고, 레이블된 데이터와 레이블 없는 데이터 간의 분포 불일치(distribution mismatch) 문제가 성능 저하를 초래할 수 있습니다. '도메인 적응(domain adaptation)' 연구에서 통찰력을 얻을 수 있습니다.

- **더 나은 아키텍처 및 파이프라인**:

  - 생물학적 및 인지적 메커니즘에서 영감을 받은 아키텍처(예: Transformer의 주의 메커니즘)가 지속적으로 중요할 것입니다.
  - 신경망 아키텍처 탐색(Neural Architecture Search, NAS)과 같은 자동화된 아키텍처 엔지니어링이 더 강력한 모델을 찾는 데 도움이 될 수 있습니다.
  - 이미지 전처리, 아키텍처 선택, 손실 함수, 데이터 증강 등 여러 하위 구성 요소를 자동으로 구성하는 자동화된 파이프라인(예: NiftyNet, nnU-Net) 개발이 임상 적용에 필수적입니다.

- **도메인 지식 통합**:

  - 의료 영상의 고유한 도전 과제(높은 클래스 유사성, 제한된 데이터 등)를 해결하기 위해 해부학적 정보, 환자 메타데이터 등 약한 도메인 지식뿐만 아니라, 양측 차이(bilateral difference)와 같은 방사선과 의사의 강력한 전문 지식을 효과적으로 통합하는 연구가 필요합니다.

- **대규모 임상 적용을 위한 과제**:
  - **데이터셋**: 단일 공용 데이터셋 사용으로 인한 "커뮤니티 전체의 과적합(community-wide overfitting)" 문제를 해결하기 위해 다양한 개인 데이터셋을 통합하고, 환자 프라이버시를 보호하는 연합 학습(federated learning)이 유망합니다.
  - **성능 평가**: 기술적 메트릭(정확도, Dice 계수) 외에 임상적 적용 가능성(환자 치료 개선 효과)을 반영하는 평가가 필요하며, 임상의와의 협력이 중요합니다.
  - **재현성 (Reproducibility)**: 데이터 선택 과정을 명확히 기술하여 연구 결과의 재현성을 높이고, 딥러닝 알고리즘에 대한 신뢰를 확보하는 것이 중요합니다.

## 📌 TL;DR

의료 영상 분석 분야 딥러닝은 대규모 주석 데이터 부족으로 어려움을 겪으며, 이를 해결하기 위해 **비지도 및 준지도 학습**이 큰 주목을 받고 있습니다. 본 논문은 분류, 분할, 탐지, 정합 등 주요 의료 영상 작업에서 **자기지도 대조 학습, GAN 기반 증강, Transformer 아키텍처** 등 최신 딥러닝 기법의 적용과 기여를 종합적으로 검토했습니다. 데이터 불균형, 작은 객체 문제, 임상 현장 적용의 한계 등 기술적 과제와 함께 **도메인 지식 통합, 자동화된 파이프라인, 연합 학습**을 통한 해결 방안과 미래 연구 방향을 제시하며, 학술적 성과와 임상적 활용 간의 간극을 줄이는 중요성을 강조합니다.
