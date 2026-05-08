# CYCADA: CYCLE-CONSISTENT ADVERSARIAL DOMAIN ADAPTATION

Judy Hoffman, Eric Tzeng, Taesung Park, Jun-Yan Zhu, Phillip Isola, Kate Saenko, Alexei A. Efros, Trevor Darrell

## 🧩 Problem to Solve

심층 신경망은 대량의 학습 데이터에서 탁월한 성능을 보이지만, 학습된 지식을 새로운 데이터셋이나 환경(도메인)에 일반화하는 데 취약합니다. 특히 비현실적인 합성 데이터(예: 게임 엔진 데이터)로 학습한 모델이 실제 이미지에 적용될 때 성능이 크게 저하되는 **도메인 변화(Domain Shift)** 문제가 발생합니다.

기존 도메인 적응 방법들은 다음과 같은 한계를 가집니다:

- **특징 수준(Feature-level) 적응:** 도메인 불변(domain invariant) 표현을 학습하지만, 의미론적 일관성을 보장하지 못하고 픽셀 수준의 저수준(low-level) 외관 변화를 포착하지 못할 수 있습니다.
- **픽셀 수준(Pixel-level) 적응:** 이미지를 다른 도메인의 "스타일"로 변환하지만, 대형 이미지나 큰 도메인 변화에는 적용하기 어렵고, 변환 과정에서 중요한 **의미론적 정보(semantic information)**가 손실될 수 있습니다 (예: 고양이 그림이 개 사진으로 변환될 수 있음).

이 논문은 정렬된(aligned) 이미지 쌍 없이도, 도메인 변환 과정에서 원본 데이터의 의미론적 내용을 보존하면서 픽셀 및 특징 수준에서 효과적인 도메인 적응을 수행하는 방법을 찾는 것을 목표로 합니다.

## ✨ Key Contributions

- **CyCADA(Cycle-Consistent Adversarial Domain Adaptation) 모델 제안:** 픽셀 수준 및 특징 수준에서 표현을 적응시키고, 사이클 일관성(cycle-consistency)과 태스크 손실(task loss)을 활용하여 의미론적 일관성을 강화하는 새로운 판별 학습 방식의 적대적 도메인 적응 모델을 제시합니다.
- **다중 수준 적응 및 일관성 강화:**
  - **픽셀 수준 적응(Pixel-level adaptation):** 이미지를 대상 도메인의 스타일로 변환하여 저수준 외관 차이를 제거합니다.
  - **특징 수준 적응(Feature-level adaptation):** 신경망의 중간 특징 표현을 정렬하여 도메인 불변 특징을 학습합니다.
  - **사이클 일관성(Cycle-consistency):** 소스 $\to$ 타겟 $\to$ 소스로의 왕복 변환이 원본 이미지를 재구성하도록 강제하여 변환 과정에서 이미지 구조를 보존합니다.
  - **의미론적 일관성(Semantic consistency):** 변환 전후의 이미지가 사전 학습된 분류기에 의해 동일한 방식으로 분류되도록 유도하여 의미론적 내용을 유지합니다.
- **선행 연구 통합:** 기존의 특징 수준 및 이미지 수준 적대적 도메인 적응 방법과 사이클 일관성 기반 이미지-이미지 변환 기법(예: CycleGAN)을 통합합니다.
- **최첨단 성능 달성:** 숫자 분류(digit classification) 및 도로 장면의 의미론적 분할(semantic segmentation of road scenes)과 같은 다양한 적응 태스크에서 새로운 최첨단(state-of-the-art) 결과를 달성했습니다. 특히 합성 데이터에서 실제 데이터로의 어려운 전환 시나리오에서 큰 성능 향상을 보였습니다.
- **해석 가능성(Interpretability) 제공:** 픽셀 수준 적응은 시각적으로 변환 결과를 확인할 수 있어, 비지도 설정에서 적응 성공 여부를 검증하는 데 유용합니다.

## 📎 Related Works

- **초기 도메인 적응:** Saenko et al. (2010)이 시각 도메인 적응 문제를 소개했습니다. Torralba & Efros (2011)는 데이터셋 편향을 연구하며 이를 대중화했습니다.
- **특징 공간 정렬(Feature space alignment):**
  - **거리 기반:** Tzeng et al. (2014), Long & Wang (2015)은 MMD(Maximum Mean Discrepancy)를 통해 특징 분포 간의 거리를 최소화했습니다. Sun & Saenko (2016)는 CORAL을 사용했습니다.
  - **적대적 학습 기반(Adversarial learning based):** Ganin & Lempitsky (2015)의 DANN, Tzeng et al. (2015, 2017)의 ADDA 등은 도메인 분류기를 속이도록 특징을 학습하여 도메인 불변 표현을 생성했습니다. 이들은 GANs (Goodfellow et al., 2014)와 관련이 있습니다.
- **픽셀 공간 적응(Pixel space adaptation):**
  - **생성적 접근:** Liu & Tuzel (2016b)의 CoGANs, Ghifary et al. (2016)은 재구성(reconstruction) 목표를 추가했습니다.
  - **GAN 기반 이미지-이미지 변환:** Isola et al. (2016)은 Conditional GAN을 사용했으나, 이는 훈련을 위해 입력-출력 이미지 쌍을 필요로 합니다.
  - **비지도 이미지-이미지 변환 (GANs without paired data):** Yoo et al. (2016), Taigman et al. (2017b)의 DTN, Shrivastava et al. (2017)은 $L_1$ 재구성 손실이나 내용 유사성 손실을 사용했습니다. Bousmalis et al. (2017b)는 내용 유사성 손실을 사용했지만, 이미지의 어떤 부분이 도메인 간에 동일하게 유지되는지 사전 지식이 필요했습니다.
  - **사이클 일관성(Cycle-consistency):** Zhu et al. (2017)의 CycleGAN, Yi et al. (2017)의 DualGAN, Kim et al. (2017)이 제안했으며, 이미지 변환에서 놀라운 결과를 보여주었으나 특정 태스크에 agnostic했습니다. CyCADA는 이 사이클 일관성 손실의 효과에 영감을 받았습니다.
- **의미론적 분할을 위한 도메인 적응(Domain Adaptation for Semantic Segmentation):** Levinkov & Fritz (2013)은 날씨 조건에 따른 적응을 연구했습니다. Hoffman et al. (2016)은 컨볼루션 도메인 적대적 접근 방식을 제안했습니다. Ros et al. (2016b), Chen et al. (2017), Zhang et al. (2017) 등도 이 분야를 연구했습니다.

## 🛠️ Methodology

CyCADA는 비지도 도메인 적응 문제를 해결하기 위해 픽셀 수준 및 특징 수준의 적응을 결합하며, 사이클 일관성 및 의미론적 일관성 제약을 활용합니다.

1. **소스 태스크 모델 $f_S$ 사전 학습:**

   - 레이블이 있는 소스 데이터 ($X_S, Y_S$)를 사용하여 소스 태스크 모델 $f_S$를 사전 학습합니다. 이는 기본적인 분류 또는 분할 성능을 확립합니다.
   - 태스크 손실: $L_{task}(f_S, X_S, Y_S) = -\mathbb{E}_{(x_s, y_s) \sim (X_S, Y_S)} \sum_{k=1}^{K} \mathbf{1}_{[k=y_s]} \log(\sigma(f^{(k)}_S(x_s)))$

2. **픽셀 수준 적대적 적응 (Image-space Adaptation):**

   - 소스 이미지를 타겟 도메인으로 변환하는 생성기 $G_{S \to T}$와 이 변환된 이미지가 실제 타겟 이미지와 얼마나 유사한지 판별하는 판별기 $D_T$를 학습합니다.
   - 목표: $G_{S \to T}$는 $D_T$를 속여 변환된 이미지가 실제 타겟 이미지처럼 보이도록 만듭니다.
   - GAN 손실: $L_{GAN}(G_{S \to T}, D_T, X_T, X_S) = \mathbb{E}_{x_t \sim X_T}[\log D_T(x_t)] + \mathbb{E}_{x_s \sim X_S}[\log(1-D_T(G_{S \to T}(x_s)))]$
   - 반대 방향 변환을 위한 $G_{T \to S}$와 $D_S$도 유사하게 학습합니다.

3. **사이클 일관성 손실 (Cycle-Consistency Loss):**

   - 변환된 이미지가 원본 이미지의 구조나 내용을 유지하도록 강제합니다.
   - 소스 이미지 $x_s$를 타겟으로 변환($G_{S \to T}(x_s)$)하고 다시 소스로 재구성($G_{T \to S}(G_{S \to T}(x_s))$)했을 때 원본과 유사하도록 $L_1$ 페널티를 부과합니다. 타겟 이미지 $x_t$에 대해서도 동일하게 적용합니다.
   - $L_{cyc}(G_{S \to T}, G_{T \to S}, X_S, X_T) = \mathbb{E}_{x_s \sim X_S}[||G_{T \to S}(G_{S \to T}(x_s)) - x_s||_1] + \mathbb{E}_{x_t \sim X_T}[||G_{S \to T}(G_{T \to S}(x_t)) - x_t||_1]$

4. **의미론적 일관성 손실 (Semantic Consistency Loss):**

   - 이미지 변환 전후의 의미론적 내용이 유지되도록 합니다.
   - 사전 학습된 소스 태스크 모델 $f_S$를 "노이즈가 많은 레이블러"로 사용하여, 변환된 이미지가 원본 이미지와 동일한 방식으로 분류되도록 유도합니다.
   - $L_{sem}(G_{S \to T}, G_{T \to S}, X_S, X_T, f_S) = L_{task}(f_S, G_{T \to S}(X_T), p(f_S, X_T)) + L_{task}(f_S, G_{S \to T}(X_S), p(f_S, X_S))$
     - 여기서 $p(f, X) = \arg\max(f(X))$는 고정된 분류기 $f$로부터 예측된 레이블입니다.

5. **특징 수준 적대적 적응 (Feature-level Adaptation):**

   - 태스크 네트워크 $f_T$의 중간 특징 표현이 변환된 소스 이미지와 실제 타겟 이미지 간에 정렬되도록 합니다.
   - 이는 특징을 판별하는 별도의 판별기 $D_{feat}$를 통해 이루어집니다.
   - 특징 GAN 손실: $L_{GAN}(f_T, D_{feat}, f_S(G_{S \to T}(X_S)), X_T)$

6. **총체적 목표 함수 (Complete Objective):**

   - 위의 모든 손실 함수를 결합하여 최종적으로 타겟 모델 $f_T$를 학습합니다.
   - $L_{CyCADA} = L_{task}(f_T, G_{S \to T}(X_S), Y_S) + L_{GAN}(G_{S \to T}, D_T, X_T, X_S) + L_{GAN}(G_{T \to S}, D_S, X_S, X_T) + L_{GAN}(f_T, D_{feat}, f_S(G_{S \to T}(X_S)), X_T) + L_{cyc}(G_{S \to T}, G_{T \to S}, X_S, X_T) + L_{sem}(G_{S \to T}, G_{T \to S}, X_S, X_T, f_S)$
   - 최적화 문제: $f^{*}_T = \arg\min_{f_T} \min_{G_{S \to T}, G_{T \to S}} \max_{D_S, D_T} L_{CyCADA}$

7. **훈련 단계:**
   - 실제 메모리 제약으로 인해 모델은 단계적으로 훈련됩니다.
   - 1단계: 소스 태스크 모델 $f_S$ 사전 학습.
   - 2단계: 이미지 공간 적응(픽셀 수준 GAN, 사이클 일관성, 의미론적 일관성 손실)을 수행하여 $G_{S \to T}, G_{T \to S}, D_S, D_T$ 및 초기 $f_T$를 학습합니다.
   - 3단계: 특징 공간 적응을 수행하여 $f_T$를 업데이트하고 $D_{feat}$를 학습합니다.
   - 세그멘테이션 실험에서는 메모리 부족으로 의미론적 손실($L_{sem}$)을 사용하지 않았습니다.

## 📊 Results

- **숫자 적응 (Digit Adaptation: MNIST, USPS, SVHN $\to$ MNIST):**
  - CyCADA (픽셀+특징)는 MNIST $\to$ USPS (95.6%), USPS $\to$ MNIST (96.5%), SVHN $\to$ MNIST (90.4%)에서 기존 최첨단 모델들보다 우수하거나 경쟁력 있는 성능을 달성했습니다.
  - 작은 도메인 변화(USPS $\leftrightarrow$ MNIST)에는 픽셀 공간 적응만으로도 좋은 성능을 보였습니다.
  - 어려운 도메인 변화(SVHN $\to$ MNIST)에서는 특징 수준 적응이 픽셀 수준 적응보다 더 큰 이점을 제공하며, 이 둘을 결합했을 때 모든 경쟁 방법을 능가하는 최고 성능을 달성했습니다.
  - **Ablation Study (의미론적 일관성 손실 없음):** 사이클 제약만으로는 의미가 맞지 않는 변환(예: 숫자 2가 7처럼 보이게 변환)이 발생할 수 있음을 확인했습니다.
  - **Ablation Study (사이클 일관성 손실 없음):** 재구성 보장이 없으며, 의미론적 손실이 일부 의미론적 일관성을 유도하더라도 레이블이 뒤바뀌는(label flipping) 경우가 여전히 발생했습니다.
- **의미론적 분할 적응 (Semantic Segmentation: SYNTHIA Fall $\to$ Winter, GTA5 $\to$ CityScapes):**
  - **SYNTHIA 계절 간 적응 (Fall $\to$ Winter):** 픽셀 수준 적응만으로도 mIoU 63.3%, 픽셀 정확도 92.1%를 달성하여 최첨단 성능을 보였고, 오라클(target-trained) 성능에 근접했습니다. 시각적으로 낙엽을 눈으로 바꾸는 등 자연스러운 변환을 보여주었습니다.
  - **합성-실제 적응 (GTA5 $\to$ CityScapes):**
    - CyCADA (픽셀+특징)는 mIoU 35.4% (VGG16-FCN8s) 또는 39.5% (DRN-26)를 달성하여, 도메인 변화로 인한 성능 손실의 약 40%를 회복하며 최첨단 결과를 보여주었습니다. 픽셀 정확도는 83.6% (VGG16-FCN8s)로 오라클 성능에 거의 근접했습니다.
    - 모든 19개 클래스에서 성능을 개선하거나 유지했습니다.
    - 이미지 공간 적응 결과는 시각적으로 GTA5 이미지의 채도와 질감을 Cityscapes에 맞춰 변화시키고, 심지어 이미지 하단에 후드 장식을 추가하는 등 현실적인 도메인 변환을 보여주었습니다.

## 🧠 Insights & Discussion

- **사이클 일관성의 중요성:** 사이클 일관성 손실이 이미지 변환 과정에서 원본 이미지의 구조적 정보를 보존하는 데 매우 효과적임을 입증했습니다. 이는 특히 픽셀 수준의 의미론적 분할과 같이 세밀한 디테일이 중요한 태스크에서 중요합니다.
- **다중 수준 적응의 시너지:** 픽셀 수준과 특징 수준에서의 적응은 상호 보완적인 개선 효과를 제공하며, 함께 사용될 때 가장 높은 모델 성능을 달성할 수 있습니다.
- **이미지 공간 적응의 해석 가능성:** 픽셀 수준 적응은 변환된 이미지를 시각적으로 검사할 수 있어, 비지도 적응 설정에서 모델이 합리적인 방식으로 작동하는지 (예: 의미론적 내용이 유지되는지) "정신 건강 검사(sanity check)" 역할을 할 수 있다는 중요한 이점을 제공합니다.
- **의미론적 일관성 손실의 역할:** 의미론적 일관성 손실은 CycleGAN의 단점인 "레이블 뒤바뀜(label flipping)" 문제를 해결하고, 변환된 이미지가 원본의 의미를 유지하도록 돕는 데 필수적입니다.
- **한계점:** CyCADA의 전체 목적 함수를 엔드-투-엔드로 최적화하는 것은 많은 메모리를 요구하므로, 현재는 단계적(staged) 훈련 방식을 사용해야 합니다. 세그멘테이션 실험에서는 메모리 제약으로 인해 의미론적 손실을 적용하지 못했습니다. 이는 향후 모델 병렬화나 더 큰 GPU 메모리 사용을 통해 해결할 수 있는 문제입니다.
- **남아있는 과제:** 숫자 분류 실험의 혼동 행렬 분석에서 보듯이, 손글씨 숫자의 경우 7과 1, 0과 2처럼 서로 매우 유사하게 보이는 클래스 간의 오분류는 여전히 발생합니다. 이러한 고도로 유사한 클래스 간의 오류를 극복할 수 있는 모델을 개발하는 것은 여전히 미해결 과제로 남아 있습니다.

## 📌 TL;DR

CyCADA는 정렬되지 않은 데이터 쌍을 사용하여 픽셀 및 특징 수준에서 동시에 도메인 적응을 수행하는 비지도 학습 모델입니다. CycleGAN의 사이클 일관성 제약과 분류 모델의 의미론적 일관성 손실을 결합하여 도메인 변환 과정에서 이미지의 구조와 의미론적 내용을 보존합니다. 숫자 분류 및 합성-실제 의미론적 분할 태스크에서 최첨단 성능을 달성했으며, 특히 픽셀 수준 적응의 시각적 해석 가능성과 다중 수준 적응의 시너지 효과를 입증했습니다.
